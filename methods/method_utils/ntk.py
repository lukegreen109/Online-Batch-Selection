import os

import torch
import torch.nn.functional as F

from methods.method_utils.build_teacher_model import build_teacher_model
from methods.method_utils.diagnostics_context import DiagnosticsRunContext

try:
    from torch.func import functional_call, jacrev, vmap

    _HAS_TORCH_FUNC = True
except ImportError:
    _HAS_TORCH_FUNC = False


class NTKDiagnostics:
    def __init__(
        self,
        logger,
        context: DiagnosticsRunContext,
        num_classes,
        ntk_max_samples,
        ntk_top_k,
        enabled,
        teacher_model_config=None,
        save_spectral_decay=False,
    ):
        self.logger = logger
        self.context = context
        self.loader = context.fixed_train_loader
        self.num_classes = int(num_classes)
        self.ntk_max_samples = int(ntk_max_samples)
        self.ntk_top_k = int(ntk_top_k)
        self.enabled = bool(enabled)
        self.teacher_model_config = teacher_model_config
        self.save_spectral_decay = bool(save_spectral_decay)

        self._ntk_init_kernel = None
        self._ntk_init_norm = None
        self._ntk_init_num_samples = 0
        self._ntk_init_top_eigenvectors = None
        self._ntk_init_top_k = 0
        self._ntk_warned_fallback = False
        self._ntk_warned_default_cap = False
        self._ntk_warned_invalid_top_k = False
        self._ntk_subset_inputs_cpu = None
        self._ntk_subset_labels_cpu = None
        self._cuda_context_initialized = False
        self._teacher_model = None
        self._teacher_kernel = None
        self._teacher_norm = None
        self._teacher_num_samples = 0
        self._teacher_eigenvalues = None
        self._teacher_kernel_initialized = False
        self._teacher_support_warned = False
        self._spectral_decay = []
        self._spectral_decay_steps = []
        self._spectral_decay_num_samples = []
        self.spectral_decay_path = None
        if self.save_spectral_decay:
            spectral_decay_dir = os.path.join(
                self.context.project_root,
                'spectral_decay',
                self.context.dataset_name,
            )
            os.makedirs(spectral_decay_dir, exist_ok=True)
            self.spectral_decay_path = os.path.join(
                spectral_decay_dir,
                f'{self.context.artifact_stem}.p',
            )

    @staticmethod
    def _symmetrize_kernel(kernel):
        return 0.5 * (kernel + kernel.t())

    @staticmethod
    def _kernel_inner_product(lhs, rhs):
        return torch.sum(lhs * rhs)

    @staticmethod
    def _center_kernel(kernel):
        row_mean = kernel.mean(dim=1, keepdim=True)
        col_mean = kernel.mean(dim=0, keepdim=True)
        total_mean = kernel.mean()
        return kernel - row_mean - col_mean + total_mean

    @staticmethod
    def _strip_module_prefix(state_dict):
        if not state_dict:
            return state_dict
        if all(key.startswith('module.') for key in state_dict.keys()):
            return {key[len('module.'):]: value for key, value in state_dict.items()}
        return state_dict

    def _warn_teacher_unsupported(self, message):
        if not self._teacher_support_warned:
            self.logger.info(message)
            self._teacher_support_warned = True

    def _build_teacher_model(self, device):
        if self._teacher_model is not None or self.teacher_model_config is None:
            return self._teacher_model

        source = self.teacher_model_config.get('diagnostics', {}).get('ntk_teacher_model', {}).get(
            'source',
            self.teacher_model_config.get('teacher_model_source'),
        )
        if source == 'clip':
            self._warn_teacher_unsupported(
                'Warning: NTK teacher-kernel diagnostics do not support clip teachers; skipping teacher NTK comparisons.'
            )
            return None

        if source not in {'timm', 'local_pretrained'}:
            self._warn_teacher_unsupported(
                f'Warning: NTK teacher model source {source} is not supported; skipping teacher NTK comparisons.'
            )
            return None

        teacher_model = build_teacher_model(self.teacher_model_config, self.logger)
        teacher_model.to(device)
        teacher_model.eval()
        self._teacher_model = teacher_model
        return self._teacher_model

    def _sorted_eigenvalues(self, kernel):
        eigenvalues = torch.linalg.eigvalsh(self._symmetrize_kernel(kernel))
        return torch.flip(eigenvalues, dims=(0,))

    def _save_spectral_decay(self):
        if not self.save_spectral_decay or self.spectral_decay_path is None:
            return
        payload = {
            'eigenvalues': self._spectral_decay,
            'steps': self._spectral_decay_steps,
            'num_samples': self._spectral_decay_num_samples,
        }
        if self._teacher_eigenvalues is not None:
            payload['teacher_eigenvalues'] = self._teacher_eigenvalues
        torch.save(payload, self.spectral_decay_path)

    def _record_spectral_decay(self, total_step, eigenvalues_desc, num_samples):
        if not self.save_spectral_decay:
            return
        self._spectral_decay.append(eigenvalues_desc.detach().cpu().to(dtype=torch.float32))
        self._spectral_decay_steps.append(int(total_step))
        self._spectral_decay_num_samples.append(int(num_samples))
        self._save_spectral_decay()

    def _ensure_teacher_kernel(self, device):
        if self._teacher_kernel_initialized:
            return
        self._teacher_kernel_initialized = True

        teacher_model = self._build_teacher_model(device)
        if teacher_model is None:
            return

        teacher_ntk, teacher_num_samples = self._compute_empirical_ntk(teacher_model, device)
        if teacher_ntk is None:
            self._warn_teacher_unsupported(
                'Warning: unable to compute teacher NTK; skipping teacher NTK comparisons.'
            )
            return

        self._teacher_kernel = self._symmetrize_kernel(teacher_ntk.detach().cpu().to(dtype=torch.float64))
        self._teacher_norm = float(torch.norm(self._teacher_kernel, p='fro').item())
        self._teacher_num_samples = int(teacher_num_samples)
        self._teacher_eigenvalues = self._sorted_eigenvalues(self._teacher_kernel).to(dtype=torch.float32)
        self._save_spectral_decay()

    def _flatten_grads(self, grads, params):
        flat = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat.append(torch.zeros_like(param, memory_format=torch.contiguous_format).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        return torch.cat(flat, dim=0)

    def _build_fixed_balanced_ntk_subset(self):
        if self._ntk_subset_inputs_cpu is not None:
            return self._ntk_subset_inputs_cpu, self._ntk_subset_inputs_cpu.size(0)
        if self.loader is None:
            return None, 0

        per_class = self.ntk_max_samples // self.num_classes
        if per_class == 0:
            self.logger.info(
                'Warning: diagnostics_ntk_max_samples is smaller than num_classes; NTK diagnostics will be skipped.'
            )
            return None, 0

        requested_total = per_class * self.num_classes
        if requested_total != self.ntk_max_samples:
            self.logger.info(
                f'NTK diagnostics: using {requested_total} samples ({per_class} per class) '
                f'instead of requested {self.ntk_max_samples} to keep class balance.'
            )

        by_class = {cls_idx: [] for cls_idx in range(self.num_classes)}
        for datas in self.loader:
            if 'target' not in datas:
                continue

            inputs_cpu = datas['input'].detach().cpu()
            targets_cpu = datas['target'].detach().cpu().long()
            for inp, tgt in zip(inputs_cpu, targets_cpu):
                cls_idx = int(tgt.item())
                if 0 <= cls_idx < self.num_classes and len(by_class[cls_idx]) < per_class:
                    by_class[cls_idx].append(inp.clone())

            if all(len(by_class[c]) >= per_class for c in range(self.num_classes)):
                break

        min_available = min(len(by_class[c]) for c in range(self.num_classes))
        if min_available == 0:
            self.logger.info('Warning: unable to build balanced NTK subset; NTK diagnostics will be skipped.')
            return None, 0

        if min_available < per_class:
            self.logger.info(
                f'NTK diagnostics: reducing to {min_available * self.num_classes} samples '
                f'({min_available} per class) due to class availability.'
            )

        selected_inputs = []
        selected_labels = []
        for cls_idx in range(self.num_classes):
            class_samples = by_class[cls_idx][:min_available]
            selected_inputs.extend(class_samples)
            selected_labels.extend([cls_idx] * len(class_samples))

        self._ntk_subset_inputs_cpu = torch.stack(selected_inputs, dim=0).contiguous()
        self._ntk_subset_labels_cpu = torch.tensor(selected_labels, dtype=torch.long)
        self.logger.info(
            f'NTK diagnostics: fixed balanced subset created with {self._ntk_subset_inputs_cpu.size(0)} '
            f'samples ({min_available} per class).'
        )
        return self._ntk_subset_inputs_cpu, self._ntk_subset_inputs_cpu.size(0)

    def _collect_ntk_inputs(self, device):
        inputs_cpu, total_samples = self._build_fixed_balanced_ntk_subset()
        if inputs_cpu is None or total_samples == 0:
            return None, 0
        return inputs_cpu.to(device), total_samples

    def _ensure_cuda_context(self, device):
        if device.type != 'cuda' or self._cuda_context_initialized:
            return
        _ = torch.empty(1, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        self._cuda_context_initialized = True

    def _compute_empirical_ntk_autograd(self, model, device):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return None, 0

        inputs, total_samples = self._collect_ntk_inputs(device)
        if inputs is None:
            return None, 0

        self._ensure_cuda_context(device)
        per_sample_jacobians = []
        for sample_idx in range(total_samples):
            sample_input = inputs[sample_idx : sample_idx + 1]
            sample_logits = model(sample_input)
            class_grads = []
            num_outputs = sample_logits.size(1)
            for class_idx in range(num_outputs):
                grads = torch.autograd.grad(
                    sample_logits[0, class_idx],
                    params,
                    retain_graph=(class_idx < num_outputs - 1),
                    create_graph=False,
                    allow_unused=True,
                )
                class_grads.append(self._flatten_grads(grads, params))

            per_sample_jacobians.append(torch.cat(class_grads, dim=0).detach().cpu())

        jacobian = torch.stack(per_sample_jacobians, dim=0)
        return jacobian @ jacobian.t(), total_samples

    def _compute_empirical_ntk_torch_func(self, model, device):
        inputs, total_samples = self._collect_ntk_inputs(device)
        if inputs is None:
            return None, 0

        self._ensure_cuda_context(device)
        params = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        if not params:
            return None, 0

        buffers = dict(model.named_buffers())

        def model_single(curr_params, curr_buffers, x):
            out = functional_call(model, (curr_params, curr_buffers), (x.unsqueeze(0),))
            return out.squeeze(0)

        jac = vmap(jacrev(model_single, argnums=0), in_dims=(None, None, 0))(params, buffers, inputs)

        ntk = None
        for param_jac in jac.values():
            j_flat = param_jac.reshape(param_jac.size(0), param_jac.size(1), -1)
            contribution = torch.einsum('nop,mop->nm', j_flat, j_flat)
            ntk = contribution if ntk is None else ntk + contribution

        if ntk is None:
            return None, 0
        return ntk.detach().cpu(), total_samples

    def _compute_empirical_ntk(self, model, device):
        if not self._ntk_warned_default_cap and self.ntk_max_samples == 1000:
            self.logger.info('NTK diagnostics: using default diagnostics_ntk_max_samples=1000.')
            self._ntk_warned_default_cap = True

        if _HAS_TORCH_FUNC:
            try:
                return self._compute_empirical_ntk_torch_func(model, device)
            except RuntimeError as err:
                err_msg = str(err).lower()
                if 'out of memory' in err_msg or ('cuda' in err_msg and 'memory' in err_msg):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.info(
                        'Warning: torch.func NTK path ran out of memory; falling back to slower autograd NTK implementation.'
                    )
                    return self._compute_empirical_ntk_autograd(model, device)
                raise

        if not self._ntk_warned_fallback:
            self.logger.info('Warning: torch.func not available; falling back to slower autograd NTK implementation.')
            self._ntk_warned_fallback = True
        return self._compute_empirical_ntk_autograd(model, device)

    def _get_ntk_targets_one_hot(self, num_samples):
        if self._ntk_subset_labels_cpu is None:
            return None
        if self._ntk_subset_labels_cpu.numel() != num_samples:
            self.logger.info(
                'Warning: NTK subset labels do not match the empirical kernel size; skipping label-based NTK diagnostics.'
            )
            return None
        return F.one_hot(
            self._ntk_subset_labels_cpu,
            num_classes=self.num_classes,
        ).to(dtype=torch.float64)

    def _compute_kernel_target_alignment_metrics(self, kernel, num_samples):
        log_data = {}
        one_hot_targets = self._get_ntk_targets_one_hot(num_samples)
        if one_hot_targets is None:
            return log_data

        target_kernel = one_hot_targets @ one_hot_targets.t()
        kernel_norm = torch.linalg.norm(kernel, ord='fro')
        target_norm = torch.linalg.norm(target_kernel, ord='fro')
        kta_denom = kernel_norm * target_norm
        if kta_denom.item() > 0.0:
            kta = self._kernel_inner_product(kernel, target_kernel) / kta_denom
            log_data['diagnostics/ntk_kernel_target_alignment'] = float(kta.item())

        centered_kernel = self._center_kernel(kernel)
        centered_target_kernel = self._center_kernel(target_kernel)
        centered_kernel_norm = torch.linalg.norm(centered_kernel, ord='fro')
        centered_target_norm = torch.linalg.norm(centered_target_kernel, ord='fro')
        cka_denom = centered_kernel_norm * centered_target_norm
        if cka_denom.item() > 0.0:
            cka = self._kernel_inner_product(centered_kernel, centered_target_kernel) / cka_denom
            log_data['diagnostics/ntk_centered_kernel_alignment'] = float(cka.item())

        return log_data

    def _compute_ntk_spectrum_metrics(self, current_ntk, num_samples, total_step):
        log_data = {}
        if num_samples <= 0:
            return log_data, None

        if self.ntk_top_k <= 0:
            if not self._ntk_warned_invalid_top_k:
                self.logger.info('Warning: diagnostics_ntk_top_k must be positive. Top-k NTK diagnostics will be skipped.')
                self._ntk_warned_invalid_top_k = True
            return log_data, None

        kernel = self._symmetrize_kernel(current_ntk.detach().cpu().to(dtype=torch.float64))

        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
        eigenvalues_desc = torch.flip(eigenvalues, dims=(0,))
        eigenvectors_desc = torch.flip(eigenvectors, dims=(1,))
        self._record_spectral_decay(total_step, eigenvalues_desc, num_samples)

        effective_k = min(self.ntk_top_k, int(num_samples), int(eigenvalues_desc.numel()))
        if effective_k <= 0:
            return log_data, eigenvalues_desc

        top_eigenvalues = eigenvalues_desc[:effective_k]
        top_eigenvectors = eigenvectors_desc[:, :effective_k]
        nonnegative_top_eigenvalues = top_eigenvalues.clamp_min(0.0)
        nonnegative_eigenvalues = eigenvalues_desc.clamp_min(0.0)
        log_data['diagnostics/ntk_top_k'] = int(effective_k)

        total_eigenvalue_mass = nonnegative_eigenvalues.sum()
        if total_eigenvalue_mass.item() > 0.0:
            concentration = nonnegative_top_eigenvalues.sum() / total_eigenvalue_mass
            log_data['diagnostics/ntk_eigenvalue_concentration'] = float(concentration.item())
        else:
            log_data['diagnostics/ntk_eigenvalue_concentration'] = 0.0

        one_hot_targets = self._get_ntk_targets_one_hot(num_samples)
        if one_hot_targets is not None:
            projected_targets = one_hot_targets.t() @ top_eigenvectors
            alignment = torch.linalg.norm(projected_targets, dim=0)
            for rank_idx in range(effective_k):
                log_data[f'diagnostics/ntk_eigenvector_target_alignment_{rank_idx + 1}'] = float(
                    alignment[rank_idx].item()
                )
            weighted_alignment = nonnegative_top_eigenvalues * alignment.pow(2)
            log_data['diagnostics/ntk_top_eigenvector_label_alignment'] = float(weighted_alignment.sum().item())

        if self._ntk_init_top_eigenvectors is None:
            self._ntk_init_top_eigenvectors = top_eigenvectors.clone()
            self._ntk_init_top_k = effective_k
            log_data['diagnostics/ntk_top_eigenspace_overlap_init'] = 1.0
            return log_data, eigenvalues_desc

        if self._ntk_init_top_eigenvectors.size(0) != top_eigenvectors.size(0):
            self.logger.info(
                'Warning: NTK eigenspace shape changed between initialization and current step; skipping eigenspace overlap logging.'
            )
            return log_data, eigenvalues_desc

        overlap_k = min(self._ntk_init_top_k, effective_k)
        if overlap_k <= 0:
            return log_data, eigenvalues_desc

        init_top_eigenvectors = self._ntk_init_top_eigenvectors[:, :overlap_k]
        current_top_eigenvectors = top_eigenvectors[:, :overlap_k]
        overlap = current_top_eigenvectors.t() @ init_top_eigenvectors
        log_data['diagnostics/ntk_top_eigenspace_overlap_init'] = float(
            overlap.norm(p='fro').pow(2).item() / overlap_k
        )
        return log_data, eigenvalues_desc

    def _compute_relative_distance(self, reference_kernel, current_kernel, reference_norm):
        denom = max(float(reference_norm), 1e-12)
        return float(torch.norm(current_kernel - reference_kernel, p='fro').item() / denom)

    def _compute_angular_distance(self, reference_kernel, current_kernel, reference_norm, current_norm):
        denom = max(float(reference_norm) * float(current_norm), 1e-12)
        cosine = float(self._kernel_inner_product(reference_kernel, current_kernel).item() / denom)
        return float(1.0 - cosine)

    def log_metrics(self, model, device, total_step):
        if not self.enabled or self.loader is None:
            return {}

        current_ntk, num_samples = self._compute_empirical_ntk(model, device)
        if current_ntk is None:
            return {}

        current_ntk = self._symmetrize_kernel(current_ntk.detach().cpu().to(dtype=torch.float64))
        current_norm = float(torch.norm(current_ntk, p='fro').item())

        log_data = {
            'diagnostics/ntk_num_samples': int(num_samples),
            'diagnostics/ntk_norm_fro': current_norm,
        }
        spectrum_metrics, _ = self._compute_ntk_spectrum_metrics(current_ntk, num_samples, total_step)
        log_data.update(spectrum_metrics)
        log_data.update(self._compute_kernel_target_alignment_metrics(current_ntk, num_samples))

        if self._ntk_init_kernel is None:
            self._ntk_init_kernel = current_ntk
            self._ntk_init_norm = current_norm
            self._ntk_init_num_samples = num_samples
            log_data['diagnostics/ntk_init_norm_fro'] = float(self._ntk_init_norm)
            log_data['diagnostics/ntk_relative_drift_init'] = 0.0
            log_data['diagnostics/ntk_angular_kernel_distance_init'] = 0.0
        elif current_ntk.shape == self._ntk_init_kernel.shape:
            relative_drift_init = self._compute_relative_distance(
                self._ntk_init_kernel,
                current_ntk,
                self._ntk_init_norm,
            )
            log_data['diagnostics/ntk_init_norm_fro'] = float(self._ntk_init_norm)
            log_data['diagnostics/ntk_relative_drift_init'] = float(relative_drift_init)
            log_data['diagnostics/ntk_angular_kernel_distance_init'] = self._compute_angular_distance(
                self._ntk_init_kernel,
                current_ntk,
                self._ntk_init_norm,
                current_norm,
            )
        else:
            self.logger.info(
                'Warning: NTK shape changed between initialization and current step; skipping relative NTK change logging.'
            )
            return log_data

        self._ensure_teacher_kernel(device)
        if self._teacher_kernel is not None:
            if current_ntk.shape == self._teacher_kernel.shape:
                log_data['diagnostics/ntk_teacher_norm_fro'] = float(self._teacher_norm)
                log_data['diagnostics/ntk_relative_distance_teacher'] = self._compute_relative_distance(
                    self._teacher_kernel,
                    current_ntk,
                    self._teacher_norm,
                )
                log_data['diagnostics/ntk_angular_kernel_distance_teacher'] = self._compute_angular_distance(
                    self._teacher_kernel,
                    current_ntk,
                    self._teacher_norm,
                    current_norm,
                )
            else:
                self.logger.info(
                    'Warning: teacher NTK shape does not match the current NTK; skipping teacher NTK comparison metrics.'
                )

        return log_data

    def finalize(self):
        self._save_spectral_decay()