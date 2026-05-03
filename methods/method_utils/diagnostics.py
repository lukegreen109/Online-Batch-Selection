import numpy as np
import torch
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

try:
    from torch.func import functional_call, jacrev, vmap
    _HAS_TORCH_FUNC = True
except ImportError:
    _HAS_TORCH_FUNC = False


class DiagnosticsLogger:
    def __init__(
        self,
        logger,
        num_classes,
        criterion,
        histogram_max_points=200000,
        lr_max_iter=300,
        ntk_enabled=True,
        ntk_max_samples=None,
        ntk_top_k=10,
    ):
        self.logger = logger
        self.num_classes = num_classes
        self.criterion = criterion
        self.histogram_max_points = histogram_max_points
        self.lr_max_iter = lr_max_iter
        self.ntk_enabled = ntk_enabled
        # Full-dataset empirical NTK is typically intractable for non-trivial models.
        # Use a safe default cap unless explicitly overridden in config.
        self.ntk_max_samples = 1000 if ntk_max_samples is None else int(ntk_max_samples)
        self.ntk_top_k = int(ntk_top_k)
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

    def _sample_for_histogram(self, tensor):
        flat = tensor.detach().float().reshape(-1).cpu()
        if flat.numel() <= self.histogram_max_points:
            return flat
        sample_idx = torch.randperm(flat.numel())[:self.histogram_max_points]
        return flat[sample_idx]

    def _collect_param_grad_stats(self, model):
        log_data = {}
        param_chunks = []
        grad_chunks = []
        last_param = None
        last_param_name = None
        last_grad = None
        last_grad_name = None

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_cpu = param.detach().float().reshape(-1).cpu()
            if param_cpu.numel() > 0:
                param_chunks.append(param_cpu)
                log_data[f'diagnostics/parameter_norms/{name}'] = torch.norm(param_cpu, p=2).item()
                last_param = param_cpu
                last_param_name = name

            if param.grad is not None:
                grad_cpu = param.grad.detach().float().reshape(-1).cpu()
                if grad_cpu.numel() > 0:
                    grad_chunks.append(grad_cpu)
                    log_data[f'diagnostics/gradient_norms/{name}'] = torch.norm(grad_cpu, p=2).item()
                    last_grad = grad_cpu
                    last_grad_name = name

        if param_chunks:
            flat_params = torch.cat(param_chunks, dim=0)
            log_data['diagnostics/parameter_norm_l2_global'] = torch.norm(flat_params, p=2).item()
            log_data['diagnostics/histograms/parameters'] = wandb.Histogram(
                self._sample_for_histogram(flat_params).numpy()
            )

        if grad_chunks:
            flat_grads = torch.cat(grad_chunks, dim=0)
            log_data['diagnostics/gradient_norm_l2_global'] = torch.norm(flat_grads, p=2).item()
            log_data['diagnostics/histograms/gradients'] = wandb.Histogram(
                self._sample_for_histogram(flat_grads).numpy()
            )

        if last_param is not None:
            log_data['diagnostics/histograms/last_layer_parameters'] = wandb.Histogram(
                self._sample_for_histogram(last_param).numpy()
            )
            log_data['diagnostics/last_layer_name'] = last_param_name

        if last_grad is not None:
            log_data['diagnostics/last_layer_gradient_norm_l2'] = torch.norm(last_grad, p=2).item()
            log_data['diagnostics/histograms/last_layer_gradients'] = wandb.Histogram(
                self._sample_for_histogram(last_grad).numpy()
            )
            log_data['diagnostics/last_gradient_layer_name'] = last_grad_name

        return log_data

    def _collect_logits_targets(self, model, loader, device):
        logits_list = []
        targets_list = []
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for datas in loader:
                inputs = datas['input'].to(device)
                targets = datas['target'].to(device)
                logits = model(inputs)
                loss = self.criterion(logits, targets)

                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_count += batch_size

                logits_list.append(logits.detach().cpu())
                targets_list.append(targets.detach().cpu())

        if total_count == 0:
            return None, None, None

        logits = torch.cat(logits_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        avg_loss = total_loss / total_count
        return logits, targets, avg_loss

    def _logit_norm_mean(self, logits):
        if logits is None or logits.numel() == 0:
            return None
        return torch.norm(logits, p=2, dim=1).mean().item()

    def _flatten_grads(self, grads, params):
        flat = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat.append(torch.zeros_like(param, memory_format=torch.contiguous_format).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        return torch.cat(flat, dim=0)

    def _build_fixed_balanced_ntk_subset(self, loader):
        if self._ntk_subset_inputs_cpu is not None:
            return self._ntk_subset_inputs_cpu, self._ntk_subset_inputs_cpu.size(0)
        if loader is None:
            return None, 0

        if self.ntk_max_samples <= 0:
            self.logger.info('Warning: diagnostics_ntk_max_samples must be positive. NTK diagnostics will be skipped.')
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
        for datas in loader:
            if 'target' not in datas:
                continue
            inputs_cpu = datas['input'].detach().cpu()
            targets_cpu = datas['target'].detach().cpu().long()

            for inp, tgt in zip(inputs_cpu, targets_cpu):
                cls_idx = int(tgt.item())
                if cls_idx < 0 or cls_idx >= self.num_classes:
                    continue
                if len(by_class[cls_idx]) < per_class:
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
        final_per_class = min_available

        selected_inputs = []
        selected_labels = []
        for cls_idx in range(self.num_classes):
            class_samples = by_class[cls_idx][:final_per_class]
            selected_inputs.extend(class_samples)
            selected_labels.extend([cls_idx] * len(class_samples))

        if len(selected_inputs) == 0:
            return None, 0

        self._ntk_subset_inputs_cpu = torch.stack(selected_inputs, dim=0).contiguous()
        self._ntk_subset_labels_cpu = torch.tensor(selected_labels, dtype=torch.long)
        self.logger.info(
            f'NTK diagnostics: fixed balanced subset created with {self._ntk_subset_inputs_cpu.size(0)} '
            f'samples ({final_per_class} per class).'
        )
        return self._ntk_subset_inputs_cpu, self._ntk_subset_inputs_cpu.size(0)

    def _collect_ntk_inputs(self, loader, device):
        inputs_cpu, total_samples = self._build_fixed_balanced_ntk_subset(loader)
        if inputs_cpu is None or total_samples == 0:
            return None, 0
        return inputs_cpu.to(device), total_samples

    def _ensure_cuda_context(self, device):
        if device.type != 'cuda' or self._cuda_context_initialized:
            return
        # Prime CUDA context once to avoid first-backward cuBLAS context warnings.
        _ = torch.empty(1, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        self._cuda_context_initialized = True

    def _compute_empirical_ntk_autograd(self, model, loader, device):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return None, 0

        inputs, total_samples = self._collect_ntk_inputs(loader, device)
        if inputs is None:
            return None, 0

        self._ensure_cuda_context(device)

        per_sample_jacobians = []

        for b in range(total_samples):
            sample_input = inputs[b:b + 1]
            sample_logits = model(sample_input)
            class_grads = []
            num_outputs = sample_logits.size(1)
            for c in range(num_outputs):
                grads = torch.autograd.grad(
                    sample_logits[0, c],
                    params,
                    retain_graph=(c < num_outputs - 1),
                    create_graph=False,
                    allow_unused=True,
                )
                class_grads.append(self._flatten_grads(grads, params))

            per_sample_jacobians.append(torch.cat(class_grads, dim=0).detach().cpu())

        jacobian = torch.stack(per_sample_jacobians, dim=0)
        ntk = jacobian @ jacobian.t()
        return ntk, total_samples

    def _compute_empirical_ntk_torch_func(self, model, loader, device):
        inputs, total_samples = self._collect_ntk_inputs(loader, device)
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
            # [N, O, ...] -> [N, O, P_i]
            j_flat = param_jac.reshape(param_jac.size(0), param_jac.size(1), -1)
            contribution = torch.einsum('nop,mop->nm', j_flat, j_flat)
            ntk = contribution if ntk is None else ntk + contribution

        if ntk is None:
            return None, 0

        return ntk.detach().cpu(), total_samples

    def _compute_empirical_ntk(self, model, loader, device):
        if not self._ntk_warned_default_cap and self.ntk_max_samples == 1000:
            self.logger.info('NTK diagnostics: using default diagnostics_ntk_max_samples=1000.')
            self._ntk_warned_default_cap = True

        if _HAS_TORCH_FUNC:
            try:
                return self._compute_empirical_ntk_torch_func(model, loader, device)
            except RuntimeError as err:
                err_msg = str(err).lower()
                if 'out of memory' in err_msg or 'cuda' in err_msg and 'memory' in err_msg:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.info(
                        'Warning: torch.func NTK path ran out of memory; falling back to slower autograd NTK implementation.'
                    )
                    return self._compute_empirical_ntk_autograd(model, loader, device)
                raise

        if not self._ntk_warned_fallback:
            self.logger.info('Warning: torch.func not available; falling back to slower autograd NTK implementation.')
            self._ntk_warned_fallback = True
        return self._compute_empirical_ntk_autograd(model, loader, device)

    def _get_ntk_targets_one_hot(self, num_samples):
        if self._ntk_subset_labels_cpu is None:
            return None
        if self._ntk_subset_labels_cpu.numel() != num_samples:
            self.logger.info(
                'Warning: NTK subset labels do not match the empirical kernel size; skipping label-based NTK diagnostics.'
            )
            return None
        return torch.nn.functional.one_hot(
            self._ntk_subset_labels_cpu,
            num_classes=self.num_classes,
        ).to(dtype=torch.float64)

    def _compute_ntk_spectrum_metrics(self, current_ntk, num_samples):
        log_data = {}
        if num_samples <= 0:
            return log_data

        if self.ntk_top_k <= 0:
            if not self._ntk_warned_invalid_top_k:
                self.logger.info('Warning: diagnostics_ntk_top_k must be positive. Top-k NTK diagnostics will be skipped.')
                self._ntk_warned_invalid_top_k = True
            return log_data

        kernel = current_ntk.detach().cpu().to(dtype=torch.float64)
        kernel = 0.5 * (kernel + kernel.t())

        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
        eigenvalues_desc = torch.flip(eigenvalues, dims=(0,))
        eigenvectors_desc = torch.flip(eigenvectors, dims=(1,))

        effective_k = min(int(self.ntk_top_k), int(num_samples), int(eigenvalues_desc.numel()))
        if effective_k <= 0:
            return log_data

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
            target_kernel = one_hot_targets @ one_hot_targets.t()
            kta_denom = torch.linalg.norm(kernel, ord='fro') * torch.linalg.norm(target_kernel, ord='fro')
            if kta_denom.item() > 0.0:
                kta = torch.sum(kernel * target_kernel) / kta_denom
                log_data['diagnostics/ntk_kernel_target_alignment'] = float(kta.item())

            projected_targets = one_hot_targets.t() @ top_eigenvectors
            label_alignment = nonnegative_top_eigenvalues * torch.sum(projected_targets.pow(2), dim=0)
            log_data['diagnostics/ntk_top_eigenvector_label_alignment'] = float(label_alignment.sum().item())

        if self._ntk_init_top_eigenvectors is None:
            self._ntk_init_top_eigenvectors = top_eigenvectors.clone()
            self._ntk_init_top_k = effective_k
            log_data['diagnostics/ntk_top_eigenspace_overlap_init'] = 1.0
            return log_data

        if self._ntk_init_top_eigenvectors.size(0) != top_eigenvectors.size(0):
            self.logger.info(
                'Warning: NTK eigenspace shape changed between initialization and current step; skipping eigenspace overlap logging.'
            )
            return log_data

        overlap_k = min(self._ntk_init_top_k, effective_k)
        if overlap_k <= 0:
            return log_data

        init_top_eigenvectors = self._ntk_init_top_eigenvectors[:, :overlap_k]
        current_top_eigenvectors = top_eigenvectors[:, :overlap_k]
        overlap = current_top_eigenvectors.t() @ init_top_eigenvectors
        log_data['diagnostics/ntk_top_eigenspace_overlap_init'] = float(
            overlap.norm(p='fro').pow(2).item() / overlap_k
        )

        return log_data

    def _log_ntk_relative_change(self, model, loader, device):
        log_data = {}
        if not self.ntk_enabled or loader is None:
            return log_data

        current_ntk, num_samples = self._compute_empirical_ntk(model, loader, device)
        if current_ntk is None:
            return log_data

        current_norm = torch.norm(current_ntk, p='fro').item()
        log_data['diagnostics/ntk_num_samples'] = int(num_samples)
        log_data['diagnostics/ntk_norm_fro'] = float(current_norm)
        log_data.update(self._compute_ntk_spectrum_metrics(current_ntk, num_samples))

        if self._ntk_init_kernel is None:
            self._ntk_init_kernel = current_ntk
            self._ntk_init_norm = current_norm
            self._ntk_init_num_samples = num_samples
            log_data['diagnostics/ntk_init_norm_fro'] = float(current_norm)
            log_data['diagnostics/ntk_relative_change'] = 0.0
            return log_data

        if current_ntk.shape != self._ntk_init_kernel.shape:
            self.logger.info(
                'Warning: NTK shape changed between initialization and current step; skipping relative NTK change logging.'
            )
            return log_data

        denom = max(float(self._ntk_init_norm), 1e-12)
        relative_change = torch.norm(current_ntk - self._ntk_init_kernel, p='fro').item() / denom
        log_data['diagnostics/ntk_init_norm_fro'] = float(self._ntk_init_norm)
        log_data['diagnostics/ntk_relative_change'] = float(relative_change)

        return log_data

    def _collect_layer_activations(self, model, loader, device, layer_names):
        named_modules = dict(model.named_modules())
        missing = [name for name in layer_names if name not in named_modules]
        if missing:
            return {}, missing

        activations = {name: [] for name in layer_names}
        targets_list = []
        hooks = []

        def make_hook(layer_name):
            def hook(_module, _input, output):
                if isinstance(output, (list, tuple)):
                    output = output[0]
                activations[layer_name].append(output.detach().reshape(output.size(0), -1).cpu())
            return hook

        for name in layer_names:
            hooks.append(named_modules[name].register_forward_hook(make_hook(name)))

        try:
            with torch.no_grad():
                for datas in loader:
                    inputs = datas['input'].to(device)
                    targets = datas['target'].cpu()
                    _ = model(inputs)
                    targets_list.append(targets)
        finally:
            for hook in hooks:
                hook.remove()

        if len(targets_list) == 0:
            return {}, []

        labels = torch.cat(targets_list, dim=0).numpy()
        layer_features = {}
        for name in layer_names:
            if len(activations[name]) == 0:
                continue
            layer_features[name] = (torch.cat(activations[name], dim=0).numpy(), labels)

        return layer_features, []

    def _log_layer_lr_metrics(self, model, train_loader, device, layer_names):
        log_data = {}
        if not layer_names:
            return log_data

        train_layer_features, missing = self._collect_layer_activations(model, train_loader, device, layer_names)
        for missing_name in missing:
            self.logger.info(f"Warning: diagnostics layer '{missing_name}' was not found in model.named_modules().")

        for layer_name, (train_features, train_labels) in train_layer_features.items():
            if train_features.shape[0] < 2 or np.unique(train_labels).shape[0] < 2:
                continue

            max_iter_try = max(int(self.lr_max_iter), 300)
            for attempt in range(2):
                clf = make_pipeline(
                    StandardScaler(with_mean=True, with_std=True),
                    LogisticRegression(
                        solver='lbfgs',
                        max_iter=max_iter_try,
                    ),
                )

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter('always', ConvergenceWarning)
                    clf.fit(train_features, train_labels)

                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught_warnings)
                if converged or attempt == 1:
                    if not converged:
                        self.logger.info(
                            f"Warning: LR probe for layer '{layer_name}' did not converge after max_iter={max_iter_try}."
                        )
                    break

                max_iter_try = max_iter_try * 3

            train_preds = clf.predict(train_features)
            train_acc = accuracy_score(train_labels, train_preds)
            sanitized_name = layer_name.replace('.', '_')
            log_data[f'diagnostics/layer_lr_train_acc/{sanitized_name}'] = float(train_acc)

        return log_data

    def log_diagnostics(self, model, trigger, total_step, device, fixed_train_loader, test_loader, layer_names):
        was_training = model.training
        model.eval()

        log_data = {
            'diagnostics/trigger': trigger,
            'diagnostics/total_step': total_step,
        }

        log_data.update(self._collect_param_grad_stats(model))

        fixed_train_logits = None
        fixed_train_labels = None
        fixed_train_avg_loss = None
        if fixed_train_loader is not None:
            fixed_train_logits, fixed_train_labels, fixed_train_avg_loss = self._collect_logits_targets(model, fixed_train_loader, device)

        test_logits = None
        test_labels = None
        test_avg_loss = None
        if test_loader is not None:
            test_logits, test_labels, test_avg_loss = self._collect_logits_targets(model, test_loader, device)

        if fixed_train_logits is not None and fixed_train_labels is not None:
            fixed_train_preds = torch.argmax(fixed_train_logits, dim=1)
            fixed_train_acc = (fixed_train_preds == fixed_train_labels).float().mean().item()
            log_data['diagnostics/fixed_train_accuracy'] = fixed_train_acc
            if fixed_train_avg_loss is not None:
                log_data['diagnostics/fixed_train_loss'] = float(fixed_train_avg_loss)

            fixed_train_logits_norm_mean = self._logit_norm_mean(fixed_train_logits)
            if fixed_train_logits_norm_mean is not None:
                log_data['diagnostics/fixed_train_logits_norm_l2_mean'] = float(fixed_train_logits_norm_mean)

        if test_logits is not None and test_labels is not None:
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = (test_preds == test_labels).float().mean().item()
            log_data['diagnostics/test_accuracy'] = test_acc
            if test_avg_loss is not None:
                log_data['diagnostics/test_loss'] = float(test_avg_loss)

        if fixed_train_loader is not None:
            log_data.update(self._log_layer_lr_metrics(model, fixed_train_loader, device, layer_names))

        log_data.update(self._log_ntk_relative_change(model, fixed_train_loader, device))

        self.logger.wandb_log(log_data)

        if was_training:
            model.train()
