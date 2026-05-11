import torch
import wandb


class ParamGradDiagnostics:
    def __init__(self, wandb_param_norms, wandb_grad_norms, histogram_max_points):
        self.wandb_param_norms = bool(wandb_param_norms)
        self.wandb_grad_norms = bool(wandb_grad_norms)
        self.histogram_max_points = int(histogram_max_points)
        self.enabled = self.wandb_param_norms or self.wandb_grad_norms

    def _sample_for_histogram(self, tensor):
        flat = tensor.detach().float().reshape(-1).cpu()
        if flat.numel() <= self.histogram_max_points:
            return flat
        sample_idx = torch.randperm(flat.numel())[:self.histogram_max_points]
        return flat[sample_idx]

    def log_metrics(self, model):
        if not self.enabled:
            return {}

        log_data = {}
        param_chunks = []
        grad_chunks = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_cpu = param.detach().float().reshape(-1).cpu()
            if self.wandb_param_norms and param_cpu.numel() > 0:
                param_chunks.append(param_cpu)

            if self.wandb_grad_norms and param.grad is not None:
                grad_cpu = param.grad.detach().float().reshape(-1).cpu()
                if grad_cpu.numel() > 0:
                    grad_chunks.append(grad_cpu)

        if self.wandb_param_norms and param_chunks:
            flat_params = torch.cat(param_chunks, dim=0)
            log_data['diagnostics/parameter_norm_l2_global'] = torch.norm(flat_params, p=2).item()
            log_data['diagnostics/histograms/parameters'] = wandb.Histogram(
                self._sample_for_histogram(flat_params).numpy()
            )

        if self.wandb_grad_norms and grad_chunks:
            flat_grads = torch.cat(grad_chunks, dim=0)
            log_data['diagnostics/gradient_norm_l2_global'] = torch.norm(flat_grads, p=2).item()
            log_data['diagnostics/histograms/gradients'] = wandb.Histogram(
                self._sample_for_histogram(flat_grads).numpy()
            )

        return log_data