import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from methods.method_utils.diagnostics_context import DiagnosticsRunContext


class ProbeDiagnostics:
    def __init__(self, logger, context: DiagnosticsRunContext, layer_names, lr_max_iter):
        self.logger = logger
        self.train_loader = context.fixed_train_loader
        self.layer_names = list(layer_names)
        self.lr_max_iter = int(lr_max_iter)
        self.enabled = bool(self.layer_names)

    @staticmethod
    def _resolve_requested_layers(named_modules, requested_layer_names):
        resolved = {}
        missing = []
        ambiguous = {}

        for requested_name in requested_layer_names:
            if requested_name in named_modules:
                resolved[requested_name] = requested_name
                continue

            # Accept unqualified layer names for wrapped models, e.g.
            # "linear1" -> "net.linear1" or "module.net.linear1".
            matches = [
                module_name
                for module_name in named_modules.keys()
                if module_name.endswith(f'.{requested_name}')
            ]

            if len(matches) == 1:
                resolved[requested_name] = matches[0]
            elif len(matches) > 1:
                ambiguous[requested_name] = matches
            else:
                missing.append(requested_name)

        return resolved, missing, ambiguous

    def _collect_layer_activations(self, model, device):
        named_modules = dict(model.named_modules())
        resolved, missing, ambiguous = self._resolve_requested_layers(named_modules, self.layer_names)
        if not resolved:
            return {}, missing, ambiguous

        activations = {name: [] for name in resolved}
        targets_list = []
        hooks = []

        def make_hook(layer_name):
            def hook(_module, _input, output):
                if isinstance(output, (list, tuple)):
                    output = output[0]
                activations[layer_name].append(output.detach().reshape(output.size(0), -1).cpu())

            return hook

        for requested_name, module_name in resolved.items():
            hooks.append(named_modules[module_name].register_forward_hook(make_hook(requested_name)))

        try:
            with torch.no_grad():
                for datas in self.train_loader:
                    inputs = datas['input'].to(device)
                    targets = datas['target'].cpu()
                    _ = model(inputs)
                    targets_list.append(targets)
        finally:
            for hook in hooks:
                hook.remove()

        labels = torch.cat(targets_list, dim=0).numpy()
        layer_features = {}
        for name in resolved:
            if not activations[name]:
                continue
            layer_features[name] = (torch.cat(activations[name], dim=0).numpy(), labels)

        return layer_features, missing, ambiguous

    def log_metrics(self, model, device):
        if not self.enabled:
            return {}

        log_data = {}
        train_layer_features, missing, ambiguous = self._collect_layer_activations(model, device)
        for missing_name in missing:
            self.logger.info(f"Warning: diagnostics layer '{missing_name}' was not found in model.named_modules().")
        for ambiguous_name, matches in ambiguous.items():
            joined_matches = ', '.join(matches)
            self.logger.info(
                f"Warning: diagnostics layer '{ambiguous_name}' matched multiple modules ({joined_matches}). "
                "Use a fully qualified layer name to disambiguate."
            )

        for layer_name, (train_features, train_labels) in train_layer_features.items():
            if train_features.shape[0] < 2 or np.unique(train_labels).shape[0] < 2:
                continue

            max_iter_try = max(self.lr_max_iter, 300)
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

                max_iter_try *= 3

            train_preds = clf.predict(train_features)
            sanitized_name = layer_name.replace('.', '_')
            log_data[f'diagnostics/layer_lr_train_acc/{sanitized_name}'] = float(
                accuracy_score(train_labels, train_preds)
            )

        return log_data