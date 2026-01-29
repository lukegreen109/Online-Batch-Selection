import yaml
import copy
import os
import itertools
# Load the base config
with open('cfg/cifar10.yaml', 'r') as f:
    base_cfg = yaml.safe_load(f)
# ensure sections exist
base_cfg.setdefault('method_opt', {})
# Define your ablation settings (list of dicts with new method_opt values)
seed = [
    {'seed': 13},
    {'seed': 14},
    {'seed': 15},
    {'seed': 16},
]
noise = [
    {"noise_percent": 0.01},
    {"noise_percent": 0.1},
    {"noise_percent": 0.05},
    {"noise_percent": 0.2},
]
out_dir = 'cfg'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "Noise_test"), exist_ok=True)
# optional short helper to make safe filename parts
def safe_part(d):
    parts = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            v_str = "-".join(map(str, v))
        else:
            v_str = str(v)
        parts.append(f"{k}-{v_str}")
    return "__".join(parts).replace(" ", "_").replace("/", "_")
# Create one config per (alpha, optimizer) combination
for seed_cfg, noise_cfg in itertools.product(seed,noise):
    cfg = copy.deepcopy(base_cfg)
    # update method_opt with alpha settings
    cfg.setdefault('method_opt', {})
    cfg['dataset'].update(noise_cfg)
    cfg.update(seed_cfg)
    # place optimizer fields under training_opt.optim_params
    noise_percent = noise_cfg['noise_percent']
    seed_part = safe_part(seed_cfg)
    out_path = os.path.join(out_dir, f"Noise_test/noise_seed{seed_part}_epochs-{noise_percent}.yaml")
    with open(out_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {out_path}")