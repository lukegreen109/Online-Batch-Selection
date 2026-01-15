<h1 align="center">Online Batch Selection Methods for Training Acceleration</h1>

## Getting Started

### Install Project Dependencies

`Online-Batch-Selection` is managed via the `uv` package manager ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/)). To install the dependencies, simply run `uv sync` from the root directory of the repository after cloning.

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting and run code quality checks. Any issues must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.


## Data Preparation
For CIFAR datasets, the data will be automatically downloaded by the code.

For Tiny-ImageNet, please download the dataset from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it to the `_TINYIMAGENET` folder. Then, run the following command to prepare the data:
```bash
cd _TINYIMAGENET
python val_folder.py
```

## Running
```bash
CUDA_VISIBLE_DEVICES=0 uv run main.py --cfg cfg/cifar10.yaml --wandb_not_upload
```
The `--wandb_not_upload` is optional and is used to keep wandb log files locally without uploading them to the wandb cloud. CUDA_VISIBLE_DEVICES specifies which GPU device to use. Multiple GPU devices are supported (i.e. CUDA_VISIBLE_DEVICES="0,2").

## Repository Structure

The repository is organized as follows:

### Core Components

- **`methods/selectionmethod.py`** - Parent class containing the main training loop and core batch selection logic. All selection methods inherit from this base class. All selection methods are implemented by overwriting the `self.beforebatch()` method.

- **`methods/`** - Implementation of various batch selection methods. Each method extends the base `SelectionMethod` class. The list of methods include Full, Uniform, DivBS, RhoLoss, Bayesian, MaxLoss, GradNorm, GradNormIS.

- **`cfg/`** – Configuration files for different datasets and experiments (e.g., `cifar10.yaml`).  
  You can specify which batch selection methods to run by listing them in the config file, for example:

  ```yaml
  methods:
    - RhoLoss
    - DivBS
    - Bayesian
    - Full
  ```

  > **Note:** `RhoLoss` and `Bayesian` require additional hyperparameters.

- **`main.py`** - Entry point for running experiments. Handles argument parsing and experiment initialization.

### Data

- **`_TINYIMAGENET/`** - Tiny-ImageNet dataset directory (see [Data Preparation](#data-preparation)).
- **`_CIFAR/`** - CIFAR datasets (automatically downloaded).

### Experiments

- **`exp/`** - Experiment results and logs (git-ignored).
- **`wandb/`** - Weights & Biases logging directory (git-ignored).

### Key Files to Review

1. Start with `selectionmethod.py` to understand the training loop architecture
2. Explore `methods/` to see specific batch selection implementations
3. Check `cfg/` for experiment configuration options

## Development

### Managing Dependencies

To add a new dependency to the project, run `uv add <package-name>`. This will install the dependency into uv's managed .venv and automatically update the `pyproject.toml` file and the `uv.lock` file, ensuring that the dependency is available for all users of the project who run `uv sync`.

To remove a dependency, run `uv remove <package-name>`. This will perform the reverse of `uv add` (including updating the `pyproject.toml` and `uv.lock` files).

See [uv's documentation](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) for more details.
