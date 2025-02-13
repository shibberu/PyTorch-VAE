(Still needs further work. Optimizer code still needs to be cleaned up. Also, data loader taking 25% of epoch time, with 8 GPUs idling most of the time.)

# PyTorch-VAE

[![Python](https://img.shields.io/badge/Python-%3E=3.8-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E=1.11-brightgreen.svg)](https://pytorch.org/)  
[![Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.0%2B-orange.svg)](https://lightning.ai/)  
[![License](https://img.shields.io/badge/license-Apache2.0-blue.svg)](LICENSE.md)

A collection of Variational Autoencoders (VAEs) implemented in PyTorch – now updated to use PyTorch Lightning 2.0. This repository focuses on reproducibility and provides working examples of several VAE models (e.g. Vanilla VAE, Beta-VAE, Conditional VAE, etc.) trained on the CelebA dataset.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Additional Notes](#additional-notes)

---

## Introduction

This project provides multiple implementations of Variational Autoencoders (VAEs) in PyTorch with a focus on reproducible research. The updated code base now uses [PyTorch Lightning 2.0](https://lightning.ai/docs/pytorch/stable/upgrade/from_1_5.html) to simplify training loops, logging, and multi-GPU support.

In this update, key modifications include:
- **Lightning 2.0 Migration:**  
  - Updated Trainer API (e.g. using `devices`/`max_epochs` instead of legacy keys).
  - Renamed lifecycle hooks (e.g. `on_validation_end` is now `on_validation_epoch_end`).
  - Import location changes (e.g. `seed_everything` is now imported from the top-level package).
- **Manual Optimization for Multiple Optimizers:**  
  If your experiment uses more than one optimizer, automatic optimization is disabled (i.e. `self.automatic_optimization = False`) and optimizer steps are handled manually in the `training_step`.

---

## Features

- **Multiple VAE Implementations:**  
  Experiment with several VAE models such as Vanilla VAE, Beta-VAE, Conditional VAE, and more.
- **Reproducibility:**  
  All models are trained on the CelebA dataset, ensuring a common benchmark for comparison.
- **PyTorch Lightning 2.0:**  
  Leverages the latest improvements in training efficiency, distributed strategies (e.g. DDPStrategy), and flexible logging.
- **Flexible Configuration:**  
  Configure model parameters, training hyperparameters, and logging options via YAML configuration files.

---

## Requirements

- **Python:** ≥ 3.8  
- **PyTorch:** ≥ 1.11  
- **PyTorch Lightning:** 2.0 or higher  
- **TorchVision:** Latest version recommended  
- **CUDA:** Optional – for GPU training  

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AntixK/PyTorch-VAE.git
   cd PyTorch-VAE
   ```

2. **Install Dependencies:**
Install the required packages using pip:
``` bash
pip install -r requirements.txt
```
Ensure that your environment has PyTorch Lightning 2.0 installed. (If needed, upgrade using pip install -U pytorch-lightning.)

## Configuration
The repository uses YAML configuration files to set up experiments. A typical config file (found in the configs/ directory) looks like:
```yaml
model_params:
  name: "VanillaVAE"      # Name of the VAE model (must match a key in your models dict)
  in_channels: 3
  latent_dim: 128
  # Add other model-specific parameters here

data_params:
  data_path: "/path/to/celeba/dataset"
  train_batch_size: 64
  val_batch_size: 64
  patch_size: [64, 64]
  num_workers: 4

exp_params:
  manual_seed: 1265
  LR: 0.001
  weight_decay: 1e-5
  kld_weight: 0.00025
  # Additional experiment parameters if needed

trainer_params:
  devices: 1            # In PL 2.0, use 'devices' instead of 'gpus'
  accelerator: "gpu"    # Or "cpu" if running on CPU
  max_epochs: 50        # Renamed from 'max_nb_epochs' in earlier versions
  gradient_clip_val: 1.5

logging_params:
  save_dir: "logs/"
  name: "VAE_Experiment"
```
Note: If you previously used keys like gpus or max_nb_epochs, update them to devices and max_epochs respectively as required by Lightning 2.0.

## Usage
Train a VAE model using the provided configuration file:
```bash
python run.py -c configs/your_config.yaml
```
This command:

    Reads the configuration file.
    Sets up logging with TensorBoard.
    Instantiates the model and experiment module.
    Creates a data module (LightningDataModule).
    Initializes the PyTorch Lightning Trainer with the updated strategy (using DDPStrategy if needed) and any specified callbacks.
    Starts the training process.

During training, if your model uses multiple optimizers, the code is set to disable automatic optimization (i.e. self.automatic_optimization = False) and manually handle optimizer steps.

## Code Structure
    dataset.py
    Contains the definition of the VAEDataset (a subclass of LightningDataModule) and custom dataset classes (e.g. MyCelebA, OxfordPets).

    experiment.py
    Defines the VAEXperiment class which subclasses pl.LightningModule. It implements the training and validation steps, loss logging, and sample image generation. Notice the updated hook name (on_validation_epoch_end) and the changes for manual optimization if using multiple optimizers.

    run.py
    The main entry point. It loads the configuration, sets up logging, seeds for reproducibility (with the new import), initializes the model and data module, and runs the training loop using the updated Trainer configuration (e.g. using DDPStrategy).

    utils.py
    Contains helper functions and decorators. The previous decorator for data loaders has been simplified to just return the function (since pl.data_loader has been removed in Lightning 2.0).

    configs/
    Contains YAML configuration files for different VAE experiments.

    models/
    Contains implementations of the various VAE models.

## Results
After training, sample reconstruction and generation images are saved to directories under your logger’s log directory (e.g., logs/VAE_Experiment/Samples and logs/VAE_Experiment/Reconstructions). You can launch TensorBoard to visualize training metrics and images:
```bash
tensorboard --logdir logs/VAE_Experiment
```

## Contributing
Contributions are welcome! If you would like to add new VAE variants or further improve the Lightning 2.0 compatibility, please open a pull request or an issue. Before contributing, make sure your changes adhere to the project’s style guidelines.

## License
This project is licensed under the Apache License 2.0. See the LICENSE.md file for details.

## Citation
If you use this code in your research, please cite it as:
```bibtex
@misc{Subramanian2020,
  author = {Subramanian, A.K},
  title = {PyTorch-VAE},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
}
```
## Additional Notes

    Lightning 2.0 Migration:
    For more details on changes in Lightning 2.0, please refer to the official upgrade guide.

    Manual Optimization:
    If your experiment uses more than one optimizer, remember that you must disable automatic optimization (by setting self.automatic_optimization = False in your LightningModule) and manually call optimizer steps within your training_step.

    Troubleshooting:
    If you encounter import errors or other issues related to Lightning 2.0 (such as changes in seed utility imports), check the updated import paths (e.g., from pytorch_lightning import seed_everything).

---

Enjoy experimenting with VAEs!
