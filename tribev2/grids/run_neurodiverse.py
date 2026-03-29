"""Grid configuration for fine-tuning TRIBE v2 on neurodiverse (autism) fMRI data.

Uses TRIBE v2's built-in transfer learning mechanism:
- resize_subject_layer=True: Resizes the subject-specific output layer
  for new subjects (ASD + TD participants)
- freeze_backbone=True: Freezes the shared Transformer backbone and
  feature extractors, only training the subject-specific layers

This allows the model to learn how neurodiverse brains respond to
the same naturalistic stimuli that TRIBE v2 was trained on.
"""

import copy

from tribev2.grids.defaults import DEFAULTS

GRID_NAME = "neurodiverse"


def get_neurodiverse_config(
    checkpoint_path: str = "facebook/tribev2",
    study_names: list[str] | None = None,
    freeze_backbone: bool = True,
    lr: float = 1e-5,
    n_epochs: int = 30,
    batch_size: int = 4,
    patience: int = 10,
):
    """Build configuration for neurodiverse fine-tuning.

    Parameters
    ----------
    checkpoint_path : str
        Path to pretrained TRIBE v2 checkpoint or HuggingFace repo.
    study_names : list[str] or None
        Which autism studies to train on.
        Default: ["Richardson2018"]
    freeze_backbone : bool
        If True, only train the subject-specific layers.
    lr : float
        Learning rate (lower than pretraining for fine-tuning).
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size (smaller due to fewer subjects).
    patience : int
        Early stopping patience.

    Returns
    -------
    dict
        Configuration dictionary for TribeExperiment.
    """
    if study_names is None:
        study_names = ["Richardson2018"]

    config = copy.deepcopy(DEFAULTS)

    # Transfer learning settings
    config["checkpoint_path"] = checkpoint_path
    config["load_checkpoint"] = True
    config["resize_subject_layer"] = True
    config["freeze_backbone"] = freeze_backbone

    # Study configuration
    config["data.study.names"] = study_names

    # Training hyperparameters (tuned for fine-tuning)
    config["n_epochs"] = n_epochs
    config["patience"] = patience
    config["data.batch_size"] = batch_size
    config["optim.optimizer.lr"] = lr
    config["optim.scheduler.max_lr"] = lr

    # Smaller segments for movie stimulus datasets
    config["data.duration_trs"] = 40
    config["data.overlap_trs_train"] = 10

    # W&B logging
    config["wandb_config.group"] = GRID_NAME
    config["wandb_config.project"] = "tribev2-neurodiverse"

    return config


# ---- Grid search configurations ----

# Single study: Richardson 2018 (Pixar video, ASD children)
RICHARDSON_CONFIG = get_neurodiverse_config(
    study_names=["Richardson2018"],
    freeze_backbone=True,
    lr=1e-5,
)

# Average subject mode: learn one "average ASD" and one "average TD"
# subject layer instead of per-subject layers (better with few subjects)
RICHARDSON_AVG_CONFIG = get_neurodiverse_config(
    study_names=["Richardson2018"],
    freeze_backbone=True,
    lr=1e-5,
)
RICHARDSON_AVG_CONFIG["average_subjects"] = True

# Unfrozen backbone: also fine-tune the Transformer
# (only if enough data -- risk of overfitting with small datasets)
RICHARDSON_FULL_CONFIG = get_neurodiverse_config(
    study_names=["Richardson2018"],
    freeze_backbone=False,
    lr=1e-6,
    n_epochs=15,
)


if __name__ == "__main__":
    from tribev2.main import TribeExperiment

    # Default: frozen backbone, Richardson 2018
    xp = TribeExperiment(**RICHARDSON_CONFIG)
    xp.run()
