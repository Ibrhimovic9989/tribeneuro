"""
Neurodiverse Brain Model - Demo Script
=======================================

This script demonstrates the full pipeline:
1. Download autism fMRI data (ABIDE + OpenNeuro)
2. Fine-tune TRIBE v2 on neurodiverse brain data
3. Compare neurotypical vs neurodiverse predictions
4. Visualize divergence on the brain surface
5. Generate a sensory processing profile

Run this script step by step or use it as a reference.

Requirements:
    pip install -e ".[plotting]"
    pip install nilearn openneuro-py statsmodels scikit-learn
"""

# %% [1] Setup and Imports
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
CACHE_DIR = Path("./cache")
REPORT_DIR = Path("./neurodiverse_report")

# %% [2] Download ABIDE Data (Resting-State)
# This downloads preprocessed resting-state fMRI for ASD and TD subjects

from tribev2.neurodiverse.download import AbideDownloader

abide = AbideDownloader(output_dir=DATA_DIR / "abide")

# Start small: download 20 subjects for testing
# Set n_subjects=None for full dataset (1100+ subjects)
phenotypic = abide.download_abide1(n_subjects=20)

print(f"\nDownloaded {len(phenotypic)} subjects:")
print(phenotypic.groupby("diagnosis").size())
print(f"\nSites: {phenotypic['SITE_ID'].nunique()}")
print(f"Age range: {phenotypic['AGE_AT_SCAN'].min():.1f} - {phenotypic['AGE_AT_SCAN'].max():.1f}")

# %% [3] Download Task-Based Autism Data (OpenNeuro)
# Richardson 2018: Children watching Pixar's "Partly Cloudy"
# This is the dataset we'll use for fine-tuning TRIBE v2

from tribev2.neurodiverse.download import OpenNeuroAutismDownloader

openneuro = OpenNeuroAutismDownloader(output_dir=DATA_DIR / "openneuro_autism")

# Download the Richardson 2018 dataset
# This may take a while (~several GB)
richardson_path = openneuro.download_richardson2018()
participants = openneuro.list_subjects(richardson_path)
print(f"\nRichardson 2018: {len(participants)} subjects")
print(participants.head())

# %% [4] Resting-State Analysis (ABIDE)
# Project ABIDE data to the same brain surface as TRIBE v2
# and compute functional connectivity differences

from tribev2.neurodiverse.resting_state import RestingStateAnalyzer

analyzer = RestingStateAnalyzer(mesh="fsaverage5")

# Project and compute connectivity for ASD vs TD
# Start with a small batch for testing
connectivity = analyzer.batch_project_and_connect(
    phenotypic,
    func_col="func_preproc_path",
    max_subjects=5,  # Set to None for all subjects
)

print(f"\nConnectivity computed:")
print(f"  ASD: {len(connectivity['ASD'])} subjects")
print(f"  TD: {len(connectivity['TD'])} subjects")

# %% [5] Compare ASD vs TD Connectivity
# Statistical comparison of resting-state functional connectivity

if len(connectivity["ASD"]) >= 2 and len(connectivity["TD"]) >= 2:
    results = analyzer.compare_groups(
        connectivity["ASD"],
        connectivity["TD"],
    )

    print("\nTop significant connections (ASD vs TD):")
    sig = results["top_connections"][results["top_connections"]["significant"]]
    if len(sig) > 0:
        print(sig.head(10).to_string(index=False))
    else:
        print("No significant connections after FDR correction")
        print("\nTop uncorrected differences:")
        print(results["top_connections"].head(10).to_string(index=False))

    # Convert to surface map for visualization
    diff_surface = analyzer.connectivity_to_surface(
        results["difference"], metric="degree"
    )
    np.save(REPORT_DIR / "connectivity_difference.npy", diff_surface)

# %% [6] Load Pretrained TRIBE v2 (Neurotypical Baseline)
from tribev2 import TribeModel

nt_model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=str(CACHE_DIR),
)
print("\nNeurotypical model loaded")

# %% [7] Run Prediction on a Sample Stimulus
# Use a video to see how the neurotypical brain model responds

sample_video = DATA_DIR / "sample_stimulus.mp4"

if sample_video.exists():
    events = nt_model.get_events_dataframe(video_path=str(sample_video))
    nt_preds, segments = nt_model.predict(events)
    print(f"\nPrediction shape: {nt_preds.shape}")
    print(f"  Timesteps: {nt_preds.shape[0]}")
    print(f"  Brain vertices: {nt_preds.shape[1]}")
else:
    print(f"\nNo sample video found at {sample_video}")
    print("To test predictions, place a video file there or use:")
    print('  events = nt_model.get_events_dataframe(video_path="path/to/video.mp4")')

# %% [8] Fine-Tune on Neurodiverse Data
# This step trains the subject-specific layers on autism fMRI data
# while keeping the backbone frozen
#
# NOTE: This requires:
# 1. Downloaded Richardson 2018 data (step 3)
# 2. A GPU with sufficient VRAM
# 3. The pretrained TRIBE v2 checkpoint
#
# For a quick test, use the test_run configuration:

print("""
To fine-tune the model, run:

    python -m tribev2.grids.run_neurodiverse

Or from Python:

    from tribev2.grids.run_neurodiverse import RICHARDSON_CONFIG
    from tribev2.main import TribeExperiment

    xp = TribeExperiment(**RICHARDSON_CONFIG)
    xp.run()

This will:
- Load the pretrained TRIBE v2 backbone
- Resize the subject layer for new ASD/TD subjects
- Freeze the backbone (only train subject layers)
- Train for 30 epochs with early stopping
- Save checkpoints to the output directory
""")

# %% [9] Compare Models (after fine-tuning)
# Once you have a fine-tuned neurodiverse model, compare predictions

from tribev2.neurodiverse.comparison import NeurodiverseComparison

# Example: loading both models
# nd_model = TribeModel.from_pretrained("./output/neurodiverse/best.ckpt")
# comparison = NeurodiverseComparison(
#     neurotypical_model=nt_model,
#     neurodiverse_model=nd_model,
# )
#
# # Run same stimulus through both
# nt_preds, nd_preds = comparison.predict_both(events)
#
# # Find where predictions diverge most
# divergence = comparison.compute_divergence_map(nt_preds, nd_preds)
# top_rois = comparison.get_top_divergent_rois(divergence)
# print(top_rois)
#
# # Generate sensory profile
# profile = comparison.sensory_profile(nt_preds, nd_preds)
# print(profile["profile_summary"])
#
# # Full report
# comparison.generate_report(events, output_dir="./neurodiverse_report")

# %% [10] Visualize Results
# Use TRIBE v2's built-in brain visualization

print("""
Visualization example (requires plotting dependencies):

    from tribev2.plotting import PlotBrain

    plotter = PlotBrain(mesh="fsaverage5")

    # Visualize neurotypical prediction
    fig = plotter.plot_timesteps(
        nt_preds[:15],
        segments=segments[:15],
        cmap="fire",
        norm_percentile=99,
        vmin=0.5,
        show_stimuli=True,
    )

    # Visualize divergence map
    fig = plotter.plot_surf(
        divergence,
        cmap="hot",
        title="NT vs ND Divergence",
    )

    # Visualize connectivity difference
    fig = plotter.plot_surf(
        diff_surface,
        cmap="coolwarm",
        symmetric_cbar=True,
        title="ASD vs TD Connectivity Difference",
    )
""")

# %% [11] Summary
print("""
=== Neurodiverse Brain Model - Architecture Summary ===

TIER A - Representation Extraction (no fine-tuning):
  Use TRIBE v2 as-is to predict brain responses to stimuli.
  Compare predictions against actual neurodiverse fMRI recordings.

TIER B - Subject Layer Fine-tuning:
  Keep TRIBE v2's backbone (LLaMA + V-JEPA2 + Wav2Vec + Transformer).
  Only retrain the subject-specific output layers on autism fMRI.
  Result: a model that predicts how the AUTISTIC brain responds to stimuli.

TIER C - Resting-State Analysis:
  Project ABIDE resting-state fMRI to TRIBE v2's brain surface.
  Compare functional connectivity between ASD and TD groups.
  Map connectivity differences onto the same surface as TRIBE v2 predictions.

Key files:
  tribev2/neurodiverse/download.py     - Data download utilities
  tribev2/neurodiverse/resting_state.py - FC analysis on fsaverage5
  tribev2/neurodiverse/comparison.py    - NT vs ND comparison tools
  tribev2/studies/abide.py              - ABIDE study definition
  tribev2/studies/openneuro_autism.py   - OpenNeuro autism studies
  tribev2/grids/run_neurodiverse.py     - Fine-tuning configuration
""")
