"""Data download utilities for autism fMRI datasets.

Downloads and organizes:
- ABIDE I/II (resting-state fMRI, via nilearn)
- OpenNeuro task-based autism datasets (naturalistic stimuli)
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AbideDownloader:
    """Downloads and organizes ABIDE I/II preprocessed resting-state fMRI data.

    Uses nilearn's fetch_abide_pcp() which provides C-PAC preprocessed data
    in MNI152 space, ready for surface projection via TribeSurfaceProjector.
    """

    def __init__(self, output_dir: str | Path = "./data/abide"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_abide1(
        self,
        pipeline: str = "cpac",
        strategy: str = "filt_noglobal",
        derivatives: list[str] | None = None,
        n_subjects: int | None = None,
    ) -> pd.DataFrame:
        """Download ABIDE I preprocessed data via nilearn.

        Parameters
        ----------
        pipeline : str
            Preprocessing pipeline. One of: cpac, css, dparsf, niak.
        strategy : str
            Denoising strategy. One of: filt_global, filt_noglobal,
            nofilt_global, nofilt_noglobal.
        derivatives : list[str] or None
            Which derivatives to download. Default: ["func_preproc"]
            for preprocessed functional data.
        n_subjects : int or None
            Limit number of subjects (for testing). None = all.

        Returns
        -------
        pd.DataFrame
            Phenotypic data with file paths for each subject.
        """
        from nilearn.datasets import fetch_abide_pcp

        if derivatives is None:
            derivatives = ["func_preproc"]

        logger.info(
            "Downloading ABIDE I (pipeline=%s, strategy=%s, derivatives=%s)",
            pipeline,
            strategy,
            derivatives,
        )

        dataset = fetch_abide_pcp(
            data_dir=str(self.output_dir),
            pipeline=pipeline,
            band_pass_filtering=strategy.startswith("filt"),
            global_signal_regression="global" in strategy,
            derivatives=derivatives,
            n_subjects=n_subjects,
        )

        phenotypic = pd.DataFrame(dataset.phenotypic)
        phenotypic["func_preproc_path"] = [
            str(p) for p in dataset.func_preproc
        ]
        phenotypic["diagnosis"] = phenotypic["DX_GROUP"].map(
            {1: "ASD", 2: "TD"}
        )

        summary_path = self.output_dir / "abide1_phenotypic.csv"
        phenotypic.to_csv(summary_path, index=False)
        logger.info(
            "ABIDE I: %d subjects (%d ASD, %d TD). Phenotypic saved to %s",
            len(phenotypic),
            (phenotypic.diagnosis == "ASD").sum(),
            (phenotypic.diagnosis == "TD").sum(),
            summary_path,
        )
        return phenotypic

    def download_abide2(
        self,
        n_subjects: int | None = None,
    ) -> pd.DataFrame:
        """Download ABIDE II preprocessed data via nilearn.

        Returns
        -------
        pd.DataFrame
            Phenotypic data with file paths.
        """
        from nilearn.datasets import fetch_abide_pcp

        logger.info("Downloading ABIDE II...")

        dataset = fetch_abide_pcp(
            data_dir=str(self.output_dir),
            n_subjects=n_subjects,
        )

        phenotypic = pd.DataFrame(dataset.phenotypic)
        phenotypic["func_preproc_path"] = [
            str(p) for p in dataset.func_preproc
        ]
        phenotypic["diagnosis"] = phenotypic["DX_GROUP"].map(
            {1: "ASD", 2: "TD"}
        )

        summary_path = self.output_dir / "abide2_phenotypic.csv"
        phenotypic.to_csv(summary_path, index=False)
        logger.info(
            "ABIDE II: %d subjects (%d ASD, %d TD)",
            len(phenotypic),
            (phenotypic.diagnosis == "ASD").sum(),
            (phenotypic.diagnosis == "TD").sum(),
        )
        return phenotypic

    def get_phenotypic(self, version: int = 1) -> pd.DataFrame:
        """Load previously downloaded phenotypic data."""
        path = self.output_dir / f"abide{version}_phenotypic.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Phenotypic file not found at {path}. Run download_abide{version}() first."
            )
        return pd.read_csv(path)


class OpenNeuroAutismDownloader:
    """Downloads task-based autism fMRI datasets from OpenNeuro.

    These datasets contain naturalistic stimuli (movies, stories) that
    directly fit TRIBE v2's multimodal encoding paradigm.
    """

    def __init__(self, output_dir: str | Path = "./data/openneuro_autism"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_richardson2018(
        self,
        include: list[str] | None = None,
    ) -> Path:
        """Download Richardson et al. 2018 (ds000228).

        Children (including ASD) watching Pixar short film "Partly Cloudy".
        Naturalistic video stimulus -- direct fit for TRIBE v2.

        Parameters
        ----------
        include : list[str] or None
            Subset of files to download (glob patterns).
            None downloads the full dataset.

        Returns
        -------
        Path
            Path to the downloaded dataset root.
        """
        import openneuro

        dataset_id = "ds000228"
        target = self.output_dir / dataset_id

        logger.info("Downloading Richardson 2018 (ds000228) from OpenNeuro...")
        openneuro.download(
            dataset=dataset_id,
            target_dir=str(target),
            include=include,
        )

        logger.info("Richardson 2018 downloaded to %s", target)
        return target

    def download_byrge_kennedy2020(
        self,
        include: list[str] | None = None,
    ) -> Path:
        """Download Byrge & Kennedy 2020 (ds002345).

        ASD and TD participants watching "Despicable Me" movie.

        Returns
        -------
        Path
            Path to the downloaded dataset root.
        """
        import openneuro

        dataset_id = "ds002345"
        target = self.output_dir / dataset_id

        logger.info("Downloading Byrge & Kennedy 2020 (ds002345)...")
        openneuro.download(
            dataset=dataset_id,
            target_dir=str(target),
            include=include,
        )

        logger.info("Byrge & Kennedy 2020 downloaded to %s", target)
        return target

    def download_primas(
        self,
        include: list[str] | None = None,
    ) -> Path:
        """Download PRIMAS dataset (ds007182).

        Precision Functional Imaging in Autism.

        Returns
        -------
        Path
            Path to the downloaded dataset root.
        """
        import openneuro

        dataset_id = "ds007182"
        target = self.output_dir / dataset_id

        logger.info("Downloading PRIMAS (ds007182)...")
        openneuro.download(
            dataset=dataset_id,
            target_dir=str(target),
            include=include,
        )

        logger.info("PRIMAS downloaded to %s", target)
        return target

    def list_subjects(self, dataset_path: str | Path) -> pd.DataFrame:
        """List subjects and their metadata from a downloaded BIDS dataset.

        Parameters
        ----------
        dataset_path : str or Path
            Root of the BIDS dataset.

        Returns
        -------
        pd.DataFrame
            Subject IDs and available sessions/runs.
        """
        dataset_path = Path(dataset_path)
        participants_tsv = dataset_path / "participants.tsv"
        if participants_tsv.exists():
            return pd.read_csv(participants_tsv, sep="\t")

        subjects = []
        for sub_dir in sorted(dataset_path.glob("sub-*")):
            if sub_dir.is_dir():
                subjects.append({"participant_id": sub_dir.name})
        return pd.DataFrame(subjects)
