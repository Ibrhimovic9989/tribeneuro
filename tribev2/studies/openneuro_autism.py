"""Task-based autism fMRI studies from OpenNeuro.

These datasets contain naturalistic movie/story stimuli with both ASD and
typically-developing (TD) participants, making them directly compatible with
TRIBE v2's multimodal stimulus-to-brain encoding paradigm.

Datasets:
    Richardson2018 (ds000228):
        Children watching Pixar short "Partly Cloudy" (~5 min).
        36 ASD, 44 TD participants.
        TR = 2.0s, MNI152NLin2009cAsym space.
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
from neuralset.events import study

logger = logging.getLogger(__name__)


def _find_bold_files(
    bids_root: Path, subject: str, task: str, space: str = "MNI152NLin2009cAsym"
) -> list[Path]:
    """Find preprocessed BOLD NIfTI files for a subject/task in fMRIPrep output."""
    sub_dir = bids_root / subject
    bold_files = []
    for ses_dir in sorted(sub_dir.iterdir()):
        if not ses_dir.name.startswith("ses-") and ses_dir.name != "func":
            continue
        func_dir = ses_dir / "func" if ses_dir.name.startswith("ses-") else ses_dir
        if not func_dir.exists():
            # Top-level func dir (no sessions)
            func_dir = sub_dir / "func"
            if not func_dir.exists():
                continue
        pattern = f"*task-{task}*space-{space}*_bold.nii.gz"
        bold_files.extend(sorted(func_dir.glob(pattern)))
    return bold_files


class Richardson2018(study.Study):
    """Children (ASD + TD) watching Pixar's "Partly Cloudy" short film.

    OpenNeuro dataset ds000228. Naturalistic video stimulus that directly
    fits TRIBE v2's video+audio encoding pipeline.

    References
    ----------
    Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018).
    Development of the social brain from age three to twelve years.
    Nature Communications, 9(1), 1027.
    """

    device: tp.ClassVar[str] = "Fmri"
    licence: tp.ClassVar[str] = "CC0"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds000228"
    TR_FMRI_S: tp.ClassVar[float] = 2.0
    _FREQUENCY: tp.ClassVar[float] = 1 / 2.0
    _TASK: tp.ClassVar[str] = "pixar"

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=155,
        num_subjects=155,
        num_events_in_query=200,
        event_types_in_query={"Fmri", "Video"},
        data_shape=(65, 77, 49, 168),
        frequency=0.5,
        fmri_spaces=("MNI152NLin2009cAsym",),
    )

    def _download(self) -> None:
        raise NotImplementedError(
            "Use OpenNeuroAutismDownloader.download_richardson2018() to download."
        )

    def _get_participants(self) -> pd.DataFrame:
        """Load participants.tsv with diagnosis info."""
        tsv = self.path / "download" / "ds000228" / "participants.tsv"
        if tsv.exists():
            return pd.read_csv(tsv, sep="\t")
        raise FileNotFoundError(f"participants.tsv not found at {tsv}")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        base = self.path / "download" / "ds000228"
        if not base.exists():
            raise RuntimeError(f"Dataset not found at {base}")

        participants = self._get_participants()
        diag_map = dict(
            zip(participants["participant_id"], participants.get("diagnosis", "unknown"))
        )

        for sub_dir in sorted(base.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            subject = sub_dir.name
            func_dir = sub_dir / "func"
            if not func_dir.exists():
                continue

            # Look for the pixar task BOLD file
            bold_files = list(func_dir.glob(f"*task-{self._TASK}*_bold.nii.gz"))
            if not bold_files:
                continue

            diagnosis = diag_map.get(subject, "unknown")
            for bold_file in bold_files:
                yield dict(
                    subject=subject,
                    task=self._TASK,
                    diagnosis=diagnosis,
                    bold_file=str(bold_file),
                )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        import nibabel

        base = self.path / "download" / "ds000228"
        bold_path = Path(timeline["bold_file"])

        nii: tp.Any = nibabel.load(bold_path, mmap=True)
        freq = self._FREQUENCY
        dur = nii.shape[-1] / freq

        events = []

        # fMRI event
        events.append(
            dict(
                type="Fmri",
                start=0,
                filepath=str(bold_path),
                frequency=freq,
                duration=dur,
            )
        )

        # Video stimulus -- Pixar's "Partly Cloudy"
        video_dir = base / "stimuli"
        video_candidates = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        if video_candidates:
            video_file = video_candidates[0]
            events.append(
                dict(type="Video", start=0, filepath=str(video_file))
            )

        df = pd.DataFrame(events)
        df["diagnosis"] = timeline["diagnosis"]
        df["split"] = "train"  # Will be overridden by SplitEvents
        return df
