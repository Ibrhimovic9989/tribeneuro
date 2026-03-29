"""Neurotypical vs Neurodiverse brain model comparison tools.

Compares predictions from the pretrained TRIBE v2 (neurotypical baseline)
against a fine-tuned neurodiverse model, identifying brain regions where
processing diverges most.
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NeurodiverseComparison:
    """Compare neurotypical and neurodiverse brain model predictions.

    Takes two TribeModel instances (pretrained baseline and fine-tuned
    neurodiverse variant) and analyzes how their predictions diverge
    for the same stimulus input.

    Parameters
    ----------
    neurotypical_model : TribeModel
        The pretrained TRIBE v2 model (baseline).
    neurodiverse_model : TribeModel
        Fine-tuned model on autism/neurodiverse fMRI data.
    mesh : str
        Brain surface mesh resolution.
    """

    def __init__(
        self,
        neurotypical_model=None,
        neurodiverse_model=None,
        mesh: str = "fsaverage5",
    ):
        self.nt_model = neurotypical_model
        self.nd_model = neurodiverse_model
        self.mesh = mesh

    def predict_both(
        self, events: pd.DataFrame, verbose: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the same stimulus through both models.

        Parameters
        ----------
        events : pd.DataFrame
            Events DataFrame (from get_events_dataframe).
        verbose : bool
            Show progress bars.

        Returns
        -------
        nt_preds : np.ndarray
            Neurotypical predictions (n_timesteps, n_vertices).
        nd_preds : np.ndarray
            Neurodiverse predictions (n_timesteps, n_vertices).
        """
        logger.info("Running neurotypical model...")
        nt_preds, nt_segments = self.nt_model.predict(events, verbose=verbose)

        logger.info("Running neurodiverse model...")
        nd_preds, nd_segments = self.nd_model.predict(events, verbose=verbose)

        # Align to same number of timesteps
        min_t = min(nt_preds.shape[0], nd_preds.shape[0])
        return nt_preds[:min_t], nd_preds[:min_t]

    def compute_divergence_map(
        self,
        nt_preds: np.ndarray,
        nd_preds: np.ndarray,
        method: str = "correlation",
    ) -> np.ndarray:
        """Compute per-vertex divergence between the two models' predictions.

        Parameters
        ----------
        nt_preds : np.ndarray
            Neurotypical predictions (n_timesteps, n_vertices).
        nd_preds : np.ndarray
            Neurodiverse predictions (n_timesteps, n_vertices).
        method : str
            - "mse": Mean squared error per vertex across time.
            - "correlation": 1 - Pearson correlation per vertex.
            - "absolute": Mean absolute difference per vertex.

        Returns
        -------
        np.ndarray
            Divergence values per vertex (n_vertices,).
            Higher = more different between NT and ND.
        """
        if method == "mse":
            divergence = np.mean((nt_preds - nd_preds) ** 2, axis=0)
        elif method == "correlation":
            n_vertices = nt_preds.shape[1]
            divergence = np.zeros(n_vertices)
            for v in range(n_vertices):
                nt_v = nt_preds[:, v]
                nd_v = nd_preds[:, v]
                if np.std(nt_v) > 1e-8 and np.std(nd_v) > 1e-8:
                    r = np.corrcoef(nt_v, nd_v)[0, 1]
                    divergence[v] = 1.0 - r
                else:
                    divergence[v] = 1.0
        elif method == "absolute":
            divergence = np.mean(np.abs(nt_preds - nd_preds), axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        return divergence

    def get_top_divergent_rois(
        self,
        divergence_map: np.ndarray,
        k: int = 20,
    ) -> pd.DataFrame:
        """Find brain regions where NT and ND predictions diverge most.

        Parameters
        ----------
        divergence_map : np.ndarray
            Per-vertex divergence values (n_vertices,).
        k : int
            Number of top ROIs to return.

        Returns
        -------
        pd.DataFrame
            Top-k most divergent ROIs with their mean divergence values.
        """
        from tribev2.utils import get_hcp_labels, summarize_by_roi

        roi_divergence = summarize_by_roi(
            divergence_map, hemi="both", mesh=self.mesh
        )
        labels = list(
            get_hcp_labels(mesh=self.mesh, combine=False, hemi="both").keys()
        )

        # Sort by divergence
        sorted_idx = np.argsort(roi_divergence)[::-1][:k]

        rows = []
        for idx in sorted_idx:
            rows.append(
                {
                    "roi": labels[idx] if idx < len(labels) else f"ROI_{idx}",
                    "divergence": roi_divergence[idx],
                    "rank": len(rows) + 1,
                }
            )

        return pd.DataFrame(rows)

    def temporal_divergence(
        self,
        nt_preds: np.ndarray,
        nd_preds: np.ndarray,
        segments: list | None = None,
    ) -> pd.DataFrame:
        """Compute how divergence changes over time.

        Useful for identifying specific moments in a stimulus
        where ASD processing diverges from NT processing.

        Parameters
        ----------
        nt_preds : np.ndarray
            (n_timesteps, n_vertices)
        nd_preds : np.ndarray
            (n_timesteps, n_vertices)
        segments : list or None
            Segment objects for time alignment.

        Returns
        -------
        pd.DataFrame
            Per-timestep divergence metrics.
        """
        n_timesteps = min(nt_preds.shape[0], nd_preds.shape[0])
        records = []

        for t in range(n_timesteps):
            nt_t = nt_preds[t]
            nd_t = nd_preds[t]
            diff = nt_t - nd_t

            record = {
                "timestep": t,
                "mse": np.mean(diff**2),
                "mean_abs_diff": np.mean(np.abs(diff)),
                "max_abs_diff": np.max(np.abs(diff)),
                "nt_mean_activation": np.mean(np.abs(nt_t)),
                "nd_mean_activation": np.mean(np.abs(nd_t)),
            }

            if np.std(nt_t) > 1e-8 and np.std(nd_t) > 1e-8:
                record["spatial_correlation"] = np.corrcoef(nt_t, nd_t)[0, 1]
            else:
                record["spatial_correlation"] = 0.0

            records.append(record)

        return pd.DataFrame(records)

    def sensory_profile(
        self,
        nt_preds: np.ndarray,
        nd_preds: np.ndarray,
    ) -> dict[str, tp.Any]:
        """Generate a sensory processing profile from divergence patterns.

        Maps divergence to known functional brain networks to identify
        which sensory/cognitive systems process differently in the
        neurodiverse brain.

        Parameters
        ----------
        nt_preds : np.ndarray
            Neurotypical predictions.
        nd_preds : np.ndarray
            Neurodiverse predictions.

        Returns
        -------
        dict
            Network-level divergence scores:
            - "visual": Visual cortex divergence
            - "auditory": Auditory cortex divergence
            - "language": Language network divergence
            - "default_mode": Default mode network divergence
            - "motor": Motor cortex divergence
            - "social": Social brain regions divergence
            - "profile_summary": Text summary of sensory profile
        """
        from tribev2.utils import get_hcp_roi_indices

        divergence = self.compute_divergence_map(nt_preds, nd_preds, method="mse")

        # Map HCP ROIs to functional networks
        network_rois = {
            "visual": ["V1*", "V2*", "V3*", "V4*", "V6*", "V8*", "FFC*", "VVC*"],
            "auditory": ["A1*", "A4*", "A5*", "RI*", "MBelt*", "LBelt*", "PBelt*"],
            "language": [
                "55b*", "STV*", "TPOJ1*", "PSL*", "SFL*",
                "44*", "45*", "47l*", "IFSa*", "IFSp*",
            ],
            "default_mode": [
                "POS1*", "POS2*", "v23ab*", "d23ab*",
                "31a*", "31pd*", "31pv*", "7m*",
                "PCV*", "RSC*",
            ],
            "motor": ["4*", "3a*", "3b*", "1*", "2*", "6a*", "6d*"],
            "social": [
                "STS*", "STSda*", "STSdp*", "STSva*", "STSvp*",
                "TE1a*", "TE1p*", "TGd*", "TGv*",
            ],
        }

        profile = {}
        for network, rois in network_rois.items():
            network_div = []
            for roi_pattern in rois:
                try:
                    indices = get_hcp_roi_indices(roi_pattern, mesh=self.mesh)
                    valid = indices[indices < len(divergence)]
                    if len(valid) > 0:
                        network_div.append(divergence[valid].mean())
                except ValueError:
                    continue
            profile[network] = float(np.mean(network_div)) if network_div else 0.0

        # Normalize to 0-1 scale
        max_div = max(profile.values()) if max(profile.values()) > 0 else 1.0
        for k in profile:
            profile[k] /= max_div

        # Generate summary
        sorted_networks = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        top_3 = [f"{name} ({score:.0%})" for name, score in sorted_networks[:3]]
        profile["profile_summary"] = (
            f"Highest processing differences in: {', '.join(top_3)}. "
            f"This suggests the neurodiverse brain processes "
            f"{sorted_networks[0][0]} stimuli most differently from "
            f"the neurotypical baseline."
        )

        return profile

    def generate_report(
        self,
        events: pd.DataFrame,
        output_dir: str | Path = "./neurodiverse_report",
    ) -> Path:
        """Generate a full comparison report with visualizations.

        Parameters
        ----------
        events : pd.DataFrame
            Stimulus events.
        output_dir : str or Path
            Directory to save the report.

        Returns
        -------
        Path
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run predictions
        nt_preds, nd_preds = self.predict_both(events)

        # Divergence map
        divergence = self.compute_divergence_map(nt_preds, nd_preds)
        np.save(output_dir / "divergence_map.npy", divergence)

        # Top divergent ROIs
        top_rois = self.get_top_divergent_rois(divergence)
        top_rois.to_csv(output_dir / "top_divergent_rois.csv", index=False)

        # Temporal divergence
        temporal = self.temporal_divergence(nt_preds, nd_preds)
        temporal.to_csv(output_dir / "temporal_divergence.csv", index=False)

        # Sensory profile
        profile = self.sensory_profile(nt_preds, nd_preds)
        profile_df = pd.DataFrame(
            [
                {"network": k, "divergence": v}
                for k, v in profile.items()
                if k != "profile_summary"
            ]
        )
        profile_df.to_csv(output_dir / "sensory_profile.csv", index=False)

        # Save summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write("Neurodiverse Brain Model Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timesteps analyzed: {nt_preds.shape[0]}\n")
            f.write(f"Brain vertices: {nt_preds.shape[1]}\n\n")
            f.write("Sensory Profile:\n")
            f.write(profile["profile_summary"] + "\n\n")
            f.write("Top 10 Most Divergent Brain Regions:\n")
            for _, row in top_rois.head(10).iterrows():
                f.write(f"  {row['rank']}. {row['roi']}: {row['divergence']:.4f}\n")

        logger.info("Report saved to %s", output_dir)
        return output_dir
