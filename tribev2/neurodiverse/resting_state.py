"""Resting-state functional connectivity analysis on fsaverage5.

Projects ABIDE resting-state fMRI onto the same cortical surface
used by TRIBE v2 (fsaverage5, ~20K vertices), then computes
functional connectivity matrices for ASD vs TD comparison.
"""

import logging
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RestingStateAnalyzer:
    """Analyze resting-state fMRI from ABIDE on TRIBE v2's brain surface.

    Projects volumetric resting-state data to fsaverage5 using TRIBE v2's
    own surface projector, computes functional connectivity, and compares
    ASD vs TD groups.

    Parameters
    ----------
    mesh : str
        Target fsaverage mesh resolution. Default "fsaverage5" matches
        TRIBE v2's output space.
    n_parcels : int
        Number of parcels for ROI-level connectivity.
        Uses HCP MMP1 parcellation (180 per hemisphere = 360 total).
    """

    def __init__(self, mesh: str = "fsaverage5", n_parcels: int = 360):
        self.mesh = mesh
        self.n_parcels = n_parcels
        self._projector = None

    def _get_projector(self):
        """Lazy-load the surface projector."""
        if self._projector is None:
            from tribev2.utils_fmri import TribeSurfaceProjector

            self._projector = TribeSurfaceProjector(
                mesh=self.mesh,
                radius=3.0,
                interpolation="linear",
                kind="ball",
            )
        return self._projector

    def project_to_surface(self, nifti_path: str | Path) -> np.ndarray:
        """Project a volumetric NIfTI to fsaverage5 surface.

        Parameters
        ----------
        nifti_path : str or Path
            Path to a 4D NIfTI file (resting-state fMRI).

        Returns
        -------
        np.ndarray
            Surface data of shape (n_vertices * 2, n_timepoints).
            For fsaverage5: (20484, n_timepoints).
        """
        import nibabel

        projector = self._get_projector()
        nii = nibabel.load(str(nifti_path))
        surface_data = projector.apply(nii)
        logger.info(
            "Projected %s -> surface shape %s",
            Path(nifti_path).name,
            surface_data.shape,
        )
        return surface_data

    def compute_connectivity(
        self,
        surface_data: np.ndarray,
        method: str = "correlation",
    ) -> np.ndarray:
        """Compute ROI-level functional connectivity from surface timeseries.

        Parameters
        ----------
        surface_data : np.ndarray
            Shape (n_vertices, n_timepoints) on fsaverage5.
        method : str
            Connectivity measure: "correlation" or "partial_correlation".

        Returns
        -------
        np.ndarray
            Connectivity matrix of shape (n_rois, n_rois).
        """
        from tribev2.utils import get_hcp_labels

        labels = get_hcp_labels(mesh=self.mesh, combine=False, hemi="both")

        # Extract mean timeseries per ROI
        roi_names = list(labels.keys())
        n_rois = len(roi_names)
        n_timepoints = surface_data.shape[1]
        roi_timeseries = np.zeros((n_rois, n_timepoints))

        for i, (name, vertices) in enumerate(labels.items()):
            valid = vertices[vertices < surface_data.shape[0]]
            if len(valid) > 0:
                roi_timeseries[i] = surface_data[valid].mean(axis=0)

        if method == "correlation":
            conn = np.corrcoef(roi_timeseries)
        elif method == "partial_correlation":
            from sklearn.covariance import LedoitWolf

            estimator = LedoitWolf()
            estimator.fit(roi_timeseries.T)
            precision = estimator.precision_
            d = np.sqrt(np.diag(precision))
            conn = -precision / np.outer(d, d)
            np.fill_diagonal(conn, 1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fisher z-transform for statistical comparisons
        conn = np.arctanh(np.clip(conn, -0.999, 0.999))
        np.fill_diagonal(conn, 0)

        return conn

    def batch_project_and_connect(
        self,
        phenotypic: pd.DataFrame,
        func_col: str = "func_preproc_path",
        max_subjects: int | None = None,
    ) -> dict[str, list[np.ndarray]]:
        """Project and compute connectivity for a batch of subjects.

        Parameters
        ----------
        phenotypic : pd.DataFrame
            Must contain columns: func_col, "diagnosis" (ASD/TD).
        func_col : str
            Column name containing NIfTI file paths.
        max_subjects : int or None
            Limit subjects per group for testing.

        Returns
        -------
        dict
            {"ASD": [conn_matrix, ...], "TD": [conn_matrix, ...]}
        """
        results: dict[str, list[np.ndarray]] = {"ASD": [], "TD": []}

        for group in ["ASD", "TD"]:
            subjects = phenotypic[phenotypic["diagnosis"] == group]
            if max_subjects is not None:
                subjects = subjects.head(max_subjects)

            logger.info("Processing %d %s subjects...", len(subjects), group)

            for _, row in subjects.iterrows():
                func_path = row[func_col]
                if not Path(func_path).exists():
                    logger.warning("Missing: %s", func_path)
                    continue
                try:
                    surface = self.project_to_surface(func_path)
                    conn = self.compute_connectivity(surface)
                    results[group].append(conn)
                except Exception as e:
                    logger.warning("Failed for %s: %s", func_path, e)
                    continue

        logger.info(
            "Processed: %d ASD, %d TD subjects",
            len(results["ASD"]),
            len(results["TD"]),
        )
        return results

    def compare_groups(
        self,
        asd_conns: list[np.ndarray],
        td_conns: list[np.ndarray],
        correction: str = "fdr_bh",
        alpha: float = 0.05,
    ) -> dict[str, tp.Any]:
        """Statistical comparison of ASD vs TD functional connectivity.

        Parameters
        ----------
        asd_conns : list[np.ndarray]
            Connectivity matrices for ASD group.
        td_conns : list[np.ndarray]
            Connectivity matrices for TD group.
        correction : str
            Multiple comparison correction method.
        alpha : float
            Significance threshold.

        Returns
        -------
        dict
            - "t_stats": t-statistic matrix (n_rois, n_rois)
            - "p_values": p-value matrix
            - "significant": boolean mask of significant connections
            - "asd_mean": mean ASD connectivity
            - "td_mean": mean TD connectivity
            - "difference": ASD mean - TD mean
            - "top_connections": DataFrame of most significant connections
        """
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        from tribev2.utils import get_hcp_labels

        asd_stack = np.stack(asd_conns)  # (n_asd, n_rois, n_rois)
        td_stack = np.stack(td_conns)  # (n_td, n_rois, n_rois)

        asd_mean = asd_stack.mean(axis=0)
        td_mean = td_stack.mean(axis=0)
        difference = asd_mean - td_mean

        n_rois = asd_mean.shape[0]
        t_stats = np.zeros((n_rois, n_rois))
        p_values = np.ones((n_rois, n_rois))

        # Independent samples t-test per connection
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                t, p = stats.ttest_ind(asd_stack[:, i, j], td_stack[:, i, j])
                t_stats[i, j] = t_stats[j, i] = t
                p_values[i, j] = p_values[j, i] = p

        # Multiple comparison correction on upper triangle
        upper_idx = np.triu_indices(n_rois, k=1)
        p_flat = p_values[upper_idx]
        reject, p_corrected, _, _ = multipletests(p_flat, alpha=alpha, method=correction)

        significant = np.zeros((n_rois, n_rois), dtype=bool)
        p_corrected_matrix = np.ones((n_rois, n_rois))
        for k, (i, j) in enumerate(zip(*upper_idx)):
            significant[i, j] = significant[j, i] = reject[k]
            p_corrected_matrix[i, j] = p_corrected_matrix[j, i] = p_corrected[k]

        # Build top connections table
        labels = list(
            get_hcp_labels(mesh=self.mesh, combine=False, hemi="both").keys()
        )
        top_connections = []
        for k in np.argsort(p_flat):
            i, j = upper_idx[0][k], upper_idx[1][k]
            top_connections.append(
                {
                    "roi_1": labels[i] if i < len(labels) else f"ROI_{i}",
                    "roi_2": labels[j] if j < len(labels) else f"ROI_{j}",
                    "t_stat": t_stats[i, j],
                    "p_corrected": p_corrected[k],
                    "asd_mean": asd_mean[i, j],
                    "td_mean": td_mean[i, j],
                    "difference": difference[i, j],
                    "significant": reject[k],
                }
            )
            if len(top_connections) >= 50:
                break

        return {
            "t_stats": t_stats,
            "p_values": p_corrected_matrix,
            "significant": significant,
            "asd_mean": asd_mean,
            "td_mean": td_mean,
            "difference": difference,
            "top_connections": pd.DataFrame(top_connections),
        }

    def connectivity_to_surface(
        self, conn_values: np.ndarray, metric: str = "degree"
    ) -> np.ndarray:
        """Convert ROI-level connectivity metric to vertex-level surface map.

        Useful for visualizing connectivity results on the brain surface
        using TRIBE v2's PlotBrain.

        Parameters
        ----------
        conn_values : np.ndarray
            Connectivity matrix (n_rois, n_rois) or vector (n_rois,).
        metric : str
            If conn_values is a matrix:
            - "degree": sum of absolute connections per ROI
            - "strength": sum of connections per ROI (signed)
            If conn_values is a vector, it's used directly.

        Returns
        -------
        np.ndarray
            Vertex-level values on fsaverage5 (n_vertices * 2,).
        """
        from neuralset.extractors.neuro import FSAVERAGE_SIZES

        from tribev2.utils import get_hcp_labels

        labels = get_hcp_labels(mesh=self.mesh, combine=False, hemi="both")
        n_vertices = FSAVERAGE_SIZES[self.mesh] * 2

        if conn_values.ndim == 2:
            if metric == "degree":
                roi_values = np.abs(conn_values).sum(axis=1)
            elif metric == "strength":
                roi_values = conn_values.sum(axis=1)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            roi_values = conn_values

        surface_map = np.zeros(n_vertices)
        for i, (name, vertices) in enumerate(labels.items()):
            if i < len(roi_values):
                valid = vertices[vertices < n_vertices]
                surface_map[valid] = roi_values[i]

        return surface_map
