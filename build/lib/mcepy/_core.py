from __future__ import annotations

import copy
import inspect
import warnings

import numpy as np
from sklearn.manifold import MDS, TSNE

try:
    from . import _speedups as _backend
    _BACKEND_NAME = "cython"
except Exception:
    from . import _speedups_fallback as _backend
    _BACKEND_NAME = "python"


def normalize_embedding(embedding):
    """Center an embedding and scale it to unit Frobenius norm.

    Parameters
    ----------
    embedding : array-like of shape (n_samples, n_features)
        Input embedding.

    Returns
    -------
    ndarray of shape (n_samples, n_features)
        Centered embedding with unit Frobenius norm. Degenerate zero-norm inputs are
        returned after centering.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError("embedding must be a 2D array")

    centered = embedding - embedding.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centered)
    if norm == 0.0:
        return centered
    return centered / norm



def _coerce_base_embeddings(base_embeddings):
    if isinstance(base_embeddings, np.ndarray):
        array = np.asarray(base_embeddings, dtype=np.float64)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        if array.ndim != 3:
            raise ValueError(
                "base_embeddings must be a 3D array with shape "
                "(n_embeddings, n_samples, n_features), or a sequence of 2D arrays"
            )
        return np.ascontiguousarray(array, dtype=np.float64)

    embeddings = [np.asarray(embedding, dtype=np.float64) for embedding in base_embeddings]
    if not embeddings:
        raise ValueError("base_embeddings must contain at least one embedding")

    shapes = {embedding.shape for embedding in embeddings}
    if len(shapes) != 1:
        raise ValueError("all base embeddings must have the same shape")
    if embeddings[0].ndim != 2:
        raise ValueError("each base embedding must be a 2D array")

    return np.ascontiguousarray(np.stack(embeddings, axis=0), dtype=np.float64)



def _as_int(value, name, *, minimum=0, strictly_positive=False):
    try:
        integer = int(value)
    except Exception as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if integer != value and not (isinstance(value, np.generic) and integer == value.item()):
        raise TypeError(f"{name} must be an integer")
    if strictly_positive and integer <= 0:
        raise ValueError(f"{name} must be positive")
    if not strictly_positive and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer



def _validate_positive(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be positive")



def _validate_non_negative(value, name):
    if value < 0:
        raise ValueError(f"{name} must be non-negative")



def _prepare_mds(n_components, rs_mds, n_init_mds, mds_max_iter, mds_kwargs):
    mds_kwargs = {} if mds_kwargs is None else copy.deepcopy(mds_kwargs)
    if "n_components" in mds_kwargs and mds_kwargs["n_components"] != n_components:
        raise ValueError("n_components must be passed via drmce, not mds_kwargs")
    if "random_state" in mds_kwargs and mds_kwargs["random_state"] != rs_mds:
        raise ValueError("random_state must be passed via rs_mds, not mds_kwargs")

    params = set(inspect.signature(MDS).parameters)
    mds_kwargs.setdefault("n_init", n_init_mds)
    mds_kwargs.setdefault("max_iter", mds_max_iter)

    if "metric_mds" in params:
        if "dissimilarity" in mds_kwargs and mds_kwargs["dissimilarity"] != "precomputed":
            raise ValueError("mds_kwargs['dissimilarity'] must be 'precomputed'")
        if "metric" in mds_kwargs and mds_kwargs["metric"] != "precomputed":
            raise ValueError("mds_kwargs['metric'] must be 'precomputed'")
        mds_kwargs.pop("dissimilarity", None)
        mds_kwargs.setdefault("metric", "precomputed")
        mds_kwargs.setdefault("metric_mds", True)
        mds_kwargs.setdefault("init", "random")
    else:
        if "metric" in mds_kwargs:
            raise ValueError(
                "This scikit-learn version does not support mds_kwargs['metric']; "
                "use mds_kwargs['dissimilarity'] instead."
            )
        if "dissimilarity" in mds_kwargs and mds_kwargs["dissimilarity"] != "precomputed":
            raise ValueError("mds_kwargs['dissimilarity'] must be 'precomputed'")
        mds_kwargs.setdefault("dissimilarity", "precomputed")

    return MDS(
        n_components=n_components,
        random_state=rs_mds,
        **mds_kwargs,
    )



def drmce(
    base_embeddings,
    n_components=2,
    rs_mds=1,
    eps=1e-15,
    tol=1e-5,
    max_iter=500,
    normalize=True,
    n_init_mds=4,
    mds_max_iter=300,
    mds_kwargs=None,
    return_info=False,
):
    """Compute a median consensus embedding from precomputed embeddings.

    This function follows the practical algorithm in the MCE paper: each base embedding is
    represented by its pairwise distance matrix, the geometric median is computed with a
    Weiszfeld update under the Frobenius norm, and the resulting consensus distance matrix is
    projected with metric MDS.

    Parameters
    ----------
    base_embeddings : sequence of array-like or ndarray of shape (n_embeddings, n_samples, n_features)
        Collection of embeddings to integrate.
    n_components : int, default=2
        Output dimensionality of the final consensus embedding.
    rs_mds : int or None, default=1
        Random state for the final MDS step.
    eps : float, default=1e-15
        Stabilizer used in the Weiszfeld update.
    tol : float, default=1e-5
        Convergence tolerance of the Weiszfeld update measured in Frobenius norm.
    max_iter : int, default=500
        Maximum number of Weiszfeld iterations.
    normalize : bool, default=True
        If True, each base embedding is centered and scaled to unit Frobenius norm before
        pairwise distances are computed.
    n_init_mds : int, default=4
        Number of SMACOF restarts used by sklearn.manifold.MDS.
    mds_max_iter : int, default=300
        Maximum number of SMACOF iterations in the final MDS step.
    mds_kwargs : dict or None, default=None
        Additional keyword arguments forwarded to sklearn.manifold.MDS. The dissimilarity is
        always fixed to "precomputed".
    return_info : bool, default=False
        If True, an additional dictionary with optimization diagnostics is returned.

    Returns
    -------
    consensus_embedding : ndarray of shape (n_samples, n_components)
        Consensus embedding returned by the final MDS step.
    consensus_distance_matrix : ndarray of shape (n_samples, n_samples)
        Consensus pairwise distance matrix.
    info : dict, optional
        Diagnostic information returned when ``return_info=True``.
    """
    n_components = _as_int(n_components, "n_components", strictly_positive=True)
    max_iter = _as_int(max_iter, "max_iter", minimum=0)
    n_init_mds = _as_int(n_init_mds, "n_init_mds", strictly_positive=True)
    mds_max_iter = _as_int(mds_max_iter, "mds_max_iter", strictly_positive=True)
    _validate_positive(eps, "eps")
    _validate_positive(tol, "tol")

    embeddings = _coerce_base_embeddings(base_embeddings)
    n_embeddings, n_samples, n_features = embeddings.shape

    condensed = _backend.embeddings_to_condensed(embeddings, normalize=normalize)
    center, n_iter, converged, objective = _backend.geometric_median_condensed(
        condensed,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
    )
    consensus_distance_matrix = _backend.condensed_to_square(center, n_samples)
    consensus_distance_matrix = np.maximum(consensus_distance_matrix, 0.0)
    np.fill_diagonal(consensus_distance_matrix, 0.0)

    mds = _prepare_mds(
        n_components=n_components,
        rs_mds=rs_mds,
        n_init_mds=n_init_mds,
        mds_max_iter=mds_max_iter,
        mds_kwargs=mds_kwargs,
    )
    consensus_embedding = mds.fit_transform(consensus_distance_matrix)

    if not converged:
        warnings.warn(
            "Weiszfeld iterations reached max_iter before convergence.",
            RuntimeWarning,
            stacklevel=2,
        )

    info = {
        "backend": _BACKEND_NAME,
        "converged": bool(converged),
        "weiszfeld_iterations": int(n_iter),
        "objective": float(objective),
        "mds_stress": float(getattr(mds, "stress_", np.nan)),
        "n_base_embeddings": int(n_embeddings),
        "n_samples": int(n_samples),
        "base_embedding_dim": int(n_features),
        "normalized_inputs": bool(normalize),
    }

    if return_info:
        return consensus_embedding, consensus_distance_matrix, info
    return consensus_embedding, consensus_distance_matrix


MCE = drmce



def _generate_seeds(n_runs, random_state):
    rng = np.random.default_rng(random_state)
    return [int(seed) for seed in rng.integers(0, np.iinfo(np.uint32).max, size=n_runs)]



def tsnemce(
    data,
    n_runs,
    n_components=2,
    random_state=None,
    tsne_kwargs=None,
    mce_kwargs=None,
    return_info=False,
    return_base_embeddings=False,
):
    """Run t-SNE repeatedly and aggregate the resulting embeddings with MCE.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input data matrix.
    n_runs : int
        Number of independent t-SNE runs.
    n_components : int, default=2
        Dimensionality of the t-SNE runs. The final MCE output uses the same dimensionality
        unless it is overridden in ``mce_kwargs``.
    random_state : int or None, default=None
        Seed used to generate run-specific random states.
    tsne_kwargs : dict or None, default=None
        Additional keyword arguments forwarded to sklearn.manifold.TSNE. ``n_components`` and
        ``random_state`` are managed internally. ``init`` defaults to ``"random"`` so that the
        repeated runs remain stochastic.
    mce_kwargs : dict or None, default=None
        Additional keyword arguments forwarded to :func:`drmce`.
    return_info : bool, default=False
        If True, return diagnostics for both the base t-SNE runs and the MCE step.
    return_base_embeddings : bool, default=False
        If True, include the stacked base embeddings in the returned info dictionary.

    Returns
    -------
    consensus_embedding : ndarray of shape (n_samples, n_components)
        Final consensus embedding.
    consensus_distance_matrix : ndarray of shape (n_samples, n_samples)
        Consensus pairwise distance matrix.
    info : dict, optional
        Diagnostics returned when ``return_info=True`` or ``return_base_embeddings=True``.
    """
    n_runs = _as_int(n_runs, "n_runs", strictly_positive=True)
    n_components = _as_int(n_components, "n_components", strictly_positive=True)

    shape = getattr(data, "shape", None)
    if shape is None:
        data = np.asarray(data)
        shape = data.shape
    if len(shape) != 2:
        raise ValueError("data must be a 2D array")

    tsne_kwargs = {} if tsne_kwargs is None else copy.deepcopy(tsne_kwargs)
    if "n_components" in tsne_kwargs and tsne_kwargs["n_components"] != n_components:
        raise ValueError("n_components must be passed via the function argument, not tsne_kwargs")
    if "random_state" in tsne_kwargs:
        raise ValueError("random_state is managed internally for each t-SNE run")
    tsne_kwargs.setdefault("init", "random")

    seeds = _generate_seeds(n_runs, random_state)
    base_embeddings = []
    for seed in seeds:
        estimator = TSNE(
            n_components=n_components,
            random_state=seed,
            **tsne_kwargs,
        )
        base_embeddings.append(estimator.fit_transform(data))

    mce_kwargs = {} if mce_kwargs is None else copy.deepcopy(mce_kwargs)
    if "return_info" in mce_kwargs:
        raise ValueError("return_info must be passed to tsnemce, not inside mce_kwargs")
    mce_kwargs.setdefault("n_components", n_components)

    consensus_embedding, consensus_distance_matrix, info = drmce(
        base_embeddings,
        return_info=True,
        **mce_kwargs,
    )
    info.update(
        {
            "base_method": "tsne",
            "n_runs": int(n_runs),
            "random_seeds": seeds,
            "tsne_kwargs": tsne_kwargs,
        }
    )
    if return_base_embeddings:
        info["base_embeddings"] = np.asarray(base_embeddings, dtype=np.float64)

    if return_info or return_base_embeddings:
        return consensus_embedding, consensus_distance_matrix, info
    return consensus_embedding, consensus_distance_matrix



def umapmce(
    data,
    n_runs,
    n_components=2,
    random_state=None,
    umap_kwargs=None,
    mce_kwargs=None,
    return_info=False,
    return_base_embeddings=False,
):
    """Run UMAP repeatedly and aggregate the resulting embeddings with MCE.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input data matrix.
    n_runs : int
        Number of independent UMAP runs.
    n_components : int, default=2
        Dimensionality of the UMAP runs. The final MCE output uses the same dimensionality
        unless it is overridden in ``mce_kwargs``.
    random_state : int or None, default=None
        Seed used to generate run-specific random states.
    umap_kwargs : dict or None, default=None
        Additional keyword arguments forwarded to ``umap.UMAP``. ``n_components`` and
        ``random_state`` are managed internally. ``init`` defaults to ``"random"`` so that the
        repeated runs remain stochastic.
    mce_kwargs : dict or None, default=None
        Additional keyword arguments forwarded to :func:`drmce`.
    return_info : bool, default=False
        If True, return diagnostics for both the base UMAP runs and the MCE step.
    return_base_embeddings : bool, default=False
        If True, include the stacked base embeddings in the returned info dictionary.

    Returns
    -------
    consensus_embedding : ndarray of shape (n_samples, n_components)
        Final consensus embedding.
    consensus_distance_matrix : ndarray of shape (n_samples, n_samples)
        Consensus pairwise distance matrix.
    info : dict, optional
        Diagnostics returned when ``return_info=True`` or ``return_base_embeddings=True``.
    """
    n_runs = _as_int(n_runs, "n_runs", strictly_positive=True)
    n_components = _as_int(n_components, "n_components", strictly_positive=True)

    shape = getattr(data, "shape", None)
    if shape is None:
        data = np.asarray(data)
        shape = data.shape
    if len(shape) != 2:
        raise ValueError("data must be a 2D array")

    try:
        import umap
    except Exception as exc:
        raise ImportError(
            "umapmce requires the 'umap-learn' package. Install mcepy with the standard "
            "dependencies, or install 'umap-learn' manually."
        ) from exc

    umap_kwargs = {} if umap_kwargs is None else copy.deepcopy(umap_kwargs)
    if "n_components" in umap_kwargs and umap_kwargs["n_components"] != n_components:
        raise ValueError("n_components must be passed via the function argument, not umap_kwargs")
    if "random_state" in umap_kwargs:
        raise ValueError("random_state is managed internally for each UMAP run")
    umap_kwargs.setdefault("init", "random")

    seeds = _generate_seeds(n_runs, random_state)
    base_embeddings = []
    for seed in seeds:
        estimator = umap.UMAP(
            n_components=n_components,
            random_state=seed,
            **umap_kwargs,
        )
        base_embeddings.append(estimator.fit_transform(data))

    mce_kwargs = {} if mce_kwargs is None else copy.deepcopy(mce_kwargs)
    if "return_info" in mce_kwargs:
        raise ValueError("return_info must be passed to umapmce, not inside mce_kwargs")
    mce_kwargs.setdefault("n_components", n_components)

    consensus_embedding, consensus_distance_matrix, info = drmce(
        base_embeddings,
        return_info=True,
        **mce_kwargs,
    )
    info.update(
        {
            "base_method": "umap",
            "n_runs": int(n_runs),
            "random_seeds": seeds,
            "umap_kwargs": umap_kwargs,
        }
    )
    if return_base_embeddings:
        info["base_embeddings"] = np.asarray(base_embeddings, dtype=np.float64)

    if return_info or return_base_embeddings:
        return consensus_embedding, consensus_distance_matrix, info
    return consensus_embedding, consensus_distance_matrix
