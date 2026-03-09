import numpy as np


def embeddings_to_condensed(embeddings, normalize=True):
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 3:
        raise ValueError("embeddings must be a 3D array with shape (n_embeddings, n_samples, n_features)")

    n_embeddings, n_samples, _ = embeddings.shape
    length = n_samples * (n_samples - 1) // 2
    out = np.empty((n_embeddings, length), dtype=np.float64)
    tri_upper = np.triu_indices(n_samples, k=1)

    for idx in range(n_embeddings):
        work = embeddings[idx].astype(np.float64, copy=True)
        if normalize:
            work -= work.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(work)
            if norm != 0.0:
                work /= norm

        sq_norm = np.sum(work * work, axis=1, dtype=np.float64)
        d2 = sq_norm[:, None] + sq_norm[None, :] - 2.0 * work.dot(work.T)
        np.maximum(d2, 0.0, out=d2)
        out[idx] = np.sqrt(d2, out=d2)[tri_upper]

    return out


def geometric_median_condensed(condensed, eps=1e-15, tol=1e-5, max_iter=500):
    condensed = np.asarray(condensed, dtype=np.float64)
    if condensed.ndim != 2:
        raise ValueError("condensed must be a 2D array")
    if condensed.shape[0] == 0:
        raise ValueError("condensed must contain at least one embedding")

    center = condensed.mean(axis=0)
    converged = False

    for iteration in range(max_iter):
        diff = condensed - center
        distances = np.sqrt(2.0 * np.sum(diff * diff, axis=1))
        weights = 1.0 / (distances + eps)
        new_center = np.sum(weights[:, None] * condensed, axis=0) / np.sum(weights)
        delta = np.sqrt(2.0 * np.sum((new_center - center) ** 2))
        center = new_center
        if delta < tol:
            converged = True
            break

    diff = condensed - center
    objective = float(np.mean(np.sqrt(2.0 * np.sum(diff * diff, axis=1))))
    n_iter = iteration + 1 if max_iter > 0 else 0
    return center, n_iter, converged, objective


def condensed_to_square(condensed, n_samples):
    condensed = np.asarray(condensed, dtype=np.float64)
    out = np.zeros((n_samples, n_samples), dtype=np.float64)
    tri_upper = np.triu_indices(n_samples, k=1)
    out[tri_upper] = condensed
    out[(tri_upper[1], tri_upper[0])] = condensed
    return out
