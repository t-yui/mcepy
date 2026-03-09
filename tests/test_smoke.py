import numpy as np

from mcepy import MCE, drmce, normalize_embedding


def test_normalize_embedding_centers_rows():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = normalize_embedding(x)
    assert np.allclose(y.mean(axis=0), 0.0)
    assert np.isclose(np.linalg.norm(y), 1.0)


def test_drmce_shapes_and_symmetry():
    rng = np.random.default_rng(0)
    base = [rng.normal(size=(20, 2)) for _ in range(5)]
    embedding, distance, info = drmce(base, return_info=True, max_iter=10)

    assert embedding.shape == (20, 2)
    assert distance.shape == (20, 20)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert info["n_base_embeddings"] == 5


def test_alias_matches_function():
    rng = np.random.default_rng(1)
    base = [rng.normal(size=(10, 2)) for _ in range(3)]
    emb1, dist1 = drmce(base, max_iter=5)
    emb2, dist2 = MCE(base, max_iter=5)
    assert emb1.shape == emb2.shape
    assert np.allclose(dist1, dist2)
