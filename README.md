# mcepy

`mcepy` is a small Python package for stable dimensionality reduction via **Median Consensus Embedding (MCE)**, based on the paper **“Median Consensus Embedding for Dimensionality Reduction”**.

## Installation

### Install directly from GitHub

```bash
pip install "git+https://github.com/t-yui/mcepy.git"
```

### Local install

```bash
git clone https://github.com/t-yui/mcepy.git
cd mcepy
pip install -e .
```

### Rebuild the Cython from `.pyx`

A generated C source file is included so ordinary users do not need Cython at install time. If you want to regenerate the extension from the `.pyx` source during development:

```bash
pip install -e .[dev]
MCEPY_USE_CYTHON=1 pip install -e .
```

If compilation is unavailable, installation still succeeds and `mcepy` falls back to a NumPy implementation.

## Requirements

- Python >= 3.10
- NumPy
- scikit-learn
- umap-learn

## Quick start

### 1. MCE from precomputed base embeddings

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

from mcepy import drmce

rng = np.random.default_rng(100)
data = load_digits().data

base_embeddings = []
for seed in rng.integers(0, 2**32 - 1, size=10):
    tsne = TSNE(
        n_components=2,
        init="random",
        random_state=int(seed),
    )
    base_embeddings.append(tsne.fit_transform(data))

consensus_embedding, consensus_distance = drmce(
    base_embeddings,
    n_components=2,
)
```

### 2. Repeated t-SNE + MCE

```python
from sklearn.datasets import load_digits

from mcepy import tsnemce

X, y = load_digits(return_X_y=True)

consensus_embedding, consensus_distance, info = tsnemce(
    X,
    n_runs=10,
    random_state=1,
    tsne_kwargs={
        "perplexity": 30
    },
    return_info=True,
)
```

### 3. Repeated UMAP + MCE

```python
from sklearn.datasets import load_digits

from mcepy import umapmce

X, y = load_digits(return_X_y=True)

consensus_embedding, consensus_distance, info = umapmce(
    X,
    n_runs=10,
    random_state=1,
    umap_kwargs={
        "n_neighbors": 15,
        "min_dist": 0.1
    },
    return_info=True,
)
```

## API

### `drmce(base_embeddings, ...)`

Compute a median consensus embedding from precomputed base embeddings.

Returns:

- `consensus_embedding`: final embedding returned by metric MDS
- `consensus_distance_matrix`: consensus pairwise distance matrix
- `info` (optional): backend name, convergence flag, iteration count, objective value, MDS stress, and basic shape metadata

### `tsnemce(data, n_runs, ...)`

Repeatedly runs `sklearn.manifold.TSNE` and then applies `drmce` to the resulting embeddings.

Notes:

- `random_state` in `tsnemce` controls the generation of per-run seeds
- `tsne_kwargs` is forwarded to scikit-learn's `TSNE`
- `init` defaults to `"random"` inside `tsnemce` to preserve run-to-run stochasticity, but you can override it explicitly

### `umapmce(data, n_runs, ...)`

Repeatedly runs `umap.UMAP` and then applies `drmce` to the resulting embeddings.

Notes:

- `random_state` in `umapmce` controls the generation of per-run seeds
- `umap_kwargs` is forwarded to `umap.UMAP`
- `init` defaults to `"random"` inside `umapmce` to preserve run-to-run stochasticity, but you can override it explicitly

### Backward-compatible alias

```python
from mcepy import MCE
```

`MCE` is an alias of `drmce`, kept for compatibility with the paper repository's original naming.

## Citation

If you use `mcepy`, cite the MCE paper:

```text
@article{medianconsensusembedding,
    title    = {Median Consensus Embedding for Dimensionality Reduction},
    author   = {Tomo, Yui and Yoneoka, Daisuke},
    journal  = {arXiv preprint arXiv:2503.08103},
    year     = {2025}
}
```
