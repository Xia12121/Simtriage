"""Task 4 acceptance: VoQ online_update lowers prediction MSE on synthetic linear data."""

import numpy as np

from src.voq import VoQKNN, VoQRidge


def _mse(model, X, y):
    return float(np.mean([(model.predict(X[i]) - y[i]) ** 2 for i in range(len(y))]))


def test_ridge_mse_decreases_with_samples():
    dim = 6
    # one shared ground-truth w so train and test come from the same linear model
    rng = np.random.default_rng(1)
    Xall = rng.normal(size=(500, dim)); Xall[:, -1] = 1.0
    w = rng.normal(size=dim)
    yall = Xall @ w + rng.normal(0, 0.05, size=500)
    Xtr, ytr, Xte, yte = Xall[:400], yall[:400], Xall[400:], yall[400:]

    model = VoQRidge(dim, l2=1.0, optimistic=1.0, warmup_n=20)
    errs = []
    for i in range(len(ytr)):
        model.online_update(Xtr[i], ytr[i])
        if (i + 1) in (40, 120, 400):
            errs.append(_mse(model, Xte, yte))
    assert errs[-1] < errs[0], f"MSE should drop with more samples: {errs}"
    assert errs[-1] < 0.05


def test_ridge_cold_start_optimistic():
    m = VoQRidge(4, optimistic=1.5, warmup_n=20)
    assert m.predict(np.ones(4)) == 1.5  # cold start returns the optimistic value


def test_ridge_cold_start_prior_used():
    m = VoQRidge(4, optimistic=1.5, warmup_n=20)
    m.cold_start_prior = lambda phi: 0.3
    assert abs(m.predict(np.ones(4)) - 0.3) < 1e-9  # transferred prior overrides optimism


def test_knn_learns():
    dim = 5
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, dim)); X[:, -1] = 1.0
    w = rng.normal(size=dim)
    y = X @ w
    m = VoQKNN(dim, k=5, warmup_n=10)
    for i in range(150):
        m.online_update(X[i], y[i])
    err = np.mean([(m.predict(X[i]) - y[i]) ** 2 for i in range(150, 200)])
    assert err < 1.0
