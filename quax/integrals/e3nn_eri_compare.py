import jax.numpy as jnp

from .e3nn_eri_train import train_e3nn_eri, evaluate_e3nn_eri
from .e3nn_eri_v2 import train_e3nn_eri_v2, e3nn_eri_v2_predict


def compare_e3nn_v1_v2(
    geom,
    basis,
    target_eri,
    steps=100,
    lr=1e-2,
    v1_rank=48,
    v2_rank=64,
    v2_hidden_dim=128,
):
    """
    One-call comparison of v1 and v2 e3nn ERI models on the same target.
    """
    p1, h1 = train_e3nn_eri(geom, basis, target_eri, steps=steps, lr=lr, rank=v1_rank)
    m1 = evaluate_e3nn_eri(geom, basis, p1, target_eri)

    p2, h2 = train_e3nn_eri_v2(
        geom,
        basis,
        target_eri,
        steps=steps,
        lr=lr,
        rank=v2_rank,
        hidden_dim=v2_hidden_dim,
    )
    pred2 = e3nn_eri_v2_predict(geom, basis, p2)
    m2 = {
        'mse': jnp.mean((pred2 - target_eri) ** 2),
        'mae': jnp.mean(jnp.abs(pred2 - target_eri)),
        'pred': pred2,
    }

    summary = {
        'v1': {'mse': m1['mse'], 'mae': m1['mae'], 'history': h1},
        'v2': {'mse': m2['mse'], 'mae': m2['mae'], 'history': h2},
        'winner': 'v2' if m2['mse'] < m1['mse'] else 'v1',
    }
    return summary
