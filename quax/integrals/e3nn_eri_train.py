import jax
import jax.numpy as jnp

from .e3nn_eri import _ao_metadata, _rbf, _direction_features, e3nn_eri_array


def build_pair_features(geom, basis, rbf_dim=8, rmax=6.0):
    """Build pair features used by the e3nn-inspired ERI model."""
    ao_atom, ao_am, ao_exp, ao_coef = _ao_metadata(basis)
    ao_pos = geom[ao_atom]

    am_max = int(jnp.max(ao_am)) + 1
    am_onehot = jax.nn.one_hot(ao_am, am_max)
    ao_scalar = jnp.concatenate([am_onehot, ao_exp[:, None], ao_coef[:, None]], axis=-1)

    rpq = ao_pos[:, None, :] - ao_pos[None, :, :]
    dpq = jnp.linalg.norm(rpq, axis=-1)
    f_rbf = _rbf(dpq, n=rbf_dim, rmax=rmax)
    f_dir = _direction_features(rpq)

    f_p = ao_scalar[:, None, :]
    f_q = ao_scalar[None, :, :]
    return jnp.concatenate([f_p + f_q, f_p * f_q, f_rbf, f_dir], axis=-1)


def init_e3nn_eri_params(in_dim, rank=48, seed=0):
    """Initialize trainable projection parameters for e3nn-style ERI model."""
    key = jax.random.PRNGKey(seed)
    W = jax.random.normal(key, (in_dim, rank)) / jnp.sqrt(float(in_dim))
    return {"W": W}


def e3nn_eri_predict(geom, basis, params, rbf_dim=8, rmax=6.0):
    """Predict ERI tensor from trainable parameters."""
    feats = build_pair_features(geom, basis, rbf_dim=rbf_dim, rmax=rmax)
    phi = jnp.tanh(jnp.einsum('pqf,fr->pqr', feats, params['W']))
    phi = 0.5 * (phi + jnp.swapaxes(phi, 0, 1))
    G = jnp.einsum('pqk,rsk->pqrs', phi, phi)
    G = 0.25 * (G + G.transpose(1, 0, 2, 3) + G.transpose(0, 1, 3, 2) + G.transpose(2, 3, 0, 1))
    return G


def eri_mse_loss(params, geom, basis, target_eri, rbf_dim=8, rmax=6.0, l2_reg=1e-8):
    pred = e3nn_eri_predict(geom, basis, params, rbf_dim=rbf_dim, rmax=rmax)
    mse = jnp.mean((pred - target_eri) ** 2)
    reg = l2_reg * jnp.mean(params['W'] ** 2)
    return mse + reg


def train_e3nn_eri(
    geom,
    basis,
    target_eri,
    steps=200,
    lr=1e-2,
    rank=48,
    rbf_dim=8,
    rmax=6.0,
    seed=0,
    l2_reg=1e-8,
):
    """
    Minimal JAX training loop for e3nn-style ERI model.
    Returns trained params and training history.
    """
    feats = build_pair_features(geom, basis, rbf_dim=rbf_dim, rmax=rmax)
    params = init_e3nn_eri_params(feats.shape[-1], rank=rank, seed=seed)

    def loss_fn(p):
        return eri_mse_loss(p, geom, basis, target_eri, rbf_dim=rbf_dim, rmax=rmax, l2_reg=l2_reg)

    grad_fn = jax.jit(jax.grad(loss_fn))
    history = []
    for _ in range(steps):
        grads = grad_fn(params)
        params = {'W': params['W'] - lr * grads['W']}
        history.append(loss_fn(params))
    return params, jnp.asarray(history)


def evaluate_e3nn_eri(geom, basis, params, target_eri, rbf_dim=8, rmax=6.0):
    """Compute simple evaluation metrics for trained e3nn ERI model."""
    pred = e3nn_eri_predict(geom, basis, params, rbf_dim=rbf_dim, rmax=rmax)
    mse = jnp.mean((pred - target_eri) ** 2)
    mae = jnp.mean(jnp.abs(pred - target_eri))
    return {"mse": mse, "mae": mae, "pred": pred}


def build_self_distillation_target(geom, basis, e3nn_options=None):
    """Utility target builder using current non-trainable e3nn_eri_array output."""
    if e3nn_options is None:
        e3nn_options = {}
    return e3nn_eri_array(geom, basis, options=e3nn_options)
