import jax
import jax.numpy as jnp

from .e3nn_eri import _ao_metadata, _rbf, _direction_features


def build_pair_features_v2(geom, basis, rbf_dim=12, rmax=8.0):
    """Richer pair features for e3nn_eri_v2."""
    ao_atom, ao_am, ao_exp, ao_coef = _ao_metadata(basis)
    ao_pos = geom[ao_atom]

    am_max = int(jnp.max(ao_am)) + 1
    am_onehot = jax.nn.one_hot(ao_am, am_max)
    ao_scalar = jnp.concatenate([am_onehot, ao_exp[:, None], ao_coef[:, None]], axis=-1)

    rpq = ao_pos[:, None, :] - ao_pos[None, :, :]
    dpq = jnp.linalg.norm(rpq, axis=-1)
    inv_d = 1.0 / (1.0 + dpq)

    f_rbf = _rbf(dpq, n=rbf_dim, rmax=rmax)
    f_dir = _direction_features(rpq)

    f_p = ao_scalar[:, None, :]
    f_q = ao_scalar[None, :, :]

    return jnp.concatenate([
        f_p + f_q,
        f_p * f_q,
        f_rbf,
        f_dir,
        dpq[..., None],
        inv_d[..., None],
    ], axis=-1)


def init_e3nn_eri_v2_params(in_dim, hidden_dim=128, rank=64, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (in_dim, hidden_dim)) / jnp.sqrt(float(in_dim))
    b1 = jnp.zeros((hidden_dim,))
    w2 = jax.random.normal(k2, (hidden_dim, rank)) / jnp.sqrt(float(hidden_dim))
    b2 = jnp.zeros((rank,))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def e3nn_eri_v2_factors(geom, basis, params, rbf_dim=12, rmax=8.0):
    """Return explicitly parameterized factors B[p,q,Q]."""
    feats = build_pair_features_v2(geom, basis, rbf_dim=rbf_dim, rmax=rmax)
    h = jnp.tanh(jnp.einsum('pqf,fh->pqh', feats, params['w1']) + params['b1'])
    B = jnp.tanh(jnp.einsum('pqh,hr->pqr', h, params['w2']) + params['b2'])
    # Ensure B_pqQ = B_qpQ
    B = 0.5 * (B + jnp.swapaxes(B, 0, 1))
    return B


def e3nn_eri_v2_predict(geom, basis, params, rbf_dim=12, rmax=8.0):
    """
    Explicit PSD-like construction in pair space:
      G[p,q,r,s] = sum_Q B[p,q,Q] * B[r,s,Q]
    """
    B = e3nn_eri_v2_factors(geom, basis, params, rbf_dim=rbf_dim, rmax=rmax)
    G = jnp.einsum('pqQ,rsQ->pqrs', B, B)
    G = 0.25 * (G + G.transpose(1, 0, 2, 3) + G.transpose(0, 1, 3, 2) + G.transpose(2, 3, 0, 1))
    return G


def e3nn_eri_v2_array(geom, basis, options=None):
    """Deterministic v2 prototype with richer embedding and MLP head."""
    if options is None:
        options = {}
    rbf_dim = int(options.get('rbf_dim', 12))
    rmax = float(options.get('rmax', 8.0))
    hidden_dim = int(options.get('hidden_dim', 128))
    rank = int(options.get('rank', 64))
    seed = int(options.get('seed', 0))

    feats = build_pair_features_v2(geom, basis, rbf_dim=rbf_dim, rmax=rmax)
    params = init_e3nn_eri_v2_params(feats.shape[-1], hidden_dim=hidden_dim, rank=rank, seed=seed)
    return e3nn_eri_v2_predict(geom, basis, params, rbf_dim=rbf_dim, rmax=rmax)


def e3nn_eri_v2_loss(params, geom, basis, target_eri, rbf_dim=12, rmax=8.0, l2_reg=1e-8):
    pred = e3nn_eri_v2_predict(geom, basis, params, rbf_dim=rbf_dim, rmax=rmax)
    mse = jnp.mean((pred - target_eri) ** 2)
    reg = l2_reg * (jnp.mean(params['w1'] ** 2) + jnp.mean(params['w2'] ** 2))
    return mse + reg


def train_e3nn_eri_v2(geom, basis, target_eri, steps=200, lr=1e-2, hidden_dim=128, rank=64, rbf_dim=12, rmax=8.0, seed=0, l2_reg=1e-8):
    feats = build_pair_features_v2(geom, basis, rbf_dim=rbf_dim, rmax=rmax)
    params = init_e3nn_eri_v2_params(feats.shape[-1], hidden_dim=hidden_dim, rank=rank, seed=seed)

    def loss_fn(p):
        return e3nn_eri_v2_loss(p, geom, basis, target_eri, rbf_dim=rbf_dim, rmax=rmax, l2_reg=l2_reg)

    grad_fn = jax.jit(jax.grad(loss_fn))
    history = []
    for _ in range(steps):
        grads = grad_fn(params)
        params = {
            'w1': params['w1'] - lr * grads['w1'],
            'b1': params['b1'] - lr * grads['b1'],
            'w2': params['w2'] - lr * grads['w2'],
            'b2': params['b2'] - lr * grads['b2'],
        }
        history.append(loss_fn(params))
    return params, jnp.asarray(history)
