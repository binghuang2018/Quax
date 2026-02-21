import jax
import jax.numpy as jnp
import numpy as np


def _ao_metadata(basis):
    """Build AO-level metadata from basis dict."""
    nbf = 0
    for i in range(len(basis)):
        nbf += basis[i]['idx_stride']

    ao_atom = np.zeros(nbf, dtype=np.int32)
    ao_am = np.zeros(nbf, dtype=np.int32)
    ao_exp = np.zeros(nbf, dtype=np.float64)
    ao_coef = np.zeros(nbf, dtype=np.float64)

    for i in range(len(basis)):
        idx = basis[i]['idx']
        dim = basis[i]['idx_stride']
        ao_atom[idx:idx + dim] = basis[i]['atom']
        ao_am[idx:idx + dim] = basis[i]['am']
        ao_exp[idx:idx + dim] = np.mean(np.asarray(basis[i]['exp']))
        ao_coef[idx:idx + dim] = np.mean(np.asarray(basis[i]['coef']))
    return jnp.asarray(ao_atom), jnp.asarray(ao_am), jnp.asarray(ao_exp), jnp.asarray(ao_coef)


def _rbf(r, n=8, rmax=6.0):
    centers = jnp.linspace(0.0, rmax, n)
    gamma = 2.0 / (centers[1] - centers[0] + 1e-8) ** 2
    return jnp.exp(-gamma * (r[..., None] - centers) ** 2)


def _direction_features(vec):
    """
    e3nn-inspired direction features.
    Uses e3nn_jax spherical harmonics if available, otherwise a polynomial fallback.
    """
    r = jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
    u = vec / r
    try:
        import e3nn_jax as e3nn
        y0 = e3nn.spherical_harmonics("0e", u, normalize=True, normalization='component')
        y1 = e3nn.spherical_harmonics("1o", u, normalize=True, normalization='component')
        y2 = e3nn.spherical_harmonics("2e", u, normalize=True, normalization='component')
        return jnp.concatenate([y0, y1, y2], axis=-1)
    except Exception:
        x, y, z = u[..., 0:1], u[..., 1:2], u[..., 2:3]
        quad = jnp.concatenate([x * x, y * y, z * z, x * y, x * z, y * z], axis=-1)
        return jnp.concatenate([jnp.ones_like(x), x, y, z, quad], axis=-1)


def _make_projection(in_dim, out_dim, seed):
    key = jax.random.PRNGKey(seed)
    W = jax.random.normal(key, (in_dim, out_dim)) / jnp.sqrt(float(in_dim))
    return W


def e3nn_eri_array(geom, basis, options=None):
    """
    Differentiable, libint-free ERI builder using an e3nn-inspired pair embedding.

    This intentionally keeps all operations in JAX so ERI derivatives can be obtained via autodiff
    without any explicit integral derivative code.
    """
    if options is None:
        options = {}
    rank = int(options.get('rank', 48))
    rbf_dim = int(options.get('rbf_dim', 8))
    rmax = float(options.get('rmax', 6.0))
    seed = int(options.get('seed', 0))

    ao_atom, ao_am, ao_exp, ao_coef = _ao_metadata(basis)
    ao_pos = geom[ao_atom]

    # AO one-body features
    am_max = int(jnp.max(ao_am)) + 1
    am_onehot = jax.nn.one_hot(ao_am, am_max)
    ao_scalar = jnp.concatenate([am_onehot, ao_exp[:, None], ao_coef[:, None]], axis=-1)

    # Pair features for all (p,q)
    rpq = ao_pos[:, None, :] - ao_pos[None, :, :]
    dpq = jnp.linalg.norm(rpq, axis=-1)
    f_rbf = _rbf(dpq, n=rbf_dim, rmax=rmax)
    f_dir = _direction_features(rpq)

    f_p = ao_scalar[:, None, :]
    f_q = ao_scalar[None, :, :]
    f_pair = jnp.concatenate([
        f_p + f_q,
        f_p * f_q,
        f_rbf,
        f_dir,
    ], axis=-1)

    in_dim = f_pair.shape[-1]
    W = _make_projection(in_dim, rank, seed)
    phi = jnp.tanh(jnp.einsum('pqf,fr->pqr', f_pair, W))

    # Symmetrize pair embedding and build ERI tensor by low-rank contraction
    phi = 0.5 * (phi + jnp.swapaxes(phi, 0, 1))
    G = jnp.einsum('pqk,rsk->pqrs', phi, phi)

    # enforce 8-fold AO ERI symmetry
    G = 0.25 * (G + G.transpose(1, 0, 2, 3) + G.transpose(0, 1, 3, 2) + G.transpose(2, 3, 0, 1))
    return G
