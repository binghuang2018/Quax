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
    B = jnp.tanh(jnp.einsum('pqf,fQ->pqQ', feats, params['W']))
    B = 0.5 * (B + jnp.swapaxes(B, 0, 1))
    G = jnp.einsum('pqQ,rsQ->pqrs', B, B)
    G = 0.25 * (G + G.transpose(1, 0, 2, 3) + G.transpose(0, 1, 3, 2) + G.transpose(2, 3, 0, 1))
    return G


def symmetry_residual_loss(pred_eri):
    t1 = pred_eri - pred_eri.transpose(1, 0, 2, 3)
    t2 = pred_eri - pred_eri.transpose(0, 1, 3, 2)
    t3 = pred_eri - pred_eri.transpose(2, 3, 0, 1)
    return (jnp.mean(t1 ** 2) + jnp.mean(t2 ** 2) + jnp.mean(t3 ** 2)) / 3.0


def coulomb_metric_loss(pred_eri, target_eri, eps=1e-9):
    # Flatten pair indices and compare in Coulomb metric: ||Delta||_C^2 = tr(Delta * C^-1 * Delta * C^-1)
    nbf = pred_eri.shape[0]
    P = nbf * nbf
    dp = pred_eri.reshape(P, P)
    dt = target_eri.reshape(P, P)
    delta = dp - dt
    C = dt + eps * jnp.eye(P)
    Cinv = jnp.linalg.pinv(C)
    metric = Cinv @ delta @ Cinv
    return jnp.mean(metric ** 2)


def two_electron_energy(eri, density):
    # E_ee ~ 0.5 * sum_{pqrs} D_pq D_rs (pq|rs)
    return 0.5 * jnp.einsum('pq,rs,pqrs->', density, density, eri)


def energy_gradient_joint_loss(pred_eri, target_eri, density, pred_grad=None, target_grad=None):
    e_pred = two_electron_energy(pred_eri, density)
    e_tgt = two_electron_energy(target_eri, density)
    loss = (e_pred - e_tgt) ** 2
    if pred_grad is not None and target_grad is not None:
        loss = loss + jnp.mean((pred_grad - target_grad) ** 2)
    return loss


def eri_total_loss(
    params,
    geom,
    basis,
    target_eri,
    density=None,
    w_mse=1.0,
    w_sym=0.1,
    w_coul=0.1,
    w_engrad=0.1,
    rbf_dim=8,
    rmax=6.0,
    l2_reg=1e-8,
):
    pred = e3nn_eri_predict(geom, basis, params, rbf_dim=rbf_dim, rmax=rmax)
    mse = jnp.mean((pred - target_eri) ** 2)
    sym = symmetry_residual_loss(pred)
    coul = coulomb_metric_loss(pred, target_eri)
    engrad = 0.0
    if density is not None:
        engrad = energy_gradient_joint_loss(pred, target_eri, density)

    reg = l2_reg * jnp.mean(params['W'] ** 2)
    total = w_mse * mse + w_sym * sym + w_coul * coul + w_engrad * engrad + reg
    return total, {'mse': mse, 'sym': sym, 'coul': coul, 'engrad': engrad}


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
    density=None,
    w_mse=1.0,
    w_sym=0.1,
    w_coul=0.1,
    w_engrad=0.1,
    target_energy_grad=None,
):
    """JAX training loop with physics-constrained losses."""
    feats = build_pair_features(geom, basis, rbf_dim=rbf_dim, rmax=rmax)
    params = init_e3nn_eri_params(feats.shape[-1], rank=rank, seed=seed)


    def energy_from_geom(g, p):
        eri = e3nn_eri_predict(g, basis, p, rbf_dim=rbf_dim, rmax=rmax)
        return two_electron_energy(eri, density)

    def scalar_loss(p):
        val, parts = eri_total_loss(
            p,
            geom,
            basis,
            target_eri,
            density=density,
            w_mse=w_mse,
            w_sym=w_sym,
            w_coul=w_coul,
            w_engrad=w_engrad,
            rbf_dim=rbf_dim,
            rmax=rmax,
            l2_reg=l2_reg,
        )
        if density is not None and w_engrad > 0.0:
            pred_grad = jax.grad(lambda x: energy_from_geom(x, p))(geom)
            val = val - w_engrad * parts['engrad'] + w_engrad * energy_gradient_joint_loss(
                e3nn_eri_predict(geom, basis, p, rbf_dim=rbf_dim, rmax=rmax),
                target_eri,
                density,
                pred_grad=pred_grad,
                target_grad=target_energy_grad,
            )
        return val

    grad_fn = jax.jit(jax.grad(scalar_loss))
    history = []
    for _ in range(steps):
        grads = grad_fn(params)
        params = {'W': params['W'] - lr * grads['W']}
        history.append(scalar_loss(params))
    return params, jnp.asarray(history)


def evaluate_e3nn_eri(geom, basis, params, target_eri, rbf_dim=8, rmax=6.0):
    """Compute simple evaluation metrics for trained e3nn ERI model."""
    pred = e3nn_eri_predict(geom, basis, params, rbf_dim=rbf_dim, rmax=rmax)
    mse = jnp.mean((pred - target_eri) ** 2)
    mae = jnp.mean(jnp.abs(pred - target_eri))
    sym = symmetry_residual_loss(pred)
    coul = coulomb_metric_loss(pred, target_eri)
    return {"mse": mse, "mae": mae, "sym": sym, "coul": coul, "pred": pred}


def build_self_distillation_target(geom, basis, e3nn_options=None):
    """Utility target builder using current non-trainable e3nn_eri_array output."""
    if e3nn_options is None:
        e3nn_options = {}
    return e3nn_eri_array(geom, basis, options=e3nn_options)
