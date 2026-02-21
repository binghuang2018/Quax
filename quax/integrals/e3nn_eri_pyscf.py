"""PySCF-driven data generation, training/testing, and learning-curve utilities for e3nn ERI models."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .e3nn_eri_train import init_e3nn_eri_params, e3nn_eri_predict

try:
    from pyscf import gto
    HAS_PYSCF = True
except Exception:  # pragma: no cover - environment dependent
    HAS_PYSCF = False


def _require_pyscf():
    if not HAS_PYSCF:
        raise ImportError("PySCF is required for e3nn_eri_pyscf utilities. Install pyscf to use this module.")


def pyscf_to_quax_basis_dict(mol) -> Dict[int, dict]:
    """Convert a PySCF molecule basis representation to Quax basis_dict format."""
    basis_dict = {}
    idx = 0
    shell_id = 0
    for ib in range(mol.nbas):
        am = int(mol.bas_angular(ib))
        atom = int(mol.bas_atom(ib))
        exps = np.asarray(mol.bas_exp(ib)).tolist()
        ctr = np.asarray(mol.bas_ctr_coeff(ib))
        if ctr.ndim == 1:
            ctr = ctr[:, None]
        ncart = (am + 1) * (am + 2) // 2

        for ic in range(ctr.shape[1]):
            basis_dict[shell_id] = {
                'am': am,
                'atom': atom,
                'exp': exps,
                'coef': np.asarray(ctr[:, ic]).tolist(),
                'idx': idx,
                'idx_stride': ncart,
            }
            idx += ncart
            shell_id += 1
    return basis_dict


def build_pyscf_mol(symbols: List[str], geom_bohr: np.ndarray, basis: str):
    _require_pyscf()
    atom_spec = ';'.join(
        [f"{s} {x} {y} {z}" for s, (x, y, z) in zip(symbols, np.asarray(geom_bohr))]
    )
    mol = gto.M(atom=atom_spec, basis=basis, unit='Bohr', cart=True, verbose=0)
    return mol


def pyscf_eri(geom_bohr: np.ndarray, symbols: List[str], basis: str):
    mol = build_pyscf_mol(symbols, geom_bohr, basis)
    eri = mol.intor('int2e', aosym='s1')
    return eri, mol


def pyscf_eri_and_gradient_fd(geom_bohr: np.ndarray, symbols: List[str], basis: str, step: float = 1e-3):
    """Compute ERI and dERI/dR by finite differences on top of PySCF ERIs."""
    eri0, mol0 = pyscf_eri(geom_bohr, symbols, basis)
    nat = geom_bohr.shape[0]
    grad = np.zeros((nat, 3) + eri0.shape, dtype=np.float64)

    for a in range(nat):
        for xyz in range(3):
            gp = np.array(geom_bohr, dtype=np.float64)
            gm = np.array(geom_bohr, dtype=np.float64)
            gp[a, xyz] += step
            gm[a, xyz] -= step
            ep, _ = pyscf_eri(gp, symbols, basis)
            em, _ = pyscf_eri(gm, symbols, basis)
            grad[a, xyz] = (ep - em) / (2.0 * step)

    return eri0, grad, mol0


@dataclass
class EriDataset:
    geometries: jnp.ndarray
    eris: jnp.ndarray
    eri_grads: jnp.ndarray
    basis_dict: dict
    symbols: List[str]


def build_dataset_pyscf(symbols: List[str], geometries_bohr: np.ndarray, basis: str, step: float = 1e-3) -> EriDataset:
    _require_pyscf()
    geoms = np.asarray(geometries_bohr, dtype=np.float64)
    eris = []
    grads = []
    basis_dict = None
    for g in geoms:
        eri, grad, mol = pyscf_eri_and_gradient_fd(g, symbols, basis, step=step)
        eris.append(eri)
        grads.append(grad)
        if basis_dict is None:
            basis_dict = pyscf_to_quax_basis_dict(mol)
    return EriDataset(
        geometries=jnp.asarray(geoms),
        eris=jnp.asarray(np.asarray(eris)),
        eri_grads=jnp.asarray(np.asarray(grads)),
        basis_dict=basis_dict,
        symbols=symbols,
    )


def standardize_dataset(ds: EriDataset):
    x = ds.geometries
    y = ds.eris
    yg = ds.eri_grads
    stats = {
        'x_mean': jnp.mean(x, axis=(0, 1), keepdims=True),
        'x_std': jnp.std(x, axis=(0, 1), keepdims=True) + 1e-8,
        'y_mean': jnp.mean(y, axis=0, keepdims=True),
        'y_std': jnp.std(y, axis=0, keepdims=True) + 1e-8,
        'yg_mean': jnp.mean(yg, axis=0, keepdims=True),
        'yg_std': jnp.std(yg, axis=0, keepdims=True) + 1e-8,
    }
    x_n = (x - stats['x_mean']) / stats['x_std']
    y_n = (y - stats['y_mean']) / stats['y_std']
    yg_n = (yg - stats['yg_mean']) / stats['yg_std']
    return EriDataset(x_n, y_n, yg_n, ds.basis_dict, ds.symbols), stats


def split_dataset(ds: EriDataset, train_ratio=0.7, val_ratio=0.15, seed=0):
    n = ds.geometries.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    ntr = max(1, int(n * train_ratio))
    nval = max(1, int(n * val_ratio))
    nte = max(1, n - ntr - nval)
    ntr = n - nval - nte

    tr, va, te = idx[:ntr], idx[ntr:ntr+nval], idx[ntr+nval:]

    def subset(ii):
        return EriDataset(ds.geometries[ii], ds.eris[ii], ds.eri_grads[ii], ds.basis_dict, ds.symbols)

    return {'train': subset(tr), 'val': subset(va), 'test': subset(te)}


def _sample_loss(params, geom, eri_ref, grad_ref, basis_dict, grad_weight=0.1):
    pred = e3nn_eri_predict(geom, basis_dict, params)
    mse_eri = jnp.mean((pred - eri_ref) ** 2)
    pred_grad = jax.jacfwd(lambda g: e3nn_eri_predict(g, basis_dict, params))(geom)
    mse_grad = jnp.mean((pred_grad - grad_ref) ** 2)
    return mse_eri + grad_weight * mse_grad


def train_e3nn_on_dataset(train_ds: EriDataset, steps=50, lr=5e-3, rank=48, grad_weight=0.1, seed=0):
    # robust init from feature size
    from .e3nn_eri_train import build_pair_features
    feat = build_pair_features(train_ds.geometries[0], train_ds.basis_dict)
    params = init_e3nn_eri_params(feat.shape[-1], rank=rank, seed=seed)

    def batch_loss(p):
        total = 0.0
        for i in range(train_ds.geometries.shape[0]):
            total += _sample_loss(
                p,
                train_ds.geometries[i],
                train_ds.eris[i],
                train_ds.eri_grads[i],
                train_ds.basis_dict,
                grad_weight=grad_weight,
            )
        return total / train_ds.geometries.shape[0]

    grad_fn = jax.jit(jax.grad(batch_loss))
    history = []
    for _ in range(steps):
        g = grad_fn(params)
        params = {'W': params['W'] - lr * g['W']}
        history.append(batch_loss(params))
    return params, jnp.asarray(history)


def evaluate_e3nn_on_dataset(params, ds: EriDataset, grad_weight=0.1):
    losses = []
    for i in range(ds.geometries.shape[0]):
        losses.append(
            _sample_loss(params, ds.geometries[i], ds.eris[i], ds.eri_grads[i], ds.basis_dict, grad_weight=grad_weight)
        )
    return jnp.mean(jnp.asarray(losses))


def learning_curve_vs_samples(splits, sample_sizes=(2, 4, 8, 12), steps=40, lr=5e-3, rank=48, grad_weight=0.1, seed=0):
    train_ds = splits['train']
    test_ds = splits['test']
    out = []
    for n in sample_sizes:
        n_use = min(int(n), train_ds.geometries.shape[0])
        sub = EriDataset(
            train_ds.geometries[:n_use],
            train_ds.eris[:n_use],
            train_ds.eri_grads[:n_use],
            train_ds.basis_dict,
            train_ds.symbols,
        )
        p, h = train_e3nn_on_dataset(sub, steps=steps, lr=lr, rank=rank, grad_weight=grad_weight, seed=seed)
        test_loss = evaluate_e3nn_on_dataset(p, test_ds, grad_weight=grad_weight)
        out.append({'n_train': n_use, 'train_final': float(h[-1]), 'test_loss': float(test_loss)})
    return out


def nh3_svp_demo(n_samples=10, disp=0.08, basis='svp', seed=0):
    """Direct NH3/SVP demo: data build (ERI+grad), train/test, and learning curve."""
    _require_pyscf()
    rng = np.random.default_rng(seed)
    symbols = ['N', 'H', 'H', 'H']
    # rough NH3 geometry in bohr
    base = np.array([
        [0.000000, 0.000000, 0.000000],
        [0.000000, 1.780000, 1.200000],
        [1.541000, -0.890000, 1.200000],
        [-1.541000, -0.890000, 1.200000],
    ])
    geoms = np.stack([base + rng.normal(scale=disp, size=base.shape) for _ in range(n_samples)], axis=0)

    ds = build_dataset_pyscf(symbols, geoms, basis=basis)
    dsn, _ = standardize_dataset(ds)
    splits = split_dataset(dsn, seed=seed)
    curve = learning_curve_vs_samples(splits, sample_sizes=(2, 4, 6, 8), steps=25, lr=3e-3, rank=32, grad_weight=0.2, seed=seed)
    return curve
