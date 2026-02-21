#!/usr/bin/env python3
"""
Standalone NH3/SVP e3nn-ERI demo without psi4 dependency.

Requirements:
  pip install jax jaxlib pyscf matplotlib

This script:
1) builds NH3 geometries
2) gets ERI + finite-difference ERI gradients from PySCF
3) trains a simple B[pqQ]-factorized ERI model
4) evaluates test loss
5) plots learning curve (test loss vs training set size)
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from pyscf import gto


def rbf(r, n=8, rmax=8.0):
    centers = jnp.linspace(0.0, rmax, n)
    gamma = 2.0 / (centers[1] - centers[0] + 1e-8) ** 2
    return jnp.exp(-gamma * (r[..., None] - centers) ** 2)


def direction_features(vec):
    r = jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12
    u = vec / r
    x, y, z = u[..., 0:1], u[..., 1:2], u[..., 2:3]
    quad = jnp.concatenate([x * x, y * y, z * z, x * y, x * z, y * z], axis=-1)
    return jnp.concatenate([jnp.ones_like(x), x, y, z, quad], axis=-1)


def pyscf_to_basis_meta(mol):
    ao_atom = []
    ao_am = []
    ao_exp = []
    ao_coef = []
    for ib in range(mol.nbas):
        am = int(mol.bas_angular(ib))
        atom = int(mol.bas_atom(ib))
        exps = np.asarray(mol.bas_exp(ib), dtype=float)
        ctr = np.asarray(mol.bas_ctr_coeff(ib), dtype=float)
        if ctr.ndim == 1:
            ctr = ctr[:, None]
        ncart = (am + 1) * (am + 2) // 2
        for ic in range(ctr.shape[1]):
            ao_atom.extend([atom] * ncart)
            ao_am.extend([am] * ncart)
            ao_exp.extend([float(np.mean(exps))] * ncart)
            ao_coef.extend([float(np.mean(ctr[:, ic]))] * ncart)
    return (
        jnp.asarray(np.array(ao_atom, dtype=np.int32)),
        jnp.asarray(np.array(ao_am, dtype=np.int32)),
        jnp.asarray(np.array(ao_exp, dtype=np.float64)),
        jnp.asarray(np.array(ao_coef, dtype=np.float64)),
    )


def build_pair_features(geom, ao_atom, ao_am, ao_exp, ao_coef, rbf_dim=8, rmax=8.0):
    ao_pos = geom[ao_atom]
    am_max = int(jnp.max(ao_am)) + 1
    am_onehot = jax.nn.one_hot(ao_am, am_max)
    ao_scalar = jnp.concatenate([am_onehot, ao_exp[:, None], ao_coef[:, None]], axis=-1)

    rpq = ao_pos[:, None, :] - ao_pos[None, :, :]
    dpq = jnp.linalg.norm(rpq, axis=-1)

    f_p = ao_scalar[:, None, :]
    f_q = ao_scalar[None, :, :]
    f = jnp.concatenate([f_p + f_q, f_p * f_q, rbf(dpq, n=rbf_dim, rmax=rmax), direction_features(rpq)], axis=-1)
    return f


def init_params(in_dim, rank=32, seed=0):
    key = jax.random.PRNGKey(seed)
    W = jax.random.normal(key, (in_dim, rank)) / jnp.sqrt(float(in_dim))
    return {"W": W}


def predict_eri(geom, params, ao_atom, ao_am, ao_exp, ao_coef):
    feat = build_pair_features(geom, ao_atom, ao_am, ao_exp, ao_coef)
    B = jnp.tanh(jnp.einsum('pqf,fQ->pqQ', feat, params['W']))
    B = 0.5 * (B + jnp.swapaxes(B, 0, 1))
    G = jnp.einsum('pqQ,rsQ->pqrs', B, B)
    G = 0.25 * (G + G.transpose(1, 0, 2, 3) + G.transpose(0, 1, 3, 2) + G.transpose(2, 3, 0, 1))
    return G


def build_mol(symbols, geom_bohr, basis='svp'):
    atom = ';'.join([f"{s} {x} {y} {z}" for s, (x, y, z) in zip(symbols, geom_bohr)])
    return gto.M(atom=atom, basis=basis, unit='Bohr', cart=True, verbose=0)


def eri_and_grad_fd(symbols, geom_bohr, basis='svp', h=1e-3):
    mol = build_mol(symbols, geom_bohr, basis=basis)
    eri0 = mol.intor('int2e', aosym='s1')
    grad = np.zeros((geom_bohr.shape[0], 3) + eri0.shape, dtype=np.float64)
    for a in range(geom_bohr.shape[0]):
        for c in range(3):
            gp = np.array(geom_bohr, dtype=np.float64)
            gm = np.array(geom_bohr, dtype=np.float64)
            gp[a, c] += h
            gm[a, c] -= h
            ep = build_mol(symbols, gp, basis=basis).intor('int2e', aosym='s1')
            em = build_mol(symbols, gm, basis=basis).intor('int2e', aosym='s1')
            grad[a, c] = (ep - em) / (2.0 * h)
    return eri0, grad, mol


@dataclass
class Dataset:
    X: jnp.ndarray
    Y: jnp.ndarray
    dY: jnp.ndarray
    ao_meta: tuple


def make_dataset(symbols, geoms, basis='svp', fd_step=1e-3):
    Ys, dYs = [], []
    ao_meta = None
    for g in geoms:
        eri, d_eri, mol = eri_and_grad_fd(symbols, g, basis=basis, h=fd_step)
        Ys.append(eri)
        dYs.append(d_eri)
        if ao_meta is None:
            ao_meta = pyscf_to_basis_meta(mol)
    return Dataset(jnp.asarray(geoms), jnp.asarray(np.asarray(Ys)), jnp.asarray(np.asarray(dYs)), ao_meta)


def split_dataset(ds, seed=0):
    n = ds.X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    ntr = max(2, int(0.7 * n))
    nva = max(1, int(0.15 * n))
    nte = max(1, n - ntr - nva)
    ntr = n - nva - nte
    tr, va, te = idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
    return (
        Dataset(ds.X[tr], ds.Y[tr], ds.dY[tr], ds.ao_meta),
        Dataset(ds.X[va], ds.Y[va], ds.dY[va], ds.ao_meta),
        Dataset(ds.X[te], ds.Y[te], ds.dY[te], ds.ao_meta),
    )


def sample_loss(params, x, y, dy, ao_meta, grad_weight=0.2):
    ao_atom, ao_am, ao_exp, ao_coef = ao_meta
    pred = predict_eri(x, params, ao_atom, ao_am, ao_exp, ao_coef)
    eri_loss = jnp.mean((pred - y) ** 2)
    pred_grad = jax.jacfwd(lambda z: predict_eri(z, params, ao_atom, ao_am, ao_exp, ao_coef))(x)
    grad_loss = jnp.mean((pred_grad - dy) ** 2)
    return eri_loss + grad_weight * grad_loss


def train(train_ds, steps=20, lr=3e-3, rank=32, seed=0):
    ao_meta = train_ds.ao_meta
    feat = build_pair_features(train_ds.X[0], *ao_meta)
    params = init_params(feat.shape[-1], rank=rank, seed=seed)

    def batch_loss(p):
        tot = 0.0
        for i in range(train_ds.X.shape[0]):
            tot += sample_loss(p, train_ds.X[i], train_ds.Y[i], train_ds.dY[i], ao_meta)
        return tot / train_ds.X.shape[0]

    grad_fn = jax.jit(jax.grad(batch_loss))
    hist = []
    for _ in range(steps):
        g = grad_fn(params)
        params = {"W": params["W"] - lr * g["W"]}
        hist.append(float(batch_loss(params)))
    return params, hist


def evaluate(params, ds):
    vals = []
    for i in range(ds.X.shape[0]):
        vals.append(float(sample_loss(params, ds.X[i], ds.Y[i], ds.dY[i], ds.ao_meta)))
    return float(np.mean(vals))


def learning_curve(train_ds, test_ds, sample_sizes=(2, 4, 6, 8), steps=20, lr=3e-3, rank=32, seed=0):
    out = []
    for n in sample_sizes:
        n = min(int(n), train_ds.X.shape[0])
        sub = Dataset(train_ds.X[:n], train_ds.Y[:n], train_ds.dY[:n], train_ds.ao_meta)
        p, h = train(sub, steps=steps, lr=lr, rank=rank, seed=seed)
        out.append({"n_train": n, "train_final": h[-1], "test_loss": evaluate(p, test_ds)})
    return out


def main(args):
    rng = np.random.default_rng(args.seed)
    symbols = ['N', 'H', 'H', 'H']
    base = np.array([
        [0.000000, 0.000000, 0.000000],
        [0.000000, 1.780000, 1.200000],
        [1.541000, -0.890000, 1.200000],
        [-1.541000, -0.890000, 1.200000],
    ], dtype=np.float64)
    geoms = np.stack([base + rng.normal(scale=args.disp, size=base.shape) for _ in range(args.n_samples)], axis=0)

    print("Building NH3/SVP dataset with PySCF (ERI + FD gradients) ...")
    ds = make_dataset(symbols, geoms, basis='svp', fd_step=args.fd_step)
    train_ds, val_ds, test_ds = split_dataset(ds, seed=args.seed)

    print("Training baseline model ...")
    p, h = train(train_ds, steps=args.steps, lr=args.lr, rank=args.rank, seed=args.seed)
    print("Validation loss:", evaluate(p, val_ds))
    print("Test loss:", evaluate(p, test_ds))

    curve = learning_curve(train_ds, test_ds, sample_sizes=args.sample_sizes, steps=args.steps, lr=args.lr, rank=args.rank, seed=args.seed)
    print("Learning curve:")
    for row in curve:
        print(row)

    try:
        import matplotlib.pyplot as plt
        xs = [r['n_train'] for r in curve]
        ys = [r['test_loss'] for r in curve]
        plt.plot(xs, ys, marker='o')
        plt.xlabel('Number of training geometries')
        plt.ylabel('Test loss (ERI + grad)')
        plt.title('NH3/SVP learning curve')
        plt.tight_layout()
        plt.savefig(args.out)
        print('Saved curve to', args.out)
    except Exception as exc:
        print('matplotlib unavailable, skip plotting:', exc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--disp', type=float, default=0.08)
    parser.add_argument('--fd-step', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[2, 4, 6, 8])
    parser.add_argument('--out', type=str, default='learning_curve_nh3_svp.png')
    main(parser.parse_args())
