import os
import tempfile

import jax.numpy as jnp
import numpy as np
import psi4

from .basis_utils import build_basis_set
from .tei import tei_array
from .e3nn_eri import e3nn_eri_array
from ..constants import libint_imported

if libint_imported:
    from ..external_integrals import TEI
    from ..external_integrals import libint_interface


def _as_geom_array(geom):
    arr = jnp.asarray(geom)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Geometry must have shape (natom, 3).")
    return arr


def _build_eri_target_libint(geom, basis_name, xyz_path):
    if not libint_imported:
        raise RuntimeError("libint source requested but libint interface is not available.")
    libint_interface.initialize(xyz_path, basis_name)
    tei_obj = TEI(basis_name, xyz_path, 0, 'core')
    G = tei_obj.tei(geom.reshape(-1))
    libint_interface.finalize()
    return G


def build_eri_target(geom, basis, source='quax', e3nn_options=None, basis_name=None, xyz_path=None):
    """
    Build one ERI supervision target.

    source:
      - 'quax': analytic JAX ERI (`tei_array`)
      - 'e3nn': self-distillation target (`e3nn_eri_array`)
      - 'libint': libint reference ERI (requires basis_name + xyz_path)
    """
    geom = _as_geom_array(geom)
    if source == 'quax':
        return tei_array(geom, basis)
    if source == 'e3nn':
        if e3nn_options is None:
            e3nn_options = {}
        return e3nn_eri_array(geom, basis, options=e3nn_options)
    if source == 'libint':
        if basis_name is None or xyz_path is None:
            raise ValueError("libint source requires basis_name and xyz_path")
        return _build_eri_target_libint(geom, basis_name, xyz_path)
    raise ValueError("source must be 'quax', 'e3nn', or 'libint'")


def build_e3nn_eri_dataset(geoms, basis, source='quax', e3nn_options=None, basis_name=None, xyz_path=None):
    """Construct ERI training dataset from a batch of geometries."""
    geoms = jnp.asarray(geoms)
    if geoms.ndim != 3 or geoms.shape[-1] != 3:
        raise ValueError("geoms must have shape (nconf, natom, 3)")

    targets = []
    for i in range(geoms.shape[0]):
        targets.append(
            build_eri_target(
                geoms[i],
                basis,
                source=source,
                e3nn_options=e3nn_options,
                basis_name=basis_name,
                xyz_path=xyz_path,
            )
        )
    return {'geometries': geoms, 'targets': jnp.stack(targets, axis=0)}


def compute_dataset_stats(dataset):
    """Compute normalization statistics for geometry/target tensors."""
    X = dataset['geometries']
    Y = dataset['targets']
    x_mean = jnp.mean(X, axis=(0, 1), keepdims=True)
    x_std = jnp.std(X, axis=(0, 1), keepdims=True) + 1e-8
    y_mean = jnp.mean(Y, axis=0, keepdims=True)
    y_std = jnp.std(Y, axis=0, keepdims=True) + 1e-8
    return {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}


def normalize_dataset(dataset, stats=None):
    """Normalize dataset using provided or computed stats."""
    if stats is None:
        stats = compute_dataset_stats(dataset)
    Xn = (dataset['geometries'] - stats['x_mean']) / stats['x_std']
    Yn = (dataset['targets'] - stats['y_mean']) / stats['y_std']
    return {'geometries': Xn, 'targets': Yn, 'stats': stats}


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=0):
    """Split dataset into train/val/test subsets."""
    X, Y = dataset['geometries'], dataset['targets']
    n = X.shape[0]
    if n < 2:
        return {'train': dataset, 'val': dataset, 'test': dataset}

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = max(1, int(n * train_ratio))
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n - n_val - n_test)

    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    if va.size == 0:
        va = te

    out = {
        'train': {'geometries': X[tr], 'targets': Y[tr]},
        'val': {'geometries': X[va], 'targets': Y[va]},
        'test': {'geometries': X[te], 'targets': Y[te]},
    }
    return out


def build_dataset_from_psi4_molecule(molecule, basis_name, displacements=None, source='quax', e3nn_options=None):
    """Build dataset from one Psi4 molecule and optional displaced geometries."""
    if not isinstance(molecule, psi4.core.Molecule):
        raise TypeError("molecule must be a psi4.core.Molecule")

    basis = build_basis_set(molecule, basis_name)
    base_geom = jnp.asarray(np.asarray(molecule.geometry()))

    if displacements is None:
        geoms = base_geom[None, :, :]
    else:
        geoms = jnp.asarray(displacements)

    if source == 'libint':
        with tempfile.NamedTemporaryFile('w', suffix='.xyz', delete=False) as fh:
            tmp_xyz = fh.name
        try:
            molecule.save_xyz_file(tmp_xyz, True)
            return build_e3nn_eri_dataset(
                geoms,
                basis,
                source='libint',
                e3nn_options=e3nn_options,
                basis_name=basis_name,
                xyz_path=tmp_xyz,
            )
        finally:
            if os.path.exists(tmp_xyz):
                os.remove(tmp_xyz)

    return build_e3nn_eri_dataset(geoms, basis, source=source, e3nn_options=e3nn_options)


def prepare_e3nn_eri_pipeline_dataset(
    molecule,
    basis_name,
    displacements,
    source='quax',
    normalize=True,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=0,
    e3nn_options=None,
):
    """End-to-end data pipeline: build -> normalize -> split."""
    dataset = build_dataset_from_psi4_molecule(
        molecule,
        basis_name,
        displacements=displacements,
        source=source,
        e3nn_options=e3nn_options,
    )
    if normalize:
        dataset = normalize_dataset(dataset)
    splits = split_dataset(dataset, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    if 'stats' in dataset:
        splits['stats'] = dataset['stats']
    return splits
