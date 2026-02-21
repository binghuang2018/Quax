import jax.numpy as jnp
import numpy as np
import psi4

from .basis_utils import build_basis_set
from .tei import tei_array
from .e3nn_eri import e3nn_eri_array


def _as_geom_array(geom):
    arr = jnp.asarray(geom)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Geometry must have shape (natom, 3).")
    return arr


def build_eri_target(geom, basis, source='quax', e3nn_options=None):
    """
    Build one ERI supervision target.

    Parameters
    ----------
    geom : jnp.ndarray
        Cartesian geometry with shape (natom, 3).
    basis : dict
        Quax basis dictionary.
    source : str
        'quax' for analytic JAX ERI (`tei_array`) or 'e3nn' for self-distillation target.
    e3nn_options : dict
        Options forwarded to `e3nn_eri_array` when source='e3nn'.
    """
    geom = _as_geom_array(geom)
    if source == 'quax':
        return tei_array(geom, basis)
    if source == 'e3nn':
        if e3nn_options is None:
            e3nn_options = {}
        return e3nn_eri_array(geom, basis, options=e3nn_options)
    raise ValueError("source must be 'quax' or 'e3nn'")


def build_e3nn_eri_dataset(geoms, basis, source='quax', e3nn_options=None):
    """
    Construct ERI training dataset from a list/batch of geometries.

    Returns
    -------
    dict
        {
          'geometries': (nconf, natom, 3),
          'targets': (nconf, nbf, nbf, nbf, nbf)
        }
    """
    geoms = jnp.asarray(geoms)
    if geoms.ndim != 3 or geoms.shape[-1] != 3:
        raise ValueError("geoms must have shape (nconf, natom, 3)")

    targets = []
    for i in range(geoms.shape[0]):
        targets.append(build_eri_target(geoms[i], basis, source=source, e3nn_options=e3nn_options))
    return {'geometries': geoms, 'targets': jnp.stack(targets, axis=0)}


def build_dataset_from_psi4_molecule(molecule, basis_name, displacements=None, source='quax', e3nn_options=None):
    """
    Convenience helper: build dataset from one Psi4 molecule and optional displaced geometries.

    Parameters
    ----------
    molecule : psi4.core.Molecule
    basis_name : str
    displacements : np.ndarray or None
        If provided, shape (nconf, natom, 3), absolute geometries in bohr.
        If None, only current molecule geometry is used.
    """
    if not isinstance(molecule, psi4.core.Molecule):
        raise TypeError("molecule must be a psi4.core.Molecule")

    basis = build_basis_set(molecule, basis_name)
    base_geom = jnp.asarray(np.asarray(molecule.geometry()))

    if displacements is None:
        geoms = base_geom[None, :, :]
    else:
        geoms = jnp.asarray(displacements)

    return build_e3nn_eri_dataset(geoms, basis, source=source, e3nn_options=e3nn_options)
