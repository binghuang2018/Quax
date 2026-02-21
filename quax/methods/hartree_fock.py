import jax 
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import psi4

from .ints import compute_integrals
from .energy_utils import nuclear_repulsion, cholesky_orthogonalization


def _jk_build_cpu(G, D):
    jk_build = jax.vmap(
        jax.vmap(lambda x, y: jnp.tensordot(x, y, axes=[(0, 1), (0, 1)]), in_axes=(0, None)),
        in_axes=(0, None),
    )
    JK = 2 * jk_build(G, D)
    JK -= jk_build(G.transpose((0, 2, 1, 3)), D)
    return JK


def _jk_build_multi_gpu(G, D):
    ndev = jax.local_device_count()
    if ndev <= 1:
        return _jk_build_cpu(G, D)

    nbf = G.shape[0]
    pad = (ndev - (nbf % ndev)) % ndev
    if pad:
        G = jnp.pad(G, ((0, pad), (0, 0), (0, 0), (0, 0)))

    nbf_padded = G.shape[0]
    chunk = nbf_padded // ndev

    G_sharded = G.reshape(ndev, chunk, G.shape[1], G.shape[2], G.shape[3])

    @jax.pmap
    def local_jk(g_local, D_global):
        j_part = jax.vmap(
            jax.vmap(lambda x, y: jnp.tensordot(x, y, axes=[(0, 1), (0, 1)]), in_axes=(0, None)),
            in_axes=(0, None),
        )(g_local, D_global)
        k_part = jax.vmap(
            jax.vmap(lambda x, y: jnp.tensordot(x, y, axes=[(0, 1), (0, 1)]), in_axes=(0, None)),
            in_axes=(0, None),
        )(g_local.transpose((0, 2, 1, 3)), D_global)
        return 2 * j_part - k_part

    D_repl = jnp.broadcast_to(D, (ndev, D.shape[0], D.shape[1]))
    JK = local_jk(G_sharded, D_repl).reshape(nbf_padded, nbf_padded)
    if pad:
        JK = JK[:nbf, :nbf]
    return JK


def restricted_hartree_fock(geom, basis_name, xyz_path, nuclear_charges, charge, options, deriv_order=0, return_aux_data=False):
    # Load keyword options
    maxit = options['maxit']
    damping = options['damping']
    damp_factor = options['damp_factor']
    spectral_shift = options['spectral_shift']
    multi_gpu = options.get('multi_gpu', False)
    convergence = 1e-10

    nelectrons = int(jnp.sum(nuclear_charges)) - charge
    ndocc = nelectrons // 2

    S, T, V, G = compute_integrals(geom, basis_name, xyz_path, nuclear_charges, charge, deriv_order, options)
    # Canonical orthogonalization via cholesky decomposition
    A = cholesky_orthogonalization(S)

    nbf = S.shape[0]

    # For slightly shifting eigenspectrum of transformed Fock for degenerate eigenvalues 
    # (JAX cannot differentiate degenerate eigenvalue eigh) 
    if spectral_shift:
        # Shifting eigenspectrum requires lower convergence.
        convergence = 1e-8 
        fudge = jnp.asarray(np.linspace(0, 1, nbf)) * convergence
        shift = jnp.diag(fudge)
    else:
        shift = jnp.zeros_like(S)

    H = T + V
    Enuc = nuclear_repulsion(geom.reshape(-1,3),nuclear_charges)
    D = jnp.zeros_like(H)
    
    def rhf_iter(F,D):
        E_scf = jnp.einsum('pq,pq->', F + H, D) + Enuc
        Fp = jnp.dot(A.T, jnp.dot(F, A))
        Fp = Fp + shift 
        eps, C2 = jnp.linalg.eigh(Fp)
        C = jnp.dot(A,C2)
        Cocc = C[:, :ndocc]
        D = jnp.dot(Cocc, Cocc.T)
        return E_scf, D, C, eps

    iteration = 0
    E_scf = 1.0
    E_old = 0.0
    Dold = jnp.zeros_like(D)
    dRMS = 1.0

    # Converge according to energy and DIIS residual to ensure eigenvalues and eigenvectors are maximally converged.
    # This is crucial for numerical stability for higher order derivatives of correlated methods.
    while ((abs(E_scf - E_old) > convergence) or (dRMS > convergence)):
        E_old = E_scf * 1
        if damping:
            if iteration < 10:
                D = Dold * damp_factor + D * damp_factor
                Dold = D * 1
        # Build JK matrix: 2 * J - K
        if multi_gpu:
            JK = _jk_build_multi_gpu(G, D)
        else:
            JK = _jk_build_cpu(G, D)
        # Build Fock
        F = H + JK
        # Update convergence error
        if iteration > 1:
            diis_e = jnp.einsum('ij,jk,kl->il', F, D, S) - jnp.einsum('ij,jk,kl->il', S, D, F)
            diis_e = A.dot(diis_e).dot(A)
            dRMS = jnp.mean(diis_e**2)**0.5
        # Compute energy, transform Fock and diagonalize, get new density
        E_scf, D, C, eps = rhf_iter(F,D)
        iteration += 1
        if iteration == maxit:
            break
    print(iteration, " RHF iterations performed")

    # If many orbitals are degenerate, warn that higher order derivatives may be unstable 
    tmp = jnp.round(eps,6)
    ndegen_orbs =  tmp.shape[0] - jnp.unique(tmp).shape[0] 
    if (ndegen_orbs / nbf) > 0.20:
        print("Hartree-Fock warning: More than 20% of orbitals have degeneracies. Higher order derivatives may be unstable due to eigendecomposition AD rule")
    if not return_aux_data:
        return E_scf
    else:
        return E_scf, C, eps, G
