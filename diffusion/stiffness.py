# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix

def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, Na, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)
        con connective (3 ddl* # eleemetn) quel noeus est connecté a quel élément
        jacobien (par élement tous les jacobé evalué aux points de quadrature) 
        det (jacob)  * ne * ngp
        xphys coordonées physique pts de gaus
        w poids de quadrature
        Na fonctions de base évaluées aux points de quadrature
        gN gradients des fonctions de base évalués aux points de quadrature (en coordonnées de  référence)
        kappa_fun  mon K(x)
        rhs_fun mon f(x)
        tag_to_dof mapping des tags de noeuds vers les indices de degrés de liberté globaux
        --> gmsh me sort des points qui sert à rien donc faire un mapping pour les dofs et les coordonnées physiques

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.

    Returns
    -------
    K : lil_matrix (nn x nn)
    F : ndarray (nn,)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            kappa_g = float(kappa_fun(xg))
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                F[Ia] += wg * f_g * N[g, a] * detg

                gradNa = invjacg @ gN[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F

def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof):
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]

            g_neu_g = float(g_neu_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                N_a = N[g, a]
                F[Ia] += wg * g_neu_g * N_a * detg

    return F

def assemble_black_scholes_operator(elemTags, conn, jac, det, xphys, w, Na, gN, sigma, r, tag_to_dof):
    
    ne = len(elemTags) # nombre d'éléments
    ngp = len(w)        # nombre de points de quadrature 
    nloc = int(len(conn) // ne)   # nombre de noeuds par élément
    nn = int(np.max(tag_to_dof) + 1) # nombre total de degrés de liberté (doit être égal au nombre de noeuds uniques)
    # --> nn on prend le mapping et on ajoute les élement car commence à 0  
    
    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N_vals = np.asarray(Na, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)


    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        elemTags_e = conn[e, :]
        dof_indices = tag_to_dof[elemTags_e]
        for g in range(ngp):
            Sg = xphys[e, g, 0]  
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            a_g = 0.5 * sigma**2 * Sg**2
            b_g = r * Sg
            for a in range(nloc):
                Ia = int(dof_indices[a])
                Na = N_vals[g, a]
                gradNa = invjacg @ gN[g, a]
                dNa_ds = gradNa[0]  # Assuming the first component corresponds to the spatial derivative in the 1D case
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    Nb = N_vals[g, b]
                    gradNb = invjacg @ gN[g, b]
                    dNb_ds = gradNb[0]  # Assuming the first component corresponds to the spatial derivative in the 1D case
                    diffusion_term = a_g * dNa_ds * dNb_ds
                    convection_term = b_g * dNb_ds * Na
                    reaction_term = r * Na * Nb
                    
                    K[Ia, Ib] += wg * (diffusion_term + convection_term + reaction_term) * detg
    
    return K, F
                                                        

