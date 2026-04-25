import numpy as np
import gmsh


def evaluate_fe_solution_1d(S0, elemType, elemTags, elemNodeTags, dof_coords, U, tag_to_dof):
    """
    Évalue la solution FEM U au point S0 en utilisant les fonctions de base Gmsh.
    Compatible avec des éléments d'ordre 1, 2, 3, ...
    """

    # Nombre de noeuds locaux par élément
    _, _, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    ne = len(elemTags)
    conn = np.asarray(elemNodeTags, dtype=int).reshape(ne, nloc)

    for e in range(ne):
        tags_e = conn[e, :]
        dofs_e = tag_to_dof[tags_e]

        x_e = dof_coords[dofs_e, 0]

        x_min = np.min(x_e)
        x_max = np.max(x_e)

        # Vérifie si S0 est dans cet élément
        if x_min - 1e-12 <= S0 <= x_max + 1e-12:

            # Les deux premiers noeuds Gmsh d'un segment sont les extrémités
            x_left = x_e[0]
            x_right = x_e[1]

            # Coordonnée de référence xi dans [-1, 1]
            xi = 2.0 * (S0 - x_left) / (x_right - x_left) - 1.0

            # Gmsh attend des coordonnées (xi, eta, zeta)
            uvw = [xi, 0.0, 0.0]

            _, bf, _ = gmsh.model.mesh.getBasisFunctions(
                elemType,
                uvw,
                "Lagrange"
            )

            N = np.asarray(bf, dtype=float).reshape(1, nloc)[0]

            Ue = U[dofs_e]

            return float(np.dot(N, Ue))

    raise ValueError(f"S0={S0} est en dehors du domaine du maillage.")