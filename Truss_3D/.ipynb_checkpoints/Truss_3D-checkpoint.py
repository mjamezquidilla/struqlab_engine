from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
# plt.style.use('fivethirtyeight')


class Truss_3D:
    def __init__(self, nodes, elements, supports, forces, elasticity, cross_area):
        """
        Initializes truss class object. User should be aware of its units for consistency of solution.

        Parameters
        ----------

        nodes: dictionary
               nodal name/mark followed by the [x,y] coordinate in an array form.

        elements: dictionary
                  member or element name followed by the connecting nodes in a form of an array.

        supports: dictionary
                  applies supports at the node name/mark followed by the an array support condition as follows:
                  if array == [1,1]: Pin support
                  if array == [0,1]: Roller support free to move abot the x-axis
                  if array == [1,0]: Roller support free to move abot the y-axis

        forces: dictionary
                applies forces at the node name/mark, followed by an array of [x,y] coordinate. Positive values for x-axis indicate right direction. Positive values for y-axis indicate up direction

        elasticity: dictionary
                    member's modulus of elasticity. Member's name/mark followed by its modulus of elasticity

        cross_area: dictionary
                    member's cross-sectional area. Member's name/mark followed by its cross-sectional area

        """

        self.nodes = nodes
        self.elements = elements
        self.supports = supports
        self.forces = forces
        self.elasticity = elasticity
        self.cross_area = cross_area
        self.K_global_ = []
        self.displacements_ = []
        self.reactions_ = []
        self.member_forces_ = []
        self.member_stresses_ = []

    def Direction_Cosines(self, element, nodes, elements):
        from_node = elements[element][0]
        to_node = elements[element][1]
        from_point = nodes[from_node]
        to_point = nodes[to_node]

        x1 = from_point[0]
        y1 = from_point[1]
        z1 = from_point[2]

        x2 = to_point[0]
        y2 = to_point[1]
        z2 = to_point[2]

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

        cx = (x2 - x1) / length
        cy = (y2 - y1) / length
        cz = (z2 - z1) / length

        return (cx, cy, cz, length)

    def Assemble_Stiffness_Matrix(self, element, elasticity, area, nodes, elements):
        cx, cy, cz, L = self.Direction_Cosines(element, nodes, elements)
        k = elasticity[element] * area[element] / L * np.array([[1, -1], [-1, 1]])
        T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
        K = np.transpose(T).dot(k).dot(T)
        return K

    def Assemble_Global_Stiffness(self, K, k, i, j):
        dofs = [3 * i - 3, 3 * i - 2, 3 * i - 1, 3 * j - 3, 3 * j - 2, 3 * j - 1]
        K[np.ix_(dofs, dofs)] += k
        return K

    def Support_Vector(self, restrained_dofs, nodes):
        dofs = np.zeros([3 * len(nodes)])

        for dof in restrained_dofs:
            dofs[dof * 3 - 3] = restrained_dofs[dof][0]
            dofs[dof * 3 - 2] = restrained_dofs[dof][1]
            dofs[dof * 3 - 1] = restrained_dofs[dof][1]

        Support_Vector = []

        for i, dof in enumerate(dofs):
            if dof == 1:
                Support_Vector.append(i + 1)

        return Support_Vector

    def Apply_Boundary_Conditions(self, restrained_dofs, K_global):
        dofs = []

        for j in restrained_dofs:
            dofs.append(j - 1)

        k = np.delete(K_global, obj=dofs, axis=0)
        k_new = np.delete(k, obj=dofs, axis=1)
        return k_new

    def Assemble_Force_Vector(self, forces, restrained_dofs, nodes):
        # Create force vector
        f = np.zeros([3 * len(nodes)])

        # extracts forces along x and y
        for force in forces:
            f[force * 3 - 3] = forces[force][0]
            f[force * 3 - 2] = forces[force][1]
            f[force * 3 - 1] = forces[force][2]

        # create dof list
        dofs = []

        # loops and appends restrained dofs and appends it to the list dof
        for j in restrained_dofs:
            dofs.append(j - 1)

        # removes force vector unecessary rows
        f_new = np.delete(f, obj=dofs, axis=0)
        return f_new

    def Truss_Global_Displacement(self, displacements, Support_Vector, nodes):
        # Create New Support Vector in python indexing
        support_vector_new = [x - 1 for x in Support_Vector]

        # Create Displacement Vector
        displacement_vector = np.zeros(3 * len(nodes))

        # Creating global displacement vector that looks for all elements in the support vector and replacing the with value of 0 of that particular index
        j = 0

        # Looping displacement vectors indexes and looks for any of the value within the array "Support_Vector". Replaces with 0 if the value of the index is equal to any of the support vector.
        for i, _ in enumerate(displacement_vector):
            if np.any(np.in1d(support_vector_new, i)):
                displacement_vector[i] = 0
            else:
                displacement_vector[i] = displacements[j]
                j += 1

        return displacement_vector

    def Solve_Reactions(self, K_global, displacement_vector):
        return np.round(K_global.dot(displacement_vector), 5)

    def Element_Displacement(self, element_number, global_displacement, elements):
        fromNode = elements[element_number][0]
        toNode = elements[element_number][1]

        u = [
            3 * fromNode - 2,
            3 * fromNode - 1,
            3 * fromNode,
            3 * toNode - 2,
            3 * toNode - 1,
            3 * toNode,
        ]

        elem_displacements = []

        for _, u_node in enumerate(u):
            elem_displacements.append(global_displacement[u_node - 1])

        return np.round(elem_displacements, 5)

    def Solve_Member_Force(
        self, element, displacement_vector, area, elasticity, nodes, elements
    ):
        cx, cy, cz, L = self.Direction_Cosines(element, nodes, elements)
        T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
        u = T.dot(displacement_vector[element - 1])
        k = elasticity[element] * area[element] / L * np.array([[1, -1], [-1, 1]])
        member_force = k.dot(u)
        return member_force  # 1st vector: positive = compression, negative = Tension

    def Solve_Member_Force_Component(
        self, element, displacement_vector, area, elasticity, nodes, elements
    ):
        cx, cy, cz, L = self.Direction_Cosines(element, nodes, elements)
        T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
        u = T.dot(displacement_vector[element - 1])
        k = elasticity[element] * area[element] / L * np.array([[1, -1], [-1, 1]])
        member_force = k.dot(u)
        member_force_component = np.transpose(T).dot(member_force)
        return member_force_component  # 1st vector: positive = compression, negative = Tension

    def Solve_Member_Stress(
        self, element, displacement_vector, elasticity, nodes, elements
    ):
        cx, cy, cz, L = self.Direction_Cosines(element, nodes, elements)
        T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
        u = T.dot(displacement_vector[element - 1])
        k = elasticity[element] / L * np.array([[1, -1], [-1, 1]])
        member_stress = k.dot(u)
        return member_stress  # 1st vector: positive = compression, negative = Tension

    def Solve_Member_Stress_Component(
        self, element, displacement_vector, elasticity, nodes, elements
    ):
        cx, cy, cz, L = self.Direction_Cosines(element, nodes, elements)
        T = np.array([[cx, cy, cz, 0, 0, 0], [0, 0, 0, cx, cy, cz]])
        u = T.dot(displacement_vector[element - 1])
        k = elasticity[element] / L * np.array([[1, -1], [-1, 1]])
        member_stress = k.dot(u)
        member_stress_component = np.transpose(T).dot(member_stress)
        return member_stress_component  # 1st vector: positive = compression, negative = Tension

    def __Displacements(self, displacements):
        displacements = {
            key + 1: displacements[key] for (key, _) in enumerate(displacements + 1)
        }

        displacements_dict = {}

        for displacement in range(1, int(len(displacements) / 3 + 1)):
            displacements_dict.update(
                {
                    displacement: [
                        displacements[3 * displacement - 2],
                        displacements[3 * displacement - 1],
                        displacements[3 * displacement],
                    ]
                }
            )

        return displacements_dict

    def __Reactions(self, reactions, supports):
        forces_dict = {key + 1: reactions[key] for (key, _) in enumerate(reactions + 1)}

        reactions_dict = {}
        for support in supports:
            reactions_dict.update(
                {
                    support: [
                        forces_dict[3 * support - 2],
                        forces_dict[3 * support - 1],
                        forces_dict[3 * support],
                    ]
                }
            )

        return reactions_dict

    def Solve(self):
        """
        Solves the 3D Truss.

        Output Parameters
        -----------------

        displacements_: dictionary
                        global displacement of each node. name of each node accompanied by its displacement values along x and y respectively

        reactions_: dictionary
                    global reactions of the truss. name of each node accompanied by its force values along x and y respectively

        member_forces_: dictionary
                        member forces of the truss. name of each member accompanied by its force values. positive (+) values are in tension and negative (-) values are in compression

        member_stresses_: dictionary
                          member stresses of the truss. name of each member accompanied by its stress values. positive (+) values are in tension and negative (-) values are in compression

        K_global: numpy array
                  returns the Global Stiffness Matrix of the Truss

        """

        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        forces = self.forces
        elasticity = self.elasticity
        cross_area = self.cross_area

        # Step 1: Get all Member Lengths
        member_lengths = []

        for element in elements:
            _, _, _, L = self.Direction_Cosines(element, nodes, elements)
            member_lengths.append(L)

        # Step 2: Assemble Stiffness Matrix for All members
        k_elems = []

        for element in elements:
            k_elems.append(
                self.Assemble_Stiffness_Matrix(
                    element, elasticity, cross_area, nodes, elements
                )
            )

        # Step 3: Assemble Global Stiffness Matrix
        K_global = np.zeros([3 * len(nodes), 3 * len(nodes)])

        for i, _ in enumerate(k_elems):
            K_global = self.Assemble_Global_Stiffness(
                K_global, k_elems[i], elements[i + 1][0], elements[i + 1][1]
            )

        # Step 4: Apply Boundary conditions to reduce the Global Stiffness Matrix
        Support_Vector = self.Support_Vector(supports, nodes)
        K_new = self.Apply_Boundary_Conditions(Support_Vector, K_global)

        # Step 5: Reduce Force Vector
        f_new = self.Assemble_Force_Vector(forces, Support_Vector, nodes)

        # Step 6: Solve for Displacement
        displacements = np.linalg.inv(K_new).dot(f_new.transpose())

        # Step 7: Create Global Displacement Vector
        global_displacements = self.Truss_Global_Displacement(
            displacements, Support_Vector, nodes
        )

        # Step 8: Solve for Reactions
        reactions = self.Solve_Reactions(K_global, global_displacements)

        # Step 9: Solve Member Displacements
        element_displacements = []

        for element in elements:
            element_displacements.append(
                self.Element_Displacement(element, global_displacements, elements)
            )

        # Step 10A: Solve Member Forces
        member_forces = []

        for element in elements:
            member_forces.append(
                self.Solve_Member_Force(
                    element,
                    element_displacements,
                    cross_area,
                    elasticity,
                    nodes,
                    elements,
                )
            )
        member_forces = {key: member_forces[key - 1][1] for key in elements}

        # Step 10B: Solve Member Forces in Components
        member_forces_components = []

        for element in elements:
            member_forces_components.append(
                self.Solve_Member_Force_Component(
                    element,
                    element_displacements,
                    cross_area,
                    elasticity,
                    nodes,
                    elements,
                )
            )
        member_forces_components = {
            key: member_forces_components[key - 1] for key in elements
        }

        # Step 11A: Solve Member Stresses
        member_stresses = []

        for element in elements:
            member_stresses.append(
                self.Solve_Member_Stress(
                    element, element_displacements, elasticity, nodes, elements
                )
            )
        member_stresses = {key: member_stresses[key - 1][1] for key in elements}

        # Step 11B: Solve Member Stresses in Components
        member_stresses_components = []

        for element in elements:
            member_stresses_components.append(
                self.Solve_Member_Stress_Component(
                    element, element_displacements, elasticity, nodes, elements
                )
            )
        member_stresses_components = {
            key: member_stresses_components[key - 1] for key in elements
        }

        # Variable lists

        self.displacements_ = self.__Displacements(np.round(global_displacements, 5))
        self.reactions_ = self.__Reactions(reactions, supports)
        self.member_forces_ = member_forces
        self.member_stresses_ = member_stresses
        self.member_forces_components_ = member_forces_components
        self.member_stresses_components_ = member_stresses_components

        self.K_global_ = K_global

        lengths = {}
        for key, length in enumerate(member_lengths):
            lengths.update({key + 1: length})
        self.member_lengths_ = lengths

        print(
            "Positive Stress/Force is in Tension, Negative Stress/Force is in Compression"
        )

    def Draw_Truss_Setup(self, figure_size=[7, 7], length_of_arrow=2):
        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        forces = self.forces

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection="3d")

        self.__scale_plot(ax)

        offset = 0.01

        # Plotting Members
        for element in elements:
            from_point = elements[element][0]
            to_point = elements[element][1]

            from_node_x = nodes[from_point][0]
            from_node_y = nodes[from_point][1]
            from_node_z = nodes[from_point][2]
            to_node_x = nodes[to_point][0]
            to_node_y = nodes[to_point][1]
            to_node_z = nodes[to_point][2]

            x = [from_node_x, to_node_x]
            y = [from_node_y, to_node_y]
            z = [from_node_z, to_node_z]

            ax.plot(x, y, z, marker="o", linewidth=1, c="black", zorder=5)

        # Plotting Supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]
            support_z = supports[support][2]

            x = nodes[support][0]
            y = nodes[support][1]
            z = nodes[support][2]

            if support_x == 1 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="^", s=200, c="r", zorder=2)
            elif support_x == 0 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="o", s=200, c="r", zorder=2)
            else:
                ax.scatter(x, y, z, marker="o", s=200, c="y", zorder=2)

        # plotting node labels
        for node in nodes:
            x = nodes[node][0]
            y = nodes[node][1]
            z = nodes[node][2]
            ax.text(x + offset, y + offset, z + offset, node, zorder=10, c="black")

        # plotting member labels
        for element in elements:
            from_point = elements[element][0]
            to_point = elements[element][1]

            from_node_x = nodes[from_point][0]
            from_node_y = nodes[from_point][1]
            from_node_z = nodes[from_point][2]
            to_node_x = nodes[to_point][0]
            to_node_y = nodes[to_point][1]
            to_node_z = nodes[to_point][2]

            x = [from_node_x, to_node_x]
            y = [from_node_y, to_node_y]
            z = [from_node_z, to_node_z]

            middle_point_x = abs((x[1] - x[0]) / 2) + min(x[0], x[1])
            middle_point_y = abs((y[1] - y[0]) / 2) + min(y[0], y[1])
            middle_point_z = abs((z[1] - z[0]) / 2) + min(z[0], z[1])

            ax.text(
                middle_point_x + offset,
                middle_point_y + offset,
                middle_point_z + offset,
                element,
                zorder=10,
                c="b",
            )

        # plotting force vectors
        # loop all x-direction forces
        for force in forces:
            x = nodes[force][0]
            y = nodes[force][1]
            z = nodes[force][2]

            f_x = forces[force][0]
            f_y = forces[force][1]
            f_z = forces[force][2]

            # plot arrow x-direction
            if f_x > 0:
                ax.quiver(
                    x,
                    y,
                    z,
                    -1,
                    0,
                    0,
                    pivot="tip",
                    length=length_of_arrow,
                    colors="r",
                    normalize=True,
                )
                ax.text(x + length_of_arrow, y, z, f_x, color="r")
            elif f_x < 0:
                ax.quiver(
                    x,
                    y,
                    z,
                    1,
                    0,
                    0,
                    pivot="tip",
                    length=length_of_arrow,
                    colors="r",
                    normalize=True,
                )
                ax.text(x - length_of_arrow, y, z, f_x, color="r")
            else:
                pass

            # plot arrow y-direction
            if f_y > 0:
                ax.quiver(
                    x,
                    y,
                    z,
                    0,
                    -1,
                    0,
                    pivot="tip",
                    length=length_of_arrow,
                    colors="b",
                    normalize=True,
                )
                ax.text(x, y + length_of_arrow, z, f_y, color="b")
            elif f_y < 0:
                ax.quiver(
                    x,
                    y,
                    z,
                    0,
                    1,
                    0,
                    pivot="tip",
                    length=length_of_arrow,
                    colors="b",
                    normalize=True,
                )
                ax.text(x, y - length_of_arrow, z, f_y, color="b")
            else:
                pass

            # plot arrow z-direction
            if f_z > 0:
                ax.quiver(
                    x,
                    y,
                    z,
                    0,
                    0,
                    1,
                    pivot="tip",
                    length=length_of_arrow,
                    colors="g",
                    normalize=True,
                )
                ax.text(x, y, z - length_of_arrow, f_z, color="g")
            elif f_z < 0:
                ax.quiver(
                    x,
                    y,
                    z,
                    0,
                    0,
                    -1,
                    pivot="tip",
                    length=length_of_arrow,
                    colors="g",
                    normalize=True,
                )
                ax.text(x, y, z + length_of_arrow, f_z, color="g")
            else:
                pass

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

    def Draw_Truss_Displacements(self, magnification_factor=100, figure_size=[7, 7]):
        nodes = self.nodes
        elements = self.elements
        supports = self.supports

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection="3d")

        self.__scale_plot(ax)

        offset = 0.01

        # Plotting Old Members
        for element in elements:
            from_point = elements[element][0]
            to_point = elements[element][1]

            from_node_x = nodes[from_point][0]
            from_node_y = nodes[from_point][1]
            from_node_z = nodes[from_point][2]
            to_node_x = nodes[to_point][0]
            to_node_y = nodes[to_point][1]
            to_node_z = nodes[to_point][2]

            x = [from_node_x, to_node_x]
            y = [from_node_y, to_node_y]
            z = [from_node_z, to_node_z]

            ax.plot(
                x,
                y,
                z,
                marker="o",
                linewidth=1,
                c="black",
                zorder=5,
                linestyle="--",
                alpha=0.10,
            )

        # Plotting Node Displacements
        new_nodes = {}
        for _ in self.displacements_:
            for node in nodes:
                x_dist = (
                    self.displacements_[node][0] * magnification_factor + nodes[node][0]
                )
                y_dist = (
                    self.displacements_[node][1] * magnification_factor + nodes[node][1]
                )
                z_dist = (
                    self.displacements_[node][2] * magnification_factor + nodes[node][2]
                )
                new_nodes.update({node: [x_dist, y_dist, z_dist]})

        # Plotting Members
        for element in elements:
            from_point = elements[element][0]
            to_point = elements[element][1]

            from_node_x = new_nodes[from_point][0]
            from_node_y = new_nodes[from_point][1]
            from_node_z = new_nodes[from_point][2]
            to_node_x = new_nodes[to_point][0]
            to_node_y = new_nodes[to_point][1]
            to_node_z = new_nodes[to_point][2]

            x = [from_node_x, to_node_x]
            y = [from_node_y, to_node_y]
            z = [from_node_z, to_node_z]

            ax.plot(x, y, z, marker="o", linewidth=1, c="black", zorder=5)

        # Plotting Supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]
            support_z = supports[support][2]

            x = new_nodes[support][0]
            y = new_nodes[support][1]
            z = new_nodes[support][2]

            if support_x == 1 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="^", s=200, c="r", zorder=2)
            elif support_x == 0 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="o", s=200, c="r", zorder=2)
            else:
                ax.scatter(x, y, z, marker="o", s=200, c="y", zorder=2)

        # plotting node labels
        for node in nodes:
            x = nodes[node][0]
            y = nodes[node][1]
            z = nodes[node][2]
            ax.text(x + offset, y + offset, z + offset, node, zorder=10, c="black")

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

    def __scale_plot(self, ax):
        nodes = self.nodes

        X = []
        Y = []
        Z = []

        for i in nodes:
            X.append(nodes[i][0])
            Y.append(nodes[i][1])
            Z.append(nodes[i][2])

        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)

        max_range = (
            np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
            / 2.0
        )
        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def Draw_Truss_Axial_Force_Map(
        self,
        figure_size=None,
        linewidth=2,
        grid=True,
        color_bar_orientation="vertical",
        color_bar_padding=0.05,
        show_member_label=True,
    ):
        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        member_forces = self.member_forces_

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection="3d")

        self.__scale_plot(ax)

        # Extract member forces
        forces = []
        for i in member_forces:
            forces.append(member_forces[i])

        normalize = mpl.colors.Normalize(vmin=min(forces), vmax=max(forces))
        colorparams = forces
        colormap = cm.plasma

        # Colorbar setup
        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
        s_map.set_array(colorparams)

        offset = 0.01

        # Plotting Members
        for i, element in enumerate(elements):
            from_point = elements[element][0]
            to_point = elements[element][1]

            from_node_x = nodes[from_point][0]
            from_node_y = nodes[from_point][1]
            from_node_z = nodes[from_point][2]
            to_node_x = nodes[to_point][0]
            to_node_y = nodes[to_point][1]
            to_node_z = nodes[to_point][2]

            x = [from_node_x, to_node_x]
            y = [from_node_y, to_node_y]
            z = [from_node_z, to_node_z]

            color = colormap(normalize(forces[i]))
            ax.plot(x, y, z, marker="o", color=color, zorder=5, linewidth=linewidth)

        # Plotting Supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]
            support_z = supports[support][2]

            x = nodes[support][0]
            y = nodes[support][1]
            z = nodes[support][2]

            if support_x == 1 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="^", s=200, c="r", zorder=2)
            elif support_x == 0 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="o", s=200, c="r", zorder=2)
            else:
                ax.scatter(x, y, z, marker="o", s=200, c="y", zorder=2)

        # plotting member labels
        if show_member_label == True:
            for element in elements:
                from_point = elements[element][0]
                to_point = elements[element][1]

                from_node_x = nodes[from_point][0]
                from_node_y = nodes[from_point][1]
                from_node_z = nodes[from_point][2]
                to_node_x = nodes[to_point][0]
                to_node_y = nodes[to_point][1]
                to_node_z = nodes[to_point][2]

                x = [from_node_x, to_node_x]
                y = [from_node_y, to_node_y]
                z = [from_node_z, to_node_z]

                middle_point_x = abs((x[1] - x[0]) / 2) + min(x[0], x[1])
                middle_point_y = abs((y[1] - y[0]) / 2) + min(y[0], y[1])
                middle_point_z = abs((z[1] - z[0]) / 2) + min(z[0], z[1])

                ax.text(
                    middle_point_x + offset,
                    middle_point_y + offset,
                    middle_point_z + offset,
                    element,
                    zorder=10,
                    c="b",
                )

        cbar = plt.colorbar(
            s_map,
            orientation=color_bar_orientation,
            extend="both",
            shrink=1,
            pad=color_bar_padding,
        )
        cbar.set_label(label="Force: (+) Tension, (-) Compression")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

    def Draw_Truss_Axial_Stress_Map(
        self,
        figure_size=None,
        linewidth=2,
        grid=True,
        color_bar_orientation="vertical",
        color_bar_padding=0.05,
        show_member_label=True,
    ):
        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        member_stresses = self.member_stresses_

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection="3d")

        self.__scale_plot(ax)

        # Extract member forces
        forces = []
        for i in member_stresses:
            forces.append(member_stresses[i])

        normalize = mpl.colors.Normalize(vmin=min(forces), vmax=max(forces))
        colorparams = forces
        colormap = cm.plasma

        # Colorbar setup
        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
        s_map.set_array(colorparams)

        offset = 0.01

        # Plotting Members
        for i, element in enumerate(elements):
            from_point = elements[element][0]
            to_point = elements[element][1]

            from_node_x = nodes[from_point][0]
            from_node_y = nodes[from_point][1]
            from_node_z = nodes[from_point][2]
            to_node_x = nodes[to_point][0]
            to_node_y = nodes[to_point][1]
            to_node_z = nodes[to_point][2]

            x = [from_node_x, to_node_x]
            y = [from_node_y, to_node_y]
            z = [from_node_z, to_node_z]

            color = colormap(normalize(forces[i]))
            ax.plot(x, y, z, marker="o", color=color, zorder=5, linewidth=linewidth)

        # Plotting Supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]
            support_z = supports[support][2]

            x = nodes[support][0]
            y = nodes[support][1]
            z = nodes[support][2]

            if support_x == 1 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="^", s=200, c="r", zorder=2)
            elif support_x == 0 and support_y == 1 and support_z == 1:
                ax.scatter(x, y, z, marker="o", s=200, c="r", zorder=2)
            else:
                ax.scatter(x, y, z, marker="o", s=200, c="y", zorder=2)

        # plotting member labels
        if show_member_label == True:
            for element in elements:
                from_point = elements[element][0]
                to_point = elements[element][1]

                from_node_x = nodes[from_point][0]
                from_node_y = nodes[from_point][1]
                from_node_z = nodes[from_point][2]
                to_node_x = nodes[to_point][0]
                to_node_y = nodes[to_point][1]
                to_node_z = nodes[to_point][2]

                x = [from_node_x, to_node_x]
                y = [from_node_y, to_node_y]
                z = [from_node_z, to_node_z]

                middle_point_x = abs((x[1] - x[0]) / 2) + min(x[0], x[1])
                middle_point_y = abs((y[1] - y[0]) / 2) + min(y[0], y[1])
                middle_point_z = abs((z[1] - z[0]) / 2) + min(z[0], z[1])

                ax.text(
                    middle_point_x + offset,
                    middle_point_y + offset,
                    middle_point_z + offset,
                    element,
                    zorder=10,
                    c="b",
                )

        cbar = plt.colorbar(
            s_map,
            orientation=color_bar_orientation,
            extend="both",
            shrink=1,
            pad=color_bar_padding,
        )
        cbar.set_label(label="Stress: (+) Tension, (-) Compression")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

