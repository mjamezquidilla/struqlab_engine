from re import S
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

# Colors
global line_color, bg_color, node_color, force_color, support_color
# line_color = '#5171A5'
# bg_color = '#e9e9e9'
# node_color = '#E5446D'
# force_color = '#d64550'
# support_color = '#69995d'

line_color = "#5171A5"
bg_color = "#e9e9e9"
node_color = "#69995d"
force_color = "#d64550"
support_color = "#e5446D"


class Truss_2D:
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

    def Extract_Coordinate_Points(self, element_number, nodes, elements):

        fromNode = elements[element_number][0]
        toNode = elements[element_number][1]

        fromPoint = nodes[fromNode]
        toPoint = nodes[toNode]

        return fromPoint, toPoint

    def __Plane_Truss_Element_Length(self, element):
        """
        This function returns the length of the
        plane truss element whose first node has
        coordinates [x1, y1] and second node has
        coordinates [x2, y2]
        """
        x1 = element[0][0]
        y1 = element[0][1]
        x2 = element[1][0]
        y2 = element[1][1]

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def __Plane_Truss_Element_Stiffness(self, element, E, A, nodes, elements):
        """
        This function returns the element
        stiffness matrix for a plane truss
        element with modulus of elasticity E,
        cross-sectional area A, length L, and
        angle theta [in degrees].
        The size of the element stiffness
        matrix is 4x4.
        """

        X = self.Extract_Coordinate_Points(element, nodes, elements)
        x1 = X[0][0]
        y1 = X[0][1]
        x2 = X[1][0]
        y2 = X[1][1]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L

        T = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, c, s], [0, 0, -s, c]])
        kk = E * A / L
        k_loc = np.array([[kk, 0, -kk, 0], [0, 0, 0, 0], [-kk, 0, kk, 0], [0, 0, 0, 0]])

        T_inv = np.linalg.inv(T)
        y = T_inv.dot(k_loc).dot(T)
        return y

    def __Plane_Truss_Assemble_Global_Stiffness(self, K, k, i, j):
        """
        This function assembles the element stiffness
        matrix k of the plane truss element with nodes
        i and j into the global stiffness matrix K.
        This function returns the global stiffness
        matrix K after the element stiffness matrix
        k is assembled.
        """

        dofs = [2 * i - 2, 2 * i - 1, 2 * j - 2, 2 * j - 1]
        K[np.ix_(dofs, dofs)] += k
        return K

    def __Apply_Boundary_Conditions(self, restrained_dofs, K_global):
        """
        Applies the boundary conditions with respect to the degrees of freedom (dof)
        """
        dofs = []

        for j in restrained_dofs:
            dofs.append(j - 1)

        k = np.delete(K_global, obj=dofs, axis=0)
        k_new = np.delete(k, obj=dofs, axis=1)
        return k_new

    def __Assemble_Force_Vector(self, forces, restrained_dofs, nodes):
        """
        Assembles the reduces force vector with respect to the restrained_dofs
        """
        # Create force vector
        f = np.zeros([2 * len(nodes)])

        # extracts forces along x and y
        for force in forces:
            f[force * 2 - 2] = forces[force][0]
            f[force * 2 - 1] = forces[force][1]

        # create dof list
        dofs = []

        # loops and appends restrained dofs and appends it to the list dof
        for j in restrained_dofs:
            dofs.append(j - 1)

        # removes force vector unecessary rows
        f_new = np.delete(f, obj=dofs, axis=0)
        return f_new

    def __Support_Vector(self, restrained_dofs, nodes):
        dofs = np.zeros([2 * len(nodes)])

        for dof in restrained_dofs:
            dofs[dof * 2 - 2] = restrained_dofs[dof][0]
            dofs[dof * 2 - 1] = restrained_dofs[dof][1]

        __Support_Vector = []

        for i, dof in enumerate(dofs):
            if dof == 1:
                __Support_Vector.append(i + 1)

        return __Support_Vector

    def Solve(self):
        """
        Solves the 2D Truss.

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
        elasticity = self.elasticity
        cross_area = self.cross_area

        # Add multiplication factor
        forces = {}
        for force in self.forces:
            x = self.forces[force][0] * 1000
            y = self.forces[force][1] * 1000
            forces.update({force: [x, y]})

        # Step 1: Determine Length and Material Properties of Truss Members

        # Step 1.1: Extract coordinates of member per member
        elems = []

        for element in elements:
            elems.append(self.Extract_Coordinate_Points(element, nodes, elements))

        # Step 1.2: Compute Length of member
        L = []

        for elem in elems:
            L.append(self.__Plane_Truss_Element_Length(elem))

        # Step 1.3: Assemble Elasticity List
        E = []

        for element in dict(elements):
            E.append(elasticity[element])

        # Step 1.4: Assemble Cross-Sectional Area List
        A = []

        for element in dict(elements):
            A.append(cross_area[element])

        # Step 2: Compute Member Element Matrix
        k_elems = []

        for i, elem in enumerate(elems):
            k_elems.append(
                self.__Plane_Truss_Element_Stiffness(i + 1, E[i], A[i], nodes, elements)
            )

        # Step 3: Compute Global Matrix
        K_global = np.zeros([2 * len(nodes), 2 * len(nodes)])

        for i, _ in enumerate(k_elems):
            K_global = self.__Plane_Truss_Assemble_Global_Stiffness(
                K_global, k_elems[i], elements[i + 1][0], elements[i + 1][1]
            )

        # Step 4: Apply Boundary conditions to reduce the Global Stiffness Matrix
        __Support_Vector = self.__Support_Vector(supports, nodes)
        K_new = self.__Apply_Boundary_Conditions(__Support_Vector, K_global)

        # Step 5: Reduce Force Vector
        # Step 5.1: Remove Forces that share same keys with Support Dictionary
        # shared_key = []
        # for key in self.forces.keys():
        #     if key in self.supports:
        #         shared_key.append(key)
        # for key in shared_key:
        #     forces.pop(key)

        # Step 5.2: Solve for Force Vector
        f_new = self.__Assemble_Force_Vector(forces, __Support_Vector, nodes)

        # Step 6: Solve for Displacement
        displacements = np.linalg.inv(K_new).dot(f_new.transpose())

        # Step 7: Create Global Displacement Vector
        global_displacements = self.__Truss_Global_Displacement(
            displacements, __Support_Vector, nodes
        )

        # Step 8: Solve for Reactions
        reactions = self.__Solve_Reactions(K_global, global_displacements)

        # Step 9: Solve Member Displacements
        __Element_Displacements = []

        for element in elements:
            __Element_Displacements.append(
                self.__Element_Displacement(element, global_displacements, elements)
            )

        # Step 10: Solve Member Forces
        member_forces = []

        for i, elem in enumerate(elems):
            member_forces.append(
                self.__Solve_Member_Forces(
                    E[i], A[i], i + 1, nodes, elements, __Element_Displacements[i]
                )
            )
        member_forces = {key: member_forces[i] for (i, key) in enumerate(elements)}

        # Step 11: Solve Member Stresses
        member_stresses = []

        for i, elem in enumerate(elems):
            member_stresses.append(
                self.__Solve_Member_Stresses(
                    E[i], i + 1, nodes, elements, __Element_Displacements[i]
                )
            )
        member_stresses = {key: member_stresses[i] for (i, key) in enumerate(elements)}

        self.displacements_ = self.__Displacements(np.round(global_displacements, 5))
        self.reactions_ = self.__Reactions(reactions, supports)
        self.member_forces_ = member_forces
        self.member_stresses_ = member_stresses
        self.K_global_ = K_global

        # Divide multiplication factor
        # Displacement
        displacements_new_ = {}
        for i in self.displacements_:
            x = round(self.displacements_[i][0] / 1000, 5)
            y = round(self.displacements_[i][1] / 1000, 5)
            displacements_new_.update({i: [x, y]})
        self.displacements_ = displacements_new_

        # Reactions
        reactions_new_ = {}
        for i in self.reactions_:
            fx = round(self.reactions_[i][0] / 1000, 5)
            fy = round(self.reactions_[i][1] / 1000, 5)
            reactions_new_.update({i: [fx, fy]})
        self.reactions_ = reactions_new_

        # Add Node forces to Reactions in Dictionary
        shared_keys = []
        for key in self.forces.keys():
            if key in self.reactions_:
                shared_keys.append(key)

        if shared_keys:
            reactions = {}
            for key in shared_keys:
                reactions.update(
                    {
                        key: [
                            -x + y
                            for x, y in zip(self.forces[key], self.reactions_[key])
                        ]
                    }
                )
            self.reactions_.update(reactions)

        # Member Forces
        member_forces_new = {}
        for i in self.member_forces_:
            P = round(self.member_forces_[i] / 1000, 5)
            member_forces_new.update({i: P})
        self.member_forces_ = member_forces_new

        # Member Stresses
        member_stresses_new = {}
        for i in self.member_stresses_:
            P = round(self.member_stresses_[i] / 1000, 5)
            member_stresses_new.update({i: P})
        self.member_stresses_ = member_stresses_new

        # Creating a Dictionary of Member Lengths
        member_length_ = {}
        for key, length in enumerate(L):
            member_length_.update({key + 1: round(length, 5)})

        self.member_lengths_ = member_length_

    def __Truss_Global_Displacement(self, displacements, __Support_Vector, nodes):
        # Create New Support Vector in python indexing
        __Support_Vector_new = [x - 1 for x in __Support_Vector]

        # Create Displacement Vector
        displacement_vector = np.zeros(2 * len(nodes))

        # Creating global displacement vector that looks for all elements in the support vector and replacing the with value of 0 of that particular index
        j = 0

        # Looping displacement vectors indexes and looks for any of the value within the array "__Support_Vector". Replaces with 0 if the value of the index is equal to any of the support vector.
        for i, _ in enumerate(displacement_vector):
            if np.any(np.isin(__Support_Vector_new, i)):
                displacement_vector[i] = 0
            else:
                displacement_vector[i] = displacements[j]
                j += 1

        return displacement_vector

    def __Solve_Reactions(self, K_global, displacement_vector):
        return np.round(K_global.dot(displacement_vector))

    def __Solve_Member_Forces(
        self, elasticity, cross_area, element, nodes, elements, displacement
    ):
        X = self.Extract_Coordinate_Points(element, nodes, elements)
        x1 = X[0][0]
        y1 = X[0][1]
        x2 = X[1][0]
        y2 = X[1][1]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L

        kk = elasticity * cross_area / L
        la = np.array([-c, -s, c, s])
        f_member = kk * la.dot(displacement)

        return np.round(f_member, 5)

    def __Element_Displacement(self, element_number, global_displacement, elements):

        fromNode = elements[element_number][0]
        toNode = elements[element_number][1]

        u = [2 * fromNode - 1, 2 * fromNode, 2 * toNode - 1, 2 * toNode]

        elem_displacements = []

        for _, u_node in enumerate(u):
            elem_displacements.append(global_displacement[u_node - 1])

        return np.round(elem_displacements, 5)

    def __Solve_Member_Stresses(
        self, elasticity, element, nodes, elements, displacement
    ):

        X = self.Extract_Coordinate_Points(element, nodes, elements)
        x1 = X[0][0]
        y1 = X[0][1]
        x2 = X[1][0]
        y2 = X[1][1]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L

        kk = elasticity / L
        la = np.array([-c, -s, c, s])
        s_member = kk * la.dot(displacement)

        return np.round(s_member, 5)

    def Draw_Truss_Setup(
        self,
        figure_size=None,
        linewidth=2,
        offset=0.12,
        length_of_arrow=1.0,
        width_of_arrow=0.05,
        arrow_line_width=2,
        grid=False,
    ):
        """
        Draws the Truss as initialized by the class

        Parameters
        ----------

        figure_size: array
                    size of the plot's figure in x, y heights.
        length_of_arrow: float
                        length of force vector arrow. default value is 1.0
        width_of_arrow: float
                        width of force vector arrow head. default value is 0.1
        arrow_line_width: float
                        width of force vector line. default value is 2.0
        grid: boolean
              activates gridlines. default value is False
        """

        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        forces = self.forces

        # plt.figure(figsize = figure_size)
        plt.grid(grid)

        ax = plt.gca()
        ax.set_aspect(aspect="equal")

        # plotting nodes and members
        node_x = []
        node_y = []

        for node in nodes:
            node_x.append(nodes[node][0])
            node_y.append(nodes[node][1])
        ax.scatter(node_x, node_y, color=line_color, zorder=10)

        for node in nodes:
            ax.annotate(
                node,
                (nodes[node][0] + offset, nodes[node][1] + offset),
                zorder=10,
                c=node_color,
            )

        # plotting members
        for element in dict(elements):
            fromPoint, toPoint = self.Extract_Coordinate_Points(
                element, nodes, elements
            )
            x1 = fromPoint[0]
            y1 = fromPoint[1]
            x2 = toPoint[0]
            y2 = toPoint[1]
            ax.scatter([x1, x2], [y1, y2], color=node_color, zorder=10)
            ax.plot(
                [x1, x2],
                [y1, y2],
                marker="o",
                color=line_color,
                zorder=5,
                linewidth=linewidth,
            )
        # plotting supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]

            x = nodes[support][0]
            y = nodes[support][1]

            if support_x == 1 and support_y == 1:
                ax.scatter(x, y, marker="^", s=200, c=support_color, zorder=2)
            elif support_x == 0 and support_y == 1:
                ax.scatter(x, y, marker="o", s=200, c=support_color, zorder=2)
            else:
                ax.scatter(x, y, marker="o", s=200, c=force_color, zorder=2)

        # plotting node labels
        # offset = 0.12

        # plotting member labels
        for element in elements:
            fromNode = elements[element][0]
            toNode = elements[element][1]
            fromPoint = nodes[fromNode]
            toPoint = nodes[toNode]

            middlePoint = [
                abs((toPoint[0] - fromPoint[0]) / 2) + min(fromPoint[0], toPoint[0]),
                abs((toPoint[1] - fromPoint[1]) / 2) + min(fromPoint[1], toPoint[1]),
            ]

            ax.annotate(
                element, (middlePoint[0], middlePoint[1]), zorder=10, c=line_color
            )

        # plotting force vectors
        # loop all x-direction forces
        for force in forces:
            x = nodes[force][0]
            y = nodes[force][1]

            f_x = forces[force][0]
            f_y = forces[force][1]

            # plot arrow x-direction
            if f_x > 0:
                ax.arrow(
                    x - length_of_arrow,
                    y,
                    length_of_arrow,
                    0,
                    shape="full",
                    head_width=width_of_arrow,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_x, ((x - length_of_arrow), y + 0.1), c=force_color)
                # ax.scatter(x - length_of_arrow, y, c='white')
            elif f_x < 0:
                ax.arrow(
                    x + length_of_arrow,
                    y,
                    -length_of_arrow,
                    0,
                    shape="full",
                    head_width=width_of_arrow,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_x, ((x + length_of_arrow), y + 0.1), c=force_color)
                # ax.scatter(x + length_of_arrow,y, c='white')
            else:
                pass

            # loop all y-direction forces
            # plot arrow y-direction
            if f_y > 0:
                ax.arrow(
                    x,
                    y - length_of_arrow,
                    0,
                    length_of_arrow,
                    shape="full",
                    head_width=width_of_arrow,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_y, (x + 0.1, (y - length_of_arrow)), c=force_color)
                # ax.scatter(x,y - length_of_arrow, c='white')
            elif f_y < 0:
                ax.arrow(
                    x,
                    y + length_of_arrow,
                    0,
                    -length_of_arrow,
                    shape="full",
                    head_width=width_of_arrow,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_y, (x + 0.1, (y + length_of_arrow)), c=force_color)
                # ax.scatter(x,y + length_of_arrow, c='white')
            else:
                pass

        # plt.show()

    def __Reactions(self, reactions, supports):

        forces_dict = {key + 1: reactions[key] for (key, _) in enumerate(reactions + 1)}

        reactions_dict = {}
        for support in supports:
            reactions_dict.update(
                {support: [forces_dict[2 * support - 1], forces_dict[2 * support]]}
            )

        return reactions_dict

    def __Displacements(self, displacements):

        displacements = {
            key + 1: displacements[key] for (key, _) in enumerate(displacements + 1)
        }

        displacements_dict = {}

        for displacement in range(1, int(len(displacements) / 2) + 1):
            displacements_dict.update(
                {
                    displacement: [
                        displacements[2 * displacement - 1],
                        displacements[2 * displacement],
                    ]
                }
            )

        return displacements_dict

    def Draw_Truss_Displacements(
        self, linewidth=2, magnification_factor=100, offset=0.12
    ):
        """
        Draws the Truss displacements after solving the truss

        Parameters
        ----------

        figure_size: array
                    size of the plot's figure in x, y heights.
        length_of_arrow: float
                        length of force vector arrow. default value is 1.0
        width_of_arrow: float
                        width of force vector arrow head. default value is 0.1
        arrow_line_width: float
                        width of force vector line. default value is 2.0
        """

        nodes = self.nodes
        elements = self.elements
        supports = self.supports

        new_nodes = {}
        for _ in self.displacements_:
            for node in nodes:
                x_dist = (
                    self.displacements_[node][0] * magnification_factor + nodes[node][0]
                )
                y_dist = (
                    self.displacements_[node][1] * magnification_factor + nodes[node][1]
                )
                new_nodes.update({node: [x_dist, y_dist]})

        # plt.figure(figsize = figure_size)
        ax = plt.gca()
        ax.set_aspect(aspect="equal")

        # Plotting Old nodes
        # plotting nodes and members
        for element in dict(elements):
            fromPoint, toPoint = self.Extract_Coordinate_Points(
                element, nodes, elements
            )
            x1 = fromPoint[0]
            y1 = fromPoint[1]
            x2 = toPoint[0]
            y2 = toPoint[1]
            ax.plot(
                [x1, x2],
                [y1, y2],
                marker="o",
                color=line_color,
                zorder=5,
                linestyle="--",
                alpha=0.50,
                linewidth=linewidth,
            )
            ax.scatter([x1, x2], [y1, y2], color=node_color, zorder=10, alpha=0.50)

        # Plotting New nodes
        # plotting nodes and members
        for element in dict(elements):
            fromPoint, toPoint = self.Extract_Coordinate_Points(
                element, new_nodes, elements
            )
            x1 = fromPoint[0]
            y1 = fromPoint[1]
            x2 = toPoint[0]
            y2 = toPoint[1]
            ax.plot(
                [x1, x2],
                [y1, y2],
                marker="o",
                color=line_color,
                zorder=5,
                linewidth=linewidth,
            )
            ax.scatter([x1, x2], [y1, y2], color=node_color, zorder=10)

        # plotting supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]

            x = new_nodes[support][0]
            y = new_nodes[support][1]

            if support_x == 1 and support_y == 1:
                ax.scatter(x, y, marker="^", s=200, c=support_color, zorder=2)
            elif support_x == 0 and support_y == 1:
                ax.scatter(x, y, marker="o", s=200, c=support_color, zorder=2)
            else:
                ax.scatter(x, y, marker="o", s=200, c=force_color, zorder=2)

        # plotting node labels
        # offset = 0.12

        for node in new_nodes:
            ax.annotate(
                node,
                (new_nodes[node][0] + offset, new_nodes[node][1] + offset),
                zorder=10,
                c=node_color,
            )

        # plotting member labels
        # for element in elements:
        #     fromNode = elements[element][0]
        #     toNode = elements[element][1]
        #     fromPoint = new_nodes[fromNode]
        #     toPoint = new_nodes[toNode]

        #     middlePoint = [abs((toPoint[0] - fromPoint[0])/2) + min(fromPoint[0], toPoint[0]),
        #                    abs((toPoint[1] - fromPoint[1])/2) + min(fromPoint[1], toPoint[1])]

        #     ax.annotate(element, (middlePoint[0], middlePoint[1]), zorder = 10, c = line_color)

        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        # plt.show()

    def Draw_Truss_Axial_Force_Map(
        self,
        figure_size=None,
        linewidth=2,
        grid=False,
        color_bar_orientation="vertical",
        color_bar_padding=0.05,
    ):
        """
        Draws the Truss Axial Force Color Map

        Parameters
        ----------

        figure_size: array
            size of the plot's figure in x, y heights.
        grid: boolean
            activates gridlines. default value is False
        color_bar_orientation: 'vertical', 'horizontal'
            places colorbar at orientation specified. Vertical at the right, Horizontal at the bottom
        """

        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        member_forces = self.member_forces_

        # Extract member forces
        forces = []
        for i in member_forces:
            forces.append(member_forces[i])

        # color_map = cm.get_cmap('PiYG')
        # heat_map = color_map(forces)

        normalize = mpl.colors.Normalize(vmin=min(forces), vmax=max(forces))
        colorparams = forces
        # colormap = cm.plasma
        # colormap = cm.Spectral
        # colormap = cm.PiYG
        # colormap = cm.viridis
        colormap = cm.coolwarm
        # colormap = cm.seismic

        # Colorbar setup
        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
        s_map.set_array(colorparams)

        # plt.figure(figsize = figure_size)
        plt.grid(grid)

        ax = plt.gca()
        ax.set_aspect(aspect="equal")

        # plotting nodes and members
        for i, element in enumerate(dict(elements)):
            fromPoint, toPoint = self.Extract_Coordinate_Points(
                element, nodes, elements
            )
            x1 = fromPoint[0]
            y1 = fromPoint[1]
            x2 = toPoint[0]
            y2 = toPoint[1]
            color = colormap(normalize(forces[i]))
            ax.scatter([x1, x2], [y1, y2], color=node_color, zorder=10)
            ax.plot(
                [x1, x2],
                [y1, y2],
                marker="o",
                color=color,
                zorder=5,
                linewidth=linewidth,
            )

        # plotting supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]

            x = nodes[support][0]
            y = nodes[support][1]

            if support_x == 1 and support_y == 1:
                ax.scatter(x, y, marker="^", s=200, c=support_color, zorder=2)
            elif support_x == 0 and support_y == 1:
                ax.scatter(x, y, marker="o", s=200, c=support_color, zorder=2)
            else:
                ax.scatter(x, y, marker="o", s=200, c=force_color, zorder=2)

        # plotting member labels
        for element in elements:
            fromNode = elements[element][0]
            toNode = elements[element][1]
            fromPoint = nodes[fromNode]
            toPoint = nodes[toNode]

            middlePoint = [
                abs((toPoint[0] - fromPoint[0]) / 2) + min(fromPoint[0], toPoint[0]),
                abs((toPoint[1] - fromPoint[1]) / 2) + min(fromPoint[1], toPoint[1]),
            ]

            ax.annotate(
                element, (middlePoint[0], middlePoint[1]), zorder=10, c=line_color
            )

        # cbar = plt.colorbar(s_map, orientation=color_bar_orientation, extend = 'both', shrink = 0.5, pad=color_bar_padding, spacing = 'proportional', ax=ax)
        # cbar.set_label(label='Force: (+) Tension, (-) Compression')
        # plt.show()

    def Draw_Truss_Axial_Stress_Map(
        self,
        figure_size=None,
        linewidth=2,
        grid=False,
        color_bar_orientation="vertical",
        color_bar_padding=0.05,
        show_member_label=True,
    ):
        """
        Draws the Truss Axial Stress Color Map

        Parameters
        ----------

        figure_size: array
            size of the plot's figure in x, y heights.
        grid: boolean
            activates gridlines. default value is False
        color_bar_orientation: 'vertical', 'horizontal'
            places colorbar at orientation specified. Vertical at the right, Horizontal at the bottom
        """

        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        member_stresses = self.member_stresses_

        # Extract member forces
        stresses = []
        for i in member_stresses:
            stresses.append(member_stresses[i])

        normalize = mpl.colors.Normalize(vmin=min(stresses), vmax=max(stresses))
        colorparams = stresses
        colormap = cm.plasma
        # colormap = cm.Spectral

        # Colorbar setup
        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
        s_map.set_array(colorparams)

        # plt.figure(figsize = figure_size)
        plt.grid(grid)

        ax = plt.gca()
        ax.set_aspect(aspect="equal")

        # plotting nodes and members
        for i, element in enumerate(dict(elements)):
            fromPoint, toPoint = self.Extract_Coordinate_Points(
                element, nodes, elements
            )
            x1 = fromPoint[0]
            y1 = fromPoint[1]
            x2 = toPoint[0]
            y2 = toPoint[1]
            color = colormap(normalize(stresses[i]))
            ax.plot(
                [x1, x2],
                [y1, y2],
                marker="o",
                color=color,
                zorder=5,
                linewidth=linewidth,
            )
            ax.scatter([x1, x2], [y1, y2], color=node_color, zorder=10)

        # plotting supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]

            x = nodes[support][0]
            y = nodes[support][1]

            if support_x == 1 and support_y == 1:
                ax.scatter(x, y, marker="^", s=200, c=support_color, zorder=2)
            elif support_x == 0 and support_y == 1:
                ax.scatter(x, y, marker="o", s=200, c=support_color, zorder=2)
            else:
                ax.scatter(x, y, marker="o", s=200, c=force_color, zorder=2)

        # plotting member labels
        if show_member_label == True:
            for element in elements:
                fromNode = elements[element][0]
                toNode = elements[element][1]
                fromPoint = nodes[fromNode]
                toPoint = nodes[toNode]

                middlePoint = [
                    abs((toPoint[0] - fromPoint[0]) / 2)
                    + min(fromPoint[0], toPoint[0]),
                    abs((toPoint[1] - fromPoint[1]) / 2)
                    + min(fromPoint[1], toPoint[1]),
                ]

                ax.annotate(
                    element, (middlePoint[0], middlePoint[1]), zorder=10, c=line_color
                )

        cbar = ax.colorbar(
            s_map,
            orientation=color_bar_orientation,
            extend="both",
            shrink=1,
            pad=color_bar_padding,
        )
        cbar.set_label(label="Stress: (-) Compression (+) Tension")
        plt.show()

    def Draw_Reactions_(
        self,
        figure_size=None,
        linewidth=2,
        offset=0.12,
        length_of_arrow=1.0,
        arrow_head_size=0.05,
        arrow_line_width=2,
        grid=False,
        show_ext_forces=False,
        darkmode=False,
    ):
        """
        Draws the Truss as initialized by the class

        Parameters
        ----------

        figure_size: array
                    size of the plot's figure in x, y heights.
        length_of_arrow: float
                        length of force vector arrow. default value is 1.0
        arrow_head_width: float
                        width of force vector arrow head. default value is 0.1
        arrow_line_width: float
                        width of force vector line. default value is 2.0
        grid: boolean
              activates gridlines. default value is False
        """

        nodes = self.nodes
        elements = self.elements
        supports = self.supports
        forces = self.forces
        reactions = self.reactions_

        # plt.figure(figsize = figure_size)
        plt.grid(grid)
        ax = plt.gca()

        # plotting nodes and members
        for element in dict(elements):
            fromPoint, toPoint = self.Extract_Coordinate_Points(
                element, nodes, elements
            )
            x1 = fromPoint[0]
            y1 = fromPoint[1]
            x2 = toPoint[0]
            y2 = toPoint[1]
            ax.scatter([x1, x2], [y1, y2], color=node_color, zorder=10)
            ax.plot(
                [x1, x2],
                [y1, y2],
                marker="o",
                color=line_color,
                zorder=5,
                linewidth=linewidth,
            )

        # plotting supports
        for support in supports:
            support_x = supports[support][0]
            support_y = supports[support][1]

            x = nodes[support][0]
            y = nodes[support][1]

            if support_x == 1 and support_y == 1:
                ax.scatter(x, y, marker="^", s=200, c=support_color, zorder=2)
            elif support_x == 0 and support_y == 1:
                ax.scatter(x, y, marker="o", s=200, c=support_color, zorder=2)
            else:
                ax.scatter(x, y, marker="o", s=200, c=force_color, zorder=2)

        # plotting node labels
        # offset = 0.12

        for node in nodes:
            ax.annotate(
                node,
                (nodes[node][0] + offset, nodes[node][1] + offset),
                zorder=10,
                c=node_color,
            )

        # plotting member labels
        # for element in elements:
        #     fromNode = elements[element][0]
        #     toNode = elements[element][1]
        #     from_point = nodes[fromNode]
        #     to_point = nodes[toNode]

        #     middlePoint = [abs((to_point[0] - from_point[0])/2) + min(from_point[0], to_point[0]),
        #                    abs((to_point[1] - from_point[1])/2) + min(from_point[1], to_point[1])]

        #     ax.annotate(element, (middlePoint[0], middlePoint[1]), zorder = 10, c = 'b')

        # plotting force vectors
        # loop all x-direction forces
        if show_ext_forces == True:
            for force in forces:
                x = nodes[force][0]
                y = nodes[force][1]

                f_x = np.round(forces[force][0], 2)
                f_y = np.round(forces[force][1], 2)

                # plot arrow x-direction
                if f_x > 0:
                    ax.arrow(
                        x - length_of_arrow,
                        y,
                        length_of_arrow,
                        0,
                        shape="full",
                        head_width=arrow_head_size,
                        length_includes_head=True,
                        color="r",
                        zorder=15,
                        linewidth=arrow_line_width,
                    )
                    ax.annotate(f_x, ((x - length_of_arrow), y + offset), c=force_color)
                    # ax.scatter(x - length_of_arrow, y, c='white')
                elif f_x < 0:
                    ax.arrow(
                        x + length_of_arrow,
                        y,
                        -length_of_arrow,
                        0,
                        shape="full",
                        head_width=arrow_head_size,
                        length_includes_head=True,
                        color="r",
                        zorder=15,
                        linewidth=arrow_line_width,
                    )
                    ax.annotate(f_x, ((x + length_of_arrow), y + offset), c=force_color)
                    # ax.scatter(x + length_of_arrow,y, c='white')
                else:
                    pass

                # plot arrow y-direction
                if f_y > 0:
                    ax.arrow(
                        x,
                        y - length_of_arrow,
                        0,
                        length_of_arrow,
                        shape="full",
                        head_width=arrow_head_size,
                        length_includes_head=True,
                        color="r",
                        zorder=15,
                        linewidth=arrow_line_width,
                    )
                    ax.annotate(f_y, (x + offset, (y - length_of_arrow)), c=force_color)
                    # ax.scatter(x,y - length_of_arrow, c='white')
                elif f_y < 0:
                    ax.arrow(
                        x,
                        y + length_of_arrow,
                        0,
                        -length_of_arrow,
                        shape="full",
                        head_width=arrow_head_size,
                        length_includes_head=True,
                        color="r",
                        zorder=15,
                        linewidth=arrow_line_width,
                    )
                    ax.annotate(f_y, (x + offset, (y + length_of_arrow)), c=force_color)
                    # ax.scatter(x,y + length_of_arrow, c='white')
                else:
                    pass

        # plotting support vectors
        # loop all x-direction forces
        for reaction in reactions:
            x = nodes[reaction][0]
            y = nodes[reaction][1]

            f_x = np.round(reactions[reaction][0], 2)
            f_y = np.round(reactions[reaction][1], 2)

            # plot arrow x-direction
            if f_x > 0:
                ax.arrow(
                    x - length_of_arrow,
                    y,
                    length_of_arrow,
                    0,
                    shape="full",
                    head_width=arrow_head_size,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_x, ((x - length_of_arrow), y + offset), c=force_color)
                # ax.scatter(x - length_of_arrow, y, c='white')
            elif f_x < 0:
                ax.arrow(
                    x + length_of_arrow,
                    y,
                    -length_of_arrow,
                    0,
                    shape="full",
                    head_width=arrow_head_size,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_x, ((x + length_of_arrow), y + offset), c=force_color)
                # ax.scatter(x + length_of_arrow,y, c='white')
            else:
                pass

            # plot arrow y-direction
            if f_y > 0:
                ax.arrow(
                    x,
                    y - length_of_arrow,
                    0,
                    length_of_arrow,
                    shape="full",
                    head_width=arrow_head_size,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_y, (x + offset, (y - length_of_arrow)), c=force_color)
                # ax.scatter(x,y - length_of_arrow, c='white')
            elif f_y < 0:
                ax.arrow(
                    x,
                    y + length_of_arrow,
                    0,
                    -length_of_arrow,
                    shape="full",
                    head_width=arrow_head_size,
                    length_includes_head=True,
                    color=force_color,
                    zorder=15,
                    linewidth=arrow_line_width,
                )
                ax.annotate(f_y, (x + offset, (y + length_of_arrow)), c=force_color)
                # ax.scatter(x,y + length_of_arrow, c='white')
            else:
                pass

        ax.axis("equal")
        # plt.show()

    def Apply_Selfweight(self, density):
        nodes = self.nodes
        elements = self.elements
        cross_area = self.cross_area

        self.self_weight = {}

        for element in elements:
            # Get Nodes list from members
            node_1 = elements[element][0]
            node_2 = elements[element][1]

            # Get node coordinates
            x1 = nodes[node_1][0]
            y1 = nodes[node_1][1]
            x2 = nodes[node_2][0]
            y2 = nodes[node_2][1]

            # compute member length
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Get member cross-sectional area
            area = cross_area[element]

            # Compute total weight of the member
            weight = area * length * density

            # Make a dictionary for selfweight nodes and Apply half of total weight to each node
            temp_selfweight = {node_1: [0, -weight / 2], node_2: [0, -weight / 2]}

            # Add to global self weight dictionary
            for key, value in temp_selfweight.items():
                if key in self.self_weight:
                    value = self.self_weight[key][1] + value[1]
                    self.self_weight.update({key: [0, np.round(value, 2)]})

                else:
                    self.self_weight[key] = np.round(value, 2)
        self.forces = self.self_weight

