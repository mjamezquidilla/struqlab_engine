from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

plt.style.use("fivethirtyeight")
plt.rcParams["hatch.color"] = "white"


class Member_2D:
    def __init__(
        self,
        area: float,
        elasticity: float,
        inertia: float,
        nodes: dict[int, list[float]] = {},
        member_number: Optional[int] = None,
        moment_release: list[int] = [0, 0],
        no_of_divs: int = 11,
    ):
        self.member_number = member_number
        self.area = area
        self.inertia = inertia
        self.elasticity = elasticity

        self.moment_release_left = moment_release[0]
        self.moment_release_right = moment_release[1]
        self.no_of_divs = no_of_divs
        self.points_of_interest: list[float] = []
        self.uniform_full_load: list[float] = []
        self.point_load: list[list[float]] = []
        self.uniform_axial_load: list[float] = []
        self.self_weight: list[float] = []
        self.uniform_full_load_fx: list[float] = []
        self.uniform_full_load_fy: list[float] = []

        # Check if nodes dictionary is not empty
        if nodes:
            self.nodes = nodes
            self.node_list: list[int] = []
            for node in nodes:
                self.node_list.append(node)

            # compute length of member
            coordinates: list[list[float]] = []
            for node in nodes:
                coordinates.append(nodes[node])
            x1 = coordinates[0][0]
            y1 = coordinates[0][1]
            x2 = coordinates[1][0]
            y2 = coordinates[1][1]
            self.length: float = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # create empty force vectors for each node
            self.forces: dict[int, list[float]] = {}
            for node in nodes:
                self.forces.update({node: [0, 0, 0]})

            # Create Shear and Moment numpy arrays
            self.division_spacing = self.length / no_of_divs
            self.x_array: NDArray[np.float64] = np.linspace(
                0, self.length, int(np.ceil(self.length) / self.division_spacing)
            )
            self.axial: NDArray[np.float64] = np.zeros(
                int(np.ceil(self.length) / self.division_spacing)
            )
            self.shear: NDArray[np.float64] = np.zeros(
                int(np.ceil(self.length) / self.division_spacing)
            )
            self.moment: NDArray[np.float64] = np.zeros(
                int(np.ceil(self.length) / self.division_spacing)
            )

            # Plotting Member Releases
            self.__Release_Node_Coordinates()

    def Compile_Member_Forces(self):
        # Reinitialize forces dictionary
        self.forces = {}
        for node in self.nodes:
            self.forces.update({node: [0, 0, 0]})

        # Insert the points of interest into x_array
        for a in self.points_of_interest:
            index = np.searchsorted(self.x_array, a)
            self.x_array = np.insert(self.x_array, index, a)

            before = np.searchsorted(self.x_array, a - 0.001)
            self.x_array = np.insert(self.x_array, before, a - 0.001)

            after = np.searchsorted(self.x_array, a + 0.001)
            self.x_array = np.insert(self.x_array, after, a + 0.001)

        # Re-initialize axial, shear, and moment arrays based on length of x_array
        self.axial = np.zeros(len(self.x_array))
        self.shear = np.zeros(len(self.x_array))
        self.moment = np.zeros(len(self.x_array))

        # Recompute the axial, moment and shear values based on new x_array
        for p in self.point_load:
            P = p[0]
            a = p[1]
            self.Add_Load_Point(P, a)

        for w in self.uniform_full_load:
            self.Add_Load_Full_Uniform(w, False)

        for a in self.uniform_axial_load:
            self.Add_Load_Axial_Uniform(a)

        for w in self.self_weight:
            self.Add_Self_Weight(w)

        for w in self.uniform_full_load_fx:
            self.Add_Load_Full_Uniform_Fx(w)

        for w in self.uniform_full_load_fy:
            self.Add_Load_Full_Uniform_Fy(w)

    def __Release_Node_Coordinates(self):
        nodes = self.nodes
        moment_release = [self.moment_release_left, self.moment_release_right]

        nodes_coordinate_List = []

        for node in nodes:
            nodes_coordinate_List.append(nodes[node].copy())

        self.release_node_coordinates = []

        for i, _ in enumerate(moment_release):
            if moment_release[i] == 1:
                self.release_node_coordinates.append(nodes_coordinate_List[i])

        self.release_node_coordinates = self.release_node_coordinates

        # compute length of member
        coordinates = []
        for node in nodes:
            coordinates.append(nodes[node])
        x1 = coordinates[0][0]
        y1 = coordinates[0][1]
        x2 = coordinates[1][0]
        y2 = coordinates[1][1]
        self.length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / self.length
        s = (y2 - y1) / self.length

        node_vert = self.length * 0.1 * s
        node_hor = self.length * 0.1 * c

        if moment_release[0] == 1 and moment_release[1] == 0:
            self.release_node_coordinates[0][0] += node_hor
            self.release_node_coordinates[0][1] += node_vert
        elif moment_release[0] == 0 and moment_release[1] == 1:
            self.release_node_coordinates[0][0] -= node_hor
            self.release_node_coordinates[0][1] -= node_vert
        elif moment_release[0] == 1 and moment_release[1] == 1:
            self.release_node_coordinates[0][0] += node_hor
            self.release_node_coordinates[0][1] += node_vert
            self.release_node_coordinates[1][0] -= node_hor
            self.release_node_coordinates[1][1] -= node_vert

    def Add_Nodes_To_Element(self, element, nodes):
        from_node = element[0]
        to_node = element[1]
        from_point = nodes[from_node]
        to_point = nodes[to_node]

        self.nodes = {from_node: from_point, to_node: to_point}
        self.node_list = []
        for node in self.nodes:
            self.node_list.append(node)

        # compute length of member
        coordinates = []
        for node in self.nodes:
            coordinates.append(nodes[node])
        x1 = coordinates[0][0]
        y1 = coordinates[0][1]
        x2 = coordinates[1][0]
        y2 = coordinates[1][1]
        self.length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # create empty force vectors for each node
        self.forces = {}
        for node in self.nodes:
            self.forces.update({node: [0, 0, 0]})

        # Create Shear and Moment numpy arrays
        self.division_spacing = self.length / self.no_of_divs
        self.x_array = np.linspace(
            0, self.length, int(np.ceil(self.length) / self.division_spacing)
        )
        self.axial = np.zeros(int(np.ceil(self.length) / self.division_spacing))
        self.shear = np.zeros(int(np.ceil(self.length) / self.division_spacing))
        self.moment = np.zeros(int(np.ceil(self.length) / self.division_spacing))

        self.__Release_Node_Coordinates()

    def Add_Load_Axial_Uniform(self, w: float, skip_part=False):
        L = self.length
        beginning_axial = w * L / 2
        end_axial = w * L / 2

        self.forces[self.node_list[0]][0] += beginning_axial
        self.forces[self.node_list[1]][0] += end_axial

        # Axial Values
        axial_values = self.x_array.copy()
        for index, x in enumerate(axial_values):
            axial_values[index] = w * x - beginning_axial

        self.axial += axial_values

        if not skip_part:
            if w not in self.uniform_axial_load:
                self.uniform_axial_load.append(w)

    def Add_Self_Weight(self, unit_weight: float):
        w = unit_weight * self.area

        nodes = self.nodes
        coordinates = []
        for node in nodes:
            coordinates.append(nodes[node])
        x1 = coordinates[0][0]
        y1 = coordinates[0][1]
        x2 = coordinates[1][0]
        y2 = coordinates[1][1]
        L = self.length

        c = (x2 - x1) / L
        s = (y2 - y1) / L

        w1 = w * c
        w2 = -w * s

        self.Add_Load_Full_Uniform(w1, True)
        self.Add_Load_Axial_Uniform(w2, True)

        if unit_weight not in self.self_weight:
            self.self_weight.append(unit_weight)

    def Add_Load_Full_Uniform_Fy(self, w: float):
        nodes = self.nodes
        coordinates = []
        for node in nodes:
            coordinates.append(nodes[node])
        x1 = coordinates[0][0]
        y1 = coordinates[0][1]
        x2 = coordinates[1][0]
        y2 = coordinates[1][1]
        L = self.length

        c = (x2 - x1) / L
        s = (y2 - y1) / L

        w1 = w * c
        w2 = -w * s

        self.Add_Load_Full_Uniform(-w1, True)
        self.Add_Load_Axial_Uniform(-w2, True)

        if w not in self.uniform_full_load_fy:
            self.uniform_full_load_fy.append(w)

    def Add_Load_Full_Uniform_Fx(self, w: float):
        nodes = self.nodes
        coordinates = []
        for node in nodes:
            coordinates.append(nodes[node])
        x1 = coordinates[0][0]
        y1 = coordinates[0][1]
        x2 = coordinates[1][0]
        y2 = coordinates[1][1]
        L = self.length

        c = (x2 - x1) / L
        s = (y2 - y1) / L

        w1 = w * s
        w2 = w * c

        self.Add_Load_Full_Uniform(w1, True)
        self.Add_Load_Axial_Uniform(w2, True)

        if w not in self.uniform_full_load_fx:
            self.uniform_full_load_fx.append(w)

    def Add_Load_Point(self, P: float, a: float):
        L = self.length
        beginning_moment = P * (L - a) ** 2 * a / L**2
        end_moment = -P * a**2 * (L - a) / L**2
        beginning_shear = (P * (L - a) + beginning_moment + end_moment) / L
        end_shear = (-P * a + end_moment + beginning_moment) / L

        if self.moment_release_left == 1 and self.moment_release_right == 0:
            beginning_shear = beginning_shear - 3 / (2 * L) * beginning_moment
            end_shear = -end_shear + 3 / (2 * L) * beginning_moment
            end_moment = end_moment - 1 / 2 * beginning_moment
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[1]][2] += end_moment
        elif self.moment_release_left == 0 and self.moment_release_right == 1:
            beginning_shear = beginning_shear - 3 / (2 * L) * end_moment
            end_shear = -end_shear + 3 / (2 * L) * end_moment
            beginning_moment = beginning_moment - 1 / 2 * end_moment
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[0]][2] += beginning_moment
        elif self.moment_release_right == 1 and self.moment_release_right == 1:
            beginning_shear = beginning_shear - 1 / L * (beginning_moment + end_moment)
            end_shear = -end_shear + 1 / L * (beginning_moment + end_moment)
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
        else:
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += end_shear
            self.forces[self.node_list[0]][2] += beginning_moment
            self.forces[self.node_list[1]][2] += end_moment

        # Shear Values
        shear_values = self.x_array.copy()
        for index, x in enumerate(self.x_array):
            if x <= a:
                shear_values[index] = beginning_shear
            else:
                shear_values[index] = beginning_shear - P

        self.shear += shear_values

        # Moment Values
        if self.moment_release_left == 1:
            beginning_moment = 0

        moment_values = self.x_array.copy()
        for index, x in enumerate(self.x_array):
            if x <= a:
                moment_values[index] = beginning_moment - beginning_shear * x
            else:
                moment_values[index] = (
                    beginning_moment - beginning_shear * x + P * (x - a)
                )
        self.moment += moment_values

        if a not in self.points_of_interest:
            self.points_of_interest.append(a)
            self.point_load.append([P, a])

    def Add_Load_Full_Uniform(self, w: float, skip_part=False):  # OKAY!
        L = self.length

        beginning_moment = w * L**2 / 12
        end_moment = -w * L**2 / 12
        beginning_shear = w * L / 2
        end_shear = w * L / 2

        if self.moment_release_left == 1 and self.moment_release_right == 0:
            beginning_shear = beginning_shear - 3 / (2 * L) * beginning_moment
            end_shear = end_shear + 3 / (2 * L) * beginning_moment
            end_moment = end_moment - 1 / 2 * beginning_moment
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[1]][2] += end_moment
        elif self.moment_release_left == 0 and self.moment_release_right == 1:
            beginning_shear = beginning_shear - 3 / (2 * L) * end_moment
            end_shear = end_shear + 3 / (2 * L) * end_moment
            beginning_moment = beginning_moment - 1 / 2 * end_moment
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[0]][2] += beginning_moment
        elif self.moment_release_right == 1 and self.moment_release_right == 1:
            beginning_shear = beginning_shear - 1 / L * (beginning_moment + end_moment)
            end_shear = end_shear + 1 / L * (beginning_moment + end_moment)
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
        else:
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[0]][2] += beginning_moment
            self.forces[self.node_list[1]][2] += end_moment

        # Shear Values
        shear_values = self.x_array.copy()
        for index, shear_value in enumerate(shear_values):
            shear_values[index] = beginning_shear - w * shear_value

        # Moment Values
        if self.moment_release_left == 1:
            beginning_moment = 0
        moment_values = self.x_array.copy()
        for index, moment_value in enumerate(moment_values):
            moment_values[index] = (
                beginning_moment
                - beginning_shear * moment_value
                + w * moment_value**2 / 2
            )

        self.shear += shear_values
        self.moment += moment_values

        if not skip_part:
            if w not in self.uniform_full_load:
                self.uniform_full_load.append(w)

    # def Add_Load_Moment(self,M,a): #TODO Recheck values for all quadrants
    #     L = self.length
    #     b = L - a
    #     beginning_moment = M * b * (2*a-b) / L**2
    #     end_moment = M * a * (2*b-a) / L**2

    #     if self.moment_release_left == 1 and self.moment_release_right == 0:
    #         end_moment = end_moment - 1 / 2 * beginning_moment
    #         self.forces[self.node_list[1]][2] += end_moment
    #     elif self.moment_release_left == 0 and self.moment_release_right == 1:
    #         beginning_moment = beginning_moment - 1 / 2 * end_moment
    #         self.forces[self.node_list[0]][2] += beginning_moment
    #     elif self.moment_release_right == 1 and self.moment_release_right == 1:
    #         pass
    #     else:
    #         self.forces[self.node_list[0]][2] += beginning_moment
    #         self.forces[self.node_list[1]][2] += end_moment

    #     # Moment Values
    #     if self.moment_release_left == 1:
    #         beginning_moment = 0
    #     moment_values = self.x_array.copy()
    #     for index, _ in enumerate(moment_values):
    #         moment_values[index] = beginning_moment
    #     self.moment += moment_values

    def Add_Load_Partial_Uniform(
        self, w: float, a: float, b: float
    ):  # TODO Recheck values for all quadrants
        L = self.length
        beginning_moment = w * L**2 / 12 * (
            6 * (b / L) ** 2 - 8 * (b / L) ** 3 + 3 * (b / L) ** 4
        ) - w * L**2 / 12 * (6 * (a / L) ** 2 - 8 * (a / L) ** 3 + 3 * (a / L) ** 4)
        end_moment = -w * L**2 / 12 * (
            4 * (b / L) ** 3 - 3 * (b / L) ** 4
        ) + w * L**2 / 12 * (4 * (a / L) ** 3 - 3 * (a / L) ** 4)
        beginning_shear = w * (b - a) / L * ((b - a) / 2 + (L - b))
        end_shear = w * (b - a) / L * ((b - a) / 2 + a)

        if self.moment_release_left == 1 and self.moment_release_right == 0:
            beginning_shear = beginning_shear - 3 / (2 * L) * beginning_moment
            end_shear = end_shear + 3 / (2 * L) * beginning_moment
            end_moment = end_moment - 1 / 2 * beginning_moment
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[1]][2] += end_moment
        elif self.moment_release_left == 0 and self.moment_release_right == 1:
            beginning_shear = beginning_shear - 3 / (2 * L) * end_moment
            end_shear = end_shear + 3 / (2 * L) * end_moment
            beginning_moment = beginning_moment - 1 / 2 * end_moment
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[0]][2] += beginning_moment
        elif self.moment_release_right == 1 and self.moment_release_right == 1:
            beginning_shear = beginning_shear - 1 / L * (beginning_moment + end_moment)
            end_shear = end_shear + 1 / L * (beginning_moment + end_moment)
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
        else:
            self.forces[self.node_list[0]][1] += -beginning_shear
            self.forces[self.node_list[1]][1] += -end_shear
            self.forces[self.node_list[0]][2] += beginning_moment
            self.forces[self.node_list[1]][2] += end_moment

        # Shear Values
        shear_values = self.x_array.copy()
        for index, shear_value in enumerate(shear_values):
            if shear_value < a:
                shear_values[index] = beginning_shear
            elif shear_value >= a and shear_value < b:
                shear_values[index] = beginning_shear - w * (shear_value - a)
            else:
                shear_values[index] = beginning_shear - w * (b - a)

        self.shear += shear_values

        # Moment Values
        # if self.moment_release_left == 1:
        #     beginning_moment = 0
        moment_values = self.x_array.copy()
        for index, moment_value in enumerate(moment_values):  # TODO FIX MOMENT VALUES
            if moment_value < a:
                moment_values[index] = beginning_moment - beginning_shear * moment_value
            elif moment_value >= a and moment_value < b:
                moment_values[index] = (
                    beginning_moment
                    - beginning_shear * moment_value
                    + w * (moment_value - a) ** 2 / 2
                )
            else:
                moment_values[index] = (
                    beginning_moment
                    - beginning_shear * moment_value
                    + w * (b - a) * ((b - a) / 2 + (moment_value - b))
                )

        self.moment += moment_values

    def Plot_Axial_Diagram(
        self, figure_size: list[float] = [10, 5], show_annoation: bool = True
    ):  # OKAY

        x_array_relative = self.x_array / np.ceil(self.length)

        plt.figure(figsize=figure_size)
        plt.plot(x_array_relative, self.axial, marker="o")
        plt.fill_between(x_array_relative, 0, self.axial, hatch="/", alpha=0.1)
        plt.xlim([-0.1, 1.1])
        # plt.plot(self.x_array,self.axial, marker='o')
        # plt.fill_between(self.x_array, 0, self.axial, hatch = '/', alpha = 0.1)
        # plt.xlim([0, self.length])
        plt.ylabel("Axial")
        plt.xlabel("Length ratio, Length = {}".format(self.length))
        plt.title("Axial Diagram of Member {}".format(self.member_number))

        if show_annoation:
            for i, text in enumerate(self.axial):
                plt.annotate(round(text, 2), (x_array_relative[i], self.axial[i]))

        plt.tight_layout()
        plt.show()

    # Plot Shear Diagram
    def Plot_Shear_Diagram(
        self, figure_size: list[float] = [10, 5], show_annotation: bool = True
    ):

        x_array_relative = self.x_array / np.ceil(self.length)

        plt.figure(figsize=figure_size)
        plt.plot(x_array_relative, self.shear, marker="o")
        plt.fill_between(x_array_relative, 0, self.shear, hatch="/", alpha=0.1)
        plt.xlim([-0.1, 1.1])
        # plt.plot(self.x_array,self.shear, marker='o')
        # plt.fill_between(self.x_array, 0, self.shear, hatch = '/', alpha = 0.1)
        # plt.xlim([0, self.length])
        plt.ylabel("Shear")
        plt.xlabel("Length ratio, Length = {}".format(self.length))
        plt.title("Shear Diagram of Member {}".format(self.member_number))

        if show_annotation:
            for i, text in enumerate(self.shear):
                plt.annotate(round(text, 2), (x_array_relative[i], self.shear[i]))

        plt.xticks(x_array_relative)
        plt.tight_layout()
        plt.show()

    # Plot Moment Diagram
    def Plot_Moment_Diagram(
        self, figure_size: list[float] = [10, 5], show_annotation: bool = True
    ):

        x_array_relative = self.x_array / np.ceil(self.length)

        plt.figure(figsize=figure_size)
        plt.plot(x_array_relative, -self.moment, marker="o")
        plt.fill_between(x_array_relative, 0, -self.moment, hatch="/", alpha=0.1)
        plt.xlim([-0.1, 1.1])
        # plt.plot(self.x_array,-self.moment, marker='o')
        # plt.fill_between(self.x_array, 0, -self.moment, hatch = '/', alpha = 0.1)
        # plt.xlim([0, self.length])
        plt.ylabel("Moment")
        plt.xlabel("Length ratio, Length = {}".format(self.length))
        plt.title("Moment Diagram of Member {}".format(self.member_number))

        if show_annotation:
            for i, text in enumerate(self.moment):
                plt.annotate(round(-text, 2), (x_array_relative[i], -self.moment[i]))

        plt.xticks(x_array_relative)
        plt.tight_layout()
        plt.show()

    def _Resolve_Forces_into_Components(self):
        self.Compile_Member_Forces()
        # solve for angle
        nodes = self.nodes

        coordinates = []
        for node in nodes:
            coordinates.append(nodes[node])
        x1 = coordinates[0][0]
        y1 = coordinates[0][1]
        x2 = coordinates[1][0]
        y2 = coordinates[1][1]
        L = self.length

        c = (x2 - x1) / L
        s = (y2 - y1) / L

        FA_1 = self.forces[self.node_list[0]][0]
        FA_2 = self.forces[self.node_list[1]][0]
        FV_1 = self.forces[self.node_list[0]][1]
        FV_2 = self.forces[self.node_list[1]][1]
        M_1 = self.forces[self.node_list[0]][2]
        M_2 = self.forces[self.node_list[1]][2]

        FV1_x = -FV_1 * s
        FV2_x = -FV_2 * s

        FA_1_x = FA_1 * c
        FA_1_y = FA_1 * s
        FA_2_x = FA_2 * c
        FA_2_y = FA_2 * s

        FV1_y = FV_1 * c
        FV2_y = FV_2 * c

        self.resolved_forces = {}

        for node in nodes:
            self.resolved_forces.update({node: [0, 0, 0]})

        self.resolved_forces[self.node_list[0]][0] = FV1_x + FA_1_x
        self.resolved_forces[self.node_list[0]][1] = FV1_y + FA_1_y

        self.resolved_forces[self.node_list[1]][0] = FV2_x + FA_2_x
        self.resolved_forces[self.node_list[1]][1] = FV2_y + FA_2_y

        if x2 >= x1 and y2 >= y1:  # 0 to 90 degrees
            self.resolved_forces[self.node_list[0]][2] = -M_1
            self.resolved_forces[self.node_list[1]][2] = -M_2
        elif x1 >= x2 and y2 >= y1:  # 91 to 180 degrees
            self.resolved_forces[self.node_list[0]][2] = -M_1
            self.resolved_forces[self.node_list[1]][2] = -M_2
        elif x1 >= x2 and y1 >= y2:  # 181 to 270 degrees
            self.resolved_forces[self.node_list[0]][2] = -M_1
            self.resolved_forces[self.node_list[1]][2] = -M_2
        else:  # 271 to 359
            self.resolved_forces[self.node_list[0]][2] = -M_1
            self.resolved_forces[self.node_list[1]][2] = -M_2

    def _Reaction_Add_Shear_At_Left_Support(self, shear: float):
        shear_values = self.x_array.copy()
        for index, _ in enumerate(shear_values):
            shear_values[index] = shear
        self.shear += shear_values

        moment_values = self.x_array.copy()
        for index, moment_value in enumerate(moment_values):
            moment_values[index] = -shear * moment_value
        self.moment += moment_values

    def _Reaction_Add_Moment_At_Left_Support(self, moment: float):
        moment_values = self.x_array.copy()
        for index, _ in enumerate(moment_values):
            moment_values[index] = moment
        self.moment += moment_values

    def _Reaction_Add_Axial_At_Left_Support(self, axial: float):
        axial_values = self.x_array.copy()
        for index, _ in enumerate(axial_values):
            axial_values[index] = axial
        self.axial += axial_values

    def Plot_Diagrams(
        self,
        figure_size: list[float] = [15, 10],
        show_annotation: bool = True,
        dpi=600,
    ):
        fig, axs = plt.subplots(3, 1)
        fig.set_figheight(figure_size[0])
        fig.set_figwidth(figure_size[1])

        # range = int(np.ceil(self.length)/self.division_spacing)
        # x_array_relative = np.linspace(0,1,range)
        x_array_relative = self.x_array / np.ceil(self.length)

        # Plot Axial Diagram
        axs[0].plot(x_array_relative, self.axial, marker="o")
        axs[0].fill_between(x_array_relative, 0, self.axial, hatch="/", alpha=0.1)
        axs[0].set_xlim([-0.1, 1.1])
        axs[0].set_ylabel("Axial")
        axs[0].set_xlabel("Length ratio, Length = {}".format(round(self.length, 2)))
        axs[0].set_title("Axial Diagram of Member {}".format(self.member_number))

        if show_annotation:
            for i, text in enumerate(self.axial):
                axs[0].annotate(
                    round(text, 2),
                    (round(x_array_relative[i], 2), round(self.axial[i], 2)),
                )

        axs[0].margins(0.2)

        # Plot Shear Diagram
        axs[1].plot(x_array_relative, self.shear, marker="o")
        axs[1].fill_between(x_array_relative, 0, self.shear, hatch="/", alpha=0.1)
        axs[1].set_xlim([-0.1, 1.1])
        axs[1].set_ylabel("Shear")
        axs[1].set_xlabel("Length ratio, Length = {}".format(round(self.length, 2)))
        axs[1].set_title("Shear Diagram of Member {}".format(self.member_number))

        if show_annotation:
            for i, text in enumerate(self.shear):
                axs[1].annotate(
                    round(text, 2),
                    (round(x_array_relative[i], 2), round(self.shear[i], 2)),
                )

        axs[1].margins(0.2)

        # Plot Moment Diagram
        axs[2].plot(x_array_relative, -self.moment, marker="o")
        axs[2].fill_between(x_array_relative, 0, -self.moment, hatch="/", alpha=0.1)
        axs[2].set_xlim([-0.1, 1.1])
        axs[2].set_ylabel("Moment")
        axs[2].set_xlabel("Length ratio, Length = {}".format(round(self.length, 2)))
        axs[2].set_title("Moment Diagram of Member {}".format(self.member_number))

        if show_annotation:
            for i, text in enumerate(self.moment):
                axs[2].annotate(
                    round(-text, 2),
                    (round(x_array_relative[i], 2), round(-self.moment[i], 2)),
                )

        axs[2].margins(0.2)

        plt.figure(dpi=dpi)
        plt.tight_layout()
        plt.show()

    def Summary(self):
        self.moment_max = -np.max(self.moment)
        self.moment_min = -np.min(self.moment)
        self.moment_at_left = -self.moment[0]
        self.moment_at_right = -self.moment[-1]

        self.shear_max = np.max(self.shear)
        self.shear_min = np.min(self.shear)
        self.shear_at_left = self.shear[0]
        self.shear_at_right = self.shear[-1]

        print("At Left End:")
        print("Axial: {}".format(self.axial[0]))
        print("Shear: {}".format(self.shear_at_left))
        print("Moment: {}".format(self.moment_at_left))
        print()
        print("At Right End:")
        print("Axial: {}".format(self.axial[-1]))
        print("Shear: {}".format(self.shear_at_right))
        print("Moment: {}".format(self.moment_at_right))
        print()
        print("Minimum and Maximum")
        print("Minimum Shear: {}".format(self.shear_min))
        print("Maximum Shear: {}".format(self.shear_max))
        print("Minimum Moment: {}".format(self.moment_min))
        print("Maximum Moment: {}".format(self.moment_max))

    def _Assemble_Plot_Loadings(self):
        # Assemble nodes and loadings into a dictionary
        plot_loadings = {
            self.member_number: {
                "nodes": self.nodes,
                "uniform_full_load": self.uniform_full_load,
                "point_load": self.point_load,
                "uniform_axial_load": self.uniform_axial_load,
                "self_weight": self.self_weight,
                "uniform_full_load_fx": self.uniform_full_load_fx,
                "uniform_full_load_fy": self.uniform_full_load_fy,
            }
        }

        # print(plot_loadings)
        return plot_loadings

    # def Assemble_Member_Stiffness_Matrix(self):
    #     # moment_releases_left = self.member_moment_releases[element][0]
    #     # moment_releases_right = self.member_moment_releases[element][1]

    #     # from_point = elements[element][0]
    #     # to_point = elements[element][1]
    #     # from_node = nodes[from_point]
    #     # to_node = nodes[to_point]

    #     # compute length of member
    #     coordinates = []
    #     for node in self.nodes:
    #         coordinates.append(self.nodes[node])
    #     x1 = coordinates[0][0]
    #     y1 = coordinates[0][1]
    #     x2 = coordinates[1][0]
    #     y2 = coordinates[1][1]

    #     L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #     c = (x2 - x1)/L
    #     s = (y2 - y1)/L

    #     A = self.area
    #     I = self.inertia
    #     E = self.elasticity

    #     # if moment_releases_left == 1 and moment_releases_right == 1:
    #     #     k = E * I / L**3 * np.array([[A*L**2/I,    0,      0, -A*L**2/I,     0,      0],
    #     #                                 [0,            0,      0,         0,     0,      0],
    #     #                                 [0,            0,      0,         0,     0,      0],
    #     #                                 [-A*L**2/I,    0,      0,  A*L**2/I,     0,      0],
    #     #                                 [0,            0,      0,         0,     0,      0],
    #     #                                 [0,            0,      0,         0,     0,      0]])

    #     # elif moment_releases_left == 1 and moment_releases_right == 0:
    #     #     k = E * I / L**3 * np.array([[A*L**2/I,    0,      0, -A*L**2/I,     0,     0],
    #     #                                 [0,          3,        0,         0,   -3,    3*L],
    #     #                                 [0,          0,        0,         0,    0,      0],
    #     #                                 [-A*L**2/I,   0,       0,  A*L**2/I,    0,      0],
    #     #                                 [0,         -3,        0,         0,    3,    -3*L],
    #     #                                 [0,         3*L,       0,         0,  -3*L, 3*L**2]])

    #     # elif moment_releases_left == 0 and moment_releases_right == 1:
    #     #     k = E * I / L**3 * np.array([[A*L**2/I,    0,      0, -A*L**2/I,    0,    0],
    #     #                                 [0,            3,    3*L,         0,   -3,    0],
    #     #                                 [0,          3*L, 3*L**2,         0, -3*L,    0],
    #     #                                 [-A*L**2/I,   0,       0,  A*L**2/I,    0,    0],
    #     #                                 [0,          -3,    -3*L,         0,    3,    0],
    #     #                                 [0,           0,       0,         0,    0,    0]])

    #     # else:
    #     k = E * I / L**3 * np.array([[A*L**2/I,    0,      0, -A*L**2/I,     0,      0],
    #                                 [0,          12,    6*L,         0,   -12,    6*L],
    #                                 [0,         6*L, 4*L**2,         0,  -6*L, 2*L**2],
    #                                 [-A*L**2/I,   0,      0,  A*L**2/I,     0,      0],
    #                                 [0,         -12,   -6*L,         0,    12,   -6*L],
    #                                 [0,         6*L, 2*L**2,         0,  -6*L, 4*L**2]])

    #     T = np.array([[c, s, 0 , 0, 0, 0],
    #                 [-s, c, 0, 0, 0, 0],
    #                 [0, 0, 1, 0, 0, 0],
    #                 [0, 0, 0, c, s, 0],
    #                 [0, 0, 0, -s, c, 0],
    #                 [0, 0, 0, 0, 0, 1]])

    #     K = np.transpose(T).dot(k).dot(T)

    #     return K
