import inspect
import itertools
from typing import Type

from .Member_2D import Member_2D


def Frame_builder(
    x_spacing: list[float],
    y_spacing: list[float],
    cls_columns: Type[Member_2D],
    cls_beams: Type[Member_2D],
    cls_column_kwargs: dict,
    cls_beams_kwargs: dict,
) -> tuple:
    """
    Frame Builder for struqlab_engine's Frame 2D class. Creates a Classic Rectangular Rigid Frame.

    ```text
                |-----------|---------------|-------------------|---------------|
                |           |               |                   |               |
           h3   |           |               |                   |               |
                |           |               |                   |               |
                |-----------|---------------|-------------------|---------------|
                |           |               |                   |               |
                |           |               |                   |               |
           h2   |           |               |                   |               |
                |           |               |                   |               |
                |           |               |                   |               |
                |-----------|---------------|-------------------|---------------|
                |           |               |                   |               |
                |           |               |                   |               |
           h1   |           |               |                   |               |
                |           |               |                   |               |
                |           |               |                   |               |
                 <---x1----> <------x2-----> <--------x3-------> <------x4----->
    ```

    Parameters
    ==========

    x_spacing: list[float] -> center-to-center column-to-column clear spacing of the 2D frame.
    y_spacing: list[float] -> center-to-center spacing of story height of the 2D frame.
    cls_columns: Type[Member_2D] -> a child of Member_2D from Frame_2D's module/package.
    cls_beams: Type[Member_2D] -> a child of Member_2D from Frame_2D's module/package.
    cls_column_kwargs: dict -> list of parameters that initializes the Member_2D.
    cls_beam_kwargs: dict -> list of parameters that initializes the Member_2D.

    Returns
    =======

    columns: dict -> returns columns with its parameters (member number, area, inertial, elasticity, node conditions, nodes dictionary, and etc.)
    beams: dict -> returns beams with its parameters (member number, area, inertial, elasticity, node conditions, nodes dictionary, and etc.)
    supports: dict -> retuns nodes with its support condition as a dictionary

    Sample Usage
    ============

    Partial Example 1: Concrete Frame 1
    -------------------------------------
    columns, beams, supports = Frame_builder(x_spacing=[6,8,6], # Define center-to-center spacing along x-axis
                                y_spacing=[3.5, 3.5], # Define center-to-center spacing along y-axis
                                cls_beams=Member_2D, # Use Frame_2D's Member_2D class or inherited class.
                                cls_columns=Member_2D, # Use Frame_2D's Member_2D class or inherited class
                                cls_beams_kwargs={'area': B_area, 'inertia': B_I, 'elasticity': B_E}, # __init__ properties of cls_beam.
                                cls_column_kwargs={'area': C_area, 'inertia': C_I, 'elasticity': C_E}) # __init__ properties of cls_column.

    Full Example 1: Concrete Frame 1
    --------------------------------

    ```python
    from struqlab_engine.Frame_2D.Frame_2D import Frame_2D, Member_2D
    from struqlab_engine.Frame_2D.Frame_2D_builders import Frame_builder

    # Define Beam Properties
    B_d = 0.5 # depth
    B_bf = 0.3 # width
    B_area = B_d * B_bf # area
    B_I = 1/12 * B_bf * B_d**3 # Inertia
    B_E = 4700 * (28)**(0.5) # elasticity

    # Define Column Properties
    C_d = 0.3 # depth
    C_bf = 0.3 # width
    C_area = C_d * C_bf # area
    C_I = 1/12 * C_bf * C_d**3 # inertia
    C_E = 4700 * (28)**(0.5) # elasticity

    #### USAGE OF MODULE ####
    columns, beams, supports = Frame_builder(x_spacing=[6,8,6], # Define center-to-center spacing along x-axis
                                            y_spacing=[3.5, 3.5], # Define center-to-center spacing along y-axis
                                            cls_beams=Member_2D, # Use Frame_2D's Member_2D class or inherited class.
                                            cls_columns=Member_2D, # Use Frame_2D's Member_2D class or inherited class
                                            cls_beams_kwargs={'area': B_area, 'inertia': B_I, 'elasticity': B_E}, # __init__ properties of cls_beam.
                                            cls_column_kwargs={'area': C_area, 'inertia': C_I, 'elasticity': C_E}) # __init__ properties of cls_column.
    #### USAGE OF MODULE ####

    # You can use for loops to quickly add loads per beam or column based on Member_2D's API
    for beam in beams:
        beams[beam].Add_Load_Full_Uniform_Fy(-10)

    # Frame_2D's 2D Frame Module
    F1 = Frame_2D() # Define 2D Frame
    members = columns | beams # Combile column and beam dictionaries together from Frame_builder return values
    F1.Compile_Frame_Member_Properties(members) # compile columns and beams and load them into Frame 2D
    F1.supports = supports # Add supports from return values of Frame_builder
    F1.Draw_Frame_Setup() # Draw Frame Setup
    F1.Solve() # Solve the 2D Frame

    # Draw relevant diagrams using matplotlib
    F1.Draw_Axial_Diagram(figure_size=[10,5],show_labels=True, dpi=300, scale_factor=50)
    F1.Draw_Shear_Diagram(figure_size=[10,5],show_labels=True, dpi=300, scale_factor=50)
    F1.Draw_Moment_Diagram(figure_size=[10,5],show_labels=True, dpi=300, scale_factor=50)
    ```
    """

    # For Columns
    params = inspect.signature(cls_columns.__init__).parameters
    valid_args_columns = {k: v for k, v in cls_column_kwargs.items() if k in params}

    # For Beams
    params = inspect.signature(cls_beams.__init__).parameters
    valid_args_beams = {k: v for k, v in cls_beams_kwargs.items() if k in params}

    x_coordinates = list(itertools.accumulate(x_spacing))
    y_coordinates = list(itertools.accumulate(y_spacing))

    x_coordinates.insert(0, 0)
    y_coordinates.insert(0, 0)

    nodes = {
        i: [x, y]
        for i, (x, y) in enumerate(
            ((x, y) for y in y_coordinates for x in x_coordinates), start=1
        )
    }

    total_columns = len(y_spacing) * (len(x_spacing) + 1)

    column_range = range(1, total_columns + 1)

    columns: dict[str, Member_2D] = {}
    beams: dict[str, Member_2D] = {}

    member_number = 1

    for col in column_range:
        member = cls_columns(
            member_number=member_number,
            nodes={k: nodes[k] for k in (col, col + len(x_coordinates))},
            **valid_args_columns,
        )
        columns.update({"C" + str(col): member})
        member_number = member_number + 1

    beam = 1
    for k in range(len(x_coordinates), len(nodes) + 1):
        if k % len(x_coordinates) != 0:
            member = cls_beams(
                member_number=member_number,
                nodes={k: nodes[k] for k in (k, k + 1)},
                **valid_args_beams,
            )
            beams.update({"B" + str(beam): member})
            member_number = member_number + 1
            beam = beam + 1

    supports: dict[int, list[int]] = {
        k: [1, 1, 1] for k in range(1, len(x_coordinates) + 1)
    }

    return columns, beams, supports
