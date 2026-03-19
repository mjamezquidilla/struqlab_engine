import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from struqlab_engine.Frame_2D.Frame_2D import Frame_2D, Member_2D
    from struqlab_engine.Frame_2D.Frame_2D_builders import Frame_builder
    return Frame_2D, Frame_builder, Member_2D


@app.cell
def _():
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
    return B_E, B_I, B_area, C_E, C_I, C_area


@app.cell
def _(B_E, B_I, B_area, C_E, C_I, C_area, Frame_builder, Member_2D):
    columns, beams, supports = Frame_builder(x_spacing=[6,8,6], # Define center-to-center spacing along x-axis
                y_spacing=[3.5, 3.5], # Define center-to-center spacing along y-axis
                cls_beams=Member_2D, # Use Frame_2D's Member_2D class or inherited class. 
                cls_columns=Member_2D, # Use Frame_2D's Member_2D class or inherited class
                cls_beams_kwargs={'area': B_area, 'inertia': B_I, 'elasticity': B_E}, # __init__ properties of cls_beam. 
                cls_column_kwargs={'area': C_area, 'inertia': C_I, 'elasticity': C_E}) # __init__ properties of cls_column.
    return beams, columns, supports


@app.cell
def _(beams):
    # You can use for loops to quickly add loads per beam or column based on Member_2D's API
    for beam in beams:
        beams[beam].Add_Load_Full_Uniform_Fy(-10)
    return


@app.cell
def _(Frame_2D, beams, columns, supports):
    # Frame_2D's 2D Frame Module
    F1 = Frame_2D() # Define 2D Frame
    members = columns | beams # Combile column and beam dictionaries together from Frame_builder return values
    F1.Compile_Frame_Member_Properties(members) # compile columns and beams and load them into Frame 2D
    F1.supports = supports # Add supports from return values of Frame_builder
    return (F1,)


@app.cell
def _(F1):
    F1.Draw_Frame_Setup() # Draw Frame Setup
    return


@app.cell
def _(F1):
    F1.Solve() # Solve the 2D Frame
    return


@app.cell
def _(F1):
    # Draw relevant diagrams using matplotlib
    F1.Draw_Axial_Diagram(figure_size=[10,5],show_labels=True, dpi=300, scale_factor=50)
    F1.Draw_Shear_Diagram(figure_size=[10,5],show_labels=True, dpi=300, scale_factor=50)
    F1.Draw_Moment_Diagram(figure_size=[10,5],show_labels=True, dpi=300, scale_factor=50)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
