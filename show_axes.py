"""
Show colored X, Y, Z axes in a 3D viewer using pythonocc.
  X = RED
  Y = GREEN
  Z = BLUE
Each axis is drawn as an arrow with a label at the tip.
Optionally loads a STEP file so you can see the model alongside the axes.
"""

import sys
import os

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax1, gp_Ax2, gp_Trsf
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeBox
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Quantity import (
    Quantity_Color,
    Quantity_TOC_RGB,
)
from OCC.Display.SimpleGui import init_display


def make_arrow(origin, direction, length=50.0, shaft_r=0.8, head_r=2.5, head_len=6.0):
    """Create an arrow (cylinder + cone) along an arbitrary direction."""
    # Build shaft along Z
    shaft = BRepPrimAPI_MakeCylinder(shaft_r, length - head_len).Shape()
    # Build cone head at tip
    cone = BRepPrimAPI_MakeCone(head_r, 0.0, head_len).Shape()
    # Move cone to top of shaft
    t_cone = gp_Trsf()
    t_cone.SetTranslation(gp_Vec(0, 0, length - head_len))
    cone = BRepBuilderAPI_Transform(cone, t_cone, True).Shape()

    # Now rotate from Z-axis to the desired direction
    from OCC.Core.gp import gp_Quaternion, gp_Vec as Vec
    z_dir = gp_Dir(0, 0, 1)
    target_dir = gp_Dir(*direction)

    trsf = gp_Trsf()
    # If direction is -Z, use special rotation
    cross = gp_Vec(z_dir).Crossed(gp_Vec(target_dir))
    if cross.Magnitude() < 1e-10:
        # Parallel or anti-parallel
        dot = gp_Vec(z_dir).Dot(gp_Vec(target_dir))
        if dot < 0:
            # 180 degree rotation around X
            from math import pi
            trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), pi)
        # else: same direction, no rotation needed
    else:
        angle = gp_Vec(z_dir).Angle(gp_Vec(target_dir))
        rot_axis = gp_Dir(cross)
        trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), rot_axis), angle)

    # Translate to origin
    t_origin = gp_Trsf()
    t_origin.SetTranslation(gp_Vec(gp_Pnt(0, 0, 0), gp_Pnt(*origin)))

    shaft = BRepBuilderAPI_Transform(shaft, trsf, True).Shape()
    shaft = BRepBuilderAPI_Transform(shaft, t_origin, True).Shape()
    cone = BRepBuilderAPI_Transform(cone, trsf, True).Shape()
    cone = BRepBuilderAPI_Transform(cone, t_origin, True).Shape()

    return shaft, cone


def main():
    step_path = None
    if len(sys.argv) > 1:
        step_path = sys.argv[1]

    display, start_display, add_menu, add_function_to_menu = init_display()

    axis_length = 60.0

    # Based on original views:
    #   front: Dir=(1,0,0)  → looking along +X   → XDir=(0,1,0) i.e. right=Y, up=Z
    #   right: Dir=(0,1,0)  → looking along +Y   → XDir=(-1,0,0) i.e. right=-X, up=Z
    #   top:   Dir=(0,0,1)  → looking along +Z   → XDir=(0,1,0) i.e. right=Y, up=-X

    # --- X axis (RED) — FRONT view direction ---
    x_shaft, x_cone = make_arrow((0, 0, 0), (1, 0, 0), length=axis_length)
    red = Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB)
    display.DisplayColoredShape(x_shaft, red, update=False)
    display.DisplayColoredShape(x_cone, red, update=False)
    display.DisplayMessage(gp_Pnt(axis_length + 5, 0, 0), "X (FRONT)", height=24, message_color=(1, 0, 0))

    # --- Y axis (GREEN) — RIGHT view direction ---
    y_shaft, y_cone = make_arrow((0, 0, 0), (0, 1, 0), length=axis_length)
    green = Quantity_Color(0.0, 0.8, 0.0, Quantity_TOC_RGB)
    display.DisplayColoredShape(y_shaft, green, update=False)
    display.DisplayColoredShape(y_cone, green, update=False)
    display.DisplayMessage(gp_Pnt(0, axis_length + 5, 0), "Y (RIGHT)", height=24, message_color=(0, 0.8, 0))

    # --- Z axis (BLUE) — TOP view direction ---
    z_shaft, z_cone = make_arrow((0, 0, 0), (0, 0, 1), length=axis_length)
    blue = Quantity_Color(0.0, 0.0, 1.0, Quantity_TOC_RGB)
    display.DisplayColoredShape(z_shaft, blue, update=False)
    display.DisplayColoredShape(z_cone, blue, update=False)
    display.DisplayMessage(gp_Pnt(0, 0, axis_length + 5), "Z (TOP)", height=24, message_color=(0, 0, 1))

    # --- ISO direction indicator (YELLOW) ---
    iso_shaft, iso_cone = make_arrow((0, 0, 0), (1, 1, 1), length=axis_length * 0.7,
                                      shaft_r=0.5, head_r=1.8, head_len=5.0)
    yellow = Quantity_Color(0.9, 0.75, 0.0, Quantity_TOC_RGB)
    display.DisplayColoredShape(iso_shaft, yellow, update=False)
    display.DisplayColoredShape(iso_cone, yellow, update=False)
    iso_tip = axis_length * 0.7 / 1.732 + 3  # 1/sqrt(3) * len + offset
    display.DisplayMessage(gp_Pnt(iso_tip, iso_tip, iso_tip), "ISO", height=20, message_color=(0.9, 0.75, 0))

    # --- Small cube at origin ---
    origin_box = BRepPrimAPI_MakeBox(gp_Pnt(-1.5, -1.5, -1.5), 3, 3, 3).Shape()
    white = Quantity_Color(0.9, 0.9, 0.9, Quantity_TOC_RGB)
    display.DisplayColoredShape(origin_box, white, update=False)

    # --- Optionally load a STEP file ---
    if step_path and os.path.isfile(step_path):
        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) == 1:
            reader.TransferRoots()
            shape = reader.OneShape()
            grey = Quantity_Color(0.6, 0.6, 0.7, Quantity_TOC_RGB)
            display.DisplayColoredShape(shape, grey, update=False)
            print(f"[OK] Loaded: {step_path}")
        else:
            print(f"[!] Failed to read: {step_path}")

    display.FitAll()
    print("\n  Axes based on original views:")
    print("    X = RED    → FRONT view looks along +X")
    print("    Y = GREEN  → RIGHT view looks along +Y")
    print("    Z = BLUE   → TOP   view looks along +Z")
    print("    ISO (yellow) → looks along (1,1,1)\n")
    start_display()


if __name__ == "__main__":
    main()
