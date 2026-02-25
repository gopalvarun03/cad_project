import os
import glob
import argparse
import math
import numpy as np
import svgwrite
from tqdm import tqdm

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE

from OCC.Core.GeomAbs import (
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_BSplineCurve,
    GeomAbs_BezierCurve
)

# Bezier approximation constant for circles/ellipses
KAPPA = 4.0 / 3.0 * (math.sqrt(2.0) - 1.0)


def extract_edges_from_compounds(compounds):
    """Extract edge data from compounds with proper geometric info."""
    edges = []
    all_pts = []

    for compound in compounds:
        if compound is None or compound.IsNull():
            continue

        explorer = TopExp_Explorer(compound, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            adaptor = BRepAdaptor_Curve(edge)

            curve_type = adaptor.GetType()
            u1, u2 = adaptor.FirstParameter(), adaptor.LastParameter()

            if curve_type == GeomAbs_Line:
                p_start = adaptor.Value(u1)
                p_end = adaptor.Value(u2)
                pt_start = np.array([p_start.X(), -p_start.Y()])
                pt_end = np.array([p_end.X(), -p_end.Y()])
                all_pts.extend([pt_start, pt_end])
                
                edges.append({
                    'type': 'line',
                    'start': pt_start,
                    'end': pt_end
                })

            elif curve_type == GeomAbs_Circle:
                circ = adaptor.Circle()
                center = circ.Location()
                radius = circ.Radius()
                
                cx, cy = center.X(), -center.Y()
                
                all_pts.append(np.array([cx - radius, cy - radius]))
                all_pts.append(np.array([cx + radius, cy + radius]))
                
                num_samples = 5
                pts = []
                tangents = []
                for i in range(num_samples):
                    u = u1 + (u2 - u1) * i / (num_samples - 1)
                    p = adaptor.Value(u)
                    pt = np.array([p.X(), -p.Y()])
                    pts.append(pt)
                    
                    from OCC.Core.gp import gp_Pnt as gp_Pnt2, gp_Vec
                    pnt = gp_Pnt2()
                    vec = gp_Vec()
                    adaptor.D1(u, pnt, vec)
                    tangent = np.array([vec.X(), -vec.Y()]) * (u2 - u1) / (num_samples - 1)
                    tangents.append(tangent)
                
                edges.append({
                    'type': 'circle',
                    'center': np.array([cx, cy]),
                    'radius': radius,
                    'start_param': u1,
                    'end_param': u2,
                    'points': pts,
                    'tangents': tangents
                })

            elif curve_type == GeomAbs_Ellipse:
                ellipse = adaptor.Ellipse()
                center = ellipse.Location()
                major_r = ellipse.MajorRadius()
                minor_r = ellipse.MinorRadius()
                
                cx, cy = center.X(), -center.Y()
                
                all_pts.append(np.array([cx - major_r, cy - major_r]))
                all_pts.append(np.array([cx + major_r, cy + major_r]))
                
                num_samples = 5
                pts = []
                tangents = []
                for i in range(num_samples):
                    u = u1 + (u2 - u1) * i / (num_samples - 1)
                    p = adaptor.Value(u)
                    pt = np.array([p.X(), -p.Y()])
                    pts.append(pt)
                    
                    from OCC.Core.gp import gp_Pnt as gp_Pnt2, gp_Vec
                    pnt = gp_Pnt2()
                    vec = gp_Vec()
                    adaptor.D1(u, pnt, vec)
                    tangent = np.array([vec.X(), -vec.Y()]) * (u2 - u1) / (num_samples - 1)
                    tangents.append(tangent)
                
                edges.append({
                    'type': 'ellipse',
                    'center': np.array([cx, cy]),
                    'major_radius': major_r,
                    'minor_radius': minor_r,
                    'start_param': u1,
                    'end_param': u2,
                    'points': pts,
                    'tangents': tangents
                })

            else:
                num_samples = 5
                pts = []
                tangents = []
                for i in range(num_samples):
                    u = u1 + (u2 - u1) * i / (num_samples - 1)
                    p = adaptor.Value(u)
                    pt = np.array([p.X(), -p.Y()])
                    pts.append(pt)
                    all_pts.append(pt)
                    
                    from OCC.Core.gp import gp_Pnt as gp_Pnt2, gp_Vec
                    pnt = gp_Pnt2()
                    vec = gp_Vec()
                    adaptor.D1(u, pnt, vec)
                    tangent = np.array([vec.X(), -vec.Y()]) * (u2 - u1) / (num_samples - 1)
                    tangents.append(tangent)
                
                edges.append({
                    'type': 'spline',
                    'points': pts,
                    'tangents': tangents
                })

            explorer.Next()

    return edges, all_pts


def fit_bezier_to_points(points, tangents=None):
    """Fit cubic bezier curves to a sequence of points."""
    if len(points) < 2:
        return []
    
    if len(points) == 2:
        p0, p1 = points[0], points[1]
        if tangents is not None and len(tangents) >= 2:
            cp1 = p0 + tangents[0] / 3.0
            cp2 = p1 - tangents[1] / 3.0
        else:
            cp1 = p0 + (p1 - p0) / 3
            cp2 = p0 + 2 * (p1 - p0) / 3
        return [(p0, cp1, cp2, p1)]
    
    segments = []
    n = len(points)
    
    for i in range(n - 1):
        p0 = points[i]
        p3 = points[i + 1]
        
        if tangents is not None and len(tangents) == n:
            cp1 = p0 + tangents[i] / 3.0
            cp2 = p3 - tangents[i + 1] / 3.0
        else:
            if i == 0:
                p_prev = points[0]
            else:
                p_prev = points[i - 1]
            
            if i + 2 >= n:
                p_next = points[-1]
            else:
                p_next = points[i + 2]
            
            t0 = (p3 - p_prev) / 6.0
            t1 = (p_next - p0) / 6.0
            
            cp1 = p0 + t0
            cp2 = p3 - t1
        
        segments.append((p0, cp1, cp2, p3))
    
    return segments


def circle_arc_to_bezier(cx, cy, r, start_angle, end_angle):
    """Convert a circular arc to cubic bezier curves."""
    angle_span = end_angle - start_angle
    num_segments = max(1, int(math.ceil(abs(angle_span) / (math.pi / 2))))
    segment_angle = angle_span / num_segments
    
    segments = []
    for i in range(num_segments):
        a1 = start_angle + i * segment_angle
        a2 = start_angle + (i + 1) * segment_angle
        
        k = 4.0 / 3.0 * math.tan((a2 - a1) / 4.0)
        
        x0 = cx + r * math.cos(a1)
        y0 = cy + r * math.sin(a1)
        
        x3 = cx + r * math.cos(a2)
        y3 = cy + r * math.sin(a2)
        
        x1 = x0 - k * r * math.sin(a1)
        y1 = y0 + k * r * math.cos(a1)
        
        x2 = x3 + k * r * math.sin(a2)
        y2 = y3 - k * r * math.cos(a2)
        
        segments.append((
            np.array([x0, y0]),
            np.array([x1, y1]),
            np.array([x2, y2]),
            np.array([x3, y3])
        ))
    
    return segments


def ellipse_arc_to_bezier(cx, cy, rx, ry, start_angle, end_angle):
    """Convert an elliptical arc to cubic bezier curves."""
    angle_span = end_angle - start_angle
    num_segments = max(1, int(math.ceil(abs(angle_span) / (math.pi / 2))))
    segment_angle = angle_span / num_segments
    
    segments = []
    for i in range(num_segments):
        a1 = start_angle + i * segment_angle
        a2 = start_angle + (i + 1) * segment_angle
        
        k = 4.0 / 3.0 * math.tan((a2 - a1) / 4.0)
        
        x0 = cx + rx * math.cos(a1)
        y0 = cy + ry * math.sin(a1)
        
        x3 = cx + rx * math.cos(a2)
        y3 = cy + ry * math.sin(a2)
        
        x1 = x0 - k * rx * math.sin(a1)
        y1 = y0 + k * ry * math.cos(a1)
        
        x2 = x3 + k * rx * math.sin(a2)
        y2 = y3 - k * ry * math.cos(a2)
        
        segments.append((
            np.array([x0, y0]),
            np.array([x1, y1]),
            np.array([x2, y2]),
            np.array([x3, y3])
        ))
    
    return segments


def save_shape_to_svg_with_scale(edges, filename, scale, use_sampled=False):
    """Save edges to SVG file with pre-computed scale."""
    dwg = svgwrite.Drawing(
        filename,
        size=("200", "200"),
        viewBox="0 0 200 200",
        profile="full"
    )

    if not edges:
        dwg.save()
        return

    all_pts = []
    for edge in edges:
        if edge['type'] == 'line':
            all_pts.extend([edge['start'], edge['end']])
        elif edge['type'] == 'circle':
            cx, cy = edge['center']
            r = edge['radius']
            all_pts.extend([np.array([cx - r, cy - r]), np.array([cx + r, cy + r])])
        elif edge['type'] == 'ellipse':
            cx, cy = edge['center']
            rx, ry = edge['major_radius'], edge['minor_radius']
            all_pts.extend([np.array([cx - rx, cy - ry]), np.array([cx + rx, cy + ry])])
        else:
            all_pts.extend(edge['points'])
    
    pts_arr = np.array(all_pts)
    min_p = pts_arr.min(axis=0)
    max_p = pts_arr.max(axis=0)
    local_center = (min_p + max_p) / 2.0

    local_scale = scale * (160.0 / 420.0)

    def T(p):
        return (
            (p[0] - local_center[0]) * local_scale + 100,
            (p[1] - local_center[1]) * local_scale + 100
        )

    for edge in edges:
        path = dwg.path(
            fill="none",
            stroke="#000000",
            stroke_width=0.75,
            stroke_linecap="round",
            stroke_linejoin="bevel"
        )
        path.attribs['fill-rule'] = 'evenodd'
        path.attribs['stroke-opacity'] = '1'

        if edge['type'] == 'line':
            p0 = T(edge['start'])
            p1 = T(edge['end'])
            path.push(f"M {p0[0]},{p0[1]}")
            path.push(f"L {p1[0]},{p1[1]}")

        elif edge['type'] == 'circle':
            if use_sampled and 'points' in edge:
                pts = edge['points']
                tangents = edge.get('tangents')
                segments = fit_bezier_to_points(pts, tangents)
                
                if segments:
                    p0 = T(segments[0][0])
                    path.push(f"M {p0[0]},{p0[1]}")
                    
                    for _, cp1, cp2, p_end in segments:
                        c1 = T(cp1)
                        c2 = T(cp2)
                        pe = T(p_end)
                        path.push(f"C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {pe[0]},{pe[1]}")
            else:
                cx, cy = edge['center']
                r = edge['radius']
                start_param = edge['start_param']
                end_param = edge['end_param']
                
                segments = circle_arc_to_bezier(cx, cy, r, -start_param, -end_param)
                
                if segments:
                    p0 = T(segments[0][0])
                    path.push(f"M {p0[0]},{p0[1]}")
                    
                    for _, cp1, cp2, p_end in segments:
                        c1 = T(cp1)
                        c2 = T(cp2)
                        pe = T(p_end)
                        path.push(f"C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {pe[0]},{pe[1]}")

        elif edge['type'] == 'ellipse':
            if use_sampled and 'points' in edge:
                pts = edge['points']
                tangents = edge.get('tangents')
                segments = fit_bezier_to_points(pts, tangents)
                
                if segments:
                    p0 = T(segments[0][0])
                    path.push(f"M {p0[0]},{p0[1]}")
                    
                    for _, cp1, cp2, p_end in segments:
                        c1 = T(cp1)
                        c2 = T(cp2)
                        pe = T(p_end)
                        path.push(f"C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {pe[0]},{pe[1]}")
            else:
                cx, cy = edge['center']
                rx = edge['major_radius']
                ry = edge['minor_radius']
                start_param = edge['start_param']
                end_param = edge['end_param']
                
                segments = ellipse_arc_to_bezier(cx, cy, rx, ry, -start_param, -end_param)
                
                if segments:
                    p0 = T(segments[0][0])
                    path.push(f"M {p0[0]},{p0[1]}")
                    
                    for _, cp1, cp2, p_end in segments:
                        c1 = T(cp1)
                        c2 = T(cp2)
                        pe = T(p_end)
                        path.push(f"C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {pe[0]},{pe[1]}")

        else:
            pts = edge['points']
            tangents = edge.get('tangents')
            segments = fit_bezier_to_points(pts, tangents)
            
            if segments:
                p0 = T(segments[0][0])
                path.push(f"M {p0[0]},{p0[1]}")
                
                for _, cp1, cp2, p_end in segments:
                    c1 = T(cp1)
                    c2 = T(cp2)
                    pe = T(p_end)
                    path.push(f"C {c1[0]},{c1[1]} {c2[0]},{c2[1]} {pe[0]},{pe[1]}")

        dwg.add(path)

    dwg.save()


def project_step_to_svg(step_path, output_folder, base_filename):
    """Project a STEP file to SVG views and save to output folder."""
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        return False

    reader.TransferRoots()
    shape = reader.OneShape()

    # Check if shape is valid
    if shape is None or shape.IsNull():
        return False

    views = {
        "Front": gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)),
        "Right": gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0), gp_Dir(-1, 0, 0)),
        "Top":   gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(0, 1, 0)),
        "FrontTopRight":   gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 1, 1), gp_Dir(-1, 1, 0))
    } 

    os.makedirs(output_folder, exist_ok=True)

    # First pass: collect all edges and points from all views
    view_data = {}
    global_all_pts = []

    for view_name, axes in views.items():
        projector = HLRAlgo_Projector(axes)
        hlr = HLRBRep_Algo()
        try:
            hlr.Add(shape)
        except Exception as e:
            # Shape might be invalid or incompatible with HLR algorithm
            return False
        hlr.Projector(projector)
        hlr.Update()
        hlr.Hide()

        hlr_shape = HLRBRep_HLRToShape(hlr)

        compounds = [
            hlr_shape.VCompound(),
            hlr_shape.OutLineVCompound(),
            hlr_shape.Rg1LineVCompound()
        ]

        edges, all_pts = extract_edges_from_compounds(compounds)
        view_data[view_name] = edges
        global_all_pts.extend(all_pts)

    if not global_all_pts:
        return False

    pts = np.array(global_all_pts)
    min_p = pts.min(axis=0)
    max_p = pts.max(axis=0)
    global_size = max(max_p - min_p)
    global_scale = 420.0 / max(global_size, 1e-9)

    # Second pass: save SVGs with unified scale
    for view_name, edges in view_data.items():
        svg_path = os.path.join(output_folder, f"{base_filename}_{view_name}.svg")
        use_sampled = True
        save_shape_to_svg_with_scale(edges, svg_path, global_scale, use_sampled)

    return True


def main():
    # Define paths
    src_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\cad_step"
    out_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\new_svg_raw"
    
    # Get all subfolders
    subfolders = sorted([d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))])
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    total_success = 0
    total_failed = 0
    failed_files = []
    
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(src_base, subfolder)
        step_files = sorted(glob.glob(os.path.join(subfolder_path, "*.step")))
        
        for step_file in step_files:
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(step_file))[0]
            
            # Create output folder: out_base/subfolder/filename/
            output_folder = os.path.join(out_base, subfolder, filename)
            
            try:
                success = project_step_to_svg(step_file, output_folder, filename)
                if success:
                    total_success += 1
                else:
                    total_failed += 1
                    failed_files.append(step_file)
            except Exception as e:
                total_failed += 1
                failed_files.append(step_file)
                print(f"\n[!] Error processing {step_file}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Success: {total_success}")
    print(f"Failed: {total_failed}")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files[:20]:  # Show first 20 failed files
            print(f"  - {f}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")


if __name__ == "__main__":
    main()
