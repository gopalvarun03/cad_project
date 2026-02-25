import os
import glob
import argparse
import math
import numpy as np
import svgwrite

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
# kappa = 4/3 * (sqrt(2) - 1) ≈ 0.5522847498
KAPPA = 4.0 / 3.0 * (math.sqrt(2.0) - 1.0)


# ------------------------------------------------------------
# Extract edges from compounds
# ------------------------------------------------------------
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
                # Lines: just need start and end points
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
                # Sample the actual projected curve points WITH tangents
                circ = adaptor.Circle()
                center = circ.Location()
                radius = circ.Radius()
                
                cx, cy = center.X(), -center.Y()
                
                # Add bounding points for scaling
                all_pts.append(np.array([cx - radius, cy - radius]))
                all_pts.append(np.array([cx + radius, cy + radius]))
                
                # Sample points and tangents along the actual projected curve
                num_samples = 5
                pts = []
                tangents = []
                for i in range(num_samples):
                    u = u1 + (u2 - u1) * i / (num_samples - 1)
                    p = adaptor.Value(u)
                    pt = np.array([p.X(), -p.Y()])
                    pts.append(pt)
                    
                    # Get derivative (tangent) at this point
                    from OCC.Core.gp import gp_Pnt as gp_Pnt2, gp_Vec
                    pnt = gp_Pnt2()
                    vec = gp_Vec()
                    adaptor.D1(u, pnt, vec)
                    # Scale tangent by parameter range for proper interpolation
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
                # Sample the actual projected curve points WITH tangents
                ellipse = adaptor.Ellipse()
                center = ellipse.Location()
                major_r = ellipse.MajorRadius()
                minor_r = ellipse.MinorRadius()
                
                cx, cy = center.X(), -center.Y()
                
                # Add bounding points
                all_pts.append(np.array([cx - major_r, cy - major_r]))
                all_pts.append(np.array([cx + major_r, cy + major_r]))
                
                # Sample points and tangents
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
                # BSpline, Bezier, or other - sample points with tangents
                num_samples = 5
                pts = []
                tangents = []
                for i in range(num_samples):
                    u = u1 + (u2 - u1) * i / (num_samples - 1)
                    p = adaptor.Value(u)
                    pt = np.array([p.X(), -p.Y()])
                    pts.append(pt)
                    all_pts.append(pt)
                    
                    # Get tangent
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
    """
    Fit cubic bezier curves to a sequence of points.
    If tangents are provided, uses Hermite interpolation for perfect curves.
    Otherwise falls back to Catmull-Rom.
    Returns list of bezier segments: [(p0, cp1, cp2, p3), ...]
    """
    if len(points) < 2:
        return []
    
    if len(points) == 2:
        p0, p1 = points[0], points[1]
        if tangents is not None and len(tangents) >= 2:
            # Hermite to Bezier conversion
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
            # Hermite interpolation: use actual tangents
            # Bezier control points from Hermite: cp1 = p0 + t0/3, cp2 = p3 - t1/3
            cp1 = p0 + tangents[i] / 3.0
            cp2 = p3 - tangents[i + 1] / 3.0
        else:
            # Catmull-Rom fallback
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
    """
    Convert a circular arc to cubic bezier curves using the standard kappa approximation.
    Returns list of bezier segments for perfect circle rendering.
    """
    # Normalize angle span
    angle_span = end_angle - start_angle
    
    # Split into segments of max 90 degrees (pi/2) for accuracy
    num_segments = max(1, int(math.ceil(abs(angle_span) / (math.pi / 2))))
    segment_angle = angle_span / num_segments
    
    segments = []
    for i in range(num_segments):
        a1 = start_angle + i * segment_angle
        a2 = start_angle + (i + 1) * segment_angle
        
        # kappa factor for this arc segment: 4/3 * tan(angle/4)
        k = 4.0 / 3.0 * math.tan((a2 - a1) / 4.0)
        
        # Start point
        x0 = cx + r * math.cos(a1)
        y0 = cy + r * math.sin(a1)
        
        # End point
        x3 = cx + r * math.cos(a2)
        y3 = cy + r * math.sin(a2)
        
        # Control point 1 (perpendicular to radius at start)
        x1 = x0 - k * r * math.sin(a1)
        y1 = y0 + k * r * math.cos(a1)
        
        # Control point 2 (perpendicular to radius at end)
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
    """
    Convert an elliptical arc to cubic bezier curves.
    """
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


# ------------------------------------------------------------
# STEP → SVG projection
# ------------------------------------------------------------
def project_step_to_svg(step_path, output_dir):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"[!] Failed to read {step_path}")
        return

    reader.TransferRoots()
    shape = reader.OneShape()

    views = {
        "front": gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, -1, 0), gp_Dir(0, 0, 1)),
        "right":   gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1),  gp_Dir(0, 1, 0)),
        "top": gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0),  gp_Dir(0, 0, 1)),
        "iso":   gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(-0.70, -1.49, 0.55), gp_Dir(-0.245, -0.24, 0.95))
    }

    base_name = os.path.splitext(os.path.basename(step_path))[0]
    save_folder = os.path.join(output_dir, base_name)
    os.makedirs(save_folder, exist_ok=True)

    # First pass: collect all edges and points from all views
    view_data = {}
    global_all_pts = []

    for view_name, axes in views.items():
        projector = HLRAlgo_Projector(axes)
        hlr = HLRBRep_Algo()
        hlr.Add(shape)
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

    # Compute global scale from all views
    if not global_all_pts:
        print(f"[!] No geometry found in {step_path}")
        return

    pts = np.array(global_all_pts)
    min_p = pts.min(axis=0)
    max_p = pts.max(axis=0)
    global_size = max(max_p - min_p)
    global_scale = 420.0 / max(global_size, 1e-9)

    # Second pass: save SVGs with unified scale but per-view centering
    for view_name, edges in view_data.items():
        svg_path = os.path.join(save_folder, f"{view_name}.svg")
        # Use sampled points with tangents for all views for accurate curves
        use_sampled = True
        save_shape_to_svg_with_scale(edges, svg_path, global_scale, use_sampled)
        print(f"[✓] {view_name} view saved")


# ------------------------------------------------------------
# SVG writer with pre-computed scale (PATH based: M / L / C)
# ------------------------------------------------------------
def save_shape_to_svg_with_scale(edges, filename, scale, use_sampled=False):
    # Use 200x200 with viewBox like reference SVGs
    dwg = svgwrite.Drawing(
        filename,
        size=("200", "200"),
        viewBox="0 0 200 200",
        profile="full"
    )

    if not edges:
        dwg.save()
        return

    # Compute local center for this view (for centering in canvas)
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

    # Scale to fit in 200x200 canvas with margin (use 160 for content)
    local_scale = scale * (160.0 / 420.0)  # Convert from 420/500 to 160/200

    def T(p):
        return (
            (p[0] - local_center[0]) * local_scale + 100,
            (p[1] - local_center[1]) * local_scale + 100
        )

    # -------- Build SVG paths ----------
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
            # For ISO view, use sampled points with tangents since 3D circles become projected ellipses
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
                # For orthographic views, use exact mathematical bezier
                cx, cy = edge['center']
                r = edge['radius']
                start_param = edge['start_param']
                end_param = edge['end_param']
                
                # Y is flipped, so negate angles
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
            # For ISO view, use sampled points with tangents
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
                # For orthographic views, use exact mathematical bezier
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
            # spline - use fitted beziers from sampled points with tangents
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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    step_files = sorted(glob.glob(os.path.join(args.src, "*.step")))

    for sf in step_files:
        print(f"Processing: {os.path.basename(sf)}")
        project_step_to_svg(sf, args.out)