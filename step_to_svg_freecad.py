"""
FreeCAD Script: Convert STEP files to Engineering Drawing SVGs
Generates 4 views with unified scaling: Front, Top, Right, and Isometric
"""
import FreeCAD as App
import FreeCADGui as Gui
import Part
import TechDraw
import TechDrawGui
import os
import re
import xml.etree.ElementTree as ET
from FreeCAD import Vector

# ==================== CONFIGURATION ====================
STEP_DIR = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\Converted_Steps"
OUT_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\FreeCAD_SVGs"
TD_TEMPLATE = r"C:\Program Files\FreeCAD 1.0\data\Mod\TechDraw\Templates\A4_Landscape_TD.svg"

# SVG output dimensions
SVG_WIDTH = 200
SVG_HEIGHT = 200

os.makedirs(OUT_ROOT, exist_ok=True)

# ==================== VIEW CONFIGURATIONS ====================
VIEWS = {
    "Front": Vector(0.0, -1.0, 0.0),
    "Top": Vector(0.0, 0.0, 1.0),
    "Right": Vector(1.0, 0.0, 0.0),
    "FrontTopRight": Vector(1.0, -1.0, 1.0)  # Isometric view
}

def extract_view_paths(svg_path, view_name):
    """Extract path elements from a TechDraw SVG."""
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  ⚠️  {view_name}: Failed to parse SVG - {e}")
        return []
    
    ns = {"svg": "http://www.w3.org/2000/svg"}
    
    # Collect all path elements directly
    paths_xml = []
    for elem in root.findall(".//svg:path", ns):
        d = elem.get("d", "")
        # Skip empty or degenerate paths
        if d.strip() and len(d.strip()) > 10:
            paths_xml.append(ET.tostring(elem, encoding="unicode"))
    
    if not paths_xml:
        all_elems = list(root.findall(".//*", ns))
        print(f"  ⚠️  {view_name}: No valid paths (found {len(all_elems)} total elements)")
        return []
    
    return paths_xml

def parse_path_data(path_elem):
    """Extract d attribute from path element."""
    if isinstance(path_elem, str):
        match = re.search(r'd="([^"]+)"', path_elem)
        if match:
            return match.group(1)
        return None
    else:
        return path_elem.get("d", "")

def get_path_bounds(d_attr):
    """Calculate bounding box of a path's d attribute."""
    # Extract all numbers including scientific notation
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', d_attr)
    if not numbers:
        return None
    
    coords = [float(n) for n in numbers]
    # Get x,y pairs
    xs = []
    ys = []
    
    i = 0
    while i < len(coords):
        xs.append(coords[i])
        if i + 1 < len(coords):
            ys.append(coords[i + 1])
        i += 2
    
    if not xs or not ys:
        return None
    
    return {
        'min_x': min(xs),
        'max_x': max(xs),
        'min_y': min(ys),
        'max_y': max(ys)
    }

def scale_path_data(d_attr, scale, tx, ty):
    """Scale SVG path data with proper coordinate transformation."""
    
    def scale_coord_pair(x_str, y_str):
        x = float(x_str) * scale + tx
        y = float(y_str) * scale + ty
        return f"{x:.4f},{y:.4f}"
    
    scaled_d = d_attr
    
    # M and L commands: M x,y or L x,y
    scaled_d = re.sub(
        r'([ML])\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,?\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        lambda m: f"{m.group(1)} {scale_coord_pair(m.group(2), m.group(3))}",
        scaled_d
    )
    
    # C command: C x1,y1 x2,y2 x,y
    scaled_d = re.sub(
        r'([C])\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,?\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,?\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*,?\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        lambda m: f"{m.group(1)} {scale_coord_pair(m.group(2), m.group(3))} {scale_coord_pair(m.group(4), m.group(5))} {scale_coord_pair(m.group(6), m.group(7))}",
        scaled_d
    )
    
    return scaled_d

def apply_unified_scale(view_paths_dict, target_width=SVG_WIDTH, target_height=SVG_HEIGHT, margin=10):
    """Apply unified scaling across all views to maintain proportions."""
    
    # Calculate global bounding box across ALL views
    all_bounds = []
    for view_name, paths_xml in view_paths_dict.items():
        for path in paths_xml:
            d_attr = parse_path_data(path)
            if d_attr:
                bounds = get_path_bounds(d_attr)
                if bounds:
                    all_bounds.append(bounds)
    
    if not all_bounds:
        print("    ⚠️  No valid bounds found across all views")
        return {}
    
    # Global bounds
    global_min_x = min(b['min_x'] for b in all_bounds)
    global_max_x = max(b['max_x'] for b in all_bounds)
    global_min_y = min(b['min_y'] for b in all_bounds)
    global_max_y = max(b['max_y'] for b in all_bounds)
    
    global_width = global_max_x - global_min_x
    global_height = global_max_y - global_min_y
    
    if global_width < 1e-6 or global_height < 1e-6:
        print(f"    ⚠️  Degenerate global bounds (w={global_width}, h={global_height})")
        return {}
    
    # Single scale factor for all views
    target_draw_width = target_width - 2 * margin
    target_draw_height = target_height - 2 * margin
    global_scale = min(target_draw_width / global_width, target_draw_height / global_height)
    
    print(f"    Global scale: {global_scale:.4f} (bounds: {global_width:.2f}×{global_height:.2f})")
    
    # Scale each view with the global scale
    scaled_views = {}
    for view_name, paths_xml in view_paths_dict.items():
        # Get this view's bounds
        view_bounds = []
        for path in paths_xml:
            d_attr = parse_path_data(path)
            if d_attr:
                bounds = get_path_bounds(d_attr)
                if bounds:
                    view_bounds.append(bounds)
        
        if not view_bounds:
            continue
        
        view_min_x = min(b['min_x'] for b in view_bounds)
        view_max_x = max(b['max_x'] for b in view_bounds)
        view_min_y = min(b['min_y'] for b in view_bounds)
        view_max_y = max(b['max_y'] for b in view_bounds)
        
        view_width = view_max_x - view_min_x
        view_height = view_max_y - view_min_y
        
        # Center this view in canvas using global scale
        tx = margin + (target_draw_width - view_width * global_scale) / 2 - view_min_x * global_scale
        ty = margin + (target_draw_height - view_height * global_scale) / 2 - view_min_y * global_scale
        
        # Scale paths
        scaled_paths = []
        for path_xml in paths_xml:
            d_attr = parse_path_data(path_xml)
            if not d_attr:
                continue
            
            scaled_d = scale_path_data(d_attr, global_scale, tx, ty)
            scaled_path = re.sub(r'd="[^"]*"', f'd="{scaled_d}"', path_xml)
            scaled_paths.append(scaled_path)
        
        scaled_views[view_name] = scaled_paths
    
    return scaled_views

def create_normalized_svg(paths_xml, out_path, width=SVG_WIDTH, height=SVG_HEIGHT):
    """Create a normalized SVG with given paths (already scaled)."""
    if not paths_xml:
        return
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:ev="http://www.w3.org/2001/xml-events" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'baseProfile="full" height="{height}" version="1.1" '
                f'viewBox="0 0 {width} {height}" width="{width}">\n')
        f.write('\t<defs/>\n')
        
        for p in paths_xml:
            # Clean namespace prefixes and remove vector-effect
            p_clean = p.replace('ns0:', '').replace('xmlns:ns0="http://www.w3.org/2000/svg"', '')
            p_clean = p_clean.replace(' vector-effect="none"', '')
            p_clean = p_clean.replace(' vector-effect="non-scaling-stroke"', '')
            
            # Add stroke attributes if not present
            if 'stroke=' not in p_clean:
                p_clean = p_clean.replace('fill-rule="evenodd"', 
                    'fill="none" fill-rule="evenodd" stroke="#000000" '
                    'stroke-linecap="round" stroke-linejoin="bevel" '
                    'stroke-opacity="1" stroke-width="0.75"')
            
            f.write("\t")
            f.write(p_clean)
            f.write("\n")
        
        f.write("</svg>\n")

def export_one_step(step_path):
    """Convert one STEP file to multiple SVG views with unified scaling."""
    base = os.path.splitext(os.path.basename(step_path))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing: {base}")
    
    # Create new document
    doc = App.newDocument(f"Doc_{base}")
    
    # Import STEP file
    try:
        shape = Part.Shape()
        shape.read(step_path)
        part_obj = doc.addObject("Part::Feature", "ImportedPart")
        part_obj.Shape = shape
    except Exception as e:
        print(f"  ❌ Failed to import STEP: {e}")
        App.closeDocument(doc.Name)
        return
    
    # Step 1: Export all views to temporary SVGs
    temp_svgs = {}
    for view_name, direction in VIEWS.items():
        try:
            # Create new page for this view
            page = doc.addObject("TechDraw::DrawPage", f"Page_{view_name}")
            template = doc.addObject("TechDraw::DrawSVGTemplate", f"Template_{view_name}")
            template.Template = TD_TEMPLATE
            page.Template = template
            
            # Add single view to page
            view_obj = doc.addObject("TechDraw::DrawViewPart", f"{view_name}View")
            view_obj.Source = [part_obj]
            view_obj.Direction = direction
            view_obj.Scale = 1.0
            
            # Set view properties
            if hasattr(view_obj, 'ShowSmoothLines'):
                view_obj.ShowSmoothLines = True
            if hasattr(view_obj, 'ShowSeamLines'):
                view_obj.ShowSeamLines = False
            if hasattr(view_obj, 'ShowIsoLines'):
                view_obj.ShowIsoLines = False
            
            page.addView(view_obj)
            
            # Recompute multiple times
            doc.recompute()
            doc.recompute()
            Gui.updateGui()
            
            # Export to temporary SVG
            tmp_svg = os.path.join(out_dir, f"_temp_{view_name}.svg")
            try:
                TechDrawGui.exportPageAsSvg(page, tmp_svg)
            except Exception as ex:
                print(f"  ⚠️  {view_name}: Export warning - {ex}")
                continue
            
            # Verify export
            if os.path.exists(tmp_svg) and os.path.getsize(tmp_svg) > 100:
                temp_svgs[view_name] = tmp_svg
            else:
                print(f"  ⚠️  {view_name}: Invalid/empty export")
                
        except Exception as e:
            print(f"  ⚠️  {view_name}: {e}")
    
    # Step 2: Extract paths from all views
    view_paths = {}
    for view_name, tmp_svg in temp_svgs.items():
        paths = extract_view_paths(tmp_svg, view_name)
        if paths:
            view_paths[view_name] = paths
    
    if not view_paths:
        print(f"  ❌ No valid views extracted")
        App.closeDocument(doc.Name)
        return
    
    # Step 3: Apply unified scaling across all views
    scaled_views = apply_unified_scale(view_paths)
    
    # Step 4: Save each view
    for view_name, scaled_paths in scaled_views.items():
        out_svg = os.path.join(out_dir, f"{base}_{view_name}.svg")
        create_normalized_svg(scaled_paths, out_svg)
        print(f"  ✅ {view_name}: {len(scaled_paths)} paths")
    
    # Step 5: Clean up temp files
    for tmp_svg in temp_svgs.values():
        if os.path.exists(tmp_svg):
            os.remove(tmp_svg)
    
    # Close document
    App.closeDocument(doc.Name)

def main():
    """Process all STEP files in the input directory."""
    step_files = sorted([f for f in os.listdir(STEP_DIR) 
                        if f.lower().endswith((".step", ".stp"))])
    
    print(f"Found {len(step_files)} STEP files")
    print("=" * 60)
    
    for i, fname in enumerate(step_files, 1):
        print(f"[{i}/{len(step_files)}] {fname}")
        try:
            export_one_step(os.path.join(STEP_DIR, fname))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        print()
    
    print("=" * 60)
    print("✅ Done! Check output directory:")
    print(f"   {OUT_ROOT}")

if __name__ == "__main__":
    main()
