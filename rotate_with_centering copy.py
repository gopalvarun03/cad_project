import os
import sys
import math
import re
import numpy as np
import xml.etree.ElementTree as ET

# --- Library Imports ---
from svgpathtools import svg2paths, wsvg
from svgpathtools import Path as ToolsPath, Line as ToolsLine, CubicBezier as ToolsCubic, QuadraticBezier as ToolsQuad, Arc as ToolsArc
from svg.path import parse_path, Line as SvgLine, CubicBezier as SvgCubic

# ==============================================================================
# PART 1: GEOMETRIC ROTATION LOGIC
# ==============================================================================

def rotate_point(x, y, cx, cy, angle_degrees_ccw):
    """
    Rotates a point (x,y) around (cx,cy) by angle_degrees_ccw.
    Positive angle = Counter-Clockwise.
    """
    theta = math.radians(angle_degrees_ccw)
    # Standard 2D rotation matrix for Counter-Clockwise
    # x' = x cos(t) - y sin(t)
    # y' = x sin(t) + y cos(t)
    # Note: SVG Y-axis is down, but the math holds relative to the center.
    
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    
    x -= cx
    y -= cy
    
    xr = cos_t * x - sin_t * y + cx
    yr = sin_t * x + cos_t * y + cy
    return xr, yr

def rotate_path_obj(path, cx, cy, angle_ccw):
    new_segments = []
    for seg in path:
        if isinstance(seg, ToolsLine):
            s = rotate_point(seg.start.real, seg.start.imag, cx, cy, angle_ccw)
            e = rotate_point(seg.end.real, seg.end.imag, cx, cy, angle_ccw)
            new_segments.append(ToolsLine(complex(*s), complex(*e)))
        
        elif isinstance(seg, ToolsCubic):
            s = rotate_point(seg.start.real, seg.start.imag, cx, cy, angle_ccw)
            c1 = rotate_point(seg.control1.real, seg.control1.imag, cx, cy, angle_ccw)
            c2 = rotate_point(seg.control2.real, seg.control2.imag, cx, cy, angle_ccw)
            e = rotate_point(seg.end.real, seg.end.imag, cx, cy, angle_ccw)
            new_segments.append(ToolsCubic(complex(*s), complex(*c1), complex(*c2), complex(*e)))
            
        elif isinstance(seg, ToolsQuad):
            s = rotate_point(seg.start.real, seg.start.imag, cx, cy, angle_ccw)
            c = rotate_point(seg.control.real, seg.control.imag, cx, cy, angle_ccw)
            e = rotate_point(seg.end.real, seg.end.imag, cx, cy, angle_ccw)
            new_segments.append(ToolsQuad(complex(*s), complex(*c), complex(*e)))
            
        elif isinstance(seg, ToolsArc):
            s = rotate_point(seg.start.real, seg.start.imag, cx, cy, angle_ccw)
            e = rotate_point(seg.end.real, seg.end.imag, cx, cy, angle_ccw)
            # In SVG coordinate space (y-down), positive rotation is usually clockwise visually.
            # But since we are calculating coord transforms manually, we just adjust the rotation param.
            # For standard math CCW, we subtract the angle because SVG 'rotation' attribute is often CW.
            # We will approximate by subtracting angle_ccw.
            new_segments.append(ToolsArc(complex(*s), seg.radius, seg.rotation - angle_ccw, 
                                         seg.large_arc, seg.sweep, complex(*e)))
            
    return ToolsPath(*new_segments)

def extract_viewbox(attributes):
    for attr in attributes:
        if "viewBox" in attr:
            return list(map(float, attr["viewBox"].split()))
    for attr in attributes:
        if "width" in attr and "height" in attr:
            w = float(attr["width"].replace("px", ""))
            h = float(attr["height"].replace("px", ""))
            return [0.0, 0.0, w, h]
    return [0.0, 0.0, 255.0, 255.0]

def compute_paths_bbox(paths):
    """
    Computes the bounding box of all paths.
    Returns (min_x, min_y, max_x, max_y) or None if no valid paths.
    """
    all_min_x, all_min_y = float('inf'), float('inf')
    all_max_x, all_max_y = float('-inf'), float('-inf')
    
    for path in paths:
        if len(path) == 0:
            continue
        try:
            bbox = path.bbox()  # Returns (xmin, xmax, ymin, ymax)
            xmin, xmax, ymin, ymax = bbox
            all_min_x = min(all_min_x, xmin)
            all_max_x = max(all_max_x, xmax)
            all_min_y = min(all_min_y, ymin)
            all_max_y = max(all_max_y, ymax)
        except Exception:
            # Some paths may not have valid bbox, skip them
            continue
    
    if all_min_x == float('inf'):
        return None
    return (all_min_x, all_min_y, all_max_x, all_max_y)

def translate_path_obj(path, dx, dy):
    """
    Translates all segments in a path by (dx, dy).
    """
    new_segments = []
    for seg in path:
        if isinstance(seg, ToolsLine):
            s = complex(seg.start.real + dx, seg.start.imag + dy)
            e = complex(seg.end.real + dx, seg.end.imag + dy)
            new_segments.append(ToolsLine(s, e))
        
        elif isinstance(seg, ToolsCubic):
            s = complex(seg.start.real + dx, seg.start.imag + dy)
            c1 = complex(seg.control1.real + dx, seg.control1.imag + dy)
            c2 = complex(seg.control2.real + dx, seg.control2.imag + dy)
            e = complex(seg.end.real + dx, seg.end.imag + dy)
            new_segments.append(ToolsCubic(s, c1, c2, e))
            
        elif isinstance(seg, ToolsQuad):
            s = complex(seg.start.real + dx, seg.start.imag + dy)
            c = complex(seg.control.real + dx, seg.control.imag + dy)
            e = complex(seg.end.real + dx, seg.end.imag + dy)
            new_segments.append(ToolsQuad(s, c, e))
            
        elif isinstance(seg, ToolsArc):
            s = complex(seg.start.real + dx, seg.start.imag + dy)
            e = complex(seg.end.real + dx, seg.end.imag + dy)
            new_segments.append(ToolsArc(s, seg.radius, seg.rotation, 
                                         seg.large_arc, seg.sweep, e))
            
    return ToolsPath(*new_segments)

def create_rotated_file(input_path, output_path, angle_ccw):
    """
    Reads an SVG, rotates it by angle_ccw (Counter-Clockwise), 
    re-centers the content, and saves it.
    """
    if not os.path.exists(input_path):
        print(f"  [!] Missing Source: {os.path.basename(input_path)}")
        return False

    try:
        paths, attributes = svg2paths(input_path)
        vx, vy, vw, vh = extract_viewbox(attributes)
        
        # Center of rotation (viewBox center)
        cx, cy = vx + vw / 2.0, vy + vh / 2.0

        # Step 1: Rotate paths around viewBox center
        rotated_paths = [rotate_path_obj(p, cx, cy, angle_ccw) for p in paths]

        # Step 2: Re-center the rotated content within the viewBox
        bbox = compute_paths_bbox(rotated_paths)
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            content_cx = (min_x + max_x) / 2.0
            content_cy = (min_y + max_y) / 2.0
            
            # Calculate translation to move content center to viewBox center
            dx = cx - content_cx
            dy = cy - content_cy
            
            # Translate all paths to center them
            rotated_paths = [translate_path_obj(p, dx, dy) for p in rotated_paths]

        wsvg(
            rotated_paths,
            filename=output_path,
            svg_attributes={
                "viewBox": f"{vx} {vy} {vw} {vh}",
                "width": str(vw),
                "height": str(vh),
                "xmlns": "http://www.w3.org/2000/svg"
            }
        )
        return True
    except Exception as e:
        print(f"  [!] Error processing {os.path.basename(input_path)}: {e}")
        return False

# ==============================================================================
# PART 2: VECTORIZATION HELPERS
# ==============================================================================

SVG_SOS_IDX, SVG_EOS_IDX, SVG_L_IDX, SVG_C_IDX = 0, 1, 2, 3
PAD_VAL = -1.0
TARGET_MIN, TARGET_MAX = 0.0, 255.0
TARGET_RANGE = TARGET_MAX - TARGET_MIN
N_COMMANDS = 100
N_PARAMS = 10 

def get_sos_vec(view_label):
    return np.array([view_label, SVG_SOS_IDX, *([PAD_VAL] * 8)], dtype=np.float32)

def get_eos_vec(view_label):
    return np.array([view_label, SVG_EOS_IDX, *([PAD_VAL] * 8)], dtype=np.float32)

def normalize_coords(x, y, vb):
    min_x, min_y, vb_width, vb_height = vb
    if vb_width == 0 or vb_height == 0: return (TARGET_MIN, TARGET_MIN)
    norm_x = ((x - min_x) / vb_width) * TARGET_RANGE + TARGET_MIN
    norm_y = ((y - min_y) / vb_height) * TARGET_RANGE + TARGET_MIN
    return np.floor(norm_x), np.floor(norm_y)

def format_line_vec(segment, view_label, vb):
    x1, y1 = normalize_coords(segment.start.real, segment.start.imag, vb)
    x2, y2 = normalize_coords(segment.end.real, segment.end.imag, vb)
    return np.array([view_label, SVG_L_IDX, x1, y1, PAD_VAL, PAD_VAL, PAD_VAL, PAD_VAL, x2, y2], dtype=np.float32)

def format_bezier_vec(segment, view_label, vb):
    x1, y1 = normalize_coords(segment.start.real, segment.start.imag, vb)
    cx1, cy1 = normalize_coords(segment.control1.real, segment.control1.imag, vb)
    cx2, cy2 = normalize_coords(segment.control2.real, segment.control2.imag, vb)
    x2, y2 = normalize_coords(segment.end.real, segment.end.imag, vb)
    return np.array([view_label, SVG_C_IDX, x1, y1, cx1, cy1, cx2, cy2, x2, y2], dtype=np.float32)

def get_viewbox_xml(root):
    viewBox_str = root.get('viewBox')
    if viewBox_str:
        vb = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", viewBox_str)]
        if len(vb) == 4: return vb
    width = root.get('width')
    height = root.get('height')
    if width and height:
        try:
            w = float(re.findall(r"[-+]?\d*\.?\d+", width)[0])
            h = float(re.findall(r"[-+]?\d*\.?\d+", height)[0])
            return [0.0, 0.0, w, h]
        except: pass
    return [0.0, 0.0, 255.0, 255.0]

def convert_svg_to_sequence(svg_file_path, view_label):
    command_sequence = []
    try:
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        vb = get_viewbox_xml(root)

        command_sequence.append(get_sos_vec(view_label))
        
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        elements = root.findall('.//svg:path', namespaces) + root.findall('.//svg:line', namespaces)
        if not elements:
            elements = root.findall('.//{http://www.w3.org/2000/svg}path') + root.findall('.//{http://www.w3.org/2000/svg}line')

        for elem in elements:
            if elem.tag.endswith('line'):
                x1, y1 = float(elem.get('x1', 0)), float(elem.get('y1', 0))
                x2, y2 = float(elem.get('x2', 0)), float(elem.get('y2', 0))
                mock = type('obj', (object,), {'start': complex(x1, y1), 'end': complex(x2, y2)})
                command_sequence.append(format_line_vec(mock, view_label, vb))
            elif elem.tag.endswith('path'):
                d_string = elem.get('d')
                if d_string:
                    for seg in parse_path(d_string):
                        if isinstance(seg, SvgLine):
                            command_sequence.append(format_line_vec(seg, view_label, vb))
                        elif isinstance(seg, SvgCubic):
                            command_sequence.append(format_bezier_vec(seg, view_label, vb))
                            
        command_sequence.append(get_eos_vec(view_label))
        return np.array(command_sequence, dtype=np.float32)
    except Exception as e:
        # print(f"  [!] Error parsing {os.path.basename(svg_file_path)}: {e}")
        return None

def pad_sequence(sequence, view_label, target_length=N_COMMANDS):
    if sequence is None: sequence = np.empty((0, N_PARAMS), dtype=np.float32)
    current_length = sequence.shape[0]
    if current_length > target_length:
        padded = sequence[:target_length]
        padded[-1] = get_eos_vec(view_label)
    elif current_length < target_length:
        pad_vec = get_eos_vec(view_label)
        padding = np.tile(pad_vec, (target_length - current_length, 1))
        padded = np.vstack((sequence, padding))
    else:
        padded = sequence
    return padded

# ==============================================================================
# PART 3: MAIN PIPELINE
# ==============================================================================

def process_folder(folder_path, output_npy_path):
    base_name = os.path.basename(folder_path)
    print(f"Processing: {base_name}")

    # --- 1. DEFINE OLD FILES (INPUTS) ---
    old_front = os.path.join(folder_path, f"{base_name}_Front.svg")
    old_top   = os.path.join(folder_path, f"{base_name}_Top.svg")
    old_right = os.path.join(folder_path, f"{base_name}_Right.svg")
    old_iso   = os.path.join(folder_path, f"{base_name}_FrontTopRight.svg")

    # --- 2. DEFINE NEW FILES (OUTPUTS) ---
    new_front = os.path.join(folder_path, f"{base_name}_Front_final.svg")
    new_top   = os.path.join(folder_path, f"{base_name}_Top_final.svg")
    new_right = os.path.join(folder_path, f"{base_name}_Right_final.svg")
    new_iso   = os.path.join(folder_path, f"{base_name}_FrontTopRight_final.svg")

    print("  [~] Performing Geometric Rotations...")

    # --- 3. APPLY ROTATIONS ---
    # New Front = Old Top (Rotated 90 CCW)
    create_rotated_file(old_top, new_front, angle_ccw=90)

    # New Top = Old Right (Rotated 90 CCW)
    create_rotated_file(old_right, new_top, angle_ccw=90)

    # New Side/Right = Old Front (Rotated 90 CCW)
    create_rotated_file(old_front, new_right, angle_ccw=90)

    # New Iso = Old Iso (Rotated 120 Clockwise -> -120 CCW)
    create_rotated_file(old_iso, new_iso, angle_ccw=-120)

    print("  [+] All 4 final SVGs saved.")

    # --- 4. CREATE NPY STACK ---
    # Order: New Front (0), New Side (1), New Top (2), New Iso (3)
    
    views_config = [
        (new_front, 0),
        (new_right, 1),
        (new_top,   2),
        (new_iso,   3)
    ]

    all_sequences = []

    for fpath, label in views_config:
        raw_seq = convert_svg_to_sequence(fpath, label)
        padded_seq = pad_sequence(raw_seq, label)
        all_sequences.append(padded_seq)

    try:
        final_stack = np.vstack(all_sequences)
        np.save(output_npy_path, final_stack)
        print(f"  [+] Saved NPY: {os.path.basename(output_npy_path)}")
        print(f"  [i] Shape: {final_stack.shape}")
    except Exception as e:
        print(f"  [!] Stacking error: {e}")

if __name__ == "__main__":
    # CONFIGURATION
    INPUT_DIR = r"C:\Users\thiri\OneDrive\Desktop\2DtoCAD\Drawing2CAD\dataset\CAD-VGDrawing-20250916T140226Z-1-001\CAD-VGDrawing\svg_raw\0000\00000007"
    
    # Auto-output path setup
    if "svg_raw" in INPUT_DIR:
        base_path = INPUT_DIR.split(r"\svg_raw")[0]
        sub_path = INPUT_DIR.split(r"\svg_raw")[1]
        OUTPUT_FILE = os.path.join(base_path, "svg_vec_new" + sub_path + ".npy")
    else:
        OUTPUT_FILE = os.path.join(INPUT_DIR, "output.npy")
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    process_folder(INPUT_DIR, OUTPUT_FILE)