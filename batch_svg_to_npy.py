import xml.etree.ElementTree as ET
import numpy as np
from svg.path import parse_path, Line, CubicBezier
import sys
import re
import os
from tqdm import tqdm

# --- Define constants based on the project ---
SVG_SOS_IDX = 0
SVG_EOS_IDX = 1
SVG_L_IDX = 2  # LineTo
SVG_C_IDX = 3  # CubicBezier

SVG_N_ARGS = 8
PAD_VAL = -1.0

# --- Normalization constants ---
TARGET_MIN = 0.0
TARGET_MAX = 255.0
TARGET_RANGE = TARGET_MAX - TARGET_MIN

# --- Sequence length constant ---
N_COMMANDS = 100
N_PARAMS = 10  # 1 (view) + 1 (cmd) + 8 (args)


def get_sos_vec(view_label):
    return np.array([view_label, SVG_SOS_IDX, *([PAD_VAL] * SVG_N_ARGS)], dtype=np.float32)


def get_eos_vec(view_label):
    return np.array([view_label, SVG_EOS_IDX, *([PAD_VAL] * SVG_N_ARGS)], dtype=np.float32)


def normalize_coords(x, y, vb):
    """Normalizes and TRUNCATES coordinates to the project's format."""
    min_x, min_y, vb_width, vb_height = vb
    if vb_width == 0 or vb_height == 0:
        return (TARGET_MIN, TARGET_MIN)
    
    norm_x = ((x - min_x) / vb_width) * TARGET_RANGE + TARGET_MIN
    norm_y = ((y - min_y) / vb_height) * TARGET_RANGE + TARGET_MIN
    
    return np.floor(norm_x), np.floor(norm_y)


def format_line_vec(segment, view_label, vb):
    x1, y1 = normalize_coords(segment.start.real, segment.start.imag, vb)
    x2, y2 = normalize_coords(segment.end.real, segment.end.imag, vb)
    return np.array([
        view_label, SVG_L_IDX,
        x1, y1, PAD_VAL, PAD_VAL, PAD_VAL, PAD_VAL, x2, y2
    ], dtype=np.float32)


def format_bezier_vec(segment, view_label, vb):
    x1, y1 = normalize_coords(segment.start.real, segment.start.imag, vb)
    cx1, cy1 = normalize_coords(segment.control1.real, segment.control1.imag, vb)
    cx2, cy2 = normalize_coords(segment.control2.real, segment.control2.imag, vb)
    x2, y2 = normalize_coords(segment.end.real, segment.end.imag, vb)
    return np.array([
        view_label, SVG_C_IDX,
        x1, y1, cx1, cy1, cx2, cy2, x2, y2
    ], dtype=np.float32)


def get_viewbox(root):
    viewBox_str = root.get('viewBox')
    if viewBox_str:
        vb = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", viewBox_str)]
        if len(vb) == 4:
            return vb
    width = root.get('width')
    height = root.get('height')
    if width and height:
        try:
            w = float(re.findall(r"[-+]?\d*\.?\d+", width)[0])
            h = float(re.findall(r"[-+]?\d*\.?\d+", height)[0])
            return [0.0, 0.0, w, h]
        except (IndexError, TypeError):
            pass
    return [0.0, 0.0, 255.0, 255.0]


def convert_svg_to_sequence(svg_file_path, view_label=0):
    """Convert a single SVG file to a sequence of commands."""
    command_sequence = []
    try:
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        vb = get_viewbox(root)
        if vb[2] == 0 or vb[3] == 0:
            return None
            
        command_sequence.append(get_sos_vec(view_label))
        
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        elements = root.findall('.//svg:path', namespaces)
        elements.extend(root.findall('.//svg:line', namespaces))
        if not elements:
            elements = root.findall('.//{http://www.w3.org/2000/svg}path')
            elements.extend(root.findall('.//{http://www.w3.org/2000/svg}line'))
            
        for elem in elements:
            if elem.tag.endswith('line'):
                x1 = float(elem.get('x1', 0))
                y1 = float(elem.get('y1', 0))
                x2 = float(elem.get('x2', 0))
                y2 = float(elem.get('y2', 0))
                mock_segment = lambda: None
                mock_segment.start = complex(x1, y1)
                mock_segment.end = complex(x2, y2)
                command_sequence.append(format_line_vec(mock_segment, view_label, vb))
            
            elif elem.tag.endswith('path'):
                d_string = elem.get('d')
                if not d_string:
                    continue
                parsed_segments = parse_path(d_string)
                for segment in parsed_segments:
                    if isinstance(segment, Line):
                        command_sequence.append(format_line_vec(segment, view_label, vb))
                    elif isinstance(segment, CubicBezier):
                        command_sequence.append(format_bezier_vec(segment, view_label, vb))
        
        command_sequence.append(get_eos_vec(view_label))
        return np.array(command_sequence, dtype=np.float32)

    except ET.ParseError:
        return None
    except Exception:
        return None


def pad_sequence(sequence, view_label, target_length=N_COMMANDS):
    """Pads or truncates a sequence to the target_length."""
    if sequence is None or len(sequence) == 0:
        # Create empty sequence with just SOS and EOS, then pad
        sequence = np.array([get_sos_vec(view_label), get_eos_vec(view_label)], dtype=np.float32)
    
    current_length = sequence.shape[0]
    
    if current_length > target_length:
        padded_sequence = sequence[:target_length]
        padded_sequence[-1] = get_eos_vec(view_label)
    elif current_length < target_length:
        num_padding = target_length - current_length
        pad_vector = get_eos_vec(view_label)
        padding = np.tile(pad_vector, (num_padding, 1))
        padded_sequence = np.vstack((sequence, padding))
    else:
        padded_sequence = sequence
        
    return padded_sequence


def process_drawing_folder(folder_path, base_filename):
    """
    Processes the 4 standard SVG views from a folder.
    New naming convention: {base_filename}_Front.svg, {base_filename}_Top.svg, etc.
    Returns a stacked 400x10 array or None if failed.
    """
    
    # Map view names to labels (matching the original convention)
    # views_to_process = [
    #     ("Front", 0),
    #     ("Top", 1),
    #     ("Right", 2),
    #     ("FrontTopRight", 3)  # This is the "iso" view
    # ]
    views_to_process = [
        ("front", 0),
        ("top", 1),
        ("right", 2),
        ("iso", 3)  # This is the "iso" view
    ]
    
    all_view_sequences = []
    
    for suffix, label in views_to_process:
        # svg_file_name = f"{base_filename}_{suffix}.svg"
        svg_file_name = f"{suffix}.svg"
        svg_file_path = os.path.join(folder_path, svg_file_name)
        
        if not os.path.exists(svg_file_path):
            raw_sequence = None
        else:
            raw_sequence = convert_svg_to_sequence(svg_file_path, view_label=label)

        padded_sequence = pad_sequence(raw_sequence, view_label=label)
        all_view_sequences.append(padded_sequence)
        
    try:
        final_stack = np.vstack(all_view_sequences)
        return final_stack
    except ValueError:
        return None


def main():
    # Define paths
    # src_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\new_svg_raw"
    src_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2"
    # out_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\new_svg_vec"
    out_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\test_npy"
    
    # Check if source exists
    if not os.path.exists(src_base):
        print(f"Error: Source folder does not exist: {src_base}")
        return
    
    # Get all subfolders
    # subfolders = sorted([d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))])
    # subfolders = [r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\test_h5_files"]
    subfolders = [r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\test_rotated2"]
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    total_success = 0
    total_failed = 0
    failed_files = []
    
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(src_base, subfolder)
        
        # Get all drawing folders (each is named like 00000007)
        drawing_folders = sorted([d for d in os.listdir(subfolder_path) 
                                  if os.path.isdir(os.path.join(subfolder_path, d))])
        
        # Create output subfolder
        out_subfolder = os.path.join(out_base, subfolder)
        os.makedirs(out_subfolder, exist_ok=True)
        
        for drawing_name in drawing_folders:
            drawing_folder_path = os.path.join(subfolder_path, drawing_name)
            
            # Create subfolder based on first 4 characters of drawing name
            first_four = drawing_name[:4]
            npy_subfolder = os.path.join(out_subfolder, first_four)
            os.makedirs(npy_subfolder, exist_ok=True)
            
            output_npy_path = os.path.join(npy_subfolder, f"{drawing_name}.npy")
            
            try:
                result = process_drawing_folder(drawing_folder_path, drawing_name)
                
                if result is not None and result.shape == (400, 10):
                    np.save(output_npy_path, result)
                    total_success += 1
                else:
                    total_failed += 1
                    failed_files.append(drawing_folder_path)
            except Exception as e:
                total_failed += 1
                failed_files.append(drawing_folder_path)
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Success: {total_success}")
    print(f"Failed: {total_failed}")
    
    if failed_files:
        print(f"\nFailed folders:")
        for f in failed_files[:20]:
            print(f"  - {f}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")


if __name__ == "__main__":
    main()


# def main():
#     # Define paths
#     # src_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\new_svg_raw"
#     src_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2"
#     # out_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\new_svg_vec"
#     out_base = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\test_npy"
    
#     # Check if source exists
#     if not os.path.exists(src_base):
#         print(f"Error: Source folder does not exist: {src_base}")
#         return
    
#     # Get all subfolders
#     # subfolders = sorted([d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))])
#     subfolders = [r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\test_h5_files"]
    
#     print(f"Found {len(subfolders)} subfolders to process")
    
#     total_success = 0
#     total_failed = 0
#     failed_files = []
    
#     for subfolder in tqdm(subfolders, desc="Processing subfolders"):
#         subfolder_path = os.path.join(src_base, subfolder)
        
#         # Get all drawing folders (each is named like 00000007)
#         drawing_folders = sorted([d for d in os.listdir(subfolder_path) 
#                                   if os.path.isdir(os.path.join(subfolder_path, d))])
        
#         # Create output subfolder
#         out_subfolder = os.path.join(out_base, subfolder)
#         os.makedirs(out_subfolder, exist_ok=True)
        
#         for drawing_name in drawing_folders:
#             drawing_folder_path = os.path.join(subfolder_path, drawing_name)
#             output_npy_path = os.path.join(out_subfolder, f"{drawing_name}.npy")
            
#             try:
#                 result = process_drawing_folder(drawing_folder_path, drawing_name)
                
#                 if result is not None and result.shape == (400, 10):
#                     np.save(output_npy_path, result)
#                     total_success += 1
#                 else:
#                     total_failed += 1
#                     failed_files.append(drawing_folder_path)
#             except Exception as e:
#                 total_failed += 1
#                 failed_files.append(drawing_folder_path)
    
#     print(f"\n{'='*50}")
#     print(f"Conversion complete!")
#     print(f"Success: {total_success}")
#     print(f"Failed: {total_failed}")
    
#     if failed_files:
#         print(f"\nFailed folders:")
#         for f in failed_files[:20]:
#             print(f"  - {f}")
#         if len(failed_files) > 20:
#             print(f"  ... and {len(failed_files) - 20} more")


# if __name__ == "__main__":
#     main()
