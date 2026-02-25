import xml.etree.ElementTree as ET
import numpy as np
from svg.path import parse_path, Line, CubicBezier
import sys
import re
import os

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
N_PARAMS = 10 # 1 (view) + 1 (cmd) + 8 (args)

# --- Vector Creation Helpers (Unchanged) ---

def get_sos_vec(view_label):
    return np.array([view_label, SVG_SOS_IDX, *([PAD_VAL] * SVG_N_ARGS)], dtype=np.float32)

def get_eos_vec(view_label):
    return np.array([view_label, SVG_EOS_IDX, *([PAD_VAL] * SVG_N_ARGS)], dtype=np.float32)

# --- ★★★ CHANGE 1 HERE ★★★ ---
def normalize_coords(x, y, vb):
    """
    Normalizes and TRUNCATES coordinates to the project's format.
    """
    min_x, min_y, vb_width, vb_height = vb
    if vb_width == 0 or vb_height == 0:
        return (TARGET_MIN, TARGET_MIN)
    
    norm_x = ((x - min_x) / vb_width) * TARGET_RANGE + TARGET_MIN
    norm_y = ((y - min_y) / vb_height) * TARGET_RANGE + TARGET_MIN
    
    # Truncate to integer values (e.g., 127.8 -> 127.0)
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

# --- SVG Parsing Function (Unchanged) ---

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
        except (IndexError, TypeError): pass
    print("Warning: Could not parse viewBox. Assuming '0 0 255 255'.", file=sys.stderr)
    return [0.0, 0.0, 255.0, 255.0]

def convert_svg_to_sequence(svg_file_path, view_label=0):
    command_sequence = []
    try:
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        vb = get_viewbox(root)
        if vb[2] == 0 or vb[3] == 0:
            print(f"Error: Invalid viewBox in {svg_file_path}.", file=sys.stderr)
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
                if not d_string: continue
                parsed_segments = parse_path(d_string)
                for segment in parsed_segments:
                    if isinstance(segment, Line):
                        command_sequence.append(format_line_vec(segment, view_label, vb))
                    elif isinstance(segment, CubicBezier):
                        command_sequence.append(format_bezier_vec(segment, view_label, vb))
        
        command_sequence.append(get_eos_vec(view_label))
        return np.array(command_sequence, dtype=np.float32)

    except ET.ParseError as e:
        print(f"Error parsing XML in {svg_file_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An error occurred in {svg_file_path}: {e}", file=sys.stderr)
        return None

# --- ★★★ CHANGE 2 HERE ★★★ ---
def pad_sequence(sequence, view_label, target_length=N_COMMANDS):
    """
    Pads or truncates a sequence to the target_length.
    Padding rows are now EOS commands.
    """
    current_length = sequence.shape[0]
    
    if current_length > target_length:
        # Truncate
        padded_sequence = sequence[:target_length]
        # Ensure the very last command is EOS
        padded_sequence[-1] = get_eos_vec(view_label)
    
    elif current_length < target_length:
        # Pad
        num_padding = target_length - current_length
        
        # Create a padding vector that is the EOS command for this view
        pad_vector = get_eos_vec(view_label)
        
        # Create the padding array by tiling the EOS vector
        padding = np.tile(pad_vector, (num_padding, 1))
        
        # Stack the sequence and the padding
        padded_sequence = np.vstack((sequence, padding))
    
    else:
        # Length is already correct
        padded_sequence = sequence
        
    return padded_sequence

def process_drawing_folder(folder_path, output_npy_file):
    """
    Processes the 4 standard SVG views from a folder,
    pads them, and stacks them into a single 400x10 array.
    """
    
    views_to_process = [
        ("front", 0),
        ("top", 1),
        ("right", 2),
        ("iso", 3)
    ]
    
    base_name = os.path.basename(folder_path)
    all_view_sequences = []
    
    print(f"Processing folder: {folder_path}")
    
    for suffix, label in views_to_process:
        svg_file_name = f"{suffix}.svg"
        svg_file_path = os.path.join(folder_path, svg_file_name)
        
        if not os.path.exists(svg_file_path):
            print(f"  [!] Warning: File not found: {svg_file_name}")
            # Create an empty sequence so pad_sequence fills it with EOS
            raw_sequence = np.empty((0, N_PARAMS), dtype=np.float32)
        else:
            raw_sequence = convert_svg_to_sequence(svg_file_path, view_label=label)
            if raw_sequence is None:
                print(f"  [!] Error processing {svg_file_name}. Skipping.")
                raw_sequence = np.empty((0, N_PARAMS), dtype=np.float32)

        # Pad the sequence (e.g., 6x10 -> 100x10)
        padded_sequence = pad_sequence(raw_sequence, view_label=label)
        print(f"  [+] Processed {svg_file_name} (Label {label}) -> {raw_sequence.shape} padded to {padded_sequence.shape}")
        all_view_sequences.append(padded_sequence)
        
    try:
        final_stack = np.vstack(all_view_sequences)
        np.save(output_npy_file, final_stack)
        
        print(f"\nSuccess! Stacked array saved to: {output_npy_file}")
        print(f"Final array shape: {final_stack.shape}")
        
    except ValueError as e:
        print(f"\nError: Could not stack sequences. {e}")

# --- Main block (pointing to your real folders) ---
if __name__ == "__main__":
    
    # 1. DEFINE YOUR REAL PATHS
    real_input_folder = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\svg_from_step\00000070"
    
    # --- Construct the output path ---
    try:
        base_path = real_input_folder.split(r"\svg_from_step")[0]
        sub_path = real_input_folder.split(r"\svg_from_step")[1]
    except IndexError:
        print(f"Error: The input path '{real_input_folder}'")
        print("Does not seem to contain the '\\svg_from_step' directory.")
        print("Please check your paths.")
        sys.exit(1)
        
    real_output_file = os.path.join(base_path, "npy_from_svg_from_step" + sub_path + ".npy")
    
    # 2. MAKE SURE THE OUTPUT DIRECTORY EXISTS
    output_dir = os.path.dirname(real_output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input folder:  {real_input_folder}")
    print(f"Output file: {real_output_file}")

    # 3. RUN THE FUNCTION
    process_drawing_folder(real_input_folder, real_output_file)

    # 4. (Optional) Load and inspect the result
    if os.path.exists(real_output_file):
        print(f"\n--- Loading '{real_output_file}' for inspection ---")
        loaded_data = np.load(real_output_file)
        # Set print options to show scientific notation
        np.set_printoptions(precision=3, suppress=False, linewidth=150)
        
        print("Final Shape (should be 400, 10):", loaded_data.shape)
        
        print("\n--- Front (Label 0) - Start [Row 0] ---")
        print(loaded_data[0:3])
        
        print("\n--- Front (Label 0) - End (Padding) [Row 97] ---")
        print(loaded_data[97:100]) # Show padding
        
        print("\n--- Iso (Label 3) - Start [Row 300] ---")
        print(loaded_data[300:303])
        
        print("\n--- Iso (Label 3) - End (Padding) [Row 397] ---")
        print(loaded_data[397:400]) # Show padding