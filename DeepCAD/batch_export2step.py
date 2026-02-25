"""
Batch convert H5 files listed in valid_h5_files.txt to STEP files.
"""
import os
import sys
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cadlib.visualize import vec2CADsolid
from utils.file_utils import ensure_dir


def convert_h5_to_step(h5_path, output_dir, subfolder, use_filter=False):
    """Convert a single H5 file to STEP format, preserving folder structure."""
    try:
        with h5py.File(h5_path, 'r') as fp:
            out_vec = fp["vec"][:].astype(np.float64)
            out_shape = vec2CADsolid(out_vec)
        
        if use_filter:
            analyzer = BRepCheck_Analyzer(out_shape)
            if not analyzer.IsValid():
                print(f"Invalid shape detected: {h5_path}")
                return False
        
        # Create subfolder in output directory (e.g., cad_step/0000/)
        subfolder_path = os.path.join(output_dir, subfolder)
        ensure_dir(subfolder_path)
        
        # Get the filename without extension
        name = os.path.basename(h5_path).replace('.h5', '')
        save_path = os.path.join(subfolder_path, name + ".step")
        write_step_file(out_shape, save_path)
        print(f"Converted: {h5_path} -> {save_path}")
        return True
    
    except Exception as e:
        print(f"Failed to convert {h5_path}: {e}")
        return False


def main():
    # Paths
    valid_files_path = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\valid_h5_files.txt"
    output_dir = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\cad_step"
    base_dir = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD"
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Read the list of valid H5 files
    with open(valid_files_path, 'r') as f:
        h5_files = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(h5_files)} H5 files to convert")
    
    success_count = 0
    fail_count = 0
    
    for i, relative_path in enumerate(h5_files):
        # Convert relative path to absolute path
        # The paths in valid_h5_files.txt are relative like: ../data2/cad_vec\0000\00000007.h5
        # We need to resolve them from the DeepCAD directory
        
        # Normalize the path (handle both / and \)
        relative_path = relative_path.replace('\\', os.sep).replace('/', os.sep)
        
        # Remove leading ../ or ..\
        if relative_path.startswith('..'):
            relative_path = relative_path[3:]  # Remove '../' or '..\\'
        
        h5_path = os.path.join(base_dir, relative_path)
        
        # Extract subfolder name (e.g., "0000" from "data2/cad_vec/0000/00000007.h5")
        path_parts = relative_path.replace('/', os.sep).replace('\\', os.sep).split(os.sep)
        # Find the subfolder (the folder containing the .h5 file, e.g., "0000")
        subfolder = path_parts[-2] if len(path_parts) >= 2 else ""
        
        # Check if file exists
        if not os.path.exists(h5_path):
            print(f"File not found: {h5_path}")
            fail_count += 1
            continue
        
        print(f"[{i+1}/{len(h5_files)}] Processing: {h5_path}")
        
        if convert_h5_to_step(h5_path, output_dir, subfolder, use_filter=False):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nConversion complete!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
