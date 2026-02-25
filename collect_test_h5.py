import json
import shutil
import os
from pathlib import Path

# Paths
json_path = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\train_val_test_split.json"
h5_base_path = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\cad_vec"
output_folder = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\test_h5_files"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Get test set entries
test_entries = data.get('test', [])
print(f"Found {len(test_entries)} test entries")

# Copy files
copied = 0
not_found = 0
not_found_list = []

for entry in test_entries:
    # entry is like "0000/00000134"
    h5_path = os.path.join(h5_base_path, entry + ".h5")
    
    if os.path.exists(h5_path):
        # Keep the filename only (e.g., 00000134.h5)
        filename = os.path.basename(h5_path)
        dest_path = os.path.join(output_folder, filename)
        
        shutil.copy2(h5_path, dest_path)
        copied += 1
    else:
        not_found += 1
        not_found_list.append(h5_path)

print(f"\nCopied: {copied} files")
print(f"Not found: {not_found} files")

if not_found_list and not_found <= 10:
    print("\nFiles not found:")
    for f in not_found_list:
        print(f"  {f}")
elif not_found_list:
    print(f"\nFirst 10 files not found:")
    for f in not_found_list[:10]:
        print(f"  {f}")

print(f"\nOutput folder: {output_folder}")
