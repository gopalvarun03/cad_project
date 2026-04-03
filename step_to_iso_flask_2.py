# import os
# import h5py
# import numpy as np
# from flask import Flask, render_template_string, send_from_directory

# # ================= PATHS =================
# INPUT_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epochs_200\evaluation_results1\test"
# # INPUT_DIR=r'C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_results'
# OUTPUT_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epochs_200\evaluation_results1\test"
# # OUTPUT_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\new_pngs_new_views"
# SVG_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\new_svg_raw"
# # SVG_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\svg_vec_vaish"
# CAD_VEC_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\cad_vec"
# H5_DIR = INPUT_DIR  # Same as INPUT_DIR for predicted h5 files

# # ================= METRIC CONSTANTS =================
# CAD_EOS_IDX = 2
# GENERATED_DATASET_NAME = "out_vec"
# TRUTH_DATASET_NAME = "vec"
# TOL = 3          # parameter tolerance
# PAD_PARAM = -1   # padded argument marker

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ================= FLASK APP =================
# app = Flask(__name__)

# # ================= METRIC FUNCTIONS =================
# def load_pair(gen_path, truth_path):
#     """Load generated and ground truth h5 files"""
#     try:
#         with h5py.File(truth_path, "r") as f:
#             truth = f[TRUTH_DATASET_NAME][:]
#         with h5py.File(gen_path, "r") as f:
#             gen = f[GENERATED_DATASET_NAME][:]
#     except:
#         return None

#     T = min(len(gen), len(truth))
#     gen = gen[:T]
#     truth = truth[:T]

#     C = min(gen.shape[1], truth.shape[1])
#     gen = gen[:, :C]
#     truth = truth[:, :C]

#     return gen, truth

# def compute_ACCcmd(pred_cmd, gt_cmd):
#     """Computes command type accuracy"""
#     valid = (gt_cmd != CAD_EOS_IDX)
#     Nc = valid.sum()
#     if Nc == 0:
#         return 0.0
#     correct = (pred_cmd == gt_cmd) & valid
#     return 100.0 * correct.sum() / Nc

# def compute_ACCparam(pred_args, gt_args, pred_cmd, gt_cmd, eta=3):
#     """Computes parameter accuracy"""
#     valid_cmd = (gt_cmd != CAD_EOS_IDX)
#     correct_cmd = (pred_cmd == gt_cmd) & valid_cmd

#     valid_param = (gt_args != PAD_PARAM)
#     mask = valid_param & correct_cmd[:, None]
#     K = mask.sum()

#     if K == 0:
#         return 0.0

#     abs_diff = np.abs(gt_args - pred_args)
#     correct_param = (abs_diff <= eta)

#     return 100.0 * (correct_param & mask).sum() / K

# def compute_metrics_for_file(gen_h5_path, truth_h5_path):
#     """Compute ACCcmd and ACCparam for a single file pair"""
#     pair = load_pair(gen_h5_path, truth_h5_path)
#     if pair is None:
#         return None, None

#     gen, truth = pair
#     pred_cmd = gen[:, 0]
#     gt_cmd = truth[:, 0]
#     pred_args = gen[:, 1:]
#     gt_args = truth[:, 1:]

#     ACCcmd = compute_ACCcmd(pred_cmd, gt_cmd)
#     ACCparam = compute_ACCparam(pred_args, gt_args, pred_cmd, gt_cmd, eta=TOL)

#     return ACCcmd, ACCparam

# def find_ground_truth_h5(step_name):
#     """Find ground truth h5 file for a given step name"""
#     # step_name looks like "00000659_vec.step"
#     base_id = step_name.replace("_vec.step", "").replace(".step", "")
#     try:
#         n = int(base_id)
#     except Exception:
#         return None
#     bucket = f"{n // 10000:04d}"
#     h5_path = os.path.join(CAD_VEC_ROOT, bucket, f"{base_id}.h5")
#     return h5_path if os.path.exists(h5_path) else None

# def read_h5_content(h5_path, dataset_name):
#     """Read h5 file content and format as CAD commands"""
#     try:
#         with h5py.File(h5_path, "r") as f:
#             data = f[dataset_name][:]
        
#         # Format as CAD commands (indices from macro.py)
#         # CAD_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
#         cmd_names = {0: 'Line', 1: 'Arc', 2: 'Circle', 3: 'EOS', 4: 'SOL', 5: 'Ext'}
        
#         # Parameters order: [x, y, α, f, r, θ, φ, γ, px, py, pz, s, e1, e2, b, u]
#         param_labels = {
#             0: 'Line',   # uses: x, y
#             1: 'Arc',    # uses: x, y, α, f
#             2: 'Circle', # uses: x, y, r
#             3: 'EOS',    # no params
#             4: 'SOL',    # no params
#             5: 'Ext'     # uses: θ, φ, γ, px, py, pz, s, e1, e2, b, u
#         }
#         # ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
        
#         formatted = []
#         for i, row in enumerate(data):
#             cmd_type = int(row[0])
#             params = row[1:].astype(int)
            
#             cmd_name = cmd_names.get(cmd_type, f'Unknown({cmd_type})')
            
#             # Format parameters based on command type
#             param_strs = []
#             if cmd_type == 0:  # Line: x, y
#                 if params[0] != -1: param_strs.append(f"x:{params[0]}")
#                 if params[1] != -1: param_strs.append(f"y:{params[1]}")
#             elif cmd_type == 1:  # Arc: x, y, α, f
#                 if params[0] != -1: param_strs.append(f"x:{params[0]}")
#                 if params[1] != -1: param_strs.append(f"y:{params[1]}")
#                 if params[2] != -1: param_strs.append(f"α:{params[2]}")
#                 if params[3] != -1: param_strs.append(f"f:{params[3]}")
#             elif cmd_type == 2:  # Circle: x, y, r
#                 if params[0] != -1: param_strs.append(f"x:{params[0]}")
#                 if params[1] != -1: param_strs.append(f"y:{params[1]}")
#                 if params[4] != -1: param_strs.append(f"r:{params[4]}")
#             elif cmd_type == 5:  # Ext: θ, φ, γ, px, py, pz, s, e1, e2, b, u
#                 if params[5] != -1: param_strs.append(f"θ:{params[5]}")
#                 if params[6] != -1: param_strs.append(f"φ:{params[6]}")
#                 if params[7] != -1: param_strs.append(f"γ:{params[7]}")
#                 if params[8] != -1: param_strs.append(f"px:{params[8]}")
#                 if params[9] != -1: param_strs.append(f"py:{params[9]}")
#                 if params[10] != -1: param_strs.append(f"pz:{params[10]}")
#                 if params[11] != -1: param_strs.append(f"s:{params[11]}")
#                 if params[12] != -1: param_strs.append(f"e1:{params[12]}")
#                 if params[13] != -1: param_strs.append(f"e2:{params[13]}")
#                 if params[14] != -1: param_strs.append(f"b:{params[14]}")
#                 if params[15] != -1: param_strs.append(f"u:{params[15]}")
            
#             if param_strs:
#                 formatted.append(f"⟨{cmd_name}⟩_{i}: ({', '.join(param_strs)})")
#             else:
#                 formatted.append(f"⟨{cmd_name}⟩_{i}: ∅")
        
#         return '\n'.join(formatted)
#     except Exception as e:
#         return f"Error reading file: {e}"

# # ============== find original SVG ==============
# def find_original_svg(step_name):
#     # step_name looks like "00000659_vec.step"
#     base_id = step_name.replace("_vec.step", "").replace(".step", "").split(".")[0]
#     try:
#         n = int(base_id)
#     except Exception as e:
#         print("⚠️ cannot parse step name:", step_name)
#         return None
#     bucket = f"{n // 10000:04d}"
#     svg_path = os.path.join(SVG_ROOT, bucket, base_id, f"{base_id}_FrontTopRight.svg")
#     return svg_path if os.path.exists(svg_path) else None

# # ============== PROCESS ALL STEPS ==============
# def process_all_steps():
#     """Process all STEP files and generate data (assumes PNGs are pre-generated)"""
#     items = []
#     for f in sorted(os.listdir(INPUT_DIR)):
#         try:
#             if not f.lower().endswith((".step", ".stp")):
#                 print(f"[SKIP] Not a STEP file: {f}")
#                 continue
#             png_name = os.path.splitext(f)[0] + ".png"
#             png_path = os.path.join(OUTPUT_DIR, png_name)

#             # Check if PNG exists (should be pre-generated)
#             if not os.path.exists(png_path):
#                 print(f"[SKIP] PNG not found for: {f}")
#                 continue

#             svg = find_original_svg(f)
#             if not svg:
#                 print(f"[INFO] SVG not found for: {f}")

#             # Compute metrics
#             base_id = f.replace("_vec.step", "").replace(".step", "").split(".")[0]
#             gen_h5_path = os.path.join(H5_DIR, base_id + ".h5")
#             truth_h5_path = find_ground_truth_h5(f)

#             acc_cmd, acc_param = None, None
#             gen_h5_content, truth_h5_content = None, None

#             if not os.path.exists(gen_h5_path):
#                 print(f"[INFO] Generated h5 missing: {gen_h5_path}")
#             if not truth_h5_path or not os.path.exists(truth_h5_path):
#                 print(f"[INFO] Ground truth h5 missing: {truth_h5_path}")

#             if truth_h5_path and os.path.exists(gen_h5_path) and os.path.exists(truth_h5_path):
#                 try:
#                     acc_cmd, acc_param = compute_metrics_for_file(gen_h5_path, truth_h5_path)
#                     if acc_cmd is None or acc_param is None:
#                         print(f"[WARN] Metrics not computed for: {f}")
#                 except Exception as e:
#                     print(f"[ERROR] Exception in metric computation for {f}: {e}")
#                 try:
#                     gen_h5_content = read_h5_content(gen_h5_path, GENERATED_DATASET_NAME)
#                     if not gen_h5_content or gen_h5_content.startswith("Error"):
#                         print(f"[WARN] Error reading generated h5 content: {gen_h5_path} | {gen_h5_content}")
#                 except Exception as e:
#                     print(f"[ERROR] Exception reading generated h5 content for {gen_h5_path}: {e}")
#                 try:
#                     truth_h5_content = read_h5_content(truth_h5_path, TRUTH_DATASET_NAME)
#                     if not truth_h5_content or truth_h5_content.startswith("Error"):
#                         print(f"[WARN] Error reading ground truth h5 content: {truth_h5_path} | {truth_h5_content}")
#                 except Exception as e:
#                     print(f"[ERROR] Exception reading ground truth h5 content for {truth_h5_path}: {e}")

#             items.append({
#                 "step": f,
#                 "png": png_name,
#                 "svg": svg,
#                 "acc_cmd": acc_cmd,
#                 "acc_param": acc_param,
#                 "gen_h5": gen_h5_content,
#                 "truth_h5": truth_h5_content
#             })
#             print(f"✅ processed: {png_name}",
#                   f"| ACCcmd: {acc_cmd:.2f}%" if acc_cmd is not None else "| ACCcmd: N/A",
#                   f"ACCparam: {acc_param:.2f}%" if acc_param is not None else "ACCparam: N/A")
#         except Exception as e:
#             print(f"[ERROR] Exception processing file {f}: {e}")

#     print(f"[SUMMARY] Total processed: {len(items)}")
#     return items

# # ============== FLASK ROUTES ==============
# @app.route('/')
# def index():
#     """Main page with gallery"""
#     items = process_all_steps()
    
#     html_template = """<!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8">
# <title>STEP Isometric Comparison - Live View</title>
# <style>
# body {
#     font-family: Arial, sans-serif;
#     background: #f2f2f2;
#     margin: 0;
#     padding: 20px 0;
# }
# .container {
#     max-width: 1200px;
#     margin: auto;
# }
# h1 {
#     text-align: center;
#     margin: 8px 0 20px 0;
# }
# .info {
#     text-align: center;
#     color: #666;
#     margin-bottom: 20px;
#     font-size: 14px;
# }
# .grid {
#     display: grid;
#     grid-template-columns: repeat(4, 1fr);
#     gap: 14px;
#     align-items: start;
# }
# .stepname {
#     grid-column: span 2;
#     font-weight: bold;
#     padding: 6px 0;
# }
# .card {
#     background: white;
#     border-radius: 6px;
#     padding: 8px;
#     box-shadow: 0 2px 5px rgba(0,0,0,0.12);
#     text-align: center;
#     min-height: 260px;
#     max-height: 500px;
#     display: flex;
#     flex-direction: column;
#     justify-content: flex-start;
#     align-items: center;
#     overflow: hidden;
# }
# .card .label {
#     font-size: 12px;
#     color: #666;
#     margin-bottom: 6px;
# }
# .metrics {
#     font-size: 11px;
#     color: #0066cc;
#     margin: 6px 0;
#     font-weight: 600;
# }
# .card img {
#     width: 100%;
#     height: auto;
#     object-fit: contain;
#     border-radius: 4px;
#     background: #fff;
# }
# .placeholder {
#     color: #999;
#     font-size: 13px;
#     margin-top: 40px;
# }
# .small {
#     font-size: 12px;
#     color: #444;
#     margin-top: 6px;
#     word-break: break-all;
# }
# .h5-info {
#     font-size: 9px;
#     color: #333;
#     margin-top: 8px;
#     padding: 6px;
#     background: #f8f8f8;
#     border-radius: 3px;
#     font-family: 'Courier New', monospace;
#     text-align: left;
#     max-height: 150px;
#     overflow-y: auto;
#     white-space: pre-wrap;
#     word-break: break-all;
# }
# </style>
# </head>
# <body>
# <div class="container">
# <h1>🔴 LIVE: STEP Isometric Comparison</h1>
# <div class="info">Flask Server Running | Total Files: {{ total_count }}</div>
# <div class="grid">
# {% set n = items|length %}
# {% for i in range(0, n, 2) %}
#     {% set a = items[i] %}
#     {% set b = items[i+1] if (i+1) < n else None %}
    
#     {# Header row #}
#     <div class="stepname">{{ a.step }}
#     {% if a.acc_cmd is not none %}
#         <span style="color:#0066cc;">[ACCcmd: {{ "%.2f"|format(a.acc_cmd) }}% | ACCparam: {{ "%.2f"|format(a.acc_param) }}%]</span>
#     {% endif %}
#     </div>
    
#     {% if b %}
#     <div class="stepname">{{ b.step }}
#     {% if b.acc_cmd is not none %}
#         <span style="color:#0066cc;">[ACCcmd: {{ "%.2f"|format(b.acc_cmd) }}% | ACCparam: {{ "%.2f"|format(b.acc_param) }}%]</span>
#     {% endif %}
#     </div>
#     {% else %}
#     <div class="stepname"></div>
#     {% endif %}
    
#     {# Generated A #}
#     <div class="card">
#         <div class="label">Generated</div>
#         <img src="/png/{{ a.png }}" alt="generated">
#         <div class="small">{{ a.png }}</div>
#         {% if a.gen_h5 %}
#         <div class="h5-info">{{ a.gen_h5 }}</div>
#         {% endif %}
#     </div>
    
#     {# Original A #}
#     <div class="card">
#         <div class="label">Original</div>
#         {% if a.svg %}
#             <img src="/svg/{{ a.svg }}" alt="original">
#             <div class="small">Original SVG</div>
#         {% else %}
#             <div class="placeholder">Not available</div>
#         {% endif %}
#         {% if a.truth_h5 %}
#         <div class="h5-info">{{ a.truth_h5 }}</div>
#         {% endif %}
#     </div>
    
#     {# Generated B #}
#     {% if b %}
#     <div class="card">
#         <div class="label">Generated</div>
#         <img src="/png/{{ b.png }}" alt="generated">
#         <div class="small">{{ b.png }}</div>
#         {% if b.gen_h5 %}
#         <div class="h5-info">{{ b.gen_h5 }}</div>
#         {% endif %}
#     </div>
#     {% else %}
#     <div class="card"><div class="placeholder">--</div></div>
#     {% endif %}
    
#     {# Original B #}
#     {% if b %}
#     <div class="card">
#         <div class="label">Original</div>
#         {% if b.svg %}
#             <img src="/svg/{{ b.svg }}" alt="original">
#             <div class="small">Original SVG</div>
#         {% else %}
#             <div class="placeholder">Not available</div>
#         {% endif %}
#         {% if b.truth_h5 %}
#         <div class="h5-info">{{ b.truth_h5 }}</div>
#         {% endif %}
#     </div>
#     {% else %}
#     <div class="card"><div class="placeholder">--</div></div>
#     {% endif %}
    
# {% endfor %}
# </div>
# </div>
# </body>
# </html>
# """
    
#     return render_template_string(html_template, items=items, total_count=len(items))

# @app.route('/png/<filename>')
# def serve_png(filename):
#     """Serve PNG files from OUTPUT_DIR"""
#     return send_from_directory(OUTPUT_DIR, filename)

# @app.route('/svg/<path:path>')
# def serve_svg(path):
#     """Serve SVG files from their original locations"""
#     directory = os.path.dirname(path)
#     filename = os.path.basename(path)
#     return send_from_directory(directory, filename)

# if __name__ == '__main__':
#     print("=" * 60)
#     print("🚀 Starting Flask server...")
#     print("=" * 60)
#     print(f"📂 Input DIR: {INPUT_DIR}")
#     print(f"📂 Output DIR: {OUTPUT_DIR}")
#     print(f"📂 SVG ROOT: {SVG_ROOT}")
#     print(f"📂 CAD VEC ROOT: {CAD_VEC_ROOT}")
#     print("=" * 60)
#     print("🌐 Server will start at: http://127.0.0.1:3000")
#     print("=" * 60)
    
#     app.run(debug=True, host='0.0.0.0', port=3000)

import os
import h5py
import numpy as np
from flask import Flask, render_template_string, send_from_directory
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

# ================= PATHS =================
INPUT_DIR = r"C:\Users\thiri\OneDrive\Desktop\2DtoCAD\Drawing2CAD\proj_log\your_exp_name\evaluation_results_rotated2\test"
# INPUT_DIR=r'C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_results'
OUTPUT_DIR = r"C:\Users\thiri\OneDrive\Desktop\2DtoCAD\Drawing2CAD\proj_log\your_exp_name\evaluation_results_rotated2\test"
# OUTPUT_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\new_pngs_new_views"
SVG_ROOT = r"C:\Users\thiri\OneDrive\Desktop\2DtoCAD\DeepCAD\data\CAD-VGDrawing\svg_raw"
# SVG_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\svg_vec_vaish"
CAD_VEC_ROOT = r"C:\Users\thiri\OneDrive\Desktop\2DtoCAD\DeepCAD\data\CAD-VGDrawing\cad_vec"
H5_DIR = INPUT_DIR  # Same as INPUT_DIR for predicted h5 files

# ================= METRIC CONSTANTS =================
CAD_EOS_IDX = 2
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
TOL = 3          # parameter tolerance
PAD_PARAM = -1   # padded argument marker

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= FLASK APP =================
app = Flask(__name__)

# ================= METRIC FUNCTIONS =================
def load_pair(gen_path, truth_path):
    """Load generated and ground truth h5 files"""
    try:
        with h5py.File(truth_path, "r") as f:
            truth = f[TRUTH_DATASET_NAME][:]
        with h5py.File(gen_path, "r") as f:
            gen = f[GENERATED_DATASET_NAME][:]
    except:
        return None

    T = min(len(gen), len(truth))
    gen = gen[:T]
    truth = truth[:T]

    C = min(gen.shape[1], truth.shape[1])
    gen = gen[:, :C]
    truth = truth[:, :C]

    return gen, truth

def compute_ACCcmd(pred_cmd, gt_cmd):
    """Computes command type accuracy"""
    valid = (gt_cmd != CAD_EOS_IDX)
    Nc = valid.sum()
    if Nc == 0:
        return 0.0
    correct = (pred_cmd == gt_cmd) & valid
    return 100.0 * correct.sum() / Nc

def compute_ACCparam(pred_args, gt_args, pred_cmd, gt_cmd, eta=3):
    """Computes parameter accuracy"""
    valid_cmd = (gt_cmd != CAD_EOS_IDX)
    correct_cmd = (pred_cmd == gt_cmd) & valid_cmd

    valid_param = (gt_args != PAD_PARAM)
    mask = valid_param & correct_cmd[:, None]
    K = mask.sum()

    if K == 0:
        return 0.0

    abs_diff = np.abs(gt_args - pred_args)
    correct_param = (abs_diff <= eta)

    return 100.0 * (correct_param & mask).sum() / K

def compute_metrics_for_file(gen_h5_path, truth_h5_path):
    """Compute ACCcmd and ACCparam for a single file pair"""
    pair = load_pair(gen_h5_path, truth_h5_path)
    if pair is None:
        return None, None

    gen, truth = pair
    pred_cmd = gen[:, 0]
    gt_cmd = truth[:, 0]
    pred_args = gen[:, 1:]
    gt_args = truth[:, 1:]

    ACCcmd = compute_ACCcmd(pred_cmd, gt_cmd)
    ACCparam = compute_ACCparam(pred_args, gt_args, pred_cmd, gt_cmd, eta=TOL)

    return ACCcmd, ACCparam

def find_ground_truth_h5(step_name):
    """Find ground truth h5 file for a given step name"""
    # step_name looks like "00000659_vec.step"
    base_id = step_name.replace("_vec.step", "").replace(".step", "")
    try:
        n = int(base_id)
    except Exception:
        return None
    bucket = f"{n // 10000:04d}"
    h5_path = os.path.join(CAD_VEC_ROOT, bucket, f"{base_id}.h5")
    return h5_path if os.path.exists(h5_path) else None

def read_h5_content(h5_path, dataset_name):
    """Read h5 file content and format as CAD commands"""
    try:
        with h5py.File(h5_path, "r") as f:
            data = f[dataset_name][:]
        
        # Format as CAD commands (indices from macro.py)
        # CAD_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
        cmd_names = {0: 'Line', 1: 'Arc', 2: 'Circle', 3: 'EOS', 4: 'SOL', 5: 'Ext'}
        
        # Parameters order: [x, y, α, f, r, θ, φ, γ, px, py, pz, s, e1, e2, b, u]
        param_labels = {
            0: 'Line',   # uses: x, y
            1: 'Arc',    # uses: x, y, α, f
            2: 'Circle', # uses: x, y, r
            3: 'EOS',    # no params
            4: 'SOL',    # no params
            5: 'Ext'     # uses: θ, φ, γ, px, py, pz, s, e1, e2, b, u
        }
        # ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
        
        formatted = []
        for i, row in enumerate(data):
            cmd_type = int(row[0])
            params = row[1:].astype(int)
            
            cmd_name = cmd_names.get(cmd_type, f'Unknown({cmd_type})')
            
            # Format parameters based on command type
            param_strs = []
            if cmd_type == 0:  # Line: x, y
                if params[0] != -1: param_strs.append(f"x:{params[0]}")
                if params[1] != -1: param_strs.append(f"y:{params[1]}")
            elif cmd_type == 1:  # Arc: x, y, α, f
                if params[0] != -1: param_strs.append(f"x:{params[0]}")
                if params[1] != -1: param_strs.append(f"y:{params[1]}")
                if params[2] != -1: param_strs.append(f"α:{params[2]}")
                if params[3] != -1: param_strs.append(f"f:{params[3]}")
            elif cmd_type == 2:  # Circle: x, y, r
                if params[0] != -1: param_strs.append(f"x:{params[0]}")
                if params[1] != -1: param_strs.append(f"y:{params[1]}")
                if params[4] != -1: param_strs.append(f"r:{params[4]}")
            elif cmd_type == 5:  # Ext: θ, φ, γ, px, py, pz, s, e1, e2, b, u
                if params[5] != -1: param_strs.append(f"θ:{params[5]}")
                if params[6] != -1: param_strs.append(f"φ:{params[6]}")
                if params[7] != -1: param_strs.append(f"γ:{params[7]}")
                if params[8] != -1: param_strs.append(f"px:{params[8]}")
                if params[9] != -1: param_strs.append(f"py:{params[9]}")
                if params[10] != -1: param_strs.append(f"pz:{params[10]}")
                if params[11] != -1: param_strs.append(f"s:{params[11]}")
                if params[12] != -1: param_strs.append(f"e1:{params[12]}")
                if params[13] != -1: param_strs.append(f"e2:{params[13]}")
                if params[14] != -1: param_strs.append(f"b:{params[14]}")
                if params[15] != -1: param_strs.append(f"u:{params[15]}")
            
            if param_strs:
                formatted.append(f"⟨{cmd_name}⟩_{i}: ({', '.join(param_strs)})")
            else:
                formatted.append(f"⟨{cmd_name}⟩_{i}: ∅")
        
        return '\n'.join(formatted)
    except Exception as e:
        return f"Error reading file: {e}"

# ============== INIT OCC VIEWER ONCE ==============
print("Initializing OCC display...")
display, _, _, _ = init_display()
display.View.SetImmediateUpdate(False)
params = display.View.RenderingParams()
params.NbMsaaSamples = 0
params.IsAntialiasingEnabled = False
display.View.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))
# Windows-safe way to set viewport size
try:
    display.View.SetWindowSize(1024, 768)
except Exception:
    pass

# ============== STEP -> PNG (robust) ==============
def step_to_iso_png(step_path, out_path):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print("❌ Read failed:", step_path)
        return False

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape is None or shape.IsNull():
        print("⚠️ Empty or invalid shape:", step_path)
        return False

    try:
        display.EraseAll()
        display.DisplayShape(shape, update=False)
        display.View_Iso()
        display.FitAll()
        display.Repaint()
        display.View.Dump(out_path)
        return True
    except Exception as e:
        print("⚠️ Render failed:", step_path, " -> ", e)
        return False

# ============== find original SVG ==============
def find_original_svg(step_name):
    # step_name looks like "00000659_vec.step"
    base_id = step_name.replace("_vec.step", "").replace(".step", "").split(".")[0]
    try:
        n = int(base_id)
    except Exception as e:
        print("⚠️ cannot parse step name:", step_name)
        return None
    bucket = f"{n // 10000:04d}"
    svg_path = os.path.join(SVG_ROOT, bucket, base_id, f"{base_id}_FrontTopRight.svg")
    return svg_path if os.path.exists(svg_path) else None

# ============== PROCESS ALL STEPS ==============
def process_all_steps():
    """Process all STEP files and generate data"""
    items = []
    for f in sorted(os.listdir(INPUT_DIR)):
        try:
            if not f.lower().endswith((".step", ".stp")):
                print(f"[SKIP] Not a STEP file: {f}")
                continue
            step_path = os.path.join(INPUT_DIR, f)
            png_name = os.path.splitext(f)[0] + ".png"
            png_path = os.path.join(OUTPUT_DIR, png_name)

            # Generate PNG if not exists
            if not os.path.exists(png_path):
                ok = step_to_iso_png(step_path, png_path)
                if not ok:
                    print(f"[SKIP] PNG not generated for: {f}")
                    continue
            elif not os.path.isfile(png_path):
                print(f"[WARN] PNG file missing after supposed generation: {png_path}")

            svg = find_original_svg(f)
            if not svg:
                print(f"[INFO] SVG not found for: {f}")

            items = []
            for f in sorted(os.listdir(INPUT_DIR)):
                try:
                    if not f.lower().endswith((".step", ".stp")):
                        print(f"[SKIP] Not a STEP file: {f}")
                        continue
                    step_path = os.path.join(INPUT_DIR, f)
                    png_name = os.path.splitext(f)[0] + ".png"
                    png_path = os.path.join(OUTPUT_DIR, png_name)

                    # Generate PNG if not exists
                    if not os.path.exists(png_path):
                        ok = step_to_iso_png(step_path, png_path)
                        if not ok:
                            print(f"[SKIP] PNG not generated for: {f}")
                            continue
                    elif not os.path.isfile(png_path):
                        print(f"[WARN] PNG file missing after supposed generation: {png_path}")

                    svg = find_original_svg(f)
                    if not svg:
                        print(f"[INFO] SVG not found for: {f}")

                    # Compute metrics
                    base_id = f.replace("_vec.step", "").replace(".step", "").split(".")[0]
                    gen_h5_path = os.path.join(H5_DIR, base_id + ".h5")
                    truth_h5_path = find_ground_truth_h5(f)

                    acc_cmd, acc_param = None, None
                    gen_h5_content, truth_h5_content = None, None

                    if not os.path.exists(gen_h5_path):
                        print(f"[INFO] Generated h5 missing: {gen_h5_path}")
                    if not truth_h5_path or not os.path.exists(truth_h5_path):
                        print(f"[INFO] Ground truth h5 missing: {truth_h5_path}")

                    if truth_h5_path and os.path.exists(gen_h5_path) and os.path.exists(truth_h5_path):
                        try:
                            acc_cmd, acc_param = compute_metrics_for_file(gen_h5_path, truth_h5_path)
                            if acc_cmd is None or acc_param is None:
                                print(f"[WARN] Metrics not computed for: {f}")
                        except Exception as e:
                            print(f"[ERROR] Exception in metric computation for {f}: {e}")
                        try:
                            gen_h5_content = read_h5_content(gen_h5_path, GENERATED_DATASET_NAME)
                            if not gen_h5_content or gen_h5_content.startswith("Error"):
                                print(f"[WARN] Error reading generated h5 content: {gen_h5_path} | {gen_h5_content}")
                        except Exception as e:
                            print(f"[ERROR] Exception reading generated h5 content for {gen_h5_path}: {e}")
                        try:
                            truth_h5_content = read_h5_content(truth_h5_path, TRUTH_DATASET_NAME)
                            if not truth_h5_content or truth_h5_content.startswith("Error"):
                                print(f"[WARN] Error reading ground truth h5 content: {truth_h5_path} | {truth_h5_content}")
                        except Exception as e:
                            print(f"[ERROR] Exception reading ground truth h5 content for {truth_h5_path}: {e}")

                    items.append({
                        "step": f,
                        "png": png_name,
                        "svg": svg,
                        "acc_cmd": acc_cmd,
                        "acc_param": acc_param,
                        "gen_h5": gen_h5_content,
                        "truth_h5": truth_h5_content
                    })
                    print(f"✅ processed: {png_name}",
                          f"| ACCcmd: {acc_cmd:.2f}%" if acc_cmd is not None else "| ACCcmd: N/A",
                          f"ACCparam: {acc_param:.2f}%" if acc_param is not None else "ACCparam: N/A")
                except Exception as e:
                    print(f"[ERROR] Exception processing file {f}: {e}")

            print(f"[SUMMARY] Total processed: {len(items)}")
            return items
            print(f"✅ processed: {png_name}",
                  f"| ACCcmd: {acc_cmd:.2f}%" if acc_cmd is not None else "| ACCcmd: N/A",
                  f"ACCparam: {acc_param:.2f}%" if acc_param is not None else "ACCparam: N/A")
        except Exception as e:
            print(f"[ERROR] Exception processing file {f}: {e}")

    print(f"[SUMMARY] Total processed: {len(items)}")
    return items

# ============== FLASK ROUTES ==============
@app.route('/')
def index():
    """Main page with gallery"""
    items = process_all_steps()
    
    html_template = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>STEP Isometric Comparison - Live View</title>
<style>
body {
    font-family: Arial, sans-serif;
    background: #f2f2f2;
    margin: 0;
    padding: 20px 0;
}
.container {
    max-width: 1200px;
    margin: auto;
}
h1 {
    text-align: center;
    margin: 8px 0 20px 0;
}
.info {
    text-align: center;
    color: #666;
    margin-bottom: 20px;
    font-size: 14px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    align-items: start;
}
.stepname {
    grid-column: span 2;
    font-weight: bold;
    padding: 6px 0;
}
.card {
    background: white;
    border-radius: 6px;
    padding: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.12);
    text-align: center;
    min-height: 260px;
    max-height: 500px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
    overflow: hidden;
}
.card .label {
    font-size: 12px;
    color: #666;
    margin-bottom: 6px;
}
.metrics {
    font-size: 11px;
    color: #0066cc;
    margin: 6px 0;
    font-weight: 600;
}
.card img {
    width: 100%;
    height: auto;
    object-fit: contain;
    border-radius: 4px;
    background: #fff;
}
.placeholder {
    color: #999;
    font-size: 13px;
    margin-top: 40px;
}
.small {
    font-size: 12px;
    color: #444;
    margin-top: 6px;
    word-break: break-all;
}
.h5-info {
    font-size: 9px;
    color: #333;
    margin-top: 8px;
    padding: 6px;
    background: #f8f8f8;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    text-align: left;
    max-height: 150px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
</style>
</head>
<body>
<div class="container">
<h1>🔴 LIVE: STEP Isometric Comparison</h1>
<div class="info">Flask Server Running | Total Files: {{ total_count }}</div>
<div class="grid">
{% set n = items|length %}
{% for i in range(0, n, 2) %}
    {% set a = items[i] %}
    {% set b = items[i+1] if (i+1) < n else None %}
    
    {# Header row #}
    <div class="stepname">{{ a.step }}
    {% if a.acc_cmd is not none %}
        <span style="color:#0066cc;">[ACCcmd: {{ "%.2f"|format(a.acc_cmd) }}% | ACCparam: {{ "%.2f"|format(a.acc_param) }}%]</span>
    {% endif %}
    </div>
    
    {% if b %}
    <div class="stepname">{{ b.step }}
    {% if b.acc_cmd is not none %}
        <span style="color:#0066cc;">[ACCcmd: {{ "%.2f"|format(b.acc_cmd) }}% | ACCparam: {{ "%.2f"|format(b.acc_param) }}%]</span>
    {% endif %}
    </div>
    {% else %}
    <div class="stepname"></div>
    {% endif %}
    
    {# Generated A #}
    <div class="card">
        <div class="label">Generated</div>
        <img src="/png/{{ a.png }}" alt="generated">
        <div class="small">{{ a.png }}</div>
        {% if a.gen_h5 %}
        <div class="h5-info">{{ a.gen_h5 }}</div>
        {% endif %}
    </div>
    
    {# Original A #}
    <div class="card">
        <div class="label">Original</div>
        {% if a.svg %}
            <img src="/svg/{{ a.svg }}" alt="original">
            <div class="small">Original SVG</div>
        {% else %}
            <div class="placeholder">Not available</div>
        {% endif %}
        {% if a.truth_h5 %}
        <div class="h5-info">{{ a.truth_h5 }}</div>
        {% endif %}
    </div>
    
    {# Generated B #}
    {% if b %}
    <div class="card">
        <div class="label">Generated</div>
        <img src="/png/{{ b.png }}" alt="generated">
        <div class="small">{{ b.png }}</div>
        {% if b.gen_h5 %}
        <div class="h5-info">{{ b.gen_h5 }}</div>
        {% endif %}
    </div>
    {% else %}
    <div class="card"><div class="placeholder">--</div></div>
    {% endif %}
    
    {# Original B #}
    {% if b %}
    <div class="card">
        <div class="label">Original</div>
        {% if b.svg %}
            <img src="/svg/{{ b.svg }}" alt="original">
            <div class="small">Original SVG</div>
        {% else %}
            <div class="placeholder">Not available</div>
        {% endif %}
        {% if b.truth_h5 %}
        <div class="h5-info">{{ b.truth_h5 }}</div>
        {% endif %}
    </div>
    {% else %}
    <div class="card"><div class="placeholder">--</div></div>
    {% endif %}
    
{% endfor %}
</div>
</div>
</body>
</html>
"""
    
    return render_template_string(html_template, items=items, total_count=len(items))

@app.route('/png/<filename>')
def serve_png(filename):
    """Serve PNG files from OUTPUT_DIR"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/svg/<path:path>')
def serve_svg(path):
    """Serve SVG files from their original locations"""
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Flask server...")
    print("=" * 60)
    print(f"📂 Input DIR: {INPUT_DIR}")
    print(f"📂 Output DIR: {OUTPUT_DIR}")
    print(f"📂 SVG ROOT: {SVG_ROOT}")
    print(f"📂 CAD VEC ROOT: {CAD_VEC_ROOT}")
    print("=" * 60)
    print("🌐 Server will start at: http://127.0.0.1:3000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=3000)