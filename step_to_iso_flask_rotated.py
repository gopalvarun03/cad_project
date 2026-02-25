import os
import h5py
import numpy as np
from flask import Flask, render_template_string, send_from_directory
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

# ================= PATHS =================
# Three variants of predictions
INPUT_DIR_ORIGINAL = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_results_original"
INPUT_DIR_ROTATED = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_results_rotated"
INPUT_DIR_ROTATED2 = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_results_rotated2"

# Corresponding output directories for PNGs
OUTPUT_DIR_ORIGINAL = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_pngs_original"
OUTPUT_DIR_ROTATED = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_pngs_rotated"
OUTPUT_DIR_ROTATED2 = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_pngs_rotated2"

SVG_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\svg_raw"
CAD_VEC_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\cad_vec"

# ================= METRIC CONSTANTS =================
CAD_EOS_IDX = 2
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
TOL = 3          # parameter tolerance
PAD_PARAM = -1   # padded argument marker

os.makedirs(OUTPUT_DIR_ORIGINAL, exist_ok=True)
os.makedirs(OUTPUT_DIR_ROTATED, exist_ok=True)
os.makedirs(OUTPUT_DIR_ROTATED2, exist_ok=True)

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
def process_variant(input_dir, output_dir, variant_name):
    """Process files from one variant directory"""
    variant_data = {}
    for f in sorted(os.listdir(input_dir)):
        try:
            if not f.lower().endswith((".step", ".stp", ".h5")):
                continue
            
            base_id = f.replace("_vec.step", "").replace(".step", "").replace(".h5", "").split(".")[0]
            png_name = base_id + "_vec.png"
            png_path = os.path.join(output_dir, png_name)
            
            # Generate PNG if STEP file and PNG doesn't exist
            if f.lower().endswith((".step", ".stp")):
                step_path = os.path.join(input_dir, f)
                if not os.path.exists(png_path):
                    ok = step_to_iso_png(step_path, png_path)
                    if not ok:
                        print(f"[{variant_name}] PNG not generated for: {f}")
            
            # Compute metrics
            gen_h5_path = os.path.join(input_dir, base_id + ".h5")
            truth_h5_path = find_ground_truth_h5(base_id)
            
            acc_cmd, acc_param = None, None
            gen_h5_content, truth_h5_content = None, None
            
            if truth_h5_path and os.path.exists(gen_h5_path) and os.path.exists(truth_h5_path):
                try:
                    acc_cmd, acc_param = compute_metrics_for_file(gen_h5_path, truth_h5_path)
                except Exception as e:
                    print(f"[{variant_name}] Metric error for {base_id}: {e}")
                try:
                    gen_h5_content = read_h5_content(gen_h5_path, GENERATED_DATASET_NAME)
                except Exception as e:
                    print(f"[{variant_name}] H5 read error for {base_id}: {e}")
                try:
                    truth_h5_content = read_h5_content(truth_h5_path, TRUTH_DATASET_NAME)
                except Exception:
                    pass
            
            variant_data[base_id] = {
                "png": png_name,
                "acc_cmd": acc_cmd,
                "acc_param": acc_param,
                "gen_h5": gen_h5_content,
                "truth_h5": truth_h5_content
            }
            
        except Exception as e:
            print(f"[{variant_name}] Error processing {f}: {e}")
    
    return variant_data

def process_all_steps():
    """Process all STEP files from all 3 variants and generate comparison data"""
    print("[INFO] Processing variant: ORIGINAL")
    original_data = process_variant(INPUT_DIR_ORIGINAL, OUTPUT_DIR_ORIGINAL, "ORIGINAL")
    
    print("[INFO] Processing variant: ROTATED")
    rotated_data = process_variant(INPUT_DIR_ROTATED, OUTPUT_DIR_ROTATED, "ROTATED")
    
    print("[INFO] Processing variant: ROTATED2")
    rotated2_data = process_variant(INPUT_DIR_ROTATED2, OUTPUT_DIR_ROTATED2, "ROTATED2")
    
    # Find common base_ids across all variants
    all_ids = set(original_data.keys()) | set(rotated_data.keys()) | set(rotated2_data.keys())
    
    items = []
    for base_id in sorted(all_ids):
        svg = find_original_svg(base_id)
        
        items.append({
            "base_id": base_id,
            "svg": svg,
            "original": original_data.get(base_id, {}),
            "rotated": rotated_data.get(base_id, {}),
            "rotated2": rotated2_data.get(base_id, {})
        })
        
        print(f"✅ {base_id} | Original: {original_data.get(base_id, {}).get('acc_cmd', 'N/A')} | "
              f"Rotated: {rotated_data.get(base_id, {}).get('acc_cmd', 'N/A')} | "
              f"Rotated2: {rotated2_data.get(base_id, {}).get('acc_cmd', 'N/A')}")
    
    print(f"[SUMMARY] Total samples: {len(items)}")
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
<title>Multi-Variant STEP Comparison</title>
<style>
body {
    font-family: Arial, sans-serif;
    background: #f2f2f2;
    margin: 0;
    padding: 20px 0;
}
.container {
    max-width: 1600px;
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
.sample-section {
    margin-bottom: 30px;
    background: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.sample-header {
    font-weight: bold;
    font-size: 16px;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 2px solid #0066cc;
}
.grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}
.card {
    background: #fafafa;
    border-radius: 6px;
    padding: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.card .label {
    font-size: 13px;
    font-weight: bold;
    color: #333;
    margin-bottom: 8px;
    padding: 4px 8px;
    border-radius: 4px;
}
.card .label.original { background: #e3f2fd; }
.card .label.rotated { background: #fff3e0; }
.card .label.rotated2 { background: #f3e5f5; }
.card .label.gt { background: #e8f5e9; }
.metrics {
    font-size: 10px;
    color: #0066cc;
    margin: 4px 0;
    font-weight: 600;
}
.card img {
    width: 100%;
    height: auto;
    max-height: 300px;
    object-fit: contain;
    border-radius: 4px;
    background: #fff;
    border: 1px solid #ddd;
}
.placeholder {
    color: #999;
    font-size: 13px;
    padding: 60px 0;
}
.small {
    font-size: 10px;
    color: #666;
    margin-top: 6px;
}
.h5-info {
    font-size: 8px;
    color: #333;
    margin-top: 8px;
    padding: 6px;
    background: #fff;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    text-align: left;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    width: 100%;
    border: 1px solid #e0e0e0;
}
</style>
</head>
<body>
<div class="container">
<h1>🔴 LIVE: Multi-Variant STEP Comparison (3 Predictions + GT)</h1>
<div class="info">Flask Server Running | Total Samples: {{ total_count }}</div>

{% for item in items %}
<div class="sample-section">
    <div class="sample-header">
        Sample ID: {{ item.base_id }}
    </div>
    <div class="grid">
        {# Variant 1: Original #}
        <div class="card">
            <div class="label original">Original View</div>
            {% if item.original.png %}
                <img src="/png_original/{{ item.original.png }}" alt="original variant">
                <div class="small">{{ item.original.png }}</div>
                {% if item.original.acc_cmd is not none %}
                <div class="metrics">ACCcmd: {{ "%.2f"|format(item.original.acc_cmd) }}% | ACCparam: {{ "%.2f"|format(item.original.acc_param) }}%</div>
                {% endif %}
                {% if item.original.gen_h5 %}
                <div class="h5-info">{{ item.original.gen_h5 }}</div>
                {% endif %}
            {% else %}
                <div class="placeholder">Not available</div>
            {% endif %}
        </div>
        
        {# Variant 2: Rotated #}
        <div class="card">
            <div class="label rotated">Rotated View</div>
            {% if item.rotated.png %}
                <img src="/png_rotated/{{ item.rotated.png }}" alt="rotated variant">
                <div class="small">{{ item.rotated.png }}</div>
                {% if item.rotated.acc_cmd is not none %}
                <div class="metrics">ACCcmd: {{ "%.2f"|format(item.rotated.acc_cmd) }}% | ACCparam: {{ "%.2f"|format(item.rotated.acc_param) }}%</div>
                {% endif %}
                {% if item.rotated.gen_h5 %}
                <div class="h5-info">{{ item.rotated.gen_h5 }}</div>
                {% endif %}
            {% else %}
                <div class="placeholder">Not available</div>
            {% endif %}
        </div>
        
        {# Variant 3: Rotated2 #}
        <div class="card">
            <div class="label rotated2">Rotated2 View</div>
            {% if item.rotated2.png %}
                <img src="/png_rotated2/{{ item.rotated2.png }}" alt="rotated2 variant">
                <div class="small">{{ item.rotated2.png }}</div>
                {% if item.rotated2.acc_cmd is not none %}
                <div class="metrics">ACCcmd: {{ "%.2f"|format(item.rotated2.acc_cmd) }}% | ACCparam: {{ "%.2f"|format(item.rotated2.acc_param) }}%</div>
                {% endif %}
                {% if item.rotated2.gen_h5 %}
                <div class="h5-info">{{ item.rotated2.gen_h5 }}</div>
                {% endif %}
            {% else %}
                <div class="placeholder">Not available</div>
            {% endif %}
        </div>
        
        {# Ground Truth #}
        <div class="card">
            <div class="label gt">Ground Truth</div>
            {% if item.svg %}
                <img src="/svg/{{ item.svg }}" alt="ground truth">
                <div class="small">GT SVG</div>
                {% if item.original.truth_h5 or item.rotated.truth_h5 or item.rotated2.truth_h5 %}
                <div class="h5-info">{{ item.original.truth_h5 or item.rotated.truth_h5 or item.rotated2.truth_h5 }}</div>
                {% endif %}
            {% else %}
                <div class="placeholder">Not available</div>
            {% endif %}
        </div>
    </div>
</div>
{% endfor %}

</div>
</body>
</html>
"""
    
    return render_template_string(html_template, items=items, total_count=len(items))

@app.route('/png_original/<filename>')
def serve_png_original(filename):
    """Serve PNG files from original variant"""
    return send_from_directory(OUTPUT_DIR_ORIGINAL, filename)

@app.route('/png_rotated/<filename>')
def serve_png_rotated(filename):
    """Serve PNG files from rotated variant"""
    return send_from_directory(OUTPUT_DIR_ROTATED, filename)

@app.route('/png_rotated2/<filename>')
def serve_png_rotated2(filename):
    """Serve PNG files from rotated2 variant"""
    return send_from_directory(OUTPUT_DIR_ROTATED2, filename)

@app.route('/svg/<path:path>')
def serve_svg(path):
    """Serve SVG files from their original locations"""
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Flask server with 3 variants...")
    print("=" * 60)
    print(f"📂 Input DIR (Original): {INPUT_DIR_ORIGINAL}")
    print(f"📂 Input DIR (Rotated): {INPUT_DIR_ROTATED}")
    print(f"📂 Input DIR (Rotated2): {INPUT_DIR_ROTATED2}")
    print(f"📂 Output DIR (Original): {OUTPUT_DIR_ORIGINAL}")
    print(f"📂 Output DIR (Rotated): {OUTPUT_DIR_ROTATED}")
    print(f"📂 Output DIR (Rotated2): {OUTPUT_DIR_ROTATED2}")
    print(f"📂 SVG ROOT: {SVG_ROOT}")
    print(f"📂 CAD VEC ROOT: {CAD_VEC_ROOT}")
    print("=" * 60)
    print("🌐 Server will start at: http://127.0.0.1:3000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=3000)