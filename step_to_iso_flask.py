import os
import io
import h5py
import zipfile
import threading
import numpy as np
from flask import Flask, render_template_string, send_from_directory, Response
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

# ================= PATHS =================
INPUT_DIR = r"C:\Users\thiri\OneDrive\Desktop\2DtoCAD\Drawing2CAD\proj_log\your_exp_name\evaluation_results_rotated2\test"
OUTPUT_DIR = r"C:\Users\thiri\OneDrive\Desktop\2DtoCAD\Drawing2CAD\proj_log\your_exp_name\evaluation_results_rotated2\test"
H5_DIR = INPUT_DIR  # Same as INPUT_DIR for predicted h5 files

# ZipFile settings
ZIP_FILE_PATH = r"C:\Users\thiri\Downloads\mini.zip"
ZIP_SVG_ROOT = "miniproject/DeepCAD/data/CAD-VGDrawing/svg_raw"
ZIP_CAD_VEC_ROOT = "miniproject/DeepCAD/data/CAD-VGDrawing/cad_vec"

# ================= METRIC CONSTANTS =================
CAD_EOS_IDX = 2
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
TOL = 3          # parameter tolerance
PAD_PARAM = -1   # padded argument marker

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= FLASK APP =================
app = Flask(__name__)

# ================= GLOBALS FOR FAST ZIPFILE ACCESS =================
zip_lock = threading.Lock()
_global_zf = None
_zip_namelist_set = set()

def get_zf():
    """Return the global ZipFile instance and the cached namelist set. Thread-safe."""
    global _global_zf, _zip_namelist_set
    with zip_lock:
        if _global_zf is None:
            print(f"Loading 75GB ZipFile central directory into memory once (this might take 5-10 seconds)...")
            _global_zf = zipfile.ZipFile(ZIP_FILE_PATH, 'r')
            print("Parsing internal index for ultra-fast lookups...")
            _zip_namelist_set = frozenset(_global_zf.namelist())
            print(f"ZipFile index loaded! Found {len(_zip_namelist_set)} files.")
    return _global_zf, _zip_namelist_set

# ================= METRIC FUNCTIONS =================
def load_pair(gen_path, truth_zip_path):
    """Load generated h5 file from disk and ground truth h5 file from zip"""
    try:
        zf, _ = get_zf()
        # Load truth from zip entirely into memory as bytes
        with zip_lock:
            with zf.open(truth_zip_path) as zfile:
                h5_bytes = zfile.read()
                
        # Use io.BytesIO to treat bytes as a file for h5py
        with io.BytesIO(h5_bytes) as bio:
            with h5py.File(bio, "r") as f:
                truth = f[TRUTH_DATASET_NAME][:]
                    
        # Load generated from disk
        with h5py.File(gen_path, "r") as f:
            gen = f[GENERATED_DATASET_NAME][:]
    except Exception as e:
        print(f"Error loading h5 pair: {e}")
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

def compute_metrics_for_file(gen_h5_path, truth_zip_path):
    """Compute ACCcmd and ACCparam for a single file pair"""
    pair = load_pair(gen_h5_path, truth_zip_path)
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
    """Find ground truth h5 file path within the zip for a given step name"""
    base_id = step_name.replace("_vec.step", "").replace(".step", "")
    try:
        n = int(base_id)
    except Exception:
        return None
    
    bucket = f"{n // 10000:04d}"
    h5_zip_path = f"{ZIP_CAD_VEC_ROOT}/{bucket}/{base_id}.h5"
    
    zf, zf_names = get_zf()
    if h5_zip_path in zf_names:
        return h5_zip_path
    return None

def read_h5_content(path, dataset_name, in_zip=False):
    """Read h5 file content from disk or zip and format as CAD commands"""
    try:
        if in_zip:
            zf, _ = get_zf()
            with zip_lock:
                with zf.open(path) as zfile:
                    h5_bytes = zfile.read()
            with io.BytesIO(h5_bytes) as bio:
                with h5py.File(bio, "r") as f:
                    data = f[dataset_name][:]
        else:
            with h5py.File(path, "r") as f:
                data = f[dataset_name][:]
        
        cmd_names = {0: 'Line', 1: 'Arc', 2: 'Circle', 3: 'EOS', 4: 'SOL', 5: 'Ext'}
        
        formatted = []
        for i, row in enumerate(data):
            cmd_type = int(row[0])
            params = row[1:].astype(int)
            
            cmd_name = cmd_names.get(cmd_type, f'Unknown({cmd_type})')
            
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
    base_id = step_name.replace("_vec.step", "").replace(".step", "").split(".")[0]
    try:
        n = int(base_id)
    except Exception as e:
        print("⚠️ cannot parse step name:", step_name)
        return None
    bucket = f"{n // 10000:04d}"
    svg_zip_path = f"{ZIP_SVG_ROOT}/{bucket}/{base_id}/{base_id}_FrontTopRight.svg"
    
    zf, zf_names = get_zf()
    if svg_zip_path in zf_names:
        return svg_zip_path
    return None

# ============== PROCESS ALL STEPS ==============
def process_all_steps():
    """Process all STEP files and generate data"""
    items = []
    
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"❌ Zip file not found: {ZIP_FILE_PATH}")
        return items
        
    for f in sorted(os.listdir(INPUT_DIR)[:1000]):
        try:
            if not f.lower().endswith((".step", ".stp")):
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

            svg_zip_path = find_original_svg(f)
            if not svg_zip_path:
                print(f"[INFO] SVG not found for: {f}")

            # Compute metrics
            base_id = f.replace("_vec.step", "").replace(".step", "").split(".")[0]
            gen_h5_path = os.path.join(H5_DIR, base_id + ".h5")
            truth_h5_zip_path = find_ground_truth_h5(f)

            acc_cmd, acc_param = None, None
            gen_h5_content, truth_h5_content = None, None

            if not os.path.exists(gen_h5_path):
                print(f"[INFO] Generated h5 missing: {gen_h5_path}")
            if not truth_h5_zip_path:
                print(f"[INFO] Ground truth h5 missing in zip for: {f}")

            if truth_h5_zip_path and os.path.exists(gen_h5_path):
                try:
                    acc_cmd, acc_param = compute_metrics_for_file(gen_h5_path, truth_h5_zip_path)
                    if acc_cmd is None or acc_param is None:
                        print(f"[WARN] Metrics not computed for: {f}")
                except Exception as e:
                    print(f"[ERROR] Exception in metric computation for {f}: {e}")
                try:
                    gen_h5_content = read_h5_content(gen_h5_path, GENERATED_DATASET_NAME, in_zip=False)
                except Exception as e:
                    print(f"[ERROR] Exception reading generated h5 content for {gen_h5_path}: {e}")
                try:
                    truth_h5_content = read_h5_content(truth_h5_zip_path, TRUTH_DATASET_NAME, in_zip=True)
                except Exception as e:
                    print(f"[ERROR] Exception reading ground truth h5 content from zip {truth_h5_zip_path}: {e}")

            items.append({
                "step": f,
                "png": png_name,
                "svg": svg_zip_path,
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
    
    <div class="card">
        <div class="label">Generated</div>
        <img src="/png/{{ a.png }}" alt="generated">
        <div class="small">{{ a.png }}</div>
        {% if a.gen_h5 %}
        <div class="h5-info">{{ a.gen_h5 }}</div>
        {% endif %}
    </div>
    
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

@app.route('/svg/<path:svg_zip_path>')
def serve_svg(svg_zip_path):
    """Serve SVG directly from the Zip file"""
    try:
        zf, _ = get_zf()
        with zip_lock:
            svg_data = zf.read(svg_zip_path)
        return Response(svg_data, mimetype='image/svg+xml')
    except Exception as e:
        return f"SVG not found in zip: {e}", 404

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Flask server (ZipFile Cache Mode)...")
    print("=" * 60)
    print(f"📂 Zip File: {ZIP_FILE_PATH}")
    print(f"📂 Zip SVG Root: {ZIP_SVG_ROOT}")
    print(f"📂 Zip CAD Vec Root: {ZIP_CAD_VEC_ROOT}")
    print("=" * 60)
    print("🌐 Server will start at: http://127.0.0.1:3000")
    print("=" * 60)
    
    # Preload the zip file to avoid hanging on the first request
    get_zf()
    app.run(debug=True, host='0.0.0.0', port=3000)
