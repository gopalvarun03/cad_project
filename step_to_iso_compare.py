import os
import h5py
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

# ================= PATHS =================
INPUT_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\evaluation_results\test"
OUTPUT_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\new_pngs"
SVG_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\svg_raw"
CAD_VEC_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\cad_vec"
H5_DIR = INPUT_DIR  # Same as INPUT_DIR for predicted h5 files
HTML_PATH = os.path.join(OUTPUT_DIR, "gallery_compare.html")

# ================= METRIC CONSTANTS =================
CAD_EOS_IDX = 2
GENERATED_DATASET_NAME = "out_vec"
TRUTH_DATASET_NAME = "vec"
TOL = 3          # parameter tolerance
PAD_PARAM = -1   # padded argument marker

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# ============== INIT OCC VIEWER ONCE ==============
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
    print(svg_path)
    return svg_path if os.path.exists(svg_path) else None

# ============== PROCESS ALL STEPS ==============
items = []   # list of dicts: {step, png, svg_or_None, acc_cmd, acc_param}
for f in sorted(os.listdir(INPUT_DIR)):
    if not f.lower().endswith((".step", ".stp")):
        continue
    step_path = os.path.join(INPUT_DIR, f)
    png_name = os.path.splitext(f)[0] + ".png"
    png_path = os.path.join(OUTPUT_DIR, png_name)

    ok = step_to_iso_png(step_path, png_path)
    if not ok:
        # skip invalid steps but keep log
        print("⚠️ skipped:", f)
        continue
    
    svg = find_original_svg(f)
    
    # Compute metrics
    base_id = f.replace("_vec.step", "").replace(".step", "").split(".")[0]
    gen_h5_path = os.path.join(H5_DIR, base_id + ".h5")
    truth_h5_path = find_ground_truth_h5(f)
    
    acc_cmd, acc_param = None, None
    if truth_h5_path and os.path.exists(gen_h5_path):
        acc_cmd, acc_param = compute_metrics_for_file(gen_h5_path, truth_h5_path)

    items.append({
        "step": f, 
        "png": png_name, 
        "svg": svg,
        "acc_cmd": acc_cmd,
        "acc_param": acc_param
    })
    print("✅ saved:", png_path, f"| ACCcmd: {acc_cmd:.2f}%" if acc_cmd else "", f"ACCparam: {acc_param:.2f}%" if acc_param else "")

# ============== BUILD HTML (2 items per row => 4 columns) ==============
# CSS:
# - grid-template-columns: repeat(4, 1fr)
# - for each pair, emit two stepname headers (each span 2 columns)
# - next emit 4 cards: genA, origA, genB, origB
html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>STEP Isometric Comparison</title>
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
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
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
</style>
</head>
<body>
<div class="container">
<h1>STEP Isometric Comparison</h1>
<div class="grid">
"""

# iterate items in pairs (2 items per row)
i = 0
n = len(items)
# n=10
while i < n:
    a = items[i]
    b = items[i+1] if (i+1) < n else None

    # header row: stepname for a (span 2), stepname for b (span 2)
    html += f'<div class="stepname">{a["step"]}'
    if a["acc_cmd"] is not None:
        html += f' <span style="color:#0066cc;">[ACCcmd: {a["acc_cmd"]:.2f}% | ACCparam: {a["acc_param"]:.2f}%]</span>'
    html += '</div>\n'
    
    if b:
        html += f'<div class="stepname">{b["step"]}'
        if b["acc_cmd"] is not None:
            html += f' <span style="color:#0066cc;">[ACCcmd: {b["acc_cmd"]:.2f}% | ACCparam: {b["acc_param"]:.2f}%]</span>'
        html += '</div>\n'
    else:
        html += f'<div class="stepname"></div>\n'  # empty header for missing second item

    # cards row: Generated A
    html += '<div class="card">\n'
    html += '<div class="label">Generated</div>\n'
    html += f'<img src="{a["png"]}" alt="generated">\n'
    html += f'<div class="small">{a["png"]}</div>\n'
    html += '</div>\n'

    # Original A

    # print(a)
    html += '<div class="card">\n'
    html += '<div class="label">Original</div>\n'
    if a["svg"]:
        rel_svg = os.path.relpath(a["svg"], OUTPUT_DIR).replace("\\", "/")
        html += f'<img src="{rel_svg}" alt="original">\n'
        html += f'<div class="small">{rel_svg}</div>\n'
    else:
        html += '<div class="placeholder">Not available</div>\n'
    html += '</div>\n'

    # If we have second item, add its generated & original, otherwise add two empty cards
    if b:
        # Generated B
        html += '<div class="card">\n'
        html += '<div class="label">Generated</div>\n'
        html += f'<img src="{b["png"]}" alt="generated">\n'
        html += f'<div class="small">{b["png"]}</div>\n'
        html += '</div>\n'

        # Original B
        html += '<div class="card">\n'
        html += '<div class="label">Original</div>\n'
        if b["svg"]:
            rel_svg_b = os.path.relpath(b["svg"], OUTPUT_DIR).replace("\\", "/")
            html += f'<img src="{rel_svg_b}" alt="original">\n'
            html += f'<div class="small">{rel_svg_b}</div>\n'
        else:
            html += '<div class="placeholder">Not available</div>\n'
        html += '</div>\n'
    else:
        # empty placeholders to keep grid layout
        html += '<div class="card"><div class="placeholder">--</div></div>\n'
        html += '<div class="card"><div class="placeholder">--</div></div>\n'

    i += 2

html += """
</div> <!-- grid -->
</div> <!-- container -->
</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("✅ DONE")
print("✅ Images saved in:", OUTPUT_DIR)
print("✅ HTML gallery:", HTML_PATH)
print("👉 Open gallery_compare.html in browser")