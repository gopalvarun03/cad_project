import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display
from OCC.Core.Graphic3d import Graphic3d_RenderingParams

def step_to_iso_png(step_path, out_path):
    display, _, _, _ = init_display()
    
    # ✅ OFFSCREEN rendering
    display.View.SetImmediateUpdate(False)
    params = display.View.RenderingParams()
    params.NbMsaaSamples = 0
    params.IsAntialiasingEnabled = False

    reader = STEPControl_Reader()
    reader.ReadFile(step_path)
    reader.TransferRoots()
    shape = reader.OneShape()

    display.DisplayShape(shape, update=False)
    display.View_Iso()
    display.FitAll()

    display.Repaint()
    display.View.Dump(out_path)

    display.EraseAll()

# ------------------ BATCH ------------------
input_dir = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_results"
output_dir = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\output_pngs"
os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(input_dir):
    if f.lower().endswith(".step") or f.lower().endswith(".stp"):
        inp = os.path.join(input_dir, f)
        out = os.path.join(output_dir, f.replace(".step", ".png").replace(".stp", ".png"))
        step_to_iso_png(inp, out)
        print("✅ saved:", out)


# ================= HTML GALLERY (APPENDED ONLY) =================

html_path = os.path.join(output_dir, "gallery.html")

images = sorted([
    f for f in os.listdir(output_dir)
    if f.lower().endswith(".png")
])

html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Isometric STEP Gallery</title>
<style>
body {
    font-family: Arial, sans-serif;
    background: #f4f4f4;
}
h1 {
    text-align: center;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
    padding: 20px;
}
.card {
    background: white;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    text-align: center;
}
.card img {
    max-width: 100%;
    border-radius: 4px;
    cursor: pointer;
}
.name {
    margin-top: 6px;
    font-size: 13px;
    word-break: break-all;
}
</style>

<script>
function zoom(src) {
    const w = window.open("");
    w.document.write('<img src="' + src + '" style="width:100%;">');
}
</script>
</head>
<body>

<h1>Isometric STEP Gallery</h1>
<div class="grid">
"""

for img in images:
    html += f"""
    <div class="card">
        <img src="{img}" onclick="zoom(this.src)">
        <div class="name">{img}</div>
    </div>
    """

html += """
</div>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("✅ HTML gallery saved at:", html_path)
print("👉 Open it in any browser")
