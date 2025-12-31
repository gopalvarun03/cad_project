import os
from flask import Flask, render_template_string, send_from_directory

# ========== PATHS ==========
SVG_ROOT = r"C:\Users\LEGION\Desktop\cad_project\DeepCAD\data2\svg_raw"
OLD_GEN_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_pngs_original"
NEW_GEN_DIR = r"C:\Users\LEGION\Desktop\cad_project\Drawing2CAD\proj_log\epoch_100_shivank\test_pngs_rotated"

app = Flask(__name__)

# ========== UTILS ==========
def get_base_ids():
    # Use intersection of base names present in all three folders
    svg_ids = set()
    # Traverse svg_raw/<bucket>/<base_id>/<base_id>_FrontTopRight.svg
    for bucket in os.listdir(SVG_ROOT):
        bucket_path = os.path.join(SVG_ROOT, bucket)
        if not os.path.isdir(bucket_path):
            continue
        for base_id in os.listdir(bucket_path):
            base_path = os.path.join(bucket_path, base_id)
            if not os.path.isdir(base_path):
                continue
            svg_file = os.path.join(base_path, f"{base_id}_FrontTopRight.svg")
            if os.path.exists(svg_file):
                svg_ids.add(base_id)
    
    # Build sets of actual PNG file base names (with _vec suffix stripped)
    old_ids = {}  # maps base_id -> actual filename
    for f in os.listdir(OLD_GEN_DIR):
        if f.lower().endswith('.png'):
            base = os.path.splitext(f)[0]
            if base.endswith('_vec'):
                actual_base = base.replace('_vec', '')
                old_ids[actual_base] = f
            else:
                old_ids[base] = f

    new_ids = {}  # maps base_id -> actual filename
    for f in os.listdir(NEW_GEN_DIR):
        if f.lower().endswith('.png'):
            base = os.path.splitext(f)[0]
            if base.endswith('_vec'):
                actual_base = base.replace('_vec', '')
                new_ids[actual_base] = f
            else:
                new_ids[base] = f

    # Find common IDs that exist in all three locations
    common = []
    for base_id in svg_ids:
        if base_id in old_ids and base_id in new_ids:
            common.append(base_id)
    return sorted(common)

def get_svg_path(base_id):
    # SVGs are in SVG_ROOT/<bucket>/<base_id>/<base_id>_FrontTopRight.svg
    try:
        n = int(base_id)
        bucket = f"{n // 10000:04d}"
        svg_path = os.path.join(SVG_ROOT, bucket, base_id, f"{base_id}_FrontTopRight.svg")
        if os.path.exists(svg_path):
            return svg_path
    except Exception:
        pass
    # fallback: search for folder named base_id
    for bucket in os.listdir(SVG_ROOT):
        bucket_path = os.path.join(SVG_ROOT, bucket)
        if not os.path.isdir(bucket_path):
            continue
        base_path = os.path.join(bucket_path, base_id)
        svg_file = os.path.join(base_path, f"{base_id}_FrontTopRight.svg")
        if os.path.exists(svg_file):
            return svg_file
    return None

# ========== ROUTES ==========
def get_png_filename(base_id, directory):
    """Find the actual PNG filename for a given base_id in a directory."""
    # Try with _vec suffix first (most common case based on user's example)
    vec_file = base_id + "_vec.png"
    if os.path.exists(os.path.join(directory, vec_file)):
        return vec_file
    # Try without _vec suffix
    plain_file = base_id + ".png"
    if os.path.exists(os.path.join(directory, plain_file)):
        return plain_file
    return None

@app.route('/')
def index():
    base_ids = get_base_ids()
    items = []
    for base_id in base_ids:
        svg_path = get_svg_path(base_id)
        
        # Find actual filenames for PNGs
        old_png_filename = get_png_filename(base_id, OLD_GEN_DIR)
        old_png_final = os.path.join(OLD_GEN_DIR, old_png_filename) if old_png_filename else None
        
        new_png_filename = get_png_filename(base_id, NEW_GEN_DIR)
        new_png_final = os.path.join(NEW_GEN_DIR, new_png_filename) if new_png_filename else None
        
        items.append({
            'base_id': base_id,
            'svg': svg_path,
            'old_png': old_png_final,
            'old_png_filename' : old_png_filename,
            'new_png': new_png_final,
            'new_png_filename': new_png_filename
        })
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset='utf-8'>
    <title>Compare Isometric Images</title>
    <style>
    body { font-family: Arial, sans-serif; background: #f2f2f2; }
    .container { max-width: 1200px; margin: auto; }
    h1 { text-align: center; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
    th { background: #eee; }
    img { max-width: 320px; max-height: 240px; background: #fff; border-radius: 4px; }
    .idcell { font-weight: bold; background: #fafafa; }
    </style>
    </head>
    <body>
    <div class="container">
    <h1>Compare Isometric Images</h1>
    <table>
      <tr>
        <th>ID</th>
        <th>Original Isometric SVG</th>
        <th>Old Generated PNG</th>
        <th>New Generated PNG</th>
      </tr>
      {% for item in items %}
      <tr>
        <td class="idcell">{{ item.base_id }}</td>
        <td>{% if item.svg %}<img src="/svg/{{ item.base_id }}">{% else %}<span>Not found</span>{% endif %}</td>
        <td>{% if item.old_png %}<img src="/old_png/{{ item.old_png_filename }}">{% else %}<span>Not found</span>{% endif %}</td>
        <td>{% if item.new_png %}<img src="/new_png/{{ item.new_png_filename }}">{% else %}<span>Not found</span>{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
    <div style="margin:20px; color:#888; font-size:13px;">Total: {{ items|length }} cases</div>
    </div>
    </body>
    </html>
    """
    return render_template_string(html, items=items)

@app.route('/svg/<base_id>')
def serve_svg(base_id):
    svg_path = get_svg_path(base_id)
    if svg_path:
        directory = os.path.dirname(svg_path)
        filename = os.path.basename(svg_path)
        return send_from_directory(directory, filename)
    return "SVG not found", 404

@app.route('/old_png/<filename>')
def serve_old_png(filename):
    return send_from_directory(OLD_GEN_DIR, filename)

@app.route('/new_png/<filename>')
def serve_new_png(filename):
    return send_from_directory(NEW_GEN_DIR, filename)

if __name__ == '__main__':
    print("Server running at http://127.0.0.1:5050")
    app.run(debug=True, host='0.0.0.0', port=5050)
