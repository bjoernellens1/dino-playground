import json
import ast
import sys

notebooks = [
    "/home/bjoern/git/dino-playground/notebooks/02_segmentation_linear_probe.ipynb",
    "/home/bjoern/git/dino-playground/notebooks/03_detection_dense_head.ipynb",
    "/home/bjoern/git/dino-playground/notebooks/04_depth_head.ipynb"
]

def verify_notebook(path):
    print(f"Verifying {path}...")
    try:
        with open(path, 'r') as f:
            nb = json.load(f)
        
        code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
        full_code = ""
        for cell in code_cells:
            # Join lines in a cell
            cell_code = "".join(cell)
            full_code += cell_code + "\n"
            
        # Check syntax
        ast.parse(full_code)
        print(f"✅ {path} is valid JSON and Python syntax.")
        return True
    except Exception as e:
        print(f"❌ {path} failed verification: {e}")
        return False

success = True
for nb in notebooks:
    if not verify_notebook(nb):
        success = False

if not success:
    sys.exit(1)
