import runpy
import sys
from pathlib import Path

# Keep the existing numerical environment first; use the original Point-MAE
# environment only to supply dependencies missing from the native environment.
sys.path.append('/workspace/home/areal_aerojepa_20260928/.venv/lib/python3.12/site-packages')
sys.path.insert(0,str(Path(__file__).resolve().parent))
script=sys.argv[1]
sys.argv=sys.argv[1:]
runpy.run_path(script,run_name='__main__')
