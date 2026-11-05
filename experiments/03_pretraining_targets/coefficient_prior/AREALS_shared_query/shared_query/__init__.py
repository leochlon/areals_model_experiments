"""Joint geometric supervision on the native AREALS spatial query pathway."""
from pathlib import Path
import sys
_VENDOR = str(Path(__file__).resolve().parents[1] / 'vendor')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
__version__ = '1.0.0'
