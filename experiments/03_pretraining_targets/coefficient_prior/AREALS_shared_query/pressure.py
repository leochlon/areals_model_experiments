#!/usr/bin/env python3
"""Pressure tuning/selected test scoring on the retained AREALS query pathway."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from shared_query.pressure import main

if __name__ == "__main__":
    main()
