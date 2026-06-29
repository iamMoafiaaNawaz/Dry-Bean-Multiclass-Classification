"""
conftest.py
-----------
Pytest configuration — adds src/ to sys.path so all test modules
can import from src/ without any manual path manipulation.
"""

import sys
from pathlib import Path

# Add src/ to path — works both locally and inside Docker (/app/src)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))