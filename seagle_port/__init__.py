"""SEAGLE → DFlash port: SpinQuant target machinery reused module-by-module.

The SEAGLE repo (/home/thahn1230/eagle_spinquant_w4a4) is read-only; we import
its third_party/SpinQuant utilities via sys.path and keep all new code here.
"""
import sys

SEAGLE_ROOT = "/home/thahn1230/eagle_spinquant_w4a4"
SPINQUANT_ROOT = f"{SEAGLE_ROOT}/third_party/SpinQuant"

if SPINQUANT_ROOT not in sys.path:
    sys.path.insert(0, SPINQUANT_ROOT)
