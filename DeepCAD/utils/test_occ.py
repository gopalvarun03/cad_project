import numpy as np
from OCC.Core.gp import gp_Pnt

try:
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pnt = gp_Pnt(*p)
    print("gp_Pnt with float32 worked")
except Exception as e:
    print(f"gp_Pnt with float32 failed: {e}")

try:
    p = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pnt = gp_Pnt(*p)
    print("gp_Pnt with float64 worked")
except Exception as e:
    print(f"gp_Pnt with float64 failed: {e}")
