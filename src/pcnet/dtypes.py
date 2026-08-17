"""Single numeric type for the whole project.

float32 and not float64: it is what exists on the ESP32 (and even that will
disappear in the ternarization phase). Fixing it here prevents NumPy from
silently promoting to float64 and keeps Python and C from diverging in bits
that are later hard to explain.
"""

from __future__ import annotations

import numpy as np

F = np.float32
