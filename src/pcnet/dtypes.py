"""Tipo numérico único para todo o projeto.

float32 e não float64: é o que existe no ESP32 (e mesmo esse vai desaparecer
na fase de ternarização). Fixar aqui evita que o NumPy promova silenciosamente
para float64 e que o Python e o C divirjam em bits que depois custam a
explicar.
"""

from __future__ import annotations

import numpy as np

F = np.float32
