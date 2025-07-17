import os
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from components.params import PARAMS
from components.utils import generar_video_trayectoria
from components.dp_model import dp

# ================================
# MOSTRAR INFORMACIÓN DEL MODELO
# ================================

print("Modelo del robot:")
print(dp)

print("\nParámetros dinámicos del robot:")
# ===============================================================================
# TEST 1: Dinámica libre sin torque aplicado. Respuesta a condiciones iniciales
# ===============================================================================

tg = dp.nofriction(coulomb=True, viscous=True).fdyn(
    T=5,                            # Duración de la simulación [s]
    q0=[0, 0],                      # Condiciones iniciales de posición
    qd0=np.zeros((2,)),            # Condiciones iniciales de velocidad
    Q=None,                        # Sin torque aplicado
    dt=1e-3                        # Paso de integración
)

# ====================================================
# GENERAR VIDEO DE LA TRAYECTORIA DEL PÉNDULO DOBLE
# ====================================================

generar_video_trayectoria(
    dp,
    tg.q,
    nombre_archivo=os.path.join('videos','condiciones_iniciales.mp4'),
    export_fps=60,
    sim_dt=tg.t[1] - tg.t[0],
    l1=PARAMS['A1'],
    l2=PARAMS['A2']
)
