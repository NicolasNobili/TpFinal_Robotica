import numpy as np
import roboticstoolbox as rtb

from components.params import PARAMS

# ========================================================
# DEFINICIÓN DEL ROBOT DOBLE PÉNDULO USANDO DH CON DINÁMICA
# ========================================================

# Crear robot con 2 articulaciones rotacionales (modelo planar)
dp = rtb.DHRobot(
    [
        # Eslabón 1
        rtb.RevoluteDH(
            a=PARAMS['A1'],        # Longitud del eslabón
            m=PARAMS['M1'],        # Masa del eslabón
            r=PARAMS['R1'],        # Vector COM relativo al marco del eslabón
            I=PARAMS['I1'],        # Tensor de inercia
            B=PARAMS['B1'],        # Fricción viscosa
            G=PARAMS['N1']         # Relación de engranajes
        ),

        # Eslabón 2
        rtb.RevoluteDH(
            a=PARAMS['A2'],
            m=PARAMS['M2'],
            r=PARAMS['R2'],
            I=PARAMS['I2'],
            B=PARAMS['B2'],
            G=PARAMS['N2']
        )
    ],
    gravity=np.array([0, -9.8, 0]),  # Gravedad actuando hacia -Y
    name="dp"
)