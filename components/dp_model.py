import numpy as np
import roboticstoolbox as rtb

from components.params import PARAMS

# ========================================================
# DEFINICIÓN DEL ROBOT DOBLE PÉNDULO USANDO DH CON DINÁMICA
# ========================================================

dp = rtb.DHRobot(
    [
        # Eslabón 1
        rtb.RevoluteDH(
            a=PARAMS['A1'],        # Longitud del eslabón
            m=PARAMS['M1'],        # Masa del eslabón
            r=PARAMS['R1'],        # Vector COM relativo al marco del eslabón
            I=PARAMS['I1'],        # Tensor de inercia
            B=PARAMS['B1'],        # Fricción viscosa
            G=PARAMS['N1'],        # Relación de engranajes
            Jm = PARAMS['JM2']     # Inercia del motor
        ),

        # Eslabón 2
        rtb.RevoluteDH(
            a=PARAMS['A2'],        # Longitud del eslabón
            m=PARAMS['M2'],        # Masa del eslabón
            r=PARAMS['R2'],        # Vector COM relativo al marco del eslabón
            I=PARAMS['I2'],        # Tensor de inercia
            B=PARAMS['B2'],        # Fricción viscosa
            G=PARAMS['N2'],        # Relación de engranajes
            Jm = PARAMS['JM2']     # Inercia del motor
        )
    ],
    gravity=np.array([0, -9.8, 0]),  # Gravedad actuando hacia -Y
    name="dp"
)