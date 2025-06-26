import numpy as np

PARAMS = {
    # === Eslabón 1 ===
    'M1': 1.0,                          # Masa [kg]
    'A1': 1.0,                          # Longitud [m]
    'R1': np.array([-0.1, 0.0, 0.0]),   # Centro de masa [m]
    'I1': np.array([0.0, 0.0, 0.0,      # Tensor de inercia 3x3 en forma de 9 elementos
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 1e-3]),
    'B1': 0.01,                         # Fricción viscosa [Nm/(rad/s)]
    'N1': 1,                            # Relación de transmisión (sin unidades)
    'JM1': 0.0,                         # Inercia del motor [kg·m²]
    'KM1': 1.0,                         # Constante del motor


    # === Eslabón 2 ===
    'M2': 2.0,
    'A2': 2.0,
    'R2': np.array([-0.2, 0.0, 0.0]),
    'I2': np.array([0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 2e-3]),
    'B2': 0.02,
    'N2': 2,
    'JM2': 0.0,
    'KM2': 1.0,
}
