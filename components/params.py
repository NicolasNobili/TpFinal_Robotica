import numpy as np

# === Parámetros físicos del sistema ===

# Eslabón 1 (primer brazo rotacional)
m_e1 = 0.972  # Masa del eslabón 1 (perfil de aluminio) [kg]
l_e1 = 0.4    # Longitud del eslabón 1 [m]
m_m1 = 2.7    # Masa del motor 1 [kg]

# Eslabón 2 (segundo brazo rotacional)
m_e2 = 0.756  # Masa del eslabón 2 (perfil de aluminio) [kg]
l_e2 = 0.4    # Longitud del eslabón 2 [m]
m_m2 = 1.55   # Masa del motor 2 (ubicado al final del eslabón 1) [kg]

# Carga útil (end-effector): cabezal Raytools
m_c = 4       # Masa del cabezal láser [kg]

# === Diccionario de parámetros del modelo dinámico ===
PARAMS = {

    # --- Eslabón 1 ---
    'M1': m_e1 + m_m2,  # Masa total del primer eslabón: incluye el eslabón + motor 2 [kg]

    'A1': l_e1,         # Longitud del eslabón 1 [m]

    # Centro de masa del eslabón 1:
    # Calculado como el centroide ponderado entre el perfil y el motor 2 (motor está al extremo).
    'R1': (
        (np.array([-l_e1/2, 0.0, 0.0]) * m_e1 + np.array([-l_e1, 0.0, 0.0]) * m_m2)
        / (m_e1 + m_m2)
    ),  # Posición relativa [m]

    # Tensor de inercia I1 (en forma de vector 3x3 aplanado):
    # Se considera la inercia del perfil y el efecto de la masa del motor en el extremo.
    'I1': np.array([
        0.0, 0.0, 0.0,
        0.0, (1/12)*m_e1*l_e1**2 + (m_e1 * m_m2 * l_e1**2) / (4*(m_e1 + m_m2)), 0.0,
        0.0, 0.0, (1/12)*m_e1*l_e1**2 + (m_e1 * m_m2 * l_e1**2) / (4*(m_e1 + m_m2))
    ]),  # [kg·m²]

    'B1': 0.01,         # Coeficiente de fricción viscosa del primer actuador [Nm/(rad/s)]

    'N1': 50,           # Relación de transmisión para el motor 1 (reductor sin fin)

    'JM1': 1.65e-4,     # Inercia del motor 1 (rotor) [kg·m²]

    'KM1': 0.645,       # Constante del motor 1 (torque por corriente, o similar)

    # --- Eslabón 2 ---
    'M2': m_c + m_e2,   # Masa total del eslabón 2 (perfil + carga útil)

    'A2': l_e2,         # Longitud del eslabón 2 [m]

    # Centro de masa del eslabón 2:
    # Se pondera entre el centro del perfil y la posición de la carga (al extremo).
    'R2': (
        (np.array([-l_e2/2, 0.0, 0.0]) * m_e2 + np.array([0, 0.0, 0.0]) * m_c)
        / (m_e2 + m_c)
    ),  # [m]

    # Tensor de inercia I2 (como vector aplanado 3x3):
    # Considera la inercia del perfil + carga al final del eslabón.
    'I2': np.array([
        0.0, 0.0, 0.0,
        0.0, (1/12)*m_e2*l_e2**2 + (m_e2 * m_c * l_e2**2) / (4*(m_e2 + m_c)), 0.0,
        0.0, 0.0, (1/12)*m_e2*l_e2**2 + (m_e2 * m_c * l_e2**2) / (4*(m_e2 + m_c))
    ]),  # [kg·m²]

    'B2': 0.01,         # Fricción viscosa del segundo actuador [Nm/(rad/s)]

    'N2': 25,           # Relación de transmisión del segundo motor

    'JM2': 0.58e-4,     # Inercia del rotor del motor 2 [kg·m²]

    'KM2': 0.645        # Constante del motor 2
}
