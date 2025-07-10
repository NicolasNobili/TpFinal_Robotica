import numpy as np

m_e1 = 0.972 # [Kg]
l_e1 = 0.4 # [m]
m_m2 = 2.7 # [Kg]

m_e2 = 0.756 # [Kg]
l_e2 = 0.4 # [m]
m_m2 = 1.55 # [kg]

m_c = 4 # [Kg]



PARAMS = {
    # === Eslabón 1 ===
    'M1': m_e1 + m_m2,                   # Masa [kg]
    'A1': l_e1,                          # Longitud [m]
    'R1': (np.array([-l_e1/2, 0.0, 0.0]) * m_e1 + np.array([-l_e1, 0.0, 0.0]) * m_m2) / (m_e1 + m_m2) ,   # Centro de masa [m]
    'I1': np.array([0.0, 0.0, 0.0,       # Tensor de inercia 3x3 en forma de 9 elementos
                    0.0, (1/12)*m_e1*l_e1**2 + (m_e1 * m_m2 * l_e1**2)/(4*(m_e1 + m_m2)), 0.0,
                    0.0, 0.0, (1/12)*m_e1*l_e1**2 + (m_e1 * m_m2 * l_e1**2)/(4*(m_e1 + m_m2))]),
    'B1': 0.01,                          # Fricción viscosa [Nm/(rad/s)]
    'N1': 50, #50,                            # Relación de transmisión (sin unidades)
    'JM1': 1.65e-4,                      # Inercia del motor [kg·m²]
    'KM1': 0.645,                        # Constante del motor


    # === Eslabón 2 ===
    'M2': m_c + m_e2,
    'A2': 0.4,
    'R2':  (np.array([-l_e2/2, 0.0, 0.0]) * m_e2 + np.array([0, 0.0, 0.0]) * m_c) / (m_e1 + m_c),
    'I2': np.array([0.0, 0.0, 0.0,      
                    0.0, (1/12)*m_e2*l_e2**2 + (m_e2 * m_c * l_e2**2)/(4*(m_e2 + m_c)), 0.0,
                    0.0, 0.0, (1/12)*m_e2*l_e2**2 + (m_e2 * m_c * l_e2**2)/(4*(m_e2 + m_c))]),
    'B2': 0.01,
    'N2': 25,#25,
    'JM2': 0.58e-4,
    'KM2': 0.645,
}
