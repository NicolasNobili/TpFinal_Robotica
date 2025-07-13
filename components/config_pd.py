
import numpy as np

# Módulos personalizados del proyecto
from components.params import PARAMS
from components.dp_model import dp

# Parámetros del modelo del actuador
Jm = np.array([[PARAMS['JM1'], 0], [0, PARAMS['JM2']]])   # Inercia del motor
N  = np.array([[PARAMS['N1'], 0], [0, PARAMS['N2']]])     # Relación de transmisión
Km = np.array([[PARAMS['KM1'], 0], [0, PARAMS['KM2']]])   # Constante del motor

# ============================================================
# CÁLCULO DE MATRIZ DE INERCIA EFECTIVA (Jef)
# ============================================================

# Configuración típica para evaluación
q_test = np.array([0, np.pi/2])

# Matriz de inercia completa M(q)
mbar_full = dp.inertia(q_test)

# Aproximación diagonal de M(q)
mbar = np.array([
    [mbar_full[0, 0], 0],
    [0, mbar_full[1, 1]]
])

# Inercia efectiva del sistema (motor + carga referida al motor)
Jef = Jm * N**2 + mbar

# ============================================================
# DISEÑO DEL CONTROLADOR PD
# ============================================================

# Frecuencia natural deseada [rad/s]
wn = 30
Bef = 0  # Fricción equivalente (despreciada)

# Ganancias proporcionales y derivativas
kp = np.divide(wn**2 * Jef, N * Km, out=np.zeros_like(Jef), where=(N * Km) != 0)
kd = np.divide(2 * np.sqrt(N * Km * kp * Jef) - Bef, N * Km, out=np.zeros_like(Jef), where=(N * Km) != 0)
