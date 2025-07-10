import os
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Módulos personalizados del proyecto
from components.params import PARAMS
from components.utils import (
    generar_video_trayectoria,
    generar_escalon_suave,
    graficar_dinamica
)
from components.dp_model import dp
from components import pdFFG_controller


# ============================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ============================================================

# Parámetros generales
T = 5                             # Duración total de la simulación [s]
q0 = [-np.pi/2, 0]               # Posición inicial de las articulaciones [rad]
qd0 = np.zeros((2,))            # Velocidad inicial de las articulaciones [rad/s]
dt = 1e-3                        # Paso de integración [s]

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

# Parámetros del modelo del actuador
Km = np.array([[PARAMS['KM1'], 0], [0, PARAMS['KM2']]])   # Constante del motor
Jm = np.array([[PARAMS['JM1'], 0], [0, PARAMS['JM2']]])   # Inercia del motor
N  = np.array([[PARAMS['N1'], 0], [0, PARAMS['N2']]])     # Relación de transmisión

# Inercia efectiva del sistema (motor + carga referida al motor)
Jef = Jm * N**2 + mbar

# ============================================================
# DISEÑO DEL CONTROLADOR PD
# ============================================================

# Frecuencia natural deseada [rad/s]
wn = 50
Bef = 0  # Fricción equivalente (despreciada)

# Ganancias proporcionales y derivativas
kp = np.divide(wn**2 * Jef, N * Km, out=np.zeros_like(Jef), where=(N * Km) != 0)
kd = np.divide(2 * np.sqrt(N * Km * kp * Jef) - Bef, N * Km, out=np.zeros_like(Jef), where=(N * Km) != 0)


# ============================================================
# GENERACIÓN DE CONTROLADOR
# ============================================================

q_final = np.array([0, 0])  # Posición objetivo final

# Genera trayectorias suaves para cada articulación
_, q_ref1 = generar_escalon_suave(T, dt, t_subida=1, q0=q0[0], ref_value=q_final[0])
_, q_ref2 = generar_escalon_suave(T, dt, t_subida=1, q0=q0[1], ref_value=q_final[1])
q_ref = np.vstack((q_ref1, q_ref2)).T

# Crear controlador PD
controller = pdFFG_controller.make_pdFFG_controller(
    dp=dp,
    params=PARAMS,
    dt=dt,
    kp=kp,
    kd=kd,
    q_ref=q_ref
)

# ============================================================
# EJECUCIÓN DE LA SIMULACIÓN DINÁMICA
# ============================================================

tg = dp.nofriction(coulomb=True, viscous=False).fdyn(
    T=T,
    q0=q0,
    Q=controller,
    qd0=qd0,
    dt=dt
)

# ============================================================
# CÁLCULO DE TORQUES APLICADOS (Q_vec)
# ============================================================

Qm_vec = np.zeros_like(q_ref)
for i in range(q_ref.shape[0]):
    Qm_vec[i, :] = controller(dp, tg.t[i], tg.q[i], tg.qd[i]) / np.array([PARAMS['N1'], PARAMS['N2']]) 

# ============================================================
# VISUALIZACIÓN: VIDEO + GRÁFICAS DE RESULTADOS
# ============================================================

# Generar animación de la trayectoria
generar_video_trayectoria(
    dp,
    tg.q,
    nombre_archivo=os.path.join('videos', 'trayectoria_prueba1_pdFF.mp4'),
    export_fps=60,
    sim_dt=tg.t[1] - tg.t[0],
    l1=PARAMS['A1'],
    l2=PARAMS['A2']
)

# Graficar dinámica: q(t), qd(t), q_ref(t), torques
graficar_dinamica(dp=dp, q=tg.q, qd=tg.qd, t=tg.t, tau = Qm_vec, q_ref=q_ref)
plt.show()
