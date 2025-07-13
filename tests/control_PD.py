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
from components import pd_controller
from components import config_pd

# ============================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ============================================================

# Parámetros generales
T = 5                             # Duración total de la simulación [s]
q0 = [-np.pi/2, 0]               # Posición inicial de las articulaciones [rad]
qd0 = np.zeros((2,))            # Velocidad inicial de las articulaciones [rad/s]
dt = 1e-3                        # Paso de integración [s]

# ============================================================
# GENERACIÓN DE TAYECTORIA
# ============================================================
q_final = np.array([0, 0])  # Posición objetivo final
q_mid = np.array([np.pi/2,0])


traj = rtb.tools.trajectory.mstraj(viapoints = np.array([q0,q_mid,q_final]),dt=dt,tacc=0.2,tsegment= np.array([T/2,T/2]))

q_ref = traj.s
qd_ref = traj.sd
qdd_ref = traj.sdd

# ============================================================
# GENERACIÓN DE REFERENCIA
# ============================================================
# Crear controlador PD
controller = pd_controller.make_pd_controller(
    dp=dp,
    params=PARAMS,
    dt=dt,
    kp=config_pd.kp,
    kd=config_pd.kd,
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
    Qm_vec[i, :] = controller(dp, tg.t[i], tg.q[i], tg.qd[i])

# ============================================================
# VISUALIZACIÓN: VIDEO + GRÁFICAS DE RESULTADOS
# ============================================================

# Generar animación de la trayectoria
generar_video_trayectoria(
    dp,
    tg.q,
    nombre_archivo=os.path.join('videos', 'trayectoria_prueba1_pd.mp4'),
    export_fps=60,
    sim_dt=tg.t[1] - tg.t[0],
    l1=PARAMS['A1'],
    l2=PARAMS['A2']
)

# Graficar dinámica: q(t), qd(t), q_ref(t), torques
graficar_dinamica(dp=dp, q=tg.q, qd=tg.qd, t=tg.t, tau = Qm_vec, q_ref=q_ref)
plt.show()