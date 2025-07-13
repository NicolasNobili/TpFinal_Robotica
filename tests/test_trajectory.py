import os
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import spatialmath as spm
import time
import math

# Módulos personalizados del proyecto
from components.params import PARAMS
from components.utils import (
    generar_video_trayectoria,
    generar_escalon_suave,
    graficar_dinamica,
    generar_vertices_poligono
)
from components.dp_model import dp
from components import pdFF_controller
from components import trajectory_generator as tg
from components import config_pd
from components import plot_examples

# ============================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ============================================================

# Parámetros generales
dt = 0.0083                      # Paso de integración [s]
# dt = 0.001

x0_cart = [0.5657 , 0.0]   # Posición inicial cartesiana
q0 = np.array([-np.pi/4, np.pi/2])
qd0  = np.zeros((2,))
qdmax = [0.2, 0.2]    # Velocidades máximas (m/s)

# ============================================================
# GENERACIÓN DE TAYECTORIA
# ============================================================

# Puntos por los que debe pasar el efector (en XY)
waypoints = generar_vertices_poligono(0.75,5,angulo_inicial=np.pi/2)
waypoints = plot_examples.batman_logo_waypoints()
waypoints = np.vstack([waypoints, waypoints[0]])

traj,q_ref,qd_ref,qdd_ref = tg.generate_cartesian_trajectory(
    dp = dp,
    viapoints=waypoints,
    tacc = 0.2,
    x0=x0_cart,
    q0=q0,
    qdmax=qdmax,
    tsegment=None,
    dt=dt
)

plt.figure(figsize=(6, 6))
plt.plot(traj.q[:, 0], traj.q[:, 1], marker='o', linestyle='-')
plt.title("Trayectoria de un Polígono Regular Inscrito")
plt.gca().set_aspect('equal')
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# ============================================================
# GENERACIÓN DE CONTROLADOR
# ============================================================

# Crear controlador PD
controller = pdFF_controller.make_pdFF_controller(
    dp=dp,
    params=PARAMS,
    dt=dt,
    kp=config_pd.kp,
    kd=config_pd.kd,
    q_ref=q_ref,
    qd_ref= qd_ref,
    qdd_ref=qdd_ref
)

# ============================================================
# EJECUCIÓN DE LA SIMULACIÓN DINÁMICA
# ============================================================
T = traj.t[-1]
print(T)
tg = dp.nofriction(coulomb=True, viscous=False).fdyn(
    T=T,
    q0=q0,
    Q=controller,
    qd0=qd0,
    dt=dt,
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
    nombre_archivo=os.path.join('videos', 'trayectoria_prueba1_poligono.mp4'),
    export_fps=60,
    sim_dt=tg.t[1] - tg.t[0],
    l1=PARAMS['A1'],
    l2=PARAMS['A2']
)

# Graficar dinámica: q(t), qd(t), q_ref(t), torques
graficar_dinamica(dp=dp, q=tg.q, qd=tg.qd, t=tg.t, tau = Qm_vec, q_ref=q_ref, qd_ref=qd_ref)
plt.show()