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
    generar_vertices_poligono,
    graficar_trayectoria
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
dt = 0.002                      # Paso de integración [s]
# dt = 0.001

x0_cart = [0.5657 , 0.0]   # Posición inicial cartesiana
q0 = np.array([-np.pi/4, np.pi/2])
qd0  = np.zeros((2,))
qdmax = [0.5, 0.5]    # Velocidades máximas (m/s)

# ============================================================
# GENERACIÓN DE TAYECTORIA
# ============================================================

# Puntos por los que debe pasar el efector (en XY)
# waypoints = generar_vertices_poligono(0.7,4,angulo_inicial=np.pi/4)
waypoints = plot_examples.snake_waypoints()
waypoints = np.vstack([waypoints, waypoints[0]])
waypoints = np.vstack([waypoints,x0_cart])
# x0_cart = waypoints[0]

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

graficar_trayectoria(traj=traj)