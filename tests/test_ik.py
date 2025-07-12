import math
import numpy as np

import os
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import spatialmath as spm
import time

# Módulos personalizados del proyecto
from components.params import PARAMS
from components.utils import (
    generar_video_trayectoria,
    generar_escalon_suave,
    graficar_dinamica
)
from components.dp_model import dp
from components import pdFF_controller


def forward_kinematics(theta1, theta2, l1, l2):
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    return np.array([x, y])

def inverse_kinematics_iterative(xy_target, l1, l2,
                                  theta_init=(0.0, 0.0),
                                  max_iter=1000, tol=1e-4, alpha=0.1):
    """
    Cinemática inversa iterativa (Jacobiano transpuesto) para péndulo doble.

    Parámetros:
    - xy_target: np.array o lista [x, y] objetivo.
    - l1, l2: longitudes de eslabones.
    - theta_init: np.array o tuple de ángulos iniciales (θ1, θ2).
    - max_iter: máximo número de iteraciones.
    - tol: tolerancia de convergencia.
    - alpha: tasa de aprendizaje.

    Retorna:
    - (θ1, θ2) si converge, o None si no converge.
    """
    xy_target = np.array(xy_target, dtype=np.float64).flatten()
    theta = np.array(theta_init, dtype=np.float64).flatten()

    for i in range(max_iter):
        # Posición actual del extremo
        end_effector = forward_kinematics(theta[0], theta[1], l1, l2)

        # Error vectorial
        error = xy_target - end_effector
        if np.linalg.norm(error) < tol:
            return tuple(theta)

        # Jacobiano transpuesto
        J = np.array([
            [-l1 * math.sin(theta[0]) - l2 * math.sin(theta[0] + theta[1]),
             -l2 * math.sin(theta[0] + theta[1])],
            [l1 * math.cos(theta[0]) + l2 * math.cos(theta[0] + theta[1]),
             l2 * math.cos(theta[0] + theta[1])]
        ])

        # Actualización
        theta += alpha * J.T @ error

    return None  # No convergió

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
wn = 30
Bef = 0  # Fricción equivalente (despreciada)

# Ganancias proporcionales y derivativas
kp = np.divide(wn**2 * Jef, N * Km, out=np.zeros_like(Jef), where=(N * Km) != 0)
kd = np.divide(2 * np.sqrt(N * Km * kp * Jef) - Bef, N * Km, out=np.zeros_like(Jef), where=(N * Km) != 0)


# ============================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ============================================================

# Parámetros generales
dt = 0.0083                      # Paso de integración [s]
# dt = 0.001

x0_cart = [0.5, 0.1]   # Posición inicial cartesiana

q = inverse_kinematics_iterative(0.5,0.1,l1=PARAMS['A1'],l2=PARAMS['A2'],theta_init=(0.0,0.0),max_iter=10000)
print(q)
print(dp.fkine(q))
print(dp.plot(q))
input('holaa')

qd0  = np.zeros((2,))
qdmax = [0.05, 0.05]    # Velocidades máximas (m/s)


