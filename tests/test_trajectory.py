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
    graficar_dinamica
)
from components.dp_model import dp
from components import pdFF_controller

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

x0_cart = [0.5657 , 0.0]   # Posición inicial cartesiana
q0 = np.array([-np.pi/4, np.pi/2])
qd0  = np.zeros((2,))
qdmax = [0.05, 0.05]    # Velocidades máximas (m/s)

# ============================================================
# GENERACIÓN DE TAYECTORIA
# ============================================================

# --------------------------------------
# 1. TRAZADO CARTESIANO CON mstraj
# --------------------------------------

# Puntos por los que debe pasar el efector (en XY)
waypoints = np.array([
    [0.5, 0.0],
    [0.0, 0.5],
    [0.5, 0.0]
])

# Trayectoria cartesiana:
traj = rtb.tools.trajectory.mstraj(
    waypoints,
    tacc=0.2,
    qdmax=qdmax,
    tsegment=None,
    q0=x0_cart,
    dt=dt
)

x = traj.s[:-1]
xd = traj.sd[:-1]
xdd = traj.sdd[:-1]

t = traj.t

# plt.figure()
# plt.plot(x[:,0], x[:,1], label='Trayectoria', linewidth=2.5)  # Línea más ancha
# plt.plot(waypoints[:,0], waypoints[:,1], 'ro', label='Waypoints', markersize=8)  # Puntos rojos más grandes
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Trayectoria cartesiana del efector')
# plt.grid(True)
# plt.axis('equal')
# plt.show()
# input('Presiona Enter para cerrar...')

# print(dp.fkine(q0))
# dp.plot(q0)
# input('Presiona Enter para cerrar...')


# --------------------------------------
# 2. CINEMÁTICA INVERSA (x -> q)
# --------------------------------------

print(q0)
q0_ik = q0
q = []
for i,xi in enumerate(x):
    # Pose actual del robot (con orientación incluida)
    Ti_actual = dp.fkine(q0_ik)

    # Extraer la rotación (3x3) de la pose actual
    R = Ti_actual.R

    t = np.array([xi[0], xi[1], 0])  # Convertimos la traslación a 3D
    Ti = spm.SE3.Rt(R, t)            # Ahora sí coincide con R (3x3) y t (3x1)

    print(Ti)
    print(dp.fkine(q0_ik))
    sol = dp.ikine_LM(Ti, q0=q0_ik,tol=1e-2,mask=[1, 1, 0, 0, 0, 0],)
    print(sol.success)
    print(sol.q)
    if not sol.success:
        time.sleep(2)
    q.append(sol.q)
    q0_ik = sol.q  # Usar esta para la próxima iteración
    print(i)
q = np.array(q)

# --------------------------------------
# 3. VELOCIDAD ARTICULAR (xd -> qd)
# --------------------------------------

qd = []
for i in range(len(q)):
    J = dp.jacob0(q[i])     # Jacobiano 6x2
    Jv = J[0:2, :]          # Tomamos solo la parte lineal (X, Y)
    qd_i = np.linalg.pinv(Jv) @ xd[i]  # Velocidad articular
    qd.append(qd_i)
qd = np.array(qd)

# --------------------------------------
# 4. ACELERACIÓN ARTICULAR (xdd -> qdd)
# --------------------------------------

# Usamos derivada numérica de qd para qdd
qdd = np.gradient(qd, dt, axis=0)

q_ref = q
qd_ref = qd
qdd_ref = qdd


# ============================================================
# GENERACIÓN DE CONTROLADOR
# ============================================================

# Crear controlador PD
controller = pdFF_controller.make_pdFF_controller(
    dp=dp,
    params=PARAMS,
    dt=dt,
    kp=kp,
    kd=kd,
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
print("== SHAPES ANTES DE ITERAR ==")
print(f"q_ref.shape      : {q_ref.shape}")
print(f"qd_ref.shape      : {qd_ref.shape}")
print(f"qdd_ref.shape      : {qdd_ref.shape}")
print(f"tg.t.shape       : {tg.t.shape}")
print(f"tg.q.shape       : {tg.q.shape}")
print(f"tg.qd.shape      : {tg.qd.shape}")  
print(f"Qm_vec.shape     : {Qm_vec.shape}")

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
graficar_dinamica(dp=dp, q=tg.q, qd=tg.qd, t=tg.t, tau = Qm_vec, q_ref=q_ref, qd_ref=qd_ref)
plt.show()