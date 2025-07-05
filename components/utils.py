import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def generar_video_trayectoria(dp,tg_q, nombre_archivo='robot_trajectory.mp4',
                              export_fps=60, sim_dt=1e-3,
                              l1=1.0, l2=1.0):
    """
    Genera un video animado de la trayectoria de un robot planar de 2 DOF,
    incluyendo una traza del extremo del péndulo.

    Parámetros:
        tg_q           : np.ndarray de forma (N, 2) con las configuraciones articulares
        nombre_archivo : str, nombre del archivo de salida (formato .mp4)
        export_fps     : int, cuadros por segundo del video
        sim_dt         : float, paso de integración usado en la simulación
        l1, l2         : float, longitudes de los eslabones del robot
    """
    skip = int(1 / (export_fps * sim_dt))
    
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-', lw=4, label='Robot')
    trace_line, = ax.plot([], [], 'r--', lw=1.5, alpha=0.7, label='Traza')
    
    ax.set_xlim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
    ax.set_ylim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_title("Animación de trayectoria del robot")
    ax.legend()

    trace_x = []
    trace_y = []

    def init():
        line.set_data([], [])
        trace_line.set_data([], [])
        return line, trace_line

    def update(i):
        q = tg_q[i]
        joints_T = dp.fkine_all(q)
        T1 = np.array(joints_T[0])
        p1 = [T1[0, 3], T1[1, 3]]
        T2 = np.array(joints_T[1])
        p2 = [T2[0, 3], T2[1, 3]]
        T3 = np.array(joints_T[2])
        p3 = [T3[0, 3], T3[1, 3]]

        points = np.array([p1, p2, p3])

        # Actualiza el robot
        line.set_data(points[:, 0], points[:, 1])
        
        # Agrega la nueva posición del extremo
        trace_x.append(points[-1, 0])
        trace_y.append(points[-1, 1])
        trace_line.set_data(trace_x, trace_y)
        
        return line, trace_line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, len(tg_q), skip),
        init_func=init,
        blit=True,
        interval=1000 / export_fps
    )

    ani.save(nombre_archivo, writer='ffmpeg', fps=export_fps)
    print(f"✅ Video guardado como '{nombre_archivo}'")

def generar_escalon_suave(T_total, Ts, t_subida, ref_value, q0=0.0):
    """
    Genera una señal escalón con subida lineal desde q0 hasta ref_value.

    Parámetros:
    - T_total   : Tiempo total de la señal [s]
    - Ts        : Tiempo de muestreo [s]
    - t_subida  : Tiempo de subida lineal [s]
    - ref_value : Valor final del escalón
    - q0        : Valor inicial de la señal (por defecto 0.0)

    Retorna:
    - t   : Vector de tiempo
    - u   : Señal del escalón suavizado
    """
    t = np.arange(0, T_total, Ts)
    u = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < t_subida:
            u[i] = q0 + (ref_value - q0) * (ti / t_subida)
        else:
            u[i] = ref_value

    return t, u

def valor_referencia_en_t(t_query, t_subida, ref_value):
    """
    Retorna el valor de la señal escalón suave en el tiempo t_query.

    Parámetros:
    - t_query   : Tiempo en el que se desea conocer el valor [s]
    - t_subida  : Tiempo de subida del escalón [s]
    - ref_value : Valor final del escalón

    Retorna:
    - Valor de la señal en t_query
    """
    if t_query < 0:
        return 0
    elif t_query < t_subida:
        return (ref_value / t_subida) * t_query
    else:
        return ref_value
    
def graficar_dinamica(dp, q, qd, t, q_ref, tau):
    """
    Grafica en una sola figura:
    - Posiciones articulares con sus referencias
    - Velocidades articulares
    - Trayectoria del extremo del robot vs referencia
    - Torques articulares

    Parámetros:
    - dp     : robot DHRobot
    - q      : (n x 2) posiciones articulares en el tiempo
    - qd     : (n x 2) velocidades articulares
    - t      : (n,) vector de tiempo
    - q_ref  : (n x 2) trayectoria articular de referencia
    - tau    : (n x 2) torques articulares en el tiempo
    """

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Posiciones articulares vs referencia
    axs[0, 0].plot(t, q[:, 0], label='q1', color='blue', linewidth=2)
    axs[0, 0].plot(t, q[:, 1], label='q2', color='orange', linewidth=2)
    axs[0, 0].plot(t, q_ref[:, 0], '--', label='q1 ref', color='blue', alpha=0.5, linewidth=2)
    axs[0, 0].plot(t, q_ref[:, 1], '--', label='q2 ref', color='orange', alpha=0.5, linewidth=2)
    axs[0, 0].set_title('Posiciones articulares')
    axs[0, 0].set_xlabel('Tiempo [s]')
    axs[0, 0].set_ylabel('Ángulo [rad]')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 2. Velocidades articulares
    axs[0, 1].plot(t, qd[:, 0], label='qd1', color='blue', linewidth=2)
    axs[0, 1].plot(t, qd[:, 1], label='qd2', color='orange', linewidth=2)
    axs[0, 1].set_title('Velocidades articulares')
    axs[0, 1].set_xlabel('Tiempo [s]')
    axs[0, 1].set_ylabel('Velocidad [rad/s]')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # 3. Trayectoria cartesiana
    x_real, y_real = [], []
    x_ref, y_ref = [], []
    for qi, qi_ref in zip(q, q_ref):
        T_real = dp.fkine(qi)
        T_ref = dp.fkine(qi_ref)
        x_real.append(T_real.t[0])
        y_real.append(T_real.t[1])
        x_ref.append(T_ref.t[0])
        y_ref.append(T_ref.t[1])

    axs[1, 0].plot(x_real, y_real, label='Trayectoria real', color='green', linewidth=2)
    axs[1, 0].plot(x_ref, y_ref, '--', label='Trayectoria ref', color='red', linewidth=2)
    axs[1, 0].set_title('Trayectoria del extremo del péndulo')
    axs[1, 0].set_xlabel('X [m]')
    axs[1, 0].set_ylabel('Y [m]')
    axs[1, 0].axis('equal')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # 4. Torques articulares
    axs[1, 1].plot(t, tau[:, 0], label='τ1', color='blue', linewidth=2)
    axs[1, 1].plot(t, tau[:, 1], label='τ2', color='orange', linewidth=2)
    axs[1, 1].set_title('Torques articulares')
    axs[1, 1].set_xlabel('Tiempo [s]')
    axs[1, 1].set_ylabel('Torque [Nm]')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
