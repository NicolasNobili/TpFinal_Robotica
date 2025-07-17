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
    
    ax.set_xlim(-l1 - l2 - 0.2, l1 + l2 + 0.2)
    ax.set_ylim(-l1 - l2 - 0.2, l1 + l2 + 0.2)
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
    

def graficar_dinamica(dp, q, qd, t, tau, q_ref=None, qd_ref=None, ):
    """
    Grafica en una sola figura:
    - Posiciones articulares con sus referencias (si se pasan)
    - Velocidades articulares con sus referencias (si se pasan)
    - Trayectoria del extremo del robot vs referencia (si se pasa q_ref)
    - Torques articulares
    - Error cuadrático medio (RMSE) de q vs q_ref (si se pasa q_ref)

    Parámetros:
    - dp      : robot DHRobot
    - q       : (n x 2) posiciones articulares en el tiempo
    - qd      : (n x 2) velocidades articulares
    - t       : (n,) vector de tiempo
    - q_ref   : (n x 2) trayectoria articular de referencia (opcional)
    - qd_ref  : (n x 2) velocidades articulares de referencia (opcional)
    - tau     : (n x 2) torques articulares en el tiempo (obligatorio)
    """

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2)

    ax_pos = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_tau = fig.add_subplot(gs[1, 1])
    ax_rmse = fig.add_subplot(gs[2, :])  # ocupa ambas columnas

    # 1. Posiciones articulares
    ax_pos.plot(t, q[:, 0], label='q1', color='blue', linewidth=2)
    ax_pos.plot(t, q[:, 1], label='q2', color='orange', linewidth=2)

    if q_ref is not None:
        ax_pos.plot(t, q_ref[:, 0], '--', label='q1 ref', color='blue', alpha=0.5, linewidth=2)
        ax_pos.plot(t, q_ref[:, 1], '--', label='q2 ref', color='orange', alpha=0.5, linewidth=2)

    ax_pos.set_title('Posiciones articulares')
    ax_pos.set_xlabel('Tiempo [s]')
    ax_pos.set_ylabel('Ángulo [rad]')
    ax_pos.grid(True)
    ax_pos.legend()

    # 2. Velocidades articulares
    ax_vel.plot(t, qd[:, 0], label='qd1', color='blue', linewidth=2)
    ax_vel.plot(t, qd[:, 1], label='qd2', color='orange', linewidth=2)

    if qd_ref is not None:
        ax_vel.plot(t, qd_ref[:, 0], '--', label='qd1 ref', color='blue', alpha=0.5, linewidth=2)
        ax_vel.plot(t, qd_ref[:, 1], '--', label='qd2 ref', color='orange', alpha=0.5, linewidth=2)

    ax_vel.set_title('Velocidades articulares')
    ax_vel.set_xlabel('Tiempo [s]')
    ax_vel.set_ylabel('Velocidad [rad/s]')
    ax_vel.grid(True)
    ax_vel.legend()

    # 3. Trayectoria cartesiana
    x_real, y_real = [], []
    x_ref, y_ref = [], []

    for i in range(len(q)):
        T_real = dp.fkine(q[i])
        x_real.append(T_real.t[0])
        y_real.append(T_real.t[1])
        if q_ref is not None:
            T_r = dp.fkine(q_ref[i])
            x_ref.append(T_r.t[0])
            y_ref.append(T_r.t[1])

    ax_traj.plot(x_real, y_real, label='Trayectoria real', color='green', linewidth=2)
    if q_ref is not None:
        ax_traj.plot(x_ref, y_ref, '--', label='Trayectoria ref', color='red', linewidth=2)

    ax_traj.set_title('Trayectoria del extremo del robot')
    ax_traj.set_xlabel('X [m]')
    ax_traj.set_ylabel('Y [m]')
    ax_traj.axis('equal')
    ax_traj.grid(True)
    ax_traj.legend()

    # 4. Torques articulares 
    ax_tau.plot(t,tau[:, 0], label='τ1', color='blue', linewidth=2)
    ax_tau.plot(t,tau[:, 1], label='τ2', color='orange', linewidth=2)
    ax_tau.set_title('Torques de Actuadores')
    ax_tau.set_xlabel('Tiempo [s]')
    ax_tau.set_ylabel('Torque [Nm]')
    ax_tau.grid(True)
    ax_tau.legend()

    # # 5. Error cuadrático medio (fila completa)
    # if q_ref is not None:
    #     error_sq_q1 = np.abs(q[:, 0] - q_ref[:, 0])
    #     error_sq_q2 = np.abs(q[:, 1] - q_ref[:, 1])
    #     # rmse_q1 = np.sqrt(np.mean(error_sq_q1))
    #     # rmse_q2 = np.sqrt(np.mean(error_sq_q2))

    #     ax_rmse.plot(t, error_sq_q1, label='Error q1', color='blue', linewidth=2)
    #     ax_rmse.plot(t, error_sq_q2, label='Error q2', color='orange', linewidth=2)
    #     # ax_rmse.axhline(rmse_q1**2, color='blue', linestyle='--', alpha=0.5, label=f'RMSE q1: {rmse_q1:.4f}')
    #     # ax_rmse.axhline(rmse_q2**2, color='orange', linestyle='--', alpha=0.5, label=f'RMSE q2: {rmse_q2:.4f}')
    #     ax_rmse.set_title('Error Absoluto entre q y q_ref')
    #     ax_rmse.set_xlabel('Tiempo [s]')
    #     ax_rmse.set_ylabel('Error² [rad²]')
    #     ax_rmse.grid(True)
    #     ax_rmse.legend()
    # else:
    #     ax_rmse.set_title('Error cuadrático entre q y q_ref')
    #     ax_rmse.text(0.5, 0.5, 'Sin q_ref', ha='center', va='center', fontsize=12)
    #     ax_rmse.set_xticks([])
    #     ax_rmse.set_yticks([])
    #     ax_rmse.grid(False)

# 5. Error cartesiano en mm
    if q_ref is not None:
        error_xy = []

        for i in range(len(q)):
            T_real = dp.fkine(q[i])
            T_ref = dp.fkine(q_ref[i])
            pos_real = T_real.t[:2]
            pos_ref = T_ref.t[:2]
            err = np.linalg.norm(pos_real - pos_ref) * 1000  # de metros a milímetros
            error_xy.append(err)

        ax_rmse.plot(t, error_xy, label='Error XY', color='purple', linewidth=2)
        ax_rmse.set_title('Error cartesiano entre trayectoria real y referencia')
        ax_rmse.set_xlabel('Tiempo [s]')
        ax_rmse.set_ylabel('Error [mm]')
        ax_rmse.grid(True)
        ax_rmse.legend()
    else:
        ax_rmse.set_title('Error cartesiano entre trayectoria real y referencia')
        ax_rmse.text(0.5, 0.5, 'Sin q_ref', ha='center', va='center', fontsize=12)
        ax_rmse.set_xticks([])
        ax_rmse.set_yticks([])
        ax_rmse.grid(False)

    plt.tight_layout()
    plt.show()


def graficar_trayectoria(traj):
    """
    Grafica la trayectoria (posición, velocidad, aceleración) de un objeto Trajectory.
    
    Parámetros:
    traj: objeto de tipo Trajectory
    """
    t = traj.t
    q = traj.s
    qd = traj.sd
    qdd = traj.sdd

    # Verificar dimensiones
    num_subplots = 3
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 8), sharex=True)

    axs[0].set_title('Posición (q)')
    axs[1].set_title('Velocidad (qd)')
    axs[2].set_title('Aceleración (qdd)')

    def plot_data(ax, t, data, ylabel):
        if data is None:
            ax.set_visible(False)
            return
        data = data if data.ndim > 1 else data[:, None]
        for i in range(data.shape[1]):
            ax.plot(t, data[:, i], label=f'Joint {i+1}', linewidth=2.5)  # Línea más gruesa
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plot_data(axs[0], t, q, 'q')
    plot_data(axs[1], t, qd, 'qd')
    plot_data(axs[2], t, qdd, 'qdd')

    axs[-1].set_xlabel('Tiempo' if traj.istime else 'Step')
    plt.tight_layout()
    plt.show()

def generar_vertices_poligono(radio, lados, angulo_inicial=0.0):
    """
    Genera las coordenadas de los vértices de un polígono regular inscrito en una circunferencia.

    Parámetros:
    - radio: float. Radio de la circunferencia circunscrita.
    - lados: int. Número de lados del polígono.
    - angulo_inicial: float. Ángulo inicial en radianes (opcional).

    Retorna:
    - np.ndarray de forma (lados, 2) con coordenadas (x, y) de los vértices.
    """
    angulos = np.linspace(0, 2 * np.pi, lados, endpoint=False) + angulo_inicial
    x = radio * np.cos(angulos)
    y = radio * np.sin(angulos)
    return np.stack((x, y), axis=-1)  # Shape: (lados, 2)
