import numpy as np

def pdFF_control(robot, t, q, qd, *, dp, params, kp, kd, dt, q_ref, qd_ref, qdd_ref,tau_p = None, t_p = None):
    """
    Controlador PD para seguimiento de trayectoria articular con compensación de gravedad.

    Parámetros:
    -----------
    robot : object
        Objeto robot (no se utiliza directamente aquí).
    t : float
        Tiempo actual en segundos.
    q : ndarray
        Vector de posiciones articulares actuales.
    qd : ndarray
        Vector de velocidades articulares actuales.
    dp : object
        Modelo del doble péndulo.
    params : dict
        Parámetros físicos del sistema (incluye N1, N2, KM1, KM2).
    kp : ndarray
        Matriz de ganancias proporcionales.
    kd : ndarray
        Matriz de ganancias derivativas.
    q_ref : ndarray
        Trayectoria de referencia articular.
    dt : float
        Intervalo de muestreo.

    Retorna:
    --------
    Q_act : ndarray
        Vector de torques de control.
    """
    step = round(t / dt)
    step = min(step, q_ref.shape[0] - 1)
    q_ref_i = q_ref[step]
    qd_ref_i = qd_ref[step]
    qdd_ref_i = qdd_ref[step]

    G = dp.rne(q_ref_i,qd_ref_i,qdd_ref_i,gravity=np.array([0,-9.8,0]))  # Torque gravitacional en la postura de referencia

    N = np.array([params['N1'], params['N2']]) 
    KM = np.array([params['KM1'], params['KM2']])

    error = q_ref_i - q
    deriv = -qd

    U_act = np.dot(kp, error) + np.dot(kd, deriv) + (1 / (N * KM)) * G

    Q = N * KM * U_act

    # Perturbacion:
    if tau_p and t_p:
        tau_p1 = tau_p[0] if t_p[0]<t<t_p[1] else 0
        tau_p2 = tau_p[1] if t_p[0]<t<t_p[1] else 0

        Q = Q + N * np.array([tau_p1,tau_p2])

    return Q


def make_pdFF_controller(dp, params, dt, kp, kd, q_ref, qd_ref, qdd_ref, tau_p=None,t_p=None):
    """
    Crea una función de control PD con compensación de gravedad preconfigurada.

    Parámetros:
    -----------
    dp : object
        Modelo del doble péndulo.
    params : dict
        Parámetros físicos del sistema.
    dt : float
        Intervalo de muestreo.
    kp : ndarray
        Matriz de ganancias proporcionales.
    kd : ndarray
        Matriz de ganancias derivativas.
    q_ref : ndarray
        Trayectoria de referencia articular.

    Retorna:
    --------
    controller : callable
        Función de control con interfaz (robot, t, q, qd) → torques.
    """
    return lambda robot, t, q, qd: pdFF_control(
        robot, t, q, qd, dp=dp, params=params, dt=dt, kp=kp, kd=kd, q_ref=q_ref, qd_ref=qd_ref, qdd_ref=qdd_ref, tau_p=tau_p,t_p=t_p
    )
