import numpy as np

def pd_control(robot, t, q, qd, *, dp, params, kp, kd, dt, q_ref, qd_ref, qdd_ref,tau_p = None, t_p = None):
    """       
    Controlador PD (Proporcional-Derivativo) para seguimiento de trayectoria articular
    de un sistema de doble péndulo definido por un modelo DH.

    Parámetros:
    ----------
    robot : object
        Objeto del robot (no se usa directamente, pero se incluye para extensibilidad).
    t : float
        Tiempo actual en segundos.
    q : ndarray
        Vector de posiciones articulares actuales (dimensión: [n_articulaciones]).
    qd : ndarray
        Vector de velocidades articulares actuales (dimensión: [n_articulaciones]).
    dp : object
        Modelo del doble péndulo creado con Denavit-Hartenberg.
    params : dict
        Diccionario o estructura con parámetros físicos del doble péndulo.
    kp : ndarray
        Matriz de ganancias proporcionales (dimensión: [n, n]).
    kd : ndarray
        Matriz de ganancias derivativas (dimensión: [n, n]).
    q_ref : ndarray
        Trayectoria de referencia articular (dimensión: [pasos, n_articulaciones]).
    dt : float
        Intervalo de muestreo o paso de simulación en segundos.

    Retorna:
    -------
    Q : ndarray
        Vector de torques de control calculados mediante la ley PD (dimensión: [n_articulaciones]).
    """
    step = round(t / dt)
    step = min(step, q_ref.shape[0] - 1)
    q_ref_i = q_ref[step]

    N = np.array([params['N1'], params['N2']]) 
    KM = np.array([params['KM1'], params['KM2']])

    # Aquí podrías usar dp y params si quisieras hacer un control más avanzado
    # Por ahora, sólo aplicamos el control PD clásico
    U = np.dot(kp, (q_ref_i - q)) - np.dot(kd, qd)

    Q = N * KM * U 

    # Perturbacion:
    if tau_p and t_p:
        tau_p1 = tau_p[0] if t_p[0]<t<t_p[1] else 0
        tau_p2 = tau_p[1] if t_p[0]<t<t_p[1] else 0

        Q = Q + N * np.array([tau_p1,tau_p2])

 
    return Q


def make_pd_controller(dp, params, dt, kp, kd, q_ref,qd_ref = None, qdd_ref = None, tau_p=None,t_p=None):
    """
    Crea una función de control PD preconfigurada con parámetros fijos para el doble péndulo.

    Parámetros:
    ----------
    dp : object
        Modelo del doble péndulo creado con Denavit-Hartenberg.
    params : dict
        Parámetros físicos del doble péndulo.
    dt : float
        Intervalo de muestreo o paso de simulación en segundos.
    kp : ndarray
        Matriz de ganancias proporcionales.
    kd : ndarray
        Matriz de ganancias derivativas.
    q_ref : ndarray
        Trayectoria articular de referencia.

    Retorna:
    -------
    controller : callable
        Función de control con la interfaz estándar:
        `controller(robot, t, q, qd)` → ndarray de torques
    """
    return lambda robot, t, q, qd: pd_control(
        robot, t, q, qd, dp=dp, params=params, dt=dt, kp=kp, kd=kd, q_ref=q_ref, qd_ref=qd_ref, qdd_ref=qdd_ref, tau_p=tau_p,t_p=t_p
    )
