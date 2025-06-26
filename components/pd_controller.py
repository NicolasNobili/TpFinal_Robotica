import numpy as np

def pd_control(robot, t, q, qd, *, dp, params, kp, kd, q_ref, dt):
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

    # Aquí podrías usar dp y params si quisieras hacer un control más avanzado
    # Por ahora, sólo aplicamos el control PD clásico
    Q = np.dot(kp, (q_ref_i - q)) - np.dot(kd, qd)
    return Q


def make_pd_controller(dp, params, dt, kp, kd, q_ref):
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
        robot, t, q, qd, dp=dp, params=params, dt=dt, kp=kp, kd=kd, q_ref=q_ref
    )
