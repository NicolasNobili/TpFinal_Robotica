import numpy as np
import math
import time
import spatialmath as sm
import roboticstoolbox as rtb


def generate_joint_trajectory(
    dp,
    viapoints,
    dt,
    tacc,
    qdmax=None,
    tsegment=None,
    q0=None,
    qd0=None,
    qdf=None,
    verbose=False
):
    """
    Generates a joint-space trajectory.

    Parameters:
    - dp: Robot (type ERobot or SerialLink with ikine_LM and jacob0)
    - viapoints: np.ndarray, intermediate Cartesian points (x, y)
    - dt: time step
    - tacc: acceleration time
    - qdmax: maximum joint velocities (per axis)
    - tsegment: segment durations (optional)
    - q0: initial Cartesian position (x, y)
    - qd0: initial velocity (optional)
    - qdf: final velocity (optional)
    - verbose: print information during processing

    Returns:
    - traj: joint trajectory object
    - q_ref: joint position trajectory (q)
    - qd_ref: joint velocity trajectory (qd)
    - qdd_ref: joint acceleration trajectory (qdd)
    """
    
    # --- 1. Generate joint trajectory ---
    traj = rtb.tools.trajectory.mstraj(
        viapoints=viapoints,
        dt=dt,
        tacc=tacc,
        qdmax=qdmax,
        tsegment=tsegment,
        q0=q0,
        qd0=qd0,
        qdf=qdf,
        verbose=verbose
    )

    q = traj.s[:-1]
    qd = traj.sd[:-1]
    qdd = traj.sdd[:-1]

    q[:, 0] = np.unwrap(q[:, 0])  # Para q1
    q[:, 1] = np.unwrap(q[:, 1])  # Para q2

    return traj, q, qd, qdd



def generate_cartesian_trajectory(
    dp,
    viapoints,
    dt,
    tacc,
    qdmax=None,
    tsegment=None,
    x0=None,
    q0=None,
    qd0=None,
    qdf=None,
    verbose=False
):
    """
    Generates a Cartesian trajectory and computes the corresponding joint-space trajectory.

    Parameters:
    - dp: Robot (type ERobot or SerialLink with ikine_LM and jacob0)
    - viapoints: np.ndarray, intermediate Cartesian points (x, y)
    - dt: time step
    - tacc: acceleration time
    - qdmax: maximum velocities (per axis)
    - tsegment: segment durations (optional)
    - q0: initial Cartesian position (x, y)
    - qd0: initial velocity (optional)
    - qdf: final velocity (optional)
    - verbose: print information during processing

    Returns:
    - traj: Cartesian trajectory object
    - q_ref: joint position trajectory (q)
    - qd_ref: joint velocity trajectory (qd)
    - qdd_ref: joint acceleration trajectory (qdd)
    """
    
    # --- 1. Generate Cartesian trajectory ---
    traj = rtb.tools.trajectory.mstraj(
        viapoints=viapoints,
        dt=dt,
        tacc=tacc,
        qdmax=qdmax,
        tsegment=tsegment,
        q0=x0,
        qd0=qd0,
        qdf=qdf,
        verbose=verbose
    )

    x = traj.s[:-1]
    xd = traj.sd[:-1]
    xdd = traj.sdd[:-1]

    # --- 2. Inverse Kinematics ---
    q0_ik = q0
    q = []
    for i, xi in enumerate(x):
        theta = math.atan2(xi[1], xi[0])
        Ti = sm.SE3(xi[0], xi[1], 0) * sm.SE3.Rz(theta)
        sol = dp.ikine_LM(Ti, q0=q0_ik, tol=1e-5, mask=[1, 1, 0, 0, 0, 0])
        if not sol.success and verbose:
            print(f"[WARN] IK failed at step {i}")
            time.sleep(1)
        q.append(sol.q)
        q0_ik = sol.q
    q = np.array(q)
    
    q[:, 0] = np.unwrap(q[:, 0])  # Para q1
    q[:, 1] = np.unwrap(q[:, 1])  # Para q2


    # --- 3. Joint Velocity ---
    qd = []
    for i in range(len(q)):
        J = dp.jacob0(q[i])
        Jv = J[0:2, :]  # Consider only XY motion
        qd_i = np.linalg.pinv(Jv) @ xd[i]
        qd.append(qd_i)
    qd = np.array(qd)

    # --- 4. Joint Acceleration ---
    qdd = np.gradient(qd, dt, axis=0)

    return traj, q, qd, qdd
