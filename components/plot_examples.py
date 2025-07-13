import numpy as np

def spiral_waypoints(num_points=200, max_radius=0.45):
    theta = np.linspace(0, 4 * np.pi, num_points)
    r = np.linspace(0, max_radius, num_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def star_waypoints():
    return np.array([
        [0.0,  0.5],
        [0.15, 0.15],
        [0.5,  0.15],
        [0.2, -0.05],
        [0.3, -0.5],
        [0.0, -0.2],
        [-0.3, -0.5],
        [-0.2, -0.05],
        [-0.5, 0.15],
        [-0.15, 0.15],
        [0.0, 0.5]
    ])

def heart_waypoints(num_points=300):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = 0.5 * 16 * (np.sin(t)**3) / 32
    y = 0.5 * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) / 32
    return np.column_stack((x, y))

def fish_waypoints():
    body = np.array([
        [-0.4, 0], [-0.3, 0.1], [0, 0.2], [0.3, 0.1], [0.4, 0],
        [0.3, -0.1], [0, -0.2], [-0.3, -0.1], [-0.4, 0]
    ])
    tail = np.array([
        [-0.4, 0], [-0.5, 0.15], [-0.5, -0.15], [-0.4, 0]
    ])
    return np.vstack((body, tail))

def puzzle_piece_waypoints():
    return np.array([
        [-0.4, 0.4], [0.0, 0.4], [0.0, 0.5], [0.1, 0.5], [0.1, 0.4], [0.4, 0.4],
        [0.4, 0.0], [0.5, 0.0], [0.5, -0.1], [0.4, -0.1], [0.4, -0.4],
        [0.0, -0.4], [0.0, -0.5], [-0.1, -0.5], [-0.1, -0.4], [-0.4, -0.4],
        [-0.4, 0.0], [-0.5, 0.0], [-0.5, 0.1], [-0.4, 0.1], [-0.4, 0.4]
    ])


def butterfly_waypoints(num_points=400):
    t = np.linspace(0, 12 * np.pi, num_points)
    x = 0.5 * np.sin(t) * (np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5) / 10
    y = 0.5 * np.cos(t) * (np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5) / 10
    return np.column_stack((x, y))

def cat_silhouette_waypoints():
    # Simplified cartoonish cat face with ears
    return np.array([
        [-0.3,  0.4], [-0.2,  0.5], [-0.1,  0.4], [0.0, 0.5], [0.1, 0.4],
        [0.2, 0.5], [0.3, 0.4], [0.4, 0.3], [0.45, 0.2], [0.48, 0.1],
        [0.5, -0.1], [0.45, -0.2], [0.4, -0.3], [0.3, -0.4], [0.1, -0.5],
        [0.0, -0.48], [-0.1, -0.5], [-0.3, -0.4], [-0.4, -0.3], [-0.45, -0.2],
        [-0.5, -0.1], [-0.48, 0.1], [-0.45, 0.2], [-0.4, 0.3], [-0.3, 0.4]
    ])

def flower_waypoints(num_petals=6, num_points=300):
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = 0.3 + 0.15 * np.sin(num_petals * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def rocket_waypoints():
    return np.array([
        [0.0, 0.5], [0.1, 0.3], [0.05, 0.3], [0.05, -0.3],
        [0.2, -0.3], [0.2, -0.5], [0.0, -0.4], [-0.2, -0.5],
        [-0.2, -0.3], [-0.05, -0.3], [-0.05, 0.3], [-0.1, 0.3], [0.0, 0.5]
    ])

def bunny_ears_waypoints():
    return np.array([
        [-0.2, -0.4], [-0.25, 0.2], [-0.2, 0.5], [-0.15, 0.2], [-0.1, -0.1],
        [0.1, -0.1], [0.15, 0.2], [0.2, 0.5], [0.25, 0.2], [0.2, -0.4],
        [0.0, -0.5], [-0.2, -0.4]
    ])


def batman_logo_waypoints():
    # Puntos aproximados para la silueta del logo de Batman estilo Nolan
    points = np.array([
        [0.0, 0.4],    # centro orejas arriba
        [0.03, 0.3], [0.1, 0.3], [0.15, 0.4], [0.18, 0.5],  # ala derecha arriba
        [0.25, 0.4], [0.4, 0.3], [0.6, 0.0], [0.55, -0.1], [0.45, -0.15], # ala derecha exterior
        [0.5, -0.3], [0.4, -0.35], [0.3, -0.3], [0.2, -0.35], [0.1, -0.5],  # ala derecha abajo
        [0.0, -0.4],  # punta abajo centro
        [-0.1, -0.5], [-0.2, -0.35], [-0.3, -0.3], [-0.4, -0.35], [-0.5, -0.3], # ala izquierda abajo
        [-0.45, -0.15], [-0.55, -0.1], [-0.6, 0.0], [-0.4, 0.3], [-0.25, 0.4], # ala izquierda exterior
        [-0.18, 0.5], [-0.15, 0.4], [-0.1, 0.3], [-0.03, 0.3], [0.0, 0.4]   # ala izquierda arriba y cierre
    ])
    return points