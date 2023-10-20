import autograd.numpy as np
from autoptim import minimize

def catenary_func(x, c, xmin, zmin):
    """
    Catenary function used internally for curve fitting.
    """
    return c*np.cosh((x-xmin)/c)-c+zmin

def euler_to_matrix(rph):
    """
    Function to convert euler (roll pitch yaw) angles to a rotation matrix.
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rph[0]), -np.sin(rph[0])],
                   [0, np.sin(rph[0]), np.cos(rph[0])]])
    
    Ry = np.array([[np.cos(rph[1]), 0, np.sin(rph[1])],
                   [0, 1, 0],
                   [-np.sin(rph[1]), 0, np.cos(rph[1])]])
    
    Rz = np.array([[np.cos(rph[2]), -np.sin(rph[2]), 0],
                   [np.sin(rph[2]), np.cos(rph[2]), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def catenary_error(pose: np.ndarray, points: np.ndarray):
    """
    Cost function for fitting a catenary equation
    """
    R = euler_to_matrix(pose[3:6])
    t = pose[:3]
    transformed_points = (R @ points) + t[:,np.newaxis]
    y_error = transformed_points[1, :]
    z_error = transformed_points[2, :] - catenary_func(transformed_points[0,:], pose[6], 0, 0)
    return np.sum(np.power(y_error, 2) + np.power(z_error, 2))

def generate_catenary_points():
    """
    Function to generate true catenary points.
    """
    x = np.linspace(0, 1, 10)
    z = catenary_func(x, 10, 0, 0)
    y = np.zeros(10)
    return np.vstack([x, y, z])

def main():
    rng = np.random.default_rng()
    # Generate exact catenary points then apply noise.
    points = generate_catenary_points() + rng.standard_normal((3, 10)) * 0.01
    # Paramerts are x, y, z, r, p, h, c (catenary variable).
    parameters = rng.standard_normal(7) * 0.2 
    parameters[6] = -10
    # Extract pose and apply it to points.
    R = euler_to_matrix(parameters[3:6])
    t = parameters[:3]
    points = R @ points + t[:, np.newaxis]

    initial = np.zeros(7)
    initial[6] = 80
    # Minimize using autodiff.
    p_min, _ = minimize(lambda p: catenary_error(p, points), initial)
    print(f"Estimated: {p_min}\n True: {parameters}\n diff {p_min + parameters}")

if __name__ == '__main__':
    main()
