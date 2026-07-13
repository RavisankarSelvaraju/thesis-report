import numpy as np

def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)

    K = skew(w / theta)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

def boxplus(x, dx):
    return {
        "R":  x["R"] @ exp_so3(dx[0:3]),
        "v":  x["v"] + dx[3:6],
        "p":  x["p"] + dx[6:9],
        "bg": x["bg"] + dx[9:12],
        "ba": x["ba"] + dx[12:15],
    }
   
def h(x, omega_measured, r_pc_body):
    omega_body = omega_measured - x["bg"]
    v_body = x["R"].T @ x["v"]

    return -(v_body + np.cross(omega_body, r_pc_body))

def jacobian_column(x, i, eps, omega_meas, r_pc_body):
    dx = np.zeros(15)
    dx[i] = eps[i]

    x_plus  = boxplus(x, dx)
    x_minus = boxplus(x, -dx)

# central difference approximation 
    return (h(x_plus, omega_meas, r_pc_body) -
            h(x_minus, omega_meas, r_pc_body)) / (2 * eps[i])


def numerical_jacobian(x, eps, omega_meas, r_pc_body):
    H = np.zeros((3, 15))
    for i in range(15):
        H[:, i] = jacobian_column(x, i, eps, omega_meas, r_pc_body)

    return H

def pprint(H, tol=1e-6):
    for row in H:
        print(" ".join(
            f"{0:4.0f}" if abs(val) < tol else f"{val:4.1f}"
            for val in row
        ))
        
# Calculate numerical jacobian for the test case
eps = np.array([
    1e-5,1e-5,1e-5,   # rotation
    1e-4,1e-4,1e-4,   # velocity
    1e-4,1e-4,1e-4,   # position
    1e-6,1e-6,1e-6,   # accel bias
    1e-6,1e-6,1e-6    # gyro bias
])
        
x = {
    "R": np.eye(3),
    "v": np.array([1.0, 0.5, -0.2]),
    "p": np.zeros(3),
    "ba": np.zeros(3),
    "bg": np.array([0.01, -0.02, 0.03])
}

omega_meas = np.array([0.2, -0.1, 0.3])
r_pc_body = np.array([0.5, 0.4, 0.8])

H = numerical_jacobian(x, eps, omega_meas, r_pc_body)

# expected jacobian matrix:
H_expected = np.array(np.zeros((3, 15)))

v_body = x["R"].T @ x["v"]
H_expected_rot = skew(v_body)
    
H_expected[:, 0:3] = - H_expected_rot  # rotation block
H_expected[:, 3:6] = - np.eye(3)         # velocity block
H_expected[:, 6:9] = np.zeros((3, 3))   # position block
H_expected[:, 9:12] = -skew(r_pc_body)  # gyro bias
H_expected[:, 12:15] = np.zeros((3, 3)) # accelerometer bias

print("Expected_Jacobian: \n")
pprint(H_expected)
print("\nComputed_Numerical_Jacobian: \n")
pprint(H)