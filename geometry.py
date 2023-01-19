import numpy as np

def line_intersection_point_matrix_solver(p1, dir1, p2, dir2):
    # Convert input to numpy arrays
    p1 = np.array(p1)
    dir1 = np.array(dir1)
    p2 = np.array(p2)
    dir2 = np.array(dir2)
    
    # Check if the lines are parallel (cross product is zero vector)
    cross_product = np.cross(dir1, dir2)
    if np.allclose(cross_product, np.zeros(3)):
        print("Error: Lines are parallel")
        return None
    
    # Calculate the intersection point by solving equation group p1 + x1 * dir1 = p2 + x2 * dir2
    A = np.array([dir1, -dir2]).T
    B = p2 - p1
    x = np.linalg.solve(A, B)
    intersection_point = p1 + x[0]*dir1
    
    return intersection_point

def line_intersection_point_algebra(p1, dir1, p2, dir2):
    """Calculates the intersection point of two lines defined by a point and a direction vector.
    p1, d1 define the first line, and p2, d2 define the second line.
    Returns None if the lines are parallel.
    """
    u = p1 - p2
    d = np.dot(dir1, dir2)
    if d == 0:
        return None
    else:
        s = np.dot(u, dir2) / d
        return p1 + dir1 * s