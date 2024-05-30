import numpy as np

def unit_vector(vector):
    return vector/np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_radians = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


v1 = np.array([1,1])
v2 = np.array([0, 1, 0])

my_lst = [(1,2), (1,8), (3,5)]
array = np.array(my_lst)

coords = array - v1
print(v2[2])
