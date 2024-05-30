import numpy as np
import math

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.values = [x, y, z]

    @property
    def x(self):
        return self.values[0]

    @x.setter
    def x(self, value):
        self.values[0] = value

    @property
    def y(self):
        return self.values[1]

    @y.setter
    def y(self, value):
        self.values[1] = value

    @property
    def z(self):
        return self.values[2]

    @z.setter
    def z(self, value):
        self.values[2] = value

    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, list):
            return Vector3(self.x + other[0], self.y + other[1], self.z + other[2])
        else:
            raise TypeError("Operand must be of type Vector3 or list")

    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, list):
            return Vector3(self.x - other[0], self.y - other[1], self.z - other[2])
        else:
            raise TypeError("Operand must be of type Vector3 or list")

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def distance(self, other):
        if isinstance(other, Vector3):
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        elif isinstance(other, list):
            return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2 + (self.z - other[2])**2)
        else:
            raise TypeError("Operand must be of type Vector3 or list")

    @staticmethod
    def angle(v1, v2):
        dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        magnitudes = v1.magnitude() * v2.magnitude()
        angle_rad = math.acos(dot_product / magnitudes)
        return math.degrees(angle_rad)

    def __repr__(self):
        return f"{self.values}"
