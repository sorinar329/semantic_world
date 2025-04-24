import enum


class Axis(int, enum.Enum):
    """
    Enum for axis identifiers in 3D.
    """
    X = 0
    Y = 1
    Z = 2

    def to_list(self):
        return [1 if i == self.value else 0 for i in range(3)]

class JointType(int, enum.Enum):
    """
    Enum for readable joint types.
    """
    REVOLUTE = 0
    PRISMATIC = 1
    SPHERICAL = 2
    PLANAR = 3
    FIXED = 4
    UNKNOWN = 5
    CONTINUOUS = 6
    FLOATING = 7