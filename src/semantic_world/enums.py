from enum import Enum


class Axis(int, Enum):
    """
    Enum for axis identifiers in 3D.
    """
    X = 0
    Y = 1
    Z = 2

    def to_list(self):
        return [1 if i == self.value else 0 for i in range(3)]


class JointType(int, Enum):
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


class WorldMode(Enum):
    """
    Enum for the different modes of the world.
    """
    GUI = "GUI"
    DIRECT = "DIRECT"


class DescriptionType(Enum):
    URDF = "urdf"
    MJCF = "mjcf"

    def get_file_extension(self):
        if self == DescriptionType.URDF:
            return ".urdf"
        elif self == DescriptionType.MJCF:
            return ".xml"
        else:
            raise ValueError("Unknown description type")