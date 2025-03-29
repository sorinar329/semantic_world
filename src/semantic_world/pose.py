from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass, field
from typing_extensions import Self, List

import numpy as np


@dataclass
class Vector3:
    """
    A 3D vector with x, y and z coordinates.
    """

    x: float = 0
    y: float = 0
    z: float = 0

    def euclidean_distance(self, other: Self) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5

    def to_list(self):
        return [self.x, self.y, self.z]

    def round(self, decimals: int = 4):
        self.x = round(self.x, decimals)
        self.y = round(self.y, decimals)
        self.z = round(self.z, decimals)

    def almost_equal(self, other: Self, tolerance: float = 1e-6) -> bool:
        return bool(np.isclose(np.array(self.to_list()), np.array(other.to_list()), atol=tolerance).all())

    def vector_to_position(self, other: Self) -> Vector3:
        return other - self

    def to_numpy(self):
        return np.array(self.to_list())

    def __add__(self, other: Self) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> Vector3:
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float) -> Vector3:
        return Vector3(self.x * other, self.y * other, self.z * other)


@dataclass
class Quaternion:
    """
    A quaternion with x, y, z and w components.
    """

    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1

    def __post_init__(self):
        self.normalize()

    def normalize(self):
        """
        Normalize the quaternion in-place.
        """
        norm = (self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2) ** 0.5
        self.x /= norm
        self.y /= norm
        self.z /= norm
        self.w /= norm

    def to_list(self):
        return [self.x, self.y, self.z, self.w]

    def to_numpy(self):
        return np.array(self.to_list())

    def round(self, decimals: int = 4):
        self.x = round(self.x, decimals)
        self.y = round(self.y, decimals)
        self.z = round(self.z, decimals)
        self.w = round(self.w, decimals)

    def almost_equal(self, other: Self, tolerance: float = 1e-6) -> bool:
        return bool(np.isclose(np.array(self.to_list()), np.array(other.to_list()), atol=tolerance).all())


@dataclass
class Pose:
    """
    A pose in 3D space.
    """
    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)

    def __repr__(self):
        return (f"Pose: {[round(v, 3) for v in [self.position.x, self.position.y, self.position.z]]}, "
                f"{[round(v, 3) for v in [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]]}")

    def to_list(self):
        return [self.position.to_list(), self.orientation.to_list()]

    def copy(self):
        return copy.copy(self)

    def round(self, decimals: int = 4):
        self.position.round(decimals)
        self.orientation.round(decimals)

    def almost_equal(self, other: Pose, tolerance: float = 1e-6) -> bool:
        return self.position.almost_equal(other.position, tolerance) and self.orientation.almost_equal(
            other.orientation, tolerance)

    def __eq__(self, other) -> bool:
        return self.almost_equal(other, tolerance=1e-4)

    @classmethod
    def from_list(cls, position: List[float], orientation: List[float]) -> Self:
        return cls(Vector3(position[0], position[1], position[2]),
                   Quaternion(orientation[0], orientation[1], orientation[2], orientation[3]))


@dataclass
class Header:
    """
    A header with a timestamp.
    """
    frame_id: str = "map"
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now, compare=False)
    sequence: int = field(default=0, compare=False)


@dataclass
class PoseStamped:
    """
    A pose in 3D space with a timestamp.
    """
    pose: Pose = field(default_factory=Pose)
    header: Header = field(default_factory=Header)

    @property
    def position(self):
        return self.pose.position

    @position.setter
    def position(self, value: Vector3):
        self.pose.position = value

    @property
    def orientation(self):
        return self.pose.orientation

    @orientation.setter
    def orientation(self, value: Quaternion):
        self.pose.orientation = value

    @property
    def frame_id(self):
        return self.header.frame_id

    @frame_id.setter
    def frame_id(self, value: str):
        self.header.frame_id = value

    def __repr__(self):
        return (f"Pose: {[round(v, 3) for v in [self.position.x, self.position.y, self.position.z]]}, "
                f"{[round(v, 3) for v in [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]]} "
                f"in frame_id {self.frame_id}")

    @classmethod
    def from_list(cls, position: List[float], orientation: List[float], frame: str) -> Self:
        return cls(pose=Pose.from_list(position, orientation),
                   header=Header(frame_id=frame, timestamp=datetime.datetime.now()))

    def to_transform_stamped(self, child_link_id: str) -> TransformStamped:
        return TransformStamped(header=self.header, pose=Transform.from_pose(self.pose), child_frame_id=child_link_id)

    def round(self, decimals: int = 4):
        self.position.round(decimals)
        self.orientation.round(decimals)


@dataclass
class Transform(Pose):
    @property
    def translation(self):
        return self.position

    @property
    def rotation(self):
        return self.orientation

    @classmethod
    def from_pose(cls, pose: Pose):
        return cls(pose.position, pose.orientation)


@dataclass
class TransformStamped(PoseStamped):
    child_frame_id: str = field(default_factory=str)

    pose: Transform

    @property
    def transform(self) -> Transform:
        return self.pose

    @transform.setter
    def transform(self, value: Transform):
        self.pose = value

    @property
    def translation(self):
        return self.pose.position

    @translation.setter
    def translation(self, value: Vector3):
        self.pose.position = value

    @property
    def rotation(self):
        return self.pose.orientation

    @rotation.setter
    def rotation(self, value: Quaternion):
        self.pose.orientation = value

    def __invert__(self):
        result = copy.deepcopy(self)
        result.header.frame_id = self.child_frame_id
        result.child_frame_id = self.header.frame_id
        result.transform = ~self.transform
        return result

    def __mul__(self, other):
        result = copy.deepcopy(self)
        result.child_frame_id = other.child_frame_id
        result.transform = self.transform * other.transform
        return result

    @classmethod
    def from_list(cls, position: List[float], orientation: List[float], frame: str, child_frame_id) -> Self:
        return cls(pose=Transform.from_list(position, orientation),
                   header=Header(frame_id=frame, timestamp=datetime.datetime.now()),
                   child_frame_id=child_frame_id)
