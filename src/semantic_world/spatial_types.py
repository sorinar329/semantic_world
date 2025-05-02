from dataclasses import dataclass

from .prefixed_name import PrefixedName


@dataclass
class ReferenceFrameMixin:
    reference_frame: PrefixedName


@dataclass
class Point3(ReferenceFrameMixin):
    ...


@dataclass
class Vector3(ReferenceFrameMixin):
    """
    A 3D vector with x, y and z coordinates.
    """


@dataclass
class Quaternion(ReferenceFrameMixin):
    """
    A quaternion with x, y, z and w components.
    """


@dataclass
class TransformationMatrix(ReferenceFrameMixin):
    ...


@dataclass
class RotationMatrix(ReferenceFrameMixin):
    ...
