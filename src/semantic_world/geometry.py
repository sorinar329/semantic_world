from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property

import trimesh

from .pose import Vector3, Pose, PoseStamped
from .utils import IDGenerator

id_generator = IDGenerator()

@dataclass
class Color:
    """
    Dataclass for storing rgba_color as an RGBA value.
    The values are stored as floats between 0 and 1.
    The default rgba_color is white.
    """
    R: float = 1
    """
    Red value of the color.
    """

    G: float = 1
    """
    Green value of the color.
    """

    B: float = 1
    """
    Blue value of the color.
    """

    A: float = 1
    """
    Opacity of the color.
    """

@dataclass
class Shape(ABC):
    """
    Base class for all shapes in the world.
    """
    origin: PoseStamped = field(default_factory=PoseStamped)

@dataclass
class Mesh(Shape):
    """
    A mesh shape.
    """

    filename: str = ""
    """
    Filename of the mesh.
    """

    scale: Vector3 = field(default_factory=Vector3)
    """
    Scale of the mesh.
    """

    @cached_property
    def mesh(self) -> trimesh.Trimesh:
        """
        The mesh object.
        """
        return trimesh.load_mesh(self.filename)


@dataclass
class Primitive(Shape):
    """
    A primitive shape.
    """
    color: Color = field(default_factory=Color)


@dataclass
class Sphere(Primitive):
    """
    A sphere shape.
    """

    radius: float = 0.5
    """
    Radius of the sphere.
    """


@dataclass
class Capsule(Sphere):
    """
    A capsule shape.
    """

    length: float = 1.0
    """
    Length of the capsule.
    """


@dataclass
class Cylinder(Capsule):
    """
    A cylinder shape.
    """


@dataclass
class Box(Primitive):
    """
    A box shape.
    Pivot point is at the center of the box.
    """
    length: float = 1.0
    """
    Length of the box.
    """

    width: float = 1.0
    """
    Width of the box. 
    """

    height: float = 1.0
    """
    Height of the box.
    """

