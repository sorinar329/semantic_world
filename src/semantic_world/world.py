from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List, Optional
import networkx as nx

from .enums import JointType, Axis
from .geometry import Shape
from .pose import PoseStamped
from .utils import IDGenerator

id_generator = IDGenerator()

@dataclass
class WorldEntity:
    """
    A class representing an entity in the world.
    """

    _world: Optional[World] = None
    """
    The backreference to the world this entity belongs to.
    """

    _views: List[View] = field(default_factory=list)
    """
    The views this entity is part of.
    """

@dataclass
class Link(WorldEntity):
    """
    Represents a link in the world.
    """

    name: str = field(default_factory=lambda: f"link_{id_generator(None)}")
    """
    The name of the link. Must be unique in the world.
    """

    origin: PoseStamped = PoseStamped()
    """
    The pose of the link in the world.
    """

    visual: List[Shape] = field(default_factory=list)
    """
    List of shapes that represent the visual appearance of the link.
    The poses of the shapes are relative to the link.
    """

    collision: List[Shape] = field(default_factory=list)
    """
    List of shapes that represent the collision geometry of the link.
    The poses of the shapes are relative to the link.
    """

    def __hash__(self):
        return hash(self.name)

    @property
    def joints(self) -> List[Joint]:
        """
        Returns all joints that are connected to this link.
        """

    def __eq__(self, other):
        return self.name == other.name


class View(WorldEntity):
    """
    Represents a view on a set of links in the world.

    This class can hold references to certain links that gain meaning in this context.
    """
    links: List[Link] = field(default_factory=list)
    """
    The links that are part of this view.
    """

    name: str = field(default_factory=lambda: f"view_{id_generator(None)}")
    """
    The name of the view. Must be unique in the world.
    """

    origin: PoseStamped = PoseStamped()
    """
    The pose of the view in the world.
    """


@dataclass
class Joint(WorldEntity):
    """
    Represents a joint in the world.
    """
    type: JointType = JointType.UNKNOWN
    """
    The type of the joint.
    """

    parent: Link = None
    """
    The parent link of the joint.
    """

    child: Link = None
    """
    The child link of the joint.
    """

    axis: Axis = None
    """
    The axis (perhaps multiple) of the joint.
    """
    value: float = 0.

    lower_limit: float = -float("inf")
    """
    The lower limit of the joint.
    """

    upper_limit: float = float("inf")
    """
    The upper limit of the joint.
    """

    damping: float = 0.
    """
    The damping of the joint.
    """

    friction: float = 0.
    """
    The friction of the joint.
    """

    origin: PoseStamped = PoseStamped()
    """
    The origin of the joint.
    """

    def __hash__(self):
        return hash((self.parent, self.child))


class World(nx.DiGraph):
    """
    A class representing the world as a directed graph.
    The nodes represent links in the world, and the edges represent joins between them.
    """

    root: Link

    def __init__(self):
        super().__init__()
        self.root = Link()


    def add_node(self, node: WorldEntity, **attr):
        """
        Add a node to the world.
        """
        super().add_node(node, **attr)
        node._world = self