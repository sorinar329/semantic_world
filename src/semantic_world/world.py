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

    _world: Optional[World] = field(default=None, init=False, repr=False)
    """
    The backreference to the world this entity belongs to.
    """

    _views: List[View] = field(default_factory=list, init=False, repr=False)
    """
    The views this entity is part of.
    """

@dataclass
class Link(WorldEntity):
    """
    Represents a link in the world.
    A link is a semantic atom. This means that a link cannot be decomposed into meaningful smaller parts.
    """

    name: str = None
    """
    The name of the link. Must be unique in the world.
    If not provided, a unique name will be generated.
    """

    origin: PoseStamped = PoseStamped()
    """
    The pose of the link in the world.
    """

    visual: List[Shape] = field(default_factory=list, repr=False)
    """
    List of shapes that represent the visual appearance of the link.
    The poses of the shapes are relative to the link.
    """

    collision: List[Shape] = field(default_factory=list, repr=False)
    """
    List of shapes that represent the collision geometry of the link.
    The poses of the shapes are relative to the link.
    """

    def __post_init__(self):
        if not self.name:
            self.name = f"link_{id_generator(self)}"

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


class World:
    """
    A class representing the world.
    The world manages a set of links and joints represented as a tree-like graph.
    The nodes represent links in the world, and the edges represent joins between them.
    """

    root: Link
    """
    The root link of the world.
    """

    kinematic_structure: nx.Graph
    """
    The kinematic structure of the world.
    The kinematic structure is a tree-like graph where the nodes represent links in the world,
    and the edges represent joints between them.
    """

    def __init__(self, root: Optional[Link] = None, kinematic_structure: Optional[nx.Graph] = None):
        super().__init__()

        if not root:
            root = Link(name="map", origin=PoseStamped())
        self.root = root

        if not kinematic_structure:
            kinematic_structure = nx.DiGraph()
        self.kinematic_structure = kinematic_structure

        self.add_link(self.root)

    def validate(self):
        """
        Validate the world.
        The world must be a tree.
        """
        if not nx.is_tree(self.kinematic_structure):
            raise ValueError("The world is not a tree.")

    def add_link(self, link: Link):
        """
        Add a link to the world.
        """
        self.kinematic_structure.add_node(link)
        link._world = self

    def add_joint(self, joint: Joint):
        """
        Add a joint to the world.
        """
        self.add_link(joint.parent)
        self.add_link(joint.child)
        self.kinematic_structure.add_edge(joint.parent, joint.child, joint=joint)