from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List, Optional
import networkx as nx

from .geometry import Shape
from .utils import IDGenerator
import spatial_types as cas


id_generator = IDGenerator()

@dataclass
class PrefixedName:
    ...

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
class Body(WorldEntity):
    """
    Represents a link in the world.
    A link is a semantic atom. This means that a link cannot be decomposed into meaningful smaller parts.
    """

    name: PrefixedName
    """
    The name of the link. Must be unique in the world.
    If not provided, a unique name will be generated.
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

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class View(WorldEntity):
    """
    Represents a view on a set of bodies in the world.

    This class can hold references to certain bodies that gain meaning in this context.
    """

@dataclass
class Connection(WorldEntity):
    """
    Represents a connection between two bodies in the world.
    """

    parent: Body = None
    """
    The parent link of the joint.
    """

    child: Body = None
    """
    The child link of the joint.
    """

    origin: cas.TransformationMatrix = None
    """
    The origin of the joint.
    """

    def __hash__(self):
        return hash((self.parent, self.child))

@dataclass
class World:
    """
    A class representing the world.
    The world manages a set of bodies and connections represented as a tree-like graph.
    The nodes represent bodies in the world, and the edges represent joins between them.
    """

    root: Body = field(default=Body(name="map", origin=PoseStamped()), kw_only=True)
    """
    The root link of the world.
    """

    kinematic_structure: nx.DiGraph = field(default_factory=nx.DiGraph, kw_only=True, repr=False)
    """
    The kinematic structure of the world.
    The kinematic structure is a tree-like directed graph where the nodes represent bodies in the world,
    and the edges represent connections between them.
    """

    def __post_init__(self):
        self.add_body(self.root)

    def validate(self):
        """
        Validate the world.
        The world must be a tree.
        """
        if not nx.is_tree(self.kinematic_structure):
            raise ValueError("The world is not a tree.")

    @property
    def bodies(self) -> List[Body]:
        return list(self.kinematic_structure.nodes())

    @property
    def connections(self) -> List[Connection]:
        return [self.kinematic_structure.get_edge_data(*edge)[Connection.__name__] for edge in self.kinematic_structure.edges()]

    def add_body(self, body: Body):
        """
        Add a link to the world.
        """
        self.kinematic_structure.add_node(body)
        body._world = self

    def add_connection(self, connection: Connection):
        """
        Add a joint to the world.
        """
        self.add_body(connection.parent)
        self.add_body(connection.child)
        kwargs = {Connection.__name__: connection}
        self.kinematic_structure.add_edge(connection.parent, connection.child, **kwargs)

    def get_joint(self, parent: Body, child: Body) -> Connection:
        return self.kinematic_structure.get_edge_data(parent, child)[Connection.__name__]

    def plot_structure(self):
        """
        Plots the kinematic structure of the world.
        The plot shows bodies as nodes and connections as edges in a directed graph.
        """
        import matplotlib.pyplot as plt

        # Create a new figure
        plt.figure(figsize=(12, 8))

        # Use spring layout for node positioning
        pos = nx.drawing.bfs_layout(self.kinematic_structure, start=self.root)

        # Draw nodes (bodies)
        nx.draw_networkx_nodes(self.kinematic_structure, pos,
                               node_color='lightblue',
                               node_size=2000)

        # Draw edges (connections)
        edges = self.kinematic_structure.edges(data=True)
        nx.draw_networkx_edges(self.kinematic_structure, pos,
                               edge_color='gray',
                               arrows=True,
                               arrowsize=20)

        # Add link names as labels
        labels = {node: node.name for node in self.kinematic_structure.nodes()}
        nx.draw_networkx_labels(self.kinematic_structure, pos, labels)

        # Add joint types as edge labels
        edge_labels = {(edge[0], edge[1]): edge[2]['Joint'].type.name
                       for edge in self.kinematic_structure.edges(data=True)}
        nx.draw_networkx_edge_labels(self.kinematic_structure, pos, edge_labels)

        plt.title("World Kinematic Structure")
        plt.axis('off')  # Hide axes
        plt.show()

