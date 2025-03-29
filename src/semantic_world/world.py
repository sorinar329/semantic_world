from __future__ import annotations

import networkx as nx

class WorldEntity:
    """
    A class representing an entity in the world.
    """

    _world: World
    """
    The backreference to the world this entity belongs to.
    """


class Link(WorldEntity):
    ...

class Joint(WorldEntity):
    ...

class World(nx.DiGraph):
    """
    A class representing the world as a directed graph.
    The nodes represent links in the world and the edges represent joins between them.
    """
