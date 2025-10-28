from __future__ import annotations

from dataclasses import dataclass, field

from semantic_annotations import *

import numpy as np
from probabilistic_model.probabilistic_circuit.rx.helper import uniform_measure_of_event
from typing_extensions import List

from ..world_description.shape_collection import BoundingBoxCollection
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import Point3
from ..datastructures.variables import SpatialVariables
from ..world_description.world_entity import SemanticAnnotation, Body, Region


@dataclass(eq=False)
class Fridge(SemanticAnnotation):
    """
    A SemanticAnnotation representing a fridge that has a door and a body.
    """
    body: Body
    door: Door

@dataclass(eq=False)
class Sink(SemanticAnnotation):
    body: Body

@dataclass(eq=False)
class CounterTop(SemanticAnnotation):
    body: Body

@dataclass(eq=False)
class Hotplate(SemanticAnnotation):
    body: Body

@dataclass(eq=False)
class Cooktop(SemanticAnnotation):
    body: Body
    hotplate: List[Hotplate] = field(default_factory=list)

@dataclass(eq=False)
class Oven(SemanticAnnotation):
    container: Container
    door: Door