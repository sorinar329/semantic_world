from semantic_world.world import World
from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Set, Union
from ripple_down_rules.utils import make_set
from .world_views_mcrdr_of_grdr_defs import *
from ripple_down_rules.rdr import MultiClassRDR


conclusion_type = (set, Drawer, list, Handle, Container,)
type_ = MultiClassRDR


def classify(case: World) -> Set[Union[Drawer, Handle, Container]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_140149685411008(case):
        conclusions.update(make_set(conclusion_140149685411008(case)))

    if conditions_140149684931312(case):
        conclusions.update(make_set(conclusion_140149684931312(case)))

    if conditions_140149683331920(case):
        conclusions.update(make_set(conclusion_140149683331920(case)))
    return conclusions
