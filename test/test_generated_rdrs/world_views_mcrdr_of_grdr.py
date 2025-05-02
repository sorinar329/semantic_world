from semantic_world.world import World
from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Set, Union
from ripple_down_rules.utils import make_set
from .world_views_mcrdr_of_grdr_defs import *
from ripple_down_rules.rdr import MultiClassRDR


conclusion_type = (Handle, set, list, Container, Drawer,)
type_ = MultiClassRDR


def classify(case: World) -> Set[Union[Handle, Container, Drawer]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_140609304379792(case):
        conclusions.update(make_set(conclusion_140609304379792(case)))

    if conditions_140609303179136(case):
        conclusions.update(make_set(conclusion_140609303179136(case)))

    if conditions_140609307238000(case):
        conclusions.update(make_set(conclusion_140609307238000(case)))
    return conclusions
