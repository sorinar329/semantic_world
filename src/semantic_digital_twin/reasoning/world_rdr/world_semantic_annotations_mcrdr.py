from semantic_digital_twin.semantic_annotations.semantic_annotations import Cabinet, Container, Door, Drawer, Fridge, Handle, Window
from ripple_down_rules.datastructures.case import Case, create_case
from ripple_down_rules.utils import make_set
from semantic_digital_twin.world import World
from ripple_down_rules.helpers import get_an_updated_case_copy
from typing_extensions import Optional, Set, Union
from .world_semantic_annotations_mcrdr_defs import *


attribute_name = 'semantic_annotations'
conclusion_type = (Door, Fridge, Handle, set, list, Drawer, Window, Container, Cabinet,)
mutually_exclusive = False
name = 'semantic_annotations'
case_type = World
case_name = 'World'


def classify(case: World, **kwargs) -> Set[Union[Door, Fridge, Handle, Drawer, Window, Container, Cabinet]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_90574698325129464513441443063592862114(case):
        conclusions.update(make_set(conclusion_90574698325129464513441443063592862114(case)))

    if conditions_14920098271685635920637692283091167284(case):
        conclusions.update(make_set(conclusion_14920098271685635920637692283091167284(case)))

    if conditions_331345798360792447350644865254855982739(case):
        conclusions.update(make_set(conclusion_331345798360792447350644865254855982739(case)))

    if conditions_35528769484583703815352905256802298589(case):
        conclusions.update(make_set(conclusion_35528769484583703815352905256802298589(case)))

    if conditions_59112619694893607910753808758642808601(case):
        conclusions.update(make_set(conclusion_59112619694893607910753808758642808601(case)))

    if conditions_10840634078579061471470540436169882059(case):
        conclusions.update(make_set(conclusion_10840634078579061471470540436169882059(case)))

    if conditions_216280842469566949273981843907929693318(case):
        conclusions.update(make_set(conclusion_216280842469566949273981843907929693318(case)))
    return conclusions
