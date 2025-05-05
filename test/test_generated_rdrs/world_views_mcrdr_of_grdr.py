from ripple_down_rules.datastructures.case import Case, create_case
from typing_extensions import Set, Union
from ripple_down_rules.utils import make_set
from .world_views_mcrdr_of_grdr_defs import *
from ripple_down_rules.rdr import MultiClassRDR


conclusion_type = (Handle, Cabinet, Container, set, list, Drawer,)
type_ = MultiClassRDR


def classify(case: World) -> Set[Union[Handle, Cabinet, Container, Drawer]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()

    if conditions_55261605499735872923247958423413495616(case):
        conclusions.update(make_set(conclusion_55261605499735872923247958423413495616(case)))

    if conditions_300822958101218002194183744330332366576(case):
        conclusions.update(make_set(conclusion_300822958101218002194183744330332366576(case)))

    if conditions_247581782537506706867009757392206822517(case):
        conclusions.update(make_set(conclusion_247581782537506706867009757392206822517(case)))

    if conditions_302270522254447256087471875001107607132(case):
        conclusions.update(make_set(conclusion_302270522254447256087471875001107607132(case)))
    return conclusions
