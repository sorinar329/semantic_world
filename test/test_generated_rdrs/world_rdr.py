from typing_extensions import List, Union, Set
from ripple_down_rules.rdr import GeneralRDR
from ripple_down_rules.datastructures.case import Case, create_case
from semantic_world.world import World
from semantic_world.world import View
from test_generated_rdrs import world_views_mcrdr_of_grdr as views_classifier


classifiers_dict = dict()
classifiers_dict['views'] = views_classifier


def classify(case: World) -> List[Union[Set[View]]]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    return GeneralRDR._classify(classifiers_dict, case)
