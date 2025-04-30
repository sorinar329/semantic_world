from semantic_world.world import World
from semantic_world.world import View
from ripple_down_rules.datastructures.case import Case, create_case
from semantic_world.views import Handle, Cabinet
from semantic_world.views import Container
from semantic_world.views import Drawer
from typing_extensions import Set
from ripple_down_rules.utils import make_set
from .world_views_mcrdr_of_grdr_defs import *
from ripple_down_rules.rdr import MultiClassRDR


conclusion_type = (View, set, list,)

type_ = MultiClassRDR

def classify(case: World) -> Set[View]:
    if not isinstance(case, Case):
        case = create_case(case, max_recursion_idx=3)
    conclusions = set()
    if True:
        conclusions.update(make_set({[Handle(b) for b in case.bodies if "handle" in b.name.lower()]}))
    if len([v for v in case.views if type(v) is Handle]) > 0:
        def conclusion_140397960440000(case):
            prismatic_connections = [c for c in case.connections if c.type == JointType.PRISMATIC]
            fixed_connections = [c for c in case.connections if c.type == JointType.FIXED]
            children_of_prismatic_connections = [c.child for c in prismatic_connections]
            handles = [v for v in case.views if type(v) is Handle]
            fixed_connections_with_handle_child = [fc for fc in fixed_connections if fc.child in [h.body for h in handles]]
            drawer_containers = set(children_of_prismatic_connections).intersection(set([fc.parent for fc in fixed_connections_with_handle_child]))
            return [Container(b) for b in drawer_containers]
            
        conclusions.update(make_set(conclusion_140397960440000(case)))
    if len([v for v in case.views if type(v) is Handle]) > 0 and len([v for v in case.views if type(v) is Container]) > 0:
        def conclusion_140397960441728(case):
            handles = [v for v in case.views if type(v) is Handle]
            containers = [v for v in case.views if type(v) is Container]
            fixed_connections = [c for c in case.connections if c.type == JointType.FIXED and c.parent in [cont.body for cont in containers] and c.child in [h.body for h in handles]]
            prismatic_connections = [c for c in case.connections if c.type == JointType.PRISMATIC and c.child in [cont.body for cont in containers]]
            drawer_handle_connections = [fc for fc in fixed_connections if fc.parent in [pc.child for pc in prismatic_connections]]
            drawers = [Drawer([cont for cont in containers if dc.parent==cont.body][0], [h for h in handles if dc.child==h.body][0]) for dc in drawer_handle_connections]
            return drawers
            
        conclusions.update(make_set(conclusion_140397960441728(case)))
    if len([v for v in case.views if type(v) is Drawer]) > 0:
        def conclusion_140397960439712(case):
            drawers = [v for v in case.views if type(v) is Drawer]
            drawer_container_bodies = [d.container.body for d in drawers]
            prismatic_connections = [c for c in case.connections if c.type == JointType.PRISMATIC
                                     and c.child in drawer_container_bodies]
            cabinets = {}
            for pc in prismatic_connections:
                if pc.parent in cabinets:
                    if pc.child not in [d.container.body for d in cabinets[pc.parent].drawers]:
                        cabinets[pc.parent].drawers.append(Drawer(pc.child, pc.parent))
                cabinet_drawer = [d for d in drawers if d.container.body == pc.child][0]
                cabinets[pc.parent] = Cabinet(Container(pc.parent), [cabinet_drawer])
                
            return cabinets
            
        conclusions.update(make_set(conclusion_140397960439712(case)))
    return conclusions
