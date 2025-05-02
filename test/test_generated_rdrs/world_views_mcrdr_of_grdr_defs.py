from semantic_world.views import Handle
from semantic_world.enums import JointType
from semantic_world.views import Container
from semantic_world.views import Drawer


def conditions_140609304379792(case):
    return True


def conclusion_140609304379792(case):
    return [Handle(b) for b in case.bodies if "handle" in b.name.lower()]


def conditions_140609303179136(case):
    return len([v for v in case.views if type(v) is Handle]) > 0


def conclusion_140609303179136(case):
    prismatic_connections = [c for c in case.connections if c.type == JointType.PRISMATIC]
    fixed_connections = [c for c in case.connections if c.type == JointType.FIXED]
    children_of_prismatic_connections = [c.child for c in prismatic_connections]
    handles = [v for v in case.views if type(v) is Handle]
    fixed_connections_with_handle_child = [fc for fc in fixed_connections if fc.child in [h.body for h in handles]]
    drawer_containers = set(children_of_prismatic_connections).intersection(set([fc.parent for fc in fixed_connections_with_handle_child]))
    return [Container(b) for b in drawer_containers]


def conditions_140609307238000(case):
    return len([v for v in case.views if type(v) is Handle]) > 0 and len([v for v in case.views if type(v) is Container]) > 0


def conclusion_140609307238000(case):
    handles = [v for v in case.views if type(v) is Handle]
    containers = [v for v in case.views if type(v) is Container]
    fixed_connections = [c for c in case.connections if c.type == JointType.FIXED and c.parent in [cont.body for cont in containers] and c.child in [h.body for h in handles]]
    prismatic_connections = [c for c in case.connections if c.type == JointType.PRISMATIC and c.child in [cont.body for cont in containers]]
    drawer_handle_connections = [fc for fc in fixed_connections if fc.parent in [pc.child for pc in prismatic_connections]]
    drawers = [Drawer([cont for cont in containers if dc.parent==cont.body][0], [h for h in handles if dc.child==h.body][0]) for dc in drawer_handle_connections]
    return drawers


