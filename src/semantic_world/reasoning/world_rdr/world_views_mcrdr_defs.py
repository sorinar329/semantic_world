from entity_query_language import rule_mode, infer, entity, in_, let, From
from typing_extensions import List, Union

from ...views.views import Cabinet, Container, Door, Drawer, Fridge, Handle
from ...world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from ...world import World
from ...world_description.world_entity import Body


def conditions_90574698325129464513441443063592862114(case) -> bool:
    def has_bodies_named_handle(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Handle."""
        return any(
            "handle" in b.name.name.lower() for b in case.kinematic_structure_entities
        )

    return has_bodies_named_handle(case)


def conclusion_90574698325129464513441443063592862114(case) -> List[Handle]:
    def get_handles(case: World) -> Union[set, list, Handle]:
        """Get possible value(s) for World.views of types list/set of Handle"""
        with rule_mode():
            body = Body(From(case.bodies))
            query = infer(entity(Handle(body=body), in_("handle", body.name.name.lower())))

        return list(query.evaluate())

    return get_handles(case)


def conditions_14920098271685635920637692283091167284(case) -> bool:
    def has_handles_and_fixed_and_prismatic_connections(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Container."""
        return (
            any(v for v in case.views if type(v) is Handle)
            and any(c for c in case.connections if isinstance(c, PrismaticConnection))
            and any(c for c in case.connections if isinstance(c, FixedConnection))
        )

    return has_handles_and_fixed_and_prismatic_connections(case)


def conclusion_14920098271685635920637692283091167284(case) -> List[Container]:
    def get_containers(case: World) -> Union[set, Container, list]:
        """Get possible value(s) for World.views of types list/set of Container"""
        prismatic_connections = [
            c for c in case.connections if isinstance(c, PrismaticConnection)
        ]
        fixed_connections = [
            c for c in case.connections if isinstance(c, FixedConnection)
        ]
        children_of_prismatic_connections = [c.child for c in prismatic_connections]
        handles = [v for v in case.views if type(v) is Handle]
        fixed_connections_with_handle_child = [
            fc for fc in fixed_connections if fc.child in [h.body for h in handles]
        ]
        drawer_containers = set(children_of_prismatic_connections).intersection(
            set([fc.parent for fc in fixed_connections_with_handle_child])
        )
        return [Container(b) for b in drawer_containers]

    return get_containers(case)


def conditions_331345798360792447350644865254855982739(case) -> bool:
    def has_handles_and_containers(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Drawer."""
        return any(v for v in case.views if type(v) is Handle) and any(
            v for v in case.views if type(v) is Container
        )

    return has_handles_and_containers(case)


def conclusion_331345798360792447350644865254855982739(case) -> List[Drawer]:
    def get_drawers(case: World) -> Union[set, list, Drawer]:
        """Get possible value(s) for World.views of types list/set of Drawer"""
        handles = [v for v in case.views if type(v) is Handle]
        containers = [v for v in case.views if type(v) is Container]
        fixed_connections = [
            c
            for c in case.connections
            if isinstance(c, FixedConnection)
            and c.parent in [cont.body for cont in containers]
            and c.child in [h.body for h in handles]
        ]
        prismatic_connections = [
            c
            for c in case.connections
            if isinstance(c, PrismaticConnection)
            and c.child in [cont.body for cont in containers]
        ]
        drawer_handle_connections = [
            fc
            for fc in fixed_connections
            if fc.parent in [pc.child for pc in prismatic_connections]
        ]
        drawers = [
            Drawer(
                [cont for cont in containers if dc.parent == cont.body][0],
                [h for h in handles if dc.child == h.body][0],
            )
            for dc in drawer_handle_connections
        ]
        return drawers

    return get_drawers(case)


def conditions_35528769484583703815352905256802298589(case) -> bool:
    def has_drawers(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Cabinet."""
        return any(v for v in case.views if type(v) is Drawer)

    return has_drawers(case)


def conclusion_35528769484583703815352905256802298589(case) -> List[Cabinet]:
    def get_cabinets(case: World) -> Union[set, Cabinet, list]:
        """Get possible value(s) for World.views of types list/set of Cabinet"""
        drawers = [v for v in case.views if type(v) is Drawer]
        prismatic_connections = [
            c
            for c in case.connections
            if isinstance(c, PrismaticConnection)
            and c.child in [drawer.container.body for drawer in drawers]
        ]
        cabinet_container_bodies = [pc.parent for pc in prismatic_connections]
        cabinets = []
        for ccb in cabinet_container_bodies:
            if ccb in [cabinet.container.body for cabinet in cabinets]:
                continue
            cc_prismatic_connections = [
                pc for pc in prismatic_connections if pc.parent is ccb
            ]
            cabinet_drawer_container_bodies = [
                pc.child for pc in cc_prismatic_connections
            ]
            cabinet_drawers = [
                d
                for d in drawers
                if d.container.body in cabinet_drawer_container_bodies
            ]
            cabinets.append(Cabinet(Container(ccb), cabinet_drawers))

        return cabinets

    return get_cabinets(case)


def conditions_59112619694893607910753808758642808601(case) -> bool:
    def has_handles_and_revolute_connections(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Door."""
        return any(v for v in case.views if isinstance(v, Handle)) and any(
            c for c in case.connections if isinstance(c, RevoluteConnection)
        )

    return has_handles_and_revolute_connections(case)


def conclusion_59112619694893607910753808758642808601(case) -> List[Door]:
    def get_doors(case: World) -> List[Door]:
        """Get possible value(s) for World.views  of type Door."""
        handles = [v for v in case.views if isinstance(v, Handle)]
        handle_bodies = [h.body for h in handles]
        connections_with_handles = [
            c
            for c in case.connections
            if isinstance(c, FixedConnection) and c.child in handle_bodies
        ]

        revolute_connections = [
            c for c in case.connections if isinstance(c, RevoluteConnection)
        ]
        bodies_connected_to_handles = [
            c.parent if c.child in handle_bodies else c.child
            for c in connections_with_handles
        ]
        bodies_that_have_revolute_joints = [
            b
            for b in bodies_connected_to_handles
            for c in revolute_connections
            if b == c.child
        ]
        body_handle_connections = [
            c
            for c in connections_with_handles
            if c.parent in bodies_that_have_revolute_joints
        ]
        doors = [
            Door(c.parent, [h for h in handles if h.body == c.child][0])
            for c in body_handle_connections
        ]
        return doors

    return get_doors(case)


def conditions_10840634078579061471470540436169882059(case) -> bool:
    def has_doors_with_fridge_in_their_name(case: World) -> bool:
        """Get conditions on whether it's possible to conclude a value for World.views  of type Fridge."""
        return any(
            v
            for v in case.views
            if isinstance(v, Door) and "fridge" in v.body.name.name.lower()
        )

    return has_doors_with_fridge_in_their_name(case)


def conclusion_10840634078579061471470540436169882059(case) -> List[Fridge]:
    def get_fridges(case: World) -> List[Fridge]:
        """Get possible value(s) for World.views  of type Fridge."""
        # Get fridge-related doors
        fridge_doors = [
            v
            for v in case.views
            if isinstance(v, Door) and "fridge" in v.body.name.name.lower()
        ]
        # Precompute bodies of the fridge doors
        fridge_doors_bodies = [d.body for d in fridge_doors]
        # Filter relevant revolute connections
        fridge_door_connections = [
            c
            for c in case.connections
            if isinstance(c, RevoluteConnection)
            and c.child in fridge_doors_bodies
            and "fridge" in c.parent.name.name.lower()
        ]
        return [
            Fridge(c.parent, fridge_doors[fridge_doors_bodies.index(c.child)])
            for c in fridge_door_connections
        ]

    return get_fridges(case)
