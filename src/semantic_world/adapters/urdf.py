import os
from dataclasses import dataclass

from typing_extensions import Optional, List, Union

from ..geometry import Shape, Box, Mesh, Cylinder
from ..spatial_types import Vector3, Quaternion, Pose, Header, PoseStamped
from ..world import World, Body, Connection
from urdf_parser_py import urdf

from ..geometry import Color
from ..enums import JointType, Axis
from ..utils import suppress_stdout_stderr
from .ros import get_ros_package_path

joint_type_map = {'unknown': JointType.UNKNOWN,
                 'revolute': JointType.REVOLUTE,
                 'continuous': JointType.CONTINUOUS,
                 'prismatic': JointType.PRISMATIC,
                 'floating': JointType.FLOATING,
                 'planar': JointType.PLANAR,
                 'fixed': JointType.FIXED}

@dataclass
class URDFParser:
    """
    Class to parse URDF files to worlds.
    """

    file_path: str
    """
    The file path of the URDF.
    """

    def parse(self) -> World:

        with open(self.file_path, 'r') as file:
            # Since parsing URDF causes a lot of warning messages which can't be deactivated, we suppress them
            with suppress_stdout_stderr():
                parsed = urdf.URDF.from_xml_string(file.read())

        links = [self.parse_link(link) for link in parsed.links]
        joints = []
        for joint in parsed.joints:
            parent = [link for link in links if link.name == joint.parent][0]
            child = [link for link in links if link.name == joint.child][0]
            parsed_joint = self.parse_joint(joint, parent, child)
            joints.append(parsed_joint)

        world = World(root=links[0])
        [world.add_connection(joint) for joint in joints]
        [world.add_body(link) for link in links]

        return world

    def parse_joint(self, joint: urdf.Joint, parent: Body, child: Body) -> Connection:
        axis = self.parse_joint_axis(joint.axis)

        lower = None
        upper = None
        if joint.limit:
            lower = joint.limit.lower
            upper = joint.limit.upper

        origin = self.urdf_pose_to_pose(joint.origin)
        origin = PoseStamped(origin, Header(frame_id=parent.name))

        result = Connection(type=joint_type_map[joint.type], parent=parent, child=child,
                            axis=axis, lower_limit=lower, upper_limit=upper, origin=origin)

        child.origin = origin
        return result

    def parse_joint_axis(self, axis) -> Axis:
        result = Axis.X
        if axis:
            if axis[0]:
                result = Axis.X
            elif axis[1]:
                result = Axis.Y
            elif axis[2]:
                result = Axis.Z
        return result

    def visual_of_link(self, link: urdf.Link) -> List[Shape]:
        if link.visuals:
            return [self.parse_shape(visual, link) for visual in link.visuals]
        else:
            return []

    def collision_of_link(self, link: urdf.Link) -> List[Shape]:
        if link.collisions:
            return [self.parse_shape(collision, link) for collision in link.collisions]
        else:
            return []

    def parse_shape(self, shape: urdf.Visual, link: urdf.Link) -> Shape:
        geometry: urdf.GeometricType = shape.geometry

        if isinstance(geometry, urdf.Box):
            return self.parse_box(geometry, shape, link)
        elif isinstance(geometry, urdf.Mesh):
            return self.parse_mesh(geometry, shape, link)
        elif isinstance(geometry, urdf.Cylinder):
            return self.parse_cylinder(geometry, shape, link)

        raise NotImplementedError(f"Parsing of {type(geometry)}: {geometry} not implemented yet.")

    def get_color(self, shape: Union[urdf.Visual, urdf.Collision]) -> Color:
        if isinstance(shape, urdf.Visual) and shape.material and shape.material.color and shape.material.color.rgba:
            return Color(shape.material.color.rgba)
        else:
            return Color()

    def parse_box(self, box: urdf.Box, shape: Union[urdf.Visual, urdf.Collision], link: urdf.Link) -> Box:
        pose = self.as_pose_stamped(self.urdf_pose_to_pose(shape.origin), link)
        color = self.get_color(shape)

        result = Box(length=box.size[0], width=box.size[1], height=box.size[2], origin=pose, color=color)
        return result

    def parse_mesh(self, mesh: urdf.Mesh, shape: Union[urdf.Visual, urdf.Collision], link: urdf.Link) -> Mesh:

        scale = Vector3(*mesh.scale) if mesh.scale else Vector3(1., 1., 1.)
        filename: str = mesh.filename

        if filename.startswith("package://"):
            package_name = filename.split('//')
            package_name = package_name[1].split('/')
            path = get_ros_package_path(package_name[0])
            filename = filename.replace("package://" + package_name[0], path)
        elif filename.startswith("file://"):
            filename = filename.replace("file://", './')

        return Mesh(filename=filename, scale=scale,
                    origin=self.as_pose_stamped(self.urdf_pose_to_pose(shape.origin), link))

    def parse_cylinder(self, cylinder: urdf.Cylinder, shape: Union[urdf.Visual, urdf.Collision], link: urdf.Link) -> Cylinder:
        color = self.get_color(shape)
        return Cylinder(radius=cylinder.radius, length=cylinder.length, origin=self.as_pose_stamped(self.urdf_pose_to_pose(shape.origin), link), color=color)


    def parse_link(self, link: urdf.Link) -> Body:
        """
        Parses a URDF link to a link object.
        :param link: The URDF link to parse.
        :return: The parsed link object.
        """
        return Body(link.name, visual=self.visual_of_link(link), collision=self.collision_of_link(link))

    def urdf_pose_to_pose(self, pose: urdf.Pose) -> Pose:
        if pose:
            return Pose(Vector3(*pose.xyz), Quaternion(*pose.rpy, 1.))
        else:
            return Pose()

    def as_pose_stamped(self, pose: Pose, link: Body):
        return PoseStamped(pose=pose, header=Header(frame_id=link.name))