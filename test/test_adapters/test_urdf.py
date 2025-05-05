import os.path
import unittest

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import GeneralRDR
from semantic_world.adapters.urdf import URDFParser
from semantic_world.enums import JointType
from semantic_world.world import View
from semantic_world.views import *


class URDFParserTestCase(unittest.TestCase):
    urdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf")
    table = os.path.join(urdf_dir, "table.urdf")
    kitchen = os.path.join(urdf_dir, "kitchen-small.urdf")
    apartment = os.path.join(urdf_dir, "apartment.urdf")

    def setUp(self):
        self.table_parser = URDFParser(self.table)
        self.kitchen_parser = URDFParser(self.kitchen)
        self.apartment_parser = URDFParser(self.apartment)

    def test_table_parsing(self):
        world = self.table_parser.parse()
        world.validate()
        self.assertEqual(len(world.bodies), 6)

        origin_left_front_leg_joint = world.get_joint(world.root, world.bodies[1])
        self.assertEqual(origin_left_front_leg_joint.type, JointType.FIXED)
        self.assertEqual(origin_left_front_leg_joint.child.origin.pose.position.x, -4)

    def test_kitchen_parsing(self):
        world = self.kitchen_parser.parse()
        world.validate()
        self.assertTrue(len(world.bodies) > 0)
        self.assertTrue(len(world.connections) > 0)

    def test_apartment_parsing(self):
        world = self.apartment_parser.parse()
        world.validate()
        self.assertTrue(len(world.bodies) > 0)
        self.assertTrue(len(world.connections) > 0)


if __name__ == '__main__':
    unittest.main()
