import os.path
import unittest
from semantic_world.adapters.urdf import URDFParser
from semantic_world.enums import JointType


class URDFParserTestCase(unittest.TestCase):
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf", "table.urdf")

    def setUp(self):
        self.parser = URDFParser(self.file)

    def test_parsing(self):
        world = self.parser.parse()
        world.validate()
        self.assertEqual(len(world.bodies), 6)

        origin_left_front_leg_joint = world.get_joint(world.root, world.bodies[1])
        self.assertEqual(origin_left_front_leg_joint.type, JointType.FIXED)
        self.assertEqual(origin_left_front_leg_joint.child.origin.pose.position.x, -4)



if __name__ == '__main__':
    unittest.main()
