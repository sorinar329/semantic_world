import os.path
import unittest
from semantic_world.enums import WorldMode
from semantic_world.worlds.bullet import Bullet


class BulletTestCase(unittest.TestCase):
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf", "table.urdf")

    def setUp(self):
        self.bullet = Bullet(mode=WorldMode.GUI)
        self.bullet.start()

    def test_load_urdf(self):
        body_id = self.bullet.load_urdf(self.file)
        self.assertTrue(body_id >= 0)


if __name__ == '__main__':
    unittest.main()
