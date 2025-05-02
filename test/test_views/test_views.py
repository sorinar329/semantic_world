import unittest

from semantic_world.views import Handle


class ViewTestCase(unittest.TestCase):

    def test_id(self):
        v1 = Handle(1)
        print(hash(v1))
        v2 = Handle(2)
        print(v1 is v2)
