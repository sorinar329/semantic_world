import unittest

import networkx as nx

from semantic_world.world import World, Link


class WorldTestCase(unittest.TestCase):

    def test_kinematic_chain_finding(self):
        world = World()
        world.validate()
        l1 = Link()
        l2 = Link()
        self.assertEqual(l1.name, "link_1")
        self.assertEqual(l2.name, "link_2")

        world.add_node(l1)
        world.add_node(l2)
        world.add_edge(world.root, l1)
        world.add_edge(world.root, l2)
        world.validate()

        path = nx.shortest_path(world, l1, l2)
        self.assertEqual(len(path), 3)


if __name__ == '__main__':
    unittest.main()
