from unittest import TestCase

from phiml import math
from phiml.math import spatial
from phiml.math.perm import all_permutations


class TestSparse(TestCase):

    def test_all_permutations_1d(self):
        p = all_permutations(spatial(x=3), index_dim=None)
        math.assert_close([0, 1, 2], p.permutations[0])
        self.assertEqual(6, p.permutations.size)

    def test_all_permutations_2d(self):
        p = all_permutations(spatial(x=2, y=2))
        self.assertEqual(24, p.permutations.size)
        math.assert_close(1, p.max)
        math.assert_close(0, p.min)
