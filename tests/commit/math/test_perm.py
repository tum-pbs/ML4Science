from unittest import TestCase

from phiml import math
from phiml.math import spatial, wrap, squeeze, channel
from phiml.math.perm import all_permutations, optimal_perm


class TestSparse(TestCase):

    def test_all_permutations_1d(self):
        p = all_permutations(spatial(x=3), index_dim=None)
        math.assert_close([0, 1, 2], p.perm.dual[0])
        self.assertEqual(6, p.perm.dual.size)

    def test_all_permutations_2d(self):
        p = all_permutations(spatial(x=2, y=2))
        self.assertEqual(24, p.perm.dual.size)
        math.assert_close(1, p.max)
        math.assert_close(0, p.min)

    def test_optimal_perm(self):
        o = wrap([0, 10, 100], 'x')
        i = wrap([90, -2, 0], '~x')
        i_perm, cost = optimal_perm((i - o) ** 2)
        math.assert_close([1, 2, 0], squeeze(i_perm, channel))
        math.assert_close(cost, (i[i_perm] - o) ** 2)
