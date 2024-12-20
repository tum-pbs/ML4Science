from unittest import TestCase

from phiml import math
from phiml.math import spatial, channel, batch, instance, non_instance, non_channel, non_spatial, non_batch
from phiml.math._shape import shape_stack, EMPTY_SHAPE, Shape, dual, resolve_index


class ShapedDummy:
    def __init__(self, shape: Shape):
        self.shape = shape


class TestShape(TestCase):

    def test_dimension_types(self):
        v = math.ones(batch(batch=10) + dual(d=1) + spatial(x=4, y=3) + channel(vector=2))
        self.assertEqual(v.x.index, 2)
        self.assertEqual(v.x.name, 'x')
        self.assertEqual(('batch', 'dual', 'spatial', 'spatial', 'channel'), tuple(v.shape.dim_types))
        b = v.x.as_batch()
        self.assertEqual(('batch', 'dual', 'batch', 'spatial', 'channel'), tuple(b.shape.dim_types))

    def test_combine(self):
        self.assertEqual(batch(batch=10) & spatial(y=4, x=3) & channel(vector=2), batch(batch=10) & channel(vector=2) & spatial(y=4, x=3))

    def test_stack(self):
        stacked = shape_stack(batch('stack'), batch(time=1) & spatial(x=3, y=3), spatial(x=3, y=4), EMPTY_SHAPE, stack_dim_first=True)
        print(stacked)
        self.assertEqual(('stack', 'time', 'x', 'y'), stacked.names)
        self.assertEqual(3, stacked.get_size('stack'))
        self.assertEqual(1, stacked.get_size('time'))
        math.assert_close(3, stacked.get_size('x'))
        math.assert_close([3, 4, 1], stacked.get_size('y'))
        self.assertEqual(('stack',), stacked.non_uniform_shape.names)
        self.assertEqual(3, stacked.non_uniform_shape.size)
        stacked = shape_stack(batch('stack'), channel(vector='x,y') & spatial(x=3, y=3), spatial(x=3, y=4), math.EMPTY_SHAPE)
        self.assertEqual(('x', 'y'), stacked['vector'].item_names[0])

    def test_subshapes(self):
        s = batch(batch=10) & spatial(x=4, y=3) & channel(vector=2) & instance(points=1)
        self.assertEqual(batch(batch=10), s.batch)
        self.assertEqual(spatial(x=4, y=3), s.spatial)
        self.assertEqual(channel(vector=2), s.channel)
        self.assertEqual(instance(points=1), s.instance)
        self.assertEqual(batch(batch=10), batch(s))
        self.assertEqual(spatial(x=4, y=3), spatial(s))
        self.assertEqual(channel(vector=2), channel(s))
        self.assertEqual(instance(points=1), instance(s))

    def test_indexing(self):
        s = batch(batch=10) & spatial(x=4, y=3) & channel(vector=2)
        self.assertEqual(batch(batch=10), s[0:1])
        self.assertEqual(batch(batch=10), s[0])
        self.assertEqual(spatial(x=4, y=3), s[1:3])
        self.assertEqual(spatial(x=4), s['x'])
        self.assertEqual(spatial(x=4, y=3), s.only('x, y', reorder=True))

    def test_after_gather(self):
        self.assertEqual(spatial(x=2), spatial(x=3).after_gather({'x': slice(None, None, 2)}))
        self.assertEqual(EMPTY_SHAPE, spatial(x=3).after_gather({'x': 0}))

    def test_item_names(self):
        c = channel(vector='r,g,b')
        self.assertEqual(('r', 'g', 'b'), c.get_item_names('vector'))

    def test_serialize_shape(self):
        s = math.concat_shapes(batch(batch=10), spatial(x=4, y=3), channel(vector=2))
        self.assertEqual(math.from_dict(math.to_dict(s)), s)

    def test_bool(self):
        self.assertFalse(math.EMPTY_SHAPE)
        self.assertTrue(math.spatial(x=3))

    def test_merge_shapes_check_item_names(self):
        s1 = channel(vector='x,y,z')
        s2 = channel(vector='r,g,b')
        try:
            math.merge_shapes(s1, s2)
            self.fail('Merging incompatible shapes did not raise an error!')
        except math.IncompatibleShapes:
            pass
        math.merge_shapes(s1, s1)

    def test_meshgrid(self):
        shape = spatial(x=2) & channel(vector='x')
        indices = list(shape.meshgrid(names=False))
        self.assertEqual([
            dict(x=0, vector=0),
            dict(x=1, vector=0),
        ], indices)

    def test_meshgrid_names(self):
        shape = spatial(x=2) & channel(vector='x')
        indices = list(shape.meshgrid(names=True))
        self.assertEqual([
            dict(x=0, vector='x'),
            dict(x=1, vector='x'),
        ], indices)

    def test_filters(self):
        b = batch(batch=10)
        s = spatial(x=4, y=3)
        i = instance(points=5)
        c = channel(vector=2)
        shape = math.concat_shapes(b, s, i, c)
        for obj in [shape, math.zeros(shape)]:
            self.assertEqual(batch(obj), b)
            self.assertEqual(spatial(obj), s)
            self.assertEqual(instance(obj), i)
            self.assertEqual(channel(obj), c)
            self.assertEqual(set(non_batch(obj)), set(s + i + c))
            self.assertEqual(set(non_spatial(obj)), set(math.concat_shapes(b, i, c)))
            self.assertEqual(set(non_instance(obj)), set(math.concat_shapes(b, s, c)))
            self.assertEqual(set(non_channel(obj)), set(math.concat_shapes(b, s, i)))

    def test_merge_shaped(self):
        self.assertEqual(spatial(x=4, y=3, z=2), math.merge_shapes(ShapedDummy(spatial(x=4, y=3)), spatial(y=3, z=2)))

    def test_concat_shaped(self):
        self.assertEqual(spatial(x=4, y=3, z=2), math.concat_shapes(ShapedDummy(spatial(x=4, y=3)), spatial(z=2)))

    def test_with_size_item_names(self):
        s = channel(vector='x,y')
        s1 = s.with_size(1)
        self.assertIsNone(s1.get_item_names('vector'))
        sxy = s1.with_sizes(s)
        self.assertEqual(('x', 'y'), sxy.get_item_names('vector'))
        sxy = s1.with_size('x,y')
        self.assertEqual(('x', 'y'), sxy.get_item_names('vector'))
        sxy = s1.with_size(['x', 'y'])
        self.assertEqual(('x', 'y'), sxy.get_item_names('vector'))
        sx = s.with_size('x')
        self.assertEqual(('x',), sx.get_item_names('vector'))
        wo = s.without_sizes()
        self.assertIsNone(wo.get_item_names('vector'))

    def test_dual_prefix(self):
        d = dual('~y,z', x=5)
        self.assertEqual(('~y', '~z', '~x'), d.names)

    def test_contains(self):
        s = batch(batch=10) & spatial(x=4, y=3) & channel(vector=2)
        self.assertTrue('x,y,batch' in s)
        self.assertTrue(['vector', 'batch'] in s)
        self.assertFalse(['other', 'batch'] in s)

    def test_as_type(self):
        s = batch(batch=10) & spatial(x=4, y=3) & channel(vector=2)
        self.assertEqual(dual(batch=10, x=4, y=3, vector=2), s.as_dual())
        self.assertEqual(channel(batch=10, x=4, y=3, vector=2), s.as_channel())
        self.assertEqual(spatial(batch=10, x=4, y=3, vector=2), s.as_spatial())
        self.assertEqual(instance(batch=10, x=4, y=3, vector=2), s.as_instance())
        self.assertEqual(batch(batch=10, x=4, y=3, vector=2), s.as_batch())

    def test_resolve_index(self):
        s = spatial(x=4, y=3) & channel(vector='x,y')
        self.assertEqual({}, resolve_index(s, {}))
        self.assertEqual({'vector': 0}, resolve_index(s, {'vector': 'x'}))
        self.assertEqual({'vector': 1}, resolve_index(s, {'vector': 'y'}))
        self.assertEqual({'vector': [0, 1]}, resolve_index(s, {'vector': 'x,y'}))
        self.assertEqual({'vector': [0, 1]}, resolve_index(s, {'vector': [0, 1]}))
        self.assertEqual({'vector': 0}, resolve_index(s, {'vector': 0}))
        self.assertEqual({'vector': slice(1)}, resolve_index(s, {'vector': slice(1)}))
        self.assertEqual({'vector': slice(0, 0)}, resolve_index(s, {'vector': []}))

    def test_auto(self):
        v = math.vec('~vector', 1, 2)
        self.assertEqual(dual(vector='1,2'), v.shape)
        v = math.vec('vec', 0)
        self.assertEqual(channel(vec='0'), v.shape)
        v = math.vec('b:b', 0)
        self.assertEqual(batch(b='0,'), v.shape)

    def test_only(self):
        s = batch(b=10) & channel(vector='x,y')
        self.assertEqual(batch(b=10), s.only([batch, spatial]))
        self.assertEqual(('vector', 'b'), s.only([channel, batch], reorder=True).names)

    def test_without(self):
        s = batch(b=10) & channel(vector='x,y')
        self.assertEqual(channel(vector='x,y'), s.without([batch, spatial]))
        self.assertEqual(channel(vector='x,y'), s - batch - spatial)
        self.assertEqual(channel(vector='x,y'), s - 'b')
        self.assertEqual(channel(vector='x,y'), s - ['b'])
        self.assertEqual(channel(vector='x,y'), s - [batch])

    def test_primitive_shape(self):
        s = non_batch([0, 1])
        self.assertEqual(2, s.size)

    def test_and_dual(self):
        shape = spatial(x=10, y=8) & channel(vector='x,y')
        and_dual = shape & dual
        self.assertEqual(shape.as_dual(), and_dual.dual)
        self.assertEqual(shape.as_dual(), and_dual[3:])
        dual_and = dual & shape
        self.assertEqual(shape.as_dual(), dual_and.dual)

    def test_hash(self):
        s = spatial(x=3, y=2)
        c = channel(vector='x,y')
        sc = s + c
        assert hash(s) == hash(sc[:2])
        assert hash(c) == hash(sc[2:])

    def test_eq(self):
        s = spatial(x=3, y=2)
        c = channel(vector='x,y')
        sc = s + c
        assert s == sc[:2]
        assert c == sc[2:]
