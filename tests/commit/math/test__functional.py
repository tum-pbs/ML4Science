import time
from functools import partial
from typing import List
from unittest import TestCase

from phiml import math
from phiml.backend import Backend
from phiml.backend._backend import init_installed_backends
from phiml.math import tensor, spatial, batch, channel, wrap, dual, Tensor

BACKENDS = init_installed_backends()


class TestFunctional(TestCase):

    def test_jit_compile(self):
        @math.jit_compile
        def scalar_mul(x, fac=1):
            return x * fac

        for backend in BACKENDS:
            with backend:
                x = math.ones(spatial(x=4))
                trace_count_0 = len(scalar_mul.traces)
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(x, fac=2), 2, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(math.zeros(spatial(x=4)), fac=2), 0, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(math.zeros(spatial(y=4)), fac=2), 0, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 2)

    def test_jit_compile_aux(self):
        @partial(math.jit_compile, auxiliary_args='fac')
        def scalar_mul(x, fac):
            return x * fac
        math.assert_close(6, scalar_mul(2, 3))

    def test_jit_compile_with_native(self):
        @math.jit_compile
        def scalar_mul(x, fac=1):
            return x * fac

        for backend in BACKENDS:
            with backend:
                x = backend.ones([3, 2])
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)

    def test_jit_compile_linear(self):
        return  # ToDo this fails on GitHub actions but is not reproducible on other systems
        x = math.random_normal(batch(batch=3) & spatial(x=4, y=3))  # , vector=2

        def linear_function(val):
            val = -val
            val *= 2
            val = math.pad(val, {'x': (2, 0), 'y': (0, 1)}, math.extrapolation.PERIODIC)
            val = val.x[:-2].y[1:] + val.x[2:].y[:-1]
            val = math.pad(val, {'x': (0, 0), 'y': (0, 1)}, math.extrapolation.ZERO)
            val = math.pad(val, {'x': (2, 2), 'y': (0, 1)}, math.extrapolation.BOUNDARY)
            return math.sum([val, val], dim='0') - val

        functions = [
            linear_function,
            lambda val: math.spatial_gradient(val, difference='forward', padding=math.extrapolation.ZERO, dims='x').gradient[0],
            lambda val: math.spatial_gradient(val, difference='backward', padding=math.extrapolation.PERIODIC, dims='x').gradient[0],
            lambda val: math.spatial_gradient(val, difference='central', padding=math.extrapolation.BOUNDARY, dims='x').gradient[0],
        ]
        for f in functions:
            direct_result = f(x)
            jit_f = math.jit_compile_linear(f)
            jit_result = jit_f(x)
            math.assert_close(direct_result, jit_result)

    def test_gradient(self):
        def f(x: math.Tensor, y: math.Tensor):
            assert isinstance(x, math.Tensor)
            assert isinstance(y, math.Tensor)
            pred = x
            loss = math.l2_loss(pred - y)
            return loss, pred

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    x_data = math.tensor(2.)
                    y_data = math.tensor(1.)
                    dx = math.gradient(f, wrt=0, get_output=False)(x_data, y_data)
                    math.assert_close(dx, 1, msg=backend.name)
                    dx, dy = math.gradient(f, [0, 1], get_output=False)(x_data, y_data)
                    math.assert_close(dx, 1, msg=backend.name)
                    math.assert_close(dy, -1, msg=backend.name)
                    (loss, pred), (dx, dy) = math.gradient(f, [0, 1], get_output=True)(x_data, y_data)
                    math.assert_close(loss, 0.5, msg=backend.name)
                    math.assert_close(pred, x_data, msg=backend.name)
                    math.assert_close(dx, 1, msg=backend.name)
                    math.assert_close(dy, -1, msg=backend.name)

    def test_custom_gradient_scalar(self):
        def f(x):
            return x

        def grad(_inputs, _y, df):
            return {'x': df * 0}

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    normal_gradient, = math.gradient(f, get_output=False)(math.ones())
                    math.assert_close(normal_gradient, 1)
                    f_custom_grad = math.custom_gradient(f, grad)
                    custom_gradient, = math.gradient(f_custom_grad, get_output=False)(math.ones())
                    math.assert_close(custom_gradient, 0)

    def test_custom_gradient_vector(self):
        def f(x):
            return x.x[:2]

        def grad(_inputs, _y, df):
            return {'x': math.flatten(math.expand(df * 0, batch(tmp=2)), flatten_batch=True)}

        def loss(x):
            fg = math.custom_gradient(f, grad)
            y = fg(x)
            return math.l1_loss(y)

        for backend in BACKENDS:
            if backend.supports(Backend.custom_gradient):
                with backend:
                    custom_loss_grad, = math.gradient(loss, get_output=False)(math.ones(spatial(x=4)))
                    math.assert_close(custom_loss_grad, 0, msg=backend.name)

    def test_map_types(self):
        def f(x, y):
            assert x.shape.batch.names == ('batch', 'x', 'y')
            assert x.shape.channel.names == ('vector',)
            assert y.shape == x.shape
            return x, y

        for f_ in [
            # math.map_types(f, 'x,y', batch),
            # math.map_types(f, spatial('x,y'), batch),
            math.map_types(f, spatial, batch),
        ]:
            x = math.random_uniform(batch(batch=10), spatial(x=4, y=3), channel(vector=2))
            x_, y_ = f_(x, x)
            assert x_.shape == x.shape
            math.assert_close(x, x_)

    # def test_hessian(self):
    #     def f(x, y):
    #         return math.l1_loss(x ** 2 * y), x, y
    #
    #     eval_hessian = math.hessian(f, wrt='x', get_output=True, get_gradient=True, dim_suffixes=('1', '2'))
    #
    #     for backend in BACKENDS:
    #         if backend.supports(Backend.hessian):
    #             with backend:
    #                 x = math.tensor([(0.01, 1, 2)], channel('vector', 'v'))
    #                 y = math.tensor([1., 2.], batch('batch'))
    #                 (L, x, y), g, H, = eval_hessian(x, y)
    #                 math.assert_close(L, [5.0001, 10.0002], msg=backend.name)
    #                 math.assert_close(g.batch[0].vector[0], (0.02, 2, 4), msg=backend.name)
    #                 math.assert_close(g.batch[1].vector[0], (0.04, 4, 8), msg=backend.name)
    #                 math.assert_close(2, H.v1[0].v2[0].batch[0], H.v1[1].v2[1].batch[0], H.v1[2].v2[2].batch[0], msg=backend.name)
    #                 math.assert_close(4, H.v1[0].v2[0].batch[1], H.v1[1].v2[1].batch[1], H.v1[2].v2[2].batch[1], msg=backend.name)

    def test_loss_batch_not_reduced(self):
        def loss_function(x):
            return math.l2_loss(x)

        gradient_function = math.gradient(loss_function, wrt=0)

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    x_test = tensor([0, 1], batch('examples'))
                    loss_direct = loss_function(x_test)
                    loss_g, _ = gradient_function(x_test)
                    math.assert_close([0, 0.5], loss_g, loss_direct)

    def test_iterate(self):
        def f(x, fac):
            return x * fac
        self.assertEqual(4, math.iterate(f, 2, 1, f_kwargs=dict(fac=2.)))
        math.assert_close([1, 2, 4], math.iterate(f, batch(trajectory=2), 1, fac=2.))
        # With measure
        r, t = math.iterate(f, 2, 1, fac=2., measure=time.perf_counter)
        self.assertEqual(4, r)
        self.assertIsInstance(t, float)
        r, t = math.iterate(f, batch(trajectory=2), 1, fac=2., measure=time.perf_counter)
        math.assert_close([1, 2, 4], r)
        self.assertEqual(batch(trajectory=2), t.shape)

    def test_iterate_more_outputs(self):
        def f(x):
            return x + 1, 1
        x_trj, one_trj = math.iterate(f, spatial(t=10), 0)
        math.assert_close(x_trj, math.range(spatial(t=11)))
        math.assert_close(one_trj, 1)
        final_x, final_one = math.iterate(f, 10, 0)
        math.assert_close(final_x, 10)
        math.assert_close(final_one, 1)

    def test_iterate_without_initial(self):
        def f(x):
            return x + 1 if x is not None else 0
        x_trj = math.iterate(f, spatial(t=10), None)
        math.assert_close(x_trj, math.range(spatial(t=10)))
        final_x = math.iterate(f, 10, None)
        math.assert_close(final_x, 9)

    def test_iterate_with_substeps(self):
        def m(x):
            return x + 1
        final = math.iterate(m, 10, 0, substeps=2)
        math.assert_close(20, final)
        trj = math.iterate(m, spatial(x=10), 0, substeps=2)
        self.assertEqual(11, trj.shape.size)
        math.assert_close(20, trj[-1])

    def test_delayed_decorator(self):
        def f(x, y):
            return x + y
        for jit in [math.jit_compile, math.jit_compile_linear]:
            f_ = jit(auxiliary_args='y', forget_traces=True)(f)
            self.assertTrue(f_.forget_traces)
            f_ = jit(auxiliary_args='y')(f)
            self.assertFalse(f_.forget_traces)
            f_ = jit(forget_traces=True)(f)
            self.assertTrue(f_.forget_traces)
            f_ = jit()(f)
            self.assertFalse(f_.forget_traces)

    def test_trace_check(self):
        @math.jit_compile(auxiliary_args='aux')
        def f(x, aux):
            return x * aux

        for backend in [b for b in BACKENDS if b.supports(Backend.jit_compile)]:
            with backend:
                x0 = math.zeros()
                aux0 = 1
                self.assertFalse(math.trace_check(f, x0, aux0)[0])
                f(x0, aux0)
                self.assertTrue(math.trace_check(f, x0, aux0)[0])
                self.assertTrue(math.trace_check(f, x=x0, aux=aux0)[0])

    def test_matrix_rename_dims(self):
        def f(x):
            return math.rename_dims(x, 'x', 'y')
        mat, _ = math.matrix_from_function(f, math.zeros(spatial(x=3)))
        self.assertEqual(dual(x=3), dual(mat))
        self.assertEqual(spatial(y=3), spatial(mat))

    def test_map(self):
        F_CALLS = []
        def f(x, y):
            F_CALLS.append((x, y))
            return x + y
        x = wrap((0, 1), spatial('x'))
        y = wrap((2, 4), spatial('y'))
        math.assert_close(wrap([(2, 4), (3, 5)], spatial('x,y')), math.map(f, x, y))
        self.assertEqual(4, len(F_CALLS), msg=F_CALLS)
        F_CALLS.clear()
        math.assert_close(wrap([(2, 4), (3, 5)], spatial('x,y')), math.map(f, x=x, y=y))
        self.assertEqual(4, len(F_CALLS), msg=F_CALLS)
        F_CALLS.clear()
        math.assert_close(wrap([(2, 4), (3, 5)], spatial('x,y')), math.map(f, x, y, dims=x.shape, unwrap_scalars=False))
        self.assertEqual(2, len(F_CALLS), msg=F_CALLS)
        F_CALLS.clear()

    def test_map_layout(self):
        l = math.layout('loss', math.EMPTY_SHAPE)
        a = math.layout([[0, 1], [2, 3]], spatial('x,y'))
        loss4 = math.map(lambda l, a: l, l, a)
        for l4 in loss4:
            self.assertEqual('loss', l4)

    def test_map_multi_output(self):
        def f(x, y):
            return x + y, x - y
        x = wrap((0, 1), spatial('x'))
        y = wrap((2, 4), spatial('y'))
        r_x, r_y = math.map(f, x, y)
        math.assert_close(wrap([(2, 4), (3, 5)], spatial('x,y')), r_x)
        math.assert_close(wrap([(-2, -4), (-1, -3)], spatial('x,y')), r_y)

    def test_broadcast(self):
        for len_ in [math.broadcast(len, channel), math.broadcast(dims=channel)(len)]:
            strings = math.vec('vector', 'a', 'bc', '')
            math.assert_close([1, 2, 0], len_(strings))

    def test_when_available(self):
        for backend in BACKENDS:
            with backend:
                TRACER: List[Tensor] = []
                CONCRETE: List[Tensor] = []

                @math.jit_compile
                def fun(x):
                    TRACER.append(x)

                    def print_x(x):
                        CONCRETE.append(x)

                    math.when_available(print_x, x)
                    return x

                fun(tensor(0))
                assert CONCRETE
                assert CONCRETE[0].available
                assert TRACER

    def test_perf_counter(self):
        @math.jit_compile
        def fun(x):
            t0 = math.perf_counter(x)
            print("fun called, time=", t0)
            for i in range(1000):
                x *= 0.5
                x += 1
            dt = math.perf_counter(x) - t0
            return x, dt

        for backend in BACKENDS:
            with backend:
                result, exec_time = fun(tensor(0))
                print("time taken", 1000_000 * float(exec_time), "result:", result)

    def test_grad_jit(self):
        for b in BACKENDS:
            if b.supports(Backend.jacobian) and b.supports(Backend.jit_compile):
                with b:
                    @math.jit_compile(auxiliary_args='a')
                    def step(x, a):
                        return x * a

                    def loss(a):
                        x = 2
                        for i in range(3):
                            x = step(x, a)
                        return math.l2_loss(x)

                    math.assert_close(2916, math.gradient(loss, 'a', get_output=False)(tensor(3.)))
