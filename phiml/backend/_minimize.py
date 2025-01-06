from threading import Barrier

import numpy

from ._backend import Backend, SolveResult, DType, ML_LOGGER
from ._dtype import BOOL, INT32
from ._linalg import _max_iter


def scipy_minimize(self, method: str, f, x0, atol, max_iter, trj: bool):
    from scipy.optimize import OptimizeResult, minimize
    from threading import Thread

    assert self.supports(Backend.jacobian)
    x0 = self.numpy(x0)
    assert x0.ndim == 2  # (batch, parameters)
    atol = self.numpy(atol)
    batch_size = x0.shape[0]
    fg = self.jacobian(f, [0], get_output=True, is_f_scalar=True)
    method_description = f"SciPy {method} with {self.name}"

    iterations = [0] * batch_size
    function_evaluations = [0] * batch_size
    xs = [None] * batch_size
    final_losses = [None] * batch_size
    converged = [False] * batch_size
    diverged = [False] * batch_size
    messages = [""] * batch_size

    f_inputs = [None] * batch_size
    f_b_losses = None
    f_b_losses_np = None
    f_grad_np = None
    f_input_available = Barrier(batch_size + 1)
    f_output_available = Barrier(batch_size + 1)
    finished = [False] * batch_size
    all_finished = False
    trajectories = [[] for _ in range(batch_size)] if trj else None
    threads = []

    for b in range(batch_size):  # Run each independent example as a scipy minimization in a new thread

        def b_thread(b=b):
            recent_b_losses = []

            def b_fun(x: numpy.ndarray):
                function_evaluations[b] += 1
                f_inputs[b] = self.as_tensor(x, convert_external=True)
                f_input_available.wait()
                f_output_available.wait()
                recent_b_losses.append(f_b_losses[b])
                if final_losses[b] is None:  # first evaluation
                    final_losses[b] = f_b_losses[b]
                    if trajectories is not None:
                        trajectories[b].append(SolveResult(method_description, x0[b], self.numpy(f_b_losses[b]), 0, 1, False, False, ""))
                return f_b_losses_np[b], f_grad_np[b]

            def callback(x, *args):  # L-BFGS-B only passes x but the documentation says (x, state)
                iterations[b] += 1
                loss = min(recent_b_losses)
                recent_b_losses.clear()
                final_losses[b] = loss
                if trajectories is not None:
                    trajectories[b].append(SolveResult(method_description, x, self.numpy(loss), iterations[b], function_evaluations[b], False, False, ""))

            res = minimize(fun=b_fun, x0=x0[b], jac=True, method=method, tol=atol[b], options={'maxiter': max_iter[b]}, callback=callback)
            assert isinstance(res, OptimizeResult)
            # res.nit, res.nfev
            xs[b] = res.x
            converged[b] = res.success
            diverged[b] = res.status not in (0, 1)  # 0=success
            messages[b] = res.message
            finished[b] = True
            while not all_finished:
                f_input_available.wait()
                f_output_available.wait()

        b_thread = Thread(target=b_thread)
        threads.append(b_thread)
        b_thread.start()

    while True:
        f_input_available.wait()
        if all(finished):
            all_finished = True
            f_output_available.wait()
            break
        f_b_losses, f_grad = fg(self.stack(f_inputs))  # Evaluate function and gradient
        f_b_losses_np = self.numpy(f_b_losses).astype(numpy.float64)
        f_grad_np = self.numpy(f_grad).astype(numpy.float64)
        f_output_available.wait()

    for b_thread in threads:
        b_thread.join()  # make sure threads exit correctly

    if trj:
        max_trajectory_length = max([len(t) for t in trajectories])
        last_points = [SolveResult(method_description, xs[b], self.numpy(final_losses[b]), iterations[b], function_evaluations[b], converged[b], diverged[b], "") for b in range(batch_size)]
        trajectories = [t[:-1] + [last_point] * (max_trajectory_length - len(t) + 1) for t, last_point in zip(trajectories, last_points)]
        trajectory = []
        for states in zip(*trajectories):
            x = numpy.stack([state.x for state in states])
            residual = numpy.stack([state.residual for state in states])
            iterations = [state.iterations for state in states]
            function_evaluations = [state.function_evaluations for state in states]
            converged = [state.converged for state in states]
            diverged = [state.diverged for state in states]
            trajectory.append(SolveResult(method_description, x, residual, iterations, function_evaluations, converged, diverged, messages))
        return trajectory
    else:
        x = self.stack(xs)
        residual = self.stack(final_losses)
        return SolveResult(method_description, x, residual, iterations, function_evaluations, converged, diverged, messages)


def gradient_descent(self: Backend, f, x0, atol, max_iter, trj: bool, step_size='adaptive', max_substeps=10):
    assert self.supports(Backend.jacobian)
    assert len(self.staticshape(x0)) == 2  # (batch, parameters)
    batch_size = self.staticshape(x0)[0]
    fg = self.jacobian(f, [0], get_output=True, is_f_scalar=True)
    method = f"Gradient descent with {self.name}"

    iterations = self.zeros([batch_size], INT32)
    function_evaluations = self.ones([batch_size], INT32)

    adaptive_step_size = step_size == 'adaptive'
    if adaptive_step_size:
        step_size = self.zeros([batch_size]) + 0.1

    loss, grad = fg(x0)  # Evaluate function and gradient
    diverged = self.any(~self.isfinite(x0), axis=(1,))
    converged = self.zeros([batch_size], BOOL)
    trajectory = [SolveResult(method, self.numpy(x0), self.numpy(loss), self.numpy(iterations), self.numpy(function_evaluations), self.numpy(converged), self.numpy(diverged), [""] * batch_size)] if trj else None
    max_iter_ = self.to_int32(max_iter)
    continue_ = ~converged & ~diverged & (iterations < max_iter_)

    def gd_step(continue_, x, loss, grad, iterations, function_evaluations, step_size, converged, diverged):
        prev_loss, prev_grad, prev_x = loss, grad, x
        continue_1 = self.to_int32(continue_)
        iterations += continue_1
        if adaptive_step_size:
            best_step_size = tested_step_size = step_size
            best_dx = - grad * self.expand_dims(best_step_size * self.to_float(continue_1), -1)
            best_x = x + best_dx
            best_loss, best_grad = fg(best_x); function_evaluations += continue_1
            for i in range(max_substeps):
                # if loss is increasing, reduce step size by parabola factor
                # if loss is decreasing, use next_grad · dx to tell us whether we can decrease the step size or not
                lin_loss_decrease = - self.sum(grad * best_dx, -1)  # >= 0, by much the loss would decrease if the function was linear
                ML_LOGGER.debug(f"Gradient: {self.numpy(best_grad)} with step_size={self.numpy(best_step_size)}")
                actual_loss_decrease = loss - best_loss  # we want > 0
                # we want actual_loss_decrease to be at least half of predicted_loss_decrease
                act_pred = self.divide_no_nan(actual_loss_decrease, lin_loss_decrease)  # we want this to be 1/2
                ML_LOGGER.debug(f"Actual/Predicted: {self.numpy(act_pred)}")
                parabola_fac = self.exp((act_pred - 0.5) * 3.)
                parabola_fac = self.maximum(parabola_fac, 0.1)  # no more than 1 order of magnitude adjustment
                gradient_direction_dot = self.sum(best_grad * best_dx, -1) / self.sqrt(self.sum(best_dx * best_dx, -1))  # gradient direction at next position
                gradient_fac = self.where(gradient_direction_dot < 0, 2, 0.5)
                step_size_fac = self.where(best_loss > loss, parabola_fac, gradient_fac)
                # --- evaluate new step size ---
                next_step_size = tested_step_size * step_size_fac
                ML_LOGGER.debug(f"step_size *= {self.numpy(step_size_fac)} = {self.numpy(next_step_size)}")
                next_dx = - grad * self.expand_dims(next_step_size * self.to_float(continue_1), -1)
                next_x = x + next_dx
                next_loss, next_grad = fg(next_x); function_evaluations += continue_1
                # print(f"\t\tstep_size: {self.numpy(next_step_size)} \t(*{self.numpy(step_size_fac)}) -> loss {self.numpy(next_loss / loss)}")
                # --- use next guess if better ---
                is_worse = next_loss >= best_loss
                best_step_size = self.where(is_worse, best_step_size, next_step_size)
                best_dx = self.where(is_worse[:, None], best_dx, next_dx)
                best_x = self.where(is_worse[:, None], best_x, next_x)
                best_loss = self.where(is_worse, best_loss, next_loss)
                best_grad = self.where(is_worse[:, None], best_grad, next_grad)
                converged = converged | ((self.sum(next_dx ** 2, axis=-1) < atol ** 2) & is_worse)
                diverged |= self.any(~self.isfinite(best_grad), axis=(1,))
                tested_step_size = next_step_size
                done = (best_loss < loss) | converged | diverged
                if self.all(done):
                    ML_LOGGER.info(f"GD iter {self.numpy(iterations).max()}: Finished step_size adjustment after {i + 1} tries. step_size={self.numpy(step_size)}\n")
                    break
                else:
                    ML_LOGGER.info(f"GD iter {self.numpy(iterations).max()} substep {i + 1}: step sizes not yet ready: {(~self.numpy(done)).astype(int)}")
            # else:
            #     converged = converged | (abs(actual_loss_decrease) < predicted_loss_decrease)
            #     ML_LOGGER.debug("Backend._minimize_gradient_descent(): No step size found!\n")
            diverged = diverged | (~converged & (best_loss > loss))
            x, loss, grad, step_size = best_x, best_loss, best_grad, best_step_size
            # print("x:", self.numpy(best_x))
            ML_LOGGER.info(f"GD iter {self.numpy(iterations).max()} new estimate x={x}")
        else:
            x -= grad * self.expand_dims(step_size * self.to_float(continue_1), -1)
            loss, grad = fg(x); function_evaluations += continue_1
            diverged = self.any(~self.isfinite(x), axis=(1,)) | (loss > prev_loss)
            converged = ~diverged & (prev_loss - loss < atol)
        if trj:
            trajectory.append(SolveResult(method, self.numpy(x), self.numpy(loss), self.numpy(iterations), self.numpy(function_evaluations), self.numpy(diverged), self.numpy(converged), [""] * batch_size))
        continue_ = ~converged & ~diverged & (iterations < max_iter_)
        return continue_, x, loss, grad, iterations, function_evaluations, step_size, converged, diverged

    not_converged, x, loss, grad, iterations, function_evaluations, step_size, converged, diverged = self.while_loop(gd_step, (continue_, x0, loss, grad, iterations, function_evaluations, step_size, converged, diverged), int(max(max_iter)))
    if trj:
        trajectory.append(SolveResult(method, self.numpy(x), self.numpy(loss), self.numpy(iterations), self.numpy(function_evaluations) + 1, self.numpy(converged), self.numpy(diverged), [""] * batch_size))
        return trajectory
    else:
        return SolveResult(method, x, loss, iterations, function_evaluations, converged, diverged, [""] * batch_size)
