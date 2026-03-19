import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import osqp
from scipy.sparse import csc_matrix, eye, kron


@dataclass
class LiPoConfig:
    chunk_size: int = 100
    blending_horizon: int = 10
    action_dim: int = 7
    len_time_delay: int = 0
    dt: float = 0.0333
    epsilon_blending: float = 0.02
    epsilon_path: float = 0.003
    osqp_eps_abs: float = 1e-4
    osqp_eps_rel: float = 1e-4
    osqp_max_iter: int = 8000


class LiPoSolver(ABC):
    def __init__(self, config: LiPoConfig):        
        self.config = config
        self.N = config.chunk_size
        self.B = config.blending_horizon
        self.D = config.action_dim
        self.TD = config.len_time_delay
        self.dt = config.dt
        self.epsilon_blending = config.epsilon_blending
        self.epsilon_path = config.epsilon_path
        self.JM = 3  # margin for jerk calculation
        self.M = self.N + self.JM  # previous + 3 to consider previous vel/acc/jrk
        self.log = []

    def _build_jerk_operator(self):
        jerk_matrix = np.zeros((self.M, self.M), dtype=np.float64)
        for i in range(max(self.N - 2, 0)):
            jerk_matrix[i, i] = -1.0
            jerk_matrix[i, i + 1] = 3.0
            jerk_matrix[i, i + 2] = -3.0
            jerk_matrix[i, i + 3] = 1.0
        return jerk_matrix / (self.dt ** 3)

    def _build_ref_array(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        ref = np.zeros((self.M, self.D), dtype=np.float64)
        blend_len = len_past_actions

        ref[self.JM:] = actions

        if blend_len > 0:
            # update last actions
            ref[: self.JM + self.TD] = past_actions[
                -blend_len - self.JM : -blend_len + self.TD
            ].copy()

            interp_len = blend_len - self.TD
            if interp_len > 0:
                ratio_space = np.linspace(0.0, 1.0, interp_len, dtype=np.float64)
                ref[self.JM + self.TD : blend_len + self.JM] = (
                    ratio_space[:, None] * actions[self.TD : blend_len]
                    + (1.0 - ratio_space)[:, None] * past_actions[-blend_len + self.TD :]
                )
        else: # blend_len == 0
            # update last actions
            ref[: self.JM] = actions[0]

        return ref

    def get_log(self):
        return self.log

    def reset_log(self):
        self.log = []

    def print_solved_times(self):
        if not self.log:
            print("No logs available.")
            return

        finite_times = np.array(
            [entry["time"] for entry in self.log if np.isfinite(entry["time"])],
            dtype=float,
        )
        num_logs = len(self.log)
        print(f"Number of logs: {num_logs}")
        if finite_times.size == 0:
            print("Average solved time: unavailable (no finite solve times logged)")
            return

        avg_time = np.mean(finite_times)
        std_time = np.std(finite_times)
        print(f"Average solved time: {avg_time:.4f} seconds, Std: {std_time:.4f} seconds")

    @abstractmethod
    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        pass


class CvxpySolver(LiPoSolver):
    def __init__(self, config: LiPoConfig):
        super().__init__(config)

        self.epsilon_var = cp.Variable((self.M, self.D))
        self.ref_param = cp.Parameter((self.M, self.D), value=np.zeros((self.M, self.D), dtype=np.float64))

        D_j = self._build_jerk_operator()
        q_total = self.epsilon_var + self.ref_param  # (N, D)
        cost = cp.sum([cp.sum_squares(D_j @ q_total[:, d]) for d in range(self.D)])

        constraints = []
        constraints += [self.epsilon_var[self.B + self.JM :] <= self.epsilon_path]
        constraints += [self.epsilon_var[self.B + self.JM :] >= -self.epsilon_path]
        constraints += [self.epsilon_var[self.JM + 1 + self.TD : self.B + self.JM] <= self.epsilon_blending]
        constraints += [self.epsilon_var[self.JM + 1 + self.TD : self.B + self.JM] >= -self.epsilon_blending]
        constraints += [self.epsilon_var[0 : self.JM + 1 + self.TD] == 0.0]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

        self.ref = np.zeros((self.M, self.D), dtype=np.float64)
        self.epsilon = np.zeros((self.M, self.D), dtype=np.float64)
        self.solved = np.zeros((self.M, self.D), dtype=np.float64)  

        # Initialize the problem & warm up
        self.problem.solve(warm_start=True, verbose=False, solver=cp.CLARABEL, time_limit=0.05)

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        self.ref[:] = self._build_ref_array(actions, past_actions, len_past_actions)
        self.ref_param.value[:] = self.ref

        t0 = time.perf_counter()
        try:
            self.problem.solve(warm_start=True, verbose=False, solver=cp.CLARABEL, time_limit=0.05)
        except Exception as e:
            return None, e
        t1 = time.perf_counter()

        if self.epsilon_var.value is None:
            return None, RuntimeError("cvxpy(CLARABEL) solve failed")

        self.epsilon[:] = self.epsilon_var.value
        np.add(self.epsilon, self.ref, out=self.solved)

        self.log.append(
            {
                "time": t1 - t0,
                "epsilon": self.epsilon.copy(),
                "ref": self.ref.copy(),
                "solved": self.solved.copy(),
            }
        )

        return self.solved[self.JM :].copy(), self.ref[self.JM :].copy()


class OsqpSolver(LiPoSolver):
    def __init__(self, config: LiPoConfig):
        super().__init__(config)

        self.num_vars = self.M * self.D
        self.ref = np.zeros((self.M, self.D), dtype=np.float64)
        self.epsilon = np.zeros((self.M, self.D), dtype=np.float64)
        self.solved = np.zeros((self.M, self.D), dtype=np.float64)
        self.linear_coeffs = np.zeros((self.M, self.D), dtype=np.float64)

        self.D_j = self._build_jerk_operator()
        self.jerk_hessian = self.D_j.T @ self.D_j

        lower_bounds, upper_bounds = self._build_box_bounds()

        P = kron(
            csc_matrix(2.0 * self.jerk_hessian),
            eye(self.D, format="csc"),
            format="csc",
        )
        A = eye(self.num_vars, format="csc")

        self.problem = osqp.OSQP()
        self.problem.setup(
            P=P,
            q=np.zeros(self.num_vars, dtype=np.float64),
            A=A,
            l=lower_bounds,
            u=upper_bounds,
            verbose=False,
            polish=False,
            warm_start=True,
            adaptive_rho=True,
            eps_abs=config.osqp_eps_abs,
            eps_rel=config.osqp_eps_rel,
            max_iter=config.osqp_max_iter,
        )

        # Initialize the problem & warm up
        self.problem.solve()

    def _build_box_bounds(self):
        row_lb = np.full(self.M, -self.epsilon_path, dtype=np.float64)
        row_ub = np.full(self.M, self.epsilon_path, dtype=np.float64)

        row_lb[self.JM + 1 + self.TD : self.B + self.JM] = -self.epsilon_blending
        row_ub[self.JM + 1 + self.TD : self.B + self.JM] = self.epsilon_blending
        row_lb[: self.JM + 1 + self.TD] = 0.0
        row_ub[: self.JM + 1 + self.TD] = 0.0

        lower_bounds = np.repeat(row_lb, self.D)
        upper_bounds = np.repeat(row_ub, self.D)

        return lower_bounds, upper_bounds

    def _linear_objective_coeffs_from_ref(self, ref: np.ndarray):
        self.linear_coeffs[:] = self.jerk_hessian @ ref
        self.linear_coeffs *= 2.0
        return self.linear_coeffs.reshape(-1)

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        self.ref[:] = self._build_ref_array(actions, past_actions, len_past_actions)
        objective_coeffs = self._linear_objective_coeffs_from_ref(self.ref)

        t0 = time.perf_counter()
        try:
            self.problem.update(q=objective_coeffs)
            result = self.problem.solve()
        except Exception as e:
            return None, e
        t1 = time.perf_counter()

        status = str(result.info.status)
        if result.x is None or status not in {"solved", "solved inaccurate"}:
            return None, RuntimeError(f"osqp solve failed (status={status})")

        self.epsilon[:] = result.x.reshape(self.M, self.D)
        np.add(self.epsilon, self.ref, out=self.solved)

        self.log.append(
            {
                "time": t1 - t0,
                "epsilon": self.epsilon.copy(),
                "ref": self.ref.copy(),
                "solved": self.solved.copy(),
            }
        )

        return self.solved[self.JM :].copy(), self.ref[self.JM :].copy()


def _create_solver(name: str, config: LiPoConfig):
    registry = {
        "CVXPY": CvxpySolver,
        "OSQP": OsqpSolver,
    }

    solver_cls = registry.get(name.upper())
    if solver_cls is None:
        raise ValueError(f"Unsupported solver: {name}. Supported solvers are: {[k.lower() for k in registry.keys()]}")
    return solver_cls(config)


class ActionLiPo:
    def __init__(
        self,
        solver="osqp",
        chunk_size=100,
        blending_horizon=10,
        action_dim=7,
        len_time_delay=0,
        dt=0.0333,
        epsilon_blending=0.02,
        epsilon_path=0.003,
        osqp_eps_abs=1e-4,
        osqp_eps_rel=1e-4,
        osqp_max_iter=8000,
    ):
        """
        ActionLiPo (Action Lightweight Post-Optimizer) for action optimization.      
        Parameters:
        - solver: The solver to use for the optimization problem. Options include "cvxpy" and "osqp".
        - chunk_size: The size of the action chunk to optimize.
        - blending_horizon: The number of actions to blend with past actions.
        - action_dim: The dimension of the action space.
        - len_time_delay: The length of the time delay for the optimization.
        - dt: Time step for the optimization.
        - epsilon_blending: Epsilon value for blending actions.
        - epsilon_path: Epsilon value for path actions.
        """
        
        self.solver = solver.upper()
        self.config = LiPoConfig(
            chunk_size=chunk_size,
            blending_horizon=blending_horizon,
            action_dim=action_dim,
            len_time_delay=len_time_delay,
            dt=dt,
            epsilon_blending=epsilon_blending,
            epsilon_path=epsilon_path,
            osqp_eps_abs=osqp_eps_abs,
            osqp_eps_rel=osqp_eps_rel,
            osqp_max_iter=osqp_max_iter,
        )
        self.strategy = _create_solver(self.solver, self.config)

        self.N = self.strategy.N
        self.B = self.strategy.B
        self.D = self.strategy.D
        self.TD = self.strategy.TD
        self.dt = self.strategy.dt
        self.epsilon_blending = self.strategy.epsilon_blending
        self.epsilon_path = self.strategy.epsilon_path
        self.JM = self.strategy.JM 
        self.M = self.strategy.M

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        return self.strategy.solve(actions, past_actions, len_past_actions)

    def get_log(self):
        return self.strategy.get_log()

    def reset_log(self):
        self.strategy.reset_log()

    def print_solved_times(self):
        self.strategy.print_solved_times()