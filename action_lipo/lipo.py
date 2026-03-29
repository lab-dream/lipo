import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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
    numpy_admm_iters: int = 25


class NumpyADMMSolverInitError(RuntimeError):
    """Raised when the specialized numpy ADMM solver is unsafe to initialize."""


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
        import cvxpy as cp

        super().__init__(config)

        self._clarabel = cp.CLARABEL

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
        self.problem.solve(warm_start=True, verbose=False, solver=self._clarabel, time_limit=0.05)

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        self.ref[:] = self._build_ref_array(actions, past_actions, len_past_actions)
        self.ref_param.value[:] = self.ref

        t0 = time.perf_counter()
        try:
            self.problem.solve(warm_start=True, verbose=False, solver=self._clarabel, time_limit=0.05)
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
        import osqp
        from scipy.sparse import csc_matrix, eye, kron

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


class NumpyADMMSolver(LiPoSolver):
    """Over-relaxed ADMM solver for box-constrained jerk-minimization QP.

    Exploits the fixed problem structure: pre-computes inv(P + rho*I) at init,
    then each solve is just matrix multiplies and clips, vectorized across all
    action dimensions simultaneously via (M, D) shaped operations.

    Over-relaxation (alpha > 1) accelerates convergence near active box
    constraints, which is critical for the blending zone.

    Safety features:
    - Init-time stability checks (reject unstable or degenerate configs)
    - Post-solve convergence check via primal residual + iterate delta
    - Automatic batched iteration extension on non-convergence
    - Returns (None, error) if convergence fails after extension
    """

    _ALPHA = 1.7  # over-relaxation parameter (theoretically stable for alpha in (0, 2))

    def __init__(self, config: LiPoConfig):
        if config.numpy_admm_iters < 1:
            raise ValueError("numpy_admm_iters must be >= 1")

        super().__init__(config)

        D_j = self._build_jerk_operator()
        # P = 2 * D_j^T @ D_j  (the QP Hessian in 0.5 x^T P x + q^T x form)
        self._P = 2.0 * (D_j.T @ D_j)

        # ADMM penalty: rho = max_eigenvalue(P) / 50
        # Lower rho improves constraint-bound convergence in the blending zone
        eigvals = np.linalg.eigvalsh(self._P)
        lam_max = float(eigvals[-1])

        # Degenerate problems (for example chunk_size < 3) do not admit the
        # specialized inverse-based path and should use the generic solver.
        if lam_max <= 0.0:
            raise NumpyADMMSolverInitError(
                "NumpyADMMSolver: jerk Hessian is degenerate for this configuration. "
                "Use the OSQP solver instead."
            )

        # Numerical stability check: detect P overflow risk
        if not np.isfinite(lam_max) or lam_max > 1e30:
            raise NumpyADMMSolverInitError(
                f"NumpyADMMSolver: P matrix has extreme eigenvalues (lam_max={lam_max:.1e}). "
                f"dt={self.dt} may be too small. Consider increasing dt."
            )

        self._rho = lam_max / 50.0

        # Pre-compute inverse of (P + rho*I) — O(M^3) once at init
        K = self._P + self._rho * np.eye(self.M, dtype=np.float64)
        try:
            self._K_inv = np.ascontiguousarray(np.linalg.inv(K), dtype=np.float64)
        except np.linalg.LinAlgError as exc:
            raise NumpyADMMSolverInitError(
                "NumpyADMMSolver: failed to invert P + rho*I. Use the OSQP solver instead."
            ) from exc
        self._P = np.ascontiguousarray(self._P, dtype=np.float64)

        # Numerical stability check: verify inverse accuracy
        inv_error = float(np.max(np.abs(K @ self._K_inv - np.eye(self.M))))
        if inv_error > 1e-6:
            raise NumpyADMMSolverInitError(
                f"NumpyADMMSolver: matrix inverse inaccurate (error={inv_error:.1e}). "
                f"Consider increasing dt or reducing chunk_size."
            )

        self._batch_iters = config.numpy_admm_iters
        self._max_total_iters = max(100, 4 * self._batch_iters)
        self._primal_tol = max(
            1e-6,
            min(0.1 * config.epsilon_blending, 0.5 * config.epsilon_path),
        )
        self._step_tol = max(
            1e-6,
            min(0.05 * config.epsilon_blending, config.epsilon_path / 6.0),
        )

        # Box bounds as (M, 1) for broadcasting over D columns
        lb = np.full(self.M, -config.epsilon_path, dtype=np.float64)
        ub = np.full(self.M, config.epsilon_path, dtype=np.float64)
        lb[self.JM + 1 + self.TD : self.B + self.JM] = -config.epsilon_blending
        ub[self.JM + 1 + self.TD : self.B + self.JM] = config.epsilon_blending
        lb[: self.JM + 1 + self.TD] = 0.0
        ub[: self.JM + 1 + self.TD] = 0.0
        self._lb2d = lb[:, None]
        self._ub2d = ub[:, None]

        # Pre-allocated working memory
        self.ref = np.zeros((self.M, self.D), dtype=np.float64)
        self.solved = np.zeros((self.M, self.D), dtype=np.float64)
        self._q = np.zeros((self.M, self.D), dtype=np.float64)
        self._rhs = np.zeros((self.M, self.D), dtype=np.float64)
        self._x = np.zeros((self.M, self.D), dtype=np.float64)
        self._x_hat = np.zeros((self.M, self.D), dtype=np.float64)
        self._z = np.zeros((self.M, self.D), dtype=np.float64)
        self._z_prev = np.zeros((self.M, self.D), dtype=np.float64)
        self._u = np.zeros((self.M, self.D), dtype=np.float64)
        self._last_primal_residual = np.inf
        self._last_step_residual = np.inf
        self._last_total_iters = 0

        # Warm-up (trigger numpy internal caches)
        self._admm_solve()

    def _admm_iterations(self, n_iters):
        """Run n_iters of over-relaxed ADMM without resetting state."""
        z = self._z
        u = self._u
        rhs = self._rhs
        x = self._x
        x_hat = self._x_hat
        z_prev = self._z_prev
        rho = self._rho
        alpha = self._ALPHA
        K_inv = self._K_inv
        lb = self._lb2d
        ub = self._ub2d
        q = self._q

        for _ in range(n_iters):
            np.subtract(z, u, out=rhs)
            rhs *= rho
            np.subtract(rhs, q, out=rhs)

            np.dot(K_inv, rhs, out=x)

            np.multiply(x, alpha, out=x_hat)
            x_hat += (1.0 - alpha) * z

            z_prev[:] = z
            np.add(x_hat, u, out=z)
            np.clip(z, lb, ub, out=z)

            u += x_hat
            u -= z

    def _residuals(self):
        """Return convergence metrics for the latest ADMM iterate."""
        primal = float(np.max(np.abs(self._x - self._z)))
        step = float(np.max(np.abs(self._z - self._z_prev)))
        return primal, step

    def _is_converged(self, primal_residual, step_residual):
        return primal_residual <= self._primal_tol and step_residual <= self._step_tol

    def _admm_solve(self):
        """Cold-start ADMM with batched convergence checks."""
        self._z[:] = 0.0
        self._z_prev[:] = 0.0
        self._u[:] = 0.0

        primal_residual = np.inf
        step_residual = np.inf
        total_iters = 0
        while total_iters < self._max_total_iters:
            n_iters = min(self._batch_iters, self._max_total_iters - total_iters)
            self._admm_iterations(n_iters)
            total_iters += n_iters

            primal_residual, step_residual = self._residuals()
            if self._is_converged(primal_residual, step_residual):
                break

        self._last_primal_residual = primal_residual
        self._last_step_residual = step_residual
        self._last_total_iters = total_iters
        return primal_residual, step_residual, total_iters

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        self.ref[:] = self._build_ref_array(actions, past_actions, len_past_actions)

        # Linear objective: q = P @ ref (vectorized for all dims)
        np.dot(self._P, self.ref, out=self._q)

        t0 = time.perf_counter()
        primal_residual, step_residual, total_iters = self._admm_solve()
        t1 = time.perf_counter()

        if not self._is_converged(primal_residual, step_residual):
            return None, RuntimeError(
                f"numpy ADMM did not converge "
                f"(primal_residual={primal_residual:.6f}, step_residual={step_residual:.6f}, "
                f"iters={total_iters}/{self._max_total_iters})"
            )

        # solved = epsilon (z) + ref
        np.add(self._z, self.ref, out=self.solved)

        self.log.append(
            {
                "time": t1 - t0,
                "epsilon": self._z.copy(),
                "ref": self.ref.copy(),
                "solved": self.solved.copy(),
                "primal_residual": primal_residual,
                "step_residual": step_residual,
                "iterations": total_iters,
            }
        )

        return self.solved[self.JM :].copy(), self.ref[self.JM :].copy()


def _create_solver(name: str, config: LiPoConfig):
    registry = {
        "CVXPY": CvxpySolver,
        "OSQP": OsqpSolver,
        "NUMPY": NumpyADMMSolver,
    }

    solver_cls = registry.get(name.upper())
    if solver_cls is None:
        raise ValueError(f"Unsupported solver: {name}. Supported solvers are: {[k.lower() for k in registry.keys()]}")
    return solver_cls(config)


class ActionLiPo:
    def __init__(
        self,
        solver="numpy",
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
        numpy_admm_iters=25,
    ):
        """
        ActionLiPo (Action Lightweight Post-Optimizer) for action optimization.      
        Parameters:
        - solver: The solver to use. Options: "numpy" (default), "osqp", "cvxpy".
        - chunk_size: The size of the action chunk to optimize.
        - blending_horizon: The number of actions to blend with past actions.
        - action_dim: The dimension of the action space.
        - len_time_delay: The length of the time delay for the optimization.
        - dt: Time step for the optimization.
        - epsilon_blending: Epsilon value for blending actions.
        - epsilon_path: Epsilon value for path actions.
        - numpy_admm_iters: Number of ADMM iterations for the numpy solver.
        """
        
        self.solver = solver.upper()
        self._log = []
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
            numpy_admm_iters=numpy_admm_iters,
        )
        if self.solver == "NUMPY":
            try:
                self.strategy = _create_solver("NUMPY", self.config)
            except NumpyADMMSolverInitError:
                self.solver = "OSQP"
                self.strategy = self._create_osqp_solver(
                    "NumpyADMMSolver could not be initialized safely"
                )
        else:
            self.strategy = _create_solver(self.solver, self.config)
        self._fallback = None  # lazy-initialized OSQP fallback

        self.N = self.strategy.N
        self.B = self.strategy.B
        self.D = self.strategy.D
        self.TD = self.strategy.TD
        self.dt = self.strategy.dt
        self.epsilon_blending = self.strategy.epsilon_blending
        self.epsilon_path = self.strategy.epsilon_path
        self.JM = self.strategy.JM
        self.M = self.strategy.M

    def _create_osqp_solver(self, context: str):
        try:
            return _create_solver("OSQP", self.config)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"{context}. OSQP fallback requires optional dependencies; "
                f"install with `pip install action-lipo[osqp]`."
            ) from exc

    def _append_last_log_entry(self, solver_name: str, strategy: LiPoSolver):
        log = strategy.get_log()
        if not log:
            return
        entry = log[-1].copy()
        entry["solver"] = solver_name
        self._log.append(entry)

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        result = self.strategy.solve(actions, past_actions, len_past_actions)

        if result[0] is not None:
            self._append_last_log_entry(self.solver, self.strategy)
            return result

        if self.solver != "NUMPY":
            return result

        # NUMPY solver failed — fallback to OSQP
        if self._fallback is None:
            try:
                self._fallback = self._create_osqp_solver("numpy ADMM fallback is unavailable")
            except RuntimeError as exc:
                return None, RuntimeError(f"{result[1]}; {exc}")

        fallback_result = self._fallback.solve(actions, past_actions, len_past_actions)
        if fallback_result[0] is not None:
            self._append_last_log_entry("OSQP", self._fallback)
        return fallback_result

    def get_log(self):
        return self._log

    def reset_log(self):
        self._log = []
        self.strategy.reset_log()
        if self._fallback is not None:
            self._fallback.reset_log()

    def print_solved_times(self):
        if not self._log:
            print("No logs available.")
            return

        finite_times = np.array(
            [entry["time"] for entry in self._log if np.isfinite(entry["time"])],
            dtype=float,
        )
        num_logs = len(self._log)
        print(f"Number of logs: {num_logs}")
        if finite_times.size == 0:
            print("Average solved time: unavailable (no finite solve times logged)")
            return

        avg_time = np.mean(finite_times)
        std_time = np.std(finite_times)
        print(f"Average solved time: {avg_time:.4f} seconds, Std: {std_time:.4f} seconds")
