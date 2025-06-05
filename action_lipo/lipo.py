import time
import numpy as np
import cvxpy as cp

class ActionLiPo:
    def __init__(self, solver="CLARABEL", 
                 chunk_size=100, 
                 blending_horizon=10, 
                 action_dim=7, 
                 len_time_delay=0,
                 dt=0.0333,
                 epsilon_blending=0.02,
                 epsilon_path=0.003):
        """
        ActionLiPo (Action Lightweight Post-Optimizer) for action optimization.      
        Parameters:
        - solver: The solver to use for the optimization problem.
        - chunk_size: The size of the action chunk to optimize.
        - blending_horizon: The number of actions to blend with past actions.
        - action_dim: The dimension of the action space.
        - len_time_delay: The length of the time delay for the optimization.
        - dt: Time step for the optimization.
        - epsilon_blending: Epsilon value for blending actions.
        - epsilon_path: Epsilon value for path actions.
        """

        self.solver = solver
        self.N = chunk_size
        self.B = blending_horizon
        self.D = action_dim
        self.TD = len_time_delay

        self.dt = dt
        self.epsilon_blending = epsilon_blending
        self.epsilon_path = epsilon_path
        
        JM = 3  # margin for jerk calculation
        self.JM = JM
        self.epsilon = cp.Variable((self.N+JM, self.D)) # previous + 3 to consider previous vel/acc/jrk
        self.ref = cp.Parameter((self.N+JM, self.D),value=np.zeros((self.N+JM, self.D))) # previous + 3
        
        D_j = np.zeros((self.N+JM, self.N+JM))
        for i in range(self.N - 2):
            D_j[i, i]     = -1
            D_j[i, i+1]   = 3
            D_j[i, i+2]   = -3
            D_j[i, i+3]   = 1
        D_j = D_j / self.dt**3

        q_total = self.epsilon + self.ref  # (N, D)
        cost = cp.sum([cp.sum_squares(D_j @ q_total[:, d]) for d in range(self.D)])

        constraints = []

        constraints += [self.epsilon[self.B+JM:] <= self.epsilon_path]
        constraints += [self.epsilon[self.B+JM:] >= - self.epsilon_path]
        constraints += [self.epsilon[JM+1+self.TD:self.B+JM] <= self.epsilon_blending]
        constraints += [self.epsilon[JM+1+self.TD:self.B+JM] >= - self.epsilon_blending]
        constraints += [self.epsilon[0:JM+1+self.TD] == 0.0]

        np.set_printoptions(precision=3, suppress=True, linewidth=100)

        self.p = cp.Problem(cp.Minimize(cost), constraints)

        # Initialize the problem & warm up
        self.p.solve(warm_start=True, verbose=False, solver=self.solver, time_limit=0.05)
        
        self.log = []

    def solve(self, actions: np.ndarray, past_actions: np.ndarray, len_past_actions: int):
        """
        Solve the optimization problem with the given actions and past actions.
        Parameters:
        - actions: The current actions to optimize.
        - past_actions: The past actions to blend with.
        - len_past_actions: The number of past actions to consider for blending.
        Returns:
        - solved: The optimized actions after solving the problem.
        - ref: The reference actions used in the optimization.
        """

        blend_len = len_past_actions
        JM = self.JM
        self.ref.value[JM:] = actions.copy()
        
        if blend_len > 0:
            # update last actions
            self.ref.value[:JM+self.TD] = past_actions[-blend_len-JM:-blend_len + self.TD].copy()
            ratio_space = np.linspace(0, 1, blend_len-self.TD) # (B,1)    
            self.ref.value[JM+self.TD:blend_len+JM] = ratio_space[:, None] * actions[self.TD:blend_len] + (1 - ratio_space[:, None]) * past_actions[-blend_len+self.TD:]
        else: # blend_len == 0
            # update last actions
            self.ref.value[:JM] = actions[0]
            
        t0 = time.time()
        try:
            self.p.solve(warm_start=True, verbose=False, solver=self.solver, time_limit=0.05)
        except Exception as e:
            return None, e
        t1 = time.time()

        solved_time = t1 - t0
        self.solved = self.epsilon.value.copy() + self.ref.value.copy()

        self.log.append({
            "time": solved_time,
            "epsilon": self.epsilon.value.copy(),
            "ref": self.ref.value.copy(),
            "solved": self.solved.copy()
        })

        return self.solved[JM:].copy(), self.ref.value[JM:].copy()

    def get_log(self):
        return self.log

    def reset_log(self):
        self.log = []

    def print_solved_times(self):
        if self.log:
            avg_time = np.mean([entry["time"] for entry in self.log])
            std_time = np.std([entry["time"] for entry in self.log])
            num_logs = len(self.log)
            print(f"Number of logs: {num_logs}")
            print(f"Average solved time: {avg_time:.4f} seconds, Std: {std_time:.4f} seconds")
        else:
            print("No logs available.")