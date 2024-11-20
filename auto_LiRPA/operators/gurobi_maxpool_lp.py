
"""Optimizable upper MaxPool relaxation using torch.autograd.Function for differentiating through LP solved by gurobi"""
from .base import *
from .activation_base import BoundOptimizableActivation
import numpy as np
from .solver_utils import grb
from tqdm import tqdm


class BatchedGurobiLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, A, b):
        """
        Args:
            c: Batch of objective coefficients, shape (batch_size, n_vars).
            A: Batch of constraint matrices, shape (batch_size, n_constraints, n_vars).
            b: Batch of constraint bounds, shape (batch_size, n_constraints).
        """
        batch_size = c.size(0)
        optimal_values = []
        solutions = []
        
        for i in range(batch_size):
            # Convert each batch item to numpy for Gurobi
            c_np = c[i].detach().cpu().numpy()
            A_np = A[i].detach().cpu().numpy()
            b_np = b[i].detach().cpu().numpy()
            
            # Set up the Gurobi model
            model = grb.Model('model')
            n_vars = c_np.shape[0]
            # by default gurobi vars are >= 0, no need to add lower bound!!!
            vars = model.addMVar(shape=n_vars, lb=-float('inf'))
            
            # Set objective
            model.setObjective(c_np @ vars, grb.GRB.MINIMIZE)
            
            # Add constraints
            model.addMConstr(A_np, vars, grb.GRB.LESS_EQUAL, b_np)

            # set method for solving to DualSimplex (hoping that this will be good for numerical precision)
            # see https://www.gurobi.com/documentation/current/refman/choosing_the_right_algorit.html
            model.setParam("Method", 1)
            
            # Optimize the model
            model.optimize()
            
            # Collect the optimal objective value and solution
            optimal_values.append(model.objVal)
            solutions.append(vars.X)
        
        # Convert lists to tensors
        optimal_values = torch.tensor(optimal_values, dtype=c.dtype, requires_grad=True)
        ctx.save_for_backward(c, A, b)
        ctx.solutions = solutions  # Save the solutions for backward
        
        return optimal_values

    @staticmethod
    def backward(ctx, grad_output):
        c, A, b = ctx.saved_tensors
        solutions = ctx.solutions
        batch_size = c.size(0)
        
        # Initialize gradients
        grad_c = torch.zeros_like(c, dtype=c.dtype)
        grad_A = grad_b = None  # No gradients for A and b in this example

        # Compute gradient w.r.t. each element of c in the batch
        for i in range(batch_size):
            grad_c[i] = torch.tensor(solutions[i], dtype=torch.float32)

        # Scale gradient by grad_output
        return grad_c * grad_output.view(-1, 1), grad_A, grad_b


class BatchedGurobiMaxPoolLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, ls, us, upper_violation):
        """
        Args:
            c: Batch of objective coefficients, shape (batch_size, n_vars).
            ls: Batch of concrete lower bounds for input neurons, shape (batch_size, n_vars)
            us: Batch of concrete upper bounds for input neurons, shape (batch_size, n_vars)
            upper_violation: (bool) whether to compute violation for the upper or lower bound
        """
        batch_size = c.size(0)
        optimal_values = []
        solutions = []

        #print("cs = ", c)
        #print("ls = ", ls)
        #print("us = ", us)

        with grb.Env(empty=True) as env:  
            env.setParam('OutputFlag', 0)
            env.start()

            for i in tqdm(range(batch_size)):
                # Convert each batch item to numpy for Gurobi
                c_np = c[i].detach().cpu().numpy()
                l_np = ls[i].detach().cpu().numpy()
                u_np = us[i].detach().cpu().numpy()
                
                n_vars = c_np.shape[0]
                optimal_values_i = []
                solutions_i = []
                
                for j in range(n_vars):
                    # solve under constraint that j is max
                    if u_np[j] < np.max(l_np):
                        # x_j cannot be max
                        continue
                    
                    # Set up the Gurobi model
                    model = grb.Model('model', env)
                    # by default gurobi vars are >= 0, no need to add lower bound!!!
                    x = model.addMVar(shape=n_vars, lb=l_np, ub=u_np)

                    # Set objective
                    if upper_violation:
                        # max f(x) - u(x) = -min u(x) - f(x) = -min u(x) - x_j if j is max
                        c_j = c_np.copy()
                        c_j[j] -= 1
                    else:
                        # min f(x) - l(x) = min x_j - l(x) if j is max
                        c_j = -c_np
                        c_j[j] += 1
                        
                    model.setObjective(c_j @ x, grb.GRB.MINIMIZE)

                    # Add constraints
                    non_max_idxs = [k for k in range(n_vars) if k != j]
                    model.addConstr(x[j] >= x[non_max_idxs], f"x_{j} is max")

                    # set method for solving to DualSimplex (hoping that this will be good for numerical precision)
                    # see https://www.gurobi.com/documentation/current/refman/choosing_the_right_algorit.html
                    model.setParam("Method", 1)

                    # Optimize the model
                    model.optimize()

                    # Collect the optimal objective value and solution
                    optimal_values_i.append(model.objVal)
                    solutions_i.append(x.X)
                    
                opt_idx = np.argmin(optimal_values_i) if len(optimal_values_i) > 0 else -1
                optimal_value = optimal_values_i[opt_idx] if opt_idx >= 0 else np.inf
                solution = solutions_i[opt_idx] if opt_idx >= 0 else np.zeros_like(l_np)
                
                optimal_values.append(optimal_value)
                solutions.append(solution)
        
        # Convert lists to tensors
        optimal_values = torch.tensor(optimal_values, dtype=c.dtype, requires_grad=True)
        ctx.save_for_backward(c)
        ctx.solutions = solutions  # Save the solutions for backward

        #print("opt: ", optimal_values)
        
        return optimal_values

    @staticmethod
    def backward(ctx, grad_output):
        (c,) = ctx.saved_tensors
        solutions = ctx.solutions
        batch_size = c.size(0)
        
        # Initialize gradients
        grad_c = torch.zeros_like(c, dtype=c.dtype)
        grad_ls = grad_us = grad_upper_violation = None  # No gradients for ls and us in this example

        # Compute gradient w.r.t. each element of c in the batch
        for i in range(batch_size):
            grad_c[i] = torch.tensor(solutions[i], dtype=torch.float32)

        # Scale gradient by grad_output
        # TODO: gradients for lbs and ubs (or maybe just straight through estimator instead of None?)
        return grad_c * grad_output.view(-1, 1), grad_ls, grad_us, grad_upper_violation

    

def get_constr_mat_torch(max_idx, lb, ub):
    """
    Compute LP constraints matrix and bias, s.t. 
    x_{max_idx} >= x_i \forall i != max_idx and
    x_i \in [l_i, u_i] \forall i

    args:
        max_idx (int) - the index of the maximum input
        lb (vec) - vector of lower bounds
        ub (vec) - vector of upper bounds

    returns:
        A, b (Matrix, vec)
    """
    n_in = len(lb)
    idxs = [i for i in range(n_in) if i != max_idx]

    # (n_in - 1) max constraints
    # n_in constraints each for lbs and ubs 
    n_ineqs = (n_in - 1) + 2*n_in
    A = torch.zeros((n_ineqs, n_in), dtype=lb.dtype)
    b = torch.zeros(n_ineqs, dtype=lb.dtype)
    A[:n_in-1,idxs] = torch.eye(n_in - 1)
    A[:n_in-1,max_idx] = -1

    A[n_in-1:2*n_in-1] = -torch.eye(n_in)
    b[n_in-1:2*n_in-1] = -lb
    
    A[2*n_in-1:] = torch.eye(n_in)
    b[2*n_in-1:] = ub

    return A, b


def compute_maxpool_bias(lower, upper, upper_d):
    """
    Computes biases for layer of maxpool neurons.

    args:
        lower (batch x out_channels x w x h) - concrete input lower bounds
        upper (batch x out_channels x w x h) - concrete input upper bounds
        upper_d (batch x out_channels x w x h) - slope value in upper relaxation for every input

    returns:
        upper_b (batch x out_channels) - bias for maxpool relaxation
    """
    # reshape to [out_channels * batch, w*h]
    # so each maxpool neuron has one row
    ls = lower.flatten(-2)
    ls = ls.view(-1, ls.shape[-1])
    us = upper.flatten(-2)
    us = us.view(-1, us.shape[-1])
    cs = upper_d.flatten(-2)
    cs = cs.view(-1, cs.shape[-1])

    # we only need the violation for the upper relaxation, so upper_violation=True
    opt_vals = -BatchedGurobiMaxPoolLP.apply(cs, ls, us, True)

    # reshape to (batch x out_channels)
    return opt_vals.view(lower.shape[:2])