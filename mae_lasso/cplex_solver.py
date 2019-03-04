import numpy as np

import cplex


def cplex_solve(X, y, reg_lambda):
    """MAE Regression solver using cplex Python API"""
    n, p = X.shape
    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.minimize)

    # Set display parameters to avoid output being printed to terminal
    model.parameters.barrier.display.set(0)
    model.parameters.simplex.display.set(0)

    # Add variables and objective function
    obj = [0.0]*(p+1) + [1.0]*n + [reg_lambda]*p
    model.variables.add(obj=obj)
    
    # Create epsilon constraints
    rhs = y.tolist()*2
    senses = 'L'*n + 'G'*n

    epsilon_upper = [cplex.SparsePair(ind=list(range(p+1)) + [p+i+1],
                                      val=[1] + row.tolist() + [-1])
                     for i, row in enumerate(X)]
    epsilon_lower = [cplex.SparsePair(ind=list(range(p+1)) + [p+i+1],
                                      val=[1] + row.tolist() + [1])
                     for i, row in enumerate(X)]
    constraints = epsilon_upper + epsilon_lower
    model.linear_constraints.add(lin_expr=constraints, rhs=rhs, senses=senses)

    # Create delta constraints
    rhs = [0]*2*p
    senses = 'G'*(2*p)
    delta_upper = [cplex.SparsePair(ind=[1+i, n+p+1+i], val=[1.0, -1.0])
                   for i in range(p)]
    delta_lower = [cplex.SparsePair(ind=[1+i, n+p+1+i], val=[-1.0, -1.0])
                   for i in range(p)]
    model.linear_constraints.add(lin_expr=delta_upper+delta_lower, rhs=rhs,
                                 senses=senses)

    # solve problem
    model.solve()
    # get coefficients
    values = model.solution.get_values()
    intercept = values[0]
    coef = values[1:p+1]
    return intercept, coef

