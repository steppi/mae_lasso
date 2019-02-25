import cplex


def cplex_solve(X, y, reg_lambda):
    """MAE Regression solver using cplex Python API"""
    n, p = X.shape
    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.minimize)
    
    # Set display parameters to avoid output being printed to terminal
    model.parameters.barrier.display.set(0)
    model.parameters.simplex.display.set(0)
    
    # Create epsilon constraints
    rhs = list(y)*2
    senses = 'L'*n + 'G'*n

    top = np.concatenate((np.full((p, 1), 1),
                          X,
                          np.full((p, 1), -1)),
                         axis=1)
    bottom = np.concatenate((np.full((p, 1), 1),
                             X,
                             np.full((p, 1), 1)),
                            axis=1)
    constraints = np.concatenate((top, bottom), axis=0)
    constraints = constraints.tolist()
                         
    model.linear_constraints.add(lin_exp=constraints, rhs=rhs, senses=senses)

    # Create delta constraints
    rhs = [reg_lambda]*p + [-reg_lambda]*p
    senses = 'L'*p + 'G'*p
    
    
