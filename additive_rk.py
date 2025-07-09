import numpy as np
from scipy.optimize import root
from butcher_tableau import EmbeddedTableau, measure_error, compute_time
from utils import process_function_list, save_result_setup, save_result_step
try:
    from firedrake import Function, inner, dx, replace, split, Constant, TestFunction, solve, DirichletBC, CheckpointFile
except:
    print("no finite element solver")
    Function = type(None)
    Constant = type(None)

def ark_step(f, step, y0, t0, methods, rtol=1e-3, atol=1e-6, control=False, order=None, J=None, **kwargs):
    """
    Take a step in time using an additive runge kutta method.
    This function assumes at least one component method is implicit, but none are fully implicit (i.e. at least one method is a dirk method)
    """
    ki = np.zeros((len(f), np.size(y0), methods[0]._b.size), dtype = y0.dtype)
    accept = False
    step_old = step
    if J is not None:
        args = (J,)
    else:
        args = ()
    if 'method' not in kwargs:
        kwargs['method'] = 'krylov'
    
    while not accept:
        for i in range(methods[0]._b.size):
            sol = root(root_func, y0.view(np.float64), args = (f, step, y0, t0, methods, ki, i, args), **kwargs)
            if not sol.success:
                print(sol.message)
                print(np.max(abs(sol.fun)))
            Y = sol.x.view(y0.dtype)

            for j in range(len(f)):
                ki[j,:,i] = f[j](t0 + step *methods[j]._c[i], Y, *args)
        y_out = y0 + step * sum([np.dot(ki[i], methods[i]._b) for i in range(len(f))])
        if control:
            y_err = y0 + step * sum([np.dot(ki[i], methods[i].b_aux) for i in range(len(f))])
            accept, err = measure_error(y0, y_out, y_err, rtol, atol)
            step_old = step
            step = compute_time(err, order, step_old)
        else:
            accept = True
    return step, step_old, y_out

def ark_step_implicit(f, step, y0, t0, methods, rtol=1e-3, atol=1e-6, control=False, order=None, J = None, **kwargs):
    """
    Take a step in time using an additive runge kutta method.
    This function assumes at least one component method is fully implicit
    """
    accept = False
    step_old = step
    if J is not None:
        args = (J,)
    else:
        args = ()

    if 'method' not in kwargs:
        kwargs['method'] = 'krylov'
    
    while not accept:
        Y = np.array([y0 for i in range(methods[0]._b.size)]).transpose().flatten(order='F').view(np.float64)

        sol = root(root_func_implicit, Y, args=(f, step, y0, t0, methods, args), **kwargs)
        if not sol.success:
            print(sol.message)
            print(sol.fun)
        Y = sol.x.view(y0.dtype)

        ki = np.zeros((len(f), np.size(y0), methods[0]._b.size), dtype=y0.dtype)
        N = y0.size
        for i in range(methods[0]._b.size):
            y = Y[i*N:(i+1)*N]
            for j in range(len(f)):
                ki[j,:,i] = f[j](t0 + step *methods[j]._c[i], y, *args)
        y_out = y0 + step * sum([np.dot(ki[i], methods[i]._b) for i in range(len(f))])
        if control:
            y_err = y0 + step * sum([np.dot(ki[i], methods[i].b_aux) for i in range(len(f))])
            accept, err = measure_error(y0, y_out, y_err, rtol, atol)
            step_old = step
            step = compute_time(err, order, step_old)
        else:
            accept = True
    return step, step_old, y_out

def root_func(Y, f, step, y0, t0, methods, ki, i, args):
    # root finding for ARK methods that have no fully implicit components
    Y = Y.view(y0.dtype)
    return (Y - step * sum([np.dot(ki[j], methods[j]._a[i]) + methods[j]._a[i,i] * f[j](t0 + step * methods[j]._c[i], Y, *args) for j in range(len(f))]) - y0).view(np.float64)

def root_func_implicit(Y, f, step, y0, t0, methods, args):
    # root finding for methods that have some fully implicit component
    Y = Y.view(y0.dtype).reshape((y0.size, -1), order='F')
    R = np.array(Y)
    ki = np.zeros((len(f), y0.size, methods[0]._b.size), dtype=y0.dtype)
    for i in range(methods[0]._b.size):
        y = Y[:,i]
        for j in range(len(f)):
            ki[j,:,i] = f[j](t0 + step *methods[j]._c[i], y, *args)

    for i in range(methods[0]._b.size):
        for j in range(len(f)):
            R[:,i] -= step * np.dot(ki[j], methods[j]._a[i])
        R[:,i] -= y0
    R = R.flatten(order='F')
    return R.view(np.float64)

def ark_step_fem(f, step, y0, t0, methods, rtol=1e-3, atol=1e-6, control = False, order=None, bc=None, solver_parameters={}):
    ki = [[None for _ in range(methods[0]._b.size)] for _ in range(len(f))]
    accept = False
    step_old = step
    test_f = f[0].arguments()[0]
    f_list = []
    for fi in f:
        tf = fi.arguments()[0]
        f_list.append(replace(fi, {tf: test_f}))
    f = f_list
    ts = Constant(t0)
    while not accept:
        for i in range(methods[0]._b.size):
            Y = Function(y0)
            F2 = inner((Y - y0) / step, test_f) * dx
            for j in range(i):
                for f_ind in range(len(f)):
                    if methods[f_ind]._a[i,j] != 0:
                        F2 -= methods[f_ind]._a[i,j] * inner(ki[f_ind][j], test_f) * dx
            for f_ind in range(len(f)):
                if methods[f_ind]._a[i,i] != 0:
                    F2 -= methods[f_ind]._a[i,i] * replace(f[f_ind], {y0: Y, t0: t0 + step * methods[f_ind]._c[i]})
            t0.assign(ts + step * methods[0]._c[i])
            solve(F2 == 0, Y, bcs=bc, solver_parameters=solver_parameters)
            for j in range(len(f)):
                ki[j][i] = Function(y0)
                F2 = inner(ki[j][i], test_f) * dx - replace(f[j], {y0: Y, t0: ts + methods[j]._c[i] * step})
                solve(F2 == 0, ki[j][i])
        y_out = Function(y0)
        for i in range(methods[0]._b.size):
            for f_ind in range(len(f)):
                if methods[f_ind]._b[i] != 0:
                    y_out += step * ki[f_ind][i] * float(methods[f_ind]._b[i])

        if control:
            y_err = Function(y0)
            for i in range(methods[0]._b.size):
                for f_ind in range(len(f)):
                    if methods[f_ind].b_aux[i] != 0:
                        y_err += step * ki[f_ind][i] * float(methods[f_ind].b_aux[i])
            accept, err = measure_error(y0, y_out, y_err, rtol, atol)
            step_old = step
            step = compute_time(err, order, step_old)
        else:
            accept = True
    y0.assign(y_out)
    return step, step_old, y0


def ark_step_implicit_fem(f, step, y0, t0, methods, rtol=1e-3, atol=1e-6, control=False, order=None, bc = None, solver_parameters={}):
    ki = [[None for _ in range(methods[0]._b.size)] for _ in range(len(f))]
    accept = False
    step_old = step
    test_f = f[0].arguments()[0]
    f_list = []
    for fi in f:
        tf = fi.arguments()[0]
        f_list.append(replace(fi, {tf: test_f}))
    f = f_list
    ts = Constant(t0)

    Vbig = y0.function_space()
    for i in range(1, methods[0]._sizeb):
        Vbig = y0.function_space() * Vbig
    test_b = TestFunction(Vbig)
    N = len(y0.function_space())
    y_save = Function(y0)
    while not accept:
        Yis = Function(Vbig)
        F = 0
        ys = split(y0)
        test_fs = split(test_f)
        if N == 1:
            ys = [y0]
            test_fs = [test_f]
        for i in range(methods[0]._sizeb):
            for j in range(methods[0]._sizeb):
                rd = {}
                for kk in range(N):
                    rd[ys[kk]] = split(Yis)[j*N + kk]
                    rd[test_fs[kk]] = split(test_b)[i * N + kk]
                for f_ind in range(len(f)):
                    rd[t0] = t0 + step * methods[f_ind]._c[j]
                    F += methods[f_ind]._a[i, j] * replace(f[f_ind], rd)
            for kk in range(N):
                F -= inner((split(Yis)[i * N + kk] - ys[kk]) / step, split(test_b)[i*N + kk]) * dx
        new_bcs = []
        if bc is not None:
            if isinstance(bc, DirichletBC):
                bc = [bc]
            for bc in bc:
                if N == 1:
                    c = bc.function_space().component
                    if c is not None:
                        Vbi = lambda i: Vbig[i].sub(c)
                    else:
                        Vbi = lambda i: Vbig[i]
                else:
                    s = bc.function_space_index()
                    c = bc.function_space().component
                    if c is not None:
                        Vbi = lambda i: Vbig[s + N * i].sub(c)
                    else:
                        Vbi = lambda i: Vbig[s + N * i]
                for j in range(methods[0]._sizeb):
                    t0.assign(ts + step * methods[0]._c[j])
                    if bc.function_arg != 0:
                        new_bcs.append(DirichletBC(Vbi(j), bc.function_arg.copy(deepcopy=True), bc.sub_domain))
                    else:
                        new_bcs.append(DirichletBC(Vbi(j), 0, bc.sub_domain))

        solve(F == 0, Yis, bcs=new_bcs, solver_parameters=solver_parameters)

        y_out = []
        for j in range(methods[0]._sizeb):
            y_i = Function(y0)
            if N == 1:
                y_i.assign(Yis.sub(j*N + kk))
            else:
                for kk in range(N):
                    y_i.sub(kk).assign(Yis.sub(j * N + kk))
            y_out.append(y_i)
        for i in range(methods[0]._b.size):
            for j in range(len(f)):
                ki[j][i] = Function(y0)
                F2 = inner(ki[j][i], test_f) * dx - replace(f[j], {y0: y_out[i], t0: t0 + methods[j]._c[i] * step})
                solve(F2 == 0, ki[j][i])
        y_out = Function(y0)
        for i in range(methods[0]._b.size):
            for f_ind in range(len(f)):
                if methods[f_ind]._b[i] != 0:
                    y_out += step * ki[f_ind][i] * float(methods[f_ind]._b[i])

        if control:
            y_err = Function(y0)
            for i in range(methods[0]._b.size):
                for f_ind in range(len(f)):
                    if methods[f_ind].b_aux[i] != 0:
                        y_err += step * ki[f_ind][i] * float(methods[f_ind].b_aux[i])
            accept, err = measure_error(y0, y_out, y_err, rtol, atol)
            step_old = step
            step = compute_time(err, order, step_old)
        else:
            accept = True
    y0.assign(y_out)
    return step, step_old, y0


def ark_solve(master_function, dt, y0, t0, tf, methods, function_labels=None, rtol=1e-3, atol=1e-6, bc=None, solver_parameters={}, fname=None, save_steps = 0, jacobian = None):
    """ This function uses an additive runge kutta method to solve a differential equation
    -----
    Inputs:
    master_function : list of functions or callable function(t, y, label)
        if list of functions:
            each is used use to approximate the differential equation in order
            the functions will be numbered 1 to n
            if any element returns np.nan, that element will not be integrated for 
            that time step
            inputs are (t, y)
            These may also be finite element Forms as provided by firedrake, or
            tuples of (Form, boundary condition)
        if callable function(t, y, label):
            each label is used to select the function to use
            the labels are provided in function_labels
            the function will be called with inputs (t, y, label)
            then each function will be processed with process_label()
            and the result will be a list of functions used to approximate the differential equation
    dt - the amount time will increase by
    y0 - the current value of y
         if the functions are Forms, this should be of type Function
    t0 - the current value of t
         if using the finite element version, this should be of type Constant
    tf - the time to solve until
    methods - a listing of the butcher tableau for each operator.  This should be a list of Tableau with length at least the number of operators provided.
        Note: if length exceeds the length of the functions list, the extra methods will be ignored
    function_labels - optional. list of strings or tuples of strings
        the labels to use for each function in master_function
        tuples should be of the form (label1, label2, ...)
        which is then mapped to the operator O(t,y) = master_function(t,y,label1) + master_function(t,y,label2) + ...
        if master_function is a list of functions, this is ignored
    bc - optional.  Only used if using the finite element version.  The boundary condition(s) to apply.
    solver_parameters - optional.  Only used for the finite element version.  Any solver parameters to use (see firedrake documentation for details)
    fname - optional.  If provided, will save intermediate results to this file.
          - if using the finite element version of the code, this is a HDF5 file
            otherwise it is a csv file.
    save_steps - the number of intermediate steps to save if fname is provided
           if it is not provided, the default is to save every step
           (or after every dt if embedded methods are being used).
    Return
    -----
    the approximate value of y at tf
    """

    functions = process_function_list(master_function, function_labels, y0)

    if len(functions) > len(methods):
        print("ERROR: not enough tableau provided")
        return 
    if np.shape(y0) == () and not isinstance(y0, Function):
        y0 = np.array([y0])

    # determine if the fully implicit solver is needed and if step size control is being used
    fully_implicit = False
    order = np.inf
    control = True
    for method in methods:
        if np.any(np.triu(method._a, 1)!=0):
            fully_implicit = True
        if not isinstance(method, EmbeddedTableau):
            control = False
        else:
            order = min(order, method.order)
    step = ark_step
    if fully_implicit:
        step = ark_step_implicit
    if isinstance(y0, Function):
        if fully_implicit:
            step = ark_step_implicit_fem
        else:
            step = ark_step_fem
        fem = True
    else:
        fem = False
    # Solve the DE
    t = t0
    if isinstance(t, Constant):
        t = t.values()[0]
        if isinstance(t, np.complex128):
            t = complex(t)
        else:
            t = float(t)
    if save_steps != 0:
        save_interval = (tf - t) / save_steps
    else:
        save_interval = dt

    if fname is not None:
        f, count_save = save_result_setup(fname, y0, t0, t)
        saved = t
    
    while t < tf:
        if fem:
            dt_new, dt, y0 = step(functions, dt, y0, t0, methods, rtol=rtol, atol=atol, order=order, control=control, bc=bc, solver_parameters=solver_parameters)
        else:
            J = None
            if jacobian is not None:
                J = jacobian(t0, y0)
            dt_new, dt, y0 = step(functions, dt, y0, t0, methods, rtol=rtol, atol=atol, order=order, control=control, J = J, **solver_parameters)
        t += dt
        if isinstance(t0, Constant):
            t0.assign(t)
        else:
            t0 = t
        if fname is not None and t - saved - save_interval > -1e-8:
            count_save = save_result_step(f, y0, t, count_save)
            saved += ((t - saved + 1e-8) // save_interval) * save_interval
        dt = dt_new
        if abs(dt) < 1e-10 and control:
            if isinstance(y0, Function):
                return y0.assign(np.nan)
            return y0 * np.nan
        if dt > tf - t:
            dt = tf - t
        if isinstance(t0, Constant):
            t0.assign(t)
        else:
            t0 = t
    if fname is not None and t - saved > -1e-8:
        count_save = save_result_step(f, y0, t, count_save)

    return y0
