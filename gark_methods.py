import numpy as np
from additive_rk import ark_solve
from butcher_tableau import Tableau

def gark_convert(A, b, order=None):
    """ Convert the provided GARK structure to an ARK structure """
    # This assumes M = N, and J = I
    tableaus = []
    Ni = sum([x.size for x in b])
    pre = 0
    for i in range(len(b)):
        si = b[i].size
        ai = np.zeros((Ni, Ni))
        ai[:,pre:pre+si] = np.concatenate([A[q][i] for q in range(len(b))])
        bi = np.zeros(Ni) 
        bi[pre:pre+si] = b[i]
        if (order is not None):
            ai = ai[:,order][order,:]
            bi = bi[order]
        ci = ai.sum(axis=1)
        tableaus.append(Tableau(ci, ai, bi))
        pre += si
    return tableaus


def gark_solve(master_function, dt, y0, t0, tf, A, b, function_labels=None, bc=None, solver_parameters={}, fname=None, save_steps = 0, jacobian=None, order=None):
    """ This function uses an GARK method to solve a differential equation
    This is done by converting to an ARK structure
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
         if using the finite element version, this should be of type Function
    t0 - the current value of t
         if using the finite element version, this should be of type Constant
    tf - the time to solve until
    A - the list of lists containing the component arrays a{J(q), i}
    b - the list containing the b{i} arrays
    function_labels : list of strings or tuples of strings
        the labels to use for each function in master_function
        tuples should be of the form (operation, label1, label2, ...)
        where operation = "prod" or "sum" if no operation is specified, it defaults to "sum"
        then the function produced by the tuple will be used as
        master_function(t,y,label1) (operation) master_function(t,y,label2) ...
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
    if isinstance(master_function, list):
        if len(master_function) > len(b):
            print("ERROR: not enough tableau provided")
            return

    tableaus = gark_convert(A, b, order)

    return ark_solve(master_function, dt, y0, t0, tf, tableaus, function_labels, bc=bc, solver_parameters=solver_parameters, fname=fname, save_steps = save_steps, jacobian=jacobian)


