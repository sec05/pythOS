import numpy as np
try:
    from firedrake import Function, CheckpointFile
    fem = True
except:
    fem = False
    Function = type(None)
    Constant = type(None)

def process_label(master_function, label, ida=False, ARKStep=False, existsIdaMethod=False):
    if label is None:
        return None
    if isinstance(label, tuple):
        if not ida and not ARKStep: 
            if existsIdaMethod:
                return lambda t, y, label=label: sum(master_function(t, y, None, l) for l in label)
            else:
                return lambda t, y, label=label: sum(master_function(t, y, l) for l in label)
        elif ARKStep and not ida:
            return (process_label(master_function, label[0], ida=False, ARKStep=False, existsIdaMethod=existsIdaMethod),
                    process_label(master_function, label[1], ida=False, ARKStep=False, existsIdaMethod=existsIdaMethod))
        else:
            return lambda t, y, dydt, label=label: sum(master_function(t, y, dydt, l) for l in label)
    else:
        if not ida:
            if existsIdaMethod:
                return lambda t, y, label=label: master_function(t, y, None, label)
            else:
                return lambda t, y, label=label: master_function(t, y, label)
        else:
            return lambda t, y, dydt, label=label: master_function(t, y, dydt, label)
    
def process_function_list(master_function, function_labels, y0, ivp_methods=None):
    if isinstance(master_function, list):
        functions = master_function
    else:
        if (callable(master_function) or isinstance(y0, Function)) and function_labels is None:
            functions = [master_function]
        else:
            functions = []
            existsIdaMethod = False
            if isinstance(ivp_methods, dict):
                for method in ivp_methods.values():
                    if method[0] == "IDA":
                        existsIdaMethod = True
                        break
            for i, label in enumerate(function_labels):
                ida = False
                ARKStep = False
                if isinstance(ivp_methods, dict):
                    try:
                        method = ivp_methods[i+1]
                        if method[0] == "IDA":
                            ida = True
                        elif (method[0][0] is None or 'ARKODE' in method[0][0]) and (method[0][1] is None or 'ARKODE' in method[0][1]):
                            ARKStep = True
                    except KeyError:
                        pass
                functions.append(process_label(master_function, label, ida=ida, ARKStep=ARKStep, existsIdaMethod=existsIdaMethod))
    return functions

def save_result_setup(fname, initial_y, initial_t, t, y=None):
    if y is None:
        y = initial_y
    if isinstance(initial_y, Function):
        f = CheckpointFile(fname, 'w')
        f.save_mesh(initial_y.function_space().mesh())
        f.save_function(initial_y, idx=0)
        f.create_group('times')
        f.set_attr('/times', '0', t)
        return f, 1
    else:
        f=open(fname, 'wb')
        np.savetxt(f, [[initial_t] + [x for x in y]], delimiter=',')
        return f, None
    
def save_result_step(f, initial_y, t, count_save, y=None):
    if y is None:
        y = initial_y
    if isinstance(initial_y, Function):
        f.save_function(y, idx=count_save)
        f.set_attr('/times', str(count_save), t)
        f.set_attr('/times', 'last_idx', count_save)
        count_save += 1
        return count_save
    else:
        np.savetxt(f, [[t]+[x for x in y]], delimiter=',')
        return None