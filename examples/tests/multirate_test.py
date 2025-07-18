import numpy as np
import sys
import fractional_step as fs # useful wrapper for the adaptive solver
from multirate import multirate_solve, mrgark_ex2_im2
from multirate_infinitesimal import multirate_infinitesimal_solve, mri_kw3, mri_imex3
from testing_utils import plot, output
from brusselator_ode_problem import *
if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == '--help'):
    print("Usage:")
    print("python3 multirate_test.py [-p|--plot] [-v|--verbose] [-s|--sundials] [-h|--help]")
    print("""Options:
    -p | --plot\tProduce plot of the results
    -v | --verbose\tPrint the solution at tf for all methods tested
    -s | --sundials\tTry the sundials adaptive solver
    -h | --help\tDisplay this help message and exit (if it is the only command line flag)""")
    exit()

verbose = "-v" in sys.argv or "--verbose" in sys.argv
enable_sundials = "-s" in sys.argv or "--sundials" in sys.argv

def master_function(t, y, label):
    if label == "f1":
        return f1(t, y)
    elif label == "f2":
        return f2(t, y)
    else:
        raise ValueError(f"Unknown label: {label}")

labels = [("f2"), "f1"]

solution = fs.fractional_step([lambda t, y: f1(t,y) + f2(t,y)], .1, y0, 0, tf, 'Godunov', None, {(0,): 'ADAPTIVE'}, fname='adaptive.csv')
if verbose:
    print("{:<30} {}".format("adaptive solution", solution))
result = multirate_solve(y0, 0, 0.1, tf, mrgark_ex2_im2, 10, fs=f2, ff=f1, fname='multirate.csv')

output(verbose, 0.2, solution, result, "mrgark_ex2_im2")

result = multirate_solve(y0, 0, 0.1, tf, mrgark_ex2_im2, 10, labels[0], labels[1], master_function=master_function, fname='multirate_labels.csv')

output(verbose, 0.2, solution, result, "mrgark_ex2_im2 (labels)")

result = multirate_infinitesimal_solve(y0,0,0.2,tf,mri_kw3,fi=f2,ff=f1, fname='mri.csv')

output(verbose, 0.1, solution, result, "mri_kw3")

result = multirate_infinitesimal_solve(y0,0,0.2,tf,mri_kw3, labels[0], labels[1], master_function=master_function, fname='mri_labels.csv')

output(verbose, 0.1, solution, result, "mri_kw3 (labels)")

result = multirate_infinitesimal_solve(y0,0,0.2,tf,mri_kw3,fi=f2,ff=f1, ivp_method='Cash-Karp', fname='mri_Cash-Karp.csv')

output(verbose, 0.1, solution, result, "mri_kw3 (Cash-Karp)")

result = multirate_infinitesimal_solve(y0,0,0.2,tf,mri_kw3, labels[0], labels[1], master_function=master_function, ivp_method='Cash-Karp', fname='mri_Cash-Karp_labels.csv')

output(verbose, 0.1, solution, result, "mri_kw3 (Cash-Karp) (labels)")

if enable_sundials:
    result = multirate_infinitesimal_solve(y0,0,0.2,tf,mri_kw3,fi=f2,ff=f1, ivp_method='CV_ADAMS', fname='mri_CV_ADAMS.csv')
    output(verbose, 0.1, solution, result, "mri_kw3 (CVode)")
    result = multirate_infinitesimal_solve(y0,0,0.2,tf,mri_kw3, labels[0], labels[1], master_function=master_function, ivp_method='CV_ADAMS', fname='mri_CV_ADAMS_labels.csv')
    output(verbose, 0.1, solution, result, "mri_kw3 (CVode) (labels)")

result = multirate_infinitesimal_solve(y0, 0, 0.3, tf, mri_imex3, fi=f2, ff=f1, fname='mri_imex.csv')

output(verbose, 0.1, solution, result, "mri_imex3")

result = multirate_infinitesimal_solve(y0, 0, 0.3, tf, mri_imex3, labels[0], labels[1], master_function=master_function, fname='mri_imex3_labels.csv')

output(verbose, 0.1, solution, result, "mri_imex3 (labels)")

if '-p' in sys.argv or '--plot' in sys.argv:
    labels={'multirate.csv': "Multirate", 'mri.csv': "MRI", 'mri_Cash-Karp.csv': "MRI with Cash-Karp", 'mri_imex.csv': "MRI-IMEX"}
    if enable_sundials:
        labels['mri_CV_ADAMS.csv'] = "MRI with CVode" 
    plot('adaptive.csv', labels)
