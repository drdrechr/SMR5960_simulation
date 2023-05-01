###########################################################################
#                                                                         #
#    Optimizing univariate reinsurance contracts in the UNRESTRICTED,     #
#    univaraite case using the C0-function as objective function.         #
#                                                                         #
#    The expected insured risk is calculated using the built-in           #
#    partial mean functions in the dist lib.                              #
#                                                                         #
###########################################################################

from math import *
from dist import *
from insureutils import *

import matplotlib.pyplot as plt
import numpy as np


seed_num = 135246780
num_points = 25000          # Numerical integration intervals

# Figure size (in inches)
HSIZE = 6.2
VSIZE = 4.7


include_title = False
include_legend = True
include_labels = False

epsilon = 0.00001


plot_pmeans = False
plot_LS_RS_1 = True
plot_objFunc_1 = True
plot_expected_R = False

save_plots = True

corr = 0.00
alpha = 0.01
gamma = 0.10
theta = 0.20

seed(seed_num)

B_epsilon = 0.0000  # By choosing B_epsilon > 0, one can avoid undefined values when calculating Phi
                    # However, by redefining Phi so that B = 0 is handled correctly, this is no
                    # longer necessary. So now we just let B_epsilon = 0.


#dist1 = Truncnormal(30, 30)
#B1 = alpha * 1.0

#dist1 = Lognormal(30, 30)
#B1 = alpha * 1.0

#dist1 = Exponential(30)
#B1 = alpha * 1.0

dist1 = Pareto(30, 30)
B1 = alpha * 1.0

dist1.calcPMean()

# Calculate the expected net result from the risk portfolio before reinsurance
m1 = dist1.getMean()
NT = gamma * m1

if plot_pmeans:
    pmean1 = dist1.getPMean()
    P = np.linspace(0, 1, num_points+1)    
    plt.plot(P, pmean1, label = 'Partial mean')
    plt.xlabel('P')
    plt.ylabel('Partial mean')
    plt.title("Partial mean as a function of P")
    plt.legend()
    if save_plots:
        plt.savefig("univariate/pmean_" + dist1.get_name() + ".pdf")
    plt.show()

#################################################################
#                                                               #
#    S T A G E   1                                              #
#                                                               #
#    Based on the initial guessed values for B1:                #
#                                                               #
#    B1 = alpha * 1.0                                           #
#                                                               #
#    we find the corresponding optimal A-value by solving:      #
#                                                               #
#     E[G] = theta * A * a1                       (12)          #
#                                                               #
#    The solution will minimizes the objective function:        #
#                                                               #
#     C0 = a1 / E[G]                                            #
#                                                               #
#    for the initial guessed values of B1                       #
#                                                               #
#################################################################

b1 = dist1.getUpperPercentile(B1)

if B1 > 0:
    Bb1 = b1 * B1
else:
    Bb1 = 0

AA = np.linspace(B1, 0.4, num_points)

LS = np.zeros(num_points)       # Left-hand side of eq (12)
RS = np.zeros(num_points)       # Right-hand side of eq (12)
C0 = np.zeros(num_points)       # Objective function


for i in range(num_points):
    a1 = dist1.getUpperPercentile(AA[i])
    int1 = dist1.getIntervalPMean(1-AA[i], 1-B1)
    LS[i] = NT - theta * (int1 - AA[i] * a1 + Bb1)
    RS[i] = theta * AA[i] * a1
    C0[i] = a1 / LS[i]

diff_min = abs(LS[0] - RS[0])
opt_i = 0

for i in range(1, num_points):
    diff = abs(LS[i] - RS[i])
    if diff < diff_min:
        diff_min = diff
        opt_i = i
        
opt_A = AA[opt_i]
opt_a1 = dist1.getUpperPercentile(opt_A)
opt_C0 = C0[opt_i]

print("")
print("------------------------------------------------------------------------")
print("*** Stage 1 results for " + dist1.get_name()                             )
print("------------------------------------------------------------------------")
print("Opt. C0 = ", opt_C0) 
print("opt_A = ", opt_A,  ",  opt_B1 = ", B1)
print("opt_a1 = ", opt_a1)
print("opt_b1 = ", b1)
print("------------------------------------------------------------------------")
print("")

if plot_LS_RS_1:  
    plt.plot(AA, LS, label='LS versus A')
    plt.plot(AA, RS, label='RS versus A')
    plt.xlabel('A')
    plt.ylabel('LS/RS')
    plt.title("Lefthand and righthand sides versus A")
    plt.legend()
    if save_plots:
        plt.savefig("univariate/LSRS_1_" + dist1.get_name() + ".pdf")
    plt.show()

if plot_objFunc_1:
    plt.plot(AA, C0, label='C0 versus A')
    plt.xlabel('A')
    plt.ylabel('C0')
    plt.title("Objective function versus A")
    plt.legend()
    if save_plots:
        plt.savefig("univariate/C0_1_" + dist1.get_name() + ".pdf")
    plt.show()


#################################################################
#                                                               #
#    S T A G E   2                                              #
#                                                               #
#    In this stage we use the current optimal value of A        #
#    found in Stage 1, and optimize B1.                         #
#                                                               #
#    Calculate expected reinsurance costs for the risk as       #
#    functions of B, and store results in R1[]                  #
#                                                               #
#################################################################

A = opt_A
a1 = dist1.getUpperPercentile(A)


########################################################################
#    Calculate expected reinsurance costs using the built-in partial   #
#    mean functions in the dist lib.                                   #
########################################################################

print("Stage 2 -- STEP 1. Calculate expected reinsurance costs")

B0 = np.linspace(B_epsilon, opt_A, num_points)
R1 = np.zeros(num_points)

R1[num_points-1] = 0   # Corresponds to the case where B = opt_A


for i in range(num_points - 1):
    b1 = dist1.getUpperPercentile(B0[i])
    int1 = dist1.getIntervalPMean(1-opt_A, 1-B0[i])
    if B0[i] > 0:
        R1[i] = int1 - a1 * opt_A + b1 * B0[i]
    else:
        R1[i] = int1 - a1 * opt_A + 0.0


if plot_expected_R:
    fig = plt.figure(figsize = (HSIZE, VSIZE))
    plt.plot(B0, R1, label = 'R1 versus B1')
    plt.xlabel('BX')
    plt.ylabel('E[RX]')
    if include_title:
        plt.title("E[RX] versus BX")
    if include_legend:
        plt.legend()
    if save_plots:
        plt.savefig("univariate/E[RX]_" + dist1.get_name() + "[" + str(corr) + "]" + ".pdf")
    plt.show()
