###########################################################################
#                                                                         #
#    Optimizing bivariate reinsurance contracts in the UNRESTRICTED,      #
#    bivariate case using the C0-function as objective function.          #
#    This script combines optimization of A using the method in           #
#    righthand_02.py (Stage 1, 3 and 4) and optimization of B1 and B2     #
#    using the method num_opt2.py (Stage 2)                               #
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
# seed_num = 104326980
num_sims = 5000000          # Simulations
num_points = 50000          # Numerical integration intervals

# Figure size (in inches)
HSIZE = 6.2
VSIZE = 4.7

# Axis ticks in iso-curve plots
# XSTEP = 0.004
# X_MAX = 0.020
# YSTEP = 0.004
# Y_MAX = 0.020

include_title = False
include_legend = True
include_labels = False

num_iter = 50              # Number of points on objective curve
epsilon = 0.00001

smooth_constraint = False
smooth_w = [0.3, 0.4, 0.3]

plot_pmeans = False
plot_LS_RS_1 = False
plot_objFunc_1 = False
plot_expected_R = False
plot_scatter1 = False
plot_scatter2 = False
plot_isocurves = True
plot_phi = True
plot_LS_RS_2 = False
plot_objFunc_2 = False
plot_LS_RS_3 = False
plot_objFunc_3 = False
plot_hazard_rate = True

save_plots = False


alpha = 0.01
gamma = 0.10
theta = 0.20
delta = 0.00

corr = 0.00
c1 = 0.5 *(sqrt(1+corr) + sqrt(1-corr))
c2 = 0.5 *(sqrt(1+corr) - sqrt(1-corr))

# Del = 0.02    # Determines the size of the importance sample region
Del = 0.10      # Determines the size of the importance sample region

probD = 1 - (1-Del)*(1-Del)
probE = (1-Del)*(1-Del)

seed(seed_num)

B_epsilon = 0.0000  # By choosing B_epsilon > 0, one can avoid undefined values when calculating Phi
                    # However, by redefining Phi so that B = 0 is handled correctly, this is no
                    # longer necessary. So now we just let B_epsilon = 0.
                    

#dist1 = Truncnormal(100, 30)
#dist2 = Truncnormal(100, 30)
#B1 = alpha * 0.5
#B2 = alpha * 0.5
#force_balanced = False
#force_unbalanced = False

#dist1 = Lognormal(50, 30)
#dist2 = Lognormal(100, 50)
#B1 = alpha * 1.0
#B2 = alpha * 0.0
#force_balanced = False
#force_unbalanced = True

#dist1 = Exponential(50)
#dist2 = Exponential(400)
#B1 = alpha * 1.0
#B2 = alpha * 0.0
#force_balanced = False
#force_unbalanced = False

# dist1 = Exponential(60)
# dist2 = Exponential(40)
# B1 = alpha * 1.0
# B2 = alpha * 0.0
# force_balanced = False
# force_unbalanced = True

dist1 = Pareto(50, 30)
dist2 = Pareto(50, 50)
B1 = alpha * 0.5
B2 = alpha * 0.5
force_balanced = False
force_unbalanced = False

# dist1 = Lomax(50, 70)
# dist2 = Lomax(50, 70)
# B1 = alpha * 0.5
# B2 = alpha * 0.5
# force_balanced = True
# force_unbalanced = False

dist1.calcPMean()
dist2.calcPMean()

# Calculate the expected net result from the risk portfolio before reinsurance
m1 = dist1.getMean()
m2 = dist2.getMean()
NT = gamma * (m1 + m2)

if plot_pmeans:
    pmean1 = dist1.getPMean()
    pmean2 = dist2.getPMean()    
    P = np.linspace(0, 1, num_points+1)    
    plt.plot(P, pmean1, label='Partial mean 1')
    plt.plot(P, pmean2, label='Partial mean 2')
    plt.xlabel('P')
    plt.ylabel('Partial mean')
    plt.title("Partial mean as a function of P")
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/pmean_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
    plt.show()

#################################################################
#                                                               #
#    S T A G E   1                                              #
#                                                               #
#    Based on the initial guessed values for B1 and B2:         #
#                                                               #
#     B1 = alpha * 0.5 and B2 = alpha * 0.5 (balanced case)     #
#     B1 = alpha * 1.0 and B2 = alpha * 0.0 (unbalanced case)   #
#                                                               #
#    we find the corresponding optimal A-value by solving:      #
#                                                               #
#     E[G] = theta * A * (a1 + a2)                (12)          #
#                                                               #
#    The solution will minimizes the objective function:        #
#                                                               #
#     C0 = (a1 + a2) / E[G]                                     #
#                                                               #
#    for the initial guessed values of B1 and B2                #
#                                                               #
#################################################################

b1 = dist1.getUpperPercentile(B1)
b2 = dist2.getUpperPercentile(B2)

if B1 > 0:
    Bb1 = b1 * B1
else:
    Bb1 = 0
    
if B2 > 0:
    Bb2 = b2 * B2
else:
    Bb2 = 0

AA = np.linspace(max(B1, B2), 0.4, num_points)

LS = np.zeros(num_points)       # Left-hand side of eq (12)
RS = np.zeros(num_points)       # Right-hand side of eq (12)
C0 = np.zeros(num_points)       # Objective function

a1 = dist1.getUpperPercentile(AA[0])
a2 = dist2.getUpperPercentile(AA[0])

int1 = 0
int2 = 0

if B1 < B2:     # i.e.: b2 < b1 and (1-B2) < (1-B1)
    int1 = dist1.getIntervalPMean(1-B2, 1-B1)
if B2 < B1:     # i.e.: b1 < b2 and (1-B1) < (1-B2)
    int2 = dist2.getIntervalPMean(1-B1, 1-B2)
    
LS[0] = NT - theta * ((int1 + int2) - AA[0] * (a1 + a2) + Bb1 + Bb2)
RS[0] = theta * AA[0] * (a1 + a2)
C0[0] = (a1 + a2) / LS[0]

for i in range(1, num_points):
    a1 = dist1.getUpperPercentile(AA[i])
    a2 = dist2.getUpperPercentile(AA[i])
    int1 = dist1.getIntervalPMean(1-AA[i], 1-B1)
    int2 = dist2.getIntervalPMean(1-AA[i], 1-B2)
    LS[i] = NT - theta * ((int1 + int2) - AA[i] * (a1 + a2) + Bb1 + Bb2)
    RS[i] = theta * AA[i] * (a1 + a2)
    C0[i] = (a1 + a2) / LS[i]

diff_min = abs(LS[0] - RS[0])
opt_i = 0

for i in range(1, num_points):
    diff = abs(LS[i] - RS[i])
    if diff < diff_min:
        diff_min = diff
        opt_i = i
        
opt_A = AA[opt_i]
opt_a1 = dist1.getUpperPercentile(opt_A)
opt_a2 = dist2.getUpperPercentile(opt_A)
opt_C0 = C0[opt_i]

print("")
print("------------------------------------------------------------------------")
print("*** Stage 1 results for " + dist1.get_name() + " vs " + dist2.get_name() )
print("------------------------------------------------------------------------")
print("Opt. C0 = ", opt_C0) 
print("opt_A = ", opt_A,  ",  opt_B1 = ", B1,  ",  opt_B2 = ", B2)
print("opt_a1 = ", opt_a1,  ",  opt_a2 = ", opt_a2)
print("opt_b1 = ", b1,  ",  opt_b2 = ", b2)
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
        plt.savefig("num_opt3/LSRS_1_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
    plt.show()

if plot_objFunc_1:
    plt.plot(AA, C0, label='C0 versus A')
    plt.xlabel('A')
    plt.ylabel('C0')
    plt.title("Objective function versus A")
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/C0_1_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
    plt.show()


#################################################################
#                                                               #
#    S T A G E   2                                              #
#                                                               #
#    In this stage we use the current optimal value of A        #
#    found in Stage 1, and optimize B1 and B2.                  #
#                                                               #
#    STEP 1. Calculate expected reinsurance costs for the       #
#            two risks as functions of Bi, and store results    #
#            in R1[] and R2[] respectively.                     #
#                                                               #
#    STEP 2. Generate risks sampling from the set D             #
#                                                               #
#    STEP 3. Determine B1_max and b1_min                        #
#                                                               #
#    STEP 4. Determine B2_max and b2_min                        #
#                                                               #
#    STEP 5. Calculate isocurves for R1 + R2 as functions of    #
#            (B1,B2)                                            #
#                                                               #
#    STEP 6. Determine the constraint set, i.e., the set of     #
#            values of B1 and B2 such that P(C) = alpha.        #
#            For each (B1,B2) in the constraint set, we         #
#            calculate Phi = expected total reinsurance cost.   #
#            The optimal value of (B1,B2) is the value that     #
#            minimizes Phi.                                     #
#                                                               #
#################################################################

A = opt_A
a1 = dist1.getUpperPercentile(A)
a2 = dist2.getUpperPercentile(A)


########################################################################
#    Calculate expected reinsurance costs using the built-in partial   #
#    mean functions in the dist lib.                                   #
########################################################################

print("Stage 2 -- STEP 1. Calculate expected reinsurance costs")

B0 = np.linspace(B_epsilon, opt_A, num_points)
R1 = np.zeros(num_points)
R2 = np.zeros(num_points)

R1[num_points-1] = 0    # Corresponds to the case where B = opt_A
R2[num_points-1] = 0    # Corresponds to the case where B = opt_A


for i in range(num_points - 1):
    b1 = dist1.getUpperPercentile(B0[i])
    b2 = dist2.getUpperPercentile(B0[i])
    int1 = dist1.getIntervalPMean(1-opt_A, 1-B0[i])
    int2 = dist2.getIntervalPMean(1-opt_A, 1-B0[i])
    if B0[i] > 0:
        R1[i] = int1 - a1 * opt_A + b1 * B0[i]
        R2[i] = int2 - a2 * opt_A + b2 * B0[i]
    else:
        R1[i] = int1 - a1 * opt_A + 0.0
        R2[i] = int2 - a2 * opt_A + 0.0


if plot_expected_R:
    fig = plt.figure(figsize = (HSIZE, VSIZE))
    plt.plot(B0, R1, label='R1 versus B1')
    plt.plot(B0, R2, label='R2 versus B2')
    plt.xlabel('BX')
    plt.ylabel('E[RX]')
    if include_title:
        plt.title("E[RX] versus BX")
    if include_legend:
        plt.legend()
    if save_plots:
        plt.savefig("num_opt3/E[RX]_" + dist1.get_name() + "_" + dist2.get_name() + "[" + str(corr) + "]" + ".png")
    plt.show()

    

#############################################################
#    Generate risks sampling from the set D                 #
#############################################################

print("Stage 2 -- STEP 2. Generate risks sampling from the set D")

x1 = np.zeros(num_sims)
x2 = np.zeros(num_sims)

if corr == 0:
    for i in range(num_sims):
        u =  bi_uniform_D(Del)
        x1[i] = dist1.getLowerPercentile(u[0])
        x2[i] = dist2.getLowerPercentile(u[1])
else:
    for i in range(num_sims):
        u =  bi_uniform_D(Del)
        g1 = gaussian_invcdf(u[0])
        g2 = gaussian_invcdf(u[1])
        h1 = c1 * g1 + c2 * g2
        h2 = c1 * g2 + c2 * g1
        v1 = gaussian_cdf(h1)
        v2 = gaussian_cdf(h2)
        x1[i] = dist1.getLowerPercentile(v1)
        x2[i] = dist2.getLowerPercentile(v2)


######################################
#    Determine B1_max and b1_min     #
######################################

print("Stage 2 -- STEP 3. Determine B1_max and b1_min:")

b2 = dist2.getUpperPercentile(0)

B1_L = 0                # B1_L = alpha
B1_U = A                # B1_U = alpha / A

# First we check if [B1_L, B1_U] contains a B1 such that P(C) = alpha
b1 = dist1.getUpperPercentile(B1_L)
probC = getCFraction(x1, a1, b1, x2, a2, b2) * probD
if probC > alpha:
    print("Error 1 (L): The interval [" + str(B1_L) + ", " + str(B1_U) + "] does not contain a solution for B1!!")
b1 = dist1.getUpperPercentile(B1_U)
probC = getCFraction(x1, a1, b1, x2, a2, b2) * probD
if probC < alpha:
    print("Error 1 (U): The interval [" + str(B1_L) + ", " + str(B1_U) + "] does not contain a solution for B1!!")

B1 = (B1_L + B1_U) / 2
b1 = dist1.getUpperPercentile(B1)

while B1_U - B1_L > epsilon:
    probC = getCFraction(x1, a1, b1, x2, a2, b2) * probD
    if probC > alpha:
        B1_U = B1
    else:
        B1_L = B1
    B1 = (B1_L + B1_U) / 2 
    b1 = dist1.getUpperPercentile(B1)

B1_max = B1
b1_min = dist1.getUpperPercentile(B1_max)
 
countC = getCCount(x1, a1, b1_min, x2, a2, b2)

# probC_given_D = countC / num_sims
# probC = probC_given_D * probD

# print("P(D) = ", probD, ", P(E) = ", probE)
# print("p(C | D) = ", probC_given_D, ", P(C) = ", probC)

print("b1_min = ", b1_min, ", B1_max = ", B1_max)

##########
# Create a scatter plot where blue dots represent simulations located in the A- or B-sets,
# while the red dots represent simulations located in the C-set. These sets are based on 
# the current values of a1, a2, b1 and b2, where in particular b1 = b1_min in this case.
#
# NOTE: Due to importance sampling there will be a set in the scatter plot with _no_ dots.
# This blank set should ideally be a subset of the A- or B-sets, which can be verified from 
# the scatter plot. If the blank set also overlaps with the C-set, this indicates that 
# the importance sampling is too agressive. If so, the Del-variable should be increased.
##########
if plot_scatter1:   
    C_set_1 = np.zeros(countC)
    C_set_2 = np.zeros(countC)
    AB_set_1 = np.zeros(num_sims - countC)
    AB_set_2 = np.zeros(num_sims - countC)
    
    j: int = 0
    k: int = 0
    
    for i in range(num_sims):
        rr: float = get_retained_risk(x1[i], a1, b1_min) + get_retained_risk(x2[i], a2, b2)
        if rr > a1 + a2:
            C_set_1[j] = x1[i]
            C_set_2[j] = x2[i]
            j += 1
        else:
            AB_set_1[k] = x1[i]
            AB_set_2[k] = x2[i]
            k += 1       
        
    plt.scatter(AB_set_1, AB_set_2, c ="blue", label = "Set A and B", s = 5, alpha = 0.5)
    plt.scatter(C_set_1, C_set_2, c ="red", label = "Set C", s = 5, alpha = 0.7)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("Importance sampling with delta = " + str(Del))
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/scat1_" + dist1.get_name() + "_" + dist2.get_name() + "[" + str(corr) + "]" + ".png")
    plt.show()   
 

######################################
#    Determine B2_max and b2_min     #
######################################

print("Stage 2 -- STEP 4. Determine B2_max and b2_min")

b1 = dist1.getUpperPercentile(0)

B2_L = 0                # B2_L = alpha
B2_U = alpha / A        # B2_U = A

# First we check if [B2_L, B2_U] contains a B2 such that P(C) = alpha
b2 = dist1.getUpperPercentile(B2_L)
probC = getCFraction(x1, a1, b1, x2, a2, b2) * probD
if probC > alpha:
    print("Error 1 (L): The interval [" + str(B2_L) + ", " + str(B2_U) + "] does not contain a solution for B2!!")
b2 = dist1.getUpperPercentile(B2_U)
probC = getCFraction(x1, a1, b1, x2, a2, b2) * probD
if probC < alpha:
    print("Error 1 (U): The interval [" + str(B2_L) + ", " + str(B2_U) + "] does not contain a solution for B1!!")

B2 = (B2_L + B2_U) / 2
b2 = dist2.getUpperPercentile(B2)

while B2_U - B2_L > epsilon:
    probC = getCFraction(x1, a1, b1, x2, a2, b2) * probD
    if probC > alpha:
        B2_U = B2
    else:
        B2_L = B2
    B2 = (B2_L + B2_U) / 2 
    b2 = dist2.getUpperPercentile(B2)

B2_max = B2
b2_min = dist2.getUpperPercentile(B2_max)
 
countC = getCCount(x1, a1, b1, x2, a2, b2_min)
        
# probC_given_D = countC / num_sims
# probC = probC_given_D * probD

# print("P(D) = ", probD, ", P(E) = ", probE)
# print("p(C | D) = ", probC_given_D, ", P(C) = ", probC)

print("b2_min = ", b2_min, ", B2_max = ", B2_max)

##########
# Create a scatter plot where blue dots represent simulations located in the A- or B-sets,
# while the red dots represent simulations located in the C-set. These sets are based on 
# the current values of a1, a2, b1 and b2, where in particular b2 = b2_min in this case.
#
# NOTE: Due to importance sampling there will be a set in the scatter plot with _no_ dots.
# This blank set should ideally be a subset of the A- or B-sets, which can be verified from 
# the scatter plot. If the blank set also overlaps with the C-set, this indicates that 
# the importance sampling is too agressive. If so, the Del-variable should be increased.
##########
if plot_scatter2:
    AB_set_1 = np.zeros(num_sims - countC)
    AB_set_2 = np.zeros(num_sims - countC)
    C_set_1 = np.zeros(countC)
    C_set_2 = np.zeros(countC)
    
    j: int = 0
    k: int = 0
    
    for i in range(num_sims):
        rr: float = get_retained_risk(x1[i], opt_a1, b1) + get_retained_risk(x2[i], opt_a2, b2_min)
        if rr > a1 + a2:
            C_set_1[j] = x1[i]
            C_set_2[j] = x2[i]
            j += 1
        else:
            AB_set_1[k] = x1[i]
            AB_set_2[k] = x2[i]
            k += 1       
        
    plt.scatter(AB_set_1, AB_set_2, c ="blue", label = "Set A and B", s = 5, alpha = 0.5)
    plt.scatter(C_set_1, C_set_2, c ="red", label = "Set C", s = 5, alpha = 0.7)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("Importance sampling with delta = " + str(Del))
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/scat2_" + dist1.get_name() + "_" + dist2.get_name() + "[" + str(corr) + "]" + ".png")
    plt.show()   
 

print("Stage 2 -- STEP 5. Calculate isocurves")

##########
# R1[0] is the expected reinsurance cost of the first risk if B1 = B_epsilon, 
# For this B1-value b1 gets its maximum calculated value (infinity if B_epsilon = 0)
# R1[i] and R2[i] are decreasing in i, and num_1 < num_2 < ... < num_8
# Hence, iso1 > iso2 > ... > iso8
# R1[num_points-1] is the expected reinsurance cost of the first risk if B1 = opt_A.
# For this B1-value b1 = a1, implying that R1[num_points-1] = 0.
##########

# Settings for iso-curve plot
num = int(0.25 * max(B1_max, B2_max) * (num_points - 1) / opt_A)

iso1 = R1[0] + R2[1 * num]  # Iso-curve 1
iso2 = R1[0] + R2[2 * num]  # Iso-curve 2
iso3 = R1[0] + R2[3 * num]  # Iso-curve 3
iso4 = R1[0] + R2[4 * num]  # Iso-curve 4
iso5 = R1[0] + R2[5 * num]  # Iso-curve 5
iso6 = R1[0] + R2[6 * num]  # Iso-curve 6
iso7 = R1[0] + R2[7 * num]  # Iso-curve 7
iso8 = R1[0] + R2[8 * num]  # Iso-curve 8


BC1_1 = []
BC1_2 = []

BC2_1 = []
BC2_2 = []

BC3_1 = []
BC3_2 = []

BC4_1 = []
BC4_2 = []

BC5_1 = []
BC5_2 = []

BC6_1 = []
BC6_2 = []

BC7_1 = []
BC7_2 = []

BC8_1 = []
BC8_2 = []


i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso1 - R1[i])
    BC1_1.append(opt_A * i / (num_points - 1))
    BC1_2.append(opt_A * j / (num_points - 1))
    i += 1
    
i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso2 - R1[i])
    BC2_1.append(opt_A * i / (num_points - 1))
    BC2_2.append(opt_A * j / (num_points - 1))
    i += 1
    
i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso3 - R1[i])
    BC3_1.append(opt_A * i / (num_points - 1))
    BC3_2.append(opt_A * j / (num_points - 1))
    i += 1
    
i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso4 - R1[i])
    BC4_1.append(opt_A * i / (num_points - 1))
    BC4_2.append(opt_A * j / (num_points - 1))
    i += 1
    
i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso5 - R1[i])
    BC5_1.append(opt_A * i / (num_points - 1))
    BC5_2.append(opt_A * j / (num_points - 1))
    i += 1    
    
i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso6 - R1[i])
    BC6_1.append(opt_A * i / (num_points - 1))
    BC6_2.append(opt_A * j / (num_points - 1))
    i += 1    
    
i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso7 - R1[i])
    BC7_1.append(opt_A * i / (num_points - 1))
    BC7_2.append(opt_A * j / (num_points - 1))
    i += 1    
    
i = 0
j = num_points - 1

while i < num_points and j > 0:
    j = valueToIndexDecr(R2, iso8 - R1[i])
    BC8_1.append(opt_A * i / (num_points - 1))
    BC8_2.append(opt_A * j / (num_points - 1))
    i += 1    


######################################
#    Calculate constraint            #
######################################

print("Stage 2 -- STEP 6. Determine constraint set and optimize (B1,B2)")

BB1 = np.linspace(B_epsilon, B1_max - B_epsilon, num_iter)
BB2 = np.zeros(num_iter)

Phi = np.zeros(num_iter)    # = Expected total reinsurance cost

opt_B1 = 0
opt_B2 = 0
minPhi = np.inf

BB2[0] = B2_max

# In the forceBalanced case we will use B1 = B2 = B_balanced
Diff_min = abs(BB1[0] - BB2[0])
B_balanced = (BB1[0] + BB2[0]) / 2

for ii in range(1, num_iter):
    b1 = dist1.getUpperPercentile(BB1[ii])
    
    # B2_L = (alpha - BB1[ii]) / (1 - BB1[ii])
    # B2_U = (alpha - A * BB1[ii]) / (A - BB1[ii])
    
    B2_L = 0
    B2_U = BB2[ii - 1]
    
    tmp_B2 = (B2_L + B2_U) / 2
    b2 = dist2.getUpperPercentile(tmp_B2)
    
    while B2_U - B2_L > epsilon:
        probC = getCFraction(x1, a1, b1, x2, a2, b2) * probD
        if probC > alpha:
            B2_U = tmp_B2
        else:
            B2_L = tmp_B2
        tmp_B2 = (B2_L + B2_U) / 2 
        b2 = dist2.getUpperPercentile(tmp_B2)
        
    if tmp_B2 > A:
        print("ERROR: tmp_B = " + str(tmp_B2) + " is out of bounds")
        tmp_B2 = A
    
    if tmp_B2 < 0:
        print("ERROR: tmp_B = " + str(tmp_B2) + " is out of bounds")
        tmp_B2 = 0.005
    
    BB2[ii] = tmp_B2
    
    # B_balanced is chosen so that abs(BB1[ii] - BB2[ii]) is minimized
    if abs(BB1[ii] - BB2[ii]) < Diff_min:
        B_balanced = (BB1[ii] + BB2[ii]) / 2
    
if smooth_constraint:
    tmpBB2 = np.zeros(num_iter)
    for s in range(1, num_iter-1):
        tmpBB2[s] = smooth_w[0] * BB2[s-1] + smooth_w[1] * BB2[s] + smooth_w[2] * BB2[s+1]
    for s in range(1, num_iter-1):
        BB2[s] = tmpBB2[s]

for ii in range(num_iter):
    b1 = dist1.getUpperPercentile(BB1[ii]) 
    b2 = dist2.getUpperPercentile(BB2[ii])
    int1 = dist1.getIntervalPMean(1-A, 1-BB1[ii])
    int2 = dist2.getIntervalPMean(1-A, 1-BB2[ii])
    if BB1[ii] > 0:
        RR1 = int1 - a1 * A + b1 * BB1[ii]
    else:
        RR1 = int1 - a1 * A + 0.0
    if BB2[ii] > 0:
        RR2 = int2 - a2 * A + b2 * BB2[ii]
    else:
        RR2 = int2 - a2 * A + 0.0

    Phi[ii] = RR1 + RR2
    
    if Phi[ii] < minPhi:
        minPhi = Phi[ii]
        opt_B1 = BB1[ii] 
        opt_B2 = BB2[ii]
        
opt_b1 = dist1.getUpperPercentile(opt_B1)
opt_b2 = dist2.getUpperPercentile(opt_B2)

print("")
print("------------------------------------------------------------------------")
print("*** Stage 2 results for " + dist1.get_name() + " vs " + dist2.get_name() )
print("------------------------------------------------------------------------")
print("minPhi = ", minPhi) 
print("opt_A = ", opt_A,  ",  opt_B1 = ", opt_B1,  ",  opt_B2 = ", opt_B2)
print("opt_a1 = ", opt_a1,  ",  opt_a2 = ", opt_a2)
print("opt_b1 = ", opt_b1,  ",  opt_b2 = ", opt_b2)
print("------------------------------------------------------------------------")
print("")

if plot_isocurves:
    plt.plot(BC1_1, BC1_2, label='Phi = ' + str(round(iso1,2)))
    plt.plot(BC2_1, BC2_2, label='Phi = ' + str(round(iso2,2)))
    plt.plot(BC3_1, BC3_2, label='Phi = ' + str(round(iso3,2)))
    plt.plot(BC4_1, BC4_2, label='Phi = ' + str(round(iso4,2)))
    plt.plot(BC5_1, BC5_2, label='Phi = ' + str(round(iso5,2)))
    plt.plot(BC6_1, BC6_2, label='Phi = ' + str(round(iso6,2)))
    plt.plot(BC7_1, BC7_2, label='Phi = ' + str(round(iso7,2)))
    plt.plot(BC8_1, BC8_2, label='Phi = ' + str(round(iso8,2)))
    plt.plot(BB1, BB2, label='Constraint')
    plt.xlabel('B1')
    plt.ylabel('B2')
    plt.title("Constraint and iso-curves")
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/iso(B)_" + dist1.get_name() + "_" + dist2.get_name() + "[" + str(corr) + "]" + ".png")
    plt.show()
        
if plot_phi:
    plt.plot(BB1, Phi, label='Phi')
    plt.xlabel('B1')
    plt.ylabel('Phi')
    plt.title("Objective function")
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/Phi(B)_" + dist1.get_name() + "_" + dist2.get_name() + "[" + str(corr) + "]" + ".png")
    plt.show()

    
#################################################################
#                                                               #
#    S T A G E   3                                              #
#                                                               #
#    Based on the optimized values for B1 and B2 we again       #
#    find the corresponding optimal A-value by solving:         #
#                                                               #
#     E[G] = theta * A * (a1 + a2)                (12)          #
#                                                               #
#    The solution will minimizes the objective function:        #
#                                                               #
#     C0 = (a1 + a2) / E[G]                                     #
#                                                               #
#    for the optimized values of B1 and B2                      #
#                                                               #
#################################################################

B1 = opt_B1
B2 = opt_B2

b1 = dist1.getUpperPercentile(B1)
b2 = dist2.getUpperPercentile(B2)

if B1 > 0:
    Bb1 = b1 * B1
else:
    Bb1 = 0
if B2 > 0:
    Bb2 = b2 * B2
else:
    Bb2 = 0

AA = np.linspace(max(B1, B2), 0.4, num_points)

LS = np.zeros(num_points)       # Left-hand side of eq (12)
RS = np.zeros(num_points)       # Right-hand side of eq (12)
C0 = np.zeros(num_points)       # Objective function

a1 = dist1.getUpperPercentile(AA[0])
a2 = dist2.getUpperPercentile(AA[0])

int1 = 0
int2 = 0

if B1 < B2:     # i.e.: b2 < b1 and (1-B2) < (1-B1)
    int1 = dist1.getIntervalPMean(1-B2, 1-B1)
if B2 < B1:     # i.e.: b1 < b2 and (1-B1) < (1-B2)
    int2 = dist2.getIntervalPMean(1-B2, 1-B1)
    
LS[0] = NT - theta * ((int1 + int2) - AA[0] * (a1 + a2) + Bb1 + Bb2)
RS[0] = theta * AA[0] * (a1 + a2)
C0[0] = (a1 + a2) / LS[0]

for i in range(1, num_points):
    a1 = dist1.getUpperPercentile(AA[i])
    a2 = dist2.getUpperPercentile(AA[i])
    int1 = dist1.getIntervalPMean(1-AA[i], 1-B1)
    int2 = dist2.getIntervalPMean(1-AA[i], 1-B2)
    LS[i] = NT - theta * ((int1 + int2) - AA[i] * (a1 + a2) + Bb1 + Bb2)
    RS[i] = theta * AA[i] * (a1 + a2)
    C0[i] = (a1 + a2) / LS[i]

diff_min = abs(LS[0] - RS[0])
opt_i = 0

for i in range(1, num_points):
    diff = abs(LS[i] - RS[i])
    if diff < diff_min:
        diff_min = diff
        opt_i = i
        
opt_A = AA[opt_i]
opt_a1 = dist1.getUpperPercentile(opt_A)
opt_a2 = dist2.getUpperPercentile(opt_A)
opt_C0 = C0[opt_i]


print("")
print("------------------------------------------------------------------------")
print("*** Stage 3 results for " + dist1.get_name() + " vs " + dist2.get_name() + " and corr = ", corr)
print("------------------------------------------------------------------------")
print("Opt. C0 = ", opt_C0) 
print("opt_A = ", opt_A,  ",  opt_B1 = ", B1,  ",  opt_B2 = ", B2)
print("opt_a1 = ", opt_a1,  ",  opt_a2 = ", opt_a2)
print("opt_b1 = ", opt_b1,  ",  opt_b2 = ", opt_b2)
print("------------------------------------------------------------------------")
print("")

if plot_LS_RS_2:  
    plt.plot(AA, LS, label='LS versus A')
    plt.plot(AA, RS, label='RS versus A')
    plt.xlabel('A')
    plt.ylabel('LS/RS')
    plt.title("Lefthand and righthand sides versus A")
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/LSRS_2_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
    plt.show()

if plot_objFunc_2:
    plt.plot(AA, C0, label='C0 versus A')
    plt.xlabel('A')
    plt.ylabel('C0')
    plt.title("Objective function versus A")
    plt.legend()
    if save_plots:
        plt.savefig("num_opt3/C0_2_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
    plt.show()


    
##############################################
#    S T A G E   4  ---  balanced            #
##############################################


if force_balanced:
    B1 = B_balanced
    B2 = B_balanced

    b1 = dist1.getUpperPercentile(B1)
    b2 = dist1.getUpperPercentile(B2)
    
    if B1 > 0:
        Bb1 = b1 * B1
    else:
        Bb1 = 0
    if B2 > 0:
        Bb2 = b1 * B2
    else:
        Bb2 = 0
    
    AA = np.linspace(max(B1, B2), 0.4, num_points)
    
    LS = np.zeros(num_points)       # Left-hand side of eq (12)
    RS = np.zeros(num_points)       # Right-hand side of eq (12)
    C0 = np.zeros(num_points)       # Objective function
    
    a1 = dist1.getUpperPercentile(AA[0])
    a2 = dist2.getUpperPercentile(AA[0])
    
    int1 = 0
    int2 = 0
    
    if B1 < B2:     # i.e.: b2 < b1 and (1-B2) < (1-B1)
        int1 = dist1.getIntervalPMean(1-B2, 1-B1)
    if B2 < B1:     # i.e.: b1 < b2 and (1-B1) < (1-B2)
        int2 = dist2.getIntervalPMean(1-B2, 1-B1)
        
    LS[0] = NT - theta * ((int1 + int2) - AA[0] * (a1 + a2) + Bb1 + Bb2)
    RS[0] = theta * AA[0] * (a1 + a2)
    C0[0] = (a1 + a2) / LS[0]
    
    for i in range(1, num_points):
        a1 = dist1.getUpperPercentile(AA[i])
        a2 = dist2.getUpperPercentile(AA[i])
        int1 = dist1.getIntervalPMean(1-AA[i], 1-B1)
        int2 = dist2.getIntervalPMean(1-AA[i], 1-B2)
        LS[i] = NT - theta * ((int1 + int2) - AA[i] * (a1 + a2) + Bb1 + Bb2)
        RS[i] = theta * AA[i] * (a1 + a2)
        C0[i] = (a1 + a2) / LS[i]
    
    diff_min = abs(LS[0] - RS[0])
    opt_i = 0
    
    for i in range(1, num_points):
        diff = abs(LS[i] - RS[i])
        if diff < diff_min:
            diff_min = diff
            opt_i = i
            
    opt_A = AA[opt_i]
    opt_a1 = dist1.getUpperPercentile(opt_A)
    opt_a2 = dist2.getUpperPercentile(opt_A)
    opt_C0 = C0[opt_i]
    
    print("")
    print("------------------------------------------------------------------------")
    print("*** Stage 4 (balanced) results for " + dist1.get_name() + " vs " + dist2.get_name() )
    print("------------------------------------------------------------------------")
    print("Opt. C0 = ", opt_C0) 
    print("opt_A = ", opt_A,  ",  opt_B1 = ", B1,  ",  opt_B2 = ", B2)
    print("opt_a1 = ", opt_a1,  ",  opt_a2 = ", opt_a2)
    print("opt_b1 = ", b1,  ",  opt_b2 = ", b2)
    print("------------------------------------------------------------------------")
    
    if plot_LS_RS_3:  
        plt.plot(AA, LS, label='LS versus A')
        plt.plot(AA, RS, label='RS versus A')
        plt.xlabel('A')
        plt.ylabel('LS/RS')
        plt.title("Lefthand and righthand sides versus A")
        plt.legend()
        if save_plots:
            plt.savefig("num_opt3/LSRS_3_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
        plt.show()
    
    if plot_objFunc_3:
        plt.plot(AA, C0, label='C0 versus A')
        plt.xlabel('A')
        plt.ylabel('C0')
        plt.title("Objective function versus A")
        plt.legend()
        if save_plots:
            plt.savefig("num_opt3/C0_3_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
        plt.show()
    

##############################################
#    S T A G E   4a  ---  unbalanced          #
##############################################

if force_unbalanced:
    B1 = B1_max
    B2 = 0

    b1 = dist1.getUpperPercentile(B1)
    b2 = dist1.getUpperPercentile(B2)
    
    Bb1 = b1 * B1
    Bb2 = 0
    
    AA = np.linspace(B1_max, 0.4, num_points)
    
    LS = np.zeros(num_points)       # Left-hand side of eq (12)
    RS = np.zeros(num_points)       # Right-hand side of eq (12)
    C0 = np.zeros(num_points)       # Objective function
    
    a1 = dist1.getUpperPercentile(AA[0])
    a2 = dist2.getUpperPercentile(AA[0])
    
    int1 = 0
    int2 = 0
    
    if B1 < B2:     # i.e.: b2 < b1 and (1-B2) < (1-B1)
        int1 = dist1.getIntervalPMean(1-B2, 1-B1)
    if B2 < B1:     # i.e.: b1 < b2 and (1-B1) < (1-B2)
        int2 = dist2.getIntervalPMean(1-B2, 1-B1)
        
    LS[0] = NT - theta * ((int1 + int2) - AA[0] * (a1 + a2) + Bb1 + Bb2)
    RS[0] = theta * AA[0] * (a1 + a2)
    C0[0] = (a1 + a2) / LS[0]
    
    for i in range(1, num_points):
        a1 = dist1.getUpperPercentile(AA[i])
        a2 = dist2.getUpperPercentile(AA[i])
        int1 = dist1.getIntervalPMean(1-AA[i], 1-B1)
        int2 = dist2.getIntervalPMean(1-AA[i], 1-B2)
        LS[i] = NT - theta * ((int1 + int2) - AA[i] * (a1 + a2) + Bb1 + Bb2)
        RS[i] = theta * AA[i] * (a1 + a2)
        C0[i] = (a1 + a2) / LS[i]
    
    diff_min = abs(LS[0] - RS[0])
    opt_i = 0
    
    for i in range(1, num_points):
        diff = abs(LS[i] - RS[i])
        if diff < diff_min:
            diff_min = diff
            opt_i = i
            
    opt_A = AA[opt_i]
    opt_a1 = dist1.getUpperPercentile(opt_A)
    opt_a2 = dist2.getUpperPercentile(opt_A)
    opt_C0 = C0[opt_i]
    
    print("")
    print("------------------------------------------------------------------------")
    print("*** Stage 4a (unbalanced) results for " + dist1.get_name() + " vs " + dist2.get_name() )
    print("------------------------------------------------------------------------")
    print("Opt. C0 = ", opt_C0) 
    print("opt_A = ", opt_A,  ",  opt_B1 = ", B1,  ",  opt_B2 = ", B2)
    print("opt_a1 = ", opt_a1,  ",  opt_a2 = ", opt_a2)
    print("opt_b1 = ", b1,  ",  opt_b2 = ", b2)
    print("------------------------------------------------------------------------")
    
    if plot_LS_RS_3:  
        plt.plot(AA, LS, label='LS versus A')
        plt.plot(AA, RS, label='RS versus A')
        plt.xlabel('A')
        plt.ylabel('LS/RS')
        plt.title("Lefthand and righthand sides versus A")
        plt.legend()
        if save_plots:
            plt.savefig("num_opt3/LSRS_4a_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
        plt.show()
    
    if plot_objFunc_3:
        plt.plot(AA, C0, label='C0 versus A')
        plt.xlabel('A')
        plt.ylabel('C0')
        plt.title("Objective function versus A")
        plt.legend()
        if save_plots:
            plt.savefig("num_opt3/C0_4a_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
        plt.show()


##############################################
#    S T A G E   4b  ---  unbalanced          #
##############################################

if force_unbalanced:
    B1 = 0
    B2 = B2_max

    b1 = dist1.getUpperPercentile(B1)
    b2 = dist1.getUpperPercentile(B2)
    
    Bb1 = 0
    Bb2 = b2 * B2
    
    AA = np.linspace(B2_max, 0.4, num_points)
    
    LS = np.zeros(num_points)       # Left-hand side of eq (12)
    RS = np.zeros(num_points)       # Right-hand side of eq (12)
    C0 = np.zeros(num_points)       # Objective function
    
    a1 = dist1.getUpperPercentile(AA[0])
    a2 = dist2.getUpperPercentile(AA[0])
    
    int1 = 0
    int2 = 0
    
    if B1 < B2:     # i.e.: b2 < b1 and (1-B2) < (1-B1)
        int1 = dist1.getIntervalPMean(1-B2, 1-B1)
    if B2 < B1:     # i.e.: b1 < b2 and (1-B1) < (1-B2)
        int2 = dist2.getIntervalPMean(1-B2, 1-B1)
        
    LS[0] = NT - theta * ((int1 + int2) - AA[0] * (a1 + a2) + Bb1 + Bb2)
    RS[0] = theta * AA[0] * (a1 + a2)
    C0[0] = (a1 + a2) / LS[0]
    
    for i in range(1, num_points):
        a1 = dist1.getUpperPercentile(AA[i])
        a2 = dist2.getUpperPercentile(AA[i])
        int1 = dist1.getIntervalPMean(1-AA[i], 1-B1)
        int2 = dist2.getIntervalPMean(1-AA[i], 1-B2)
        LS[i] = NT - theta * ((int1 + int2) - AA[i] * (a1 + a2) + Bb1 + Bb2)
        RS[i] = theta * AA[i] * (a1 + a2)
        C0[i] = (a1 + a2) / LS[i]
    
    diff_min = abs(LS[0] - RS[0])
    opt_i = 0
    
    for i in range(1, num_points):
        diff = abs(LS[i] - RS[i])
        if diff < diff_min:
            
            diff_min = diff
            opt_i = i
            
    opt_A = AA[opt_i]
    opt_a1 = dist1.getUpperPercentile(opt_A)
    opt_a2 = dist2.getUpperPercentile(opt_A)
    opt_C0 = C0[opt_i]
    
    print("")
    print("------------------------------------------------------------------------")
    print("*** Stage 4b (unbalanced) results for " + dist1.get_name() + " vs " + dist2.get_name() )
    print("------------------------------------------------------------------------")
    print("Opt. C0 = ", opt_C0) 
    print("opt_A = ", opt_A,  ",  opt_B1 = ", B1,  ",  opt_B2 = ", B2)
    print("opt_a1 = ", opt_a1,  ",  opt_a2 = ", opt_a2)
    print("opt_b1 = ", b1,  ",  opt_b2 = ", b2)
    print("------------------------------------------------------------------------")
    
    if plot_LS_RS_3:  
        plt.plot(AA, LS, label='LS versus A')
        plt.plot(AA, RS, label='RS versus A')
        plt.xlabel('A')
        plt.ylabel('LS/RS')
        plt.title("Lefthand and righthand sides versus A")
        plt.legend()
        if save_plots:
            plt.savefig("num_opt3/LSRS_4b_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
        plt.show()
    
    if plot_objFunc_3:
        plt.plot(AA, C0, label='C0 versus A')
        plt.xlabel('A')
        plt.ylabel('C0')
        plt.title("Objective function versus A")
        plt.legend()
        if save_plots:
            plt.savefig("num_opt3/C0_4b_" + dist1.get_name() + "_" + dist2.get_name() + ".png")
        plt.show()
        
#x1 = np.zeros(num_points)
#r1 = np.zeros(num_points)
#x2 = np.zeros(num_points)
#r2 = np.zeros(num_points)
#x3 = np.zeros(num_points)
#r3 = np.zeros(num_points)
#x4 = np.zeros(num_points)
#r4 = np.zeros(num_points)
#
#def check_Hazard_Type(r, num_points):
#    if (max(r) == r[num_points]):
#        return 0
#    else:
#        return 1
#
#def hazard_rate(dist, x, r):
#
#    x_min = dist.getLowerPercentile(B1)
#    x_max = dist.getLowerPercentile(B2)
#    
#    for i in range(num_points):
#        x[i] = x_min + i * (x_max - x_min) / num_points
#        r[i] = dist.getHazardRate(x[i])
#    return x, r        
#
#rHazard1 = hazard_rate(dist1, x1, r1)
#rHazard2 = hazard_rate(dist1, x2, r2)
#rHazard3 = hazard_rate(dist1, x3, r3)
#rHazard4 = hazard_rate(dist1, x4, r4)
#
#if plot_hazard_rate:
#    plt.plot(xHazard1, rHazard1)
#    plt.plot(xHazard2, rhazard2)
#    plt.plot(xHazard3, rHazard3)
#    plt.plot(xHazard4, rHazard4)
 
