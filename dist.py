###########################################################################
#                                                                         #
#    Fast jit version of the dist-library                                 #
#                                                                         #
###########################################################################

from random import *
from math import *
from scipy.optimize import minimize
from scipy.stats import gamma
from cmath import pi

from numba import jit
import numpy as np


# Alternative distributions defined in random:
#   uniform(a, b)
#   triangular(low, high, mode)
#   betavariate(alpha, beta)
#   expovariate(lambd)
#   gammavariate(alpha, beta)
#   gauss(mu, sigma)
#   lognormvariate(mu, sigma)
#   normalvariate(mu, sigma)
#   vonmisesvariate(mu, kappa)
#   paretovariate(alpha)
#   weibullvariate(alpha, beta)

SQRTPI: float = 1.77245385090551602729816748334     # sqrt(pi)
SQRT2PI: float = 2.50662827463100029                # sqrt(2*pi)

EULER_MASCHERONI: float = 0.57721566490153286060

P_MEAN_SIZE = 25000



@jit(nopython=True)
def gaussian_pdf(x) -> float:
    return exp(-x * x / 2.0) / SQRT2PI


@jit(nopython=True)
def gaussian_cdf(x) -> float:
    if x >= 0.0:
        t = 1.0/(1.0 + 0.33267 * x)
        return 1.0 - gaussian_pdf(x) * (0.4361836*t - 0.1201676*t*t + 0.9372980*t*t*t)
    else:
        t = 1.0/(1.0 - 0.33267 * x);
        return gaussian_pdf(x) * (0.4361836*t - 0.1201676*t*t + 0.9372980*t*t*t)


@jit(nopython=True)
def gaussian_sdf(x) -> float:
    if x >= 0.0:
        t = 1.0/(1.0 + 0.33267 * x)
        return gaussian_pdf(x) * (0.4361836*t - 0.1201676*t*t + 0.9372980*t*t*t)
    else:
        t = 1.0/(1.0 - 0.33267 * x);
        return 1.0 - gaussian_pdf(x) * (0.4361836*t - 0.1201676*t*t + 0.9372980*t*t*t)


@jit(nopython=True)
def gaussian_invcdf(u) -> float:
    if u < 0.5:
        t = sqrt(log(1.0/(u*u)))
        return -t + (2.515517 + 0.802853*t + 0.010328*t*t) / (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
    elif u == 0.5:
        return 0.0
    else:
        t = sqrt(log(1.0/((1.0 - u)*(1.0 - u))))
        return t - (2.515517 + 0.802853*t + 0.010328*t*t) / (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)


@jit(nopython=True)
def gaussian_invsdf(u) -> float:
    if u < 0.5:
        t = sqrt(log(1.0/(u*u)))
        return t - (2.515517 + 0.802853*t + 0.010328*t*t) / (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
    elif u == 0.5:
        return 0.0
    else:
        t = sqrt(log(1.0/((1.0 - u)*(1.0 - u))))
        return -t + (2.515517 + 0.802853*t + 0.010328*t*t) / (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)



###############################################################
#    Generate a univariate uniform variable restricted to     #
#    the importance sample region:                            #
#        D = [1-r, 1]                                         #
###############################################################
def uni_uniform_D(r):
    return uniform(1-r,1)


###############################################################
#    Generate a univariate uniform variable restricted to     #
#    the importance sample region:                            #
#        E = [0, 1-r]                                         #
###############################################################
def uni_uniform_E(r):
    return uniform(0,1-r)


###############################################################
#    Generate bivariate uniform variables restricted to       #
#    the importance sample region:                            #
#        D = ([1-r, 1] x [0, 1]) union ([0, 1] x [1-r, 1])    #
###############################################################
def bi_uniform_D(r):
    u0 = uniform(0,1) * (2-r)
    u1 = uniform(0,1)
    u2 = uniform(0,1)
    v1 = 0
    v2 = 0
    if u0 < (1-r):
        v1 = 1-(r * u1)
        v2 = (1-r) * u2
    elif u0 < 2*(1-r):
        v1 = (1-r) * u1
        v2 = 1-(r * u2)
    else:
        v1 = 1-(r * u1)
        v2 = 1-(r * u2)
    return v1, v2

###############################################################
#     Generate bivariate uniform variables restricted to      #
#     the complement of importance sample region:             #
#         E = [0, 1-r] x [0, 1-r]                             #
###############################################################
def bi_uniform_E(r):
    u1 = uniform(0,1-r)
    u2 = uniform(0,1-r)
    return u1, u2


################################################################
#                                                              #
#     Calculate mean and standard deviation of a mixture       #
#     of distributions.                                        #
#                                                              #
################################################################
def mixtureMoments(dists, weights):
    n = len(weights)
    eMean = 0.0
    eMean2 = 0.0
    eVar = 0.0
    for i in range(n):
        eMean += dists[i].getMean() * weights[i]
        eMean2 += dists[i].getMean() * dists[i].getMean() * weights[i]
        eVar += dists[i].getVar() * weights[i]
    mean = eMean
    variance = eVar + (eMean2 - eMean * eMean)
    return mean, sqrt(variance)

################################################################
#                                                              #
#     Calculate min and max of a mixture                       #
#     of distributions.                                        #
#                                                              #
################################################################
def mixtureMinMax(dists, weights):
    n = len(weights)
    min = dists[0].getMin()
    max = dists[0].getMax()
    for i in range(1, n):
        if dists[i].getMin() < min:
            min = dists[i].getMin()
        if dists[i].getMax() > max:
            max = dists[i].getMax()
    return min, max


################################################################
#                                                              #
#     The base Distribution class                              #
#                                                              #
################################################################

class Distribution:
    def __init__(self, name, mean, stdev):
        self.name: str = name
        self.mean: float = mean
        self.stdev: float = stdev
        self.min: float = -np.inf
        self.max: float = np.inf
        self.pmean = []
        self.pmean_len = 0

    def print_name(self):
        print(self.get_name())

    def get_name(self):
        namestr = self.name + "({0}, {1})"
        return namestr.format(self.mean, self.stdev)

    def getMean(self):
        return self.mean

    def getStdev(self):
        return self.stdev

    def getVar(self):
        return self.stdev * self.stdev

    def getMin(self):
        return self.min

    def getMax(self):
        return self.max

    def getPDF(self, x):
        pass

    def getCDF(self, x):
        pass

    def getSDF(self, x):
        pass

    def calcPMean(self, num_points = P_MEAN_SIZE):
        self.pmean = np.zeros(num_points + 1)
        self.pmean_len = num_points + 1
        integral = 0
        x_old = self.min
        f_old = self.getPDF(x_old)
        for i in range(1, num_points):
            p = i / num_points
            x = self.lowerPercentile(p)
            f = self.getPDF(x)
            dx = x - x_old
            integral += (x * f + x_old * f_old) * 0.5 * dx
            self.pmean[i] = integral
            x_old = x
            f_old = f
        self.pmean[num_points] = self.mean

    def getPMeanLength(self):
        if self.pmean_len == 0:
            self.calcPMean()
        return self.mean_len

    def getLowerPMeanAt(self, i):
        if self.pmean_len == 0:
            self.calcPMean()
        if i <= 0:
            return 0
        elif i >= self.pmean_len:
            return self.mean
        else:
            return self.pmean[i]

    def getPMean(self):
        if self.pmean_len == 0:
            self.calcPMean()
        return self.pmean

    def getLowerPMean(self, p):
        if self.pmean_len == 0:
            self.calcPMean()
        if p == 0:
            return 0
        elif p == 1:
            return self.mean
        else:
            i_L = floor((self.pmean_len - 1) * p)
            i_U = ceil((self.pmean_len - 1) * p)
            if i_L == i_U:
                return self.pmean[i_L]
            elif i_L == i_U - 1:
                i_M = (self.pmean_len - 1) * p
                return self.pmean[i_L] * (i_U - i_M) + self.pmean[i_U] * (i_M - i_L)
            else:
                print("ERROR: Interpolation failed. i_L = " + str(i_L) + ", i_U = " + str(i_U))
                return self.pmean[i_L + 1]

    def getUpperPMean(self, p):
        return self.mean - self.getLowerPMean(1-p)

    def getIntervalPMean(self, p1, p2):
        return self.getLowerPMean(p2) - self.getLowerPMean(p1)

    def getHazardRate(self, x):
        f = self.getPDF(x)
        s = self.getSDF(x)
        if s > 0:
            return f/s
        else:
            return 0

    def lowerPercentile(self, p):
        pass

    def getLowerPercentile(self, p):
        if p <= 0:
            return self.min
        elif p >= 1:
            return self.max
        else:
            return self.lowerPercentile(p)

    def getUpperPercentile(self, p):
        if p <= 0:
            return self.max
        elif p >= 1:
            return self.min
        else:
            return self.lowerPercentile(1-p)

    def getStochasticValue(self):
        u = uniform(0,1)
        return self.getLowerPercentile(u)


################################################################
#                                                              #
#     The Uniform distribution class                           #
#                                                              #
################################################################

SQRT12 = 3.464101615137755

class Uniform(Distribution):
    def __init__(self, low, high):
        super().__init__("UNIFORM", (low + high)/2, (high - low) / SQRT12)
        self.a = low
        self.b = high
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ")")

    def getPDF(self, x):
        if x < self.a:
            return 0
        elif x <= self.b:
            return 1 / (self.b - self.a)
        else:
            return 0

    def getCDF(self, x):
        if x < self.a:
            return 0
        elif x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1

    def getSDF(self, x):
        if x < self.a:
            return 1
        elif x <= self.b:
            return (self.b - x) / (self.b - self.a)
        else:
            return 0

    def lowerPercentile(self, p):
        return self.a + (self.b - self.a) * p

    def getStochasticValue(self):
        return uniform(self.a, self.b)


################################################################
#                                                              #
#     The Normal distribution class                            #
#                                                              #
################################################################

class Normal(Distribution):
    def __init__(self, mean, stdev):
        super().__init__("NORMAL", mean, stdev)
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ")")

    def getPDF(self, x):
        return gaussian_pdf((x-self.mean) / self.stdev) / self.stdev

    def getCDF(self, x) -> float:
        return gaussian_cdf((x-self.mean) / self.stdev)

    def getSDF(self, x):
        return gaussian_sdf((x-self.mean) / self.stdev)

    def lowerPercentile(self, p):
        return self.stdev * gaussian_invcdf(p) + self.mean


################################################################
#                                                              #
#     The Lognormal distribution class                         #
#                                                              #
################################################################

class Lognormal(Distribution):
    def __init__(self, mean, stdev):
        super().__init__("LOGNORMAL", mean, stdev)
        self.h = 1.0 + (stdev * stdev) / (mean * mean)
        self.s = sqrt(log(self.h))
        self.m = log(self.mean) - 0.5 * log(self.h)
        self.min = 0
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ", logmean = ", self.m, ", logstdev = ", self.s, ")")

    def getPDF(self, x):
        if x > self.min:
            return gaussian_pdf((log(x)-self.m) / self.s) / (self.s * x)
        else:
            return 0

    def getCDF(self, x):
        if x > self.min:
            return gaussian_cdf((log(x)-self.m) / self.s)
        else:
            return 0

    def getSDF(self, x):
        if x > self.min:
            return gaussian_sdf((log(x)-self.m) / self.s)
        else:
            return 1

    def lowerPercentile(self, p):
        return self.mean * exp(self.s * gaussian_invcdf(p)) / sqrt(self.h)


################################################################
#                                                              #
#     The Exponential distribution class                       #
#                                                              #
################################################################

class Exponential(Distribution):
    def __init__(self, mean):
        super().__init__("EXPONENTIAL", mean, mean)
        self.stdev = mean
        self.rate = 1 / mean
        self.min = 0
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev,  ", rate = ", self.rate, ")")

    def getPDF(self, x):
        if x > self.min:
            return self.rate * exp(-self.rate * x)
        else:
            return 0

    def getCDF(self, x):
        if x > self.min:
            return 1 - exp(-self.rate * x)
        else:
            return 0

    def getSDF(self, x):
        if x > self.min:
            return exp(-self.rate * x)
        else:
            return 1

    def lowerPercentile(self, p):
        return -log(1-p) * self.mean


################################################################
#                                                              #
#     The Gamma distribution class                             #
#                                                              #
################################################################

class Gamma(Distribution):
    def __init__(self, mean, stdev):
        super().__init__("GAMMA", mean, stdev)
        self.alpha = (mean * mean) / (stdev * stdev)
        self.beta = self.alpha / mean
        self.min = 0
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ", alpha = ", self.alpha,  ", beta = ", self.beta, ")")

    def getPDF(self, x):
        if x > self.min:
            return gamma.pdf(self.beta * x, self.alpha) * self.beta
        else:
            return 0

    def getCDF(self, x):
        if x > self.min:
            return gamma.cdf(self.beta * x, self.alpha)
        else:
            return 0

    def getSDF(self, x):
        if x > self.min:
            return 1 - gamma.cdf(self.beta * x, self.alpha)
        else:
            return 1

    def lowerPercentile(self, p):
        return  gamma.ppf(p, self.alpha) / self.beta



################################################################
#                                                              #
#      Moment matching for truncnormal distributions           #
#                                                              #
#      The algorithm assumes that the ratio (s*s) / (m*m)      #
#      is between 0.04 and 65.0                                #
#                                                              #
#      The method is descibed in:                              #
#        Huseby (2022) The Truncated normal distribution       #
#                                                              #
################################################################

ratio_min = 0.04
ratio_max = 65.0

# Search interval for alpha = - mu / sigma
alpha_min = -5
alpha_max = 5

epsilon: float = (alpha_max - alpha_min) / 2**20


def phi_ratio(x):
    return gaussian_pdf(x) / (1 - gaussian_cdf(x))


def righthand(x):
    pr_x = phi_ratio(x)
    return (1 + x * pr_x - pr_x * pr_x) / ((pr_x - x)*(pr_x - x))


def find_alpha(m, s):
    ratio = (s * s) / (m * m)
    if ratio < ratio_min:
        print("ERROR: Ratio between variance and mean square is too small!")
    elif ratio > ratio_max:
        print("ERROR: Ratio between variance and mean square is too large!")
    L = alpha_min
    U = alpha_max
    x = (L + U) / 2
    while U - L > epsilon:
        r = righthand(x)
        if r < ratio:
            L = x
        else:
            U = x
        x = (L + U) / 2
    return x


################################################################
#                                                              #
#     The Truncnormal distribution class                       #
#                                                              #
################################################################

class Truncnormal(Distribution):
    def __init__(self, mean, stdev):
        super().__init__("TNORMAL", mean, stdev)
        alpha = find_alpha(mean, stdev)
        self.sigma = mean * (1 - gaussian_cdf(alpha)) / (gaussian_pdf(alpha) - alpha*(1 - gaussian_cdf(alpha)))
        self.mu = -alpha * self.sigma
        self.omega = gaussian_cdf(-self.mu/self.sigma)
        self.min = 0
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ", alpha = ", alpha, ", sigma = ", self.sigma, ", mu = ", self.mu, ", omega = ", self.omega, ")")

    def getPDF(self, x):
        if x > self.min:
            return (gaussian_pdf((x-self.mu) / self.sigma) / self.sigma) / (1 - self.omega)
        else:
            return 0

    def getCDF(self, x):
        return 1 - self.getSDF(x)

    def getSDF(self, x):
        if x > self.min:
            return gaussian_sdf((x-self.mu) / self.sigma) / (1 - self.omega)
        else:
            return 1

    def lowerPercentile(self, p):
        return self.mu + self.sigma * gaussian_invcdf(self.omega + p * (1 - self.omega))


################################################################
#                                                              #
#      The Pareto distribution class                           #
#                                                              #
#      For details see:                                        #
#        Huseby (2021)  Pareto distributions                   #
#      or alternatively:                                       #
#        https://en.wikipedia.org/wiki/Pareto_distribution     #
#                                                              #
################################################################

class Pareto(Distribution):
    def __init__(self, mean, stdev):
        super().__init__("PARETO", mean, stdev)
        self.alpha: float = sqrt((mean * mean) / (stdev * stdev) + 1) + 1
        self.sigma: float = mean * (self.alpha - 1) / self.alpha
        self.min = self.sigma
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ", alpha = ", self.alpha, ", sigma = ", self.sigma, ")")

    def getPDF(self, x):
        if x > self.min:
            return self.alpha * (self.min**self.alpha) / (x**(self.alpha+1))
        else:
            return 0

    def getCDF(self, x):
        if x > self.min:
            return 1 - (self.min / x)**self.alpha
        else:
            return 0

    def getSDF(self, x):
        if x > self.min:
            return (self.min / x)**self.alpha
        else:
            return 1

    def lowerPercentile(self, p):
        return self.sigma * ((1-p)**(-1/self.alpha))


################################################################
#                                                              #
#      The Pareto2 distribution class                          #
#                                                              #
#      For details see:                                        #
#        Huseby (2021)  Pareto distributions                   #
#      or alternatively:                                       #
#        https://en.wikipedia.org/wiki/Pareto_distribution     #
#                                                              #
################################################################

class Pareto2(Distribution):
    def __init__(self, mean, stdev, x0):
        super().__init__("PARETO2", mean, stdev)
        self.mu = x0
        self.alpha = 2 * stdev**2 /(stdev**2 - (mean - x0)**2)
        self.sigma = (mean - x0) * (self.alpha - 1)
        self.min = self.mu
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ", alpha = ", self.alpha, ", sigma = ", self.sigma, ", mu = ", self.mu, ")")

    def getPDF(self, x):
        if x > self.min:
            return self.alpha * (1 + (x - self.mu) / self.sigma)**(-self.alpha - 1) / self.sigma
        else:
            return 0

    def getCDF(self, x):
        if x > self.min:
            return 1 - (1 + (x - self.mu) / self.sigma)**(-self.alpha)
        else:
            return 0

    def getSDF(self, x):
        if x > self.min:
            return (1 + (x - self.mu) / self.sigma)**(-self.alpha)
        else:
            return 1

    def lowerPercentile(self, p):
        return self.sigma * ((1-p)**(-1/self.alpha)) + self.mu - self.sigma


################################################################
#                                                              #
#      The Lomax distribution class                            #
#                                                              #
#      For details see:                                        #
#        Huseby (2021)  Pareto distributions                   #
#      or alternatively:                                       #
#        https://en.wikipedia.org/wiki/Pareto_distribution     #
#                                                              #
################################################################

class Lomax(Distribution):
    def __init__(self, mean, stdev):
        super().__init__("LOMAX", mean, stdev)
        self.alpha = 2 * stdev**2 /(stdev**2 - mean**2)
        self.sigma = mean * (self.alpha - 1)
        self.min = 0
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ", alpha = ", self.alpha, ", sigma = ", self.sigma, ")")

    def getPDF(self, x):
        if x > self.min:
            return self.alpha * (1 + x / self.sigma)**(-self.alpha - 1) / self.sigma
        else:
            return 0

    def getCDF(self, x):
        if x > self.min:
            return 1 - (1 + x / self.sigma)**(-self.alpha)
        else:
            return 0

    def getSDF(self, x):
        if x > self.min:
            return (1 + x / self.sigma)**(-self.alpha)
        else:
            return 1

    def lowerPercentile(self, p):
        return self.sigma * ((1-p)**(-1/self.alpha)) - self.sigma


################################################################
#                                                              #
#    Fast jit-compiled function for finding the                #
#    lower percentile of a Mixture distribution                #
#                                                              #
################################################################

def mixtureLowerPercentile(mdist, p):
    n = len(mdist.weights)
    lo = np.inf
    hi = -np.inf
    for i in range(n):
        y = mdist.dists[i].lowerPercentile(p)
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    if lo >= hi:
        return 0.5 * (lo + hi)

    y = 0.5 * (lo + hi)

    for _ in range(20):
        p0 = mdist.getCDF(y)
        if p0 <= p:
            lo = y
        elif p0 >= p:
            hi = y
        y = 0.5 * (lo + hi)

    return y


################################################################
#                                                              #
#     The Mixture distribution class                           #
#                                                              #
################################################################

class Mixture(Distribution):
    def __init__(self, d, w):
        res = mixtureMoments(d, w)
        super().__init__("MIXTURE", res[0], res[1])
        self.dists = d
        self.weights = w
        res2 = mixtureMinMax(d, w)
        self.min = res2[0]
        self.max = res2[1]
        print(self.name, "(mean = ", res[0], ", stdev = ", res[1], ", min = ", res2[0], ", max = ", res2[1], ")")

    def getPDF(self, x):
        n = len(self.weights)
        pdf = 0
        for i in range(n):
            pdf += self.dists[i].getPDF(x) * self.weights[i]
        return pdf

    def getCDF(self, x):
        n = len(self.weights)
        cdf = 0
        for i in range(n):
            cdf += self.dists[i].getCDF(x) * self.weights[i]
        return cdf

    def getSDF(self, x):
        n = len(self.weights)
        sdf = 0
        for i in range(n):
            sdf += self.dists[i].getSDF(x) * self.weights[i]
        return sdf

    def lowerPercentile(self, p):
        return mixtureLowerPercentile(self, p)

    def getStochasticValue(self):
        n = len(self.weights)
        j = 0
        v = uniform(0,1)
        for i in range(n):
            if self.weights[i] < v:
                j+=1
            v -= self.weight[i]
        return self.dists[j].getStochasticValue()


################################################################
#                                                              #
#     The MixedMeanTruncnormal distribution class              #
#                                                              #
################################################################

@jit(nopython=True)
def mixedMeanTruncnormalLowerPercentile(min_val, mu, sigma, omega, mixcnt, p):
    lo = np.inf
    hi = -np.inf
    for i in range(mixcnt):
        y = mu[i] + sigma * gaussian_invcdf(omega + p * (1 - omega))
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    if lo >= hi:
        return 0.5 * (lo + hi)

    y = 0.5 * (lo + hi)
    mixwgt = 1 / mixcnt

    for _ in range(20):
        sdf = 1
        if y > min_val:
            sdf = 0
            for i in range(mixcnt):
                sdf += mixwgt * gaussian_sdf((y-mu[i]) / sigma) / (1 - omega)
        p0 = 1 - sdf
        if p0 <= p:
            lo = y
        elif p0 >= p:
            hi = y
        y = 0.5 * (lo + hi)

    return y

class MixedMeanTruncnormal(Distribution):
    def __init__(self, mean, stdev, mixdist, mixcnt):
        super().__init__("MIXEDMEANTNORMAL", mean, stdev)
        alpha = find_alpha(mean, stdev)
        self.sigma = mean * (1 - gaussian_cdf(alpha)) / (gaussian_pdf(alpha) - alpha*(1 - gaussian_cdf(alpha)))
        m = -alpha * self.sigma
        self.omega = gaussian_cdf(-m/self.sigma)
        self.mixdist = mixdist
        self.mixcnt = mixcnt
        self.mixwgt = 1 / mixcnt
        self.mu = np.zeros(mixcnt)
        for i in range(mixcnt):
            perc = self.mixdist.lowerPercentile(self.mixwgt * (i + 0.5))
            self.mu[i] = m * perc
        self.min = 0
        print(self.name, "(mean = ", mean, ", stdev = ", self.stdev, ")")

    def getPDF(self, x):
        if x > self.min:
            pdf = 0
            for i in range(self.mixcnt):
                pdf += self.mixwgt * (gaussian_pdf((x-self.mu[i]) / self.sigma) / self.sigma) / (1 - self.omega)
            return pdf
        else:
            return 0

    def getCDF(self, x):
        return 1 - self.getSDF(x)

    def getSDF(self, x):
        if x > self.min:
            sdf = 0
            for i in range(self.mixcnt):
                sdf += self.mixwgt * gaussian_sdf((x-self.mu[i]) / self.sigma) / (1 - self.omega)
            return sdf
        else:
            return 1

    def lowPerc(self, p, i):
        return self.mu[i] + self.sigma * gaussian_invcdf(self.omega + p * (1 - self.omega))

    def lowerPercentile(self, p):
        return mixedMeanTruncnormalLowerPercentile(self.min, self.mu, self.sigma, self.omega, self.mixcnt, p)

    def getStochasticValue(self):
        i = floor(uniform(0, self.mixcnt))
        p = uniform(0, 1)
        return self.lowPerc(p, i)



################################################################
#                                                              #
#     The MixedStdevTruncnormal distribution class             #
#                                                              #
################################################################

@jit(nopython=True)
def mixedStdevTruncnormalLowerPercentile(minimum, mu, sigma, omega, mixcnt, p):
    lo = np.inf
    hi = -np.inf
    for i in range(mixcnt):
        y = mu + sigma[i] * gaussian_invcdf(omega[i] + p * (1 - omega[i]))
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    if lo >= hi:
        return 0.5 * (lo + hi)

    y = 0.5 * (lo + hi)
    mixwgt = 1 / mixcnt

    for _ in range(20):
        sdf = 1
        if y > minimum:
            sdf = 0
            for i in range(mixcnt):
                sdf += mixwgt * gaussian_sdf((y-mu) / sigma[i]) / (1 - omega[i])
        p0 = 1 - sdf
        if p0 <= p:
            lo = y
        elif p0 >= p:
            hi = y
        y = 0.5 * (lo + hi)

    return y

class MixedStdevTruncnormal(Distribution):
    def __init__(self, mean, stdev, mixdist, mixcnt):
        super().__init__("MIXEDSTDEVTNORMAL", mean, stdev)
        alpha = find_alpha(mean, stdev)
        sig = mean * (1 - gaussian_cdf(alpha)) / (gaussian_pdf(alpha) - alpha*(1 - gaussian_cdf(alpha)))
        self.mu = -alpha * sig
        self.mixdist = mixdist
        self.mixcnt = mixcnt
        self.mixwgt = 1 / mixcnt
        self.sigma = np.zeros(mixcnt)
        self.omega = np.zeros(mixcnt)
        for i in range(mixcnt):
            perc = self.mixdist.lowerPercentile(self.mixwgt * (i + 0.5))
            self.sigma[i] = sig * perc
            self.omega[i] = gaussian_cdf(-self.mu/(sig * perc))
        self.min = 0
        print(self.name, "(mean = ", mean, ", stdev = ", self.stdev, ")")

    def getPDF(self, x):
        if x > self.min:
            pdf = 0
            for i in range(self.mixcnt):
                pdf += self.mixwgt * (gaussian_pdf((x-self.mu) / self.sigma[i]) / self.sigma[i]) / (1 - self.omega[i])
            return pdf
        else:
            return 0

    def getCDF(self, x):
        return 1 - self.getSDF(x)

    def getSDF(self, x):
        if x > self.min:
            sdf = 0
            for i in range(self.mixcnt):
                sdf += self.mixwgt * gaussian_sdf((x-self.mu) / self.sigma[i]) / (1 - self.omega[i])
            return sdf
        else:
            return 1

    def lowPerc(self, p, i):
        return self.mu + self.sigma[i] * gaussian_invcdf(self.omega[i] + p * (1 - self.omega[i]))

    def lowerPercentile(self, p):
        return mixedStdevTruncnormalLowerPercentile(self.min, self.mu, self.sigma, self.omega, self.mixcnt, p)

    def getStochasticValue(self):
        i = floor(uniform(0, self.mixcnt))
        p = uniform(0, 1)
        return self.lowPerc(p, i)



################################################################
#                                                              #
#     The MixedExponential distribution class                  #
#                                                              #
################################################################

@jit(nopython=True)
def mixedExponentialLowerPercentile(minimum, mu, mixcnt, p):
    lo = np.inf
    hi = -np.inf
    for i in range(mixcnt):
        y = -log(1-p) * mu[i]
        if y < lo:
            lo = y
        if y > hi:
            hi = y
    if lo >= hi:
        return 0.5 * (lo + hi)

    y = 0.5 * (lo + hi)
    mixwgt = 1 / mixcnt

    for _ in range(20):
        p0 = 0
        if y > minimum:
            for i in range(mixcnt):
                p0 += mixwgt * (1 - exp(- y / mu[i]))
        if p0 <= p:
            lo = y
        elif p0 >= p:
            hi = y
        y = 0.5 * (lo + hi)

    return y

class MixedExponential(Distribution):
    def __init__(self, mean, mixdist, mixcnt):
        super().__init__("MIXEDEXPONENTIAL", mean, mean)
        self.mixdist = mixdist
        self.mixcnt = mixcnt
        self.mixwgt = 1 / mixcnt
        self.mu = np.zeros(mixcnt)
        for i in range(self.mixcnt):
            perc = self.mixdist.lowerPercentile(self.mixwgt * (i + 0.5))
            self.mu[i] = self.mean * perc
        self.min = 0
        print(self.name, "(mean = ", self.mean, ", stdev = ", self.stdev, ")")

    def getPDF(self, x):
        if x > self.min:
            pdf = 0
            for i in range(self.mixcnt):
                pdf += self.mixwgt * exp(- x / self.mu[i]) / self.mu[i]
            return pdf
        else:
            return 0

    def getCDF(self, x):
        if x > self.min:
            cdf = 0
            for i in range(self.mixcnt):
                cdf += self.mixwgt * (1 - exp(- x / self.mu[i]))
            return cdf
        else:
            return 0

    def getSDF(self, x):
        if x > self.min:
            sdf = 0
            for i in range(self.mixcnt):
                sdf += self.mixwgt * exp(- x / self.mu[i])
            return sdf
        else:
            return 1

    def lowPerc(self, p, i):
        return -log(1-p) * self.mu[i]

    def lowerPercentile(self, p):
        return mixedExponentialLowerPercentile(self.min, self.mu, self.mixcnt, p)

    def getStochasticValue(self):
        i = floor(uniform(0, self.mixcnt))
        p = uniform(0, 1)
        return self.lowPerc(p, i)
