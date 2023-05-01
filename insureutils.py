###########################################################################
#                                                                         #
#    Fast jit version of the insureutils-library                          #
#                                                                         #
###########################################################################

from numba import jit


#############################################
#        RETAINED AND INSURED RISK          #
#############################################
@jit(nopython=True)
def get_retained_risk(x, a, b) -> float:
    if x < a:
        return x
    elif x < b:
        return a
    else:
        return x - (b - a)


@jit(nopython=True)
def get_insured_risk(x, a, b) -> float:
    if x < a:
        return 0
    elif x < b:
        return x - a
    else:
        return b - a


@jit(nopython=True)
def get_expected_risk(xx) -> float:
    s: float = 0
    for i in range(len(xx)):
        s += xx[i]
    return s / len(xx)


@jit(nopython=True)
def get_expected_retained_risk(xx, a, b) -> float:
    s: float = 0
    for i in range(len(xx)):
        s += get_retained_risk(xx[i], a, b)
    return s / len(xx)


@jit(nopython=True)
def get_expected_insured_risk(xx, a, b) -> float:
    s = 0
    for i in range(len(xx)):
        s += get_insured_risk(xx[i], a, b)
    return s / len(xx)


# The set C is the set of points where retained risk is greater than (aa1 + aa2)
# NOTE: We assume that len(xx1) = len(xx2)
@jit(nopython=True)
def getCCount(xx1, aa1, bb1, xx2, aa2, bb2) -> float:
    count = 0
    for i in range(len(xx1)):
        rr: float = get_retained_risk(xx1[i], aa1, bb1) + get_retained_risk(xx2[i], aa2, bb2)
        if rr > aa1 + aa2:
            count += 1
    return count

@jit(nopython=True)
def getCFraction(xx1, aa1, bb1, xx2, aa2, bb2) -> float:
    count = 0
    for i in range(len(xx1)):
        rr: float = get_retained_risk(xx1[i], aa1, bb1) + get_retained_risk(xx2[i], aa2, bb2)
        if rr > aa1 + aa2:
            count += 1
    return count / len(xx1)


#############################################
#             UTILITY METHODS               #
#############################################
@jit(nopython=True)
def probabilityToIndex(p, n) -> int:
    if p <= 1:
        return int(n * p + 0.5) - 1
    else:
        return n - 1


@jit(nopython=True)
def survivalToIndex(p, n) -> int:
    if p >= 0:
        return int(n * (1.0 - p) + 0.5) - 1
    else:
        return n - 1


@jit(nopython=True)
def indexToProbability(i, n) -> float:
    return (i + 1.0) / n


@jit(nopython=True)
def indexToSurvival(i, n) -> float:
    return 1.0 - (i + 1.0) / n

##############################################################
#    Find the index of the value in xx closest to val        #
#    assuming that the values in xx is sorted in increasing  #
#    order                                                   #
##############################################################
@jit(nopython=True)
def valueToIndexIncr(xx, val) -> int:
    lower = 0
    upper = len(xx) - 1
    if val <= xx[lower]:
        return lower
    elif val >= xx[upper]:
        return upper
    while lower < upper - 1:
        mid = (lower + upper) // 2
        if val < xx[mid]:
            upper = mid
        else:
            lower = mid
    return upper


##############################################################
#    Find the index of the value in xx closest to val        #
#    assuming that the values in xx is sorted in decreasing  #
#    order                                                   #
##############################################################
@jit(nopython=True)
def valueToIndexDecr(xx, val) -> int:
    lower = 0
    upper = len(xx) - 1
    if val >= xx[lower]:
        return lower
    elif val <= xx[upper]:
        return upper
    while lower < upper - 1:
        mid = (lower + upper) // 2
        if val > xx[mid]:
            upper = mid
        else:
            lower = mid
    return upper
