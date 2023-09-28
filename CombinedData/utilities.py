import numpy as np

#this is a fitfunc intended for use with the scipy.ODR
#class. It uses t0 as a fixed, known parameter, the start
#time of the falling exponential. 
def fitfunc_exp(beta, x, t0):
    return beta[0]*np.exp(-(x - t0)/beta[1])

#fit to exponential with a fixed start time, which 
#may vary from event to event (depending on the digitizer settings)
def exp_wrapper_fixedt0(fixed_t0):
    return lambda beta, x: fitfunc_exp(beta, x, fixed_t0)

def exp_wrapper_fixedtau(fixed_t0, fixed_tau):
    return lambda beta, x: fitfunc_exp([beta[0], fixed_tau], x, fixed_t0)