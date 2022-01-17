import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
n_noise = 1
n_Drift = 6 # simple model
random.seed(1234)

T = 1000
dt = 0.01

def D1(x,t):
 #   return(-1.*x + 1. *np.sin(2.*np.pi *t) )
    return(- 6*x**2 + 1.*t)

def D2(x,t):
    return(0.1 *random.gauss(0,1))

x0 = 1.0
X = np.zeros(T)
X[0] = x0
t = np.arange(T)*dt


for i in range(1,T):
    X[i] = X[i-1] + D1(X[i-1], t[i-1]) *dt + D2(X[i-1], t[i-1])

plt.plot(t, X)
plt.show()



Data = np.empty((T,2))
Data[:,0] = t
Data[:,1] = X

####

## Define all the functions

#@jit
def poly(x,sigma):
    #x_vec=np.array([1,x[0],x[1],x[0]*x[0],x[1]*x[0],
    #                x[1]*x[1],x[0]*x[0]*x[0],x[1]*x[0]*x[0],x[1]*x[1]*x[0],x[1]*x[1]*x[1],
    #               x[0]**4, x[0]**3 * x[1], x[0]**2 * x[1]**2, x[0]*x[1]**3, x[1]**4],
    #                dtype=object)

    x_vec = np.array([1,x[0], x[1], x[0]*x[0], x[1]*x[0],
                        x[1]*x[1]], ) # simple form
    return np.dot(sigma,x_vec)



#@jit
def D1(sigma,x):
    # Make use of the fact that here, x_1 = t and hence dx_1/dt = 1
    sigma_d1 = sigma[n_noise:] # without noise parameters
    return poly(x.T, sigma_d1)


#@jit
def D2(alpha,x):
    return alpha[0]*np.ones(x.shape[0])


#  Log Likelihood and negative logL

def log_likelihood(alpha,x,dt):
    # alpha is the current set of parameters
    # x is the entire data set N x 2
    # dt is the time difference

    log_like = 0 # initial value of sum

    if alpha[0]>0  : # noise must be positive

        dx = x[1:,:]-x[:-1,:]

        # only the x[:,1] component is the actual time series
        d1 = dx[:,1] - D1(alpha,x)[:-1]*dt
        d2 = D2(alpha,x)[:-1]
        d2_inv = d2**(-1.)
        log_like = ( -0.5*np.log(d2) - 0.5 * d2_inv * d1**2.).sum()

        return log_like
    else:
        return -np.inf


def neg_log_likelihood(alpha,x,dt): #L Threshold Lambdac
    return -1*log_likelihood(alpha,x,dt)


# NEVER cut off all noise terms!
def cutIndex(Parameters, Threshold):
    Index = (np.abs(Parameters) < Threshold)
    Index[0:n_noise] = False
    return(Index)

def second_neg_log_likelihood(Coeff, Index,x,dt):
    # Index: Index of those coefficients which are set to 0: Boolean
    Index = cutIndex(Coeff,L_thresh)
    Coeff[Index] = 0
    return -1*log_likelihood(Coeff,x,dt)


# BIC as goodness criterion for a threshold value

def BIC(alpha,x,dt,L): # mit Lambda Threshold

    logi = np.abs(alpha)>L # which are larger than Lambda?
    logi[0:n_noise] = True  # noise is always included
    return np.log(x[:,0].size)*np.sum(  logi ) - 2*log_likelihood(alpha, x,dt )


# Calculate BIC in the Loop with thresholding

def Loop(x, dt, L, a_Ini):
    # estimates alpha parameters based on starting values a_Ini for a given threshold L
    a_hat = optimize.minimize(neg_log_likelihood, a_Ini,args=(x,dt)) # max likelihood
    Estimation = a_hat["x"]
    Estimation[n_noise:] =  Estimation[n_noise:] * ( abs(Estimation[n_noise:] ) > L )

    for i in np.arange(0,n_Cut):
        Cut = (np.abs(Estimation)<L) # Boolean of the values that are cut off
        # second optimization with maxLikelEstimator as start:
        a_hat = optimize.minimize(second_neg_log_likelihood,Estimation,args = (Cut,x,dt))
        Estimation = a_hat["x"]
        Estimation[n_noise:] =  Estimation[n_noise:] * ( abs(Estimation[n_noise:] ) > L )
    return(Estimation)

######
# For consistency with past codes: call Data x
x = Data

# Set up variables for the hyperparameter search on threshold

n_Cut = 5 # Number of reiterating
hp1 = np.arange(0.00,1.5, 0.1) # list of possible thresholds
n_Iteration = len(hp1) # Number of Hyperparameter search iterations
score = np.empty(n_Iteration) # score for Hyperparameters



TestAl = np.ones(n_noise+n_Drift)   # sample parameters to start the search


#### Some educated initial guesses


def Diff_D1(Coefficients,x,dt):
    dx = x[1:]-x[:-1]
    D1 = poly(x.T,Coefficients)[:-1]
    return(sum((dx[:,1]-D1)**2.))
TestLS = optimize.minimize(Diff_D1,np.ones(n_Drift),args=(x,dt))

TestAl[0] = 0.01
TestAl[1:] = TestLS["x"]


TestAl = optimize.minimize(neg_log_likelihood,TestAl,args=(x,dt))

print(TestAl)









