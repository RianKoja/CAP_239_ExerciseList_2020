
# Modified from: https://github.com/reinaldo-rosa-inpe/cap239/blob/master/Codigos/cullen_frey_andre_from_R.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from math import ceil
#np.seterr(under='ignore')
#np.seterr(over='ignore')


def graph(df, method='unbiased', discrete=False, boot=None):

    if len(np.shape(df)) > 1: 
        raise TypeError('Samples must be a list with N x 1 dimensions')
        
    if not isinstance(df, list):
        df = list(df)

    if len(df) < 4:
        raise ValueError('The number of samples needs to be greater than 4')

    if boot is not None:    
        if not isinstance(boot, int):
            raise ValueError('boot must be integer')

    if method =='unbiased':
        skewdata = skew(df, bias=False)
        kurtdata = kurtosis(df, bias=False)+3

    elif method =='sample':
        skewdata = skew(df, bias=True)
        kurtdata = kurtosis(df, bias=True)+3
 
    if boot is not None:
        if boot < 10:
            raise ValueError('boot must be greater than 10')

        n = len(df)

        nrow = n
        ncol = boot
        databoot = np.reshape(np.random.choice(df, size=n*boot, replace=True),(nrow,ncol)) 

        s2boot = (skew(pd.DataFrame(databoot)))**2
        kurtboot = kurtosis(pd.DataFrame(databoot))+3

        kurtmax = max(10, ceil(max(kurtboot)))
        xmax = max(4, ceil(max(s2boot)))

    else:
        kurtmax = max(10, ceil(kurtdata))
        xmax = max(4, ceil(skewdata**2))

    ymax = kurtmax-1

    res = [min(df), max(df), np.median(df), np.mean(df), np.std(df), skew(df), kurtosis(df)]

    # If discrete = False
    if not discrete:
        #Beta distribution
        p = np.exp(-100)
        lq = np.arange(-100, 100.1, 0.1)
        q = np.exp(lq)
        s2a = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
        ya = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
        p = np.exp(100)
        lq = np.arange(-100, 100.1, 0.1)
        q = np.exp(lq)
        s2b = (4*(q-p)**2*(p+q+1))/((p+q+2)**2*p*q)
        yb = kurtmax-(3*(p+q+1)*(p*q*(p+q-6)+2*(p+q)**2)/(p*q*(p+q+2)*(p+q+3)))
        s2 = [*s2a,*s2b]
        y = [*ya,*yb]
        
        # Gama distribution
        lshape_gama = np.arange(-100, 100, 0.1)
        shape_gama = np.exp(lshape_gama)
        s2_gama = 4/shape_gama
        y_gama = kurtmax-(3+6/shape_gama) 
        
        # Lognormal distribution
        lshape_lnorm = np.arange(-100, 100, 0.1)
        shape_lnorm = np.exp(lshape_lnorm)
        es2_lnorm = np.exp(shape_lnorm**2, dtype=np.float64)
        s2_lnorm = (es2_lnorm+2)**2*(es2_lnorm-1)
        y_lnorm = kurtmax-(es2_lnorm**4+2*es2_lnorm**3+3*es2_lnorm**2-3)

        plt.figure(figsize=(12, 9))

        # Observations
        obs = plt.scatter(skewdata**2, kurtmax-kurtdata, s=200, c='blue',
                      label='Observations',zorder=10)
        # beta
        beta = plt.fill(s2,y,color='lightgrey',alpha=0.6, label='beta', zorder=0)
        # gama
        gama = plt.plot(s2_gama,y_gama, '--', c='k', label='gama')
        # lognormal
        lnormal = plt.plot(s2_lnorm,y_lnorm, c='k', label='lognormal')
    
        if boot is not None:
            # bootstrap
            bootstrap = plt.scatter(s2boot, kurtmax-kurtboot, marker='$\circ$', c='orange', s=50,
                                    label='Bootstrap values', zorder=5)

        # markers
        normal = plt.scatter(0, kurtmax-3, marker=(8, 2, 0), s=400, c='k', label='normal', zorder=5)
        
        uniform = plt.scatter(0, kurtmax-9/5, marker='$\\bigtriangleup$', s=400, c='k', label='uniform', zorder=5)
        
        exp_dist = plt.scatter(2**2, kurtmax-9, marker='$\\bigotimes$', s=400, c='k', label='exponential', zorder=5)
        
        logistic = plt.scatter(0, kurtmax-4.2, marker='+', s=400, c='k', label='logistic', zorder=5)
        
        # Adjusting the axis
        yax = [str(kurtmax - i) for i in range(0, ymax+1)]
        plt.xlim(-0.08, xmax+0.4)
        plt.ylim(-1, ymax+0.08)
        plt.yticks(list(range(0, ymax+1)), labels=yax)
        
        # Adding the labels
        plt.xlabel('square of skewness', fontsize=13)
        plt.ylabel('kurtosis', fontsize=13)
        plt.title('Cullen and Frey graph', fontsize=15) 
        
        # Adding the legends
        legenda2 = plt.legend(handles=[obs, bootstrap], loc='upper center', labelspacing=1, frameon=False)
        plt.gca().add_artist(legenda2)

        plt.legend(handles=[normal, uniform, exp_dist, logistic, beta[0], lnormal[0], gama[0]],
               title='Theoretical distributions', loc='upper right', labelspacing=1.4, frameon=False)
    
        plt.show()

    # If discrete = True
    else:
        # negbin distribution
        p = np.exp(-10)
        lr = np.arange(-100, 100, 0.1)
        r = np.exp(lr)
        s2a = (2-p)**2/(r*(1-p))
        ya = kurtmax-(3+6/r+p**2/(r*(1-p)))
        p = 1-np.exp(-10)
        lr = np.arange(100, -100, -0.1)
        r = np.exp(lr)
        s2b = (2-p)**2/(r*(1-p))
        yb = kurtmax-(3+6/r+p**2/(r*(1-p)))
        s2_negbin = [*s2a, *s2b]
        y_negbin = [*ya, *yb]

        # poisson distribution
        llambda = np.arange(-100, 100, 0.1)
        lambda_ = np.exp(llambda)
        s2_poisson = 1/lambda_
        y_poisson = kurtmax-(3+1/lambda_)
    
        plt.figure(figsize=(12, 9))
    
        # observations
        obs = plt.scatter(skewdata**2, kurtmax-kurtdata, s=200, c='blue',
                          label='Observations', zorder=10)

        # negative binomial
        negbin = plt.fill(s2_negbin, y_negbin, color='lightgrey', alpha=0.6, label='negative binomial', zorder=0)

        # poisson
        poisson = plt.plot(s2_poisson,y_poisson, '--', c='k', label='poisson')

        if boot is not None:
            # bootstrap
            bootstrap = plt.scatter(s2boot, kurtmax-kurtboot, marker='$\circ$', c='orange', s=50,
                                    label='Bootstrap values', zorder=5)
  

        # markers
        normal = plt.scatter(0, kurtmax-3, marker=(8, 2, 0), s=400, c='k', label='normal', zorder=5)
    
        # adjusting the axis
        yax = [str(kurtmax - i) for i in range(0, ymax+1)]
        plt.xlim(-0.08, xmax+0.4)
        plt.ylim(-1, ymax+0.08)
        plt.yticks(list(range(0, ymax+1)), labels=yax)
    
        # adding the labels
        plt.xlabel('square of skewness', fontsize=13)
        plt.ylabel('kurtosis', fontsize=13)
        plt.title('Cullen and Frey graph', fontsize=15) 
    
        # adding the legends
        legenda1 = plt.legend(handles=[obs, bootstrap], loc='upper center', labelspacing=1, frameon=False)
        plt.gca().add_artist(legenda1)
    
        plt.legend(handles=[normal, negbin[0], poisson[0]], title='Theoretical distributions', loc='upper right',
                   labelspacing=1.4, frameon=False)
    
        plt.show()

    # print some statistical information
    print('=== summary statistics ===')
    print(f'min:{res[0]:.4f}\nmax:{res[1]:.4f}\nmean:{res[3]:.4f}\nstandard deviation:{res[4]:.4f}'f'\nskewness:{res[5]:.4f}\nkurtosis:{res[6]:.4f} +3 for the plot')


if __name__ == 'main':
    graph()