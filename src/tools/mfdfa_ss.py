# MFDFA-Analytics-by-SKDataScience
# multifractal DFA singularity spectra - module 04
# Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss.py
# This module is the entry point for testing the modified first-order uni- and multifractal DFA methods.
# It should be possible to execute this file without modifications if all modules
# (mfdfa_ss_m1, mfdfa_ss_m2, mfdfa_ss_m3 and mfdfa_ss_m4) are in the same repository.
# Modified by Leonardo S. Cassara for publishing in github repository.

import numpy as np
import matplotlib.pyplot as plt

import tools.mfdfa_ss_m1 as mfdfa1
import tools.mfdfa_ss_m2 as mfdfa2
import tools.mfdfa_ss_m3 as mfdfa3


def main(dx):
    # Computing
    # Modified first-order DFA
    [time_measure, mean_data_measure, scales] = mfdfa1.getHurstByUpscaling(dx)

    [b_scale, b_dm, bs_index, h_major, h_minor] = mfdfa3.get_scaling_exponents(time_measure, mean_data_measure)

    # Modified first-order MF-DFA
    [_, data_measure, _, stats, q] = mfdfa2.get_mss_by_upscaling(dx, is_normalised=1)

    # Unnecessary plots for the purpose of the test are commented out
    # Output
    # Modified first-order DFA
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.loglog(time_measure, mean_data_measure, 'ko-')
    # plt.xlabel(r'$\mu(t)$')
    # plt.ylabel(r'$\mu(\Delta x)$')
    # plt.grid('on', which='minor')
    # plt.title('Modified First-Order DFA of a Multifractal Noise')

    # plt.subplot(2, 1, 2)
    # plt.loglog(scales, mean_data_measure, 'ko-')
    # plt.loglog(b_scale, b_dm, 'ro')
    # plt.xlabel(r'$j$')
    # plt.ylabel(r'$\mu(\Delta x)$')
    # plt.grid('on', which='minor')

    # Modified first-order MF-DFA
    stats['delta_alpha'] = stats['LH_max'] - stats['LH_min']
    # print('alpha_min = %g, alpha_max = %g, dalpha = %g'
    #      % (stats['LH_min'], stats['LH_max'], stats['delta_alpha']))
    index_max = np.argmax(stats['f'])
    stats['alpha_zero'] = stats['LH'][index_max][0]

    # print('alpha_zero =', stats['alpha_zero'])
    stats['a_alpha'] = (stats['alpha_zero'] - stats['LH_min'])/(stats['LH_max'] - stats['alpha_zero'])
    # print('a_alpha =', stats['a_alpha'])
    stats['Psi'] = stats['delta_alpha'] / stats['LH_max']
    # print('Psi = %g' % stats['Psi'])
    # print('h_min = %g, h_max = %g, dh = %g\n' % (stats['h_min'], stats['h_max'], stats['delta_alpha']))

    plt.figure()
    nq = np.int(len(q))
    leg_txt = []
    for qi in range(1, nq + 1):
        llh = plt.loglog(scales, data_measure[qi - 1, :], 'o-')
        leg_txt.append(r'$\tau$' + ' = %2.1e (q = %2.1e)' % (stats['tau'][qi - 1], q[qi - 1]))
    plt.xlabel(r'$j$')
    plt.ylabel(r'$\mu(\Delta x, q)$')
    plt.grid('on', which='minor')
    plt.title('Modified First-Order MF-DFA of a Multifractal Noise')
    plt.legend(leg_txt, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.figure()
    plt.plot(q, stats['tau'], 'ko-')
    plt.xlabel(r'$q$')
    plt.ylabel(r'$\tau(q)$')
    plt.grid('on', which='major')
    plt.title('Statistics of Modified First-Order MF-DFA of a Multifractal Noise')

    # Plot singularity spectra:
    plt.figure()
    plt.plot(stats['LH'], stats['f'], 'ko-')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$f(\alpha)$')
    plt.grid('on', which='both')
    plt.title("Singularity Spectra\n" + r' $\alpha_{min}$ = %g,' % stats['LH_min'] +
              r'$\alpha_{max}$ = %g' % stats['LH_max'] + r' $\Delta\alpha$ = %g ' % stats['delta_alpha'] + "\n" +
              r' $\alpha_0$ = %g ' % stats['alpha_zero'] + r' $A_\alpha$ = %g ' % stats['a_alpha'] +
              r' $\Psi$ = %g ' % stats['Psi'])
    
    plt.tight_layout()

    return stats


if __name__ == "__main__":
    # Create sample test data
    mean, cov = [1, -1], [(1, .5), (.5, 1)]
    sample_data, _ = np.random.multivariate_normal(mean, cov, size=256).T
    sample_data = sample_data.tolist()
    main(sample_data)
    plt.show()
