
# Standard imports:
import numpy as np
import waipy

# Local imports:
from generators import GRNG
from generators import colorednoise
from generators import pmodel
from generators import logis
from generators import henon


if __name__ == '__main__':
    z = np.linspace(0, 2048, 2048)
    x = np.sin(50 * np.pi * z) + 3.5 * np.random.randn(len(z))
    y = np.cos(50 * np.pi * z + np.pi / 4) + 4 * np.random.randn(len(z))

    data_norm = waipy.normalize(x)
    result = waipy.cwt(data_norm, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet')
    waipy.wavelet_plot('Sine with noise', z, data_norm, 0.03125, result)

    data_norm1 = waipy.normalize(y)
    result1 = waipy.cwt(data_norm1, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet', name='y')
    waipy.wavelet_plot('Cosine with noise', z, data_norm1, 0.03125, result1)

    cross_power, coherence, phase_angle = waipy.cross_wavelet(result['wave'], result1['wave'])
    figname = 'example3.png'
    waipy.plot_cross('Crosspower sine and cosine', cross_power, phase_angle, z, result, result1, figname)