
# Standard imports:
import os
import numpy as np
import matplotlib.pyplot as plt
from tools.waipy.lib import waipy
from pandas.compat import BytesIO
from docx.shared import Inches

# Local imports:
from generators import grng
from generators import colorednoise
from generators import pmodel
from generators import logis
from generators import henon


def run(doc_report):
    # Time series with GNRG:
    data_grng = grng.time_series(2 ** 13, 1)
    figname_gnrg = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mount', 'GNRG_waipy')
    figname_gnrg = os.path.relpath(figname_gnrg, os.getcwd())
    result_grng = waipy.cwt(data_grng, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet', name='test name')
    waipy.wavelet_plot(figname_gnrg, np.linspace(0, len(data_grng), len(data_grng)), data_grng, 0.03125, result_grng)
    doc_report.document.add_picture(figname_gnrg + '.png', width=Inches(6))

    plt.close('all')


# Sample execution:
if __name__ == '__main__':

    z = np.linspace(0, 2048, 2048)
    x = np.sin(50 * np.pi * z) + 3.5 * np.random.randn(len(z))
    y = np.cos(50 * np.pi * z + np.pi / 4) + 4 * np.random.randn(len(z))

    data_norm = waipy.normalize(x)
    result = waipy.cwt(data_norm, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet', name='test name')
    waipy.wavelet_plot('../mount/Sine with noise', z, data_norm, 0.03125, result)

    data_norm1 = waipy.normalize(y)
    result1 = waipy.cwt(data_norm1, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet', name='y')
    waipy.wavelet_plot('../mount/Cosine with noise', z, data_norm1, 0.03125, result1)

    cross_power, coherence, phase_angle = waipy.cross_wavelet(result['wave'], result1['wave'])
    figname = '../mount/example3.png'
    waipy.plot_cross('../mount/Crosspower sine and cosine', cross_power, phase_angle, z, result, result1, figname)

    plt.show()
