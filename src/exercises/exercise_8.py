
# Standard imports:
import os
import numpy as np
import matplotlib.pyplot as plt
from tools.waipy.lib import waipy
from docx.shared import Inches

# Local imports:
from generators import grng
from generators import colorednoise
from generators import pmodel
from generators import logis
from generators import henon


def run(doc_report):
    doc_report.document.add_heading('Exercise 8', level=2)
    doc_report.document.add_paragraph("""
      Here we compare the Continuous Wavelet Spectrum for time series generated with each signal generator used so far.
      Both Morley and DOG wavelet charts are used.""")

    # Henon map not used as it causes issues with the waipy module.
    names = ('GNRG', 'Color', 'P_model_025_exogen_beta04', 'logistic_rho3.88_tau1.1')  # 'henon_a1.38_b0.22',
    generators = (lambda: grng.time_series(8192, 1),
                  lambda: colorednoise.powerlaw_psd_gaussian(1, 8192),
                  lambda: pmodel.pmodel(8192, 0.25, 0.4)[1],
                  lambda: logis.logistic_series(3.88, 1.1, 8192)[1])
    # lambda: np.array(henon.henon_series(1.38, 0.22, 8192 - 1)[1], dtype=np.float32),
    full_names = ('Non Gaussian Random Generator', 'Colored Noise Generator', 'P-Model', 'Logistic Map')  # 'Henon Map'
    for (name, func, full_name) in zip(names, generators, full_names):
        data = func()
        doc_report.document.add_heading(full_name, level=3)

        fig_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mount', name + '_waipy')
        fig_name = os.path.relpath(fig_name, os.getcwd())
        result = waipy.cwt(data, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet', name='test name')
        waipy.wavelet_plot(fig_name, np.linspace(0, len(data), len(data)), data, 0.03125, result)
        doc_report.document.add_heading("Morley:", level=4)
        doc_report.document.add_picture(fig_name + '.png', width=Inches(6))
        result = waipy.cwt(data, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='DOG', name='test name')
        waipy.wavelet_plot(fig_name, np.linspace(0, len(data), len(data)), data, 0.03125, result)
        doc_report.document.add_heading("DOG:", level=4)
        doc_report.document.add_picture(fig_name + '.png', width=Inches(6))

        plt.close('all')


# Sample execution:
if __name__ == '__main__':

    z = np.linspace(0, 2048, 2048)
    x = np.sin(50 * np.pi * z) + 3.5 * np.random.randn(len(z))
    y = np.cos(50 * np.pi * z + np.pi / 4) + 4 * np.random.randn(len(z))

    data_norm = waipy.normalize(x)
    result0 = waipy.cwt(data_norm, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet', name='test name')
    waipy.wavelet_plot('../mount/Sine with noise', z, data_norm, 0.03125, result0)

    data_norm1 = waipy.normalize(y)
    result1 = waipy.cwt(data_norm1, 1, 1, 0.25, 4, 4 / 0.25, 0.72, 6, mother='Morlet', name='y')
    waipy.wavelet_plot('../mount/Cosine with noise', z, data_norm1, 0.03125, result1)

    cross_power, coherence, phase_angle = waipy.cross_wavelet(result0['wave'], result1['wave'])
    figname = '../mount/example3.png'
    waipy.plot_cross('../mount/Crosspower sine and cosine', cross_power, phase_angle, z, result0, result1, figname)

    plt.show()
