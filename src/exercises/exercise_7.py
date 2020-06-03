
# Standard imports:
import os
import numpy as np
import matplotlib.pyplot as plt
import tools.mfdfa_ss as mfdfa

# Local imports:
from generators import grng
from generators import colorednoise
from generators import pmodel
from generators import logis
from generators import henon


def run(doc_report):
    # Time series with GNRG:
    data_grng = grng.time_series(2 ** 13, 1)
    mfdfa.main(data_grng)


# Sample execution:
if __name__ == '__main__':
    run(None)
    plt.show()
