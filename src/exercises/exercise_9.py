
# Standard imports:
import matplotlib.pyplot as plt

# Local imports:
from generators import pmodel
from tools import soc


if __name__ == '__main__':

    p_value = 0.2
    _, data = pmodel.pmodel(8192, p_value, False)

    soc.soc_plot(data, "SOC chart for p-model series with p = " + str(p_value))

    plt.show()
