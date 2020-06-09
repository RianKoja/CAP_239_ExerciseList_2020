
# Standard imports:
import numpy as np
import matplotlib.pyplot as plt

# Local imports:
from generators import pmodel
from tools import createdocument, soc


def run(doc):
    doc.add_heading("Exercise 8.1", level=2)

    # Control the random seed so results are consistent between runs:
    np.random.seed(182745949)

    doc.add_heading("For endogenous series:", level=2)
    plt.figure()
    for ii in range(0, 50):
        data = pmodel.pmodel(8192, np.random.uniform(0.32, 0.42), 0.4)[1]
        prob_gamma, this_counts = soc.soc_main(data)
        plt.plot(prob_gamma, label=str(ii), linestyle='None', marker=".",)
    plt.legend()



if __name__ == '__main__':

    p_value = 0.2
    _, data_y = pmodel.pmodel(8192, p_value, slope=0.4)

    soc.soc_plot(data_y, "soc_main chart for p-model series with p = " + str(p_value))

    # Initialize report for debugging:
    test_report = createdocument.ReportDocument()
    # Run function:
    run(test_report)
    test_report.finish()
    print("Finished ", __file__)
    plt.show()


