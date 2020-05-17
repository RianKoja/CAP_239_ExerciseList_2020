########################################################################################################################
# Entry point function to run all analysis from CAP-239 exercise list 2020. (in progress)
#
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################


# Standard imports:
import numpy as np

# Local imports:
from tools import createdocument
from tools import auxfunctions
from generators import GRNG

print("Starting ", __file__)
# Use fixed seed, so results don't change between runs of the same algorithm:
np.random.seed(82745949)
# Start the report:
doc_report = createdocument.ReportDocument()

# Run the script for exercises 1 to 3:

algorithm = GRNG()

auxfunctions.exercises_1_3(algorithm, doc_report)

# Finish the report:
doc_report.finish()

print("Finished ", __file__)
