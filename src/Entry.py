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
from generators import colorednoise
from generators import pmodel

print("Starting ", __file__)
# Use fixed seed, so results don't change between runs of the same algorithm:
np.random.seed(82745949)
# Start the report:
doc_report = createdocument.ReportDocument()

# Run the script for exercises 1 to 3:
doc_report.document.add_heading("Exercise 1", level=2)
doc_report.document.add_paragraph("\n\n")
algorithm1 = GRNG()
auxfunctions.exercises_1_3(algorithm1, doc_report)
doc_report.document.add_paragraph("\n\n")

doc_report.document.add_heading("Exercise 2", level=2)
doc_report.document.add_paragraph("\n\n")
algorithm2 = colorednoise.coloredgenerator()
auxfunctions.exercises_1_3(algorithm2, doc_report)
doc_report.document.add_paragraph("\n\n")

doc_report.document.add_heading("Exercise 3", level=2)
doc_report.document.add_paragraph("\n\n")
algorithm3 = pmodel.PModelGenerator()
auxfunctions.exercises_1_3(algorithm3, doc_report)
doc_report.document.add_paragraph("\n\n")

# Finish the report:
doc_report.finish()

print("Finished ", __file__)
