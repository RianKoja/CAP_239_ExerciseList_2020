########################################################################################################################
# Entry point function to run all analysis from CAP-239 exercise list 2020. (in progress)
#
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################


# Standard imports:
import numpy as np
import matplotlib.pyplot as plt

# Local imports:
from tools import createdocument
from exercises import exercise_1_to_4_1, exercise_4_2, exercise_5, exercise_8
from generators import grng, colorednoise, pmodel

print("Starting ", __file__)
# Use fixed seed, so results don't change between runs of the same algorithm:
np.random.seed(82745949)
# Start the report:
doc_report = createdocument.ReportDocument()

# Run the script for exercises 1 to 4.1:
doc_report.document.add_heading("Exercise 1", level=2)
doc_report.document.add_heading("GRNG", level=3)
doc_report.document.add_paragraph("\n\n")
algorithm1 = grng.Grng()
exercise_1_to_4_1.exercises_1_3(algorithm1, doc_report)
doc_report.document.add_paragraph("\n\n")
plt.close('all')

doc_report.document.add_heading("Colored Noise", level=3)
doc_report.document.add_paragraph("\n\n")
algorithm2 = colorednoise.coloredgenerator()
exercise_1_to_4_1.exercises_1_3(algorithm2, doc_report)
doc_report.document.add_paragraph("\n\n")
plt.close('all')

doc_report.document.add_heading("P-Model", level=3)
doc_report.document.add_paragraph("\n\n")
algorithm3 = pmodel.PModelGenerator()
exercise_1_to_4_1.exercises_1_3(algorithm3, doc_report)
doc_report.document.add_paragraph("\n\n")
plt.close('all')

# Run for exercise 4.2:
exercise_4_2.plot_estimates_noises(doc_report)

# Run for exercise 5:
exercise_5.report_ex5(doc_report)
plt.close('all')

# Run for exercise 8:
exercise_8.run(doc_report)
plt.close('all')


# Finish the report:
doc_report.finish()

print("Finished ", __file__)
