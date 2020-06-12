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
from exercises import exercise_1_to_4_1, exercise_4_2, exercise_5_1, exercise_6_1, exercise_6_2, exercise_6_3
from exercises import exercise_7, exercise_8, exercise_9
from generators import grng, colorednoise, pmodel

print("Starting ", __file__)
# Use fixed seed, so results don't change between runs of the same algorithm:
np.random.seed(82745949)
# Start the report:
doc_report = createdocument.ReportDocument()

# Run the script for exercises 1 to 4.1:
doc_report.add_heading("Exercise 1", level=2)
doc_report.add_heading("GRNG", level=3)
doc_report.add_paragraph("\n\n")
algorithm1 = grng.Grng()
exercise_1_to_4_1.exercises_1_3(algorithm1, doc_report)
doc_report.add_paragraph("\n\n")
plt.close('all')

doc_report.add_heading("Exercise 2", level=2)
doc_report.add_heading("Colored Noise", level=3)
doc_report.add_paragraph("\n\n")
algorithm2 = colorednoise.ColoredGenerator()
exercise_1_to_4_1.exercises_1_3(algorithm2, doc_report)
doc_report.add_paragraph("\n\n")
plt.close('all')

doc_report.add_heading("Exercise 3", level=2)
doc_report.add_heading("P-Model", level=3)
doc_report.add_paragraph("\n\n")
algorithm3 = pmodel.PModelGenerator()
exercise_1_to_4_1.exercises_1_3(algorithm3, doc_report)
doc_report.add_paragraph("\n\n")
plt.close('all')

# Exercise 4.1
doc_report.add_heading("Exercise 4", level=2)
doc_report.add_heading("Exercise 4.1", level=3)
doc_report.add_paragraph("This exercise produces results for each of the generators used in the prior exercises, thus " +
                         "such results were shown in the previous sections.")

# Run for exercise 4.2:
exercise_4_2.plot_estimates_noises(doc_report)

# Run for exercise 5:
exercise_5_1.report_ex5(doc_report)
plt.close('all')

# Run for exercise 6.1:
k_means_list = exercise_6_1.run(doc_report)
plt.close('all')

# Run for exercise 6.2:
exercise_6_2.run(doc_report, k_means_list)
plt.close('all')

# Run for exercise 6.3:
exercise_6_3.run(doc_report)
plt.close('all')

# Run for exercise 7:
exercise_7.run(doc_report)
plt.close('all')

# Run for exercise 8:
exercise_8.run(doc_report)
plt.close('all')

# Run for exercise 9:
exercise_9.run(doc_report)
plt.close('all')

# Finish the report:
doc_report.finish()

print("Finished ", __file__)
