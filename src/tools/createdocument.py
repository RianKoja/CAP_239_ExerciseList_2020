########################################################################################################################
# Defines a class that handles writing results to a word document.
#
#
# Adapted by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################
import os
from docx import Document
from docx.shared import Inches

class ReportDocument:
    def __init__(self):
        self.document = Document()
        self.document.add_heading('Exercise List', level=0)
        self.document.add_heading('Rian Koja', level=1)
        self.document.add_heading('CAP 239 Computational Mathematics', level=1)

        self.document.add_heading("Comment #" + str("22"), level=1)
        self.document.add_paragraph("loooool")
        self.document.add_heading('Response:', level=2)
        self.document.add_paragraph("\n\n")
        self.document.add_heading('Status:', level=2)
        self.document.add_paragraph("To be implemented")

        self.file_path = os.path.dirname(os.path.abspath(__file__))
        file_version = 0

        while os.path.isfile(os.path.join(self.file_path, "mount", "List_RianKoja_v" + str(file_version) + ".docx")):
            file_version += 1

        self.file_name = "List_RianKoja_v" + str(file_version) + ".docx"

    def add_fig(self, memfile, wid=6):
        # Calling function must do:
        # from pandas.compat import StringIO
        # memfile = StringIO()
        # plt.savefig(memfile)
        self.document.add_picture(memfile, width=Inches(wid))
        memfile.close()

    def finish(self):
        self.document.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mount', self.file_name))

        print("finished word document file.")


if __name__ == "__main__":
    # Just test the functions:
    testDoc = ReportDocument()
    testDoc.finish()
