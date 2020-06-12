# CAP_239_ExerciseList_2020
This is a repository to store the code and documents related to the exercise list proposed by prof. Reinaldo Rosa during INPE's 2020 class of Computational Mathematics

# Execution and Results
To execute all functions, go to the `src` directory and run `Entry.py` with a Python 3.7 interpreter. This will generate a `.docx` file named `List_RianKoja_vNNN.docx` where `NNN` is increased numerically to avoid overwriting an existing file, a 0 version is made available and should be used as reference. This document contains most of the content to respond the exercises stated on `Lista_CAP239_2020_Atualizada_e_com_Tabela.pdf`. Other parts which were written manually rather than with Python functions are provided on `Report_Rian_manual.docx`.

# Docker
After cloning the target repository, to get the files of the Docker submodule, run the commands:
`git submodule init`
`git submodule update`

The docker image should allow running the python files in a docker container, but is currently untested. It removes the need for the appropriate Python version and environment, but requires Docker and some display surrogate, such as `xhost` in Linux.