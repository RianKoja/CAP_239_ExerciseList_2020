########################################################################################################################
# Stochastic series generator based on universality class.
#
#
# Adapted by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# Gerador de Série Temporal Estocástica - V.1.2 por R.R.Rosa
# Trata-se de um gerador randômico não-gaussiano sem classe de universalidade via PDF.
# Input: n=número de pontos da série
# res: resolução
import numpy as np
import pandas as pd
from numpy import sqrt
import matplotlib.pyplot as plt

from tools import stat

class GRNG:
    def __init__(self):
        self.name = 'grng'
        self.N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    def generator(self, points, resolution):
        df = pd.DataFrame(np.random.randn(points) * sqrt(resolution / points)).cumsum()
        #  df = pd.Series(np.random.randn(no) * sqrt(re) * sqrt(1 / 128.)).cumsum()
        ret = df[0].tolist()
        new_df = pd.DataFrame(stat.series2datasetline(ret), index=[1])
        return new_df

    def makedataframe(self, df):
        for N in self.N:
            for trial in range(0, 10):  # Gerando 10 sinais.
                data = self.generator(N, N / 12)
                data['Type'] = N
                df = df.append(data, ignore_index=True)
        return df


if __name__ == "__main__":
    # Define sample parameters:
    n = 128
    res = n/12

    # Generate data
    generator = GRNG()
    a = GRNG().generator(n, res)

    # Create figures:
    plt.figure()
    plt.plot(a)
    plt.ylabel("Valores de Amplitude, A(t)")
    plt.xlabel("Tempo (t)")
    plt.draw()

    b = int(n/10)
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.style.use("ggplot")
    plt.hist(a, bins=b, ec="k", alpha=0.6, color='royalblue')
    plt.xlabel("Valores de Amplitude")
    plt.ylabel("Contagem")
    plt.draw()

    # Printing the Time Series
    # print('\n'.join(map(str, a)))

    plt.show()
    print("Finished ", __file__)