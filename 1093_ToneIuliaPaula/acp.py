from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


class acp():
    def __init__(self, t, variabile=None):
        assert isinstance(t, DataFrame)
        if variabile is None:
            variabile = list(t)
        self.__x = t[variabile].values

    def creare_model(self, std=True, nlib=0):
        if std:
            x_ = (self.__x - np.mean(self.__x, axis=0)) / np.std(self.__x, axis=0)
        else:
            x_ = self.__x - np.mean(self.__x, axis=0)
        n, m = np.shape(self.__x)
        mat = (1 / (n - nlib)) * x_.T @ x_
        valp, vecp = np.linalg.eig(mat)
        k = np.flipud(np.argsort(valp))
        self.__alpha = valp[k]
        self.__a = vecp[:, k]
        self.__c = x_ @ self.__a
        self.etichete_componente = ["comp" + str(i + 1) for i in range(m)]
        # Aplicare criterii de semnificatie pentru componente
        if std:
            self.nrcomp_k = len(np.where(self.__alpha > 1)[0])
        else:
            self.nrcomp_k = None
        pondere = np.cumsum(self.alpha / sum(self.alpha))
        self.nrcomp_p = np.where(pondere >= 0.8)[0][0] + 1
        eps = self.alpha[:(m - 1)] - self.alpha[1:]
        # print(eps)
        sigma = eps[:(m - 2)] - eps[1:]
        # print(sigma)
        negative = sigma < 0
        if any(negative):
            self.nrcomp_c = np.where(negative)[0][0] + 2
        else:
            self.nrcomp_c = None
        # Calcul corelatii dintre componente si variabilele observate
        self.r_x_c = np.corrcoef(self.__x,self.__c,rowvar=False)[:m,m:]


    def tabelare_varianta(self):
        procent_varianta = self.alpha * 100 / sum(self.alpha)
        tabel_varianta = DataFrame(data={
            "Varianta": self.alpha,
            "Varianta cumulata": np.cumsum(self.alpha),
            "Procent varianta": procent_varianta,
            "Procent cumulat": np.cumsum(procent_varianta)},
            index=self.etichete_componente
        )
        return tabel_varianta

    def plot_varianta(self):
        fig = plt.figure("Plot varianta", figsize=(12, 7))
        ax = fig.add_subplot(1, 1, 1)
        assert isinstance(ax, plt.Axes)
        ax.set_title("Plot varianta componente", fontsize=18, color='b')
        ax.set_xlabel("Componente")
        ax.set_ylabel("Varianta")
        m = len(self.alpha)
        x = np.arange(1, m + 1)
        ax.set_xticks(x)
        ax.plot(x, self.alpha)
        ax.scatter(x, self.alpha, c='r')
        if self.nrcomp_k is not None:
            ax.axhline(1, c='g', label="Kaiser")
        ax.axhline( self.alpha[self.nrcomp_p-1],c='m',label = "Acoperire minimala" )
        if self.nrcomp_c is not None:
            ax.axhline(self.alpha[self.nrcomp_c-1], c='c', label="Cattell")
        ax.legend()
        # plt.show()

    @property
    def x(self):
        return self.__x

    @property
    def alpha(self):
        return self.__alpha

    @property
    def a(self):
        return self.__a

    @property
    def c(self):
        return self.__c
