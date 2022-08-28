import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np


def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    nume_variabile = list(t.columns)
    for v in nume_variabile:
        if any(t[v].isna()):
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                modulul = t[v].mode()[0]
                t[v].fillna(modulul, inplace=True)


def tabelare_matrice(x, nume_linii=None, nume_coloane=None, out=None):
    t = pd.DataFrame(x, nume_linii, nume_coloane)
    if out is not None:
        t.to_csv(out)
    return t


def tabelare_varianta(alpha, etichete):
    procent_varianta = alpha * 100 / sum(alpha)
    tabel_varianta = pd.DataFrame(data={
        "Varianta": alpha,
        "Varianta cumulata": np.cumsum(alpha),
        "Procent varianta": procent_varianta,
        "Procent cumulat": np.cumsum(procent_varianta)},
        index=etichete
    )
    return tabel_varianta
