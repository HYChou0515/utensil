import pickle
import os

_pkldir = os.path.dirname(__file__)


def svpkl(*obj):
    if len(obj) == 1:
        pickle.dump(obj[0], open(f"{_pkldir}/a.pkl", "wb"))
    else:
        pickle.dump(obj, open(f"{_pkldir}/a.pkl", "wb"))


def ldpkl():
    return pickle.load(open(f"{_pkldir}/a.pkl", "rb"))
