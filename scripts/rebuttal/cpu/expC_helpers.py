import numpy as np
from numba import njit
from policysim import doc_starts
@njit(cache=True)
def _ver(match, starts):
    ver = np.zeros(len(match), np.bool_)
    for d in range(len(starts) - 1):
        c, e = starts[d], starts[d + 1]
        while c < e:
            j = 0; mis = -1
            while c + j < e and j < 4:
                ver[c + j] = True
                if not match[c + j]:
                    mis = c + j; break
                j += 1
            c = mis + 1 if mis >= 0 else c + j + 1
    return ver
def verified_k4(df):
    return _ver((df.d_arg.values == df.y.values), doc_starts(df.doc.values))
