"""Figure out hex transforms"""

import numpy as np

S = 5
S2 = S * S

i = np.arange(S2)
I = i.reshape((S, S))


def sq(f):
    return f.reshape((S, S))


def fl(s):
    return s.flatten()


def X(b):
    return b[:, ::-1]


def Y(b):
    return b[::-1, :]


def T(b):
    return b.transpose()


from typing import Callable


def test(*ts: Callable):
    print(", ".join(f.__name__ for f in ts))
    print(I)
    F = I
    for t in ts:
        F = t(F)
        print(F)
    f = fl(F)
    print("forward", f)
    assert np.all(F == sq(fl(I)[f]))

    i = np.random.choice(fl(I))
    v = f[i]

    print("reverse")
    R = I
    for t in ts[::-1]:
        R = t(R)
        print(R)
    r = fl(R)
    print("reverse", r)
    assert np.all(I == sq(f[r]))

    print(i, v, f[i], r[i])
    assert v == f[i]

    return f


TS = [
    [X],
    [Y],
    [X, Y],
    [T],
    [T, X],
    [T, Y],
    [T, X, Y],
]

ts = [test(*T) for T in TS]

for t in ts:
    print(t.tolist())
