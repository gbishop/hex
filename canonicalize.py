import numpy as np
import numpy.typing as npt


class Canonicalizer:
    """Reflect and transpose the hex board to a canonical state"""

    def __init__(self, size, byvalue=True, identity=False):
        self.size = size
        self.byvalue = byvalue
        self.identity = identity

        # identity transform
        I: npt.NDArray[np.int64] = np.arange(size * size)

        def square(A: npt.NDArray[np.int64]):
            return A.reshape((size, size))

        def flat(A: npt.NDArray[np.int64]):
            return A.flatten()

        self.x = x = flat(square(I)[:, ::-1])  # reflect x
        self.y = y = flat(square(I)[::-1, :])  # reflect y
        self.xy = xy = x[y]  # relect both
        self.t = t = flat(square(I).transpose())  # transpose
        self.tx = tx = I[t][x]  # transpose then flip x
        self.ty = ty = I[t][y]  # transpose then flip y
        self.txy = txy = I[t][x][y]  # transpose then flip both

        self.transforms = np.array(
            [
                I,
                x,
                y,
                xy,
                t,
                tx,
                ty,
                txy,
            ]
        )
        # transposing requires reversing player identities
        self.signs = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],
            dtype=np.float32,
        )

    def canocalize(self, obs: npt.NDArray[np.float32], player: int):
        # convert the observation to the players view
        if player == -1:
            obs = -obs[self.t]
        # get all transforms of this observation
        if not self.byvalue:
            obs = obs != 0
        canons = obs[self.transforms]
        # sort it to choose the canonical transform
        if self.identity:
            ci = 0
        else:
            ci = np.lexsort(canons.transpose())[0]
        # get the map
        inverse = self.transforms[ci]
        # get the sign
        sign = self.signs[ci]
        # get the transformed observation
        canon = obs[inverse] * sign
        if player == -1:
            inverse = inverse[self.t]
            sign = -sign

        return canon, inverse, sign


if __name__ == "__main__":
    import random

    SIZE = 5
    CELLS = SIZE * SIZE
    COMBINATIONS = 2**CELLS

    N = Canonicalizer(SIZE)

    In = set()
    Out = set()

    def count(board: np.ndarray, S: set[tuple[int, ...]]):
        key = tuple(int(b) for b in board)
        S.add(key)

    p = 1
    samples = min(COMBINATIONS, 1_000_000)
    print(f"{samples=}")
    for i in random.sample(range(COMBINATIONS), samples):
        obs = np.zeros(CELLS, dtype=np.float32)
        for ip in [1]:
            p = ip
            for j in range(CELLS):
                if i & (1 << j):
                    obs[j] = p
                    p = -p

            count(obs, In)
            canon, inverse, sign = N.canocalize(obs, ip)
            count(canon, Out)

    scale = COMBINATIONS / samples
    rate = 100 * (len(In) - len(Out)) / len(In)
    print(f"scaled rate={scale * rate:.1f}")
    print(f"In={len(In)} Out={len(Out)}")
