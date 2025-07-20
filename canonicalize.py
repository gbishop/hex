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
        self.I = I

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

    def canocalize(
        self, obs: npt.NDArray[np.float32], player: int, ci: int | None = None
    ):
        canon = obs
        tobs = obs
        map = self.I
        sign = 1

        # convert the observation to the players view
        if player == -1:
            tobs = -obs
            sign = -1

        # get all transforms of this observation
        if not self.byvalue:
            # choose canon on occupied status
            tobs = tobs != 0
        canons = tobs[self.transforms]
        if self.identity:
            ci = 0
        elif ci is None:
            # sort it to choose the canonical transform
            ci = np.lexsort(canons.transpose())[0]

        assert ci is not None

        if player == -1:
            # swap sides for player 2 by jumping to the transposed transforms
            ci = (ci + 4) % len(canons)
        # get the map
        map = self.transforms[ci]
        # get the sign
        sign = sign * self.signs[ci]
        # get the transformed observation
        canon = obs[map] * sign

        return canon, map, sign


def visualize():
    test = np.array(
        [
            [0, 0, 0, 0, -1],
            [0, 0, 0, -1, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    SIZE = len(test)
    test = test.flatten()

    def draw(*boards):
        chars = {1: "X", -1: "O", 0: "_"}
        N = len(boards)
        S = SIZE
        w = 3 * S
        W = N * w
        lines = []
        for r in range(S):
            line = [" " for _ in range(W)]
            for i, board in enumerate(boards):
                for c in range(S):
                    line[i * w + c * 2 + r] = chars[board[r * S + c]]
            lines.append("".join(line))
        return "\n".join(lines)

    C = Canonicalizer(SIZE, byvalue=True, identity=True)
    for ci in range(8):
        for p in [1, -1]:
            b = test.copy()
            c, inverse, _ = C.canocalize(b, p, ci=ci)
            i = np.where(c == 0)[0][0]
            d = b.copy()
            print(f"{ci=} {p=} {i=} {inverse[i]=}")
            ok = d[inverse[i]] == 0
            d[inverse[i]] = p
            print(draw(b, c, d))
            assert ok


if __name__ == "__main__":
    visualize()


def efficiency():
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
            canon, _, _ = N.canocalize(obs, ip)
            count(canon, Out)

    scale = COMBINATIONS / samples
    rate = 100 * (len(In) - len(Out)) / len(In)
    print(f"scaled rate={scale * rate:.1f}")
    print(f"In={len(In)} Out={len(Out)}")
