import random
import time


player1 = 1
player2 = -1


class Connections:
    def __init__(self, N: list[int], P1: list[int] = [], P2: list[int] = []):
        # offsets to neighbors
        self.neighbors = N
        # edge connections
        self.edges = {player1: P1, player2: P2}

    def __repr__(self):
        return f"Connections(N={self.neighbors}, E={self.edges})"


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size  # Used for union by rank optimization

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by rank: attach smaller tree under root of taller tree
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True  # Successfully united
        return False  # Already in the same set


class Hex:
    def __init__(self, size: int):
        S = self.size = size

        self.board = [0] * (S * S)

        # room for every cell + 4 virtual cells
        self.uf = UnionFind(S**2 + 4)

        # edges
        TE = S * S
        BE = S * S + 1
        LE = S * S + 2
        RE = S * S + 3
        self.edges = {
            player1: (TE, BE),
            player2: (LE, RE),
        }

        # map index to connections
        i2c: dict[int, Connections] = {}
        C = Connections

        # board coordinates
        T = 0
        B = S * (S - 1)
        L = 0
        R = S - 1
        # corners
        i2c[T + L] = C(N=[0 + 1, S + 0], P1=[TE], P2=[LE])
        i2c[T + R] = C(N=[0 - 1, S - 1, S + 0], P1=[TE], P2=[RE])
        i2c[B + L] = C(N=[-S + 0, -S + 1, 0 + 1], P1=[BE], P2=[LE])
        i2c[B + R] = C(N=[-S + 0, 0 - 1], P1=[BE], P2=[RE])
        # edges
        top = C(N=[0 - 1, 0 + 1, S - 1, S + 0], P1=[TE])
        for c in range(1, S - 1):
            i2c[T + c] = top
        bottom = C(N=[-S + 0, -S + 1, 0 - 1, 0 + 1], P1=[BE])
        for c in range(1, S - 1):
            i2c[B + c] = bottom
        left = C(N=[-S + 0, -S + 1, 0 + 1, S + 0], P2=[LE])
        for r in range(S, S * S - S, S):
            i2c[r + L] = left
        right = C(N=[-S + 0, 0 - 1, S - 1, S - 0], P2=[RE])
        for r in range(S, S * S - S, S):
            i2c[r + R] = right
        # remainder
        middle = C(N=[-S + 0, -S + 1, 0 - 1, 0 + 1, S - 1, S + 0])
        for r in range(1, S - 1):
            for c in range(1, S - 1):
                i2c[r * S + c] = middle

        self.index_to_connections = i2c

    def index(self, r: int, c: int) -> int:
        return self.size * r + c

    def rc(self, index: int) -> tuple[int, int]:
        return index // self.size, index % self.size

    def move(self, index: int, player: int) -> bool:
        board = self.board
        assert board[index] == 0
        board[index] = player
        connections = self.index_to_connections[index]
        for step in connections.neighbors:
            if board[index + step] == player:
                self.uf.union(index, index + step)
        for edge in connections.edges.get(player, []):
            self.uf.union(index, edge)
        E1, E2 = self.edges[player]
        return self.uf.find(E1) == self.uf.find(E2)

    def legal_moves(self) -> list[int]:
        return [index for index in range(self.size**2) if not self.board[index]]

    def __str__(self):
        chars = {player1: "X", player2: "O", 0: "_"}
        S = self.size
        lines = []
        for r in range(S):
            index = r * S
            line = " " * r + " ".join(chars[c] for c in self.board[index : index + S])
            lines.append(line)
        return "\n".join(lines)


N = 11

game = Hex(N)

moves = [index for index in range(N**2)]
random.shuffle(moves)

player = 1
t0 = time.time()
for index in moves:
    if game.move(index, player):
        t1 = time.time()
        print(game)
        moves = sum(cell != 0 for cell in game.board)
        print(
            f"player { {1:1, -1: 2}[player]} wins after {moves} moves {moves/(t1-t0):.1f} per second"
        )
        break
    player = -player
