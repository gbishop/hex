import numpy as np

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


class HexGame:
    def __init__(self, size: int):
        S = self.size = size

        self.board = np.zeros(S * S, dtype=np.float32)

        # room for every cell + 4 virtual cells
        self.uf = UnionFind(S**2 + 4)

        # edges
        TE = S * S  # Top Edge virtual node
        BE = S * S + 1  # Bottom Edge virtual node
        LE = S * S + 2  # Left Edge virtual node
        RE = S * S + 3  # Right Edge virtual node
        self.edges = {
            player1: (TE, BE),  # Player 1 (X) connects Top to Bottom
            player2: (LE, RE),  # Player 2 (O) connects Left to Right
        }

        # map index to connections
        i2c: dict[int, Connections] = {}
        C = Connections

        # board coordinates
        T = 0  # Row offset for Top edge
        B = S * (S - 1)  # Row offset for Bottom edge
        L = 0  # Column offset for Left edge
        R = S - 1  # Column offset for Right edge

        # corners
        # Top-Left Corner
        i2c[T + L] = C(N=[0 + 1, S + 0], P1=[TE], P2=[LE])
        # Top-Right Corner
        i2c[T + R] = C(N=[0 - 1, S - 1, S + 0], P1=[TE], P2=[RE])
        # Bottom-Left Corner
        i2c[B + L] = C(N=[-S + 0, -S + 1, 0 + 1], P1=[BE], P2=[LE])
        # Bottom-Right Corner
        i2c[B + R] = C(N=[-S + 0, 0 - 1], P1=[BE], P2=[RE])

        # edges (excluding corners)
        # Top Edge cells
        top = C(N=[0 - 1, 0 + 1, S - 1, S + 0], P1=[TE])
        for c in range(1, S - 1):
            i2c[T + c] = top
        # Bottom Edge cells
        bottom = C(N=[-S + 0, -S + 1, 0 - 1, 0 + 1], P1=[BE])
        for c in range(1, S - 1):
            i2c[B + c] = bottom
        # Left Edge cells
        left = C(N=[-S + 0, -S + 1, 0 + 1, S + 0], P2=[LE])
        for r in range(S, S * S - S, S):
            i2c[r + L] = left
        # Right Edge cells
        right = C(N=[-S + 0, 0 - 1, S - 1, S - 0], P2=[RE])
        for r in range(S, S * S - S, S):
            i2c[r + R] = right

        # remainder (middle cells)
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
        # Ensure the move is to an empty cell
        assert 0 <= index < len(board) and abs(board[index]) == 0
        board[index] = player
        connections = self.index_to_connections[index]
        for step in connections.neighbors:
            neighbor_index = index + step
            # Check if neighbor is within board bounds and occupied by the same player
            if 0 <= neighbor_index < len(board) and board[neighbor_index] == player:
                self.uf.union(index, neighbor_index)
        for edge in connections.edges.get(player, []):
            self.uf.union(index, edge)
        E1, E2 = self.edges[player]
        return self.uf.find(E1) == self.uf.find(E2)

    def legal_moves(self):
        # return [index for index in range(self.size**2) if self.board[index] == 0]
        return np.where(self.board == 0)[0]

    def __str__(self):
        chars = {player1: "X", player2: "O", 0: "_"}
        S = self.size
        lines = []
        for r in range(S):
            index = r * S
            line = " " * r + " ".join(chars[c] for c in self.board[index : index + S])
            lines.append(line)
        return "\n".join(lines)
