class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size  # Used for union by rank optimization

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Path compression
        return self.parent[i]

    def union(self, i, j):
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


def get_hex_neighbors(row, col, N):
    neighbors = []
    # Standard 6 Hex neighbors
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    for dr, dc in deltas:
        n_row, n_col = row + dr, col + dc
        if 0 <= n_row < N and 0 <= n_col < N:
            neighbors.append((n_row, n_col))
    return neighbors


def check_hex_win(board, N, player):
    # board is N x N, binary representation (1 for player's pieces, 0 otherwise)

    # Map (row, col) to a unique integer index
    def to_index(r, c):
        return r * N + c

    # Define virtual nodes
    # N*N is virtual_top_P1, N*N + 1 is virtual_bottom_P1
    # N*N + 2 is virtual_left_P2, N*N + 3 is virtual_right_P2
    # Adjust for board size
    virtual_top_P1 = N * N
    virtual_bottom_P1 = N * N + 1
    virtual_left_P2 = N * N + 2
    virtual_right_P2 = N * N + 3

    uf = UnionFind(N * N + 4)  # N*N cells + 4 virtual nodes

    for r in range(N):
        for c in range(N):
            if board[r][c] == player:
                current_cell_idx = to_index(r, c)

                # Connect to adjacent pieces of the same player
                for nr, nc in get_hex_neighbors(r, c, N):
                    if board[nr][nc] == player:
                        uf.union(current_cell_idx, to_index(nr, nc))

                # Connect to virtual boundary nodes
                if player == 1:  # Player 1 connects top to bottom
                    if r == 0:
                        uf.union(current_cell_idx, virtual_top_P1)
                    if r == N - 1:
                        uf.union(current_cell_idx, virtual_bottom_P1)
                elif player == 2:  # Player 2 connects left to right
                    if c == 0:
                        uf.union(current_cell_idx, virtual_left_P2)
                    if c == N - 1:
                        uf.union(current_cell_idx, virtual_right_P2)

    # Check for win condition
    if player == 1:
        return uf.find(virtual_top_P1) == uf.find(virtual_bottom_P1)
    elif player == 2:
        return uf.find(virtual_left_P2) == uf.find(virtual_right_P2)
    return False


# Example usage for an 11x11 board (N=11)
N = 11
# Example board state (replace with actual game board)
# Let's imagine a winning board for player 1 (connecting top to bottom)
board = [[0 for _ in range(N)] for _ in range(N)]

# Simulate a winning path for player 1
board[0][5] = 1
board[1][5] = 1
board[2][4] = 1
board[3][4] = 1
board[4][3] = 1
board[5][3] = 1
board[6][2] = 1
board[7][2] = 1
board[8][1] = 1
board[9][1] = 1
board[10][0] = 1


def show_board(board):
    for r, row in enumerate(board):
        line = " " * r + " ".join("_XO"[c] for c in row)
        print(line)
    print()


show_board(board)

print(f"Player 1 wins: {check_hex_win(board, N, 1)}")

# Simulate a winning path for player 2 (connecting left to right)
board_p2 = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N):
    board_p2[0][i] = 2
    board_p2[1][i] = 2  # example to force a chain

board_p2[5][0] = 2
board_p2[5][1] = 2
board_p2[4][2] = 2
board_p2[6][2] = 2
board_p2[5][3] = 2
board_p2[5][4] = 2
board_p2[4][5] = 2
board_p2[6][5] = 2
board_p2[5][6] = 2
board_p2[5][7] = 2
board_p2[4][8] = 2
board_p2[6][8] = 2
board_p2[5][9] = 2
board_p2[5][10] = 2

show_board(board_p2)

print(f"Player 2 wins: {check_hex_win(board_p2, N, 2)}")

# No win example
board_no_win = [[0 for _ in range(N)] for _ in range(N)]
board_no_win[0][0] = 1
board_no_win[N - 1][N - 1] = 1

show_board(board_no_win)

print(f"Player 1 wins (no win): {check_hex_win(board_no_win, N, 1)}")

board_random = [[0 for _ in range(N)] for _ in range(N)]

import random

moves = [(r, c) for r in range(N) for c in range(N)]
random.shuffle(moves)

player = 1
for r, c in moves:
    board_random[r][c] = player
    if check_hex_win(board_random, N, player):
        show_board(board_random)
        print(f"player {player} wins")
        break
    player = 3 - player
