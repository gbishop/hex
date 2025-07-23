import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# --- Your provided Hex class (copied here for self-containment) ---
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
        if not (0 <= index < len(board) and board[index] == 0):
            # print(f"Invalid move: index {index} is out of bounds or cell is not empty.")
            return False  # Indicate invalid move
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

    def legal_moves(self) -> list[int]:
        return [index for index in range(self.size**2) if self.board[index] == 0]

    def __str__(self):
        chars = {player1: "X", player2: "O", 0: "_"}
        S = self.size
        lines = []
        for r in range(S):
            index = r * S
            line = " " * r + " ".join(chars[c] for c in self.board[index : index + S])
            lines.append(line)
        return "\n".join(lines)


# --- PyTorch Implementation for Self-Learning ---


class HexPolicyNet(nn.Module):
    """
    A simple feed-forward neural network for the Hex game policy.
    It takes the flattened board state as input and outputs logits
    for each possible move.
    """

    def __init__(self, board_size: int):
        super(HexPolicyNet, self).__init__()
        self.board_dim = board_size * board_size
        self.fc1 = nn.Linear(self.board_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.board_dim)  # Output logits for each cell

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): The input board state, flattened.
                              Shape: (batch_size, board_dim)
        Returns:
            torch.Tensor: Logits for each possible move.
                          Shape: (batch_size, board_dim)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Return logits, not probabilities yet


class HexAgent:
    """
    Agent that uses a neural network to decide moves and learns
    from self-play using a policy gradient method.
    """

    def __init__(
        self, board_size: int, learning_rate: float = 0.001, gamma: float = 0.99
    ):
        self.board_size = board_size
        self.policy_net = HexPolicyNet(board_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # Discount factor for rewards

        # Store experiences for an episode
        self.log_probs = []
        self.rewards = []

    def get_action(self, hex_game: Hex, player: int, epsilon: float = 0.1) -> int:
        """
        Selects an action (move) based on the current policy and an
        epsilon-greedy exploration strategy.
        Args:
            hex_game (Hex): The current Hex game instance.
            player (int): The current player (1 or -1).
            epsilon (float): Probability of taking a random action.
        Returns:
            int: The chosen move (index on the board).
        """
        legal_moves = hex_game.legal_moves()
        if not legal_moves:
            return -1  # No legal moves

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.choice(legal_moves)

        # Convert board to tensor, normalizing player values to be consistent for the network
        # The network should learn to interpret board states regardless of which player is '1'
        # For simplicity, we'll represent the current player's pieces as 1, opponent's as -1, empty as 0
        board_tensor = self._board_to_tensor(hex_game.board, player)

        # Get logits from the policy network
        with torch.no_grad():  # No need to calculate gradients for action selection
            logits = self.policy_net(board_tensor)

        # Mask out illegal moves by setting their logits to a very small number
        # This ensures they have negligible probability
        mask = torch.full_like(logits, -float("inf"))
        for move_idx in legal_moves:
            mask[0, move_idx] = (
                0  # Set valid move logits to 0 (or their original value)
            )

        masked_logits = logits + mask

        # Convert logits to probabilities and then to a categorical distribution
        probs = torch.softmax(masked_logits, dim=-1)

        # Sample an action from the distribution
        action_distribution = torch.distributions.Categorical(probs)
        action = action_distribution.sample()

        # Store the log probability of the chosen action for learning
        self.log_probs.append(action_distribution.log_prob(action))

        return int(action.item())

    def store_reward(self, reward: float):
        """
        Stores the reward received for the last action.
        Args:
            reward (float): The reward value.
        """
        self.rewards.append(reward)

    def learn(self):
        """
        Performs a policy gradient update using the collected log probabilities
        and rewards from an episode.
        """
        if not self.log_probs:  # No moves made in the episode
            return

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in self.rewards[::-1]:  # Iterate backwards
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)  # Insert at the beginning to maintain order

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards (optional, but often helps training stability)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9
            )

        # Calculate policy loss
        # We want to maximize expected reward, so we minimize -log_prob * reward
        policy_loss = []
        for log_prob, G in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * G)

        loss = torch.cat(policy_loss).sum()

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode history
        self.log_probs = []
        self.rewards = []

    def _board_to_tensor(self, board: list[int], current_player: int) -> torch.Tensor:
        """
        Converts the Hex board list into a PyTorch tensor.
        The board is normalized such that the current player's pieces are 1,
        the opponent's are -1, and empty cells are 0.
        Args:
            board (list[int]): The game board state.
            current_player (int): The player whose turn it is (1 or -1).
        Returns:
            torch.Tensor: Flattened board tensor. Shape: (1, board_dim)
        """
        tensor_board = np.array(board, dtype=np.float32)
        # Normalize board values relative to the current player
        # If current_player is 1, board values remain as is (1 for player1, -1 for player2)
        # If current_player is -1, board values are inverted (-1 for player1, 1 for player2)
        normalized_board = tensor_board * current_player
        return torch.tensor(normalized_board, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension


# --- Training Configuration ---
BOARD_SIZE = 5  # Example board size (e.g., 5x5)
NUM_EPISODES = 5000  # Number of games to simulate for training
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.995  # Decay rate for epsilon per episode

# --- Main Training Loop ---
if __name__ == "__main__":
    agent_p1 = HexAgent(BOARD_SIZE)  # Agent for Player 1 (X)
    agent_p2 = HexAgent(
        BOARD_SIZE
    )  # Agent for Player 2 (O) - can share weights or be separate

    epsilon = EPSILON_START
    win_counts = {player1: 0, player2: 0}

    print(f"Starting Hex self-play training on a {BOARD_SIZE}x{BOARD_SIZE} board...")

    for episode in range(1, NUM_EPISODES + 1):
        game = Hex(BOARD_SIZE)
        current_player = player1  # Player 1 starts
        done = False
        winner = None

        # Store states and actions for both players in this episode
        # We need to apply rewards at the end of the episode
        states_p1 = []
        log_probs_p1 = []
        states_p2 = []
        log_probs_p2 = []

        while not done:
            # Get legal moves for the current player
            legal_moves = game.legal_moves()
            if not legal_moves:
                # Game is a draw if no legal moves remain (shouldn't happen in Hex normally)
                done = True
                break

            if current_player == player1:
                # Player 1's turn
                board_tensor = agent_p1._board_to_tensor(game.board, current_player)
                logits = agent_p1.policy_net(board_tensor)
                mask = torch.full_like(logits, -float("inf"))
                for move_idx in legal_moves:
                    mask[0, move_idx] = 0
                masked_logits = logits + mask
                probs = torch.softmax(masked_logits, dim=-1)
                action_distribution = torch.distributions.Categorical(probs)

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    chosen_move = random.choice(legal_moves)
                    log_prob_chosen = action_distribution.log_prob(
                        torch.tensor(chosen_move)
                    )
                else:
                    chosen_move = action_distribution.sample().item()
                    log_prob_chosen = action_distribution.log_prob(
                        torch.tensor(chosen_move)
                    )

                # Store for learning
                states_p1.append(board_tensor)
                log_probs_p1.append(log_prob_chosen)

                # Make the move
                win = game.move(int(chosen_move), current_player)
                if win:
                    winner = current_player
                    done = True
                current_player = player2  # Switch player

            else:  # current_player == player2
                # Player 2's turn
                board_tensor = agent_p2._board_to_tensor(game.board, current_player)
                logits = agent_p2.policy_net(board_tensor)
                mask = torch.full_like(logits, -float("inf"))
                for move_idx in legal_moves:
                    mask[0, move_idx] = 0
                masked_logits = logits + mask
                probs = torch.softmax(masked_logits, dim=-1)
                action_distribution = torch.distributions.Categorical(probs)

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    chosen_move = random.choice(legal_moves)
                    log_prob_chosen = action_distribution.log_prob(
                        torch.tensor(chosen_move)
                    )
                else:
                    chosen_move = action_distribution.sample().item()
                    log_prob_chosen = action_distribution.log_prob(
                        torch.tensor(chosen_move)
                    )

                # Store for learning
                states_p2.append(board_tensor)
                log_probs_p2.append(log_prob_chosen)

                # Make the move
                win = game.move(int(chosen_move), current_player)
                if win:
                    winner = current_player
                    done = True
                current_player = player1  # Switch player

        # --- Episode End: Assign Rewards and Learn ---
        if winner == player1:
            reward_p1 = 1.0
            reward_p2 = -1.0
            win_counts[player1] += 1
        elif winner == player2:
            reward_p1 = -1.0
            reward_p2 = 1.0
            win_counts[player2] += 1
        else:  # Draw (should be rare in Hex)
            reward_p1 = 0.0
            reward_p2 = 0.0

        # Assign rewards to all steps in the episode for each player
        for _ in log_probs_p1:
            agent_p1.store_reward(reward_p1)
        for _ in log_probs_p2:
            agent_p2.store_reward(reward_p2)

        # Perform learning update for both agents
        # Note: In self-play, you often train one agent and copy its weights,
        # or train two agents against each other. Here, we train two distinct agents.
        # For a single self-learning agent, you'd have one agent playing against itself
        # by switching the board state perspective.
        agent_p1.log_probs = log_probs_p1  # Set the collected log_probs for P1
        agent_p1.learn()

        agent_p2.log_probs = log_probs_p2  # Set the collected log_probs for P2
        agent_p2.learn()

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"Episode {episode}/{NUM_EPISODES}, Epsilon: {epsilon:.3f}")
            print(
                f"Win Counts - Player 1 (X): {win_counts[player1]}, Player 2 (O): {win_counts[player2]}"
            )
            # Optional: Print the final board of a recent game
            # if episode % 1000 == 0:
            #     print("Final Board:")
            #     print(game)

    print("\nTraining complete!")
    print(
        f"Final Win Counts - Player 1 (X): {win_counts[player1]}, Player 2 (O): {win_counts[player2]}"
    )

    # --- Example of playing a trained agent (without exploration) ---
    print("\n--- Playing a game with trained agents (no exploration) ---")
    test_game = Hex(BOARD_SIZE)
    current_player = player1
    done = False

    while not done:
        legal_moves = test_game.legal_moves()
        if not legal_moves:
            print("No legal moves left. Game is a draw.")
            break

        print(f"\nPlayer { {1:'X', -1:'O'}[current_player]}'s turn:")
        print(test_game)

        if current_player == player1:
            chosen_move = agent_p1.get_action(
                test_game, current_player, epsilon=0.0
            )  # No exploration
        else:
            chosen_move = agent_p2.get_action(
                test_game, current_player, epsilon=0.0
            )  # No exploration

        if chosen_move == -1:  # No legal moves
            print("Agent chose no legal move. Game ends.")
            break

        print(f"Agent chooses move: {test_game.rc(chosen_move)}")
        win = test_game.move(chosen_move, current_player)

        if win:
            print(f"\nPlayer { {1:'X', -1:'O'}[current_player]} wins!")
            print(test_game)
            done = True
        else:
            current_player = -current_player  # Switch player
