# player1 : MCTS 

import random
import math
import time
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Set

ROWS, COLS = 20, 30
DIRECTIONS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)] 

class GameState:
    def __init__(self, players: List[Tuple[int, int]], stars: List[Tuple[int, int]], 
                 scores: List[int], current_player: int):
        self.players = players[:]
        self.stars = set(stars)
        self.scores = scores[:]
        self.current_player = current_player
        self.num_players = len(players)
    
    def copy(self):
        return GameState(self.players, list(self.stars), self.scores, self.current_player)
    
    def is_valid_move(self, player_idx: int, new_pos: Tuple[int, int]) -> bool:
        """Check if a move is valid"""
        old_pos = self.players[player_idx]
        r, c = new_pos
        
        # Check bounds
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return False
        
        # Check movement constraint (one step or stay)
        if abs(r - old_pos[0]) + abs(c - old_pos[1]) > 1:
            return False
        
        return True
    
    def get_possible_moves(self, player_idx: int) -> List[Tuple[int, int]]:
        """Get all possible moves for a player"""
        current_pos = self.players[player_idx]
        moves = []
        
        for dr, dc in DIRECTIONS:
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)
            if self.is_valid_move(player_idx, new_pos):
                moves.append(new_pos)
        
        return moves
    
    def simulate_turn(self, moves: List[Tuple[int, int]]) -> 'GameState':
        """Simulate one complete turn with all players' moves"""
        new_state = self.copy()
        
        # Apply the same conflict resolution as the server
        proposed = moves[:]
        old_positions = new_state.players[:]
        
        # Resolve same-target conflicts randomly
        target_to_indices = {}
        for i, tgt in enumerate(proposed):
            target_to_indices.setdefault(tgt, []).append(i)
        
        winners = set()
        for tgt, idxs in target_to_indices.items():
            if len(idxs) == 1:
                winners.add(idxs[0])
            else:
                winner = random.choice(idxs)
                winners.add(winner)
        
        # Apply occupancy rule
        will_move = [False] * len(proposed)
        for i in range(len(proposed)):
            will_move[i] = (i in winners) and (proposed[i] != old_positions[i])
        
        vacating = {old_positions[i]: will_move[i] for i in range(len(old_positions))}
        
        final_positions = list(old_positions)
        for i in range(len(proposed)):
            if not will_move[i]:
                continue
            target = proposed[i]
            if target in vacating and not vacating[target]:
                continue
            final_positions[i] = target
        
        new_state.players = final_positions
        
        # Update scores for stars collected
        for idx, pos in enumerate(final_positions):
            if pos in new_state.stars:
                new_state.scores[idx] += 1
                new_state.stars.discard(pos)
        
        return new_state
    
    def get_reward(self, player_idx: int) -> float:
        """Calculate reward for the current player"""
        if not self.stars:  # Game over
            # Bonus for winning
            max_score = max(self.scores)
            if self.scores[player_idx] == max_score:
                winners = [i for i, score in enumerate(self.scores) if score == max_score]
                if len(winners) == 1:
                    return 100.0  # Win bonus
                else:
                    return 50.0 / len(winners)  # Tie bonus
        
        # Score difference reward
        my_score = self.scores[player_idx]
        avg_score = sum(self.scores) / len(self.scores)
        return my_score - avg_score

class MCTSNode:
    """Node in the MCTS tree"""
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Tuple[int, int]] = None):
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves: List[Tuple[int, int]] = []
        
        if state:
            self.untried_moves = state.get_possible_moves(state.current_player)
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        return len(self.state.stars) == 0 or self.visits > 0 and len(self.children) == 0
    
    def ucb1_score(self, exploration_weight: float = 1.41) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self) -> 'MCTSNode':
        """Select child with highest UCB1 score"""
        return max(self.children.values(), key=lambda child: child.ucb1_score())
    
    def expand(self) -> 'MCTSNode':
        """Expand tree by adding a new child node"""
        if not self.untried_moves:
            return self
        
        move = self.untried_moves.pop(random.randint(0, len(self.untried_moves) - 1))
        
        # Create new state by simulating all players' moves
        # For simplicity, assume other players make random moves
        all_moves = []
        for i in range(self.state.num_players):
            if i == self.state.current_player:
                all_moves.append(move)
            else:
                possible = self.state.get_possible_moves(i)
                all_moves.append(random.choice(possible) if possible else self.state.players[i])
        
        new_state = self.state.simulate_turn(all_moves)
        new_state.current_player = self.state.current_player
        
        child = MCTSNode(new_state, parent=self, move=move)
        self.children[move] = child
        return child
    
    def simulate(self, max_depth: int = 20) -> float:
        """Run a random simulation from this node"""
        current_state = self.state.copy()
        depth = 0
        
        while len(current_state.stars) > 0 and depth < max_depth:
            # Generate random moves for all players
            all_moves = []
            for i in range(current_state.num_players):
                possible = current_state.get_possible_moves(i)
                if possible:
                    # Add some intelligence: prefer moves towards nearest star
                    if random.random() < 0.3:  # 30% chance to be greedy
                        player_pos = current_state.players[i]
                        nearest_star = min(current_state.stars, 
                                         key=lambda s: abs(s[0] - player_pos[0]) + abs(s[1] - player_pos[1]))
                        
                        best_move = min(possible, 
                                      key=lambda m: abs(m[0] - nearest_star[0]) + abs(m[1] - nearest_star[1]))
                        all_moves.append(best_move)
                    else:
                        all_moves.append(random.choice(possible))
                else:
                    all_moves.append(current_state.players[i])
            
            current_state = current_state.simulate_turn(all_moves)
            depth += 1
        
        return current_state.get_reward(self.state.current_player)
    
    def backpropagate(self, reward: float):
        """Backpropagate the reward up the tree"""
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

class MCTS:
    """Monte Carlo Tree Search implementation"""
    def __init__(self, time_limit: float = 0.8):
        self.time_limit = time_limit
    
    def search(self, root_state: GameState) -> Tuple[int, int]:
        """Perform MCTS and return the best move"""
        root = MCTSNode(root_state)
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < self.time_limit:
            # Selection
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child()
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation
            reward = node.simulate()
            
            # Backpropagation
            node.backpropagate(reward)
            iterations += 1
        
        # Return the most visited child (most robust)
        if root.children:
            best_move = max(root.children.keys(), 
                          key=lambda move: root.children[move].visits)
            return best_move
        else:
            # Fallback to current position
            return root_state.players[root_state.current_player]

# Global MCTS instance
mcts_ai = MCTS()

def move1(players: List[Tuple[int, int]], stars: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Main function called by the game server"""
    try:
        # Create game state (assuming we are player 0)
        scores = [0] * len(players)  # We don't have access to actual scores, so initialize
        state = GameState(players, stars, scores, current_player=0)
        
        # Use MCTS to find the best move
        best_move = mcts_ai.search(state)
        return best_move
        
    except Exception as e:
        # Fallback: move towards nearest star or stay
        current_pos = players[0]
        if stars:
            nearest_star = min(stars, key=lambda s: abs(s[0] - current_pos[0]) + abs(s[1] - current_pos[1]))
            
            # Simple greedy move towards nearest star
            dr = 0 if nearest_star[0] == current_pos[0] else (1 if nearest_star[0] > current_pos[0] else -1)
            dc = 0 if nearest_star[1] == current_pos[1] else (1 if nearest_star[1] > current_pos[1] else -1)
            
            new_pos = (current_pos[0] + dr, current_pos[1] + dc)
            
            # Validate move
            if (0 <= new_pos[0] < ROWS and 0 <= new_pos[1] < COLS and 
                abs(dr) + abs(dc) <= 1):
                return new_pos
        
        return current_pos  # Stay in place if all else fails