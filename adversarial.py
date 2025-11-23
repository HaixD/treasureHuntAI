import numpy as np
from utils import euclidean_distance
from utils import get_neighbors
from constants import Cell
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class Agent:
    position: tuple[float, float]
    treasures: int = 0

class Adversarial:
    class Node:
        def __init__(self, adversarial_state, *, parent=None, is_partial=False):
            self.state = adversarial_state
            self.value = adversarial_state.get_utility_value() if not is_partial else None
            self.parent = parent
            self.children = []
            self.depth = 0 if parent is None else parent.depth + 1
            self.expanded = False

        def is_partial(self):
            return self.value is None
        
        def is_leaf(self):
            return len(self.children) == 0

        def get_agent_index(self, swap=False):
            return (self.depth + swap) % 2 == 1

        def shallow_expand(self):
            if self.expanded:
                return
            
            agentIndex = self.get_agent_index(True)
            grid, agents = self.state.grid, self.state.agents,

            valid_moves = get_neighbors(grid, agents[agentIndex].position, include_traps=True)
            for move in valid_moves:
                self.children.append(Adversarial.Node(move, parent=self, is_partial=True))

            self.expanded = True

        def build_node(self):
            if not self.is_partial():
                return
            
            move, agentIndex = self.state, self.get_agent_index()
            
            new_state = deepcopy(self.parent.state)
            if move in new_state.treasures:
                new_state.treasures.remove(move)
                new_state.agents[agentIndex].treasures += 1

            new_state.agents[agentIndex].position = move
            new_state.paths[agentIndex].append(move)

            self.state = new_state
            self.value = self.state.get_utility_value()

        def alpha_beta_minimax(self, limit):
            def dfs(node, alpha, beta):
                if (node.expanded and node.is_leaf()) or (node.depth == self.depth + limit - 1):
                    node.build_node()
                    return node.value
                
                node.build_node()
                node.shallow_expand()
                
                agentIndex = node.get_agent_index()

                for child in node.children:
                    v = dfs(child, alpha, beta)
                    if agentIndex == 0:
                        node.value = node.value or -float('inf')
                        node.value = max(node.value, v)
                    else:
                        node.value = node.value or float('inf')
                        node.value = min(node.value, v)
                return node.value
                
                # if agentIndex == 0:
                #     for child in node.children:
                #         v = dfs(child, alpha, beta)
                #         alpha = max(alpha, v)
                #         if alpha >= beta:
                #             break
                #     node.value = alpha
                #     return alpha
                # else:
                #     for child in node.children:
                #         v = dfs(child, alpha, beta)
                #         beta = max(alpha, v)
                #         if beta <= alpha:
                #             break
                #     node.value = beta
                #     return beta

            dfs(self, -float('inf'), float('inf'))

        def get_next_node(self):
            agentIndex = self.get_agent_index(True)
            
            if agentIndex == 0:
                next_node = max(self.children, key=lambda node : node.value)
            else:
                next_node = min(self.children, key=lambda node : node.value)
            
            return next_node

        def foreach_leaf(self, function):
            if not self.children:
                return function(self, self.depth)
            
            for child in self.children:
                child.foreach_leaf(function)
    
    def __init__(self, grid, start_pos1, start_pos2):
        self.grid = grid
        self.diagonal_length = euclidean_distance((0, 0), grid.shape) #this is the max distance possible between a point and treasure
        self.agents = Agent(start_pos1), Agent(start_pos2)
        self.paths = [[], []]

        self.treasures = set()
        for r, row in enumerate(self.grid):
            for c, col in enumerate(row):
                if col == Cell.TREASURE:
                    self.treasures.add((r, c))

    def __str__(self):
        grid = deepcopy(self.grid)
        grid[*self.agents[0].position] = 8
        grid[*self.agents[1].position] = 9
        return str(grid)

    def __deepcopy__(self, memo):
        copied = Adversarial(np.zeros((0, 0), dtype=int), self.agents[0].position, self.agents[1].position)
        copied.grid = deepcopy(self.grid)
        copied.diagonal_length = self.diagonal_length
        copied.agents[0].treasures = self.agents[0].treasures
        copied.agents[1].treasures = self.agents[1].treasures
        copied.treasures = set(treasure for treasure in self.treasures)
        copied.paths = deepcopy(self.paths)
        
        return copied

    def get_utility_value(self):
        MAX, MIN = self.agents
        
        if len(self.treasures) == 0:
            if MAX.treasures > MIN.treasures:
                return float('inf')
            elif MAX.treasures < MIN.treasures:
                return -float('inf')
            return 0
        
        max_cloest_treasure = float('inf')
        for treasure in self.treasures:
            max_cloest_treasure = min(max_cloest_treasure, euclidean_distance(treasure, MAX.position))

        min_cloest_treasure = float('inf')
        for treasure in self.treasures:
            min_cloest_treasure = min(min_cloest_treasure, euclidean_distance(treasure, MIN.position))

        closest_treasure_score = (self.diagonal_length - max_cloest_treasure) - (self.diagonal_length - min_cloest_treasure)
        owned_treasure_score = (MAX.treasures - MIN.treasures) * self.diagonal_length

        return closest_treasure_score + owned_treasure_score
    
    def search(self, limit=5):
        root = Adversarial.Node(deepcopy(self))

        try:
            while root.state.treasures:
                root.alpha_beta_minimax(limit)
                root = root.get_next_node()
                print(str(root.state).replace('0', ' '), end='\n\n')
        except Exception: # PLACEHOLDER DEBUG
            print('ERROR:')
            print(root.children)
            root.shallow_expand()
            print(root.children)

        return root
    
if __name__ == '__main__':
    grid = np.zeros((15, 15), dtype=int)
    grid[14, 13] = Cell.TREASURE
    adversarial = Adversarial(grid, (0, 0), (0, 5))
    node = adversarial.search()