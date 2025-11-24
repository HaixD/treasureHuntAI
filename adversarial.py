import numpy as np
from utils import euclidean_distance
from utils import get_neighbors
from algorithms import a_star
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
            self.debug = {}

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

        def build_node(self, build_value=False):
            if not self.is_partial():
                return
            
            move, agentIndex = self.state, self.get_agent_index()
            
            new_state = deepcopy(self.parent.state)
            new_state.apply_move(move, agentIndex)

            self.state = new_state

            if build_value:
                self.value = self.state.get_utility_value()

        def alpha_beta_minimax(self, limit, prune=False):
            def dfs(node, alpha, beta):
                if (node.expanded and node.is_leaf()) or (node.depth == self.depth + limit - 1):
                    node.build_node(True)
                    return node.value
                
                node.build_node()
                node.shallow_expand()
                
                agentIndex = node.get_agent_index(True)

                if not prune:
                    for child in node.children:
                        v = dfs(child, alpha, beta)
                        if agentIndex == 0:
                            node.value = node.value or -float('inf')
                            node.value = max(node.value, v)
                        else:
                            node.value = node.value or float('inf')
                            node.value = min(node.value, v)
                    return node.value
                
                if agentIndex == 0:
                    for child in node.children:
                        v = dfs(child, alpha, beta)
                        alpha = max(alpha, v)
                        if alpha >= beta:
                            break
                    node.value = alpha
                    return alpha
                else:
                    for child in node.children:
                        v = dfs(child, alpha, beta)
                        beta = max(alpha, v)
                        if beta <= alpha:
                            break
                    node.value = beta
                    return beta

            dfs(self, -float('inf'), float('inf'))

        def get_next_node(self):
            agentIndex = self.get_agent_index(True)
            
            if agentIndex == 0:
                next_node = max(self.children, key=lambda node : node.value)
            else:
                next_node = min(self.children, key=lambda node : node.value)

            next_node.expanded = False
            next_node.children = []
            
            return next_node
    
    def __init__(self, grid, start_pos1, start_pos2):
        self.grid = grid
        self.diagonal_length = euclidean_distance((0, 0), grid.shape) #this is the max distance possible between a point and treasure
        self.agents = (Agent(start_pos1), Agent(start_pos2))
        self.paths = ([], [])

        self.treasures = set()
        for r, row in enumerate(self.grid):
            for c, col in enumerate(row):
                if col == Cell.TREASURE:
                    self.treasures.add((r, c))

    def __str__(self):
        grid = deepcopy(self.grid)
        grid[*self.agents[0].position] = 8
        grid[*self.agents[1].position] = 9
        
        text = str(grid)
        text = text.replace('0', ' ').replace('8', 'A').replace('9', 'B')

        return text

    def __deepcopy__(self, memo):
        copied = Adversarial(np.zeros((0, 0), dtype=int), self.agents[0].position, self.agents[1].position)
        copied.grid = deepcopy(self.grid)
        copied.diagonal_length = self.diagonal_length
        copied.agents[0].treasures = self.agents[0].treasures
        copied.agents[1].treasures = self.agents[1].treasures
        copied.treasures = set(treasure for treasure in self.treasures)
        copied.paths = deepcopy(self.paths)
        
        return copied

    def apply_move(self, move, agentIndex):
        if move in self.treasures:
            self.treasures.remove(move)
            self.agents[agentIndex].treasures += 1

        self.agents[agentIndex].position = move
        self.paths[agentIndex].append(move)

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
            max_cloest_treasure = min(max_cloest_treasure, len(a_star(self.grid, MAX.position, treasure, euclidean_distance)[0]))

        min_cloest_treasure = float('inf')
        for treasure in self.treasures:
            min_cloest_treasure = min(min_cloest_treasure, len(a_star(self.grid, MIN.position, treasure, euclidean_distance)[0]))

        closest_treasure_score = (self.diagonal_length - max_cloest_treasure) - (self.diagonal_length - min_cloest_treasure)
        owned_treasure_score = (MAX.treasures - MIN.treasures) * self.diagonal_length

        return closest_treasure_score + owned_treasure_score
    
    def search(self, limit=5, max_iterations=10000):
        root = Adversarial.Node(deepcopy(self))

        for _ in range(max_iterations - 1):
            if not root.state.treasures:
                break
            root.alpha_beta_minimax(limit)
            root = root.get_next_node()
            print(root.state, end='\n\n')
        else:
            root.alpha_beta_minimax(limit)


        return root

if __name__ == '__main__':
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 2],
    ], dtype=int)
    adversarial = Adversarial(grid, (3, 1), (2, 1))
    node = adversarial.search()
    def print_tree(node, depth=0):
        rows = ['        ' * depth + row for row in str(node.state).split('\n')]
        if 'punishment' in node.debug:
            rows[0] += f' {node.state.get_utility_value()}'
            rows[1] += f' {node.debug["punishment"]}'
        rows[2] += f' {node.value}'
        text = '\n'.join(rows)

        print(text)
        for child in node.children:
            print_tree(child, depth + 1)

    # print_tree(node)