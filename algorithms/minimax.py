"""Minimax algorithm implementation for adversarial grid-based treasure hunt.

This module provides a Minimax search algorithm with optional alpha-beta pruning for two-player
competitive pathfinding where agents compete to collect treasures on a grid with obstacles and
traps.
"""

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from algorithms import a_star
from constants import Cell
from utils import euclidean_distance, get_neighbors

@dataclass
class Agent:
    position: tuple[int, int]
    current_goal: tuple[int, int] # treasure that agent wants to reach first

    treasures: int = 0
    traps: int = 0

class Minimax:
    """Adversarial search using Minimax algorithm with optional alpha-beta pruning.

    This class implements a two-player zero-sum game where agents compete to collect treasures on a
    grid. The maximizing agent tries to maximize their advantage while the minimizing agent tries to
    minimize it.

    Attributes:
        grid (np.ndarray): The game grid containing cells, treasures, traps, and walls.
        max_treasure_distance (float): Maximum possible distance to any treasure.
        agents (tuple[Agent, Agent]): Tuple of two agents (maximizer, minimizer).
        paths (tuple[list, list]): Paths taken by each agent.
        treasures (set): Set of remaining treasure positions.
    """

    class Node:
        def __init__(self, state:"Minimax|tuple", *, parent:"Minimax.Node|None"=None, is_partial=False):
            self.state = state
            self.value = None
            self.parent = parent
            self.children = []
            self.depth = 0 if parent is None else parent.depth + 1
            self.expanded = False
            self.debug = {"expansions": 0}

            if not is_partial:
                self.generate_value()

        def is_partial(self):
            return self.value is None or type(self.state) is tuple

        def is_leaf(self):
            """Check if this is a leaf node with no children.

            Returns:
                bool: True if the node has no children.
            """
            return len(self.children) == 0

        def get_agent_index(self, swap=False):
            return (self.depth + swap) % 2

        def shallow_expand(self):
            """Create child nodes for all valid moves without building full states.

            This creates partial nodes that will be built later as needed, improving
            efficiency by avoiding unnecessary state copying.
            """
            if self.expanded:
                return

            grid = self.state.grid
            agent = self.state.agents[self.get_agent_index()]
            other = self.state.agents[self.get_agent_index(True)]

            valid_moves = get_neighbors(
                grid, agent.position, include_traps=True
            )
            for move in valid_moves:
                if move == other.position:
                    continue
                child = Minimax.Node(move, parent=self, is_partial=True)
                self.children.append(child)

            self.expanded = True

        def generate_value(self):
            MAX_value, cells_expanded1 = self.state.get_utility_value_expansions(0)
            MIN_value, cells_expanded2 = self.state.get_utility_value_expansions(1)
            
            self_value = (MAX_value, MIN_value)[self.get_agent_index()]

            self.value = (MAX_value * abs(MAX_value) - MIN_value * abs(MIN_value), self_value)
            # self.value = (MAX_value - MIN_value, 0)
            self.debug["expansions"] = cells_expanded1 + cells_expanded2

        def build_node(self, build_value=False):
            """Build the full game state for this node if it's partial.

            Args:
                build_value (bool, optional): Whether to compute the utility value for this node.
                    Defaults to False.
            """
            if not self.is_partial():
                return

            move = self.state
            agent_index = self.get_agent_index(True)

            if self.parent:
                new_state = self.parent.state.copy()
                new_state.apply_move(move, agent_index)

                self.state = new_state

                if build_value:
                    self.generate_value()

        def alpha_beta_minimax(self, limit, prune=False):
            def dfs(node, alpha=None, beta=None):
                alpha = alpha or (-float('inf'), 0)
                beta = beta or (float('inf'), 0)
                
                if (node.expanded and node.is_leaf()) or (
                    node.depth == self.depth + limit - 1
                ):
                    node.build_node(True)
                    return node.value

                node.build_node()
                node.shallow_expand()

                agent_index = node.get_agent_index()

                if agent_index == 0:
                    for child in node.children:
                        v = dfs(child, alpha, beta) if prune else dfs(child)
                        alpha = max(alpha, v)
                        if prune and alpha >= beta:
                            break
                    node.value = alpha
                    return alpha
                else:
                    for child in node.children:
                        v = dfs(child, alpha, beta) if prune else dfs(child)
                        beta = min(beta, v)
                        if prune and beta <= alpha:
                            break
                    node.value = beta
                    return beta

            dfs(self)

        def get_next_node(self):
            agent_index = self.get_agent_index()

            if agent_index == 0:
                next_node = max(self.children, 
                                key=lambda node: node.value if node.value is not None else (-float('inf'), 0))
            else:
                next_node = min(self.children, 
                                key=lambda node: node.value if node.value is not None else (float('inf'), 0))
            
            print(agent_index, list(map(lambda node : node.value, self.children)))

            next_node.expanded = False
            next_node.children = []

            return next_node

    def __init__(self, grid, start_pos1, start_pos2):
        """Initialize a Minimax game instance.

        Args:
            grid (np.ndarray): 2D grid containing cells, treasures, traps, and walls.
            start_pos1 (tuple): Starting position for agent 1 (maximizer) as (row, col).
            start_pos2 (tuple): Starting position for agent 2 (minimizer) as (row, col).
        """
        self.grid = grid
        self.paths = ([], [])
        self.length_cache = {} # length_cache[treasure_pos][pos] = cached length

        # initialize treasures
        self.treasures = set()
        for r, row in enumerate(self.grid):
            for c, col in enumerate(row):
                if col == Cell.TREASURE:
                    self.treasures.add((r, c))

        # initialize agents
        self.agents = (Agent(start_pos1, (-1, -1)), Agent(start_pos2, (-1, -1))) # PLACEHOLDER
        self.update_treasures()

    def __str__(self):
        """Return a string representation of the current game state.

        Returns:
            str: Grid visualization with 'A' for agent 1 and 'B' for agent 2.
        """
        grid = deepcopy(self.grid)
        grid[*self.agents[0].position] = 8
        grid[*self.agents[1].position] = 9

        grid[grid == 2] = 0
        for treasure in self.treasures:
            grid[*treasure] = 2

        agent1_char = chr(ord('A') + self.agents[0].treasures)
        agent2_char = chr(ord('Z') - self.agents[1].treasures)

        text = str(grid)
        text = (text.replace("0", " ")
                .replace("8", agent1_char)
                .replace("9", agent2_char)
                .replace(str(Cell.WALL), "#")
                .replace(str(Cell.TRAP), "@"))

        return text
    
    def update_treasures(self):
        total_expansions = 0
        
        for i, agent in enumerate(self.agents):
            if agent.current_goal not in self.treasures:
                agent.current_goal, cells_expanded = self.get_cloest_treasure(i)
                total_expansions += cells_expanded
        
        return total_expansions

    # consider get_best_treasure where treasure quality = other agent distance - given agent distance to treasure
    def get_cloest_treasure(self, agent_index):
        other_index = (agent_index + 1) % 2
        shortest = (float('inf'), (0, 0), 0) # (length, position, cells expanded)
        for treasure in self.treasures:
            self_length, cells_expanded1 = self.get_a_star_length(self.agents[agent_index].position, treasure)
            other_length, cells_expanded2 = self.get_a_star_length(self.agents[other_index].position, treasure)

            length = self_length - other_length
            cells_expanded = cells_expanded1 + cells_expanded2
            
            shortest = min(shortest, (length, treasure, cells_expanded), key=lambda x : x[0])

        return shortest[1:]

    def get_a_star_length(self, start, goal):
        if goal in self.length_cache: 
            if start in self.length_cache[goal]:
                return self.length_cache[goal][start], 0
        else:
            self.length_cache[goal] = {}
        
        path, expanded_cells = a_star(self.grid, start, goal, euclidean_distance)

        if path:
            for i, (_, position) in enumerate(path):
                self.length_cache[goal][position] = len(path) - 1 - i
        
        return self.length_cache[goal][start], expanded_cells

    def copy(self):
        copied = Minimax(
            np.zeros((0, 0), dtype=int),
            self.agents[0].position,
            self.agents[1].position,
        )
        copied.grid = self.grid
        copied.agents= deepcopy(self.agents)
        copied.treasures = set(treasure for treasure in self.treasures)
        copied.paths = deepcopy(self.paths)
        copied.length_cache = self.length_cache

        return copied

    def apply_move(self, move, agent_index):
        if move in self.treasures:
            self.treasures.remove(move)
            self.agents[agent_index].treasures += 1
            self.update_treasures()
        elif self.grid[move] == Cell.TRAP:
            self.agents[agent_index].traps += 1
            self.agents[agent_index].traps += 1

        self.agents[agent_index].position = move
        self.paths[agent_index].append(move)
        self.agents[agent_index].position = move
        self.paths[agent_index].append(move)

    def get_utility_value_expansions(self, agent_index):
        agent = self.agents[agent_index]

        if len(self.treasures) == 0:
            other = self.agents[(agent_index + 1) % 2]
            if agent.treasures > other.treasures:
                return float("inf"), 0
            elif agent.treasures < other.treasures:
                return -float("inf"), 0

            return 0, 0

        distance, cells_expanded = self.get_a_star_length(agent.position, agent.current_goal)

        distance_score = -distance * 0.5
        treasure_score = agent.treasures * 500
        trap_score = -agent.traps * 5 # BUGGED

        total_score = distance_score + treasure_score + trap_score

        return total_score, cells_expanded
    
    def get_utility_value(self, *args, **kwargs):
        return self.get_utility_value_expansions(*args, **kwargs)[0]

    def search(self, limit=5, prune=False, max_iterations=10000):
        root = Minimax.Node(self.copy())
        curr = root

        for _ in range(max_iterations):
            if not curr.state.treasures:
                break
            curr.alpha_beta_minimax(limit, prune)
            curr = curr.get_next_node()
            print(curr.state, end='\n\n')

        def get_expansions(root):
            """Recursively count total expansions in the tree."""
            total = 0
            for child in root.children:
                total += get_expansions(child)

            return total + root.debug["expansions"]

        return curr, get_expansions(root)

    def search_increment(self, state=None, *, limit=5, prune=False, move=None):
        """Perform one incremental step of Minimax search.

        This allows stepping through the search one move at a time, useful for visualization or
        interactive applications.

        Args:
            state (dict, optional): Current search state containing 'curr', 'limit',
                and 'prune'. Defaults to None (creates initial state).
            limit (int, optional): Depth limit for minimax search. Defaults to 5.
            prune (bool, optional): Whether to use alpha-beta pruning. Defaults to False.
            move (int, optional): Index of child to select. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - state (dict): Updated search state.
                - moves (list): List of possible next moves (empty if game over).
        """
        state = state or {
            "curr": Minimax.Node(self.copy()),
            "limit": limit,
            "prune": prune,
        }

        if move is not None:
            state["curr"] = state["curr"].children[move]

        state["curr"].alpha_beta_minimax(state["limit"], state["prune"])
        state["curr"] = state["curr"].get_next_node()
        state["curr"].shallow_expand()

        if not state["curr"].state.treasures:
            return state, []

        return state, list(map(lambda node: node.state, state["curr"].children))

    def print_tree(self, node, depth=0):
        """Recursively print the minimax game tree.

        Args:
            node (Node): The node to print.
            depth (int, optional): Current depth in the tree. Defaults to 0.
        """
        rows = ["  " * depth + row for row in str(node.state).split("\n")]
        rows[0] += f" {round(node.value[0], 2)}"
        text = "\n".join(rows)

        print(text, end="\n\n")
        node.children = sorted(
            node.children, key=lambda child: child.value, reverse=depth % 2
        )
        for child in node.children:
            self.print_tree(child, depth + 1)


if __name__ == "__main__":
    from random import randint, choice, seed

    seed(0)
    
    def place_random(grid, value):
        h, w = grid.shape

        while True:
            position = randint(0, h - 1), randint(0, w - 1)
            if grid[position] != value:
                break
        grid[position] = value

    def place_around(grid, position, value):
        h, w = grid.shape
        options = [
            (position[0] - 1, position[1]),
            (position[0] + 1, position[1]),
            (position[0], position[1] - 1),
            (position[0], position[1] + 1),
        ]
        for position in options.copy():
            for coordinate in position:
                if not 0 <= coordinate < h or grid[position] == value:
                    options.remove(position)
                    break

        grid[choice(options)] = value
    
    # test 1
    grid1 = np.array([[0] * 15 for _ in range(15)], dtype=int)
    grid1[7, 7] = Cell.TREASURE

    for i in range(10):
        if i < 6:
            place_random(grid1, Cell.TRAP)
        place_random(grid1, Cell.WALL)
    minimax1 = Minimax(grid1, (1, 1), (13, 13))
    # print(minimax1, end='\n\n')
    # minimax1.search(limit=4, prune=True)

    # test 2
    grid2 = np.array([[0] * 15 for _ in range(15)], dtype=int)
    treasures = [(6, 5), (7, 10), (10, 8)]
    for pos in treasures:
        grid2[pos] = Cell.TREASURE
        for _ in range(2):
            place_around(grid2, pos, Cell.TRAP)
    for _ in range(10):
        place_random(grid2, Cell.WALL)
    
    minimax2 = Minimax(grid2, (2, 12), (12, 2))
    # print(minimax2, end='\n\n')
    # node, _ =  minimax2.search(limit=5, prune=False)

    # test 3
    grid3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,],])
    minimax3 = Minimax(grid3, (10, 18), (6, 18))

    print(minimax3, end='\n\n')
    node, _ = minimax3.search(limit=3, prune=False, max_iterations=101)

    # test 4
    grid4 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
                      [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 4, 0,],
                      [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],])
    minimax4 = Minimax(grid4, (14, 15), (6, 1))

    # print(minimax4, end='\n\n')
    # node, _ = minimax4.search(limit=3, prune=False, max_iterations=100)