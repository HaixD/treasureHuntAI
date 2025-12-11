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

class Minimax2:
    class Node:
        def __init__(self, state:"Minimax2|tuple", *, parent:"Minimax2.Node|None"=None, is_partial=False):
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
            return len(self.children) == 0

        def get_agent_index(self, swap=False):
            return (self.depth + swap) % 2 == 0

        def shallow_expand(self):
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
                child = Minimax2.Node(move, parent=self, is_partial=True)
                self.children.append(child)

            self.expanded = True

        def generate_value(self):
            # player, cells_expanded1 = self.state.get_utility_value_expansions(self.get_agent_index())
            # other, cells_expanded2 = self.state.get_utility_value_expansions(self.get_agent_index(True))
            
            MAX_value, cells_expanded1 = self.state.get_utility_value_expansions(0)
            MIN_value, cells_expanded2 = self.state.get_utility_value_expansions(1)

            # self.value = player - other
            self.value = MAX_value - MIN_value
            self.debug["expansions"] = cells_expanded1 + cells_expanded2

        def build_node(self, build_value=False):
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
            def dfs(node, alpha=-float('inf'), beta=float('inf')):
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
                print(list(map(lambda node: node.value, self.children)))
                next_node = max(self.children, key=lambda node: node.value if node.value is not None else -float('inf'))
            else:
                print(list(map(lambda node: node.value, self.children)))
                next_node = min(self.children, key=lambda node: node.value if node.value is not None else float('inf'))

            next_node.expanded = False
            next_node.children = []

            return next_node

    def __init__(self, grid, start_pos1, start_pos2):
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
        grid = deepcopy(self.grid)
        grid[*self.agents[0].position] = 8
        grid[*self.agents[1].position] = 9

        grid[grid == 2] = 0
        for treasure in self.treasures:
            grid[*treasure] = 2

        agent1_char = chr(ord('A') + self.agents[0].treasures)
        agent2_char = chr(ord('Z') + self.agents[1].treasures)

        text = str(grid)
        text = text.replace("0", " ").replace("8", agent1_char).replace("9", agent2_char)

        return text
    
    def update_treasures(self):
        for i, agent in enumerate(self.agents):
            if agent.current_goal not in self.treasures:
                agent.current_goal, cells_expanded = self.get_cloest_treasure(i)

    # consider get_best_treasure where treasure quality = other agent distance - given agent distance to treasure
    def get_cloest_treasure(self, agent_index):
        shortest = (float('inf'), (0, 0), 0) # (length, position, cells expanded)
        for treasure in self.treasures:
            length, cells_expanded = self.get_a_star_length(self.agents[agent_index].position, treasure)
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
        copied = Minimax2(
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
        treasure_score = agent.treasures * 10
        trap_score = -agent.traps * 5

        total_score = distance_score + treasure_score + trap_score

        return total_score, cells_expanded
    
    def get_utility_value(self, agent_index):
        return self.get_utility_value_expansions(agent_index)[0]

    def search(self, limit=5, prune=False, max_iterations=10000):
        root = Minimax2.Node(self.copy())
        curr = root

        for _ in range(max_iterations - 1):
            if not curr.state.treasures:
                break
            curr.alpha_beta_minimax(limit, prune)
            curr = curr.get_next_node()
            print(curr.state, end='\n\n')

        def get_expansions(root):
            total = 0
            for child in root.children:
                total += get_expansions(child)

            return total + root.debug["expansions"]

        return curr, get_expansions(root)

    def search_increment(self, state=None, *, limit=5, prune=False, move=None):
        state = state or {
            "curr": Minimax2.Node(self.copy()),
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
        rows = ["  " * depth + row for row in str(node.state).split("\n")]
        rows[0] += f" {round(node.value, 2)}"
        text = "\n".join(rows)

        print(text, end="\n\n")
        node.children = sorted(
            node.children, key=lambda child: child.value, reverse=depth % 2
        )
        for child in node.children:
            self.print_tree(child, depth + 1)


if __name__ == "__main__":
    grid = np.array(
        [
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    minimax = Minimax2(grid, (3, 8), (3, len(grid[0]) - 1))
    print(minimax, end='\n\n')
    node, _ = minimax.search(prune=True)