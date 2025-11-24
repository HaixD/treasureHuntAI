from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from algorithms import a_star
from constants import Cell
from utils import euclidean_distance, get_neighbors


@dataclass
class Agent:
    position: tuple[float, float]
    treasures: int = 0
    traps: int = 0


class Minimax:
    class Node:
        def __init__(self, adversarial_state, *, parent=None, is_partial=False):
            self.state = adversarial_state
            self.value = (
                adversarial_state.get_utility_value() if not is_partial else None
            )
            self.parent = parent
            self.children = []
            self.depth = 0 if parent is None else parent.depth + 1
            self.expanded = False
            self.debug = {"expansions": 0}

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
            grid, agents = (
                self.state.grid,
                self.state.agents,
            )

            valid_moves = get_neighbors(
                grid, agents[agentIndex].position, include_traps=True
            )
            for move in valid_moves:
                child = Minimax.Node(move, parent=self, is_partial=True)
                self.children.append(child)

            self.expanded = True

        def build_node(self, build_value=False):
            if not self.is_partial():
                return

            move, agentIndex = self.state, self.get_agent_index()

            new_state = deepcopy(self.parent.state)
            new_state.apply_move(move, agentIndex)

            self.state = new_state

            if build_value:
                self.value, expansions = self.state.get_utility_value_expansions()
                self.debug["expansions"] = expansions

        def alpha_beta_minimax(self, limit, prune=False):
            def dfs(node, alpha, beta):
                if (node.expanded and node.is_leaf()) or (
                    node.depth == self.depth + limit - 1
                ):
                    node.build_node(True)
                    return node.value

                node.build_node()
                node.shallow_expand()

                agentIndex = node.get_agent_index(True)

                if not prune:
                    for child in node.children:
                        v = dfs(child, alpha, beta)
                        if agentIndex == 0:
                            node.value = node.value or -float("inf")
                            node.value = max(node.value, v)
                        else:
                            node.value = node.value or float("inf")
                            node.value = min(node.value, v)
                    return node.value

                children = node.children.copy()
                node.children = []
                if agentIndex == 0:
                    for child in children:
                        v = dfs(child, alpha, beta)
                        node.children.append(child)
                        alpha = max(alpha, v)
                        if alpha >= beta:
                            break
                    node.value = alpha
                    return alpha
                else:
                    for child in children:
                        v = dfs(child, alpha, beta)
                        node.children.append(child)
                        beta = max(alpha, v)
                        if beta <= alpha:
                            break
                    node.value = beta
                    return beta

            dfs(self, -float("inf"), float("inf"))

        def get_next_node(self):
            agentIndex = self.get_agent_index(True)

            if agentIndex == 0:
                next_node = max(self.children, key=lambda node: node.value)
            else:
                next_node = min(self.children, key=lambda node: node.value)

            next_node.expanded = False
            next_node.children = []

            return next_node

    def __init__(self, grid, start_pos1, start_pos2):
        self.grid = grid
        self.max_treasure_distance = (
            sum(grid.shape) * 2
        )  # this is the max distance possible between a point and treasure
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
        text = text.replace("0", " ").replace("8", "A").replace("9", "B")

        return text

    def __deepcopy__(self, memo):
        copied = Minimax(
            np.zeros((0, 0), dtype=int),
            self.agents[0].position,
            self.agents[1].position,
        )
        copied.grid = deepcopy(self.grid)
        copied.max_treasure_distance = self.max_treasure_distance
        copied.agents[0].treasures = self.agents[0].treasures
        copied.agents[1].treasures = self.agents[1].treasures
        copied.treasures = set(treasure for treasure in self.treasures)
        copied.paths = deepcopy(self.paths)

        return copied

    def apply_move(self, move, agentIndex):
        if move in self.treasures:
            self.grid[move] = Cell.EMPTY
            self.treasures.remove(move)
            self.agents[agentIndex].treasures += 1
        elif self.grid[move] == Cell.TRAP:
            self.agents[agentIndex].traps += 1

        self.agents[agentIndex].position = move
        self.paths[agentIndex].append(move)

    def get_utility_value_expansions(self):
        MAX, MIN = self.agents

        if len(self.treasures) == 0:
            if MAX.treasures > MIN.treasures:
                return float("inf"), 0
            elif MAX.treasures < MIN.treasures:
                return -float("inf"), 0
            return 0, 0

        total_cells_expanded = 0

        max_cloest_treasure = float("inf")
        for treasure in self.treasures:
            path, cells_expanded = a_star(
                self.grid, MAX.position, treasure, euclidean_distance
            )
            total_cells_expanded += cells_expanded
            max_cloest_treasure = min(max_cloest_treasure, len(path))

        min_cloest_treasure = float("inf")
        for treasure in self.treasures:
            path, cells_expanded = a_star(
                self.grid, MIN.position, treasure, euclidean_distance
            )
            total_cells_expanded += cells_expanded
            min_cloest_treasure = min(min_cloest_treasure, len(path))

        closest_treasure_score = (self.max_treasure_distance - max_cloest_treasure) - (
            self.max_treasure_distance - min_cloest_treasure
        )
        traps_difference_score = (
            (MAX.traps - MIN.traps) * self.max_treasure_distance * 0.5
        )
        owned_treasure_score = (
            MAX.treasures - MIN.treasures
        ) * self.max_treasure_distance

        score = closest_treasure_score + traps_difference_score + owned_treasure_score

        return score, cells_expanded

    def get_utility_value(self):
        return self.get_utility_value_expansions()[0]

    def search(self, limit=5, prune=False, max_iterations=10000):
        root = Minimax.Node(deepcopy(self))
        curr = root

        for _ in range(max_iterations - 1):
            if not curr.state.treasures:
                break
            curr.alpha_beta_minimax(limit, prune)
            curr = curr.get_next_node()
        else:
            curr.alpha_beta_minimax(limit, prune)

        def get_expansions(root):
            total = 0
            for child in root.children:
                total += get_expansions(child)

            return total + root.debug["expansions"]

        return curr, get_expansions(root)

    def search_increment(self, state=None, *, limit=5, prune=False, move=None):
        state = state or {
            "curr": Minimax.Node(deepcopy(self)),
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
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2],
        ],
        dtype=int,
    )
    minimax = Minimax(grid, (2, 2), (3, 4))
    # node, expansions = adversarial.search(limit=2, prune=False)
    # def print_tree(node, depth=0):
    #     rows = ['      ' * depth + row for row in str(node.state).split('\n')]
    #     rows[0] += f' {round(node.value, 2)}'
    #     text = '\n'.join(rows)

    #     print(text, end='\n\n')
    #     node.children = sorted(node.children, key=lambda child : child.value, reverse=depth % 2)
    #     for child in node.children:
    #         print_tree(child, depth + 1)

    # root = node
    # while root.parent is not None:
    #     root = root.parent

    # # print_tree(root)
    state, moves = minimax.search_increment()
    print(state["curr"].state)
    while moves:
        state, moves = minimax.search_increment(state, move=0)
        print(state["curr"].state)
