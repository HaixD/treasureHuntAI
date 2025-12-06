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
    """Represents an agent in the adversarial search game.

    Attributes:
        position (tuple[float, float]): Current position as (row, col).
        treasures (int): Number of treasures collected. Defaults to 0.
        traps (int): Number of traps triggered. Defaults to 0.
    """

    position: tuple[float, float]
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
        """Represents a node in the Minimax game tree.

        Each node contains a game state and can be expanded to generate child nodes representing
        possible moves in the game.

        Attributes:
            state: The game state (Minimax instance or move tuple).
            value (float): The minimax value of this node.
            parent (Node): Parent node in the tree.
            children (list[Node]): Child nodes representing possible moves.
            depth (int): Depth of this node in the tree.
            expanded (bool): Whether this node has been expanded.
            debug (dict): Debug information including expansion count.
        """

        def __init__(self, adversarial_state, *, parent=None, is_partial=False):
            """Initialize a Minimax tree node.

            Args:
                adversarial_state: Either a Minimax game state or a move tuple.
                parent (Node, optional): Parent node. Defaults to None.
                is_partial (bool, optional): Whether this is a partial node that needs
                    building. Defaults to False.
            """
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
            """Check if this is a partial node that needs building.

            Returns:
                bool: True if the node's value is None (needs building).
            """
            return self.value is None

        def is_leaf(self):
            """Check if this is a leaf node with no children.

            Returns:
                bool: True if the node has no children.
            """
            return len(self.children) == 0

        def get_agent_index(self, swap=False):
            """Get the index of the agent that moves at this node.

            Args:
                swap (bool, optional): Whether to swap agent turn. Defaults to False.

            Returns:
                int: 0 for maximizing agent, 1 for minimizing agent.
            """
            return (self.depth + swap) % 2 == 1

        def shallow_expand(self):
            """Create child nodes for all valid moves without building full states.

            This creates partial nodes that will be built later as needed, improving
            efficiency by avoiding unnecessary state copying.
            """
            if self.expanded:
                return

            agent_index = self.get_agent_index(True)
            grid, agents = (
                self.state.grid,
                self.state.agents,
            )

            valid_moves = get_neighbors(
                grid, agents[agent_index].position, include_traps=True
            )
            for move in valid_moves:
                child = Minimax.Node(move, parent=self, is_partial=True)
                self.children.append(child)

            self.expanded = True

        def build_node(self, build_value=False):
            """Build the full game state for this node if it's partial.

            Args:
                build_value (bool, optional): Whether to compute the utility value for this node.
                    Defaults to False.
            """
            if not self.is_partial():
                return

            move, agent_index = self.state, self.get_agent_index()

            new_state = deepcopy(self.parent.state)
            new_state.apply_move(move, agent_index)

            self.state = new_state

            if build_value:
                self.value, expansions = self.state.get_utility_value_expansions()
                self.debug["expansions"] = expansions

        def alpha_beta_minimax(self, limit, prune=False):
            """Run minimax search with optional alpha-beta pruning from this node.

            This method recursively explores the game tree up to a specified depth limit, computing
            minimax values for each node. With pruning enabled, it uses alpha-beta pruning to skip
            branches that cannot affect the result.

            Args:
                limit (int): Maximum depth to search from this node.
                prune (bool, optional): Whether to use alpha-beta pruning. Defaults to False.
            """

            def dfs(node, alpha, beta):
                if (node.expanded and node.is_leaf()) or (
                    node.depth == self.depth + limit - 1
                ):
                    node.build_node(True)
                    return node.value

                node.build_node()
                node.shallow_expand()

                agent_index = node.get_agent_index(True)

                if not prune:
                    for child in node.children:
                        v = dfs(child, alpha, beta)
                        if agent_index == 0:
                            node.value = node.value or -float("inf")
                            node.value = max(node.value, v)
                        else:
                            node.value = node.value or float("inf")
                            node.value = min(node.value, v)
                    return node.value

                children = node.children.copy()
                node.children = []
                if agent_index == 0:
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
            """Select the best child node based on minimax values.

            The maximizing agent selects the child with maximum value, while the minimizing agent
            selects the child with minimum value.

            Returns:
                Node: The best child node according to minimax values.
            """
            agent_index = self.get_agent_index(True)

            if agent_index == 0:
                next_node = max(self.children, key=lambda node: node.value)
            else:
                next_node = min(self.children, key=lambda node: node.value)

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
        """Return a string representation of the current game state.

        Returns:
            str: Grid visualization with 'A' for agent 1 and 'B' for agent 2.
        """
        grid = deepcopy(self.grid)
        grid[*self.agents[0].position] = 8
        grid[*self.agents[1].position] = 9

        text = str(grid)
        text = text.replace("0", " ").replace("8", "A").replace("9", "B")

        return text

    def __deepcopy__(self, memo):
        """Create a deep copy of this Minimax instance.

        Args:
            memo (dict): Memoization dictionary for copy.deepcopy.

        Returns:
            Minimax: A deep copy of this instance.
        """
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

    def apply_move(self, move, agent_index):
        """Apply a move for the specified agent, updating game state.

        Args:
            move (tuple): Target position as (row, col).
            agentIndex (int): Index of the agent making the move (0 or 1).
        """
        if move in self.treasures:
            self.grid[move] = Cell.EMPTY
            self.treasures.remove(move)
            self.agents[agent_index].treasures += 1
        elif self.grid[move] == Cell.TRAP:
            self.agents[agent_index].traps += 1

        self.agents[agent_index].position = move
        self.paths[agent_index].append(move)

    def get_utility_value_expansions(self):
        """Calculate the utility value of the current state and track expansions.

        The utility is computed based on:
        - Treasure ownership advantage
        - Distance to nearest treasure for each agent
        - Trap penalties

        Returns:
            tuple: A tuple containing:
                - score (float): The utility value from the maximizer's perspective.
                - cells_expanded (int): Total cells expanded during A* calls.
        """
        max_agent, min_agent = self.agents

        if len(self.treasures) == 0:
            if max_agent.treasures > min_agent.treasures:
                return float("inf"), 0

            if max_agent.treasures < min_agent.treasures:
                return -float("inf"), 0

            return 0, 0

        total_cells_expanded = 0

        max_cloest_treasure = float("inf")
        for treasure in self.treasures:
            path, cells_expanded = a_star(
                self.grid, max_agent.position, treasure, euclidean_distance
            )
            total_cells_expanded += cells_expanded
            max_cloest_treasure = min(max_cloest_treasure, len(path))

        min_cloest_treasure = float("inf")
        for treasure in self.treasures:
            path, cells_expanded = a_star(
                self.grid, min_agent.position, treasure, euclidean_distance
            )
            total_cells_expanded += cells_expanded
            min_cloest_treasure = min(min_cloest_treasure, len(path))

        closest_treasure_score = (self.max_treasure_distance - max_cloest_treasure) - (
            self.max_treasure_distance - min_cloest_treasure
        )
        traps_difference_score = (
            (max_agent.traps - min_agent.traps) * self.max_treasure_distance * 0.5
        )
        owned_treasure_score = (
            max_agent.treasures - min_agent.treasures
        ) * self.max_treasure_distance

        score = closest_treasure_score + traps_difference_score + owned_treasure_score

        return score, cells_expanded

    def get_utility_value(self):
        """Calculate the utility value of the current state.

        Returns:
            float: The utility value from the maximizer's perspective.
        """
        return self.get_utility_value_expansions()[0]

    def search(self, limit=5, prune=False, max_iterations=10000):
        """Perform iterative Minimax search until all treasures are collected.

        This method runs minimax search repeatedly, selecting the best move at each step until all
        treasures are collected or max iterations is reached.

        Args:
            limit (int, optional): Depth limit for each minimax search. Defaults to 5.
            prune (bool, optional): Whether to use alpha-beta pruning. Defaults to False.
            max_iterations (int, optional): Maximum search iterations. Defaults to 10000.

        Returns:
            tuple: A tuple containing:
                - curr (Node): The final node reached after all moves.
                - total_expansions (int): Total number of node expansions.
        """
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
        """Recursively print the minimax game tree.

        Args:
            node (Node): The node to print.
            depth (int, optional): Current depth in the tree. Defaults to 0.
        """
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
    test_grid = np.array(
        [
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2],
        ],
        dtype=int,
    )
    minimax = Minimax(test_grid, (2, 2), (3, 4))
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
    test_state, moves = minimax.search_increment()
    print(test_state["curr"].state)
    while moves:
        test_state, moves = minimax.search_increment(test_state, move=0)
        print(test_state["curr"].state)
