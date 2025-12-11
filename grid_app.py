"""Treasure Hunt AI visualization application.

This module provides a GUI-based grid environment for visualizing and comparing various search
algorithms, including uninformed search, informed search, adversarial search, and Bayesian search,
in a treasure hunt scenario with traps and walls.
"""

import copy
import random
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from algorithms import bfs, dfs, ucs, greedy, a_star, Minimax, BeliefGrid
from constants import Cell, PATH_GRADIENT_END, PATH_GRADIENT_START
from utils import (
    euclidean_distance,
    generate_gradient_colors,
    get_closest_point,
    get_moves,
    get_random_seed,
    manhattan_distance,
)


class GridApp:
    """GUI application for visualizing pathfinding algorithms in a treasure hunt game.

    This class creates an interactive grid-based environment where various search algorithms can be
    visualized finding paths to treasures while avoiding traps (only going through when capable and
    necessary) and walls. It supports both single-agent and adversarial multi-agent scenarios.

    Attributes:
        grid_size (int): The size of the square grid (default 20x20).
        treasure_total (int): Number of treasures to place on the grid.
        trap_total (int): Number of traps to place on the grid.
        wall_total (int): Number of walls to place on the grid.
        first_start_pos (tuple): Starting position for the first agent.
        second_start_pos (tuple): Starting position for the second agent (adversarial mode).
        treasure_pos (list): List of treasure positions on the grid.
        cell_size (int): Size of each grid cell in pixels.
        seed (int): Random seed for grid generation.
        animation_speed (int): Milliseconds between animation steps.
        is_animating (bool): Flag indicating if an animation is in progress.
        minimax_depth (int): Search depth limit for minimax algorithms.
    """

    def __init__(
        self,
        set_prompt=None,
    ):
        """Initialize the GridApp with specified parameters.

        Args:
            grid_size (int, optional): Size of the square grid. Defaults to 20.
            treasure_total (int, optional): Number of treasures. Defaults to 2.
            trap_total (int, optional): Number of traps. Defaults to 4.
            wall_total (int, optional): Number of walls. Defaults to 15.
            set_prompt (int, optional): Predefined scenario number (1 or 2).
                Defaults to None for random generation.
        """
        match set_prompt:
            case 1:
                self.set_grid_size = 25
                self.set_first_start = None
                self.set_second_start = None
                self.set_treasure_total = 3
                self.set_treasure_pos = []
                self.set_trap_total = 4
                self.set_trap_pos = []
                self.set_trap_range = 0
                self.set_wall_total = 20
            case 2:
                self.set_grid_size = 30
                self.set_first_start = None
                self.set_second_start = None
                self.set_treasure_total = 4
                self.set_treasure_pos = []
                self.set_trap_total = 16
                self.set_trap_pos = []
                self.set_trap_range = 1
                self.set_wall_total = 20
            case 3:
                self.set_grid_size = 20
                self.set_first_start = None
                self.set_second_start = None
                self.set_treasure_total = 3
                self.set_treasure_pos = []
                self.set_trap_total = 18
                self.set_trap_pos = []
                self.set_trap_range = 2
                self.set_wall_total = 20
            case 4:
                self.set_grid_size = 30
                self.set_first_start = None
                self.set_second_start = None
                self.set_treasure_total = 3
                self.set_treasure_pos = []
                self.set_trap_total = 0
                self.set_trap_pos = []
                self.set_trap_range = 0
                self.set_wall_total = 20
            case _:
                self.set_grid_size = 20
                self.set_first_start = None
                self.set_second_start = None
                self.set_treasure_total = 2
                self.set_treasure_pos = []
                self.set_trap_total = 4
                self.set_trap_pos = []
                self.set_trap_range = 0
                self.set_wall_total = 15

        self.grid_size = self.set_grid_size
        self.treasure_total = self.set_treasure_total
        self.trap_total = self.set_trap_total
        self.wall_total = self.set_wall_total
        self.first_start_pos = None
        self.second_start_pos = None
        self.treasure_pos = None

        # Inversely scale cell size based on grid size
        self.total_grid_pixels = 500
        self.cell_size = self.total_grid_pixels // self.grid_size

        # Seed for random maze
        self.seed = get_random_seed()
        self.rand = random.Random(self.seed)

        # Gradient path colors
        self.path_colors = []

        # Animation settings
        self.animation_speed = 25  # milliseconds between steps
        self.is_animating = False

        # Minimax depth setting
        self.minimax_depth = 3  # Default depth

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Treasure Hunt AI")

        self.ttk = ttk

        self.canvas = tk.Canvas(
            self.root,
            width=self.grid_size * self.cell_size,
            height=self.grid_size * self.cell_size,
        )
        self.canvas.pack(padx=10, pady=10)

        # Stats display
        self.stats_label = tk.Label(
            self.root,
            text="Run a search algorithm to see statistics",
            font=("Arial", 12, "bold"),
            justify=tk.CENTER,
            bg="white",
            fg="#333333",
            padx=15,
            pady=10,
            relief=tk.SOLID,
            borderwidth=1,
        )
        self.stats_label.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Frame for search algorithm buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        # Label that spans both rows, vertically centered
        run_label = tk.Label(button_frame, text="Run:", font=("Arial", 11))
        run_label.grid(row=0, column=0, rowspan=2, padx=10, sticky="ns")

        # Row 1: uninformed search
        tk.Button(
            button_frame, text="BFS", command=lambda: self.run_search_algorithm("BFS")
        ).grid(row=0, column=1, padx=5, pady=3)
        tk.Button(
            button_frame, text="DFS", command=lambda: self.run_search_algorithm("DFS")
        ).grid(row=0, column=2, padx=5, pady=3)
        tk.Button(
            button_frame, text="UCS", command=lambda: self.run_search_algorithm("UCS")
        ).grid(row=0, column=3, padx=5, pady=3)

        # Row 2: informed search
        tk.Button(
            button_frame,
            text="Greedy",
            command=lambda: self.run_search_algorithm("Greedy"),
        ).grid(row=1, column=1, padx=5, pady=3)
        tk.Button(
            button_frame,
            text="A* (Manhattan)",
            command=lambda: self.run_search_algorithm("A* (Manhattan)"),
        ).grid(row=1, column=2, padx=5, pady=3)
        tk.Button(
            button_frame,
            text="A* (Euclidean)",
            command=lambda: self.run_search_algorithm("A* (Euclidean)"),
        ).grid(row=1, column=3, padx=5, pady=3)

        # Row 3: adversarial search
        tk.Button(
            button_frame,
            text="Minimax",
            command=lambda: self.run_search_algorithm("Minimax"),
        ).grid(row=2, column=1, padx=5, pady=3)
        tk.Button(
            button_frame,
            text="Alpha-Beta",
            command=lambda: self.run_search_algorithm("Alpha-Beta"),
        ).grid(row=2, column=2, padx=5, pady=3)

        # Row 4: Bayesian search
        tk.Button(
            button_frame,
            text="Bayes (Low Noise)",
            command=lambda: self.run_bayesian_agent("Low"),
        ).grid(row=3, column=1, padx=5, pady=3)
        tk.Button(
            button_frame,
            text="Bayes (Med Noise)",
            command=lambda: self.run_bayesian_agent("Med"),
        ).grid(row=3, column=2, padx=5, pady=3)
        tk.Button(
            button_frame,
            text="Bayes (High Noise)",
            command=lambda: self.run_bayesian_agent("High"),
        ).grid(row=3, column=3, padx=5, pady=3)

        # Frame for Minimax depth slider
        depth_frame = tk.Frame(self.root)
        depth_frame.pack(pady=(0, 10))

        tk.Label(depth_frame, text="Depth:", font=("Arial", 11)).pack(
            side=tk.LEFT, padx=5
        )

        self.depth_label = tk.Label(
            depth_frame, text=str(self.minimax_depth), font=("Arial", 11, "bold")
        )
        self.depth_label.pack(side=tk.LEFT, padx=5)

        self.depth_slider = ttk.Scale(
            depth_frame,
            from_=2,
            to=5,
            orient=tk.HORIZONTAL,
            command=self.update_depth,
        )
        self.depth_slider.set(self.minimax_depth)
        self.depth_slider.pack(side=tk.LEFT, padx=5)

        # Frame for getting maze seed
        cur_seed_frame = tk.Frame(self.root)
        cur_seed_frame.pack(pady=(0, 10))

        self.cur_seed_label = tk.Label(
            cur_seed_frame, text=f"Current Seed: {self.seed}", font=("Arial", 11)
        )
        self.cur_seed_label.pack(side=tk.LEFT, padx=5)
        tk.Button(cur_seed_frame, text="Copy", command=self.copy_seed).pack(
            side=tk.LEFT, padx=5
        )

        # Frame for setting maze seed
        set_seed_frame = tk.Frame(self.root)
        set_seed_frame.pack(pady=(0, 10))

        def validate_seed_input(text):
            """Validate that seed input is a valid 32-bit unsigned integer."""
            return text.isdigit() and int(text) < 2**32 and int(text) >= 0 or text == ""

        vcmd = (self.root.register(validate_seed_input), "%P")

        self.set_seed_label = tk.Label(
            set_seed_frame, text="Set Seed:", font=("Arial", 11)
        )
        self.set_seed_label.pack(side=tk.LEFT, padx=5)
        self.set_seed_entry = tk.Entry(
            set_seed_frame, width=15, validate="key", validatecommand=vcmd
        )
        self.set_seed_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(
            set_seed_frame,
            text="Set",
            command=lambda: self.set_seed(self.get_seed_entry()),
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(set_seed_frame, text="Random", command=self.set_seed).pack(
            side=tk.LEFT, padx=5
        )

        # Draw grid
        self.grid = self.create_grid()
        self.draw_grid()

    def update_depth(self, value):
        """Update the minimax search depth when slider changes.

        Args:
            value (str): The new depth value from the slider.
        """
        self.minimax_depth = int(float(value))
        self.depth_label.config(text=str(self.minimax_depth))

    def get_seed_entry(self):
        """Get the seed value from the entry field.

        Returns:
            int: The entered seed value, or current seed if entry is empty.
        """
        value = self.set_seed_entry.get().strip()
        return int(value) if value != "" else self.seed

    def copy_seed(self):
        """Copy the current seed to the system clipboard."""
        self.root.clipboard_clear()
        self.root.clipboard_append(str(self.seed))

    def set_seed(self, new_seed=None):
        """Set the maze generation seed and regenerate the grid.

        Args:
            new_seed (int, optional): The seed to use. If None, generates a random seed.
        """
        self.seed = new_seed if new_seed is not None else get_random_seed()
        self.regenerate_grid()

    def create_grid(self):
        """Create a new grid with treasures, traps, walls, and starting position.

        Returns:
            np.ndarray: A 2D numpy array representing the grid with cell types.
        """
        self.rand = random.Random(self.seed)

        # Create empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Place treasures
        self.treasure_pos = []
        treasure_count = 0
        while treasure_count < self.treasure_total:
            if self.set_treasure_pos is not None and treasure_count < len(
                self.set_treasure_pos
            ):
                cur_treasure_pos = self.set_treasure_pos[treasure_count]
                treasure_pos = (cur_treasure_pos[0], cur_treasure_pos[1])
            else:
                treasure_pos = (
                    self.rand.randrange(self.grid_size),
                    self.rand.randrange(self.grid_size),
                )

            if grid[treasure_pos] == Cell.EMPTY:
                grid[treasure_pos] = Cell.TREASURE
                self.treasure_pos.append(treasure_pos)
                treasure_count += 1

        # Place traps
        trap_count = 0
        max_attempts_per_trap = 100
        while trap_count < self.trap_total:
            if self.set_trap_pos is not None and trap_count < len(self.set_trap_pos):
                cur_trap_pos = self.set_trap_pos[trap_count]
                trap_pos = (cur_trap_pos[0], cur_trap_pos[1])
            else:
                # Try to place near a treasure
                placed = False
                for attempt in range(max_attempts_per_trap):
                    random_treasure_pos = random.choice(self.treasure_pos)

                    # Expand range if failing to place
                    trap_range = (
                        self.set_trap_range
                        if self.set_trap_range > 0
                        else self.grid_size
                    )
                    if attempt >= max_attempts_per_trap // 2:
                        trap_range += 1

                    trap_pos = (
                        self.rand.randrange(
                            max(random_treasure_pos[0] - trap_range, 0),
                            min(
                                random_treasure_pos[0] + trap_range + 1, self.grid_size
                            ),
                        ),
                        self.rand.randrange(
                            max(random_treasure_pos[1] - trap_range, 0),
                            min(
                                random_treasure_pos[1] + trap_range + 1, self.grid_size
                            ),
                        ),
                    )

                    if grid[trap_pos] == Cell.EMPTY:
                        placed = True
                        break

                if not placed:
                    # Skip placing trap
                    print(
                        f"WARNING: Could only place {trap_count} of {self.trap_total} traps"
                    )
                    break

            if grid[trap_pos] == Cell.EMPTY:
                grid[trap_pos] = Cell.TRAP
                trap_count += 1

        # Place walls
        wall_count = 0
        while wall_count < self.wall_total:
            r, c = self.rand.randrange(self.grid_size), self.rand.randrange(
                self.grid_size
            )

            if grid[r, c] == Cell.EMPTY:
                grid[r, c] = Cell.WALL
                wall_count += 1

        # Place start
        if self.set_first_start is not None and isinstance(self.set_first_start, tuple):
            first_start_pos = self.set_first_start
        else:
            while True:
                first_start_pos = (
                    self.rand.randrange(self.grid_size),
                    self.rand.randrange(self.grid_size),
                )

                if first_start_pos not in [Cell.TREASURE, Cell.TRAP]:
                    break

        grid[first_start_pos] = Cell.START
        self.first_start_pos = first_start_pos

        return grid

    def regenerate_grid(self):
        """Regenerate the grid with the current seed and reset the display."""
        if self.is_animating:
            return

        self.grid = self.create_grid()
        self.path_colors = []
        self.stats_label.config(text="Run a search algorithm to see statistics")
        self.draw_grid()
        self.cur_seed_label.config(text=f"Current Seed: {self.seed}")

    def draw_grid(self, *, callback=lambda: None):
        """Draw the current grid state on the canvas.

        Args:
            callback (callable, optional): Function to call after drawing. Defaults to no-op.
        """
        self.canvas.delete("all")

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                value = Cell(self.grid[r, c])

                # Use stored path color if available
                if value == Cell.PATH and (r, c) in self.path_colors:
                    color = self.path_colors[(r, c)]
                else:
                    color = value.color

                # Draw background
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="black"
                )

                # Add symbol
                if value.symbol:
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=value.symbol,
                        font=("Arial", int(self.cell_size / 2), "bold"),
                        fill="black",
                    )

        callback()

    def run_bfs(self):
        """Execute and visualize the Breadth-First Search algorithm."""
        if self.is_animating:
            return

        self.clear_path()
        start_time = time.time()
        result = bfs(self.grid, self.first_start_pos)
        end_time = time.time()

        if result[0] is None:
            self.stats_label.config(text="BFS: No path found!")
            return

        path, cells_expanded = result
        execution_time = (end_time - start_time) * 1000

        # Animate solution path
        self.animate_path(path, cells_expanded, execution_time, "BFS")

    def run_dfs(self):
        """Execute and visualize the Depth-First Search algorithm.

        Tries all possible move orders and returns the best path found.
        """
        if self.is_animating:
            return

        self.clear_path()

        start_time = time.time()

        min_result = (float("inf"), None)
        for move_order in get_moves():  # try all moves with include_traps=False
            result = dfs(
                self.grid,
                self.first_start_pos,
                move_order=move_order,
                include_traps=False,
            )

            if result[0] and -self.get_path_score(result[0]) < min_result[0]:
                min_result = (len(result[0]), result)

        for move_order in get_moves():  # try all moves with include_traps=True
            result = dfs(
                self.grid,
                self.first_start_pos,
                move_order=move_order,
                include_traps=True,
            )

            if result[0] and -self.get_path_score(result[0]) < min_result[0]:
                min_result = (len(result[0]), result)

        end_time = time.time()

        # relay best result to user
        if min_result[1] is None:
            self.stats_label.config(text="DFS: No path found!")
            return

        path, cells_expanded = min_result[1]
        execution_time = (end_time - start_time) * 1000

        # Animate solution path
        self.animate_path(path, cells_expanded, execution_time, "DFS")

    def run_minimax(self, use_pruning):
        """Execute and visualize the Minimax algorithm with optional alpha-beta pruning.

        Args:
            use_pruning (bool): Whether to use alpha-beta pruning optimization.
        """
        if self.is_animating:
            return

        self.clear_path()

        if self.set_second_start is not None and isinstance(
            self.set_second_start, tuple
        ):
            self.second_start_pos = self.set_second_start
        else:
            while True:
                if self.second_start_pos is not None and self.second_start_pos not in [
                    Cell.TREASURE,
                    Cell.TRAP,
                ]:
                    break
                self.second_start_pos = (
                    self.rand.randrange(self.grid_size),
                    self.rand.randrange(self.grid_size),
                )

        self.grid[self.first_start_pos] = Cell.START_FIRST
        self.grid[self.second_start_pos] = Cell.START_SECOND
        self.draw_grid()

        start_time = time.time()

        minimax = Minimax(self.grid, self.first_start_pos, self.second_start_pos)
        node, expansions, pruning_ratio = minimax.search(
            limit=self.minimax_depth, prune=use_pruning, max_iterations=100
        )

        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        agents = node.state.agents
        paths = node.state.paths

        # Determine winner by highest treasure count; if not, lowest path cost
        if agents[0].treasures != agents[1].treasures:
            winner = agents.index(max(agents, key=lambda a: a.treasures))
        elif len(paths[0]) != len(paths[1]):
            winner = paths.index(min(paths, key=len))
        else:
            winner = [0, 1]

        self.animate_adversarial_path(
            winner,
            paths[0],
            paths[1],
            expansions,
            execution_time,
            "Minimax",
            pruning_ratio=pruning_ratio,
        )

        root = node
        while root.parent is not None:
            root = root.parent
        # Minimax.print_tree(minimax, root)

    def run_search_algorithm(self, algorithm):
        """Execute and visualize the specified search algorithm.

        Args:
            algorithm (str): Name of the algorithm to run (e.g., "BFS", "A* (Manhattan)").
        """
        if self.is_animating:
            return

        if algorithm.lower() == "minimax":
            self.run_minimax(use_pruning=False)
            return

        if algorithm.lower() == "alpha-beta":
            self.run_minimax(use_pruning=True)
            return

        self.second_start_pos = None

        if algorithm.lower() == "bfs":
            self.run_bfs()
            return

        if algorithm.lower() == "dfs":
            self.run_dfs()
            return

        self.clear_path()

        cur_pos = self.first_start_pos
        treasure_pos = copy.deepcopy(self.treasure_pos)
        treasure_count = len(self.treasure_pos)

        path = []
        cells_expanded = 0
        start_time = time.time()
        while treasure_count > 0:
            closest_treasure_pos = get_closest_point(
                cur_pos, treasure_pos, manhattan_distance
            )
            match algorithm.lower():
                case "ucs":
                    result = ucs(self.grid, cur_pos, closest_treasure_pos)
                case "greedy":
                    result = greedy(
                        self.grid,
                        cur_pos,
                        closest_treasure_pos,
                        manhattan_distance,
                    )
                case "a* (manhattan)":
                    result = a_star(
                        self.grid,
                        cur_pos,
                        closest_treasure_pos,
                        manhattan_distance,
                    )
                case "a* (euclidean)":
                    result = a_star(
                        self.grid,
                        cur_pos,
                        closest_treasure_pos,
                        euclidean_distance,
                    )
            path += result[0]
            cells_expanded += result[1]
            cur_pos = closest_treasure_pos
            treasure_pos.remove(closest_treasure_pos)
            treasure_count -= 1
        end_time = time.time()

        if path is None:
            self.stats_label.config(text=f"{algorithm}: No path found!")
            return

        execution_time = (end_time - start_time) * 1000

        # Animate solution path
        if algorithm.lower() in ["bfs", "dfs", "ucs"]:
            self.animate_path(path, cells_expanded, execution_time, algorithm)
        else:
            path_costs, path_positions = zip(*path)
            self.animate_path(
                path_positions,
                cells_expanded,
                execution_time,
                algorithm,
                path_costs=path_costs,
            )

    def run_bayesian_agent(self, noise_level="Low"):
        """Execute and visualize the Bayesian agent with specified noise level.

        Args:
            noise_level (str, optional): Sensor noise level ("Low", "Medium", or "High").
                Defaults to "Low".
        """
        if self.is_animating:
            return

        self.clear_path()
        match noise_level.lower():
            case "low":
                fp, fn = 0.05, 0.05
            case "medium":
                fp, fn = 0.1, 0.2
            case "high":
                fp, fn = 0.2, 0.3
            case _:
                fp, fn = 0.1, 0.1

        print(f"\n{'='*20} Starting Bayesian Search ({noise_level}) {'='*20}")

        def print_belief_grid(bg_obj, title):
            print(f"\n--- {title} ---")
            header = "      " + " ".join([f"{c:^5}" for c in range(self.grid_size)])
            print(header)

            for r in range(self.grid_size):
                row_str = f"{r:^4} |"
                for c in range(self.grid_size):
                    val = bg_obj.beliefs[r][c]
                    if val is None:
                        val = bg_obj.prior

                    # [ ] for high probability (> 50%)
                    #  *  for zero (found/impossible)
                    if val > 0.5:
                        row_str += f"[{val:.2f}]"
                    elif val == 0.0:
                        row_str += "  *  "
                    elif val < 0.001:
                        row_str += " ... "
                    else:
                        row_str += f" {val:.3f} "
                print(row_str)
            print("-" * len(header))

        def handle_found_treasure(bg_obj, r, c):
            # Zero out the found cell
            bg_obj.beliefs[r][c] = 0.0

            total_prob = sum(sum(row) for row in bg_obj.beliefs)

            if total_prob > 0:
                for row_idx in range(self.grid_size):
                    for col_idx in range(self.grid_size):
                        bg_obj.beliefs[row_idx][col_idx] /= total_prob

            # Clear the 'popped' history so the agent can re-evaluate nearby cells for the SECOND
            # treasure
            bg_obj.popped.clear()

        bg = BeliefGrid(self.grid, self.rand, false_positive=fp, false_negative=fn)
        curr_pos = self.first_start_pos
        path_history = []
        scans = 0

        treasures_found = 0
        total_treasures = self.treasure_total

        print_belief_grid(bg, "Belief Map at t=0 (Uniform Prior)")

        start_time = time.time()
        max_steps = self.grid_size * self.grid_size * 3

        while treasures_found < total_treasures and len(path_history) < max_steps:
            if (
                self.grid[curr_pos] == Cell.TREASURE
                or self.grid[curr_pos] == Cell.TREASURE_COLLECTED
            ) and bg.beliefs[curr_pos[0]][curr_pos[1]] > 0.0:
                treasures_found += 1
                print_belief_grid(
                    bg,
                    f"Belief Map at Detection #{treasures_found} (Location: {curr_pos})",
                )
                handle_found_treasure(bg, curr_pos[0], curr_pos[1])

                if treasures_found == total_treasures:
                    print("All treasures found!")
                    break

            bg.scan(curr_pos)
            scans += 1

            if scans == 10:
                print_belief_grid(bg, "Belief Map after 10 scans")

            try:
                target = bg.pop(position=curr_pos, error=False)
            except (IndexError, TypeError):
                print("Bayesian Agent: No more targets found.")
                break

            if target is None:
                break

            path_result = a_star(self.grid, curr_pos, target, manhattan_distance)
            path_segment = path_result[0]

            if path_segment is None:
                bg.beliefs[target[0]][target[1]] = 0.0
                continue

            if (
                len(path_segment) > 0
                and isinstance(path_segment[0], tuple)
                and len(path_segment[0]) == 2
            ):
                _, positions = zip(*path_segment)
            else:
                positions = path_segment

            if len(path_history) > 0:
                path_history.extend(positions[1:])
            else:
                path_history.extend(positions)

            curr_pos = target

        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        final_entropy = bg.get_entropy()

        print(f"\nResults for {noise_level} Noise:")
        print(f"  - Scans: {scans}")
        print(f"  - Moves: {len(path_history)}")
        print(f"  - Final Entropy: {final_entropy:.4f}")

        print("\nfalse positive/negative table:", *bg.false_table, sep="\n")

        self.animate_path(path_history, scans, execution_time, f"Bayes ({noise_level})")

    def get_path_score(self, path):
        """Calculate the score for a given path.

        Args:
            path (list): List of (row, col) positions in the path.

        Returns:
            float: Path score (higher is better). +10 per treasure, -5 per trap, -0.5 per step.
        """
        cost = -len(path) / 2  # -0.5 pts per length

        for r, c in path:
            if self.grid[r, c] == Cell.TRAP or self.grid[r, c] == Cell.TRAP_TRIGGERED:
                cost -= 5  # -5 pts per trap
            elif self.grid[r, c] == Cell.TREASURE:
                cost += 10

        return cost

    def clear_path(self):
        """Clear all path markings from the grid and reset to initial state."""
        grid = self.grid
        grid[grid == Cell.START_SECOND] = Cell.EMPTY
        grid[grid == Cell.PATH] = Cell.EMPTY
        grid[grid == Cell.PATH_FIRST] = Cell.EMPTY
        grid[grid == Cell.PATH_SECOND] = Cell.EMPTY
        grid[grid == Cell.PATH_BOTH] = Cell.EMPTY
        grid[grid == Cell.TREASURE_COLLECTED] = Cell.TREASURE
        grid[grid == Cell.TRAP_TRIGGERED] = Cell.TRAP

        self.draw_grid()

    def animate_path(
        self, path, cells_expanded, execution_time, algorithm_name, *, path_costs=None
    ):
        """Animate the solution path cell by cell with gradient coloring.

        Args:
            path (list): List of (row, col) positions in the path.
            cells_expanded (int): Number of cells expanded during search.
            execution_time (float): Algorithm execution time in milliseconds.
            algorithm_name (str): Name of the algorithm for display.
            path_costs (list, optional): List of costs for each position. Defaults to None.
        """
        self.is_animating = True
        self.path_colors = {}

        # Calculate gradient colors for each path cell
        gradient_colors = generate_gradient_colors(
            len(path), PATH_GRADIENT_START, PATH_GRADIENT_END
        )
        for i, pos in enumerate(path):
            self.path_colors[pos] = gradient_colors[i]

        def animate_costs():
            """Display path costs on the grid cells."""
            if path_costs is None:
                return

            drawn = set()
            for i, position in zip(range(len(path) - 1, -1, -1), path[::-1]):
                r, c = position
                if position in drawn:
                    continue
                elif self.grid[r, c] != Cell.PATH:
                    continue

                drawn.add(position)
                path_cost = path_costs[i]

                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=str(round(path_cost)),
                    font=("Arial", int(self.cell_size / 2), "bold"),
                    fill="black",
                )

        # Animate step by step
        expected_draw_time = time.time()
        for i, (r, c) in enumerate(path):
            match self.grid[r, c]:
                case Cell.EMPTY:
                    self.grid[r, c] = Cell.PATH
                case Cell.TREASURE:
                    self.grid[r, c] = Cell.TREASURE_COLLECTED
                case Cell.TRAP:
                    self.grid[r, c] = Cell.TRAP_TRIGGERED
                case _:
                    pass

            expected_draw_time += self.animation_speed / 1000
            if expected_draw_time > time.time():  # too fast:
                time.sleep(expected_draw_time - time.time())

                self.draw_grid(callback=animate_costs)
                self.root.update()

        self.draw_grid(callback=animate_costs)
        self.root.update()

        # Animation complete
        self.is_animating = False
        path_cost = len(path) - 1
        stats_text = (
            f"{algorithm_name} Results:\nPath Cost: {path_cost} | "
            + f"Cells Expanded: {cells_expanded} | Time: {execution_time:.3f} ms"
        )
        self.stats_label.config(text=stats_text)

    def animate_adversarial_path(
        self,
        winner,
        path1,
        path2,
        total_cells_expanded,
        total_execution_time,
        algorithm_name,
        *,
        path1_costs=None,
        path2_costs=None,
        pruning_ratio=None,
    ):
        """Animate two competing paths in adversarial search.

        Args:
            winner (int or list): Index of winning agent (0 or 1) or [0, 1] for tie.
            path1 (list): First agent's path.
            path2 (list): Second agent's path.
            total_cells_expanded (int): Total cells expanded by both agents.
            total_execution_time (float): Total execution time in milliseconds.
            algorithm_name (str): Name of the algorithm for display.
            path1_costs (list, optional): Costs for first path. Defaults to None.
            path2_costs (list, optional): Costs for second path. Defaults to None.
            pruning_ratio (float, optional): Ratio of pruned nodes. Defaults to None.
        """
        self.is_animating = True
        self.path_colors = {}

        def animate_costs():
            """Display path costs for both agents on the grid cells."""
            if path1_costs is None:
                return

            drawn = set()
            # Draw costs for first path
            for i, position in zip(range(len(path1) - 1, -1, -1), path1[::-1]):
                r, c = position
                if position in drawn:
                    continue
                elif self.grid[r, c] != Cell.PATH:
                    continue

                drawn.add(position)
                path_cost = path1_costs[i]

                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    text=str(round(path_cost)),
                    font=("Arial", int(self.cell_size / 2), "bold"),
                    fill="black",
                )

            # Draw costs for second path
            if path2 is not None and path2_costs is not None:
                for i, position in zip(range(len(path2) - 1, -1, -1), path2[::-1]):
                    r, c = position
                    if position in drawn:
                        continue
                    elif self.grid[r, c] != Cell.PATH:
                        continue

                    drawn.add(position)
                    path_cost = path2_costs[i]

                    x1, y1 = c * self.cell_size, r * self.cell_size
                    x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=str(round(path_cost)),
                        font=("Arial", int(self.cell_size / 2), "bold"),
                        fill="white",  # Different color for second path
                    )

        # Track which cells have been visited by each path
        path1_cells = set()
        path2_cells = set()

        # Animate step by step, alternating between paths
        expected_draw_time = time.time()
        max_length = max(len(path1), len(path2) if path2 is not None else 0)

        for i in range(max_length):
            # Animate first path step
            if i < len(path1):
                r, c = path1[i]
                path1_cells.add((r, c))

                match self.grid[r, c]:
                    case Cell.EMPTY:
                        self.grid[r, c] = Cell.PATH_FIRST
                    case Cell.PATH_SECOND:
                        self.grid[r, c] = Cell.PATH_BOTH
                    case Cell.TREASURE:
                        self.grid[r, c] = Cell.TREASURE_COLLECTED
                    case Cell.TRAP:
                        self.grid[r, c] = Cell.TRAP_TRIGGERED
                    case _:
                        pass

                expected_draw_time += self.animation_speed / 1000
                if expected_draw_time > time.time():
                    time.sleep(expected_draw_time - time.time())

                self.draw_grid(callback=animate_costs)
                self.root.update()

            # Animate second path step
            if path2 is not None and i < len(path2):
                r, c = path2[i]
                path2_cells.add((r, c))

                match self.grid[r, c]:
                    case Cell.EMPTY:
                        self.grid[r, c] = Cell.PATH_SECOND
                    case Cell.PATH_FIRST:
                        self.grid[r, c] = Cell.PATH_BOTH
                    case Cell.TREASURE:
                        self.grid[r, c] = Cell.TREASURE_COLLECTED
                    case Cell.TRAP:
                        self.grid[r, c] = Cell.TRAP_TRIGGERED
                    case _:
                        pass

                expected_draw_time += self.animation_speed / 1000
                if expected_draw_time > time.time():
                    time.sleep(expected_draw_time - time.time())

                self.draw_grid(callback=animate_costs)
                self.root.update()

        self.draw_grid(callback=animate_costs)
        self.root.update()

        # Animation complete
        self.is_animating = False

        if winner == 0:
            winner_text = "A"
        elif winner == 1:
            winner_text = "B"
        else:
            winner_text = "Tie"
        path1_cost = len(path1) - 1
        path2_cost = len(path2) - 1

        stats_text = (
            f"{algorithm_name} Results:\nWinner: {winner_text} | Path A Cost: {path1_cost} | "
            + f"Path B Cost: {path2_cost}\n"
            + f"Total Cells Expanded: {total_cells_expanded} | "
            + f"Total Time: {total_execution_time:.3f} ms"
        )
        if pruning_ratio is not None:
            stats_text += f"\nPruning Ratio: {pruning_ratio:.2f}"

        self.stats_label.config(text=stats_text)

    def visualize_belief(self, bg):
        """Visualize the belief probabilities on the grid.

        Args:
            bg (BeliefGrid): The belief grid containing probability distributions.
        """
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Skip walls/static items if you want, or just draw over them
                if self.grid[r, c] == Cell.WALL:
                    continue

                prob = bg.beliefs[r][c]
                if prob is None:
                    prob = bg.prior

                # Draw a circle with opacity based on probability by scaling color from white (0%)
                # to red (100%)
                intensity = int(255 * (1 - prob))  # 1.0 -> 0 (Dark), 0.0 -> 255 (Light)
                color = f"#ff{intensity:02x}{intensity:02x}"

                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size

                # Draw small indicator in corner
                self.canvas.create_oval(
                    x1 + 2, y1 + 2, x1 + 10, y1 + 10, fill=color, outline=""
                )
        self.root.update()

    def run(self):
        """Start the GUI main event loop."""
        self.root.mainloop()
