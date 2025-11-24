import copy
import random
import time
import tkinter as tk
import numpy as np
from algorithms import bfs, dfs, ucs, greedy, a_star
from algorithms.minimax import Minimax
from constants import Cell, PATH_GRADIENT_START, PATH_GRADIENT_END
from utils import (
    euclidean_distance,
    manhattan_distance,
    get_closest_point,
    get_moves,
    get_random_seed,
    generate_gradient_colors,
)


class GridApp:
    def __init__(
        self,
        grid_size=20,
        treasure_total=2,
        trap_total=4,
        wall_total=15,
        set_prompt=None,
    ):
        self.grid_size = grid_size
        self.treasure_total = treasure_total
        self.trap_total = trap_total
        self.wall_total = wall_total
        self.treasure_total = treasure_total
        self.first_start_pos = None
        self.second_start_pos = None
        self.treasure_pos = None

        match set_prompt:
            case 1:
                self.set_start = (0, 0)
                self.set_treasures = [(19, 19)]
                self.set_traps = [(5, 5), (10, 14), (14, 7)]
            case 2:
                self.set_start = (10, 10)
                self.set_treasures = [(3, 17), (17, 3)]
                self.set_traps = [(3, 16), (4, 17), (16, 3), (17, 4)]
            case _:
                self.set_start = None
                self.set_treasures = None
                self.set_traps = None

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

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Treasure Hunt AI")

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

        # Top row: non-heuristics algorithms
        tk.Button(
            button_frame, text="BFS", command=lambda: self.run_search_algorithm("BFS")
        ).grid(row=0, column=1, padx=5, pady=3)
        tk.Button(
            button_frame, text="DFS", command=lambda: self.run_search_algorithm("DFS")
        ).grid(row=0, column=2, padx=5, pady=3)
        tk.Button(
            button_frame, text="UCS", command=lambda: self.run_search_algorithm("UCS")
        ).grid(row=0, column=3, padx=5, pady=3)

        # Bottom row: heuristic-based algorithms
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

        tk.Button(
            button_frame,
            text="Minimax",
            command=lambda: self.run_search_algorithm("Minimax"),
        ).grid(row=2, column=2, padx=5, pady=3)

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

    def get_seed_entry(self):
        value = self.set_seed_entry.get().strip()

        return int(value) if value != "" else self.seed

    # Copies current seed to the clipboard
    def copy_seed(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(str(self.seed))

    # Sets maze seed based on a given seed or a random 32-bit seed otherwise
    def set_seed(self, new_seed=None):
        self.seed = new_seed if new_seed is not None else get_random_seed()
        self.regenerate_grid()

    def create_grid(self):
        self.rand = random.Random(self.seed)

        # Create empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Place treasures
        self.treasure_pos = []
        treasure_count = 0
        while treasure_count < self.treasure_total:
            if self.set_treasures is not None and treasure_count < len(
                self.set_treasures
            ):
                cur_treasure_pos = self.set_treasures[treasure_count]
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
        while trap_count < self.trap_total:
            if self.set_traps is not None and trap_count < len(self.set_traps):
                cur_trap_pos = self.set_traps[trap_count]
                trap_pos = (cur_trap_pos[0], cur_trap_pos[1])
            else:
                trap_pos = (
                    self.rand.randrange(
                        max(treasure_pos[0] - 2, 0),
                        min(treasure_pos[0] + 2, self.grid_size),
                    ),
                    self.rand.randrange(
                        max(treasure_pos[1] - 2, 0),
                        min(treasure_pos[1] + 2, self.grid_size),
                    ),
                )
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
        if isinstance(self.set_start, tuple):
            start_pos = (self.set_start[0], self.set_start[1])
        else:
            while True:
                start_pos = (
                    self.rand.randrange(self.grid_size),
                    self.rand.randrange(self.grid_size),
                )
                if start_pos not in [Cell.TREASURE, Cell.TRAP]:
                    break
        grid[start_pos] = Cell.START
        self.first_start_pos = start_pos

        return grid

    def regenerate_grid(self):
        if self.is_animating:
            return

        self.grid = self.create_grid()
        self.path_colors = []
        self.stats_label.config(text="Run a search algorithm to see statistics")
        self.draw_grid()
        self.cur_seed_label.config(text=f"Current Seed: {self.seed}")

    def draw_grid(self, *, callback=lambda: None):
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

    def run_minimax(self):
        if self.is_animating:
            return

        self.clear_path()

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
        node, expansions = minimax.search(limit=2, prune=False)

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
            winner, paths[0], paths[1], expansions, execution_time, "Minimax"
        )

    def run_search_algorithm(self, algorithm):
        if self.is_animating:
            return

        if algorithm.lower() == "minimax":
            self.run_minimax()
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

    def get_path_score(self, path):
        cost = -len(path) / 2  # -0.5 pts per length

        for r, c in path:
            if self.grid[r, c] == Cell.TRAP or self.grid[r, c] == Cell.TRAP_TRIGGERED:
                cost -= 5  # -5 pts per trap
            elif self.grid[r, c] == Cell.TREASURE:
                cost += 10

        return cost

    def clear_path(self):
        grid = self.grid
        grid[grid == Cell.PATH] = Cell.EMPTY
        grid[grid == Cell.PATH_FIRST] = Cell.EMPTY
        grid[grid == Cell.PATH_SECOND] = Cell.EMPTY
        grid[grid == Cell.PATH_BOTH] = Cell.EMPTY
        grid[grid == Cell.TREASURE_COLLECTED] = Cell.TREASURE
        grid[grid == Cell.TRAP_TRIGGERED] = Cell.TRAP

        self.draw_grid()

    # Animate the path cell by cell
    def animate_path(
        self, path, cells_expanded, execution_time, algorithm_name, *, path_costs=None
    ):
        self.is_animating = True
        self.path_colors = {}

        # Calculate gradient colors for each path cell
        gradient_colors = generate_gradient_colors(
            len(path), PATH_GRADIENT_START, PATH_GRADIENT_END
        )
        for i, pos in enumerate(path):
            self.path_colors[pos] = gradient_colors[i]

        def animate_costs():
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
    ):
        self.is_animating = True
        self.path_colors = {}

        def animate_costs():
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
            f"{algorithm_name} Results:\nWinner: {winner_text} | Path A Cost: {path1_cost} | Path B Cost: {path2_cost}\n"
            + f"Total Cells Expanded: {total_cells_expanded} | Total Time: {total_execution_time:.3f} ms"
        )

        self.stats_label.config(text=stats_text)

    def run(self):
        self.root.mainloop()
