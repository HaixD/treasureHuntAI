import copy
import random
import tkinter as tk
import heapq
import time
from collections import deque
from itertools import permutations
import numpy as np

class GridApp:
    def __init__(self, grid_size=10, treasure_total=2, trap_total=2, wall_total=5):
        self.grid_size = grid_size
        self.treasure_total = treasure_total
        self.trap_total = trap_total
        self.wall_total = wall_total
        self.treasure_total = treasure_total

        # Inversely scale cell size based on grid size
        self.total_grid_pixels = 500
        self.cell_size = self.total_grid_pixels // self.grid_size

        # Seed for random maze
        self.seed = self.get_random32()

        # Grid codes
        self.EMPTY = 0
        self.WALL = 1
        self.TREASURE = 2
        self.TREASURE_COLLECTED = 3
        self.TRAP = 4
        self.TRAP_TRIGGERED = 5
        self.START = 6
        self.PATH = 7

        # Grid colors
        self.COLORS = {
            self.EMPTY: "white",
            self.WALL: "gold",
            self.TREASURE: "pink",
            self.TREASURE_COLLECTED: "hot pink",
            self.TRAP: "sky blue",
            self.TRAP_TRIGGERED: "royal blue",
            self.START: "light green"
        }

        # Gradient path colors
        self.path_colors = []

        # Grid symbols
        self.SYMBOLS = {
            self.WALL: "#",
            self.TREASURE: "T",
            self.TREASURE_COLLECTED: "+",
            self.TRAP: "X",
            self.TRAP_TRIGGERED: "!",
            self.START: "S"
        }

        # Animation settings
        self.animation_speed = 25  # milliseconds between steps
        self.is_animating = False

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Treasure Hunt AI")

        self.canvas = tk.Canvas(
            self.root,
            width=self.grid_size * self.cell_size,
            height=self.grid_size * self.cell_size
        )
        self.canvas.pack(padx=10, pady=10)

        # Stats display
        self.stats_label = tk.Label(
            self.root,
            text="Run a search algorithm to see statistics",
            font=("Arial", 11, "bold"),
            justify=tk.CENTER,
            bg="white",
            fg="#333333",
            padx=15,
            pady=10,
            relief=tk.SOLID,
            borderwidth=1
        )
        self.stats_label.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Frame for search algorithm buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        # Label that spans both rows, vertically centered
        run_label = tk.Label(button_frame, text="Run:", font=("Arial", 11))
        run_label.grid(row=0, column=0, rowspan=2, padx=10, sticky="ns")

        # Top row: non-heuristics algorithms
        tk.Button(button_frame, text="BFS", command=lambda: self.run_search("BFS")).grid(row=0, column=1, padx=5, pady=3)
        tk.Button(button_frame, text="DFS", command=lambda: self.run_search("DFS")).grid(row=0, column=2, padx=5, pady=3)
        tk.Button(button_frame, text="UCS", command=lambda: self.run_search("UCS")).grid(row=0, column=3, padx=5, pady=3)

        # Bottom row: heuristic-based algorithms
        tk.Button(button_frame, text="Greedy", command=lambda: self.run_search("Greedy")).grid(row=1, column=1, padx=5, pady=3)
        tk.Button(button_frame, text="A* (Manhattan)", command=lambda: self.run_search("A* (Manhattan)")).grid(row=1, column=2, padx=5, pady=3)
        tk.Button(button_frame, text="A* (Euclidean)", command=lambda: self.run_search("A* (Euclidean)")).grid(row=1, column=3, padx=5, pady=3)

        # Frame for getting maze seed
        cur_seed_frame = tk.Frame(self.root)
        cur_seed_frame.pack(pady=(0, 10))

        self.cur_seed_label = tk.Label(cur_seed_frame, text=f"Current Seed: {self.seed}", font=("Arial", 11))
        self.cur_seed_label.pack(side=tk.LEFT, padx=5)
        tk.Button(cur_seed_frame, text="Copy", command=self.copy_seed).pack(side=tk.LEFT, padx=5)

        # Frame for setting maze seed
        set_seed_frame = tk.Frame(self.root)
        set_seed_frame.pack(pady=(0, 10))

        self.set_seed_label = tk.Label(set_seed_frame, text="Set Seed:", font=("Arial", 11))
        self.set_seed_label.pack(side=tk.LEFT, padx=5)
        self.set_seed_entry = tk.Entry(set_seed_frame, width=15)
        self.set_seed_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(set_seed_frame, text="Set", command=lambda: self.set_seed(self.get_seed_entry())).pack(side=tk.LEFT, padx=5)
        tk.Button(set_seed_frame, text="Random", command=self.set_seed).pack(side=tk.LEFT, padx=5)

        # Draw grid
        self.grid = self.create_grid()
        self.draw_grid()

    def get_seed_entry(self):
        value = self.set_seed_entry.get().strip()

        if value == "":
            return self.seed

        return int(value)

    def get_path_score(self, path):
        cost = -len(path) / 2 # -0.5 pts per length

        for r, c in path:
            if self.grid[r, c] == self.TRAP or self.grid[r, c] == self.TRAP_TRIGGERED:
                cost -= 5  # -5 pts per trap
            elif self.grid[r, c] == self.TREASURE:
                cost += 10

        return cost

    # Copies current seed to the clipboard
    def copy_seed(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(str(self.seed))

    # Sets maze seed based on a given seed or a random 32-bit seed otherwise
    def set_seed(self, new_seed=None):
        self.seed = new_seed if new_seed is not None else self.get_random32()
        self.regenerate_grid()

    # Returns a random positive 32-bit number
    def get_random32(self):
        return random.randint(0, 2**32 - 1)

    def create_grid(self):
        rand = random.Random(self.seed)

        # Create empty grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Place treasures
        self.treasure_pos = []
        treasure_count = 0
        while treasure_count < self.treasure_total:
            treasure_pos = (rand.randrange(self.grid_size), rand.randrange(self.grid_size))
            if grid[treasure_pos] == self.EMPTY:
                grid[treasure_pos] = self.TREASURE
                self.treasure_pos.append(treasure_pos)
                treasure_count += 1

        # Place traps
        trap_count = 0
        while trap_count < self.trap_total:
            trap_pos = (
                rand.randrange(
                    max(treasure_pos[0] - 2, 0), min(treasure_pos[0] + 2, self.grid_size)
                ),
                rand.randrange(
                    max(treasure_pos[1] - 2, 0), min(treasure_pos[1] + 2, self.grid_size)
                )
            )
            if grid[trap_pos] == self.EMPTY:
                grid[trap_pos] = self.TRAP
                trap_count += 1

        # Place walls
        wall_count = 0
        while wall_count < self.wall_total:
            r, c = rand.randrange(self.grid_size), rand.randrange(self.grid_size)
            if grid[r, c] == self.EMPTY:
                grid[r, c] = self.WALL
                wall_count += 1

        # Place start
        start_pos = (rand.randrange(self.grid_size), rand.randrange(self.grid_size))
        while start_pos in [treasure_pos, trap_pos]:
            start_pos = (rand.randrange(self.grid_size), rand.randrange(self.grid_size))
        grid[start_pos] = self.START
        self.start_pos = start_pos

        self.moves = []

        # make get neighbors point in the direction of the target first, prioritize horizontal
        if abs(treasure_pos[1] - start_pos[1]) > abs(treasure_pos[0] - start_pos[0]):
            if start_pos[1] < treasure_pos[1]: # go right first, left third
                self.moves.append(lambda r, c : (r, c + 1))
                self.moves.append(lambda r, c : (r, c - 1))
            else:
                self.moves.append(lambda r, c : (r, c - 1))
                self.moves.append(lambda r, c : (r, c + 1))

            if start_pos[0] < treasure_pos[0]: # go down second, up fourth
                self.moves.insert(1, lambda r, c : (r + 1, c))
                self.moves.append(lambda r, c : (r - 1, c))
            else:
                self.moves.insert(1, lambda r, c : (r - 1, c))
                self.moves.append(lambda r, c : (r + 1, c))
        else: # prioritize vertical
            if start_pos[0] < treasure_pos[0]: # go down second, up fourth
                self.moves.append(lambda r, c : (r + 1, c))
                self.moves.append(lambda r, c : (r - 1, c))
            else:
                self.moves.append(lambda r, c : (r - 1, c))
                self.moves.append(lambda r, c : (r + 1, c))

            if start_pos[1] < treasure_pos[1]:  # go right first, left third
                self.moves.insert(1, lambda r, c : (r, c + 1))
                self.moves.append(lambda r, c : (r, c - 1))
            else:
                self.moves.insert(1, lambda r, c : (r, c - 1))
                self.moves.append(lambda r, c : (r, c + 1))

        return grid

    @staticmethod
    def get_moves():
        return permutations([
            lambda r, c : (r, c + 1), # right
            lambda r, c : (r, c - 1), # left
            lambda r, c : (r - 1, c), # up
            lambda r, c : (r + 1, c)  # down
        ])

    # Return valid neighbors for a given position
    def get_neighbors(self, pos, moves=None, include_traps=False):
        r, c = pos
        valid_neighbors = []

        moves = moves or next(GridApp.get_moves())

        for move in moves:
            nr, nc = move(r, c)
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if (not include_traps and self.grid[nr, nc] not in [self.WALL, self.TRAP]
                        or include_traps and self.grid[nr, nc] not in [self.WALL]):
                    valid_neighbors.append((nr, nc))

        return valid_neighbors

    @staticmethod
    def euclidean_distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def manhattan_distance(p1, p2):
        if len(p1) != len(p2):
            raise ValueError("Points must have the same number of dimensions.")

        distance = 0
        for a, b in zip(p1, p2):
            distance += abs(a - b)

        return distance

    def get_closest_point(self, start, targets):
        if not isinstance(targets, list):
            return targets

        return min(targets, key=lambda cur: self.manhattan_distance(start, cur))

    def greedy(self, start, goal):
        pq = [(0, start)]           # Priority queue: (cost, position)
        visited = set()             # Record all fully explored cells so far
        parent = {start: None}      # Record best parents of each cell for path reconstruction
        cost = {start: 0}           # Record lowest costs to reach each cell
        heuristics = {start: self.manhattan_distance(start, goal), goal: 0}
        cells_expanded = 0

        while pq:
            # Get next best cost and position from priority queue
            _, current_pos = heapq.heappop(pq)

            # Skip current position if already visited
            if current_pos in visited:
                continue

            # Otherwise, record that current position has been visited
            visited.add(current_pos)
            cells_expanded += 1

            # If goal state reached
            if current_pos == goal:
                # Reconstruct path from start to goal
                path = []
                while current_pos is not None:
                    path.append((heuristics[current_pos], current_pos))
                    current_pos = parent[current_pos]
                return path[::-1], cells_expanded

            # Otherwise, explore neighbors and get their costs
            for neighbor in self.get_neighbors(current_pos, include_traps=True):
                heuristic = self.manhattan_distance(neighbor, goal)
                heuristics[neighbor] = heuristic

                new_cost = self.manhattan_distance(neighbor, goal)
                if self.grid[neighbor[0], neighbor[1]] == self.TRAP:
                    new_cost += 4

                # Update if this is a better path to neighbor
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = current_pos
                    heapq.heappush(pq, (new_cost, neighbor))

        return None, cells_expanded

    def a_star(self, start, goal, heuristic_func):
        # pq = [(0, start)]           # Priority queue: (cost, position)
        visited = set()             # Record all fully explored cells so far
        parent = {start: None}      # Record best parents of each cell for path reconstruction
        cost = {start: 0}           # Record lowest costs to reach each cell
        heuristics = {start: self.manhattan_distance(start, goal), goal: 0}
        cells_expanded = 0
        cost = {start: 0}
        h_start = self.manhattan_distance(start,goal)
        f_start = cost[start] + h_start
        pq = [(f_start, start)]
        while pq:
            # Get next best cost and position from priority queue
            _, current_pos = heapq.heappop(pq)

            # Skip current position if already visited
            if current_pos in visited:
                continue

            # Otherwise, record that current position has been visited
            visited.add(current_pos)
            cells_expanded += 1

            # If goal state reached
            if current_pos == goal:
                #Reconstruct path from start to goal
                path = []
                while current_pos is not None:
                    path.append((heuristics[current_pos], current_pos))
                    current_pos = parent[current_pos]
                return path[::-1], cells_expanded

            # Otherwise, explore neighbors and get their costs
            for neighbor in self.get_neighbors(current_pos, include_traps=True):
                step_cost = 1
                if self.grid[neighbor[0], neighbor[1]] == self.TRAP:
                    step_cost += 4
                new_cost = cost[current_pos] + step_cost
                # Update if this is a better path to neighbor
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = current_pos
                    h_cost = heuristic_func(neighbor, goal)
                    heuristics[neighbor] = h_cost
                    new_f_cost = new_cost + h_cost
                    heapq.heappush(pq, (new_f_cost, neighbor))

        return None, cells_expanded

    def bfs(self):
        queue = deque([[self.start_pos]])
        visited = {self.start_pos}
        cells_expanded = 0
        while queue:
            path = queue.popleft()
            position = path[-1]
            cells_expanded += 1

            if self.grid[position[0], position[1]] == self.TREASURE:
                return path, cells_expanded

            for neighbor in self.get_neighbors(position, include_traps= True):
                nr, nc = neighbor
                if neighbor not in visited and self.grid[nr, nc] != self.WALL:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return None, cells_expanded

    def run_bfs(self):
        if self.is_animating:
            return

        self.clear_path()
        # print(self.manhattan_distance(self.start_pos, self.get_closest_point(self.start_pos, self.treasure_pos)))
        start_time = time.time()
        result = self.bfs()
        end_time = time.time()

        if result[0] is None:
            self.stats_label.config(text="BFS: No path found!")
            return

        path, cells_expanded = result
        execution_time = (end_time - start_time) * 1000

        # Animate solution path
        self.animate_path(path, cells_expanded, execution_time, "BFS")

    def dfs(self, position=None, path=None, visited=None, cells_expanded=None, *, move_order=None, include_traps=True):
        position = position or self.start_pos
        path = path or []
        visited = visited or set()
        if cells_expanded is None:
            cells_expanded = [0]  # Use list to keep track across recursive calls

        r, c = position

        if self.grid[r, c] == self.WALL: # invalid position
            return None, cells_expanded[0]
        elif position in visited: # already visited (avoid loops)
            return None, cells_expanded[0]
        elif self.grid[r, c] == self.TREASURE: # found treasure
            cells_expanded[0] += 1
            return path + [position], cells_expanded[0]

        cells_expanded[0] += 1
        visited.add(position)
        for cell in self.get_neighbors(position, moves=move_order, include_traps=include_traps):
            result, _ = self.dfs(cell, path + [position], visited, cells_expanded, move_order=move_order, include_traps=include_traps)
            if result is not None:
                return result, cells_expanded[0]

        return None, cells_expanded[0]

    def run_dfs(self):
        if self.is_animating:
            return

        self.clear_path()

        start_time = time.time()

        min_result = (float('inf'), None)
        for move_order in GridApp.get_moves(): # try all moves with include_traps=False
            result = self.dfs(move_order=move_order, include_traps=False)

            if result[0] and -self.get_path_score(result[0]) < min_result[0]:
                min_result = (len(result[0]), result)


        for move_order in GridApp.get_moves(): # try all moves with include_traps=True
            result = self.dfs(move_order=move_order, include_traps=True)

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

    def ucs(self, start, goal):
        pq = [(0, start)]           # Priority queue: (cost, position)
        visited = set()             # Record all fully explored cells so far
        parent = {start: None}      # Record best parents of each cell for path reconstruction
        cost = {start: 0}           # Record lowest costs to reach each cell
        cells_expanded = 0

        while pq:
            # Get next best cost and position from priority queue
            current_cost, current_pos = heapq.heappop(pq)

            # Skip current position if already visited
            if current_pos in visited:
                continue

            # Otherwise, record that current position has been visited
            visited.add(current_pos)
            cells_expanded += 1

            # If goal state reached
            if current_pos == goal:
                # Reconstruct path from start to goal
                path = []
                while current_pos is not None:
                    path.append(current_pos)
                    current_pos = parent[current_pos]
                return path[::-1], cells_expanded

            # Otherwise, explore neighbors and get their costs
            for neighbor in self.get_neighbors(current_pos, include_traps=True):
                new_cost = current_cost + 1
                if self.grid[neighbor[0], neighbor[1]] == self.TRAP:
                    new_cost += 4

                # Update if this is a better path to neighbor
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = current_pos
                    heapq.heappush(pq, (new_cost, neighbor))

        return None, cells_expanded

    def run_search(self, algorithm="UCS"):
        if self.is_animating:
            return

        if algorithm.lower() == "bfs":
            self.run_bfs()
            return

        if algorithm.lower() == "dfs":
            self.run_dfs()
            return

        self.clear_path()

        cur_pos = self.start_pos
        treasure_pos = copy.deepcopy(self.treasure_pos)
        treasure_count = len(self.treasure_pos)

        path = []
        cells_expanded = 0
        start_time = time.time()
        while treasure_count > 0:
            closest_treasure_pos = self.get_closest_point(cur_pos, treasure_pos)
            match algorithm.lower():
                case "ucs":
                    result = self.ucs(cur_pos, closest_treasure_pos)
                case "greedy":
                    result = self.greedy(cur_pos, closest_treasure_pos)
                case "a* (manhattan)":
                    result = self.a_star(cur_pos, closest_treasure_pos, self.manhattan_distance)
                case "a* (euclidean)":
                    result = self.a_star(cur_pos, closest_treasure_pos, self.euclidean_distance)
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
            self.animate_path(path_positions, cells_expanded, execution_time, algorithm, path_costs=path_costs)

    def clear_path(self):
        grid = self.grid
        grid[grid == self.PATH] = self.EMPTY
        grid[grid == self.TREASURE_COLLECTED] = self.TREASURE
        grid[grid == self.TRAP_TRIGGERED] = self.TRAP

        self.draw_grid()

    # Interpolate color for current cell of path between light green (for start) and pink (for treasure)
    def interpolate_path_color(self, ratio):
        # Start RGB:    (144, 238, 144)
        # Treasure RGB: (255, 192, 203)
        start_r, start_g, start_b = 144, 238, 144
        end_r, end_g, end_b = 255, 192, 203
        # end_r, end_g, end_b = 238, 114, 178

        r = int(start_r + (end_r - start_r) * ratio)
        g = int(start_g + (end_g - start_g) * ratio)
        b = int(start_b + (end_b - start_b) * ratio)

        return f'#{r:02x}{g:02x}{b:02x}'

    # Generate gradient colors for the entire path
    def generate_gradient_colors(self, path_length):
        colors = []
        for i in range(path_length):
            ratio = i / max(path_length - 1, 1)
            colors.append(self.interpolate_path_color(ratio))
        return colors

    # Animate the path cell by cell
    def animate_path(self, path, cells_expanded, execution_time, algorithm_name, *, path_costs=None):
        self.is_animating = True
        self.path_colors = {}

        # Calculate gradient colors for each path cell
        gradient_colors = self.generate_gradient_colors(len(path))
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
                elif self.grid[r, c] != self.PATH:
                    continue

                drawn.add(position)
                path_cost = path_costs[i]

                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2,
                    text=str(round(path_cost)),
                    font=("Arial", int(self.cell_size / 2), "bold"),
                    fill="black"
                )

        # Animate step by step
        expected_draw_time = time.time()
        for i, (r, c) in enumerate(path):
            match self.grid[r, c]:
                case self.EMPTY:
                    self.grid[r, c] = self.PATH
                case self.TREASURE:
                    self.grid[r, c] = self.TREASURE_COLLECTED
                case self.TRAP:
                    self.grid[r, c] = self.TRAP_TRIGGERED
                case _:
                    pass

            expected_draw_time += self.animation_speed / 1000
            if expected_draw_time > time.time(): # too fast:
                time.sleep(expected_draw_time - time.time())

                self.draw_grid(callback=animate_costs)
                self.root.update()

        self.draw_grid(callback=animate_costs)
        self.root.update()

        # Animation complete
        self.is_animating = False
        path_cost = len(path) - 1
        stats_text = f"{algorithm_name} Results:\nPath Cost: {path_cost} | Cells Expanded: {cells_expanded} | Time: {execution_time:.3f} ms"
        self.stats_label.config(text=stats_text)

    def draw_grid(self, *, callback=lambda : None):
        self.canvas.delete("all")

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                value = self.grid[r, c]

                # Use stored path color if available
                if value == self.PATH and (r, c) in self.path_colors:
                    color = self.path_colors[(r, c)]
                else:
                    color = self.COLORS[value]

                # Draw background
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

                # Add symbol
                if value in self.SYMBOLS:
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=self.SYMBOLS[value],
                        font=("Arial", int(self.cell_size / 2), "bold"),
                        fill="black"
                    )

        callback()

    def regenerate_grid(self):
        if self.is_animating:
            return

        self.grid = self.create_grid()
        self.path_colors = []
        self.stats_label.config(text="Run a search algorithm to see statistics")
        self.draw_grid()
        self.cur_seed_label.config(text=f"Current Seed: {self.seed}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GridApp()
    app.run()
