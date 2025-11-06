import copy
import random
import tkinter as tk
import heapq
import time
import sys
from itertools import permutations
from collections import deque
import numpy as np

class GridApp:
    def __init__(self, grid_size=20, treasure_total=10, trap_total=2, wall_total=30):
        self.grid_size = grid_size
        self.treasure_total = treasure_total
        self.trap_total = trap_total
        self.wall_total = wall_total
        self.treasure_total = treasure_total

        # Inversely scale cell size based on grid size
        self.total_grid_pixels = 500
        self.cell_size = self.total_grid_pixels // self.grid_size

        # Seed for random maze
        self.seed = random.randrange(sys.maxsize)

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

        tk.Button(button_frame, text="Run BFS", command=self.run_bfs).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run DFS", command=self.run_dfs).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run UCS", command=self.run_ucs).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run Greedy", command=self.run_greedy).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run A*", command=self.run_a_star).pack(side=tk.LEFT, padx=5)

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
        self.set_seed_entry = tk.Entry(set_seed_frame, width=20)
        self.set_seed_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(set_seed_frame, text="Set", command=lambda: self.set_seed(int(self.set_seed_entry.get()))).pack(side=tk.LEFT, padx=5)
        tk.Button(set_seed_frame, text="Random", command=self.set_seed).pack(side=tk.LEFT, padx=5)

        # Draw grid
        self.grid = self.create_grid()
        self.draw_grid()

    def get_path_score(self, path):
        cost = -len(path) / 2 # -0.5 pts per length
        
        for r, c in path:
            if self.grid[r, c] == self.TRAP or self.grid[r, c] == self.TRAP_TRIGGERED: # -5 pts per trap
                cost -= 5
            elif self.grid[r, c] == self.TREASURE:
                cost += 10

        return cost

    def copy_seed(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(str(self.seed))

    def set_seed(self, new_seed=None):
        if new_seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = new_seed
        self.regenerate_grid()

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
                    trap_pos = (rand.randrange(max(treasure_pos[0] - 2, 0), min(treasure_pos[0] + 2, self.grid_size)),
                                rand.randrange(max(treasure_pos[1] - 2, 0), min(treasure_pos[1] + 2, self.grid_size)))
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

        # make get neighbors point in the direction of the target first
        if abs(treasure_pos[1] - start_pos[1]) > abs(treasure_pos[0] - start_pos[0]): # prioritize horizontal
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
                    path.append(current_pos)
                    current_pos = parent[current_pos]
                return path[::-1], cells_expanded

            # Otherwise, explore neighbors and get their costs
            for neighbor in self.get_neighbors(current_pos, include_traps=True):
                new_cost = self.manhattan_distance(neighbor, goal)
                if self.grid[neighbor[0], neighbor[1]] == self.TRAP:
                    new_cost += 4

                # Update if this is a better path to neighbor
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = current_pos
                    heapq.heappush(pq, (new_cost, neighbor))

        return None, cells_expanded

    def run_greedy(self):
        if self.is_animating:
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
            result = self.greedy(cur_pos, closest_treasure_pos)
            path += result[0]
            cells_expanded += result[1]
            cur_pos = closest_treasure_pos
            treasure_pos.remove(closest_treasure_pos)
            treasure_count -= 1
        end_time = time.time()

        if path is None:
            self.stats_label.config(text="greedy: No path found!")
            return

        execution_time = (end_time - start_time) * 1000

        # Animate solution path
        self.animate_path(path, cells_expanded, execution_time, "greedt")


    def a_star(self, start, goal):
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
                #Reconstruct path from start to goal
                path = []
                while current_pos is not None:
                    path.append(current_pos)
                    current_pos = parent[current_pos]
                return path[::-1], cells_expanded

            # Otherwise, explore neighbors and get their costs
            for neighbor in self.get_neighbors(current_pos, include_traps=True):
                new_cost = self.manhattan_distance(neighbor, goal) + current_cost + 1
                if self.grid[neighbor[0], neighbor[1]] == self.TRAP:
                    new_cost += 4

                # Update if this is a better path to neighbor
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = current_pos
                    heapq.heappush(pq, (new_cost, neighbor))

        return None, cells_expanded



    def run_a_star(self):
        if self.is_animating:
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
            result = self.a_star(cur_pos, closest_treasure_pos)
            path += result[0]
            cells_expanded += result[1]
            cur_pos = closest_treasure_pos
            treasure_pos.remove(closest_treasure_pos)
            treasure_count -= 1
        end_time = time.time()

        if path is None:
            self.stats_label.config(text="astar: No path found!")
            return

        execution_time = (end_time - start_time) * 1000

        # Animate solution path
        self.animate_path(path, cells_expanded, execution_time, "a_star")

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
        print(self.manhattan_distance(self.treasure_pos, self.start_pos))
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

    def run_ucs(self):
        if self.is_animating:
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
            result = self.ucs(cur_pos, closest_treasure_pos)
            path += result[0]
            cells_expanded += result[1]
            cur_pos = closest_treasure_pos
            treasure_pos.remove(closest_treasure_pos)
            treasure_count -= 1
        end_time = time.time()

        if path is None:
            self.stats_label.config(text="UCS: No path found!")
            return

        execution_time = (end_time - start_time) * 1000

        # Animate solution path
        self.animate_path(path, cells_expanded, execution_time, "UCS")

    def clear_path(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                match self.grid[r, c]:
                    case self.PATH:
                        self.grid[r, c] = self.EMPTY
                    case self.TREASURE_COLLECTED:
                        self.grid[r, c] = self.TREASURE
                    case self.TRAP_TRIGGERED:
                        self.grid[r, c] = self.TRAP

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
    def animate_path(self, path, cells_expanded, execution_time, algorithm_name):
        """Animate the path drawing with gradient colors"""
        self.is_animating = True
        self.path_colors = {}

        # Calculate gradient colors for each path cell
        gradient_colors = self.generate_gradient_colors(len(path))
        for i, pos in enumerate(path):
            self.path_colors[pos] = gradient_colors[i]

        # Animate step by step
        expected_draw_time = time.time()
        for i, (r, c) in enumerate(path):
            match self.grid[r, c]:
                case self.EMPTY:
                    self.grid[r, c] = self.PATH
                case self.TRAP:
                    self.grid[r, c] = self.TRAP_TRIGGERED
                case _:
                    pass

            expected_draw_time += self.animation_speed / 1000
            if expected_draw_time > time.time(): # too fast:
                time.sleep(expected_draw_time - time.time())

                self.draw_grid()
                self.root.update()

        self.draw_grid()
        self.root.update()

        # Animation complete
        self.is_animating = False
        path_cost = len(path) - 1
        stats_text = f"{algorithm_name} Results:\nPath Cost: {path_cost} | Cells Expanded: {cells_expanded} | Time: {execution_time:.3f} ms"
        self.stats_label.config(text=stats_text)

    def draw_grid(self):
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

    def regenerate_grid(self):
        if self.is_animating:
            return

        self.grid = self.create_grid()
        self.path_colors = []
        self.stats_label.config(text="Run a search algorith m to see statistics")
        self.draw_grid()
        self.cur_seed_label.config(text=f"Current Seed: {self.seed}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GridApp()
    app.run()
