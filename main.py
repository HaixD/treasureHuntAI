import tkinter as tk
import numpy as np
import random
import heapq
import time
from collections import deque

class GridApp:
    def __init__(self, size=10, wall_total=10, cell_size=50):
        self.size = size
        self.wall_total = wall_total
        self.cell_size = cell_size

        # Grid codes
        self.EMPTY = 0
        self.WALL = 1
        self.TREASURE = 2
        self.TRAP = 3
        self.START = 4
        self.PATH = 5

        # Grid colors
        self.COLORS = {
            self.EMPTY: "white",
            self.WALL: "gold",
            self.TREASURE: "pink",
            self.TRAP: "sky blue",
            self.START: "light green"
        }

        # Gradient path colors
        self.path_colors = []

        # Grid symbols
        self.SYMBOLS = {
            self.WALL: "#",
            self.TREASURE: "T",
            self.TRAP: "X",
            self.START: "S"
        }

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("Treasure Hunt AI")

        self.canvas = tk.Canvas(
            self.root,
            width=self.size * self.cell_size,
            height=self.size * self.cell_size
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

        # Button frame for better organization
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        tk.Button(button_frame, text="Regenerate", command=self.regenerate_grid).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run BFS", command=self.run_bfs).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run DFS", command=self.run_dfs).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Run UCS", command=self.run_ucs).pack(side=tk.LEFT, padx=5)

        # Draw grid
        self.grid = self.create_grid()
        self.draw_grid()

    def create_grid(self):
        grid = np.zeros((self.size, self.size), dtype=int)

        # Place treasure
        treasure_placed = False
        while not treasure_placed:
            treasure_pos = (random.randrange(self.size), random.randrange(self.size))
            if grid[treasure_pos] == self.EMPTY:
                grid[treasure_pos] = self.TREASURE
                treasure_placed = True

        # Place trap
        trap_placed = False
        while not trap_placed:
            trap_pos = (random.randrange(self.size), random.randrange(self.size))
            if grid[trap_pos] == self.EMPTY:
                grid[trap_pos] = self.TRAP
                trap_placed = True

        # Place walls
        wall_count = 0
        while wall_count < self.wall_total:
            r, c = random.randrange(self.size), random.randrange(self.size)
            if grid[r, c] == self.EMPTY:
                grid[r, c] = self.WALL
                wall_count += 1

        # Place start
        start_pos = (random.randrange(self.size), random.randrange(self.size))
        while start_pos in [treasure_pos, trap_pos]:
            start_pos = (random.randrange(self.size), random.randrange(self.size))
        grid[start_pos] = self.START

        self.start_pos = start_pos
        self.treasure_pos = treasure_pos

        return grid

    # Return valid neighbors for a given position
    def get_neighbors(self, pos):
        r, c = pos
        moves = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        valid_neighbors = []
        for nr, nc in moves:
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.grid[nr, nc] not in [self.WALL, self.TRAP]:
                    valid_neighbors.append((nr, nc))
        return valid_neighbors

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

            for neighbor in self.get_neighbors(position):
                nr, nc = neighbor
                if neighbor not in visited and self.grid[nr, nc] != self.WALL:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return None, cells_expanded


    def run_bfs(self):
        self.clear_path()

        start_time = time.time()
        result = self.bfs()
        end_time = time.time()

        if result[0] is None:
            print("No path found!")
            self.stats_label.config(text="BFS: No path found!")
            return

        path, cells_expanded = result
        execution_time = (end_time - start_time) * 1000
        path_cost = len(path) - 1

        # Update stats display
        stats_text = f"BFS Results:\nPath Cost: {path_cost} | Cells Expanded: {cells_expanded} | Time: {execution_time:.3f} ms"
        self.stats_label.config(text=stats_text)

        # Generate gradient colors
        self.path_colors = self.generate_gradient_colors(len(path))

        for (r, c) in path:
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.PATH

        self.draw_grid(path)

    def dfs(self, position=None, path=None, visited=None, cells_expanded=None):
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
        for cell in self.get_neighbors(position):
            result, _ = self.dfs(cell, path + [position], visited, cells_expanded)
            if result is not None:
                return result, cells_expanded[0]

        return None, cells_expanded[0]

    def run_dfs(self):
        self.clear_path()

        start_time = time.time()
        result = self.dfs()
        end_time = time.time()

        if result[0] is None:
            self.stats_label.config(text="DFS: No path found!")
            return

        path, cells_expanded = result
        execution_time = (end_time - start_time) * 1000
        path_cost = len(path) - 1

        # Update stats display
        stats_text = f"DFS Results:\nPath Cost: {path_cost} | Cells Expanded: {cells_expanded} | Time: {execution_time:.3f} ms"
        self.stats_label.config(text=stats_text)

        # Generate gradient colors
        self.path_colors = self.generate_gradient_colors(len(path))

        for (r, c) in path:
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.PATH

        self.draw_grid(path)

    def ucs(self, start, goal):
        pq = [(0, start)]           # Priority queue: (cost, position)
        visited = set()             # Record all fully explored cells so far
        parent = {start: None}      # Record best parents of each cell for path reconstruction
        cost = {start: 0}           # Record lowest costs to reach each cell
        cells_expanded = 0

        while pq:
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

            # Explore neighbors and get their costs
            for neighbor in self.get_neighbors(current_pos):
                new_cost = current_cost + 1

                # Update if this is a better path to neighbor
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    parent[neighbor] = current_pos
                    heapq.heappush(pq, (new_cost, neighbor))

        return None, cells_expanded

    def run_ucs(self):
        self.clear_path()

        start_time = time.time()
        result = self.ucs(self.start_pos, self.treasure_pos)
        end_time = time.time()

        if result[0] is None:
            print("No path found!")
            self.stats_label.config(text="UCS: No path found!")
            return

        path, cells_expanded = result
        execution_time = (end_time - start_time) * 1000
        path_cost = len(path) - 1

        # Update stats display
        stats_text = f"UCS Results:\nPath Cost: {path_cost} | Cells Expanded: {cells_expanded} | Time: {execution_time:.3f} ms"
        self.stats_label.config(text=stats_text)

        # Generate gradient colors
        self.path_colors = self.generate_gradient_colors(len(path))

        for (r, c) in path:
            if self.grid[r, c] == self.EMPTY:
                self.grid[r, c] = self.PATH

        self.draw_grid(path)

    # Clear all path cells from the grid
    def clear_path(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == self.PATH:
                    self.grid[r, c] = self.EMPTY

    # Interpolate color for current cell of path between light green (for start) and pink (for treasure)
    def interpolate_path_color(self, ratio):
        # Start RGB:    (144, 238, 144)
        # Treasure RGB: (255, 192, 203)
        start_r, start_g, start_b = 144, 238, 144
        end_r, end_g, end_b = 255, 192, 203

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

    def draw_grid(self, path=None):
        self.canvas.delete("all")

        # Create a mapping of positions to path indices for gradient coloring
        path_index_map = {}
        if path is not None:
            for i, pos in enumerate(path):
                path_index_map[pos] = i

        for r in range(self.size):
            for c in range(self.size):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                value = self.grid[r, c]

                # Determine color
                if value == self.PATH and (r, c) in path_index_map:
                    color = self.path_colors[path_index_map[(r, c)]]
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
        self.grid = self.create_grid()
        self.path_colors = []
        self.stats_label.config(text="Run a search algorithm to see statistics")
        self.draw_grid()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GridApp(size=10, wall_total=10)
    app.run()