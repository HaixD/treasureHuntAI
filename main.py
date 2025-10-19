import tkinter as tk
import numpy as np
import random

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

        # Grid colors
        self.COLORS = {
            self.EMPTY: "white",
            self.WALL: "gold",
            self.TREASURE: "pink",
            self.TRAP: "sky blue",
            self.START: "light green"
        }

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
        self.canvas.pack()

        self.regenerate_button = tk.Button(
            self.root,
            text="Regenerate",
            command=self.regenerate_grid
        )
        self.regenerate_button.pack()

        # Draw grid
        self.grid, start_pos = self.create_grid()

        self.DFS_path = self.DFS(start_pos)
        
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

        return grid, start_pos

    def draw_grid(self):
        self.canvas.delete("all")
        for r in range(self.size):
            for c in range(self.size):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                value = self.grid[r, c]
                color = self.COLORS[value]

                # draw path
                if (r, c) in self.DFS_path:
                    color = "red"

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
        self.draw_grid()

    def run(self):
        self.root.mainloop()


    def expand(self, row, col):
        if row + 1 in range(0, self.size): 
            yield (row + 1, col)
        if row - 1 in range(0, self.size): 
            yield (row - 1, col)
        if col + 1 in range(0, self.size): 
            yield (row, col + 1)
        if col - 1 in range(0, self.size): 
            yield (row, col - 1)

    def DFS(self, position, path=None):
        path = path or set()
        r, c = position

        if self.grid[r, c] == self.WALL:       # invalid position
            return
        elif position in path:                 # already visited (avoid loops)
            return
        elif self.grid[r, c] == self.TREASURE: # found treasure
            path.add(position)
            return path.copy()
        
        path.add(position)
        for cell in self.expand(r, c):
            result = self.DFS(cell, path)
            if result is not None:
                return result
        path.remove(position)

if __name__ == "__main__":
    app = GridApp(size=10, wall_total=10)
    app.run()
