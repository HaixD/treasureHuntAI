import numpy as np
from os import mkdir, remove
from os.path import exists
from random import randrange
from pickle import dump, load

class Grid:
    # Grid codes
    EMPTY = 0
    WALL = 1
    TREASURE = 2
    TRAP = 3
    TRAP_TRIGGERED = 4
    START = 5
    PATH = 6
    
    def __init__(self, grid_size=None, treasure_total=None, trap_total=None, wall_total=None, *, grid=None, start_pos=None, treasure_pos=None):
        self.grid_size = grid_size
        self.treasure_total = treasure_total
        self.trap_total = trap_total
        self.wall_total = wall_total
        
        self.grid = grid if grid is not None else np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.start_pos = start_pos
        self.treasure_pos = treasure_pos

    def generate(self):
        # Place treasures
        treasure_count = 0
        while treasure_count < self.treasure_total:
            treasure_pos = (randrange(self.grid_size), randrange(self.grid_size))
            if self.grid[treasure_pos] == Grid.EMPTY:
                self.grid[treasure_pos] = Grid.TREASURE
                treasure_count += 1

        # Place traps
        trap_count = 0
        while trap_count < self.trap_total:
            trap_pos = (randrange(self.grid_size), randrange(self.grid_size))
            if self.grid[trap_pos] == Grid.EMPTY:
                self.grid[trap_pos] = Grid.TRAP
                trap_count += 1

        # Place walls
        wall_count = 0
        while wall_count < self.wall_total:
            r, c = randrange(self.grid_size), randrange(self.grid_size)
            if self.grid[r, c] == Grid.EMPTY:
                self.grid[r, c] = Grid.WALL
                wall_count += 1

        # Place start
        start_pos = (randrange(self.grid_size), randrange(self.grid_size))
        while start_pos in [treasure_pos, trap_pos]:
            start_pos = (randrange(self.grid_size), randrange(self.grid_size))
        self.grid[start_pos] = self.START

        self.start_pos = start_pos
        self.treasure_pos = treasure_pos

    def save(self, filename):
        try:            
            with open(f'test grids/{filename}.pickle', 'wb') as file:
                dump(self, file)
        except FileNotFoundError:
            mkdir('test grids')
            self.save(filename)

    @staticmethod
    def load_file(filename):
        with open(f'test grids/{filename}.pickle', 'rb') as file:
            return load(file)

    @staticmethod
    def load_grid(grid):
        grid_size = len(grid)
        treasure_total = 0
        trap_total = 0
        wall_total = 0
        start_pos = None
        treasure_pos = None
        
        for r, row in enumerate(grid):
            for c, col in enumerate(row):
                match col:
                    case Grid.START:
                        start_pos = (r, c)
                    case Grid.TREASURE:
                        treasure_pos = (r, c)
                        treasure_total += 1
                    case Grid.TRAP | Grid.TRAP_TRIGGERED:
                        grid[r, c] = Grid.TRAP
                        trap_total += 1
                    case Grid.WALL:
                        wall_total += 1
                    case Grid.PATH:
                        grid[r, c] = Grid.EMPTY

        print(grid)

        return Grid(grid_size, treasure_total, trap_total, wall_total,
                    grid=grid, start_pos=start_pos, treasure_pos=treasure_pos)

def test():
    # load grid through np array and compare original to loaded grid
    grid1 = Grid.load_grid(np.array([
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [6, 6, 6, 6, 6, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [0, 0, 0, 0, 0, 0, 0, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 6, 6],
       [0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [6, 6, 6, 6, 6, 6, 6, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    grid1.save('test1')
    grid2 = Grid.load_file('test1')

    if np.all(grid2.grid != grid1.grid):
        raise Exception('Grid save and load mismatch')
    
    # generate new grid and save
    
    grid3 = Grid(20, 3, 3, 10)
    grid3.generate()
    grid3.save('test2')

    # delete generated files
    if exists('test grids/test1.pickle'): # delete test1.pickle
        remove('test grids/test1.pickle')
    if exists('test grids/test2.pickle'): # delete test2.pickle
        remove('test grids/test2.pickle')

    print('test complete')

if __name__ == '__main__':
    # test()
    pass