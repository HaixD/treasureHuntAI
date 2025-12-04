from math import log
from random import randint, seed
from enum import IntEnum

class Cell(IntEnum): # PLACEHOLDER
    EMPTY = 0
    WALL = 1
    TREASURE = 2
    TREASURE_COLLECTED = 3
    TRAP = 4
    TRAP_TRIGGERED = 5
    START = 6
    START_FIRST = 7
    START_SECOND = 8
    PATH = 9
    PATH_FIRST = 10
    PATH_SECOND = 11
    PATH_BOTH = 12

class BeliefGrid:
    def __init__(self, grid, false_positive, false_negative):
        self.grid = grid
        self.false_positive = false_positive
        self.false_negative = false_negative
        
        self.observations = [[[] for _ in range(len(grid[0]))] for _ in range(len(grid))]
        self.beliefs = [[None] * len(grid[0]) for _ in range(len(grid))]
        self.overrides = [[None] * len(grid[0]) for _ in range(len(grid))]
        self.popped = set()

        self.treasures = 1
        for row in grid:
            for col in row:
                self.treasures += col == Cell.TREASURE
        self.decrement_treasure()

    def override_belief(self, position, *, pop=True):
        r, c = position
        is_treasure = self.grid[r][c] == Cell.TREASURE
        
        self.overrides[r][c] = int(is_treasure)
        self.beliefs[r][c] = self.overrides[r][c]

        if is_treasure:
            self.decrement_treasure()

        if pop:
            self.popped.add(position)

    def decrement_treasure(self):
        self.treasures -= 1
        self.prior = self.treasures / (len(self.grid) * len(self.grid[0]))

        for r, row in enumerate(self.beliefs):
            for c, col in enumerate(row):
                if col is not None:
                    self.update_posterior((r, c))

    def scan_neighbors(self, position):
        r, c = position
        h, w = len(self.grid), len(self.grid[0])
       
        def try_scan(r, c):
            if 0 <= r < h and 0 <= c < w:
                self.scan((r, c))

        try_scan(r - 1, c)
        try_scan(r + 1, c)
        try_scan(r, c - 1)
        try_scan(r, c + 1)

    def scan(self, position):
        r, c = position
        if self.grid[r][c] == Cell.TREASURE:
            if randint(1, 100) / 100 <= self.false_negative:
                self.observations[r][c].append(0)
            else:
                self.observations[r][c].append(1)
        else:
            if randint(1, 100) / 100 <= self.false_positive:
                self.observations[r][c].append(1)
            else:
                self.observations[r][c].append(0)

        self.update_posterior(position)

    def update_posterior(self, position):
        r, c = position
        if self.overrides[r][c] is not None:
            return
        
        true_positive = 1 - self.false_positive
        true_negative = 1 - self.false_negative

        T_likelihood = 1
        not_T_likelihood = 1
        for observation in self.observations[r][c]:
            if observation:
                T_likelihood *= true_positive
                not_T_likelihood *= self.false_positive
            else:
                T_likelihood *= self.false_negative
                not_T_likelihood *= true_negative

        T_posterior = self.prior * T_likelihood
        not_T_posterior = (1 - self.prior)  * not_T_likelihood

        self.beliefs[r][c] = T_posterior / (T_posterior + not_T_posterior)

    def pop(self, position=None, *, error=True):
        output = (float('inf'), float('inf'), None) # (belief, distance, position)
        for r, row in enumerate(self.beliefs):
            for c, belief in enumerate(row):
                cell = (r, c)
                if belief is None or cell in self.popped:
                    continue

                distance = (abs(r - position[0]) + abs(c - position[1])) if position else float('inf')
                output = min(output, (-belief, distance, cell))

        cell = output[-1]

        if cell is None:
            if error:
                raise IndexError("pop from empty list")
        else:
            self.popped.add(cell)

        return cell

    def get_entropy(self):
        entropy = 0

        for r, row in enumerate(self.beliefs):
            for c, belief in enumerate(row):
                if self.overrides[r][c] is not None:
                    belief = self.overrides[r][c]
                elif belief is None:
                    belief = self.prior
                entropy += -belief * log(belief + 1e-12)

        return entropy

if __name__ == '__main__':
    seed(0)
    
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    h, w = len(grid), len(grid[0])
            
    belief_grid = BeliefGrid(grid, 0.2, 0.3)
    for i in range(len(grid[0])):
        belief_grid.scan_neighbors((2, i))
    belief_grid.override_belief((5, 7), pop=True)
    belief_grid.get_entropy()
    