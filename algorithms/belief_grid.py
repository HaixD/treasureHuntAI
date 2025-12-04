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

def update_posterior(observations, prior, fp, fn):
    true_positive = 1 - fp
    true_negative = 1 - fn

    T_likelihood = 1
    not_T_likelihood = 1
    for observation in observations:
        if observation:
            T_likelihood *= true_positive
            not_T_likelihood *= fp
        else:
            T_likelihood *= fn
            not_T_likelihood *= true_negative

    T_posterior = prior * T_likelihood
    not_T_posterior = (1 - prior)  * not_T_likelihood

    return T_posterior / (T_posterior + not_T_posterior)

res = update_posterior(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    2 / 70,
    0.1,
    0.2
)
##print(res)

class BeliefGrid:
    def __init__(self, grid, false_positive, false_negative):
        self.grid = grid
        self.false_positive = false_positive
        self.false_negative = false_negative
        
        self.observations = [[[] for _ in range(len(grid[0]))] for _ in range(len(grid))]
        self.beliefs = [[None] * len(grid[0]) for _ in range(len(grid))]
        self.popped = set()

        self.treasures = 1
        for row in grid:
            for col in row:
                self.treasures += col == Cell.TREASURE
        self.decrement_treasure()

    def decrement_treasure(self):
        self.treasures -= 1
        self.prior = self.treasures / (len(self.grid) * len(self.grid[0]))

        for r, row in enumerate(self.beliefs):
            for c, col in enumerate(row):
                if col is not None:
                    self.update_posterior(r, c)

    def scan(self, r, c):
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

        self.update_posterior(r, c)

    def scan_neighbors(self, r, c):
        h, w = len(self.grid), len(self.grid[0])
       
        def try_scan(r, c):
            if 0 <= r < h and 0 <= c < w:
                self.scan(r, c)

        try_scan(r - 1, c)
        try_scan(r + 1, c)
        try_scan(r, c - 1)
        try_scan(r, c + 1)

    def update_posterior(self, r, c):
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
        output = (-1, None) # (belief, distance, position)
        for r, row in enumerate(self.beliefs):
            for c, belief in enumerate(row):
                position = (r, c)
                if belief is None or position in self.popped:
                    continue

                distance = abs(r - position[0]) + abs(c - position[1])

                if (belief > output[0]) or (position and belief == output[0] and distance < output[1]):
                    output = (belief, distance, position)

        position = output[-1]

        if position is None:
            if error:
                raise IndexError("pop from empty list")
        else:
            self.popped.add(position)

        return position

    def get_entropy(self):
        entropy = 0

        for row in self.beliefs:
            for belief in row:
                if belief is None:
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
    belief_grid.scan_neighbors(0, 0)
    belief_grid.scan_neighbors(0, 1)
    belief_grid.scan_neighbors(1, 0)
    print(belief_grid.pop())