from math import log
from random import randint
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

    def pop(self, *, error=True):
        output = (-1, None) # (belief, position)
        for r, row in enumerate(self.beliefs):
            for c, belief in enumerate(row):
                position = (r, c)
                if belief is None or position in self.popped:
                    continue

                output = max(output, (belief, position), key=lambda v : v[0])

        position = output[1]

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
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    h, w = len(grid), len(grid[0])
            
    belief_grid = BeliefGrid(grid, 0.1, 0.2)
    print(belief_grid.get_entropy())
    for _ in range(len(grid) * len(grid[0]) * 100):
        belief_grid.scan(randint(0, h - 1), randint(0, w - 1))
    print(belief_grid.get_entropy())
    for _ in range(4):
        try:
            r, c = belief_grid.pop()
            print((r, c), grid[r][c] == 2)
        except IndexError:
            break
