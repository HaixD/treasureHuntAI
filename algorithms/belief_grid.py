from random import randint
from enum import IntEnum
from heapq import heappush, heappop

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
        self.beliefs_heap = []
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

        old_heap = self.beliefs_heap
        self.beliefs_heap = []
        for _, position in old_heap:
            r, c = position
            heappush((-self.beliefs[r][c], position))

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
        false_negative = 1 - self.false_negative

        T_likelihood = 1
        not_T_likelihood = 1
        for observation in self.observations[r][c]:
            if observation:
                T_likelihood *= true_positive
                not_T_likelihood *= self.false_positive
            else:
                T_likelihood *= false_negative
                not_T_likelihood *= self.false_negative

        T_posterior = self.prior * T_likelihood
        not_T_posterior = (1 - self.prior)  * not_T_likelihood

        self.beliefs[r][c] = T_posterior / (T_posterior + not_T_posterior)

        position = (r, c)
        if position not in self.popped:
            self.beliefs_heap.append((-self.beliefs[r][c], position))

    def pop(self):
        _, position = heappop(self.beliefs_heap)
        (r, c) = position

        if position in self.popped:
            return self.pop()
        self.popped.add(position)

        return r, c

if __name__ == '__main__':
    grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    belief_grid = BeliefGrid(grid, 0.1, 0.2)
    for _ in range(100):
        belief_grid.scan(randint(0, len(grid) - 1), randint(0, len(grid[0]) - 1))

    while True:
        try:
            print(belief_grid.pop())
        except Exception:
            break