from math import log
import random

try:
    from constants import Cell
except ModuleNotFoundError:
    from collections import namedtuple

    Cell = namedtuple("Cell", ["TREASURE"])(TREASURE=2)


class BeliefGrid:
    def __init__(self, grid, rand, false_positive, false_negative):
        self.grid = grid
        self.rand = rand
        self.false_positive = false_positive
        self.false_negative = false_negative

        self.popped = set()

        self.treasures = 0
        for row in grid:
            for col in row:
                if col == Cell.TREASURE:
                    self.treasures += 1
        self.prior = 1.0 / (len(self.grid) * len(self.grid[0]))

        self.beliefs = [[self.prior] * len(grid[0]) for _ in range(len(grid))]

    def scan(self, position):
        r, c = position
        true_negative = 1 - self.false_positive
        true_positive = 1 - self.false_negative

        if self.grid[r][c] == Cell.TREASURE:
            if self.rand.randint(1, 100) / 100 <= self.false_negative:
                self.update_posterior(position, self.false_negative, true_negative)
            else:
                self.update_posterior(position, true_positive, self.false_positive)
        else:
            if self.rand.randint(1, 100) / 100 <= self.false_positive:
                self.update_posterior(position, self.false_positive, true_positive)
            else:
                self.update_posterior(position, true_negative, self.false_negative)

    def update_posterior(self, position, self_chance, other_chance):
        h, w = len(self.grid), len(self.grid[0])

        total = 0
        for r in range(h):
            for c in range(w):
                if (r, c) == position:
                    self.beliefs[r][c] *= self_chance
                else:
                    self.beliefs[r][c] *= other_chance

                total += self.beliefs[r][c]

        for r in range(h):
            for c in range(w):
                self.beliefs[r][c] /= total

    def pop(self, position=None, *, error=True):
        output = (float("inf"), float("inf"), None)  # (belief, distance, position)
        for r, row in enumerate(self.beliefs):
            for c, belief in enumerate(row):
                cell = (r, c)
                if belief is None or cell in self.popped:
                    continue

                distance = (
                    (abs(r - position[0]) + abs(c - position[1]))
                    if position
                    else float("inf")
                )
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

        for row in self.beliefs:
            for belief in row:
                entropy += -belief * log(belief + 1e-12)

        return entropy


if __name__ == "__main__":
    test_grid = [[2, 2], [0, 0]]

    belief_grid = BeliefGrid(
        test_grid, random.Random(), false_positive=0.1, false_negative=0.2
    )
    belief_grid.scan((0, 0))
    belief_grid.scan((0, 1))
    print(
        *map(
            lambda row: list(map(lambda col: round(col, 3), row)), belief_grid.beliefs
        ),
        sep="\n"
    )
    print(belief_grid.get_entropy())
