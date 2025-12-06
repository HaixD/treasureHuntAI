"""Bayesian belief tracking for treasure location estimation.

This module implements a belief grid that maintains probability distributions over treasure
locations using Bayesian inference with noisy sensor readings. The system handles false positives
and false negatives in treasure detection.
"""

from math import log
import random

try:
    from constants import Cell
except ModuleNotFoundError:
    from collections import namedtuple

    Cell = namedtuple("Cell", ["TREASURE"])(TREASURE=2)


class BeliefGrid:
    """Maintains probabilistic beliefs about treasure locations on a grid.

    This class uses Bayesian inference to update beliefs about where treasures are located based on
    noisy sensor readings. It handles sensor errors including false positives (detecting treasure
    when there is none) and false negatives (failing to detect treasure when it exists).

    Attributes:
        grid (list): The actual grid containing treasure and obstacle information.
        rand (random.Random): Random number generator for simulating sensor noise.
        false_positive (float): Probability of false positive sensor reading (0-1).
        false_negative (float): Probability of false negative sensor reading (0-1).
        popped (set): Set of positions that have been selected/visited.
        treasures (int): Total number of treasures in the grid.
        prior (float): Prior probability of treasure at any given position.
        beliefs (list): 2D list of probability beliefs for each grid position.
    """

    def __init__(self, grid, rand, false_positive, false_negative):
        """Initialize a BeliefGrid with sensor noise parameters.

        Args:
            grid (list): 2D grid containing cell types including treasures.
            rand (random.Random): Random number generator for sensor simulation.
            false_positive (float): Probability of false positive (0-1).
            false_negative (float): Probability of false negative (0-1).
        """
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
        """Perform a noisy sensor scan at the given position and update beliefs.

        This simulates a sensor reading that may produce false positives or false negatives based on
        the configured error rates. The belief probabilities for all grid positions are updated
        using Bayesian inference.

        Args:
            position (tuple): Position to scan as (row, col).
        """
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
        """Update belief probabilities using Bayesian inference.

        This method applies Bayes' rule to update the posterior probabilities for all grid positions
        based on a sensor reading. The scanned position is updated with self_chance while all other
        positions are updated with other_chance. Probabilities are then normalized to sum to 1.

        Args:
            position (tuple): Position that was scanned as (row, col).
            self_chance (float): Probability factor for the scanned position.
            other_chance (float): Probability factor for all other positions.
        """
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
        """Select and return the most likely treasure position.

        This method selects the grid cell with the highest belief probability that hasn't been
        selected before. When a position is provided, distance is used as a tiebreaker (closer
        positions are preferred).

        Args:
            position (tuple, optional): Current position for distance calculation.
                Defaults to None (distance not considered).
            error (bool, optional): Whether to raise an error if no positions available.
                Defaults to True.

        Returns:
            tuple or None: The selected position as (row, col), or None if no
                positions are available.

        Raises:
            IndexError: If no positions are available and error=True.
        """
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
        """Calculate the entropy of the current belief distribution.

        Entropy measures the uncertainty in the belief distribution. Higher entropy indicates more
        uncertainty about treasure locations, while lower entropy indicates more confident beliefs.

        Returns:
            float: The Shannon entropy of the belief distribution.
        """
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
