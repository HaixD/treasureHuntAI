"""Greedy Best-First Search algorithm implementation for grid pathfinding.

This module provides a Greedy search implementation that finds a path by always moving toward the
goal based on a heuristic estimate, without considering path cost.
"""

import heapq
from utils import get_neighbors
from constants import Cell


def greedy(grid, start, goal, heuristic_func):
    """Find a path to the goal using Greedy Best-First Search.

    Greedy search expands nodes based solely on their heuristic estimate to the goal, always
    choosing the node that appears closest. This approach is fast but does not guarantee an optimal
    path. Movement through traps incurs an additional cost penalty of 4, while regular moves cost 1.

    Args:
        grid (np.ndarray): 2D grid array containing cell types (treasures, walls, traps, etc.).
        start (tuple): Starting position as (row, col).
        goal (tuple): Goal position as (row, col).
        heuristic_func (callable): Function that estimates distance to goal.
            Takes two positions (current, goal) and returns a numeric estimate.
            Common choices: manhattan_distance, euclidean_distance.

    Returns:
        tuple: A tuple containing:
            - path (list or None): List of (heuristic_cost, (row, col)) tuples from
                start to goal, or None if no path exists.
            - cells_expanded (int): Number of cells explored during the search.
    """
    pq = [(0, start)]  # Priority queue: (cost, position)
    visited = set()  # Record all fully explored cells so far
    parent = {start: None}  # Record best parents of each cell for path reconstruction
    cost = {start: 0}  # Record lowest costs to reach each cell
    heuristics = {start: heuristic_func(start, goal), goal: 0}
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
                path.append((heuristics[current_pos], current_pos))
                current_pos = parent[current_pos]
            return path[::-1], cells_expanded

        # Otherwise, explore neighbors and get their costs
        for neighbor in get_neighbors(grid, current_pos, include_traps=True):
            heuristic = heuristic_func(neighbor, goal)
            heuristics[neighbor] = heuristic

            new_cost = heuristic_func(neighbor, goal)
            if grid[neighbor[0], neighbor[1]] == Cell.TRAP:
                new_cost += 4

            # Update if this is a better path to neighbor
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = current_pos
                heapq.heappush(pq, (new_cost, neighbor))

    return None, cells_expanded
