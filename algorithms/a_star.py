"""A* Search algorithm implementation for grid pathfinding.

This module provides an A* implementation that finds the optimal lowest-cost path by combining
actual path cost with heuristic estimates to the goal.
"""

import heapq
from utils import get_neighbors
from constants import Cell


def a_star(grid, start, goal, heuristic_func):
    """Find the optimal path to the goal using A* Search.

    A* combines the actual cost to reach a node (g-cost) with a heuristic estimate of the remaining
    distance to the goal (h-cost) to efficiently find the optimal path. The f-cost (g + h)
    determines node expansion order. With an admissible heuristic, A* guarantees the optimal
    solution.

    Args:
        grid (np.ndarray): 2D grid array containing cell types (treasures, walls, traps, etc.).
        start (tuple): Starting position as (row, col).
        goal (tuple): Goal position as (row, col).
        heuristic_func (callable): Admissible heuristic function that estimates distance to goal.
            Takes two positions (current, goal) and returns a numeric estimate.
            Common choices: manhattan_distance, euclidean_distance.

    Returns:
        tuple: A tuple containing:
            - path (list or None): List of (heuristic_cost, (row, col)) tuples from
                start to goal, or None if no path exists.
            - cells_expanded (int): Number of cells explored during the search.

    Notes:
        - Regular moves cost 1, traps add an additional cost of 4.
        - The heuristic must be admissible (never overestimate) for optimality.
    """
    visited = set()  # Record all fully explored cells so far
    parent = {start: None}  # Record best parents of each cell for path reconstruction
    cost = {start: 0}  # Record lowest costs to reach each cell
    heuristics = {start: heuristic_func(start, goal), goal: 0}
    cells_expanded = 0
    h_start = heuristic_func(start, goal)
    f_start = cost[start] + h_start
    pq = [(f_start, start)]  # Priority queue

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
            step_cost = 1
            if grid[neighbor[0], neighbor[1]] == Cell.TRAP:
                step_cost += 4
            new_cost = cost[current_pos] + step_cost
            # Update if this is a better path to neighbor
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = current_pos
                h_cost = heuristic_func(neighbor, goal)
                heuristics[neighbor] = h_cost
                new_f_cost = new_cost + h_cost
                heapq.heappush(pq, (new_f_cost, neighbor))

    return None, cells_expanded
