"""Breadth-First Search algorithm implementation for grid pathfinding.

This module provides a BFS implementation that finds the shortest path from a start position to a
treasure on a grid with obstacles.
"""

from collections import deque
from utils import get_neighbors
from constants import Cell


def bfs(grid, start_pos):
    """Find a path to the nearest treasure using Breadth-First Search.

    BFS explores all nodes at the current depth before moving to nodes at the next depth level,
    guaranteeing the shortest path in terms of number of steps.

    Args:
        grid (np.ndarray): 2D grid array containing cell types (treasures, walls, etc.).
        start_pos (tuple): Starting position as (row, col).

    Returns:
        tuple: A tuple containing:
            - path (list or None): List of (row, col) positions from start to treasure, or None if
                no path exists.
            - cells_expanded (int): Number of cells explored during the search.
    """
    queue = deque([[start_pos]])
    visited = {start_pos}
    cells_expanded = 0

    while queue:
        path = queue.popleft()
        position = path[-1]
        cells_expanded += 1

        if grid[position[0], position[1]] == Cell.TREASURE:
            return path, cells_expanded

        for neighbor in get_neighbors(grid, position, include_traps=True):
            nr, nc = neighbor
            if neighbor not in visited and grid[nr, nc] != Cell.WALL:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None, cells_expanded
