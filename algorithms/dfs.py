"""Depth-First Search algorithm implementation for grid pathfinding.

This module provides a DFS implementation that explores paths deeply before backtracking to find a
treasure on a grid with obstacles.
"""

from utils import get_neighbors
from constants import Cell


def dfs(
    grid,
    start_pos,
    position=None,
    path=None,
    visited=None,
    cells_expanded=None,
    *,
    move_order=None,
    include_traps=True
):
    """Find a path to a treasure using Depth-First Search.

    DFS explores as far as possible along each branch before backtracking. The path found is not
    guaranteed to be the shortest. This implementation uses recursion to explore the grid.

    Args:
        grid (np.ndarray): 2D grid array containing cell types (treasures, walls, etc.).
        start_pos (tuple): Starting position as (row, col).
        position (tuple, optional): Current position in recursion. Defaults to start_pos.
        path (list, optional): Current path taken. Defaults to empty list.
        visited (set, optional): Set of visited positions. Defaults to empty set.
        cells_expanded (list, optional): List containing count of expanded cells.
            Uses list to maintain reference across recursive calls. Defaults to [0].
        move_order (list, optional): Order of moves to try (e.g., up, down, left, right).
            Defaults to None.
        include_traps (bool, optional): Whether to allow moving through traps.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - path (list or None): List of (row, col) positions from start to treasure,
                or None if no path exists.
            - cells_expanded (int): Number of cells explored during the search.
    """
    position = position or start_pos
    path = path or []
    visited = visited or set()
    if cells_expanded is None:
        cells_expanded = [0]  # Use list to keep track across recursive calls

    r, c = position

    # invalid position
    if grid[r, c] == Cell.WALL:
        return None, cells_expanded[0]

    # already visited (avoid loops)
    if position in visited:
        return None, cells_expanded[0]

    # found treasure
    if grid[r, c] == Cell.TREASURE:
        cells_expanded[0] += 1
        return path + [position], cells_expanded[0]

    cells_expanded[0] += 1
    visited.add(position)
    for cell in get_neighbors(
        grid, position, moves=move_order, include_traps=include_traps
    ):
        result, _ = dfs(
            grid,
            start_pos,
            cell,
            path + [position],
            visited,
            cells_expanded,
            move_order=move_order,
            include_traps=include_traps,
        )
        if result is not None:
            return result, cells_expanded[0]

    return None, cells_expanded[0]
