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
