from collections import deque
from constants import Cell

def bfs(grid, start_pos, get_neighborus_func):
        queue = deque([[start_pos]])
        visited = {start_pos}
        cells_expanded = 0
        while queue:
            path = queue.popleft()
            position = path[-1]
            cells_expanded += 1

            if grid[position[0], position[1]] == Cell.TREASURE:
                return path, cells_expanded

            for neighbor in get_neighborus_func(position, include_traps=True):
                nr, nc = neighbor
                if neighbor not in visited and grid[nr, nc] != Cell.WALL:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return None, cells_expanded
