import heapq
from utils import get_neighbors
from constants import Cell


def ucs(grid, start, goal):
    pq = [(0, start)]  # Priority queue: (cost, position)
    visited = set()  # Record all fully explored cells so far
    parent = {start: None}  # Record best parents of each cell for path reconstruction
    cost = {start: 0}  # Record lowest costs to reach each cell
    cells_expanded = 0

    while pq:
        # Get next best cost and position from priority queue
        current_cost, current_pos = heapq.heappop(pq)

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
                path.append(current_pos)
                current_pos = parent[current_pos]
            return path[::-1], cells_expanded

        # Otherwise, explore neighbors and get their costs
        for neighbor in get_neighbors(grid, current_pos, include_traps=True):
            new_cost = current_cost + 1
            if grid[neighbor[0], neighbor[1]] == Cell.TRAP:
                new_cost += 4

            # Update if this is a better path to neighbor
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = current_pos
                heapq.heappush(pq, (new_cost, neighbor))

    return None, cells_expanded
