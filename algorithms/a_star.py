import heapq
from constants import Cell


def a_star(grid, start, goal, get_neighbors_func, heuristic_func):
    # pq = [(0, start)]           # Priority queue: (cost, position)
    visited = set()  # Record all fully explored cells so far
    parent = {start: None}  # Record best parents of each cell for path reconstruction
    cost = {start: 0}  # Record lowest costs to reach each cell
    heuristics = {start: heuristic_func(start, goal), goal: 0}
    cells_expanded = 0
    cost = {start: 0}
    h_start = heuristic_func(start, goal)
    f_start = cost[start] + h_start
    pq = [(f_start, start)]
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
        for neighbor in get_neighbors_func(current_pos, include_traps=True):
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
