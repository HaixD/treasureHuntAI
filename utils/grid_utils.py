from itertools import permutations
import random
from constants import Cell


def get_closest_point(start, targets, distance_func):
    """
    Get the closest target point to start position.

    Args:
        start: Starting position
        targets: Single target or list of target positions
        distance_func: Function to calculate distance between two points

    Returns:
        tuple: Closest target position
    """
    if not isinstance(targets, list):
        return targets

    return min(targets, key=lambda cur: distance_func(start, cur))


def get_moves():
    """
    Generate all permutations of the four cardinal direction moves.

    Returns:
        iterator: Generator of move tuples (each tuple contains 4 lambda functions)
    """
    return permutations(
        [
            lambda r, c: (r, c + 1),  # right
            lambda r, c: (r, c - 1),  # left
            lambda r, c: (r - 1, c),  # up
            lambda r, c: (r + 1, c),  # down
        ]
    )


def get_neighbors(grid, pos, moves=None, include_traps=False):
    """
    Get valid neighbors for a given position.

    Args:
        grid: The grid array
        pos: Current position tuple (r, c)
        grid_size: Size of the grid
        moves: Optional tuple of move functions
        include_traps: Whether to include trap cells as valid neighbors

    Returns:
        list: List of valid neighbor positions
    """

    r, c = pos
    valid_neighbors = []

    moves = moves or next(get_moves())

    for move in moves:
        nr, nc = move(r, c)
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            if (
                not include_traps
                and grid[nr, nc] not in [Cell.WALL, Cell.TRAP]
                or include_traps
                and grid[nr, nc] not in [Cell.WALL]
            ):
                valid_neighbors.append((nr, nc))

    return valid_neighbors


def get_random_seed():
    """Generate a random positive 32-bit integer seed."""
    return random.randint(0, 2**32 - 1)


def interpolate_color(ratio, start_rgb, end_rgb):
    """
    Interpolate between two RGB colors.

    Args:
        ratio: Float between 0 and 1
        start_rgb: Starting RGB tuple
        end_rgb: Ending RGB tuple

    Returns:
        str: Hex color string
    """
    start_r, start_g, start_b = start_rgb
    end_r, end_g, end_b = end_rgb

    r = int(start_r + (end_r - start_r) * ratio)
    g = int(start_g + (end_g - start_g) * ratio)
    b = int(start_b + (end_b - start_b) * ratio)

    return f"#{r:02x}{g:02x}{b:02x}"


def generate_gradient_colors(path_length, start_rgb, end_rgb):
    """
    Generate gradient colors for a path.

    Args:
        path_length: Length of the path
        start_rgb: Starting RGB tuple
        end_rgb: Ending RGB tuple

    Returns:
        list: List of hex color strings
    """
    colors = []
    for i in range(path_length):
        ratio = i / max(path_length - 1, 1)
        colors.append(interpolate_color(ratio, start_rgb, end_rgb))
    return colors
