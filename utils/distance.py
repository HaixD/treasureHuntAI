def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.

    Args:
        p1: First point tuple (x, y)
        p2: Second point tuple (x, y)

    Returns:
        float: Euclidean distance
    """
    if len(p1) != len(p2):
        raise ValueError("Points must have the same number of dimensions.")

    (x1, y1), (x2, y2) = p1, p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def manhattan_distance(p1, p2):
    """
    Calculate Manhattan distance between two points.

    Args:
        p1: First point tuple
        p2: Second point tuple

    Returns:
        int: Manhattan distance
    """
    if len(p1) != len(p2):
        raise ValueError("Points must have the same number of dimensions.")

    return sum(abs(a - b) for a, b in zip(p1, p2))
