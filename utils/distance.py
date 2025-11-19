def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point tuple (x, y)
        p2: Second point tuple (x, y)
    
    Returns:
        float: Euclidean distance
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


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

    distance = 0
    for a, b in zip(p1, p2):
        distance += abs(a - b)

    return distance