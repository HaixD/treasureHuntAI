from itertools import permutations

def get_moves():
    """
    Generate all permutations of the four cardinal direction moves.
    
    Returns:
        iterator: Generator of move tuples (each tuple contains 4 lambda functions)
    """
    return permutations([
        lambda r, c: (r, c + 1),  # right
        lambda r, c: (r, c - 1),  # left
        lambda r, c: (r - 1, c),  # up
        lambda r, c: (r + 1, c)   # down
    ])