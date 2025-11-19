from enum import IntEnum

class Cell(IntEnum):
    EMPTY = 0
    WALL = 1
    TREASURE = 2
    TREASURE_COLLECTED = 3
    TRAP = 4
    TRAP_TRIGGERED = 5
    START = 6
    PATH = 7

    @property
    def color(self):
        """Get associated color with cell type."""
        colors = {
            Cell.EMPTY: "white",
            Cell.WALL: "gold",
            Cell.TREASURE: "pink",
            Cell.TREASURE_COLLECTED: "hot pink",
            Cell.TRAP: "sky blue",
            Cell.TRAP_TRIGGERED: "royal blue",
            Cell.START: "light green"
        }

        return colors.get(self)

    @property
    def symbol(self):
        """Get associated symbol with cell type."""
        symbols = {
            Cell.WALL: "#",
            Cell.TREASURE: "T",
            Cell.TREASURE_COLLECTED: "+",
            Cell.TRAP: "X",
            Cell.TRAP_TRIGGERED: "!",
            Cell.START: "S"
        }

        return symbols.get(self)

# Path gradient colors (start and end RGB values)
# Start RGB:    (144, 238, 144)
# Treasure RGB: (255, 192, 203)
PATH_GRADIENT_START = (144, 238, 144)
PATH_GRADIENT_END = (255, 192, 203)
