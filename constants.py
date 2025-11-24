from enum import IntEnum


class Cell(IntEnum):
    EMPTY = 0
    WALL = 1
    TREASURE = 2
    TREASURE_COLLECTED = 3
    TRAP = 4
    TRAP_TRIGGERED = 5
    START = 6
    START_FIRST = 7
    START_SECOND = 8
    PATH = 9
    PATH_FIRST = 10
    PATH_SECOND = 11
    PATH_BOTH = 12

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
            Cell.START: "light green",
            Cell.START_FIRST: "tomato",
            Cell.START_SECOND: "cornflower blue",
            Cell.PATH_FIRST: "tomato",
            Cell.PATH_SECOND: "cornflower blue",
            Cell.PATH_BOTH: "dark orchid",
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
            Cell.START: "S",
            Cell.START_FIRST: "A",
            Cell.START_SECOND: "B",
        }

        return symbols.get(self)


# Path gradient colors (start and end RGB values)
# Start RGB:    (144, 238, 144)
# Treasure RGB: (255, 192, 203)
PATH_GRADIENT_START = (144, 238, 144)
PATH_GRADIENT_END = (255, 192, 203)
