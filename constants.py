"""Cell types and visual properties for a treasure hunt grid game.

This module defines the Cell enumeration which represents different states of cells in a grid-based
treasure hunt game. Each cell type has associated visual properties (colors and symbols) for
rendering the grid.

The module also defines gradient colors for rendering paths from the start position to treasure
locations.
"""

from enum import IntEnum


class Cell(IntEnum):
    """Enumeration of cell types in a treasure hunt grid.

    This enum defines the various states a cell can have in the grid, including empty spaces, walls,
    treasures, traps, starting positions, and path markers. Each cell type has associated visual
    properties like colors and symbols for rendering.
    """

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
        """Get the display color associated with this cell type.

        Returns:
            str: A color name (e.g., 'white', 'gold', 'pink') representing the visual appearance of
                this cell type, or None if no color is defined.
        """
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
        """Get the character symbol associated with this cell type.

        Returns:
            str: A single character representing this cell type (e.g., '#' for walls, 'T' for
                treasure), or None if no symbol is defined.
        """
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


# Start RGB:    (144, 238, 144)
PATH_GRADIENT_START = (144, 238, 144)
"""tuple: RGB color values for the start of path gradient (light green)."""

# Treasure RGB: (255, 192, 203)
PATH_GRADIENT_END = (255, 192, 203)
"""tuple: RGB color values for the end of path gradient (light pink)."""
