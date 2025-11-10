# How to run:
1. Install NumPy if you don't have it already
2. run main.py
- This will launch the GUI window where you can interact with the grid and run the search algorithms.
# Implementation Details
- **A\*:** A\* is almost the same as UCS except the value we pass to the priority queue along with any node is the manhattan distance from the goal + true cost
- **Greedy:** Greedy also uses similar code as UCS except we no longer consider true cost and all nodes passed into the priority queue are paired with only their manhattan distance
- **UCS Priority Queue:** The Uniform-Cost Search uses Python's heapq library to implement a priority queue
- **DFS Stack (Recursive):** The Depth-First Search is implemented using recursion. The program's function call stack implicitly acts the data structure required for DFS.
- **BFS Queue:** The Breadth-First Search algorithm uses a collections.deque object to store the paths to explore
- **Grid:** The maze is represented by a 2D array. Each cell stores an integer code representing its state (TRAP, WALL, TREASURE, EMPTY)
# Comparison
|Algorithm|Seed               |Path Cost|Cells Expanded|Time (ms)|
|---------|-------------------|---------|--------------|---------|
|A\*      |6033585196692522394|108      |592           |5.997    |
|Greedy   |6033585196692522394|120      |137           |2.002    |
|A\*      |7210453983052381200|92       |422           |4.999    |
|Greedy   |7210453983052381200|98       |115           |2.001    |
|A\*      |2700272591481403539|74       |395           |4.000    |
|Greedy   |2700272591481403539|82       |85            |1.000    |

From the table above, we can see that A\* consistently has a lower path cost but it also has a higher number of cells expanded (and time). This is to be expected since A\* has to check more nodes to truly determine a path to the goal has the lowest true cost.
# Screenshot
![Screenshot](./gui_screenshot.png)
# Generative AI Statement
No AI was used for Assignment 2.
