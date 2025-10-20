How to run:
Install NumPy if you don't have it already:
run main.py

This will launch the GUI window where you can interact with the grid and run the search algorithms.

Implementation Details
Data Structures
Grid: The maze is represented by a 2D array. Each cell stores an integer code representing its state (TRAP, WALL, TREASURE, EMPTY)
BFS Queue: The Breadth-First Search algorithm uses a collections.deque object to store the paths to explore
DFS Stack (Recursive): The Depth-First Search is implemented using recursion. The program's function call stack implicitly acts the data structure required for DFS.

UCS Priority Queue: The Uniform-Cost Search uses Python's heapq library to implement a priority queue

Design Decisions

The application is encapsulated within a single GridApp class, which manages the grid state, GUI elements, and algorithm execution.

GUI: Tkinter was chosen for its simplicity and inclusion in the Python standard library.The grid is drawn on a Canvas widget, with each cell represented by a colored rectangle and sometimes a symbol symbol.

Algorithm Integration: Each search algorithm (bfs, dfs, ucs) is implemented as a method within the class. Helper methods (run_bfs, run_dfs, run_ucs) handle the setup ,clearing previous paths, execution.

Path Visualization: A unique design choice was to render the found path not with a single color, but with a color gradient to make the direction of the path obvious. 
<img width="1512" height="982" alt="Screenshot 2025-10-19 at 9 31 10 PM" src="https://github.com/user-attachments/assets/d6d0b1f2-bd9d-4c32-8513-c55a7ec8c54e" />
<img width="1512" height="982" alt="Screenshot 2025-10-19 at 9 31 12 PM" src="https://github.com/user-attachments/assets/90d0e3d9-f5c7-4305-aa2c-b104ed01e170" />
<img width="1512" height="982" alt="Screenshot 2025-10-19 at 9 31 14 PM" src="https://github.com/user-attachments/assets/743449a0-2567-46cd-9adc-d72e3dc43469" />

