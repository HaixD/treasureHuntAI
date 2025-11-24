# How to run:
1. Install NumPy if you don't have it already
2. run main.py
    - this will launch the GUI window where you can interact with the grid and run the search algorithms.
3. set the minimax depth to a to any desired value and click minimax or minimax (pruning) to perform minimax and alpha-beta pruning respectively
    - these buttons are for **AI vs AI**. For **AI vs human**, click on a valid adjacent tile next to the **B tile**.
# Utility Value Formula
There are 2 components which make up the utility value
- a treasure difference (number of teasures owned by MAX - number of treasured owned by MIN)
- the distance between MAX and its closest treasure (based on A*), and the distance between MIN and its closest treasure (also based on A*).
    - Because we want to penalize large distances and encourage shorter distances, we subtract the diagonal length of the grid by the A* distance (this assumes the diagonal length is the longest a path could be)
    - We take the adjusted distance of MAX and subtract it by MIN's adjusted distance
- when there are no treasures left the utility value will be either `float('inf')`, `-float('inf')`, or 0 depending who has more treasures
# Comparison
|Algorithm      |depth|Nodes Expanded|Execution Time (ms)|Winner |
|---------------|-----|--------------|-------------------|-------|
|Minimax        |2    |3595          |325                |MAX(A) |
|Alpha-Beta     |2    |3593          |324                |MAX(A) |
|Minimax        |3    |31181         |2154               |TIE    |
|Alpha-Beta     |3    |20726         |1433               |TIE    |
|Minimax        |4    |40427         |4581               |MAX(A) |
|Alpha-Beta     |4    |16638         |2402               |MAX(A) |

Here, we observed that alpha-beta runs noitceably faster while achieving the same end result. It also results in less nodes expanded. Another big factor for run time and number of nodes expanded is depth. In our testing, we found that high nodes (e.g. 10) are impractical because it takes too long.

In regards to the evaluation function, we saw that individually they weren't ideal but combining them yielded better results.

# Screenshot
![Screenshot](./gui_screenshot.png)
# Generative AI Statement
No AI was used for Assignment 3.
