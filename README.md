# How to run:
1. install NumPy if you don't have it already
2. run main.py
    - this will launch the GUI window where you can interact with the grid and run the search algorithms.
3. select 1 of the following options: Bayes(High Noise), Bayes (Medium Noise), or Bayes(Low Noise)

# Sensor Model Description and Bayesian Update
When we scan a cell, there will be a chance that we "intentionally" flip our result to simulate a false positive or false negative. Based on the type of scan result, we update all beliefs differently.
- if the scan result is a **false negative**: the belief for the scanned cell is multiplied by the *false negative chance* and all other cells are multiplied by the *true negative chance*
- if the scan result is a **true positive**: the belief for the scanned cell is multiplied by the *true positive chance* and all other cells are multiplied by the *false positive chance*
- if the scan result is a **false positive**: the belief for the scanned cell is multiplied by the *false positive chance* and all other cells are multiplied by the *true positive chance*
- if the scan result is a **true negative**: the belief for the scanned cell is multiplied by the *true negative chance* and all other cells are multiplied by the *false engative chance*

After applying the multiplications, all beliefs are normalized (by summing them up and dividing each of them by the total).

# Belief Implementation
Beliefs are stored in a 2D list where each value is initialized to number of treasures / total number of cells. This list is updated whenever we scan.

# Decision Policy
WRITE DECISION POLICY HERE PLS

# Experiments and Results
|Noise Level|Average Steps|Average Scans|Average Entropy at Detection|Detection Accuracy|
|-----------|-------------|-------------|----------------------------|------------------|
|Low        |             |             |                            |100%              |
|Medium     |             |             |                            |100%              |
|High       |410.9        |509.6        |5.55823                     |100%              |