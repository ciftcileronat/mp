### In this DP model, p1,p0 and q0 are the decision variables, They all be found by running the DP code.
### This is PhD Dissertation Chapter-1: Scenario 3.
### Average cost bi-objective dynamic programming model, we use weighted-sum method to maximize profit and minimize waste.

# Average cost

import numpy as np
import time

start = time.time()
import math
from scipy.stats import uniform
from scipy.stats import binom


def findMaxNumber(qList):
    qDict = {}
    qListMax = float("-inf")
    for i in range(len(qList)):
        currentMax, currentMaxIdx = float("-inf"), [[None, None], None]
        for j in range(len(qList[i])):
            for x in range(len(qList[i][j])):
                currentNumber = qList[i][j][x]
                if currentNumber > currentMax:
                    currentMax, currentMaxIdx = currentNumber, [j, x]
        qDict[i] = [currentMaxIdx, currentMax]
        qListMax = max(qListMax, currentMax)

    qDict["listMaxValue"] = qListMax
    return qDict


from scipy import stats

# N: pazar payi
N_a = 5;
N_b = 5;
#N_a = 3;
#N_b = 3;
# h:stok tutma maliyet
h = 0.01
# yeni urun satis fiyati, deltadan kucuk

# weighting factor
W=(0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
#W=(0.1,0.2,0.5,0.6,0.7,0.8,0.9,1)
#W = (0,0.4,0.3,0.5)
# F:tasima maliyeti (fixed)
F = 0.1
# c: br siparis maliyeti
##c=0.1
c = 0.1
# a_q0: stok alt sinir, u_q0:stok ust sinir
a_q0 = 0;
u_q0 = 6;
#u_q0 = 4;
epsilon=0.005
delta_a = 0.6;
delta_b = 0.6;
# eski urun satis fiyati
# P1 = ( 0.05,0.1,0.15,0.2,0.25, 0.3, 0.35,0.4, 0.45,0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8,0.85,0.9,0.95,1)
p0=0.55
# P0= ( 0.05,0.1,0.15,0.2,0.25, 0.3, 0.35,0.4, 0.45,0.5, 0.55,0.6, 0.65,0.7, 0.75,0.8,0.85,0.9,0.95,1)
##ilk fiyat a subesinin, ikinci fiyat b subesinin

#P = [[0.15, 0.15],[0.15, 0.25],[0.15, 0.35],[0.15, 0.45],[0.25, 0.15],[0.25, 0.25],[0.25, 0.35],[0.25, 0.45],[0.35, 0.15],[0.35, 0.25],[0.35, 0.35],[0.35, 0.45],[0.45, 0.15],[0.45, 0.25],[0.45, 0.35],[0.45, 0.45]]
#P = [[0.05, 0.05],[0.05, 0.15],[0.05, 0.25],[0.05, 0.35],[0.05, 0.45],[0.05, 0.55],[0.15, 0.05],[0.15, 0.15],[0.15, 0.25],[0.15, 0.35],[0.15, 0.45],[0.15, 0.55],[0.25, 0.05],[0.25, 0.15],[0.25, 0.25],[0.25, 0.35],[0.25, 0.45],[0.25, 0.55],[0.35, 0.05],[0.35, 0.15],[0.35, 0.25],[0.35, 0.35],[0.35, 0.45],[0.35, 0.55],[0.45, 0.05],[0.45, 0.15],[0.45, 0.25],[0.45, 0.35],[0.45, 0.45],[0.45, 0.55],[0.55, 0.05],[0.55, 0.15],[0.55, 0.25],[0.55, 0.35],[0.55, 0.45],[0.55, 0.55]]
#P = [[0.35, 0.35],[0.35, 0.45],[0.45, 0.35],[0.45, 0.45]]

#P=[[0.05, 0.05], [0.05, 0.1], [0.05, 0.15], [0.05, 0.2], [0.05, 0.25], [0.05, 0.3], [0.05, 0.35], [0.05, 0.4], [0.05, 0.45], [0.1, 0.05], [0.1, 0.1], [0.1, 0.15], [0.1, 0.2], [0.1, 0.25], [0.1, 0.3], [0.1, 0.35], [0.1, 0.4], [0.1, 0.45], [0.15, 0.05], [0.15, 0.1], [0.15, 0.15], [0.15, 0.2], [0.15, 0.25], [0.15, 0.3], [0.15, 0.35], [0.15, 0.4], [0.15, 0.45], [0.2, 0.05], [0.2, 0.1], [0.2, 0.15], [0.2, 0.2], [0.2, 0.25], [0.2, 0.3], [0.2, 0.35], [0.2, 0.4], [0.2, 0.45], [0.25, 0.05], [0.25, 0.1], [0.25, 0.15], [0.25, 0.2], [0.25, 0.25], [0.25, 0.3], [0.25, 0.35], [0.25, 0.4], [0.25, 0.45], [0.3, 0.05], [0.3, 0.1], [0.3, 0.15], [0.3, 0.2], [0.3, 0.25], [0.3, 0.3], [0.3, 0.35], [0.3, 0.4], [0.3, 0.45], [0.35, 0.05], [0.35, 0.1], [0.35, 0.15], [0.35, 0.2], [0.35, 0.25], [0.35, 0.3], [0.35, 0.35], [0.35, 0.4], [0.35, 0.45], [0.4, 0.05], [0.4, 0.1], [0.4, 0.15], [0.4, 0.2], [0.4, 0.25], [0.4, 0.3], [0.4, 0.35], [0.4, 0.4], [0.4, 0.45], [0.45, 0.05], [0.45, 0.1], [0.45, 0.15], [0.45, 0.2], [0.45, 0.25], [0.45, 0.3], [0.45, 0.35], [0.45, 0.4], [0.45, 0.45]]
P=[[0.05, 0.05], [0.05, 0.1], [0.05, 0.15], [0.05, 0.2], [0.05, 0.25], [0.05, 0.3], [0.05, 0.35], [0.05, 0.4], [0.1, 0.05], [0.1, 0.1], [0.1, 0.15], [0.1, 0.2], [0.1, 0.25], [0.1, 0.3], [0.1, 0.35], [0.1, 0.4], [0.15, 0.05], [0.15, 0.1], [0.15, 0.15], [0.15, 0.2], [0.15, 0.25], [0.15, 0.3], [0.15, 0.35], [0.15, 0.4], [0.2, 0.05], [0.2, 0.1], [0.2, 0.15], [0.2, 0.2], [0.2, 0.25], [0.2, 0.3], [0.2, 0.35], [0.2, 0.4],[0.25, 0.05], [0.25, 0.1], [0.25, 0.15], [0.25, 0.2], [0.25, 0.25], [0.25, 0.3], [0.25, 0.35], [0.25, 0.4],  [0.3, 0.05], [0.3, 0.1], [0.3, 0.15], [0.3, 0.2], [0.3, 0.25], [0.3, 0.3], [0.3, 0.35], [0.3, 0.4], [0.35, 0.05], [0.35, 0.1], [0.35, 0.15], [0.35, 0.2], [0.35, 0.25], [0.35, 0.3], [0.35, 0.35], [0.35, 0.4],  [0.4, 0.05], [0.4, 0.1], [0.4, 0.15], [0.4, 0.2], [0.4, 0.25], [0.4, 0.3], [0.4, 0.35], [0.4, 0.4]  ]

Q0=[
   [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
   [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5],
   [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
   [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
   [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5],
   [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
 ]
#Q0=[
  #[0, 0], [0, 1], [0, 2], [0, 3],
  #[1, 0], [1, 1], [1, 2], [1, 3],
  #[2, 0], [2, 1], [2, 2], [2, 3],
  #[3, 0], [3, 1], [3, 2], [3, 3],
#]
# P=[[0.05, 0.05], [0.1, 0.05], [0.1, 0.1], [0.15, 0.05], [0.15, 0.1], [0.15, 0.15], [0.2, 0.05], [0.2, 0.1], [0.2, 0.15], [0.2, 0.2], [0.25, 0.05], [0.25, 0.1], [0.25, 0.15], [0.25, 0.2], [0.25, 0.25], [0.3, 0.05], [0.3, 0.1], [0.3, 0.15], [0.3, 0.2], [0.3, 0.25], [0.3, 0.3], [0.35, 0.05], [0.35, 0.1], [0.35, 0.15], [0.35, 0.2], [0.35, 0.25], [0.35, 0.3], [0.35, 0.35], [0.4, 0.05], [0.4, 0.1], [0.4, 0.15], [0.4, 0.2], [0.4, 0.25], [0.4, 0.3], [0.4, 0.35], [0.4, 0.4], [0.45, 0.05], [0.45, 0.1], [0.45, 0.15], [0.45, 0.2], [0.45, 0.25], [0.45, 0.3], [0.45, 0.35], [0.45, 0.4], [0.45, 0.45], [0.5, 0.05], [0.5, 0.1], [0.5, 0.15], [0.5, 0.2], [0.5, 0.25], [0.5, 0.3], [0.5, 0.35], [0.5, 0.4], [0.5, 0.45], [0.5, 0.5], [0.55, 0.05], [0.55, 0.1], [0.55, 0.15], [0.55, 0.2], [0.55, 0.25], [0.55, 0.3], [0.55, 0.35], [0.55, 0.4], [0.55, 0.45], [0.55, 0.5], [0.55, 0.55], [0.6, 0.05], [0.6, 0.1], [0.6, 0.15], [0.6, 0.2], [0.6, 0.25], [0.6, 0.3], [0.6, 0.35], [0.6, 0.4], [0.6, 0.45], [0.6, 0.5], [0.6, 0.55], [0.6, 0.6], [0.65, 0.05], [0.65, 0.1], [0.65, 0.15], [0.65, 0.2], [0.65, 0.25], [0.65, 0.3], [0.65, 0.35], [0.65, 0.4], [0.65, 0.45], [0.65, 0.5], [0.65, 0.55], [0.65, 0.6], [0.65, 0.65], [0.7, 0.05], [0.7, 0.1], [0.7, 0.15], [0.7, 0.2], [0.7, 0.25], [0.7, 0.3], [0.7, 0.35], [0.7, 0.4], [0.7, 0.45], [0.7, 0.5], [0.7, 0.55], [0.7, 0.6], [0.7, 0.65], [0.7, 0.7], [0.75, 0.05], [0.75, 0.1], [0.75, 0.15], [0.75, 0.2], [0.75, 0.25], [0.75, 0.3], [0.75, 0.35], [0.75, 0.4], [0.75, 0.45], [0.75, 0.5], [0.75, 0.55], [0.75, 0.6], [0.75, 0.65], [0.75, 0.7], [0.75, 0.75], [0.8, 0.05], [0.8, 0.1], [0.8, 0.15], [0.8, 0.2], [0.8, 0.25], [0.8, 0.3], [0.8, 0.35], [0.8, 0.4], [0.8, 0.45], [0.8, 0.5], [0.8, 0.55], [0.8, 0.6], [0.8, 0.65], [0.8, 0.7], [0.8, 0.75], [0.8, 0.8], [0.85, 0.05], [0.85, 0.1], [0.85, 0.15], [0.85, 0.2], [0.85, 0.25], [0.85, 0.3], [0.85, 0.35], [0.85, 0.4], [0.85, 0.45], [0.85, 0.5], [0.85, 0.55], [0.85, 0.6], [0.85, 0.65], [0.85, 0.7], [0.85, 0.75], [0.85, 0.8], [0.85, 0.85], [0.9, 0.05], [0.9, 0.1], [0.9, 0.15], [0.9, 0.2], [0.9, 0.25], [0.9, 0.3], [0.9, 0.35], [0.9, 0.4], [0.9, 0.45], [0.9, 0.5], [0.9, 0.55], [0.9, 0.6], [0.9, 0.65], [0.9, 0.7], [0.9, 0.75], [0.9, 0.8], [0.9, 0.85], [0.9, 0.9], [0.95, 0.05], [0.95, 0.1], [0.95, 0.15], [0.95, 0.2], [0.95, 0.25], [0.95, 0.3], [0.95, 0.35], [0.95, 0.4], [0.95, 0.45], [0.95, 0.5], [0.95, 0.55], [0.95, 0.6], [0.95, 0.65], [0.95, 0.7], [0.95, 0.75], [0.95, 0.8], [0.95, 0.85], [0.95, 0.9], [0.95, 0.95], [1, 0.05], [1, 0.1], [1, 0.15], [1, 0.2], [1, 0.25], [1, 0.3], [1, 0.35], [1, 0.4], [1, 0.45], [1, 0.5], [1, 0.55], [1, 0.6], [1, 0.65], [1, 0.7], [1, 0.75], [1, 0.8], [1, 0.85], [1, 0.9], [1, 0.95], [1, 1]]
# Q0=[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [10, 0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10]]
# P=[[0.05, 0.05], [0.05, 0.1], [0.05, 0.15], [0.05, 0.2], [0.05, 0.25], [0.05, 0.3], [0.05, 0.35], [0.05, 0.4], [0.05, 0.45], [0.05, 0.5], [0.05, 0.55], [0.05, 0.6], [0.05, 0.65], [0.05, 0.7], [0.05, 0.75], [0.05, 0.8], [0.05, 0.85], [0.05, 0.9], [0.05, 0.95], [0.05, 1.0], [0.1, 0.05], [0.1, 0.1], [0.1, 0.15], [0.1, 0.2], [0.1, 0.25], [0.1, 0.3], [0.1, 0.35], [0.1, 0.4], [0.1, 0.45], [0.1, 0.5], [0.1, 0.55], [0.1, 0.6], [0.1, 0.65], [0.1, 0.7], [0.1, 0.75], [0.1, 0.8], [0.1, 0.85], [0.1, 0.9], [0.1, 0.95], [0.1, 1.0], [0.15, 0.05], [0.15, 0.1], [0.15, 0.15], [0.15, 0.2], [0.15, 0.25], [0.15, 0.3], [0.15, 0.35], [0.15, 0.4], [0.15, 0.45], [0.15, 0.5], [0.15, 0.55], [0.15, 0.6], [0.15, 0.65], [0.15, 0.7], [0.15, 0.75], [0.15, 0.8], [0.15, 0.85], [0.15, 0.9], [0.15, 0.95], [0.15, 1.0], [0.2, 0.05], [0.2, 0.1], [0.2, 0.15], [0.2, 0.2], [0.2, 0.25], [0.2, 0.3], [0.2, 0.35], [0.2, 0.4], [0.2, 0.45], [0.2, 0.5], [0.2, 0.55], [0.2, 0.6], [0.2, 0.65], [0.2, 0.7], [0.2, 0.75], [0.2, 0.8], [0.2, 0.85], [0.2, 0.9], [0.2, 0.95], [0.2, 1.0], [0.25, 0.05], [0.25, 0.1], [0.25, 0.15], [0.25, 0.2], [0.25, 0.25], [0.25, 0.3], [0.25, 0.35], [0.25, 0.4], [0.25, 0.45], [0.25, 0.5], [0.25, 0.55], [0.25, 0.6], [0.25, 0.65], [0.25, 0.7], [0.25, 0.75], [0.25, 0.8], [0.25, 0.85], [0.25, 0.9], [0.25, 0.95], [0.25, 1.0], [0.3, 0.05], [0.3, 0.1], [0.3, 0.15], [0.3, 0.2], [0.3, 0.25], [0.3, 0.3], [0.3, 0.35], [0.3, 0.4], [0.3, 0.45], [0.3, 0.5], [0.3, 0.55], [0.3, 0.6], [0.3, 0.65], [0.3, 0.7], [0.3, 0.75], [0.3, 0.8], [0.3, 0.85], [0.3, 0.9], [0.3, 0.95], [0.3, 1.0], [0.35, 0.05], [0.35, 0.1], [0.35, 0.15], [0.35, 0.2], [0.35, 0.25], [0.35, 0.3], [0.35, 0.35], [0.35, 0.4], [0.35, 0.45], [0.35, 0.5], [0.35, 0.55], [0.35, 0.6], [0.35, 0.65], [0.35, 0.7], [0.35, 0.75], [0.35, 0.8], [0.35, 0.85], [0.35, 0.9], [0.35, 0.95], [0.35, 1.0], [0.4, 0.05], [0.4, 0.1], [0.4, 0.15], [0.4, 0.2], [0.4, 0.25], [0.4, 0.3], [0.4, 0.35], [0.4, 0.4], [0.4, 0.45], [0.4, 0.5], [0.4, 0.55], [0.4, 0.6], [0.4, 0.65], [0.4, 0.7], [0.4, 0.75], [0.4, 0.8], [0.4, 0.85], [0.4, 0.9], [0.4, 0.95], [0.4, 1.0], [0.45, 0.05], [0.45, 0.1], [0.45, 0.15], [0.45, 0.2], [0.45, 0.25], [0.45, 0.3], [0.45, 0.35], [0.45, 0.4], [0.45, 0.45], [0.45, 0.5], [0.45, 0.55], [0.45, 0.6], [0.45, 0.65], [0.45, 0.7], [0.45, 0.75], [0.45, 0.8], [0.45, 0.85], [0.45, 0.9], [0.45, 0.95], [0.45, 1.0], [0.5, 0.05], [0.5, 0.1], [0.5, 0.15], [0.5, 0.2], [0.5, 0.25], [0.5, 0.3], [0.5, 0.35], [0.5, 0.4], [0.5, 0.45], [0.5, 0.5], [0.5, 0.55], [0.5, 0.6], [0.5, 0.65], [0.5, 0.7], [0.5, 0.75], [0.5, 0.8], [0.5, 0.85], [0.5, 0.9], [0.5, 0.95], [0.5, 1.0], [0.55, 0.05], [0.55, 0.1], [0.55, 0.15], [0.55, 0.2], [0.55, 0.25], [0.55, 0.3], [0.55, 0.35], [0.55, 0.4], [0.55, 0.45], [0.55, 0.5], [0.55, 0.55], [0.55, 0.6], [0.55, 0.65], [0.55, 0.7], [0.55, 0.75], [0.55, 0.8], [0.55, 0.85], [0.55, 0.9], [0.55, 0.95], [0.55, 1.0], [0.6, 0.05], [0.6, 0.1], [0.6, 0.15], [0.6, 0.2], [0.6, 0.25], [0.6, 0.3], [0.6, 0.35], [0.6, 0.4], [0.6, 0.45], [0.6, 0.5], [0.6, 0.55], [0.6, 0.6], [0.6, 0.65], [0.6, 0.7], [0.6, 0.75], [0.6, 0.8], [0.6, 0.85], [0.6, 0.9], [0.6, 0.95], [0.6, 1.0], [0.65, 0.05], [0.65, 0.1], [0.65, 0.15], [0.65, 0.2], [0.65, 0.25], [0.65, 0.3], [0.65, 0.35], [0.65, 0.4], [0.65, 0.45], [0.65, 0.5], [0.65, 0.55], [0.65, 0.6], [0.65, 0.65], [0.65, 0.7], [0.65, 0.75], [0.65, 0.8], [0.65, 0.85], [0.65, 0.9], [0.65, 0.95], [0.65, 1.0], [0.7, 0.05], [0.7, 0.1], [0.7, 0.15], [0.7, 0.2], [0.7, 0.25], [0.7, 0.3], [0.7, 0.35], [0.7, 0.4], [0.7, 0.45], [0.7, 0.5], [0.7, 0.55], [0.7, 0.6], [0.7, 0.65], [0.7, 0.7], [0.7, 0.75], [0.7, 0.8], [0.7, 0.85], [0.7, 0.9], [0.7, 0.95], [0.7, 1.0], [0.75, 0.05], [0.75, 0.1], [0.75, 0.15], [0.75, 0.2], [0.75, 0.25], [0.75, 0.3], [0.75, 0.35], [0.75, 0.4], [0.75, 0.45], [0.75, 0.5], [0.75, 0.55], [0.75, 0.6], [0.75, 0.65], [0.75, 0.7], [0.75, 0.75], [0.75, 0.8], [0.75, 0.85], [0.75, 0.9], [0.75, 0.95], [0.75, 1.0], [0.8, 0.05], [0.8, 0.1], [0.8, 0.15], [0.8, 0.2], [0.8, 0.25], [0.8, 0.3], [0.8, 0.35], [0.8, 0.4], [0.8, 0.45], [0.8, 0.5], [0.8, 0.55], [0.8, 0.6], [0.8, 0.65], [0.8, 0.7], [0.8, 0.75], [0.8, 0.8], [0.8, 0.85], [0.8, 0.9], [0.8, 0.95], [0.8, 1.0], [0.85, 0.05], [0.85, 0.1], [0.85, 0.15], [0.85, 0.2], [0.85, 0.25], [0.85, 0.3], [0.85, 0.35], [0.85, 0.4], [0.85, 0.45], [0.85, 0.5], [0.85, 0.55], [0.85, 0.6], [0.85, 0.65], [0.85, 0.7], [0.85, 0.75], [0.85, 0.8], [0.85, 0.85], [0.85, 0.9], [0.85, 0.95], [0.85, 1.0], [0.9, 0.05], [0.9, 0.1], [0.9, 0.15], [0.9, 0.2], [0.9, 0.25], [0.9, 0.3], [0.9, 0.35], [0.9, 0.4], [0.9, 0.45], [0.9, 0.5], [0.9, 0.55], [0.9, 0.6], [0.9, 0.65], [0.9, 0.7], [0.9, 0.75], [0.9, 0.8], [0.9, 0.85], [0.9, 0.9], [0.9, 0.95], [0.9, 1.0], [0.95, 0.05], [0.95, 0.1], [0.95, 0.15], [0.95, 0.2], [0.95, 0.25], [0.95, 0.3], [0.95, 0.35], [0.95, 0.4], [0.95, 0.45], [0.95, 0.5], [0.95, 0.55], [0.95, 0.6], [0.95, 0.65], [0.95, 0.7], [0.95, 0.75], [0.95, 0.8], [0.95, 0.85], [0.95, 0.9], [0.95, 0.95], [0.95, 1.0], [1.0, 0.05], [1.0, 0.1], [1.0, 0.15], [1.0, 0.2], [1.0, 0.25], [1.0, 0.3], [1.0, 0.35], [1.0, 0.4], [1.0, 0.45], [1.0, 0.5], [1.0, 0.55], [1.0, 0.6], [1.0, 0.65], [1.0, 0.7], [1.0, 0.75], [1.0, 0.8], [1.0, 0.85], [1.0, 0.9], [1.0, 0.95], [1.0, 1.0]]

# P1 = (0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7);
# argmax fonksiyonu maksimum degerin konumunu getirir, orn: L=[[1,2,3],[3,4,5],[9,999,8]] argmax(L)=2
print(len(P))



# kod her bir w degeri icin calisiyor.
for w in W:

    # tetalarin tutuldugu liste, her bir satir q1'leri ifade ederken, her bir sutun da k'lari ifade eder. teta=[[1,2,3],[3,4,5],[9,999,8]] olsun. q1=0 ve k=2 icin, teta degeri teta[0][2]=3'tur
    teta = [[[] for i in range(u_q0)] for i in range(u_q0)]

    # teta[q1][k]-teta[q1][k-1]'in tutuldugu listedir, durma kosulu icin kullanilir. while dongusunun calisabilmesii icin ilk basta elemanlari 9 ama farklari epsilona yakinsayinca 9 siliniyor.
    ##duzenlemedim
    tetafark = [[[9] for i in range(u_q0)] for i in range(u_q0)]

    k = 1
    # tetanin ilk elemanini yani her q1 icin k=0'i 0'a esitliyor.
    for q1_a in range(a_q0, len(teta)):
        for q1_b in range(a_q0, len(teta)):
            teta[q1_a][q1_b].insert(0, 0)

        # teta[0].insert(0, 0)

    print("w1=", w, "w2=", 1 - w, "p0", p0, "icin")

    # while (teta[q1][k] - teta[q1][k -1]) > 0.0001:
    # farklarin maksimumu epsilondan buyuk olduguu surece calisiyor.
    while findMaxNumber(tetafark)['listMaxValue'] > epsilon:
        P1_i = [0];
        Q0_i = [0];
        QS = [0]
        # Lm=[[[],[],...[]],[[],[],...[]]....] seklinde bir liste, L[q1][p1][q0] sekinde temsil ediliyor, her sistem durumu q1 icin ve KD'lere gore AF tutuluyor. Daha sonra optimal KD degerlerini bulbilmek icin.
        # Lm =[[[]for i in range(u_q0)] for i in range(u_q0)]; KD_p1= [[]for i in range(u_q0) ];KD_q0= [[[]for i in range(u_q0)] for i in range(u_q0)];KD_qs= [[[]for i in range(u_q0)] for i in range(u_q0)]
        Lm = [[[] for i in range(u_q0)] for i in range(u_q0)];
        KD_p1 = [[] for i in range(u_q0)];
        KD_q0 = [[] for i in range(u_q0)];
        KD_qs = [[] for i in range(u_q0)]

        # # her sistem durumu icin kod hesaplamalar yapiliyor
        for q1_a in range(a_q0, u_q0):
            for q1_b in range(a_q0, u_q0):

                # print("q1", q1)
                # L: her bir sistem durumu icin yenileniyor ve karar degiskenlerine gore AF degerlerinin tutuldugu bir liste L[p1][q0] matematiksel ifadesi--> g(i,u)+∑p_i_j(u)h_k(j)
                L = -1;
                # L1: her bir sistem durumu icin yenileniyor. referans noktasi=0 icin tutuluyor ve karar degiskenlerine gore AF degerlerinin tutuldugu bir liste L[p1][q0] matematiksel ifadesi--> g(s,u)+∑p_s_j(u)h_k(j)
                L1 = -1;

                # p1 karar degiskeni

                for p in range(0, len(P)):

                    for q0 in range(0, len(Q0)):
                        # q0 karar degiskeni
                        for qs in range(-q1_a, q1_b + 1):
                            # SumAF: AF'nin  beklenen degerli kismi; SumAF1: referans noktasi icin
                            SumAF = SumAF1 = 0;
                            # he bir KD icin Amac fonksiyonu
                            AF1 = AF = 0
                            for d0_a in range(0, u_q0):
                                for d1_a in range(0, u_q0):
                                    for d0_b in range(0, u_q0):
                                        for d1_b in range(0, u_q0):
                                            # alfa0 = 1 - p1 / p0
                                            # alfa1 = p1 / p0

                                            # ilk durum; p0<delta oldugu icin ve suanki p1 degerlerine gore model hep ilk durumun kosulunu sagliyor, ikinci duruma girmiyor.
                                            #if P[p][0] / delta_a <= p0 <= (p0 - P[p][0]) / (1 - delta_a):
                                            if  p0 <= (p0 - P[p][0]) / (1 - delta_a):
                                                v0_a = uniform.sf((p0 - P[p][0]) / (1 - delta_a))

                                                v2_a = uniform.cdf(P[p][0] / delta_a)
                                                v1_a = uniform.cdf((p0 - P[p][0]) / (1 - delta_a)) - v2_a
                                                if v1_a == 0:
                                                    if d1_a > 0:
                                                        olasilik1_a = 0
                                                    else:
                                                        olasilik1_a = 1
                                                else:

                                                    olasilik1_a = binom.pmf(d1_a, N_a, v1_a)
                                                if v0_a == 0:
                                                    if d0_a > 0:
                                                        olasilik0_a = 0
                                                    else:
                                                        olasilik0_a = 1
                                                else:

                                                    olasilik0_a = binom.pmf(d0_a, N_a, v0_a)
                                            # ikinci durum
                                            #if (p0 - P[p][0]) / (1 - delta_a) < p0 < P[p][0] / delta_a:
                                            if  (p0 - P[p][0]) / (1 - delta_a) < p0 :
                                                v0_a = uniform.sf(P[p][0])
                                                v2_a = uniform.cdf(P[p][0])
                                                v1_a = 0

                                                # Ortalama(M) ve varyans(V)
                                                if v0_a == 0:
                                                    if d0_a > 0:
                                                        olasilik0_a = 0
                                                    else:
                                                        olasilik0_a = 1
                                                else:

                                                    olasilik0_a = binom.pmf(d0_a, N_a, v0_a)
                                                if d1_a > 0:
                                                    olasilik1_a = 0
                                                else:
                                                    olasilik1_a = 1
                                            # ilk durum
                                            #if P[p][1] / delta_b <= p0 <= (p0 - P[p][1]) / (1 - delta_b):
                                            if p0 <= (p0 - P[p][1]) / (1 - delta_b):
                                                v0_b = uniform.sf((p0 - P[p][1]) / (1 - delta_b))

                                                v2_b = uniform.cdf(P[p][1] / delta_b)
                                                v1_b = uniform.cdf((p0 - P[p][1]) / (1 - delta_b)) - v2_b
                                                if v1_b == 0:
                                                    if d1_b > 0:
                                                        olasilik1_b = 0
                                                    else:
                                                        olasilik1_b = 1
                                                else:

                                                    olasilik1_b = binom.pmf(d1_b, N_b, v1_b)
                                                if v0_b == 0:
                                                    if d0_b > 0:
                                                        olasilik0_b = 0
                                                    else:
                                                        olasilik0_b = 1
                                                else:

                                                    olasilik0_b = binom.pmf(d0_b, N_b, v0_b)
                                            # ikinci durum
                                            #if (p0 - P[p][1]) / (1 - delta_b) < p0 < P[p][1] / delta_b:
                                            if (p0 - P[p][1]) / (1 - delta_b) < p0 :
                                                v0_b = uniform.sf(P[p][1])
                                                v2_b = uniform.cdf(P[p][1])
                                                v1_b = 0
                                                # Ortalama(M) ve varyans(V)
                                                if v0_b == 0:
                                                    if d0_b > 0:
                                                        olasilik0_b = 0
                                                    else:
                                                        olasilik0_b = 1
                                                else:

                                                    olasilik0_b = binom.pmf(d0_b, N_b, v0_b)
                                                if d1_b > 0:
                                                    olasilik1_b = 0
                                                else:
                                                    olasilik1_b = 1

                                            # AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                            # w=0.1, p1=0.2 q1=2, q0=0, d0=1 ve d1=0 iken olasilik0=0 ama boyle bir durumda -w*max(0,2-0) maliyetini goz ardi etmis oluyoruz. elimizde eski urunden var talebi 0 ise maliyet olusmali
                                            alfa0_a = 1 - 0.5 * (P[p][0] / p0)
                                            alfa1_a = (P[p][0] / p0)
                                            alfa0_b = 1 - 0.5 * (P[p][1] / p0)
                                            alfa1_b = (P[p][1] / p0)

                                            e = 1

                                            if (d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) <= 0) and (
                                                    d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) <= 0):
                                                d01_a = 0;
                                                olasilik01_a = 1;
                                                d10_a = 0;
                                                olasilik10_a = 1;
                                                d01_b = 0;
                                                olasilik01_b = 1;
                                                d10_b = 0;
                                                olasilik10_b = 1;

                                                SumOlasilik = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                          w * (P[p][0] * min(d1_a + d01_a, q1_a + qs) +
                                                                               P[p][1] * min(d1_b + d01_b,q1_b - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1])))
                                                                          - (1 - w) * (max(0, q1_a - (d1_a + d01_a) + qs) + max(0, q1_b - (d1_b + d01_b) - qs)) +
                                                                          teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                SumAF = SumOlasilik + SumAF

                                                # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                # olasiliklarla carpildiktan sonra toplaniyor

                                                SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + 0) + P[p][1] * min(d1_b + d01_b, 0 - 0) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + 0) + max(0, 0 - (d1_b + d01_b) - 0)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                SumAF1 = SumOlasilik1 + SumAF1
                                                # print("IF1","p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p], "q0",
                                                # Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            elif (d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) > 0) and (
                                                    d0_b - Q0[q0][1] > 0) and (d1_b - (q1_b-qs) <= 0):
                                                d01_a = 0;
                                                olasilik01_a = 1;

                                                d10_b = 0;
                                                olasilik10_b = 1;

                                                for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                    for d01_b in range(0, d0_b- Q0[q0][1] + e):
                                                        olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a)
                                                        olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1], alfa0_b)
                                                        SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,q1_a + qs) + P[p][1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min((d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (max(0, q1_a - (d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF = SumOlasilik + SumAF

                                                # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                        #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF1 = SumOlasilik1 + SumAF1
                                            elif (d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) > 0) and (
                                                    d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) > 0):
                                                d01_a = 0;
                                                olasilik01_a = 1;

                                                d01_b = 0;
                                                olasilik01_b = 1;

                                                for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                    for d10_b in range(0, d1_b- (q1_b-qs) + e):
                                                        olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a)
                                                        olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b)
                                                        SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,q1_a + qs) + P[p][1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min((d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (max(0, q1_a - (d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF = SumOlasilik + SumAF

                                                # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                        SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                        # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF1 = SumOlasilik1 + SumAF1


                                            elif (d0_a - Q0[q0][0] > 0) and (d1_a - (q1_a+qs) <= 0) and (
                                                    d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) > 0):
                                                d10_a = 0;
                                                olasilik10_a = 1;

                                                d01_b = 0;
                                                olasilik01_b = 1;

                                                for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                    for d10_b in range(0, d1_b- (q1_b-qs) + e):
                                                        olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0], alfa0_a)
                                                        olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b)
                                                        SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,q1_a + qs) + P[p][1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min((d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (max(0, q1_a - (d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF = SumOlasilik + SumAF

                                                # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                        #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF1 = SumOlasilik1 + SumAF1

                                            elif (d0_a - Q0[q0][0] > 0) and (d1_a - (q1_a+qs) <= 0) and (
                                                    d0_b - Q0[q0][1] > 0) and (d1_b - (q1_b-qs) <= 0):
                                                d10_a = 0;
                                                olasilik10_a = 1;

                                                d10_b = 0;
                                                olasilik10_b = 1;

                                                for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                    for d01_b in range(0, d0_b- Q0[q0][1] + e):
                                                        olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0], alfa0_a)
                                                        olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1], alfa0_b)
                                                        SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,q1_a + qs) + P[p][1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min((d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (max(0, q1_a - (d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF = SumOlasilik + SumAF

                                                # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                        #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF1 = SumOlasilik1 + SumAF1

                                            elif (d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) > 0) and (
                                                    d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) <= 0):
                                                d01_a = 0;
                                                olasilik01_a = 1;
                                                d01_b = 0;
                                                olasilik01_b = 1;
                                                d10_b = 0;
                                                olasilik10_b = 1;

                                                for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                    olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a)
                                                    SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,
                                                                                                 q1_a + qs) + P[p][
                                                                                       1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          0] - (
                                                                                                                                  d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (
                                                                                               min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min(
                                                                                           (d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (
                                                                                          max(0, q1_a - (
                                                                                                      d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][
                                                                                  k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF = SumOlasilik + SumAF

                                                # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                    #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF1 = SumOlasilik1 + SumAF1
                                            # print("ELIF1", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p], "q0",
                                            # Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)

                                            elif (d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) > 0) and (
                                                    d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) <= 0):
                                                d01_b = 0;
                                                olasilik01_b = 1;
                                                d01_a = 0;
                                                olasilik01_a = 1;
                                                d10_a = 0;
                                                olasilik10_a = 1;

                                                for d10_b in range(0, d1_b - (q1_b-qs) + e):
                                                    olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b)
                                                    SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,
                                                                                                 q1_a + qs) + P[p][
                                                                                       1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          0] - (
                                                                                                                                  d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (
                                                                                               min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min(
                                                                                           (d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (
                                                                                          max(0, q1_a - (
                                                                                                      d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][
                                                                                  k - 1])
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF = SumOlasilik + SumAF

                                                    # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                    #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF2", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]",
                                                #           P[p], "q0",
                                                #           Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)

                                            elif (d0_a - Q0[q0][0] > 0) and (d1_a - (q1_a+qs) <= 0) and (
                                                    d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) <= 0):
                                                d10_a = 0;
                                                olasilik10_a = 1;
                                                d01_b = 0;
                                                olasilik01_b = 1;
                                                d10_b = 0;
                                                olasilik10_b = 1;
                                                for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                    olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0], alfa0_a)

                                                    SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,
                                                                                                 q1_a + qs) + P[p][
                                                                                       1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          0] - (
                                                                                                                                  d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (
                                                                                               min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min(
                                                                                           (d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (
                                                                                          max(0, q1_a - (
                                                                                                      d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][
                                                                                  k - 1])
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF = SumOlasilik + SumAF

                                                    # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                    #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF3", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)

                                            elif (d0_b - Q0[q0][1] > 0) and (d1_b - (q1_b-qs) <= 0) and (
                                                    d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) <= 0):
                                                d10_b = 0;
                                                olasilik10_b = 1;
                                                d01_a = 0;
                                                olasilik01_a = 1;
                                                d10_a = 0;
                                                olasilik10_a = 1;
                                                for d01_b in range(0, d0_b - Q0[q0][1] + e):
                                                    olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1], alfa0_b)

                                                    SumOlasilik = (
                                                                              olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                              w * (P[p][0] * min(d1_a + d01_a,
                                                                                                 q1_a + qs) + P[p][
                                                                                       1] * min(d1_b + d01_b,
                                                                                                q1_b - qs) - h * (max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          0] - (
                                                                                                                                  d0_a + d10_a)) + max(
                                                                          0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (
                                                                                               min((d0_a + d10_a),
                                                                                                   Q0[q0][0]) + min(
                                                                                           (d0_b + d10_b),
                                                                                           Q0[q0][1]))) - (1 - w) * (
                                                                                          max(0, q1_a - (
                                                                                                      d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                              teta[max(0, Q0[q0][0] - (d0_a + d10_a))][
                                                                                  max(0, Q0[q0][1] - (d0_b + d10_b))][
                                                                                  k - 1])
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF = SumOlasilik + SumAF

                                                    # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                    #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                    # olasiliklarla carpildiktan sonra toplaniyor
                                                    SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF4", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            elif (d0_b - Q0[q0][1] > 0) and (d1_b - (q1_b-qs) > 0) and (
                                                    d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) <= 0):

                                                d01_a = 0;
                                                olasilik01_a = 1;
                                                d10_a = 0;
                                                olasilik10_a = 1;
                                                for d01_b in range(0, d0_b - Q0[q0][1] + e):
                                                    for d10_b in range(0, d1_b - (q1_b-qs) + e):
                                                        olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1], alfa0_b)
                                                        olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b)

                                                        SumOlasilik = (
                                                                                  olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                                  w * (P[p][0] * min(d1_a + d01_a,
                                                                                                     q1_a + qs) + P[p][
                                                                                           1] * min(d1_b + d01_b,
                                                                                                    q1_b - qs) - h * (
                                                                                                   max(0, Q0[q0][0] - (
                                                                                                               d0_a + d10_a)) + max(
                                                                                               0, Q0[q0][1] - (
                                                                                                           d0_b + d10_b))) + p0 * (
                                                                                                   min((d0_a + d10_a),
                                                                                                       Q0[q0][0]) + min(
                                                                                               (d0_b + d10_b),
                                                                                               Q0[q0][1]))) - (
                                                                                              1 - w) * (max(0, q1_a - (
                                                                                      d1_a + d01_a) + qs) + max(0,
                                                                                                                q1_b - (
                                                                                                                            d1_b + d01_b) - qs)) +
                                                                                  teta[max(0,
                                                                                           Q0[q0][0] - (d0_a + d10_a))][
                                                                                      max(0,
                                                                                          Q0[q0][1] - (d0_b + d10_b))][
                                                                                      k - 1])
                                                        # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF = SumOlasilik + SumAF

                                                        # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                        # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                        #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF5", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            elif (d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) <= 0) and (
                                                    d0_a - Q0[q0][0] > 0) and (d1_a - (q1_a+qs) > 0):

                                                d01_b = 0;
                                                olasilik01_b = 1;
                                                d10_b = 0;
                                                olasilik10_b = 1;
                                                for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                    for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                        olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0], alfa0_a)
                                                        olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a)

                                                        SumOlasilik = (
                                                                                  olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                                  w * (P[p][0] * min(d1_a + d01_a,
                                                                                                     q1_a + qs) + P[p][
                                                                                           1] * min(d1_b + d01_b,
                                                                                                    q1_b - qs) - h * (
                                                                                                   max(0, Q0[q0][0] - (
                                                                                                               d0_a + d10_a)) + max(
                                                                                               0, Q0[q0][1] - (
                                                                                                           d0_b + d10_b))) + p0 * (
                                                                                                   min((d0_a + d10_a),
                                                                                                       Q0[q0][0]) + min(
                                                                                               (d0_b + d10_b),
                                                                                               Q0[q0][1]))) - (
                                                                                              1 - w) * (max(0, q1_a - (
                                                                                      d1_a + d01_a) + qs) + max(0,
                                                                                                                q1_b - (
                                                                                                                            d1_b + d01_b) - qs)) +
                                                                                  teta[max(0,
                                                                                           Q0[q0][0] - (d0_a + d10_a))][
                                                                                      max(0,
                                                                                          Q0[q0][1] - (d0_b + d10_b))][
                                                                                      k - 1])
                                                        # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF = SumOlasilik + SumAF

                                                        # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                        # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                        #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                        # olasiliklarla carpildiktan sonra toplaniyor
                                                        SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF6", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            elif (d0_b - Q0[q0][1] <= 0) and (d1_b - (q1_b-qs) > 0) and (
                                                    d0_a - Q0[q0][0] > 0) and (d1_a - (q1_a+qs) > 0):
                                                d01_b = 0;
                                                olasilik01_b = 1;
                                                for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                    for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                        for d10_b in range(0, d1_b - (q1_b-qs) + e):
                                                            olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0],
                                                                                     alfa0_a)
                                                            olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a)
                                                            olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b)

                                                            SumOlasilik = (
                                                                                      olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                                      w * (P[p][0] * min(d1_a + d01_a,
                                                                                                         q1_a + qs) +
                                                                                           P[p][1] * min(d1_b + d01_b,
                                                                                                         q1_b - qs) - h * (
                                                                                                       max(0,
                                                                                                           Q0[q0][0] - (
                                                                                                                       d0_a + d10_a)) + max(
                                                                                                   0, Q0[q0][1] - (
                                                                                                               d0_b + d10_b))) + p0 * (
                                                                                                       min((
                                                                                                                       d0_a + d10_a),
                                                                                                           Q0[q0][
                                                                                                               0]) + min(
                                                                                                   (d0_b + d10_b),
                                                                                                   Q0[q0][1]))) - (
                                                                                                  1 - w) * (max(0,
                                                                                                                q1_a - (
                                                                                                                            d1_a + d01_a) + qs) + max(
                                                                                  0, q1_b - (d1_b + d01_b) - qs)) +
                                                                                      teta[max(0, Q0[q0][0] - (
                                                                                                  d0_a + d10_a))][max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          1] - (
                                                                                                                                  d0_b + d10_b))][
                                                                                          k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF = SumOlasilik + SumAF

                                                            # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                            #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF7", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            elif (d0_b - Q0[q0][1] > 0) and (d1_b - (q1_b-qs) <= 0) and (
                                                    d0_a - Q0[q0][0] > 0) and (d1_a - (q1_a+qs) > 0):
                                                d10_b = 0;
                                                olasilik10_b = 1;
                                                for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                    for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                        for d01_b in range(0, d0_b - Q0[q0][1] + e):
                                                            olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0], alfa0_a)
                                                            olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a)
                                                            olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1], alfa0_b)

                                                            SumOlasilik = (
                                                                                      olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                                      w * (P[p][0] * min(d1_a + d01_a,
                                                                                                         q1_a + qs) +
                                                                                           P[p][1] * min(d1_b + d01_b,
                                                                                                         q1_b - qs) - h * (
                                                                                                       max(0,
                                                                                                           Q0[q0][0] - (
                                                                                                                       d0_a + d10_a)) + max(
                                                                                                   0, Q0[q0][1] - (
                                                                                                               d0_b + d10_b))) + p0 * (
                                                                                                       min((
                                                                                                                       d0_a + d10_a),
                                                                                                           Q0[q0][
                                                                                                               0]) + min(
                                                                                                   (d0_b + d10_b),
                                                                                                   Q0[q0][1]))) - (
                                                                                                  1 - w) * (max(0,
                                                                                                                q1_a - (
                                                                                                                            d1_a + d01_a) + qs) + max(
                                                                                  0, q1_b - (d1_b + d01_b) - qs)) +
                                                                                      teta[max(0, Q0[q0][0] - (
                                                                                                  d0_a + d10_a))][max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          1] - (
                                                                                                                                  d0_b + d10_b))][
                                                                                          k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF = SumOlasilik + SumAF

                                                            # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                            #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF8", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            elif (d0_b - Q0[q0][1] > 0) and (d1_b - (q1_b-qs) > 0) and (
                                                    d0_a - Q0[q0][0] <= 0) and (d1_a - (q1_a+qs) > 0):
                                                d01_a = 0;
                                                olasilik01_a = 1;
                                                for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                    for d10_b in range(0, d1_b - (q1_b-qs) + e):
                                                        for d01_b in range(0, d0_b - Q0[q0][1] + e):
                                                            olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a)
                                                            olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b)
                                                            olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1], alfa0_b)

                                                            SumOlasilik = (
                                                                                      olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                                      w * (P[p][0] * min(d1_a + d01_a,
                                                                                                         q1_a + qs) +
                                                                                           P[p][1] * min(d1_b + d01_b,
                                                                                                         q1_b - qs) - h * (
                                                                                                       max(0,
                                                                                                           Q0[q0][0] - (
                                                                                                                       d0_a + d10_a)) + max(
                                                                                                   0, Q0[q0][1] - (
                                                                                                               d0_b + d10_b))) + p0 * (
                                                                                                       min((
                                                                                                                       d0_a + d10_a),
                                                                                                           Q0[q0][
                                                                                                               0]) + min(
                                                                                                   (d0_b + d10_b),
                                                                                                   Q0[q0][1]))) - (
                                                                                                  1 - w) * (max(0,
                                                                                                                q1_a - (
                                                                                                                            d1_a + d01_a) + qs) + max(
                                                                                  0, q1_b - (d1_b + d01_b) - qs)) +
                                                                                      teta[max(0, Q0[q0][0] - (
                                                                                                  d0_a + d10_a))][max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          1] - (
                                                                                                                                  d0_b + d10_b))][
                                                                                          k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF = SumOlasilik + SumAF

                                                            # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                            #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF9", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            elif (d0_b - Q0[q0][1] > 0) and (d1_b - (q1_b-qs) > 0) and (
                                                    d0_a - Q0[q0][0] > 0) and (d1_a - (q1_a+qs) <= 0):
                                                d10_a = 0;
                                                olasilik10_a = 1;
                                                for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                    for d10_b in range(0, d1_b - (q1_b-qs) + e):
                                                        for d01_b in range(0, d0_b - Q0[q0][1] + e):
                                                            olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0], alfa0_a)
                                                            olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b)
                                                            olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1], alfa0_b)
                                                            # print("olasilik01_a,olasilik01_b,olasilik10_a,olasilik10_b",olasilik01_a,olasilik01_b,olasilik10_a,olasilik10_b)
                                                            SumOlasilik = (
                                                                                      olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                                      w * (P[p][0] * min(d1_a + d01_a,
                                                                                                         q1_a + qs) +
                                                                                           P[p][1] * min(d1_b + d01_b,
                                                                                                         q1_b - qs) - h * (
                                                                                                       max(0,
                                                                                                           Q0[q0][0] - (
                                                                                                                       d0_a + d10_a)) + max(
                                                                                                   0, Q0[q0][1] - (
                                                                                                               d0_b + d10_b))) + p0 * (
                                                                                                       min((
                                                                                                                       d0_a + d10_a),
                                                                                                           Q0[q0][
                                                                                                               0]) + min(
                                                                                                   (d0_b + d10_b),
                                                                                                   Q0[q0][1]))) - (
                                                                                                  1 - w) * (max(0,
                                                                                                                q1_a - (
                                                                                                                            d1_a + d01_a) + qs) + max(
                                                                                  0, q1_b - (d1_b + d01_b) - qs)) +
                                                                                      teta[max(0, Q0[q0][0] - (
                                                                                                  d0_a + d10_a))][max(0,
                                                                                                                      Q0[
                                                                                                                          q0][
                                                                                                                          1] - (
                                                                                                                                  d0_b + d10_b))][
                                                                                          k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF = SumOlasilik + SumAF

                                                            # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                            #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                            # olasiliklarla carpildiktan sonra toplaniyor
                                                            SumAF1 = SumOlasilik1 + SumAF1
                                                # print("ELIF10", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                                            else:
                                                for d10_a in range(0, d1_a - (q1_a+qs) + e):
                                                    for d01_a in range(0, d0_a - Q0[q0][0] + e):
                                                        for d10_b in range(0, d1_b - (q1_b-qs) + e):
                                                            for d01_b in range(0, d0_b - Q0[q0][1] + e):
                                                                olasilik10_a = binom.pmf(d10_a, d1_a - (q1_a+qs), alfa1_a);
                                                                olasilik01_a = binom.pmf(d01_a, d0_a - Q0[q0][0],
                                                                                         alfa0_a);
                                                                olasilik10_b = binom.pmf(d10_b, d1_b - (q1_b-qs), alfa1_b);
                                                                olasilik01_b = binom.pmf(d01_b, d0_b - Q0[q0][1],
                                                                                         alfa0_b)

                                                                SumOlasilik = (
                                                                                          olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (
                                                                                          w * (P[p][0] * min(
                                                                                      d1_a + d01_a, q1_a + qs) + P[p][
                                                                                                   1] * min(
                                                                                      d1_b + d01_b, q1_b - qs) - h * (
                                                                                                           max(0,
                                                                                                               Q0[q0][
                                                                                                                   0] - (
                                                                                                                           d0_a + d10_a)) + max(
                                                                                                       0, Q0[q0][1] - (
                                                                                                                   d0_b + d10_b))) + p0 * (
                                                                                                           min((
                                                                                                                           d0_a + d10_a),
                                                                                                               Q0[q0][
                                                                                                                   0]) + min(
                                                                                                       (d0_b + d10_b),
                                                                                                       Q0[q0][1]))) - (
                                                                                                      1 - w) * (max(0,
                                                                                                                    q1_a - (
                                                                                                                                d1_a + d01_a) + qs) + max(
                                                                                      0, q1_b - (d1_b + d01_b) - qs)) +
                                                                                          teta[max(0, Q0[q0][0] - (
                                                                                                      d0_a + d10_a))][
                                                                                              max(0, Q0[q0][1] - (
                                                                                                          d0_b + d10_b))][
                                                                                              k - 1])
                                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                                SumAF = SumOlasilik + SumAF

                                                                # referans noktasi icin: AF'nun beklenen deger hesaplamasi, olasiliklarla carpiliyor
                                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                                SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) + P[p][1] * min(d1_b + d01_b, 0 - qs) - h * (max(0, Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a), Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) + teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])

                                                                #SumOlasilik1 = (olasilik1_a * olasilik0_a * olasilik1_b * olasilik0_b * olasilik10_a * olasilik01_a * olasilik10_b * olasilik01_b) * (w * (P[p][0] * min(d1_a + d01_a, 0 + qs) +P[p][1] * min(d1_b + d01_b,0 - qs) - h * (max(0,Q0[q0][0] - (d0_a + d10_a)) + max(0, Q0[q0][1] - (d0_b + d10_b))) + p0 * (min((d0_a + d10_a),Q0[q0][0]) + min((d0_b + d10_b), Q0[q0][1]))) - (1 - w) * (max(0, 0 - (d1_a + d01_a) + qs) + max(0, 0 - (d1_b + d01_b) - qs)) +teta[max(0, Q0[q0][0] - (d0_a + d10_a))][max(0, Q0[q0][1] - (d0_b + d10_b))][k - 1])
                                                                # olasiliklarla carpildiktan sonra toplaniyor
                                                                SumAF1 = SumOlasilik1 + SumAF1
                                                                # print('en ice hesaplamalara girebildi2')
                                                                # print("2",'p',p,'q0',q0,'qs',qs)
                                                # print("ELIF11", "p0", p0, "q1_a", q1_a, "q1_b", q1_b, "P[p]", P[p],
                                                #       "q0",
                                                #       Q0[q0], "SumAF", SumAF,"d0_a",d0_a,"d1_a",d1_a,"d0_b",d0_b,"d1_b",d1_b)
                            # her bir KD icin Amac fonskiyonu hesaplaniyor,

                            AF = w * (-c * (Q0[q0][0] + Q0[q0][1]) - F * min(1, abs(qs))) + SumAF
                            AF1 = w * (-c * (Q0[q0][0] + Q0[q0][1]) - F * min(1, abs(qs)) ) + SumAF1

                            #AF1 = w * (-c * (Q0[q0][0] + Q0[q0][1])- F * min(1, abs(qs))) + SumAF1
                            # print("p0",p0,"q1_a",q1_a,"q1_b",q1_b,"P[p]",P[p],"q0",Q0[q0],"SumAF",SumAF)
                            # print("P[p]",P[p],"AF",AF,"L",L)

                            # print(P1_i)
                            # AF: her bir KD kombinasyonu icin hesaplanan amac fonksiyonu degeri. L ise o sistem durumu icin hesaplanan AF degeri. her bir KD kombinasyonu icin AF hesaplanip
                            #L ise kiyaslaniyor eger AF>=L ise o zaman L=AF oluyor yani en iyi amac fonk degeri guncelleniyor ve en iyi KD de listede tutuliyor.
                            if AF > L:
                                P1_i = P[p]
                                Q0_i = Q0[q0]
                                QS = qs
                                L = AF

                            if AF1 > L1:
                                L1 = AF1

                    # print('P1_i2', P1_i)
                # if skip_to_next_p0 == True:
                #     break
                # O andaki k ve q1 icin listeyi dolduruyor.
                # teta[q1_a][q1_b][1]= L - L1
                teta[q1_a][q1_b].insert(k, L - L1)
                # teta[q1].insert(k,  maxdegerL - maxdegerL1)
                Lm[q1_a][q1_b].append(L)

                KD_p1[q1_a].insert(q1_b, P1_i);
                KD_q0[q1_a].insert(q1_b, Q0_i);
                KD_qs[q1_a].insert(q1_b, QS)

            # print("Lm", Lm)
            # print("teta", teta)

            # print("teta", teta)

        print('q1 dongusu bitti, k=', k)

        # tum q1 lere gore hesaplamalar yapildigi icin burada tekrar q1'lere gore tetalarin yakinsayp yakinsamadigina bakiliyor
        for q1_A in range(a_q0, u_q0):
            for q1_B in range(a_q0, u_q0):
                if (abs(teta[q1_A][q1_B][k] - teta[q1_A][q1_B][k - 1])) < epsilon:

                    #     yakinsayan sistem durumlari icin farklarin tutuldugu liste yani tetafark'tan 9 elamanini sildiriyor ki durma sarti saglansin,

                    if tetafark[q1_A][q1_B][len(tetafark[q1_A][q1_B]) - 1] == 9:
                        tetafark[q1_A][q1_B].remove(9)

                    # tetafark listesinden 9 elemani silindikten sonra, listeye tetalarin farklari ekleniyor, en basta k=1 oldugu icin de k-1 kullanildi
                    tetafark[q1_A][q1_B].insert(k - 1, abs(teta[q1_A][q1_B][k] - teta[q1_A][q1_B][k - 1]))
                    # tetafark[q1_A][q1_B].insert(0, abs(teta[q1_A][q1_B][1] - teta[q1_A][q1_B][0]))
                # print("teta",teta)
        # durma sarti hala saglanmadiysa k=k+1 oluyor
        k = k + 1
        check = time.time()
        print('k', k, 'time', check - start)

        # print(tetafark)

    else:
        # durma sarti saglandi yani tetalarin farklarinin maksimumu epsilondan kucuk bi degere ulasti, bu nedenle optimum ort kar ve optimal KD bulunabilir.
        P1a_list = [];
        P1b_list = [];
        Q0a_list = [];
        Q0b_list = [];
        Qs_list = []
        for q1_A in range(a_q0, u_q0):
            for q1_B in range(a_q0, u_q0):
                print('KD_p1', KD_p1, KD_p1[q1_A][q1_B][0], KD_p1[q1_A][q1_B][1])
                optp1_a = KD_p1[q1_A][q1_B][0]
                optp1_b = KD_p1[q1_A][q1_B][1]
                optq0_a = KD_q0[q1_A][q1_B][0]
                optq0_b = KD_q0[q1_A][q1_B][1]
                optqs = KD_qs[q1_A][q1_B]
                P1a_list.append(optp1_a);
                P1b_list.append(optp1_b)
                Q0a_list.append(optq0_a);
                Q0b_list.append(optq0_b);
                Qs_list.append(optqs);
                #print("w", w, "p0", p0, "k", k - 1, "q1_a=", q1_A, "icin", "q1_b=", q1_B, "icin",
                      #" A subesi optimal indirimli fiyat", optp1_a, " B subesi optimal indirimli fiyat", optp1_b,
                      #"A subesi optimal siparis miktari", optq0_a, "B subesi optimal siparis miktari", optq0_b,
                      #"stok paylasim miktari", optqs)
        print(P1a_list)
        print(P1b_list)
        print(Q0a_list)
        print(Q0b_list)
        print(Qs_list)
    # print("teta", teta)
    # print("Lmax",Lmax)
    # optimum ortalama kar bulma:

    print("Lm", Lm)
    print("teta", teta)
    # λ∗+teta(q1)=J(q1)'dir. J(q1) Lmax listesinde tutuluyor ve tum q1ler icin tetalar hesaplandi. Bu nedenle herhangi bir q1 icin yakinsadigi k'daki Lmax'tan teta cikarilarak λ∗ bulunabilir.
    print("w=", w, "icin", max(Lm[1][1]) - teta[1][1][k - 1],max(Lm[2][2]) - teta[2][2][k - 1])
    # print("w=", w, "icin", max(Lm[1][1]) - teta[1][1][k - 1])
    print(L1)


    # print(Lmax1)
end = time.time()
print(end - start)