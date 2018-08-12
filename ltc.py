# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:17:05 2018

@author: U586086
"""

# leetcode continue...


"""
Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1.

Example 1: 
Input:
0 0 0
0 1 0
0 0 0

Output:
0 0 0
0 1 0
0 0 0

Example 2: 
Input:
0 0 0
0 1 0
1 1 1

Output:
0 0 0
0 1 0
1 2 1

Note:
1.The number of elements of the given matrix will not exceed 10,000.
2.There are at least one 0 in the given matrix.
3.The cells are adjacent in only four directions: up, down, left and right.
"""
from collections import deque
def updateMatrix(matrix):
    # trasverse each 0: check the four neighbors, if has larger, then needs 
    # to be trasversed
    q = deque()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                matrix[i][j] = float('inf')
            else:
                q.append((i,j))

    while len(q) != 0:
        locs = q.popleft()
        update(locs[0], locs[1], matrix, q)
        
    return matrix
       
def update(i, j, matrix, q):
    for (irow, icol) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
        if 0 <= irow <= len(matrix) - 1 and 0 <= icol <= len(matrix[0]) - 1:
            if matrix[irow][icol] > matrix[i][j] + 1:
                matrix[irow][icol] = matrix[i][j] + 1
                q.append((irow, icol))
                
    return None


updateMatrix([[0 ,0 ,0],[0, 1, 0],[1, 1, 1]])
matrix = [[0 ,0 ,0],[0, 1, 0],[1, 1, 1]]

"""
Open the lock 

You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.

"""
format(10, '04d')

def openlock(deadends, target):
    hs = set()
    q = deque()
    hs.add("0000")
    q.append("0000")
    depth = 0
    
    while len(q) > 0:
        for i in range(len(q)):
            curr = q.popleft()
            if target == curr:
                return depth
            for j in neighbor(curr):
                if j not in hs and j not in deadends:
                    q.append(j)
                    hs.add(j)
        depth = depth + 1
    return -1
        
def neighbor(s):
    nbs = []
    for i in range(4):
        tmp = str(int(s[i]) + 1 if int(s[i]) < 9 else '0')
        nbs.append(s[:i] + tmp + s[i+1:])
        tmp = str(int(s[i]) - 1  if int(s[i]) > 0 else '9')
        nbs.append(s[:i] + tmp + s[i+1:])
        
    return nbs

openlock(deadends, target)
deadends, target = ["0201","0101","0102","1212","2002"], "0202"
deadends, target = ["8887","8889","8878","8898","8788","8988","7888","9888"],   "8888"

                   
                   