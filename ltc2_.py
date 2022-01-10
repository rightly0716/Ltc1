# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:17:05 2018

@author: U586086
"""

# leetcode continue...


"""01 matrix
01matrix Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.

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
Open the lock. You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.
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

"""
You are asked to cut off trees in a forest for a golf event. The forest is represented as a non-negative 2D map, in this map:
1.0 represents the obstacle can't be reached.
2.1 represents the ground can be walked through.
3.The place with number bigger than 1 represents a tree can be walked through, and this positive number represents the tree's height.

You are asked to cut off all the trees in this forest in the order of tree's height - always cut off the tree with lowest height first. And after cutting, the original place has the tree will become a grass (value 1).
You will start from the point (0, 0) and you should output the minimum steps you need to walk to cut off all the trees. If you can't cut off all the trees, output -1 in that situation.
You are guaranteed that no two trees have the same height and there is at least one tree needs to be cut off.

Example 1:
Input: 
[
 [1,2,3],
 [0,0,4],
 [7,6,5]
]
Output: 6
"""
from collections import deque

def cutOffTree(forest):
    heights = []
    for i in range(len(forest)):
        for j in range(len(forest[0])):
            if forest[i][j] != 0:
                heights.append([forest[i][j], (i, j)])
    
    heights.sort(key = lambda x:x[0])
    nstep = 0
    curr_i, curr_j = 0, 0

    for tree in heights:
        # height = tree[0]
        dest_i ,dest_j = tree[1]
        curr_step = dist(curr_i, curr_j, dest_i, dest_j, forest)
        curr_i, curr_j = dest_i ,dest_j 
#        if curr_step == float('inf'):
#            return -1
        # forest[dest_i][dest_j] = 1
        # print(curr_step)
        nstep = nstep + curr_step
    
    return nstep    

def dist(curr_i, curr_j, dest_i, dest_j, forest):
    # calc the distance, change forest[dest_i][dest_j] to 1
    #if curr_i < 0 or curr_i > len(forest) - 1 or curr_j < 0 or curr_j > len(forest[0]) - 1:
    #    return float('inf')
    #if forest[curr_i][curr_j] == 0:
    #    return float('inf')
    d = 0
    q = deque()
    hs = set()
    q.append((curr_i, curr_j))
    hs.add((curr_i, curr_j))
    while len(q) != 0:
        for item in range(len(q)):
            i, j = q.popleft()
            if i == dest_i and j == dest_j:
                #print(d)
                return d
            for (irow, icol) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if 0 <= irow <= len(forest)-1 and 0 <= icol <= len(forest[0])-1 and (irow, icol) not in hs:
                    if forest[irow][icol] != 0:
                        q.append((irow,icol))
                        hs.add((irow,icol))
        d = d + 1
    return float('inf')

forest = [
 [1,2,3],
 [0,0,4],
 [7,6,5]
]
cutOffTree(forest)

"""
LeetCode 743. Network Delay Time

There are N network nodes, labelled 1 to N.

Given times, a list of travel times as directed edges times[i] = (u, v, w), where u is the source node, v is the target node, and w is the time it takes for a signal to travel from source to target.

Now, we send a signal from a certain node K. How long will it take for all nodes to receive the signal? If it is impossible, return -1.

Note:
1.N will be in the range [1, 100].
2.K will be in the range [1, N].
3.The length of times will be in the range [1, 6000].
4.All edges times[i] = (u, v, w) will have 1 <= u, v <= N and 1 <= w <= 100.

"""

from heapq import heappush
from heapq import heappop

def networkDelayTime(times, N, k):
    G = []
    for i in range(N):
        G.append([float('inf')] * N)

    for item in times:
        G[item[0] - 1, item[1] - 1] = item[2]
    dists = Dijkstra(G, k - 1)
    if float('Inf') in dists:
        return -1
    else:
        return max(dists)

# Dijkstra: every time heappop the min dist node (mnode), and check whether 
# (start, mnode) + (mnode, node i) is less than (start, node i) for all the i
# remained in the heap
def Dijkstra(G, k):
    # 0 <= k <= len(G[0]) - 1
    dists = [float('inf')] * len(G[0])
    dists[k] = 0
    previous = [None] * len(G[0])

    q = []
    for i in range(len(G[0])):
        heappush(q, (dists[i], i))
        
    while len(q) != 0:
        d, node = heappop(q)
        for i in range(len(G[0])):
            if dist[i] < dist[node] + G[node, i]:
                dist[i] = dist[node] + G[node, i]
                previous[i] = node
    
    return dists    

    
times = []

"""
Maze II
There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.
Given the ball's start position, the destination and the maze, find the shortest distance for the ball to stop at the destination. The distance is defined by the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included). If the ball cannot stop at the destination, return -1.
The maze is represented by a binary 2D array. 1 means the wall and 0 means the empty space. You may assume that the borders of the maze are all walls. The start and destination coordinates are represented by row and column indexes.
example:
Input 1: a maze represented by a 2D array

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (4, 4)
Output: 12
"""
from collections import deque

def shortestDistance(maze, start, destination):
    nr = len(maze)
    nc = len(maze[0])
    
    q = deque()
    q.append(start)
    
    dists = [[float('inf')] * nc for _ in range(nr)]
    dists[start[0]][start[1]] = 0
    direct = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while len(q) != 0:
        for _ in range(len(q)):
            loc_i, loc_j = q.popleft()
            for (d_i, d_j) in direct:
                x, y = loc_i, loc_j
                dist = dists[x][y]
                while 0 <= x < nr and 0 <= y < nc and maze[x][y] == 0:
                    x = x + d_i
                    y = y + d_j
                    dist = dist + 1
                    
                x = x - d_i
                y = y - d_j
                dist = dist - 1
                if dist < dists[x][y]:
                    dists[x][y] = dist # update this stop cell
                    if x != destination[0] or y != destination[1]:
                        q.append((x, y)) # does not need to cont. at arrival
    res =  dists[destination[0]][destination[1]]                    
    return res if res != float('inf') else -1
    



maze = [[0,0,1,0,0],
[0,0,0,0,0],
[0,0,0,1,0],
[1,1,0,1,1],
[0,0,0,0,0]]

shortestDistance(maze, (0, 4), (4, 4))

"""
Perfect Squares
Given a positive integer n, find the least number of perfect square numbers 
(for example, 1, 4, 9, 16, ...) which sum to n.
For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.
"""
# may have TLE issue
def numSquares(n):
    if n <= 0:
        return 0
    
    dp = [float('inf')] * (n+1)
    dp[0] = 0
    
    for i in range(int(n**0.5)):
        dp[i**2] = 1
    
    for i in range(int(n**0.5)):
        for j in [j for j in range(n) if i**2 + j <= n]:
            dp[i**2 + j] = min(dp[j] + 1, dp[i**2 + j])
    return dp[n]

"""
Number of Connected Components in an Undirected Graph 
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes),
 write a function to find the number of connected components in an undirected graph.
"""
from collections import deque
n = 5 
edges = [[0, 1], [1, 2], [3, 4]]
edges = [[0, 1], [1, 2], [2, 3], [3, 4]]

def countComponents(n ,edges):
    if n == 0:
        return 0
    hs = set()
    adj_list = [[] for _ in range(n)] # contains the adj edge of each node
    for i in range(n):
        for j in edges:
            if i == j[0]:
                adj_list[i].append(j) 
            elif i == j[1]:
                adj_list[i].append(j[::-1]) 
                
    num = 0
    for i in range(n):
        if i not in hs:
            num = num + 1
            explore(i, edges, hs, adj_list)
    return num

def explore(node, edges, hs, adj_list):
    
    q = deque()
    q.append(node)
    hs.add(node)
    while len(q) != 0:
        curr_node = q.popleft()
        for nb_edge in adj_list[curr_node]:
            if nb_edge[1] not in hs:
                q.append(nb_edge[1])
                hs.add(nb_edge[1])
    return None

countComponents(n, edges)

"""
Graph Valid Tree 图验证树 

Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), 
write a function to check whether these edges make up a valid tree.

For example:

Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.
Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.
Hint:
1.Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], what should your return? Is this case a valid tree?
2.According to the definition of tree on Wikipedia: “a tree is an undirected graph 
in which any two vertices are connected by exactly one path. In other words, any 
connected graph without simple cycles is a tree.”

Note: you can assume that no duplicate edges will appear in edges. Since all edges 
are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

"""
# all nodes need to be visited; no cycle is allowed

from collections import deque

def validTree(n ,edges):
    if n == 0:
        return True
    g = [[] for _ in range(n)]
    for e in edges:
        g[e[0]].append(e[1])
        g[e[1]].append(e[0])
    hs = set()
    
    q = deque()
    q.append(0)
    hs.add(0)
    
    while len(q) != 0:
        curr_node = q.popleft()
        for nb_node in g[curr_node]:
            if nb_node not in hs:
                q.append(nb_node)
                hs.add(nb_node)
                # need to remove the "back edge"
                g[nb_node] = [i for i in g[nb_node] if i != curr_node]
            else:
                return False
    return len(hs) == n    

validTree(n, edges)

n = 5
edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
n = 1; edges = []

# union  find
def validTree(n, edges):
    if n == 0:
        return True
    roots = [-1] * n
    for e in edges:
        x = find_root(roots, e[0])
        y = find_root(roots, e[1])
        if x == y:
            return False
        else:
            roots[x] = y
    #return len(edges) == n-1
    return len([i for i,x in enumerate(roots) if x==-1 and roots.count(i)==0]) == 0

def find_root(roots, i):
    while roots[i] != -1:
        i = roots[i]
    return i


"""Course Schedule 课程清单 
There are a total of n courses you have to take, labeled from 0 to n - 1.
Some courses may have prerequisites, for example to take course 0 you have to 
first take course 1, which is expressed as a pair: [0,1]
Given the total number of courses and a list of prerequisite pairs, 
is it possible for you to finish all courses?

For example:
2, [[1,0]]

There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.
2, [[1,0],[0,1]]

There are a total of 2 courses to take. To take course 1 you should have 
finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
"""

def canFinish(numCourses, prerequisites):
    # The core idea is to start with courses with 0 in_degree
    # remove it from the graph and update the in_degree of its "post-course"
    # the push the post courses with 0 in_degree into q;
    # return false if any in_degree is not zero in the end
    
    if numCourses == 0:
        return True
    g = [[] for _ in range(numCourses)]
    in_degree = [0] * numCourses
    for pre in prerequisites:
        g[pre[1]].append(pre[0]) 
        # each g saves the crs it goes to
        in_degree[pre[0]] = in_degree[pre[0]]  + 1
    
    q = deque()
    for i in range(numCourses):
        if in_degree[i] == 0:
            q.append(i)
    
    while len(q) != 0:
        curr_course = q.popleft()
        for post_course in g[curr_course]:
            in_degree[post_course] = in_degree[post_course] - 1
            if in_degree[post_course] == 0:
                q.append(post_course)
                
    return max(in_degree) == 0

numCourses = 3
prerequisites = [[1,0],[0,1], [0,2]]
canFinish(numCourses, prerequisites)

"""Course Schedule II 课程清单之二 
There are a total of n courses you have to take, labeled from 0 to n - 1.
Some courses may have prerequisites, for example to take course 0 you have to 
first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the 
ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If 
it is impossible to finish all courses, return an empty array.

For example:
2, [[1,0]]
There are a total of 2 courses to take. To take course 1 you should have 
finished course 0. So the correct course order is [0,1]

4, [[1,0],[2,0],[3,1],[3,2]]
There are a total of 4 courses to take. To take course 3 you should have 
finished both courses 1 and 2. Both courses 1 and 2 should be taken after you 
finished course 0. So one correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].
"""

def findOrder(numCourses, prerequisites):
    if numCourses == 0:
        return []
    
    g = [[] for _ in range(numCourses)]
    in_degree = [0] * numCourses
    for pre in prerequisites:
        g[pre[1]].append(pre[0])
        in_degree[pre[0]] = in_degree[pre[0]] + 1
    q = deque()
    for i in range(numCourses):
        if in_degree[i] == 0:
            q.append(i)
    order = []
    while len(q) != 0:
        curr = q.popleft()
        order.append(curr)
        for curr_post in g[curr]:
            in_degree[curr_post] = in_degree[curr_post] - 1
            if in_degree[curr_post] == 0:
                q.append(curr_post)
    
    if max(in_degree) > 0:
        return []
    else:
        return order

findOrder(4, [[1,0],[2,0],[3,1],[3,2]])


"""Binary Tree Right Side View 二叉树的右侧视图 
Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

For example:
Given the following binary tree,
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
  
You should return [1, 3, 4].
"""

def rightSideView(root):
    if root == None:
        return []
    output = []
    q = deque()
    q.append(root)
    
    while len(q) != 0:
        temp = []
        for _ in range(len(q)):
            curr_node = q.popleft()
            temp.append(curr_node.val)
            if curr_node.left != None:
                q.append(curr_node.left)
            if curr_node.right != None:
                q.append(curr_node.right)
        output.append(temp[len(temp) - 1])
    
    return output

""" Clone Graph
Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.

How we serialize an undirected graph:
Nodes are labeled uniquely.
We use # as a separator for each node, and , as a separator for node label 
and each neighbor of the node.
As an example, consider the serialized graph {0,1,2#1,2#2,2}.
The graph has a total of three nodes, and therefore contains three parts as separated by #.
First node is labeled as 0. Connect node 0 to both nodes 1 and 2. Second node 
is labeled as 1. Connect node 1 to node 2. Third node is labeled as 2. Connect 
node 2 to node 2 (itself), thus forming a self-cycle. Visually, the graph looks like the following:
"""


"""leetcode Question 131: Surrounded Regions 
Surrounded Regions

Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region .

For example,
X X X X
X O O X
X X O X
X O X X

After running your function, the board should be:
X X X X
X X X X
X X X X
X O X X

"""


""" 103. Binary Tree Zigzag Level Order Traversal
Given a binary tree, return the zigzag level order traversal of its nodes' values. 
(ie, from left to right, then right to left for the next level and alternate between).

Example 

Given binary tree {3,9,20,#,#,15,7},
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]

"""
from collections import deque

def zigzagLevelOrder(root):
    if root == None:
        return []
    isOdd = True
    
    q = deque()
    q.append(root)
    output = []
    
    while len(q) != 0:
        level = []
        for _ in range(len(q)):
            curr_node = q.popleft()
            level.append(curr_node.val)
            if curr_node.left != None:
                q.append(curr_node.left)
            if curr_node.right != None:
                q.append(curr_node.right)
        if isOdd:
            output.append(level)
            isOdd = False
        elif not isOdd:
            output.append(level[::-1])
            isOdd = True
    return output

""" 756. Pyramid Transition Matrix
We are stacking blocks to form a pyramid. Each block has a color which is a one letter string, like 'Z'.
For every block of color C we place not in the bottom row, we are placing it on
 top of a left block of color A and right block of color B. We are allowed to
 place the block there only if (A, B, C) is an allowed triple.
We start with a bottom row of bottom, represented as a single string. We also
 start with a list of allowed triples allowed. Each allowed triple is represented as a string of length 3.

Return true if we can build the pyramid all the way to the top, otherwise false

"""
bottom = "XXYX"
allowed = ["XXX", "XXY", "XYX", "XYY", "YXZ"]

bottom = "XYZ"
allowed = ["XYD", "YZE", "DEA", "FFF"]


def pyramidTransition(bottom, allowed):
    n = len(bottom)
    if n == 1:
        return True
    ht = {}
    for i in allowed:
        if i[:2] not in ht:
            ht[i[:2]] = i[2]
        else:
            ht[i[:2]] = ht[i[:2]] + i[2]
    # i - level, j - location
    pyramid = [['']*n for _ in range(n)]
    for j in range(n):
        pyramid[0][j] = bottom[j]
        
    for i in range(1, n):
        for j in range(n - i):   
            for a in pyramid[i-1][j]:
                for b in pyramid[i-1][j+1]:
                    if a+b in ht and ht[a+b] not in pyramid[i][j]:
                        pyramid[i][j] = pyramid[i][j] + ht[a+b]
            
    return len(pyramid[n-1][0]) > 0

pyramidTransition(bottom, allowed)

"""Sentence Similarity II  
Given two sentences words1, words2 (each represented as an array of strings),
 and a list of similar word pairs pairs, determine if two sentences are similar.
For example, words1 = ["great", "acting", "skills"] and
 words2 = ["fine", "drama", "talent"] are similar, if the similar word pairs are
 pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]].
Note that the similarity relation is transitive. For example, if "great" and
 "good" are similar, and "fine" and "good" are similar, then "great" and "fine" are similar.
Similarity is also symmetric. For example, "great" and "fine" being similar is
 the same as "fine" and "great" being similar.
Also, a word is always similar with itself. For example, the sentences
 words1 = ["great"], words2 = ["great"], pairs = [] are similar, even though
 there are no specified similar word pairs.
Finally, sentences can only be similar if they have the same number of words. 
So a sentence like words1 = ["great"] can never be similar to words2 = ["doubleplus","good"].
"""
words1 = ["great", "acting", "skills"]
words2 = ["fine", "drama", "talent"] 
pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]

from collections import deque

def areSentencesSimilarTwo(words1, words2, pairs):
    if len(words1) != len(words2):
        return False
    ht = {}
    for i in pairs:
        if i[0] not in ht:
            ht[i[0]] = [i[1]]
        else:
            ht[i[0]] = ht[i[0]] + [i[1]]
        if i[1] not in ht:
            ht[i[1]] = [i[0]]
        else:
            ht[i[1]] = ht[i[1]] + [i[0]]
    
    for i in range(len(words1)):
        if not isSimilar(words1[i], words2[i], ht):
            return False
        
    return True

def isSimilar(w1, w2, ht):
    if w1 == w2:
        return True
    q = deque()
    hs = set()
    for i in ht[w1]:
        q.append(i)
        hs.add(i)
    
    while len(q) != 0:
        for _ in range(len(q)):
            curr_w = q.popleft()
            if curr_w  == w2:
                return True
            for nb_w in ht[curr_w]:
                if nb_w not in hs:
                    hs.add(nb_w)
                    q.append(nb_w)
    return False

areSentencesSimilarTwo(words1, words2, pairs)


words1 = ["great"]; words2 = ["great"]; pairs = []

"""Accounts Merge 账户合并 
Given a list accounts, each element accounts[i] is a list of strings, where the
 first element accounts[i][0] is a name, and the rest of the elements are emails
 representing emails of the account.
Now, we would like to merge these accounts. Two accounts definitely belong to
 the same person if there is some email that is common to both accounts. Note
 that even if two accounts have the same name, they may belong to different
 people as people could have the same name. A person can have any number of accounts
 initially, but all of their accounts definitely have the same name.
After merging the accounts, return the accounts in the following format: the
 first element of each account is the name, and the rest of the elements are
 emails in sorted order. The accounts themselves can be returned in any order.

Example 1:
Input: 
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], 
 ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  
 ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
Explanation: 
The first and third John's are the same person as they have the common email "johnsmith@mail.com".
The second John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
"""
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], 
 ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]

def accountsMerge(accounts):
    return None
    
"""Two Sum III - Data structure design 两数之和之三 - 数据结构设计 
Design and implement a TwoSum class. It should support the following operations:add and find.
add - Add the number to an internal data structure.
find - Find if there exists any pair of numbers which sum is equal to the value.

For example,
add(1); add(3); add(5);
find(4) -> true
find(7) -> false 
"""
# hashtable and watch out the case when x = k - x, or k = 2 * x
# hashtable needs to save the time of x as well

"""Design Tic-Tac-Toe 设计井字棋游戏 
Design a Tic-tac-toe game that is played between two players on a n x n grid.

You may assume the following rules:

A move is guaranteed to be valid and is placed on an empty block.
Once a winning condition is reached, no more moves is allowed.
A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.

"""



"""Keyboard Row 键盘行 
Given a List of words, return the words that can be typed using letters of 
alphabet on only one row's of American keyboard like the image below.

Example 1:
Input: ["Hello", "Alaska", "Dad", "Peace"]
Output: ["Alaska", "Dad"]
"""
row1 = ['q','w','e','r','t','y','u','i','o','p']

def explore(words, row1):
    for s in words:
        if s not in row1:
            return False
    return True

explore('game', row1)
explore('yiop', row1)


"""Number of Boomerangs 回旋镖的数量 
Given n points in the plane that are all pairwise distinct, a "boomerang" is a 
tuple of points (i, j, k) such that the distance between i and j equals the 
distance between i and k (the order of the tuple matters).
Find the number of boomerangs. You may assume that n will be at most 500 and 
coordinates of points are all in the range [-10000, 10000] (inclusive).

Input:
[[0,0],[1,0],[2,0]]

Output:
2

Explanation:
The two boomerangs are [[1,0],[0,0],[2,0]] and [[1,0],[2,0],[0,0]]
"""
points = [[0,0],[1,0],[2,0], [3,0], [4,0]]
points = [[0,0],[1,0],[2,0]]

def numberOfBoomerangs(points):
    num = 0
    for pt in points:
        ht = {}
        for i in points:
            if cal_dist(pt, i) in ht:
                # ht[cal_dist(pt, i)] = ht[cal_dist(pt, i)] + [i]
                ht[cal_dist(pt, i)].append(i)
            else:
                ht[cal_dist(pt, i)] = [i]
        for item in ht:
            if len(ht[item]) > 1:
                num = num + len(ht[item]) * (len(ht[item]) - 1)
    return num

numberOfBoomerangs(points)

def cal_dist(pt1, pt2):
    return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2


"""Maximum Size Subarray Sum Equals k 最大子数组之和为k 
Given an array nums and a target value k, find the maximum length of a subarray
 that sums to k. If there isn't one, return 0 instead.

Example 1:

Given nums = [1, -1, 5, -2, 3], k = 3,
 return 4. (because the subarray [1, -1, 5, -2] sums to 3 and is the longest) 

Example 2:

Given nums = [-2, -1, 2, 1], k = 1,
 return 2. (because the subarray [-1, 2] sums to 1 and is the longest) 

Follow Up:
 Can you do it in O(n) time? 

"""
nums = [1, -1, 5, -2, 3]
nums = [1, -1, 5, -2, 3, 6, -3, 3, -5, -3]
k = 3

def maxSubArrayLen(nums, k):
    ht = {}
    sum_nums = [0] * len(nums)
    sum_val = 0
    for i,x in enumerate(nums):
        sum_val = sum_val + x
        sum_nums[i] = sum_val
        ht[sum_nums[i]] = i
            
    length = 0
    for i,x in enumerate(sum_nums):
        if x == k:
            length = max(length, i + 1)
        for item in ht:
            if x + k in ht:
                length = max(length, ht[x + k] - i)
    return length

maxSubArrayLen(nums, k)


"""Group Shifted Strings 群组偏移字符串 
Given a string, we can "shift" each of its letter to its successive letter, 
for example: "abc" -> "bcd". We can keep "shifting" which forms the sequence:
"abc" -> "bcd" -> ... -> "xyz"

Given a list of strings which contains only lowercase alphabets, group all 
strings that belong to the same shifting sequence.

For example, given: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"], 
Return:
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]

Note: For the return value, each inner list's elements must follow the lexicographic order.

"""
strings = ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]

def groupStrings(strings):
    ht = {}
    for s in strings:
        if hashcode(s) not in ht:
            ht[hashcode(s)] = [s]
        else:
            ht[hashcode(s)].append(s)

    res = []
    for item in ht:
        res.append(sorted(ht[item]))
    return res

def hashcode(s):
    out = [0] * len(s)
    for i in range(len(s)):
        out[i] = ord(s[i]) - ord(s[0]) if ord(s[i]) - ord(s[0]) >= 0 else ord(s[i]) - ord(s[0]) + 26
    return tuple(out)

groupStrings(strings)


""" Largest number
1 # Time:  O(nlogn) 
2 # Space: O(1) 
3 # 
4 # Given a list of non negative integers, arrange them such that they form the largest number. 
5 #  
6 # For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330. 
7 #  
8 # Note: The result may be very large, so you need to return a string instead of an integer. 
9 # 
"""
nums = [3, 30, 34, 5, 9]
class my_cmp(str):
    def __lt__(a,b):
        return int(a + b) < int(b+a)
    def __gt__(a,b):        
        return int(a + b) >= int(b+a)

def largestNumber(nums):
    s_nums = [str(i) for i in nums]
    s = sorted(map(my_cmp, s_nums))[::-1]
    return "".join(s)
    
largestNumber(nums)

"""Sort Colors 
Given an array with n objects colored red, white or blue, sort them so that 
objects of the same color are adjacent, with the colors in the order red, white and blue.
Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
Note:
You are not suppose to use the library's sort function for this problem.
Follow up:
A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array 
with total number of 0's, then 1's and followed by 2's.
Could you come up with an one-pass algorithm using only constant space?
"""

A = [1,2,1,0,0,1,2,2,0]

def sortColors(A):
    i = 0
    begin = 0
    end = len(A) - 1
    while i < end:
        while A[i] == 0 and i > begin:
            A[i] = A[begin]
            A[begin] = 0
            begin = begin + 1
        while A[i] == 2 and i < end:
            A[i] = A[end]
            A[end] = 2
            end = end - 1
        i = i + 1
    return A


sortColors(A)


""" Exclusive Time of Functions
Given the running logs of n functions that are executed in a nonpreemptive single 
threaded CPU, find the exclusive time of these functions.

Each function has a unique id, start from 0 to n-1. A function may be called 
recursively or by another function.

A log is a string has this format : function_id:start_or_end:timestamp. For 
example, "0:start:0" means function 0 starts from the very beginning of time 0. 
"0:end:0" means function 0 ends to the very end of time 0.

Exclusive time of a function is defined as the time spent within this function, 
the time spent by calling other functions should not be considered as this function's exclusive time. You should return the exclusive time of each function sorted by their function id.

"""
logs = ["0:start:0",  "1:start:2",  "1:end:5",  "0:end:6"]
n = 2
def exclusiveTime(n, logs):
    res = [0] * n
    stack = []
    for i in logs:
        fid, status, time = i.split(':')
        fid = int(fid)
        time = int(time)
        if status == "start" and len(stack) == 0:
            stack.append([fid, time])
        elif status == "start" and len(stack) != 0:
            res[stack[-1][0]] = res[stack[-1][0]] + time - stack[-1][1]
            stack.append([fid, time])
        elif status == "end":
            res[stack[-1][0]] = res[stack[-1][0]] + time - stack[-1][1] + 1
            stack.pop()
            if len(stack) != 0:
                stack[-1][1] = time + 1
    return res

exclusiveTime(n, logs)


"""Next Greater Element II 下一个较大的元素之二 
Given a circular array (the next element of the last element is the first 
element of the array), print the Next Greater Number for every element. The 
Next Greater Number of a number x is the first greater number to its 
traversing-order next in the array, which means you could search circularly to 
find its next greater number. If it doesn't exist, output -1 for this number.

Example 1:
Input: [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number; 
The second 1's next greater number needs to search circularly, which is also 2.

"""

nums = [1,3,2,1,4,2,3,1]

def nextGreaterElements(nums):
    n = len(nums)
    res = [-1] * n
    stack = []
    for i in range(n*2):
        curr_i = i % n
        if len(stack) != 0:
            while len(stack) != 0 and nums[curr_i] > nums[stack[-1]]:
                res[stack[-1]] = nums[curr_i]
                stack.pop()
        if i < n:
            stack.append(i)
    return res

nextGreaterElements(nums)



"""Different Ways to Add Parentheses 添加括号的不同方式 
Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are +, -and *.

Example 2

Input: "2*3-4*5"
(2*(3-(4*5))) = -34
((2*3)-(4*5)) = -14
((2*(3-4))*5) = -10
(2*((3-4)*5)) = -10
(((2*3)-4)*5) = 10

Output: [-34, -14, -10, -10, 10]
"""
# JAVA code with a clear idea on how to use DP
class Solution {
public:
    vector<int> diffWaysToCompute(string input) {
        vector<int> res;
        for (int i = 0; i < input.size(); ++i) {
            if (input[i] == '+' || input[i] == '-' || input[i] == '*') {
                vector<int> left = diffWaysToCompute(input.substr(0, i));
                vector<int> right = diffWaysToCompute(input.substr(i + 1));
                for (int j = 0; j < left.size(); ++j) {
                    for (int k = 0; k < right.size(); ++k) {
                        if (input[i] == '+') res.push_back(left[j] + right[k]);
                        else if (input[i] == '-') res.push_back(left[j] - right[k]);
                        else res.push_back(left[j] * right[k]);
                    }
                }
            }
        }
        if (res.empty()) res.push_back(atoi(input.c_str()));
        return res;
    }
};
                
"""Partition Equal Subset Sum 相同子集和分割 
Given a non-empty array containing only positive integers, find if the array 
can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note:
 Both the array size and each of the array element will not exceed 100. 
Example 1: 
Input: [1, 5, 11, 5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].

Example 2:
Input: [1, 2, 3, 5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
"""
# use the dp udea, create a T/F dp[] with length sum/2 + 1
# its element i represents whether i can be made by summing
# for i in range(len(nums)):
    #for j in range(target, nums[i]-1):
# each i loop will update all the dp[] that can be made by summing nums[:i]

                    
"""Partition to K Equal Sum Subsets 分割K个等和的子集 
Given an array of integers nums and a positive integer k, find whether it's 
possible to divide this array into knon-empty subsets whose sums are all equal.

Example 1:
Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
Output: True
Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.

Note:
•1 <= k <= len(nums) <= 16.
•0 < nums[i] < 10000.
"""

"""Predict the Winner 预测赢家 
Given an array of scores that are non-negative integers. Player 1 picks one of 
the numbers from either end of the array followed by the player 2 and then player 
1 and so on. Each time a player picks a number, that number will not be available 
for the next player. This continues until all the scores have been chosen. The 
player with the maximum score wins.

Given an array of scores, predict whether player 1 is the winner. You can assume 
each player plays to maximize his score.

Example 1:
Input: [1, 5, 2]
Output: False
Explanation: Initially, player 1 can choose between 1 and 2. 
If he chooses 2 (or 1), then player 2 can choose from 1 (or 2) and 5. If player 2 
chooses 5, then player 1 will be left with 1 (or 2). 
So, final score of player 1 is 1 + 2 = 3, and player 2 is 5. 
Hence, player 1 will never be the winner and you need to return False.

Example 2:
Input: [1, 5, 233, 7]
Output: True
Explanation: Player 1 first chooses 1. Then player 2 have to choose between 5 
and 7. No matter which number player 2 choose, player 1 can choose 233.
Finally, player 1 has more score (234) than player 2 (12), so you need to 
return True representing player1 can win.
"""
# dp records whether the calculation was done before to avoid duplicates
nums = [1, 5, 233, 7]
nums =  [1, 5, 2]

PredictTheWinner(nums)

def PredictTheWinner(nums):
    if len(nums) == 0:
        return False
    dp = [[-1] * len(nums) for i in range(len(nums))] 
    return helper(nums, 0, len(nums)-1, dp) > 0

def helper(nums, i, j, dp):
    if dp[i][j] == -1:
        if i == j:
            dp[i][j] = nums[i]
            return dp[i][j]
        else:
            dp[i][j] = max(nums[i] - helper(nums, i+1, j, dp), nums[j] - helper(nums, i, j-1, dp))
            return dp[i][j]
    else:
        return dp[i][j]

"""My Calendar III 我的日历之三 

Implement a MyCalendarThree class to store your events. A new event can always 
be added.

Your class will have one method, book(int start, int end). Formally, this 
represents a booking on the half open interval [start, end), the range of real 
    numbers x such that start <= x < end.

A K-booking happens when K events have some non-empty intersection (ie., there 
is some time that is common to all K events.)

For each call to the method MyCalendar.book, return an integer K representing 
the largest integer such that there exists a K-booking in the calendar.
Your class will be called like this: MyCalendarThree cal = new MyCalendarThree(); 
MyCalendarThree.book(start, end)

Example 1:
MyCalendarThree();
MyCalendarThree.book(10, 20); // returns 1
MyCalendarThree.book(50, 60); // returns 1
MyCalendarThree.book(10, 40); // returns 2
MyCalendarThree.book(5, 15); // returns 3
MyCalendarThree.book(5, 10); // returns 3
MyCalendarThree.book(25, 55); // returns 3
Explanation: 
The first two events can be booked and are disjoint, so the maximum K-booking is a 1-booking.
The third event [10, 40) intersects the first event, and the maximum K-booking is a 2-booking.
The remaining events cause the maximum K-booking to be only a 3-booking.
Note that the last event locally causes a 2-booking, but the answer is still 3 because
eg. [10, 20), [10, 40), and [5, 15) are still triple booked.
"""


class MyCalendarThree:
    def __init__(self):
        self.max_booking = 0
        self.ht = {}

    def book(self, start, end):
        if start not in self.ht:
            self.ht[start] = 1
        else:
            self.ht[start] = self.ht[start] + 1
        if end not in self.ht:
            self.ht[end] = -1
        else:
            self.ht[end] = self.ht[end] - 1
                   
        cnt = 0
        self.max_booking = 0
        for e in sorted(list(self.ht.keys())):
            cnt = cnt + self.ht[e]
            self.max_booking = max(self.max_booking, cnt)
        return self.max_booking
    
a = MyCalendarThree();
a.book(10, 20)# returns 1
a.book(50, 60)# returns 1
a.book(10, 40)# returns 2
a.book(5, 15)# returns 3
a.book(5, 10)# returns 3
a.book(25, 55)# returns 3


"""Word Break I
Given a string s and a dictionary of words dict, determine if s can be 
segmented into a space-separated sequence of one or more dictionary words.

For example, given s = "leetcode", dict = ["leet", "code"].
Return true because "leetcode" can be segmented as "leet code".
"""
s = "leetcode"
dict1 = ["leet", "code"]

def wordBreak(s, dict1):
    if len(s) == 0:
        return True
    hs = set()
    for i in dict1:
        hs.add(i)
    n = len(s)
    res = [False]*(n+1)
    res[0] = True
    for i in range(1, n+1):
       str1 = s[:i]
       for j in range(i)[::-1]:
           if res[j] == True and s[j:i] in hs:
               res[i] = True
               break
    return res[n]

wordBreak(s, dict1)

"""
Word Break II 拆分词句之二 

Given a string s and a dictionary of words dict, add spaces in s to construct 
a sentence where each word is a valid dictionary word.

Return all such possible sentences.

For example, given
s = "catsanddog",
dict = ["cat", "cats", "and", "sand", "dog"]. 

A solution is ["cats and dog", "cat sand dog"]. 
"""

def wordBreak():
    return

"""Construct Binary Tree from String  
You need to construct a binary tree from a string consisting of parenthesis and integers.

The whole input represents a binary tree. It contains an integer followed by zero, 
one or two pairs of parenthesis. The integer represents the root's value and a 
pair of parenthesis contains a child binary tree with the same structure.

You always start to construct the left child node of the parent first if it exists.
Input: "4(2(3)(1))(6(5))"
Output: return the tree root node representing the following tree:

       4
     /   \
    2     6
   / \   / 
  3   1 5 
"""
s = "4(2(3)(1))(6(5))"
s = "6(5)"
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def str2tree(s):
    if s == "":
        return TreeNode(None)
    if "(" not in s:
        root = TreeNode()
        root.val = int(s)
        return root
    left_set = [i for i,x in enumerate(s) if x == '(']
    right_set = [i for i,x in enumerate(s) if x == ')']
    root = TreeNode(int(s[:left_set[0]]))
    n = len(s)
    cnt = 0
    for i in range(left_set[0], n):
        if s[i] == '(':
            cnt = cnt + 1
        elif s[i] == ')':
            cnt = cnt - 1
        if cnt == 0:  
            right_loc = i
            break
    
    left_str = s[(left_set[0] + 1):(right_loc)]
    # may not need if for the right because s[3:1] will return "" (1<3)
    if right_loc+2 < n -1:
        right_str = s[(right_loc+2):(n-1)]
    else:
        right_str = ""
    root.left = str2tree(left_str)
    root.right = str2tree(right_str)
    return root


"""Longest Increasing Path in a Matrix  
Given an integer matrix, find the length of the longest increasing path.
From each cell, you can either move to four directions: left, right, up or down.
 You may NOT move diagonally or move outside of the boundary 
 (i.e. wrap-around is not allowed).

Example 1:
nums = [
  [9,9,4],
  [6,6,8],
  [2,1,1]
]
Return 4
 The longest increasing path is [1, 2, 6, 9].
"""
# use dp to save calculated results
nums = [
  [9,9,4],
  [6,6,8],
  [2,1,1]
]
def longestIncreasingPath(nums):
    nr = len(nums)
    nc = len(nums[0])
    dp = [[0]*nc for _ in range(nr)]
    #direct = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #hs = set()
    max_val = [0]
    for i in range(nr):
        for j in range(nc):
            # the longest path with i,j as the starting point
            if dp[i][j] == 0:
                helper(i, j, nums, dp, max_val)
    
    return max_val[0]
    
def helper(i, j, nums, dp, max_val):
    nr = len(nums)
    nc = len(nums[0])
    res = 1
    #if i < 0 or i > nr - 1 or j < 0 or j > nc:
    #    return 0
    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        curr_i, curr_j = i+dr, j+dc
        if -1 < curr_i < nr and -1 < curr_j < nc:
            if nums[curr_i][curr_j] > nums[i][j]:
                if dp[curr_i][curr_j] != 0:
                    res = max(res, 1 + dp[curr_i][curr_j])
                else:
                    dp[curr_i][curr_j] = helper(curr_i, curr_j, nums, dp, max_val)
                    res = max(res, dp[curr_i][curr_j] + 1)
    dp[i][j] = res
    max_val[0] = max(res, max_val[0])
    return res

longestIncreasingPath(nums)

"""Best Meeting Point 最佳开会地点 
A group of two or more people wants to meet and minimize the total travel distance. 
You are given a 2D grid of values 0 or 1, where each 1 marks the home of someone 
in the group. The distance is calculated using Manhattan Distance, where 
distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.

For example, given three people living at (0,0), (0,4), and (2,2):
1 - 0 - 0 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

The point (0,2) is an ideal meeting point, as the total travel distance of 
2+2+2=6 is minimal. So return 6.

Hint:
1.Try to solve it in one dimension first. How can this solution apply to the two dimension case?
"""
# two dimension = sum of two one dimension

def minTotalDistance(grid):
    nr = len(grid)
    nc = len(grid[0])
    row_set = []
    col_set = []
    for i in range(nr):
        for j in range(nc):
            if grid[i][j] == 1:
                row_set.append(i)
                col_set.append(j)
                
    return helper(row_set) + helper(col_set)

def helper(set1):
    sort_set = sorted(set1)
    start = 0
    end = len(set1) - 1
    res = 0
    while start < end:
        res = res + set1[end] - set1[start]
        start = start + 1
        end = end - 1
    return res

"""Shortest Distance from All Buildings 建筑物的最短距离 
You want to build a house on an empty land which reaches all buildings in the 
shortest amount of distance. You can only move up, down, left and right. You 
are given a 2D grid of values 0, 1 or 2, where:
•Each 0 marks an empty land which you can pass by freely.
•Each 1 marks a building which you cannot pass through.
•Each 2 marks an obstacle which you cannot pass through.

For example, given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2):
1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

The point (1,2) is an ideal empty land to build a house, as the total travel 
distance of 3+3+1=7 is minimal. So return 7.

Note:
There will be at least one building. If it is not possible to build such house 
according to the above rules, return -1.
"""


"""Minimum Height Trees 最小高度树 
For a undirected graph with tree characteristics, we can choose any node as the root. The result graph is then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs). Given such a graph, write a function to find all the MHTs and return a list of their root labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. You will be given the number n and a list of undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

Example 1:

Given n = 4, edges = [[1, 0], [1, 2], [1, 3]]
        0
        |
        1
       / \
      2   3
return [1]
Example 2:
Given n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]
     0  1  2
      \ | /
        3
        |
        4
        |
        5

return [3, 4]

Hint:
1.How many MHTs can a graph have at most?
"""
from collections import deque
n = 6
edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

findMinHeightTrees(n, edges)

def findMinHeightTrees(n, edges):
    if n == 0:
        return None
    if n == 1:
        return [0]
    ht = {}
    for e in edges:
        start, end = e[0], e[1]
        if e[0] in ht:
            ht[e[0]].append(e[1])
        else:
            ht[e[0]] = [e[1]]
        
        if e[1] in ht:
            ht[e[1]].append(e[0])
        else:
            ht[e[1]] = [e[0]]
    
    q = deque()
    for key in ht.keys():
        if len(ht[key]) == 1:
            q.append(key)
            
    rem_edges = edges
    while len(q) != 0 and len(rem_edges) > 1:
        for _ in range(len(q)):
            node = q.popleft()
            start, end = [edge for edge in rem_edges if node in edge][0]
            rem_edges = [edge for edge in rem_edges if edge != [start, end]]
            ht[start] = [i for i in ht[start] if i != end]
            ht[end] = [i for i in ht[end] if i != start]
            if len(ht[start]) == 1:
                q.append(start)
            if len(ht[end]) == 1:
                q.append(end)
            
        
    return [i for i in q]


"""Decode String
Given an encoded string, return it's decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the 
square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, 
square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits 
and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

Examples:
s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
"""
s = "2[abc]3[cd]ef"

decodeString("3[a]2[bc]")
decodeString("3[a2[c]]")
decodeString("2[abc]3[cd]ef")

def decodeString(s):

    n = len(s)
    res = ""
    temp_num = ""
    temp_str = ""
    s_num = [] # stack for num
    s_str = [] # sttack for string
    for i in range(n):
        if s[i].isdigit():
            temp_num = temp_num + s[i]
        if s[i].isalpha():
            temp_str = temp_str + s[i]
            if i < n - 1 and (s[i+1].isdigit() or s[i+1] == ']'):
                s_str.append(temp_str)
                temp_str = ""
        if s[i] == '[':
            s_num.append(int(temp_num))
            temp_num = ""
        if s[i] == ']':
            num0 = s_num.pop()
            str0 = s_str.pop()
            rep_str = str0 * num0
            if len(s_str) == 0:
                res = res + rep_str
            else:
                s_str.append(s_str.pop() + rep_str)
            temp_str = ""
            temp_num = ""
    return res + temp_str

"""Basic Calculator 基本计算器 
Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), 
the plus + or minus sign -, non-negative integers and empty spaces .

You may assume that the given expression is always valid.

Some examples:

"1 + 1" = 2
" 2-1 + 2 " = 3
"(1+(4+5+2)-3)+(6+8)" = 23
Note: Do not use the eval built-in library function.
"""
s = "(1+(4+5+2)-3)+(6+8)"
s1 = "(1+(4+5+2)-3)+(6+8)   +  10"
calculate(s)
calculate(s1)

def calculate(s):
    n = len(s)
    sign = 1
    s_num = []
    s_sym = []
    temp_num = ""
    res = 0
    for i in range(n):
        if s[i].isdigit():
            temp_num = temp_num + s[i]
            if i == n-1 or (i < n - 1 and not s[i+1].isdigit()):
                res = res + sign * int(temp_num)
                temp_num = ""
        if s[i] == '+':
            sign = 1
        if s[i] == '-':
            sign = -1
        if s[i] == '(':
            s_num.append(res)
            res = 0
            s_sym.append(sign)
            sign = 1
        if s[i] == ')':
            res = s_num.pop() + s_sym.pop() * res
                           
    
    return res                      
            

"""Check if a given array can represent Preorder Traversal of Binary Search Tree
Given an array of numbers, return true if given array can represent preorder 
traversal of a Binary Search Tree, else return false. Expected time complexity is O(n).

Example:
Input:  pre[] = {2, 4, 1}
Output: false
Given array cannot represent preorder traversal
of a Binary Search Tree.

A Simple Solution is to do following for every node pre[i] starting from first one.
1) Find the first greater value on right side of current node. 
   Let the index of this node be j. Return true if following 
   conditions hold. Else return false
    (i)  All values after the above found greater value are 
         greater than current node.
    (ii) Recursive calls for the subarrays pre[i+1..j-1] and 
         pre[j+1..n-1] also return true. 
An Efficient Solution can solve this problem in O(n) time. The idea is to use a 
stack. This problem is similar to Next (or closest) Greater Element problem. 
Here we find next greater element and after finding next greater, if we find a 
smaller element, then return false.
"""

def canRepresentBST(pre):
    return None

"""Add Bold Tag in String
Given a string s and a list of strings dict, you need to add a closed pair of 
bold tag <b> and </b> to wrap the substrings in s that exist in dict. If two 
such substrings overlap, you need to wrap them together by only one pair of 
closed bold tag. Also, if two substrings wrapped by bold tags are consecutive, you need to combine them.
Example 2:
Input: 
s = "aaabbcc"
dict = ["aaa","aab","bc"]
Output:
"<b>aaabbc</b>c"

"""
# KMP algorithm in string matching
# https://blog.csdn.net/turkeyzhou/article/details/5660959
s = "eaaabbcc"
dict1 = ["aaa","aab","bc"]

def addBoldTag(s, dict1):
    # brutal force: create a indicator vector to see whether each i is in dict,
    # for each i and find the longest word that match
    n = len(s)
    end = 0
    isin = [False] * n
    for i in range(n):
        for word in dict1:
            if i + len(word) < n and s[i:(i+len(word))] == word:
                end = max(end, i+len(word))
        if end > i:
            isin[i] = True

    res = ""
    temp = ""
    i = 0
    while i < n:
        if isin[i] == False:
            res = res + s[i]
            i = i + 1
            continue
        while isin[i]:
            temp = temp + s[i]
            i = i+1
        res = res + '<b>' + temp + '</b>'
        temp = ""
    
    return res

addBoldTag(s, dict1)
# index = {'Year':5, 'Month':8, 'Day':11}


"""Next Closest Time
Given a time represented in the format "HH:MM", form the next closest time by 
reusing the current digits. There is no limit on how many times a digit can be reused.

You may assume the given input string is always valid. For example, "01:34", 
"12:09" are all valid. "1:34", "12:9" are all invalid.

Example 1:
Input: "19:34"
Output: "19:39"
Explanation: The next closest time choosing from digits 1, 9, 3, 4, is 19:39,
which occurs 5 minutes later.  It is not 19:33, because this occurs 23 hours and 59 minutes later.


Example 2:
Input: "23:59"
Output: "22:22"
Explanation: The next closest time choosing from digits 2, 3, 5, 9, is 22:22. 
It may be assumed that the returned time is next day's time since it is smaller than the input time numerically

"""
import numpy as np
time = "19:34"
nextClosestTime(time)
nextClosestTime("23:59")
def nextClosestTime(time):
    s = list(np.unique([time[0], time[1], time[3], time[4]]))
    ns = len(s)
    hours = time[:2]
    mins = time[3:]
    all_mins = []
    all_hours = []
    for i in range(ns):
        for j in range(ns):
            if int(s[i]+s[j]) < 60:
                all_mins.append(s[i]+s[j])
            if int(s[i]+s[j]) < 24:
                all_hours.append(s[i]+s[j])
    all_mins.sort(key = lambda x: int(x))
    all_hours.sort(key = lambda x: int(x))                

    if mins != max(all_mins):
        return hours + ":" + all_mins[[i for i,x in enumerate(all_mins) if x == mins][0]+1]
    elif hours != max(all_hours):
        return all_hours[[i for i,x in enumerate(all_hours) if x == hours][0]+1] + ":" + all_mins[0]
    else:
        return all_hours[0] + ":" + all_mins[0]


"""Valid Parenthesis String 验证括号字符串 
Given a string containing only three types of characters: '(', ')' and '*', 
write a function to check whether this string is valid. We define the validity of a string by these rules:
1.Any left parenthesis '(' must have a corresponding right parenthesis ')'.
2.Any right parenthesis ')' must have a corresponding left parenthesis '('.
3.Left parenthesis '(' must go before the corresponding right parenthesis ')'.
4.'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string.
5.An empty string is also valid.
Example 1:
    Input: "*("
    Output: False

Example 2:
Input: "(*)"
Output: True

Example 3:
Input: "(*))"
Output: True

"""
# http://www.cnblogs.com/grandyang/p/7617017.html


"""Palindromic Substrings 回文子字符串 
Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different 
substrings even they consist of same characters.

Example 1:
Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

Example 2:
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
"""
s = "aaa"
countSubstrings(s)

def countSubstrings(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    # dp[i][j] records whether s[i:j+1] is Palindromic
    res = 0
    # the order of i/j is crucial
    for i in range(n)[::-1]:
        for j in range(i, n):
            if i == j:
                dp[i][j] = True
                res = res + 1
            elif j == i + 1 or j == i + 2:
                if s[i] == s[j]:
                    dp[i][j] = True
                    res = res + 1
            else:
                dp[i][j] = dp[i+1][j-1] and (s[i] == s[j])
                if dp[i][j]:
                    res = res + 1
    return res

"""Delete Operation for Two Strings
Given two words word1 and word2, find the minimum number of steps required to 
make word1 and word2 the same, where in each step you can delete one character in either string.

Example 1:
Input: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".
"""

word1 = "sea2"
word2 = "eat1as1e1=a2"

minDistance(word1, word2)

def minDistance(word1, word2):
    # longest common subseq, lw1+lw2-2*lcs
    n1 = len(word1)
    n2 = len(word2)
    dp = [[0]*(n2+1) for _ in range(n1+1)]
    # dp[i][j] the longest lcs of word1[:i] and word2[:j]
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return n1 + n2 - 2 * dp[n1][n2]


"""Find Duplicate File in System
Given a list of directory info including directory path, and all the files with 
contents in this directory, you need to find out all the groups of duplicate files
 in the file system in terms of their paths.

A group of duplicate files consists of at least two files that have exactly the same content.

A single directory info string in the input list has the following format:

"root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"

It means there are n files (f1.txt, f2.txt ... fn.txt with content f1_content, 
f2_content ... fn_content, respectively) in directory root/d1/d2/.../dm. Note that
 n >= 1 and m >= 0. If m = 0, it means the directory is just the root directory.

The output is a list of group of duplicate file paths. For each group, it contains
 all the file paths of the files that have the same content. A file path is a 
 string that has the following format:

"directory_path/file_name.txt"

Example 1:

Input:
["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]
Output:  
[["root/a/2.txt","root/c/d/4.txt","root/4.txt"],["root/a/1.txt","root/c/3.txt"]]

Note:

1.No order is required for the final output.
2.You may assume the directory name, file name and file content only has letters 
and digits, and the length of file content is in the range of [1,50].
3.The number of files given is in the range of [1,20000].
4.You may assume no files or directories share the same name in the same directory.
5.You may assume each given directory info represents a unique directory. Directory
 path and file info are separated by a single blank space.
 
 Follow-up beyond contest: 1.Imagine you are given a real file system, how will you search files? DFS or BFS?
2.If the file content is very large (GB level), how will you modify your solution?
3.If you can only read the file by 1kb each time, how will you modify your solution?
4.What is the time complexity of your modified solution? What is the most 
time-consuming part and memory consuming part of it? How to optimize?
5.How to make sure the duplicated files you find are not false positive?


"""
def findDuplicate(s):
    return None

"""Next Greater Element III 下一个较大的元素之三 
Given a positive 32-bit integer n, you need to find the smallest 32-bit integer which has exactly the same digits existing in the integer n and is greater in value than n. If no such positive 32-bit integer exists, you need to return -1.

Example 1:
Input: 12443322
Output: 13222344

Example 2:
Input: 21
Output: -1
"""
n = 12443322
n = 12123213212
n=431

nextGreaterElement(n)

def nextGreaterElement(n):
    n = str(n)
    ns = len(n)
    i = ns - 1
    while i > 0:
        if int(n[i]) > int(n[i - 1]):
            break
        i = i - 1
        
    if i == 0:
        # mono desc
        return -1
    # replace p1[i] with the smallest value in p2 that is greater than p1[i], temp
    # sort the p2 by asc but replace one temp value with the previous p1[i]
    p1 = list(n[:i])
    p2 = list(n[i:])
    temp = p1[-1]
    p1[-1] = str(min([int(x) for x in p2 if int(x) > int(temp)]))
    p2[[i for i,x in enumerate(p2) if x == p1[-1]][0]] = temp
    p2.sort(key = lambda x:int(x))
    return int("".join(p1+p2))


"""Print all combinations of balanced parentheses
Write a function to generate all possible n pairs of balanced parentheses. 

For example,
Input : n=1
Output: ()

Input : n=2
Output: 
()()
(())
"""

def generateParenthesis(n):
    res = []
    curr = ""
    helper(n, n, curr, res)
    return res

def helper(n_left, n_right, curr, res):
    # n_left: left pare remained
    if n_left < 0 or n_left > n_right:
        return None
    if n_left == 0 and n_right == 0:
        res.append(curr)
        return None
    else:
        helper(n_left - 1, n_right, curr + "(", res)
        helper(n_left, n_right - 1, curr + ")", res)
        return None

generateParenthesis(1)
generateParenthesis(2)
generateParenthesis(3)
generateParenthesis(5)



"""Letter Combinations of a Phone Number
"""

"""Converting Decimal Number lying between 1 to 3999 to Roman Numerals
Given a number, find its corresponding Roman numeral.
 Example: 
Input : 9
Output : IX

Input : 40
Output : XL

Input :  1904
Output : MCMIV

3549 - MMMDXLIX
Following is the list of Roman symbols which include subtractive cases also:
SYMBOL       VALUE
I             1
IV            4
V             5
IX            9
X             10
XL            40
L             50
XC            90
C             100
CD            400
D             500
CM            900 
M             1000   
"""

def printRoman(n):
    roman = ["I","IV","V","IX","X","XL","L","XC","C","CD","D","CM","M"]
    values = [1,4,5,9,10,40,50,90,100,400,500, 900,1000]
    curr_i = len(values) - 1
    res = ""
    while curr_i > -1:
        while n >= values[curr_i]:
            res = res + roman[curr_i]
            n = n - values[curr_i]
        curr_i = curr_i - 1
    return res

printRoman(3549)
printRoman(1904)
"""Length of the longest substring without repeating characters
Given a string, find the length of the longest substring without repeating 
characters. For example, the longest substrings without repeating characters 
for "ABDEFGABEF" are "BDEFGA" and "DEFGAB", with length 6. For "BBBB" the longest
substring is “B”, with length 1. For “GEEKSFORGEEKS”, there are two longest 
substrings shown in the below diagrams, with length 7.
"""

s = "ABDEFGABEF"
# idea is to use start to seperate the part that is abandoned
def longestUniqueSubsttr(s):
    ht = {}
    res = 0
    temp_len = 0
    start = 0
    for i in range(len(s)):
        if s[i] not in ht or ht[s[i]] < start:
            ht[s[i]] = i
            temp_len = temp_len + 1
            #res = max(res, temp_len)
        else:
            start = ht[s[i]] + 1
            ht[s[i]] = i
            res = max(res, temp_len)
            temp_len = i - start + 1
                
    return max(res, temp_len)

longestUniqueSubsttr("ECBA")
longestUniqueSubsttr("AAAABBCBA")
longestUniqueSubsttr("ABDEFGABEF")
longestUniqueSubsttr("GEEKSFORGEEKS")


"""Simplify Path
Given an absolute path for a file (Unix-style), simplify it. Note that absolute
 path always begin with ‘/’ ( root directory ), a dot in path represent current
 directory and double dot represents parent directory.

Examples:
"/a/./"   --> means stay at the current directory 'a'
"/a/b/.." --> means jump to the parent directory
              from 'b' to 'a'
"////"    --> consecutive multiple '/' are a  valid  
              path, they are equivalent to single "/".
Input : /home/
Output : /home

Input : /a/./b/../../c/
Output : /c

Input : /a/..
Output : /

Input : /a/../
Ouput : /

Input : /../../../../../a
Ouput : /a

Input : /a/./b/./c/./d/
Ouput : /a/b/c/d

Input : /a/../.././../../.
Ouput : /

Input : /a//b//c//////d
Ouput : /a/b/c/d
"""
s = "/a/./b/../../c/"
s = "/a/./b/./c/../d/"

def simplify(s):
    s1 = s.split("/")
    stack = []
    for i in s1:
        if i == '' or i == '.':
            continue
        elif i == '..' and len(stack) > 0:
            stack.pop()
        else:
            stack.append(i)
    return '/' + '/'.join(stack)

simplify("/a/./b/./c/../d/")
simplify('/a/./b/./c/./d/')


"""Decode Ways
A message containing letters from A-Z is being encoded to numbers using the following mapping:
'A' -> 1
'B' -> 2
...
'Z' -> 26

Given an encoded message containing digits, determine the total number of ways to decode it.
For example,
Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).

The number of ways decoding "12" is 2.
how about "01" ? is this valid?
"""
from string import ascii_uppercase
from random import choice

s = "1213"
s = "01"
def numDecodings(s):
    if len(s) == 0:
        return ''
    letters = ascii_uppercase
    ht = {}
    for i in range(1, 27):
        ht[str(i)] = letters[i-1]
    n = len(s)
    dp = [[] for i in range(n)]
    if s[0] == '0':
        return ''
    else:
        dp[0] = ht[s[0]]
    if n == 1:
        return hp[0]
    
    if s[0] + s[1] in ht:
        dp[1].append(ht[s[0] + s[1]])
    if s[1] in ht:
        dp[1].append(dp[0][0] + ht[s[1]])
    if n == 2:
        return dp[1]
    
    # starting from i = 2
    for i in range(2, n):
        if s[i] != '0':
            dp[i] = list(map(lambda x: x + ht[s[i]], dp[i-1]))
        if s[i-1] + s[i] in ht:
            dp[i] = dp[i] + list(map(lambda x: x + ht[s[i-1] + s[i]], dp[i-2]))
        
        
    return dp[n-1] if len(dp[n-1]) > 0 else ''
   

numDecodings('01')
numDecodings('101')
numDecodings('1201')
numDecodings('121010299')


""" Restore IP Addresses 复原IP地址 
Given a string containing only digits, restore it by returning all possible 
valid IP address combinations.

For example:
Given "25525511135",

return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
"""
s = "25525511135"
def restoreIpAddresses(s):
    res = []
    helper(s, 4, "", res)
    return res

def helper(rem_s, k, temp, res):
    if k == 0:
        if rem_s == "":
            res.append(temp)
            return None
        else:
            return None
    else:
        for i in range(1,4):
            if len(rem_s) >= i and isValid(rem_s[:i]):
                if k == 4:
                    helper(rem_s[i:], k-1, rem_s[:i], res)
                else:
                    helper(rem_s[i:], k-1, temp+'.'+rem_s[:i], res)
    return None

def isValid(ip):
    if len(ip) > 1 and ip[0] == '0':
        return False
    if int(ip) > 255:
        return False
    return True

# 2^n complexity
restoreIpAddresses(s)
restoreIpAddresses("0201211135")
restoreIpAddresses("1020211135")


"""

"""





