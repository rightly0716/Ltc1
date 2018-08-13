# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:17:05 2018

@author: U586086
"""

# leetcode continue...

# Union Find
# Dykstra Algo

"""01 matrix
01matrix Given a matrix consists of 0 and 1, find the distance of the nearest 0
 for each cell.

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
Open the lock. You have a lock in front of you with 4 circular wheels. Each 
wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels 
can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' 
to be '9'. Each move consists of turning one wheel one slot.
The lock initially starts at '0000', a string representing the state of the 4 wheels.
You are given a list of deadends dead ends, meaning if the lock displays any of 
these codes, the wheels of the lock will stop turning and you will be unable to open it.
Given a target representing the value of the wheels that will unlock the lock, 
return the minimum total number of turns required to open the lock, or -1 if 
it is impossible.
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
You are asked to cut off trees in a forest for a golf event. The forest is 
represented as a non-negative 2D map, in this map:
1.0 represents the obstacle can't be reached.
2.1 represents the ground can be walked through.
3.The place with number bigger than 1 represents a tree can be walked through,
 and this positive number represents the tree's height.

You are asked to cut off all the trees in this forest in the order of tree's 
height - always cut off the tree with lowest height first. And after cutting, 
the original place has the tree will become a grass (value 1).
You will start from the point (0, 0) and you should output the minimum steps 
you need to walk to cut off all the trees. If you can't cut off all the trees, 
output -1 in that situation.
You are guaranteed that no two trees have the same height and there is at least 
one tree needs to be cut off.

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

Given times, a list of travel times as directed edges times[i] = (u, v, w), 
where u is the source node, v is the target node, and w is the time it takes 
for a signal to travel from source to target.

Now, we send a signal from a certain node K. How long will it take for all nodes 
to receive the signal? If it is impossible, return -1.

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
There is a ball in a maze with empty spaces and walls. The ball can go through 
empty spaces by rolling up, down, left or right, but it won't stop rolling until 
hitting a wall. When the ball stops, it could choose the next direction.
Given the ball's start position, the destination and the maze, find the shortest
 distance for the ball to stop at the destination. The distance is defined by 
 the number of empty spaces traveled by the ball from the start position (excluded) 
 to the destination (included). If the ball cannot stop at the destination, return -1.
The maze is represented by a binary 2D array. 1 means the wall and 0 means the 
empty space. You may assume that the borders of the maze are all walls. The start
 and destination coordinates are represented by row and column indexes.
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
For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, 
return 2 because 13 = 4 + 9.
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
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge
 is a pair of nodes),
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
Redundant Connection: can use the same idea

Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge 
is a pair of nodes), 
write a function to check whether these edges make up a valid tree.

For example:

Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.
Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.
Hint:
1.Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], what should your return? Is
 this case a valid tree?
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

# union find ： check whether two points connect to each other in a graph, cannot return
# the path
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

There are a total of 2 courses to take. To take course 1 you should have finished 
course 0. So it is possible.
2, [[1,0],[0,1]]

There are a total of 2 courses to take. To take course 1 you should have 
finished course 0, and to take course 0 you should also have finished course 1.
 So it is impossible.
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
finished course 0. So one correct course order is [0,1,2,3]. Another correct 
ordering is[0,2,1,3].
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
Given a binary tree, imagine yourself standing on the right side of it, return
 the values of the nodes you can see ordered from top to bottom.

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
Clone an undirected graph. Each node in the graph contains a label and a list of 
its neighbors.

How we serialize an undirected graph:
Nodes are labeled uniquely.
We use # as a separator for each node, and , as a separator for node label 
and each neighbor of the node.
As an example, consider the serialized graph {0,1,2#1,2#2,2}.
The graph has a total of three nodes, and therefore contains three parts as separated by #.
First node is labeled as 0. Connect node 0 to both nodes 1 and 2. Second node 
is labeled as 1. Connect node 1 to node 2. Third node is labeled as 2. Connect 
node 2 to node 2 (itself), thus forming a self-cycle. Visually, the graph looks 
like the following:
"""
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

def cloneGraph(node):
    return None



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
board = [['X', 'X', 'X', 'X'], ['X', 'O', 'O', 'X'], ['X', 'X', 'O','X'], 
         ['X', 'O', 'X', 'X']]

def solve(board):
    m,n = len(board), len(board[0])
    if m < 2 or n < 2:
        return board
    for i in range(n):
        dfs(0, i, board)
        dfs(m-1, i, board)
    for j in range(m):
        dfs(j, 0, board)
        dfs(j, n-1, board)
        
    for i in range(n):
        for j in range(m):
            if board[j][i] == 'O':
                board[j][i] = 'X'
            elif board[j][i] == 'I':
                 board[j][i] = 'O'
    return board

def dfs(j, i, board):
    if j < 0 or j > m-1 or i < 0 or i > n-1 or board[j][i] in ['I', 'X']:
        return None
    else:
        board[j][i] = 'I'
        dfs(j-1, i, board)
        dfs(j+1, i, board)
        dfs(j, i-1, board)
        dfs(j, i+1, board)
    return None

solve([['X', 'X', 'X', 'X'], ['X', 'O', 'O', 'X'], ['X', 'X', 'O','X'], 
         ['X', 'O', 'X', 'X']])
solve([['O', 'X', 'X', 'X'], ['X', 'X', 'O', 'X'], ['O', 'O', 'X','X'], 
         ['X', 'O', 'X', 'X']])
    
"""
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
We are stacking blocks to form a pyramid. Each block has a color which is a one 
letter string, like 'Z'.
For every block of color C we place not in the bottom row, we are placing it on
 top of a left block of color A and right block of color B. We are allowed to
 place the block there only if (A, B, C) is an allowed triple.
We start with a bottom row of bottom, represented as a single string. We also
 start with a list of allowed triples allowed. Each allowed triple is represented 
 as a string of length 3.

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
The first and third John's are the same person as they have the common email 
"johnsmith@mail.com".
The second John and Mary are different people as none of their email addresses 
are used by other accounts.
We could return these lists in any order, for example the answer 
[['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would 
still be accepted.
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
the time spent by calling other functions should not be considered as this function's 
exclusive time. You should return the exclusive time of each function sorted by their function id.
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
Given a string of numbers and operators, return all possible results from 
computing all the different possible ways to group numbers and operators. The 
valid operators are +, -and *.

Example 2

Input: "2*3-4*5"
(2*(3-(4*5))) = -34
((2*3)-(4*5)) = -14
((2*(3-4))*5) = -10
(2*((3-4)*5)) = -10
(((2*3)-4)*5) = 10

Output: [-34, -14, -10, -10, 10]

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
"""

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
    #for j in range(target, nums[i]-1): # start at a larger val target
    # dp[j] = dp[j] || dp[j - nums[i]]
# each i loop will update all the dp[] that can be made by summing nums[:i]

# A clear idea: Actually, this is a 0/1 knapsack problem, for each number, we can pick it 
# or not. Let us assume dp[i][j] means whether the specific sum j can be gotten
# from the first i numbers. If we can pick such a series of numbers from 0-i 
# whose sum is j, dp[i][j] is true, otherwise it is false.

# Base case: dp[0][0] is true; (zero number consists of sum 0 is true)
# Transition function: For each number, if we don’t pick it, dp[i][j] = dp[i-1][j],
# which means if the first i-1 elements has made it to j, dp[i][j] would also 
# make it to j (we can just ignore nums[i]). If we pick 
# nums[i]. dp[i][j] = dp[i-1][j-nums[i]], which represents that j is composed of 
# the current value nums[i] and the remaining composed of other previous numbers.
# Thus, the transition function is dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]

# another idea:
def canPartition(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    sums = sum(nums)
    if sums & 1: return False
    nset = set([0])
    for n in nums:
        for m in nset.copy():
            nset.add(m + n)
    return sums / 2 in nset

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
    # res[i] indicates whether s[:i] can be seged
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
    return None

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
For a undirected graph with tree characteristics, we can choose any node as the
 root. The result graph is then a rooted tree. Among all possible rooted trees,
 those with minimum height are called minimum height trees (MHTs). Given such a
 graph, write a function to find all the MHTs and return a list of their root labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. You will be given
 the number n and a list of undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. Since all edges 
are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

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
square brackets is being repeated exactly k times. Note that k is guaranteed to
 be a positive integer.

You may assume that the input string is always valid; No extra white spaces, 
square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits 
and that digits are only for those repeat numbers, k. For example, there won't 
be input like 3a or 2[4].

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
    s_str = [] # stack for string
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
            

"""Basic Calculator II 基本计算器之二
Implement a basic calculator to evaluate a simple expression string.
The expression string contains only non-negative integers, +, -, *, / operators and empty spaces . 
The integer division should truncate toward zero.
You may assume that the given expression is always valid.

Some examples:
"3+2*2" = 7
" 3/2 " = 1
" 13+5 / 2 " = 15
Note: Do not use the eval built-in library function.
"""
# use stack to save numbers in +/- operations
s = " -1 + 2 - 13+5 / 2 " #-10
def calculate(s):
    Stack = []
    n = len(s)
    temp = ''
    op = '+' # previous operator
    for i in range(n):
        if s[i].isdigit():
            temp = temp + s[i]
        if s[i] in ['+', '-', '*', '/'] or i == n-1:
            # if operator, need previous operator
            if op == '-':
                if temp != '':
                    Stack.append(int(temp) * -1)
            elif op == '+':
                if temp != '':
                    Stack.append(int(temp))
            elif op == '*':
                top_num = Stack.pop()
                Stack.append(top_num*int(temp))
            elif op == '/':
                top_num = Stack.pop()
                Stack.append(top_num // int(temp))
            op = s[i]
            temp = ''
    res = 0
    while len(Stack) > 0:
        res  = res + Stack.pop()
    return res

calculate(s)
calculate(" -1 + 2*2 - 13+5 / 2 /2 + 1")

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
closed bold tag. Also, if two substrings wrapped by bold tags are consecutive, 
you need to combine them.

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
which occurs 5 minutes later.  It is not 19:33, because this occurs 23 hours 
and 59 minutes later.

Example 2:
Input: "23:59"
Output: "22:22"
Explanation: The next closest time choosing from digits 2, 3, 5, 9, is 22:22. 
It may be assumed that the returned time is next day's time since it is smaller
 than the input time numerically
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
write a function to check whether this string is valid. We define the validity
 of a string by these rules:
1.Any left parenthesis '(' must have a corresponding right parenthesis ')'.
2.Any right parenthesis ')' must have a corresponding left parenthesis '('.
3.Left parenthesis '(' must go before the corresponding right parenthesis ')'.
4.'*' could be treated as a single right parenthesis ')' or a single left 
            parenthesis '(' or an empty string.
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
# 当循环结束后，我们希望left中没有多余的左括号，就算有，我们可以尝试着用星号来抵消，
# 当star和left均不为空时，进行循环，如果left的栈顶左括号的位置在star的栈顶星号的右边，
# 那么就组成了 *( 模式，直接返回false

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
make word1 and word2 the same, where in each step you can delete one character 
in either string.

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
["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)",
 "root 4.txt(efgh)"]
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
 
 Follow-up beyond contest: 1.Imagine you are given a real file system, how will
 you search files? DFS or BFS?
2.If the file content is very large (GB level), how will you modify your solution?
3.If you can only read the file by 1kb each time, how will you modify your solution?
4.What is the time complexity of your modified solution? What is the most 
time-consuming part and memory consuming part of it? How to optimize?
5.How to make sure the duplicated files you find are not false positive?
"""
def findDuplicate(s):
    return None

"""Next Greater Element III 下一个较大的元素之三 
Given a positive 32-bit integer n, you need to find the smallest 32-bit integer
 which has exactly the same digits existing in the integer n and is greater in
 value than n. If no such positive 32-bit integer exists, you need to return -1.

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
Given a digit string, return all possible letter combinations that the number could represent.
A mapping of digit to letters (just like on the telephone buttons) is given below.
1 - ""  2 - "abc" 3 - "def" 4 - "ghi" 5 - "jkl" 6 - "mno"
7 - "pqrs" 8 - "tuv" 9 - "wxyz"

Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
"""
ht = {}
ht['1'] = ''   ; ht['2'] = 'abc'; ht['3'] = 'def'
ht['4'] = 'ghi'; ht['5'] = 'jkl'; ht['6'] = 'mno'
ht['7'] = 'pqrs'; ht['8'] = 'tuv'; ht['9'] = 'wxyz'

s = "23"
s = "12213"
from collections import deque

def phone(s):
    q = deque()
    q.append('')
    for i in s:
        if i == '1':
            continue
        for _ in range(len(q)):
            tmp = q.popleft()
            for j in ht[i]:
                q.append(tmp + j)
    return list(q)
            
phone(s)


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
s = '121010299'

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


"""Best Time to Buy and Sell Stock with Cooldown 买股票的最佳时间含冷冻期 

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions 
as you like (ie, buy one and sell one share of the stock multiple times) with 
the following restrictions:
•You may not engage in multiple transactions at the same time (ie, you must sell 
the stock before you buy again).
•After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)

Example:
prices = [1, 2, 3, 0, 2]
maxProfit = 3
transactions = [buy, sell, cooldown, buy, sell]

对于每一天，有三种动作，buy, sell, cooldown, sell 和 cooldown 可以合并成一种状态，
因为手里最终没有股票。最终需要的结果是 sell，即手里股票卖了获得最大利润。我们可以用两
个数组来记录当前持股和未持股的状态，令 sell[i]  表示第i天未持股时，获得的最大利润， 
buy[i] 表示第i天持有股票时，获得的最大利润。
对于 sell[i] ，最大利润有两种可能，一是今天没动作跟昨天未持股状态一样，二是今天卖了股票，
所以状态转移方程如下：
sell[i] = max{sell[i - 1], buy[i-1] + prices[i]} 
对于 buy[i] ，最大利润有两种可能，一是今天没动作跟昨天持股状态一样，二是前天卖了股票，
今天买了股票，因为 cooldown 只能隔天交易，所以今天买股票要追溯到前天的状态。状态转移方程如下：
buy[i] = max{buy[i-1], sell[i-2] - prices[i]} 
最终我们要求的结果是 sell[n - 1] ，表示最后一天结束时，手里没有股票时的最大利润。
"""


"""Count Primes 质数的个数 
Description:
Count the number of prime numbers less than a non-negative number, n
我们从2开始遍历到根号n，先找到第一个质数2，然后将其所有的倍数全部标记出来，然后到下一个质数3，
标记其所有倍数，一次类推，直到根号n，此时数组中未被标记的数字就是质数。我们需要一个n-1长度的
bool型数组来记录每个数字是否被标记，长度为n-1的原因是题目说是小于n的质数个数，并不包括n

For each i = 2,3,...,int(sqrt(n)), we can start with i^2 after observation
limit = sqrt(n)
for (int i = 2; i <= limit; ++i) {
            if (num[i - 1]) {
                for (int j = i * i; j < n; j += i) {
                    num[j - 1] = false;
                }
            }
        }
"""

"""Lowest Common Ancestor of a Binary Search Tree BST的最小共同父节点 
Given a binary search tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
"""
# the idea is to report the two paths to the two nodes, and return the last overlap
# node of them
# while p.val and q.val are on one side of the root.val:
#   update root
# return root

"""Lowest Common Ancestor of a Binary Tree 二叉树的最小共同父节点
Given a binary tree, find the lowest common ancestor (LCA) of two given NODEs in the tree.
According to the definition of LCA on Wikipedia: “The lowest common ancestor is 
defined between two nodes v and w as the lowest node in T that has both v and w
 as descendants (where we allow a node to be a descendant of itself).”
        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example
is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according
to the LCA definition.
"""
def lowestCommonAncestor(root, p, q):
    if root == None or p == root or q == root:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left is not None:
        return left
    if right is not None:
        return right
    return None

"""Longest Palindromic Subsequence 最长回文子序列 
Given a string s, find the longest palindromic subsequence's length in s. 
You may assume that the maximum length of s is 1000.

Example 1:
Input:
"bbbab"
Output:
4

One possible longest palindromic subsequence is "bbbb".
"""
s = "bbbab"

def longestPalindromeSubseq(s):
    n = len(s)
    if n == 0:
        return None
    elif n == 1:
        return 1
    dp = [[0]*n for _ in range(n)]
    # dp[i][j] stores the length of LPS in s[i:j+1]
    # the order of filling dp is from bottom left to top right
    # OR from bottom right to the top
    for i in range(n):
        dp[i][i] = 1
    
    for i in range(n-1):
        j = i + 1
        dp[i][j] = 2 if s[i] == s[j] else 1
          
    for k in range(2,n):
        for i in range(n-k):
            j = i + k
            dp[i][j] = dp[i+1][j-1] + 2 if s[i] == s[j] else max(dp[i][j-1], dp[i+1][j])
    
    return dp[0][n-1]

longestPalindromeSubseq(s)

"""Solve the Equation
Input: "2x+3x-6x=x+2"
Output: "x=-1"

Input: "x=x+2"
Output: "No solution"

Input: "x=x"
Output: "Infinite solution"
"""
eq = "2x+3x-6x=x+2"
s = eq_left
def solveEquation(eq):
    eq_left, eq_right = eq.split('=')
    a_left, b_left = helper(eq_left)
    a_right, b_right = helper(eq_right)
    if a_left != a_right:
        return "x="+(b_right-b_left)/(a_left-a_right)
    elif b_left == b_right:
        return "Infinite solution"
    else: 
        return "No solution"

def helper(s):
    a = 0
    b = 0
    n = len(s)
    tmp_num = ''
    sign = 1
    i = 0
    while i < n:
        if s[i] == '+':
            sign = 1
            i = i + 1
        elif s[i] == '-':
            sign = -1
            i = i + 1
        elif s[i] == 'x':
            a = a + sign
            i = i + 1
        else:
            while i < n and s[i].isdigit():
                tmp_num = tmp_num + s[i]
                i = i + 1
            
            if i == n:
                b = b + int(tmp_num) * sign
            elif s[i] == 'x':
                a = a + int(tmp_num) * sign
                i = i + 1
            else:
                b = b + int(tmp_num) * sign
            tmp_num = ''

    return [a,b]

"""Generate Parentheses 生成括号 
Given n pairs of parentheses, write a function to generate all combinations of 
well-formed parentheses.

For example, given n = 3, a solution set is:
"((()))", "(()())", "(())()", "()(())", "()()()"
"""
n = 3

def geneParenthesis(n):
    out = []
    helper(out, '', n, n)
    return out

def helper(out, s, nl, nr):
    # nl and nr are the num of (/) left
    if nl == 0 and nr == 0:
        out.append(s)
        return None
    if nl > nr or nl < 0 or nr < 0:
        return None
    helper(out, s+'(', nl-1, nr)
    helper(out, s+')', nl, nr-1)
    return None
    

geneParenthesis(3)
len(geneParenthesis(4))


"""Longest Increasing Subsequence (LIS) (Size )

https://en.wikipedia.org/wiki/Longest_increasing_subsequence
https://www.geeksforgeeks.org/longest-monotonically-increasing-subsequence-size-n-log-n/
https://stackoverflow.com/questions/3992697/longest-increasing-subsequence
M数组内任意元素M[i]，记录的是最长递增子序列长度为i的序列的末尾元素的值，
也就是这个最长递增子序列的最大元素的大小值。
P saves the index of previous number in the LIS
Use  M to find the last element, then use P to recover the whole seq (reversely)
"""

s = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
LIS(s)

def LIS(s):
    n = len(s)
    M = [None] * n
    P = [None] * n
    M[0] = 0
    L = 1 # length of LIS
    
    for i in range(1, n):
        if s[i] > s[M[L-1]]:
            P[i] = M[L-1]
            M[L] = i
            L = L + 1
        else:
            lower = BST(M[:L], s[i]) # find the smallest i in M that > s[i]
            M[lower] = i
            P[i] = M[lower - 1]
    
    pos = M[L-1]
    res = []
    for _ in range(L):
        res.append(s[pos])
        pos = P[pos]
    return res[::-1]
            
def BST(s, x):
    lower = 0
    upper = len(s)
    while upper - lower > 1:
        mid = (upper + lower) // 2
        if s[mid] > x:
            upper = mid
        else:
            lower = mid
    
    return lower

"""Pow(x, n) 求x的n次方 
Implement pow(x, n).
"""
def pow(x, n):
    if n == 0:
        if x == 0:
            return None
        else:
            return 1
    if n < 0:
        if x == 0:
            return None
        else:
            return 1/pow(x, -1*n)
    else:
        if x == 0:
            return 0
        if n == 1:
            return x
        elif n % 2 == 0:
            tmp = pow(x, n/2)
            return tmp * tmp
        else: 
            tmp = pow(x, n//2)
            return x * tmp * tmp


pow(2, 10)
pow(2, -1)
pow(-2, 0)
pow(0, -2)
pow(2, 11)


"""[LeetCode] Spiral Matrix 螺旋矩阵 
Given a matrix of m x n elements (m rows, n columns), return all elements of 
the matrix in spiral order.

For example,
Given the following matrix:
mat = [
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
You should return [1,2,3,6,9,8,7,4,5].

Each while loop completes a circle, need to check when finishing each edge of 
the circle
"""
def spiral(mat):
    M, N = len(mat), len(mat[0])
    # suppose m, n > 0
    m, n = 0, -1
    res = []
    while min(M, N) > 0:
        # from left to right
        for i in range(N):
            n = n + 1
            res.append(mat[m][n])
        M = M - 1
        if M == 0:
            break
        # from up to down
        for j in range(M):
            m = m + 1
            res.append(mat[m][n])
        N = N - 1
        if N == 0:
            break
        # from right to left
        for i in range(N):
            n = n - 1
            res.append(mat[m][n])
        M = M - 1
        if M == 0:
            break
        # from down to up
        for j in range(M):
            m = m - 1
            res.append(mat[m][n])
        N = N - 1
        if N == 0:
            break
    return res

spiral(mat)
spiral([  [ 1, 2, 3, 31], [ 4, 5, 6, 61], [ 7, 8, 9, 91] ])

"""[LeetCode] Merge Intervals 合并区间 
Given a collection of intervals, merge all overlapping intervals.

For example,
 Given [1,3],[2,6],[8,10],[15,18],
 return [1,6],[8,10],[15,18]. 
"""
s = [[2,6],[8,10],[1,3],[15,18]]

def mergeInterval(s):
    s1 = sorted(s, key = lambda x: x[0])
    ns = len(s1)
    res = []
    begin, end = s1[0]
    for i in range(1, ns):
        tmp_b, tmp_e = s1[i]
        if end < tmp_b:
            res.append([begin, end])
            begin, end = tmp_b, tmp_e
        else:
            end = max(end, tmp_e)
    res.append([begin, end])
    return res

mergeInterval(s)
mergeInterval([[2,6],[8,10],[1,3],[15,18],[-1,2],[17,20]])

"""[LeetCode] Longest Consecutive Sequence 求最长连续序列
Given an unsorted array of integers, find the length of the longest consecutive
elements sequence.

For example,
 Given [100, 4, 200, 1, 3, 2],
 The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4. 

Your algorithm should run in O(n) complexity. 
"""
# use a hashset to store, go thourhg each element to find its  longest conse seq
s = [100, 4, 200, 1, 3, 2]
s = [100, 4, 200, 1, 3, 2,0]
def longestConsecutive(s):
    hs = set(s)
    max_length = 1
    for i in set(s):
        if i in hs:
            begin, end = i,i
            hs.remove(i)
            while (begin - 1) in hs:
                hs.remove(begin - 1)
                begin = begin - 1
            while (end + 1) in hs:
                hs.remove(end + 1)
                end = end + 1
            max_length = max(max_length, end - begin + 1)
        
    return max_length

longestConsecutive(s)
LCS(s)
# use hash table / dict to record the visit status
def LCS(s):
    ht = {x: True for x in s}
    max_length = 1
    for i in ht:
        if ht[i]:
            begin, end = i,i
            ht[i] = False
            while (begin - 1) in ht:
                ht[begin - 1] = False
                begin = begin - 1
            while (end + 1) in ht:
                ht[end + 1] = False
                end = end + 1
            max_length = max(max_length, end - begin + 1)
    return max_length


"""[LeetCode] Find Peak Element 求数组的局部峰值 
A peak element is an element that is greater than its neighbors.

Given an input array where num[i] != num[i+1], find a peak element and return its index.
The array may contain multiple peaks, in that case return the index to any one 
of the peaks is fine. You may imagine that num[-1] = num[n] = -inf (guaratee 
existence).

For example, in array [1, 2, 3, 1], 3 is a peak element and your function 
should return the index number 2.

The runetime should be better then O(n)
"""
nums = [1,2,3,4,2,1]

def findPeakElement(nums):
    n = len(nums)
    if n == 1:
        return nums[0]
    high = n - 1
    low = 0
    while low < high:
        mid = (high + low) // 2
#        if nums[mid + 1] < nums[mid] and nums[mid - 1] < nums[mid]:
#            return mid
        if nums[mid + 1] > nums[mid]:
            low = mid + 1
        else: 
            #if nums[mid - 1] > nums[mid]:
            high = mid
            # must contain mid as the endpoint here
    return high

findPeakElement(nums)
findPeakElement([1,2,3,-2,-1,1,2])
findPeakElement([1,2,3,4,5])
findPeakElement([1,2,3,4,5][::-1])


""" Rotate Image 旋转图像
You are given an n x n 2D matrix representing an image.
Rotate the image by 90 degrees (clockwise).
1  2  3　　　 　　 7  4  1　

4  5  6　　-->　　 8  5  2　　

7  8  9 　　　 　　9  6  3
Follow up:
Could you do this in-place?
Idea1:对于当前位置，计算旋转后的新位置，然后再计算下一个新位置，
第四个位置又变成当前位置了，所以这个方法每次循环换四个数字，如下所示：
1  2  3                 7  2  1         7  4  1

4  5  6      -->       4  5  6　　 -->  　 8  5  2　　

7  8  9                 9  8  3　　　　　  9  6  3
Idea 2
还有一种解法，首先以从对角线为轴翻转，然后再以x轴中线上下翻转即可得到结果，
如下图所示(其中蓝色数字表示翻转轴)：

1  2  3　　　 　　 9  6  3　　　　　      7  4  1

4  5  6　　-->　　 8  5  2　　 -->   　 8  5  2　　

7  8  9 　　　 　　7  4  1　　　　　      9  6  3
"""
import numpy as np
m = np.array(range(16)).reshape(4,4)
m = np.array(range(1,10)).reshape(3,3)
def rotate(m):
    n = len(m)
    for ir in range(n):
        for ic in range(ir, n):
            if ir != ic:
                temp = m[ir][ic]
                m[ir][ic] = m[ic][ir]
                m[ic][ir] = temp
    
    for ir in range(n):
        for ic in range(n//2):
            temp = m[ir][ic]
            m[ir][ic] = m[ir][n-ic-1]
            m[ir][n-ic-1] = temp
    return m

rotate(m)


"""Jump Game 跳跃游戏
Given an array of non-negative integers, you are initially positioned at the 
first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.

Example 1:
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
             jump length is 0, which makes it impossible to reach the last index.
"""
s1 = [2,3,1,1,4]
s2 = [3,2,1,0,4]
s3 = [3,0,1,0]
s3 = [1,1,1,0]
s3 = [0,0,0,0]
# Greedy: use reach to record the longest place that can go
def canJump(s):
    n = len(s)
    reach = 0
    for i in range(n):
        reach = max(i+s[i], reach)
        if i == reach:
            break
        if reach >= n - 1:
            return True
    return False

canJump(s1)
canJump(s2)
canJump(s3)

"""Jump Game II 跳跃游戏之二
Given an array of non-negative integers, you are initially positioned at the 
first index of the array.
Each element in the array represents your maximum jump length at that position.
Your goal is to reach the last index in the minimum number of jumps.

For example:
Given array A = [2,3,1,1,4]

The minimum number of jumps to reach the last index is 2. (Jump 1 step from 
index 0 to 1, then 3 steps to the last index.)
"""
# greedy idea.
# use last to record the previous largest step
# curr to explore the current largest, replace last when i exceeds it
# step plus one evert time we replace last with curr
# the places that last visits are not the actual route

nums = [2,3,1,1,4]
nums = [2,1,0,0,4]
def jump(nums):
    n = len(nums)
    last = 0
    curr = 0
    step = 0
    for i in range(n):
        curr = max(nums[i] + i, curr)
        if i == last:
            last = curr
            step = step + 1
            if curr >= n - 1:
                break
        if i == curr:
            return -1

    return step

jump(nums)
jump([3,0,1,0])

"""Combinations 组合项
Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

For example,
If n = 4 and k = 2, a solution is:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4]]
"""
def combine(n, k):
    if k>n:
        return []
    out = []
    helper(out, [], list(range(1, n+1)), k)
    return out

def helper(out, s_curr, s_remain, k):
    if len(s_curr) == k:
        out.append(s_curr)
        return None
    for i in range(len(s_remain)):
        helper(out, s_curr + [s_remain[i]], s_remain[(i+1):], k)
    return None

combine(4,2)
combine(2,3)
# combine(10,3)

"""Permutations II 全排列之二
Given a collection of numbers that might contain duplicates, return all possible 
unique permutations.
For example,
[1,1,2] have the following unique permutations:
[1,1,2], [1,2,1], and [2,1,1].
"""
nums = [1,2,3,3]
def permute(nums):
    out = []
    hs = {}
    helper(out, [], nums, hs)
    return out

def helper(out, s_curr, s_remain, hs):
    if len(s_remain) == 0:
        out.append(s_curr)
        return None
    hs = set()
    for i in range(len(s_remain)):
        if s_remain[i] not in hs:
            helper(out, s_curr + [s_remain[i]], s_remain[:i]+s_remain[(i+1):], hs)
            hs.add(s_remain[i])
    return None
    
permute([1,2,3])
permute([1,2,3,3])

"""Flatten Binary Tree to Linked List 453
Given a binary tree, flatten it to a linked list in-place.

For example,
Given
         1
        / \
       2   5
      / \   \
     3   4   6
The flattened tree should look like:
   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6
"""
# idea: recursively do left and right, and then combine 
# when combine, 
def flatten(root):
    if root == None:
        return None
    if root.left != None:
        flatten(root.left)
    if root.right != None:
        flatten(root.right)
    head = root
    tmp = root.right
    root.right = root.left
    root.left = None
    while root.right is not None:
        root = root.right
    root.right = tmp
    return head
    
"""Populating Next Right Pointers in Each Node 每个节点的右向指针
Given a binary tree
    struct TreeLinkNode {
      TreeLinkNode *left;
      TreeLinkNode *right;
      TreeLinkNode *next;
    }
Populate each next pointer to point to its next right node. If there is no next
 right node, the next pointer should be set to NULL.
Initially, all next pointers are set to NULL.
Note:
You may only use constant extra space.
You may assume that it is a perfect binary tree (ie, all leaves are at the same
 level, and every parent has two children).
For example,
Given the following perfect binary tree,
         1
       /  \
      2    3
     / \  / \
    4  5  6  7
After calling your function, the tree should look like:
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \  / \
    4->5->6->7 -> NULL
"""
# idea, left is easy, for right, see whether its father.next is None 
def connect(root):
    if root is None:
        return None
    head = root
    if root.left is not None:
        root.left.next = root.right
        if root.next is None:
            root.right.next = None
        else:
            root.right.next = root.next.left
    connect(root.left)
    connect(root.right)
    return head

from collections import deque
def connect_nonrec(root):
    if root is None:
        return None
    q = deque()
    q.append(root)
    while len(q) != 0:
        for i in range(len(q)):
            curr = q.popleft()
            if i == len(q):
                curr.next = None
            else:
                curr.next = q[0]
            if curr.left is not None:
                q.append(curr.left)
            if curr.right is not None:
                q.append(curr.right)
    return root
    
"""Populating next right pointers in each node
Write a function to connect all the adjacent nodes at the same level in a binary 
tree. Structure of the given Binary Tree node is like following.
struct node{
  int data;
  struct node* left;
  struct node* right;
  struct node* nextRight;  
}
Initially, all the nextRight pointers point to garbage values. Your function 
should set these pointers to point next right for each node.
Example:
Input Tree
       A
      / \
     B   C
    / \   \
   D   E   F
Output Tree
       A--->NULL
      / \
     B-->C-->NULL
    / \   \
   D-->E-->F-->NULL
"""
# for each level, find the nexRight of the right kid at next level.
def connect2(root):
    if root is None:
        return root
    p = root.nextRight
    while p is not None:
        if p.left is not None:
            p = p.left
            break
        if node.right is not None:
            p = p.right
            break
        p = p.nextRight
    # p is the nextRight of the right kid if exists
    if root.right is not None:
        root.right.nextRight = p
    if root.left is not None:
        if root.right is not None:
            root.left.nextRight = root.right
        else:
            root.left.nextRight = p
    connect(root.left)
    connect(root.right)
    return root

# non recurrsive can refer to the previous solution

"""Find the maximum path sum between two leaves of a binary tree
Given a binary tree in which each node element contains a number. 
Find the maximum possible sum from one leaf node to another.
The maximum sum path may or may not go through root. 
 Expected time complexity is O(n).
 
If one side of root is empty, then function should return minus infinite
"""

def maxPathSum(root):
    if root is None:
        return float('-Inf')
    res = [float('-Inf')]
    helper(root, res)
    return res[0]

def helper(node, res):
    # return the one-side largest path including node, from left or right
    # gives res the value of the largest path in its subtree (may not pass it)
    if node is None:
        return 0
    left = max(helper(node.left, res), 0)
    right = max(helper(node.right, res), 0)
    res[0] = max(res[0], left + right + node.val)
    return max(left, right) + node.val

"""Maximum Subarray 最大子数组
Find the contiguous subarray within an array (containing at least one number) 
which has the largest sum.

For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
the contiguous subarray [4,−1,2,1] has the largest sum = 6.
"""
nums = [-2,1,-3,4,-1,2,1,-5,4]

def maxSubArray(nums):
    cursum = 0
    res = 0
    n = len(nums)
    for i in range(n):
        cursum = max(cursum + nums[i], nums[i])
        # cursum contains the largest sum ends with nums[i]
        res = max(res, cursum)
    
    return res

maxSubArray(nums)

"""Valid Palindrome
Question
Given a string, determine if it is a palindrome, considering only alphanumeric 
characters and ignoring cases.

Notice
Have you consider that the string might be empty? This is a good question to 
ask during an interview.
For the purpose of this problem, we define empty string as valid palindrome.

Example
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.
Challenge
O(n) time without extra memory.
"""
s = "A man, a plan, a canal: Panama"
s1 = "race a car"
s2= 'ba   3314134 A   b'
s3 = 'a ba    a    '
def isPalindrome(s):
    n = len(s)
    if n == 0:
        return True
    i,j = 0,n-1
    while i < j:
        while (not s[i].isalpha()) and i < n:
            i = i + 1
        while not s[j].isalpha() and j > -1:
            j = j - 1
        if s[i].isalpha() and s[j].isalpha():
            if s[i].lower() != s[j].lower():
                return False
        else:
            return True
        i = i+1
        j = j-1
    return True
            
isPalindrome(s)
isPalindrome(s1)
isPalindrome(s2)
isPalindrome(s3)


"""Copy List with Random Pointer 拷贝带有随机指针的链表
A linked list is given such that each node contains an additional random pointer
 which could point to any node in the list or null.

Return a deep copy of the list.
考虑用Hash map来缩短查找时间，第一遍遍历生成所有新节点时同时建立一个原节点和新节点的
哈希表，第二遍给随机指针赋值时，查找时间是常数级
"""

"""Reverse words in a string
Example: Let the input string be "i like this program very much". 
The function should change the string to "much very program this like i"
"""
# 先把每个单词翻转一遍，再把整个字符串翻转一遍，或者也可以调换个顺序，先翻转整个字符串，再翻转每个单词
s = "i like this program very much"
def reverse(s):
    return ' '.join(s.split(' ')[::-1])
reverse(s)

"""ind Minimum in Rotated Sorted Array II
Suppose a sorted array is rotated at some pivot unknown to you beforehand.
(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
Find the minimum element.
The array may contain duplicates.
"""
nums = [4,4,5,6,7,0,1,2] #return 0
num2 = [2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2] # return 0
def findMin(nums):
    n = len(nums)
    if n == 0:
        return None
    if n == 1:
        return nums[0]

    left = int(0)
    right = int(n-1)
    while left < right:
        mid = left + (right - left)//2
        if nums[mid] < nums[left]:
            right = mid
        elif nums[mid] > nums[left]:
            # the left side must be monotonic
            left = mid + 1
        else:
            # nums[mid] == nums[left]
            left = left + 1
    return nums[left]
    #return min(nums[left], nums[right]), while left < right - 1


findMin(nums)
findMin(num2)
findMin([3,4,5,6,7,0,1,2])
findMin([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

"""Number of 1 Bits 位1的个数
Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).

For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
"""
n = '00000000000000000000000000001011'
def hammingWeight(n):
    res = 0
    n = int(n, 2)
    for i in range(32):
        res = res + (n&1)
        n = n >> 1
    return res
hammingWeight(n)

"""Implement Trie (Prefix Tree) 实现字典树(前缀树)
Implement a trie with insert, search, and startsWith methods.
Note:
You may assume that all inputs are consist of lowercase letters a-z.

字典树主要有如下三点性质：

1. 根节点不包含字符，除根节点意外每个节点只包含一个字符。
2. 从根节点到某一个节点，路径上经过的字符连接起来，为该节点对应的字符串。
3. 每个节点的所有子节点包含的字符串不相同。

Idea: 对每个结点开一个字母集大小的数组，对应的下标是儿子所表示的字母，内容则是这个儿子对应在大数组上的位置，即标号；
"""


"""House Robber II 打家劫舍之二
Note: This is an extension of House Robber.

After robbing those houses on that street, the thief has found himself a new place for his thievery so that he will not 
get too much attention. This time, all houses at this place are arranged in a circle. That means the first house is the 
neighbor of the last one. Meanwhile, the security system for these houses remain the same as for those in the previous street.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of 
money you can rob tonight without alerting the police.
"""

# hint: 这道题是之前那道House Robber 打家劫舍的拓展，现在房子排成了一个圆圈，则如果抢了第一家，就不能抢最后一家，因为首尾相连了，
# 所以第一家和最后一家只能抢其中的一家，或者都不抢，那我们这里变通一下，如果我们把第一家和最后一家分别去掉，各算一遍能抢的最大值，
# 然后比较两个值取其中较大的一个即为所求。

"""Integer to English Words 整数转为英文单词
Convert a non-negative integer to its english words representation. Given input is guaranteed to be less than 231 - 1.

For example,
123 -> "One Hundred Twenty Three"
12345 -> "Twelve Thousand Three Hundred Forty Five"
1234567 -> "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
Hint:

Did you see a pattern in dividing the number into chunk of words? For example, 123 and 123000.
Group the number by thousands (3 digits). You can write a helper function that takes a number less than 1000 and 
convert just that chunk to words.
There are many edge cases. What are some good test cases? Does your code work with input such as 0? Or 1000010? 
(middle chunk is zero and should not be printed out)
"""
num = 1000010
num = 11210010
num = 1
def numberToWords(num):
    if num == 0:
        return 'Zero'
    # first split it
    n = len(str(num))
    s = ''
    split_num = []
    for i in range(n):
        s = str(num)[n-1-i] + s
        if (i+1)%3 == 0:
            split_num = [s] + split_num
            s = ''
    if len(s) > 0:
        split_num = [s] + split_num
    v = ["", "Thousand", "Million", "Billion"]
    res = ''
    for i,unit in zip(split_num, v[:len(split_num)][::-1]):
        res = res + helper(i, unit)
    return res[1:]

def helper(i_str, unit):
    v1 = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
          "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    v2 = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    i = int(i_str)
    if i == 0:
        return ''
    if i < 20:
        out = [v1[i]]
    elif i < 100:
        out = [v2[i//10]] + [v1[i%10]]
    else:
        tmp = i - 100 * (i//100)
        out = [v1[i//100]] + ['Hundred'] + [v2[tmp//10]] + [v1[tmp%10]]
    res1 = " ".join(' '.join(out).split())
    return ' ' + res1 if unit == '' else ' ' + res1 + ' ' + unit

numberToWords(12343)
numberToWords(1000001)

"""[LeetCode] Largest BST Subtree 最大的二分搜索子树
Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), where largest means subtree with 
largest number of nodes in it.

Note:
A subtree must include all of its descendants.
Here's an example:
    10
    / \
   5  15
  / \   \ 
 1   8   7
The Largest BST Subtree in this case is the highlighted one. 
The return value is the subtree's size, which is 3.

Hint:
You can recursively use algorithm similar to 98. Validate Binary Search Tree at each node of the tree, which will 
result in O(nlogn) time complexity.
Follow up:
Can you figure out ways to solve it with O(n) time complexity?
"""
def largestBSTSubtree(root):
    return None

"""Water and Jug Problem 水罐问题
You are given two jugs with capacities x and y litres. There is an infinite amount of water supply available. You need 
to determine whether it is possible to measure exactly z litres using these two jugs.

If z liters of water is measurable, you must have z liters of water contained within one or both buckets by the end.

Operations allowed:

Fill any of the jugs completely with water.
Empty any of the jugs.
Pour water from one jug into another till the other jug is completely full or the first jug itself is empty.
Example 1: (From the famous "Die Hard" example)

Input: x = 3, y = 5, z = 4
Output: True
Example 2:

Input: x = 2, y = 6, z = 5
Output: False
hint: greatest common divisor (gcd)
"""
# 这道问题其实可以转换为有一个很大的容器，我们有两个杯子，容量分别为x和y，问我们通过用两个杯子往里倒水，和往出舀水，问能不能使容器中的水
# 刚好为z升。那么我们可以用一个公式来表达：z = m * x + n * y; 其中m，n为舀水和倒水的次数，正数表示往里舀水，负数表示往外倒水，那么
# 题目中的例子可以写成: 4 = (-2) * 3 + 2 * 5，即3升的水罐往外倒了两次水，5升水罐往里舀了两次水。那么问题就变成了对于任意给定的x,y,z，
# 存不存在m和n使得上面的等式成立。根据裴蜀定理，ax + by = d的解为 d = gcd(x, y)，那么我们只要只要z % d == 0，上面的等式就有解，
# 所以问题就迎刃而解了，我们只要看z是不是x和y的最大公约数的倍数就行了，别忘了还有个限制条件x + y >= z，因为x和y不可能称出比它们之和还多的水
"""
bool canMeasureWater(int x, int y, int z) {
    return z == 0 || (x + y >= z && z % gcd(x, y) == 0);
}
int gcd(int x, int y) {
    return y == 0 ? x : gcd(y, x % y);
}
"""

"""Add Two Numbers II 两个数字相加之二
You are given two linked lists representing two non-negative numbers. The most significant digit comes first and each 
of their nodes contain a single digit. Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.
Example:

Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7
"""
def addTwoNumbers(node1, node2):
    s1 = []
    s2 = []

    while node1:
        s1.append(node1.val)
        node1 = node1.next
    while node2:
        s2.append(node2.val)
        node2 = node2.next
    sum = 0

    curr = listNode(None)
    while len(s1) > 0 or len(s2) > 0 or sum > 0:
        if len(s1) > 0:
            sum = sum + s1[-1]
        if len(s2) > 0:
            sum = sum + s2[-1]
        curr.val = sum%10
        sum = sum//10
        head = ListNode(None)
        head.next = curr
        curr = head

    return curr

"""Minimum Number of Arrows to Burst Balloons 最少数量的箭引爆气球
There are a number of spherical balloons spread in two-dimensional space. For each balloon, provided input is the start 
and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter and hence the 
x-coordinates of start and end of the diameter suffice. Start is always smaller than end. There will be at most 104 balloons.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with xstart and xend 
bursts by an arrow shot at x if xstart ≤ x ≤ xend. There is no limit to the number of arrows that can be shot. 
An arrow once shot keeps travelling up infinitely. The problem is to find the minimum number of arrows that must be shot
 to burst all balloons.
Example:
Input:
[[10,16], [2,8], [1,6], [7,12]]
Output:
2
Explanation:
One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6]) and another arrow at x = 11
 (bursting the other two balloons).
"""
# greedy algo. sort balloons by its end point. First arrow goes to the end of the first ballon; remove
# all balloons shot by this arrow. Then repeat this process.

balls = [[10,16], [2,8], [1,6], [7,12]]

def findMinArrowShots(balls):
    if len(balls) == 0:
        return 0
    balls.sort(key = lambda x: x[1])
    arrow = balls[0][1]
    count = 1
    for i in range(len(balls)):
        if balls[i][0] <= arrow:
            continue
        else:
            count = count + 1
            arrow = balls[i][1]

    return count

findMinArrowShots([[10,16], [2,8], [1,6], [7,12]])


"""4 Keys Keyboard 四键的键盘
Imagine you have a special keyboard with the following keys:
Key 1: (A): Print one 'A' on screen.
Key 2: (Ctrl-A): Select the whole screen.
Key 3: (Ctrl-C): Copy selection to buffer.
Key 4: (Ctrl-V): Print buffer on screen appending it after what has already been printed.

Now, you can only press the keyboard for N times (with the above four keys), find out the maximum numbers of 'A' you can
 print on screen.

Example:
Input: N = 7
Output: 9
Explanation: 
We can at most get 9 A's on screen by pressing following key sequence:
A, A, A, Ctrl A, Ctrl C, Ctrl V, Ctrl V
"""
# Idea: for i, 最多的A数量是maxA(i)，剩下粘贴 (N-i-2+1) * maxA(i)
# use dp to save the previous results
N = 7
def maxA(N):
    dp = list(range(1, N+1))#len = N
    for i in range(N):
        for j in range(1, i-1):
            # when j = 0, never copy and paste
            dp[i] = max(dp[i], dp[j] * (i-j-2+1))

    return dp[N-1]

maxA(7)
maxA(20)
maxA(1)
maxA(3)


"""[LeetCode] Bulb Switcher 灯泡开关
There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On 
the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the nth round, 
you only toggle the last bulb. Find how many bulbs are on after n rounds.

Example:
Given n = 3. 

At first, the three bulbs are [off, off, off].
After first round, the three bulbs are [on, on, on].
After second round, the three bulbs are [on, off, on].
After third round, the three bulbs are [on, off, off]. 

So you should return 1, because there is only one bulb is on.
"""
# 一个完全平方数的因子个数必然为奇数；反之，任何一个自然数若有奇数个因子，这个自然数必为完全平方数
# only bulbs with numbers that have odd factors will stay on


"""Bulb Switcher II 灯泡开关之二
There is a room with n lights which are turned on initially and 4 buttons on the wall. After performing exactly m 
unknown operations towards buttons, you need to return how many different kinds of status of the n lights could be.

Suppose n lights are labeled as number [1, 2, 3 ..., n], function of these 4 buttons are given below:

Flip all the lights.
Flip lights with even numbers.
Flip lights with odd numbers.
Flip lights with (3k + 1) numbers, k = 0, 1, 2, ...

Example 3:

Input: n = 3, m = 1.
Output: 4
Explanation: Status can be: [off, on, off], [on, off, on], [off, off, off], [off, on, on].
"""
# 1. treat it as number of diff combinations. you have four operations A,B,C,D, then at most you have
#    4 + 4choose2  4choose3 + 4choose4 = 1 + 6+ 4 +1=12 types of results. But there are duplicates
# 2. only need to discuss the status of first 6 lights

# below is a BF solution which is not efficient. a "smarter" idea is
"""    int flipLights(int n, int m) {
        if (n == 0 || m == 0) return 1;
        if (n == 1) return 2;
        if (n == 2) return m == 1 ? 3 : 4;
        if (m == 1) return 4;
        return m == 2 ? 7 : 8;
    }
"""

def flipLights(n, m):

    return

"""Remove Comments 移除注释
Given a C++ program, remove comments from it.

Example 1:
Input: 
source = ["/*Test program */", "int main()", "{ ", "  // variable declaration ", "int a, b, c;", "/* This is a test", 
"   multiline  ", "   comment for ", "   testing */", "a = b + c;", "}"]

The line by line code is visualized as below:
/*Test program */
int main()
{ 
  // variable declaration 
int a, b, c;
/* This is a test
   multiline  
   comment for 
   testing */
a = b + c;
}
Output: ["int main()","{ ","  ","int a, b, c;","a = b + c;","}"]

Example 2:
Input: 
source = ["a/*comment", "line", "more_comment*/b"]
Output: ["ab"]
"""
def removeComments(source):
    skip = False
    res = []
    for i,x in enumerate(source):
        if skip:
            # in the range of /*
            n3 = find_s1(x, '*/')
            if n3 < float("Inf"):
                skip = False
                res[-1] = res[-1] + x[(n3+2):]
                # res.append(x[(n3+2):])
            continue
        else:
            n1, n2, n3 = find_s1(x, '//'), find_s1(x, '/*'), find_s1(x, '*/')
            if min(min(n1, n2), n3) == float("Inf"):
                res.append(x)
            elif n1 < n2 and n1 < n3:
                # // first
                res.append(x[:n1])
            elif n2 < n1 and n1 <= n3:
                # /* // */ or /* or /* //
                if n3 < float("Inf"):
                    res.append(x[:n2])
                    res[-1] = res[-1] + x[(n3+2):]
                    # res.append(x[(n3+2):])
                else:
                    res.append(x[:n2])
                    skip = True
                    continue
            elif n2 < n3 and n3 <= n1:
                if n1 < float("Inf"):
                    # /* */ //
                    res.append(x[:n2])
                    res[-1] = res[-1] + x[(n3+2):n1]
                    # res.append(x[(n3+2):n1])
                else:
                    # /* */
                    res.append(x[:n2])
                    res[-1] = res[-1] + x[(n3+2):]
                    # res.append(x[(n3+2):])

    return [i for i in res if i != ""]

def find_s1(s, s0):
    # return the index of //, if no return Inf
    if s0 not in s:
        return float("Inf")
    for i in range(len(s) - len(s0) + 1):
        if s[i:(i+len(s0))] == s0:
            return i


removeComments(source)
removeComments(["a/*comment", "line", "more_comment*/b"])

""" Closest Leaf in a Binary Tree 二叉树中最近的叶结点
Given a binary tree where every node has a unique value, and a target key k, find the value of the nearest leaf node to 
target k in the tree.
Here, nearest to a leaf means the least number of edges travelled on the binary tree to reach any leaf of the tree. 
Also, a node is called a leaf if it has no children.
In the following examples, the input tree is represented in flattened form row by row. The actual root tree given will 
be a TreeNode object.
Input:
root = [1,2,3,4,null,null,null,5,null,6], k = 2
Diagram of binary tree:
             1
            / \
           2   10
          /
         4
        /
       5
      /
     6
Output: 10
Explanation: The leaf node with value 10 (and not the leaf node with value 6) is nearest to the node with value 2.
# Idea: build track back map, for each node, return its parent
"""
from collections import deque
def findClosestLeaf(root, k):
    if root is None:
        return None
    ht = {}
    create_map(root, ht)
    start_node = findk(root, k)
    q = deque()
    q.append(start_node)
    hs = set()
    while len(q) > 0:
        for _ in range(len(q)):
            curr_node = q.popleft()
            # if leave, then return
            if curr_node.left is None and curr_node.right is None:
                return curr_node.val
            # check its parent, left child and right child
            parent_node = ht[curr_node]
            if parent_node not in hs:
                hs.add(parent_node)
                q.append(parent_node)
            if curr_node.left is not None and curr_node.left not in hs:
                hs.add(curr_node.left)
                q.append(curr_node.left)
            if curr_node.right is not None and curr_node.right not in hs:
                hs.add(curr_node.right)
                q.append(curr_node.right)
    return None

# return the node whose value is k
def findk(root, k):
    if root is None:
        return None
    if root.val == k:
        return root
    res_left = findk(root.left, k)
    res_right = findk(root.right, k)
    return res_left if res_left is not None else res_right

# create the track back ht, return its parental node
def create_map(root, ht):
    if root.left is not None:
        ht[root.left] = root
        createTrack(root.left, ht)
    if root.right is not None:
        ht[root.right] = root
        createTrack(root.right, ht)
    return None



"""[LeetCode] Longest Univalue Path 最长相同值路径
Given a binary tree, find the length of the longest path where each node in the path has the same value. This path may 
or may not pass through the root.
Note: The length of path between two nodes is represented by the number of edges between them.
Example 1:
Input:

              5
             / \
            4   5
           / \   \
          1   1   5
Output:
2
"""

def longestUnivaluePath(root):
    if root is None:
        return 0
    res = longestPath(root.left, root.val) + longestPath(root.right, root.val) #+1
    res = max(res, longestUnivaluePath(root.left))
    res = max(res, longestUnivaluePath(root.right))
    return res

# the longest path of val k, starting from root
def longestPath(root, k):
    if root is None or root.val != k:
        return 0
    return 1 + max(longestPath(root.left, k), longestPath(root.right, k))

"""Path Sum IV 二叉树的路径和之四
 

If the depth of a tree is smaller than 5, then this tree can be represented by a list of three-digits integers.

For each integer in this list:

The hundreds digit represents the depth D of this node, 1 <= D <= 4.
The tens digit represents the position P of this node in the level it belongs to, 1 <= P <= 8. The position is the same 
as that in a full binary tree.
The units digit represents the value V of this node, 0 <= V <= 9.
 
Given a list of ascending three-digits integers representing a binary with the depth smaller than 5. You need to return 
the sum of all paths from the root towards the leaves.

Example 1:

Input: [113, 215, 221]
Output: 12
Explanation: 
The tree that the list represents is:
    3
   / \
  5   1
The path sum is (3 + 5) + (3 + 1) = 12.
"""
# idea: figure out the location relationship (2nd number) of parents and kids
nums = [113, 215, 221]
def pathSum(nums):
    m = {}
    for i in nums:
        m[str(i)[:2]] = str(i)[2]
    res = 0
    for i in nums:
        lv = str(i)[0]
        loc = str(i)[1]
        left2 = str(int(lv)+1) + str(2 * int(loc) - 1)
        right2 = str(int(lv)+1) + str(2 * int(loc))
        if left2 in m:
            res = res + int(str(i)[2]) + int(m[left2])
        if right2 in m:
            res = res + int(str(i)[2]) + int(m[right2])
    return res

pathSum(nums)


"""[LeetCode] Equal Tree Partition 划分等价树
Given a binary tree with n nodes, your task is to check if it's possible to partition the tree to two trees which have 
the equal sum of values after removing exactly one edge on the original tree.
Example 1:
Input:     
    5
   / \
  10 10
    /  \
   2   3
Output: True
Explanation: 
    5
   / 
  10
      
Sum: 15
   10
  /  \
 2    3

Sum: 15
 

Example 2:

Input:     
    1
   / \
  2  10
    /  \
   2   20
Output: False
Explanation: You can't split the tree into two trees with equal sum after removing exactly one edge on the tree.
Idea: use a ht to save all the subtree sums, ca
"""

def checkEqualTree(root):
    if root is None:
        return False
    hs = set()
    total_sum = calcsum(root, hs)
    return total_sum % 2 == 0 and int(total_sum / 2) in hs

def savesum(root, hs):
    if root is None:
        return 0
    left_sum = savesum(root.left)
    hs.add(left_sum)
    right_sum = savesum(root.right)
    hs.add(right_sum)
    res = left_sum + right_sum + root.val
    hs.add(res)
    return res


"""[LeetCode] Find Duplicate Subtrees 寻找重复树
Given a binary tree, return all duplicate subtrees. For each kind of duplicate subtrees, 
you only need to return the root node of any oneof them. 
Two trees are duplicate if they have the same structure with same node values.
Example 1: 

        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
The following are two duplicate subtrees:
      2
     /
    4
and
    4
Therefore, you need to return above trees' root in the form of a list.

# Idea: 还有数组(Tree)序列化，并且建立序列化跟其出现次数的映射
class Solution {
public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        vector<TreeNode*> res;
        unordered_map<string, int> m;
        helper(root, m, res);
        return res;
    }
    string helper(TreeNode* node, unordered_map<string, int>& m, vector<TreeNode*>& res) {
        if (!node) return "#";
        string str = to_string(node->val) + "," + helper(node->left, m, res) + "," + helper(node->right, m, res);
        if (m[str] == 1) res.push_back(node);
        ++m[str];
        return str;
    }
};
"""

"""Add One Row to Tree 二叉树中增加一行
Given the root of a binary tree, then value v and depth d, you need to add a row of nodes with value v at the given depth d. The root node is at depth 1.

The adding rule is: given a positive integer depth d, for each NOT null tree nodes N in depth d-1, create two tree nodes
 with value v as N's left subtree root and right subtree root. And N's original left subtree should be the left subtree 
 of the new left subtree root, its original right subtree should be the right subtree of the new right subtree root. If 
 depth d is 1 that means there is no depth d-1 at all, then create a tree node with value v as the new root of the whole 
 original tree, and the original tree is the new root's left subtree.
 
Example 1:

Input: 
A binary tree as following:
       4
     /   \
    2     6
   / \   / 
  3   1 5   

v = 1

d = 2

Output: 
       4
      / \
     1   1
    /     \
   2       6
  / \     / 
 3   1   5   
"""

def addOneRow(root, v, d):
    if d == 1:
        newroot = TreeNode(v)
        newroot.left = root
        return newroot

    q = deque()
    q.append(root)

    while d > 1:
        for _ in range(len(q)):
            curr_node = q.popleft()
            if curr_node.left is not None:
                q.append(curr_node.left)
            if curr_node.right is not None:
                q.append(curr_node.right)
        d = d - 1

    for _ in range(len(q)):
        curr_node = q.popleft()
        curr_l = curr_node.left
        curr_r = curr_node.right
        curr_node.left = TreeNode(v)
        curr_node.right = TreeNode(v)
        curr_node.left.left = curr_l
        curr_node.right.right = curr_r
    return root

""" Minimum Time Difference 最短时间差
Given a list of 24-hour clock time points in "Hour:Minutes" format, find the minimum minutes difference between any 
two time points in the list.
Example 1:
Input: ["23:59","00:00", "12:00"]
Output: 1
"""
timePoints = ["07:56","19:58","19:12","01:59","04:27"] # 46
def findMinDifference(timePoints):
    res = float('Inf')
    timePoints.sort(key = lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    for i in range(1, len(timePoints)):
        res = min(res, timeDiff(timePoints[i-1], timePoints[i]))
    res = min(res, timeDiff(timePoints[0], timePoints[len(timePoints)-1]))
    return int(res)

def timeDiff(t0, t1):
    h1,m1 = [int(i) for i in t0.split(':')]
    h2, m2 = [int(i) for i in t1.split(':')]
    n1 = (h1 - h2) * 60 + m1 - m2 if h1 >= h2 else (24 + h1 - h2) * 60 + m1 - m2
    n2 = (h2 - h1) * 60 + m2 - m1 if h2 >= h1 else (24 + h2 - h1) * 60 + m2 - m1
    res = min(abs(n1), abs(n2))
    return res

findMinDifference(timePoints)

"""Odd Even Linked List 奇偶链表
Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking 
about the node number and not the value in the nodes.
You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example:
Given 1->2->3->4->5->NULL,
return 1->3->5->2->4->NULL.

Note:
The relative order inside both the even and odd groups should remain as it was in the input. 
The first node is considered odd, the second node even and so on ...
"""
def oddEvenList(head):
    if head is None or head.next is None:
        return head
    odd = head
    even = odd.next
    first_even = odd.next
    while even is not None and even.next is not None:
        odd.next = even.next
        even.next = even.next.next
        odd = odd.next
        even = even.next
    odd.next = first_even
    return head


"""Palindrome Linked List 回文链表
Given a singly linked list, determine if it is a palindrome. Follow up: Could you do it in O(n) time and O(1) space?
"""
def isPalindrome(head):
    fast = head
    slow = head
    while fast is not None and fast.next is not None:
        fast = fast.next.next
        slow = slow.next
    #mid = slow
    tail = reverseList(slow)
    curr = tail
    while head != curr:
        if head.val != curr.val:
            reverseList(tail)
            return False
        head = head.next
        curr = curr.next
    reverseList(tail)
    return True

def reverseList(node):
    if node is None:
        return node
    tail = None
    head = node
    while head is not None:
        tmp = head.next
        head.next = tail
        tail = head
        head = tmp
    return tail

# recursive reverse list
# 1->2->3->4->None
# first=1->2<-3<-4=head
# None<-first=1<-2<-3<-4=head
def reverseList2(node):
    if node is None or node.next is None:
        return node
    first = node
    head = reverseList2(node.next)
    first.next.next = first
    first.next = None
    return head

"""Sequence Reconstruction 序列重建
 

Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. The org sequence is a 
permutation of the integers from 1 to n, with 1 ≤ n ≤ 104. Reconstruction means building a shortest common supersequence
 of the sequences in seqs (i.e., a shortest sequence so that all sequences in seqs are subsequences of it). Determine 
 whether there is only one sequence that can be reconstructed from seqs and it is the org sequence.

Example 1:

Input:
org: [1,2,3], seqs: [[1,2],[1,3]]

Output:
false

Example 3:

Input:
org: [1,2,3], seqs: [[1,2],[1,3],[2,3]]

Output:
true
"""
# idea, [1,2,3] the position of 2 is confirmed only when we see [1,2], therefore [1,3] does not mean anything
def sequenceReconstruction(org, seqs):

    return None



"""Insertion Sort List 链表插入排序
Sort a linked list using insertion sort.
"""
def insertionSortList(head):
    new_head = Node(float('-Inf'))
    new_head.next = head
    curr = new_head
    while curr.next is not None:
        if curr.val <= curr.next.val:
            curr = curr.next
            continue
        else:
            next = curr.next
            curr.next = next.next
            pre = new_head
            while next.val > pre.next.val:
                pre = pre.next
            tmp = pre.next
            pre.next = next
            next.next = tmp
    return new_head.next

"""N Queens
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
Given an integer n, return all distinct solutions to the n-queens puzzle.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a 
queen and an empty space respectively.

For example,
There exist two distinct solutions to the 4-queens puzzle:

[
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]
"""
from copy import deepcopy
n = 4
n = 8
all_board = solveNQueens(n)
len(all_board)

def solveNQueens(n):
    board = ['e'*n for i in range(n)] # e means empty, not controlled; . means controlled but no queen
    res = []
    helper(board, n, 0, res)
    return res

# put the queen on the row_index th row, and move on
def helper(board, n, row_index, res):
    if row_index == n:
        res.append(board)
        return None
    for j in range(n):
        #print('col=' + str(j))
        if board[row_index][j] == 'e':
            #print('Find e')
            prev_board = deepcopy(board)
            board[row_index] = board[row_index][:j] + 'Q' + board[row_index][(j+1):]
            update(board, row_index, j)
            helper(board, n, row_index+1, res)
            # very important to recover the board!!
            board = deepcopy(prev_board)
    return None

# update the board according to the new Queen that is placed at [i,j]
def update(board, i, j):
    for irow in range(n):
        if board[irow][j] == 'e':
            board[irow] = board[irow][:j] + '.' + board[irow][(j+1):]
    for icol in range(n):
        if board[i][icol] == 'e':
            board[i] = board[i][:icol] + '.' + board[i][(icol+1):]
    # four diagonal
    cont = True
    step = 1
    while cont:
        cnt = 0
        for (curr_i, curr_j) in [(i-step, j-step), (i-step, j+step), (i+step, j-step), (i+step, j+step)]:
            if 0 <= curr_i < n and 0 <= curr_j < n:
                cnt = 1
                if board[curr_i][curr_j] == 'e':
                    board[curr_i] = board[curr_i][:curr_j] + '.' + board[curr_i][(curr_j+1):]
        step = step + 1
        if cnt == 0:
            cont = False
    return None













