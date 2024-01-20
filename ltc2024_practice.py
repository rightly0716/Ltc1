"""[LeetCode] 215. Kth Largest Element in an Array 数组中第k大的数字
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the 
sorted order, not the kth distinct element.

Example 1:
Input: 
[3,2,1,5,6,4] 
and k = 2
Output: 5

Example 2:
Input: 
[3,2,3,1,2,4,5,5,6] 
and k = 4
Output: 4
Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.
"""

class kthlargest:
    def findKthLargest(self, arr, k):
        n=len(arr)
        left, right = 0, n-1
        while left <= right:
            # find 
            pivot_index = self.partition(left, right, arr)
            if pivot_index == n-k:
                return arr[pivot_index]
            if pivot_index < n-k:
                # dest is on right
                left = pivot_index + 1
            if pivot_index > n-k:
                right = pivot_index - 1
    
    def partition(self, left, right, arr):
        # move all smaller to left
        pivot = arr[left]
        l_index, r_index = left+1, right
        while l_index <= r_index:
            if arr[l_index] > pivot and arr[r_index] < pivot:
                arr[l_index], arr[r_index] = arr[r_index], arr[l_index]
                l_index += 1
                r_index -= 1
            if arr[l_index] <= pivot:
                l_index += 1
            if arr[r_index] >= pivot:
                r_index -= 1

        arr[left], arr[r_index] = arr[r_index], arr[left]
        return r_index

arr=[3,2,3,1,2,4,5,5,6] 
k=4
solution=kthlargest()
solution.findKthLargest(arr, k)


from collections import defaultdict
def getHint(secret, guess):
    bulls, cows = 0, 0
    d1, d2 = defaultdict(lambda: 0), defaultdict(lambda: 0) # keep dictionaries of values which are not bulls
    for n1, n2 in zip(secret, guess):
        if n1 == n2:
            bulls += 1
        else:
            d1[n1] = 1 if n1 not in d1 else d1[n1] + 1
            d2[n2] = 1 if n2 not in d2 else d2[n2] + 1
    for k2, v2 in d2.items(): # go through your guess, determine if each digit is a cow
        v1 = d1[k2] 
        cows += min(v1, v2)
    return f"{bulls}A{cows}B"

getHint("2962","7236") # '0A2B'


"""[LeetCode] 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit !!!
Given an array of integers nums and an integer limit, return the size of the longest non-empty 
subarray such that the absolute difference between any two elements of this subarray is 
less than or equal to limit.

Input: nums = [10,1,2,4,7,2], limit = 5
Output: 4 
Explanation: The subarray [2,4,7,2] is the longest since the maximum absolute diff is |2-7| = 5 <= 5.

Input: nums = [4,2,2,2,4,4,2,2], limit = 0
Output: 3
1. left side end starts at 0, iterate on right end
2. deque pop from right when adding new element
3. deque pop from left when updating left side
"""
from collections import deque
def longestSubarray(nums, limit):
    # max_q is mono decreasing
    max_q, min_q = deque(), deque()
    res = 0
    left = 0
    for right in range(len(nums)):
        # push 
        while max_q and nums[max_q[-1]] < nums[right]:
            max_q.pop()
        while min_q and nums[min_q[-1]] > nums[right]:
            min_q.pop()
        max_q.append(right)
        min_q.append(right)
        while nums[max_q[0]] - nums[min_q[0]] > limit:
            if max_q[0] == left:
                max_q.popleft()
            if min_q[0] == left:
                min_q.popleft()
            left += 1
        res = max(res, right - left + 1)
    return res

longestSubarray(nums = [10,1,2,4,7,2], limit = 5)


"""[LeetCode] 895. Maximum Frequency Stack 最大频率栈 !!!
Implement `FreqStack`, a class which simulates the operation of a stack-like data structure.
FreqStack has two functions:
push(int x), which pushes an integer x onto the stack.
pop(), which removes and returns the most frequent element in the stack.
If there is a tie for most frequent element, the element closest to the top of the stack is removed and returned.

Input:
["FreqStack","push","push","push","push","push","push","pop","pop","pop","pop"],
[[],[5],[7],[5],[7],[4],[5],[],[],[],[]]
Output: [null,null,null,null,null,null,null,5,7,5,4]
Explanation:
After making six .push operations, the stack is [5,7,5,7,4,5] from bottom to top.  Then:

pop() -> returns 5, as 5 is the most frequent.
The stack becomes [5,7,5,7,4].

pop() -> returns 7, as 5 and 7 is the most frequent, but 7 is closest to the top.
The stack becomes [5,7,5,4].

pop() -> returns 5.
The stack becomes [5,7,4].

pop() -> returns 4.
The stack becomes [5,7].
"""
from collections import defaultdict
class FreqStack:
    def __init__(self):
        self.max_Freq = 0
        self.m_f2n = defaultdict(list)  # freq -> [numbers]
        self.m_n2f = defaultdict(lambda: 0)  # number -> freq
        
    def push(self, val: int) -> None:
        # update n2f
        self.m_n2f[val] += 1
        # update f2n
        self.m_f2n[self.m_n2f[val]].append(val)  # later to right
        # update max_Freq
        self.max_Freq = max(self.max_Freq, self.m_n2f[val])

    def pop(self) -> int:
        val = self.m_f2n[self.max_Freq].pop()  # take from right
        if len(self.m_f2n[self.max_Freq]) == 0:
            self.max_Freq = self.max_Freq - 1
        self.m_n2f[val] = self.m_n2f[val] - 1
        return val
        

""" follow up
Given an array with positive and negative numbers, find the maximum average subarray 
which length should be less or equal to given length k.

Input: nums=[1,12,-5,-6,-3,50,3]; k=3
Output: 53

单调队列存最小的cumsum index作为left, iterate on right
"""
from collections import deque
def maxSumSubarray(arr, k):
    cumsums = get_cumsum(arr)  # start with 0, [0, 1, 13, ...]
    print(cumsums)
    res = 0
    min_q = deque()  # mono increasing, min of cumsum
    for right in range(1, len(cumsums)):
        while min_q and cumsums[min_q[-1]] >= cumsums[right]:
            min_q.pop()
        min_q.append(right)
        while right - min_q[0] > k:
            min_q.popleft()
        res = max(res, cumsums[right] - cumsums[min_q[0]])
    return res

def get_cumsum(arr):
    get_cumsum = [0]
    for num in arr:
        get_cumsum.append(get_cumsum[-1] + num)
    return get_cumsum

maxSumSubarray(arr=[1,12,-5,-6,-3,50,3], k=3)


"""[LeetCode] 1485. Clone Binary Tree With Random Pointer !!!
A binary tree is given such that each node contains an additional random pointer which could point 
to any node in the tree or null.

Return a deep copy of the tree.

The tree is represented in the same input/output way as normal binary trees where each node is 
represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (in the input) where the random pointer points to, or null 
if it does not point to any node.
You will be given the tree in class Node and you should return the cloned tree in class NodeCopy. 
NodeCopy class is just a clone of Node class with the same attributes and constructors.

Input: root = [[1,null],null,[4,3],[7,0]]
Output: [[1,null],null,[4,3],[7,0]]
Explanation: The original binary tree is [1,null,4,7].
The random pointer of node one is null, so it is represented as [1, null].
The random pointer of node 4 is node 7, so it is represented as [4, 3] 
where 3 is the index of node 7 in the array representing the tree.
The random pointer of node 7 is node 1, so it is represented as [7, 0] 
where 0 is the index of node 1 in the array representing the tree.
"""
class TreeNode:
    def __init__(self, val=0, left=None, right=None, random=None):
        self.val = val
        self.left = left
        self.right = right
        self.random = random

class Solution_BFS:
    def copyRandomBinaryTree(self, root):
        if root is None:
            return None
        m = dict()
        q = deque()
        q.append(root)
        m[root] = TreeNode(root.val)
        while q:
            curr_node = q.popleft()
            # m[curr_node] = TreeNode(curr_node.val)
            if curr_node.left:
                if curr_node.left not in m:
                    m[curr_node.left] = TreeNode(curr_node.left.val)
                m[curr_node].left = m[curr_node.left]
                q.append(curr_node.left)
            if curr_node.right:
                if curr_node.right not in m:
                    m[curr_node.right] = TreeNode(curr_node.left.val)
                m[curr_node].right = m[curr_node.right]
                q.append(curr_node.right)
            if curr_node.random:
                if curr_node.random not in m:
                    m[curr_node.random] = TreeNode(curr_node.random.val)
                m[curr_node].random = m[curr_node.random]

        return m[root]

"""LeetCode 1644. Lowest Common Ancestor of a Binary Tree II
Similar to 236, but possible that p or q not in the binary tree. 

Given the root of a binary tree, return the lowest common ancestor (LCA) of two given nodes, p and q. 
If either node p or q does not exist in the tree, return null. All values of the nodes in the tree are unique.

According to the definition of LCA on Wikipedia: 
"The lowest common ancestor of two nodes p and q in a binary tree T is the lowest node that 
has both p and q as descendants (where we allow a node to be a descendant of itself)". 
A descendant of a node x is a node y that is on the path from node x to some leaf node.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5. A node can be a descendant of itself according to the definition of LCA.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 10
Output: null
Explanation: Node 10 does not exist in the tree, so return null.

Follow up: what if treenode has a attribute of parental node? - use a hash set to store parents of p (till root),
then from q go to root till the node is in set(). Return the node


Solution 1: ltc236 + first pass check whether both p and q are in tree
Solution 2: use (node, num of nodes) as return, only when node and num = 2 return true
https://www.bilibili.com/video/BV1sf4y1x7Kn/
"""
class Solution:
    def lowestCommonAncestor2(self, root, p, q):
        LCA, num_node_found = self.getLCAAndNum(root, p, q)
        if num_node_found == 2:
            return LCA
        return 
    
    def getLCAAndNum(self, root, p, q):
        # return 1) LCA 2) number of p and q exists
        if root is None:
            return (None, 0)
        left, num_left = self.getLCAAndNum(root.left, p, q)
        right, num_right = self.getLCAAndNum(root.right, p, q)
        if root == p or root == q:
            num = 1 + num_left + num_right
            return (root, num)
        if num_left > 0 and num_right > 0:
            return (root, 2)
        return (left, left_num) if num_left > 0 else (right, right_num)


class Solution:
    def delNodes(self, root, to_delete):
        delete_set = set(to_delete)
        res = []
        self.delNodesHelper(root, None, delete_set, res)
        return res
    
    def delNodesHelper(self, node, parent, delete_set, res, from_left):
        if node is None:
            return 
        if node.val in delete_set:
            if parent:
                if from_left:
                    parent.left = None
                else:
                    parent.right = None
            self.delNodesHelper(node.left, None, delete_set, res, True)
            self.delNodesHelper(node.right, None, delete_set, res, False)
        else:
            if parent is None:
                res.append(node) # new root
                self.delNodesHelper(node.left, node, delete_set, res, True)
                self.delNodesHelper(node.right, node, delete_set, res, False)
        return 


"""[LeetCode] Trim a Binary Search Tree 修剪一棵二叉搜索树
 
Given a binary search tree and the lowest and highest boundaries as L and R, 
trim the tree so that all its elements lies in [L, R] (R >= L). 
You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

Example 2:

Input: 
    3
   / \
  0   4
   \
    2
   /
  1

  L = 1
  R = 3

Output: 
      3
     / 
   2   
  /
 1
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class TrimBST:
    def TrimBST(self, root, L, R):
        if not root:
            return None
        if root.val < L:
            root = self.TrimBST(root.right, L, R)
        elif root.val > R:
            root = self.TrimBST(root.left, L, R)
        else:
            root.left = self.TrimBST(root.left, L, R)
            root.right = self.TrimBST(root.right, L, R)

        return root


"""[LeetCode] 333. Largest BST Subtree 最大的二分搜索子树
Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), 
where largest means subtree with largest number of nodes in it.

Note:
A subtree must include all of its descendants.

Input: [10,5,15,1,8,null,7]
   10 
   / \ 
  5  15 
 / \   \ 
1   8   7

Output: 3
Explanation: The Largest BST Subtree in this case is the highlighted one.
             The return value is the subtree's size, which is 3.

Follow up:
Can you figure out ways to solve it with O(n) time complexity?
"""
class Solution:
    def largestBSTSubtree(self, root):
        size, isBST, mn, mx = self.largestBST_helper(self, root)
        return size
    
    def largestBST_helper(self, node):
        if not node:
            return 0, True, float('-inf'), float('inf')
        left_size, left_is_BST, left_min, left_max = self.largestBST_helper(node.left)
        right_size, right_is_BST, right_min, right_max = self.largestBST_helper(node.right)
        if left_max < node.val < right_min and left_is_BST and right_is_BST:
            return left_size + 1 + right_size, True, left_min, right_max
        else:
            return 0, False, left_min, right_max

from copy import deepcopy
class Solution:
    def solveNQueens(self, n: int):
        self.res = []
        curr_board = ["e"*n]*n
        self.Qhelper(0, curr_board)
        return self.res
    
    def Qhelper(self, irow, board):
        if irow == len(board):
            self.res.append(board)
            return 
        for j in board[irow]:
            if board[irow][j] == 'e':
                new_board = self.place(board, irow, j)
                self.Qhelper(irow+1, newboard)
                # del new_board
        return 

    def update_board(self, i, j, curr_board):
        # put Q at i,j, O(n)
        board = deepcopy(curr_board)  # do not change the input
        board[i][j] = 'Q'
        for (di, dj) in [(-1, -1), (-1, 1), (1, -1), (1,1), (-1,0), (1,0), (0,-1),(0,1)]:
            curr_i, curr_j = i+di, j+dj
            while 0<=curr_i<self.size and 0<=curr_j<self.size:
                if board[curr_i][curr_j]=='e':
                    board[curr_i][curr_j]='.'
                curr_i, curr_j = curr_i+di, curr_j+dj
        
        return board

class WP2:
    def wp2(self, pattern, str1):
        self.p2s = dict()
        self.used = set() # save how many map values, diff keys cannot map to the same value
        return self.wp2_helper(pattern, str1, 0, 0)
    
    def wp2_helper(self, pattern, str1, pi, si):
        if pi == len(pattern) and si == len(str1):
            return True
        if pi == len(pattern) or si == len(str1): 
            return False
        if pattern[pi] in self.p2s:
            if not str1[si:].startswith(self.p2s[pattern[pi]]):
                return False
            return self.wp2_helper
        
        for i in range(si, len(str1)):
            if str1[si:i+1] not in self.used:
                self.used.add(str1[si:i+1])
                self.p2s[pattern[pi]] = str1[si:i+1]
                if self.wp2_helper():
                    return True
                del self.p2s[pattern[pi]]
                self.used.remove(str1[si:i+1])
        return False


"""37. Sudoku Solver !!!
Write a program to solve a Sudoku puzzle by filling the empty cells.
"""

"""79. Word Search !!!
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally 
or vertically neighboring. The same letter cell may not be used more than once.

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
"""
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        nrows, ncols = len(board), len(board[0])
        for i in range(nrows):
            for j in range(ncols):
                visited = [[False for _ in range(ncols)] for _ in range(nrows)]
                res = self.dfs(board, i, j, 0, word, visited)
                if res:
                    return True
                    
        return False
    
    def dfs(self, board, i, j, idx, word, visited):
        if idx == len(word):
            return True
        if board[i][j] != word[idx]:
            return False
        visited[i][j] = True
        for next_i, next_j in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
            if 0<=next_i<len(board) and 0<=next_j<len(board[0]) and not visited[next_i][next_j]:
                if self.dfs(board, next_i, next_j, idx+1, word, visited):
                    return True
        visited[i][j] = False
        return False
        

"""1376. Time Needed to Inform All Employees
A company has n employees with a unique ID for each employee from 0 to n - 1. 
The head of the company is the one with headID.

Each employee has one direct manager given in the manager array where manager[i] 
is the direct manager of the i-th employee, manager[headID] = -1. Also, it is guaranteed that the subordination relationships have a tree structure.

The head of the company wants to inform all the company employees of an urgent 
piece of news. He will inform his direct subordinates, and they will inform their 
subordinates, and so on until all employees know about the urgent news.

The i-th employee needs informTime[i] minutes to inform all of his direct subordinates 
(i.e., After informTime[i] minutes, all his direct subordinates can start spreading the news).

Return the number of minutes needed to inform all the employees about the urgent news.

Input: n = 6, headID = 2, manager = [2,2,-1,2,2,2], informTime = [0,0,1,0,0,0]
Output: 1
Explanation: The head of the company with id = 2 is the direct manager of all the employees in the company and needs 1 minute to inform them all.
The tree structure of the employees in the company is shown.

Solution 1: BFS:
build a graph with key as manager and value as his/her direct report 
start with the headID, use a vsited to store employees informed, use queue to save the current
use a res to update the max distance to headID so far. return res 

Solution 2: DFS
the time takes to inform all employee under manager A is 1 + (the max of time spent for his/her direct report)
"""
from collections import defaultdict
class Solution_BFS:
    def numOfMinutes(self, n: int, headID: int, manager, informTime):
        mgr2ic = defaultdict(list) # manager -> direct report 
        for report_id, manager_id in enumerate(manager):
            mgr2ic[manager_id].append(report_id)

        q = deque()
        q.append((headID, 0))
        res = 0
        while q:
            pid, curr_time = q.popleft()
            if len(mgr2ic[pid]) == 0:
                res = max(res, curr_time)
            for report_id in mgr2ic[pid]:
                q.append((report_id, curr_time+informTime[pid]))
        
        return res
    

"""[LeetCode] Combination Sum III 组合之和之三
Find all possible combinations of k numbers that add up to a number n, given that
 only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Ensure that numbers within the set are sorted in ascending order.

Example 1:
Input: k = 3, n = 7
Output:
[[1,2,4]]

Example 2:
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
"""
class CombSum3:
    def combSum3(self, k, n):
        nums = list(range(1, 10))
        self.res = []
        self.dfs(0, nums, k, n, [])
        return self.res
    
    def dfs(self, start, nums, rem_k, rem_target, curr_out):
        if rem_target == 0 and rem_k == 0:
            self.res.append(curr_out)
        if rem_k == 0 or rem_target < 0:
            return 
        for i in range(start, len(nums)):
            if nums[i] > rem_target:
                continue
            self.dfs(i+1, nums, rem_k-1, rem_target-nums[i], curr_out+[nums[i]])
        return 

sol=CombSum3()
sol.combSum3(3,9)

"""72. Edit Distance
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character

Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

# Solution 1
比较的时候, 要尝试三种操作, 因为谁也不知道当前的操作会对后面产生什么样的影响。
对于当前比较的两个字符 word1[i] 和 word2[j], 若二者相同, 一切好说, 直接跳到下一个位置。
若不相同, 有三种处理方法, 
首先是直接插入一个 word2[j], 那么 word2[j] 位置的字符就跳过了, 接着比较 word1[i] 和 word2[j+1] 即可。
第二个种方法是删除, 即将 word1[i] 字符直接删掉, 接着比较 word1[i+1] 和 word2[j] 即可。
第三种则是将 word1[i] 修改为 word2[j], 接着比较 word1[i+1] 和 word[j+1] 即可。

分析到这里, 就可以直接写出递归的代码, 但是很可惜会 Time Limited Exceed, 所以必须要优化时间复杂度, 
需要去掉大量的重复计算, 这里使用记忆数组 memo 来保存计算过的状态, 从而可以通过 OJ, 
"""
class Solution_DFS:
    def minDistance(self, word1: str, word2: str):
        memo = [[0 for _ in range(len(word2))] for _ in range(len(word1))]
        # memo[i1][i2] steps needed to make word1[i1:] and word2[i2:] the same
        return self.dfs(0, 0, word1, word2, memo)

    def dfs(self, i1, i2, word1, word2, memo):
        if i1 == len(word1):
            # delete the rest of word2
            return len(word2) - i2
        if i2 == len(word2):
            return len(word1) - i1
        if memo[i1][i2] > 0:
            return memo[i1][i2]
        
        if word1[i1] == word2[i2]:
            return self.dfs(i1+1, i2+1, word1, word2, memo)
        
        insertCnt = self.dfs(i1, i2+1, word1, word2,memo)
        deleteCnt = self.dfs(i1+1, i2, word1, word2, memo)
        replaceCnt = self.dfs(i1+1, i2+1, word1, word2, memo)

        res = min(insertCnt, min(deleteCnt, replaceCnt)) + 1
        memo[i1][i2] = res
        return memo[i1][i2]


word1 = "intention"; word2 = "execution"
word1="horse"; word2="ros"
sol = Solution_DFS()
sol.minDistance(word1, word2)

""" Solution 2: DP
dp[i][j] 表示从 word1 的前i个字符转换到 word2 的前j个字符所需要的步骤
先给这个二维数组 dp 的第一行第一列赋值, 这个很简单, 因为第一行和第一列对应的总有一个字符串是空串, 于是转换步骤完全是另一个字符串的长度
  Ø a b c d
Ø 0 1 2 3 4
b 1 1 1 2 3
b 2 2 1 2 3
c 3 3 2 1 2
通过观察可以发现, 当 word1[i] == word2[j] 时, dp[i][j] = dp[i - 1][j - 1], 
其他情况时, dp[i][j] 是其左, 左上, 上的三个值中的最小值加1, 那么可以得到状态转移方程为

dp[i][j] =      /    dp[i - 1][j - 1]    if word1[i - 1] == word2[j - 1]

                \    min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1        else
"""
class Solution_DP:
    def minDistance(self, word1: str, word2: str):
        if word1 == "":
            return len(word2)
        if word2 == "":
            return len(word1)
        dp = [[0 for _ in range(len(word2)+1)] for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1): 
            for j in range(1, len(word2)+1): 
                if word1[i-1] == word2[j - 1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1

        return dp[len(word1)][len(word2)]
    

sol = Solution_DP()
sol.minDistance(word1, word2)
sol.minDistance(word1="a", word2="b")

def dfs(stones, start_pos, prev_k, stone_set):
    if start_pos = stones[-1]:
        return True
    if stones[start_idx] + prev_k + 1 < stones[start_idx+1]:
        return False
    for next_k in []:
        if stones[start_idx] + next_k in stones_set:
            res = dfs(stones, stones[start_idx] + next_k, next_k, stone_set)


"""[LeetCode] 42. Trapping Rain Water 收集雨水
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it is able to trap after raining.

The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

Example:
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
"""

# Solution 2: DP
# left[i]: max height on the left of height[i]
# right[i]:max height on the right of height[i]
# the water that point i can contribute is: min(l, r) - height[i]
# left[i] = max(height[i], left[i-1]) if i>0 else height[i]
# right[i] = max(height[i], right[i+1]) if i<len(height)-1 else height[i]
"""
// Author: Huahua
class Solution {
public:
  int trap(vector<int>& height) {
    const int n = height.size();
    vector<int> l(n);
    vector<int> r(n);
    int ans = 0;
    for (int i = 0; i < n; ++i)
      l[i] = i == 0 ? height[i] : max(l[i - 1], height[i]);
    for (int i = n - 1; i >= 0; --i)
      r[i] = i == n - 1 ? height[i] : max(r[i + 1], height[i]);
    for (int i = 0; i < n; ++i)
      ans += min(l[i], r[i]) - height[i];
    return ans;
  }
};
"""

"""673. Number of Longest Increasing Subsequence
Given an integer array nums, return the number of longest increasing subsequences.
Notice that the sequence has to be strictly increasing.

Example 1:
Input: nums = [1,3,5,4,7]
Output: 2
Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].

Example 2:
Input: nums = [2,2,2,2,2]
Output: 5
Explanation: The length of longest continuous increasing subsequence is 1, 
and there are 5 subsequences' length is 1, so output 5.

** Follow up: what if you need to print all the LIS? 

Idea 1: n^2
将 dp[i] 定义为以 nums[i] 为结尾的递推序列的个数
用 len[i] 表示以 nums[i] 为结尾的递推序列的长度

In the Longest Increasing Subsequence problem, the DP array simply had to store the longest length. 
In this variant, each element in the DP array needs to store two things: 
(1) Length of longest subsequence ending at this index and 
(2) Number of longest subsequences that end at this index. 
I use a two element list for this purpose.
In each loop as we build up the DP array, find the longest length for this index and then 
sum up the numbers at these indices that contribute to this longest length.
https://leetcode.com/problems/number-of-longest-increasing-subsequence/discuss/107320/Python-DP-with-explanation-(Beats-88)
"""


"""639. Decode Ways II
A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters 
using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:
"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

In addition to the mapping above, an encoded message may contain the '*' character, 
which can represent any digit from '1' to '9' ('0' is excluded). 
For example, the encoded message "1*" may represent any of the encoded messages 
"11", "12", "13", "14", "15", "16", "17", "18", or "19". Decoding "1*" is equivalent to decoding 
any of the encoded messages it can represent.

Given a string s consisting of digits and '*' characters, return the number of ways to decode it.

Since the answer may be very large, return it modulo 10**9 + 7.

Example 1:
Input: s = "*"
Output: 9

Example 2:
Input: s = "1*"
Output: 18
Explanation: The encoded message can represent any of the encoded messages "11", "12", "13", "14", "15", "16", "17", "18", or "19".
Each of these encoded messages have 2 ways to be decoded (e.g. "11" can be decoded to "AA" or "K").
Hence, there are a total of 9 * 2 = 18 ways to decode "1*".

Example 3:
Input: s = "2*"
Output: 15
Explanation: The encoded message can represent any of the encoded messages "21", "22", "23", "24", "25", "26", "27", "28", or "29".
"21", "22", "23", "24", "25", and "26" have 2 ways of being decoded, but "27", "28", and "29" only have 1 way.
Hence, there are a total of (6 * 2) + (3 * 1) = 12 + 3 = 15 ways to decode "2*".
"""
class Solution:
    def numDecodings(self, s: str) -> int:
        if len(s) == 0:
            return 1
        if s[0] == '0':
            return 0
        n = len(s)
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 9 if s[0]=='*' else (1 if int(s[0])>0 else 0)
        for i in range(2, n+1):
            dp[i] = dp[i-1] * helper(s[i-1]) + dp[i-2] * helper(s[i-2:i])
            dp[i] = dp[i] % (10**9 + 7)
        return dp[n]

def helper(s):
    if s[0] == '0':
        return 0
    if len(s) == 1:
        return 9 if s=='*' else 1
    if len(s) == 2:
        if s[0] == '*': # "*A" or "**"
            return 15 if s[1]=='*' else (2 if int(s[1]) <= 6 else 1)
        else: # "A*" or "AB"
            if s[1] == '*': 
                return 9 if s[0] == '1' else (6 if s[0] == '2' else 0)
            else:
                return 0 if int(s)>26 else 1
