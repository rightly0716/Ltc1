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

