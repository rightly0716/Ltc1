"""
In Ltc, some questions need hard memory
This file contains such questions
"""


"""
in python, one can use OrderedDict (a dict that has popitem() to remove the last or first key)
"""
from collections import OrderedDict
hashmap = OrderedDict()
hashmap.pop(key)
hashmap.popitem(last=False) # O(1)


""" TreeMap: 红黑树，用logN 来存取，能用logN来读第k大元素
python里面可以用SortedDict来实现
SortedDict: sorted dict inherits from dict to store items and maintains a sorted list of keys.
d.popitem(k) will pop the kth smallest item  - logN
d.bisect_left(val) # return the index of val (from left) if inserted - logN
可以和bisect.bisect(sd.keys(), new_start)配合使用用logN查找插入的index

有时候也可以用SortedList来实现, 比如Sliding Window Median, Count of Smaller Numbers After Self
"""
# SortedDict
from sortedcontainers import SortedDict
sd = SortedDict({'a': 1, 'b': 2, 'c': 3})
sd.pop('c') # 3, O(logn)
sd.popitem(0) # ('a', 1), O(logn)
sd.peekitem(0) # ('a', 1), O(logn)
sd['a'] # return 1, logn
sd['d'] = 4 # O(logn)
# bisect_left: return an index to insert value in the sorted list, left if exists
sd.bisect_left('b') # return 1, logn
# bisect_right=bisect_left except when exists
sd.bisect_right('b') # return 2, logn

# SortedList
lst = SortedList()
lst.add(x) 
lst.remove(x)
lst[k] # kth smallest 

""" Stack and (de)queue in python
"""
stack = []
stack.append('a')
print(stack.pop())   # pop the last, O(1)
print(stack.pop(0))  # pop the first, O(n)

from collections import deque
d = deque()
d.append('j')
d.appendleft('f')
d.pop()  # pop from right
d[-1] # peek from right
d.popleft() 
d[0]  # peek from left
d.extend('jkl')  # add multiple elements at once

""" Heap/Priority Queue
# time complexity of building a heap is O(n)
# https://www.geeksforgeeks.org/time-complexity-of-building-a-heap/
# insert/remove max from max heap is O(logn)
# find max is O(1)
# use case: k largest/smallest, k most frequent
"""
import heapq # python only support min heap
q = [1,2,3,1]
heapq.heapify(q) # make q a heap, O(n)
heapq.heappush(q, x); x=heapq.heappop(q) # min
# the way to customize the heap order is to have each element on the heap to be a tuple, with the first tuple element being one that accepts normal Python comparisons.

heapq.nsmallest(k, q, key) # return the k smallest, klogn, [note: sum(logn) ~ nlogn]
heapq.nlargest(k, q, key=None)  # return k largest , klogn
# example: 
tags = [ ("python", 30), ("ruby", 25), ("c++", 50), ("lisp", 20) ]
heapq.nlargest(2, tags, key=lambda e:e[1]) # Gives [ ("c++", 50), ("python", 30) ]


"""单调栈与单调队列(Monotone Stack/Queue)
栈还是普通栈, 不论单调栈还是单调队列, 单调的意思是保留在栈或者队列中的数字是单调递增或者单调递减的
1. longest subarray with limitation (such as limit diff < k)
2. Largest Rectangle in Histogram: mono increasing
    - add sentinel 0 on right of heights
    - prev_area = heights[prev_i] * (i - 1 - stack[-1])
    - prev_area = heights[prev_i] * i if not stack
"""

""" Dictionary
"""
# In python, define a defaultdict/Counter in this
from collections import defaultdict, Counter
d1 = defaultdict(lambda: 0)
words = ["apple", "banana", "apple", "orange", "mango"]
word_counts = Counter(words)


"""
二分法（Binary Search)
Summary https://www.cnblogs.com/grandyang/p/6854825.html

若 right 初始化为了 nums.size(), 那么就必须用 left < right, 而最后的right 的赋值必须用 right = mid。这种情况一般用于不会中间停止的
但是如果我们 right 初始化为 nums.size() - 1, 那么就必须用 left <= right, 并且right的赋值要写成 right = mid - 1, 不然就会出错。

Find the mtn peak is an exceptional. Use left, right = 0, len-1 and while left < right. 
Because peak cannot be on nums[n-1] and condition is nums[mid] < nums[mid+1] and mid+1 can out of range if right=len

还有一类是排序区间的插入, 删除, 和合并。一般思路是对于待处理的区间, 查找第一个和最后一个与它overlap的区间(可以左闭右开), 然后用for来处理中间的, 
一个例子是715. Range Module.
intervals=[[1,2], [3,4], [5,6]], interval=[2.5,4.5] -> first=0 and last=2 (左闭右开, i.e. intervals[2]是第一个不overlap的)
一般可以考虑
1) 新建一个list存放,O(n) -- 添加区间的一个常用技巧是bool inserted=False, 一般添加完毕后设成True然后后面所有区间直接加进来
2) treemap(SortedDict) 的数据结构, 用bisect_left和bisect_right来查找头尾(logn) -- 见715. Range Module
"""
def bin_search(nums, target):
    l, r = 0, len(nums) 
    while l < r:
        mid = l + (r-l)//2
        if nums[mid] == target:
            return mid # or do nothing
        if nums[mid] < target:
            l = mid + 1 # not soln except for last
        else: # (nums[mid] >= target)
            r = mid # r is soln candidate
    return l # (or r)


""" 双指针: Max Consecutive Ones III
Given a binary array nums and an integer k, return the maximum number of consecutive 
1's in the array if you can flip at most k 0's.

Example 1:
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
"""
class Solution:
    def longestOnes(self, nums, k: int) -> int:
        num_ones = 0
        l = 0
        res = 0
        for r in range(len(nums)):
            num_ones = num_ones + nums[r]
            # move l
            while num_ones + k < r + 1 - l:
                num_ones = num_ones - nums[l]
                l = l + 1
            res = max(r - l + 1, res)
        
        return res


""" BFS: 用deque来存, for loop对每一层来处理

BFS 1. 二叉树层序遍历
For example: Given binary tree {3,9,20,#,#,15,7},
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
recursive的写法是引入一个level parameter
self.getLevelOrder(root, 0, res)
"""
from collections import deque
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class TreeLevelOrder:
    def levelOrder(self, root):
        if not root:
            return 
        q = deque()
        res = []
        q.append(root)
        while q:
            curr_level = []
            for _ in range(len(q)):
                curr_node = q.popleft()
                curr_level.append(curr_node.val)
                if curr_node.left:
                    q.append(curr_node.left)
                if curr_node.right:
                    q.append(curr_node.right)
                if curr_level:
                    res.append(curr_level)
        return res

# recursive的写法
class Solution:
    def levelOrder(self, root: Optional[TreeNode]):
        res = []
        self.getLevelOrder(root, 0, res)
        return res

    def getLevelOrder(self, root, level, res):
        if root is None:
            return None
        if len(res) < level+1:
            # key: create [] for new level
            res.append([])
        res[level].append(root.val)
        self.getLevelOrder(root.left, level+1, res)
        self.getLevelOrder(root.right, level+1, res)
        return None

"""
BFS 2. 无权重的最短路径. 记得用一个visited set去记录避免重复, q里面存的一般是(row,col,distance), 停止条件是(row,col)=target or distance=target.push进q的条件一般是没有visit过和在边界以内. 

如果要返回所有最短路径，那么每一次for之前用一个空的curr_visited来记录当前这一步visit到的点，for loop结束的时候统一用visited.update(curr_visited)的方式加到visited。这样避免漏掉比如从1->2 / 3->4->5. 另外注意有一个flag mark是否最短路线已经到达target,作为停止条件。例子见126. Word Ladder II
"""
from collections import defaultdict, deque
graph = [[3,1],[3,2,4],[3],[4],[]]; start=0; target=4
def getShortestPaths_bfs1(graph, start, target):
    # save path in q, drawback is memory usage
    q = deque()
    parent = defaultdict(list)
    q.append((start, [start]))
    visited = set()
    find_target = False  # mark whether stop
    res = []
    while len(q) > 0:
        curr_visited = set() # for current step
        for _ in range(len(q)):
            curr_node, curr_path = q.popleft()
            if curr_node == target:
                find_target = True
                res.append(curr_path)
                continue
            for next_node in graph[curr_node]:
                if next_node not in visited:
                    curr_visited.add(next_node)
                    q.append((next_node, curr_path + [next_node]))
        visited.update(curr_visited) # add for current step
        if find_target:
            return res
    return res   # no path is found

"""
3. 根据prerequisite关系构建次序列:比如course schedule. 需要构建indegree和pre_hashmap.每次从indegree为0的拿,如果多于1个就不唯一。拿出来后根据pre_hashmap更新indegree，如果为0就push进deque。
q = deque([node for node in org if indegree[node] == 0])
res = []  # reconstruct results
while q:
    if len(q) > 1:
        # have +1 option for the next
        return False
    curr_num = q.popleft()
    res.append(curr_num)
    # update indegree based on pre_m
    for num in pre_m[curr_num]:
        indegree[num] -= 1
        if indegree[num] == 0:
            q.append(num)
    # print(res)
"""


""" DFS
1.Merge trees (BFS也可以做，需要修改其中一棵当成结果，需要两个deque)
2.Deep copy tree
3.图中(有向无向皆可)的符合某种特征(比如最长)的路径以及长度
4.图中两个点的距离（边有weights）:不是最短距离(dijkstra).用visited来记录已经走过的路径，注意trackback
5. Permutation和combination,注意如果permutation里有重复，比如[1,1,2]的所有perm，需要一个visited=[0]*len(nums)来dedup以及如果: nums[i] == nums[i-1] and visited[i-1] == 0, continue (we can use this number only if its previous is used)
6. LIS matrix里的最长上升路径 (用dp记录结果避免重复)
没有特别好的优化 runtime一般都是2^N
"""


""" Union Find 并查集
如果数据不是实时变化, 本类问题可以用BFS或者DFS的方式遍历,
如果数据实时变化, 则并查集每次的时间复杂度可以视为O(1)

- Functions: 一般是先union建图，在find连接
    - init(N): 
        - self.parents = list(range(N))
        - self.weights = [1] * N  
    - Union(a, b): find root of child and parent, link child's root to parent's root, O(a(n))=O(1)  #  inverse Ackermann function
    - Find(x): link x's parent to the root, return, O(a(n))=O(1)

Application 1: Accounts merge
Application 2: number of islands when addLand
    Input: m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]]
    Output: [1,1,2,3]
"""
from collections import defaultdict
class UF:
    def __init__(self, N):
        self.parents = list(range(N))
        # when union, less weights node -> larger
        self.weights = [1] * N  
        
    def find(self, x):
        # link x's parent to the root, return
        if x!= self.parents[x]:
            # not root
            curr_parent = self.parents[x]
            self.parents[x] = self.find(curr_parent)
        return self.parents[x]
    
    def union(self, a, b):
        # find root of child and parent
        # link child's root to parent's root
        a_root = self.find(a)
        b_root = self.find(b)
        if a_root == b_root:
            return None
        if self.weights[a_root] <= self.weights[b_root]:
            self.parents[a_root] = b_root
            self.weights[b_root] += self.weights[a_root]
        else:
            self.parents[b_root] = a_root
            self.weights[a_root] += self.weights[b_root]
        return None


""" Trie 
一般用Trie来做word search problem。

Example: Word Search II
"""
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = dict()
        self.is_word = False
        
class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None: # O(len(word))
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_word = True

    # Returns if the word is in the trie.
    def search(self, word: str) -> bool:
        current = self.root
        for letter in word:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return current.is_word
    
    # Returns if there is any word in the trie that starts with the given prefix.
    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for letter in prefix:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return True


""" 前缀和(Prefix Sum), cumsum
cumsum[i] = sum(nums[:i+1])
cumsum[j] - cumsum[i] = sum(nums[i+1:j+1])
cumsum[j-1] - cumsum[i-1] = sum(nums[i:j])

1. Maximum Subarray O(n) 
nums = [-2,1,-3,4,-1,2,1,-5,4], Output: 6
hint: reset when cumsum<0

2. expand to 2D, range sum query
hint: 
self.dp = [[0 for _ in range(len(matrix[0])+1)] for _ in range(len(matrix)+1)]
self.dp[i][j] = self.dp[i-1][j] + self.dp[i][j-1] - self.dp[i-1][j-1] + matrix[i-1][j-1]
"""

"""
线段树 (Segment Tree or binary index tree): 一般用于处理上述Prefix sum但array会更新的情况. 

FenwickTree/binary index tree: 
- Each element whose index i is a power of 2 contains the sum of the first i elements. 

- Functions
    - init: self.cumsum # cumsum of numbers in the same branch instead of not nums[:i] https://en.wikipedia.org/wiki/Fenwick_tree
    - update(i, delta): add delta to nums[i-1]  # logn
    - query(i): sum(nums[:i])   # logn
- runtime: Init nlogn, query logn, update logn


Application 1: (307. Range Sum Query - Mutable)
1. Update the value of an element in nums.
2. Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
"""




""" DP
Print all Longest Increasing Subsequence: patience sort ???

Sometimes greedy can work: Jump Game II


"""


""" 01 Knapsack Problem 
一共有N件物品, 第i(i从1开始)件物品的重量为weights[i], 价值为values[i]。在总重量不超过背包承载上限W的情况下, 能够装入背包的最大价值是多少？

动态规划的核心思想避免重复计算在01背包问题中体现得淋漓尽致。第i件物品装入或者不装入而获得的最大价值
完全可以由前面i-1件物品的最大价值决定, 暴力枚举忽略了这个事实。
""" 
def Knapsack01(values, weights, W):
    # dp[i][j]: max value by taking from first i items, with weight at most j
    n = len(values)
    dp = [[0 for _ in range(W+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, W+1):
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]] + values[i-1]) if \
                j-weights[i-1]>=0 else dp[i-1][j]
    return dp[n][W]

def Knapsack01(values, weights, W):
    # dp[i]: max value with weight at most j
    n = len(values)
    dp = [0 for _ in range(W+1)]
    for i in range(1, n+1):
        for j in range(W+1)[::-1]:
            if j-weights[i-1] < 0:
                continue
            dp[j] = max(dp[j], dp[j-weights[i-1]] + values[i-1])
    return dp[W]


"""Unbounded Knapsack problem
完全背包(unbounded knapsack problem)与01背包不同就是每种物品可以有无限多个：一共有N种物品, 每种物品有无限多个, 
第i(i从1开始)种物品的重量为weights[i], 价值为v[i]。在总重量不超过背包承载上限W的情况下, 能够装入背包的最大价值是多少？

dp[i][j]表示将前i种物品装进限重为j的背包可以获得的最大价值, 0<=i<=N, 0<=j<=W
这个状态转移方程与01背包问题唯一不同就是max第二项不是dp[i-1]而是dp[i]。
i.e., 
dp[i][j] = max(dp[i-1][j], dp[i][j-weights[i-1]] + values[i-1])

"""

#
# Start of practices 
#
"""写一个quicksort来找arr的第k大的数字
1. partition 
- 函数 while是<=, 先写swap的逻辑(不然可能在不满足left<=right的条件下错误swap)
- 非swap条件都包括=
2. 主函数while可以是true,想清楚什么时候left=pivot_index+1

[LeetCode] 215. Kth Largest Element in an Array 数组中第k大的数字

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

arr=[3,2,3,1,2,4,5,5,6]; k=4
solution=kthlargest()
solution.findKthLargest(arr, k)


""" [LeetCode] 142. Linked List Cycle II 单链表中的环之二
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

To represent a cycle in the given linked list, we use an integer pos which represents the position 
(0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Note: Do not modify the linked list.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.

这个求单链表中的环的起始点是之前那个判断单链表中是否有环的延伸, 可参之前那道 Linked List Cycle。这里还是要设快慢指针, 
不过这次要记录两个指针相遇的位置, 当两个指针相遇了后, 让其中一个指针从链表头开始, 此时再相遇的位置就是链表中环的起始位置
因为快指针每次走2, 慢指针每次走1, 快指针走的距离是慢指针的两倍。而快指针又比慢指针多走了一圈。所以 head到环的起点+环的起点到他们相遇的点的距离 与 环一圈的距离相等。现在重新开始, head 运行到环起点 和 相遇点到环起点的距离也是相等的, 相当于他们同时减掉了 环的起点到他们相遇的点的距离。
"""



"""[LeetCode] 772. Basic Calculator III 基本计算器之三 !!!
Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .
The expression string contains only non-negative integers, +, -, *, / operators , open ( and closing parentheses ) and empty spaces . The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-2147483648, 2147483647].

Some examples:
"1 + 1" = 2
" 6-4 / 2 " = 4
"2*(5+5*2)/3+(6/2+8)" = 21
"(2+6* 3+5- (3*14/7+2)*5)+3"=-12

Note: Do not use the eval built-in library function.
"""
def calculator3(s):
    m = dict()  # map all ( and ) pair location
    stack = []  # for parenthesis location
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        if s[i] == ')':
            m[stack.pop()] = i

    op, curr_num = '+', ''  # previous operator
    n = len(s)
    stack = []  # can use prev_num and curr_num to save ram
    sign = 1
    i = 0
    while i < n: # cannot use FOR because i needs to update in loop at '('
        if s[i].isdigit(): # 0-9
            curr_num = curr_num + s[i]
        if s[i] == '(':
            # treat the part between ( and ) as a number (curr_num)
            j = m[i]
            sub_string = s[(i+1):j]
            curr_num = calculator3(sub_string)
            i = j  # set i at the location of )
        if s[i] in ['+', '-', '*', '/'] or i == n-1:
            if i==0 or (i<n-1 and not s[i-1].isdigit() and s[i-1] not in '()'):
                # sign, not a op
                if s[i] == '-':
                    sign = sign * -1
            else:
                # if s[i] is a operator, not a sign
                if op == '+':
                    stack.append(int(curr_num) * sign)
                if op == '-':
                    stack.append(int(curr_num) * -1 * sign)
                if op =='*':
                    previous_num = stack.pop()
                    stack.append(previous_num * int(curr_num) * sign)
                if op == '/':
                    previous_num = stack.pop()
                    stack.append(int(previous_num / int(curr_num)) * sign)
                sign = 1  # reset sign!
                op = s[i]
                curr_num = ''
        i = i + 1

    return sum(stack)

calculator3("1-(-2)*5")
calculator3("2+6*3+5-(3*14/7+2)*5+3")==eval("2+6*3+5-(3*14/7+2)*5+3")
calculator3("-4*(-1-2)")

class node:
    def __init__(self, val=None, prev=None, next=None):
        self.val = val
        self.next = None
        self.prev = None
    
class LinkedList:
    def __init__(self):
        self.head = self.tail = Node()
        self.head.next, self.tail.prev = self.tail, self.head
    
    def remove(self, node):
        node.next.prev, node.prev.next = node.prev, node.next
    
    def appendleft(self, node):
        old_first_node = self.head.next
        self.head.next, node.prev = node, self.head
        old_first_node.prev, node.next = node, old_first_node

    def pop(self):



class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.llist = LinkedList()
        self.node2val = dict()
    
    def get(self, key):
        curr_node = node(key)
        if curr_node in self.node2val:
            self.llist.remove(curr_node)
            self.llist.appendleft(curr_node)
            return self.node2val[curr_node]
        return -1

    def put(self, key, value):
        curr_node = node(key)
        if curr_node in self.node2val:
            self.llist.remove(curr_node)
            self.llist.appendleft(curr_node)
        else:
            self.llist.appendleft(curr_node)
            if len(self.node2val) > capacity:
                key_rm = self.llist.pop()
                del self.node2val[node(key_rm)]
        self.node2val[curr_node] = value


""" follow up
Given an array with positive and negative numbers, find the maximum average subarray 
which length should be less or equal to given length k.

单调队列存[0,right]中最小的cumsum右端的index(left),那只要cumsums[right]-cumsums[q[0]]就好,只是要随时保证right-q[0]<=k

Input: nums=[1,12,-5,-6,50,3]; k=3
Output: 53
"""
class maxSumSubarray:
    def maxSumSubarray(self, arr, k):
        minCumsumq = deque()  # minCumsumq[0] has min cumsum index so far
        cumsums = self.cumsum(arr)
        res = 0
        for r in range(len(cumsums)):
            # minCumsumq: mono increasing, with [0] smallest cumsum index in [0,r]
            while len(minCumsumq) > 0 and cumsums[r] <= cumsums[minCumsumq[-1]]:
                minCumsumq.pop()
            minCumsumq.append(r)
            if r - minCumsumq[0] > k:
                minCumsumq.popleft()
            res = max(res, cumsums[r] - cumsums[minCumsumq[0]])
        return res
    
    def cumsum(self, arr):
        cumsumArr = [0]
        for num in arr:
            cumsumArr.append(cumsumArr[-1] + num)  # O(1) cost
        return cumsumArr

sol = maxSumSubarray()
sol.maxSumSubarray( [3,-2,12,-6,3], 3) 


"""[LeetCode] 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit !!!
Given an array of integers nums and an integer limit, return the size of the longest non-empty 
subarray such that the absolute difference between any two elements of this subarray is 
less than or equal to limit.

Input: nums = [10,1,2,4,7,2], limit = 5
Output: 4 
Explanation: The subarray [2,4,7,2] is the longest since the maximum absolute diff is |2-7| = 5 <= 5.

Input: nums = [4,2,2,2,4,4,2,2], limit = 0
Output: 3

Solution
sliding window + 单调queue。注意不是单调栈, 而是单调queue。

我们维护一个单调递减queue maxQueue和一个单调递增queue minQueue, 里面存的都是下标。
maxQueue的首元素是当前遍历到的最大元素的下标, minQueue的首元素是当前遍历到的最小元素的下标。
注意存元素也可以, 但是存下标的好处是如果有重复元素, 对下标是没有影响的。

同时我们需要两个指针start和end。一开始end往后走, 当发现

maxQueue不为空且maxQueue的最后一个元素小于当前元素nums[end]了, 则不断往外poll元素, 直到整个maxQueue变回降序
minQueue不为空且minQueue的最后一个元素大于当前元素nums[end]了, 则不断往外poll元素, 直到整个minQueue变回升序
此时再判断, 如果两个queue都不为空但是两个queue里的最后一个元素（一个是最大值, 一个是最小值）的差值大于limit了, 
则开始左移start指针, 左移的同时, 如果两个queue里有任何元素的下标<= start, 则往外poll, 因为不需要了。
这里也是存下标的另一个好处, 因为下标一定是有序被放进两个queue的, 所以如果大于limit了, 
你是需要从最一开始的start指针那里开始检查的。

时间O(n)

空间O(n)
"""
# Solution 2: two deques O(n), to track max and min of sliding window
# while diff>limit, move left till diff<=limit, pop from queue if q[0]==nums[left]
class Solution:
    def longestSubarray(self, nums, limit):
        max_q, min_q = deque(), deque()  # mono q
        # max_q mono decrease and min_q mono increase
        # so that max_q[0] is max and min_q[0] is min in the window
        l = 0 # left side of window
        res = 0
        for r in range(len(nums)):
            while len(max_q) > 0 and nums[max_q[-1]] < nums[r]:
                max_q.pop()  # popright
            while len(min_q) > 0 and nums[min_q[-1]] > nums[r]:
                min_q.pop()
            max_q.append(r)
            min_q.append(r)
            while nums[max_q[0]] - nums[min_q[0]] > limit:
                # remove very left elem
                if nums[max_q[0]] == nums[l]:
                    max_q.popleft()
                if nums[min_q[0]] == nums[l]:
                    min_q.popleft()
                l += 1  # can't be min(min_q[0], max_q[0]), [1,3,2,4], limit=2
            res = max(res, r-l+1)
        return res


"""[LeetCode] 424. Longest Repeating Character Replacement !!!
Given a string that consists of only uppercase English letters, you 
can replace any letter in the string with another letter at most k times. 
Find the length of a longest substring containing all repeating letters 
you can get after performing the above operations.

Note:
Both the string's length and k will not exceed 10^4.

Example 1:
Input:
s = "ABAB", k = 2
Output:
4
Explanation:
Replace the two 'A's with two 'B's or vice versa.

Example 2:
Input:
s = "AABABBA", k = 1
Output:
4
"""
from collections import Counter, defaultdict, deque
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # two pters
        # condition: if r - l + 1 - maxCnt <= k then okay else move left pters
        m = defaultdict(lambda x:0)
        l = 0
        maxCnt = 0  # maxCnt of letter in s[l:r+1]
        res = 0
        for r in range(len(s)):
            m[s[r]] = m[s[r]] + 1
            maxCnt = max(maxCnt, m[s[r]])  # can be costy
            while r - l + 1 - maxCnt > k:
                # will not make r-l+1 as output
                m[s[l]] = m[s[l]] - 1
                l = l + 1
                maxCnt = max(m.values())  # O(26n), can be removed, but why?
            res = max(res, r-l+1)
        
        return res

class Solution2:
    def characterReplacement(self, s: str, k: int) -> int:
        n = len(s)
        if k + 1 >= n: # cannot > n
            return n
        res = 0
        d_idx = defaultdict(deque)  # idx 
        d = defaultdict(lambda:0)  # freq
        for r in range(len(s)):
            curr_char = s[r]
            d_idx[curr_char].append(r)
            d[curr_char] += 1
            while r - d_idx[curr_char][0] + 1 > k + d[curr_char]:
                d_idx[curr_char].popleft()
                d[curr_char] -= 1
            res = max(res,k + d[curr_char])
            if res >= n:
                return n
            # print("{} / {}".format(d_idx[curr_char], r))
        return res


"""Serialize and Deserialize Binary Tree 二叉树的序列化和去序列化
Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your
serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a
 string or list and this can be deserialized to the original tree structure.

For example, you may serialize the following tree
    1
   / \
  2   3
     / \
    4   5
as [1,2,3,null,null,4,5,null,null,null,null], just the same as how LeetCode OJ serializes a binary tree. You do not necessarily need to
follow this format, so please be creative and come up with different approaches yourself.
"""
class node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


# DFS
def serialize_dfs(root):
    res = deque()
    serialize_helper(root, res)
    return res

def serialize_helper(root, res):
    res.append(root.val)
    if res is not None:
        serialize_helper(root.left, res)
        serialize_helper(root.right, res)
    return 

def deserialize_dfs(res):
    curr_val = res.popleft()
    root = node(val=curr_val)
    if root.val:
        root.left = deserialize_dfs(res)
        root.right = deserialize_dfs(res)
    return root

# BFS
def serialize(root):
    res = []
    q = deque()
    q.append(root)
    while q:
        curr_node = q.popleft()
        if curr_node is None:
            res.append(None)
        else:
            res.append(curr_node.val)
            q.append(curr_node.left)
            q.append(curr_node.right)
    return res

def deserialize(res):
    if res is None:
        return None
    root = node(res[0])
    q = deque()
    q.append(root)
    curr_idx = 1
    while q:
        for _ in range(len(q)):
            curr_node = q.popleft()
            curr_node.left = node(res[curr_idx])
            curr_node.right = node(res[curr_idx+1])
            if curr_node.left:
                q.append(curr_node.left)
            if curr_node.right:
                q.append(curr_node.right)
            curr_idx += 2

    return root
            

""" Clone Graph
Given a reference of a node in a connected undirected graph.
Return a deep copy (clone) of the graph.
Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
"""
class Solution_DFS:
    def cloneGraph(self, node: 'Node') -> 'Node':
        n2n = dict()
        self.cloneGraphDict(node, n2n)
        return n2n[node]

    def cloneGraphDict(self, node, n2n):
        if node in n2n:
            return n2n[node]
        newnode = Node(val=node.val)
        n2n[node] = newnode
        for old_nb in node.neighbor:
            new_nb = cloneGraphDict(old_nb, n2n)
            newnode.neighbor.append(new_nb)
        return newnode

"""[LeetCode] Diameter of Binary Tree 二叉树的直径 !!!
Given a binary tree, you need to compute the length of the diameter of the tree. 
The diameter of a binary tree is the length of the longestpath between any two nodes 
in a tree. This path may or may not pass through the root.

Example:
Given a binary tree 

          1
         / \
        2   3
       / \     
      4   5    

Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

Follow up: what if N ary tree instead of binary?  https://www.cnblogs.com/cnoodle/p/14349421.html

根结点1的左右两个子树的深度之和呢。那么我们只要对每一个结点求出其左右子树深度之和, 
这个值作为一个候选值, 然后再对左右子结点分别调用求直径对递归函数, 这三个值相互比较, 
取最大的值更新结果res, 因为直径不一定会经过根结点, 所以才要对左右子结点再分别算一次。

为了减少重复计算, 我们用哈希表建立每个结点和其深度之间的映射, 这样某个结点的深度之前计算过了, 就不用再次计算了
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root):
        if root is None:
            return 0
        self.res = 0
        root_height = self.getHeight(root)
        return self.res
    
    def getHeight(self, node):
        if node is None:
            return 0
        left_height = self.getHeight(node.left)
        right_height = self.getHeight(node.right)
        self.res = max(self.res, left_height + right_height)
        return 1 + max(left_height, right_height)


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
            if curr_node.left:
                if curr_node.left not in m:
                    # chance that curr_node.left exists due to random ptr
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

"""124. Binary Tree Maximum Path Sum
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in 
the sequence has an edge connecting them. A node can only appear in the sequence at 
most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.
Example 1:

Input: [1,2,3]
       1
      / \
     2   3
Output: 6

Example 2:
Input: [-10,9,20,null,null,15,7]
   -10
   / \
  9  20
    /  \
   15   7
Output: 42
"""
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = float('-Inf')
        if root is None:
            return 0
        self.maxPathOnOneSide(root)
        # res = max(node.val + max(maxPathOnOneSide(node.left), 0) + max(maxPathOnOneSide(node.right), 0)) for all node
        return self.res
    
    def maxPathOnOneSide(self, node):
        # 返回值的定义是以当前结点为终点的 path 之和
        if node is None:
            return 0
        # always first calculate the child before node
        left_max = max(0, self.maxPathOnOneSide(node.left))
        right_max = max(0, self.maxPathOnOneSide(node.right))
        # if child is larger, it will not update res
        self.res = max(self.res, left_max + right_max + node.val)
        return node.val + max(left_max, right_max)


"""中序遍历的非递归写法: use stack first push left till end, then right (repeat)
       5
      / \
     3   6
    / \
   2   4
  /
 1
"""
def kthSmallest(root, k):
    res = []
    stack = []
    curr_node = root
    while curr_node or stack:
        while curr_node:
            stack.append(p)
            curr_node = curr_node.left
        curr_node = stack.pop()
        res.append(curr_node.val)
        curr_node = curr_node.right
    return res[k-1]


"""[LeetCode] 126. Word Ladder II 词语阶梯之二
Given two words (beginWord and endWord), and a dictionary's word list, 
find all shortest transformation sequence(s) from beginWord to endWord, 
such that:

Only one letter can be changed at a time
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return an empty list if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
Example 2:

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: []

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.
"""


"""698. Partition to K Equal Sum Subsets 分割K个等和的子集 
Given an array of integers nums and a positive integer k, find whether it's 
possible to divide this array into k non-empty subsets whose sums are all equal.

Example 1:
Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
Output: True
Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.

Note:
•1 <= k <= len(nums) <= 16.
•0 < nums[i] < 10000.

Idea: it is like combination, but record all that sum up to target
"""
class Solution:
    def canPartitionKSubsets(self, nums, k: int):
        self.raw_target = sum(nums) / k
        if self.raw_target != int(self.raw_target):
            return False
        nums.sort(reverse=True)
        visited = ['0' for _ in range(len(nums))]
        memo = dict()
        def dfs(k, rem_target, start_idx, memo):
            if k == 1: # if k-1 targets, then the last one must be sum to target(target=sums/k)
                return True
            visited_str = ''.join(visited)
            if visited_str in memo:
                return memo[visited_str]
            for i in range(start_idx, len(nums)):
                if visited[i] == '1':
                    continue
                if nums[i] == rem_target:
                    visited[i] = '1'
                    res = dfs(k-1, self.raw_target, 0, memo)
                    if res:
                        memo[''.join(visited)] = True
                        return True
                    visited[i] = '0'
                elif nums[i] < rem_target:
                    visited[i] = '1'
                    res = dfs(k, rem_target - nums[i], i+1, memo)
                    if res:
                        memo[''.join(visited)] = True
                        return True
                    visited[i] = '0'
                else:
                    continue
            memo[''.join(visited)] = False
            return False
        return dfs(k, self.raw_target, 0, memo)


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

"""218. The Skyline Problem
https://leetcode.com/problems/the-skyline-problem/description/

Pseudo code:
events = {{x: left, h: heights, type:enter}, {x:right, h:heights, type:leaving}}
events.sorted(by=(x, type, h*type))
ds = DS()  # provide 1) max, 2) remove a given element
for e in events:
    if e.type == enter:
        if e.h > ds.max():
            res.append([e.x, e.h])
        ds.add(e.h)
    if e.type == leaving:
        ds.remove(e.h)
        if e.h > ds.max():
            # current e.h is the max before removing
            res.append([x, ds.max()])
"""
# O(nlogn) time. O(n) space
# ds: max heap, use a active set to pseudo "remove"



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

Idea 2: 
"""
# ???


"""45 Jump Game II
Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Your goal is to reach the last index in the minimum number of jumps.
You can assume that you can always reach the last index.

Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [2,3,0,1,4]
Output: 2
"""
# solution 1: dp[i]: denotes minimum jumps required from current index to reach till the end.
# initial: dp = [inf]*n, dp[0]=1
# iteration: dp[i] = min(dp[j] + 1) if nums[i] >= i-j for j in range(i,n)
# O(n^2)

# solution 2: Gready: O(n)
# dp[i] represent furthest reachable distance at ith jump
# We can iterate over all indices maintaining the furthest reachable position from current index - maxReachable 
#  and currently furthest reached position - lastJumpedPos. 
# Everytime we will try to update lastJumpedPos to furthest possible reachable index - maxReachable.
class Solution:
    def jump(self, nums) -> int:
        current_end = 0  # maximum reachable for current
        res = 0
        next_end = 0  # maximum reachable for next 
        for i in range(len(nums)): # may only need range(len(nums)-1)
            if current_end >= len(nums) - 1:
                return res
            next_end = max(next_end, nums[i]+i)
            if i == current_end:
                res += 1
                current_end = next_end
                if next_end <= i:
                    return -1
        
        return res


""" [LeetCode] 518. Coin Change 2 硬币找零之二
You are given an integer array coins representing coins of different denominations and 
an integer amount representing a total amount of money.
Return the number of combinations that make up that amount. If that amount of money cannot 
be made up by any combination of the coins, return 0.
You may assume that you have an infinite number of each kind of coin.
The answer is guaranteed to fit into a signed 32-bit integer.

Example 1:
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

Example 2:
Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.

Example 3:
Input: amount = 10, coins = [10]
Output: 1

Solution: 
makeChange(11) = makeChange(11 using 0-5) + makeChange(11 using 1-5) + makeChange(11 using 2-5) + ...

Each child will continue using 
makeChange(11 using 1-5) = makeChange(6 using 0-2) + makeChange(6 using 1-2) + ... 

Continue will be
makeChange(6 using 0-2) = makeChange(6-0 using 1's) = 1 (because 6%1==0)
makeChange(6 using 1-2) = makeChange(6-2 using 1's) = 1
"""
# 用到最后一个硬币时, 判断当前还剩的钱数是否能整除这个硬币, 不能的话就返回0, 否则返回1
# memo: key is (remain_amount, curr_coin_idx)
class Solution:
    def change(self, amount: int, coins) -> int:
        memo = dict()
        sorted_coins = sorted(coins, reverse=True)
        return self.dfs(sorted_coins, amount, memo, 0)

    def dfs(self, coins, rem_amount, memo, coin_idx):
        if coin_idx == len(coins) - 1:
            # last coin
            # rem=0 will also return 1
            return 1 if rem_amount % coins[coin_idx] == 0 else 0
        if (rem_amount, coin_idx) in memo:
            return memo[(rem_amount, coin_idx)]
        res = 0
        for num_curr_coin in range(1+rem_amount//coins[coin_idx]):
            next_rem_amount = rem_amount - num_curr_coin*coins[coin_idx]
            res += self.dfs(coins, next_rem_amount, memo, coin_idx+1)
        memo[(rem_amount, coin_idx)] = res
        # print(memo)
        return res

sol=Solution()
sol.change(5, [1, 2, 5])
sol.change(10, [1, 2, 5])

# DFS+cache trick
class Solution:
    def change(self, amount: int, coins) -> int:
        self.coins = sorted(coins)
        self.dp = [[-1]*(amount+1) for _ in range(len(coins)+1)]
        return self.dfs(amount, 0)
    
    def dfs(self, rem_amount, start_idx):
        if self.dp[start_idx][rem_amount] > 0:
            return self.dp[start_idx][rem_amount]
        if rem_amount == 0:
            return 1
        if start_idx == len(self.coins):
            return 0
        if start_idx == len(self.coins) - 1:
            # last coin
            # rem=0 will also return 1
            return 1 if rem_amount % self.coins[start_idx] == 0 else 0
        res = 0
        for i in range(start_idx, len(self.coins)):
            if rem_amount - self.coins[i] >= 0:
                res += self.dfs(rem_amount-self.coins[i], i)
            else:
                break
        self.dp[start_idx][rem_amount] = res
        return res

# DP: Unbounded Knapsack
# dp[i][j]: number of comb to achieve j amount, if only using first i kinds of coins
class Solution:
    def change(self, amount: int, coins) -> int:
        n = len(coins)
        dp = [[0 for _ in range(amount+1)] for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            for j in range(amount+1):
                if j == 0: # note if i=0 but j>0, should be 0
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j] # first include all sols that only use first i-1 kinds
                    if j >= coins[i-1]:
                        # add coins[i-1]: note that not i-1 due to unbounded! 
                        # we can use multiple coins[i] till j-coins[i]<0
                        dp[i][j] += dp[i][j-coins[i-1]]
        return dp[n][amount]

# may reduce space complexity by using 1-d array
"""
def change(self, amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for i in coins:
        for j in range(1, amount + 1):
            if j >= i:
                dp[j] += dp[j - i]
    return dp[amount]
"""

