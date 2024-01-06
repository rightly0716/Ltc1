class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head
        mid = self.breakList(head)
        sorted_head = self.sortList(head)
        sorted_mid = self.sortList(mid)
        new_head = self.mergeSortedList(sorted_head, sorted_mid)
        return new_head
    
    def breakList(self, node):
        fast = slow = ListNode(next=node)
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        mid = slow.next
        slow.next = None
        return mid
    
    def mergeSortedList(self, node1, node2):
        if not node1:
            return node2
        if not node2:
            return node1
        prev = head = ListNode()
        while node1 and node2:
            if node1.val < node2.val:
                head.next = node1
                node1 = node1.next
            else:
                head.next = node2
                node2 = node2.next
            head = head.next
        if not node1:
            head.next = node2
        else:
            head.next = node1
        return prev.next


class BrowserHistory:
    def __init__(self, homepage: str):
        self.pages = [homepage]
        self.index = 0
        self.size = 1
    
    def visit(self, url: str) -> None:
        # pop out forward 
        num_pop = len(self.pages) - self.index - 1
        for _ in range(num_pop):
            self.pages.pop()
        self.pages.append(url)
        self.index += 1
        self.size += 1
    
    def back(self, steps: int) -> str:
    
    def forward(self, steps: int) -> str:



class LRUCache:
    def __init__(self, capacity):


from collections import deque
class maxSumSubarray:
    def maxSumSubarray(self, arr, k):
        minCumsumq = deque()  # mono increasing
        cumsums = self.cumsum(arr)
        res = float('-Inf')
        for r in range(len(cumsums)):
            # update mono q
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
            cumsumArr.append(cumsumArr[-1] + num)
        return cumsumArr


arr = [2, 1, -2, 3, 2]
k = 2
sol = maxSumSubarray()
sol.maxSumSubarray( [2, -1, 2, -2, 1, -1], 3)
sol.maxSumSubarray([0, 2, 3, 1, 4, 6], 2)

from collections import defaultdict
class Node:
    def __init__(self, val=0, nb=[]):
        self.val = val
        self.nb = nb

class cloneGraph:
    def cloneBFS(self, node):
        old2new = defaultdict(Node)
        old2new[node] = Node(val=node.val)
        q = deque()
        q.append(node)
        while q:
            curr_node = q.popleft()
            for old_nb in curr_node.nb:
                if old_nb not in old2new:
                    old2new[old_nb] = Node(val=old_nb.val)
                    q.append(old_nb)
                old2new[curr_node].nb.append(old2new[old_nb])
        
        return old2new[node]
                    
    def cloneDFShelper(self, node, old2new):
        if node in old2new:
            return old2new[node]
        old2new[node] = Node(val=node.val)
        for old_nb in curr_node.nb:
            old2new[node].nb.append(self.cloneDFShelper(old_nb, old2new))
        
        return old2new[node]
    
    def cloneDFS(self, node):
        old2new = defaultdict(Node)
        self.cloneDFShelper(node, old2new)
        return old2new[node]


class mergeTwoTrees:
    def mergeTwoTrees(self, root1, root2):
        if root1 and root2:
            new_root = Node(root1.val + root2.val)
            new_root.left = self.mergeTwoTrees(root1.left, root2.left)
            new_root.left = self.mergeTwoTrees(root1.left, root2.left)
            return new_root
        if root1:


class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not (t1 and t2):
            return t1 or t2
        new_root = TreeNode()
        queue1, queue2, queue3 = collections.deque([t1]), collections.deque([t2]), collections.deque([new_root])
        while queue1 and queue2:
            node1, node2, node3 = queue1.popleft(), queue2.popleft(), queue3.popleft()
            if node1 or node2:
                if node1 and node2:
                    node3 = TreeNode(node1.val + node2.val)
                    queue1.append(node1.left)
                    queue1.append(node1.right)
                    queue2.append(node2.left)
                    queue2.append(node2.right)
                    queue3.append(node3.left)
                    queue3.append(node3.right)
                if node1: # node 2 None
                    node3 = TreeNode(node1.val)
                    queue1.append(node1.left)
                    queue1.append(node1.right)
                    queue2.append(None)
                    queue2.append(None)
                    queue3.append(node3.left)
                    queue3.append(node3.right)
                if node2:
                    node3 = TreeNode(node2.val)
                    queue1.append(None)
                    queue1.append(None)
                    queue2.append(node2.left)
                    queue2.append(node2.right)
                    queue3.append(node3.left)
                    queue3.append(node3.right)                    

        return t1
    

class Diameter:
    def getDiameter(self, root):
        self.res = 0
        self.getDepth(root)
        return self.res
    
    def getDepth(self, node):
        if not node:
            return 0
        leftDepth = self.getDepth(node.left)
        rightDepth = self.getDepth(node.right)
        depth = 1 + leftDepth + rightDepth
        self.res = max(self.res, depth)
        return 1 + max()


def helper(node1, node2):
    if node1.val != node2.val:
        return False


def copy(root):
    old2new = dict()
    copynode()
    return o2n[]


def copynode(node, old2new):
    if node is None:
        return node
    if node in old2new:
        return old2new[node]
    old2new[node] = TreeNode(node.val)
    old2new[node].left = copynode(node.left, old2new)
    old2new[node].left = copynode(node.left, old2new)
    old2new[node].left = copynode(node.left, old2new)
    return old2new[node]


def copynode(node, old2new):
    old2new = dict()
    q = deque()
    q.append(node)
    old2new[node] = TreeNode(node.val)
    while q:
        curr_node = q.popleft()
        if curr_node.left:
            if curr_node.left not in old2new:
                old2new[curr_node.left] = TreeNode(node.left.val)
            old2new[curr_node].left = old2new[curr_node.left]
            q.append(curr_node.left)



class Solution:
    def restoreIpAddresses(self, s: str):
        res = []
        self.restoreHelper(s, [], 0, res)
        return res
    
    def restoreHelper(self, rem_str, curr_res, curr_len, res):
        if rem_str == "" and curr_len == 4:
            res.append(".".join(curr_res))
            return 
        if rem_str == "":
            return 
        for i in range(1, 4):
            if i <= len(rem_str): # o.w. duplicate
                if i == 1 or (rem_str[0] != '0' and int(rem_str[:i]) <= 255):
                    self.restoreHelper(rem_str[i:], curr_res+[rem_str[:i]], curr_len+1, res)
        return 

sol=Solution()
sol.restoreIpAddresses("25525511135")


graph = [[3,1],[3,2,4],[3],[4],[]]; start=0; target=4


def getShortestPaths_bfs(graph, start, target):
    q = deque()
    parent = defaultdict(list)
    q.append((start, [start]))
    visited = set()
    find_target = False
    res = []
    while len(q) > 0:
        curr_visited = set()
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
        visited.update(curr_visited)
        if find_target:
            return res
    return res


getShortestPaths_bfs([[3,1],[3,2,4],[3],[4],[]], 0, 4)


# def findWord(board, i, j, rem_word):
#     if rem_word == "":
#         return True
#     if rem_word[0] != board[i][j]:
#         return False
#     if (i, j) in visited:
#         return False
#     visited.add((i, j))
#     for next_i, next_j in neighbor(i, j):
#         if 0 <= next_i < nrow and 0<= next_j < ncol:
#             if findWord(board, next_i, next_j, rem_word[1:]):
#                 return True
#     visited.delete((i, j))
#     return False


class editDistance:
    def editDistance(self, w1, w2):

    
    def partialDistance(self, w1, i1, w2, i2):
        if i1 == len(w1) - 1 and i2 == len(w2) - 1:
            
        

class editDistance:
    def editDistance(self, w1, w2):
        dp = [[float('Inf') for _ in range(len(w2)+1)] for _ in range(len(w1)+1)]
        # partdist[i][j] = w1[:i], w2[:j]
        for i in range(len(dp)):
            dp[i][0] = i
        for j in range(len(dp[0])):
            dp[0][j] = j
        
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if w1[i-1] == w2[j-1]:
                    dp[][] = dp[i-1][j-1]
                else:
                    dp[][] = 1 + min(min(), )
                    

                dp[i][j] = 1 + min()
                        
def maxProfit(startTime, endTime, profit):
    # sort by endTime
    # max profit at endTime[i]
    prev_i = find_last_non_overlap_job(i) # last job with endtime no later than startTime[i]
    dp[i] = dp[i-1] + job[i] if startTime[i] >= endTime[i-1] else max(dp[i-1], dp[prev_i]+profit[i])


    
    
def assignJob(arr, d, start_idx, curr_res):
    if d == 1:
        self.res = min(self.res, curr_res + max(arr[start_idx:]))
        return 
    if d > len(d[start_idx:]):
        return 
    for i in range(start_idx+1, len(d)):
        assignJob(arr, d-1, i, curr_res + max(arr[start_idx:i]))
    
    return 


def jumpstones(arr, start_idx, step):
    if start_idx = target:
        return True
    for k in [step-1, step, step+1]:
        next_idx = start_idx + k
        if next_idx > len(start_idx)-1:
            continue
        if jumpstones(arr, next_idx, k):
            return True
    return False


class Solution:
    def longestIncreasingPath(self, matrix) -> int:
        visited = [[-1 for _ in range()] for _ in range()]
        res = 0
        for i in range():
            for j in range():
                res = max(res, self.getlongestIncreasingPath(matrix, i, j, visited, 1))
        
        return res
    
    def getlongestIncreasingPath(self, matrix, i, j, visited, curr_len):
        visited[][] > 0:
            return visited[i][j]
        
        for ni, nj in getNB(i, j):
            if 0<=ni<n and 0<=nj<m:
                if matrix[ni][nj] > matrix[i][j] and visited[][] < curr_len + 1:
                    visited[][] = curr_len + 1
                    curr_res = max(1 + self.getlongestIncreasingPath(matrix, ni, nj, visited, curr_len+1))
        return visited[i][j]
                

class UF:
    def __init__(self, n):
        self.parents = list(range(n))
        self.weights = [1] * n
    
    def find(self, i):
        if self.parents[i] == i:
            return i
        curr_parent = self.parents[i]
        self.parents[i] = self.find(curr_parent)
        return self.parents[i]
    
    def union(self, i, j):
        if self.parents[i] == self.[j]:
            return
        if weight_i < weight_j:
            self.parents[i] = self.parents[j]
            self.weights[j] += self.wegiht_i
        

accounts = [[name1, email1, ...], []]


class Solution:
    def merge(self, accounts):
        email2id = dict()
        for id, account in enumerate(accounts):
            for email in account:
                if email in email2id:
                    uf.union(id, email2id[email])
                else:
                    email2id[email] = id
        id2email = dict()
        for email in email2id.keys():
            curr_id = email2id[email]
            root_id = uf.find(curr_id)
            id2email[root_id].append(email)
        


"""
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
Explanation: 
"""
class Solution:
    def merge(self, accounts):
        email2id = defaultdict(list)
        email2name = dict()
        for idx, account in enumerate(accounts):
            for i in range(1, len(account)):
                email2id[account[i]].append(idx)
                email2name[account[i]] = account[0]
        
        visited = set()
        for email in email2id.keys():
            curr_res = self.bfs(email, accounts, email2id, visited)
            if len(curr_res) > 0:
                res.append([] + [])
        
        return res
    
    def bfs():
        curr_res = []
        if email in visited:
            return curr_res
        q = deque()
        q.append(email)
        visited.add()
        curr_res.append()
        while q:
            curr_email = q.popleft()
            for next_id in email2id:
                for next_email in id2email[next_id]:
                    if next_email not in visited:
                        q.append(email)
                        curr_res.append(email)
                        visited.add()
        return curr_res


class FenwickTree2:
    # [0, 1, 1+2, 3, 1+2+3+4, 5, 5+6, 7, 1+...+8]
    def __init__(self, n):
        # https://en.wikipedia.org/wiki/Fenwick_tree
        # sum(nums[:(i+1)])
        self.sums = [0] * n
    
    def update(self, i, delta):
        # nums[i]
        if i == 0:
            self.sums[0] += delta
            return 
        while i < len(self.sums):
            self.sums[i] += delta
            i += FenwickTree2.lowbit(i)
        return 
    
    def getSum(self, i):
        # sum(nums[:i+1])
        if i < 0:
            return 0
        res = self.sums[0]
        while i > 0:
            res += self.sums[i]
            i -= FenwickTree2.lowbit(i)
        return res
    
    @staticmethod
    def lowbit(x):
        return x & (-x)

class NumArray2:
    def __init__(self, nums):
        self.tree = FenwickTree2(len(nums))
        self.nums = nums
        for i, num in enumerate(nums):
            self.tree.update(i, num)
        
    def update(self, i, val):
        # nums[i] set as val
        self.tree.update(i, val-self.nums[i])
        self.nums[i] = val
    
    def sumRange(self, i, j):
        # sum(nums[i:j+1]) = sum(nums[:j+1]) - sum(nums[:i])
        return self.tree.getSum(j) - self.tree.getSum(i-1)


sol=NumArray2([1, 3, 5])
sol.sumRange(0, 2) # 9
sol.update(1, 2)
sol.nums
sol.sumRange(0, 2) # 8
sol.tree.getSum(2)


def coinChange(rem_amount, start_idx, res):
    if rem_amount == 0:
        return 1
    res = 0
    for i in range(start_idx, len(coins)):
        if rem_amount >= coins[i]:            
            res += coinChange(rem_amount - coins[i], i)
    memo[(rem_amount, start_idx)] = res
    return res


def removeLeaves(root, res):
    if not root:
        return
    if isLeave(root):
        res.append(root)
        return
    root.left = removeLeaves(root.left, res)
    root.right = removeLeaves(root.right, res)
    return root

def getLeaves(root):
    res = []
    while root:
        leaves = []
        root = removeLeaves(root, leaves)
        res.append(leaves)
    return res








"""
Input: n = 2
Output: 8
Explanation: There are 8 records with length 2 that are eligible for an award:
"PP", "AP", "PA", "LP", "PL", "AL", "LA", "LL"
Only "AA" is not eligible because there are 2 absences (there need to be fewer than 2).
"""
        


class Solution:
    def amountPainted(self, paint: List[List[int]]) -> List[int]:
        dic = collections.defaultdict(list)
        for i, (start, end) in enumerate(paint):
            dic[start].append([i, 1])
            dic[end].append([i, -1])
        daySet = SortedList()
        ans = [0 for _ in paint]
        arr = [(k, dic[k]) for k in sorted(dic)]
        for i, (pos, flags) in enumerate(arr):
            for idx, flag in flags:
                if flag == -1:
                    daySet.pop(idx) # logN
                else:
                    daySet.add(idx) # logN
            if i < len(arr) - 1 and daySet:
                ans[daySet[0]] += arr[i+1][0]- arr[i][0]
        return ans


x = [[1,2], [0,3]]
x.sort(key=lambda x: x[1])