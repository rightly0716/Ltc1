
""" 
############################################################################
Meta / FB
############################################################################
https://www.1point3acres.com/bbs/interview/facebook-software-engineer-693957.html
近期FB 离口算法题总结， *为地里多次出现的面经5，10，23*，27，29，32，38，39，40，52，56*，57，67，69，88，92，
98，114，124*，125，126，127，130，136，138，139，140，154，155，158*，163，173，199*，200，210，211*，215*，
227*，235，236*，238*，239，246，273，278，282*，297，301，322，333，339*，348*，378，392*，408，415，438*，
468，485，498，528，540，547，560*（变体，全是正数和带有负数两种情况），621，637，674，680，766，778, 785，824，
919，921，938*，940，953*，958，973，977，986，987，1004，1026，1029，1123，1197，1249*，1428*，1570*，1574

https://imgbb.com/6JbqSmF
"""

# need to review
# binary segment tree, binary index tree (307)

# 
# 426 Convert BST to linked list O(1) solution
# 65 valid number
# 636 Exclusive Time of Functions
# 173. Binary Search Tree Iterator
# 708. Insert into a Cyclic Sorted List
# 282. expression add ope
# 616 add bold tag (idea)
# 398 reservior sampling
# 1216 valid palindrom iii
# 825 Friends of appropriate age
# 1944
# 2076
# 43
# 10
# 333
# 489. Robot Room Cleaner
# 463  Island Perimeter
# 117. Populating Next Right Pointers in Each Node II
# 416. Partition Equal Subset Sum

"""[LeetCode] 408. Valid Word Abbreviation
Given a non-empty string s and an abbreviation abbr, return whether the string matches with the given abbreviation.

A string such as "word" contains only the following valid abbreviations:
["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

Given s = "internationalization", abbr = "i12iz4n":
Return true

class Solution {
public:
    bool validWordAbbreviation(string word, string abbr) {
        int i = 0, j = 0, m = word.size(), n = abbr.size();
        while (i < m && j < n) {
            if (abbr[j] >= '0' && abbr[j] <= '9') {
                if (abbr[j] == '0') return false;
                int val = 0;
                while (j < n && abbr[j] >= '0' && abbr[j] <= '9') {
                    val = val * 10 + abbr[j++] - '0';
                }
                i += val;
            } else {
                if (word[i++] != abbr[j++]) return false;
            }
        }
        return i == m && j == n;
    }
};
"""


"""173. Binary Search Tree Iterator
Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):

BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. 
The root of the BST is given as part of the constructor. The pointer should be initialized to a 
non-existent number smaller than any element in the BST.

boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise returns false.
int next() Moves the pointer to the right, then returns the number at the pointer.
    9
   / \
  3  20
    /  \
   15   27

Follow up: 
Could you implement next() and hasNext() to run in average O(1) time and use O(h) memory, where h is the height of the tree?

Hints: 
For O(n) solution, we just need to in order traverse the tree and save value into a stack/array

下面的做法的空间复杂度是O(h)，做法是每次保存要遍历的节点的所有左孩子。这样，每次最多也就是H个节点被保存，
当遍历了这个节点之后，需要把该节点的右孩子的所有左孩子放到栈里，这就是个中序遍历的过程。
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self.push_left(root) # recursively push left child

    def push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left
        return 

    def next(self) -> int:
        next_node = self.stack.pop()
        self.push_left(next_node.right)
        return next_node.val
        
    def hasNext(self) -> bool:
        return len(self.stack) > 0


# iterative: just for practice
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self.in_order(root) # recursively push left child

    def in_order(self, node):
        temp_stack = []
        curr = node
        while True:
            if curr:
                temp_stack.append(curr)
                curr = curr.right
            elif len(temp_stack) > 0:
                curr = temp_stack.pop()
                self.stack.append(curr)
                curr = curr.left
            else:
                break
        return 

    def next(self) -> int:
        next_node = self.stack.pop()
        return next_node.val
        
    def hasNext(self) -> bool:
        return len(self.stack) > 0


""" 498. Diagonal Traverse
Given an m x n matrix mat, return an array of all the elements of the array in a diagonal order.

Input: mat = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,4,7,5,3,6,8,9]

# https://www.cnblogs.com/grandyang/p/6414461.html
"""
class Solution:
    def findDiagonalOrder(self, mat):
        r, c = 0, 0
        res = []
        upright = True
        while True:
            res.append(mat[r][c])
            if r == len(mat) - 1 and c == len(mat[0]) - 1:
                break
            if upright:
                c += 1
                r -= 1
            else:
                c -= 1
                r += 1
            # four cases of go beyond
            if r == len(mat):
                r -= 1
                c += 2
                upright = not upright
            if c == len(mat[0]):
                c -= 1
                r += 2
                upright = not upright
            if r == -1:
                r = 0
                upright = not upright
            if c == -1:
                c = 0
                upright = not upright
        return res

# Output: [1,2,4,7,5,3,6,8,9]
sol=Solution()
sol.findDiagonalOrder([[1,2,3],[4,5,6],[7,8,9]])

"""1424. Diagonal Traverse II
Given a 2D integer array nums, return all elements of nums in diagonal order as shown in the below images.

Input: nums = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,4,2,7,5,3,8,6,9]

Input: nums = [[1,2,3,4,5],[6,7],[8],[9,10,11],[12,13,14,15,16]]
Output: [1,6,2,8,7,3,9,4,12,10,5,13,11,14,15,16]
"""
# Solution1: use a hashmap: r+c -> val, merge vals after sort by keys
# BFS: 
# The top-left number, nums[0][0], is the root node. nums[1][0] is its left child and nums[0][1] is 
# its right child. Same analogy applies to all nodes nums[i][j].

# Can further improve memory usage by not using visited
# Note that nums[i][j] is both the left child of nums[i-1][j] and the right child of nums[i][j-1]. 
# To avoid double counting, we only consider a number's left child when we are at the left-most column (j == 0).
# https://leetcode.com/problems/diagonal-traverse-ii/discuss/597690/Python-Simple-BFS-solution-with-detailed-explanation
from collections import deque
class Solution:
    def findDiagonalOrder(self, nums):
        q = deque()
        q.append((0, 0))
        visited = set((0,0))
        res = []
        while q:
            r, c = q.popleft()
            res.append(nums[r][c])
            if r + 1 < len(nums) and c < len(nums[r+1]) and (r+1, c) not in visited:
                visited.add((r+1, c))
                q.append((r+1, c))
            if c + 1 < len(nums[r]) and (r, c+1) not in visited:
                visited.add((r, c+1))
                q.append((r, c+1))
        
        return res

sol=Solution()
sol.findDiagonalOrder([[1,2,3,4,5],[6,7],[8],[9,10,11],[12,13,14,15,16]])


""" 636 Exclusive Time of Functions
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

Input: n = 2, logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
Output: [3,4]

Input: n = 1, logs = ["0:start:0","0:start:2","0:end:5","0:start:6","0:end:6","0:end:7"]
Output: [8]
"""
logs = ["0:start:0",  "1:start:2",  "1:end:5",  "0:end:6"]
n = 2
class Solution:
    def exclusiveTime(self, n, logs):
        res = [0] * n
        stack = []
        for log in logs:
            jid, status, t = log.split(':')
            jid, t = int(jid), int(t)
            if status == 'start':
                if len(stack) == 0:
                    stack.append([jid, t])
                else:
                    prev_jid, prev_t = stack[-1]
                    res[prev_jid] += t - prev_t
                    stack.append([jid, t])
            else:
                prev_jid, prev_t = stack.pop()
                res[prev_jid] += t - prev_t + 1
                if len(stack) > 0:
                    stack[-1][1] = t + 1 # end defined at end of time
        return res

exclusiveTime(n, logs)

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
from collections import defaultdict
class Solution:
    def groupStrings(self, strings):
        d = defaultdict(list)
        for s in strings:
            d[self.hashcode(s)].append(s)
        res = []
        for k in d:
            res.append(sorted(d[k]))
        return res

    def hashcode(self, s):
        out = [0] * len(s)
        for i in range(len(s)):
            out[i] = ord(s[i]) - ord(s[0]) if ord(s[i]) - ord(s[0]) >= 0 else ord(s[i]) - ord(s[0]) + 26
        return tuple(out)

sol=Solution()
sol.groupStrings(["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"])


"""1091. Shortest Path in Binary Matrix
Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. 
If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:

All the visited cells of the path are 0.
All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).

Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4

Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
Output: -1
"""
# BFS
from collections import deque
class Solution:
    def shortestPathBinaryMatrix(self, grid) -> int:
        if grid[0][0] == 1 or grid[-1][-1] == 1:
            return -1
        visited = set()
        q = deque()
        q.append((0, 0, 1))
        while len(q) > 0:
            for _ in range(len(q)):
                curr_i, curr_j, curr_dist = q.popleft()
                if (curr_i, curr_j) == (len(grid)-1, len(grid[0])-1):
                    return curr_dist
                for next_i, next_j in self.get_nb(curr_i, curr_j, grid):
                    if grid[next_i][next_j] == 0 and (next_i, next_j) not in visited:
                        visited.add((next_i, next_j))
                        q.append((next_i, next_j, curr_dist+1))
        return -1
    
    def get_nb(self, i, j, grid):
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        res = []
        for di, dj in dirs:
            if 0 <= i + di < len(grid) and 0 <= j + dj < len(grid[0]):
                res.append((i+di, j+dj))
        return res


sol = Solution()
sol.shortestPathBinaryMatrix([[0,1],[1,0]])
sol.shortestPathBinaryMatrix([[0,0,0],[1,1,0],[1,1,0]])
sol.shortestPathBinaryMatrix([[1,0,0],[1,1,0],[1,1,0]])
sol.shortestPathBinaryMatrix([[0,1,0,0,0,0],[0,1,0,1,1,0],[0,1,1,0,1,0],[0,0,0,0,1,0],[1,1,1,1,1,0],[1,1,1,1,1,0]])
sol.shortestPathBinaryMatrix([[0,0,1,1,0,0],[0,0,0,0,1,1],[1,0,1,1,0,0],[0,0,1,1,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0]])

# DFS: must use a dp to record otherwise TLE!!!
# Still TLE
class Solution:
    def shortestPathBinaryMatrix(self, grid) -> int:
        if grid[0][0] == 1 or grid[-1][-1] == 1:
            return -1
        dist = [[float('Inf') for _ in range(len(grid[0]))] for _ in range(len(grid))]
        dist[0][0] = 1  # path length to -1, -1 is 1
        self.traversal(0, 0, grid, dist)
        print(dist)
        return dist[-1][-1] if dist[-1][-1] < float('Inf') else -1
    
    def traversal(self, i, j, grid, dist):
        # print(self.get_nb(start, grid))
        for next_i, next_j in self.get_nb((i, j), grid):
            if grid[next_i][next_j] == 0: # and abs(dist[next_i][next_j] - dist[i][j]) > 1:
                if dist[next_i][next_j] > dist[i][j] + 1:
                    dist[next_i][next_j] = dist[i][j] + 1
                # else: # dist[i][j] > dist[next_i][next_j] + 1:
                #     dist[i][j] = dist[next_i][next_j] + 1
                    self.traversal(next_i, next_j, grid, dist)
        return None
    
    def get_nb(self, curr_point, grid):
        i, j = curr_point
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        res = []
        for di, dj in dirs:
            if 0 <= i + di < len(grid) and 0 <= j + dj < len(grid[0]):
                res.append((i+di, j+dj))
        return res


"""189. Rotate Array
Given an array, rotate the array to the right by k steps, where k is non-negative.

Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]

# hint on optimal
We can simulate the rotation with three reverses.
reverse the whole array O(n) [7,6,5,4,3,2,1]
reverse the left part 0 ~ k – 1 O(k) [5,6,7,4,3,2,1]
reverse the right part k ~ n – 1 O(n-k) [5,6,7,1,2,3,4]
"""
class Solution:
    def rotate(self, nums, k: int) -> None:
        n = len(nums)
        k = k % n
        self.inverse(nums, 0, n-1)
        self.inverse(nums, 0, k-1)
        self.inverse(nums, k, n-1)
        return None

    def inverse(self, nums, i, j):
        while i < j:
            nums[i], nums[j] = nums[j], nums[i]
            i, j = i + 1, j - 1
        return None

sol=Solution
nums = [-1, -100, 3, 99, 2]; k = 2


"""65. Valid Number
Validate if a given string can be interpreted as a decimal number.

Some examples:
"0" => true
" 0.1 " => true
"abc" => false
"1 a" => false
"2e10" => true
" -90e3   " => true
" 1e" => false
"e3" => false
" 6e-1" => true
" 99e2.5 " => false
"53.5e93" => true
" --6 " => false
"-+3" => false
"95a54e53" => false

Note: It is intended for the problem statement to be ambiguous. You should gather all requirements up front before implementing one. 
However, here is a list of characters that can be in a valid decimal number:

Numbers 0-9
Exponent - "e"
Positive/negative sign - "+"/"-"
Decimal point - "."
Of course, the context of these characters also matters in the input.

## Hint
We use three flags: met_dot, met_e, met_digit, mark if we have met ., e or any digit so far. 
First we strip the string, then go through each char and make sure:

If char == + or char == -, then prev char (if there is) must be e
. cannot appear twice or after e
e cannot appear twice, and there must be at least one digit before and after e
All other non-digit char is invalid
https://leetcode.com/problems/valid-number/discuss/173977/Python-with-simple-explanation
"""
import re
class Solution:
    def isNumber(self, s):
        s = s.strip()
        s_split_by_e = re.split('e|E', s)
        if len(s_split_by_e) > 2:
            return False
        if len(s_split_by_e) == 1:
            return self.is_decimal(s_split_by_e[0])
        else:
            decimal, integer = s_split_by_e
            if len(decimal) == 0 or len(integer) == 0:
                return False
            return self.is_decimal(decimal) and self.is_decimal(integer, True)

    def is_decimal(self, s, is_integer=False):
        met_dot = False
        met_digit = False
        for i, char in enumerate(s):
            if char in ['-', '+']:
                if i != 0 or i == len(s)-1:
                    return False
            elif char == '.':
                if is_integer:
                    return False
                if met_dot:
                    return False
                met_dot = True
            elif char.isdigit():
                met_digit = True
            else:
                return False
        return met_digit


sol=Solution()
sol.isNumber("4.e+19")
sol.isNumber("4e+")
sol.isNumber("e")
sol.isNumber(".")

class Solution:
    def isNumber(self, s):
        s = s.strip()
        met_dot = met_e = met_digit = False
        for i, char in enumerate(s):
            if char in ['+', '-']:
                if i > 0 and s[i-1] not in ['e', 'E']:
                    return False
            elif char == '.':
                if met_dot or met_e: return False
                met_dot = True
            elif char in ['e', 'E'] :
                if met_e or not met_digit:
                    return False
                met_e, met_digit = True, False
            elif char.isdigit():
                met_digit = True
            else:
                return False
        return met_digit


""" LeetCode 426. Convert Binary Search Tree to Sorted Doubly Linked List
Convert a Binary Search Tree to a sorted Circular Doubly-Linked List in place.

We want to transform this BST into a circular doubly linked list. Each node in a doubly linked 
list has a predecessor and successor. For a circular doubly linked list, the predecessor of 
the first element is the last element, and the successor of the last element is the first element. 

Specifically, we want to do the transformation in place. After the transformation, 
the left pointer of the tree node should point to its predecessor, and the right pointer 
should point to its successor. We should return the pointer to the first element of the linked list
"""
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
# O(n) memory
class Solution:
    def treeToDoublyList(self, root):
        if root is None:
            return root
        self.stack = []
        self.inorder(root)
        for i in range(1, len(self.stack)):
            self.stack[i].right = self.stack[i+1]
            self.stack[i].left = self.stack[i-1]
        self.stack[0].left = self.stack[-1]
        self.stack[-1].right = self.stack[0]
        return self.stack[0]
    
    def inorder(self, node):
        if not node:
            return
        self.inorder(node.left)
        self.stack.append(node)
        self.inorder(node.right)
        return 

# O(1) memory: use self.prev node
class Solution:
    def treeToDoublyList(self, root):
        if root is None:
            return root
        self.prev = dummy = Node(-1)
        self.inorder(root)
        self.prev.right = dummy.right
        dummy.right.left = self.prev
        return dummy.right
    
    def inorder(self, node):
        if not node:
            return
        self.inorder(node.left)
        self.prev.right = node
        node.left = self.prev
        self.prev = self.prev.right
        self.inorder(node.right)
        return 


"""[LeetCode] 31. Next Permutation 下一个排列
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

hint
一个数组
1　　2　　7　　4　　3　　1

下一个排列为：
1　　3　　1　　2　　4　　7

那么是如何得到的呢，我们通过观察原数组可以发现，如果从末尾往前看，数字逐渐变大，到了2时才减小的，
然后再从后往前找第一个比2大的数字，是3，那么我们交换2和3，再把此时3后面的所有数字转置一下即可，步骤如下：
1　　2　　7　　4　　3　　1
1　　2　　7　　4　　3　　1
1　　3　　7　　4　　2　　1
1　　3　　1　　2　　4　　7
"""
class Solution:
    def nextPermutation(self, nums) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(1, len(nums))[::-1]:
            if nums[i] > nums[i-1]:
                # nums[i:] is decreasing order
                idx = self.find_last_larger_num(nums, i, nums[i-1])
                nums[i-1], nums[idx] = nums[idx], nums[i-1]
                nums[i:] = nums[i:][::-1]
                return None
        # means nums are sorted descreaing
        nums.reverse()  # nums = nums[::-1] does not update nums!!
        return None
    
    def find_last_larger_num(self, arr, left, x):
        low, high = left, len(arr)
        while low < high:
            mid = low + (high - low) // 2
            if arr[mid] > x:
                low = mid + 1
            else:
                high = mid
        return low - 1


sol = Solution()
nums = [1,2,3]
nums = [3, 2, 1]
sol.nextPermutation(nums)
nums


"""556. Next Greater Element III
Given a positive integer n, find the smallest integer which has exactly the same digits existing in the 
integer n and is greater in value than n. If no such positive integer exists, return -1.

Input: n = 12443322
Output: 13222344

Idea is similar to previous
"""
class Solution:
    def nextGreaterElement(self, n: int) -> int:
        # find first non-increasing element from right hand side
        str_n = [int(i) for i in str(n)]
        for i in range(1, len(str_n))[::-1]:
            if str_n[i-1] < str_n[i]:
                # find last element > str_n[i]
                idx = self.find_larger_element(str_n, str_n[i-1], i)
                str_n[i-1], str_n[idx] = str_n[idx], str_n[i-1]
                res_list = str_n[:i] + str_n[i:][::-1]
                res = int(''.join([str(i) for i in res_list]))
                return res if res < 2 ** 31 else -1
                
        return -1

    def find_larger_element(self, s, target, start_left):
        left, right = start_left, len(s)
        while left < right:
            mid = left + (right - left) // 2
            if s[mid] > target:
                left = mid + 1
            else:
                right = mid
        return left - 1


sol=Solution()
sol.nextGreaterElement(12443322)
        
"""827. Making A Large Island
You are given an n x n binary matrix grid. You are allowed to change at most one 0 to be 1.
Return the size of the largest island in grid after applying this operation.
An island is a 4-directionally connected group of 1s.

Input: grid = [[1,0],[0,1]]
Output: 3
Explanation: Change one 0 to 1 and connect two 1s, then we get an island with area = 3.

遇到1就遍历当前的island，然后存入一个dict with key as island_id and value as size, 同时把岛上的grid[i][j]
全变成island_id，
"""

class Solution:
    def largestIsland(self, grid) -> int:
        cur_id = 2
        id2size = dict()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    id2size[cur_id] = self.getCurrIslandSize(grid, i, j, cur_id)
                    cur_id += 1
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    curr_max = self.get_max_size(grid, i, j, id2size)
                    res = max(res, curr_max)
        
        return res if res > 0 else len(grid) * len(grid[0])
    
    def get_max_size(self, grid, i, j, id2size):
        res = 1
        id_set = set()
        for n_i, n_j in self.get_nb(grid, i, j):
            if grid[n_i][n_j] != 0 and grid[n_i][n_j] not in id_set:
                res += id2size[grid[n_i][n_j]]
                id_set.add(grid[n_i][n_j])
        return res

    def getCurrIslandSize(self, grid, i, j, cur_id):
        res = 1
        grid[i][j] = cur_id
        for n_i, n_j in self.get_nb(grid, i, j):
            if grid[n_i][n_j] == 1:
                res += self.getCurrIslandSize(grid, n_i, n_j, cur_id)
        return res

    def get_nb(self, grid, i, j):
        res = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_i, next_j = i+di, j+dj
            if 0 <= next_i < len(grid) and 0<= next_j < len(grid[0]):
                res.append((next_i, next_j))
        return res


sol=Solution()
sol.largestIsland([[1,0],[0,1]])
sol.largestIsland([[1,1],[1,0]])
sol.largestIsland([[0,0],[0,0]])

"""791. Custom Sort String
You are given two strings order and s. All the words of order are unique and were sorted 
in some custom order previously.
Permute the characters of s so that they match the order that order was sorted. More 
specifically, if a character x occurs before a character y in order, then x should occur 
before y in the permuted string.

Return any permutation of s that satisfies this property.

Input: order = "cba", s = "abcd"
Output: "cbad"

Input: order = "cbafg", s = "abcd"
Output: "cbad"
"""
class Solution:
    def customSortString(self, S: str, T: str) -> str:        
        counter =collections.Counter(T)
        res = ""
        for i in S:
            if i in counter:
                res += i*counter[i]
                counter.pop(i)
        for k,v in counter.items():
            res += k*v
        return res


"""438. Find All Anagrams in a String
Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Input: s = "cbaebabacd"; p = "abc"
Output: [0,6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".

Input: s = "abab", p = "ab"
Output: [0,1,2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
"""
from collections import Counter
class Solution:
    def findAnagrams(self, s: str, p: str):
        d = Counter(p)
        n = len(p)
        if len(s) < n:
            return []
        res = []
        cnt = 0
        for i in range(n):
            if s[i] in d:
                d[s[i]] -= 1
                if d[s[i]] >= 0:
                    cnt += 1
        if cnt == n:
            res.append(0)
        
        for i in range(n, len(s)):
            # remove i-n, add i
            if s[i-n] in d:
                d[s[i-n]] += 1
                if d[s[i-n]] > 0:
                    cnt -= 1
            if s[i] in d:
                d[s[i]] -= 1
                if d[s[i]] >= 0:
                    cnt += 1
            if cnt == n:
                res.append(i-n+1)
        return res

# another way is to create two hashmaps to compare, because ther are 26 letters, comparing two hm takes O(1)
from collections import Counter
class Solution:
    def findAnagrams(self, s, p):
        res = []
        n = len(p)
        if len(s) < n:
            return []
        pCounter = Counter(p)
        sCounter = Counter(s[:n])
        if pCounter == sCounter:
            res.append(0)
        for i in range(n, len(s)):
            sCounter[s[i]] += 1   # include a new char in the window
            sCounter[s[i-n]] -= 1   # decrease the count of oldest char in the window
            if sCounter[s[i-n]] == 0:
                del sCounter[s[i-n]]   # remove the count if it is 0
            if sCounter == pCounter:    # This step is O(1), since there are at most 26 English letters 
                res.append(i-n+1)   # append the starting index
        return res


"""392. Is Subsequence
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) 
of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

Input: s = "abc"; t = "ahbgdc"
Output: true

Follow up: Suppose there are lots of incoming s, say s1, s2, ..., sk where k >= 109, 
and you want to check one by one to see if t has its subsequence. In this scenario, how would you change your code?

建立字符串t中的每个字符跟其位置直接的映射,相同的字符出现的所有位置按顺序加到一个数组中
使用二分搜索来加快搜索速度
需要一个变量 pre 来记录当前匹配到t字符串中的位置，对于当前s串中的字符c，即便在t串中存在，但是若其在位置 pre 之前，也是不能匹配的
"""
s = "acb"; t="ahbgdc"
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        d = defaultdict(list)
        for i, letter in enumerate(t):
            d[letter].append(i)
        # given a new s
        curr_loc = -1
        for letter in s:
            next_loc = find_next_location(d[letter], curr_loc) # find location greater than curr_loc
            if next_loc == -1:
                return False
            curr_loc = next_loc
        return True

def find_next_location(loc_list, idx):
    # return the location after idx
    l, r = 0, len(loc_list)
    while l < r:
        mid = l + (r - l) // 2
        if loc_list[mid] <= idx:
            l = mid + 1
        else:
            r = mid
    return loc_list[r] if r < len(loc_list) else -1


"""238. Product of Array Except Self
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
You must write an algorithm that runs in O(n) time and without using the division operation.

Example 1:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Follow up: use O(1) space
如果我们知道其前面所有数字的乘积，同时也知道后面所有的数乘积，那么二者相乘就是我们要的结果，
所以我们只要分别创建出这两个数组即可，分别从数组的两个方向遍历就可以分别创建出乘积累积数组
Follow up: 先从前面遍历一遍，将乘积的累积存入结果 res 中，然后从后面开始遍历，用到一个临时变量 right，初始化为1，然后每次不断累积
"""
class Solution:
    # @param {integer[]} nums
    # @return {integer[]}
    def productExceptSelf(self, nums):
        n = len(nums)
        output = [1] * n
        for i in range(1,n):
            # for now output saves prodct of all nums <i (on left)
            output[i] = output[i-1] * nums[i-1]
            
        p = 1 # product of all nums >i
        for i in range(n-1,-1,-1):
            output[i] = output[i] * p
            p = p * nums[i]
        return output

"""560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

Example 1:
Input: nums = [1,1,1], k = 2
Output: 2

Example 2:
Input: nums = [1,2,3], k = 3
Output: 2
"""
# 用一个 HashMap 来建立连续子数组之和跟其出现次数之间的映射，初始化要加入 {0,1} 这对映射
from collections import defaultdict
class Solution:
    def subarraySum(self, nums, k):
        d = defaultdict(lambda: 0)
        d[0] = 1 # cumsum to freq
        cumsum = 0
        res = 0
        for i, num in enumerate(nums):
            cumsum += num
            if cumsum - k in d:
                res += d[cumsum - k]
            d[cumsum] += 1
        return res

sol=Solution()
sol.subarraySum(nums=[-1,-1, 1], k=0)
sol.subarraySum(nums=[-1,-1, 1], k=1)
sol.subarraySum(nums=[1,1, 1], k=2)
sol.subarraySum(nums=[1], k=0)


"""67. Add Binary
Given two binary strings a and b, return their sum as a binary string.
Example 1:
Input: a = "11", b = "1"
Output: "100"

Example 2:
Input: a = "1010", b = "1011"
Output: "10101"
"""

"""129. Sum Root to Leaf Numbers
You are given the root of a binary tree containing digits from 0 to 9 only.
Each root-to-leaf path in the tree represents a number.
For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
Return the total sum of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a 32-bit integer.

A leaf node is a node with no children.

Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root) -> int:
        if root is None:
            return 0
        self.res = 0
        self.dfs(root, 0)
        return self.res
        
    def dfs(self, node, curr_val):
        # add curr_val into self.res if leaf
        if node.left is None and node.right is None:
            self.res += curr_val * 10 + node.val
            return 
        if node.left:
            self.dfs(node.left, curr_val * 10 + node.val)
        if node.right:
            self.dfs(node.right, curr_val * 10  + node.val)
        
        return None


"""787. Cheapest Flights Within K Stops
There are n cities connected by some number of flights. You are given an array flights where 
flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.
You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. 
If there is no such route, return -1.

Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200
Explanation: The graph is shown.
The cheapest price from city 0 to city 2 with at most 1 stop costs 200, as marked red in the picture.
src != dst
"""
# BFS: V^2 the worst, Dijkstra's Algorithm O ( V + E l o g V ) .
from collections import deque, defaultdict
class Solution:
        def findCheapestPrice(self, n, flights, src, dst, k):
            m = defaultdict(list)
            for src_stop, tgt_stop, price in flights:
                m[src_stop].append([tgt_stop, price])
            
            q = deque()
            q.append((src, 0, 0)) # src, price, k
            res_cities = [float('Inf')] * n
            while len(q) > 0:
                for _ in range(len(q)):
                    curr_src, curr_price, curr_step = q.popleft()
                    if curr_step <= k:
                        for curr_tgt, next_flight_price in m[curr_src]:
                            if curr_price + next_flight_price < res_cities[curr_tgt]:
                                res_cities[curr_tgt] = curr_price + next_flight_price
                                q.append((curr_tgt, curr_price+next_flight_price, curr_step+1))

            return res_cities[dst] if  res_cities[dst] < float('Inf') else -1

sol= Solution()
sol.findCheapestPrice(n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1)


"""71. Simplify Path
Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert 
it to the simplified canonical path.
In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, 
and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods 
such as '...' are treated as file/directory names.

The canonical path should have the following format:

The path starts with a single slash '/'.
Any two directories are separated by a single slash '/'.
The path does not end with a trailing '/'.
The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')

Input: path = "/home/"
Output: "/home"

Input: path = "/../"
Output: "/"

Input: path = "/home//foo/"
Output: "/home/foo"
"""
class Solution(object):
    def simplifyPath(self, path):
        places = [p for p in path.split("/") if p!="." and p!=""]
        stack = []
        for p in places:
            if p == "..":
                if len(stack) > 0:
                    stack.pop()
            else:
                stack.append(p)
        return "/" + "/".join(stack)


"""650. 2 Keys Keyboard
There is only one character 'A' on the screen of a notepad. You can perform two operations on this notepad for each step:

Copy All: You can copy all the characters present on the screen (a partial copy is not allowed).
Paste: You can paste the characters which are copied last time.
Given an integer n, return the minimum number of operations to get the character 'A' exactly n times on the screen.

Input: n = 3
Output: 3
Explanation: Intitally, we have one character 'A'.
In step 1, we use Copy All operation.
In step 2, we use Paste operation to get 'AA'.
In step 3, we use Paste operation to get 'AAA'.
"""
# DP + greedy
"""
public int minSteps(int n) {
    int[] dp = new int[n+1];
    for (int i = 2; i <= n; i++) {
        dp[i] = i; # if prime number
        for (int j = i//2; j > 1; j--) {
            if (i % j == 0) {
                dp[i] = dp[j] + (i/j);
                break;
            }
            
        }
    }
    return dp[n];
}
"""

"""类似 思久斯 494 和 尔捌迩 282
输入是一串数字“1234567‍‌‌‌‍‌‌‍‌‌‍‌‌‌‍‌‌‍‌89”，在中间随意增加+或-，输出所有结果等于一百的组合，比如123+45-67+8-9
"""
from functools import lru_cache
def findTargetSumWays(num_str, target: int) -> int:
    # get all the possible combinations 
    all_comb = dfs(num_str)
    # see how manny = target
    res = []
    for comb in all_comb:
        if eval(comb) == target:
            clean_comb = comb if comb[0] != '+' else comb[1:]
            res.append(clean_comb)
    return res

@lru_cache(None)
def dfs(num_str):
    if num_str[0] == '0' and len(num_str) > 1:
        # illegal string
        return []
    res = ['+'+num_str, '-'+num_str]
    for i in range(1, len(num_str)):
        pre_comb = ['+' + num_str[:i], '-' + num_str[:i]]
        post_comb = dfs(num_str[i:])
        for pre in pre_comb:
            for post in post_comb:
                res.append(pre+post)
    return res

dfs("123")
num_str = "123456789"
target = 100

"""282. Expression Add Operators
Given a string num that contains only digits and an integer target, return all possibilities to insert the binary operators 
'+', '-', and/or '*' between the digits of num so that the resultant expression evaluates to the target value.

You may also merge two neighboring digits directly

Input: num = "123", target = 6
Output: ["1*2*3","1+2+3"]
Explanation: Both "1*2*3" and "1+2+3" evaluate to 6.

Input: num = "123", target = 15
Output: ["12+3"]
"""
num = "105"; target=5
# TLE
from functools import lru_cache
class Solution:
    def addOperators(self, num: str, target: int):
        all_comb = self.get_all_combination(num)
        res = []
        for comb in all_comb:
            eval_output = eval(comb)
            if eval_output== target:
                res.append(comb)
        return res

    @lru_cache(None)
    def get_all_combination(self, num_str):
        # return all combinations in a list
        res = [] if num_str[0] == '0' and len(num_str) > 1 else [num_str]
        for i in range(1, len(num_str)):
            post_num = num_str[i:] # do not split any more
            if post_num[0] == '0' and len(post_num) > 1:
                continue
            pre_num_list = self.get_all_combination(num_str[:i])
            for pre_num in pre_num_list:
                res.append(pre_num + '+' + post_num)
                res.append(pre_num + '-' + post_num)
                res.append(pre_num + '*' + post_num)
        return res


# Imagine you are currently evaluating the expression 5 + 2 * 3, the dfs method has last = 2, cur= 7,
# To evaluate expression A + B * C, it should be read with multiplication taking precedence, A + (B * C), 
# so result should be 5 + (2 * 3) => 11. Without last, one could end up calculating result as (5+2)*3 => 21
# Hence the expression, cur - last + last * val => 7-2 + (2 * 3) = 11
class Solution:
    def addOperators(self, num: str, target: int):
        res = []
        dfs(num, "", 0, 0, target, res)
        return res

    # add cache trick is even slower
    def dfs(self, rem_num, output, curr, last, target, res):
        """
        rem_num: remaining num string
        output: temporally string with operators added
        cur: current result of "temp" string
        last: last multiply-level number in "temp". if next operator is "multiply", "cur" and "last" will be updated
        """
        if curr == target and len(rem_num) == 0:
            res.append(output)
            return None
        for i in range(1, len(rem_num)+1):
            pre_num = rem_num[:i] # no split anymore
            if pre_num[0] == '0' and len(pre_num) > 1:
                # illegal 
                return None
            val = int(pre_num)
            if len(output) > 0:
                # Case 1: 2 + 3*4   *    5
                # last: 3*4  -> 3*4*5 (last * val)
                # curr: 2+3*4  -> 2 + 3*4*5  (curr-last) + last * val
                # val: 5
                # Case 2: 2 + 3*4   +    5
                # last: 3*4  -> 5 (val)
                # curr: 2+3*4  -> 2 + 3*4 + 5 (curr+val)
                self.dfs(rem_num[i:], output + "+" + pre_num, curr+val, val, target, res)
                self.dfs(rem_num[i:], output + "-" + pre_num, curr-val, -1*val, target, res)
                self.dfs(rem_num[i:], output + "*" + pre_num, (curr-last) + last * val, last * val, target, res)
            else:
                # at the first step, when output = ""
                self.dfs(rem_num[i:], pre_num, curr+val, val, target, res)
        return None

sol=Solution()
sol.addOperators("123", 6)
sol.addOperators("00", 0)


"""477. Total Hamming Distance
The Hamming distance between two integers is the number of positions at which the corresponding bits are different.
Given an integer array nums, return the sum of Hamming distances between all the pairs of the integers in nums.

Example 1:
Input: nums = [4,14,2]
Output: 6
Explanation: In binary representation, the 4 is 0100, 14 is 1110, and 2 is 0010 (just
showing the four bits relevant in this case).
The answer will be:
HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.
"""
# O(n) solution:
# at ith position, count how many have 1's and 0's, then multiple to get the distance at ith position
class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        ans = 0
        for i in range(32):
            # largest is 32 bit int
            zero = one = 0
            # x << y: Returns x with the bits shifted to the left by y places 
            # (and new bits on the right-hand-side are zeros). This is the same as multiplying x by 2**y.
            mask = 1 << i  # 100000 -> i 0's on right (left are all zeros by default)
            for num in nums:
                # ‘&’ is a bitwise operator in Python that acts on bits and performs bit by bit operation
                # https://wiki.python.org/moin/BitwiseOperators
                # Does a "bitwise and". Each bit of the output is 1 if the corresponding bit of x AND of y is 1, otherwise it's 0.
                if mask & num: one += 1  # if true, means the ith position of num is 1, else 0
                else: zero += 1    
            ans += one * zero        
        return ans  

"""939. Minimum Area Rectangle
You are given an array of points in the X-Y plane points where points[i] = [xi, yi].
Return the minimum area of a rectangle formed from these points, with sides parallel to the X and Y axes. If there is not any such rectangle, return 0.

Example 1:
Input: [[1,1],[1,3],[3,1],[3,3],[2,2]]
Output: 4

Example 2:
Input: points = [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]
Output: 2
"""
# use hashtable to save points
# for each two points, if they do not share x or y, then check whether other two poitns are in hast
# update size
class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        m = defaultdict(set)
        for p in points:
            # m: x_axis -> set of y_axis
            m[p[0]].add(p[1])
        n = len(points)
        res = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                if points[i][0] != points[j][0] and points[i][1] != points[j][1]:
                    # check whether (points[i][0], points[j][1])
                    if points[j][1] in m[points[i][0]] and points[i][1] in m[points[j][0]]:
                        res = min(res, abs(points[i][0] - points[j][0]) * abs(points[i][1] - points[j][1]))
        
        return res if res < float('inf') else 0


"""[LeetCode] 963. Minimum Area Rectangle II 面积最小的矩形之二
Given a set of points in the xy-plane, determine the minimum area of any rectangle formed from these points, 
with sides not necessarily parallel to the x and y axes. ( can be a diamond)

If there isn't any rectangle, return 0.

Input: [[1,2],[2,1],[1,0],[0,1]]
Output: 2.00000
Explanation: The minimum area rectangle occurs at [1,2],[2,1],[1,0],[0,1], with an area of 2.

只要找到了两组对顶点，它们的中心重合，并且表示的对角线长度相等，则一定可以组成矩形。基于这种思想，可以遍历任意两个顶点，
求出它们之间的距离，和中心点的坐标，将这两个信息组成一个字符串，建立和顶点在数组中位置之间的映射，这样能组成矩形的点就被归类到一起了。
接下来就是遍历这个 HashMap 了，只能取出两组顶点及更多的地方，开始遍历，分别通过顶点的坐标算出两条边的长度，然后相乘用来更新结果 res 即可
"""
"""
class Solution {
public:
    double minAreaFreeRect(vector<vector<int>>& points) {
        int n = points.size();
        if (n < 4) return 0.0;
        double res = DBL_MAX;
        unordered_map<string, vector<vector<int>>> m;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                long dist = getLength(points[i], points[j]);
                double centerX = (points[i][0] + points[j][0]) / 2.0;
                double centerY = (points[i][1] + points[j][1]) / 2.0;
                string key = to_string(dist) + "_" + to_string(centerX) + "_" + to_string(centerY);
                m[key].push_back({i, j});
            }
        }
        for (auto &a : m) {
            vector<vector<int>> vec = a.second;
            if (vec.size() < 2) continue;
            for (int i = 0; i < vec.size(); ++i) {
                for (int j = i + 1; j < vec.size(); ++j) {
                    int p1 = vec[i][0], p2 = vec[j][0], p3 = vec[j][1];
                    double len1 = sqrt(getLength(points[p1], points[p2]));
                    double len2 = sqrt(getLength(points[p1], points[p3]));
                    res = min(res, len1 * len2);
                }
            }
        }
        return res == DBL_MAX ? 0.0 : res;
    }
    long getLength(vector<int>& pt1, vector<int>& pt2) {
        // triangle rule: x^2+y^2
        return (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]);
    }
};
"""

"""670. Maximum Swap
You are given an integer num. You can swap two digits at most once to get the maximum valued number.
Return the maximum valued number you can get.
Input: num = 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.

Input: num = 9973
Output: 9973
Explanation: No swap.

72736  -> 77236
77632
"""
# NlogN
class Solution: 
    def maximumSwap(self, num: int) -> int:
        num_str = list(str(num))
        sorted_num = sorted(num_str, reverse=True)
        i = 0
        # find the first swap idx
        while i < len(num_str):
            if num_str[i] != sorted_num[i]:
                break
            i += 1
        
        if i == len(num_str):
            return num
        # second swap is the lowest digit in num_str, = sorted_num[i]
        for j in range(len(num_str))[::-1]:
            if num_str[j] == sorted_num[i]:
                break
        num_str[i], num_str[j] = num_str[j], num_str[i]
        return int(''.join(num_str))

# N
"""
只关注两个需要交换的位置即可，即高位上的小数字和低位上的大数字，分别用 pos1 和 pos2 指向其位置，均初始化为 -1，然后用一个指针 mx 指向低位最大数字的位置，
初始化为 n-1，然后从倒数第二个数字开始往前遍历，假如 str[i] 小于 str[mx]，说明此时高位上的数字小于低位上的数字，
是一对儿潜在可以交换的对象（但并不保证上最优解），此时将 pos1 和 pos2 分别赋值为 i 和 mx；若 str[i] 大于 str[mx]，
说明此时 str[mx] 不是低位最大数，将 mx 更新为 i。循环结束后，若 pos1 不为 -1，说明此时找到了可以交换的对象，而且找到的一定是最优解，直接交换即可，
class Solution {
public:
    int maximumSwap(int num) {
        string str = to_string(num);
        int n = str.size(), mx = n - 1, pos1 = -1, pos2 = -1;
        for (int i = n - 2; i >= 0; --i) {
            if (str[i] < str[mx]) {
                pos1 = i;
                pos2 = mx;
            } else if (str[i] > str[mx]) {
                mx = i;
            }
        }
        if (pos1 != -1) swap(str[pos1], str[pos2]);
        return stoi(str);
    }
};
"""

"""[LeetCode] 708. Insert into a Cyclic Sorted List 在循环有序的链表中插入结点
Given a node from a Circular Linked List which is sorted in ascending order, write a function to 
insert a value insertVal into the list such that it remains a sorted circular list. 
The given node can be a reference to any single node in the list, and may not be necessarily 
the smallest value in the circular list.

If there are multiple suitable places for insertion, you may choose any place to insert the new value. 
After the insertion, the circular list should remain sorted.
3 -> 4 -> 1 -> 3 (head again)
3 -> 4 -> 1 -> 2 -> 3 (head again)
Input: head = [3,4,1], insertVal = 2
Output: [3,4,1,2]
"""
class Solution:
    def insert(self, head, insertVal):
        if head is None:
            head = Node(insertVal)
            head.next = head  # point to itself
            return head
        pre = head; curr = head.next
        # 
        while curr != head: # loop the linkedList for once
            if pre.val <= insertVal <= curr.val:
                # pre.next = val, val.next = curr
                break
            if pre.val > curr.val and (pre.val <= insertVal or curr >= insertVal):
                # means pre is largest and curr is smallest
                # 3 -> 4(pre) -> 1(curr) -> 3(back), and insertVal = 6 or 0
                # pre.next = val, val.next = curr
                break
            pre = curr
            curr = curr.next
            # if 4 -> 4 -> 4 (back), then insertval can be anywhere
        pre.next = Node(insertVal)
        pre.next.next = curr
        return head


"""766. Toeplitz Matrix
Given an m x n matrix, return true if the matrix is Toeplitz. Otherwise, return false.

A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same elements.

Input: matrix = [[1,2,3,4],[5,1,2,3],[9,5,1,2]]
Output: true
Explanation:
In the above grid, the diagonals are:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]".
In each diagonal all elements are the same, so the answer is True.

Follow up:
What if the matrix is stored on disk, and the memory is limited such that you can only load at most one 
row of the matrix into the memory at once?
What if the matrix is so large that you can only load up a partial row into the memory at once?
"""
class Solution:
    def isToeplitzMatrix(self, matrix):
        for i in range(len(matrix) - 1):
            for j in range(len(matrix[0]) - 1):
                if matrix[i][j] != matrix[i + 1][j + 1]:
                    return False
        return True

# follow up: we just need to save prev and curr rows, compare prev[:-1] == curr[:-1]
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        if not matrix:
            return False
        n_rows, n_cols = len(matrix), len(matrix[0])
        prev = matrix[0][:-1]  # need previous row up to last element
        for i in range(1, n_rows):
            if prev != matrix[i][1:]:
                return False
            prev = matrix[i][:-1]
        return True

"""986. Interval List Intersections
You are given two lists of closed intervals, firstList and secondList, 
where firstList[i] = [starti, endi] and secondList[j] = [startj, endj]. 
Each list of intervals is pairwise disjoint and in sorted order

Return the intersection of these two interval lists.

Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
"""
class Solution:
    def intervalIntersection(self, A, B):
        i = 0; j = 0
        result = []
        while i < len(A) and j < len(B):
            a_start, a_end = A[i]
            b_start, b_end = B[j]
            if a_start <= b_end and b_start <= a_end:                       # Criss-cross lock
                result.append([max(a_start, b_start), min(a_end, b_end)])   # Squeezing
            if a_end < b_end:         # Exhausted this range in A
                i += 1               # Point to next range in A
            elif a_end > b_end:                      # Exhausted this range in B
                j += 1               # Point to next range in B
            else: # else both move next, can be merged into any of previous condition
                i += 1
                j += 1
        return result


"""1011. Capacity To Ship Packages Within D Days
A conveyor belt has packages that must be shipped from one port to another within days days.

The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship 
with packages on the conveyor belt (in the order given by weights). 
We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the 
conveyor belt being shipped within days days. Note that the cargo must be shipped in the order given

Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10
"""
# binary search: nlogn
class Solution:
    def shipWithinDays(self, weights, days):
        low, high = max(weights), sum(weights)
        while low < high:
            # guess the capacity of ship
            mid = (low+high)//2
            num_days = self.get_days(weights, mid)
            # we need too many ships, so we need to increase capacity to reduce num of ships needed
            if num_days > days:
                low = mid+1
            # we are able to ship with good num of ships, but we still need to find the optimal max capacity
            else:
                high = mid
        return low
    
    def get_days(self, weights, ship_size):
        cur_cap = 0 # loaded capacity of current ship
        num_days = 1 # number of days needed
        #----simulating loading the weight to ship one by one----#
        for w in weights:
            cur_cap += w
            if cur_cap > ship_size: # current ship meets its capacity
                cur_cap = w
                num_days += 1
        #---------------simulation ends--------------------------#
        return num_days


"""1868 - Product of Two Run-Length Encoded Arrays
encode = [[num1, repeat1], [num2, repeat2], ...]
Input: encoded1 = [[1,3],[2,3]], encoded2 = [[6,3],[3,3]]
Output: [[6,6]]

Explanation: encoded1 expands to [1,1,1,2,2,2] and encoded2 expands to [6,6,6,3,3,3].

Input: encoded1 = [[1,3],[2,1],[3,2]], encoded2 = [[2,3],[3,3]]
Output: [[2,3],[6,1],[9,2]]
"""
class Solution:
    def findRLEArray(self, encoded1, encoded2):
        res = [] 
        prevProduct = -1
        prevCount = 0
        i, j = 0, 0
        while i < len(encoded1) and j < len(encoded2):
            val1 = encoded1[i][0]
            val2 = encoded2[j][0]
            freq = min(encoded1[i][1], encoded2[j][1])
            # new result
            curProduct = val1 * val2
            if curProduct == prevProduct:
                prevCount += freq
            else:
                if prevCount > 0:
                    res.append([prevProduct, prevCount])
                prevProduct = curProduct
                prevCount = freq
            # update 
            encoded1[i][1] -= freq
            encoded2[j][1] -= freq
            if encoded1[i][1] == 0:
                i += 1
            if encoded2[j][1] == 0:
                j += 1
        
        res.append([prevProduct, prevCount])
        return res

sol=Solution()
sol.findRLEArray([[1,3],[2,3]], [[6,3],[3,3]])           
sol.findRLEArray([[1,3],[2,1],[3,2]], [[2,3],[3,3]])           


"""138. Copy List with Random Pointer
Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, 
where each new node has its value set to the value of its corresponding original node. 
Both the next and random pointer of the new nodes should point to new nodes in the copied 
list such that the pointers in the original list and copied list represent the same list state. 
None of the pointers in the new list should point to nodes in the original list.

Return the head of the copied linked list

Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
"""
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head):
        m = dict()
        new_head = self.copyhelper(head, m)
        return new_head
    
    def copyhelper(self, node, m):
        if node is None:
            return None
        if node in m:
            return m[node]
        m[node] = Node(x=node.val)
        m[node].next = self.copyhelper(node.next, m)
        m[node].random = self.copyhelper(node.random, m)
        return m[node]

# solution 2: no extra space
class Solution:
    def copyRandomList1(self, head):
        if not head:
            return 
        # copy nodes
        cur = head
        while cur:
            nxt = cur.next
            cur.next = Node(cur.val)
            cur.next.next = nxt
            cur = nxt
        # copy random pointers
        cur = head
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            cur = cur.next.next
        # separate two parts
        new_head = cur = head.next
        while cur.next:
            head.next = cur.next
            head = head.next
            cur.next = head.next
            cur = cur.next
        head.next = None
        return new_head


"""LeetCode 616 - Add Bold Tag in String

Given a string s and a list of strings dict, you need to add a closed pair of bold tag <b> and </b> to 
wrap the substrings in s that exist in dict. If two such substrings overlap, you need to wrap them together 
by only one pair of closed bold tag. Also, if two substrings wrapped by bold tags are consecutive, 
you need to combine them.

Input:  s = "abcxyz123", dict = ["abc","123"]
Output: "<b>abc</b>xyz<b>123</b>"

Input: s = "aaabbcc", dict = ["aaa","aab","bc"]
Output: "<b>aaabbc</b>c"

hint: 使用一个数组bold，标记所有需要加粗的位置为true，初始化所有为false。我们首先要判断每个单词word是否是S的子串，
判断的方法就是逐个字符比较，遍历字符串S，找到和word首字符相等的位置，并且比较随后和word等长的子串，如果完全相同，
则将子串所有的位置在bold上比较为true。等我们知道了所有需要加粗的位置后，我们就可以来生成结果res了，
我们遍历bold数组，如果当前位置是true的话，表示需要加粗
"""
# O(nd) 
"""
class Solution {
public:
    string boldWords(vector<string>& words, string S) {
        int n = S.size();
        string res = "";
        vector<bool> bold(n, false);      
        for (string word : words) {
            int len = word.size();
            for (int i = 0; i <= n - len; ++i) {
                if (S[i] == word[0] && S.substr(i, len) == word) {
                    for (int j = i; j < i + len; ++j) bold[j] = true;
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (bold[i]) {
                if (i == 0 || !bold[i - 1]) res += "<b>";
                res.push_back(S[i]);
                if (i == n - 1 || !bold[i + 1]) res += "</b>";
            } else {
                res.push_back(S[i]);
            }
        }
        return res;
    }
};
"""

"""398. Random Pick Index
Given an integer array nums with possible duplicates, randomly output the index of a given target number. 
You can assume that the given target number must exist in the array.

Implement the Solution class:

Solution(int[] nums) Initializes the object with the array nums.
int pick(int target) Picks a random index i from nums where nums[i] == target. If there are multiple 
valid i's, then each index should have an equal probability of returning.

Input
["Solution", "pick", "pick", "pick"]
[[[1, 2, 3, 3, 3]], [3], [1], [3]]
Output
[null, 4, 0, 2]

The array size can be very large. Solution that uses too much extra space will not pass the judge.

Hint
这道题指明了我们不能用太多的空间，那么省空间的随机方法只有水塘抽样Reservoir Sampling了，
LeetCode之前有过两道需要用这种方法的题目Shuffle an Array和Linked List Random Node。那么如果了解了水塘抽样，
这道题就不算一道难题了，我们定义两个变量，计数器cnt和返回结果res，我们遍历整个数组，如果数组的值不等于target，
直接跳过；如果等于target，计数器加1，然后我们在[0,cnt)范围内随机生成一个数字，如果这个数字是0，我们将res赋值为i即可
"""
import random
class Solution:
    def __init__(self, nums):
        self.nums = nums

    def pick(self, target: int) -> int:
        cnt = 0
        res = -1
        for i, num in enumerate(self.nums):
            if num != target:
                continue
            else:
                rand = random.uniform(0, 1)
                if rand <= 1/(cnt+1):
                    res = i
                cnt += 1
        return res

"""778. Swim in Rising Water
You are given an n x n integer matrix grid where each value grid[i][j] represents the elevation at that point (i, j).

The rain starts to fall. At time t, the depth of the water everywhere is t. You can swim from a square to another 4-directionally 
adjacent square if and only if the elevation of both squares individually are at most t. You can swim infinite distances in zero 
time. Of course, you must stay within the boundaries of the grid during your swim.

Return the least time until you can reach the bottom right square (n - 1, n - 1) if you start at the top left square (0, 0).
"""
# solution 1: BFS + binary search (n^2*log(max-min))
# https://leetcode.com/problems/swim-in-rising-water/discuss/113765/PythonC%2B%2B-Binary-Search

# solution 2: use heap (n^2 * logn)
# https://leetcode.com/problems/swim-in-rising-water/discuss/1284843/Python-2-solutions%3A-Union-FindHeap-explained


"""1382. Balance a Binary Search Tree
Given the root of a binary search tree, return a balanced binary search tree with the same node values. 
If there is more than one answer, return any of them.

A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than 1.

Input: root = [1,null,2,null,3,null,4,null,null]
Output: [2,1,3,null,null,null,4]
Explanation: This is not the only correct answer, [3,1,4,null,2] is also correct.
"""
# similar to 108, first in-order traverse the tree and save the value in a array, then 
# create balanced BST using the array. 
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        nodes = []
        def in_order_traverse(root):
            if root is None:return
            in_order_traverse(root.left)
            nodes.append(root)
            in_order_traverse(root.right)
        
        def build_balanced_tree(left, right):
            if left>right:return None
            mid = (left+right)//2
            root = nodes[mid]
            root.left = build_balanced_tree(left, mid-1)
            root.right = build_balanced_tree(mid+1, right)
            return root
        in_order_traverse(root)
        return build_balanced_tree(0, len(nodes)-1)  


"""1344. Angle Between Hands of a Clock
Given two numbers, hour and minutes, return the smaller angle (in degrees) formed between the hour and the minute hand.

Input: hour = 12, minutes = 30
Output: 165
"""
class Solution:
    def angleClock(self, hour: int, minutes: int):
        min_loc = 6 * minutes
        hour_loc = (30 * hour + 0.5 * minutes) % 360
        diff = abs(hour_loc - min_loc)
        return diff if diff <= 180 else 360 - diff

"""489. Robot Room Cleaner
Given a robot cleaner in a room modeled as a grid. Each cell in the grid can be empty or blocked.
The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 90 degrees.
When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it stays on the current cell.

Design an algorithm to clean the entire room using only the 4 given APIs shown below.
hint: can only use DFS considering robot 's path is continuous, need to record curr direction (cur)
https://grandyang.com/leetcode/489/

class Solution {
public:
    vector<vector<int>> dirs{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    void cleanRoom(Robot& robot) {
        unordered_set<string> visited;
        helper(robot, 0, 0, 0, visited);
    }
    void helper(Robot& robot, int x, int y, int dir, unordered_set<string>& visited) {
        robot.clean();
        visited.insert(to_string(x) + "-" + to_string(y));
        for (int i = 0; i < 4; ++i) {
            int cur = (i + dir) % 4, newX = x + dirs[cur][0], newY = y + dirs[cur][1];
            if (!visited.count(to_string(newX) + "-" + to_string(newY)) && robot.move()) {
                helper(robot, newX, newY, cur, visited);
                robot.turnRight();
                robot.turnRight();
                robot.move();
                robot.turnLeft();
                robot.turnLeft();
            }
            robot.turnRight();
        }
    }
};
"""

"""463. Island Perimeter
You are given row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water.

Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
Output: 16
Explanation: The perimeter is the 16 yellow stripes in the image above.
"""
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    curr_new_sides = 4
                    if i > 0 and grid[i-1][j] == 1:
                        curr_new_sides -= 1
                    if j > 0 and grid[i][j-1] == 1:
                        curr_new_sides -= 1
                    if i < len(grid)-1 and grid[i+1][j] == 1:
                        curr_new_sides -= 1
                    if j < len(grid[0])-1 and grid[i][j+1] == 1:
                        curr_new_sides -= 1
                    res += curr_new_sides
        return res


"""234. Palindrome Linked List
Given the head of a singly linked list, return true if it is a palindrome.

Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false

Follow up: Could you do it in O(n) time and O(1) space?

我们可以在找到中点后，将后半段的链表翻转一下，这样我们就可以按照回文的顺序比较了
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head) -> bool:
        # pre = ListNode(next=head)
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            # last = slow
            slow = slow.next
        
        # reverse slow.next and then compare
        node1 = head
        node2 = self.reverse_linkedlist(slow)
        
        while node2:
            if node1.val != node2.val:
                return False
            node1 = node1.next
            node2 = node2.next
        return True

    def reverse_linkedlist(self, node):
        prev = None
        curr = node
        while curr:
            tmp = curr
            curr = curr.next
            tmp.next = prev
            prev = tmp
        return prev


"""109. Convert Sorted List to Binary Search Tree
Given the head of a singly linked list where elements are sorted in ascending order, convert 
it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth 
of the two subtrees of every node never differ by more than 1.

Given the sorted linked list: [-10,-3,0,5,9], One possible answer is: [0,-3,9,-10,null,5], 
which represents the following height balanced BST:
      0
     / \
   -3   9
   /   /
 -10  5
"""
class Solution:
    def sortedListToBST(self, head):
        if not head:
            return None
        if head.next is None:
            return TreeNode(val=head.val)
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            last = slow  # prev of slow
            slow = slow.next
        root = TreeNode(val=slow.val)
        root.right = self.sortedListToBST(slow.next)
        last.next = None  # need to cut last o.w. dead loop
        root.left = self.sortedListToBST(head)

        return root

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
# Solution 1: DFS + memo
class Solution:
    def canPartition(self, nums) -> bool:
        target = sum(nums) / 2
        if target != int(target):
            return False
        nums.sort()
        memo=dict()
        return self.dfs(nums, 0, target, memo)
    
    def dfs(self, nums, start_idx, target, memo):
        if target in memo:
            return memo[target]
        for i in range(start_idx, len(nums)):
            if nums[i] == target:
                return True
            if nums[i] < target:
                if self.dfs(nums, i+1, target-nums[i], memo):
                    return True
            else:
                break
        memo[target] = False
        return False

sol=Solution()
sol.canPartition([1, 5, 11, 5])

# DP: kranpack : much slower than previous
# dp[i][j] := whether using nums[:i] can find a subset sum to j
# dp[n][target]
# dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
class Solution:
    def canPartition(self, nums) -> bool:
        target = sum(nums) / 2
        if target != int(target):
            return False
        target = int(target)
        dp = [[False for _ in range(target+1)] for _ in range(len(nums)+1)]
        for i in range(len(nums)+1):
            dp[i][0] = True
        for j in range(target+1):
            dp[0][j] = False
        
        for i in range(1, len(nums)+1):
            for j in range(1, target+1):
                dp[i][j] = dp[i-1][j]
                if j-nums[i-1] >= 0:
                    dp[i][j] = dp[i][j] or dp[i-1][j-nums[i-1]]
        
        return dp[-1][-1]


"""865. Smallest Subtree with all the Deepest Nodes
Given the root of a binary tree, the depth of each node is the shortest distance to the root.
Return the smallest subtree such that it contains all the deepest nodes in the original tree.

A node is called the deepest if it has the largest depth possible among any node in the entire tree.
The subtree of a node is a tree consisting of that node, plus the set of all descendants of that node.
Input: {2,3,5,4,1}
    2
   / \
  3   5
 / \
4   1
Output: {3,4,1}
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode):
        ans, ans_depth = self.getDepth(root)
        return ans
        
    
    def getDepth(self, node):
        # return depth and node
        if node is None:
            return None, 0
        left, left_depth = self.getDepth(node.left)
        right, right_depth = self.getDepth(node.right)
        if left_depth == right_depth:
            return node, 1 + left_depth
        if left_depth > right_depth:
            return left, 1 + left_depth
        else:
            return right, 1 + right_depth


""" 
############################################################################
Linkedin
############################################################################
"""

"""127. Word Ladder
"""

""" 50. Pow(x, n)
Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).

n is integer (can be negative)
runtime: O(logn)

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25
"""
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n % 2 == 0:
            ans =  self.myPow(x, n / 2)
            return ans * ans
        return x * self.myPow(x, n-1)


"""360. Sort Transformed Array
Given a sorted array of integers nums and integer values a, b and c. Apply a function of the form f(x) = ax2 + bx + c to each element x in the array.
The returned array must be in sorted order.
Expected time complexity: O(n)
Example:
nums = [-4, -2, 2, 4], a = 1, b = 3, c = 5,
Result: [3, 9, 15, 33]
nums = [-4, -2, 2, 4], a = -1, b = 3, c = 5
Result: [-23, -5, 1, 7]
"""
# if a=0
# if not, l, r = 0, n-1, based on a>0 or a<0 to narrow down l,r. while l<r
class Solution:
    def sortTransformedArr(self, nums, a, b, c):
        f_nums = [self.f(x, a, b, c) for x in nums]
        if a == 0:
            if b >= 0:
                return f_nums
            else:
                return f_nums[::-1]
        
        l, r = 0, len(nums) - 1  # left, right
        res = []
        while l < r:
            if f_nums[l] >= f_nums[r]:
                if a > 0: # decrease
                    res.append(f_nums[l])
                    l += 1
                else: # increase
                    res.append(f_nums[r])
                    r -= 1
            else:
                if a < 0:
                    res.append(f_nums[l])
                    l += 1
                else:
                    res.append(f_nums[r])
                    r -= 1
        res.append(f_nums[l])
        return res if a < 0 else res[::-1]

    def f(self, x, a, b, c):
        return a * x**2 + b * x + c

sol = Solution()
sol.sortTransformedArr(nums=[-4, -2, 2, 4], a = 1, b = 3, c = 5)
sol.sortTransformedArr(nums=[-4, -2, 2, 4], a = -1, b = 3, c = 5)


"""152. Maximum Product Subarray
Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.
The test cases are generated so that the answer will fit in a 32-bit integer.
A subarray is a contiguous subsequence of the array.

Example 1:
Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Example 2:
Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
"""
nums = [-4,-3,-2]
class Solution:
    def maxProduct(self, nums) -> int:
        curr_max = nums[0]
        curr_min = nums[0]  # smallest number ends with nums[i]
        res = nums[0]
        for num in nums[1:]:
            if num > 0:
                curr_max = max(curr_max * num, num)
                curr_min = min(curr_min * num, num)
            elif num < 0:
                curr_max_next = max(curr_min * num, num)
                curr_min = min(curr_max * num, num)
                curr_max = curr_max_next
            else:
                curr_max = 0
                curr_min = 0
            res = max(res, curr_max)
        return res


""" 634. Find the Derangement of An Array
In combinatorial mathematics, a derangement is a permutation of the elements of a set, such that no element appears in its original position.
There's originally an array consisting of n integers from 1 to n in ascending order, you need to find the number of derangement it can generate.
Also, since the answer may be very large, you should return the output mod 109 + 7.

Example 1:
Input: 3
Output: 2
Explanation: The original array is [1,2,3]. The two derangements are [2,3,1] and [3,1,2].

我们来想n = 4时该怎么求，我们假设把4排在了第k位，这里我们就让k = 3吧，那么我们就把4放到了3的位置，变成了：
x x 4 x
我们看被4占了位置的3，应该放到哪里，这里分两种情况，如果3放到了4的位置，那么有：
x x 4 3
那么此时4和3的位置都确定了，实际上只用排1和2了，那么就相当于只排1和2，就是dp[2]的值，是已知的。那么再来看第二种情况，3不在4的位置，那么此时我们把4去掉的话，就又变成了：
x x x
这里3不能放在第3个x的位置，在去掉4之前，这里是移动4之前的4的位置，那么实际上这又变成了排1，2，3的情况了，就是dp[3]的值。
再回到最开始我们选k的时候，我们当时选了k = 3，其实k可以等于1,2,3，也就是有三种情况，所以dp[4] = 3 * (dp[3] + dp[2])。
那么递推公式也就出来了：
dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])
"""
def findDerangement(n):
    dp = [0] * (n+1)
    dp[0] = 0
    dp[1] = 0
    dp[2] = 1
    for i in range(3, n+1):
        dp[i] = (i-1) * (dp[i-1] + dp[i-2]) % (10e+9 + 7)
    return dp[-1]


"""[LeetCode] 254. Factor Combinations 因子组合
Numbers can be regarded as product of its factors. For example,
8 = 2 x 2 x 2;
  = 2 x 4.
Write a function that takes an integer n and return all possible combinations of its factors.
Note:
You may assume that n is always positive. Factors should be greater than 1 and less than n.

Example 3:
Input: 
12
Output:
[
  [2, 6],
  [2, 2, 3],
  [3, 4]
]

class Solution {
public:
    vector<vector<int>> getFactors(int n) {
        vector<vector<int>> res;
        helper(n, 2, {}, res);
        return res;
    }
    void helper(int n, int start, vector<int> out, vector<vector<int>>& res) {
        if (n == 1) {
            if (out.size() > 1) res.push_back(out);
            return;
        }
        for (int i = start; i <= int(sqrt(n)); ++i) { // add sqrt
            if (n % i != 0) continue;
            out.push_back(i);
            helper(n / i, i, out, res);
            out.pop_back();
        }
    }
};
"""

"""[LeetCode] 156. Binary Tree Upside Down
Given a binary tree where all the right nodes are either leaf nodes with a sibling (a left node that shares 
the same parent node) or empty, flip it upside down and turn it into a tree where the original right nodes 
turned into left leaf nodes. Return the new root.

For example:
Given a binary tree {1,2,3,4,5},
    1
   / \
  2   3
 / \
4   5
return the root of the binary tree [4,5,2,#,#,3,1].
   4
  / \
 5   2
    / \
   3   1  
confused what "{1,#,2,3}" means? > read more on how binary tree is serialized on OJ.

The serialization of a binary tree follows a level order traversal, where '#' signifies a path terminator where no node exists below.
Here's an example:
   1
  / \
 2   3
    /
   4
    \
     5
The above binary tree is serialized as [1,2,3,#,#,4,#,#,5].

Follow up: how do you validate whether input meets requirement or not? 

对于一个根节点来说，目标是将其左子节点变为根节点，右子节点变为左子节点，原根节点变为右子节点，首先判断这个根节点是否存在，
且其有没有左子节点，如果不满足这两个条件的话，直接返回即可，不需要翻转操作。那么不停的对左子节点调用递归函数，
直到到达最左子节点开始翻转，翻转好最左子节点后，开始回到上一个左子节点继续翻转即可，直至翻转完整棵树
"""
def upsideDownBinaryTree(root):
    if root is None or root.left is None:
        return root
    left = root.left
    right = root.right
    upsideDownLeft = upsideDownBinaryTree(root.left)
    # right becomes left, left becomes root, and root becomes right
    new_root = upsideDownLeft
    left.left = right
    left.right = root
    # ! root is leaf now
    root.left=None
    root.right=None
    return new_root


"""[LeetCode] 57. Insert Interval 插入区间
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
You may assume that the intervals were initially sorted according to their start times.
Example 1:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:
[[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]
[3,5],[6,7],[8,10]

# O(n)
"""
intervals = [[1,3],[6,9]]; newInterval = [2,5]
class Solution:
    def insert(self, intervals, newInterval):
        res = []
        i = 0
        while i < len(intervals):
            if intervals[i][1] < newInterval[0]:
                res.append(intervals[i])
                i += 1
            else:
                break
        while i < len(intervals):
            if intervals[i][0] <= newInterval[1]: # overlap
                merged_interval = self.mergeInterval(intervals[i], newInterval)
                newInterval = merged_interval
                i += 1
            else:
                break
        res.append(newInterval)
        while i < len(intervals):
            res.append(intervals[i])
            i += 1
        return res
    
    def mergeInterval(self, interval1, interval2):
        return [min(interval1[0], interval2[0]), max(interval1[1], interval2[1])]


sol = Solution()
sol.insert(intervals = [[1,3],[6,9]], newInterval = [2,5])
sol.insert(intervals = [[1,5]], newInterval = [2,3])


class Solution2:
    # another way is to use inserted bool to decide whether insertion happens or not
    def insert(self, intervals, newInterval):
        res = []
        newStart, newEnd = newInterval
        for i, interval in enumerate(intervals):
            start, end = interval
            if end < newStart:
                res.append(interval)
            elif newEnd < start:
                res.append([newStart, newEnd])
                return res + intervals[i:]  # can return earlier
            else:  # overlap case
                newStart = min(newStart, start)
                newEnd = max(newEnd, end)
                
        res.append([newStart, newEnd])
        return res

# follow up: what if you need to remove an interval instead of insert? 
# same except for overlap case, if overlap, 
# 1) if end > newStart: res.append([start, newStart])
# 2) if start < newEnd: res.append([newEnd, end])

"""381. Insert Delete GetRandom O(1) - Duplicates allowed
RandomizedCollection is a data structure that contains a collection of numbers, possibly duplicates (i.e., a multiset).
 It should support inserting and removing specific elements and also removing a random element.

Implement the RandomizedCollection class:

RandomizedCollection() Initializes the empty RandomizedCollection object.
bool insert(int val) Inserts an item val into the multiset, even if the item is already present. Returns true if the item is not present, false otherwise.

bool remove(int val) Removes an item val from the multiset if present. Returns true if the item is present, false otherwise. 
Note that if val has multiple occurrences in the multiset, we only remove one of them.

int getRandom() Returns a random element from the current multiset of elements. The probability of each element being returned 
is linearly related to the number of same values the multiset contains.
You must implement the functions of the class such that each function works on average O(1) time complexity.
"""
# use self.d = defaultdict(set) to make sure O(1) to remove
import random
class RandomizedCollection:
    def __init__(self):
        self.vector = []
        self.d = defaultdict(set)
        self.size = 0

    def insert(self, val: int) -> bool:
        res = True
        if len(self.d[val]) > 0:
            res = False
        self.vector.append(val)
        self.d[val].add(self.size)
        self.size += 1
        return res

    def remove(self, val: int) -> bool:
        if len(self.d[val]) == 0:
            return False
        # swap last with the val to delete
        val_idx = self.d[val].pop()
        self.vector[val_idx], self.vector[-1] = self.vector[-1], self.vector[val_idx]
        self.d[self.vector[val_idx]].add(val_idx)
        self.d[self.vector[val_idx]].remove(self.size-1)  # self.size-1 must be in self.d[self.vector[val_idx]]
        self.vector.pop()
        self.size -= 1
        return True
        
    def getRandom(self) -> int:
        return random.choice(self.vector)


"""237. Delete Node in a Linked List
Write a function to delete a node in a singly-linked list. You will not be given access to the head of the list, 
instead you will be given access to the node to be deleted directly.
It is guaranteed that the node to be deleted is not a tail node in the list.
"""

def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next


"""[LeetCode] 311. Sparse Matrix Multiplication
Given two sparse matrices A and B, return the result of AB.
You may assume that A's column number is equal to B's row number.

Example
A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]

B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]

     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                  | 0 0 1 |
"""
# How do I use the information that the matrices are sparse? Instead of iterating through the product matrix, 
# I can iterate through A, and add the contribution of each number to the result matrix. If A[i][j] == 0, I can just skip the calculation.
"""
class Solution {
    public int[][] multiply(int[][] A, int[][] B) {
        int ARow = A.length;
        int AColumn = A[0].length;
        int BRow = B.length;
        int BColumn = B[0].length;
        
        int[][] product = new int[ARow][BColumn];
        for (int i = 0; i < ARow; i++) {
            for (int j = 0; j < AColumn; j++) {
                if (A[i][j] == 0) {
                    continue;
                }
                for (int k = 0; k < BColumn; k++) {
                    product[i][k] += A[i][j] * B[j][k];
                }
            }
        }
        return product;
    }
}
"""
# another idea is to use dict to save row for A and col for B
# then use the following to calculate product of two sparse vectors

def product(d1, d2):
    if d1['len'] ! =d2['len'] # needs to be re-designed
        return None
    if d1['len1'] > d2['len2']:
        res = 0
        for k, v in d2.items():
            if k in d1:
                res += d1[k] * d2[k]
    else:
        res = 0
        for k, v in d1.items():
            if k in d2:
                res += d1[k] * d2[k]
    return res

# Another way to reduce runtime but increase storage: 
Arow = [[(0, 1)], [(0, -1), (2, 3)]]  # A[i] means ith row, [[1, 0, 0], [-1, 0, 3]]
Bcol = [[(0, 7)], [], [(2, 1)]]  # B[k] means kth col, [[7, 0], [0, 0], [1, 0]]
colB = 3
C = [[0]*colB for _ in range(len(A))]
# C is not in sparse format, needs to convert
class Solution:
    def multiply(self, A, B, colB):
        # C = [[0]*colB for _ in range(len(A))]
        C = [[] for _ in range(len(A))]
        for i in range(len(A)):
            res = 0
            for k, val in A[i]:
                # C[i][j] = A[i][k] * B[k][j]
                for j, valb in B[k]:
                    res += val * valb  # C[i][j]
            if res != 0:
                C[i].append((j, res))  # rowi, colj
        return C


from copy import deepcopy
class SparseMatrix:
    def __init__(self, nrow, ncol, S):
        self.nrow = nrow
        self.ncol = ncol
        self.S = S
    
    def matmul(self, B):
        # S * B
        res = dict()
        for (i, k), val in self.S.items():
            for j in range(B.ncol):
                if (k, j) in B.S:
                    res[(i, j)] = res.get((i, j), 0) + self.S[(i, k)] * B.S[(k, j)]
        return SparseMatrix(self.nrow, B.ncol, res)

    def add(self, B):
        assert self.nrow==B.nrow and self.ncol==B.ncol
        res_dict = deepcopy(self.S)
        for (i, j), val in B.S.items():
            res_dict[(i, j)] = res_dict.get((i,j), 0) + val
        return SparseMatrix(self.nrow, self.ncol, res_dict)


matA = SparseMatrix(3, 1, {(2,0):5})
matB = SparseMatrix(1, 3, {(0,2):5})
matA.matmul(matB).S

"""68. Text Justification
Given an array of strings words and a width maxWidth, format the text such that each line has exactly 
maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. 
Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a 
line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified and no extra space is inserted between words.

Example 1:
Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
"""
class Solution:
    def fullJustify(self, words, maxWidth):
        res = []
        return self.JustifyRemWords(words, "", res, maxWidth)
    
    def JustifyRemWords(self, rem_words, out, res, maxWidth):
        if len(rem_words) == 0 and len(out) == 0:
            return res
        if len(rem_words) == 0 or \
            (len(out) > 0 and len(out) + len(rem_words[0]) + 1 > maxWidth) or \
            (len(out) == 0 and len(rem_words[0]) > maxWidth):
            # cannot put next word in the same line
            aligned_out = self.alignLine(out, maxWidth, last=(len(rem_words) == 0))
            res.append(aligned_out)
            return self.JustifyRemWords(rem_words, "", res, maxWidth)
        else:
            out = out + " " + rem_words[0]
            return self.JustifyRemWords(rem_words[1:], out.strip(), res, maxWidth)
    
    def alignLine(self, line, maxWidth, last=False):
        words_in_out = line.split(" ")
        if len(words_in_out) == 1:
            return words_in_out[0] + " "*(maxWidth - len(words_in_out[0]))
        num_spaces = maxWidth - sum([len(w) for w in words_in_out])
        if last: # if last line
            return line + " "*(maxWidth - len(line))
        num_spaces_btw = num_spaces // (len(words_in_out) - 1)
        num_extra_spaces = num_spaces % (len(words_in_out) - 1)
        output = ""
        for i, w in enumerate(words_in_out[:-1]):
            if i < num_extra_spaces:
                output += w + " "*(num_spaces_btw+1)
            else:
                output += w + " "*(num_spaces_btw)
        return output + words_in_out[-1]

class Solution2:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        """
        Time complexity: O(m*n)
        Space complexity: O(m*n)
        :param words:
        :param maxWidth:
        :return:
        """
        def justify(line, width, max_width):
            """
            :param line: the actual words
            :param width: sum of length of all words (without overall_spaces_count)
            :param max_width: maximum allowed width
            :return:
            """
            overall_spaces_count = max_width - width
            words_count = len(line)
            if len(line) == 1:
                # if there is only word in line
                # just insert overall_spaces_count for the remainder of line
                return line[0] + ' ' * overall_spaces_count
            else:
                spaces_to_insert_between_words = (words_count - 1)
                # num_spaces_between_words_list[i] : tells you to insert num_spaces_between_words_list[i] spaces
                # after word on line[i]
                num_spaces_between_words_list = spaces_to_insert_between_words * [overall_spaces_count // spaces_to_insert_between_words]
                spaces_count_in_locations = overall_spaces_count % spaces_to_insert_between_words
                # distribute spaces via round robin to the left words
                for i in range(spaces_count_in_locations):
                    num_spaces_between_words_list[i] += 1
                aligned_words_list = []
                for i in range(spaces_to_insert_between_words):
                    # add the word
                    aligned_words_list.append(line[i])
                     # add the spaces to insert
                    aligned_words_list.append(num_spaces_between_words_list[i] * ' ')
                # just add the last word to the sentence
                aligned_words_list.append(line[-1])
                # join the alligned words list to form a justified line
                return ''.join(aligned_words_list)

        answer = []
        line, width = [], 0
        for word in words:
            if width + len(word) + len(line) <= maxWidth:
                # keep adding words until we can fill out maxWidth
                # width = sum of length of all words (without overall_spaces_count)
                # len(word) = length of current word
                # len(line) = number of overall_spaces_count to insert between words
                line.append(word)
                width += len(word)
            else:
                # justify the line and add it to result
                answer.append(justify(line, width, maxWidth))
                # reset new line and new width
                line, width = [word], len(word)
        remaining_spaces = maxWidth - width - len(line)
        answer.append(' '.join(line) + (remaining_spaces + 1) * ' ')
        return answer


"""149. Max Points on a Line
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, 
return the maximum number of points that lie on the same straight line.
Input: points = [[1,1],[2,2],[3,3]]
Output: 3

Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4
http://zxi.mytechroad.com/blog/geometry/leetcode-149-max-points-on-a-line/
Hint: for each point, 1) count duplicate points, 2) count lines by slope (same slope + same starting point -> same line)
for pi:
    for pj: (j>i)
        if pi==pj: dup++
        else: ++count[slope(pi, pj)]
    ans = max(ans, max(count) + dup)
"""
from collections import defaultdict
class Solution:
    def maxPoints(self, points):
        # count num of duplicates, use as base for each pt
        dup_m = self.count_dup(points)
        res = 0
        for i in range(len(points)):
            m = defaultdict(lambda: dup_m[tuple(points[i])])
            for j in range(i+1, len(points)):
                if points[i] == points[j]:
                    continue
                if points[i][0] == points[j][0]:
                    m['inf'] += 1
                else:
                    h = (points[j][1] - points[i][1]) / (points[j][0] - points[i][0])
                    m[h] += 1
            res = max(max(m.values()), res) if len(m) > 0 else max(res, dup_m[tuple(points[i])])
        return res
    
    def count_dup(self, points):
        m = defaultdict(lambda: 0)
        for point in points:
            m[tuple(point)] += 1
        return m

sol = Solution()
sol.maxPoints([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]])
sol.maxPoints([[1,1],[2,2],[3,3]])
sol.maxPoints([[1,1]])

class Solution:
    def maxPoints(self, points):
        l = len(points)
        m = 0
        for i in range(l):
            dic = dict()
            same = 1  #  itself
            for j in range(i+1, l):
                tx, ty = points[j][0], points[j][1]
                if tx == points[i][0] and ty == points[i][1]: 
                    same += 1
                    continue
                if points[i][0] == tx: slope = 'i'
                else:slope = (points[i][1]-ty) * 1.0 /(points[i][0]-tx)
                if slope not in dic: dic[slope] = 1
                else: dic[slope] += 1
            m = max(m, max(dic.values()) + same) if len(dic) > 0 else max(m, same)
        return m

points = [[1,1],[2,2],[3,3]]


"""7. Reverse Integer
Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to 
go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).
Example 1:
Input: x = 123
Output: 321

Example 2:
Input: x = -123
Output: -321
"""
class Solution:
    def reverse(self, x: int) -> int:
        revarr = []
        n = len(str(x))
        strint = ""
        if x == 0:
            return 0
        if x > 0:
            for i in range(n):
                revarr.append(x % 10)
                x = int(x/10)
            for i in range(n):
                strint += str(revarr[i])
            return int(strint) if int(strint) < (2**31 - 1) else 0
        else:
            return -1 * + self.reverse(-1*x)

class Solution:
    def reverse(self, x: int) -> int:
        if x == 0:
            return 0
        if x > 0:
            res = 0
            while x > 0:
                res = x % 10 + res * 10
                x = x // 10
            return res if res < (2**31 - 1) else 0
        else:
            return -1 * + self.reverse(-1*x)

sol=Solution()
sol.reverse(123)


"""
############################################################################
Microsoft
############################################################################
"""
""" Implement heap and quicksort
"""
# two versions of partition
class Solution:
    def partition1(self, arr, l, r):
        pivot = l
        i = l
        for j in range(l+1, r+1):
            # if 
            if arr[j] < arr[pivot]:
                i += 1 # can ensure arr[i] < arr[pivot]
                arr[i], arr[j] = arr[j], arr[i]
        arr[pivot], arr[i], = arr[i], arr[pivot]
        return i

    def partition(self, nums, left, right):
        # choose a pivot, put smaller elements on right, larger on left
        # return rank of pivot (put on final r position)
        pivot = nums[left]
        l = left + 1
        r = right
        while l <= r:
            if nums[l] > pivot and nums[r] < pivot:
                nums[l], nums[r] = nums[r], nums[l]
                l = l+1
                r = r-1
            if nums[l] <= pivot:
                l = l+1
            if nums[r] >= pivot:
                r = r-1
        nums[left], nums[r] = nums[r], nums[left]
        return r

    def qs(self, arr, l, r):
        if l < r:
            pivot = self.partition(arr, l, r)  # nums[l:r+1]
            self.qs(arr, l, pivot-1)
            self.qs(arr, pivot+1, r)
        return None


arr = [1,3,4,2,3,5,3, 10,1]
sol=Solution()
sol.qs(arr, 0, len(arr)-1)
arr = [10, 9, 8, 7, 5, 3, -1]
arr = [3,2,1]
"""merge sort
"""
from copy import deepcopy
def ms(arr, l, r):
    if r > l:
        mid = l + (r-l) // 2
        ms(arr, l, mid)
        ms(arr, mid+1, r)
        merge(arr, l, mid, r)
    return None

def merge(arr, l, m, r):
    left_merged_arr = deepcopy(arr[l:m+1])
    right_merged_arr = deepcopy(arr[m+1:r+1])
    k = l
    i = j = 0
    while i < m+1-l and j < r-m:
        if left_merged_arr[i] < right_merged_arr[j]:
            arr[k] = left_merged_arr[i]
            i += 1
        else:
            arr[k] = right_merged_arr[j]
            j += 1
        k += 1
    if i == m+1-l:
        while j < r-m:
            arr[k] = right_merged_arr[j]
            k += 1
            j += 1
    else:
        while i < m+1-l:
            arr[k] = left_merged_arr[i]
            k += 1
            i += 1
    return None

arr = [1,3,4,2,3,5,10,1]
ms(arr, 0, len(arr)-1)


"""545. Boundary of Binary Tree (Medium)
Given a binary tree, return the values of its boundary in anti-clockwise direction starting from root. 
Boundary includes left boundary, leaves, and right boundary in order without duplicate nodes.
Input:
    ____1_____
   /          \
  2            3
 / \          / 
4   5        6   
   / \      / \
  7   8    9  10  

Ouput:
[1,2,4,7,8,9,10,6,3]

Explanation:
The left boundary are node 1,2,4. (4 is the left-most node according to definition)
The leaves are node 4,7,8,9,10.
The right boundary are node 1,3,6,10. (10 is the right-most node).
So order them in anti-clockwise without duplicate nodes we have [1,2,4,7,8,9,10,6,3].
"""
# first find left b, then leaves, then right b
# https://xiaoguan.gitbooks.io/leetcode/content/LeetCode/545-boundary-of-binary-tree-medium.html


"""
############################################################################
Roku
############################################################################
"""
"""287. Find the Duplicate Number
Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums, and uses only constant extra space.
Can you find in O(nlogn) and further O(n)? 

Input: nums = [1,3,4,2,2]
Output: 2
"""
# nlogn: binary search : l,r=1,n, then mid = (l+r)/2, count # of nums < mid -> see whether duplicate happens at [1, mid]

# n
# http://bookshadow.com/weblog/2015/09/28/leetcode-find-duplicate-number/
# 快慢指针在之前的题目 Linked List Cycle II 中就有应用，这里应用的更加巧妙一些，由于题目限定了区间 [1,n]，
# 所以可以巧妙的利用坐标和数值之间相互转换，而由于重复数字的存在，那么一定会形成环，用快慢指针可以找到环并确定环的起始位置
"""
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0, t = 0;
        while (true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast) break;
        }
        while (true) {
            slow = nums[slow];
            t = nums[t];
            if (slow == t) break;
        }
        return slow;
    }
};
"""

"""442. Find All Duplicates in an Array
Given an integer array nums of length n where all the integers of nums are in the range [1, n] 
and each integer appears once or twice, return an array of all the integers that appears twice.

You must write an algorithm that runs in O(n) time and uses only constant extra space.

Input: nums = [4,3,2,7,8,2,3,1]
Output: [2,3]
"""
# use nums as a hashmap: abs(num)-1, and reverse the value
"""
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> ans;
        for(auto num : nums) {
            int index = abs(num)-1; // num >= 1
            if (nums[index] < 0)
                ans.push_back(index+1);
            nums[index]*=-1;
        }
        return ans;
    }
};
"""

"""1326. Minimum Number of Taps to Open to Water a Garden
There is a one-dimensional garden on the x-axis. The garden starts at the point 0 and ends at the point n. (i.e The length of the garden is n).

There are n + 1 taps located at points [0, 1, ..., n] in the garden.

Given an integer n and an integer array ranges of length n + 1 where ranges[i] (0-indexed) means the i-th tap can water the area [i - ranges[i], i + ranges[i]] if it was open.

Return the minimum number of taps that should be open to water the whole garden, If the garden cannot be watered return -1.

Input: n = 5, ranges = [3,4,1,1,0,0]
Output: 1
Explanation: The tap at point 0 can cover the interval [-3,3]
The tap at point 1 can cover the interval [-3,5]
The tap at point 2 can cover the interval [1,3]
The tap at point 3 can cover the interval [2,4]
The tap at point 4 can cover the interval [4,4]
The tap at point 5 can cover the interval [5,5]
Opening Only the second tap will water the whole garden [0,5]
"""
# Similar to 45 Jump Game II
class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        max_range = [0] * (n + 1)
        
        for i, r in enumerate(ranges):
            left, right = max(0, i - r), min(n, i + r)
            max_range[left] = max(max_range[left], right - left)
        
		# it's a jump game now
        start = end = step = 0
        
        while end < n:
            step += 1
            start, end = end, max(i + max_range[i] for i in range(start, end + 1))
            if start == end:
                return -1
            
        return step


"""
############################################################################
Twitter
############################################################################
"""

""" interview 
A bucket that has W white and R red beans, for each round, take a bean:
- if it is R, put it back and do again, eat whatever next bean is
- else (W), eat it
What is the probability that the last bean is R? 
"""
# dp[i][j]: probability that the bucket has i white and j red beans left
# ans = dp[0][1]
# dp[W][R] = 1
# dp[i][j] = dp[i+1][j] * (1 - ((i+1)/(i+j+1)) ** 2) + dp[i][j+1] * ((j+1)/(i+j+1)) ** 2
W=2; R=2
class Solution:
    def lastProb(self, W, R):
        dp = [[0 for _ in range(R+1)] for _ in range(W+1)]
        dp[W][R] = 1
        for j in range(R)[::-1]:
            # eat R
            dp[W][j] = dp[W][j+1] * ((j+1)/(W+j+1)) ** 2
        for i in range(W)[::-1]:
            # eat W
            dp[i][R] = dp[i+1][R] * (1 - (R/(i+R+1)) ** 2)
        for i in range(W)[::-1]:
            for j in range(R)[::-1]:
                dp[i][j] = dp[i+1][j] * (1 - (j/(i+j+1)) ** 2) + dp[i][j+1] * ((j+1)/(i+j+1)) ** 2
        print(dp)
        return dp[0][1]

sol = Solution()
sol.lastProb(2,2)
sol.lastProb(2,1)

"""1817. Finding the Users Active Minutes
You are given the logs for users' actions on LeetCode, and an integer k. 
The logs are represented by a 2D integer array logs where each logs[i] = [IDi, timei] 
indicates that the user with IDi performed an action at the minute timei.

Multiple users can perform actions simultaneously, and a single user can perform multiple actions in the same minute.

The user active minutes (UAM) for a given user is defined as the number of unique minutes in which 
the user performed an action on LeetCode. A minute can only be counted once, even if multiple actions occur during it.

You are to calculate a 1-indexed array answer of size k such that, for each j (1 <= j <= k), answer[j] is 
the number of users whose UAM equals j.

Return the array answer as described above.

Input: logs = [[0,5],[1,2],[0,2],[0,5],[1,3]], k = 5
Output: [0,2,0,0,0]
The user with ID=0 performed actions at minutes 5, 2, and 5 again. Hence, they have a UAM of 2 (minute 5 is only counted once).
The user with ID=1 performed actions at minutes 2 and 3. Hence, they have a UAM of 2.
Since both users have a UAM of 2, answer[2] is 2, and the remaining answer[j] values are 0.

Input: logs = [[1,1],[2,2],[2,3]], k = 4
Output: [1,1,0,0]
The user with ID=1 performed a single action at minute 1. Hence, they have a UAM of 1.
The user with ID=2 performed actions at minutes 2 and 3. Hence, they have a UAM of 2.
There is one user with a UAM of 1 and one with a UAM of 2.
Hence, answer[1] = 1, answer[2] = 1, and the remaining values are 0.
"""
import collections
class Solution:
    def findingUsersActiveMinutes(self, logs, k: int):
        UAMs = collections.defaultdict(set)
        ans = [0] * k
        for ID, time in logs:
            UAMs[ID].add(time)
        for UAM in UAMs.values():
            ans[len(UAM)-1] += 1
        return ans

sol=Solution()
sol.findingUsersActiveMinutes([[1,1],[2,2],[2,3]], 4)


"""
############################################################################
Amazon
############################################################################
"""
""" delete a element from list (replace with -1), do it in-place
"""
nums = [1,2,3,4,3,2,1]
def rm_target(nums, target):
    target_pt = 0
    for i in range(len(nums)):
        if nums[i] != target:
            nums[i], nums[target_pt] = nums[target], nums[i]
            target_pt += 1
    
    for i in range(target_pt, len(nums)):
        nums[i] = -1
    return None

rm_target(nums, 3)


"""
############################################################################
Google
############################################################################
"""

"""975. Odd Even Jump
At odd, jump up to smalles num >= curr, at even, jump up to largest num <= curr,
if more than one indeces, then only to the cloest j
https://leetcode.com/problems/odd-even-jump/

Input: arr = [10,13,12,14,15]
Output: 2
In total, there are 2 different starting indices i = 3 and i = 4, 
where we can reach the end with some number of jumps.

Input: arr = [2,3,1,1,4]
Output: 3
i= 1, 3, 4
"""
# two arrs to store the smallest next number >= curr (not first next number), 
# and largest next num <= curr
# then dp or dfs + memo
A = [10,13,12,14,15]
A = [1,2,3,2,1,4,4,5]
class Solution:
    def oddEvenJumps(self, A):
        n = len(A)
        next_higher, next_lower = [0] * n, [0] * n  # idx of the number >= or <= curr

        stack = []
        for a, i in sorted([a, i] for i, a in enumerate(A)):
            while stack and stack[-1] < i: # saves index, can guarantee later >= earlier
                next_higher[stack.pop()] = i
            stack.append(i)

        stack = []
        for a, i in sorted([-a, i] for i, a in enumerate(A)):
            while stack and stack[-1] < i:
                next_lower[stack.pop()] = i
            stack.append(i)

        higher, lower = [0] * n, [0] * n   # whether can achieve at odd / even jump
        higher[-1] = lower[-1] = 1
        for i in range(n - 1)[::-1]: # from n-1 -> 0
            higher[i] = lower[next_higher[i]] 
            lower[i] = higher[next_lower[i]]
        return sum(higher)


"""LeetCode 1102. Path With Maximum Minimum Value
Given a matrix of integers A with R rows and C columns, find the maximum score of a path starting at [0,0] and ending at [R-1,C-1].
The score of a path is the minimum value in that path.  For example, the value of the path 8 →  4 →  5 →  9 is 4.
A path moves some number of times from one visited cell to any neighbouring unvisited cell in north, east, west, south.

Input: [[5,4,5],[1,2,6],[7,4,6]]
Output: 4
Explanation: 
The path with the maximum score is highlighted in yellow. 

Input: [[3,4,6,3,4],[0,2,1,1,7],[8,8,3,2,7],[3,2,4,9,8],[4,1,2,0,0],[4,6,5,4,3]]
Output: 3

Hint: BFS, use maxHeap so always pop the maximum neighbors
From A[0][0], put element with index into maxHeap, sorted by element. Mark it as visited.
When polling out the currrent, check its surroundings. If not visited before, put it into maxHeap.
Until we hit the A[m-1][n-1].
Time Complexity: O(m*n*logmn). m = A.length. n = A[0].length. maxHeap add and poll takes O(logmn).

Space: O(m*n).

# solution 2: DP XXXX (????)
Define: dp[i][j] is the max score from [0][0] to [i][j]
Recurrence: dp[i][j] = max( min(dp[i-1][j], grid[i][j]), min(dp[i][j-1]), grid[i][j] )
https://www.cnblogs.com/Dylan-Java-NYC/p/11297106.html
"""
class Solution:
    def maximumMinimumPath(self, A):
        return 


""" Actual interview: compare two strings
# . represents backspace, decide whether two strings are equivalent
# you may only use constant space
Example
str1 = "abc.cd" # -> abcd
str2 = "ab.cd"  # -> acd
Return: False
"""
compare("abc.cd",  "ab.cd")
compare("ab.c.cd",  "ab.cd")
class CompareStrings:
    def compare(self, str1, str2):
        n1, n2 = len(str1), len(str2)
        b1, b2 = 0, 0
        # str1[:n1], str2[:n2]
        return self.compare_wt_back(str1, str2, n1, n2, b1, b2)

    def compare_wt_back(self, str1, str2, i1, i2, b1, b2):
        print("{} / {}".format(str1[:i1], str2[:i2]))
        if i1 == 0 and i2 == 0:
            return True
        if str1[i1-1] == ".":
            return self.compare_wt_back(str1, str2, i1-1, i2, b1+1, b2)
        if str2[i2-1] == ".":
            return self.compare_wt_back(str1, str2, i1, i2-1, b1, b2+1)
        if b1 > 0:
            return self.compare_wt_back(str1, str2, i1-1, i2, b1-1, b2)
        if b2 > 0:
            return self.compare_wt_back(str1, str2, i1, i2-1, b1, b2-1)
        if str1[i1-1] != str2[i2-1]:
            return False
        return self.compare_wt_back(str1, str2, i1-1, i2-1, b1, b2)


"""[LeetCode] Find Leaves of Binary Tree 找二叉树的叶节点
Given a binary tree, collect a tree's nodes as if you were doing this: Collect and remove all leaves, 
repeat until the tree is empty.

Example:
Input: [1,2,3,4,5]

          1
         / \
        2   3
       / \     
      4   5    

Output: [[4,5,3],[2],[1]]
"""
# https://www.cnblogs.com/grandyang/p/5616158.html
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


"""LeetCode 2096. Step-By-Step Directions From a Binary Tree Node to Another
You are given the root of a binary tree with n nodes. Each node is uniquely assigned a value from 1 to n. 
You are also given an integer startValue representing the value of the start node s, and a different integer 
destValue representing the value of the destination node t.

Find the shortest path starting from node s and ending at node t. Generate step-by-step directions of such 
path as a string consisting of only the uppercase letters 'L', 'R', and 'U'. Each letter indicates a specific direction:

'L' means to go from a node to its left child node.
'R' means to go from a node to its right child node.
'U' means to go from a node to its parent node.
Return the step-by-step directions of the shortest path from node s to node t.

Input: root = [5,1,2,3,null,6,4], startValue = 3, destValue = 6
          5
         / \
        1   2
       /   / \ 
      3   6   4

Output: "UURL"
Explanation: The shortest path is: 3 → 1 → 5 → 2 → 6.
"""
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        def getLCA(root,p,q):
            if not root: return None
            if root.val==p or root.val==q:
                return root
            L=getLCA(root.left,p,q)
            R=getLCA(root.right,p,q)
            if L and R:
                return root
            return L or R
        
        def getPath_dfs(node1, node2, path): #---> Problem here
            # can also use BFS
            if not node1: 
                return
            if node1.val==node2: 
                return path
            left_path = getPath(node1.left,node2,path+["L"])
            right_path = getPath(node1.right,node2,path+["R"])
            return left_path if left_path else right_path
        
        def getPath_bfs(root, startValue, destValue):
            # option 2
            startPath = ""
            endPath = ""
            q = deque()
            q.append(root)
            while q:
                node, path = q.popleft()
                if node.val == startValue:
                    startPath = path
                if node.val == destValue:
                    endPath = path
                if startPath and endPath:
                    startPath, endPath
                if node.left:
                    q.append((node.left, path+["L"]))
                if node.right:
                    q.append((node.right, path+["R"]))
                
            return -1, -1
        
        LCA=getLCA(root,startValue,destValue)
        path1=getPath_dfs(LCA,startValue,[]) 
        path2=getPath_dfs(LCA,destValue,[])
        path=["U"]*len(path1) + path2 if path1 else path2 
        return "".join(path)


"""2128 Remove All Ones With Row and Column Flips
Given a binary matrix, tell if it is possible to convert entire matrix to zeroes using below operations :
flip entire column
flip entire row

Input :
[[0,0,1,0],
[1,1,0,1],
[0,0,1,0],
[1,1,0,1]]

Output : Yes
[[0,0,0,0],
[0,0,0,0],
[0,0,0,0],
[0,0,0,0]]
这题只需要判断第一行, 见到0不动, 见到1翻过来,这样第一行就是全0. 然后通过判断其他行是不是全1或者全0, 即可知道答案.

证: 如果第一行已经全0, 其他某行, 非全1或者全0, 那么这行通过行翻转, 肯定不能得到全1或者全0, 
但是通过列翻转, 又破坏第一行的全0,故此.
https://www.chenguanghe.com/remove-all-ones-with-row-and-column-flips/
"""

"""1937. Maximum Number of Points with Cost

To gain points, you must pick one cell in each row. Picking the cell at coordinates (r, c) will add points[r][c] to your score.
However, you will lose points if you pick a cell too far from the cell that you picked in the previous row. 
For every two adjacent rows r and r + 1 (where 0 <= r < m - 1), picking cells at coordinates (r, c1) and 
(r + 1, c2) will subtract abs(c1 - c2) from your score.

Return the maximum number of points you can achieve.

Input: points = [[1,2,3],[1,5,1],[3,1,1]]
Output: 9
Explanation:
The blue cells denote the optimal cells to pick, which have coordinates (0, 2), (1, 1), and (2, 0).
You add 3 + 5 + 3 = 11 to your score.
However, you must subtract abs(2 - 1) + abs(1 - 0) = 2 from your score.
Your final score is 11 - 2 = 9.
"""
# DFS is likely to TLE
# https://leetcode.com/problems/maximum-number-of-points-with-cost/discuss/1344888/C%2B%2B-dp-from-O(m-*-n-*-n)-to-O(m-*-n)
# DP[i][j]: means the maximum points ends with points[i][j]. The value is from previous row with costs abs(j - k).
# dp[i][j] = max(dp[i - 1][k] + point[i][j] - abs(j - k)) for each 0 <= k < points[i - 1].szie()  O(m*n*n)
# can still improve because the comparison of row i-1 can be indep of j
# from left side: dp[i][j] = max(dp[i - 1][k] + k-j) + points[i][j], for all 0 <= k <= j
#                           = max(dp[i - 1][k] + k) + points[i][j] - j -> no j in max()!!
# In this way, we can compute all the dp[i-1][k] for all j's, and then just compute diff j to find the biggest dp[i][j]
# same for right side: dp[i][j] = max(dp[i - 1][k] - k) + points[i][j] + j for all j <= k < n



"""2034. Stock Price Fluctuation
You are given a stream of records about a particular stock. Each record contains a timestamp and the corresponding price of the stock at that timestamp.

Unfortunately due to the volatile nature of the stock market, the records do not come in order. 
Even worse, some records may be incorrect. Another record with the same timestamp may appear later in the 
stream correcting the price of the previous wrong record.

Design an algorithm that:
Updates the price of the stock at a particular timestamp, correcting the price from any previous records at the timestamp.
Finds the latest price of the stock based on the current records, at the latest timestamp recorded.
Finds the maximum price the stock has been based on the current records.
Finds the minimum price the stock has been based on the current records.

Implement the StockPrice class:
StockPrice() Initializes the object with no price records.
void update(int timestamp, int price) Updates the price of the stock at the given timestamp.
int current() Returns the latest price of the stock.
int maximum() Returns the maximum price of the stock.
int minimum() Returns the minimum price of the stock

Example 1:
Input
["StockPrice", "update", "update", "current", "maximum", "update", "maximum", "update", "minimum"]
[[], [1, 10], [2, 5], [], [], [1, 3], [], [4, 2], []]
Output
[null, null, null, 5, 10, null, 5, null, 2]
"""
# https://leetcode.com/problems/stock-price-fluctuation/discuss/1513628/Python3-SortedDict-and-SortedList
# SOlution 1: map + sortedlist
from sortedcontainers import SortedList
class StockPrice:
    def __init__(self):
        self.latest_time = -1
        self.time2prices = dict()
        self.prices = SortedList()  # logN to insert, remove, peek by index/value
        
    def update(self, timestamp: int, price: int) -> None:
        if timestamp in self.time2prices:
            old_price = self.time2prices[timestamp]
            self.time2prices[timestamp] = price
            self.prices.remove(old_price)
            self.prices.add(price)
        else:
            self.time2prices[timestamp] = price
            self.prices.add(price)
        self.latest_time = max(self.latest_time, timestamp)
        return

    def current(self) -> int:
        return self.time2prices[self.latest_time]
        
    def maximum(self) -> int:
        return self.prices[-1]
        
    def minimum(self) -> int:
        return self.prices[0]


sol = StockPrice()
sol.update(1, 10)
sol.update(2, 5) 
sol.current()
sol.maximum()

# Solutin 2: use two heaps, a dict and a latest_time constant; (not optimal to use two priority queue + dict)
# Keep a dict of timestamp to prices. When updating a timestamp, push it in a heap along with it's timestamp.
# When maximum or minimum is called, if the timestamp doesn't match it's current price, this indiciates 
# it's value has been updated in the meantime. 
# Keeping popping elements off the heap until an element matches it's current price
# https://leetcode.com/problems/stock-price-fluctuation/discuss/1513293/Python-Clean-2-Heaps-Commented-Code


""" 1146. Snapshot Array
Implement a SnapshotArray that supports the following interface:

SnapshotArray(int length) initializes an array-like data structure with the given length.  Initially, each element equals 0.
void set(index, val) sets the element at the given index to be equal to val.
int snap() takes a snapshot of the array and returns the snap_id: the total number of times we called snap() minus 1.
int get(index, snap_id) returns the value at the given index, at the time we took the snapshot with the given snap_id

Input: ["SnapshotArray","set","snap","set","get"]
[[3],[0,5],[],[0,6],[0,0]]
Output: [null,null,0,null,5]
Explanation: 
SnapshotArray snapshotArr = new SnapshotArray(3); // set the length to be 3
snapshotArr.set(0,5);  // Set array[0] = 5
snapshotArr.snap();  // Take a snapshot, return snap_id = 0
snapshotArr.set(0,6);
snapshotArr.get(0,0);  // Get the value of array[0] with snap_id = 0, return 5
"""
from sortedcontainers import SortedDict
class SnapshotArray:
    def __init__(self, length: int):
        self.array = [SortedDict({-1: 0}) for _ in range(length)] # -1 handle if no snap_id 
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        self.array[index][self.snap_id] = val
        return 

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1
        
    def get(self, index: int, snap_id: int) -> int:
        if index < 0 or index >= len(self.array):
            return 0
        if snap_id in self.array[index]:
            return self.array[index][snap_id]
        # find the id before 
        prev_snap_id = self.array[index].bisect_left(snap_id)  # location to insert snap_id
        # prev_snap_id = self.find_last_smaller_than_id(snap_id)  # location to insert snap_id
        if prev_snap_id == 1: # snap_id < all prev snap_id since first is -1
            return 0
        sorted_ids = self.array[index].keys()
        insert_key = sorted_ids[prev_snap_id-1]
        return self.array[index][insert_key]
        

"""2013. Detect Squares
Use diagnal point to iterate 
p1  p4
p3  p2

given p1, for each p2, find cnt of p2, p3, p4 to multiple
"""

"""833. Find And Replace in String
Input: s = "abcd", indices = [0, 2], sources = ["a", "cd"], targets = ["eee", "ffff"]
Output: "eeebffff"
Explanation:
"a" occurs at index 0 in s, so we replace it with "eee".
"cd" occurs at index 2 in s, so we replace it with "ffff".

Input: s = "abcd", indices = [0, 2], sources = ["ab","ec"], targets = ["eee","ffff"]
Output: "eeecd"
Explanation:
"ab" occurs at index 0 in s, so we replace it with "eee".
"ec" does not occur at index 2 in s, so we do nothing.
"""
# use list to simplify the operation 
def findReplaceString(self, S, indices, sources, targets):
        """
        :type S: str
        :type indexes: List[int]
        :type sources: List[str]
        :type targets: List[str]
        :rtype: str
        """
        modified = list(S)
        for index, source, target in zip(indices, sources, targets):
            if not S[index:].startswith(source):
                continue
            else:
                modified[index] = target
                for i in range(index+1, len(source) + index):
                    modified[i] = ""

        return "".join(modified)


"""2115. Find All Possible Recipes from Given Supplies
Example 2
Input: recipes = ["bread","sandwich"], ingredients = [["yeast","flour"],["bread","meat"]], 
supplies = ["yeast","flour","meat"]
Output: ["bread","sandwich"]
Explanation:
We can create "bread" since we have the ingredients "yeast" and "flour".
We can create "sandwich" since we have the ingredient "meat" and can create the ingredient "bread"
"""
from collections import defaultdict
class Solution:
    def findAllRecipes(self, recipes, ingredients, supplies):
        rec2ing = defaultdict(set)
        ing2rec = defaultdict(set)  # may only need indegree instead of a set
        supplies = set(supplies)
        q = deque()
        for i, rec in enumerate(recipes):
            for ing in ingredients[i]:
                if ing not in supplies:
                    # ing is a rec
                    rec2ing[rec].add(ing)
                    ing2rec[ing].add(rec)
        res = []
        for rec in recipes:
            if len(rec2ing[rec]) == 0: # enough to only count
                q.append(rec)
                res.append(rec)
        
        while q:
            curr_rec = q.popleft() # rec can be made
            for next_rec in ing2rec[curr_rec]:
                rec2ing[next_rec].remove(curr_rec)
                if len(rec2ing[next_rec]) == 0:
                    q.append(next_rec)
                    res.append(next_rec)
        
        return res


""" 552. Student Attendance Record II
Any student is eligible for an attendance award if they meet both of the following criteria:

The student was absent ('A') for strictly fewer than 2 days total.
The student was never late ('L') for 3 or more consecutive days.

Input: n = 2
Output: 8
Explanation: There are 8 records with length 2 that are eligible for an award:
"PP", "AP", "PA", "LP", "PL", "AL", "LA", "LL"
Only "AA" is not eligible because there are 2 absences (there need to be fewer than 2).
"""
# 下面这种方法来自 大神 dettier 的帖子，这里面定义了两个数组P和 PorL，其中 P[i] 表示数组前i个数字中1以P结尾的排列个数，
# PorL[i] 表示数组前i个数字中已P或者L结尾的排列个数。这个解法的精髓是先不考虑字符A的情况，而是先把定义的这个数组先求出来，
# 由于P字符可以再任意字符后面加上，所以 P[i] = PorL[i-1]；而 PorL[i] 由两部分组成，P[i] + L[i]，其中 P[i] 已经更新了，
# L[i] 只能当前一个字符是P，或者前一个字符是L且再前一个字符是P的时候加上，即为 P[i-1] + P[i-2]，
# 所以 PorL[i] = P[i] + P[i-1] + P[i-2]。

# 那么这里就已经把不包含A的情况求出来了，存在了 PorL[n] 中，下面就是要求包含一个A的情况，那么就得去除一个字符，
# 从而给A留出位置。就相当于在数组的任意一个位置上加上A，数组就被分成左右两个部分了，而这两个部分当然就不能再有A了，
# 实际上所有不包含A的情况都已经在数组 PorL 中计算过了，而分成的子数组的长度又不会大于原数组的长度，
# 所以直接在 PorL 中取值就行了，两个子数组的排列个数相乘，然后再把所有分割的情况累加起来就是最终结果啦，参见代码如下：
"""
class Solution {
public:
    int checkRecord(int n) {
        int M = 1e9 + 7;
        vector<long> P(n + 1), PorL(n + 1);
        P[0] = 1; PorL[0] = 1; PorL[1] = 2;
        for (int i = 1; i <= n; ++i) {
            P[i] = PorL[i - 1];
            if (i > 1) PorL[i] = (P[i] + P[i - 1] + P[i - 2]) % M;
        }
        long res = PorL[n];
        for (int i = 0; i < n; ++i) {
            long t = (PorL[i] * PorL[n - 1 - i]) % M;
            res = (res + t) % M;
        }
        return res;
    }
};
"""

"""2158. Amount of New Area Painted Each Day
input: [[1,4], [4, 7], [5,8]]
output: [3, 3, 1]
Explanation, day 1 paint 4-1 = 3, day 2 paints 7-4=3, day 3 paint 8-7, 
because 5-7 already painted

psuedo code:
for time, (idx, in) in dict.items():
    if in: # add in
        res[idx] += time - ds[ds.find_first_idx()]
        ds[ds.find_first_idx()] = time
        ds[idx] = time # store start time of job idx
    else: # remove
        if idx == ds.find_first_idx():
            res[idx] += time - ds[idx]
            ds.remove(idx)
            ds[ds.find_first_idx] = time # current top idx, important! 
        else:
            ds.remove(idx)
"""
from collections import defaultdict
class Solution:
    def amountPainted(self, paint):
        day2idx = defaultdict(list)
        for i, (start, end) in enumerate(paint):
            day2idx[start].append([i, 1]) # start
            day2idx[end].append([i, -1]) # end
        
        res = [0] * len(paint)
        idx2startTime = SortedDict()
        for time, vals in sorted(day2idx.items()):
            for idx, status in vals:
                if status == 1:
                    if len(idx2startTime) > 0:
                        top_idx = idx2startTime.keys()[0]
                        res[top_idx] += time - idx2startTime[top_idx]
                        idx2startTime[top_idx] = time
                    idx2startTime[idx] = time
                else:
                    if idx == idx2startTime.keys()[0]:
                        res[idx] += time - idx2startTime[idx]
                        idx2startTime.pop(idx)
                        if len(idx2startTime) > 0:
                            idx2startTime[idx2startTime.keys()[0]] = time
                    else:
                        idx2startTime.pop(idx)
        
        return res


sol=Solution()
sol.amountPainted(paint = [[1,4], [4, 7], [5,8]])

# https://www.youtube.com/watch?v=BuPTkTw2dC4
# sweepline, always add the new segment into the earliest day index 
class Solution:
    def amountPainted(self, paint: List[List[int]]) -> List[int]:
        dic = collections.defaultdict(list)
        # construct sweepline dict
        for i, (start, end) in enumerate(paint):
            dic[start].append([i, 1]) # start
            dic[end].append([i, -1]) # end
        daySet = SortedList() # only store index
        ans = [0 for _ in paint]
        arr = [(k, dic[k]) for k in sorted(dic)]  # (time, [idx, status])
        # go through sweepline
        for i, (pos, flags) in enumerate(arr):
            for idx, flag in flags:
                if flag == -1: # rm idx
                    daySet.pop(idx) # logN
                else:
                    daySet.add(idx) # logN
            if i < len(arr) - 1 and daySet:
                ans[daySet[0]] += arr[i+1][0]- arr[i][0] # into the earliest day index 
        return ans


"""777. Swap Adjacent in LR String
In a string composed of 'L', 'R', and 'X' characters, like "RXXLRXRXL", a move consists of either replacing 
one occurrence of "XL" with "LX", or replacing one occurrence of "RX" with "XR". Given the starting string 
start and the ending string end, return True if and only if there exists a sequence of moves to transform 
one string to the other.

Input: start = "RXXLRXRXL", end = "XRLXXRRLX"
Output: true
Explanation: We can transform start to end following these steps:
RXXLRXRXL ->
XRXLRXRXL ->
XRLXRXRXL ->
XRLXXRRXL ->
XRLXXRRLX
"""
# https://leetcode.com/problems/swap-adjacent-in-lr-string/discuss/873004/Easy-to-understand-explanation-with-PICTURE
# L can only move to left, and R can only move to right, they cannot cross
# check the index of L in start, and that of end -> start index should be on right of end
# same for R
class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        if len(start) != len(end): return False
        
        # check L R orders are the same
        if start.replace('X','') != end.replace('X', ''): return False
        
        n = len(start)
        Lstart = [i for i in range(n) if start[i] == 'L']
        Lend = [i for i in range(n) if end[i] == 'L']
        
        Rstart = [i for i in range(n) if start[i] == 'R']
        Rend = [i for i in range(n) if end[i] == 'R']
		# check L positions are correct
        for i, j in zip(Lstart, Lend):
            if i < j:
                return False
            
        # check R positions are correct
        for i, j in zip(Rstart, Rend):
            if i > j:
                return False
            
        return True


"""2135. Count Words Obtained After Adding a Letter
For each string in targetWords, check if it is possible to choose a string from startWords and perform a conversion 
operation on it to be equal to that from targetWords
- Append any lowercase letter that is not present in the string to its end
- Rearrange the letters of the new string in any arbitrary order
Return the number of strings in targetWords that can be obtained by performing the operations on any string of startWords

Input: startWords = ["ant","act","tack"], targetWords = ["tack","act","acti"]
Output: 2
Explanation:
- In order to form targetWords[0] = "tack", we use startWords[1] = "act", append 'k' to it, and rearrange "actk" to "tack".
- There is no string in startWords that can be used to obtain targetWords[1] = "act".
  Note that "act" does exist in startWords, but we must append one letter to the string before rearranging it.
- In order to form targetWords[2] = "acti", we use startWords[1] = "act", append 'i' to it, and rearrange "acti" to "acti" itself.

No letter occurs more than once in any string of startWords or targetWords!
"""
# a more efficient way is bitmask, which can quickly decide whether two strings are the same after sorting O(1)
# If not using bitmask, we can use a Trie to store all sorted strings from startWords, and also store
# the original start words in a set, sort it, and search in the Trie
# note that we should allow exact one mismatch due to append operation (give a bool app=False)


"""1554. Strings Differ by One Character
Given a list of strings dict where all the strings are of the same length.

Return True if there are 2 strings that only differ by 1 character in the same index, otherwise return False.

Follow up: Could you solve this problem in O(n*m) where n is the length of dict and m is the length of each string.

Example 1:
Input: dict = ["abcd","acbd", "aacd"]
Output: true
Output: Strings "abcd" and "aacd" differ only by one character in the index 1
"""

# 举个例子，对于上面的示例1，dict = [“abcd”,“acbd”, “aacd”]，首先遍历第一个字符串的第一个字符，将其用*代替，
# “abcd"变为”*bcd"，然后dicts["*bcd"]+=1；剩下的同理；
# 我们希望字符串之间只有一个字符不同，我们分别将字符串中的某一个字符变为"*"，然后比较某一种字符串类别是否有两个，
# 如果出现两个，则返回True。

"""[LeetCode] 660. Remove 9 移除9
Start from integer 1, remove any integer that contains 9 such as 9, 19, 29...
So now, you will have a new integer sequence: 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, ...

Given a positive integer n, you need to return the n-th integer after removing. Note that 1 will be the first integer.

Example 1:
Input: 9
Output: 10
Hint: n will not exceed 9 x 10^8.

这道题让我们移除所有包含数字9的数字，然后得到一个新的数列，给了一个数字n，求在这个新的数组中第n个数字。多写些数字来看看：
0，1，2，3，4，5，6，7，8 （移除了9）
10，11，12，13，14，15，16，17，18 （移除了19）
.....
80，81，82，83，84，85，86，87，88 （移除了89）
（移除了 90 - 99 ）
100，101，102，103，104，105，106，107，108 （移除了109）

可以发现，8的下一位就是10了，18的下一位是20，88的下一位是100，实际上这就是九进制的数字的规律，那么这道题就变成了将十进制数n转为九进制数，
这个就没啥难度了，就每次对9取余，然后乘以 base，n每次自除以9，base 每次扩大10倍，参见代码如下：

class Solution {
public:
   int newInteger(int n) {
        long res = 0, base = 1;
        while (n > 0) {
            multiplier += n % 9;
            res += multiplier * base
            n = n / 9;
            base *= 10;
        }
        return res;
   }
};

此解法只限于对9， 如果对其他数字， 参见
https://www.cnblogs.com/grandyang/p/8261714.html
"""


"""792. Number of Matching Subsequences
Given a string s and an array of strings words, return the number of words[i] that is a subsequence of s.

Example: 
Input: s = "abcde", words = ["a","bb","acd","ace"]
Output: 3
Explanation: There are three strings in words that are a subsequence of s: "a", "acd", "ace".
"""

"""
Solution: 
Given the input
"abcde"
["a","bb","acd","ace"]
Create a word dict to keep track of prefix -> candidates

{
  "a": ["a", "acd", "ace"],
  "b": ["bb"]
}
Go through each char in s and find candidates that start with char. The candidate has length of 1, 
then we found one of the subsequences
For char "a", we found these candidates
["a", "acd", "ace"]
Only "a" is a subsequence, then we increment count
Then we update the word dict by 
taking out all the candidates start with "a" becuase we have processed them
add new candidates with the prefix "a" removed, aka ["cd", "ce"]
{
  "a": [],
  "b": ["bb"],
  "c": ["cd", "ce"]
}
Keep repeating until the input s is processed.
"""

"""[LeetCode] 652. Find Duplicate Subtrees 寻找重复树
Given the root of a binary tree, return all duplicate subtrees.
For each kind of duplicate subtrees, you only need to return the root node of any one of them.
Two trees are duplicate if they have the same structure with the same node values.

Example 1:

    4
   / \
  9   9
 /   /  \
3   3   27

Output: [node(val=9),node(val=3)]
"""
#后序遍历，还有数组序列化，并且建立序列化跟其出现次数的映射，这样如果得到某个结点的序列化字符串，而该字符串正好出现的次数为1，
# 说明之前已经有一个重复树了，将当前结点存入结果res，这样保证了多个重复树只会存入一个结点，参见代码如下：
"""
class Solution {
public:
    vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
        vector<TreeNode*> res;
        unordered_map<string, int> m;
        postorder(root, m, res);
        return res;
    }
    string postorder(TreeNode* node, unordered_map<string, int>& m, vector<TreeNode*>& res) {
        if (!node) return "#";
        string str = to_string(node->val) + "," + postorder(node->left, m, res) + "," + postorder(node->right, m, res);
        if (m[str] == 1) res.push_back(node); // postorder
        ++m[str];
        return str;
    }
};
"""

"""562 Longest Line of Consecutive One in Matrix
Given a 01 matrix M, find the longest line of consecutive one in the matrix. The line could be horizontal, 
vertical, diagonal or anti-diagonal.

Input:
[[0,1,1,0],
 [0,1,1,0],
 [0,0,0,1]]
Output: 3

Solution: DP, use four dp[i][j][0-3] matrix to record longest one line ends at mat[i][j] for horizontal, vertical, diag, and anti-diag
anti-diag: dp[i][j][3] = dp[i - 1][j + 1][3] + 1 if M[i][j]==1 else 0
"""


"""302. Smallest Rectangle Enclosing Black Pixels
An image is represented by a binary matrix with 0 as a white pixel and 1 as a black pixel. 
The black pixels are connected, i.e., there is only one black region. Pixels are connected horizontally and vertically. 
Given the location (x, y) of one of the black pixels, 
return the area of the smallest (axis-aligned) rectangle that encloses all black pixels.

For example, given the following image:
[
  "0010",
  "0110",
  "0100"
]
and x = 0, y = 2, return 6.
"""
# can you think of a solution at runtime O(mlogn + nlogm)? m=nrow, n=ncol
# binary search to find the four boundraies
# for example, for horizontal, everytime take mid, and then check each elem from mat[mid][0] to mat[mid][ncol] - logn * m


""" 889 · Sentence Screen Fitting ???
Given a rows x cols screen and a sentence represented by a list of non-empty words, find how many times the given sentence can be fitted on the screen.

Example 1:
	Input: rows = 4, cols = 5, sentence = ["I", "had", "apple", "pie"]
	Output: 1
	
	Explanation:
	I-had
	apple
	pie-I
	had--
	
	The character '-' signifies an empty space on the screen.
	
Example 2:
	Input:  rows = 2, cols = 8, sentence = ["hello", "world"]
	Output:  1
	
	Explanation:	
	hello---
	world---

Example 3:
	Input: rows = 3, cols = 6, sentence = ["a", "bcd", "e"]
	Output:  2
	
	Explanation:
	a-bcd-
	e-a---
	bcd-e-
"""
class Solution:
    def wordsTyping(self, sentence: List[str], rows: int, cols: int) -> int:
        num_words = len(sentence)
        dp = [0] * num_words
        #dp[i] denotes if a new row starts with word[i], # of words we can put there, inclusive
        for i in range(num_words):
            length = cols
            j = i 
            while len(sentence[j % num_words]) <= length:
                length = length - len(sentence[j % num_words]) - 1
                j += 1
            dp[i] = j - i 
        print(dp)
        
        k, index = 0, 0 #initialization 
        total_num_words = 0 
        for k in range(rows):
            total_num_words += dp[index]
            index = (index + dp[index%num_words]) % num_words
        return total_num_words//num_words


"""[LeetCode] 920. Number of Music Playlists 音乐播放列表的个数
Your music player contains `N` different songs and she wants to listen to `L` (not necessarily different) songs during your trip.  You create a playlist so that:
Every song is played at least once
A song can only be played again only if K other songs have been played

Example
Input: N = 2, L = 3, K = 0
Output: 6 Explanation: There are 6 possible playlists. 
[1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2, 1], [2, 1, 2], [1, 2, 2]

Input: N = 2, L = 3, K = 1
Output: 2
Explanation: There are 2 possible playlists. [1, 2, 1], [2, 1, 2]

一个是每首歌都必须至少播放1次，第二个是两首重复歌的中间至少要有K首其他的歌
dp[i][j] 表示当一共有N首unique的歌时, 放一个有i首unique歌曲的长度为j的歌单有多少种。
下面来考虑状态转移方程，在加入一首歌的时候，此时有两种情况：
- 当加入的是一首新歌，则表示之前的 j-1 首歌中有 i-1 首不同的歌曲，其所有的组合情况都可以加上这首新歌，
那么当前其实有 N-(i-1) 首新歌可以选。
- 当加入的是一首重复的歌，则表示之前的 j-1 首歌中已经有了 i 首不同的歌，那么若没有K的限制，则当前有 i 首重复的歌可以选。
但是现在有了K的限制，意思是两首重复歌中间必须要有K首其他的歌，则当前只有 i-K 首可以选。而当 i<K 时，其实这种情况是为0的。
            dp[i-1][j-1] * (N-(i-1)) + dp[i][j-1] * (i-K)    (i > K)
           /
dp[i][j] = 
           \
            dp[i-1][j-1] * (N-(i-1))   (j <= K)

"""
# dp[i][j] 表示当一共有N首unique的歌时, 放一个有i首unique歌曲的长度为j的歌单有多少种
class Solution:
    def numMusicPlaylists(self, N, L, K):
        dp = [[0 for j in range(L + 1)] for i in range(N + 1)]
        mod = 10**9 + 7
        for i in range(1, N+1): # unique songs
            for j in range(1, L+1): # len of list
                if j == 1:
                    dp[i][j] = N if i == 1 else 0
                    continue
                if i == 1: # j > 1
                    dp[i][j] = N if K == 0 else 0
                    continue
                if i > K:
                    # jth is a new or old song from first j-1 + 
                    dp[i][j] = (dp[i-1][j-1] * (N-(i-1)) + dp[i][j-1] * (i-K)) % mod
                else:
                    # jth can only be a new song from first j-1
                    dp[i][j] = (dp[i-1][j-1] * (N-(i-1))) % mod

        return dp[N][L]


# https://leetcode.com/problems/number-of-music-playlists/discuss/178415/C%2B%2BJavaPython-DP-Solution
# dp[i][j] 表示当一共有i首unique, 歌单长度为j首，不同的方法的数量
class Solution2:
    # F(N,L,K) = F(N - 1, L - 1, K) * N + F(N, L - 1, K) * (N - K)
    def numMusicPlaylists(self, N, L, K):
        dp = [[0 for i in range(L + 1)] for j in range(N + 1)]
        for i in range(K + 1, N + 1): # i unique songs
            for j in range(i, L + 1): # j-long songs
                if i == j or i == K + 1:
                    dp[i][j] = math.factorial(i)
                else:
                    dp[i][j] = dp[i - 1][j - 1] * i + dp[i][j - 1] * (i - K)
        return dp[N][L] % (10**9 + 7)


"""664. Strange Printer
There is a strange printer with the following two special properties:

The printer can only print a sequence of the same character each time.
At each turn, the printer can print new characters starting from and ending at any place and will cover the original existing characters.

Given a string s, return the minimum number of turns the printer needed to print it.

Input: s = "aaabbb"
Output: 2
Explanation: Print "aaa" first and then print "bbb".

Input: s = "aba"
Output: 2
Explanation: Print "aaa" first and then print "b" from the second place of the string, which will cover the existing character 'a'.

Solution: https://leetcode.com/problems/strange-printer/discuss/152758/Clear-Logical-Thinking-with-Code

divide the problem:
we keep dividing s until the substring contains 1 or 2 characters (as the base case)
Take s = "abc" for example,

  abc
 /    \
a,bc ab,c (base case here)

conquer the subproblems:
turns s needed = min(turns one substring needed + turns the other needed) (since there are many ways to divide s, 
we pick the one needs minimum turns)
Please note that, if s = "aba", and we divide it into "a,ba", turns "aba" needed = turns "a" needed + turns "ba" needed - 1 
(aaa => aba rather than a => ab => aba).

state: state[i][j] turns needed to print s[i .. j] (both inclusive)
aim state: state[0][n - 1] (n = s.length() - 1)
state transition:

state[i][i] = 1;
state[i][i + 1] = 1 if s[i] == s[i + 1]
state[i][i + 1] = 2 if s[i] != s[i + 1]
state[i][j] = min(state[i][k] + state[k + 1][j]) for i <= k <= j - 1
	please note that, if s[i] equals to s[j] , state[i][j] should -1

# Code: 
public int strangePrinter(String s) {

    if (s == null || s.length() == 0) {
        return 0;
    }

    int n = s.length();
    int[][] state = new int[n][n];

    for (int i = 0; i < n; i++) {
        state[i][i] = 1;
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int dist = 1; dist + i < n; dist++) {
            int j = dist + i;
            if (dist == 1) {
                state[i][j] = (s.charAt(i) == s.charAt(j)) ? 1 : 2;
                continue;
            }
            state[i][j] = Integer.MAX_VALUE;
            for (int k = i; k + 1 <= j; k++) {
                int tmp = state[i][k] + state[k + 1][j];
                state[i][j] = Math.min(state[i][j], tmp);
            }
            if (s.charAt(i) == s.charAt(j)) {
                state[i][j]--;
            }
        }
    }

    return state[0][n - 1];
}
"""

"""642. Design Search Autocomplete System
Design a search autocomplete system for a search engine. 

Example:
Operation: AutocompleteSystem(["i love you", "island","ironman", "i love leetcode"], [5,3,2,2]) 
The system have already tracked down the following sentences and their corresponding times: 
"i love you" : 5 times 
"island" : 3 times 
"ironman" : 2 times 
"i love leetcode" : 2 times 
Now, the user begins another search: 

Operation: input('i') 
Output: ["i love you", "island","i love leetcode"] 
Explanation: 
There are four sentences that have prefix "i". Among them, "ironman" and "i love leetcode" have same hot degree. 
Since ' ' has ASCII code 32 and 'r' has ASCII code 114, "i love leetcode" should be in front of "ironman". 
Also we only need to output top 3 hot sentences, so "ironman" will be ignored. 

Operation: input(' ') 
Output: ["i love you","i love leetcode"] 
Explanation: 
There are only two sentences that have prefix "i ".

Operation: input('a') 
Output: [] 
Explanation: 
There are no sentences that have prefix "i a". 
"""
# similar to word auto completion, design a Trie and save the auto-complete at each tree node with maxheap 


"""715. Range Module
A Range Module is a module that tracks ranges of numbers. Design a data structure to track the ranges represented as 
half-open intervals and query about them.
A half-open interval [left, right) denotes all the real numbers x where left <= x < right.

Implement: 
void addRange(int left, int right) Adds the half-open interval [left, right)
boolean queryRange(int left, int right) Returns true if every real number in the interval [left, right) is currently being tracked
void removeRange(int left, int right) Stops tracking every real number currently being tracked in the half-open interval [left, right)

Input
["RangeModule", "addRange", "removeRange", "queryRange", "queryRange", "queryRange"]
[[], [10, 20], [14, 16], [10, 14], [13, 15], [16, 17]]
Output
[null, null, null, true, false, true]

Explanation
RangeModule rangeModule = new RangeModule();
rangeModule.addRange(10, 20);
rangeModule.removeRange(14, 16);
rangeModule.queryRange(10, 14); // return True,(Every number in [10, 14) is being tracked)
rangeModule.queryRange(13, 15); // return False,(Numbers like 14, 14.03, 14.17 in [13, 15) are not being tracked)
rangeModule.queryRange(16, 17); // return True, (The number 16 in [16, 17) is still being tracked, despite the remove operation)
"""
# http://zxi.mytechroad.com/blog/data-structure/leetcode-715-range-module/
from sortedcontainers import SortedDict
class RangeModule:
    def __init__(self):
        self.start2end = SortedDict()

    def addRange(self, left: int, right: int):
        # find the first and last interval overlaps with [left, right]
        # [[1,3], [4,6], [7,8]], [3,4] -> first=0, last=2 although [)
        first, last = self.find_first_and_last(left, right)
        if first < last:
            # merge from first to last, pop out all the intervals
            sorted_start_time = self.start2end.keys()
            for key in sorted_start_time[first:last]:
                curr_start = key
                left = min(left, curr_start)
                right = max(right, self.start2end[curr_start])
                del self.start2end[key]
        self.start2end[left] = right
        return 

    def queryRange(self, left: int, right: int) -> bool:
        # return whether overlap
        first, last = self.find_first_and_last(left, right)
        if first < last:
            start, end = self.start2end.peekitem(first) # only need to check left overlap interval
            return start <= left and right <= end
        return False

    def removeRange(self, left: int, right: int) -> None:
        first, last = self.find_first_and_last(left, right)
        if first < last:
            sorted_start_time = self.start2end.keys()
            # delete all overlap intervals
            # [[1,3], [4,6]], [3,5] 
            left_start = min(sorted_start_time[first], left)  # keep left_start -> left
            right_end = max(self.start2end[sorted_start_time[last-1]], right) # keep right -> right_end
            for key in sorted_start_time[first:last]:
                del self.start2end[key]
            if left_start < left:
                self.start2end[left_start] = left
            if right_end > right:
                self.start2end[right] = right_end
        return 
        
    def find_first_and_last(self, left, right):
        # find the first and last interval overlaps with [left, right]
        start_idx = self.start2end.bisect_left(left) # [[1,2], [3,4]], [5,6] -> return 2, [3.5,5] -> return 2 (needs adjust)
        end_idx = self.start2end.bisect_right(right)  # end_idx -> first interval with start > right
        if start_idx != 0:
            # if prev interval overlap with [left, right], then start_idx -= 1
            if self.start2end[self.start2end.keys()[start_idx - 1]] >= left: # start_idx : first interval with end >= left
                # no move when self.start2end[start_idx] == left, e.g., [[2,3], [4,6]], [6,8], then start_idx=1
                start_idx -= 1
        return start_idx, end_idx


#  [[1,3], [4,6], [7,8]]
sol=RangeModule()
sol.addRange(1, 4)
sol.addRange(5, 6)
sol.start2end
sol.addRange(7,8)
sol.addRange(6.5, 6.6)
sol.addRange(5,8)


"""1631. Path With Minimum Effort
You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right cell, (rows-1, columns-1) (i.e., 0-indexed). 
You can move up, down, left, or right, and you wish to find a route that requires the minimum effort

A route's effort is the maximum absolute difference in heights between two consecutive cells of the route.

Input: heights = [
    [1,2,2],
    [3,8,2],
    [5,3,5]]
Output: 2
Explanation: The route of [1,3,5,3,5] has a maximum absolute difference of 2 in consecutive cells.
This is better than the route of [1,2,2,2,5], where the maximum absolute difference is 3.
"""
from heapq import heappush, heappop
class Solution:
    def minimumEffortPath(self, heights) -> int:
        q = []  # (min heap)
        visited = set()
        heappush(q, (0, 0, 0))
        while q:
            max_diff, x, y  = heappop(q)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            if x == len(heights) - 1 and y== len(heights[0]) - 1:
                return max_diff
            for next_x, next_y in self.get_nb(x, y, len(heights), len(heights[0])):
                if (next_x, next_y) not in visited:
                    curr_diff = abs(heights[next_x][next_y] - heights[x][y])
                    heappush(q, (max(max_diff, curr_diff), next_x, next_y))
        
        return 
    
    def get_nb(self, x, y, m, n):
        res = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= x+dx < m and 0 <= y+dy < n:
                res.append([x+dx, y+dy])
        return res


"""1834. Single-Threaded CPU
You are given n​​​​​​ tasks labeled from 0 to n - 1 represented by a 2D integer array tasks, where tasks[i] = [enqueueTimei, processingTimei] 
means that the i​​​​​​th​​​​ task will be available to process at enqueueTimei and will take processingTimei to finish processing.

If the CPU is idle and there are available tasks, the CPU will choose the one with the shortest processing time. If multiple tasks 
have the same shortest processing time, it will choose the task with the smallest index.

Return the order in which the CPU will process the tasks.

Input: tasks = [[1,2],[2,4],[3,2],[4,1]]
Output: [0,2,3,1]

Input: tasks = [[7,10],[7,12],[7,5],[7,4],[7,2]]
Output: [4,3,2,0,1]
"""
# idea:
# sort tasks by start_time, length
# start from the 1st job, add jobs into min heap by (start_time, original_idx) until curr_time
# pop heap, and update curr_time
# repeat

from heapq import heappush, heappop
class Solution:
    def getOrder(self, tasks):
        res = []
        h = []
        sorted_tasks = sorted([(task[0], task[1], i) for i, task in enumerate(tasks)], key=lambda x:x[0])  # start_time, last_time, idx
        curr_time = sorted_tasks[0][0]
        job_id = 0
        while len(res) < len(sorted_tasks):
            while job_id < len(sorted_tasks) and sorted_tasks[job_id][0] <= curr_time:
                heappush(h, (sorted_tasks[job_id][1], sorted_tasks[job_id][2], sorted_tasks[job_id][0])) # last_time, idx, start_time
                job_id += 1
            
            if h:
                curr_last_time, curr_idx, curr_start_time  = heappop(h)
                res.append(curr_idx)
                curr_time += curr_last_time
            else:
                if job_id < len(sorted_tasks):
                    # h is empty
                    curr_time = sorted_tasks[job_id][0] # go to the next start time
        
        return res

        
"""1477. Find Two Non-overlapping Sub-arrays Each With Target Sum
You are given an array of integers arr and an integer target.

You have to find two non-overlapping sub-arrays of arr each with a sum equal target. There can be multiple answers so you 
have to find an answer where the sum of the lengths of the two sub-arrays is minimum.

Return the minimum sum of the lengths of the two required sub-arrays, or return -1 if you cannot find such two sub-arrays.

1 <= arr[i] <= 1000  !! 

Input: arr = [3,2,2,4,3], target = 3
Output: 2
Explanation: Only two sub-arrays have sum = 3 ([3] and [3]). The sum of their lengths is 2.

Input: arr = [7,3,4,7], target = 7
Output: 2
Explanation: Although we have three non-overlapping sub-arrays of sum = 7 ([7], [3,4] and [7]), 
but we will choose the first and third sub-arrays as the sum of their lengths is 2.

Input: arr = [4,3,2,6,2,3,4], target = 6
Output: -1
Explanation: We have only one sub-array of sum = 6.
"""
# 1) create prefix sum, use a dict (cumsum -> idx) because all positive numbers! 
# 2) for i, track of min_len on left (idx of cumsum[i] - target), and update left + right (idx of cumsum[i] + target)
class Solution:
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        s, lsize, res = 0, float('inf'), float('inf')
        prefixSum = { 0: -1 }
        for i, val in enumerate(arr):
            s += val
            prefixSum[s] = i
            
        s = 0  # curr sum
        for i, val in enumerate(arr):
            s += val
            if s - target in prefixSum: # left can be found
                lsize = min(i - prefixSum[s - target], lsize)
            if s + target in prefixSum and lsize != float('inf'): # right soln exists
                rsize = prefixSum[s + target] - i
                res = min(res, rsize + lsize)
                
        return res if res != float('inf') else -1


"""489. Robot Room Cleaner
Given a robot cleaner in a room modeled as a grid.
Each cell in the grid can be empty or blocked.
The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 90 degrees.
When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it stays on the current cell.
Design an algorithm to clean the entire room using only the 4 given APIs shown below.

interface Robot {
  // returns true if next cell is open and robot moves into the cell.
  // returns false if next cell is obstacle and robot stays on the current cell.
  boolean move();

  // Robot will stay on the same cell after calling turnLeft/turnRight.
  // Each turn will be 90 degrees.
  void turnLeft();
  void turnRight();

  // Clean the current cell.
  void clean();
}

Input:
room = [
  [1,1,1,1,1,0,1,1],
  [1,1,1,1,1,0,1,1],
  [1,0,1,1,1,1,1,1],
  [0,0,0,1,0,0,0,0],
  [1,1,1,1,1,1,1,1]
],
row = 1, col = 3

Explanation:
All grids in the room are marked by either 0 or 1.
0 means the cell is blocked, while 1 means the cell is accessible.
The robot initially starts at the position of row=1, col=3.
From the top left corner, its position is one row below and three columns right.
"""


"""Robot clean up
/**
 * // This is the robot's control interface.
 * // You should not implement it, or speculate about its implementation
 * class Robot {
 *   public:
 *     // Returns true if the cell in front is open and robot moves into the cell.
 *     // Returns false if the cell in front is blocked and robot stays in the current cell.
 *     bool move();
 *
 *     // Robot will stay in the same cell after calling turnLeft/turnRight.
 *     // Each turn will be 90 degrees.
 *     void turnLeft();
 *     void turnRight();
 *
 *     // Clean the current cell.
 *     void clean();
 * };
 */
 
class Solution {
    
    void dfs(Robot &robot, set<pair<int, int>>& visited, int i, int j, int dir) {
        robot.clean();
        visited.insert({i, j});
        pair<int, int> directions[4] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        for (int k = 0; k < 4; k++) {
            int new_i = i + directions[dir].first;
            int new_j = j + directions[dir].second;
            if (visited.count({new_i, new_j}) == 0 && robot.move()) {
                dfs(robot, visited, new_i, new_j, dir);
                robot.turnRight();
                robot.turnRight();
                robot.move();
                robot.turnRight();
                robot.turnRight();
            }
            dir = (dir + 1) % 4;
            robot.turnRight();
        }
    }
    
public:
    void cleanRoom(Robot& robot) {
        set<pair<int, int>> visited;
        dfs(robot, visited, /*i=*/0, /*j=*/0, /*dir=*/0);
    }
};

"""



"""
############################################################################
Snapchat
############################################################################
"""

"""
Given an m * n matrix, find the sliding square submatrix mean.
比如：
input matrix=[
[2,2,3,0],
[3,1,3,6],
[5,0,7,0]
], and an integer k = 2, k代表submatrix的size，那么答案就是
[[2, 2.25, 3], [2.25, 2.75, 4]]
"""
import numpy as np
class Solution:
    def sliding_average(self, matrix, k):
        matrix = np.array(matrix)
        kernel = np.array([[1/k**k for _ in range(k)] for _ in range(k)])
        output = np.zeros((len(matrix)-k+1, len(matrix[0])-k+1))
        for i in range(len(output)):
            for j in range(len(output[0])):
                # m * n * k^2
                output[i, j] = np.sum(matrix[i:i+k, j:j+k] * kernel)
        return output

sol=Solution()
sol.sliding_average([[2,2,3,0], [3,1,3,6], [5,0,7,0]], 2)
# improve time complexity
class Solution:
    def sliding_average(self, matrix, k):
        matrix = np.array(matrix)
        output = np.zeros((len(matrix)-k+1, len(matrix[0])-k+1))
        cumsum_mat = self.get_cumsum(matrix)
        for i in range(len(output)):
            for j in range(len(output[0])):
                # m * n
                # cumsum_mat[i, j]: sum(matrix[:i, :j])
                # output[i, j] = sum(matrix[i:i+k, j:j+k])
                # = sum(matrix[:i+k, :j+k]) - sum(matrix[:i, :j+k]) - sum(matrix[:i+k, :j]) + sum(matrix[:i, :j])
                output[i, j] = cumsum_mat[i+k, j+k] - cumsum_mat[i, j+k] - cumsum_mat[i+k, j] + cumsum_mat[i, j]
        return output / k**2
    
    def get_cumsum(self, matrix):
        mat = [[0 for _ in range(len(matrix[0])+1)] for _ in range(len(matrix)+1)]
        for i in range(1, len(mat)):
            for j in range(1, len(mat[0])):
                mat[i][j] = mat[i-1][j] + mat[i][j-1] - mat[i-1][j-1] + matrix[i-1][j-1]
        return np.array(mat)


"""1498. Number of Subsequences That Satisfy the Given Sum Condition
You are given an array of integers nums and an integer target.

Return the number of non-empty subsequences of nums such that the sum of the minimum and maximum 
element on it is less or equal to target. Since the answer may be too large, return it modulo 109 + 7.

Input: nums = [3,5,6,7], target = 9
Output: 4
Explanation: There are 4 subsequences that satisfy the condition.
[3] -> Min value + max value <= target (3 + 3 <= 9)
[3,5] -> (3 + 5 <= 9)
[3,5,6] -> (3 + 6 <= 9)
[3,6] -> (3 + 6 <= 9)

Input: nums = [3,3,6,8], target = 10
Output: 6
Explanation: There are 6 subsequences that satisfy the condition. (nums can have repeated numbers).
[3] , [3] , [3,3], [3,6] , [3,6] , [3,3,6]

Hint: Since order of the elements in the subsequence doesn’t matter, we can sort the input array.
Very similar to two sum, we use two pointers (i, j) to maintain a window, s.t. nums[i] +nums[j] <= target.
Then fix nums[i], any subset of (nums[i+1~j]) gives us a valid subsequence, 
thus we have 2^(j-(i+1)+1) = 2^(j-i) valid subsequence for window (i, j).
"""
class Solution:
    def numSubseq(self, nums, target: int) -> int:
        nums.sort()
        i, j = 0, len(nums) - 1
        res = 0
        mod = 10**9 + 7
        while i <= j:
            if nums[i] + nums[j] > target:
                j -= 1
            else:
                # always include nums[i], For each elements in the subarray A[i+1] ~ A[j]
                # we can pick or not pick,
                res += 2 ** (j-i)  # pow(2, j-i, mod) is much faster
                res = res % mod
                i += 1
        return res

sol=Solution()
sol.numSubseq([3,5,6,7], 9)
sol.numSubseq([2,3,3,4,6,7], 12) # 61


""" 按频率对数组排序
Example:
input: [1, 3, 1, 1, 4, 2, 2, 3]
output: [1, 1, 1, 3, 3, 2, 2, 4]
Have an O(nlogn) solution, but needs one O(n) solution

bucket sort: index as freq
"""


"""241. Different Ways to Add Parentheses 添加括号的不同方式 
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

DFS + memo
dfs("2*3-4*5") = 
dfs("2") * dfs ("3-4*5")
dfs("2*3") - dfs ("4*5")
dfs("2*3-4") * dfs ("5")

T(n) = T(1)*T(n-1) + ... + T(n-1)*T(1) (no memo)
"""
from functools import lru_cache
class Solution:
    @lru_cache(None)
    def diffWaysToCompute(self, expression: str):
        res = []
        for i in range(len(expression)):
            if expression[i] in ["+", "-", "*"]:
                left = self.diffWaysToCompute(expression[:i])
                right = self.diffWaysToCompute(expression[i+1:])
                for l in left:
                    for r in right:
                        if expression[i] == "+":
                            curr_res = l + r
                        elif expression[i] == "-":
                            curr_res = l - r
                        else:
                            curr_res = l * r
                        res.append(curr_res)

        return res if len(res) > 0 else [int(expression)]

sol=Solution()
sol.diffWaysToCompute("2*3-4*5")


"""
a list of tasks with project id, start time, and end time [[p_i, s_i, e_i],...], \
    为了让员工不无聊, 员工不能连续处理两个属于相同project的task. 
    问given 这个list，员工最多能处理的tasks有多少?
similar to 1235. Maximum Profit in Job Scheduling ? 
"""
# sort + dp
# dp[i] max at time i
tasks = [[1, 0, 2], [2, 1, 3], [1, 3, 4], [2, 5, 6]] # 3
class Solution:
    def max_tasks(self, tasks):
        return 
    

"""1926. Nearest Exit from Entrance in Maze

Return the number of steps in the shortest path from the entrance to the nearest exit, 
or -1 if no such path exists.

Input: maze = [["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], entrance = [1,2]
Output: 1
Explanation: There are 3 exits in this maze at [1,0], [0,2], and [2,3].
Initially, you are at the entrance cell [1,2].
- You can reach [1,0] by moving 2 steps left.
- You can reach [0,2] by moving 1 step up.
It is impossible to reach [2,3] from the entrance.
Thus, the nearest exit is [0,2], which is 1 step away.
"""
from collections import deque
class Solution:
    def nearestExit(self, maze, entrance) -> int:
        visited = set()
        q = deque()
        entrance = tuple(entrance)
        q.append((entrance[0], entrance[1], 0))
        visited.add(entrance)
        while q:
            for _ in range(len(q)):
                curr_i, curr_j, step = q.popleft()
                if self.is_exit(curr_i, curr_j, maze) and (curr_i, curr_j) != entrance:
                    return step
                for next_i, next_j in self.get_nb(curr_i, curr_j, maze):
                    if (next_i, next_j) not in visited and maze[next_i][next_j] != "+":
                        visited.add((next_i, next_j))
                        q.append((next_i, next_j, step+1))
        
        return -1
    
    def is_exit(self, curr_i, curr_j, maze):
        return curr_i in [0, len(maze)-1] or curr_j in [0, len(maze[0])-1]
    
    def get_nb(self, i, j, maze):
        res = []
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_i, next_j = di + i, dj + j
            if 0 <= next_i < len(maze) and 0 <= next_j < len(maze[0]):
                res.append((next_i, next_j))
        return res

sol=Solution()
sol.nearestExit([["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], [1,2])


"""Check if thief can get from bottom to top without triggering any sensors
https://leetcode.com/discuss/interview-question/1743143/airbnb-onsite

You are a thief standing in a room. The room has length L and width W. Your goal is to go from the bottom wall to 
anywhere on the top wall. The room has a set of sensors. Each sensor consists of x, y coordinates, and radius r. 
If you go within r of the center, the sensor is triggered and you get caught. You can move in any continuous path. 
Given the dimensions of the room L, W, Sensors. Return true if the thief can get from bottom to top without triggering any sensors? 
"""