# 560. Subarray Sum Equals K
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

# 523. Continuous Subarray Sum
# 若数字a和b分别除以数字c，若得到的余数相同，那么 (a-b) 必定能够整除c
# 注意k=0的时候 无法取余
class Solution:
    def checkSubarraySum(self, nums, k: int) -> bool:
        m = dict()
        cumsum = 0
        m[0] = -1  # !!!
        for i, num in enumerate(nums):
            cumsum = cumsum + num
            curr_key = cumsum % k if k!=0 else cumsum
            if curr_key in m:
                # never update m[curr_key] because need the longer subarray
                if i - m[curr_key] > 1:
                    # length at least 2
                    return True
                    # print(curr_key)
            else:
                # note, should not update if curr_key in m!!!
                m[curr_key] = i
        
        return False

# 109. Convert Sorted List to Binary Search Tree
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

# LeetCode 426. Convert Binary Search Tree to Sorted Doubly Linked List
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

# UF
def find(self, x):
    # link x's parent to the root, return
    if x!= self.parents[x]:
        # not root
        curr_parent = self.parents[x]
        self.parents[x] = self.find(curr_parent)
    return self.parents[x]


# 636 Exclusive Time of Functions
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

"""282. Expression Add Operators
"""
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

"""140. Word Break II
Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
"""
class Solution:
    def wordBreak(self, s: str, wordDict):
        m = defaultdict(list)  # 
        max_len = max([len(word) for word in wordDict])
        wordSet = set(wordDict)
        return self.dfs(s, m, wordSet, max_len)

    def dfs(self, rem_str, m, wordSet, max_len):
        # question: what to return if rem_str = ""
        # option 1: avoid that by adding if rem_str in wordSet -> curr_res.append(rem_str) (below)
        # option 2: if "" return [""] but need to handle :
        #   rem_str[:i+1] + " " + rem_item when rem_item="", because extra space is in the end
        if rem_str in m:
            # avoid duplicated calculation
            return m[rem_str]
        curr_res = []
        if rem_str in wordSet:
            # if rem_str in dict, add it to the answer array
            curr_res.append(rem_str)
        
        # further split rem_str into word + m[rest]
        for i in range(max_len):
            if rem_str[:i+1] in wordSet:
                rem_list = self.dfs(rem_str[i+1:], m, wordSet, max_len)
                # ["cats and dog","cat sand dog"]
                for rem_item in rem_list:
                    # 
                    curr_res.append(rem_str[:i+1] + " " + rem_item)
        
        m[rem_str] = curr_res
        return m[rem_str]

"""129. Sum Root to Leaf Numbers
Input: root = [1,2,3]
Output: 25
"""
class Solution:
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

"""1868 - Product of Two Run-Length Encoded Arrays
encode = [[num1, repeat1], [num2, repeat2], ...]
Input: encoded1 = [[1,3],[2,3]], encoded2 = [[6,3],[3,3]]
Output: [[6,6]]
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

"""138. Copy List with Random Pointer
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

"""LeetCode 616 - Add Bold Tag in String
Input:  s = "abcxyz123", dict = ["abc","123"]
Output: "<b>abc</b>xyz<b>123</b>"
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

"""[LeetCode] 42. Trapping Rain Water 收集雨水
# left[i]: max height on the left of height[i]
# right[i]:max height on the right of height[i]
# the water that point i can contribute is: min(l, r) - height[i]
# left[i] = max(height[i], left[i-1])
# right[i] = max(height[i], right[i+1])
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

# 273. Integer to English Words
class Solution:
    def numberToWords(self, num: int) -> str:
        to19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve ' \
           'Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
        tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
        thousand = 'Thousand Million Billion'.split()
        
        def word(num, idx=0):
            if num == 0:
                return []
            if num < 20:
                return [to19[num-1]]
            if num < 100:
                return [tens[num//10-2]] + word(num%10)
            if num < 1000:
                return [to19[num//100-1]] + ['Hundred'] + word(num%100)

            p, r = num//1000, num%1000
            space = [thousand[idx]] if p % 1000 !=0 else []
            return  word(p, idx+1) + space + word(r)
        return ' '.join(word(num, 0)) or 'Zero'


# 398. Random Pick Index
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

# LeetCode 1216. Valid Palindrome III
# dp[i][j] = min k needed to remove, to make s[i:j+1] palin
def isValidPalindrome(s, k):
    if len(s) == 0:
        return True
    n = len(s)
    dp = [[float('inf') for _ in range(len(s))] for _ in range(len(s))]
    for i in range(n):
        for j in range(n):
            if i >= j:
                dp[i][j] = 0
    for i in list(range(n))[::-1]:
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1])
    
    return  dp[0][n-1] <= k

# Leetcode 1060 Missing Element in Sorted Array
class Solution:
    def missingElement(self, nums, k):
        while low < high:
            mid = low + (high - low) // 2
            num_miss_mid = self.get_num_miss(nums, mid)
            if num_miss_mid >= k:
                high = mid
            else:
                low = mid + 1
        return nums[low-1] + k - self.get_num_miss(nums, low-1)
    
    def get_num_miss(self, nums, idx):
        # number of missing on left of nums[idx]
        return nums[idx] - nums[0] + 1 - (idx + 1)

# 380. Insert Delete GetRandom O(1)
class RandomizedSet:
    def remove(self, val: int) -> bool:
        if val not in self.d:
            return False
        # swap last with the val to delete
        val_idx = self.d[val]
        self.vector[val_idx], self.vector[-1] = self.vector[-1], self.vector[val_idx]
        self.d[self.vector[val_idx]] = val_idx
        del self.d[val]
        self.vector.pop()
        self.idx -= 1
        return True


# 416. Partition Equal Subset Sum
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
            if target - nums[i] == 0:
                return True
            if nums[i] < target:
                if self.dfs(nums, i+1, target-nums[i], memo):
                    return True
            else:
                break
        memo[target] = False
        return False

# 116. Populating Next Right Pointers in Each Node
class Solution:
    def connect(self, root):
        if root is None:
            return None
        if root.left is not None:
            root.left.next = root.right
            if root.next is None:
                root.right.next = None
            else:
                root.right.next = root.next.left
        self.connect(root.left)
        self.connect(root.right)
        return root

# 333. Largest BST Subtree 最大的二分搜索子树
class Solution:
    def __init__():
        self.res = 0
    
    def largestBSTSubtree(self, root):
        self.GetBSTSize(root, float('-Inf'), float('Inf'))
        return self.res

    def GetBSTSize(self, node, mn, mx):
        if node is None:
            return 0
        if node.val <= mn or node.val >= mx:
            curr_res = -1 # not a BST            
        left = self.GetBSTSize(node.left, mn, node.val)
        right = self.GetBSTSize(node.right, node.val, mx)
        if left >= 0 and right >= 0:
            self.res = max(self.res, left + right + 1)
            return left + right + 1
        # node is not BST but still
        return curr_res

# 10 regular expression
if (j > 1 && p[j - 1] == '*') {
    dp[i][j] = dp[i][j - 2] || (i > 0 && (s[i - 1] == p[j - 2] || p[j - 2] == '.') && dp[i - 1][j]);
} else {
    dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
}