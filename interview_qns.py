
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
        res_list = self.dfs(root)
        res = 0
        for i in res_list:
            curr_res = int(i) if i != "" else 0
            res += curr_res
        return res
        
    def dfs(self, node):
        if node is None:
            return [""]
        res = []
        if node.left is None and node.right is None:
            return [str(node.val)]
        if node.left:
            left_res = self.dfs(node.left)
            for i in left_res:
                res.append(str(node.val) + i)
        if node.right:
            right_res = self.dfs(node.right)
            for i in right_res:
                res.append(str(node.val) + i)
        return res

class Solution:
    def sumNumbers(self, root) -> int:
        self.res = 0
        self.dfs(root, 0)
        return self.res
        
    def dfs(self, node, curr_val):
        # add curr_val into self.res if leaf
        if node is None:
            return 0
        if node.left is None and node.right is None:
            self.res += curr_val + node.val
            return None
        if node.left:
            self.dfs(node.left, (curr_val + node.val) * 10)
        if node.right:
            self.dfs(node.right, (curr_val + node.val) * 10)
        
        return None


"""787. Cheapest Flights Within K Stops
There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.
You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.

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
# Runtime Error
from functools import lru_cache
class Solution:
    def addOperators(self, num: str, target: int):
        all_comb = dfs(num)
        res = []
        for comb in all_comb:
            eval_output = eval(comb)
            if eval_output== target:
                res.append(comb)
        return res
    
@lru_cache(None)
def dfs(num_str):
    # return all combinations in a list
    res = [] if num_str[0] == '0' and len(num_str) > 1 else [num_str]
    for i in range(1, len(num_str)):
        post_num = num_str[i:] # do not split any more
        if post_num[0] == '0' and len(post_num) > 1:
            continue
        pre_num_list = dfs(num_str[:i])
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

def dfs(rem_num, output, curr, last, target, res):
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
            dfs(rem_num[i:], output + "+" + pre_num, curr+val, val, target, res)
            dfs(rem_num[i:], output + "-" + pre_num, curr-val, -1*val, target, res)
            dfs(rem_num[i:], output + "*" + pre_num, (curr-last) + last * val, last * val, target, res)
        else:
            # at the first step, when output = ""
            dfs(rem_num[i:], pre_num, curr+val, val, target, res)
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
        for (int i = start; i <= n; ++i) {
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
[[1,2],[3,5],[6,7],[8,10],[12,16]]
[4,8]
[4,8]
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


"""381. Insert Delete GetRandom O(1) - Duplicates allowed
RandomizedCollection is a data structure that contains a collection of numbers, possibly duplicates (i.e., a multiset).
 It should support inserting and removing specific elements and also removing a random element.

Implement the RandomizedCollection class:

RandomizedCollection() Initializes the empty RandomizedCollection object.
bool insert(int val) Inserts an item val into the multiset, even if the item is already present. Returns true if the item is not present, false otherwise.
bool remove(int val) Removes an item val from the multiset if present. Returns true if the item is present, false otherwise. Note that if val has multiple occurrences in the multiset, we only remove one of them.
int getRandom() Returns a random element from the current multiset of elements. The probability of each element being returned is linearly related to the number of same values the multiset contains.
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

# 
A = [[(0, 1)], [(0, -1), (2, 3)]]
B = [[(0, 7)], [], [(2, 1)]]
colB = 3
C = [[0]*colB for _ in range(len(A))]
# C is not in sparse format, needs to convert
class Solution:
    def multiply(self, A, B, colB):
        C = [[0]*colB for _ in range(len(A))]
        for i in range(len(A)):
            for k, val in A[i]:
                # C[i][j] = A[i][k] * B[k][j]
                for j, valb in B[k]:
                    C[i][j] += val * valb
        return C


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
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.
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
Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

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

"""
############################################################################
Microsoft
############################################################################
"""
""" Implement heap and quicksort
"""
# two versions of partition
def partition(arr, l, r):
    pivot = l
    i = l
    for j in range(l+1, r+1):
        # if 
        if arr[j] < arr[pivot]:
            i += 1 # can ensure arr[i] < arr[pivot]
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[pivot], arr[i], = arr[i], arr[pivot]
    return i

def partition(nums, left, right):
    # choose a pivot, put smaller elements on right, larger on left
    # return rank of pivot (put on final r position)
    pivot = nums[left]
    l = left + 1
    r = right
    while l<=r:
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

def qs(arr, l, r):
    if l<r:
        pivot = partition(arr, l, r)
        qs(arr, l, pivot-1)
        qs(arr, pivot+1, r)
    return None

arr = [1,3,4,2,3,5,10,1]
qs(arr, 0, len(arr)-1)


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

