class ListNode:
    def __init__(self, val=0, next=None):
        self.val=val
        self.next=next

class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head
        dummy = ListNode(next=head)
        fast = slow = dummy
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
        
        mid = slow.next
        slow.next = None
        sorted_head = self.sortList(head)
        sorted_mid = self.sortList(mid)
        return self.merge(sorted_head, sorted_mid)
    
    def merge(self, h1, h2):
        dummy = ListNode()
        head = dummy
        while h1 and h2:
            if h1.val >= h2.val:
                head.next = h2
                head = head.next
                h2 = h2.next
            else:
                head.next = h1
                head = head.next
                h1 = h1.next
        if h1:
            head.next = h1
        else:
            head.next = h2
        return dummy.next


# 
class Solution:
    def reverseBetween(self, head, left: int, right: int):
        dummy = ListNode(next = head)
        node = dummy
        for i in range(right):
            if i == left - 1:
                pre_left_node = node
            node = node.next
        # identify four key nodes
        right_node = node
        right_node_next = right_node.next
        left_node = pre_left_node.next
        # disconnect
        pre_left_node.next = None
        right_node.next = None

        reversed_head = self.reverse_list(left_node)
        pre_left_node.next = reversed_head
        left_node.next = right_node_next
        return dummy.next
    
    def reverse_list(self, node):
        if not node:
            return node
        head = ListNode(next = node)
        tail = head.next
        while tail:
            temp = tail
            tail = tail.next
            temp.next = head
            head = temp
        return head
        

# [LeetCode] 281. Zigzag Iterator 之字形迭代器
vk = [[1,2,3], [4,5,6,7], [8,9]]
class Zigzag:
    def __init__(self, vk):
        self.vk = vk
        self.size = sum([len(v) for v in vk])
        self.k = len(vk)
        self.iter_vector = [iter(v) for v in vk]
        self.curr_idx = 0 # which vector to take value next
    
    def _next(self):
        if self.size == 0:
            return -1
        while True:
            try:
                next_num = next(self.iter_vector[self.curr_idx])
                self.size -= 1
                self.curr_idx = (self.curr_idx + 1) % self.k
                break
            except:
                self.curr_idx = (self.curr_idx + 1) % self.k
        return next_num
    
    def hasNext(self):
        if self.size == 0:
            return False
        return True

zig=Zigzag(vk)
res = []
for i in range(10):
    zig.hasNext()
    res.append(zig._next())
        

# [LeetCode] 1472. Design Browser History
class browserHistory:
    def __init__(self):
        self.curr_node = ListNode()
    

# 
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


# 394. Decode String !!!
decodeString("3[a]2[bc]")
decodeString("3[a2[c]]")
decodeString("2[abc]3[cd]ef")
decodeString("abc3[cd]xyz")
decodeString("2[abc]3[cd]ef")

class Solution:
    def decodeString(self, s: str) -> str:
        # left and right pair
        self.m = self.get_parenthesis_pairs(s)
        res = self.decode_dfs(s, 0, len(s)-1)
        return res
    
    def decode_dfs(self, s, start, end):
        temp_num = ''
        res = ''
        i = start
        while i <= end:
            if s[i].isalpha():
                res += s[i]
                i += 1
            elif s[i].isdigit():
                temp_num += s[i]
                i += 1
            elif s[i] == '[':
                rep_num = int(temp_num)
                close_idx = self.m[i]
                temp_res = self.decode_dfs(s, i+1, close_idx-1)
                res += temp_res * rep_num
                i = close_idx + 1
                temp_num = ''
        return res
    
    def get_parenthesis_pairs(self, s):
        m = dict()
        stack = []
        for i in range(len(s)):
            if s[i] == '[':
                stack.append(i)
            if s[i] == ']':
                m[stack.pop()] = i
        return m



sol=Solution()
sol.decodeString("2[abc]3[cd]")
sol.decodeString("2[abc]3[cd]ef")



class Solution:
    def scoreOfParentheses(self, s):
        self.m = self.get_parenthesis_pairs(s)
        return self.scoreOfSubstring(s, 0, len(s)-1)
    
    def scoreOfSubstring(self, s, start, end):
        if end - start == 1:
            return 1
        i = start
        res = 0
        while i <= end:
            if s[i] == '(':
                if s[i+1] == ')':
                    res += 1
                    i = i + 2
                else:
                    res += 2 * self.scoreOfSubstring(s, i+1, self.m[i]-1)
                    i = self.m[i] + 1
        
        return res
    
    def get_parenthesis_pairs(self, s):
        m = dict()
        stack = []
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            if s[i] == ')':
                m[stack.pop()] = i
        return m


sol=Solution()
sol.scoreOfParentheses("(())")
sol.scoreOfParentheses("()()")
sol.scoreOfParentheses("(())()")

from collections import defaultdict
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList):
        self.nb_dict = self.create_nb_graph(beginWord, wordList)
        print(self.nb_dict)
        self.visited = set([beginWord])
        res = []
        self.shortest_steps = float('Inf')
        self.findLaddersDFS([beginWord], endWord, res)
        return [i for i in res if len(i) == self.shortest_steps]
    
    def findLaddersDFS(self, path, target, res):
        if path[-1] == target:
            if len(path) <= self.shortest_steps:
                self.shortest_steps = min(self.shortest_steps, len(path))
                res.append(path)
            return None
        for next_word in self.nb_dict[path[-1]]:
            if next_word not in self.visited:
                self.visited.add(next_word)
                self.findLaddersDFS(path + [next_word], target, res)
                self.visited.remove(next_word)
        return None
        
    def create_nb_graph(self, beginWord, wordList):
        m = defaultdict(list)  # save neighbors
        full_word_list = [beginWord] + wordList
        for i in range(len(full_word_list)):
            for j in range(i+1, len(full_word_list)):
                if self.is_neighbor(full_word_list[i], full_word_list[j]):
                    if full_word_list[j] not in m[full_word_list[i]]:
                        m[full_word_list[i]].append(full_word_list[j])
                    if full_word_list[i] not in m[full_word_list[j]]:
                        m[full_word_list[j]].append(full_word_list[i])
        return m
    
    def is_neighbor(self, word1, word2):
        if len(word1) == len(word2):
            ct = 0
            for i in range(len(word1)):
                if word1[i] != word2[i]:
                    ct += 1
            return ct == 1
        return False 



wordList = ["hot","dot","dog","lot","log","cog"]
beginWord = "hit"
endWord = "cog"
sol=Solution()
sol.findLadders("hit", "cog", ["hot","dot","dog","lot","log","cog"])



class Solution:
    def solveSudoku(self, board):



from functools import lru_cache
class Solution:
    def wordBreak(self, s: str, wordDict):
        self.wordSet = set(wordDict)
        self.max_len = max([len(word) for word in wordDict])
        self.res = []
        self.dfs(s, 0, "")
        return [i.strip() for i in self.res]
    
    @lru_cache(None)
    def dfs(self, s, start_idx, out):
        if start_idx == len(s):
            self.res.append(out) 
            return None
        # if s[start_idx:] in self.wordSet:
        #     res.append(out + " " + s[start_idx:])
        for i in range(start_idx, len(s)):
            if i > start_idx + self.max_len:
                break
            if s[start_idx:i+1] in self.wordSet:
                self.dfs(s, i+1, out + " " + s[start_idx:i+1])
        return None

wordDict = ["cat","cats","and","sand","dog"]
sol = Solution()
sol.wordBreak("catsanddog", ["cat","cats","and","sand","dog"]) 
sol.wordBreak("catsand", ["cat","cats","and","sand","dog"]) 


import re
s = "{a,b}c{d,e}f"
class Solution:
    def expand(self, s):
        s_list = re.split("{|}", s)
        s_list = [i.split(",") for i in s_list if len(i)>0]
        res = []
        self.expand_list(s_list, 0, "", res)
        return res
    
    def expand_list(self, s_list, start_idx, out, res):
        if start_idx == len(s_list):
            res.append(out)
            return None
        
        if isinstance(s_list[start_idx], str):
            self.expand_list(s_list, start_idx+1, out+s_list[start_idx], res)
            return None

        # list
        for i in s_list[start_idx]:
            self.expand_list(s_list, start_idx+1, out+i, res)
        return None

sol=Solution()
sol.expand(s)
sol.expand("{a,b,c}d{e,f}")
sol.expand("{a,b,c}{e,f}")


# G[a] = [(b, 2)] -> a/b = 2
# G[b] = [(c, 2)] -> b/c = 3
class Solution:
    def eval_division(self, curr_letter, target, curr_val, G, visited):
        if curr_letter not in G:
            return None
        if curr_letter == target:
            self.res = curr_val
            return None
        
        visited.add(curr_letter)
        for next_letter, curr_div in G[curr_letter]:
            if next_letter not in visited:
                self.eval_division(next_letter, curr_val*curr_div, G, visitied)
        visited.remove(curr_letter)
        return None
        

"""[LeetCode] 57. Insert Interval 插入区间
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


class Solution:
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

A = [[(0, 1)], [(0, -1), (2, 3)]]
B = [[(0, 7)], [], [(2, 1)]]
colB = 3
C = [[0]*colB for _ in range(len(A))]

class Solution:
    def multiply(self, A, B, colB):
        C = [[0]*colB for _ in range(len(A))]
        for i in range(len(A)):
            for k, val in A[i]:
                # C[i][j] = A[i][k] * B[k][j]
                for j, valb in B[k]:
                    C[i][j] += val * valb
        return C



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
    
from collections import defaultdict
class Solution:
    def maxPoints(self, points):
        dup_m = self.count_dup(points)
        res = 0
        # print(dup_m)
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
            # print(m)
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
    def fullJustify(self, words, maxWidth):
        res = []
        return self.JustifyRemWords(words, "", res, maxWidth)
    
    def JustifyRemWords(self, rem_words, out, res, maxWidth):
        if len(rem_words) == 0 and len(out) == 0:
            return res
        if len(rem_words) == 0 or \
            (len(out) > 0 and len(out) + len(rem_words[0]) + 1 > maxWidth) or \
            (len(out) == 0 and len(rem_words[0]) > maxWidth):
            aligned_out = self.alignLine(out, maxWidth, last=(len(rem_words) == 0))
            res.append(aligned_out)
            return self.JustifyRemWords(rem_words, "", res, maxWidth)
        else:
            # print(out)
            out = out + " " + rem_words[0]
            return self.JustifyRemWords(rem_words[1:], out.strip(), res, maxWidth)
    
    def alignLine(self, line, maxWidth, last=False):
        words_in_out = line.split(" ")
        if len(words_in_out) == 1:
            return words_in_out[0] + " "*(maxWidth - len(words_in_out[0]))
        num_spaces = maxWidth - sum([len(w) for w in words_in_out])
        if last:
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

words = ["This", "is", "an", "example", "of", "text", "justification."]; maxWidth = 16


sol = Solution()
sol.fullJustify(words, 16)
sol.fullJustify(["What","must","be","acknowledgment","shall","be"], 16)
sol.fullJustify(["Listen","to","many,","speak","to","a","few."], 6)



# k means 

# randomly pick k points, use them  as centroids
# repeat: 
# assign other points into the k clusters based on distance to the centroids
# re-calculate centroids of each cluster
# stop when prev and next_centroid diff less than eps
import numpy as np 
# Size of dataset to be generated. The final size is 4 * data_size
data_size = 10
num_clusters = 3

# sample from Gaussians 
data1 = np.random.normal((5,5,5), (4, 4, 4), (data_size,3))
data2 = np.random.normal((4,20,20), (3,3,3), (data_size, 3))
data3 = np.random.normal((25, 20, 5), (5, 5, 5), (data_size,3))
data = np.concatenate((data1,data2, data3), axis = 0)

from numpy.random import randint
class Solution:
    def k_means(self, raw_data, k, niter, eps):
        init_centroids = self.random_pick_points(raw_data, k)
        # print(init_centroids)
        curr_centroids = init_centroids
        for i in range(niter):
            print(curr_centroids)
            curr_clusters = self.cluster_into_k(raw_data, curr_centroids)
            print(curr_clusters)
            next_centroids = self.calculate_centroids(raw_data, curr_clusters)
            print(next_centroids)
            centroids_diff = self.get_centroid_dist(next_centroids, curr_centroids)
            print("Iter {}: dist is: {}".format(i, centroids_diff))
            if centroids_diff < eps:
                break
            curr_centroids = next_centroids
        
        return curr_clusters

    def random_pick_points(self, raw_data, k):
        n_sample = raw_data.shape[0]
        idx = np.random.choice(raw_data.shape[0], k)
        return raw_data[idx, :]
    
    def cluster_into_k(self, raw_data, curr_centroids):
        shortest_dist = [float('Inf')] * len(raw_data)
        clusters = [-1] * len(raw_data)
        k = curr_centroids.shape[0]
        for rowi, row in enumerate(raw_data):
            for i in range(k):
                centroid = curr_centroids[i]
                curr_dist = self.get_dist(centroid, row)
                if curr_dist < shortest_dist[rowi]:
                    clusters[rowi] = i
                    shortest_dist[rowi] = curr_dist
        return np.array(clusters)
    
    def calculate_centroids(self, raw_data, curr_clusters):
        centroids = []
        k = len(np.unique(curr_clusters))
        for i in range(k):
            centroids.append(np.mean(raw_data[[idx for idx in curr_clusters if idx == i], :], axis=0))
        
        return np.array(centroids)
    
    def get_dist(self, arr1, arr2):
        return sum((arr1 - arr2) ** 2) / len(arr1)

    def get_centroid_dist(self, centroid1, centroid2):
        dist = 0
        for i in range(len(centroid1)):
            dist += self.get_dist(centroid1[i], centroid2[i])
        return dist

sol=Solution()
sol.k_means(data, k, 10, 0.01)




from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)



"""1235. Maximum Profit in Job Scheduling
"""
startTime = [1,2,3,4,6]; endTime = [3,5,10,6,9]; profit = [20,20,100,70,60]
sol = Solution()
sol.jobScheduling(startTime, endTime, profit)
import bisect 
class Solution:
    def jobScheduling(self, startTime, endTime, profit):
        job_triplet = sorted(zip(startTime, endTime, profit), key= lambda x:x[1])
        sorted_endTime = [job[1] for job in job_triplet]
        # job_triplet = self.create_triplet(startTime, endTime, profit)
        # job_triplet = sorted(job_triplet, key=lambda x:x[1])
        print(job_triplet)
        dp = [0] * (len(job_triplet) + 1)
        for i in range(len(job_triplet)):
            # print(dp)
            starti, endi, profiti = job_triplet[i]
            # if take job i
            prev_i = self.find_previous_endtime(starti, sorted_endTime[:i])
            # prev_i = bisect.bisect(sorted_endTime[:i], starti) - 1
            take_max_profit = profiti + dp[prev_i+1]
            not_take_max_profit = dp[i]
            dp[i+1] = max(take_max_profit, not_take_max_profit)
        return dp[-1]

    # def create_triplet(self, startTime, endTime, profit):
    #     res = []
    #     for i, j, k in zip(startTime, endTime, profit):
    #         res.append((i, j, k))
    #     return res
    
    def find_previous_endtime(self, ti, endTime):
        # find idx of the first endTime <= ti
        low, high = 0, len(endTime)
        while low < high:
            mid = low + (high - low) // 2
            if endTime[mid] <= ti:
                low = mid + 1
            else:
                high = mid
        return low - 1


"""1335. Minimum Difficulty of a Job Schedule
"""
from functools import lru_cache
class Solution:
    def minDifficulty(self, jobDifficulty, d: int):
        if len(jobDifficulty) < d:
            return -1
        self.jobDifficulty = jobDifficulty
        return self.RemMinDifficulty(0, d)
    
    @lru_cache(None)
    def RemMinDifficulty(self, start, rem_d):
        if rem_d == 1:
            return max(self.jobDifficulty[start:])
        res = float('Inf')
        max_so_far = float('-Inf')
        for i in range(start, len(self.jobDifficulty)):
            if len(self.jobDifficulty) - i < rem_d:
                break
            max_so_far = max(max_so_far, self.jobDifficulty[i])
            curr_res = max_so_far + self.RemMinDifficulty(i+1, rem_d-1)
            res = min(res, curr_res)

        return res


from functools import lru_cache
class Solution:
    def minDifficulty(self, jobDifficulty, d: int):
        if len(jobDifficulty) < d:
            return -1
        self.jobDifficulty = jobDifficulty
        self.res = float('Inf')
        self.minRemDifficulty(0, 0, d)
        return self.res
    
    @lru_cache(None)
    def minRemDifficulty(self, start, cum_diff, rem_d):
        if start == len(self.jobDifficulty) and rem_d == 0:
            self.res = min(self.res, cum_diff)
            return None
        for i in range(start, len(self.jobDifficulty)):
            self.minRemDifficulty(i+1, cum_diff + max(self.jobDifficulty[start:i+1]), rem_d-1)
            if len(self.jobDifficulty) - i < rem_d:
                break
        return None

# jobDifficulty = [6,5,4,3,2,1], d = 2
sol=Solution()
sol.minDifficulty([6,5,4,3,2,1], 2)
sol.minDifficulty([6,5,4,3,2,1], 2)

s = "abcdeca"; k = 2
class Solution:
    def isValidPalindrome(self, s, k):
        return self.isValidPalindromeSubset(s, 0, len(s)-1, k)
    
    @lru_cache(None)
    def isValidPalindromeSubset(self, s, i, j, k):
        # s[i:j+1]
        if i == j:
            return True
        left, right = i, j
        while left < right:
            if s[left] != s[right]:
                if k == 0:
                    return False
                return self.isValidPalindromeSubset(s, left+1, right, k-1) or \
                    self.isValidPalindromeSubset(s, left, right-1, k-1)
            left += 1
            right -= 1
        return True

sol=Solution()
sol.isValidPalindrome("abcdeca", 2)
sol.isValidPalindrome("abcdecda", 2)


"""[LeetCode] Accounts Merge 账户合并
"""
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], 
["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
# Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  
# ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
from collections import defaultdict
sol = Solution()
sol.accountsMerge(accounts)

class Solution:
    def accountsMerge(self, accounts):
        uf = UF(len(accounts))

        email2account = dict()
        for account_id, account in enumerate(accounts):
            for i in range(1, len(account)):
                if account[i] in email2account:
                    uf.Union(account_id, email2account[account[i]])
                else:
                    email2account[account[i]] = account_id
        print(uf.parents)
        account2email = defaultdict(list)
        for i in range(len(accounts)):
            rootid = uf.Find(i)
            account2email[rootid] += accounts[i][1:]
        print(account2email)
        res = []
        for k in account2email.keys():
            res.append([accounts[k][0]] + sorted(list(set(account2email[k]))))
        
        return res

class UF:
    def __init__(self, n):
        self.parents = list(range(n))
        self.weights = [1]*n
    
    def Find(self, i):
        if self.parents[i] != i:
            current_parent = self.parents[i]
            self.parents[i] = self.Find(current_parent)
        return self.parents[i]
    
    def Union(self, i, j):
        root_i = self.Find(i)
        root_j = self.Find(j)
        if root_i == root_j:
            return None
        if self.weights[root_i] >= self.weights[root_j]:
            self.parents[root_j] = root_i
            self.weights[root_i] += self.weights[root_j]
        else:
            self.parents[root_i] = root_j
            self.weights[root_j] += self.weights[root_i]
        return None


accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], 
["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]


nb_dict = dict(email: {"name": , "nb": set()})
email_list = nb_dict.keys()
visited = set()
res = []
for email in email_list:
    if email not in visited:
        curr_email_list = bfs(email, nb_dict)
        res.append([nb_dict[email]["name"]] + sorted(curr_email_list))


# WS II
res = []
for i:
    for j:
        dfs(b, i, j, "", visited, trie.root, res)


def dfs(b, i, j, word, visited, node, res):
    if node.isWord:
        res.append(word)
        node.isWord = False 
        return None
    if b[i][j] not in node.children():
        return None
    if (i, j) in visited:
        return None
    visited.add((i, j))
    for next_i, next_j in get_nb((i, j, b)):
        dfs(b, next_i, next_j, word+b[i][j], visited, node.children[b[i][j]], res)
    visited.remove((i, j))
    return None
        


heights = [2,1,5,6,2,3]; 
class Solution:
    def largestRectangleArea(self, heights) -> int:
        heights = heights + [0]
        s = [] # stack mono increasing
        res = 0
        for i, h, in enumerate(heights):
            print(s)
            if len(s) == 0 or h > s[-1][1]:
                s.append([i, h])
            else:
                while len(s) > 0 and s[-1][1] >= h:
                    prev_i, prev_h = s.pop()
                    if len(s) > 0:
                        prev_area = (i-1-s[-1][0]) * prev_h
                    else:
                        prev_area = i * prev_h
                    res = max(prev_area, res)
                s.append([i, h])
        return res

sol=Solution()
sol.largestRectangleArea([2,1,5,6,2,3])
sol.largestRectangleArea([2,4])
sol.largestRectangleArea([2,1,2])
sol.largestRectangleArea([4,2,0,3,2,5])



"""[LeetCode] 42. Trapping Rain Water 收集雨水
"""
[0,1,0,2,1,0,1,3,2,1,2,1]
class Solution:
    def trap(self, height):
        res = 0
        s = []
        for i, h in enumerate(height):
            if len(s) == 0 or height[i] <= s[-1][1]:
                s.append([i, h])
            else:
                while len(s) > 0 and height[i] > s[-1][1]:
                    prev_i, prev_h = s.pop() # bottom
                    if len(s) == 0:
                        # no height on left higher than prev_h
                        break
                    curr_width = i - s[-1][0] - 1
                    curr_depth = min(height[i], height[s[-1][0]]) - prev_h
                    # print(curr_width * curr_depth)
                    res += curr_width * curr_depth
                s.append([i, h])
        return res

sol=Solution()
sol.trap([0,1,0,2,1,0,1,3,2,1,2,1])

"""739. Daily Temperatures
"""
class Solution:
    def dailyTemperatures(self, temperatures):
        s = []
        res = [0] * len(temperatures)
        for i, temp in enumerate(temperatures):
            if len(s) == 0 or temp <= s[-1][1]:
                s.append([i, temp])
            else:
                while len(s) > 0 and temp > s[-1][1]:
                    prev_i, prev_temp = s.pop()
                    res[prev_i] = i - prev_i
                s.append([i, temp])
        return res



"""
1. sort: (x, h, type)
 by: x, h, if type == -1 large h first else small h first
x, h*type, increasing
2. if enter: if >max then res.append, heap.append
3. if leaving: remove, if heap.max == res[-1] then res.append

"""
import heapq
buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
class Solution:
    def getSkyline(self, buildings):
        active = set()
        start_points = [(idx, l, h, -1) for idx, (l, r, h) in enumerate(buildings)]
        end_points = [(idx, r, h, 1) for idx, (l, r, h) in enumerate(buildings)]
        buildings = sorted(start_points + end_points, key=lambda x: (x[1], x[2]*x[3]))
        # print(buildings)
        q = [[0, -1]]   # max heap on height, 0 in case all pop out, then record 0
        active.add(-1) # -1 will never be pop out
        res = []
        for (idx, x, h, status) in buildings:
            # print(q)
            if status == -1:
                active.add(idx)
            else:
                active.remove(idx)
            
            if status == -1:
                if len(q) == 0 or h > -1 * q[0][0]:
                    res.append([x, h])
                heapq.heappush(q, [-1*h, idx])
            else:
                while len(q) > 0 and q[0][1] not in active:
                    heapq.heappop(q)
                if res[-1][1] > -1 * q[0][0]:
                    res.append([x, -1 * q[0][0]])
        return res


sol=Solution()
sol.getSkyline(buildings)


"""368. Largest Divisible Subset
"""
nums = [1,2,4,8]

class Solution:
    def largestDivisibleSubset(self, nums):
        dp = [1] * len(nums)  # len ends with i
        lds = [[num] for num in nums]
        print(lds)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    if dp[i] < dp[j] + 1:
                        dp[i] = dp[j] + 1
                        lds[i] = lds[j] + [nums[i]]
        print(lds)
        max_id = [i for i, val in enumerate(dp) if val == max(dp)]
        return lds[max_id[0]]


sol=Solution()
sol.largestDivisibleSubset([1,2,3,6])



"""300 Longest Increasing Subsequence (接龙型dp)
"""
nums = [1,3,5,4,7]
class Solution:
    def findNumberOfLIS(self, nums) -> int:
        return 

"""45 Jump Game II
"""
nums = [2,3,0,1,4]
class Solution:
    def jump(self, nums) -> int:
        if len(nums) == 1:
            return 0
        step = 1
        maxCurr = nums[0]
        maxNext = nums[0]
        i = 0
        while i < len(nums):
            if i > maxCurr:
                step += 1
                maxCurr = maxNext
            maxNext = max(maxNext, i+nums[i])
            if maxCurr >= len(nums) - 1:
                return step
            i += 1
            if i > maxNext:
                return -1
        return -1


from functools import lru_cache
class Solution:
    @lru_cache(None)
    def minCut(self, s: str) -> int:
        if self.isPalid(s):
            return 0
        res = float('Inf')
        for i in range(1, len(s)):
            if self.isPalid(s[:i]):
                res = min(res, 1 + self.minCut(s[i:]))

        return res
    
    @lru_cache(None)
    def isPalid(self, s):
        left, right = 0, len(s)-1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        
        return True


sol=Solution()
sol.minCut("ab")
sol.minCut("aab")
sol.minCut("ababen")
sol.minCut("cabababcbc")
ss="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
sol.minCut(ss)

dp = [range(n+1)]  # dp[i]: cut for s[:i]
for i in range(n):


from functools import lru_cache
class Solution:
    def minCut(self, s: str) -> int:
        # dp[i]: cut for s[:i]
        m = dict()
        dp = [0] + list(range(len(s)))
        for i in range(1, len(dp)):
            if self.isPalid(s[:i], m):
                dp[i] = 0
                continue
            for j in range(i):
                if self.isPalid(s[j:i], m):
                    dp[i] = min(dp[i], dp[j] + 1)
        # print(dp)
        return dp[-1]

    def isPalid(self, s, m):
        if s in m:
            return m[s]
        left, right = 0, len(s)-1
        while left < right:
            if s[left] != s[right]:
                m[s] = False
                return m[s]
            left += 1
            right -= 1
        
        m[s] = True
        return m[s]

class Solution:
    @lru_cache(None)
    def numDistinct(self, s: str, t: str) -> int:
        if len(t) == 0:
            return 1
        if len(s) == 0:
            return 0
        if s[0] == t[0]:
            return self.numDistinct(s[1:], t[1:]) + \
                self.numDistinct(s[1:], t)
        return self.numDistinct(s[1:], t)


sol=Solution()
sol.numDistinct("rabbbit", "rabbit")


class Solution:
    @lru_cache(None)
    def numDecodings(self, s: str) -> int:
        if len(s) > 0 and s[0] == '0':
            return 0
        if len(s) <= 1:
            return 1
        if int(s[:2]) <= 26:
            return self.numDecodings(s[1:]) + self.numDecodings(s[2:])
        else:
            return self.numDecodings(s[1:])

sol=Solution()
sol.numDecodings("226")
sol.numDecodings("1023412342")
sol.numDecodings("301")
sol.numDecodings("12")


class Solution:
    @lru_cache(None)
    def numDecodings(self, s: str) -> int:
        if len(s) > 0 and s[0] == '0':
            return 0
        if len(s) == 0:
            return 1
        if len(s) == 1:
            res = self.numDecodings(s[1:]) * self.get_number(s[:1]) 
        else:
            res = self.numDecodings(s[1:]) * self.get_number(s[:1]) + \
                self.numDecodings(s[2:]) * self.get_number(s[:2])
        return res % (10**9 + 7)

    def get_number(self, s):
        # when len(s) <= 2
        if s[0] == '0':
            return 0
        if len(s) == 1:
            if s == '*':
                return 9
            else:
                return 1
        if len(s) == 2:
            # **, *d, d*, dd
            if s[0] == '*':
                if s[1] == '*': # cannot be 01, 02, ...
                    # 11 - 26 \ 20
                    return 15
                else: # *d
                    return 2 if s[1] <= '6' else 1
            else: # d* or dd
                if s[1] == '*':
                    return 9 if s[0] == '1' else (6 if s[0] == '2' else 0)
                else: # dd
                    return 1 if s <= '26' else 0


sol=Solution()
sol.numDecodings("226")
sol.numDecodings("1**8638")
sol.numDecodings("*********")
sol.numDecodings("2*")   
sol.get_number("2*")
sol.get_number(ss)
        
class Solution:
    def change(self, amount: int, coins) -> int:
        self.coins = sorted(coins)
        return self.dfs(amount, 0)
    
    @lru_cache(None)
    def dfs(self, rem_amount, start_idx):
        if rem_amount == 0:
            return 1
        if start_idx == len(self.coins):
            return 0
        res = 0
        for i in range(start_idx, len(self.coins)):
            if rem_amount - self.coins[i] >= 0:
                res += self.dfs(rem_amount-self.coins[i], i)
            else:
                break
        return res


sol=Solution()
sol.change(5, [1,2,5])


arr = [1,4,1,5,7,3,6,1,9,9,3], k = 4
from functools import lru_cache
class Solution:
    def maxSumAfterPartitioning(self, arr, k: int) -> int:
        res = self.maxSumhelper(tuple(arr), 0, k)
        return res
    
    @lru_cache(None)
    def maxSumhelper(self, arr, idx, k):
        if len(arr) - idx <= k:
            return self.get_maxsum(arr[idx:])
        res = 0
        for i in range(1, k+1):
            if i + idx > len(arr):
                print("Error")
            # print(self.arr[idx:idx+i])
            res = max(res, self.get_maxsum(arr[idx:idx+i]) + self.maxSumhelper(arr, idx+i, k))
        return res

    def get_maxsum(self, arr):
        return max(arr) * len(arr)


sol=Solution()
sol.maxSumAfterPartitioning([1,15,7,9,2,5,10], 3)
sol.maxSumAfterPartitioning([1,4,1,5,7,3,6,1,9,9,3], 4)
sol.maxSumAfterPartitioning([1,1,7,7], 2)
sol.maxSumAfterPartitioning([1,1,7], 2)
sol.maxSumAfterPartitioning([1,7], 2)
sol.maxSumhelper(0, 2)
sol.arr


s = "mississippi"
p = "mis*is*p*."
s = "aab"
p = "c*a*b"

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        return self.isMatchhelper(s, p, 0, 0)

    @lru_cache(None)
    def isMatchhelper(self, s, p, i, j):
        if i == len(s) and j == len(p):
            return True
        if j == len(p):
            return False
        if j == len(p) - 1 or p[j+1] != '*':
            if i == len(s):
                return False
            if p[j] == '.' or (p[j] != '.' and s[i] == p[j]):
                return self.isMatchhelper(s, p, i+1, j+1)
            else:
                return False
        else: # j < len(p) - 1 and p[j+1] == '*'
            if p[j] == '.':
                for idx in range(i, len(s)+1):
                    # print(idx)
                    if self.isMatchhelper(s, p, idx, j+2):
                            return True
                return False
            else:
                if i == len(s) or s[i] != p[j]:
                    return self.isMatchhelper(s, p, i, j+2)
                else: # s[i] == p[j]
                    for idx in range(i, len(s)+1):
                        # print(idx)
                        if idx == i or idx == len(s) or (idx > i and s[idx] == s[idx-1]):
                            # need to include len(s), i.e. the last one
                            # e.g., (aa, a*)
                            if self.isMatchhelper(s, p, idx, j+2):
                                return True
                        else: # s[idx] != s[idx-1]
                            return self.isMatchhelper(s, p, idx, j+2)


sol=Solution()
sol.isMatch("aa", "a")
sol.isMatch("aa", "a*")
sol.isMatch("ab", ".*")
sol.isMatch("ab", ".*c")
sol.isMatch("a", "ab*")
sol.isMatchhelper("ab", ".*c", 2, 2)
sol.isMatchhelper("a","ab*", 1,1 )
sol.isMatch("aab", "c*a*b")
sol.isMatch("mississippi", "mis*is*p*.")
        
