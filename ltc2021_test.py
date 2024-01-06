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


"""[LeetCode] 146. LRU Cache 最近最少使用页面置换缓存器 !!! (linked list solution)
"""
class Node:
    def __init__(self, key=-1, val=-1, nxt=None, prev=None):
        self.key, self.val, self.next, self.prev = key, val, nxt, prev

class LinkedList:
    def __init__(self):
        self.head = self.tail = Node()
        self.head.next, self.tail.prev = self.tail, self.head
    
    def remove(self, node):
        prev_node, next_node = node.prev, node.next
        prev_node.next, next_node.prev = next_node, prev_node
        return None
    
    def appendleft(self, node):
        head_next = self.head.next
        self.head.next, head_next.prev, node.prev, node.next = \
            node, node, self.head, head_next
        return None

    def pop(self):
        if self.tail.prev != self.head:
            node = self.tail.prev
            self.remove(node)
            return node
    
    def popleft(self):
        if self.head.next != self.tail:
            node = self.head.next
            self.remove(node)
            return node


class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.key2node = dict()
        self.l = LinkedList()
        
    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.key2node:
            return -1
        node = self.key2node[key]
        self.l.remove(node)
        self.l.appendleft(node)  # add from head side
        return node.val
        
    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.key2node:
            self.l.remove(self.key2node[key])
            self.key2node[key] = Node(key=key, val=value)
            self.l.appendleft(self.key2node[key])
        else:
            self.key2node[key] = Node(key=key, val=value)
            self.l.appendleft(self.key2node[key])
            if len(self.key2node) > self.capacity:
                key_to_del = self.l.pop().key
                del self.key2node[key_to_del]
        return None


# Definition for a binary tree node.
class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self.in_order(root) # recursively push left child

    def in_order(self, node):
        temp_stack = []
        curr = node
        while curr or temp_stack:
            if curr:
                temp_stack.append(curr)
                curr = curr.left
            else:
                curr = temp_stack.pop()
                # self.stack.append(curr)
                print(curr.val)
                curr = curr.right
        print(self.stack)
        return 

    def next(self) -> int:
        next_node = self.stack.pop()
        return next_node.val
        
    def hasNext(self) -> bool:
        return len(self.stack) > 0


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

sol=BSTIterator(root)
sol.inOrder(root)

s = "4(2(3)(1))(6(5))"
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def str2tree(self, s):
        self.m = self.match_pair_idx(s)
        print(self.m)
        root = self.str2treehelper(s, 0, len(s)-1)
        return root
    
    def str2treehelper(self, s, start, end):
        if start > end:
            return None
        root = TreeNode(int(s[start]))
        if start == end:
            return root
        left = self.str2treehelper(s, start+2, self.m[start+1]-1)
        root.left = left
        if end > self.m[start+1]:
            right = self.str2treehelper(s, self.m[start+1]+2, end-1)
            root.right = right
        return root

    def match_pair_idx(self, s):
        d = dict()
        stack = []
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            elif s[i] == ')':
                d[stack.pop()] = i
            else:
                continue
        return d

sol=Solution()
root = sol.str2tree("4(2(3)(1))(6(5))")
inorder(root)
def inorder(root):
    if root is None:
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)
    return 



class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordSet = set(wordDict)
        dp = [False] * len(s) # s[:i+1] 
        for i in range(len(s)):
            if s[:i+1] in wordSet:
                dp[i] = True
                continue
            for j in range(i):
                if s[j+1:i+1] in wordSet and s[:j+1]:
                    dp[i] = True
        return dp[-1]
                
sol = SOlu
"catsandog"
["cats","dog","sand","and","cat"]
        
"""1110. Delete Nodes And Return Forest
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def delNodes(self, root, to_delete):
        to_del_set = set(to_delete)
        res = []
        self.delNodesHelper(root, None, None, to_del_set, res)
        return res
    
    def delNodesHelper(self, node, prev, from_left, to_del_set, res):
        if node is None:
            return 
        if node.val in to_del_set:
            if prev:
                if from_left:
                    prev.left = None
                else:
                    prev.right = None
            self.delNodesHelper(node.left, None, True, to_del_set, res)
            self.delNodesHelper(node.right, None, False, to_del_set, res) 
        else:
            if prev is None:
                res.append(node)
            self.delNodesHelper(node.left, node, True, to_del_set, res)
            self.delNodesHelper(node.right, node, False, to_del_set, res)
        return 
            


"""
Input: [10,5,15,1,8,null,7]

   10 
   / \ 
  5  15 
 / \   \ 
1   8   7
"""
def largestBSTSubtree(root):
    self.res = 1
    self.dfs(root, float('-Inf'), float('Inf'))
    return self.res

def getBST(node, mn, mx):
    if node is None:
        return 0
    if mx <= node.val or mn >= node.val:
        return -1

class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList):
        L = len(beginWord)
        graph = defaultdict(list)

        for word in wordList:
            for i in range(L):
                graph[word[:i] + '*' + word[i + 1:]].append(word)
        print(graph)
        queue = deque([(beginWord, [beginWord])])
        seen = set()
        seen.add(beginWord)
        res = []

        while queue:
            queue_length = len(queue)
            # bot -> hot -> cot, bot -> dot -> cot is valid,
            # so we cannot add cot to global set before finishing current level.
            new_nodes = set()
            found = False

            for _ in range(queue_length):
                word, path = queue.popleft()

                for i in range(L):
                    transformed_word = word[:i] + '*' + word[i + 1:]

                    for w in graph[transformed_word]:

                        if w in seen:
                            continue

                        if w == endWord:
                            path.append(endWord)
                            res.append(path[:])
                            path.pop()
                            found = True
                            continue

                        new_nodes.add(w)
                        path.append(w)
                        queue.append((w, path[:]))
                        path.pop()

            # Found the shortest paths, bail early.
            if found:
                return res

            seen.update(new_nodes)

        return res

sol=Solution()
sol.findLadders(beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"])

[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]


for i in range(niter):
    clusters = estimate_cluster(X, curr_centroids)
    new_centroids = calc_centroids(X, clusters)
    diff = get_dist(new_centroids, curr_centroids)
    if diff < eps:
        break
    curr_centroids = new_centroids



from heapq import heappush, heappop
import numpy as np
from scipy.stats import mode
# Test distance function
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

X_train = [d[:2] for d in dataset[1:]]
y_train = [d[2] for d in dataset[1:]]
X_test = [d[:2] for d in dataset[:2]]
y_test = [d[2] for d in dataset[:2]]
class KNN:
    def knn(self, X_train, y_train, X_test, k):
        predictions = []
        for x in X_test:
            idx = self.find_nn(x, X_train, k)
            pred = mode(np.array(y_train)[idx])[0][0]
            predictions.append(pred)
        return predictions

    def find_nn(self, x, train_data, k):
        k = min(k, len(train_data))
        q = [] # can also directly sort res by dist
        for idx, train_i in enumerate(train_data):
            curr_dist = self.get_dist(train_i, x)
            heappush(q, (curr_dist, idx))
        
        res = []
        for i in range(k):
            res.append(heappop(q)[1])
        return res

    def get_dist(self, l1, l2):
        return sum((np.array(l1) - np.array(l2)) ** 2)

m1 = KNN()
m1.knn(X_train, y_train, X_test, k)
y_test

nestedList = [[1,1],2,[1,1]]
from collections import defaultdict
class Solution:
    def depthSumInverse(self, nestedList):
        self.d = defaultdict(lambda :0)
        self.getSum(nestedList, 1)
        print(self.d)
        max_lv = max(self.d.keys())
        res = 0
        for k, v in self.d.items():
            res += (max_lv-k+1) * v
        
        return res

    def getSum(self, nestedList, lv):
        res = 0
        for i in range(len(nestedList)):
            if isinstance(nestedList[i], int):
                res +=  nestedList[i]
            else:
                self.getSum(nestedList[i], lv+1)
        self.d[lv] += res
        return 

sol=Solution()
sol.depthSumInverse([[1,1],2,[1,1]])
sol.depthSumInverse([1,[4,[6]]])



class Solution:
    def largestIsland(self, grid) -> int:
        cur_id = 2
        id2size = dict()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    id2size[cur_id] = self.getCurrIslandSize(grid, i, j, cur_id)
                    cur_id += 1
        print(grid)
        print(id2size)
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
sol.largestIsland([[1,1],[1,1]])


class UF:
    def __init__(self, N):
        self.parents = list(range(N))
        self.weights = [0]*N

    def find(self, x):
        if self.parents[x] != x:
            curr_parent = self.parents[x]
            self.parents[x] = self.find(curr_parent)
        return self.parents[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.weights[x] < self.weights[y]:
            self.parents[root_x] = root_y
            self.weights[root_y] += self.weights[root_x]
        else:
            self.parents[root_y] = root_x
            self.weights[root_x] += self.weights[root_y]
        return 
    

class Solution:
    def accountsMerge(self, accounts):
        uf = UF(len(accounts))  # account num 1-N, now need to find which nums should be unioned
        email2id = defaultdict(int)
        for idx, account in enumerate(accounts):
            for i in range(1, len(account)):
                if account[i] in email2id:
                    uf.union(idx, email2id[account[i]])
                else:
                    email2id[account[i]] = idx
        
        # merge union accounts
        id2emails = defaultdict(list)
        for email in email2id.keys():
            root_id = uf.find(email2id[email])
            id2emails[root_id].append(email)

        res = []
        for root_id in id2emails.keys():
            curr_res = [accounts[root_id][0]] + sorted(id2emails[root_id])
            res.append(curr_res)
        return res



accounts = [["Hanzo","Hanzo2@m.co","Hanzo3@m.co"],["Hanzo","Hanzo4@m.co","Hanzo5@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co"],
["Hanzo","Hanzo3@m.co","Hanzo4@m.co"],["Hanzo","Hanzo7@m.co","Hanzo8@m.co"],["Hanzo","Hanzo1@m.co","Hanzo2@m.co"],
["Hanzo","Hanzo6@m.co","Hanzo7@m.co"],["Hanzo","Hanzo5@m.co","Hanzo6@m.co"]]
sol=Solution()
sol.accountsMerge(accounts)


class Solution:
    def checkSubarraySum(self, nums, k: int) -> bool:
        d = dict{0: -1}
        cumsum = 0
        for i, num in enumerate(nums):
            cumsum += num
            curr_key = cumsum % k if k != 0 else cumsum
            if curr_key in d:
                if i - d[curr_key] > 1:
                    return True
            else:
                d[curr_key] = i
        return False


logs = ["0:start:0",  "1:start:2",  "1:end:5",  "0:end:6"]
n = 2
class Solution:
    def exclusiveTime(self, n, logs):



"""301. Remove Invalid Parentheses  !!!
Given a string s that contains parentheses and letters, 
remove the minimum number of invalid parentheses to make the 
input string valid.

Return all the possible results. You may return the answer in any order.

Example 1:
Input: s = "()())()"
Output: ["(())()","()()()"]
"""
class Solution:
    def removeInvalidParentheses(self, s: str):
        l, r = self.count_invalid(s)
        res = []
        self.rmParenthesis(s, idx, l, r, res)
        return res
    
    def rmParenthesis(s, start_idx, l_count, r_count, res):
        if l_count == 0 and r_count == 0 and isValid(s):
            res.append(s)
            return 
        for i in range(start_idx, len(s)):
            if s[i] == '(' and l_count > 0:
                self.rmParenthesis(s[:i] + s[i+1:], l_count-1, r_count, res)
            if s[i] == ')' and r_count > 0:
                self.rmParenthesis(s[:i] + s[i+1:], l_count, r_count-1, res)
        return


"""[LeetCode] Merge Intervals 合并区间 
Given a collection of intervals, merge all overlapping intervals.

For example,
 Given [1,3],[2,6],[8,10],[15,18],
 return [1,6],[8,10],[15,18]. 
"""
s = [[2,6],[8,10],[1,3],[15,18]]
class Solution:
    def mergeInterval(self, s):
        s = sorted(s, key=lambda x:x[0])
        res = []
        start, end = s[0]
        for interval in s[1:]:
            if interval[0] <= end:
                end = max(end, interval[1])
            else:
                res.append(start, end)
                start, end = interval
        
        res.append([start, end])
        return res


class Solution:
    def findDiagonalOrder(self, mat):
        r, c= 0, 0
        up = True
        res = []

        while True:
            res.append(mat[r][c])
            if stop:
                break

            if up:
                r -= 1
                c += 1
            else:
                r += 1
                c -= 1

            if r < 0 and c > m - 1:
                r += 2
                c -= 1
            elif r < 0 :
                r += 1
            elif r > m-1 and c < 0:
                r -= 1
                c += 2
            elif r>m-1:
                c += 1
                r -= 1


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


sol=Solution()
sol.addOperators("123", 6)
sol.addOperators("00", 0)
class Solution:
    def addOperators(self, num: str, target: int):

    
    def dfs(self, num, start_idx, curr, target, out, res):
        if curr == target and start_idx == len(num):
            res.append(out)
            return
        
        for i in range(start_idx, len(num)):
            prev_num = num[start_idx:i+1]
            if prev_num[0] == '0' and len(prev_num) > 1:
                return 
            # +
            self.dfs(num, i+1, curr + int(prev_num), target, out + "+" + prev_num, res)
            # -
            self.dfs(num, i+1, curr - int(prev_num), target, out + "-" + prev_num, res)
            # * 

        return


class Solution:
    def sumNumbers(self, root) -> int:
        return self.helper(root, 0)

    def helper(node, res):
        if node is None:
            return res
        if node.left is None and node.right is None:
            return node.val + res*10
        res_left = self.helper(node.left, node.val + res*10)
        res_right = self.helper(node.right, node.val + res*10)
        return res_left + res_right



class Codec:
    def serialize(self, root):
    

    def deserialize(self, data):
        # data: list
        if len(data) == 0:
            return 
        root = TreeNode(data[0])
        q = deque()
        q.append(root)
        idx = 1
        while q:
            curr_node = q.popleft()
            if idx < len(data) and data[idx]:
                curr_node.left = TreeNode(data[idx])
                q.append(curr_node.left)
            idx += 1
            if idx < len(data) and data[idx]:
                curr_node.right = TreeNode(data[idx])
                q.append(curr_node.right)
            idx += 1
        return root


    

def construct(input):
    self.num_nodes = 0
    construct_at_depth(input, 1)
    

def construct_at_depth(input, level):
    if not input:
        return None
    node = TreeNode(val=0)
    if input[0][1] < level:
        return False
    if input[0][1] == level:
        node.left = input[0][0]
        node.right = construct_at_depth(input[1:], level+1)
    if input[0][1] 
    


class Pair {
    char c;
    int level;
}
class TreeNode {
      char val;
      TreeNode left;
      TreeNode right;
      TreeNode(char val) { this.val = val; }
  }
public class TreeBuilder {
    public static TreeNode buildTree(Pair[] inputs) {
        TreeMap<Integer, PriorityQueue<Character>> charsByLevel = new TreeMap<>();
        for (Pair p: inputs) {
            charsByLevel.computeIfAbsent(p.level, key -> new PriorityQueue<>()).offer(p.c);
        }
        TreeNode root = recursiveHelper(charsByLevel, 0);
        if (!charsByLevel.isEmpty()) {
            return null;
        }
        return root;
    }
    private static TreeNode recursiveHelper(TreeMap<Integer, PriorityQueue<Character>> charsByLevel, int level) {
        // we stop in this case because no leaves above this level exist.
        if (charsByLevel.ceilingKey(level) == null) {
            return null;
        }
        TreeNode curNode;
        if (charsByLevel.containsKey(level)) {
            curNode = new TreeNode(charsByLevel.get(level).poll());
            if (charsByLevel.get(level).isEmpty()) {
                charsByLevel.remove(level);
            }
        } else {
            curNode = new TreeNode('*');
            curNode.left = recursiveHelper(charsByLevel, level + 1);
            curNode.right = recursiveHelper(charsByLevel, level + 1);
        }
        return curNode;
    }
}


def bin_search(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            right = mid
        else:
            left = mid + 1
    
    return -1

class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val


class Codec:
    def serialize(self, root):
        if root is None:
            return [None]
        q = deque()
        q.append(root)
        res = []
        while len(q) > 0:
            for _ in range(len(q)):
                curr_node = q.popleft()
                if curr_node is None:
                    res.append(None)
                else:
                    res.append(curr_node.val)
                    q.append(curr_node.left)
                    q.append(curr_node.right)
        return res
    
    def deserialize(self, data):
        root = Node(data[0]) if data[0] else None
        q = deque()
        q.append(root)
        idx = 1
        while idx < len(val):
            curr_node = q.popleft()
            if data[idx] is not None:
                curr_node.left = Node(data[idx])
                q.append(curr_node.left)
            idx += 1
            if data[idx] is not None:
                curr_node.right = Node(data[idx])
                q.append(curr_node.right)
            idx += 1
        return root

"""
For example, you may serialize the following tree
    1
   / \
  2   3
     / \
    4   5
as "[1,2,3,null,null,4,5,null,null,null,null]", just the same as how LeetCode OJ serializes a binary tree. You do not necessarily need to
follow this format, so please be creative and come up with different approaches yourself.
"""
class Codec:
    def serialize_helper(self, node, res):
        if node is None:
            res.append(None)
            return 
        res.append(node.val)
        self.serialize_helper(node.left, res)
        self.serialize_helper(node.right, res)
        return

    def serialize(self, root):
        res = []
        self.serialize_helper(root, res)
        return res

    def deserialize_helper(self, data):
        if len(data) == 0 or data[0] == None:
            return None
        node = Node(data[0])
        data = data[1:]
        node.left = self.deserialize_helper(data)
        node.right = self.deserialize_helper(data)
        return node

    def deserialize(self, data):
        root = self.deserialize_helper(data)
        return root
