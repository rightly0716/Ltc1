# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:17:05 2018

@author: U586086
"""

# leetcode continue...


"""
Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.

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
Open the lock 

You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.

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
[LeetCode] Accounts Merge 账户合并
 

Given a list accounts, each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some email that is common to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.

Example 1:

Input: 
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
Explanation: 
The first and third John's are the same person as they have the common email "johnsmith@mail.com".
The second John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
 

Note:
The length of accounts will be in the range [1, 1000].
The length of accounts[i] will be in the range [1, 10].
The length of accounts[i][j] will be in the range [1, 30].
"""
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"], ["John", "a@gmail.com", "b@gmail.com"], 
["John", "c@gmail.com", "d@gmail.com"],
["John", "a@gmail.com", "c@gmail.com"]]
# need a hash map from email to its account nameidx, for each nameidx
from collections import deque

def merge_accounts(accounts):
    map_from_email_to_idx = {}
    n = len(accounts)
    for i in range(n):
        emails = accounts[i][1:]
        for email in emails:
            if email not in map_from_email_to_idx:
                map_from_email_to_idx[email] = [i]
            else:
                map_from_email_to_idx[email].append(i)
        
    visited = [False] * len(accounts)

    out = []

    for name_id in range(len(accounts)):
        if visited[name_id]:
            # if the current row of names was visited before
            continue
        visited[name_id] = True
        q = deque() # queue saves name_id to be visited
        q.append(name_id)

        curr_name = accounts[name_id][0]
        # curr_emails = accounts[name_id][0]
        res = [curr_name]

        while len(q) > 0:
            for i in range(len(q)):
                #if visited[curr_name_id]:
                #    continue
                # visited[curr_name_id] = True
                curr_name_idx = q.popleft()
                curr_emails = accounts[curr_name_idx][1:]
                for email in curr_emails:
                    if email not in res:
                        res.append(email)
                    for j in map_from_email_to_idx[email]:
                        if visited[j]:
                            continue
                        q.append(j)
                        visited[j] = True

        out.append(res)

    return out


"""[LeetCode] Binary Search Tree Iterator 二叉搜索树迭代器
Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.

Credits:
Special thanks to @ts for adding this problem and creating all test cases.
"""




"""
322. Coin Change2 /knapsnak problem
You are given coins of different denominations and a total amount of money. Write a function to compute the number of combinations 
that make up that amount. You may assume that you have infinite number of each kind of coin.

Input: amount = 5, coins = [1, 2, 5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.

这里需要一个二维的 dp 数组，其中 dp[i][j] 表示用前i个硬币组成钱数为j的不同组合方法
dp[i][j] = dp[i - 1][j] + (j >= coins[i - 1] ? dp[i][j - coins[i - 1]] : 0)
"""




                   