# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:17:05 2018

@author: U586086
"""

"""
BFS: 
542: 01 matrix; 752. Open the Lock

DFS:





"""


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




                   