"""Serialize and Deserialize Binary Tree 二叉树的序列化和去序列化
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored
in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or
another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your
serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a
 string and this string can be deserialized to the original tree structure.

For example, you may serialize the following tree
    1
   / \
  2   3
     / \
    4   5
as "[1,2,3,null,null,4,5]", just the same as how LeetCode OJ serializes a binary tree. You do not necessarily need to
follow this format, so please be creative and come up with different approaches yourself.
"""
# can do both DFS (pre-order) and BFS
# when using BFS, do not push None into the deque(); only do while once, no for loop

"""Swap nodes in a linked list without swapping data
Given a linked list and two keys in it, swap nodes for two given keys. Nodes should be swapped by changing links. 
Swapping data of nodes may be expensive in many situations when data contains many fields.

It may be assumed that all keys in linked list are distinct.

Examples:

Input:  10->15->12->13->20->14,  x = 12, y = 20
Output: 10->15->20->13->12->14

Input:  10->15->12->13->20->14,  x = 10, y = 20
Output: 20->15->12->13->10->14

Input:  10->15->12->13->20->14,  x = 12, y = 13
Output: 10->15->13->12->20->14
"""
# The idea it to first search x and y in given linked list. If any of them is not present, then return.
# While searching for x and y, keep track of current and previous pointers. First change next of previous pointers,
# then change next of current pointers. Following are C and Java implementations of this approach.

"""Choose k randome value equally from a stream of integers.
Reservoir sampling
"""
#Example: Sample size 10 Suppose we see a sequence of items, one at a time. We want to keep ten items in memory,
# and we want them to be selected at random from the sequence. If we know the total number of items (n), then the
# solution is easy: select ten distinct indices i between 1 and n with equal probability, and keep the i-th elements.
# The problem is that we do not always know n in advance. A possible solution is the following:
#Keep the first ten items in memory.
#When the i-th item arrives:
#with probability 10/i, keep the new item (discard an old one, selecting which to replace at random, each with chance 1/10)
#with probability 1-10/i, keep the old items (ignore the new one)



"""Write a program to print all permutations of a given string
Below are the permutations of string ABC:
ABC ACB BAC BCA CBA CAB
"""
# https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/

# Python program to print all permutations with
# duplicates allowed

def toString(List):
    return ''.join(List)


# Function to print permutations of string
# This function takes three parameters:
# 1. String
# 2. Starting index of the string
# 3. Ending index of the string.
def permute(a, l, r):
    if l == r:
        print
        toString(a)
    else:
        for i in xrange(l, r + 1):
            a[l], a[i] = a[i], a[l]
            permute(a, l + 1, r)
            a[l], a[i] = a[i], a[l]  # backtrack


# Driver program to test the above function
string = "ABC"
n = len(string)
a = list(string)
permute(a, 0, n - 1)



"""
For the given binary tree, return a deep copy of it.
Given a binary tree:

    1
   / \
  2   3
 / \
4   5

return the new binary tree with same structure and same value:

    1
   / \
  2   3
 / \
4   5
"""
class Node:
    def __init__(self, val):
        self.val = val
        self.left = Node(None)
        self.right = Node(None)

def copytree(node):
    if node is None:
        return None
    else:
        head = Node(node.val)
        head.left = copytree(node.left)
        head.right = copytree(node.right)
    return head

