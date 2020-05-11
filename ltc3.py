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
        for i in range(l, r + 1):
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


"""
68. Text Justification
Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left justified and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.
Example 1:

Input:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

Example 2:

Input:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be",
             because the last line must be left-justified instead of fully-justified.
             Note that the second line is also left-justified becase it contains only one word.

Example 3:

Input:
words = ["Science","is","what","we","understand","well","enough","to","explain",
         "to","a","computer.","Art","is","everything","else","we","do"]
maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]
"""

def text_justification(words, maxwidth):
  cum_len = 0
  output = []
  curr_words = []
  rest_words = words
  while rest_words:
      curr_line, rest_words = extract_line(rest_words, maxwidth)
      output.append(curr_line)

  return output

def extract_line():
  # get evenly distributed word line

  return 




# Given an array of integers and an integer k, 
# you need to find the total number of continuous subarrays whose sum equals to k.

# Example 1:
# Input:nums = [1,1,1], k = 2

# Output: 2
def num_subarrays(nums, k):
  # first cumsum 
  res = 0
  cumsum_arr, ht1 = cumsum(nums)
  for i in cumsum_arr:
    if i == k:
      res = res + 1
    if k-i in ht1:
      res = res + len(ht1[k-i])

  return res


def cumsum(nums):
  cumsum_arr = []
  sum_so_far = 0
  ht = {}
  for i in range(len(nums)):
    sum_so_far = sum_so_far + nums[i]
    cumsum_arr.append(sum_so_far)
    if sum_so_far not in ht:
      ht[sum_so_far] = 1
    else:
      ht[sum_so_far]+=1
  return cumsum_arr, hs







