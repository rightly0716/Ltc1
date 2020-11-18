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

NOTE:
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
text_justification(words, maxwidth=20)

def text_justification(words, maxwidth):
  # check whether any words length > maxwidth
  cum_len = 0
  output = []
  curr_words = []
  rest_words = words
  while len(rest_words) > 0:
      curr_line, rest_words = extract_line(rest_words, maxwidth)
      output.append(curr_line)
  return output

def extract_line(words, maxwidth):
  # get evenly distributed word line
  curr_len = 0
  curr_words = []
  rest_words = []
  for i in range(len(words)):
    curr_len += len(words[i])
    if curr_len > maxwidth:
      rest_words = words[i:]
      break
    else:
      curr_words.append(words[i])
      curr_len += 1
  
  if len(rest_words) == 0:
    # if last line
    curr_line = ''
    for word in curr_words:
      curr_line += word + ' '
    curr_line += ' '*(maxwidth-len(curr_line))
  else:
    # put spaces between curr_words
    total_space_num = maxwidth
    for word in curr_words:
      total_space_num -= len(word)
    space_list = distribute_space(total_space_num, len(curr_words)-1)
    curr_line = ''
    for num_space, word in zip(space_list, curr_words):
      curr_line = curr_line + word + ' '*num_space
  return curr_line, rest_words

def distribute_space(n, k):
  # split n spaces into k parts, left more
  baseline_list = [n//k]*k
  remainder_list = [1]*(n%k) + [0]*(k-n%k)
  output = [i+j for i,j in zip(baseline_list, remainder_list)]
  output.append(0)
  return output

"""
# Given an array of integers and an integer k, 
# you need to find the total number of continuous subarrays whose sum equals to k.
# Example 1:
# Input:nums = [1,1,1], k = 2
# Output: 2
"""

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

"""
# s1 = [(1, 2), (1, 3), (3, 4), (4,6), (2,3)]
# output: [[1,2,3,4,6], [1,3,4,6]]
"""
get_all_routes(s1)
def get_all_routes(input_list):
  # build a hash table saving: from -> to
  map_to_dest = build_map(input_list)
  start_city = 1
  res = []
  curr_res = [1]
  dfs(res, curr_res, map_to_dest)
  return res

def dfs(res, curr_res, map_to_dest):
  if curr_res[-1] not in map_to_dest:
    res.append(curr_res)
  else:
    for next_city in map_to_dest[curr_res[-1]]:
      dfs(res, curr_res + [next_city], map_to_dest)

def build_map(input_list):
  # build a hash table saving: from -> to
  output_set = {}
  for pair in input_list:
    if pair[0] not in output_set:
      output_set[pair[0]] = [pair[1]]
    else:
      output_set[pair[0]].append(pair[1])
  
  return output_set


"""
[LeetCode] 209. Minimum Size Subarray Sum 最短子数组之和
Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum >= s. If there isn't one, return 0 instead.

Example: 
Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
Follow up:
If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n). 

讨论：本题有一个很好的 Follow up，就是去掉所有数字是正数的限制条件，而去掉这个条件会使得累加数组不一定会是递增的了，那么就不能使用二分法，同时双指针的方法也会失效，只能另辟蹊径了。其实博主觉得同时应该去掉大于s的条件，只保留 sum=s 这个要求，因为这样就可以在建立累加数组后用 2sum 的思路，快速查找 s-sum 是否存在，如果有了大于的条件，还得继续遍历所有大于 s-sum 的值，效率提高不了多少。
"""






