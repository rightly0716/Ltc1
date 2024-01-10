"""
In Ltc, some questions need hard memory
This file contains such questions
"""


"""
in python, one can use OrderedDict (a dict that has popitem() to remove the last or first key)
"""
from collections import OrderedDict
hashmap = OrderedDict()
hashmap.pop(key)
hashmap.popitem(last=False) # O(1)


""" Stack and (de)queue in python
stack = []
stack.append('a')
print(stack.pop())   # pop the last, O(1)
print(stack.pop(0))  # pop the first, O(n)

from collections import deque
d = deque()
d.append('j')
d.appendleft('f')
d.pop()  # pop from right
d[-1] # peek from right
d.popleft() 
d[0]  # peek from left
d.extend('jkl')  # add multiple elements at once
"""


"""写一个quicksort来找arr的第k大的数字
1. partition 
- 函数 while是<=, 先写swap的逻辑(不然可能在不满足left<=right的条件下错误swap)
- 非swap条件都包括=
2. 主函数while可以是true,想清楚什么时候left=pivot_index+1

[LeetCode] 215. Kth Largest Element in an Array 数组中第k大的数字

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the 
sorted order, not the kth distinct element.

Example 1:
Input: 
[3,2,1,5,6,4] 
and k = 2
Output: 5

Example 2:
Input: 
[3,2,3,1,2,4,5,5,6] 
and k = 4
Output: 4
Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.
"""
class kthlargest:
    def findKthLargest(self, arr, k):
        n=len(arr)
        left, right = 0, n-1
        while left <= right:
            # find 
            pivot_index = self.partition(left, right, arr)
            if pivot_index == n-k:
                return arr[pivot_index]
            if pivot_index < n-k:
                # dest is on right
                left = pivot_index + 1
            if pivot_index > n-k:
                right = pivot_index - 1
    
    def partition(self, left, right, arr):
        # move all smaller to left
        pivot = arr[left]
        l_index, r_index = left+1, right
        while l_index <= r_index:
            if arr[l_index] > pivot and arr[r_index] < pivot:
                arr[l_index], arr[r_index] = arr[r_index], arr[l_index]
                l_index += 1
                r_index -= 1
            if arr[l_index] <= pivot:
                l_index += 1
            if arr[r_index] >= pivot:
                r_index -= 1

        arr[left], arr[r_index] = arr[r_index], arr[left]
        return r_index

arr=[3,2,3,1,2,4,5,5,6]; k=4
solution=kthlargest()
solution.findKthLargest(arr, k)


""" [LeetCode] 142. Linked List Cycle II 单链表中的环之二
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

To represent a cycle in the given linked list, we use an integer pos which represents the position 
(0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Note: Do not modify the linked list.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.

这个求单链表中的环的起始点是之前那个判断单链表中是否有环的延伸，可参之前那道 Linked List Cycle。这里还是要设快慢指针，
不过这次要记录两个指针相遇的位置，当两个指针相遇了后，让其中一个指针从链表头开始，此时再相遇的位置就是链表中环的起始位置
因为快指针每次走2，慢指针每次走1，快指针走的距离是慢指针的两倍。而快指针又比慢指针多走了一圈。所以 head到环的起点+环的起点到他们相遇的点的距离 与 环一圈的距离相等。现在重新开始，head 运行到环起点 和 相遇点到环起点的距离也是相等的，相当于他们同时减掉了 环的起点到他们相遇的点的距离。
"""

"""
############################################################################
堆（Heap or Priority Queue）、栈（Stack）、队列（Queue）、哈希表类（Hashmap、Hashset）
############################################################################
Stack: string processing, calculator, 
(de)Queue: streaming data
"""

""" Stack and (de)queue in python
stack = []
stack.append('a')
print(stack.pop())   # pop the last
print(stack.pop(0))  # pop the first

from collections import deque
d = deque()
d.append('j')
d.appendleft('f')
d.pop()  # pop from right
d[-1] # peek from right
d.popleft() 
d[0]  # peek from left
d.extend('jkl')  # add multiple elements at once
"""

"""[LeetCode] 772. Basic Calculator III 基本计算器之三 !!!
Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .
The expression string contains only non-negative integers, +, -, *, / operators , open ( and closing parentheses ) and empty spaces . The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-2147483648, 2147483647].

Some examples:
"1 + 1" = 2
" 6-4 / 2 " = 4
"2*(5+5*2)/3+(6/2+8)" = 21
"(2+6* 3+5- (3*14/7+2)*5)+3"=-12

Note: Do not use the eval built-in library function.
"""
def calculator3(s):
    m = dict()  # map all ( and ) pair location
    stack = []  # for parenthesis location
    for i in range(len(s)):
        if s[i] == '(':
            stack.append(i)
        if s[i] == ')':
            m[stack.pop()] = i

    op, curr_num = '+', ''  # previous operator
    n = len(s)
    stack = []  # can use prev_num and curr_num to save ram
    sign = 1
    i = 0
    while i < n: # cannot use FOR because i needs to update in loop at '('
        if s[i].isdigit(): # 0-9
            curr_num = curr_num + s[i]
        if s[i] == '(':
            # treat the part between ( and ) as a number (curr_num)
            j = m[i]
            sub_string = s[(i+1):j]
            curr_num = calculator3(sub_string)
            i = j  # set i at the location of )
        if s[i] in ['+', '-', '*', '/'] or i == n-1:
            if i==0 or (i<n-1 and not s[i-1].isdigit() and s[i-1] not in '()'):
                # sign, not a op
                if s[i] == '-':
                    sign = sign * -1
            else:
                # if s[i] is a operator, not a sign
                if op == '+':
                    stack.append(int(curr_num) * sign)
                if op == '-':
                    stack.append(int(curr_num) * -1 * sign)
                if op =='*':
                    previous_num = stack.pop()
                    stack.append(previous_num * int(curr_num) * sign)
                if op == '/':
                    previous_num = stack.pop()
                    stack.append(int(previous_num / int(curr_num)) * sign)
                sign = 1  # reset sign!
                op = s[i]
                curr_num = ''
        i = i + 1

    return sum(stack)

calculator3("1-(-2)*5")
calculator3("2+6*3+5-(3*14/7+2)*5+3")==eval("2+6*3+5-(3*14/7+2)*5+3")
calculator3("-4*(-1-2)")

class node:
    def __init__(self, val=None, prev=None, next=None):
        self.val = val
        self.next = None
        self.prev = None
    
class LinkedList:
    def __init__(self):
        self.head = self.tail = Node()
        self.head.next, self.tail.prev = self.tail, self.head
    
    def remove(self, node):
        node.next.prev, node.prev.next = node.prev, node.next
    
    def appendleft(self, node):
        old_first_node = self.head.next
        self.head.next, node.prev = node, self.head
        old_first_node.prev, node.next = node, old_first_node
        


    def pop(self):

    
    def popleft(self):


class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.llist = LinkedList()
        self.node2val = dict()
    
    def get(self, key):
        curr_node = node(key)
        if curr_node in self.node2val:
            self.llist.remove(curr_node)
            self.llist.appendleft(curr_node)
            return self.node2val[curr_node]
        return -1

    def put(self, key, value):
        curr_node = node(key)
        if curr_node in self.node2val:
            self.llist.remove(curr_node)
            self.llist.appendleft(curr_node)
        else:
            self.llist.appendleft(curr_node)
            if len(self.node2val) > capacity:
                key_rm = self.llist.pop()
                del self.node2val[node(key_rm)]
        self.node2val[curr_node] = value



