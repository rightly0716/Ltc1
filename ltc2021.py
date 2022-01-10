"""
Leetcode by categories and ds
"""
# cheatsheet
# https://www.hackerearth.com/practice/notes/big-o-cheatsheet-series-data-structures-and-algorithms-with-thier-complexities-1/
# !!! means interview qn candidates

"""
############################################################################
Sort 排序类
############################################################################
OrderedDict: popitem() will pop the most recent item
"""

"""[LeetCode] 148. Sort List 链表排序
Sort a linked list in O(n log n) time using constant space complexity.

Example 1:

Input: 4->2->1->3
Output: 1->2->3->4
Example 2:

Input: -1->5->3->4->0
Output: -1->0->3->4->5
"""
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head
        dummy = ListNode(next=head)
        fast = dummy
        slow = dummy
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        mid = slow.next
        slow.next = None
        sorted_mid = self.sortList(mid)
        sorted_head = self.sortList(head)
        return self.mergeSortedLists(sorted_mid, sorted_head)

    def mergeSortedLists(self, nodeA, nodeB):
        if nodeA is None:
            return nodeB
        elif nodeB is None:
            return nodeA
        
        dummy = ListNode(next=None)
        pre = dummy
        while nodeA and nodeB:
            if nodeA.val < nodeB.val:
                pre.next = nodeA
                nodeA = nodeA.next    
            else:
                pre.next = nodeB
                nodeB = nodeB.next
            pre = pre.next
        pre.next = nodeA if nodeA else nodeB
        return dummy.next


"""[LeetCode] 215. Kth Largest Element in an Array 数组中第k大的数字
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
# solution 1: use maxheap
from heapq import heappush, heappop 

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # use a min heap to save the n nums, 
        hp = []
        for i in nums:
            heappush(hp, i)
        
        for i in range(len(nums)-k):
            heappop(hp)
        
        return heappop(hp)


# solution 2: use quick sort 
class Solution2(object):
    def findKthLargest(self, nums, k):
        left = 0
        right = len(nums) - 1
        while True:
            curr = self.partition(nums, left, right, k)
            if curr == k - 1:
                return nums[curr]
            if curr > k - 1:
                # on the right
                right = curr - 1
            if curr < k - 1:
                left = curr + 1

    def partition(self, nums, left, right, k):
        # choose a pivot, put smaller elements on right, larger on left
        # return rank of pivot (put on final r position)
        pivot = nums[left]
        l = left + 1
        r = right
        while l<=r:
            if nums[l] < pivot and nums[r] > pivot:
                nums[l], nums[r] = nums[r], nums[l]
                l = l+1
                r = r-1
            if nums[l] >= pivot:
                l = l+1
            if nums[r] <= pivot:
                r = r-1
        nums[left], nums[r] = nums[r], nums[left]
        return r



"""[LeetCode] 4. Median of Two Sorted Arrays 两个有序数组的中位数
There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume nums1 and nums2 cannot be both empty.

Example 1:

nums1 = [1, 3]
nums2 = [2]

The median is 2.0
Example 2:

nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5
"""
# https://zxi.mytechroad.com/blog/algorithms/binary-search/leetcode-4-median-of-two-sorted-arrays/

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        if n1 > n2:
            return self.findMedianSortedArrays(nums2, nums1)
        
        k = int((n1 + n2 + 1)/2)
        l = 0
        r = n1
        # find m1, m2 such that if n1+n2 is odd, then c1 is median
        # else c1 + c2 /2 
        while l < r:
            m1 = int(l + (r-l)/2)
            m2 = k - m1
            if nums1[m1] < nums2[m2-1]:
                l = m1 + 1
            else:
                r = m1
        
        # l=r
        m1 = l
        m2 = k - l
        # now need to decide whether m1 exceeds k (m1<=0 or m2<=0)
        if m1 <= 0:
            # nums1 all larger than nums2
            c1 = nums2[m2 - 1]
        elif m2 <= 0:
            # nums2 all larger than nums1
            c1 = nums1[m1 - 1]
        else:
            # max of the two chosen
            c1 = max(nums2[m2 - 1], nums1[m1 - 1])
        
        if (n1 + n2) % 2 == 1:
            # odd
            return c1

        # if even, take the number next to c1 (c2), c1+c2/2
        if m1 >= n1:
            c2 = nums2[m2]
        elif m2 >= n2:
            c2 = nums1[m1]
        else:
            # min of the next numbers to the two chosen
            c2 = min(nums2[m2], nums1[m1])
        
        return (c1 + c2) * 0.5




"""
############################################################################
Linked List
############################################################################
"""

""" 876 Middle of the Linked List
Given the head of a singly linked list, return the middle node of the linked list.
If there are two middle nodes, return the second middle node.

Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head):
        if head is None:
            return head
        fast = head
        slow = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        if fast.next is None:
            return slow
        else:
            return slow.next

"""[LeetCode 160] Intersection of Two Linked Lists 求两个链表的交点
Write a program to find the node at which the intersection of two singly linked lists begins.
For example, the following two linked lists:

A:          a1 → a2
                      ↘
                        c1 → c2 → c3
                      ↗            
B:     b1 → b2 → b3
begin to intersect at node c1.

Notes:

If the two linked lists have no intersection at all, return null.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in O(n) time and use only O(1) memory.
 

Credits:
Special thanks to @stellari for adding this problem and creating all test cases.
"""
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode):
        len_listA = self.get_length(headA)
        len_listB = self.get_length(headB)
        if len_listA >= len_listB:
            for i in range(len_listA - len_listB):
                headA = headA.next
        else:
            for i in range(len_listB - len_listA):
                headB = headB.next
        while headA != headB:
            headA = headA.next
            headB = headB.next
        return headA
    
    def get_length(self, head):
        res = 0
        while head:
            head = head.next
            res = res + 1
        return res


class Solution2(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA is None or headB is None:
            return None
        pa = headA # 2 pointers
        pb = headB
        while pa is not pb:
            # if either pointer hits the end, switch head and continue the second traversal, 
            # if not hit the end, just move on to next
            pa = headB if pa is None else pa.next
            pb = headA if pb is None else pb.next

        return pa # only 2 ways to get out of the loop, they meet or the both hit the end=None


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
不过这次要记录两个指针相遇的位置，当两个指针相遇了后，让其中一个指针从链表头开始，一步两步，一步一步似爪牙，似魔鬼的步伐。。。
哈哈，打住打住。。。此时再相遇的位置就是链表中环的起始位置，为啥是这样呢，这里直接贴上热心网友「飞鸟想飞」的解释哈，
因为快指针每次走2，慢指针每次走1，快指针走的距离是慢指针的两倍。而快指针又比慢指针多走了一圈。所以 head 
到环的起点+环的起点到他们相遇的点的距离 与 环一圈的距离相等。现在重新开始，head 运行到环起点 和 相遇点到环起点 
的距离也是相等的，相当于他们同时减掉了 环的起点到他们相遇的点的距离。

class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) break;
        }
        if (!fast || !fast->next) return NULL;
        slow = head;
        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return fast;
    }
};
"""

"""92. Reverse Linked List II
Given the head of a singly linked list and two integers left and right where left <= right, 
reverse the nodes of the list from position left to position right, and return the reversed list.

Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseBetween(self, head, left: int, right: int):
        H1 = ListNode(next=head)
        newhead = H1
        for i in range(left-1):
            H1 = H1.next
        T1 = H1.next
        
        H2 = ListNode(next=head)
        for i in range(right):
            H2 = H2.next
        T2 = H2.next
        
        H1.next = None
        H2.next = None

        H2 = self.reverseList(T1)
        H1.next = H2
        T1.next = T2

        return newhead.next
    
    def reverseList(self, head):
        if head is None:
            return head
        newhead = ListNode(next=head)
        curr = newhead.next
        while curr:
            prev = newhead
            newhead = curr
            curr = curr.next
            newhead.next = prev
        return newhead



    




"""
############################################################################
堆（Heap or Priority Queue）、栈（Stack）、队列（Queue）、哈希表类（Hashmap、Hashset）
############################################################################
"""

""" Stack and queue in python
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



"""[LeetCode] 346. Moving Average from Data Stream 从数据流中移动平均值
 

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Example:

MovingAverage m = new MovingAverage(3);
m.next(1) = 1
m.next(10) = (1 + 10) / 2
m.next(3) = (1 + 10 + 3) / 3
m.next(5) = (10 + 3 + 5) / 3
"""
# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param = obj.next(val)

class MovingAverage:
    """
    @param: size: An integer
    """
    def __init__(self, size):
        # do intialization if necessary
        self.size = size
        self.queue = []
        self.sum = 0

    """
    @param: val: An integer
    @return:  
    """
    def next(self, val):
        # write your code here
        self.queue.add(val)
        self.sum = self.sum + val
        # pop more than 
        if len(self.queue) > self.size:
            self.sum = self.sum - self.queue[0]
            self.queue.pop()
        
        return self.sum / len(self.queue)
        

"""[LeetCode] 281. Zigzag Iterator 之字形迭代器
Given two 1d vectors, implement an iterator to return their elements alternately.
Input: v1 = [1, 2] and v2 = [3, 4, 5, 6]
Output: [1, 3, 2, 4, 5, 6]
Explanation: 
By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1, 3, 2, 4, 5, 6].

Follow up: What if you are given k 1d vectors? How well can your code be extended to such cases?
Clarification for the follow up question:
The "Zigzag" order is not clearly defined and is ambiguous for k > 2 cases. If "Zigzag" does not look right to you, 
replace "Zigzag" with "Cyclic". For example:

Input:
[1,2,3]
[4,5,6,7]
[8,9]

Output: 
[1,4,8,2,5,9,3,6,7]
"""
# Your ZigzagIterator object will be instantiated and called as such:
# solution, result = ZigzagIterator(v1, v2), []
# while solution.hasNext(): result.append(solution.next())
# Output result

class ZigzagIterator:
    """
    @param: v1: A 1d vector
    @param: v2: A 1d vector
    """
    def __init__(self, v1, v2):
        # do intialization if necessary
        n1 = len(v1)
        n2 = len(v2)
        it1 = iter(v1)
        it2 = iter(v2)
        # max_len = max(n1, n2)
        # self.zigzag_vec = []  # extra space, not good
        # for i in range(max_len):
        #     if i < n1:
        #         self.zigzag_vec.append(v1[i])
        #     if i < n2:
        #         self.zigzag_vec.append(v2[i])
        self.size = n1 + n2
        self.index = 0
    
    """
    @return: An integer
    """
    def _next(self):
        # write your code here
        if self.index % 2 == 0:
            try:
                curr_num = next(it1)
            except:
                curr_num = next(it2)
        else:
            try:
                curr_num = next(it2)
            except:
                curr_num = next(it1)
        # next_num = self.zigzag_vec[self.index]
        self.index = self.index + 1
        return curr_num

    """
    @return: True if has next
    """
    def hasNext(self):
        # write your code here
        if self.index >= self.size:
            return False
        else:
            return True



"""[LeetCode] 362. Design Hit Counter 设计点击计数器
 

Design a hit counter which counts the number of hits received in the past 5 minutes.

Each function accepts a timestamp parameter (in seconds granularity) and you may assume that calls are 
being made to the system in chronological order (ie, the timestamp is monotonically increasing). 
You may assume that the earliest timestamp starts at 1.

It is possible that several hits arrive roughly at the same time.

Example:

HitCounter counter = new HitCounter();

// hit at timestamp 1.
counter.hit(1);

// hit at timestamp 2.
counter.hit(2);

// hit at timestamp 3.
counter.hit(3);

// get hits at timestamp 4, should return 3.
counter.getHits(4);

// hit at timestamp 300.
counter.hit(300);

// get hits at timestamp 300, should return 4.
counter.getHits(300);

// get hits at timestamp 301, should return 3.
counter.getHits(301); 
Follow up:
What if the number of hits per second could be very large? Does your design scale?

Follow up:
What if the number of hits per second could be very large? Does your design scale?
"""

""" Solution
这道题让我们设计一个点击计数器，能够返回五分钟内的点击数，提示了有可能同一时间内有多次点击。由于操作都是按时间顺序的，
下一次的时间戳都会大于等于本次的时间戳，那么最直接的方法就是用一个队列queue，每次点击时都将当前时间戳加入queue中，
然后在需要获取点击数时，我们从队列开头开始看，如果开头的时间戳在5分钟以外了，就删掉，直到开头的时间戳在5分钟以内停止，
然后返回queue的元素个数即为所求的点击数，参见代码如下：

由于Follow up中说每秒中会有很多点击，下面这种方法就比较巧妙了，定义了两个大小为300的一维数组times和hits，
分别用来保存时间戳和点击数，在点击函数中，将时间戳对300取余，然后看此位置中之前保存的时间戳和当前的时间戳是否一样，
一样说明是同一个时间戳，那么对应的点击数自增1，如果不一样，说明已经过了五分钟了，那么将对应的点击数重置为1。
那么在返回点击数时，我们需要遍历times数组，找出所有在5分中内的位置，然后把hits中对应位置的点击数都加起来即可，参见代码如下：

https://www.cnblogs.com/grandyang/p/5605552.html
"""


"""[LeetCode] Max Stack 最大栈
Design a max stack that supports push, pop, top, peekMax and popMax.

push(x) -- Push element x onto stack.
pop() -- Remove the element on top of the stack and return it.
top() -- Get the element on the top.
peekMax() -- Retrieve the maximum element in the stack.
popMax() -- Retrieve the maximum element in the stack, and remove it. If you find more than one maximum elements, 
only remove the top-most one.
 

Example 1:

MaxStack stack = new MaxStack();
stack.push(5); 
stack.push(1);
stack.push(5);
stack.top(); -> 5
stack.popMax(); -> 5
stack.top(); -> 1
stack.peekMax(); -> 5
stack.pop(); -> 1
stack.top(); -> 5
"""
# solution 2 uses iterator map to locate the elements 
# https://www.cnblogs.com/grandyang/p/7823424.html
class MaxStack:
    def __init__(self):
        self.stack1 = []  # data 
        self.stack2 = []  # for max so far
        self.max_num = float('-Inf')

    def push(self, x: int) -> None:
        self.stack1.append(x)
        if x >= self.max_num:
            self.stack2.append(x)
            self.max_num = x

    def pop(self) -> int:
        x = self.stack1.pop()
        if x == self.max_num:
            self.stack2.pop()
        return x

    def top(self) -> int:
        return self.stack1[-1]

    def peekMax(self) -> int:
        return self.stack2[-1]

    def popMax(self) -> int:
        # use a tmp stack to store elements above max
        # push to both stacks! 
        temp = []
        while self.top() != self.max_num:
            temp.push(self.stack1.pop())
        
        x = self.stack1.pop()
        self.stack2.pop()

        while len(temp)>0:
            self.push(temp.pop())
        
        return x


"""[LeetCode] Evaluate Reverse Polish Notation 计算逆波兰表达式
 

Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are +, -, *, /. Each operand may be an integer or another expression.

Note:

Division between two integers should truncate toward zero.
The given RPN expression is always valid. That means the expression would always evaluate to a result and there won't be any divide by zero operation.
Example 1:

Input: ["2", "1", "+", "3", "*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
Example 2:

Input: ["4", "13", "5", "/", "+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
Example 3:

Input: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
Output: 22
Explanation: 
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
"""
# use stack to save time
class Solution:
    # @param tokens, a list of string
    # @return an integer
    def evalRPN(self, tokens):
        stack = []
        for i in tokens:
            if i not in ('+', '-', '*', '/'):
                stack.append(int(i))
            else:
                op2 = stack.pop()
                op1 = stack.pop()
                if i == '+': stack.append(op1 + op2)
                elif i == '-': stack.append(op1 - op2)
                elif i == '*': stack.append(op1 * op2)
                else: stack.append(int(op1 * 1.0 / op2))
        return stack[0]


"""[LeetCode] 772. Basic Calculator III 基本计算器之三 !!!
Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .

The expression string contains only non-negative integers, +, -, *, / operators , open ( and closing parentheses ) 
and empty spaces . The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-2147483648, 2147483647].

Some examples:

"1 + 1" = 2
" 6-4 / 2 " = 4
"2*(5+5*2)/3+(6/2+8)" = 21
"(2+6* 3+5- (3*14/7+2)*5)+3"=-12
 

Note: Do not use the eval built-in library function.
"""
# based on calculator 2, use recursive function to process ()
s1 = "(2+6* 3+5- (3*14/7+2)*5)+3"
s2 = "2*(5+5*2)/3+(6/2+8)"
s3 = "6-4 / 2"

def calculator(s):
    op = '+'  # previous operator
    curr_num = ''
    n = len(s)
    stack = []  # may not need 
    i = 0
    while i < n: # cannot use FOR because i needs to update in loop at '('
        if s[i].isdigit():
            # 1-9
            curr_num = curr_num + s[i]
        if s[i] in ['(']:
            # treat the part between ( and ) as a number (curr_num)
            cnt = 0
            curr_i = i
            for j in range(i, n):
                if s[j] == '(':
                    cnt = cnt + 1
                if s[j] == ')':
                    cnt = cnt - 1
                if cnt == 0:
                    # the string is between a pair of ( and )
                    break
            i = j  # set i at the location of )
            sub_string = s[curr_i+1:j]
            curr_num = calculator(sub_string)
        if s[i] in ['+', '-', '*', '/'] or i == n-1:
            if op == '+':
                stack.append(int(curr_num))
            if op == '-':
                stack.append(int(curr_num) * -1)
            if op =='*':
                previous_num = stack.pop()
                stack.append(previous_num * int(curr_num))
            if op == '/':
                previous_num = stack.pop()
                stack.append(int(previous_num / int(curr_num)))
            op = s[i]
            curr_num = ''

        i = i + 1
    
    res = 0
    while len(stack) > 0:
        res = res + stack.pop()
    
    return res


calculator(s1) # -12
calculator(s2) # 21
calculator(s3) # 4
                

"""[LeetCode] 1472. Design Browser History
You have a browser of one tab where you start on the homepage and you can visit another url, get 
back in the history number of steps or move forward in the history number of steps.

Implement the BrowserHistory class:

BrowserHistory(string homepage) Initializes the object with the homepage of the browser.
void visit(string url) Visits url from the current page. It clears up all the forward history.
string back(int steps) Move steps back in history. If you can only return x steps in the history and steps > x, 
you will return only x steps. Return the current url after moving back in history at most steps.
string forward(int steps) Move steps forward in history. If you can only forward x steps in the history and steps > x,
 you will forward only x steps. Return the current url after forwarding in history at most steps.
Example:

Input:
["BrowserHistory","visit","visit","visit","back","back","forward","visit","forward","back","back"]
[["leetcode.com"],["google.com"],["facebook.com"],["youtube.com"],[1],[1],[1],["linkedin.com"],[2],[2],[7]]
Output:
[null,null,null,null,"facebook.com","google.com","facebook.com",null,"linkedin.com","google.com","leetcode.com"]

Explanation:
BrowserHistory browserHistory = new BrowserHistory("leetcode.com");
browserHistory.visit("google.com");       // You are in "leetcode.com". Visit "google.com"
browserHistory.visit("facebook.com");     // You are in "google.com". Visit "facebook.com"
browserHistory.visit("youtube.com");      // You are in "facebook.com". Visit "youtube.com"
browserHistory.back(1);                   // You are in "youtube.com", move back to "facebook.com" return "facebook.com"
browserHistory.back(1);                   // You are in "facebook.com", move back to "google.com" return "google.com"
browserHistory.forward(1);                // You are in "google.com", move forward to "facebook.com" return "facebook.com"
browserHistory.visit("linkedin.com");     // You are in "facebook.com". Visit "linkedin.com"
browserHistory.forward(2);                // You are in "linkedin.com", you cannot move forward any steps.
browserHistory.back(2);                   // You are in "linkedin.com", move back two steps to "facebook.com" then to "google.com". return "google.com"
browserHistory.back(7);                   // You are in "google.com", you can move back only one step to "leetcode.com". return "leetcode.com"
Constraints:

1 <= homepage.length <= 20
1 <= url.length <= 20
1 <= steps <= 100
homepage and url consist of  '.' or lower case English letters.
At most 5000 calls will be made to visit, back, and forward.
"""

# for example 
# suppose [1,2,(3),4] # () means where curr page is
# if back(1) -> [1,(2),3,4]
# if forward(2) -> [1,2,3,(4)]
# if visit(5) -> [1,2,3,(5)]
class BrowserHistory:

    def __init__(self, homepage: str):
        self.stack = [homepage]
        self.size = 1
        self.curr = 0  # curr location
        

    def visit(self, url: str) -> None:
        while len(self.stack) > self.curr + 1:
            self.stack.pop()
        self.stack.append(url)
        self.curr = self.curr + 1
        self.size = self.curr + 1

        
    def back(self, steps: int) -> str:
        self.curr = max(0, self.curr - steps)
        return self.stack[self.curr]
        

    def forward(self, steps: int) -> str:
        self.curr = min(self.curr+steps, self.size-1)
        return self.stack[self.curr]
        

""" 1209. Remove All Adjacent Duplicates in String II !!!
You are given a string s and an integer k, a k duplicate removal consists of choosing k adjacent and equal letters 
from s and removing them, causing the left and the right side of the deleted substring to concatenate together.
We repeatedly make k duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique.

Example 1:

Input: s = "abcd", k = 2
Output: "abcd"
Explanation: There's nothing to delete.
Example 2:

Input: s = "deeedbbcccbdaa", k = 3
Output: "aa"
Explanation: 
First delete "eee" and "ccc", get "ddbbbdaa"
Then delete "bbb", get "dddaa"
Finally delete "ddd", get "aa"
Example 3:

Input: s = "pbbcggttciiippooaais", k = 2
Output: "ps"
"""
# note that recursively call the function will raise time limit error
# so can only run through the string once
s = 'abcd'
k=2
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        n = len(s)
        stack = []
        stack.append((s[0], 1))  # letter, cnts
        for i in range(1, n):
            if len(stack) > 0:
                if s[i] == stack[-1][0]:
                    letter, val = stack.pop()
                    if val + 1 < k:
                        # if =k, pop
                        stack.append((s[i], val + 1))
                else:
                    # if diff from previous
                    stack.append((s[i], 1))
            else:
                stack.append((s[i], 1))
        
        res = ''
        while len(stack) > 0:
            letter, val = stack.pop()
            res = letter * val + res
        
        return res


# if duplicate is defined as equal or larger than k instead
def removeDuplicates2(s: str, k: int) -> str:
    n = len(s)
    stack = []
    stack.append((s[0], 1))  # letter, cnts
    i = 1
    while i < n:
        if len(stack) > 0:
            if s[i] == stack[-1][0]:
                letter, val = stack.pop()
                stack.append((s[i], val + 1))
                if i == n - 1 and val + 1 >= k:
                    stack.pop()
                i = i + 1
            else:
                # if diff from previous
                if stack[-1][1] >= k:
                    stack.pop()
                    continue
                else:
                    stack.append((s[i], 1))
                    i = i + 1
        else:
            stack.append((s[i], 1))
            i = i + 1
    
    res = ''
    while len(stack) > 0:
        letter, val = stack.pop()
        res = letter * val + res
    
    return res


s1 = 'cabbbaabbc'
s2 = 'cabbbdaabbc'
removeDuplicates2(s1, 2)
removeDuplicates2(s2, 2)


"""[LeetCode] 1249. Minimum Remove to Make Valid Parentheses 移除无效的括号
Given a string s of '(' , ')' and lowercase English characters.

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the 
resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
Example 1:

Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
"""
s = "lee(t(c)o)de)"
class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        n = len(s)
        index_right_tbd = []
        index_left_tbd_stack = []  #
        for i in range(n):
            if s[i] == '(':
                index_left_tbd_stack.append(i)
            elif s[i] == ')':
                if len(index_left_tbd_stack) == 0:
                    # illegal right
                    index_right_tbd.append(i)
                else:
                    index_left_tbd_stack.pop()
        index_tbd = index_left_tbd_stack + index_right_tbd
        res = [s[i] for i in range(n) if i not in index_tbd]
        return ''.join(res)
                

sol = Solution()
sol.minRemoveToMakeValid(s)


"""[LeetCode] Asteroid Collision 行星碰撞
We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction
 (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one 
will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

Example 1:

Input: 
asteroids = [5, 10, -5]
Output: [5, 10]
Explanation: 
The 10 and -5 collide resulting in 10.  The 5 and 10 never collide.
"""

class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        res = []
        n = len(asteroids)
        for i in range(n):
            if asteroids[i] >= 0 or len(res) == 0 or res[-1] < 0:
                # no collision happens
                res.append(asteroids[i])
            else:
                # left aste can collide
                while len(res) > 0 and res[-1] > 0 and res[-1] < asteroids[i] * -1:
                    # while curr aste wins and collides happening
                    res.pop()
                if len(res) == 0 or res[-1] < 0:
                    # if new aste wins 
                    res.append(asteroids[i])
                if res[-1] > asteroids[i] * -1:
                    # if old aste wins
                    continue
                if res[-1] == asteroids[i] * -1:
                    # both killed
                    res.pop()
                    continue
        
        return res

asteroids = [10,2,-5]
sol = Solution()
sol.asteroidCollision(asteroids)

        
"""[LeetCode] 146. LRU Cache 最近最少使用页面置换缓存器
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its 
capacity, it should invalidate the least recently used item before inserting a new item.

Follow up:
Could you do both operations in O(1) time complexity?

Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
"""
# in python, one can use OrderedDict (a dict that has pipitem() to remove the last or first key)
from collections import OrderedDict

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.hashmap = OrderedDict()
        self.capacity = capacity
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.hashmap:
            return -1
        else:
            value = self.hashmap.pop(key)
            self.hashmap[key] = value
            return value
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.hashmap:
            self.hashmap.pop(key)
            self.hashmap[key] = value
        else:
            if len(self.hashmap) < self.capacity:
                self.hashmap[key] = value
            else:
                self.hashmap.popitem(last=False)
                self.hashmap[key] = value


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# if not using OrderedDict, need to use linked list to achieve O(1) runtime
# https://www.lintcode.com/problem/134/solution/16614
# https://leetcode.com/problems/lru-cache/discuss/1655397/Python3-Dictionary-%2B-LinkedList-or-Clean-%2B-Detailed-Explanation-or-O(1)-Time-For-All-Operations


"""[LeetCode] Set Matrix Zeroes 矩阵赋零
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.

click to show follow up.

Follow up:
Did you use extra space?
A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
"""
# idea: to avoid using extra space, we use first row and col of matrix to record zeros
# omit




"""[LeetCode] 380. Insert Delete GetRandom O(1) 常数时间内插入删除和获得随机数
Design a data structure that supports all following operations in average O(1) time.

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom: Returns a random element from current set of elements. Each 
element must have the same probability of being returned. 

Example:

// Init an empty set.
RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomSet.insert(1);

// Returns false as 2 does not exist in the set.
randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
randomSet.remove(1);

// 2 was already in the set, so return false.
randomSet.insert(2);

// Since 1 is the only number in the set, getRandom always return 1.
randomSet.getRandom();
"""
import random

class RandomizedSet:

    def __init__(self):
        self.hashmap = dict()
        self.vector = []
        self.size = 0

    def insert(self, val: int) -> bool:
        if val in self.hashmap:
            return False
        else:
            self.vector.append(val)
            self.hashmap[val] = self.size
            self.size = self.size + 1
            return True

    def remove(self, val: int) -> bool:
        if val in self.hashmap:
            location = self.hashmap[val]
            self.vector[location], self.vector[-1] = self.vector[-1], self.vector[location]
            self.hashmap[self.vector[location]] = location
            del self.hashmap[val]
            self.vector.pop()
            self.size = self.size - 1
            return True
        else:
            return False

    def getRandom(self) -> int:
        rand_index = random.randint(0, self.size-1)
        return self.vector[rand_index]


# 删除操作是比较 tricky 的，还是要先判断其是否在 HashMap 里，如果没有，直接返回 false。由于 HashMap 的
# 删除是常数时间的，而数组并不是，为了使数组删除也能常数级，实际上将要删除的数字和数组的最后一个数字
# 调换个位置，然后修改对应的 HashMap 中的值，这样只需要删除数组的最后一个元素即可，保证了常数时间
# 内的删除。而返回随机数对于数组来说就很简单了，只要随机生成一个位置，返回该位置上的数字即可
# Your RandomizedSet object will be instantiated and called as such:
obj = RandomizedSet()
param_1 = obj.insert(0); obj.vector; obj.hashmap
param_1 = obj.insert(1); obj.vector; obj.hashmap
param_2 = obj.remove(0); obj.vector; obj.hashmap
param_2 = obj.insert(2); obj.vector; obj.hashmap
param_2 = obj.remove(1); obj.vector; obj.hashmap
param_3 = obj.getRandom(); param_3


"""[LeetCode] 49. Group Anagrams 群组错位词
Given an array of strings, group anagrams together.

Example:

Input: 
["eat", "tea", "tan", "ate", "nat", "bat"]
,
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
Note:

All inputs will be in lowercase.
The order of your output does not matter.
"""
# use sort to tell whether Anagrams

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashmap = dict() # record anagrams
        for s in strs:
            s_key = ''.join(sorted(s))
            if s_key in hashmap:
                hashmap[s_key].append(s)
            else:
                hashmap[s_key] = [s]
        
        return list(hashmap.values())


""" 299. Bulls and Cows
You are playing the Bulls and Cows game with your friend.

You write down a secret number and ask your friend to guess what the number is. 
When your friend makes a guess, you provide a hint with the following info:

The number of "bulls", which are digits in the guess that are in the correct position.
The number of "cows", which are digits in the guess that are in your secret number but are 
located in the wrong position. Specifically, the non-bull digits in the guess that could be
 rearranged such that they become bulls.
Given the secret number secret and your friend's guess guess, return the hint for your 
friend's guess.

The hint should be formatted as "xAyB", where x is the number of bulls and y is the number of cows.
 Note that both secret and guess may contain duplicate digits.

 

Example 1:

Input: secret = "1807", guess = "7810"
Output: "1A3B"
Explanation: Bulls are connected with a '|' and cows are underlined:
"1807"
  |
"7810"
"""
# pay attention that count of B should not contain A. 
# omit
# you can only go through strings for once (see below)
# can also run through twice, note that when secret[i] == guess[i], do not add in hashmap (B)
def (secret, guess):
    bulls, cows = 0, 0
        dp1, dp2 = {}, {} # keep dictionaries of values which are not bulls
        for n1, n2 in zip(secret, guess):
            if n1 == n2:
                bulls += 1
            else:
                dp1[n1] = 1 if n1 not in dp1 else dp1[n1] + 1
                dp2[n2] = 1 if n2 not in dp2 else dp2[n2] + 1
        for k, v in dp2.items(): # go through your guess, determine if each digit is a cow
            v2 = dp1[k] if k in dp1 else None
            if v2 is not None:
                cows += min(v, v2)
        return str(bulls) + "A" + str(cows) + "B"



"""Design Tic-Tac-Toe 设计井字棋游戏 
Design a Tic-tac-toe game that is played between two players on a n x n grid.

You may assume the following rules:

A move is guaranteed to be valid and is placed on an empty block.
Once a winning condition is reached, no more moves is allowed.
A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.

https://www.cnblogs.com/grandyang/p/5467118.html

我们首先来O(n2)的解法，这种方法的思路很straightforward，就是建立一个nxn大小的board，其中0表示该位置没有棋子，
1表示玩家1放的子，2表示玩家2。那么棋盘上每增加一个子，我们都扫描当前行列，对角线，
和逆对角线(只有在row==col时才check)，看看是否有三子相连的情况，有的话则返回对应的玩家，没有则返回0，参见代码如下：

Follow up中让我们用更高效的方法，那么根据提示中的，我们建立一个大小为n的一维数组rows和cols，
还有变量对角线diag和逆对角线rev_diag，这种方法的思路是，如果玩家1在第一行某一列放了一个子，那么rows[0]自增1，
如果玩家2在第一行某一列放了一个子，则rows[0]自减1，那么只有当rows[0]等于n或者-n的时候，
表示第一行的子都是一个玩家放的，则游戏结束返回该玩家即可，其他各行各列，对角线和逆对角线都是这种思路，参见代码如下：

"""



"""
############################################################################
Heap／Priority Queue题目
# time complexity of building a heap is O(n)
# https://www.geeksforgeeks.org/time-complexity-of-building-a-heap/
# insert/remove max from max heap is O(logn)
# find max is O(1)
############################################################################
"""

"""[LeetCode] 973. K Closest Points to Origin 最接近原点的K个点
We have a list of points on the plane.  Find the K closest points to the origin (0, 0).
(Here, the distance between two points on a plane is the Euclidean distance.)
You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)

Example 1:

Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].
Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], K = 2
Output: [[3,3],[-2,4]]
(The answer [[-2,4],[3,3]] would also be accepted.)
"""
points = [[3,3],[5,-1],[-2,4]]; k=2
# solution 1: nlogn
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        return sorted(points, key = lambda p: p[0]**2 + p[1]**2)[0:k]

# solution2 : use heap, O(n)+O(klogn)
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        
        q = []
        for x,y in points:
            heapq.heappush(q, [x*x + y*y,[x,y]])
        result = heapq.nsmallest(k, q, key=lambda x: x[0])
        result = [j for i,j in result]
        return result


"""[LeetCode] 347. Top K Frequent Elements 前K个高频元素
Given a non-empty array of integers, return the k most frequent elements.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
Note:

You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
"""

import heapq
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = Counter(nums)
        q = []
        for key,val in counter.items():
            heapq.heappush(q, (key,val))
        
        topk = heapq.nlargest(k, q, lambda x: x[1])
        return [key for key,val in topk]


"""[LeetCode] 23. Merge k Sorted Lists 合并k个有序链表
 

Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6

利用了最小堆这种数据结构，首先把k个链表的首元素都加入最小堆中，它们会自动排好序。然后每次取出最小的那个元素
加入最终结果的链表中，然后把取出元素的下一个元素再加入堆中，下次仍从堆中取出最小的元素做相同的操作，
以此类推，直到堆中没有元素了，此时k个链表也合并为了一个链表，返回首节点即可

Anther idea is merge sort based on merging two sorted linked list
"""


# Definition for singly-linked list.
lists = [ListNode(5), ListNode(2), ListNode(3)]

import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        q = [(head.val, i) for i, head in enumerate(lists) if head]
        heapq.heapify(q)
        head = ListNode()
        dummy = head
        while len(q) > 0:
            _, min_idx = heapq.heappop(q)
            head.next = lists[min_idx]
            head = head.next
            lists[min_idx] = lists[min_idx].next
            if lists[min_idx]:
                heapq.heappush(q, (lists[min_idx].val, min_idx))

        return dummy.next


"""[LeetCode] 264. Ugly Number II 丑陋数之二
 

Write a program to find the n-th ugly number.

Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. 

Example:

1, 2, 3, 4, 5, 6, 8, 9, 10, 12
10
Note:  

1 is typically treated as an ugly number.
n does not exceed 1690.
Hint:
1) The naive approach is to call isUgly for every number until you reach the nth one. 
Most numbers are not ugly. Try to focus your effort on generating only the ugly ones.
2) An ugly number must be multiplied by either 2, 3, or 5 from a smaller ugly number.
3) The key is how to maintain the order of the ugly numbers. Try a similar approach of merging from three sorted lists: L1, L2, and L3.
4) Assume you have Uk, the kth ugly number. Then Uk+1 must be Min(L1 * 2, L2 * 3, L3 * 5).

这道题是之前那道 Ugly Number 的拓展，这里让找到第n个丑陋数，还好题目中给了很多提示，
基本上相当于告诉我们解法了，根据提示中的信息，丑陋数序列可以拆分为下面3个子列表：

(1) 1x2,  2x2, 2x2, 3x2, 3x2, 4x2, 5x2...
(2) 1x3,  1x3, 2x3, 2x3, 2x3, 3x3, 3x3...
(3) 1x5,  1x5, 1x5, 1x5, 2x5, 2x5, 2x5...
仔细观察上述三个列表，可以发现每个子列表都是一个丑陋数分别乘以 2，3，5，
而要求的丑陋数就是从已经生成的序列中取出来的，每次都从三个列表中取出当前最小的那个加入序列，请参见代码如下：


我们也可以使用最小堆来做，首先放进去一个1，然后从1遍历到n，每次取出堆顶元素，为了确保没有重复数字，
进行一次 while 循环，将此时和堆顶元素相同的都取出来，然后分别将这个取出的数字乘以 2，3，5，
并分别加入最小堆。这样最终 for 循环退出后，堆顶元素就是所求的第n个丑陋数
"""
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        res = [1]
        i2 = i3 = i5 = 0
        curr_ugly = 1
        while len(res) < n:
            num2, num3, num5 = res[i2]*2, res[i3]*3, res[i5]*5
            next_ugly = min(num2, min(num3, num5))
            if next_ugly == num2:
                i2 = i2 + 1
            if next_ugly == num3:
                i3 = i3 + 1
            if next_ugly == num5:
                i5 = i5 + 1
            if next_ugly > res[-1]:
                res.append(next_ugly)
        
        return res[-1]


soln = Solution()
soln.nthUglyNumber(11)


""" [LeetCode] 1086. High Five
Description
Each student has two attributes ID and scores. Find the average of the top five scores for each student.

Example
Example 1:

Input: 
[[1,91],[1,92],[2,93],[2,99],[2,98],[2,97],[1,60],[1,58],[2,100],[1,61]]
Output:
1: 72.40
2: 97.40

时间O(nlogn)
空间O(n)
"""
# use a max heap to store top 5 scores

results = [[1,91],[1,92],[2,93],[2,99],[2,98],[2,97],[1,60],[1,58],[2,100],[1,61]]

import heapq

class Solution:
    # @param {Record[]} results a list of <student_id, score>
    # @return {dict(id, average)} find the average of 5 highest scores for each person
    # <key, value> (student_id, average_score)
    def highFive(self, results):
        # Write your code here
        hashmap = dict()
        top5_res = dict()
        for result in results:
            id, score = result
            if id not in hashmap:
                hashmap[id] = [score]
                heapq.heapify(hashmap[id])
            else:
                heapq.heappush(hashmap[id], score)
        
        for k,v in hashmap.items():
            top5_scores = heapq.nlargest(5, v)
            top5_res[k] = sum(top5_scores) / 5
        
        return top5_res


sol = Solution()
sol.highFive(results)


""" [LeetCode] 88. Merge Sorted Array 混合插入有序数组
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and 
two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored 
inside the array nums1. To accommodate this, nums1 has a length of m + n, where the 
first m elements denote the elements that should be merged, and the last n elements 
are set to 0 and should be ignored. nums2 has a length of n.

Example 1:

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.

Follow up: Can you come up with an algorithm that runs in O(m + n) time?

"""
# O(m + n)
# fill from the end (largest)

class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        end1 = m-1
        end2 = n-1
        end = m+n-1

        while end1 >= 0 and end2 >= 0:
            if nums1[end1] < nums2[end2]:
                nums1[end] = nums2[end2]
                end2 = end2 - 1
            else:
                nums1[end] = nums1[end1]
                end1 = end1 - 1
            end = end - 1
        
        if end2 >= 0:
            # nums1 all used
            nums1[:end2+1] = nums2[:end2+1]
        
        return None


nums1 = [1,2,3,0,0,0]; m = 3; nums2 = [2,5,6]; n = 3
nums1 = [2,2,7,0,0,0]; m = 3; nums2 = [1,5,6]; n = 3

sol = Solution()
sol.merge(nums1, m, nums2, n); nums1


"""[LeetCode] 378. Kth Smallest Element in a Sorted Matrix 有序矩阵中第K小的元素
Given an n x n matrix where each of the rows and columns is sorted in ascending order, 
return the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

You must find a solution with a memory complexity better than O(n2).

Example 1:

Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13

Follow up:

Could you solve the problem with a constant memory (i.e., O(1) memory complexity)?
Could you solve the problem in O(n) time complexity? 
The solution may be too advanced for an interview but you may find reading this paper fun.
"""

# memory O(k): use a max heap of size k, and pop when size>k, top will be kth smallest
# memory O(1): !!!
matrix = [[1,5,9],[10,11,13],[12,13,15]]
k = 8
val = 13

class Solution:
    def kthSmallest(self, matrix, k: int) -> int:
        low, high = matrix[0][0], matrix[-1][-1]
        while low < high:
            mid = low + (high - low) // 2
            small_num = self.findRank(matrix, mid)
            if small_num < k:
                low = mid + 1
            else:
                # if =k, then should be higher bound
                high = mid
        
        return low
    
    def findRank(self, matrix, val):
        # find how many ele smaller or equal than val
        res = 0
        for row in matrix:
            low_idx, high_idx = 0, len(row) - 1
            while low_idx <= high_idx:
                mid_idx = low_idx + (high_idx - low_idx)//2
                if row[mid_idx] <= val:
                    low_idx = mid_idx + 1
                else:
                    high_idx = mid_idx - 1
            res = res + low_idx
        return res
    
    def count(self, matrix, val): #btw given = mid
        count = 0
        for i in matrix:
            for j in i:
                if j <= val:
                    count +=  1         
        return count


sol = Solution()
sol.kthSmallest(matrix, k)



"""295. Find Median from Data Stream
Share
The median is the middle value in an ordered integer list. If the size of the list is even, 
there is no middle value and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.

Follow up:

If all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?

Solution: logn for median
The basic idea is to have 2 heaps that hold two halfs of the array. A max_heap will hold the first half 
where the top will be the largest of the 1st half, and a min_heap for the upper half where its top will 
hold the minimum of the next half. Insert the elements alternatively, and if top of the max_heap is greater 
than the top of the min_heap, swap the elements to maintain lower half and upper half properties of the heaps. 
If the number of elements is even, take both tops / 2 and return the median, if it is odd return the top 
of the lower half.

Follow up: bucket search

"""
import heapq

class MedianFinder:
    def __init__(self):
        self.size = 0
        self.q_low_maxheap = []  # top is max
        self.q_high_minheap = [] # top is min
        heapq.heapify(self.q_low_maxheap)
        heapq.heapify(self.q_high_minheap)
        
    def addNum(self, num: int) -> None:
        if self.size % 2 == 0:
            # odd go to low heap (curr is even)
            heapq.heappush(self.q_low_maxheap, -1 * num)
        else:
            heapq.heappush(self.q_high_minheap, num)
        
        self.size = self.size + 1

        if self.size > 1:
            # adjust to make sure q_low always < q_high
            if -1 * self.q_low_maxheap[0] > self.q_high_minheap[0]:
                low = -1 * heapq.heappop(self.q_low_maxheap)
                high =  heapq.heappop(self.q_high_minheap)
                heapq.heappush(self.q_low_maxheap, -1 * high)
                heapq.heappush(self.q_high_minheap, low)
        
    def findMedian(self) -> float:
        if self.size % 2 == 1:
            # odd, take low heap
            return -1 * self.q_low_maxheap[0]
        else:
            low_median = -1 * self.q_low_maxheap[0]
            high_median = self.q_high_minheap[0]
            return (low_median + high_median) / 2

        
# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()


"""[LeetCode] 767. Reorganize String 重构字符串
Given a string S, check if the letters can be rearranged so that two characters that 
are adjacent to each other are not the same.

If possible, output any possible result.  If not possible, return the empty string.

Example 1:

Input: S = "aab"
Output: "aba"
Example 2:

Input: S = "aaab"
Output: ""
Note:

S will consist of lowercase letters and have length in range [1, 500].
"""
s = "aacdddb"
s = "aab"
import heapq
from collections import Counter

class Solution:
    def reorganizeString(self, s: str) -> str:
        countmap = Counter(s)
        q = [(-1*v, k) for k,v in countmap.items()]
        heapq.heapify(q)
        prev_freq, prev_letter = heapq.heappop(q)
        res = prev_letter

        while len(q) > 0:
            freq, letter = heapq.heappop(q)
            res = res + letter
            if prev_freq + 1 != 0:
                heapq.heappush(q, (prev_freq + 1, prev_letter))  # minus 1 freq
            prev_freq, prev_letter = freq, letter
        
        if len(res) == len(s):
            return res
        else:
            return ''



"""[LeetCode] 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
Given an array of integers nums and an integer limit, return the size of the longest non-empty 
subarray such that the absolute difference between any two elements of this subarray is 
less than or equal to limit.

Input: nums = [10,1,2,4,7,2], limit = 5
Output: 4 
Explanation: The subarray [2,4,7,2] is the longest since the maximum absolute diff is |2-7| = 5 <= 5.

Input: nums = [4,2,2,2,4,4,2,2], limit = 0
Output: 3

Solution
sliding window + 单调queue。注意不是单调栈，而是单调queue。

我们维护一个单调递减queue maxQueue和一个单调递增queue minQueue，里面存的都是下标。
maxQueue的首元素是当前遍历到的最大元素的下标，minQueue的首元素是当前遍历到的最小元素的下标。
注意存元素也可以，但是存下标的好处是如果有重复元素，对下标是没有影响的。

同时我们需要两个指针start和end。一开始end往后走，当发现

maxQueue不为空且maxQueue的最后一个元素小于当前元素nums[end]了，则不断往外poll元素，直到整个maxQueue变回降序
minQueue不为空且minQueue的最后一个元素大于当前元素nums[end]了，则不断往外poll元素，直到整个minQueue变回升序
此时再判断，如果两个queue都不为空但是两个queue里的最后一个元素（一个是最大值，一个是最小值）的差值大于limit了，
则开始左移start指针，左移的同时，如果两个queue里有任何元素的下标<= start，则往外poll，因为不需要了。
这里也是存下标的另一个好处，因为下标一定是有序被放进两个queue的，所以如果大于limit了，
你是需要从最一开始的start指针那里开始检查的。

时间O(n)

空间O(n)
"""

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        



"""[LeetCode] 895. Maximum Frequency Stack 最大频率栈

Implement `FreqStack`, a class which simulates the operation of a stack-like data structure.
FreqStack has two functions:

push(int x), which pushes an integer xonto the stack.
pop(), which removes and returns the most frequent element in the stack.
If there is a tie for most frequent element, the element closest to the top of the stack is removed and returned.

Input:
["FreqStack","push","push","push","push","push","push","pop","pop","pop","pop"],
[[],[5],[7],[5],[7],[4],[5],[],[],[],[]]
Output: [null,null,null,null,null,null,null,5,7,5,4]
Explanation:
After making six .push operations, the stack is [5,7,5,7,4,5] from bottom to top.  Then:

pop() -> returns 5, as 5 is the most frequent.
The stack becomes [5,7,5,7,4].

pop() -> returns 7, as 5 and 7 is the most frequent, but 7 is closest to the top.
The stack becomes [5,7,5,4].

pop() -> returns 5.
The stack becomes [5,7,4].

pop() -> returns 4.
The stack becomes [5,7].
"""

class FreqStack:

    def __init__(self):
        self.max_Freq = 0
        self.m_f2n = dict()  # freq -> [numbers]
        self.m_n2f = dict()  # number -> freq
        

    def push(self, val: int) -> None:
        # update n2f
        if val in self.m_n2f:
            self.m_n2f[val] = self.m_n2f[val] + 1
        else:
            self.m_n2f[val] = 1
        
        # update f2n
        if self.m_n2f[val] in self.m_f2n:
            self.m_f2n[self.m_n2f[val]].append(val)  # later to right
        else:
            self.m_f2n[self.m_n2f[val]] = [val]
        
        # update max_Freq
        if self.m_n2f[val] > self.max_Freq:
            self.max_Freq = self.m_n2f[val]


    def pop(self) -> int:
        val = self.m_f2n[self.max_Freq].pop()  # take from right
        if len(self.m_f2n[self.max_Freq]) == 0:
            self.max_Freq = self.max_Freq - 1
        
        self.m_n2f[val] = self.m_n2f[val] - 1
        return val
        

# Your FreqStack object will be instantiated and called as such:
obj = FreqStack()
obj.push(5) ; obj.m_f2n ; obj.m_n2f
obj.push(7) ; obj.m_f2n
obj.push(5) ; obj.m_f2n
obj.push(7) ; obj.m_f2n
obj.push(4) ; obj.m_f2n
obj.pop()
obj.pop()
obj.pop()
obj.pop()



"""
############################################################################
二分法（Binary Search)
############################################################################
Summary
https://www.cnblogs.com/grandyang/p/6854825.html

若 right 初始化为了 nums.size()，那么就必须用 left < right，而最后的 right 的赋值必须用 right = mid。
但是如果我们 right 初始化为 nums.size() - 1，那么就必须用 left <= right，并且right的赋值要写成 right = mid - 1，不然就会出错
"""
# one kind of code
def bin_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r-l)//2
        if nums[mid] == target:
            return mid # or do nothing
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1

# note: if ascending
# if finish: l = r + 1
# (if do nothing at meet condition): l (or r+1) returns the first element that is greater or equal than target 
# if search for the first number greater than target: do nothing and change line 1982 to nums[mid] <= target
# alternative
def bin_search(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r-l)//2
        if nums[mid] == target:
            return mid # or do nothing
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return -1

# note: if ascending
# if finish, l = r
# (if do nothing at meet condition): l (or r) returns first element that is greater or equal than target 


"""34. Find First and Last Position of Element in Sorted Array
Given an array of integers nums sorted in non-decreasing order, find the 
starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
"""
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        ind = self.bs(nums, target)
        if ind == -1:
            return [-1, -1]
        low, high = ind, ind
        while low >= 0 and nums[low] == target:
            low = low - 1
        while high < len(nums) and nums[high] == target:
            high = high + 1
        
        return [low+1, high-1]

    def bs(self, nums, target):
        # return index
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r-l)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        
        return -1


"""81. Search in Rotated Sorted Array II !!!
There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values).

Before being passed to your function, nums is rotated at an unknown pivot index k 
(0 <= k < nums.length) such that the resulting array is 
[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
For example, [0,1,2,4,4,4,5,6,6,7] might be rotated at pivot index 5 and become [4,5,6,6,7,0,1,2,4,4].

Given the array nums after the rotation and an integer target, return true if target is in nums, 
or false if it is not in nums.

You must decrease the overall operation steps as much as possible.

Example 1:

Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true

Solution
数组中允许出现重复数字，这个也会影响我们选择哪半边继续搜索，由于之前那道题不存在相同值，
我们在比较中间值和最右值时就完全符合之前所说的规律：如果中间的数小于最右边的数，
则右半段是有序的，若中间数大于最右边数，则左半段是有序的。而如果可以有重复值，
就会出现来面两种情况，[3 1 1] 和 [1 1 3 1]，对于这两种情况中间值等于最右值时，
目标值3既可以在左边又可以在右边，那怎么办么，对于这种情况其实处理非常简单，只要把最右值向左一位即可继续循环，
如果还相同则继续移，直到移到不同值为止，然后其他部分还采用 Search in Rotated Sorted Array 中的方法
"""

class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = low + (high - low) // 2
            if nums[mid] == target:
                return True
            if nums[mid] < nums[high]:
                # then right half is sorted
                if target > nums[mid] and target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
            elif nums[mid] > nums[high]:
                # left half is sorted
                if target >= nums[low] and target < nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                high = high - 1
        
        return False


""" 1095. Find in Mountain Array !!!
You may recall that an array arr is a mountain array if and only if:

arr.length >= 3
There exists some i with 0 < i < arr.length - 1 such that:
arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
arr[i] > arr[i + 1] > ... > arr[arr.length - 1]
Given a mountain array mountainArr, return the minimum index such that mountainArr.get(index) == target. 
If such an index does not exist, return -1.

You cannot access the mountain array directly. You may only access the array using a MountainArray interface:

MountainArray.get(k) returns the element of the array at index k (0-indexed).
MountainArray.length() returns the length of the array.
Submissions making more than 100 calls to MountainArray.get will be judged Wrong Answer. 
Also, any solutions that attempt to circumvent the judge will result in disqualification.

Example 1:

Input: array = [1,2,3,4,5,3,1], target = 3
Output: 2
Explanation: 3 exists in the array, at index=2 and index=5. Return the minimum index, which is 2.
"""

# """
# This is MountainArray's API interface.
# You should not implement it, or speculate about its implementation
# """
class MountainArray:
   def get(self, index: int) -> int:
   def length(self) -> int:


class Solution:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        # find the peak
        n = mountain_arr.length()
        peak_idx = self.find_peak(mountain_arr)
        left_idx = self.find_target(mountain_arr, 0, peak_idx, target, asc=True)
        right_idx = self.find_target(mountain_arr, peak_idx, n-1, target, asc=False)
        return left_idx if left_idx!=-1 else (right_idx if right_idx!=-1 else -1)

    def find_peak(self, mountain_arr):
        low, high = 0, mountain_arr.length() - 1
        while low < high:
            mid = low + (high - low) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid+1):
                low = mid + 1
            else:
                high = mid
        return low

    def find_target(self, mountain_arr, low, high, target, asc):
        while low <= high:
            mid = low + (high - low) // 2
            if mountain_arr.get(mid) == target:
                return mid
            if mountain_arr.get(mid) < target:
                if asc:
                    low = mid + 1
                else:
                    high = mid - 1
            else:
                if asc:
                    high = mid - 1
                else:
                    low = mid + 1
        return -1

        
"""[LeetCode] Find Peak Element 求数组的局部峰值 !!!
A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

Example 1:

Input: nums = 
[1,2,3,1]

Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
Example 2:

Input: nums = 
[
1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.
Note:

Your solution should be in logarithmic complexity!

由于题目中提示了要用对数级的时间复杂度，那么我们就要考虑使用类似于二分查找法来缩短时间，
由于只是需要找到任意一个峰值，那么我们在确定二分查找折半后中间那个元素后，和紧跟的那个元素比较下大小，
如果大于，则说明峰值在前面，如果小于则在后面。这样就可以找到一个峰值了
"""
# omit

"""[LeetCode] 240. Search a 2D Matrix II 搜索一个二维矩阵之二

Write an efficient algorithm that searches for a target value in an m x n integer matrix. 
The matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.

Input: matrix = 
[[1,4,7,11,15],
[2,5,8,12,19],
[3,6,9,16,22],
[10,13,14,17,24],
[18,21,23,26,30]], target = 5
Output: true

Solution
如果我们观察题目中给的那个例子，可以发现有两个位置的数字很有特点，左下角和右上角的数。左下角的 18，
往上所有的数变小，往右所有数增加，那么就可以和目标数相比较，如果目标数大，就往右搜，如果目标数小，
就往上搜。这样就可以判断目标数是否存在。当然也可以把起始数放在右上角，往左和下搜，停止条件设置正确就行。
"""
# omit



"""[LeetCode] 69. Sqrt(x) 求平方根
Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.

Example 1:

Input: 4
Output: 2
Example 2:

Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned.
"""
# first element greater than - 1
class Solution:
    def mySqrt(self, x: int) -> int:
        l, r = 0, x
        while l <= r:
            mid = l + (r-l)//2
            if mid * mid <= x:
                l = mid + 1
            else:
                r = mid - 1
        return l - 1
        

"""[LeetCode] Single Element in a Sorted Array 有序数组中的单独元素

Given a sorted array consisting of only integers where every element appears twice except 
for one element which appears once. Find this single element that appears only once.

Example 1:
Input: [1,1,2,3,3,4,4,8,8]
Output: 2

Example 2:
Input: [3,3,7,7,10,11,11]
Output: 10

Note: Your solution should run in O(log n) time and O(1) space.
"""

class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = low + (high - low) // 2
            if mid == len(nums)-1 or mid == 0:
                return nums[mid]
            if nums[mid] == nums[mid + 1]:
                if mid % 2 == 0:
                    # single on right
                    low = mid + 1
                else:
                    high = mid - 1
            elif nums[mid] == nums[mid - 1]:
                if mid % 2 == 0:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                return nums[mid]
        
        return nums[low-1]


""" 617 · Maximum Average Subarray II
Description
Given an array with positive and negative numbers, find the maximum average subarray 
which length should be greater or equal to given length k.
Example 1:
Input:
[1,12,-5,-6,50,3]
3
Output:
15.667
Explanation:
 (-6 + 50 + 3) / 3 = 15.667

Solution 1: 双指针遍历 用hashmap存cumsum方便计算 O(n^2)
Solution 2: O(nlogn)
先更新累加和数组 sums，注意这个累加和数组不是原始数字的累加，而是它们和 mid 相减的差值累加。
我们的目标是找长度大于等于k的子数组的平均值大于 mid，由于每个数组都减去了 mid，
那么就转换为找长度大于等于k的子数组的差累积值大于0。建立差值累加数组的意义就在于通过 sums[i] - sums[j] 
来快速算出j和i位置中间数字之和，那么只要j和i中间正好差k个数字即可，
然后 minSum 就是用来保存j位置之前的子数组差累积的最小值，所以当 i >= k 时，我们用 sums[i - k] 来
更新 minSum，这里的 i - k 就是j的位置，然后判断如果 sums[i] - minSum > 0了，
说明找到了一段长度大于等k的子数组平均值大于 mid 了，就可以更新 left 为 mid 了，我们标记 check 为 true，
并退出循环。在 for 循环外面，当 check 为 true 的时候，left 更新为 mid，否则 right 更新为 mid

# https://www.cnblogs.com/grandyang/p/8021421.html
# https://www.lintcode.com/problem/617/solution/23436 (suspicious dp)
"""
class Solution:
    """
    @param nums: an array with positive and negative numbers
    @param k: an integer
    @return: the maximum average
    """
    def maxAverage(self, nums, k):
        # write your code here
        low, high = min(nums), max(nums)

        while low <= high - 1e-05:
            mid = low + (high - low) / 2
            is_exist = self.check(nums, k, mid)  # whether exists a subarray (>k) with avg>=mid
            if is_exist:
                low = mid
            else:
                high = mid
        
        return low

    def check(self, nums, k, target):
        # whether exists a subarray (>k) with avg>=target
        cumsums = [nums[0]]
        for i in range(1, len(nums)):
            cumsums[i] = cumsums[i - 1] + nums[i] - target
        
        minSum = 0
        for i in range(len(nums)):
            minSum = min(minSum, cumsums[i-k])
            if i>=k and cumsums[i-k] > minSum:
                return True
        
        return False



"""[LeetCode] Random Pick with Weight 根据权重随机取点
Given an array w of positive integers, where w[i] describes the weight of index i,
write a function pickIndex which randomly picks an index in proportion to its weight.

Note:
1 <= w.length <= 10000
1 <= w[i] <= 10^5
pickIndex will be called at most 10000 times.

For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 
(i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%)

"""
import random

class Solution:
    def __init__(self, w):
        self.size = len(w)
        total_wts = sum(w)
        self.cumsums = [0] * self.size
        self.cumsums[0] = w[0] / total_wts
        for i in range(1, self.size):
            self.cumsums[i] = self.cumsums[i - 1] + w[i]/total_wts
        

    def pickIndex(self) -> int:
        rand = random.uniform(0, 1)
        low, high = 0, self.size - 1
        # find the smallest index that larger than rand
        while low <= high:
            mid = low + (high - low) // 2
            if self.cumsums[mid] > rand:
                high = mid - 1
            else:
                low = mid + 1

        return low


# Your Solution object will be instantiated and called as such:
obj = Solution(w)
obj.cumsums
obj.pickIndex()
obj.pickIndex()
obj.pickIndex()


"""[LeetCode] 1300. Sum of Mutated Array Closest to Target
Given an integer array arr and a target value target, return the integer value such that when we 
change all the integers larger than value in the given array to be equal to value, the sum of 
the array gets as close as possible (in absolute difference) to target.

In case of a tie, return the minimum such integer.

Notice that the answer is not necessarily a number from arr. 

Example 1:

Input: arr = [4,9,3], target = 10
Output: 3
Explanation: When using 3 arr converts to [3, 3, 3] which sums 9 and that's the optimal answer.
Example 2:

Input: arr = [2,3,5], target = 10
Output: 5
Example 3:

Input: arr = [60864,25176,27249,21296,20204], target = 56803
Output: 11361

Solution
思路是二分法。首先遍历input数组，得到数组所有元素的和sum以及数组中最大的数字max。需要寻找的这个res一定介于0 - max之间。
为什么呢？因为被修改的数字一定是需要小于value的，如果这个value大于数组中的最大元素，意味着没有任何一个数字被修改过，
所以value大于max是不成立的。所以在0 - max之间做二分搜索，并且每找到一个mid，就算一次sum和，
二分法逼近最接近target的sum之后，找到对应的mid即可。NlogN

# suspicious O(N) solution
# https://leetcode.com/problems/sum-of-mutated-array-closest-to-target/discuss/1390461/Python-average-O(N)-solution
"""

# omit





""" Leetcode 1060 Missing Element in Sorted Array  !!!
Given a sorted array A of unique numbers, find the K-th missing number starting from the leftmost number of the array.
 
Example 1:
Input: A = [4,7,9,10], K = 1
Output: 5
Explanation: 
The first missing number is 5.
Example 2:
Input: A = [4,7,9,10], K = 3
Output: 8
Explanation: 
The missing numbers are [5,6,8,...], hence the third missing number is 8.
Example 3:
Input: A = [1,2,4], K = 3
Output: 6
Explanation: 
The missing numbers are [3,5,6,7,...], hence the third missing number is 6.
 
Note:
1 <= A.length <= 50000
1 <= A[i] <= 1e7
1 <= K <= 1e8

Solution: 
If the missing numbers count of the whole array < k, then missing number must be 
after nums[n-1].  res = nums[n-1] + missingCount.

Otherwise, need to find out the starting index to calculate the missing number.

Use binary search to have mid as candidate. 
If missing count < k, then must fall on the right side. l = mid + 1.

Time Complexity: O(logn). n = nums.length.
Space: O(1).
"""
nums = [1,2,4]; K = 3
nums = [4,7,9,10,11,12,13,14]; K = 3  # 8
nums = [4,7,9,10]; K = 1

class Solution:
    def missingElement(self, nums, k):
        n = len(nums)
        total_miss = nums[-1] - nums[0] + 1 - n
        if k > total_miss:
            return nums[-1] + k - total_miss
        low, high = 0, n - 1
        # find the first idx s.t. num_miss on left >= k
        while low <= high:
            mid = low + (high - low) // 2
            num_miss_mid = self.get_num_miss(nums, mid)
            if num_miss_mid >= k:
                high = mid - 1
            else:
                low = mid + 1
        
        # low: smallest idx that has miss>=k
        # find the missing idx: delta is self.get_num_miss(nums, low) - k
        # even no delta, still move left by 1
        # e.g., nums = [4,7,9,10]; K = 1
        return nums[low]-1 - (self.get_num_miss(nums, low) - k)
    
    def get_num_miss(self, nums, idx):
        # number of missing on left of nums[idx]
        return nums[idx] - nums[0] + 1 - (idx + 1)


sol = Solution()
sol.missingElement(nums, K)


"""[LeetCode] 1044. Longest Duplicate Substring 最长重复子串
Given a string s, consider all duplicated substrings: (contiguous) substrings of s 
that occur 2 or more times. The occurrences may overlap.

Return any duplicated substring that has the longest possible length. If s does not 
have a duplicated substring, the answer is "".

Example 1:

Input: s = "banana"
Output: "ana"
Example 2:

Input: s = "abcd"
Output: ""
Constraints:

2 <= s.length <= 3 * 104
s consists of lowercase English letters.

# https://www.cnblogs.com/grandyang/p/14497723.html

"""




""" 1891 - Cutting Ribbons
Given an array of integers with elements representing lengths of ribbons. Your goal is to obtain k 
ribbons of equal length cutting the ribbons into as many pieces as you want. 
Find the maximum integer length L to obtain at least k ribbons of length L.

Example 1:

Input: arr = [1, 2, 3, 4, 9], k = 5
Output: 3
Explanation: cut ribbon of length 9 into 3 pieces of length 3, length 4 into two pieces 
one of which is length 3 and the other length 1,
and one piece is already is of length 3. So you get 5 total pieces (satisfying k) and 
the greatest length L possible which would be 3.

//OverallRuntime - O(N Log N), Space - O(1)

Solution:
given a length x, can calculate k O(N), then binary search, min=0, max=max(arr)
"""
arr = [1, 2, 3, 4, 9]; k = 5 # Output: 3

def greatestLength(arr, k):
    # assume k > 0
    low, high = 1, max(arr)  # size
    # find size <= size(k), i.e. smallest size making >=k ribbons
    while low <= high:
        mid = low + (high - low) // 2  # candidate size
        num_ribbon = get_num_ribbons(arr, mid)
        # if num_ribbon == k:
        #     return mid
        if num_ribbon > k:
            # bigger size
            low = mid + 1
        else:
            # need smaller size
            high = mid - 1

    return low

def get_num_ribbons(arr, size):
    res = 0
    for i in arr:
        res = res + i//size
    return res

greatestLength(arr, k)



"""
############################################################################
双指针（2 Pointer）
############################################################################
"""


"""[LeetCode] 647. Palindromic Substrings 回文子字符串
 
Given a string, your task is to count how many palindromic substrings in this string.
The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

Example 1:

Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
 
Example 2:

Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
 

Note:

The input string length won't exceed 1000.

Solution
用递归来做，而且思路非常的简单粗暴。就是以字符串中的每一个字符都当作回文串中间的位置，然后向两边扩散，
每当成功匹配两个左右两个字符，结果 res 自增1，然后再比较下一对。注意回文字符串有奇数和偶数两种形式，
如果是奇数长度，那么i位置就是中间那个字符的位置，所以左右两遍都从i开始遍历；如果是偶数长度的，
那么i是最中间两个字符的左边那个，右边那个就是 i+1，这样就能 cover 所有的情况啦，而且都是不同的回文子字符串
"""
# omit


"""[LeetCode] 15. 3Sum 三数之和  !!!

Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
Find all unique triplets in the array which gives the sum of zero.

Note:
Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
The solution set must not contain duplicate triplets.
 

    For example, given array S = {-1 0 1 2 -1 -4},

    A solution set is:
    (-1, 0, 1)
    (-1, -1, 2)
"""

def sum3(nums):
    n = len(nums)
    res = list()
    nums.sort()
    for i in range(n-2):
        if i and nums[i] == nums[i-1]:
            continue # no duplicate
        target = 0 - nums[i]
        l = i + 1  # no need to look backward bcz already calculated
        r = n - 1
        while(l < r):
            if nums[l]  + nums[r] > target: 
                r -= 1 
            elif nums[l] + nums[r] < target: 
                l += 1 
            else:
                res.append(tuple([nums[i], nums[l], nums[r]]))
                l += 1
                r -= 1              
    return  list(set(res))


"""[LeetCode] 16. 3Sum Closest 最近三数之和
 

Given an array nums of n integers and an integer target, find three integers 
in nums such that the sum is closest to target. Return the sum of the three integers.
 You may assume that each input would have exactly one solution.

Example:

Given array nums = [-1, 2, 1, -4], and target = 1.

The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

# similar to 3sum except use a closst to record the answer

# c++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int closest = nums[0] + nums[1] + nums[2];
        int diff = abs(closest - target);
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size() - 2; ++i) {
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                int newDiff = abs(sum - target);
                if (diff > newDiff) {
                    diff = newDiff;
                    closest = sum;
                }
                if (sum < target) ++left;
                else --right;
            }
        }
        return closest;
    }
};

我们还可以稍稍进行一下优化，每次判断一下，当 nums[i]*3 > target 的时候，
就可以直接比较 closest 和 nums[i] + nums[i+1] + nums[i+2] 的值，
返回较小的那个，因为数组已经排过序了，后面的数字只会越来越大，就不必再往后比较了
"""


"""[LeetCode] 18. 4Sum 四数之和
Given an array nums of n integers, return an array of all the unique quadruplets
 [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.

 

Example 1:

Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
Example 2:

Input: nums = [2,2,2,2,2], target = 8
Output: [[2,2,2,2]]

Solution: do a double loop for 2sum

# c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        set<vector<int>> res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < int(nums.size() - 3); ++i) {
            for (int j = i + 1; j < int(nums.size() - 2); ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                int left = j + 1, right = nums.size() - 1;
                while (left < right) {
                    long sum = (long)nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        vector<int> out{nums[i], nums[j], nums[left], nums[right]};
                        res.insert(out);
                        ++left; --right;
                    } else if (sum < target) ++left;
                    else --right;
                }
            }
        }
        return vector<vector<int>>(res.begin(), res.end());
    }
};
那么这道题使用 HashMap 是否也能将时间复杂度降到 O(n2) 呢？答案是肯定的，
如果把A和B的两两之和都求出来，在 HashMap 中建立两数之和跟其出现次数之间的映射，
那么再遍历C和D中任意两个数之和，只要看哈希表存不存在target-这两数之和就行了
但要注意把index有overlap的去掉
"""


"""[LeetCode] 454. 4Sum II 四数之和之二
 

Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) 
there are such that A[i] + B[j] + C[k] + D[l] is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. 
All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.

Example:

Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2

Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0

Solution: 
遍历所有的情况，时间复杂度为 O(n4)。但是既然 Two Sum 那道都能将时间复杂度缩小一倍，
那么这道题使用 HashMap 是否也能将时间复杂度降到 O(n2) 呢？答案是肯定的，
如果把A和B的两两之和都求出来，在 HashMap 中建立两数之和跟其出现次数之间的映射，
那么再遍历C和D中任意两个数之和，只要看哈希表存不存在这两数之和的相反数就行了

class Solution {
public:
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        int res = 0;
        unordered_map<int, int> m;
        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < B.size(); ++j) {
                ++m[A[i] + B[j]];
            }
        }
        for (int i = 0; i < C.size(); ++i) {
            for (int j = 0; j < D.size(); ++j) {
                int target = -1 * (C[i] + D[j]);
                res += m[target];
            }
        }
        return res;
    }
};
"""


"""[LeetCode] 277. Find the Celebrity 寻找名人
Suppose you are at a party with n people (labeled from 0 to n - 1) and among them, 
there may exist one celebrity. The definition of a celebrity is that all the other n - 1
 people know him/her but he/she does not know any of them.

Now you want to find out who the celebrity is or verify that there is not one. 
The only thing you are allowed to do is to ask questions like: "Hi, A. Do you know B?" 
to get information of whether A knows B. You need to find out the celebrity 
(or verify there is not one) by asking as few questions as possible (in the asymptotic sense).

You are given a helper function bool knows(a, b)which tells you whether A knows B. 
Implement a function int findCelebrity(n). There will be exactly one celebrity if 
he/she is in the party. Return the celebrity's label if there is a celebrity in the 
party. If there is no celebrity, return -1.

optimal solution: (O(n))
首先loop一遍找到一个人i使得对于所有j(j>=i)都不认识i。
然后再loop一遍判断是否有人不认识i或者i认识某个人

bool knows(int a, int b);
class Solution {
public:
    int findCelebrity(int n) {
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (knows(res, i)) res = i;  # if not (res, i) then we know i is not cele
        }
        # before res, no information
        # after res, res does not know anyone
        for (int i = 0; i < res; ++i) {
            if (knows(res, i) || !knows(i, res)) return -1;
        }
        for (int i = res + 1; i < n; ++i) {
            if (!knows(i, res)) return -1;
        }
        return res;
    }
};
"""


"""11. Container With Most Water
You are given an integer array height of length n. There are n vertical 
lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the 
container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.

Solution:
定义i和j两个指针分别指向数组的左右两端，然后两个指针向中间搜索，
每移动一次算一个值和结果比较取较大的，容器装水量的算法是找出左右两个边缘中较小的那个乘以两边缘的距离
移动长边指针是不可能增加容量的
# C++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int res = 0, i = 0, j = height.size() - 1;
        while (i < j) {
            res = max(res, min(height[i], height[j]) * (j - i));
            if (height[i] < height[j]){++i} # we'll never increase volume if we move longer side
            else {--j}
        }
        return res;
    }
};
"""

"""[LeetCode] Move Zeroes 移动零
Given an array nums, write a function to move all 0's to the end of it while maintaining 
the relative order of the non-zero elements.

For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

Note:
You must do this in-place without making a copy of the array.
Minimize the total number of operations.

Hint: 
A two-pointer approach could be helpful here. The idea would be to have one pointer for 
iterating the array and another pointer that just works on the non-zero elements of the array.

class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        j = 0  #  index of first-zero
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] != 0) {
                swap(nums[i], nums[j]);
                j++
            }
        }
    }
};

"""


"""[LeetCode] 395. Longest Substring with At Least K Repeating Characters 至少有K个重复字符的最长子字符串
Find the length of the longest substring T of a given string (consists of lowercase letters only) such that 
every character in T appears no less than k times.

Example 1:

Input:
s = "aaabb", k = 3

Output:
3

The longest substring is "aaa", as 'a' is repeated 3 times.


Two pointer
https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/discuss/1655754/Python-2-Pointer-solution
"""
from collections import Counter

class Solution:
    def helper(self, s, k, start, end):
        # length of longest substring of s[strat:end]
        if len(s[start:end]) < k:
            return 0
        
        m = Counter(s[start:end])
        for i in range(start, end):
            if m[s[i]] < k:
                # curr_letter cannot be included 
                return max(self.helper(s, k, start, i), self.helper(s, k, i+1, end))
        
        return end-start # all letters >=k

    def longestSubstring(self, s: str, k: int) -> int:
        return self.helper(s=s, k=k, start=0, end=len(s))


sol = Solution()
sol.longestSubstring("ababacb", 3)
sol.longestSubstring("ababa", 3)
sol.longestSubstring("a", 3)


""" 386 · Longest Substring with At Most K Distinct Characters
Description
Given a string S, find the length of the longest substring T that contains at most k distinct characters.

Example 1:

Input: S = "eceba" and k = 3
Output: 4
Explanation: T = "eceb"

Example 2:

Input: S = "WORLD" and k = 4
Output: 4
Explanation: T = "WORL" or "ORLD"
Challenge
O(n) time

# solution
two pointer and one dict
"""

def lengthOfLongestSubstringKDistinct(s, k):
    l, r = 0, 0
    n = len(s)
    m = dict()
    res = 0
    while r < n:
        m[s[r]] = 1 if r not in m else m[s[r]] + 1
        while len(m) > k:
            # if more than k in s[l:r+1] then move l till =k
            m[s[l]] = m[s[l]] - 1
            if m[s[l]] == 0:
                del m[s[l]]
            l = l + 1
        r = r + 1
        res = max(res, r-l+1)
    return res
        


"""[LeetCode] Longest Repeating Character Replacement 最长重复字符置换

Given a string that consists of only uppercase English letters, you 
can replace any letter in the string with another letter at most k times. 
Find the length of a longest substring containing all repeating letters 
you can get after performing the above operations.

Note:
Both the string's length and k will not exceed 104.

Example 1:

Input:
s = "ABAB", k = 2

Output:
4

Explanation:
Replace the two 'A's with two 'B's or vice versa.
Example 2:

Input:
s = "AABABBA", k = 1

Output:
4

"""
from collections import Counter

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # two pters
        # condition: if r - l + 1 - maxCnt <= k then okay else move left pters
        m = dict()
        l = 0
        maxCnt = 0  # maxCnt of letter in s[l:r+1]
        res = 0
        for r in range(len(s)):
            m[s[r]] = m[s[r]] + 1 if s[r] in m else 1
            maxCnt = max(maxCnt, m[s[r]])
            while r - l + 1 - maxCnt > k:
                # will not make r-l+1 as output
                m[s[l]] = m[s[l]] - 1
                l = l + 1
                
            res = max(res, r-l+1)
        
        return res


"""[LeetCode] 76. Minimum Window Substring 最小窗口子串
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

Example:
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"

Example 2:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.

Note:
If there is no such window in S that covers all characters in T, return the empty string "".
If there is such window, you are guaranteed that there will always be only one unique minimum window in S.

"""
from collections import Counter

s = "ADOBECODEBANC"; t = "ABC"
s="cabwefgewcwaefgcf"; t="cae"

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        m = Counter(t)  # number of remaining letters
        num_letters_covered = 0
        l = 0
        min_len = len(s) + 1
        res = ""
        for r in range(len(s)):
            if s[r] in m:
                m[s[r]] = m[s[r]] - 1
                if m[s[r]] >= 0:
                    num_letters_covered = num_letters_covered + 1

            while num_letters_covered == len(t):
                # move l 
                if r+1-l < min_len:
                    res = s[l:r+1]
                    min_len = r+1-l
                # if s[l:r+1] covers T
                if s[l] in m:
                    m[s[l]] = m[s[l]] + 1
                    if m[s[l]] > 0:
                        num_letters_covered = num_letters_covered - 1
                l = l + 1
                print(s[l:r+1])
                print(m)
                
        return res

class Solution2:
    def minWindow(self, s: str, t: str) -> str:
        m = Counter(t)  # number of remaining letters
        l = 0
        min_len = len(s) + 1
        res = ""
        for r in range(len(s)):
            if s[r] in m:
                m[s[r]] = m[s[r]] - 1
            while max(m.values()) <= 0:
                # if all gets covered
                if r+1-l < min_len:
                    res = s[l:r+1]
                    min_len = r+1-l
                # if s[l:r+1] covers T
                if s[l] in m:
                    m[s[l]] = m[s[l]] + 1
                l = l + 1
                
        return res


sol = Solution()
sol.minWindow("cabwefgewcwaefgcf", "cae")
sol.minWindow(s="ewcwae", t="cae")


""" 1004. Max Consecutive Ones III
Given a binary array nums and an integer k, return the maximum number of consecutive 
1's in the array if you can flip at most k 0's.

Example 1:

Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
"""
class Solution:
    def longestOnes(self, nums, k: int) -> int:
        num_ones = 0
        l = 0
        res = 0
        for r in range(len(nums)):
            if nums[r] == 1:
                num_ones = num_ones + 1
            if num_ones + k >= r + 1 - l:
                # fit
                res = max(r - l + 1, res)
            else:
                # move l
                while num_ones + k < r + 1 - l:
                    num_ones = num_ones - 1 if nums[l] == 1 else num_ones
                    l = l + 1
        
        return res


nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]; k = 3
nums = [1,1,1,0,0,0,1,1,1,1,0]; k = 2
sol = Solution()
sol.longestOnes(nums, k)



"""
############################################################################
宽度优先搜索（BFS）
############################################################################
迷宫遍历求最短路径
建立距离场
有向图遍历
拓扑排序
"""

"""[LeetCode] 102. Binary Tree Level Order Traversal 二叉树层序遍历
 

Given a binary tree, return the level order traversal of its nodes' values. 
(ie, from left to right, level by level).

For example:
Given binary tree {3,9,20,#,#,15,7},

    3
   / \
  9  20
    /  \
   15   7
 

return its level order traversal as:

[
  [3],
  [9,20],
  [15,7]
]
"""

# Definition for a binary tree node.
from collections import deque
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return root
        res = []
        q = deque()
        q.append(root)
        while len(q) > 0:
            curr_res = []
            for _ in range(len(q)):
                curr_node = q.popleft()
                if curr_node:
                    curr_res.append(curr_node.val)
                    q.append(curr_node.left)
                    q.append(curr_node.right)
            if len(curr_res) > 0:
                res.append(curr_res)
        
        return res


# recursive solution
# use a list of list
# use level to indicate where to push
# use level and res.size() to decide whether add a [] to res
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        self.helper(root, 0, res)
        return res

    def helper(self, root, level, res):
        if root is None:
            return None
        
        if len(res) < level+1:
            # key: create [] for level
            res.append([])
        res[level].append(root.val)

        self.helper(root.left, level+1, res)
        self.helper(root.right, level+1, res)
        return None



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
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# solution 1: BFS
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = ""
        q = deque()
        q.append(root)
        while len(q) > 0:
            for _ in range(len(q)):
                curr_node = q.popleft()
                if curr_node:
                    res = f"{res}{curr_node.val},"
                    q.append(curr_node.left)
                    q.append(curr_node.right)
                else:
                    res = f"{res}#,"
        # 1,2,3,#,#,4,5,#,#,#,#
        return res.strip(',')  # remove the end comma

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        data = data.split(',')
        if data[0] == '#':
            return None
        root = TreeNode(int(data[0]))
        q = deque()
        q.append(root)
        idx = 1
        while len(q) > 0:
            for _ in range(len(q)):
                curr_node = q.popleft()
                if data[idx] != '#':
                    curr_node.left = TreeNode(int(data[idx]))
                    q.append(curr_node.left)
                idx = idx + 1
                if data[idx] != '#':
                    curr_node.right = TreeNode(int(data[idx]))
                    q.append(curr_node.right)
                idx = idx + 1
        
        return root

        
# solution 2: di gui ???
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = ""
        self.serialize_helper(root, res)
        return res.strip(",")
    
    def serialize_helper(self, root, res):
        if root is None:
            res = f"{res}#,"
        else:
            res = f"{res}{root.val},"
            self.serialize_helper(root.left, res)
            self.serialize_helper(root.right, res)
        return None

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        data = data.split(',')
        if data[0] == '' or data[0] == '#':
            return None
        return self.deserialize_helper(data)

        # root = TreeNode(int(data[0]))
        # data = ','.join(data[1:])
        # root.left = self.deserialize(data)
        # root.right = self.deserialize(data)
        # return root
        
    def deserialize_helper(self, res):
        # res is a list
        if res[0] == '#':
            return None
        root = TreeNode(int(res[0]))
        res = res[1:]
        root.left = self.deserialize_helper(res)
        root.right = self.deserialize_helper(res)
        return root


class Codec: # ???
    def serialize(self, root):
        return ','.join(self.serialize_helper(root))
    
    def serialize_helper(self, root):
        # write your code here
        if not root:
            return ['#']
        ans = []
        ans.append(str(root.val))
        ans += self.serialize(root.left)
        ans += self.serialize(root.right)
        return ans

    def deserialize(self, data):
        return self.deserialize_helper(data.split(','))

    def deserialize_helper(self, data):
        # write your code here
        ch = data.pop(0)
        if ch == '#':
            return None
        else:
            root = TreeNode(int(ch))
        root.left = self.deserialize(data)
        root.right = self.deserialize(data)
        return root


"""[LeetCode] 314. Binary Tree Vertical Order Traversal 二叉树的竖直遍历

Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.

Examples 1:
Input: 
[3,9,20,null,null,15,7]

   3
  /\
 /  \
 9  20
    /\
   /  \
  15   7 

Output:

[
  [9],
  [3,15],
  [20],
  [7]
]

BFS: use (node, order_idx (vertical order), time (horizontal order)) to push, 
left child has order_idx - 1, right + 1
for every new node, has time+1  # in case order, if order and val is the same, need to use time to sort
用一个 TreeMap 来建立序号和其对应的节点值的映射，用 TreeMap 的另一个好处是其自动排序功能可以让列从左到右
"""
# Definition for a binary tree node.
from collections import defaultdict

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution_BFS:
    # recommend DFS below
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        
        # when level + 1, then left child order =- 1, right order =+ 1
        q = deque()
        q.append((0, root, 0))  # order, node, time
        m = defaultdict(list)  # key as order, value as (time, node.val)
        while len(q) > 0:
            for _ in range(len(q)):
                order_idx, curr_node, t = q.popleft()
                if curr_node:
                    m[order_idx].append((t, curr_node.val))
                    if curr_node.left:
                        q.append((order_idx-1, curr_node.left, t+1))  # time(horizontal order)+1
                    if curr_node.right:
                        q.append((order_idx+1, curr_node.right, t+1))
        
        res = []
        for k in sorted(m.keys()):
            sorted_tup = sorted(m[k], key=lambda x:(x[0], x[1])) # sort by time and val
            res.append([tup[1] for tup in sorted_tup])  
        
        return res

# DFS: similarly, build a hashmap, and recursively save vertical, horizontal locations and val
# then sort by vertical, horizontal and val
class Solution_DFS:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        m = defaultdict(list)
        self.dfs_helper(root, 0, 0, m)
        res = []
        for k in sorted(m.keys()): # sort by v
            sorted_tup = sorted(m[k], key=lambda x: (x[0], x[1])) # sort by h and val
            res.append([tup[1] for tup in sorted_tup])
        return res

    def dfs_helper(self, node, v, h, m):
        if node is None:
            return None
        m[v].append((h, node.val))
        self.dfs_helper(node.left, v-1, h+1, m)
        self.dfs_helper(node.right, v+1, h+1, m)
        return None


""" Clone Graph
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
"""
from collections import deque, defaultdict

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:
    # BFS
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return None
        m = defaultdict()  # map from old node to new node
        q = deque()
        q.append(node)
        new_node = Node(node.val)
        m[node] = new_node
        while len(q) > 0:
            curr_old_node = q.popleft()
            for old_neighbor in curr_old_node.neighbors:
                if old_neighbor not in m:
                    new_neighbor = Node(old_neighbor.val)
                    m[old_neighbor] = new_neighbor
                    q.append(old_neighbor)
                
                m[curr_old_node].neighbors.append(m[old_neighbor])

        return m[node]


# DFS
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return None
        m = defaultdict()  # map from old node to new node
        # new_node = Node(node.val)
        # m[node] = new_node
        self.helper(node, m)
        return m[node]

    def helper(self, node, m):
        if node in m:
            return m[node]
        m[node] = Node(node.val)
        # given old node and m, given the new node
        for old_neighbor in node.neighbors:
            m[old_neighbor] = self.helper(old_neighbor, m)
            m[node].neighbors.append(m[old_neighbor])

        return m[node]



"""[LeetCode] 815. Bus Routes 公交线路

We have a list of bus routes. Each routes[i] is a bus route that the i-th bus repeats forever. 
For example if routes[0] = [1, 5, 7], this means that the first bus (0-th indexed) 
travels in the sequence 1->5->7->1->5->7->1->... forever.

We start at bus stop S (initially not on a bus), and we want to go to bus stop T. 
Travelling by buses only, what is the least number of buses we must take to reach our 
destination? Return -1 if it is not possible.

Example:
Input: 
routes = [[1, 2, 7], [3, 6, 7]]
S = 1
T = 6
Output: 2
"""

routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]]
source = 15
target = 12

from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        if source == target:
            return 0
        
        stop2bus = defaultdict(list)
        for bus_idx, route in enumerate(routes):
            for stop in route:
                stop2bus[stop].append(bus_idx)
        
        res = 1
        q = deque()
        visited_bus = set()
        for bus in stop2bus[source]:
            q.append(bus)
            visited_bus.add(bus)

        while len(q) > 0:
            for _ in range(len(q)):
                curr_bus = q.popleft()
                if target in routes[curr_bus]:
                    return res
                for stop_in_curr_bus in routes[curr_bus]:
                    for next_bus in stop2bus[stop_in_curr_bus]:
                        if next_bus not in visited_bus:
                            q.append(next_bus)
                            visited_bus.add(next_bus)

            res = res + 1
        return -1


routes = [[1,2,7],[3,6,7]]
source = 1
target = 6
# use stop instead of bus
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:        
        stop2bus = defaultdict(list)
        for bus_idx, route in enumerate(routes):
            for stop in route:
                stop2bus[stop].append(bus_idx)
        
        q = deque()
        q.append((source, 0))
        visited_stop = set([source])

        while len(q) > 0:
            for _ in range(len(q)):
                curr_stop, curr_step = q.popleft()
                if curr_stop == target:
                    return curr_step
                for curr_bus in stop2bus[curr_stop]:
                    for stop in routes[curr_bus]:
                        if stop not in visited_stop:
                            q.append((stop, curr_step+1))
                            visited_stop.add(stop)
                    routes[curr_bus] = []  # set empty 
        
        return -1


"""Shortest Distance from All Buildings 建筑物的最短距离 
You want to build a house on an empty land which reaches all buildings in the 
shortest amount of distance. You can only move up, down, left and right. You 
are given a 2D grid of values 0, 1 or 2, where:
•Each 0 marks an empty land which you can pass by freely.
•Each 1 marks a building which you cannot pass through.
•Each 2 marks an obstacle which you cannot pass through.

For example, given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2):
1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

The point (1,2) is an ideal empty land to build a house, as the total travel 
distance of 3+3+1=7 is minimal. So return 7.

Note:
There will be at least one building. If it is not possible to build such house 
according to the above rules, return -1.

BFS遍历，对每个building都建立一个dist的距离场，加起来。实际中可以有一个去cumulate
"""
grid = [
    [1, 0, 2, 0, 1], 
    [0, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0]
    ]

grid = [
    [1, 0, 1, 0, 1], 
    [0, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0]
    ]
"""
Step 0 initial dist matrix
[[inf, 0, inf, 0, inf], 
[0, 0, 0, 0, 0], 
[0, 0, inf, 0, 0]]

Step 1 dist matrix
[[<inf>, 1, inf, 5, inf], 
[1, 2, 3, 4, 5], 
[2, 3, inf, 5, 6]]

Step 2 cumulatively add dist
[[inf, 2, <inf>, 6, inf], 
[4, 4, 4, 6, 8], 
[6, 6, inf, 8, 10]]
"""

def shortestDistance(grid):
    num_buildings = 0
    nrows = len(grid)
    ncols = len(grid[0])
    dist = [[float('Inf') for _ in range(ncols)] for _ in range(nrows)]
    visitable = [[0 for _ in range(ncols)] for _ in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            if grid[i][j] == 1:
                num_buildings += 1
                update_distance(i, j, grid, dist, visitable)
    res = float('Inf')
    for i in range(nrows):
        for j in range(ncols):
            if visitable[i][j] == num_buildings:
                # (i,j) reachable by num of buildings
                res = min(res, dist[i][j])
    
    return res if res < float('Inf') else -1


def update_distance(i, j, grid, dist, visitable):
    nrows = len(grid)
    ncols = len(grid[0])
    # for building X at gird[i][j], update dist matrix
    visited = [[False for _ in range(ncols)] for _ in range(nrows)]
    q = deque()
    q.append((i, j, 0))
    visited[i][j] = True
    directs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    while len(q) > 0:
        for _ in range(len(q)):
            curr_i, curr_j, curr_dist = q.popleft()
            for direct in directs:
                next_i, next_j = curr_i+direct[0], curr_j+direct[1]
                if 0<=next_i<nrows and 0<=next_j<ncols and grid[next_i][next_j]==0 and visited[next_i][next_j] is False:
                    q.append((next_i, next_j, curr_dist+1))
                    visitable[next_i][next_j] += 1
                    visited[next_i][next_j] = True
                    if dist[next_i][next_j] == float('Inf'):
                        dist[next_i][next_j] = curr_dist+1
                    else:
                        dist[next_i][next_j] = dist[next_i][next_j] + curr_dist+1

    return None


shortestDistance(grid)


"""1293. Shortest Path in a Grid with Obstacles Elimination
You are given an m x n integer matrix grid where each cell is either 0 (empty) or 1 (obstacle). 
You can move up, down, left, or right from and to an empty cell in one step.

Return the minimum number of steps to walk from the upper left corner (0, 0) to the lower 
right corner (m - 1, n - 1) given that you can eliminate at most k obstacles. If it is not possible to find such walk return -1.

hint: use a triple to save status, (row, col, curr_k) (how many elimination happens) in stead of (row, col)
"""

grid = [[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]]; k = 1  # 6
grid = [[0,1,1],[1,1,1],[1,0,0]]; k = 1  # -1
grid = [[0,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,0],[0,1],[0,1],[0,1],[0,0],[1,0],[1,0],[0,0]]; k=4  # 14

class Solution:
    def shortestPath(self, grid, k: int) -> int:
        nrows, ncols = len(grid), len(grid[0])
        visited = set((0, 0, 0))
        q = deque()
        q.append((0, 0, 0, 0))  # i, j, dist, k
        directs = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        while len(q) > 0:
            for _ in range(len(q)):
                i, j, curr_dist, curr_k = q.popleft()
                if i==nrows-1 and j == ncols - 1:
                    return curr_dist
                    # print(curr_dist)
                for direct in directs:
                    next_i, next_j = i+direct[0], j+direct[1]
                    if 0<=next_i<nrows and 0<=next_j<ncols:
                        # two scenarios:
                        if grid[next_i][next_j] == 0 and (next_i, next_j, curr_k) not in visited:
                            visited.add((next_i, next_j, curr_k))
                            q.append((next_i, next_j, curr_dist+1, curr_k))
                        elif grid[next_i][next_j] == 1 and curr_k + 1 <= k and (next_i, next_j, curr_k+1) not in visited:
                            visited.add((next_i, next_j, curr_k+1))
                            q.append((next_i, next_j, curr_dist+1, curr_k+1))
        
        return -1


sol=Solution()
sol.shortestPath(grid, k)



"""Sequence Reconstruction 序列重建
Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. The org sequence is a 
permutation of the integers from 1 to n, with 1 ≤ n ≤ 104. Reconstruction means building a shortest common supersequence
 of the sequences in seqs (i.e., a shortest sequence so that all sequences in seqs are subsequences of it). Determine 
 whether there is only one sequence that can be reconstructed from seqs and it is the org sequence.

Example 1:

Input:
org: [1,2,3], seqs: [[1,2],[1,3]]

Output:
false

Example 3:

Input:
org: [1,2,3], seqs: [[1,2],[1,3],[2,3]]

Output:
true

Input:org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
Output:true

和course schedule ii 思路相似，利用seqs来重构org，如果发现1）无法重构2）有多种情况就返回false
多种情况的判断基于是否每一步q里都只有一个元素加进来（indegree为0的只有一个）
# https://www.lintcode.com/problem/605/solution/34974
"""
from collections import defaultdict, deque

org = [1,2,3]; seqs = [[1,2],[1,3]]
org = [4,1,5,2,6,3]; seqs = [[5,2,6,3],[4,1,5,2]]
# idea, [1,2,3] the position of 2 is confirmed only when we see [1,2], therefore [1,3] does not mean anything
def sequenceReconstruction(org, seqs):
    pre_m = defaultdict(list)  # m[p] = c: p is before c
    indegree = dict()  # how many numbers in front
    for seq in seqs:
        for i in range(len(seq)-1):
            pre_m[seq[i]].append(seq[i+1])
            indegree[seq[i+1]] = 1 if seq[i+1] not in indegree else indegree[seq[i+1]] + 1
            # indegree[seq[i+1]] = indegree.get(seq[i+1], 0) + 1
    
    n_org = len(org)
    q = deque()
    for seq in seqs:
        for i in range(len(seq)): 
            if seq[i] < 1 or seq[i] > n_org:
                # 1-n
                # print(-1)
                return False
            # put all numbers that have no pre in the queue
            if seq[i] not in indegree:
                q.append(seq[i])
    res = []  # reconstruct results
    while len(q) > 0:
        if len(q) > 1:
            # have +1 option for the next
            return False
        curr_num = q.popleft()
        res.append(curr_num)
        # update indegree based on pre_m
        for num in pre_m[curr_num]:
            indegree[num] -= 1
            if indegree[num] == 0:
                q.append(num)
    if res == org:
        return True 
    return False


sequenceReconstruction(org, seqs)
sequenceReconstruction([4,1,5,2,6,3], [[5,2,6,3],[4,1,5,2]])


"""[LeetCode] 269. Alien Dictionary 另类字典
There is a new alien language which uses the latin alphabet. However, the order 
among letters are unknown to you. You receive a list of non-empty words from the 
dictionary, where words are sorted lexicographically by the rules of this new language. 
Derive the order of letters in this language.

Example 1:
Input: ["wrt","wrf","er","ett","rftt"]
Output："wertf"

Example 2
Input：["z","x"]
Output："zx"
Explanation：
from "z" and "x"，we can get 'z' < 'x'
So return "zx"

# 如果同时在indegree==0的字符里出现了 b 和 c，先输出正常字典序最小的，所以这里使用最小堆很对口。
"""

from collections import defaultdict
import heapq
words = ["wrt","wrf","er","ett","rftt"]
words = ["ca", "cb"]  # abc
alienOrder(words)

def alienOrder(words):
    # Write your code here
    n_words = len(words)
    pre_m = defaultdict(list)
    indegree = dict()
    set_unique_letters = set(''.join(words))
    num_unique_letters = len(set_unique_letters)

    # first go through word by word: no such limitation!!!!
    # for word in words:
    #     for i in range(len(word)-1):
    #         if word[i] != word[i+1] and word[i+1] not in pre_m[word[i]]:
    #             pre_m[word[i]].append(word[i+1])  # p -> c
    #             indegree[word[i+1]] = indegree.get(word[i+1], 0) + 1

    # second go through index of diff words
    for word_idx in range(n_words - 1):
        curr_word, next_word = words[word_idx], words[word_idx+1]
        min_word_len = min(len(curr_word), len(next_word))
        for loc_idx in range(min_word_len):
            if curr_word[loc_idx] == next_word[loc_idx]:
                continue
            if next_word[loc_idx] not in pre_m[curr_word[loc_idx]]:
                pre_m[curr_word[loc_idx]].append(next_word[loc_idx])  # p -> c
                indegree[next_word[loc_idx]] = indegree.get(next_word[loc_idx], 0) + 1
                break
    
    # q = deque()  # not meet lexicograph condition
    q = []
    heapq.heapify(q)  # due to lexigraphical order, ["ca", "cb"] -> abc not cab
    for letter in set_unique_letters:
        if letter not in indegree:
            heapq.heappush(q, letter)
    
    res = ""
    while len(q) > 0:
        curr_letter = heapq.heappop(q)
        res = res + curr_letter
        for next_letter in pre_m[curr_letter]:
            indegree[next_letter] = indegree[next_letter] - 1
            if indegree[next_letter] <= 0:
                # same letter can be de-indegree for several times, "attt"
                q.append(next_letter)
    
    return res if len(res) == num_unique_letters else ""

        
"""
##############################################################################
深度优先搜索（DFS）
##############################################################################
1) 图中（有向无向皆可）的符合某种特征（比如最长）的路径以及长度
2）排列组合
3）遍历一个图（或者树）
4）找出图或者树中符合题目要求的全部方案
"""

"""[LeetCode] Diameter of Binary Tree 二叉树的直径 !!!
Given a binary tree, you need to compute the length of the diameter of the tree. 
The diameter of a binary tree is the length of the longestpath between any two nodes 
in a tree. This path may or may not pass through the root.

Example:
Given a binary tree 

          1
         / \
        2   3
       / \     
      4   5    


Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].
根结点1的左右两个子树的深度之和呢。那么我们只要对每一个结点求出其左右子树深度之和，
这个值作为一个候选值，然后再对左右子结点分别调用求直径对递归函数，这三个值相互比较，
取最大的值更新结果res，因为直径不一定会经过根结点，所以才要对左右子结点再分别算一次。

为了减少重复计算，我们用哈希表建立每个结点和其深度之间的映射，这样某个结点的深度之前计算过了，就不用再次计算了
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def __init__(self):
        self.m = dict()
    
    def diameterOfBinaryTree(self, root) -> int:
        if root is None:
            return 0
        res = self.getHeight(root.left) + self.getHeight(root.right)
        res = max(res, self.diameterOfBinaryTree(root.left))
        res = max(res, self.diameterOfBinaryTree(root.right))
        return res
    
    def getHeight(self, root):
        if root is None:
            return 0
        if root in self.m:
            return self.m[root]
        return 1 + max(self.getHeight(root.left), self.getHeight(root.right))
    

"""[LeetCode] Invert Binary Tree 翻转二叉树

Invert a binary tree.
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

     4
   /   \
  2     7
 / \   / \
1   3 6   9
to

     4
   /   \
  7     2
 / \   / \
9   6 3   1

# can you use two approaches: recursive and non-recursive? 
"""

class Solution_DFS:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        temp_node = root.left
        root.left = self.invertTree(root.right)
        root.right = self.invertTree(temp_node)
        return root


class Solution_BFS:
    # 先把根节点排入队列中，然后从队中取出来，交换其左右节点，如果存在则分别将
    # 左右节点在排入队列中，以此类推直到队列中木有节点了停止循环，返回root即可
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        q = deque()
        q.append(root)
        while len(q) > 0:
            curr_node = q.popleft()
            # swap left and right
            temp_node = curr_node.left
            curr_node.left = curr_node.right
            curr_node.right = temp_node
            if curr_node.left:
                q.append(curr_node.left)
            if curr_node.right:
                q.append(curr_node.right)
        
        return root 

            # add left and right into queue


"""[LeetCode] Symmetric Tree 判断对称树
 

Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree is symmetric:
    1
   / \
  2   2
 / \ / \
3  4 4  3

But the following is not:

    1
   / \
  2   2
   \   \
   3    3

Note:
Bonus points if you could solve it both recursively and iteratively.
"""
class Solution_DFS:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None or (root.left is None and root.right is None):
            return True
        return self.isSymPair(root.left, root.right)
    
    def isSymPair(self, node1, node2):
        # whether node1 and node2 are sym trees
        if node1 is None and node2 is None:
            return True
        if (node1 and not node2) or (not node1 and node2):
            return False
        if node1.val != node2.val:
            return False
        return self.isSymPair(node1.left, node2.right) and self.isSymPair(node1.right, node2.left)


class Solution_BFS:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # use two queues for left and right child
        if root is None or (root.left is None and root.right is None):
            return True
        if (root.left and not root.right) or (not root.left and root.right):
            return False
        
        q1 = deque()
        q2 = deque()
        q1.append(root.left)
        q2.append(root.right)

        while len(q1) >0 and len(q2)>0:
            curr_l_node = q1.popleft()
            curr_r_node = q2.popleft()
            if not curr_l_node and not curr_r_node:
                continue
            elif curr_l_node and curr_r_node: 
                if curr_l_node.val != curr_r_node.val:
                    return False
                q1.append(curr_l_node.left)
                q2.append(curr_r_node.right)
                q1.append(curr_l_node.right)
                q2.append(curr_r_node.left)
            else:
                return False
        
        return True


"""[LeetCode] 951. Flip Equivalent Binary Trees 翻转等价二叉树

For a binary tree T, we can define a flip operation as follows: 
choose any node, and swap the left and right child subtrees.

A binary tree X is flip equivalent to a binary tree Y if and only 
if we can make X equal to Y after some number of flip operations.

Write a function that determines whether two binary trees are flip equivalent.  
The trees are given by root nodes root1 and root2.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if root1 is None and root2 is None:
            return True
        if (root1 and not root2) or (not root1 and root2):
            return False
        if root1.val != root2.val:
            return False
        opt1 = self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right)
        opt2 = self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left)
        return opt1 or opt2


"""Binary Tree Maximum Path Sum !!!
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in 
the sequence has an edge connecting them. A node can only appear in the sequence at 
most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.
Example 1:

Input: [1,2,3]

       1
      / \
     2   3

Output: 6
Example 2:

Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42


讨论：这道题有一个很好的 Follow up，就是返回这个最大路径，那么就复杂很多，因为这样递归函数就不能返回路径和了，
而是返回该路径上所有的结点组成的数组，递归的参数还要保留最大路径之和，同时还需要最大路径结点的数组，
然后对左右子节点调用递归函数后得到的是数组，要统计出数组之和，并且跟0比较，如果小于0，和清零，数组清空。
然后就是更新最大路径之和跟数组啦，还要拼出来返回值数组，代码长了很多
"""
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = float('-Inf')
        if root is None:
            return 0
        self.helper(root)
        return self.res
    
    def helper(self, node):
        # 返回值的定义是以当前结点为终点的 path 之和
        # res saves the required answer so far (based on current node)
        if node is None:
            return 0
        # always first calculate the child before node
        left_max = max(0, self.helper(node.left))
        right_max = max(0, self.helper(node.right))
        # if child is larger, it will not update res
        self.res = max(self.res, left_max + right_max + node.val)
        return node.val + max(left_max, right_max)


class Solution:
    # TLE
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        res_including_root = root.val + \
            max(self.maxPathOnOneSide(root.left), 0) + \
            max(self.maxPathOnOneSide(root.right), 0)
        max_left = self.maxPathSum(root.left) if root.left else float('-Inf')  # repeat maxPathOnOneSide, should use a dict to save
        max_right = self.maxPathSum(root.right) if root.right else float('-Inf')
        res = max(max_left, max_right)
        return max(res, res_including_root)

    def maxPathOnOneSide(self, root):
        # via root
        if root is None:
            return 0
        left_max = max(self.maxPathOnOneSide(root.left), 0)
        right_max = max(self.maxPathOnOneSide(root.right), 0)
        return root.val + max(left_max, right_max)



""" LeetCode 1644. Lowest Common Ancestor of a Binary Tree II
Similar to 236, but possible that p or q not in the binary tree. 

Given the root of a binary tree, return the lowest common ancestor (LCA) of two given nodes, p and q. 
If either node p or q does not exist in the tree, return null. All values of the nodes in the tree are unique.

According to the definition of LCA on Wikipedia: 
"The lowest common ancestor of two nodes p and q in a binary tree T is the lowest node that 
has both p and q as descendants (where we allow a node to be a descendant of itself)". 
A descendant of a node x is a node y that is on the path from node x to some leaf node.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5. A node can be a descendant of itself according to the definition of LCA.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 10
Output: null
Explanation: Node 10 does not exist in the tree, so return null.

Follow up: what if treenode has a attribute of parental node? - use a hash set to store parents of p (till root),
then from q go to root till the node is in set(). Return the node


Solution 1: ltc236 + first pass check whether both p and q are in tree
Solution 2: use (node, num of nodes) as return, only when node and num = 2 return true
https://www.bilibili.com/video/BV1sf4y1x7Kn/

"""
class Solution:
    def lowestCommonAncestor2(self, root, p, q):
        res, num = helper(root, p, q)
        if res and num == 2:
            return res
        return None

    def helper(self, root, p, q):
        if root is None:
            return (None, 0)
        left_node, left_num = self.helper(root.left)
        right_node, right_num = self.helper(root.left)
        if root == p or root == q:
            return (root, 1 + left_num + right_num)
        if left_node and right_node:
            return (root, 2)
        return (left_node, left_num) if left_node else (right_node, right_num)



"""[LeetCode] 105. Construct Binary Tree from Preorder and Inorder Traversal 
由先序和中序遍历建立二叉树
 
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7
Output: [3,9,20,null,null,15,7]


Solution: 由于先序的顺序的第一个肯定是根，所以原二叉树的根节点可以知道，题目中给了一个很关键的条件就是树中没有相同元素，
有了这个条件就可以在中序遍历中也定位出根节点的位置，并以根节点的位置将中序遍历拆分为左右两个部分，分别对其递归调用原函数
preorder:
    3,      [9,                   ]   [20,                 15,7]
l   rleft   rleft+1, rleft+i-ileft 
r                                     rleft+i-ileft+1,    pright

inorder: 
    [9,        ] 3,   [15,20,7]
l   ileft,  i-1, (i)
r                      i+1,  iright    
"""
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        return self.helper(preorder, 0, len(preorder)-1, inorder, 0, len(inorder)-1)
    
    def helper(self, preorder, pleft, pright, inorder, ileft, iright):
        # use preorder to find the root, then locate its position in inorder
        # then use the location to split both
        if ileft > iright or pleft > pright:
            # otherwise ileft may be out of index 
            return None

        curr_node_val = preorder[pleft]
        curr_node = TreeNode(val=curr_node_val)

        # locate left and right
        # pleft + i - ileft -> left
        # pleft 
        loc = 0
        for i in range(ileft, iright+1):
            if inorder[i] == curr_node_val:
                loc = i
                break
        # based on diff of i and ileft, i and iright to move p
        curr_node.left = self.helper(preorder, pleft+1, pleft+loc-ileft, inorder, ileft, loc-1)
        curr_node.right = self.helper(preorder, pleft+loc-ileft+1, pright, inorder, loc+1, iright) 
        return curr_node


"""[LeetCode] 106. Construct Binary Tree from Inorder and Postorder Traversal 由中序和后序遍历建立二叉树
Given inorder and postorder traversal of a tree, construct the binary tree.
Note:
You may assume that duplicates do not exist in the tree.

inorder = [9,3,15,20,7]
postorder = [9,15,7,20,3]

Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7
hint: the last num in postorder is root
"""
# omit 


"""[LeetCode] 1485. Clone Binary Tree With Random Pointer
A binary tree is given such that each node contains an additional random pointer which could point 
to any node in the tree or null.

Return a deep copy of the tree.

The tree is represented in the same input/output way as normal binary trees where each node is 
represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (in the input) where the random pointer points to, or null 
if it does not point to any node.
You will be given the tree in class Node and you should return the cloned tree in class NodeCopy. 
NodeCopy class is just a clone of Node class with the same attributes and constructors.

Input: root = [[1,null],null,[4,3],[7,0]]
Output: [[1,null],null,[4,3],[7,0]]
Explanation: The original binary tree is [1,null,4,7].
The random pointer of node one is null, so it is represented as [1, null].
The random pointer of node 4 is node 7, so it is represented as [4, 3] 
where 3 is the index of node 7 in the array representing the tree.
The random pointer of node 7 is node 1, so it is represented as [7, 0] 
where 0 is the index of node 1 in the array representing the tree.

Solution: 
既然是深度复制，又是树的遍历，所以比较直观的感受是用BFS或者DFS做，在遍历树的每个节点的同时，将每个节点复制，用hashmap储存。

首先是BFS。还是常规的层序遍历的思路去遍历树的每个节点，但是注意在做层序遍历的时候，
只需要将左孩子和右孩子加入queue进行下一轮遍历，不需要加入random节点。

时间O(n)
空间O(n)
"""



