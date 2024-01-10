"""
Leetcode by categories and ds
"""
# cheatsheet
# https://www.hackerearth.com/practice/notes/big-o-cheatsheet-series-data-structures-and-algorithms-with-thier-complexities-1/
# !!! means interview qn candidates

""" 额外知识点归纳
1. 如果判断一个图中是否有环？
- 无向图
求出图中所有结点的度。
将所有度 <= 1 的结点入队。（独立结点的度为 0）
当队列不空时，弹出队首元素，把与队首元素相邻节点的度减一。如果相邻节点的度变为一，则将相邻结点入队。
循环结束时判断已经访问的结点数是否等于 n。等于 n 说明全部结点都被访问过，无环；反之，则有环。
- 有向图
在判断无向图中是否存在环时，是将所有**度 <= 1** 的结点入队；
在判断有向图中是否存在环时，是将所有**入度 = 0** 的结点入队

也可以用DFS，详见https://zhuanlan.zhihu.com/p/214747022
"""

"""
############################################################################
Sort 排序类
############################################################################
OrderedDict: popitem() will pop the most recent item
-- hashmap.popitem(last=False) # O(1)
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
# solution 1: use maxheap nlogn+klogn
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

# solution 2: use quick sort , n in avg: T(n) = n + T(n/2)
# [4, 3, 2, 2, 0, -1]: 2 (idx=2) is the 3rd largest
# https://en.wikipedia.org/wiki/Quickselect
class Solution2(object):
    # kth largest = (n-k+1)th smallest
    def findKthLargest(self, nums, k):
        left = 0
        right = len(nums) - 1
        while True:
            curr = self.partition(nums, left, right)
            if curr == k - 1:
                return nums[curr]
            if curr > k - 1:
                # on the left
                right = curr - 1
            if curr < k - 1:
                left = curr + 1

    def partition(self, nums, left, right):
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


"""[LeetCode] 4. Median of Two Sorted Arrays 两个有序数组的中位数 !!!
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
        k = int((n1 + n2 + 1)/2)  # if n1+n2 is odd, then kth largest is median, else avg of k and k+1
        
        # find m1 from nums1, m2 from nums2, such that nums1[m1]=nums[m2] and m1+m2=k, (not always exists)
        # c1 = max(nums1[m1-1], nums2[m2-1]) if m1>0 and m2>0
        # if n1+n2 is odd, then c1 is median
        # else (c1 + c2) /2, where c2 is next number to c1, which is min(nums2[m2], nums1[m1]) (ignore corner case)
        l = 0; r = n1
        while l < r:
            m1 = l + (r-l)//2
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
            # nums1 all larger than nums2, m2=k
            c1 = nums2[m2 - 1]
        elif m2 <= 0:
            # nums2 all larger than nums1, m1=k
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
不过这次要记录两个指针相遇的位置，当两个指针相遇了后，让其中一个指针从链表头开始，
此时再相遇的位置就是链表中环的起始位置，为啥是这样呢，这里直接贴上热心网友「飞鸟想飞」的解释哈，
因为快指针每次走2，慢指针每次走1，快指针走的距离是慢指针的两倍。而快指针又比慢指针多走了一圈。所以 head到环的起点+环的起点到他们相遇的点的距离 与 环一圈的距离相等。现在重新开始，head 运行到环起点 和 相遇点到环起点的距离也是相等的，相当于他们同时减掉了 环的起点到他们相遇的点的距离。
because AB = m*l, then BC-AC=l-CB-AC=l-m*l  (A:head, B: fast catch slow, C: head of cycle)
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
# if no limitation on memory, then use a hash set to record 
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        hs = set()
        while head:
            if head in hs:
                return head
            hs.add(head)
            head = head.next
        
        return None

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
Stack: string processing, calculator, 
(de)Queue: streaming data
"""

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

#
# follow up
#
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
popMax() -- Retrieve the maximum element in the stack, and remove it. If you find more than one maximum elements, only remove the top-most one.

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
            temp.append(self.stack1.pop())
        
        x = self.stack1.pop()
        self.stack2.pop()

        while len(temp)>0:
            # self.push(temp.pop())
            self.stack1.append(temp.pop())
        
        return x


"""[LeetCode] 150 Evaluate Reverse Polish Notation 计算逆波兰表达式
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
The expression string contains only non-negative integers, +, -, *, / operators , open ( and closing parentheses ) and empty spaces . The integer division should truncate toward zero.

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
s4 = "2-(4-5)"

calculator3(s4)
# Solution 1: based on calculator 2, get the pair of parenthesis and recursively call it
class Solution:
    def calculator3(self, s):
        left2right = dict()  # map all ( and ) pair location
        stack=[]
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            if s[i] == ')':
                left2right[stack.pop()] = i
        return self.helper(s, 0, len(s), left2right)

    def helper(self, s, start_idx, end_idx, m):
        # get results for s[start_idx:end_ix]
        # problemtic because cannot handle 2*(2-4)
        clean_str = ""
        i = start_idx
        while i < end_idx:
            if s[i] not in ['(', ')']:
                clean_str += s[i]
                i += 1
            else:
                if s[i] == '(':
                    # find the corresponding pair
                    j = m[i]  # ) location
                    curr_res = self.helper(s, i+1, j, m)
                    clean_str += str(curr_res)
                    i = j + 1
        return self.calculator2(clean_str)

    def calculator2(self, s):
        # if consider negative number, then first remove spaces, then consider previous ops
        # https://www.youtube.com/watch?v=Tm_hHBhQgII&t=682
        Stack = []
        n = len(s)
        curr_num_str = ''
        sign = 1
        op = '+' # previous operator
        for i in range(n):
            if s[i].isdigit():
                curr_num_str = curr_num_str + s[i]
            if s[i] in ['+', '-', '*', '/'] or i == n-1:
                if i==0 or (i < n-1 and not s[i-1].isdigit()):
                    # not an operator but a sign, can only be + or -
                    # *-4, --4, +-4
                    if s[i] == '-':
                        sign = sign * -1
                        # curr_num_str = '-' + curr_num_str  # if ---4 then fails
                else:
                    # if operator, need previous operator
                    if op == '-':
                        Stack.append(int(curr_num_str) * sign * -1)
                    elif op == '+':
                        Stack.append(int(curr_num_str) * sign)
                    elif op == '*':
                        top_num = Stack.pop()
                        Stack.append(top_num*int(curr_num_str) * sign)
                    elif op == '/':
                        top_num = Stack.pop()
                        Stack.append(int(top_num  * sign/ int(curr_num_str)))
                    sign = 1
                    op = s[i]
                    curr_num_str = ''
        res = 0
        while len(Stack) > 0:
            res  = res + Stack.pop()
        return res

# calculator 2: solution 2 O(1)
def calculator2(s):
    # if consider negative number, then first remove spaces, then consider previous ops
    # https://www.youtube.com/watch?v=Tm_hHBhQgII&t=682
    s = str.replace(s, " ", "") 
    # Stack = []
    prev_num, curr_num = 0, 0  # 2-size stack: [prev_num, curr_num]
    s = '+' + s
    i = 0
    while i < len(s):
        if s[i] in ['+', '-']:
            prev_num += curr_num
            curr_num = 0
            j = i + 1  # + -456 -> + int(-456)
            if s[j] in ['+', '-']:
                j += 1  # sign instead of op
            while j < len(s) and s[j].isdigit():
                j += 1
            if s[i] == '+':
                # Stack.append(int(s[i+1:j]))
                curr_num = int(s[i+1:j])
            else:
                # Stack.append(-1 * int(s[i+1:j]))
                curr_num = -1 * int(s[i+1:j])
            i = j
        elif s[i] in ['*', '/']:
            j = i + 1  # + -456 -> + int(-456)
            if s[j] in ['+', '-']:
                j += 1  # sign instead of op
            while j < len(s) and s[j].isdigit():
                j += 1
            if s[i] == '*':
                # Stack[-1] = Stack[-1]*int(s[i+1:j])
                curr_num *= int(s[i+1:j])
            else:
                # Stack[-1] = int(Stack[-1] / int(s[i+1:j]))
                curr_num = int(curr_num / int(s[i+1:j]))
            i = j  
        else:
            i += 1            

    return curr_num + prev_num

calculator2("2 * -4")
calculator2("2 * -4 --5")

# Solution 2 for calculator 3: combine together, can handle 2*(2-4)
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
# Another design is to use linkedlist, the visit takes O(1) but back/forward takes O(m)

# for example 
# suppose [1,2,(3),4] # () means where curr page is
# if back(1) -> [1,(2),3,4]  # idx=1
# if forward(1) -> [1,2,(3),4]  # idx=2
# if visit(5) -> [1,2,3,(5)]
class BrowserHistory:
    def __init__(self, homepage: str):
        self.stack = [homepage]
        self.size = 1
        self.curr = 0  # curr location
        
    def visit(self, url: str) -> None:
        num_pops = len(self.stack) - self.curr - 1
        for _ in range(num_pops):
            self.stack.pop()
        # while len(self.stack) > self.curr + 1:
        #     self.stack.pop()
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

Follow-up: can you do it at O(n)
"""
# note that recursively call the function will raise time limit error
# so can only run through the string once
s = 'abcd'
k=2
# duplicate is defined as exact k rep
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        n = len(s)
        letter_freq_stack = []
        for i in range(n):
            if len(letter_freq_stack) > 0:
                if s[i] == letter_freq_stack[-1][0]:
                    letter, val = letter_freq_stack.pop()
                    if val + 1 < k:
                        # if =k, pop
                        letter_freq_stack.append((s[i], val + 1))
                else:
                    # if diff from previous
                    letter_freq_stack.append((s[i], 1))
            else:
                letter_freq_stack.append((s[i], 1))
        res = ''
        while len(letter_freq_stack) > 0:
            letter, val = letter_freq_stack.pop()
            res = letter * val + res
        
        return res

# if duplicate is defined as equal or larger than k instead
def removeDuplicates2(s: str, k: int) -> str:
    if not s:
        return s
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
                    continue # no i+=1!
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


"""[LeetCode] 146. LRU Cache 最近最少使用页面置换缓存器 !!! (linked list solution)
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

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
# in python, one can use OrderedDict (a dict that has popitem() to remove the last or first key)
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
            value = self.hashmap.pop(key) # O(1)
            self.hashmap[key] = value  # reset time order of this key
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
                # https://stackoverflow.com/questions/51800639/complexity-of-deleting-a-key-from-python-ordered-dict
                self.hashmap.popitem(last=False) # O(1)
                self.hashmap[key] = value


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# if not using OrderedDict, need to use linked list to achieve O(1) runtime
# need to define linked list, and always append the latest node from left
# use O(1) to remove any node (def remove(self, node):)
# https://leetcode.com/problems/lru-cache/discuss/1655397/Python3-Dictionary-%2B-LinkedList-or-Clean-%2B-Detailed-Explanation-or-O(1)-Time-For-All-Operations
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
        # remove the right most node
        if self.tail.prev != self.head:
            node = self.tail.prev
            self.remove(node)
            return node
    
    def popleft(self):
        if self.head.next != self.tail:
            node = self.head.next
            self.remove(node)
            return node


class LRUCache:
    # 另一种思路是用node2val作为dict，node里面存key
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


"""[LeetCode] Set Matrix Zeroes 矩阵赋零
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
Follow up: Did you use extra space?

A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
"""
# idea: to avoid using extra space, we use first row and col of matrix to record zeros
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m = len(matrix)
        n = len(matrix[0])
        first_row_has_zero = False
        first_col_has_zero = False
        # iterate through matrix to mark the zero row and cols
        for row in range(m):
            for col in range(n):
                if matrix[row][col] == 0:
                    if row == 0:
                        first_row_has_zero = True
                    if col == 0:
                        first_col_has_zero = True
                    matrix[row][0] = matrix[0][col] = 0
        # iterate through matrix to update the cell to be zero if it's in a zero row or col
        for row in range(1, m):
            for col in range(1, n):
                matrix[row][col] = 0 if matrix[0][col] == 0 or matrix[row][0] == 0 else matrix[row][col]
        # update the first row and col if they're zero
        if first_row_has_zero:
            for col in range(n):
                matrix[0][col] = 0
        if first_col_has_zero:
            for row in range(m):
                matrix[row][0] = 0
        return None


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
# 删除操作是比较 tricky 的，还是要先判断其是否在 HashMap 里，如果没有，直接返回 false。由于 HashMap 的
# 删除是常数时间的，而数组并不是，为了使数组删除也能常数级，实际上将要删除的数字和数组的最后一个数字
# 调换个位置，然后修改对应的 HashMap 中的值，这样只需要删除数组的最后一个元素即可，保证了常数时间
# 内的删除。而返回随机数对于数组来说就很简单了，只要随机生成一个位置，返回该位置上的数字即可
import random
class RandomizedSet:
    def __init__(self):
        self.val2index = dict() # val -> location of ele in vector
        self.vector = []
        self.size = 0

    def insert(self, val: int) -> bool:
        if val in self.val2index:
            return False
        else:
            self.vector.append(val)
            self.val2index[val] = self.size
            self.size = self.size + 1
            return True

    def remove(self, val: int) -> bool:
        # swap val location with the last one, then pop from vector
        if val in self.val2index:
            location = self.val2index[val]
            self.val2index[self.vector[-1]] = location
            self.vector[location], self.vector[-1] = self.vector[-1], self.vector[location]
            del self.val2index[val]
            self.vector.pop()
            self.size = self.size - 1
            return True
        else:
            return False

    def getRandom(self) -> int:
        rand_index = random.randint(0, self.size-1)
        return self.vector[rand_index]


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
["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]]
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
def getHint(secret, guess):
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
class TicTacToe:
    def __init__(self, n):
        self.curr_num = 0
        row_sums = [0] * n
        col_sums = [0] * n
        diag = 0
        rev_diag = 0
        board = dict()
    
    def move(self, row, col, player):
        # if (row, col) in board:
        #     return False
        direction = 1 if player == 1 else -1
        row_sums[row] += direction
        if row_sums[row] == n*direction:
            return player
        col_sums[col] += direction
        if col_sums[row] == n*direction:
            return player
        if row == col:
            diag += direction
            rev_diag += direction
        if diag == n*direction or rev_diag == n*direction:
            return player



"""
############################################################################
Heap／Priority Queue题目
# time complexity of building a heap is O(n)
# https://www.geeksforgeeks.org/time-complexity-of-building-a-heap/
# insert/remove max from max heap is O(logn)
# find max is O(1)
# use case: k largest/smallest, k most frequent
############################################################################
import heapq # python only support min heap
heapq.heappush(q, x); x=heapq.heappop(q) # min
heapq.nsmallest(k, q, key) # return the k smallest, klogn, [note: sum(logn) ~ nlogn]
heapq.nlargest(k, q, key=None)  # return k largest , klogn

# example: 
tags = [ ("python", 30), ("ruby", 25), ("c++", 50), ("lisp", 20) ]
heapq.nlargest(2, tags, key=lambda e:e[1]) # Gives [ ("c++", 50), ("python", 30) ]
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

# solution2 : use heap, O(nlogk)
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        q = []
        for (x, y) in points:
            dist = -(x*x + y*y)
            if len(heap) == K:
                heapq.heappushpop(q, (dist, x, y))
            else:
                heapq.heappush(q, (dist, x, y))
        return [(x,y) for (dist,x, y) in q]


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
            heapq.heappush(q, (val, key))
            if len(q) > k:
                heapq.heappop(q)

        return [key for val, key in q]


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
以此类推，直到堆中没有元素了，此时k个链表也合并为了一个链表，返回首节点即可, logk * n
Anther idea is merge sort based on merging two sorted linked list, n*k
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
            _, min_idx = heapq.heappop(q) # from which linked list
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

# use min heap
class Solution(object):
    # The idea is to keep a heap that stores a tuple (val, fact).
    # The fact value stores the last factor the number was multiplied to.
    # So every time I extract the minimum and look at the queue, I know
    # it must be multiplied by queue and next factor, otherwise I will have a
    # duplicate. 
    # 2(*2,3,5): 2, 4, 6 ...; 3(*3,5): 3, 9, 15, ...; 5(*5): 5, 25, ..
    # For example, we have 2 and 5 with fact 2 and fact 5. When I extract
    # 2 and fact 2, I will insert 2 * 5 = 10 with fact 5. Then, when I extract 5 from queue
    # 5, if I do 5 * 2 (fact 2) = 10,
    # I will have a duplicate. So every number coming from
    # the k-th fact must be multiplied only by the factor fact and higher.
    def nthUglyNumber(self, n):
        if n is 1:
            return 1
        h, val = [(2, 2), (3, 3), (5, 5)], 1
        for i in range(1, n):
            val, fact = heapq.heappop(h)
            heapq.heappush(h, (val * 5, 5))
            if fact <= 3:
                heapq.heappush(h, (val * 3, 3))
            if fact is 2:
                heapq.heappush(h, (val * 2, 2))
        return val


""" [LeetCode] 1086. High Five
Each student has two attributes ID and scores. Find the average of the top five scores for each student.
Example 1:
Input: 
[[1,91],[1,92],[2,93],[2,99],[2,98],[2,97],[1,60],[1,58],[2,100],[1,61]]
Output:
1: 72.40
2: 97.40
时间O(nlogn)
空间O(n)
"""
# use a max heap to store all scores of a student
results = [[1,91],[1,92],[2,93],[2,99],[2,98],[2,97],[1,60],[1,58],[2,100],[1,61]]
import heapq
from collections import defaultdict
class Solution:
    # @param {Record[]} results a list of <student_id, score>
    # @return {dict(id, average)} find the average of 5 highest scores for each person
    # <key, value> (student_id, average_score)
    def highFive(self, results):
        # Write your code here
        id2scores = defaultdict(list)
        top5_res = dict()
        for result in results:
            sid, score = result
            heapq.heappush(id2scores[sid], score)

        for k,v in id2scores.items():
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


"""[LeetCode] 378. Kth Smallest Element in a Sorted Matrix 有序矩阵中第K小的元素 !!!
Given an n x n matrix where each of the rows and columns is sorted in ascending order, 
return the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

You must find a solution with a memory complexity better than O(n^2).

Example 1:
Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13

Follow up:
Could you solve the problem with a constant memory (i.e., O(1) memory complexity)?
Could you solve the problem in O(n) time complexity? 
The solution may be too advanced for an interview but you may find reading this paper fun.
"""
# sol1: memory O(k): use a max heap of size k, and pop when size>k, top will be kth smallest, skip
# sol2: memory O(n), runtime O(k*logn), similar to ugly number, use a min heap (size n) to save 
# first column, then deque/pop, and enque matrix[row_idx][col_idx+1], repeat for k-1 times
import heapq
class Solution:
    def kthSmallest(self, matrix, k: int) -> int:
        q = []  # min heap
        for i in range(min(k, len(matrix))):
            heapq.heappush(q, (matrix[i][0], i, 0))
        
        for _ in range(k-1):
            curr_val, curr_row_idx, curr_col_idx = heapq.heappop(q)
            if curr_col_idx < len(matrix[0]) - 1:
                heapq.heappush(q, (matrix[curr_row_idx][curr_col_idx+1], curr_row_idx, curr_col_idx+1))
        
        return heapq.heappop(q)[0]

# sol3: memory O(1): !!!
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
                # note that mid may not be in matrix
                low = mid + 1
            else:
                # if =k, then should be higher bound
                high = mid
        
        return low
    
    def findRank(self, matrix, val):
        # find how many ele smaller or equal than val
        res = 0
        for row in matrix:
            low_idx, high_idx = 0, len(row)
            while low_idx < high_idx:
                mid_idx = low_idx + (high_idx - low_idx)//2
                if row[mid_idx] <= val:
                    low_idx = mid_idx + 1
                else:
                    high_idx = mid_idx
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


"""295. Find Median from Data Stream !!!
Share The median is the middle value in an ordered integer list. If the size of the list is even, 
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
# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
# T(n) = logn + T(n-1) = logn! = nlogn
import heapq
class MedianFinder:
    def __init__(self):
        self.size = 0
        self.q_low_maxheap = []  # top is max, keep low set size in [high size, high size + 1]
        self.q_high_minheap = [] # top is min
        
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


"""[LeetCode] 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit !!!
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
# Solution 1: binary insert and remove, O(n^2), sliding window left-right
# Keep an increasing list L. Binary insert the current element nums[right].
# If the L[L.size() - 1] - L[0] > limit, binary search the position of nums[left] and remove it from the list.
import bisect
class Solution:
    def longestSubarray(self, nums, limit):
        l, L, n = 0, [], len(nums)
        res = 0
        for r in range(n):
            bisect.insort(L, nums[r])
            while L[-1] - L[0] > limit:
                L.pop(bisect.bisect(L, nums[l]) - 1) # can only pop the most left
                l += 1
            res = max(res, r-l+1)
        return res

# Solution 2: two deques O(n), to track max and min of sliding window
# while diff>limit, move left till diff<=limit, pop from queue if q[0]==nums[left]
class Solution:
    def longestSubarray(self, nums, limit):
        max_q, min_q = deque(), deque()  # mono q
        # max_q mono decrease and min_q mono increase
        # so that max_q[0] is max and min_q[0] is min in the window
        l = 0 # left side of window
        res = 0
        for r in range(len(nums)):
            while len(max_q) > 0 and max_q[-1] < nums[r]:
                max_q.pop()  # popright
            while len(min_q) > 0 and min_q[-1] > nums[r]:
                min_q.pop()
            max_q.append(nums[r])
            min_q.append(nums[r])
            while max_q[0] - min_q[0] > limit:
                # remove very left elem
                if max_q[0] == nums[l]:
                    max_q.popleft()
                if min_q[0] == nums[l]:
                    min_q.popleft()
                l += 1
            res = max(res, r-l+1)
        return res

# Solution 3: use heap to replace deque, each operation takes logn, O(nlogn)
class Solution:
    def longestSubarray(self, nums, limit):
        maxq, minq = [], []
        res = left = 0
        for right, num in enumerate(nums):
            heapq.heappush(maxq, [-num, right])  # val, idx
            heapq.heappush(minq, [num, right])
            while -maxq[0][0] - minq[0][0] > limit:
                # as long as diff > limit, remove the prev elems
                left = min(maxq[0][1], minq[0][1]) + 1 # min idx before which needs to remove
                while maxq[0][1] < left: 
                    heapq.heappop(maxq)
                while minq[0][1] < left: 
                    heapq.heappop(minq)
            res = max(res, right - left + 1)
        return res


"""[LeetCode] 895. Maximum Frequency Stack 最大频率栈 !!!
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
from collections import defaultdict
class FreqStack:
    def __init__(self):
        self.max_Freq = 0
        self.m_f2n = defaultdict(list)  # freq -> [numbers]
        self.m_n2f = defaultdict(lambda: 0)  # number -> freq
        
    def push(self, val: int) -> None:
        # update n2f
        self.m_n2f[val] += 1
        # update f2n
        self.m_f2n[self.m_n2f[val]].append(val)  # later to right
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

Find the mtn peak is an exceptional. Use left, right = 0, len-1 and while left < right. 
Because condition is nums[mid] < nums[mid+1] and mid+1 can out of range if right=len

还有一类是排序区间的插入，删除，和合并。一般思路是对于待处理的区间，查找第一个和最后一个与它overlap的区间(可以左闭右开)，然后用for来处理中间的，
一个例子是715. Range Module.
intervals=[[1,2], [3,4], [5,6]], interval=[2.5,4.5] -> first=0 and last=2 (左闭右开，i.e. intervals[2]是第一个不overlap的)
一般可以考虑
1) 新建一个list存放,O(n) -- 添加区间的一个常用技巧是bool inserted=False，一般添加完毕后设成True然后后面所有区间直接加进来
2) treemap(SortedDict) 的数据结构，用bisect_left和bisect_right来查找头尾(logn) -- 见715. Range Module
"""
# note: if ascending
# if finish: l = r + 1
# (if do nothing at meet condition): l (or r+1) returns the first element that is greater or equal than target 
# if search for the first number greater than target: do nothing and change line 1982 to nums[mid] <= target
# alternative
def bin_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r-l)//2
        if nums[mid] == target:
            return mid # or do nothing
        elif nums[mid] < target:
            l = mid + 1
        else: # nums[mid] >= target
            r = mid - 1
    return -1

# note: if ascending
# if finish, l = r
# (if do nothing at meet condition): l (or r) returns first element that is greater or equal than target 
def bin_search(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r-l)//2
        # if nums[mid] == target:
        #     return mid # or do nothing
        if nums[mid] < target:
            l = mid + 1 # not soln except for last
        else: # (nums[mid] >= target)
            r = mid # r is soln candidate
    return l # (or r)

# note: if ascending
# if finish, l = r
# (if do nothing at meet condition): l-1 (or r-1) returns first element that is smaller or equal than target 
def bin_search(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r-l)//2
        if nums[mid] <= target:
            l = mid + 1 # l-1 is soln candidate
        else: # (nums[mid] >= target)
            r = mid # not soln, upper bound
    return l-1


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
        l, r = 0, len(nums)
        while l < r:
            mid = l + (r-l)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid
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
        low, high = 0, len(nums)
        while low < high: # [, )
            mid = low + (high - low) // 2
            if nums[mid] == target or nums[high-1] == target or nums[low] == target:
                return True
            if nums[mid] < nums[high-1]:  # nums[high] if [, ]
                # then right half is sorted
                if target > nums[mid] and target <= nums[high-1]:
                    low = mid + 1
                else:
                    high = mid
            elif nums[mid] > nums[high-1]:
                # left half is sorted
                if target >= nums[low] and target < nums[mid]:
                    high = mid
                else:
                    low = mid + 1
            else:
                high = high - 1
        return False


""" 1095. Find in Mountain Array
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
        # not mountain_arr.length() o.w. mid+1 is out of range
        while low < high:
            mid = low + (high - low) // 2
            if mountain_arr.get(mid) < mountain_arr.get(mid+1):
                low = mid + 1
            else:
                high = mid
        return low

    def find_target(self, mountain_arr, low, high, target, asc):
        # include high as a candidate res
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

mountain_arr = [0,1,2,4,2,1]; target = 3
mountain_arr = [1,2,3,5,3]; target = 0


"""[LeetCode] Find Peak Element 求数组的局部峰值
A peak element is an element that is greater than its neighbors.
Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.
The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
You may imagine that nums[-1] = nums[n] = -∞.

Example 1:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Example 2:
Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.
Note:
Your solution should be in logarithmic complexity!

由于题目中提示了要用对数级的时间复杂度，那么我们就要考虑使用类似于二分查找法来缩短时间，
由于只是需要找到任意一个峰值，那么我们在确定二分查找折半后中间那个元素后，和紧跟的那个元素比较下大小，
如果大于，则说明峰值在前面，如果小于则在后面。这样就可以找到一个峰值了
"""
class Solution(object):
    def findPeakElement(self, nums):
        if nums[-1] > nums[-2]:
            return len(nums)-1
        l, r = 0, len(nums)-1
        while l < r:
            mid = l + (r-l)//2
            if (mid-1<0 or nums[mid]>nums[mid-1]) and (nums[mid]>nums[mid+1]):
                return mid
            if nums[mid] < nums[mid+1]:
                l = mid + 1
            else:
                r = mid 
        return l


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

Solution O(m+n)
如果我们观察题目中给的那个例子，可以发现有两个位置的数字很有特点，左下角和右上角的数。左下角的 18，
往上所有的数变小，往右所有数增加，那么就可以和目标数相比较，如果目标数大，就往右搜，如果目标数小，
就往上搜。这样就可以判断目标数是否存在。当然也可以把起始数放在右上角，往左和下搜，停止条件设置正确就行。

Solution 2: O(mlogn)
For each row, if row[0]<target<row[-1], then use binary search
"""
# skip



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
        l, r = 0, x+1
        while l < r:
            mid = l + (r-l)//2
            if mid * mid <= x:
                l = mid + 1
            else:
                r = mid
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
        low, high = 0, len(nums)
        while low < high:
            mid = low + (high - low) // 2
            if mid == len(nums)-1 or mid == 0:
                return nums[mid]
            if nums[mid] == nums[mid + 1]:
                if mid % 2 == 0:
                    # single on right
                    low = mid + 1 # + 2 also works
                else:
                    high = mid
            elif nums[mid] == nums[mid - 1]:
                if mid % 2 == 0:
                    high = mid # - 1 also works 
                else:
                    low = mid + 1
            else:
                return nums[mid]
        
        return nums[low-1]


""" 617 · Maximum Average Subarray II ???
Given an array with positive and negative numbers, find the maximum average subarray 
which length should be greater or equal to given length k.
Example 1:
Input: nums=[1,12,-5,-6,50,3]; k=3
Output: 15.667
Explanation: (-6 + 50 + 3) / 3 = 15.667

Solution 1: 双指针遍历 用hashmap存cumsum方便计算 O(n^2)
Solution 2: O(nlog (max - min))
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
            is_exist = self.check(nums, k, mid)  # whether exists a subarray (>=k) with avg>=mid
            if is_exist:
                low = mid
            else:
                high = mid
        
        return low

    def check(self, nums, k, target):
        # whether exists a subarray (>=k) with avg>=target
        # index starts with k
        # 那么在nums[0:index]中，长度>=k的子数组，区间和最大为pre[index - 1] - min{pre[0 : index - k]}
        cumsums = [0] * (len(nums) + 1)
        for i in range(1, len(nums)+1):
            cumsums[i] = cumsums[i - 1] + nums[i-1] - target
        # find a subarray with len >=k and avg > 0
        minSum = 0
        for i in range(k, len(nums)+1):
            minSum = min(minSum, cumsums[i-k])
            if cumsums[i] > minSum:
                return True
        
        return False


sol=Solution()
sol.maxAverage(nums, k)

# follow up: Maximum Sum Subarray with length less or equal to k
class maxSumSubarray:
    def maxSumSubarray(self, arr, k):
        minCumsumq = deque()  # mono increasing
        cumsums = self.cumsum(arr)
        res = float('-Inf')
        for r in range(len(cumsums)):
            # update mono q
            while len(minCumsumq) > 0 and cumsums[r] <= cumsums[minCumsumq[-1]]:
                minCumsumq.pop()
            minCumsumq.append(r)
            if r - minCumsumq[0] > k:
                minCumsumq.popleft()
            res = max(res, cumsums[r] - cumsums[minCumsumq[0]])
        return res
    
    def cumsum(self, arr):
        cumsumArr = [0]
        for num in arr:
            cumsumArr.append(cumsumArr[-1] + num)  # O(1) cost
        return cumsumArr

sol = maxSumSubarray()
sol.maxSumSubarray( [2, -1, 2, -2, 1, -1], 3)


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
        # 1, 3, -> 0.25, 1
        for i in range(1, self.size):
            self.cumsums[i] = self.cumsums[i - 1] + w[i]/total_wts
        
    def pickIndex(self) -> int:
        rand = random.uniform(0, 1)
        low, high = 0, self.size
        # find the smallest index that larger than rand
        while low < high:
            mid = low + (high - low) // 2
            if self.cumsums[mid] > rand:
                high = mid
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
思路是二分法。首先遍历input数组，得到数组所有元素的和sum以及数组中最大的数字max。需要寻找的这个res一定介于target//len(arr) - max之间。
为什么呢？因为被修改的数字一定是需要小于value的，如果这个value大于数组中的最大元素，意味着没有任何一个数字被修改过，
所以value大于max是不成立的。所以在target//len(arr) - max之间做二分搜索，并且每找到一个mid，就算一次sum和，
二分法逼近最接近target的sum之后，找到对应的mid即可。NlogN

# suspicious O(N) solution
# https://leetcode.com/problems/sum-of-mutated-array-closest-to-target/discuss/1390461/Python-average-O(N)-solution
"""
class Solution:
    def findBestValue(self, arr, target: int) -> int:
        # Binary search to find the largest v (as the final lo) 
        # such that the sum is smaller than or equal to target
        lo, hi = target//len(arr), max(arr)+1  # or, lo = 0
        while lo < hi:
            mid = lo + (hi-lo)//2
            curr_value = get_sum(arr, mid)
            if curr_value == target:
                return mid
            if curr_value < target:
                lo = mid + 1
            else:
                hi = mid
        # Check if (lo + 1) results in a smaller absolute difference
        # Note: get_sum(lo) <= target
        #       get_sum(lo + 1) > target if lo < max(arr)
        if target - get_sum(arr, lo-1) <= abs(get_sum(arr, lo) - target):
            return lo - 1
        return lo
    
def get_sum(arr, threshold):
    return sum([threshold if a > threshold else a for a in arr])


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
        low, high = 0, n
        # find the first idx s.t. num_miss on left >= k
        # or the last idx s.t. num_miss on left < k (commented return line)
        while low < high:
            mid = low + (high - low) // 2
            num_miss_mid = self.get_num_miss(nums, mid)
            if num_miss_mid >= k:
                high = mid
            else:
                low = mid + 1
        # low: smallest idx that has miss>=k
        # find the missing idx: delta is self.get_num_miss(nums, low) - k
        # even no delta, still move left by 1
        # e.g., nums = [4,7,9,10]; K = 1
        # return nums[low]-1 - (self.get_num_miss(nums, low) - k)
        return nums[low-1] + k - self.get_num_miss(nums, low-1)
    
    def get_num_miss(self, nums, idx):
        # number of missing on left of nums[idx]
        return nums[idx] - nums[0] + 1 - (idx + 1)


sol = Solution()
sol.missingElement(nums, K)
nums = [4,7,9,10,11,12,13,14]; K = 3  # 8


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
    low, high = 1, max(arr) + 1 # size
    # find size <= size(k), i.e. smallest size making >=k ribbons
    while low < high:
        mid = low + (high - low) // 2  # candidate size
        num_ribbon = get_num_ribbons(arr, mid)
        if num_ribbon >= k:
            # bigger size
            low = mid + 1
        else:
            # need smaller size
            high = mid

    return low-1

def get_num_ribbons(arr, size):
    res = 0
    for i in arr:
        res = res + i//size
    return res

greatestLength(arr, k)


"""715. Range Module
A Range Module is a module that tracks ranges of numbers. Design a data structure to track the ranges represented as 
half-open intervals and query about them.
A half-open interval [left, right) denotes all the real numbers x where left <= x < right.

Implement: 
void addRange(int left, int right) Adds the half-open interval [left, right)
boolean queryRange(int left, int right) Returns true if every real number in the interval [left, right) is currently being tracked
void removeRange(int left, int right) Stops tracking every real number currently being tracked in the half-open interval [left, right)

Input
["RangeModule", "addRange", "removeRange", "queryRange", "queryRange", "queryRange"]
[[], [10, 20], [14, 16], [10, 14], [13, 15], [16, 17]]
Output
[null, null, null, true, false, true]

Explanation
RangeModule rangeModule = new RangeModule();
rangeModule.addRange(10, 20);
rangeModule.removeRange(14, 16);
rangeModule.queryRange(10, 14); // return True,(Every number in [10, 14) is being tracked)
rangeModule.queryRange(13, 15); // return False,(Numbers like 14, 14.03, 14.17 in [13, 15) are not being tracked)
rangeModule.queryRange(16, 17); // return True, (The number 16 in [16, 17) is still being tracked, despite the remove operation)
"""
# http://zxi.mytechroad.com/blog/data-structure/leetcode-715-range-module/
from sortedcontainers import SortedDict
class RangeModule:
    def __init__(self):
        self.start2end = SortedDict()

    def addRange(self, left: int, right: int):
        # find the first and last interval overlaps with [left, right]
        # [[1,3], [4,6], [7,8]], [3,4] -> first=0, last=2 although [)
        first, last = self.find_first_and_last(left, right)
        if first < last:
            # merge from first to last, pop out all the intervals
            sorted_start_time = self.start2end.keys()
            for key in sorted_start_time[first:last]:
                curr_start = key
                left = min(left, curr_start)
                right = max(right, self.start2end[curr_start])
                del self.start2end[key]
        self.start2end[left] = right
        return 

    def queryRange(self, left: int, right: int) -> bool:
        # return whether overlap
        first, last = self.find_first_and_last(left, right)
        if first < last:
            start, end = self.start2end.peekitem(first) # only need to check left overlap interval
            return start <= left and right <= end
        return False

    def removeRange(self, left: int, right: int) -> None:
        first, last = self.find_first_and_last(left, right)
        if first < last:
            sorted_start_time = self.start2end.keys()
            # delete all overlap intervals
            # [[1,3], [4,6]], [3,5] 
            left_start = min(sorted_start_time[first], left)  # keep left_start -> left
            right_end = max(self.start2end[sorted_start_time[last-1]], right) # keep right -> right_end
            for key in sorted_start_time[first:last]:
                del self.start2end[key]
            if left_start < left:
                self.start2end[left_start] = left
            if right_end > right:
                self.start2end[right] = right_end
        return 
        
    def find_first_and_last(self, left, right):
        # find the first and last interval overlaps with [left, right]
        start_idx = self.start2end.bisect_left(left) # [[1,2], [3,4]], [5,6] -> return 2, [3.5,5] -> return 2 (needs adjust)
        end_idx = self.start2end.bisect_right(right)  # end_idx -> first interval with start > right
        if start_idx != 0:
            # if prev interval overlap with [left, right], then start_idx -= 1
            if self.start2end[self.start2end.keys()[start_idx - 1]] >= left: # start_idx : first interval with end >= left
                # no move when self.start2end[start_idx] == left, e.g., [[2,3], [4,6]], [6,8], then start_idx=1
                start_idx -= 1
        return start_idx, end_idx


#  [[1,3], [4,6], [7,8]]
sol=RangeModule()
sol.addRange(1, 4)
sol.addRange(5, 6)
sol.start2end
sol.addRange(7,8)
sol.addRange(6.5, 6.6)
sol.addRange(5,8)

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
Input: s = "aaa"
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
class Solution:
    def countSubstrings(self, s: str) -> int:
        L, res = len(s), 0
        for i in range(L):
            a,b = i,i
            while a >= 0 and b < L and s[a] == s[b]: 
                res += 1
                a -= 1; b += 1
            a, b = i, i+1
            while a >= 0 and b < L and s[a] == s[b]: 
                res += 1
                a -= 1; b += 1
        return res


"""[LeetCode] 15. 3Sum 三数之和  !!!
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
Find all unique triplets in the array which gives the sum of zero.

Note:
Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
The solution set must not contain duplicate triplets.

For example, given array S = {-1 0 1 2 -1 -4}, a solution set is:
(-1, 0, 1)
(-1, -1, 2)
"""
class Solution:
    def threeSum(self, nums):
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
                    if l == i + 1 or nums[l] != nums[l-1]:
                        res.append(tuple([nums[i], nums[l], nums[r]]))
                    l += 1
                    r -= 1              
        return  list(res)


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

You are given a helper function bool knows(a, b) which tells you whether A knows B. 
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
"""
class Solution:
    def maxArea(self, height):
        i, j = 0, len(height) - 1  # l, r
        res = 0
        while i < j:
            res = max(res, (j - i) * min(height[i], height[j]))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return res

"""[LeetCode] 283 Move Zeroes 移动零
Given an array nums, write a function to move all 0's to the end of it while maintaining 
the relative order of the non-zero elements.

For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

Note:
You must do this in-place without making a copy of the array.
Minimize the total number of operations.

Hint: 
A two-pointer approach could be helpful here. The idea would be to have one pointer for 
iterating the array and another pointer that just works on the non-zero elements of the array.
"""
class Solution:
    def moveZeroes(self, nums) -> None:
        slow = 0  # zero idx, always <= fast
        for fast in range(len(nums)):
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
            # wait while we find a non-zero element to
            # swap with you
            if nums[slow] != 0: # stay at zero, "while" rather than "if" is safer?
                slow += 1
        return None

class Solution:
    def moveZeroes(self, nums):
        zero = 0  #  min(first zero, curr index)
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero] = nums[zero], nums[i]
                zero += 1
        return None

nums = [0, 1, 0, 3, 12]
nums = [0]
sol = Solution()
sol.moveZeroes(nums)

# class Solution:
#     def moveZeroes(self, nums):
#         i = j = 0 # j-> non zero, i -> zero
#         while j < len(nums) and i < len(nums):
#             if nums[j] != 0 and nums[i] == 0 and j > i:
#                 nums[i], nums[j] = nums[j], nums[i]
#                 i += 1
#                 j += 1
#             if j < len(nums) and (nums[j] == 0 or j < i):
#                 j += 1
#             if i < len(nums) and nums[i] != 0:
#                 i += 1
#         return None


"""[LeetCode] 395. Longest Substring with At Least K Repeating Characters ??? (two pointer)
Find the length of the longest substring T of a given string (consists of lowercase letters only) such that 
every character in T appears no less than k times.

Example 1:
Input: s = "aaabb", k = 3
Output: 3
The longest substring is "aaa", as 'a' is repeated 3 times.

Example 2:
Input: s = "ababbc", k = 2
Output: 5
Explanation: The longest substring is "ababb", as 'a' is repeated 2 times and 'b' is repeated 3 times.

Two pointer: O(n), solution3
https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/discuss/1655754/Python-2-Pointer-solution
"""
from collections import Counter
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        if len(s) == 0 or len(s) < k:
            return 0
        if k <= 1:
            return len(s)
            
        for char, freq in Counter(s).items():
            if freq < k:
                return max(self.longestSubstring(w,k) for w in s.split(char))
            
        return len(s)

class Solution2: # ~nlogn
    def longestSubstring_substring(self, s, k, start, end):
        # length of longest substring of s[start:end]
        if len(s[start:end]) < k:
            return 0
        m = Counter(s[start:end])
        for i in range(start, end):
            if m[s[i]] < k:
                # curr_letter cannot be included 
                return max(self.longestSubstring_substring(s, k, start, i), self.longestSubstring_substring(s, k, i+1, end))
        
        return end-start # all letters >=k

    def longestSubstring(self, s: str, k: int) -> int:
        return self.longestSubstring_substring(s=s, k=k, start=0, end=len(s))


class Solution3: 
    def longestSubstring(self, s: str, k: int) -> int:
        res = 0
        max_number_of_unique_letters = len(Counter(s))  # 26 is hard max
        for i in range(max_number_of_unique_letters):
            res = max(res, self.longestSubstring_k_unique(s, i+1, k))
        return res

    def longestSubstring_k_unique(self, s, num_unique_letters, k):
        # output: solution that has num_unique_letters unique letters
        l = 0
        res = 0
        letter2freq = defaultdict(lambda: 0)
        num_qualify_letters = 0
        for r in range(len(s)): # two ptrs
            letter2freq[s[r]] += 1
            if letter2freq[s[r]] == k:
                num_qualify_letters += 1
            while len(letter2freq) > num_unique_letters:
                letter2freq[s[l]] -= 1
                if letter2freq[s[l]] == k - 1: # if s[l] has a letter with freq=k
                    num_qualify_letters -= 1
                if letter2freq[s[l]] == 0:
                    del letter2freq[s[l]]
                l += 1
            if num_qualify_letters == num_unique_letters:
                res = max(res, r-l+1)
        return res



sol = Solution3()
sol.longestSubstring("ababacbc", 3)
sol.longestSubstring("ababa", 3)
sol.longestSubstring("aaabb", 3)


""" 386 · Longest Substring with At Most K Distinct Characters
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
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s, k):
        l = 0
        m = dict()
        res = 0
        num_unique_letter = 0
        for r in range(len(s)):
            if s[r] not in m:
                m[s[r]] = 1 
                num_unique_letter += 1
            else:
                m[s[r]] + 1

            while num_unique_letter > k:
                # if more than k in s[l:r+1] then move l till =k
                m[s[l]] = m[s[l]] - 1
                if m[s[l]] == 0:
                    del m[s[l]]
                    num_unique_letter -= 1
                l = l + 1
            res = max(res, r-l+1)
        return res

sol=Solution()
sol.lengthOfLongestSubstringKDistinct("eceba", 3)
sol.lengthOfLongestSubstringKDistinct("worrrld", 3)

"""[LeetCode] 424. Longest Repeating Character Replacement !!!
Given a string that consists of only uppercase English letters, you 
can replace any letter in the string with another letter at most k times. 
Find the length of a longest substring containing all repeating letters 
you can get after performing the above operations.

Note:
Both the string's length and k will not exceed 10^4.

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
from collections import Counter, defaultdict, deque
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        # two pters
        # condition: if r - l + 1 - maxCnt <= k then okay else move left pters
        m = defaultdict(lambda x:0)
        l = 0
        maxCnt = 0  # maxCnt of letter in s[l:r+1]
        res = 0
        for r in range(len(s)):
            m[s[r]] = m[s[r]] + 1
            maxCnt = max(maxCnt, m[s[r]])
            while r - l + 1 - maxCnt > k:
                # will not make r-l+1 as output
                m[s[l]] = m[s[l]] - 1
                l = l + 1
                maxCnt = max(m.values())  # O(26n), can be removed, but why?
            res = max(res, r-l+1)
        
        return res

# O(n): AABAC
# use d: char -> freq
# use a dict of 26 queues to save the idx
# for example when i=3: 
# d_idx['a'] = [0, 1, 3]; d_idx['b'] = [2]
# d['a'] = 3, d['b'] = 1
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        n = len(s)
        if k + 1 >= n: # cannot > n
            return n
        res = 0
        d_idx = defaultdict(deque)  # idx 
        d = defaultdict(lambda:0)  # freq
        for r in range(len(s)):
            curr_char = s[r]
            d_idx[curr_char].append(r)
            d[curr_char] += 1
            while r - d_idx[curr_char][0] + 1 > k + d[curr_char]:
                d_idx[curr_char].popleft()
                d[curr_char] -= 1
            res = max(res,k + d[curr_char])
            if res >= n:
                return n
            # print("{} / {}".format(d_idx[curr_char], r))
        return res

sol=Solution()
sol.characterReplacement("ABAB", 2)
sol.characterReplacement("AAAABC", 3)

"""[LeetCode] 76. Minimum Window Substring 最小窗口子串
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

Example 1:
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
                # s contains t, move l 
                if r+1-l < min_len:
                    res = s[l:r+1]
                    min_len = r+1-l
                # if s[l:r+1] covers T
                if s[l] in m:
                    m[s[l]] = m[s[l]] + 1
                    if m[s[l]] > 0:
                        num_letters_covered = num_letters_covered - 1
                l = l + 1
        
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
            while max(m.values()) <= 0:  # o(26n)
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
(i.e., from left to right, level by level).
For example: Given binary tree {3,9,20,#,#,15,7},
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
    def levelOrder(self, root: Optional[TreeNode]):
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
    def levelOrder(self, root: Optional[TreeNode]):
        res = []
        self.getLevelOrder(root, 0, res)
        return res

    def getLevelOrder(self, root, level, res):
        if root is None:
            return None
        if len(res) < level+1:
            # key: create [] for new level
            res.append([])
        res[level].append(root.val)
        self.getLevelOrder(root.left, level+1, res)
        self.getLevelOrder(root.right, level+1, res)
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
as "[1,2,3,null,null,4,5,null,null,null,null]", just the same as how LeetCode OJ serializes a binary tree. You do not necessarily need to
follow this format, so please be creative and come up with different approaches yourself.
"""
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# solution 1: BFS, using list is easier
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
                if idx < len(data) and data[idx] != '#':
                    # idx < len(data) can be rm, will never violate
                    curr_node.left = TreeNode(int(data[idx]))
                    q.append(curr_node.left)
                idx = idx + 1
                if idx < len(data) and data[idx] != '#':
                    curr_node.right = TreeNode(int(data[idx]))
                    q.append(curr_node.right)
                idx = idx + 1
        
        return root


# solution 2: dfs
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
        
    def deserialize_helper(self, res):
        # res is a list
        if res[0] == '#':
            return None
        root = TreeNode(int(res[0]))
        res = res[1:] # need to change res
        root.left = self.deserialize_helper(res)
        root.right = self.deserialize_helper(res)
        return root


class Codec: # ???
    def serialize(self, root):
        return ','.join(self.serialize_helper(root))
    
    def serialize_helper(self, root):
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
    def verticalTraversal(self, root: Optional[TreeNode]):
        # when level + 1, then left child order =- 1, right order =+ 1
        q = deque()
        q.append((0, root, 0))  # v_level, node, h_level
        m = defaultdict(list)  # key as order, value as (h_level, node.val)
        while len(q) > 0:
            for _ in range(len(q)):
                v_level, curr_node, h_level = q.popleft() # vertical, node, horizontal
                if curr_node:
                    m[v_level].append((h_level, curr_node.val))
                    if curr_node.left:
                        q.append((v_level-1, curr_node.left, h_level+1))  # time(horizontal order)+1
                    if curr_node.right:
                        q.append((v_level+1, curr_node.right, h_level+1))
        
        res = []
        for k in sorted(m.keys()):
            sorted_tup = sorted(m[k], key=lambda x:(x[0], x[1])) # sort by h_level and val
            res.append([tup[1] for tup in sorted_tup])  
        
        return res

# DFS: similarly, build a hashmap, and recursively save vertical, horizontal locations and val
# then sort by vertical, horizontal and val
class Solution_DFS:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        # may not need a dict, just a tuple (v, h, node), sort by x[0], x[1], x[2]
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

class Solution_BFS:
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
                    m[old_neighbor] = Node(old_neighbor.val)
                    q.append(old_neighbor)
                
                m[curr_old_node].neighbors.append(m[old_neighbor])

        return m[node]


class Solution_DFS:
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
            clone_nb = self.helper(old_neighbor, m)
            m[node].neighbors.append(clone_nb)

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
# use stop instead of bus
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:        
        stop2bus = defaultdict(list)
        for bus_idx, route in enumerate(routes):
            for stop in route:
                stop2bus[stop].append(bus_idx)
        q = deque()
        q.append((source, 0))
        visited_bus = set()  # visit at bus level is more efficient
        while len(q) > 0:
            size = len(q)
            for _ in range(size):
                curr_stop, curr_step = q.popleft()
                if curr_stop == target:
                    return curr_step
                for curr_bus in stop2bus[curr_stop]:
                    if curr_bus not in visited_bus:
                        visited_bus.add(curr_bus)
                        for stop in routes[curr_bus]:
                            q.append((stop, curr_step+1))
        
        return -1

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


"""317 Shortest Distance from All Buildings 建筑物的最短距离 
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
    dist = [[float('Inf') for _ in range(ncols)] for _ in range(nrows)] # cum dist matrix
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
    # update cum distance matrix, and visitable matrix
    nrows = len(grid)
    ncols = len(grid[0])
    # for building X at gird[i][j], update dist matrix
    visited = [[False for _ in range(ncols)] for _ in range(nrows)]  # visited map
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

#
# Try to simplify it
#
def shortestDistance(grid):
    num_buildings = 0
    nrows = len(grid)
    ncols = len(grid[0])
    dist = [[0 for _ in range(ncols)] for _ in range(nrows)]
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
    print(dist)
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
            dist[curr_i][curr_j] = dist[curr_i][curr_j] + curr_dist
            for direct in directs:
                next_i, next_j = curr_i+direct[0], curr_j+direct[1]
                if 0<=next_i<nrows and 0<=next_j<ncols and grid[next_i][next_j]==0 and visited[next_i][next_j] is False:
                    q.append((next_i, next_j, curr_dist+1))
                    visitable[next_i][next_j] += 1
                    visited[next_i][next_j] = True

    return None


shortestDistance(grid)


"""1293. Shortest Path in a Grid with Obstacles Elimination !!!
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
        if k >= nrows + ncols - 2: # speed up from 1000 ms to 100 ms  
            return nrows + ncols - 2
        visited = set((0, 0, 0))
        q = deque()
        q.append((0, 0, 0, 0))  # i, j, dist, k
        directs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        while q:
            for _ in range(len(q)):
                i, j, curr_dist, curr_k = q.popleft()
                if i == nrows-1 and j == ncols - 1:
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

Input:
org: [1,2,3], seqs: [[1,2],[1,3]]
Output:false

Input:
org: [1,2,3], seqs: [[1,2],[1,3],[2,3]]
Output:true

Input:org = [4,1,5,2,6,3], seqs = [[5,2,6,3],[4,1,5,2]]
Output:true

和course schedule ii 思路相似，利用seqs来重构org，如果发现1）无法重构2）有多种情况就返回false
多种情况的判断基于是否每一步q里都只有一个元素加进来（indegree为0的只有一个）
# https://www.lintcode.com/problem/605/solution/34974
"""
from collections import defaultdict, deque
# idea, [1,2,3] the position of 2 is confirmed only when we see [1,2], therefore [1,3] does not mean anything
class Solution:
    def sequenceReconstruction(self, org, seqs):
        if set(org) != set([num for seq in seqs for num in seq]):
            return False
        pre_m = defaultdict(list)  # m[p] = c: p is before c
        indegree = defaultdict(lambda: 0)  # how many numbers in front
        for seq in seqs:
            for i in range(len(seq)-1):
                pre_m[seq[i]].append(seq[i+1])
                indegree[seq[i+1]] = 1 if seq[i+1] not in indegree else indegree[seq[i+1]] + 1

        q = deque([node for node in org if indegree[node] == 0])
        res = []  # reconstruct results
        while q:
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
            # print(res)
        if res == org:
            return True 
        return False

sol=Solution()
sol.sequenceReconstruction([1,2,3], [[1,2],[1,3]])
sol.sequenceReconstruction([1,2,3], [[1,2],[1,3], [2,3]])
sol.sequenceReconstruction([4,1,5,2,6,3], [[5,2,6,3],[4,1,5,2]])


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
class Solution:
    def alienOrder(self, words):
        # Write your code here
        n_words = len(words)
        pre_m = defaultdict(set)  # better use set
        indegree = defaultdict(lambda:0)
        set_unique_letters = set(''.join(words))
        num_unique_letters = len(set_unique_letters)
        # go through index of diff words
        for word_idx in range(n_words - 1):
            curr_word, next_word = words[word_idx], words[word_idx+1]
            min_word_len = min(len(curr_word), len(next_word))
            for loc_idx in range(min_word_len):
                if curr_word[loc_idx] == next_word[loc_idx]:
                    continue
                if next_word[loc_idx] not in pre_m[curr_word[loc_idx]]:
                    pre_m[curr_word[loc_idx]].add(next_word[loc_idx])  # p -> c
                    indegree[next_word[loc_idx]] += 1
                    break
        # q = deque()  # not meet lexicograph condition !!
        q = [] # due to lexigraphical order, ["ca", "cb"] -> abc not cab 
        for letter in set_unique_letters:
            if indegree[letter] == 0:
                heapq.heappush(q, letter)
        
        res = ""
        while q:
            curr_letter = heapq.heappop(q)
            res = res + curr_letter
            for next_letter in pre_m[curr_letter]:
                indegree[next_letter] = indegree[next_letter] - 1
                if indegree[next_letter] == 0:
                    # same letter can be de-indegree for several times, "attt" ??
                    heapq.heappush(q, next_letter)

        return res if len(res) == num_unique_letters else ""

sol=Solution()
sol.alienOrder(["wrt","wrf","er","ett","rftt"])
sol.alienOrder( ["ca", "cb"])

"""
##############################################################################
深度优先搜索（DFS）
##############################################################################
1) 图中（有向无向皆可）的符合某种特征（比如最长）的路径以及长度
2）排列组合
3）遍历一个图（或者树）, 复制图或者树
4）找出图或者树中符合题目要求的全部方案
About the time and space complexity of recursion 
https://www.youtube.com/watch?v=OQi4n8EKRD8
"""

"""617. Merge Two Binary Trees
You are given two binary trees root1 and root2. return sum of them. 
Input: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
"""
# DFS is simple
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if root1 is None:
            return root2
        if root2 is None:
            return root1

        node = TreeNode(root1.val + root2.val)
        node.left = self.mergeTrees(root1.left, root2.left)
        node.right = self.mergeTrees(root1.right, root2.right)
        return node

# BFS is doable
class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not (t1 and t2):
            return t1 or t2
        queue1, queue2 = collections.deque([t1]), collections.deque([t2])
        while queue1 and queue2:
            node1, node2 = queue1.popleft(), queue2.popleft()
            if node1 and node2:
                node1.val = node1.val + node2.val
                if (not node1.left) and node2.left:
                    node1.left = TreeNode(0)
                if (not node1.right) and node2.right:
                    node1.right = TreeNode(0)
                queue1.append(node1.left)
                queue1.append(node1.right)
                queue2.append(node2.left)
                queue2.append(node2.right)
        return t1


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

Follow up: what if N ary tree instead of binary?  https://www.cnblogs.com/cnoodle/p/14349421.html

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

# donot need m! 
class Solution2:
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
            # add left and right into queue
            if curr_node.left:
                q.append(curr_node.left)
            if curr_node.right:
                q.append(curr_node.right)
        
        return root 


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
    def flipEquiv(self, root1, root2) -> bool:
        if root1 is None and root2 is None:
            return True
        if (root1 and not root2) or (not root1 and root2):
            return False
        if root1.val != root2.val:
            return False
        opt1 = self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right)
        opt2 = self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left)
        return opt1 or opt2


"""124. Binary Tree Maximum Path Sum
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
"""
// Author: Huahua
// Runtime: 19 ms
class Solution {
public:
    int maxPathSum(TreeNode* root) {
        if (!root) return 0;
        int ans = INT_MIN;
        maxPathSum(root, ans);
        return ans;
    }
private:
    int maxPathSum(TreeNode* root, int& ans) {
        if (!root) return 0;
        int l = max(0, maxPathSum(root->left, ans));
        int r = max(0, maxPathSum(root->right, ans));
        int sum = l + r + root->val;
        ans = max(ans, sum);
        return max(l, r) + root->val;
    }
};
"""
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = float('-Inf')
        if root is None:
            return 0
        self.maxPathOnOneSide(root)
        # res = max(node.val + max(maxPathOnOneSide(node.left), 0) + max(maxPathOnOneSide(node.right), 0)) for all node
        return self.res
    
    def maxPathOnOneSide(self, node):
        # 返回值的定义是以当前结点为终点的 path 之和
        if node is None:
            return 0
        # always first calculate the child before node
        left_max = max(0, self.maxPathOnOneSide(node.left))
        right_max = max(0, self.maxPathOnOneSide(node.right))
        # if child is larger, it will not update res
        self.res = max(self.res, left_max + right_max + node.val)
        return node.val + max(left_max, right_max)


class Solution2:
    # TLE
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        m = dict()
        res_including_root = root.val + \
            max(self.maxPathOnOneSide(root.left), 0) + \
            max(self.maxPathOnOneSide(root.right), 0)
        max_left = self.maxPathSum(root.left) if root.left else float('-Inf')  # repeat maxPathOnOneSide, should use a dict to save
        max_right = self.maxPathSum(root.right) if root.right else float('-Inf')
        res = max(max_left, max_right)
        return max(res, res_including_root)

    @lru_cache(None)
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
        LCA, num = self.getLCAAndNum(root, p, q)
        if LCA and num == 2:
            return LCA
        return None

    def getLCAAndNum(self, root, p, q):
        if root is None:
            return (None, 0)
        left_node, left_num = self.getLCAAndNum(root.left)
        right_node, right_num = self.getLCAAndNum(root.right)
        if root == p or root == q:
            return (root, 1 + left_num + right_num)
        if left_node and right_node:
            return (root, 2)
        return (left_node, left_num) if left_node else (right_node, right_num)


"""[LeetCode] 105. Construct Binary Tree from Preorder and Inorder Traversal 
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
"""
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if len(preorder) == 0:
            return None
        root_val = preorder[0]
        # for idx, val in enumerate(inorder):
        #     if val == root_val:
        #         break
        idx = inorder.index(root_val)  # speed up
        left_child_inorder = inorder[:idx]
        left_child_preorder = preorder[1:idx+1]
        right_child_inorder = inorder[idx+1:]
        right_child_preorder = preorder[idx+1:]
        root = TreeNode(root_val)
        root.left = self.buildTree(left_child_preorder, left_child_inorder)
        root.right = self.buildTree(right_child_preorder, right_child_inorder)
        return root

""" use hashmap to save location idx, will need entire inorder for recursion
left_child_len = i-ileft
right_child_len = iright-i
preorder:
    3,      [9,                   ]   [20,                 15,7]
l   pleft   pleft+1, pleft+i-ileft 
r                                     pleft+i-ileft+1,    pright

inorder: 
    [9,        ] 3,   [15,20,7]
l   ileft,  i-1, (i)
r                      i+1,  iright  
"""
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        self.m = dict()
        for index, value in enumerate(inorder):
            self.m[value] = index
        return self.buildSubTree(preorder, 0, len(preorder)-1, inorder, 0, len(inorder)-1)
    
    def buildSubTree(self, preorder, pleft, pright, inorder, ileft, iright):
        # use preorder to find the root, then locate its position in inorder
        # then use the location to split both
        if ileft > iright or pleft > pright:
            # otherwise ileft may be out of index 
            return None
        curr_node_val = preorder[pleft]
        curr_node = TreeNode(val=curr_node_val)
        loc = self.m[curr_node_val]  # current root index
        # based on diff of i and ileft, i and iright to move p
        # left_child_len = i-ileft
        # right_child_len = iright-i
        curr_node.left = self.buildSubTree(preorder, pleft+1, pleft+loc-ileft, inorder, ileft, loc-1)
        curr_node.right = self.buildSubTree(preorder, pleft+loc-ileft+1, pright, inorder, loc+1, iright) 
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
# skip 


"""[LeetCode] 1485. Clone Binary Tree With Random Pointer !!!
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
from collections import queue
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None, random=None):
        self.val = val
        self.left = left
        self.right = right
        self.random = random

class Solution:
    def copyRandomBinaryTree(self, root):
        if root is None:
            return None
        m = dict()
        q = deque()
        q.append(root)
        m[root] = TreeNode(root.val)

        while len(q) > 0:
            curr_node = q.popleft()

            if curr_node.left:
                if curr_node.left not in m:
                    # chance that curr_node.left exists due to random ptr
                    m[curr_node.left] = TreeNode(curr_node.left.val)
                # no matter whether random ptr covers, still need to add
                q.append(curr_node.left)  # ganrantee that all nodes in q also in m
                m[curr_node].left = m[curr_node.left]
            
            if curr_node.right:
                if curr_node.right not in m:
                    m[curr_node.right] = TreeNode(curr_node.right.val)
                q.append(curr_node.right)  # ganrantee that all nodes in q also in m
                m[curr_node].right = m[curr_node.right]
            
            if curr_node.random:
                if curr_node.random not in m:
                    m[curr_node.random] = TreeNode(curr_node.random.val)
                m[curr_node].random = m[curr_node.random]
        
        return m[root]

        
class Solution_DFS:
    def __init__(self):
        self.m = dict()
    
    def copyRandomBinaryTree(self, root):
        if root is None:
            return None
        if root in self.m:
            return self.m[root]
        self.m[root] = TreeNode(root.val)
        self.m[root].left = self.copyRandomBinaryTree(root.left)
        self.m[root].right = self.copyRandomBinaryTree(root.right)
        self.m[root].random = self.copyRandomBinaryTree(root.random)
        return self.m[root]


"""[LeetCode] 863. All Nodes Distance K in Binary Tree 二叉树距离为K的所有结点 !!! (follow up)
We are given a binary tree (with root node `root`), a `target` node, and an integer value `K`.
Return a list of the values of all nodes that have a distance K from the target node.  The answer can be returned in any order.

Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2
Output: [7,4,1]
Explanation:
The nodes that are a distance 2 from the target node (with value 5)
have values 7, 4, and 1.

Note that the inputs "root" and "target" are actually TreeNodes.
The descriptions of the inputs above are just serializations of these objects.

Solution: 
建立一个邻接链表，即每个结点最多有三个跟其相连的结点，左右子结点和父结点，
使用一个 HashMap 来建立每个结点和其相邻的结点数组之间的映射，这样就几乎完全将其当作图来对待了

Follow up:
Can you use O(1) extra space? 
https://www.cnblogs.com/grandyang/p/10686922.html
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        m = defaultdict(list)  # curr: [left, right, parent]
        self.find_neighbor(root, None, m)
        q = deque()
        q.append((target, 0))  # 0 is the distance
        visited = set()
        visited.add(target)
        res = []
        while len(q) > 0:
            for _ in range(len(q)):
                curr_node, curr_dist = q.popleft()
                if curr_dist == k:
                    res.append(curr_node.val)
                elif curr_dist < k:
                    for neighbor in m[curr_node]:
                        if neighbor not in visited and curr_dist+1 <=k:
                            q.append((neighbor, curr_dist+1))
                        visited.add(neighbor)
        
        return res

    def find_neighbor(self, curr, pre, m):
        if curr is None or curr in m:
            return None
        if pre:
            m[curr].append(pre)
        if curr.left:
            m[curr].append(curr.left)
        if curr.right:
            m[curr].append(curr.right)
        self.find_neighbor(curr.left, curr, m)
        self.find_neighbor(curr.right, curr, m)
        return None


"""1110. Delete Nodes And Return Forest  !!!
Given the root of a binary tree, each node in the tree has a distinct value.

After deleting all nodes with a value in to_delete, we are left with a forest (a disjoint union of trees).

Return the roots of the trees in the remaining forest. You may return the result in any order.
 
Example 1:
Input: root = [1,2,3,4,5,6,7], to_delete = [3,5]
Output: [[1,2,null,4],[6],[7]]

Example 2:
Input: root = [1,2,4,null,3], to_delete = [3]
Output: [[1,2,4]]
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# helper does not return anything
# O(N) time, O(N) space
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

# 
class Solution2:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        delete_set = set(to_delete)
        res = []
        self.helper(root, True, delete_set, res)
        return res

    def helper(self, node, isRoot, delete_set, res):
        # return node if not delete, else None; add root in subtree into res
        # isRoot: whether current node is root (no parent)
        if node is None:
            return None
        if isRoot and node.val not in delete_set:
            res.append(node)
        if node.val in delete_set:
            # if left, right are root if not none
            node.left = self.helper(node.left, True, delete_set, res)
            node.right = self.helper(node.right, True, delete_set, res)
        else:
            node.left = self.helper(node.left, False, delete_set, res)
            node.right = self.helper(node.right, False, delete_set, res)
        return node if node.val not in delete_set else None


"""[LeetCode] 230. Kth Smallest Element in a BST 二叉搜索树中的第K小的元素 !!!
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note: 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Example 1:
Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1

Example 2:
Input: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
Output: 3
Follow up:
What if the BST is modified (insert/delete operations) often and you need to find 
the kth smallest frequently? How would you optimize the kthSmallest routine?
"""

# Solution 1: inorder traverse 
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res = []
        self.inorder(root, res)
        return res[k-1]
    
    def inorder(self, node, res):
        if node is None:
            return 
        if node.left:
            self.inorder(node.left, res)
        res.append(node.val)
        if node.right:
            self.inorder(node.right, res)
        return None

"""中序遍历的非递归写法: use stack first push left till end, then right (repeat)
def kthSmallest(root, k):
    res = []
    stack = []
    p, pre = root, None
    while p or stack:
        while p:
            stack.append(p)
            p = p.left
        p = stack.pop()
        res.append(p)
        p = p.right
    return res[k-1]
"""

# Solution 2: binary search 
"""
class Solution {
public:
    int kthSmallest(TreeNode* root, int k) {
        int cnt = count(root->left);
        if (k <= cnt) {  
            return kthSmallest(root->left, k);
        } else if (k > cnt + 1) {
            return kthSmallest(root->right, k - cnt - 1);
        }
        return root->val; //cnt = k-1
    }
    int count(TreeNode* node) {
        if (!node) return 0;
        return 1 + count(node->left) + count(node->right);
    }
};
"""
# Follow up: 
# we can modify the sol 2 to handle follow up
# modify the tree node structure to let it also save cnt information
# so no need to run "int cnt = count(root->left)"
# https://www.cnblogs.com/grandyang/p/4620012.html


"""98. Validate Binary Search Tree
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

Input: root = [2,1,3]
Output: true

Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
"""
# 也可以通过利用中序遍历结果为有序数列来做
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self.valid(root, float('-Inf'), float('Inf'))
    
    def valid(self, node, mn, mx):
        # need to pass the curr lower and upper limit
        if node is None:
            return True
        if node.val <= mn or node.val >= mx:
            return False
        is_left_valid = self.valid(node.left, mn, node.val)
        is_right_valid = self.valid(node.right, node.val, mx)
        return is_left_valid and is_right_valid

"""中序遍历的非递归写法
public class Solution {
    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> s = new Stack<TreeNode>();
        TreeNode p = root, pre = null;
        while (p != null || !s.empty()) {
            while (p != null) {
                s.push(p);
                p = p.left;
            }
            p = s.pop();
            if (pre != null && p.val <= pre.val) return false;
            pre = p;
            p = p.right;
        }
        return true;
    }
}
"""

"""[LeetCode] 270. Closest Binary Search Tree Value 最近的二分搜索树的值
Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

Note:
Given target value is a floating point.
You are guaranteed to have only one unique value in the BST that is closest to the target.
Example:

Input: root = [4,2,5,1,3], target = 3.714286

    4
   / \
  2   5
 / \
1   3

Output: 4
"""
# Solution 1: inorder traverse 
# Solution 2: binary search
class Solution:
    def closestValue(self, root, target):
        self.res = root.val
        self.helper(root, target)
        return self.res

    def helper(root, target, res):
        if root is None:
            return None
        if root.val == target:
            self.res = root.val
            return None
        if abs(root.val - target) < abs(self.res - target):
            self.res = root.val
        if root.val > target:
            self.helper(root.left, target)
        else:
            self.helper(root.right, target)
        return
            
# Another way:
"""
class Solution {
public:
    int closestValue(TreeNode* root, double target) {
        int a = root->val;
        TreeNode *t = target < a ? root->left : root->right;
        if (!t) return a;
        int b = closestValue(t, target);
        return abs(a - target) < abs(b - target) ? a : b;
    }
};
"""

"""[LeetCode] Trim a Binary Search Tree 修剪一棵二叉搜索树
 
Given a binary search tree and the lowest and highest boundaries as L and R, 
trim the tree so that all its elements lies in [L, R] (R >= L). 
You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

Example 1:

Input: 
    1
   / \
  0   2

  L = 1
  R = 2

Output: 
    1
      \
       2
 

Example 2:

Input: 
    3
   / \
  0   4
   \
    2
   /
  1

  L = 1
  R = 3

Output: 
      3
     / 
   2   
  /
 1
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if root is None:
            return None
        if root.val < low:
            # left part is trimed
            return self.trimBST(root.right, low, high)
        if root.val > high:
            return self.trimBST(root.left, low, high)
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root


"""[LeetCode] 333. Largest BST Subtree 最大的二分搜索子树
Given a binary tree, find the largest subtree which is a Binary Search Tree (BST), 
where largest means subtree with largest number of nodes in it.

Note:
A subtree must include all of its descendants.

Input: [10,5,15,1,8,null,7]
   10 
   / \ 
  5  15 
 / \   \ 
1   8   7

Output: 3
Explanation: The Largest BST Subtree in this case is the highlighted one.
             The return value is the subtree's size, which is 3.

Follow up:
Can you figure out ways to solve it with O(n) time complexity?
"""
# 下面我们来看一种更简洁的写法，对于每一个节点，都来验证其是否是 BST，
# 如果是的话，就统计节点的个数即可，参见代码如下
"""
class Solution {
public:
    int largestBSTSubtree(TreeNode* root) {
        if (!root) return 0;
        if (isValid(root, INT_MIN, INT_MAX)) return count(root);
        return max(largestBSTSubtree(root->left), largestBSTSubtree(root->right));
    }
    bool isValid(TreeNode* root, int mn, int mx) {
        if (!root) return true;
        if (root->val <= mn || root->val >= mx) return false;
        return isValid(root->left, mn, root->val) && isValid(root->right, root->val, mx);
    }
    int count(TreeNode* root) {
        if (!root) return 0;
        return count(root->left) + count(root->right) + 1;
    }
};


class Solution {
public:
    vector<int> DFS(TreeNode* root, int& ans)
    {
        if(!root) return vector<int>{0, INT_MAX, INT_MIN};
        auto left=DFS(root->left, ans), right=DFS(root->right, ans);
        if(root->val > left[2] && root->val < right[1])
        {
            int Min =min(root->val, left[1]), Max =max(root->val, right[2]);
            ans = max(ans, left[0] + right[0] + 1);
            return vector<int>{left[0] +right[0] +1, Min , Max};
        }
        return vector<int>{0, INT_MIN, INT_MAX};
    }
    
    int largestBSTSubtree(TreeNode* root) {
        int ans = 0;
        DFS(root, ans);
        return ans;
    }
"""
class Solution:
    def __init__():
        self.res = 0
    
    def largestBSTSubtree(self, root):
        self.GetBSTSize(root, float('-Inf'), float('Inf'))
        return self.res

    def GetBSTSize(self, node, mn, mx):
        if node is None:
            return 0
        # curr_res = 1
        # if node.val <= mn or node.val >= mx:
        #     curr_res = -1 # not a BST            
        left = self.GetBSTSize(node.left, mn, node.val)
        right = self.GetBSTSize(node.right, node.val, mx)
        if left >= 0 and right >= 0 and mn < node.val < mx:
            self.res = max(self.res, left + right + 1)
            return left + right + 1
        # node is not BST but still
        return -1


class Solution:
    def __init__():
        self.res = 0
    
    def largestBSTSubtree(self, root):
        self.GetBSTSize(root)
        return self.res

    def GetBSTSize(self, node):
        size = self.helper(node, float('-Inf'), float('Inf'))
        if size != -1:
            # node is root of a BST, any of is subtree is BST
            # no need to go deeper
            self.res = max(size, self.res)
            return None
        # else check its kids
        self.GetBSTSize(node.left)
        self.GetBSTSize(node.right)

    def helper(self, node, mn, mx):
        # return the size of BST with node as root,
        # return -1 if node is not root of a BST
        if node is None:
            return 0
        if node.val <= mn or node.val >= mx:
            return -1 # not a BST
        left = self.helper(node.left, mn, node.val)
        if left == -1:
            # if left node is not a root of BST, then node is not
            return -1
        right = self.helper(node.right, node.val, mx)
        if right == -1:
            return -1
        return left + right + 1

# follow up:
# https://www.cnblogs.com/grandyang/p/5188938.html
"""
class Solution {
public:
    int largestBSTSubtree(TreeNode* root) {
        int res = 0, mn = INT_MIN, mx = INT_MAX;
        isValidBST(root, mn, mx, res);
        return res;
    }
    void isValidBST(TreeNode* root, int& mn, int& mx, int& res) {
        if (!root) return;
        int left_cnt = 0, right_cnt = 0, left_mn = INT_MIN;
        int right_mn = INT_MIN, left_mx = INT_MAX, right_mx = INT_MAX;
        isValidBST(root->left, left_mn, left_mx, left_cnt);
        isValidBST(root->right, right_mn, right_mx, right_cnt);
        if ((!root->left || root->val > left_mx) && (!root->right || root->val < right_mn)) {
            res = left_cnt + right_cnt + 1;
            mn = root->left ? left_mn : root->val;
            mx = root->right ? right_mx : root->val;
        } else {
            res = max(left_cnt, right_cnt);    
        }
    }
};
"""


"""[LeetCode] 285. Inorder Successor in BST 二叉搜索树中的中序后继节点 !!!
Given a binary search tree and a node in it, find the in-order successor of that node in the BST.

The successor of a node p is the node with the smallest key greater than p.val.
"""
# 首先要确定中序遍历的后继:
# - 如果该节点有右子节点, 那么后继是其右子节点的子树中最左端的节点
# - 如果该节点没有右子节点, 那么后继是离它最近的祖先, 该节点在这个祖先的左子树内.
# 使用循环实现:
# - 查找该节点, 并在该过程中维护上述性质的祖先节点
# - 查找到后, 如果该节点有右子节点, 则后继在其右子树内; 否则后继就是维护的那个祖先节点
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root == None:
            return None
        if root.val <= p.val:
            # make sure root.val > p
            return self.inorderSuccessor(root.right, p)
        
        left = self.inorderSuccessor(root.left, p)  # find the successor node in left subtree
        if left != None:
            return left
        else:
            # means all nodes in left subtree < p
            return root
        
# 递归iterative：这种方法充分地利用到了 BST 的性质，首先看根节点值和p节点值的大小，
# 如果根节点值大，说明p节点肯定在左子树中，那么此时先将 res 赋为 root，然后 root 移到其左子节点
# ，循环的条件是 root 存在，再比较此时 root 值和p节点值的大小，如果还是 root 值大，重复上面的操作
# ，如果p节点值，那么将 root 移到其右子节点，这样当 root 为空时，res 指向的就是p的后继节点，
class Solution:
    def inorderSuccessor(self, root, p):
        res = None
        while root:
            if root.val > p.val:
                # only record when root > p value
                # if root>p, target can either root or in left subtree
                res = root.val
                root = root.left
            else:
                # if root<=p, target can only previous that>p, or in right subtree
                root = root.right
        
        return res
                

"""[LeetCode] 510. Inorder Successor in BST II 二叉搜索树中的中序后继节点之二 !!!
Given a binary search tree and a node in it, find the in-order successor of that node in the BST.
The successor of a node p is the node with the smallest key greater than p.val.

You will have direct access to the node but not to the root of the tree. Each node will have a reference to its parent node.

Input: tree below, 5
Output: 7
    8 
   / \ 
  5  15 
 / \   \ 
1   7   16

Follow up:
Could you solve it without looking up any of the node's values?

Solution:
这道题并没有确定给我们根结点，只是给了树的任意一个结点.

仔细观察例子不难发现，当某个结点存在右子结点时，其中序后继结点就在子孙结点中，反之则在祖先结点中。
这样我们就可以分别来处理，当右子结点存在时，我们需要找到右子结点的最左子结点，这个不难，就用个 while 循环就行了。
当右子结点不存在，我们就要找到第一个比其值大的祖先结点，也是用个 while 循环去找即可，参见代码如下
"""
def inorderSuccessor(node):
    if node is None:
        return None
    if node.right:
        node = node.right
        while node and node.left:
            node = node.left
        return node
    # else
    while node.parent and node.val >= node.parent.val:
        # if node.parent.val > node.val, then parent is on the right 
        # and will gt all the previous node seq
        node = node.parent
    
    return node.parent  # node.parent.val > node.val

# 本题的 Follow up 让我们不要访问结点值，那么上面的解法就不行了。因为当 node 没有右子结点时，
# 我们没法通过比较结点值来找到第一个大于 node 的祖先结点。虽然不能比较结点值了，
# 我们还是可以通过 node 相对于其 parent 的位置来判断，当 node 是其 parent 的左子结点时，
# 我们知道此时 parent 的结点值一定大于 node，因为这是二叉搜索树的性质。
# 若 node 是其 parent 的右子结点时，则将 node 赋值为其 parent，继续向上找，直到其 parent 结点不存在了，
# 此时说明不存在大于 node 值的祖先结点，这说明 node 是 BST 的最后一个结点了，没有后继结点，直接返回 nullptr 即可

def inorderSuccessor(node):
    if node is None:
        return None
    if node.right:
        node = node.right
        while node and node.left:
            node = node.left
        return node
    while node.parent and node == node.parent.right:
        node = node.parent
    
    return node.parent  # node.parent.left = node (node.val is smaller)
    # while node:
    #     if node.parent is None:
    #         return None
    #     if node == node.parent.left:
    #         # node is left child, then parent.val > node.val
    #         return node.parent
    #     node = node.parent
    # return None


"""341. Flatten Nested List Iterator
You are given a nested list of integers nestedList. Each element is either an integer or a list 
whose elements may also be integers or other lists. Implement an iterator to flatten it.

Implement the NestedIterator class:
NestedIterator(List<NestedInteger> nestedList) Initializes the iterator with the nested list nestedList.
int next() Returns the next integer in the nested list.
boolean hasNext() Returns true if there are still some integers in the nested list and false otherwise.

Input: nestedList = [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
Explanation: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,1,2,1,1].

Input: nestedList = [1,[4,[6]]]
Output: [1,4,6]
Explanation: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,4,6].

res = []
while iterator.hasNext()
    append iterator.next() to the end of res
return res
"""

class NestedInteger:
   def isInteger(self) -> bool:
       """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       """

   def getInteger(self) -> int:
       """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       Return None if this NestedInteger holds a nested list
       """

   def getList(self) -> [NestedInteger]:
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       """

from collections import deque
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.q = deque()
        self.make_queue(nestedList)
        
    
    def next(self) -> int:
        return self.q.popleft()
    
    def hasNext(self) -> bool:
        return len(self.q) > 0
    
    def make_queue(self, nestedList):
        for i in range(len(nestedList)):
            if nestedList[i].isInteger():
                self.q.append(nestedList[i].getInteger())
            else:
                self.make_queue(nestedList[i].getList())
         

"""[LeetCode] 364. Nested List Weight Sum II 嵌套链表权重和之二
Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Different from the previous question where weight is increasing from root to leaf, now 
the weight is defined from bottom up. i.e., the leaf level integers have weight 1, and the root level integers have the largest weight.

Example 1:
Input: [[1,1],2,[1,1]]
Output: 8 
Explanation: Four 1's at depth 1, one 2 at depth 2.

Example 2:
Input: [1,[4,[6]]]
Output: 17 
Explanation: One 1 at depth 3, one 4 at depth 2, and one 6 at depth 1; 1*3 + 4*2 + 6*1 = 17.
# idea: use a vector of size K to save the sum of numbers at each depth, then sum up with weights
"""
class Solution:
    def depthSumInverse(self, nestedList):
        res_vector = [] # or can use a dict: depth -> sum at depth
        self.depthSumByLevel(nestedList, res_vector, 1)
        wsum = 0
        for i in range(len(res_vector)):
            wsum = res_vector[i] * (len(res_vector) - i)
        return wsum

    def depthSumByLevel(self, nestedList, res_vector, depth):
        if len(res_vector) < depth:
            res_vector.append(0)
        for i in range(len(nestedList)):
            if nestedList[i].isInteger():
                res_vector[depth] += nestedList[i].getInteger()
            else:
                self.depthSumByLevel(nestedList[i], res_vector, depth+1)
        return None

from collections import defaultdict
class Solution2:
    def depthSumInverse(self, nestedList):
        self.d = defaultdict(lambda: 0)
        self.getSum(nestedList, 1)
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
# Solution 2 把每一层的数字都先加起来放到一个变量 unweighted 中，然后每层遍历完了之后，
# 就加到 weighted 变量中。再遍历下一层，再把数字加到 unweighted 中，当前层遍历完成了之后再次加到 weighted 变量中，
# 注意此时 unweighted 中还包含上一层的数字和，此时就相当于上一层的数字和加了两次
"""
class Solution {
public:
    int depthSumInverse(vector<NestedInteger>& nestedList) {
        int unweighted = 0, weighted = 0;
        queue<vector<NestedInteger>> q;
        q.push(nestedList);
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                vector<NestedInteger> t = q.front(); q.pop();
                for (auto a : t) {
                    if (a.isInteger()) unweighted += a.getInteger();
                    else if (!a.getList().empty()) q.push(a.getList());
                }
            }
            weighted += unweighted;
        }
        return weighted;
    }
};
"""


"""394. Decode String !!!
Given an encoded string, return it's decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the 
square brackets is being repeated exactly k times. Note that k is guaranteed to
 be a positive integer.

You may assume that the input string is always valid; No extra white spaces, 
square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits 
and that digits are only for those repeat numbers, k. For example, there won't 
be input like 3a or 2[4].

Examples:
s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
"""
s = "2[abc]3[cd]ef"

decodeString("3[a]2[bc]")
decodeString("3[a2[c]]")
decodeString("2[abc]3[cd]ef")
decodeString("abc3[cd]xyz")
decodeString("2[abc]3[cd]ef")
s = "3[a2[c]]"

# Recursion idea: 3[f(ab)]b2[g(b)] -> 1) decoder fcn loop through 3[f(ab)], b, 2[g(b)]
# 2) helper process 3[f(ab)] -> [decoder(f(ab))] * 3  [may combine the two fcn into one]
# one core is to find close parenthesis given open idx, use a hashmap to reduce to O(1) per search
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

# class Solution:
#     def decodeString(self, s: str) -> str:
#         self.m = dict() # map open idx [ to close idx ] location
#         # find all pairs
#         open_idx = []
#         for i in range(len(s)):
#             if s[i] == '[':
#                 open_idx.append(i)
#             elif s[i] ==']':
#                 self.m[open_idx.pop()] = i
#         return self.decoder(s, 0, len(s)) # s[:len(s)]

#     def decoder(self, s, start_idx, end_idx) -> str:
#         # break 3[ab]2[b] -> 3[ab], 2[b], call helper for each
#         res = ""
#         start, i = start_idx, start_idx
#         # curr_str = ""
#         while i < end_idx:
#             if s[i] == '[':
#                 # find the location of closing parenthesis
#                 close_idx = self.m[i]  # right parent location
#                 # res += self.helper(s[start:close_idx+1])
#                 res += self.helper(s, start, close_idx)
#                 start, i = close_idx + 1, close_idx + 1
#                 curr_str = ""
#             else:
#                 if s[i].isdigit():
#                     # cannot be ] because i = close_idx + 1
#                     # curr_str += s[i]
#                     i += 1
#                 else: # if letter, add to s, adjust start
#                     res += s[i]
#                     start = i + 1
#                     i += 1                
#         return res
    
#     def helper(self, s, start_idx, end_idx):
#         # handle 3[ab] -> decodestring(ab) * 3
#         # s[end_idx] must be ]
#         num_str = ""
#         i = start_idx
#         while s[i].isdigit():
#             num_str += s[i]
#             i += 1
#         string = self.decoder(s, i+1, end_idx) # btw [ and ]
#         return string * int(num_str)


sol=Solution()
sol.decodeString(s="2[abc]3[cd]ef")
sol.decodeString(s="3[a2[c]]")
sol.decodeString(s="3[z]2[2[y]pq4[2[jk]e1[f]]]ef")

# Non-recursion. Stack, solution 1: 
# always push before meeting ]
# if ], pop all the way until the number under [
class Solution:
    def decodeString(self, s):
        res = ''
        stack = []
        temp_num = ''  # cum number until [
        # temp_str = ''  # cum str until ]
        for i in range(len(s)):
            if s[i].isdigit():
                temp_num += s[i]
            elif s[i] == '[':
                stack.append(int(temp_num))
                stack.append(s[i]) # separate number from letter
                temp_num = ''
            elif s[i] == ']':
                # stack.append(temp_str)
                temp_str = ''
                while stack[-1].isalpha(): # pop all letters
                    temp_str = stack.pop() + temp_str # reverse order
                stack.pop()  # must be [ after letter, followed by number
                curr_num = stack.pop()
                curr_str = temp_str * curr_num
                if len(stack) > 0:
                    stack.append(curr_str)
                else: # if empty, then [] is most outside, add to final res
                    res += curr_str
                    temp_str = ''
            else: # letter
                if len(stack) == 0:
                    res += s[i]
                else:
                    stack.append(s[i])

        return res

# Stack sol2, faster
def decodeString(s):
    n = len(s)
    res = ""
    temp_num = ""
    temp_str = ""
    s_num = [] # stack for num
    s_str = [] # stack for string
    for i in range(n):
        if s[i].isdigit():
            temp_num = temp_num + s[i]
        elif s[i] == '[':
            # 如果遇到左括号，我们把当前 temp_num 压入数字栈中，把当前temp_str压入字符串栈中；
            s_num.append(int(temp_num))
            s_str.append(temp_str)
            temp_num = ""
            temp_str = ""
        elif s[i] == ']':
            # 如果遇到右括号时，我们取出数字栈中顶元素，存入变量num0，
            # 然后给字符串栈的顶元素循环加上num0个t字符串，然后取出顶元素存入字符串t中
            num0 = s_num.pop()  # current multipler
            str0 = s_str.pop()  # previous string 
            temp_str = str0 + temp_str * num0
        else:
            # alpha
            temp_str = temp_str + s[i]

    return temp_str


"""51. N-Queens !!!
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
Given an integer n, return all distinct solutions to the n-queens puzzle.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a 
queen and an empty space respectively.

For example,
There exist two distinct solutions to the 4-queens puzzle:

[
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]

Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
经典的N皇后问题，基本所有的算法书中都会包含的问题，经典解法为回溯递归，一层一层的向下扫描，需要用到一个pos数组，
其中pos[i]表示第i行皇后的位置，初始化为-1，然后从第0开始递归，每一行都一次遍历各列，判断如果在该位置放置皇后会不会有冲突，
以此类推，当到最后一行的皇后放好后，一种解法就生成了，将其存入结果res中，然后再还会继续完成搜索所有的情况，代码如下：
"""
from copy import deepcopy
class Solution:
    def solveNQueens(self, n: int):
        self.res = []
        self.size = n
        self.add_a_queen(0, [["e"]*n for _ in range(n)]) # e means unattacked
        final_res = [[''.join(row) for row in board] for board in self.res]
        return final_res # [''.join(row) for row in self.res]
    
    def add_a_queen(self, row_i, curr_board):
        if row_i == self.size:
            self.res.append(curr_board)
            return
        for j in range(self.size):
            if curr_board[row_i][j] == 'e':
                next_board = self.update_board(row_i, j, curr_board)  # put a Q at row_i, j
                self.add_a_queen(row_i+1, next_board)
        return 
    
    def update_board(self, i, j, curr_board):
        # put Q at i,j
        board = deepcopy(curr_board)  # do not change the input
        board[i][j] = 'Q'
        for (di, dj) in [(-1, -1), (-1, 1), (1, -1), (1,1), (-1,0), (1,0), (0,-1),(0,1)]:
            curr_i, curr_j = i+di, j+dj
            while 0<=curr_i<self.size and 0<=curr_j<self.size:
                if board[curr_i][curr_j]=='e':
                    board[curr_i][curr_j]='.'
                curr_i, curr_j = curr_i+di, curr_j+dj
        
        return board


sol=Solution()
len(sol.solveNQueens(8))
len(sol.solveNQueens(10))

# optimize the memory usage
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        queens = [-1] * n
        # queens is a one-dimension array, like [1, 3, 0, 2] means
        # [.Q..]
        # [...Q]
        # [Q...]
        # [..Q.]
        # index represents row no and value represents col no
        def dfs(index):
            if index == len(queens): # n queens have been placed correctly
                res.append(queens[:])
                return  # backtracking
            for i in range(len(queens)):
                queens[index] = i
                if valid(index):  # pruning
                    dfs(index + 1, )

        # check whether nth queens can be placed
        def valid(n):
            for i in range(n):
                if abs(queens[i] - queens[n]) == n - i:  # same digonal
                    return False
                if queens[i] == queens[n]:  # same column
                    return False
            return True

        # given queens = [1,3,0,2] this function returns 
        # [.Q..]
        # [...Q]
        # [Q...]
        # [..Q.]
        def make_board(queens):
            n = len(queens)
            board = []
            board_temp = [['.'] * n for _ in range(n)]
            for row, col in enumerate(queens):
                board_temp[row][col] = 'Q'
            for row in board_temp:
                board.append("".join(row))
            return board

        def make_all_boards(res):
            actual_boards = []
            for queens in res:
                actual_boards.append(make_board(queens))
            return actual_boards

        dfs(0)
        return make_all_boards(res)


""" 829 · Word Pattern II !!!
Description
Given a pattern and a string str, find if str follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern 
and a non-empty substring in str.(i.e if a corresponds to s, then b cannot correspond to s. 
For example, given pattern = "ab", str = "ss", return false.)

Example 1
Input:
pattern = "abab"
str = "redblueredblue"
Output: true
Explanation: "a"->"red","b"->"blue"

Example 2
Input:
pattern = "aaaa"
str = "asdasdasdasd"
Output: true
Explanation: "a"->"asd"
Example 3

Input:
pattern = "aabb"
str = "xyzabcxzyabc"
Output: false

取出当前位置的模式字符，然后从单词串的r位置开始往后遍历，每次取出一个单词，如果模式字符已经存在 HashMap 中，
而且对应的单词和取出的单词也相等，那么再次调用递归函数在下一个位置，如果返回 true，那么就返回 true。
反之如果该模式字符不在 HashMap 中，要看有没有别的模式字符已经映射了当前取出的单词，如果没有的话，
建立新的映射，并且调用递归函数，注意如果递归函数返回 false 了，要在 HashMap 中删去这个映射
"""
class Solution:
    """
    @param pattern: a string,denote pattern string
    @param str1: a string, denote matching string
    @return: a boolean
    """
    def wordPatternMatch(self, pattern, str1):
        # write your code here
        m = dict()
        used = set() # save how many map values, diff keys cannot map to the same value
        res = self.helper(pattern, 0, str1, 0, m, used)
        print(m)
        return res
    
    def helper(self, pattern, p, str1, r, m, used):
        if p == len(pattern) and r == len(str1):
            return True
        if p == len(pattern) or r == len(str1):
            return False
        curr_c = pattern[p]
        if curr_c in m:
            # if already mapped, then stick to it
            if len(str1[r:]) < len(m[curr_c]) or str1[r:r+len(m[curr_c])] != m[curr_c]:
                return False
            return self.helper(pattern, p+1, str1, r+len(m[curr_c]), m, used)
        
        for i in range(r, len(str1)):
            if len(str1) - i < len(pattern) - p:
                # trim, if str1 rem even shorter, then quit (map to non-empty)
                break
            rem_str = str1[r:(i+1)]  # not contain i+1
            if rem_str not in used:
                # diff k cannot map to the same chars
                m[curr_c] = rem_str
                used.add(rem_str)
                if self.helper(pattern, p+1, str1, i+1, m, used):
                    return True
                del m[curr_c]
                used.remove(rem_str)
        return False


sol=Solution()
sol.wordPatternMatch("abab", "redblueredblue")
sol.wordPatternMatch("aaaa", "asdasdasdasd")
sol.wordPatternMatch("aabb", "xyzabcxzyabc")


"""[LeetCode] 126. Word Ladder II 词语阶梯之二
Given two words (beginWord and endWord), and a dictionary's word list, 
find all shortest transformation sequence(s) from beginWord to endWord, 
such that:

Only one letter can be changed at a time
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return an empty list if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"]
]
Example 2:

Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: []

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.

"""          
from collections import defaultdict, deque
wordList = ["hot","dot","dog","lot","log","cog"]
graph = defaultdict(list)
beginWord = "hit"
endWord = "cog"


"cat"
"fin"
["ion","rev","che","ind","lie","wis","oct","ham","jag","ray","nun","ref","wig","jul","ken","mit",
"eel","paw","per","ola","pat","old","maj","ell","irk","ivy","beg","fan","rap","sun","yak","sat",
"fit","tom","fin","bug","can","hes","col","pep","tug","ump","arc","fee","lee","ohs","eli","nay",
"raw","lot","mat","egg","cat","pol","fat","joe","pis","dot","jaw","hat","roe","ada","mac"]

findLadders(beginWord, endWord, wordList)

class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList):
        m = self.create_nb_graph(beginWord, wordList)  # save neighbors
        res = []
        visited = set()
        q = deque()
        q.append((beginWord, [beginWord], 1))  # save both word and path
        visited.add(beginWord)
        shortest_len = float('Inf')
        while len(q) > 0:
            for _ in range(len(q)):
                curr_visited = set()
                curr_word, curr_path, curr_path_len = q.popleft()
                for nb_word in m[curr_word]:
                    if nb_word not in visited:
                        curr_visited.add(nb_word)  # add to current visted
                        if nb_word == endWord:
                            shortest_len = curr_path_len + 1
                            # print(curr_path+[nb_word])
                            res.append(curr_path+[nb_word])
                        else:
                            q.append((nb_word, curr_path+[nb_word], curr_path_len+1))
            
            visited.update(curr_visited)
            if shortest_len == curr_path_len + 1:
                return res
        
        return res
                        
    def is_neighbor(self, word1, word2):
        if len(word1) == len(word2):
            ct = 0
            for i in range(len(word1)):
                if word1[i] != word2[i]:
                    ct += 1
            return ct == 1
        return False 
    
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

# DFS: cannot decided shortest path on the way, 
# so have to print all and then filter
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
                # trim a bit, still cannot determine final shortest path
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

# another BFS
def findLadders(beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
    L = len(beginWord)
    graph = defaultdict(list)

    for word in wordList:
        for i in range(L):
            graph[word[:i] + '*' + word[i + 1:]].append(word)

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


""" Print all shortest path from start to target
https://stackoverflow.com/questions/20257227/how-to-find-all-shortest-paths
GIven a graph and start and target node, print all the shortest path from start to target
Dijkstra algorithm will do the trick. But can you use BFS?

Input: graph = [[1,2],[3],[3],[]], 0, 3
Output: [[0,1,3],[0,2,3]]
Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.

Input: graph = [[3,1],[3,2,4],[3],[4],[]], 0, 4
Output: [[0,3,4],[0,1,4]]

# need a parent dict to store the previous node and distance in path
# note: if for weighted graph, then use dijkstra algorithm
"""
from collections import defaultdict, deque

graph = [[3,1],[3,2,4],[3],[4],[]]; start=0; target=4

def getShortestPaths_bfs1(graph, start, target):
    # save path in q, drawback is memory usage
    q = deque()
    parent = defaultdict(list)
    q.append((start, [start]))
    visited = set()
    find_target = False
    res = []
    while len(q) > 0:
        curr_visited = set()
        for _ in range(len(q)):
            curr_node, curr_path = q.popleft()
            if curr_node == target:
                find_target = True
                res.append(curr_path)
                continue
            for next_node in graph[curr_node]:
                if next_node not in visited:
                    curr_visited.add(next_node)
                    q.append((next_node, curr_path + [next_node]))
        visited.update(curr_visited)
        if find_target:
            return res
    return res

def getShortestPaths_bfs2(graph, start, target):
    # use dict to track path
    q = deque()
    parent = defaultdict(list)
    q.append((start, 0))
    visited = set()
    while len(q) > 0:
        for _ in range(len(q)):
            curr_node, curr_dist = q.popleft()
            if curr_node == target:
                break
            for next_node in graph[curr_node]:
                if next_node not in visited:
                    visited.add(next_node)
                    parent[next_node].append((curr_node, curr_dist+1))
                    q.append((next_node, curr_dist+1))
                else:
                    _, prev_dist = parent[next_node][0]
                    if prev_dist == curr_dist + 1:
                        parent[next_node].append((curr_node, curr_dist+1))
                        # if curr_dist + 1 bigger, then should not add into
    
    # print out path based on parent
    res = []
    q1 = deque()
    q1.append([target])
    while len(q1) > 0:
        for _ in range(len(q1)):
            curr_path = q1.popleft()
            for next_node, _ in parent[curr_path[-1]]:
                q1.append(curr_path + [next_node])
                if next_node == start:
                    res.append((curr_path + [next_node])[::-1])
    
    return res


"""Restore IP Addresses 复原IP地址 
Given a string containing only digits, restore it by returning all possible 
valid IP address combinations.

For example:
Given "25525511135",

return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
"""
class Solution:
    def restoreIpAddresses(self, s: str):
        res = []
        self.restoreHelper(s, [], 0, res)
        return res
    
    def restoreHelper(self, rem_str, curr_res, curr_len, res):
        if rem_str == "" and curr_len == 4:
            res.append(".".join(curr_res))
            return 
        if rem_str == "":
            return 
        for i in range(1, 4):
            if i <= len(rem_str): # o.w. duplicate
                if i == 1 or (rem_str[0] != '0' and int(rem_str[:i]) <= 255):
                    self.restoreHelper(rem_str[i:], curr_res+[rem_str[:i]], curr_len+1, res)
        return 

sol=Solution()
sol.restoreIpAddresses("25525511135")


"""856. Score of Parentheses
Given a balanced parentheses string s, return the score of the string.
The score of a balanced parentheses string is based on the following rule:

"()" has score 1.
AB has score A + B, where A and B are balanced parentheses strings.
(A) has score 2 * A, where A is a balanced parentheses string.

Example 1:
Input: s = "()"
Output: 1

Example 2:
Input: s = "(())"
Output: 2

Example 3:
Input: s = "()()"
Output: 2
"""

# Solution: recursive
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
                    # "()" cannot be taken care by dfs directly, i never reach end so i+1 is valid
                    res += 1
                    i = i + 2
                else:
                    res += 2 * self.scoreOfSubstring(s, i+1, self.m[i]-1)
                    i = self.m[i] + 1
            else:
                # never happens unless s contains letter
                i += 1
        
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

# stack: if (, push into stack, else keep poping till (. then add curr together
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        curr = 0
        stack = []
        res = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(s[i])
            else:
                curr = 0
                while stack[-1] != '(':
                    # can be integer (prev curr)! 
                    curr += stack.pop()
                stack.pop() # previous "("
                temp_res = 1 if curr == 0 else 2*curr
                if len(stack) == 0: # outside () finished
                    res += temp_res
                else:
                    stack.append(temp_res)
        return res


# idea: only () returns score, other (**) only adds or minus multiplier
# ( () (()) ) -> (()) + ((()))  again, only consider adjacent (), others are multiplier
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        res = 0
        multiplier = 0
        for i in range(len(s)):
            if s[i] == '(':
                multiplier += 1
            if s[i] == ')' and s[i-1] == '(':
                res = res + 2 ** (multiplier - 1)
                multiplier -= 1
            elif s[i] == ')':
                multiplier -= 1

        return res


# Solution 2: use stack to store current score: (C (A))
# When move at ( before A, push C into stack, reset curr=0
# handle (A) and pop the stack to add 
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        curr = 0
        stack = []
        for i in range(len(s)):
            if s[i] == ')':
                # if curr=0 then add by 1 else add by curr * 2
                # keep curr til the next (
                if curr == 0:
                    curr = 1
                else:
                    curr = curr * 2
                # then add into the previous stack (pushed before '(' at same level)
                curr = curr + stack.pop()
            else:
                # if (, then will move into next level, push and reset curr
                stack.append(curr)
                curr = 0
        return curr


# sol 4: recursion 2
# 使用一个计数器，遇到左括号，计数器自增1，反之右括号计数器自减1，那么当计数器为0的时候，就是一个合法的字符串了，
# 我们对除去最外层的括号的中间内容调用递归，然后把返回值乘以2，并和1比较，取二者间的较大值加到结果 res 中
# https://www.youtube.com/watch?v=tiAaVfMcL9w
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        return self.helper(s, 0, len(s))
    
    def helper(self, s, l, r):
        # s[l:r] must be a valid paranthesis, 
        if r-l == 2:
            # ()
            return 1
        cnt = 0
        for i in range(l, r-1):
            # O(n) to identify whether balance parenthesis
            # i < r-1 because the last one must be balanced
            # then = self.helper(s, l, r) + self.helper(s, r, r) : repeat
            if s[i] == '(':
                cnt += 1
            else:
                cnt -= 1
            if cnt == 0:
                # score("(A)(B)(C)") = score("(A)") + score("(B)(C)")
                # s[l:i+1], s[i+1:r]
                return self.helper(s, l, i+1) + self.helper(s, i+1, r)
    
        # else, s[l:r] cannot seperated into valid parenthesis, such as ((()))
        # worst time is O(n^2): for each layer, takes O(N) to identify balance
        return 2 * self.helper(s, l+1, r-1)


sol = Solution()
sol.scoreOfParentheses("(()(()))")


"""301. Remove Invalid Parentheses  !!!
Given a string s that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.

Return all the possible results. You may return the answer in any order.

Example 1:
Input: s = "()())()"
Output: ["(())()","()()()"]

Example 2:
Input: s = "(a)())()"
Output: ["(a())()","(a)()()"]

Example 3:
Input: s = ")("
Output: [""]
"""
# first remove closed parenthesis, then left
# if consecutive, only consider the first one to avoid duplication
# time complexity O(2^(l+r)) where l and r are the number of open/close parentheses need to be removed 
# space com O(l+r)^2 - O(n^2)
class Solution:
    def removeInvalidParentheses(self, s: str):
        # first calculate number of l and r need to remove 
        l, r = 0, 0
        cnt = 0
        for i in range(len(s)):
            if s[i] == '(':
                l += 1
            elif s[i] == ')':
                if l == 0:
                    r += 1
                else:
                    l -= 1
        
        self.res = []
        self.validSet = dict() # record whether a string is valid
        self.helper(s, 0, l, r)
        return self.res

    def helper(self, s, start, l, r):
        if l == 0 and r == 0 and self.is_valid(s):
            self.res.append(s)
            return 
        for i in range(start, len(s)):
            # only remove the first parenthes if there are consecutive ones to avoid duplications.
            if i != start and s[i] == s[i-1]:
                continue
            if s[i] == ')' and r > 0:
                self.helper(s[:i]+s[i+1:], i, l, r-1)
            elif s[i] == '(' and l > 0:
                self.helper(s[:i]+s[i+1:], i, l-1, r)
        
        return None
    
    def is_valid(self, s):
        if s in self.validSet:
            return self.validSet[s]
        # whether s is valid parenthesis 
        cnt = 0
        for i in range(len(s)):
            if s[i] == '(':
                cnt += 1
            elif s[i] == ')':
                cnt -= 1
                if cnt < 0:
                    self.validSet[s] = False
                    return False
        self.validSet[s] = (cnt==0)
        return cnt==0


sol = Solution()
sol.removeInvalidParentheses("(a)())()")

"""Copy a solution: looks like a brutal force but: (326 ms vs 71 ms)
1) use a set to record all word that been tested
2) record the longest revised parenthesis so that shorter ones will not be tested
"""
class Solution2:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def is_valid(w):
            if not w:
                return True
            bal = 0
            for ch in w:
                if ch == '(':
                    bal += 1
                elif ch ==')':
                    bal -= 1
                    if bal < 0:
                        return False
            return bal == 0
        
        def dfs(w):
            nonlocal res, seen, max_so_far
            
            if is_valid(w):
                res.add(w)
                max_so_far = max(max_so_far, len(w))
            else:
                for i,ch in enumerate(w):
                    if ch in '()':
                        word = w[:i] + w[i+1:]
                        if word in seen or len(word) < max_so_far:
                            continue
                        else:
                            seen.add(word)
                            dfs(word)
        
        res = set()
        seen = set()
        max_so_far = 0
        dfs(s)
        maxlen = max(map(len, res))
        return filter(lambda x: len(x) == maxlen, res)


"""37. Sudoku Solver !!!
Write a program to solve a Sudoku puzzle by filling the empty cells.

A sudoku solution must satisfy all of the following rules:

Each of the digits 1-9 must occur exactly once in each row.
Each of the digits 1-9 must occur exactly once in each column.
Each of the digits 1-9 must occur exactly once in each of the 9 3x3 sub-boxes of the grid.
The '.' character indicates empty cells.

Input: board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],
[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],
["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],
[".",".",".",".","8",".",".","7","9"]]
Output: [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],
["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],
["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],
["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],
["3","4","5","2","8","6","1","7","9"]]
"""
board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],
[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],
["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],
[".",".",".",".","8",".",".","7","9"]]

sol = Solution()
sol.solveSudoku(board)

class Solution:
    def solveSudoku(self, board):
        """
        Do not return anything, modify board in-place instead.
        """
        # create three set to record whether a give number is at:
        # given row, col, and small block
        row_set = [set() for _ in range(len(board))]
        col_set = [set() for _ in range(len(board[0]))]
        block_set = [set() for _ in range(len(board))]

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != ".":
                    row_set[i].add(board[i][j])
                    col_set[j].add(board[i][j])
                    block_set[self.block_map(i, j)].add(board[i][j])
        
        res = []
        self.dfs(board, 0, 0, row_set, col_set, block_set, res)
        return res[0]

    def dfs(self, board, i, j, row_set, col_set, block_set, res):
        # return true if solution is found, and add into res
        if i == 9:
            res.append(board)
            return True
        if j == 9:
            return self.dfs(board, i+1, 0, row_set, col_set, block_set, res)
        if board[i][j] != ".":
            return self.dfs(board, i, j+1, row_set, col_set, block_set, res)
        else:
            for num in range(1, 10):
                str_num = str(num)
                if str_num in row_set[i] or str_num in col_set[j] or str_num in block_set[self.block_map(i, j)]:
                    continue
                board[i][j] = str_num
                row_set[i].add(str_num)
                col_set[j].add(str_num)
                block_set[self.block_map(i, j)].add(str_num)
                if self.dfs(board, i, j+1, row_set, col_set, block_set, res):
                    return True
                row_set[i].remove(str_num)
                col_set[j].remove(str_num)
                block_set[self.block_map(i, j)].remove(str_num)
                board[i][j] = "."

        return False

    def block_map(self, i, j):
        # (i,j) -> bloack id
        num = 9*(i//3) + j
        return num // 3


"""79. Word Search !!!
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally 
or vertically neighboring. The same letter cell may not be used more than once.

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
"""

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if word is None:
            return True
        if len(board) == 0 or len(board[0]) == 0:
            return False
        nrows, ncols = len(board), len(board[0])
        visited = [[False for _ in range(ncols)] for _ in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                if self.dfs(board, i, j, word, visited):
                    return True
        return False

    def dfs(self, board, i, j, rem_str, visited):
        if rem_str == "":
            return True
        if 0<=i<len(board) and 0<=j<len(board[0]) and not visited[i][j]:
            if board[i][j] == rem_str[0]:
                visited[i][j] = True
                res = self.dfs(board, i-1, j, rem_str[1:], visited) or \
                    self.dfs(board, i+1, j, rem_str[1:], visited) or \
                        self.dfs(board, i, j-1, rem_str[1:], visited) or \
                            self.dfs(board, i, j+1, rem_str[1:], visited)
                visited[i][j] = False
                return res
                
        return False


"""212. Word Search II
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, 
where adjacent cells are horizontally or vertically neighboring. 
The same letter cell may not be used more than once in a word.

Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Solution: Trie
https://leetcode.com/problems/word-search-ii/discuss/59790/Python-dfs-solution-(directly-use-Trie-implemented).

board = [["o","a","a","n"],
["e","t","a","e"],
["i","h","k","r"],
["i","f","l","v"]]
words = ["oath","pea","eat","rain","oathi","oathk","oathf","oate","oathii","oathfi","oathfii"]

See Trie section below
"""
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = dict()
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_word = True

    # Returns if the word is in the trie.
    def search(self, word: str) -> bool:
        current = self.root
        for letter in word:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return current.is_word
    
    # Returns if there is any word in the trie that starts with the given prefix.
    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for letter in prefix:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return True

class Solution:
    def findWords(self, board, words):
        trie = Trie()
        node = trie.root
        for word in words:
            trie.insert(word)

        visited = [[False for _ in range(len(board[0]))] for _ in range(len(board))]
        self.res = []

        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(i, j, board, node, "", visited)
        
        return self.res
    
    def dfs(self, i, j, board, node, curr_str, visited):
        if node.is_word: #and curr_str not in self.res:
            # curr_str is in trie
            self.res.append(curr_str)
            node.is_word = False  # avoid replicate

        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or visited[i][j]:
            return None
        if board[i][j] not in node.children:
            return None
        next_node = node.children[board[i][j]]
        visited[i][j] = True
        self.dfs(i-1, j, board, next_node, curr_str+board[i][j], visited)
        self.dfs(i+1, j, board, next_node, curr_str+board[i][j], visited)
        self.dfs(i, j-1, board, next_node, curr_str+board[i][j], visited)
        self.dfs(i, j+1, board, next_node, curr_str+board[i][j], visited)
        visited[i][j] = False
        return None

sol = Solution()
sol.findWords(board, words)


""" LeetCode 1087. Brace Expansion
A string S represents a list of words.

Each letter in the word has 1 or more options.  If there is one option, the letter is represented as is. 
 If there is more than one option, then curly braces delimit the options.  
 For example, "{a,b,c}" represents options ["a", "b", "c"].

For example, "{a,b,c}d{e,f}" represents the list ["ade", "adf", "bde", "bdf", "cde", "cdf"].

Return all words that can be formed in this manner, in lexicographical order.

Example 1:

Input: "{a,b}c{d,e}f"
Output: ["acdf","acef","bcdf","bcef"]
Example 2:

Input: "abcd"
Output: ["abcd"]

1 <= S.length <= 50
There are no nested curly brackets.
All characters inside a pair of consecutive opening and ending curly brackets are different
"""
s = "{a,b}c{d,e}f"
s = "abcd"
expand("abcd")
expand("{a,b}c{d,e}f")

def expand(s):
    res = []
    dfs("", s, res)
    return res

def dfs(curr_str, rem_str, res):
    if rem_str == "":
        res.append(curr_str)
        return None
    # find the idx of first { and }
    has_left, has_right = False, False
    for i in range(len(rem_str)):
        if rem_str[i] == '{' and not has_left:
            left_idx = i
            has_left = True
        if rem_str[i] == '}' and not has_right:
            right_idx = i
            has_right = True
        if has_left and has_right:
            break
    
    if not has_left and not has_right:
        # no curl
        dfs(curr_str + rem_str, "", res)
        return None
    str_in_first_brace = rem_str[left_idx+1:right_idx]
    for letter in str_in_first_brace.split(','):
        dfs(curr_str + rem_str[:left_idx] + letter, rem_str[right_idx+1:], res)
    
    return None

# Another solution on list operations
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
sol.expand("{a,b,c}d{e,f}")

""" 399 Evaluate Division
You are given an array of variable pairs equations and an array of real numbers values, 
where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. 
Each Ai or Bi is a string that represents a single variable.

You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query 
where you must find the answer for Cj / Dj = ?.

Return the answers to all queries. If a single answer cannot be determined, return -1.0.

Example 1:
Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
Explanation: 
Given: a / b = 2.0, b / c = 3.0
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
return: [6.0, 0.5, -1.0, 1.0, -1.0 ]

Example 2:
Input: equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
Output: [3.75000,0.40000,5.00000,0.20000]

Example 3:
Input: equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
Output: [0.50000,2.00000,-1.00000,-1.00000]


Hint:
Binary relationship is represented as a graph usually.
Does the direction of an edge matters? -- Yes. Take a / b = 2 for example, it indicates a --2--> b as well as b --1/2--> a.
Thus, it is a directed weighted graph.
In this graph, how do we evaluate division?
Take a / b = 2, b / c = 3, a / c = ? for example,

a --2--> b --3--> c
We simply find a path using DFS from node a to node c and multiply the weights of edges passed, i.e. 2 * 3 = 6.
"""
# weighted graph: 
# {start_node1: {end_node1:weight1, end_node2:weight2}, }
from collections import defaultdict

class Solution:
    def calcEquation(self, equations, values, queries):
        wt_graph = self.build_graph(equations, values)
        res = []
        for query in queries:
            visited = set()
            visited.add(query[0])
            wt_dist = self.get_weight(query[0], query[1], wt_graph, visited, 1)
            res.append(wt_dist)

        return res
    
    def get_weight(self, start, target, wt_graph, visited, curr_dist):
        # return -1 if not achievable
        if start not in wt_graph:
            return -1
        if start == target:
            return curr_dist
        
        for next_num, next_dist in wt_graph[start].items():
            # next_node: next -> weight
            if next_num not in visited:
                visited.add(next_num)
                curr_res = self.get_weight(next_num, target, wt_graph, visited, curr_dist*next_dist)
                if curr_res != -1:
                    return curr_res
                visited.remove(next_num)

        return -1

    def build_graph(self, equations, values):
        wt_graph = defaultdict(dict)
        for i in range(len(equations)):
            start, end = equations[i]
            weight = values[i]
            wt_graph[start].update({end: weight})
            wt_graph[end].update({start: 1/weight})
        return wt_graph


equations = [["a","b"],["b","c"],["bc","cd"]]
values = [1.5,2.5,5.0]
queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
# Output: [3.75000,0.40000,5.00000,0.20000]

sol = Solution()
sol.calcEquation(equations, values, queries)


"""LeetCode 1274. Number of Ships in a Rectangle
On the sea represented by a cartesian plane, each ship is located at an integer point, 
and each integer point may contain at most 1 ship.

You have a function Sea.hasShips(topRight, bottomLeft) which takes two points as 
arguments and returns true if and only if there is at least one ship in the 
rectangle represented by the two points, including on the boundary.

Given two points, which are the top right and bottom left corners of a rectangle, 
return the number of ships present in that rectangle.  It is guaranteed that 
there are at most 10 ships in that rectangle.

Submissions making more than 400 calls to hasShips will be judged Wrong Answer.  
Also, any solutions that attempt to circumvent the judge will result in disqualification.

Input: 
ships = [[1,1],[2,2],[3,3],[5,5]], topRight = [4,4], bottomLeft = [0,0]
Output: 3
Explanation: From [0,0] to [4,4] we can count 3 ships within the range.


Solution: 
If the current rectangle contains ships, subdivide it into 4 smaller ones until
1) no ships contained
2) the current rectangle is a single point (e.g. topRight == bottomRight)

Time complexity: O(logn)
Space complexity: O(logn)
"""
# Sea.hasShips(topRight, bottomLeft)
ships = [[1,1],[2,2],[3,3],[5,5]]
topRight = [4,4]
bottomLeft = [0,0]

def countShips(sea, topRight, bottomLeft):
    cnt = dfs(sea, topRight, bottomLeft)
    return cnt

def dfs(sea, topRight, bottomLeft):
    if not sea.hasShips(topRight, bottomLeft) or topRight[0]<bottomLeft[0] or topRight[1]<bottomLeft[1]:
        return 0
    if topRight[0] == bottomLeft[0] and topRight[1] == bottomLeft[1]:
        return 1
    mid_x = (topRight[0] + bottomLeft[0]) // 2
    mid_y = (topRight[1] + bottomLeft[1]) // 2
    cnt1 = dfs(sea, [mid_x, mid_y], bottomLeft)
    cnt2 = dfs(sea, [mid_x, topRight[1]], [bottomLeft[0], mid_y+1])
    cnt3 = dfs(sea, [topRight[0], mid_y], [mid_x+1, bottomLeft[1]])
    cnt4 = dfs(sea, topRight, [mid_x+1, mid_y+1])
    return cnt1 + cnt2 + cnt3 + cnt4


"""1376. Time Needed to Inform All Employees
A company has n employees with a unique ID for each employee from 0 to n - 1. 
The head of the company is the one with headID.

Each employee has one direct manager given in the manager array where manager[i] 
is the direct manager of the i-th employee, manager[headID] = -1. Also, it is guaranteed that the subordination relationships have a tree structure.

The head of the company wants to inform all the company employees of an urgent 
piece of news. He will inform his direct subordinates, and they will inform their 
subordinates, and so on until all employees know about the urgent news.

The i-th employee needs informTime[i] minutes to inform all of his direct subordinates 
(i.e., After informTime[i] minutes, all his direct subordinates can start spreading the news).

Return the number of minutes needed to inform all the employees about the urgent news.

Input: n = 6, headID = 2, manager = [2,2,-1,2,2,2], informTime = [0,0,1,0,0,0]
Output: 1
Explanation: The head of the company with id = 2 is the direct manager of all the employees in the company and needs 1 minute to inform them all.
The tree structure of the employees in the company is shown.

Solution 1: BFS:
build a graph with key as manager and value as his/her direct report 
start with the headID, use a vsited to store employees informed, use queue to save the current
use a res to update the max distance to headID so far. return res 

Solution 2: DFS
the time takes to inform all employee under manager A is 1 + (the max of time spent for his/her direct report)
"""
from collections import defaultdict
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager, informTime):
        m = defaultdict(list) # manager -> direct report 
        for report_id, manager_id in enumerate(manager):
            m[manager_id].append(report_id)
        visited = set()
        visited.add(headID)
        res = self.dfs(headID, m, visited, informTime)
        return res

    def dfs(self, manager_id, m, visited, informTime):
        if manager_id not in m:
            return 0 # informTime[manager_id] = 0
        res = 0
        for direct_report in m[manager_id]:
            if direct_report not in visited:
                visited.add(direct_report)  # may not required for this qn because no overlap
                res = max(res, informTime[manager_id] + self.dfs(direct_report, m, visited, informTime))
        
        return res


sol = Solution()
sol.numOfMinutes(6, 2, [2,2,-1,2,2,2], [0,0,1,0,0,0])


"""131. Palindrome Partitioning 
Given a string s, partition s such that every substring of the partition is a
 palindrome. Return all possible palindrome partitioning of s.
A palindrome string is a string that reads the same backward as forward.

Example 1:

Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
Example 2:

Input: s = "a"
Output: [["a"]]

Idea: recursive: split s into s[:start_idx], s[stat_idx:i], s[i:]
s[:start_idx] -> previous already explored
s[stat_idx:i] -> Palindrome
s[i:] -> next to recursively explore
"""
class Solution:
    def partition(self, s: str):
        self.res = []
        curr_out = []
        self.dfs(s, 0, curr_out)
        return self.res
    
    def dfs(self, s, start_idx, curr_out):
        if start_idx == len(s):
            self.res.append(curr_out)
            return None
        for i in range(start_idx, len(s)):
            if self.is_Palindrome(s[start_idx:i+1]):
                # s[:start_idx], s[stat_idx:i], s[i:]
                self.dfs(s, i+1, curr_out + [s[start_idx:i+1]])
        return None
        
    def is_Palindrome(self, s):
        l, r = 0, len(s) - 1
        while l<r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

 # DP
class Solution:
    def partition(self, s: str):
        dp = [[] for _ in range(len(s) + 1)]
        dp[0] = [[]]
        for j in range(1, len(s)+1):
            for i in range(j):
                if s[i:j] == s[i:j][::-1]:
                    for each in dp[i]:
                        dp[j].append(each + [s[i:j]])
        return dp[-1]

"""77. Combinations
Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].
You may return the answer in any order.

Example 1:
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
"""
class Solution:
    def combine(self, n: int, k: int):
        nums = list(range(1, n+1))
        res = []
        self.dfs(nums, k, 0, [], res)
        return res
    
    def dfs(self, nums, k, start, out, res):
        if k==0:
            res.append(out)
            return None
        
        for i in range(start, len(nums)):
            self.dfs(nums, k-1, i+1, out+[nums[i]], res)
        return None


sol=Solution()
sol.combine(4, 2)

""" 40. Combination Sum II
Given a collection of candidate numbers (candidates) and a target number (target), 
find all unique combinations in candidates where the candidate numbers sum to target.
Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.
All numbers (including target) will be positive integers.

Example 1:
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[[1,1,6],[1,2,5],[1,7],[2,6]]

Example 2:

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[[1,2,2],[5]]
"""

# THe general format of combination, C(nums, N) -> nums choose N, order not matter:
"""
def dfs(nums, d, N, start, curr, res):
    # better sort nums 
    if d==N:
        res.append(curr)
        return None
    for i in range(s, len(nums)):
        # pick i at THIS LEVEL d, will never choose the same because start of next is i+1
        dfs(nums, d+1, N, i+1, curr + [nums[i]], res)
    return None
"""
class Solution:
    def combinationSum2(self, candidates, target: int):
        candidates.sort()  # avoid necessary calculation, and reduce redundants
        self.res = []  # another way is to use a set to avoid redundancy 
        self.dfs(candidates, target, [], 0)
        return self.res

    def dfs(self, candidates, rem_target, out, start_idx):
        if rem_target == 0:
            self.res.append(out)
            return None
        if rem_target < 0:
            return None
        for i in range(start_idx, len(candidates)):
            if i > start_idx and candidates[i] == candidates[i-1]:
                # already add this number at THIS LEVEL
                continue 
            if candidates[i] > rem_target:
                # prune
                return None
            self.dfs(candidates, rem_target-candidates[i], out + [candidates[i]], i+1)
        
        return None

# time: O(2^N)  worst 
# space: O(len(res) * N)
sol = Solution()
sol.combinationSum2(candidates, target)


"""[LeetCode] Combination Sum III 组合之和之三
Find all possible combinations of k numbers that add up to a number n, given that
 only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

Ensure that numbers within the set are sorted in ascending order.

Example 1:
Input: k = 3, n = 7
Output:
[[1,2,4]]

Example 2:
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
"""
# add a arg depth to represent the number of digits taken 
# omit 
"""
class Solution {
public:
    vector<vector<int> > combinationSum3(int k, int n) {
        vector<vector<int> > res;
        vector<int> out;
        combinationSum3DFS(k, n, 1, out, res);
        return res;
    }
    void combinationSum3DFS(int k, int n, int level, vector<int> &out, vector<vector<int> > &res) {
        if (n < 0) return;
        if (n == 0 && out.size() == k) res.push_back(out);
        for (int i = level; i <= 9; ++i) {
            out.push_back(i);
            combinationSum3DFS(k, n - i, i + 1, out, res);
            out.pop_back();
        }
    }
};
"""

""" 90. Subsets II
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

Example 2:
Input: nums = [0]
Output: [[],[0]]

# special case of Combination but not restriction, i.e., return all comb
"""
class Solution:
    def subsetsWithDup(self, nums):
        res = []
        nums.sort()  # necessary to remove deplicate
        self.dfs(nums, 0, [], res)
        return res

    def dfs(self, nums, start_idx, out, res):
        res.append(out)  # no restriction
        for i in range(start_idx, len(nums)):
            if i>start_idx and nums[i] == nums[i-1]:
                continue
            self.dfs(nums, i+1, out + [nums[i]], res)
        return None

sol=Solution()
sol.subsetsWithDup( [1,2,2])
sol.subsetsWithDup( [1,2,3])


"""47. Permutations II !!!
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

Example 1:
Input: nums = [1,1,2]
Output:
[[1,1,2],
 [1,2,1],
 [2,1,1]]

Example 2:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

Constraints:
1 <= nums.length <= 8
-10 <= nums[i] <= 10

Question is how to de-duplicate
"""

# THe general format of permutation, P(nums, N) -> nums choose N, order matters:
"""
def dfs(nums, d, N, visited, curr, res):
    if d==N:
        res.append(curr)
        return None
    for i in range(0, len(nums)):
        if visited[i] is False:
            visited[i] = True
            dfs(nums, d+1, N, visited, curr+[nums[i]], res)
            visited[i] = False
    
    return None

"""
nums = [1,1,2]

class Solution:
    def permuteUnique(self, nums):
        res = []
        visited = [0 for _ in range(len(nums))]
        nums.sort()
        self.dfs(nums, [], res, visited)
        return res
    
    def dfs(self, nums, out, res, visited):
        if len(out) == len(nums):
            res.append(out)
            return None
        
        for i in range(0, len(nums)):
            if i > 0 and nums[i] == nums[i-1] and visited[i-1] == 0:
                # when a number has the same value with its previous, 
                # we can use this number only if his previous is used
                continue
            if visited[i] == 0:
                visited[i] = 1
                self.dfs(nums, out+[nums[i]], res, visited)
                visited[i] = 0
        
        return None

sol = Solution()
sol.permuteUnique([1,3,2])
sol.permuteUnique([1,1,2])


"""698. Partition to K Equal Sum Subsets 分割K个等和的子集 
Given an array of integers nums and a positive integer k, find whether it's 
possible to divide this array into knon-empty subsets whose sums are all equal.

Example 1:
Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
Output: True
Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.

Note:
•1 <= k <= len(nums) <= 16.
•0 < nums[i] < 10000.

Idea: it is like combination, but record all that sum up to target
"""
# TLE
# class Solution:
#     def canPartitionKSubsets(self, nums, k: int):
#         self.raw_target = sum(nums) / k
#         if self.raw_target != int(self.raw_target):
#             return False
#         nums.sort(reverse=True)
#         visited = [0 for _ in range(len(nums))]
#         return self.dfs(nums, k, self.raw_target, 0, visited)
    
#     def dfs(self, nums, k, target, start_idx, visited):
#         if k == 0:
#             return True

#         for i in range(start_idx, len(nums)):
#             if visited[i] == 1:
#                 continue
#             if nums[i] == target:
#                 visited[i] = 1
#                 if self.dfs(nums, k-1, self.raw_target, 0, visited):
#                     return True
#                 visited[i] = 0
#             elif nums[i] < target:
#                 visited[i] = 1
#                 if self.dfs(nums, k, target - nums[i], i+1, visited):
#                     return True
#                 visited[i] = 0
#             else:
#                 continue
#         return False
class Solution:
    def canPartitionKSubsets(self, nums, k: int):
        self.raw_target = sum(nums) / k
        if self.raw_target != int(self.raw_target):
            return False
        nums.sort(reverse=True)
        visited = ['0' for _ in range(len(nums))]
        memo = dict()
        def dfs(k, target, start_idx, memo):
            if k == 1: # if k-1 targets, then the last one must be sum to target(target=sums/k)
                return True
            visited_str = ''.join(visited)
            if visited_str in memo:
                return memo[visited_str]
            for i in range(start_idx, len(nums)):
                if visited[i] == '1':
                    continue
                if nums[i] == target:
                    visited[i] = '1'
                    res = dfs(k-1, self.raw_target, 0, memo)
                    if res:
                        memo[''.join(visited)] = True
                        return True
                    visited[i] = '0'
                elif nums[i] < target:
                    visited[i] = '1'
                    res = dfs(k, target - nums[i], i+1, memo)
                    if res:
                        memo[''.join(visited)] = True
                        return True
                    visited[i] = '0'
                else:
                    continue
            memo[''.join(visited)] = False
            return False
        return dfs(k, self.raw_target, 0, memo)

sol = Solution()
sol.canPartitionKSubsets([4, 3, 2, 3, 5, 2, 1], 4)
sol.canPartitionKSubsets([4, 1, 5], 2)
sol.canPartitionKSubsets([724,3908,1444,522,325,322,1037,5508,1112,724,424,2017,1227,6655,5576,543], 4)
sol.canPartitionKSubsets([2,9,4,7,3,2,10,5,3,6,6,2,7,5,2,4], 7)
sol.canPartitionKSubsets([2,6,16,13,3,4,1,1,2,12], 3)
sol.canPartitionKSubsets([2,9,4,7,3,2,10,5,3,6,6,2,7,5,2,4], 7)
sol.canPartitionKSubsets([4,5,9,3,10,2,10,7,10,8,5,9,4,6,4,9], 5)


class Solution:
    def canPartitionKSubsets(self, nums, k: int):
        self.raw_target = sum(nums) / k
        if self.raw_target != int(self.raw_target):
            return False
        nums.sort(reverse=True)
        visited = [0 for _ in range(len(nums))]
        v_str = ''.join([str(i) for i in visited])
        # visit_set = set()
        
        @lru_cache(None)
        def dfs(k, target, start_idx, v_str):
            if k == 0:
                return True
            for i in range(start_idx, len(nums)):
                if visited[i] == 1:
                    continue
                if nums[i] == target:
                    visited[i] = 1
                    if dfs(k-1, self.raw_target, 0, ''.join([str(i) for i in visited])):
                        return True
                    visited[i] = 0
                elif nums[i] < target:
                    visited[i] = 1
                    if dfs(k, target - nums[i], i+1, ''.join([str(i) for i in visited])):
                        return True
                    visited[i] = 0
                else:
                    continue
            return False
        return dfs(k, self.raw_target, 0, ''.join([str(i) for i in visited]))


"""526. Beautiful Arrangement
Suppose you have n integers labeled 1 through n. A permutation of those n integers perm 
(1-indexed) is considered a beautiful arrangement if for every i (1 <= i <= n), either of the following is true:

perm[i] is divisible by i.
i is divisible by perm[i].

Given an integer n, return the number of the beautiful arrangements that you can construct.

Example 1:
Input: n = 2
Output: 2
Explanation: 
The first beautiful arrangement is [1,2]:
    - perm[1] = 1 is divisible by i = 1
    - perm[2] = 2 is divisible by i = 2
The second beautiful arrangement is [2,1]:
    - perm[1] = 2 is divisible by i = 1
    - i = 2 is divisible by perm[2] = 1
"""
# Solution: same as permutation but add the beatiful arrangment when adding new element into current out
class Solution:
    def countArrangement(self, n: int):
        res = []
        visited = [0 for _ in range(n)]
        nums = list(range(1, n+1))
        self.dfs(nums, 1, [], res, visited)
        return len(res)
    
    def dfs(self, nums, pos, out, res, visited):
        # pos: location index of current input 
        if pos == len(nums)+1:
            res.append(out)
            return None
        for i in range(len(nums)):
            if visited[i] == 1:
                continue
            if nums[i] % pos != 0 and pos % nums[i] != 0:
                # def of beatiful
                continue
            visited[i] = 1
            self.dfs(nums, pos+1, out+[nums[i]], res, visited)
            visited[i] = 0
        return None

sol=Solution()
sol.countArrangement(2)
sol.countArrangement(4)


"""140. Word Break II
Given a string s and a dictionary of strings wordDict, add spaces in s to construct 
a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Example 1:
Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]

Example 2:
Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
Explanation: Note that you are allowed to reuse a dictionary word.

Example 3:
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: []

Solution: 
dfs(catsanddog) = wordSet[cat] + dfs[sanddog] or wordSet[cats] + dfs[anddog]

use a dict m to help, for any substr, take "catsanddog" as an example
m["catsanddog"] = ["cats and dog","cat sand dog"]

"""
from collections import defaultdict

class Solution:
    def wordBreak(self, s: str, wordDict):
        m = defaultdict(list)  # 
        max_len = max([len(word) for word in wordDict])
        wordSet = set(wordDict)
        return self.dfs(s, m, wordSet, max_len)

    def dfs(self, rem_str, m, wordSet, max_len):
        # question: what to return if rem_str = ""
        # option 1: avoid that by adding if rem_str in wordSet -> curr_res.append(rem_str) (below)
        # option 2: if "" return [""] but need to handle :
        #   rem_str[:i+1] + " " + rem_item when rem_item="", because extra space is in the end
        if rem_str in m:
            # avoid duplicated calculation
            return m[rem_str]
        curr_res = []
        if rem_str in wordSet:
            # if rem_str in dict, add it to the answer array
            curr_res.append(rem_str)
        
        # further split rem_str into word + m[rest]
        for i in range(max_len):
            if rem_str[:i+1] in wordSet:
                rem_list = self.dfs(rem_str[i+1:], m, wordSet, max_len)
                # ["cats and dog","cat sand dog"]
                for rem_item in rem_list:
                    # 
                    curr_res.append(rem_str[:i+1] + " " + rem_item)
        
        m[rem_str] = curr_res
        return m[rem_str]


wordDict = ["cat","cats","and","sand","dog"]
sol = Solution()
sol.wordBreak("catsanddog", ["cat","cats","and","sand","dog"]) 
sol.wordBreak("catsand", ["cat","cats","and","sand","dog"]) 

# Another format: no return value for fcn
from functools import lru_cache
class Solution:
    def wordBreak(self, s: str, wordDict):
        self.wordSet = set(wordDict)
        self.max_len = max([len(word) for word in wordDict])
        self.res = []
        self.dfs(s, 0, "")
        return [i.strip() for i in self.res]
    
    @lru_cache(None)  # may not be useful?
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

# DP
class Solution:
    def wordBreak(self, s: str, wordDict) -> bool:
        wordSet = set(wordDict)
        dp = [False] * len(s) # s[:i+1] 
        res = [[] for _ in range(len(s))] # s[:i+1] 
        for i in range(len(s)):
            if s[:i+1] in wordSet:
                dp[i] = True
                res[i].append(s[:i+1])
            for j in range(i):
                if s[j+1:i+1] in wordSet and dp[j]:
                    dp[i] = True
                    res[i] += [stri + " " + s[j+1:i+1] for stri in res[j]]
        return res[-1]

sol=Solution()
sol.wordBreak("catsanddog", ["cat","cats","and","sand","dog"])


"""72. Edit Distance
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character

Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:

Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

# Solution 1
比较的时候，要尝试三种操作，因为谁也不知道当前的操作会对后面产生什么样的影响。
对于当前比较的两个字符 word1[i] 和 word2[j]，若二者相同，一切好说，直接跳到下一个位置。
若不相同，有三种处理方法，
首先是直接插入一个 word2[j]，那么 word2[j] 位置的字符就跳过了，接着比较 word1[i] 和 word2[j+1] 即可。
第二个种方法是删除，即将 word1[i] 字符直接删掉，接着比较 word1[i+1] 和 word2[j] 即可。
第三种则是将 word1[i] 修改为 word2[j]，接着比较 word1[i+1] 和 word[j+1] 即可。

分析到这里，就可以直接写出递归的代码，但是很可惜会 Time Limited Exceed，所以必须要优化时间复杂度，
需要去掉大量的重复计算，这里使用记忆数组 memo 来保存计算过的状态，从而可以通过 OJ，
"""
class Solution_DFS:
    def minDistance(self, word1: str, word2: str):
        memo = [[0 for _ in range(len(word2))] for _ in range(len(word1))]
        # memo[i1][i2] steps needed to make word1[i1:] and word2[i2:] the same
        return self.dfs(0, 0, word1, word2, memo)

    def dfs(self, i1, i2, word1, word2, memo):
        if i1 == len(word1):
            # delete the rest of word2
            return len(word2) - i2
        if i2 == len(word2):
            return len(word1) - i1
        if memo[i1][i2] > 0:
            return memo[i1][i2]
        
        if word1[i1] == word2[i2]:
            return self.dfs(i1+1, i2+1, word1, word2, memo)
        
        insertCnt = self.dfs(i1, i2+1, word1, word2,memo)
        deleteCnt = self.dfs(i1+1, i2, word1, word2, memo)
        replaceCnt = self.dfs(i1+1, i2+1, word1, word2, memo)

        res = min(insertCnt, min(deleteCnt, replaceCnt)) + 1
        memo[i1][i2] = res
        return memo[i1][i2]


word1 = "intention"; word2 = "execution"
word1="horse"; word2="ros"
sol = Solution_DFS()
sol.minDistance(word1, word2)

""" Solution 2: DP
dp[i][j] 表示从 word1 的前i个字符转换到 word2 的前j个字符所需要的步骤
先给这个二维数组 dp 的第一行第一列赋值，这个很简单，因为第一行和第一列对应的总有一个字符串是空串，于是转换步骤完全是另一个字符串的长度
  Ø a b c d
Ø 0 1 2 3 4
b 1 1 1 2 3
b 2 2 1 2 3
c 3 3 2 1 2
通过观察可以发现，当 word1[i] == word2[j] 时，dp[i][j] = dp[i - 1][j - 1]，
其他情况时，dp[i][j] 是其左，左上，上的三个值中的最小值加1，那么可以得到状态转移方程为

dp[i][j] =      /    dp[i - 1][j - 1]    if word1[i - 1] == word2[j - 1]

                \    min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1        else
"""
class Solution_DP:
    def minDistance(self, word1: str, word2: str):
        if word1 == "":
            return len(word2)
        if word2 == "":
            return len(word1)
        dp = [[0 for _ in range(len(word2)+1)] for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1): 
            for j in range(1, len(word2)+1): 
                if word1[i-1] == word2[j - 1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1

        return dp[len(word1)][len(word2)]
    

sol = Solution_DP()
sol.minDistance(word1, word2)
sol.minDistance(word1="a", word2="b")


"""377. Combination Sum IV
Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.
The test cases are generated so that the answer can fit in a 32-bit integer.

1 <= nums[i] <= 1000

Example 1:

Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.
Example 2:

Input: nums = [9], target = 3
Output: 0

nums=[4,1,2], target=32
output: 39882198
"""
#DFS without saving the intermediate results will TLE
# may use DP as well

""" DP 
需要一个一维数组 dp，其中 dp[i] 表示目标数为i的解的个数，然后从1遍历到 target，对于每一个数i，
遍历 nums 数组，如果 i>=x, dp[i] += dp[i - x]。这个也很好理解，比如说对于 [1,2,3] 4，
这个例子，当计算 dp[3] 的时候，3可以拆分为 1+x，而x即为 dp[2]，3也可以拆分为 2+x，
此时x为 dp[1]，3同样可以拆为 3+x，此时x为 dp[0]
"""
class Solution:
    def combinationSum4(self, nums, target: int) -> int:
        dp = [0 for _ in range(target+1)]  # start with 0, to target
        dp[0] = 1
        nums.sort()  # sort so can break when num is too large
        for i in range(1, len(dp)):
            # how many comb add to i
            for num in nums:
                if num <= i:
                    dp[i] = dp[i] + dp[i-num]
                if num > i:
                    # no need to continue
                    break
        
        return dp[-1]
        
sol=Solution()
sol.combinationSum4([4,1,2], 32)

nums=[4,1,2]
target=32


"""DFS: need to save intermediate results
"""
class Solution:
    def combinationSum4(self, nums, target: int) -> int:
        memo = defaultdict(int)  # target -> combs add to target
        memo[0] = 1
        return self.dfs(nums, target, memo)
    
    def dfs(self, nums, target, memo):
        # return combs add to target through memo
        if target in memo:
            return memo[target]
        if target < 0:
            # positive nums
            return 0
        res = 0
        for num in nums:
            # because can use multiple times
            res = res + self.dfs(nums, target-num, memo)
        memo[target] = res
        return memo[target]


"""1235. Maximum Profit in Job Scheduling
We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays, return the maximum profit you can take 
such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time X you will be able to start another job that starts at time X.

Input: startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
Output: 120
Explanation: The subset chosen is the first and fourth job. 
Time range [1-3]+[3-6] , we get profit of 120 = 50 + 70.

Input: startTime = [1,2,3,4,6], endTime = [3,5,10,6,9], profit = [20,20,100,70,60]
Output: 150
Explanation: The subset chosen is the first, fourth and fifth job. 
Profit obtained 150 = 20 + 70 + 60.


Solution: DP
Sort jobs by ending time.
dp[t] := max profit by end time t.

for a job = (s, e, p)
dp[e] = dp[u] + p, u <= s, and if dp[u] + p > last_element in dp.
If we do this job, binary search in the dp to find the largest profit we can make before start time s.

Time complexity: O(nlogn)
Space complexity: O(n)

# https://leetcode.com/problems/maximum-profit-in-job-scheduling/discuss/409009/JavaC%2B%2BPython-DP-Solution
"""
startTime = [1,2,3,4,6]; endTime = [3,5,10,6,9]; profit = [20,20,100,70,60]
sol = Solution()
sol.jobScheduling(startTime, endTime, profit)

class Solution:
    def jobScheduling(self, startTime, endTime, profit) -> int:
        # sort jobs by end time
        sorted_jobs = sorted(zip(startTime, endTime, profit), key= lambda x:x[1])
        # create dp
        dp = []
        dp.append([0, 0])  # time, max profit by time
        for job in sorted_jobs:
            curr_start, curr_end, curr_profit = job
            # find the last dp[t0, p0] that t0 <= start of curr job
            prev_dp = self.find_early_or_equal(dp, curr_start)
            if curr_profit + prev_dp[1] > dp[-1][1]:
                # dp[-1][1] was the best profit by curr_end
                dp.append([curr_end, curr_profit + prev_dp[1]])
        return dp[-1][1]
    
    def find_early_or_equal(self, dp, start_time):
        # find the last dp[t0, p0] that t0 <= start of curr job
        l, r = 0, len(dp)
        while l < r:
            mid = l + (r-l)//2
            if dp[mid][0] > start_time:
                r = mid
            else:
                l = mid + 1
        return dp[l-1]


# Almost TLE ... 
startTime = [1,2,3,4,6]; endTime = [3,5,10,6,9]; profit = [20,20,100,70,60]
sol = Solution()
sol.jobScheduling(startTime, endTime, profit)
import bisect 
class Solution2:
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
            # prev_i = self.find_previous_endtime(starti, sorted_endTime[:i])
            prev_i = bisect.bisect(sorted_endTime[:i], starti) - 1  # faster
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

"""1335. Minimum Difficulty of a Job Schedule  !!!
You want to schedule a list of jobs in d days. Jobs are dependent 
(i.e To work on the ith job, you have to finish all the jobs j where 0 <= j < i).

You have to finish at least one task every day. 
The difficulty of a job schedule is the sum of difficulties of each day of the d days. 
The difficulty of a day is the maximum difficulty of a job done on that day.

You are given an integer array jobDifficulty and an integer d. The difficulty of the ith job is jobDifficulty[i].

Return the minimum difficulty of a job schedule. If you cannot find a schedule for the jobs return -1.

Example 1
Input: jobDifficulty = [6,5,4,3,2,1], d = 2
Output: 7
Explanation: First day you can finish the first 5 jobs, total difficulty = 6.
Second day you can finish the last job, total difficulty = 1.
The difficulty of the schedule = 6 + 1 = 7 

Example 2:
Input: jobDifficulty = [9,9,9], d = 4
Output: -1
Explanation: If you finish a job per day you will still have a free day. you cannot find a schedule for the given jobs.

Example 3:
Input: jobDifficulty = [1,1,1], d = 3
Output: 3
Explanation: The schedule is one job per day. total difficulty will be 3.

1 <= jobDifficulty.length <= 300
0 <= jobDifficulty[i] <= 1000
1 <= d <= 10

# DP: 
dp[i][k] the min difficulty to schedule the first i jobs in k days
j jobs in first k-1 days and i-j in last day
dp[i][k] = min(dp[j][k-1] + max(jobDifficulty[j+1:i])) for k-1<=j<=i-1

Time complexity: O(n^2 * k)
space complexity: O(n * k)
"""
class Solution:
    def minDifficulty(self, jobDifficulty, d: int) -> int:
        n = len(jobDifficulty)
        dp = [[float('Inf') for _ in range(d+1)] for _ in range(n+1)]  # dp[0][0] = 0
        dp[0][0] = 0
        for i in range(1, n+1):
            for k in range(1, d+1):
                max_difficult = 0
                for j in list(range(k-1, i))[::-1]:
                    max_difficult = max(max_difficult, jobDifficulty[j])  # max(jobs[j+1:i]), cannot take i
                    dp[i][k] = min(dp[i][k], dp[j][k-1] + max_difficult)

        return dp[n][d] if dp[n][d]<float('Inf') else -1

sol = Solution()
sol.minDifficulty(jobDifficulty=[6,5,4,3,2,1], d=2)

# DFS + memo
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


""" LeetCode 1216. Valid Palindrome III
Given a string s and an integer k, find out if the given string is a K-Palindrome or not.
A string is K-Palindrome if it can be transformed into a palindrome by removing at most k characters from it.

Example 1:
Input: s = "abcdeca", k = 2
Output: true
Explanation: Remove 'b' and 'e' characters.

Constraints:
1 <= s.length <= 1000
s has only lowercase English letters.
1 <= k <= s.length

Time Complexity: O(n^2). n = s.length().
Space: O(n^2).
"""
# dfs + memo
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
sol.isValidPalindrome("abcdadecbaa", 2) # true

# dp[i][j] = min k needed to remove, to make s[i:j+1] palin
def isValidPalindrome(s, k):
    if len(s) == 0:
        return True
    n = len(s)
    dp = [[float('inf') for _ in range(len(s))] for _ in range(len(s))]
    for i in range(n):
        for j in range(n):
            if i >= j:
                dp[i][j] = 0
    for i in list(range(n))[::-1]:
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i+1][j], dp[i][j-1])
    
    return  dp[0][n-1] <= k

# dp[i][j]:  max length to make s[i:j+1] palindrome, after removing elements
# i.e., check the difference between s and longest palindrome length, if it is <= k, then return true.
def isValidPalindrome(s, k):
    if len(s) == 0:
        return True
    n = len(s)
    dp = [[1 for _ in range(len(s))] for _ in range(len(s))]  # dp[i][i] = 1
    # dp[i][j] = dp[i+1][j-1] + 2 if equal, or max(dp[i+1][j], dp[i][j-1])
    for i in list(range(n))[::-1]:
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2 if j > i + 1 else 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    
    num_to_delete = len(s) - dp[0][n-1]
    return num_to_delete <= k


"""97. Interleaving String 交织相错的字符串
Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.
An interleaving of two strings s and t is a configuration where they are divided into non-empty substrings such that:
s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
Note: a + b is the concatenation of strings a and b.

Constraints:
0 <= s1.length, s2.length <= 100
0 <= s3.length <= 200
s1, s2, and s3 consist of lowercase English letters.

Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true

Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false

Follow up: Could you solve it using only O(s2.length) additional memory space?

DP: whether s3[0:i+j] can be formed by interleaving s1[0:i] and s2[0:j].
"""
sol=Solution()
sol.isInterleave("aabcc", "dbbca", "aadbbcbcac")
sol.isInterleave("aabcc", "dbbca", "aadbbbaccc")
sol.isInterleave("aaaa", "aa", "aaa")
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        return self.isInterleave_dfs(s1, s2, s3, 0, 0, 0)

    @lru_cache(None)
    def isInterleave_dfs(self, s1, s2, s3, i, j, k):
        if i == len(s1):
            return s2[j:] == s3[k:]
        if j == len(s2):
            return s1[i:] == s3[k:]
        if s1[i] == s3[k]:
            if self.isInterleave_dfs(s1, s2, s3, i+1, j, k+1):
                return True
        if s2[j] == s3[k]:
            if self.isInterleave_dfs(s1, s2, s3, i, j+1, k+1):
                return True
        return False

#
# DP
#
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s3) != len(s1) + len(s2):
            return False
        dp = [[False for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
        dp[0][0] = True
        for i in range(0, len(s1)+1):
            # need including 0 to initialize row 0 and col 0
            for j in range(0, len(s2)+1):
                # whether s3[0:i+j] can be formed by interleaving s1[0:i] and s2[0:j].
                if i > 0:
                    dp[i][j] = dp[i][j] or (dp[i-1][j] and s1[i-1] == s3[i+j-1])  # i,j starts with 1
                if j > 0:
                    dp[i][j] = dp[i][j] or (dp[i][j-1] and s2[j-1] == s3[i+j-1])  # i,j starts with 1

        return dp[len(s1)][len(s2)]


"""472. Concatenated Words
Given an array of strings words (without duplicates), return all the concatenated words in the given list of words.
A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.

Example 1:

Input: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats"; 
"dogcatsdog" can be concatenated by "dog", "cats" and "dog"; 
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".

Example 2:
Input: words = ["cat","dog","catdog"]
Output: ["catdog"]

Solution 1: DFS with memorization
"""
from collections import defaultdict
class Solution:
    def findAllConcatenatedWordsInADict(self, words):
        memo = defaultdict(list)
        wordDict = set(words)
        res = []
        for word in words:
            if self.dfs(word, wordDict, memo):
                res.append(word)

        return res
    
    def dfs(self, word, wordDict, memo):
        if word in memo:
            return memo[word]
        curr_res = False
        for i in range(1, len(word)):
            # note i!=0, o.w. always be true because word itself in wordDict
            prefix = word[:i]  # again, cannot be word[:n]
            suffix = word[i:]
            if prefix in wordDict:
                if suffix in wordDict or self.dfs(suffix, wordDict, memo):
                    curr_res = True
        
        memo[word] = curr_res
        return memo[word]


sol=Solution()
sol.findAllConcatenatedWordsInADict(["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"])
"""Solution 2 : DP
思路是：对于words中的每个单词w，我们定义一个数组dp[n+1]，如果dp[i] == true，
则表示w.substr(0, i)可以由words中的已有单词连接而成。那么状态转移方程就是
dp[i] = || {dp[j] && w[j:i] is in words}，
其中j < i。最终检查dp[n]是否为true，如果是则将其加入结果集中。为了加速对words中的单词的查找，
我们用一个哈希表来保存各个单词。这样时间复杂度可以降低到O(n * m^2)，其中n是words中的单词的个数，m是每个单词的平均长度（或者最大长度？）
"""
class Solution:
    def findAllConcatenatedWordsInADict(self, words):
        wordDict = set(words)
        res = []
        for word in words:
            if len(word) == 0:
                continue
            wordDict.remove(word)  # remove itself
            dp = [False for _ in range(len(word)+1)]
            dp[0] = True
            for i in range(1, len(word)+1):
                # i: 1 - n
                for j in range(i):
                    # j : 0 - i-1
                    dp[i] = dp[j] & (word[j:i] in wordDict)
                    if dp[i]:
                        break
            if dp[-1]:
                res.append(word)
            wordDict.add(word)
        
        return res


"""403. Frog Jump
A frog is crossing a river. The river is divided into some number of units, and at each unit, 
there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of stones' positions (in units) in sorted ascending order, 
determine if the frog can cross the river by landing on the last stone. 
Initially, the frog is on the first stone and assumes the first jump must be 1 unit.

If the frog's last jump was k units, its next jump must be either k - 1, k, or k + 1 units. 
The frog can only jump in the forward direction.

Example 1:
Input: stones = [0,1,3,5,6,8,12,17]
Output: true
Explanation: The frog can jump to the last stone by jumping 1 unit to the 2nd stone, 
then 2 units to the 3rd stone, then 2 units to the 4th stone, then 3 units to the 6th stone, 
4 units to the 7th stone, and 5 units to the 8th stone.

Example 2:
Input: stones = [0,1,2,3,4,8,9,11]
Output: false
Explanation: There is no way to jump to the last stone as the gap between the 5th and 6th stone is too large.

Constraints:
2 <= stones.length <= 2000
0 <= stones[i] <= 231 - 1
stones[0] == 0
stones is sorted in a strictly increasing order.
"""

"""Solution 1: DFS with memo"""
from collections import defaultdict

class Solution:
    def canCross(self, stones) -> bool:
        if len(stones) == 1:
            return True
        if len(stones) == 2:
            return stones[1] == 1
        if stones[1] != 1:
            return False
        memo = defaultdict(list)
        return self.dfs(stones, 1, 1, memo)  # first jump is fixed as 1
        # return self.dfs(stones, 0, 0, memo)  # can remove previous checks
    
    def dfs(self, stones, start_idx, jump_step, memo):
        if (start_idx, jump_step) in memo:
            return memo[(start_idx, jump_step)]
        
        curr_res = False
        for i in range(start_idx+1, len(stones)):
            # will never jump 0 if jump_step is 1
            curr_jump = stones[i] - stones[start_idx]
            if curr_jump < jump_step - 1:
                continue
            if curr_jump > jump_step + 1:
                break
            if i == len(stones) - 1:
                curr_res = True
                break
            if self.dfs(stones, i, curr_jump, memo):
                curr_res = True
                break
        
        memo[(start_idx, jump_step)] = curr_res
        return curr_res


sol = Solution()
sol.canCross([0,1,3,5,6,8,12,17])
sol.canCross([0,1,2,3,4,8,9,11])
sol.canCross([0,2,4,5,6,8,9,11,14,17,18,19,20,22,23,24,25,27,30])

# SOlution 2: DP
# 其中 dp[i] 表示在位置为i的石头青蛙的弹跳力(只有青蛙能跳到该石头上，dp[i] 才大于0)
"""
Idea 1:
            
index:        0   1   2   3   4   5   6   7 
            +---+---+---+---+---+---+---+---+
stone pos:  | 0 | 1 | 3 | 5 | 6 | 8 | 12| 17|
            +---+---+---+---+---+---+---+---+
k:          | 1 | 0 | 1 | 1 | 0 | 1 | 3 | 5 |
            |   | 1 | 2 | 2 | 1 | 2 | 4 | 6 |
            |   | 2 | 3 | 3 | 2 | 3 | 5 | 7 |
            |   |   |   |   | 3 | 4 |   |   |
            |   |   |   |   | 4 |   |   |   |
            |   |   |   |   |   |   |   |   |

for any j < i,
dist = stones[i] - stones[j];
if dist is in dp(j):
    put dist - 1, dist, dist + 1 into dp(i). 

More efficient if using a hashmap to sve k's: stone_idex -> jumps

Idea 2:
// Sub-problem and state:
let dp[i][j] denote at stone i, the frog can or cannot make jump of size j

index:        0   1   2   3   4   5   6   7 
            +---+---+---+---+---+---+---+---+
stone pos:  | 0 | 1 | 3 | 5 | 6 | 8 | 12| 17|
            +---+---+---+---+---+---+---+---+
k:        0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
          1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
          2 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
          3 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
          4 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
          5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
          6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
          7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
// Recurrence relation:
for any j < i,
dist = stones[i] - stones[j];
if dp[j][dist]:
    dp[i][dist - 1] = ture
    dp[i][dist] = ture
    dp[i][dist + 1] = ture
"""
class Solution:
    def canCross(self, stones) -> bool:
        m = defaultdict(list)
        dp = [0 for _ in range(len(stones))]
        dp[0] = 1
        m[0].append(0)  # not 1, because first jump fix to be 1!

        for i in range(1, len(dp)):
            for j in range(i):
                curr_dist = stones[i] - stones[j]
                if curr_dist in m[j] or curr_dist-1 in m[j] or curr_dist+1 in m[j]:
                    # can jump from j to i with curr_dist
                    m[i].append(curr_dist)
                    dp[i] = 1
        
        return dp[-1] == 1

sol = Solution()
sol.canCross([0,1,3,5,6,8,12,17])
# one way to improve its speed, is to use dp[i] to save the largest jump, so can quickly decide
# whether j is too far: 
"""
class Solution {
public:
    bool canCross(vector<int>& stones) {
        unordered_map<int, unordered_set<int>> m;
        vector<int> dp(stones.size(), 0);
        m[0].insert(0);
        int k = 0;
        for (int i = 1; i < stones.size(); ++i) {
            while (dp[k] + 1 < stones[i] - stones[k]) ++k;
            for (int j = k; j < i; ++j) {
                int t = stones[i] - stones[j];
                if (m[j].count(t - 1) || m[j].count(t) || m[j].count(t + 1)) {
                    m[i].insert(t);
                    dp[i] = max(dp[i], t);
                }
            }
        }
        return dp.back() > 0;
    }
};
"""


"""329. Longest Increasing Path in a Matrix
Given an m x n integers matrix, return the length of the longest increasing path in matrix.

From each cell, you can either move in four directions: left, right, up, or down. 
You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

Input: matrix = [[9,9,4],[6,6,8],[2,1,1]]
Output: 4
Explanation: The longest increasing path is [1, 2, 6, 9].

Input: matrix = [[3,4,5],[3,2,6],[2,2,1]]
Output: 4
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.

Input: matrix = [[1]]
Output: 1

0 <= matrix[i][j] <= 231 - 1
"""
# DFS: 
# 用 DP 的原因是为了提高效率，避免重复运算。这里需要维护一个二维动态数组dp，
# 其中 dp[i][j] 表示数组中以 (i,j) 为起点的最长递增路径的长度，初始将 dp 数组都赋为0，
# 当用递归调用时，遇到某个位置 (x, y), 如果 dp[x][y] 不为0的话，直接返回 dp[x][y] 即可，
class Solution:
    def longestIncreasingPath(self, matrix) -> int:
        dp = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        res = 1
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                res = max(res, self.LIP_from_pt(matrix, i, j, dp))
        
        return res

    def LIP_from_pt(self, matrix, i, j, dp):
        if dp[i][j] > 0:
            return dp[i][j]
        curr_mx = 1
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_i = di+i
            next_j = dj+j
            if 0<=next_i<len(matrix) and 0<=next_j<len(matrix[0]) and matrix[next_i][next_j] > matrix[i][j]:
                curr_mx = max(curr_mx, 1+self.LIP_from_pt(matrix, next_i, next_j, dp))
        
        dp[i][j] = curr_mx
        return curr_mx


sol=Solution()
sol.dfs(matrix, 0, 0, dp)
sol.longestIncreasingPath(matrix)
# BFS: 
# Every element in dp can be updated for several times but not as initial point
# 需要优化的是，如果当前点的 dp 值大于0了，说明当前点已经计算过了
# will be faster if we sort the matrix, and visit from largest to smallest
from collections import deque
class Solution_BFS:
    # much slower than DFS+memorization
    def longestIncreasingPath(self, matrix) -> int:
        q = deque()
        dp = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        res = 1
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if dp[i][j] == 0:
                    q.append((i, j, 1))
                    while len(q) > 0:
                        curr_i, curr_j, curr_mx = q.popleft()
                        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            next_i, next_j = curr_i+di, curr_j+dj
                            if 0<=next_i<len(matrix) and 0<=next_j<len(matrix[0]) and matrix[next_i][next_j] > matrix[curr_i][curr_j]:
                                if dp[next_i][next_j] < curr_mx + 1:
                                    dp[next_i][next_j] = curr_mx + 1
                                    q.append((next_i, next_j, curr_mx + 1))
                                    res = max(res, curr_mx + 1)
                # else:
                #     res = max(res, dp[i][j])

        return res

sol=Solution_BFS()
sol.longestIncreasingPath(matrix)


"""
############################################################################
前缀和（Prefix Sum）
############################################################################
基础知识：前缀和本质上是在一个list当中，用O（N）的时间提前算好从第0个数字到第i个数字之和，
在后续使用中可以在O（1）时间内计算出第i到第j个数字之和，一般很少单独作为一道题出现，而是很多题目中的用到的一个小技巧
cumsum[j] - cumsum[i] = sum(nums[i+1:j+1])
cumsum[j-1] - cumsum[i-1] = sum(nums[i:j])
[1,2,3,4]
[1,3,6,10]
cumsum[3] - cumsum[1] = sum(nums[2:4])
"""

"""53. Maximum Subarray
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23

-104 <= nums[i] <= 104

Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.
"""
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = float('-Inf')
        cumsum = 0
        for num in nums:
            cumsum = cumsum + num
            res = max(res, cumsum)
            if cumsum < 0:
                # reset and start from here again
                cumsum = 0
        
        return res

# 题目还要求我们用分治法 Divide and Conquer Approach 来解，这个分治法的思想就类似于二分搜索法，
# 需要把数组一分为二，分别找出左边和右边的最大子数组之和，然后还要从中间开始向左右分别扫描，
# 求出的最大值分别和左右两边得出的最大值相比较取最大的那一个
"""
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if (nums.empty()) return 0;
        return helper(nums, 0, (int)nums.size() - 1);
    }
    int helper(vector<int>& nums, int left, int right) {
        if (left >= right) return nums[left];
        int mid = left + (right - left) / 2;
        int lmax = helper(nums, left, mid - 1);
        int rmax = helper(nums, mid + 1, right);
        int mmax = nums[mid], t = mmax;
        for (int i = mid - 1; i >= left; --i) {
            t += nums[i];
            mmax = max(mmax, t);
        }
        t = mmax;
        for (int i = mid + 1; i <= right; ++i) {
            t += nums[i];
            mmax = max(mmax, t);
        }
        return max(mmax, max(lmax, rmax));
    }
};
"""

""" 1423. Maximum Points You Can Obtain from Cards
There are several cards arranged in a row, and each card has an associated number of points. 
The points are given in the integer array cardPoints.

In one step, you can take one card from the beginning or from the end of the row. You have to take exactly k cards.

Your score is the sum of the points of the cards you have taken.

Given the integer array cardPoints and the integer k, return the maximum score you can obtain.

Example 1:
Input: cardPoints = [1,2,3,4,5,6,1], k = 3
Output: 12
Explanation: After the first step, your score will always be 1. However, choosing the 
rightmost card first will maximize your total score. The optimal strategy is to 
take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.

Example 2:
Input: cardPoints = [2,2,2], k = 2
Output: 4
Explanation: Regardless of which two cards you take, your score will always be 4.
Example 3:

Input: cardPoints = [9,7,7,9,7,7,9], k = 7
Output: 55
Explanation: You have to take all the cards. Your score is the sum of points of all cards.
"""

# can you think of a O(n) answer? 
# Let the sum of all points be total_pts. You need to remove a sub-array from cardPoints with length n - k.
# Keep a window of size n - k over the array. The answer is max(answer, total_pts - sumOfCurrentWindow)
class Solution:
    def maxScore(self, cardPoints, k: int) -> int:
        total_sum = sum(cardPoints)
        n = len(cardPoints)
        cumsum = 0
        for i in range(n-k):
            cumsum += cardPoints[i]
        res = cumsum
        for i in range(n-k, n):
            cumsum = cumsum+cardPoints[i]-cardPoints[i-(n-k)]
            res = min(res, cumsum)
        return total_sum - res

# DFS + memorazation 
class Solution:
    # MLT
    def maxScore(self, cardPoints, k: int) -> int:
        memo = dict()
        return self.dfs(tuple(cardPoints), k, memo)
    
    def dfs(self, cardPoints, k, memo):
        if (cardPoints, k) in memo:
            return memo[(cardPoints, k)]
        if k == 1:
            return max(cardPoints[0], cardPoints[-1])
        left = self.dfs(cardPoints[1:], k-1, memo) + cardPoints[0]
        right = self.dfs(cardPoints[:-1], k-1, memo) + cardPoints[-1]
        res = max(left, right)
        memo[(cardPoints, k)] = res
        return res


# [30,88,33,37,18,77,54,73,31,88,93,25,18,31,71,8,97,20,98,16,65,40,18,25,13,51,59]
# 26
sol = Solution()
sol.maxScore([1,2,3,4,5,6,1], 3)
sol.maxScore([30,88,33,37,18,77,54,73,31,88,93,25,18,31,71,8,97,20,98,16,65,40,18,25,13,51,59], 26)


"""031. Maximum Sum of Two Non-Overlapping Subarrays
Given an integer array nums and two integers firstLen and secondLen, return the maximum sum 
of elements in two non-overlapping subarrays with lengths firstLen and secondLen.

The array with length firstLen could occur before or after the array with length secondLen, but they have to be non-overlapping.

A subarray is a contiguous part of an array.

Example 1:
Input: nums = [0,6,5,2,2,5,1,9,4], firstLen = 1, secondLen = 2
Output: 20
Explanation: One choice of subarrays is [9] with length 1, and [6,5] with length 2.

Example 2:
Input: nums = [3,8,1,3,2,1,8,9,0], firstLen = 3, secondLen = 2
Output: 29
Explanation: One choice of subarrays is [3,8,1] with length 3, and [8,9] with length 2.

Example 3:
Input: nums = [2,1,5,6,0,9,5,0,3,8], firstLen = 4, secondLen = 3
Output: 31
Explanation: One choice of subarrays is [5,6,0,9] with length 4, and [3,8] with length 3.

Solution: Basically it can be broken it into 2 cases: L is always before M vs M is always before L.
L is always before M, we maintain a Lmax to keep track of the max sum of L subarray, and 
sliding the window of M from left to right to cover all the M subarray.
The same for the case where M is before L

<-            n         ->
<- L ->   <-  M ->
<-      i       ->
for i in range(M+L, n):
    M is always [i-M:i]
    L keep update [0:L], [1:L+1], ...

Explanation
Lsum, sum of the last L elements
Msum, sum of the last M elements
Lmax, max sum of contiguous L elements before the last M elements.
Mmax, max sum of contiguous M elements before the last L elements/

Complexity
Two pass, O(N) time,
O(1) extra space.
"""
class Solution:
    def maxSumTwoNoOverlap(self, nums, firstLen: int, secondLen: int) -> int:
        cumsum = [nums[0]]
        for num in nums[1:]:
            cumsum.append(cumsum[-1] + num)
        
        L = firstLen
        M = secondLen
        n = len(nums)
        res = cumsum[M+L-1]
        L_sum = cumsum[L-1]  # start at left L = sum(nums[:L])
        # when L is to left of M
        for i in range(M+L, n):
            M_sum = cumsum[i] - cumsum[i-M]
            L_sum = max(L_sum, cumsum[i-M] - cumsum[i-M-L])  # sliding window to nums[i-M-L:i-M]
            res = max(res, M_sum + L_sum)

        # when M is to left of L
        M_sum = cumsum[M-1]
        for i in range(M+L, n):
            L_sum = cumsum[i] - cumsum[i-L]
            M_sum = max(M_sum, cumsum[i-L] - cumsum[i-M-L])  # sliding window to nums[i-M-L:i-L]
            res = max(res, M_sum + L_sum)
        
        return res


sol=Solution()
sol.maxSumTwoNoOverlap(nums=[0,6,5,2,2,5,1,9,4], firstLen=1, secondLen=2)  # 20
sol.maxSumTwoNoOverlap(nums=[3,8,1,3,2,1,8,9,0], firstLen=3, secondLen=2)  # 29
sol.maxSumTwoNoOverlap(nums=[1,0,3], firstLen=1, secondLen=2)  # 29


""" 523. Continuous Subarray Sum
Given an integer array nums and an integer k, return true if nums has a continuous subarray 
of size at least two whose elements sum up to a multiple of k, or false otherwise.

An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k

Example 1:
Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.

Example 2:
Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.

Example 3:
Input: nums = [23,2,6,4,7], k = 13
Output: false

0 <= nums[i] <= 10**9

Hint
若数字a和b分别除以数字c，若得到的余数相同，那么 (a-b) 必定能够整除c
注意k=0的时候 无法取余
"""
class Solution:
    def checkSubarraySum(self, nums, k: int) -> bool:
        m = dict()
        cumsum = 0
        m[0] = -1  # !!!
        for i, num in enumerate(nums):
            cumsum = cumsum + num
            curr_key = cumsum % k if k!=0 else cumsum
            if curr_key in m:
                # never update m[curr_key] because need the longer subarray
                if i - m[curr_key] > 1:
                    # length at least 2
                    return True
                    # print(curr_key)
            else:
                # note, should not update if curr_key in m!!!
                m[curr_key] = i
        
        return False


sol=Solution()
sol.checkSubarraySum([23,2,6,4,7], 6)
sol.checkSubarraySum([23,2,6,4,7], 13)
sol.checkSubarraySum([23,2,4,6,6], 7)
sol.checkSubarraySum([5,0,0,0], 3)


"""304. Range Sum Query 2D - Immutable
Given a 2D matrix matrix, handle multiple queries of the following type:

Calculate the sum of the elements of matrix inside the rectangle defined by its upper 
left corner (row1, col1) and lower right corner (row2, col2).
Implement the NumMatrix class:

NumMatrix(int[][] matrix) Initializes the object with the integer matrix matrix.
int sumRegion(int row1, int col1, int row2, int col2) Returns the sum of the elements 
of matrix inside the rectangle defined by its upper left corner (row1, col1) and 
lower right corner (row2, col2).

Input
["NumMatrix", "sumRegion", "sumRegion", "sumRegion"]
[
    [[[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]], 
[2, 1, 4, 3], [1, 1, 2, 2], [1, 2, 2, 4]
]
Output
[null, 8, 11, 12]

# 维护一个二维数组dp，其中dp[i][j]表示累计区间(0, 0)到(i, j)这个矩形区间所有的数字之和，
# 那么此时如果我们想要快速求出(r1, c1)到(r2, c2)的矩形区间时，
# 只需dp[r2][c2] - dp[r2][c1 - 1] - dp[r1 - 1][c2] + dp[r1 - 1][c1 - 1]即可，
"""
class NumMatrix:
    def __init__(self, matrix):
        self.dp = [[0 for _ in range(len(matrix[0])+1)] for _ in range(len(matrix)+1)]
        for i in range(1, len(self.dp)):
            for j in range(1, len(self.dp[0])):
                self.dp[i][j] = self.dp[i-1][j] + self.dp[i][j-1] - self.dp[i-1][j-1] + matrix[i-1][j-1]
    
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = self.dp[row2+1][col2+1] - self.dp[row2+1][col1] - self.dp[row1][col2+1] + self.dp[row1][col1]
        return res

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)

"""
############################################################################
线段树 (Segment Tree or binary index tree): 一般用于处理上述Prefix sum但array会更新的情况. 
Init nlogn, query logn, update logn
############################################################################
"""
"""
307. Range Sum Query - Mutable
Given an integer array nums, handle multiple queries of the following types:
1. Update the value of an element in nums.
2. Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
Implement the NumArray class:
 
Input
["NumArray", "sumRange", "update", "sumRange"]
[[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
Output
[null, 9, null, 8]

Explanation
NumArray numArray = new NumArray([1, 3, 5]);
numArray.sumRange(0, 2); // return 1 + 3 + 5 = 9
numArray.update(1, 2);   // nums = [1, 2, 5]
numArray.sumRange(0, 2); // return 1 + 2 + 5 = 8
"""
# Solution 1: FenwickTree/binary index tree
# https://www.youtube.com/watch?v=WbafSgetDDk
class FenwickTree:
    # https://en.wikipedia.org/wiki/Fenwick_tree
    # Each element whose index i is a power of 2 contains the sum of the first i elements. 
    # Elements whose indices are the sum of two (distinct) powers of 2 contain the sum of the elements 
    # since the preceding power of 2. 
    # In general, each element contains the sum of the values since its parent in the tree, 
    # and that parent is found by clearing the least significant bit in the index.
    # Value as below: (treat zero differently becase lowbit(0)=0)
    # [0, 1, 1+2, 3, 1+2+3+4, 5, 3+4+5+6, 7, 1+...+8]
    def __init__(self, n):
        # sum(nums[:(i+1)])
        self.sums = [0] * n
    
    def update(self, i, delta):
        # update nums[i]
        if i == 0:
            self.sums[0] += delta
            return 
        while i < len(self.sums):
            self.sums[i] += delta
            i += FenwickTree.lowbit(i)
        return 
    
    def getSum(self, i):
        # sum(nums[:i+1])
        if i < 0:
            return 0
        res = self.sums[0]
        while i > 0:
            res += self.sums[i]
            i -= FenwickTree.lowbit(i)
        return res
    
    @staticmethod
    def lowbit(x):
        return x & (-x)

class NumArray:
    def __init__(self, nums):
        self.tree = FenwickTree(len(nums))
        self.nums = nums
        for i, num in enumerate(nums):
            self.tree.update(i, num)
        
    def update(self, i, val):
        # nums[i] set as val
        self.tree.update(i, val-self.nums[i])
        self.nums[i] = val
    
    def sumRange(self, i, j):
        # sum(nums[i:j+1]) = sum(nums[:j+1]) - sum(nums[:i])
        return self.tree.getSum(j) - self.tree.getSum(i-1)


sol=NumArray([1, 3, 5])
sol.sumRange(0, 2) # 9
sol.update(1, 2)
sol.nums
sol.sumRange(0, 2) # 8
sol.tree.getSum(2)

# another implementation - do not treat zero differently
class FenwickTree: 
    def __init__(self, n):
        # cumsum of numbers in the same branch (not nums[:i]!)
        self.cumsum = [0] * (n+1)
    
    def update(self, i, delta):
        # add delta to nums[i-1], logn, update all its parents
        # i >= 1, note lowbit(0) = 0 so dead loop if i=0
        while i < len(self.cumsum):
            self.cumsum[i] += delta
            i += FenwickTree.lowbit(i)  # add 1 at the lowest 1 
        
    def query(self, i):
        # cumsum of prefix i (nums[:i]), logn
        res = 0
        while i > 0:
            res += self.cumsum[i]
            i -= FenwickTree.lowbit(i)
        return res
    
    @staticmethod
    def lowbit(x):
        # find the lowest 1 in x
        # 0101 -> 1, 0110 -> 10
        return x & -x

class NumArray:
    def __init__(self, nums):
        self.tree = FenwickTree(len(nums))  # BIT
        for i, num in enumerate(nums):
            self.tree.update(i+1, num)
        self.nums = [0] + nums # just be consistent

    def update(self, index: int, val: int) -> None:
        self.tree.update(index+1, val - self.nums[index+1])
        self.nums[index+1] = val

    def sumRange(self, left: int, right: int) -> int:
        # nums[left:right+1]
        # sum(nums[:right+1]) - sum(nums[:left])
        return self.tree.query(right+1) - self.tree.query(left)

sol=NumArray([1, 3, 5])
sol.sumRange(0, 2) # 9
sol.update(1, 2)
sol.sumRange(0, 2) # 8
sol.tree.cumsum
sol.tree.query(2)

# Segment tree
# https://www.youtube.com/watch?v=rYBtViWXYeI
class TreeNode:
    def __init__(self, start, end):
        self.start = start # nums[start:end+1]
        self.end = end
        self.left = None # kid
        self.right = None
        self.sum = 0 # sum(nums[start:end+1])

class SegmentTree:
    def __init__(self, nums):
        self.root = self.buildTree(nums, 0, len(nums)-1)
        
    def buildTree(self, nums, left, right):
        # return a TreeNode has sum(nums[left:right+1]) as .sum
        if left > right:
            return None # in case typo input
        root = TreeNode(left, right)
        if left == right:
            root.sum = nums[left]
            return root
        mid = left + (right - left) // 2
        root.left = self.buildTree(nums, left, mid)
        root.right = self.buildTree(nums, mid+1, right)
        root.sum = root.left.sum + root.right.sum
        return root

    def update(self, index, val, node=None):
        # update nums[index] to val
        if not node:
            node = self.root
        if node.start == node.end:
            # leaf and should be =index
            node.sum = val
            return
        mid = node.start + (node.end - node.start) // 2
        if index <= mid:
            self.update(index, val, node.left)
        else:
            self.update(index, val, node.right)
        node.sum = node.left.sum + node.right.sum
        
    def sumRange(self, start, end, node=None): 
        # return sum(nums[start:end+1])
        if node is None:
            node = self.root
        if node.start == start and node.end == end:
            # exact match of curr node range
            return node.sum
        mid = node.start + (node.end - node.start) // 2
        if end <= mid: # [start, end] should locate on left child
            return self.sumRange(start, end, node.left)
        if start >= mid + 1:
            return self.sumRange(start, end, node.right)
        # part go to left, part to right
        return self.sumRange(start, mid, node.left) + \
            self.sumRange(mid+1, end, node.right)

class NumArray:
    def __init__(self, nums):
        self.tree = SegmentTree(nums)

    def update(self, i: int, val: int) -> None:
        self.tree.update(i, val, self.tree.root)

    def sumRange(self, i: int, j: int) -> int:
        return self.tree.sumRange(i, j, self.tree.root)

"""
############################################################################
并查集（Union Find）：把两个或者多个集合合并为一个集合
############################################################################
如果数据不是实时变化，本类问题可以用BFS或者DFS的方式遍历，如果数据实时变化（data stream）
则并查集每次的时间复杂度可以视为O（1）；需要牢记合并与查找两个操作的模板
"""

"""[LeetCode] Accounts Merge 账户合并
Given a list accounts, each element accounts[i] is a list of strings, where the first element 
accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there 
is some email that is common to both accounts. Note that even if two accounts have the same name, 
they may belong to different people as people could have the same name. A person can have any number 
of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of 
each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.

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
# Solution 1: Union Find
# 首先遍历每个账户和其中的所有邮箱，先将每个邮箱的 root 映射为其自身，然后将 owner 赋值为用户名。
# 然后开始另一个循环，遍历每一个账号，首先对帐号的第一个邮箱调用 find 函数，得到其父串p，
# 然后遍历之后的邮箱，对每个遍历到的邮箱先调用 find 函数，将其父串的 root 值赋值为p，
# 这样做相当于将相同账号内的所有邮箱都链接起来了。接下来要做的就是再次遍历每个账户内的所有邮箱，
# 先对该邮箱调用 find 函数，找到父串，然后将该邮箱加入该父串映射的集合汇总，这样就就完成了合并。
# 最后只需要将集合转为字符串数组，加入结果 res 中，通过 owner 映射找到父串的用户名，加入字符串数组的首位置
from collections import defaultdict

class UF:
    def __init__(self, N):
        self.parents = list(range(N))
        # when union, less weights node -> larger
        self.weights = [1] * N  
        
    def find(self, x):
        # link x's parent to the root, return
        if x!= self.parents[x]:
            # not root
            curr_parent = self.parents[x]
            self.parents[x] = self.find(curr_parent)
        return self.parents[x]
    
    def union(self, a, b):
        # find root of child and parent
        # link child's root to parent's root
        a_root = self.find(a)
        b_root = self.find(b)
        if a_root == b_root:
            return None
        if self.weights[a_root] <= self.weights[b_root]:
            self.parents[a_root] = b_root
            self.weights[b_root] += self.weights[a_root]
        else:
            self.parents[b_root] = a_root
            self.weights[a_root] += self.weights[b_root]
        return None


class Solution:
    def accountsMerge(self, accounts):
        uf = UF(len(accounts))  # account num 1-N, now need to find which nums should be unioned
        # prepare a hash with unique email address as key and index in accouts as value
        emailtoId = defaultdict(int)
        for i in range(len(accounts)):
            # curr_account_name = accounts[i][0]
            curr_emails = accounts[i][1:]
            for j in range(len(curr_emails)):
                email = curr_emails[j]
                # if we have already seen this email before, merge the account num "i" with previous account
                # else add it to hash
                if email in emailtoId:
                    # union the two ids
                    uf.union(i, emailtoId[email])
                else:
                    emailtoId[email] = i
        
        # prepare a hash with index in accounts as key and list of unique email address for that account as value
        # still do not have the account name
        idtoEmails = defaultdict(list)
        for curr_email, curr_id in emailtoId.items():
            root_id = uf.find(curr_id)
            idtoEmails[root_id].append(curr_email)  # add email to
        
        # collect the emails from idToEmails, sort it and add account name at index 0 to get the final list to add to final return List
        mergeDetails = []
        for id_num in idtoEmails.keys():
            # id_num is still integer
            curr_emails = sorted(idtoEmails[id_num])
            curr_account_name = accounts[id_num][0]
            mergeDetails.append([curr_account_name] + curr_emails)
        
        return mergeDetails


accounts = [["Hanzo","Hanzo2@m.co","Hanzo3@m.co"],["Hanzo","Hanzo4@m.co","Hanzo5@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co"],
["Hanzo","Hanzo3@m.co","Hanzo4@m.co"],["Hanzo","Hanzo7@m.co","Hanzo8@m.co"],["Hanzo","Hanzo1@m.co","Hanzo2@m.co"],
["Hanzo","Hanzo6@m.co","Hanzo7@m.co"],["Hanzo","Hanzo5@m.co","Hanzo6@m.co"]]
sol=Solution()
sol.accountsMerge(accounts)

# Solution2: DFS
from collections import defaultdict
class Solution:
    def accountsMerge(self, accounts):
        # construct graph
        email2id = defaultdict(list)
        email2name = dict()
        for idx, account in enumerate(accounts):
            for i in range(1, len(account)):
                email2id[account[i]].append(idx)
                email2name[account[i]] = account[0]
        
        visited = set()
        res = []
        for email in email2id.keys():
            curr_res = []
            self.dfs(email, visited, email2id, accounts, curr_res)
            if len(curr_res) > 0:
                res.append([email2name[email]]+sorted(curr_res))

        return res
    
    def dfs(self, email, visited, email2id, accounts, curr_emails):
        if email in visited:
            return 
        visited.add(email)
        curr_emails.append(email)
        for next_account in email2id[email]:
            for next_email in accounts[next_account][1:]:
                self.dfs(next_email, visited, email2id, accounts, curr_emails)
        
        return 


"""547. Number of Provinces
There are n cities. Some of them are connected, while some are not. If city a is 
connected directly with city b, and city b is connected directly with city c, 
then city a is connected indirectly with city c.
A province is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and 
the jth city are directly connected, and isConnected[i][j] = 0 otherwise. Return the total number of provinces.

Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
Output: 2

Input: isConnected = [[1,0,0],[0,1,0],[0,0,1]]
Output: 3

# BFS should work. can you use union find, what is the time complexity? O(n^2)?
"""
class Solution:
    def findCircleNum(self, isConnected) -> int:
        n_cities = len(isConnected)
        uf = UF(n_cities)
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                # avoid duplicate by j>i
                if isConnected[i][j] == 1:
                    uf.union(i, j)
        
        province_set = set()
        for i in range(n_cities):
            root = uf.find(i)
            if root not in province_set:
                province_set.add(root)
        
        return len(province_set)
text 

"""[LeetCode] 737. Sentence Similarity II 句子相似度之二

Given two sentences words1, words2 (each represented as an array of strings),
 and a list of similar word pairs pairs, determine if two sentences are similar.

For example, 
words1 = ["great", "acting", "skills"]
words2 = ["fine", "drama", "talent"]
pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]

Note that the similarity relation is transitive. For example, if "great" and "good" are similar, 
and "fine" and "good" are similar, then "great" and "fine" are similar.

Similarity is also symmetric. For example, "great" and "fine" being similar is the 
same as "fine" and "great" being similar.

Also, a word is always similar with itself. For example, the sentences 
words1 = ["great"], words2 = ["great"], pairs = [] are similar, 
even though there are no specified similar word pairs.

Finally, sentences can only be similar if they have the same number of words. 
So a sentence like words1 = ["great"] can never be similar to words2 = ["doubleplus","good"].

Note:
The length of words1 and words2 will not exceed 1000.
The length of pairs will not exceed 2000.
The length of each pairs[i] will be 2.
The length of each words[i] and pairs[i][j] will be in the range [1, 20]
"""
# Use the UF above 
def areSentencesSimilarTwo(words1, words2, pairs):
    # you can revise the UF by making parents a dict (from word to its parent word)
    # if keep using the same, then first build a word to id map 
    if len(words1) != len(words2):
        return False
    word2id = dict()
    curr_id = 0
    for pair in pairs:
        w1, w2 = pair
        if w1 not in word2id:
            word2id[w1] = curr_id
            curr_id += 1
        if w2 not in word2id:
            word2id[w2] = curr_id
            curr_id += 1

    uf = UF(len(word2id))
    for pair in pairs:
        w1, w2 = pair
        uf.union(word2id[w1], word2id[w2])

    for i in range(len(words1)):
        w1, w2 = words1[i], words2[i]
        if w1 == w2:
            continue
        w1_root = uf.find(word2id[w1])
        w2_root = uf.find(word2id[w2])
        if w1_root != w2_root:
            return False
            # print(w1, w2)

    return True


words1 = ["great", "acting", "skills"]; words2 = ["fine", "drama", "talent"]
pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]

words1 = ["great"]; words2 = ["great"]; pairs = [] 
areSentencesSimilarTwo(words1, words2, pairs)

# another version of UF: use string as input directly
from collections import defaultdict
class UF_word:
    def __init__(self):
        self.parents = defaultdict(str)
        # when union, less weights node -> larger
        self.weights = defaultdict(lambda:1)  # default 1
        
    def find(self, x, create=True):
        # link x's parent to the root, return
        # if x is a new element, its parent is itself
        if x not in self.parents:
            if create:
                self.parents[x] = x
                return self.parents[x]
            else:
                raise ValueError(f"{x} not in current UF set!")
        if x!= self.parents[x]:
            # not root
            curr_parent = self.parents[x]
            self.parents[x] = self.find(curr_parent)
        return self.parents[x]
    
    def union(self, a, b):
        # find root of child and parent
        # link child's root to parent's root
        a_root = self.find(a)
        b_root = self.find(b)
        if a_root == b_root:
            return None
        if self.weights[a_root] <= self.weights[b_root]:
            self.parents[a_root] = b_root
            self.weights[b_root] += self.weights[a_root]
        else:
            self.parents[b_root] = a_root
            self.weights[a_root] += self.weights[b_root]
        return None


def areSentencesSimilarTwo(words1, words2, pairs):
    # you can revise the UF by making parents a dict (from word to its parent word)
    # if keep using the same, then first build a word to id map 
    if len(words1) != len(words2):
        return False

    uf = UF_word()
    for pair in pairs:
        w1, w2 = pair
        uf.union(w1, w2)

    for i in range(len(words1)):
        w1, w2 = words1[i], words2[i]
        if w1 == w2:
            continue
        if w1 not in uf.parents or w2 not in uf.parents:
            return False
        w1_root = uf.find(w1)
        w2_root = uf.find(w2)
        if w1_root != w2_root:
            return False
            # print(w1, w2)

    return True


"""[LeetCode] 305. Number of Islands II 岛屿的数量之二
A 2d grid map of m rows and n columns is initially filled with water. 
We may perform an addLand operation which turns the water at position (row, col) into a land. 
Given a list of positions to operate, count the number of islands after each addLand operation. 
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
You may assume all four edges of the grid are all surrounded by water.

Example:
Input: m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]]
Output: [1,1,2,3]

Follow up:
Can you do it in time complexity O(k log mn), where k is the length of the positions?
"""
# 将二维数组 encode 为一维的，于是需要一个长度为 m*n 的一维数组来标记各个位置属于哪个岛屿
# Every time a new land is built, check its surroundings, if also island, then union
# slightly modify UF to update current number of island
class UF:
    def __init__(self, N):
        self.parents = list(range(N))
        # when union, less weights node -> larger
        self.weights = [1] * N  
        self.count = 0
        
    def find(self, x):
        # link x's parent to the root, return
        if x!= self.parents[x]:
            # not root
            curr_parent = self.parents[x]
            self.parents[x] = self.find(curr_parent)
        return self.parents[x]
    
    def union(self, a, b):
        # find root of child and parent
        # link child's root to parent's root
        a_root = self.find(a)
        b_root = self.find(b)
        if a_root == b_root:
            return None
        if self.weights[a_root] <= self.weights[b_root]:
            self.parents[a_root] = b_root
            self.weights[b_root] += self.weights[a_root]
        else:
            self.parents[b_root] = a_root
            self.weights[a_root] += self.weights[b_root]
        return None

def numIslands2(m, n, positions):
    island_vector = [0 for _ in range(m*n)]
    uf = UF(m*n)
    res = []
    for position in positions:
        x, y = position
        island_vector[map2dto1d(x, y, n)] = 1
        uf.count += 1
        for dx, dy in [(-1 ,0), (1, 0), (0, -1), (0, 1)]:
            next_x, next_y = x+dx, y+dy
            if 0<=next_x<m and 0<=next_y<n:
                if island_vector[map2dto1d(next_x, next_y, n)] == 1:
                    # island next to island
                    uf.union(map2dto1d(x, y, n), map2dto1d(next_x, next_y, n))
                    uf.count -= 1
        res.append(uf.count)
    return res

def map2dto1d(x, y, n_cols):
    # (x,y) -> location in 1d
    return x*n_cols + y


m = 3; n = 3; positions = [[0,0], [0,1], [1,2], [2,1]]
numIslands2(3, 3, [[0,0], [0,1], [1,2], [2,1]])


"""
############################################################################
字典树（Trie）
############################################################################
多数情况下可以通过用一个set来记录所有单词的prefix来替代，时间复杂度不变，但空间复杂度略高
"""

"""208. Implement Trie (Prefix Tree)
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently 
store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
"""
from collections import defaultdict

class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = dict()
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_word = True

    # Returns if the word is in the trie.
    def search(self, word: str) -> bool:
        current = self.root
        for letter in word:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return current.is_word
    
    # Returns if there is any word in the trie that starts with the given prefix.
    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for letter in prefix:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return True


t1= Trie()
t1.insert("apple")
t1.insert("pear")
t1.root.children['a'].children

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)


"""212. Word Search II
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, 
where adjacent cells are horizontally or vertically neighboring. 
The same letter cell may not be used more than once in a word.

Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Solution: Trie
https://leetcode.com/problems/word-search-ii/discuss/59790/Python-dfs-solution-(directly-use-Trie-implemented).

board = [["o","a","a","n"],
["e","t","a","e"],
["i","h","k","r"],
["i","f","l","v"]]
words = ["oath","pea","eat","rain","oathi","oathk","oathf","oate","oathii","oathfi","oathfii"]
"""
# Use the trie defined as above
class Solution:
    def findWords(self, board, words):
        trie = Trie()
        node = trie.root
        for word in words:
            trie.insert(word)

        visited = [[False for _ in range(len(board[0]))] for _ in range(len(board))]
        self.res = []

        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(i, j, board, node, "", visited)
        
        return self.res
                    
    def dfs(self, i, j, board, node, curr_str, visited):
        if node.is_word: #and curr_str not in self.res:
            # curr_str is in trie
            self.res.append(curr_str)
            node.is_word = False  # avoid replicate

        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or visited[i][j]:
            return 
        if board[i][j] not in node.children:
            return 
        next_node = node.children[board[i][j]]
        visited[i][j] = True
        self.dfs(i-1, j, board, next_node, curr_str+board[i][j], visited)
        self.dfs(i+1, j, board, next_node, curr_str+board[i][j], visited)
        self.dfs(i, j-1, board, next_node, curr_str+board[i][j], visited)
        self.dfs(i, j+1, board, next_node, curr_str+board[i][j], visited)
        visited[i][j] = False
        return 

sol = Solution()
sol.findWords(board, words)


""" 211. Design Add and Search Words Data Structure
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. 
word may contain dots '.' where dots can be matched with any letter.

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation:
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
"""
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = dict()
        self.is_word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_word = True

    def search(self, word: str) -> bool:
        return self.dfs(self.root, word)
    
    def dfs(self, node, word):
        if len(word) == 0:
            return True if node.is_word else False
        if word[0] != '.':
            if word[0] in node.children:
                return self.dfs(node.children[word[0]], word[1:])
            else:
                return False
        else:
            for letter in node.children.keys():
                if self.dfs(node.children[letter], word[1:]):
                    return True
            return False
        
      
# ["WordDictionary","addWord","addWord","addWord","addWord","search","search","addWord","search","search","search","search","search","search"]
# [[],["at"],["and"],["an"],["add"],["a"],[".at"],["bat"],[".at"],["an."],["a.d."],["b."],["a.d"],["."]]

class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str) -> None:
        current = self.root
        for letter in word:
            if letter not in current.children:
                current.children[letter] = TrieNode()
            current = current.children[letter]
        current.is_word = True

    # Returns if the word is in the trie.
    def search(self, word: str) -> bool:
        current = self.root
        for letter in word:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return current.is_word
    
    # Returns if there is any word in the trie that starts with the given prefix.
    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for letter in prefix:
            if letter in current.children:
                current = current.children[letter]
            else:
                return False
        return True

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)


""" 1268. Search Suggestions System
You are given an array of strings products and a string searchWord.

Design a system that suggests at most three product names from products 
after each character of searchWord is typed. Suggested products should have common prefix 
with searchWord. If there are more than three products with a common prefix 
return the three lexicographically minimums products.

Return a list of lists of the suggested products after each character of searchWord is typed.

Example 1:

Input: products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"
Output: [
["mobile","moneypot","monitor"],
["mobile","moneypot","monitor"],
["mouse","mousepad"],
["mouse","mousepad"],
["mouse","mousepad"]
]
Explanation: products sorted lexicographically = ["mobile","moneypot","monitor","mouse","mousepad"]
After typing m and mo all products match and we show user ["mobile","moneypot","monitor"]
After typing mou, mous and mouse the system suggests ["mouse","mousepad"]

All the strings of products are unique.
"""
from collections import defaultdict
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.suggestion = []
    
    def add_suggestion(self, product, k=3):
        if len(self.suggestion) < k:
            self.suggestion.append(product)

class Solution:
    def suggestedProducts(self, products, searchWord: str):
        products = sorted(products)  # keep lexicographical order
        root = TrieNode()
        for product in products:
            curr = root
            for char in product:
                if char not in curr.children:
                    curr.children[char] = TrieNode()
                curr = curr.children[char]
                curr.add_suggestion(product) # suggestion will be product
        
        res = []
        curr = root
        for char in searchWord:
            curr_suggestion = curr.children[char].suggestion
            res.append(curr_suggestion)
            curr = curr.children[char]
        return res


sol=Solution()
sol.suggestedProducts(products=["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse")
sol.suggestedProducts(products=["havana"], searchWord = "tatiana")

#
# Solution 2: sort + binary search
# Time O(NlogN) for sorting
# Time O(logN) for each query
# https://leetcode.com/problems/search-suggestions-system/discuss/436674/C%2B%2BJavaPython-Sort-and-Binary-Search-the-Prefix
import bisect
def suggestedProducts(self, products, word):
    products.sort()
    res, prefix, i = [], '', 0
    for c in word:
        prefix += c
        # bisect.bisect_left returns the leftmost place in the sorted list to insert the given element.
        i = bisect.bisect_left(products, prefix, lo=i)  # lo just set the left_start=i for binary search 
        res.append([w for w in products[i:i + 3] if w.startswith(prefix)])
    return res


"""
############################################################################
单调栈与单调队列（Monotone Stack／Queue）
############################################################################
栈还是普通栈，不论单调栈还是单调队列，单调的意思是保留在栈或者队列中的数字是单调递增或者单调递减的
"""

"""84. Largest Rectangle in Histogram
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, 
return the area of the largest rectangle in the histogram.

Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.

Input: heights = [2,4]
Output: 4
"""

# Solution 1: Brutal force O(n^2): 
# for every height, get the width by going to left and right till lower than it O(n)
# skip

# Solution 2: 
"""Use a monotonic stack to maintain the higher bars's indices in ascending order.
When encounter a lower bar, pop the tallest bar and use it as the bottleneck to compute the area.

https://leetcode-cn.com/problems/largest-rectangle-in-histogram/solution/zhu-zhuang-tu-zhong-zui-da-de-ju-xing-by-leetcode-/
Time complexity: O(n)
Space complexity: O(n)
"""
heights = [2,1,5,6,2,3]; 
class Solution:
    def largestRectangleArea(self, heights) -> int:
        stack = []
        res = 0
        heights.append(0)  # sentinel, avoid another loop to clean up stacks
        # one idea to avoid discussion on len(s) > 0 in loop is to add -1 into stack
        # so s will never be empty
        n = len(heights)
        for i in range(n):
            if len(stack) == 0 or heights[i] > heights[stack[-1]]:
                stack.append(i)
            else:
                while len(stack) > 0 and heights[i] <= heights[stack[-1]]:
                    # if equal height, current i will have same area as previous, so pop previous
                    prev_i = stack.pop()
                    # prev_area can take up from stack[-1] (not included) to i-1 (included) if stack not empty
                    # if stack empty, take from 0 to i-1 -> len = i
                    prev_area = heights[prev_i] * (i - 1 - stack[-1]) if len(stack)>0 else heights[prev_i] * i
                    res = max(res, prev_area)
                stack.append(i)
        
        # clean up remaining in the stack
        # one way to avoid this is sentinel: heights.append(0), see above
        # while len(stack) > 0 and heights[i] <= heights[stack[-1]]:
        #     # if equal height, current i will have same area as previous, so pop previous
        #     prev_i = stack.pop()
        #     # prev_area can take up from stack[-1] (not included) to i-1 (included)
        #     prev_area = heights[prev_i] * (n - 1 - stack[-1]) if len(stack)>0 else heights[prev_i] * n
        #     res = max(res, prev_area)
        return res

sol=Solution()
sol.largestRectangleArea([2,4])
sol.largestRectangleArea([1])
sol.largestRectangleArea([2,1,5,6,2,3])

"""85. Maximal Rectangle
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Input: matrix = [
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.

Input: matrix = [["1"]]
Output: 1
"""
# Solution 1: use the solution of 84. Largest Rectangle in Histogram
# for each row, apply this o(n)
# total O(m*n)

# Solution 2: DP: skip
"""
Time complexity: O(m^2*n)
Space complexity: O(mn)
https://zxi.mytechroad.com/blog/dynamic-programming/leetcode-85-maximal-rectangle/

dp[i][j] := max length of all 1 sequence ends with col j, at the i-th row.
transition:
dp[i][j] = 0 if matrix[i][j] == 0
= dp[i][j-1] + 1 if matrix[i][j] == 1

class Solution {
public:
  int maximalRectangle(vector<vector<char> > &matrix) {
    int r = matrix.size();
    if (r == 0) return 0;
    int c = matrix[0].size();
    // dp[i][j] := max len of all 1s ends with col j at row i.
    vector<vector<int>> dp(r, vector<int>(c));
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j)
        dp[i][j] = (matrix[i][j] == '1') ? (j == 0 ? 1 : dp[i][j - 1] + 1) : 0;
 
    int ans = 0;
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) {
        int len = INT_MAX;
        for (int k = i; k < r; k++) {
          len = min(len, dp[k][j]);
          if (len == 0) break;
          ans = max(len * (k - i + 1), ans);
        }
    }
    return ans;
  }
};
"""


"""[LeetCode] 42. Trapping Rain Water 收集雨水
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it is able to trap after raining.

The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

Example:
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

遍历高度，如果此时栈为空，或者当前高度小于等于栈顶高度，则把当前高度的坐标压入栈，注意这里不直接把高度压入栈，
而是把坐标压入栈，这样方便在后来算水平距离。当遇到比栈顶高度大的时候，就说明有可能会有坑存在，可以装雨水。
此时栈里至少有一个高度，如果只有一个的话，那么不能形成坑，直接跳过，如果多余一个的话，
那么此时把栈顶元素取出来当作坑，新的栈顶元素就是左边界，当前高度是右边界，只要取二者较小的，减去坑的高度，
长度就是右边界坐标减去左边界坐标再减1，二者相乘就是盛水量啦
Time complexity: O(n)
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

# Solution 2: DP
# left[i]: max height on the left of height[i]
# right[i]:max height on the right of height[i]
# the water that point i can contribute is: min(l, r) - height[i]
# left[i] = max(height[i], left[i-1]) if i>0 else height[i]
# right[i] = max(height[i], right[i+1]) if i<len(height)-1 else height[i]
"""
// Author: Huahua
class Solution {
public:
  int trap(vector<int>& height) {
    const int n = height.size();
    vector<int> l(n);
    vector<int> r(n);
    int ans = 0;
    for (int i = 0; i < n; ++i)
      l[i] = i == 0 ? height[i] : max(l[i - 1], height[i]);
    for (int i = n - 1; i >= 0; --i)
      r[i] = i == n - 1 ? height[i] : max(r[i + 1], height[i]);
    for (int i = 0; i < n; ++i)
      ans += min(l[i], r[i]) - height[i];
    return ans;
  }
};
"""


""" 907. Sum of Subarray Minimums !!!
Given an array of integers arr, find the sum of min(b), where b ranges over every (contiguous) subarray of arr. 
Since the answer may be large, return the answer modulo 10**9 + 7.

Example 1:
Input: arr = [3,1,2,4]
Output: 17
Explanation: 
Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4]. 
Minimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1.
Sum is 17.

Example 2:
Input: arr = [11,81,94,43,3]
Output: 444

Idea:
                            [2, 9, 7, 8, 3, 4, 6, 1]
			                    |        |     |
	                  the previous less       the next less 
	                     element of 3          element of 3

How many subarrays with 3 being its minimum value?
left : 4 (distance of 3 to 2)
right: 3
Combine of left and right: 12
What if equal? always put it on right. 

For each A[i]
Denote by left[i] the distance between element A[i] and its PLE (strictly).
Denote by right[i] the distance between element A[i] and its NLE (can equal).

Complexity:
All elements will be pushed twice and popped at most twice
O(n) time, O(n) space
"""
class Solution:
    def sumSubarrayMins(self, arr) -> int:
        n = len(arr)
        left, right = [0]*n, [0]*n
        stack_left, stack_right = [], []
        # first left
        for i in range(n):
            cnt = 1  # itself
            if len(stack_left) == 0 or arr[i] >= arr[stack_left[-1]]:
                left[i] = cnt  # no element on left < arr[i]
                stack_left.append(i)
            else:
                while len(stack_left) > 0 and arr[i] < arr[stack_left[-1]]:
                    # left element > arr[i] (not including equal)
                    prev_i = stack_left.pop()
                    cnt += left[prev_i]
                left[i] = cnt
                stack_left.append(i)
        
        # second, right 
        for i in range(n)[::-1]:
            cnt = 1
            if len(stack_right) == 0 or arr[i] > arr[stack_right[-1]]:
                right[i] = cnt # no element on right <= arr[i]
                stack_right.append(i)
            else:
                while len(stack_right) > 0 and arr[i] <= arr[stack_right[-1]]:
                    prev_i = stack_right.pop()
                    cnt += right[prev_i]
                right[i] = cnt
                stack_right.append(i)
        
        # combine
        res = 0
        for i in range(n):
            res += arr[i] * (left[i]) * (right[i])
        
        return res % (10**9 + 7)


sol= Solution()
sol.sumSubarrayMins(arr=[3,1,2,4])


"""739. Daily Temperatures
Given an array of integers temperatures represents the daily temperatures, 
return an array answer such that answer[i] is the number of days you have to wait 
after the ith day to get a warmer temperature. If there is no future day for which 
this is possible, keep answer[i] == 0 instead.

Example 1:
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

Example 2:
Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]

Example 3:
Input: temperatures = [30,60,90]
Output: [1,1,0]
"""
class Solution:
    def dailyTemperatures(self, temperatures):
        stack = []
        n = len(temperatures)
        res = [0] * n
        for i in range(n):
            if len(stack) == 0 or temperatures[i] <= temperatures[stack[-1]]:
                stack.append(i)
            else:
                while len(stack) > 0 and temperatures[i] > temperatures[stack[-1]]:
                    prev_i = stack.pop()
                    res[prev_i] = i - prev_i
                stack.append(i)
        
        return res
        
sol=Solution()
sol.dailyTemperatures([73,74,75,71,69,72,76,73])
sol.dailyTemperatures([30,40,50,60])


"""901. Online Stock Span
Design an algorithm that collects daily price quotes for some stock and 
returns the span of that stock's price for the current day.

The span of the stock's price today is defined as the maximum number of consecutive days
 (starting from today and going backward) for which the stock price was 
 less than or equal to today's price.

For example, if the price of a stock over the next 7 days were [100,80,60,70,60,75,85], 
then the stock spans would be [1,1,1,2,1,4,6].

Implement the StockSpanner class:
StockSpanner() Initializes the object of the class.
int next(int price) Returns the span of the stock's price given that today's price is price.

Input
["StockSpanner", "next", "next", "next", "next", "next", "next", "next"]
[[], [100], [80], [60], [70], [60], [75], [85]]
Output
[null, 1, 1, 1, 2, 1, 4, 6]
"""
# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)
# another solution is DP: dp[i] = dp[i-1] + 1 if stocks[i] > stocks[i-1] else 1
class StockSpanner:
    def __init__(self):
        self.stack = []  # save idx
        self.stocks = []
        self.i = -1

    def next(self, price: int) -> int:
        self.stocks.append(price)
        self.i += 1
        if len(self.stack) == 0 or self.stocks[self.stack[-1]] > price:
            self.stack.append(self.i)
            return 1
        else:
            while len(self.stack) > 0 and self.stocks[self.stack[-1]] <= price:
                prev_i = self.stack.pop()
            res = self.i - self.stack[-1] if len(self.stack) > 0 else self.i + 1
            self.stack.append(self.i)
            
            return res


"""Next Greater Element II 下一个较大的元素之二 
Given a circular array (the next element of the last element is the first 
element of the array), print the Next Greater Number for every element. The 
Next Greater Number of a number x is the first greater number to its 
traversing-order next in the array, which means you could search circularly to 
find its next greater number. If it doesn't exist, output -1 for this number.

Example 1:
Input: [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number; 
The second 1's next greater number needs to search circularly, which is also 2.
"""
class Solution:
    def nextGreaterElements(self, nums):
        n = len(nums)
        res = [-1] * n
        stack = []
        for i in range(n*2):
            curr_i = i % n
            if len(stack) != 0:
                # 如果此时栈不为空，且栈顶元素小于当前数字，说明当前数字就是栈顶元素的右边第一个较大数，那么建立二者的映射
                while len(stack) != 0 and nums[curr_i] > nums[stack[-1]]:
                    res[stack[-1]] = nums[curr_i]
                    stack.pop()
            if i < n:
                # 因为 res 的长度必须是n，超过n的部分我们只是为了给之前栈中的数字找较大值，所以不能压入栈，
                # actually it does not matter...
                stack.append(i)
        return res


sol = Solution()
sol.nextGreaterElements([1,2,1])


"""239. Sliding Window Maximum
You are given an array of integers nums, there is a sliding window of size k 
which is moving from the very left of the array to the very right. 
You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

Example 1:

Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

Follow up: how about runtime complexity O(n)? 
 How about using a data structure such as deque (double-ended queue)?
 The queue size need not be the same as the window's size.
 Remove redundant elements and the queue should store only elements that need to be considered.

Solution1: O(n)
We scan the array from 0 to n-1, keep "promising" elements in the deque. 
The algorithm is amortized O(n) as each element is put and polled once.
At each i, we keep "promising" elements, which are potentially max number in window [i-(k-1),i] or 
any subsequent window. This means:

1) If an element in the deque and it is out of i-(k-1), we discard them. We just need to poll from the head, 
as we are using a deque and elements are ordered as the sequence in the array
2) Now only those elements within [i-(k-1),i] are in the deque. We then discard elements smaller 
than a[i] from the tail. This is because if a[x] <a[i] and x<i, then a[x] has no chance to be the 
"max" in [i-(k-1),i], or any other subsequent window: a[i] would always be a better candidate.
3) As a result elements in the deque are ordered in both sequence in array and their value. 
At each step the head of the deque is the max element in [i-(k-1),i]
"""
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums, k: int):
        q = deque()  # save idx, need to measure k length
        res = []
        for i in range(len(nums)):
            if len(q) > 0 and i - q[0] == k:
                # q is k+1 long
                q.popleft()
            while len(q)>0 and nums[i] > nums[q[-1]]:
                q.pop()
            q.append(i)  # q may not k-long but in desc order
            if i >= k-1:
                # q[0] is always the largest of q
                res.append(nums[q[0]])
        return res

sol=Solution()
sol.maxSlidingWindow(nums = [1,3,-1,-3,5,3,6,7], k = 3)

""" Solution 2: max heap O(Nlgk)
Add k elements and their indices to heap. Python has min-heap. So for max-heap, multiply by -1.
Set start = 0 and end = k-1 as the current range.
Extract the max from heap which is in range i.e. >= start. Add the max to the result list. 
Now add the max back to the heap - it could be relevant to other ranges.
Move the range by 1. Add the new last number to heap.

Note that we need not invest into thinking about deleting the obsolete entry every time 
the window slides.That would be very hard to implement. 
Instead we maintain the index in heap and "delete" when the maximum number is out of bounds.

https://leetcode.com/problems/sliding-window-maximum/discuss/65957/Python-solution-with-detailed-explanation
"""
import heapq as h
class Solution(object):
    def get_next_max(self, heap, start):
        while True:
            x,idx = h.heappop(heap)
            if idx >= start:
                # only take when in the window
                # up to O(n) in total (sum across i from k+1 - n)
                return x*-1, idx
    
    def maxSlidingWindow(self, nums, k):
        if k == 0:
            return []
        heap = []
        for i in range(k):
            h.heappush(heap, (nums[i]*-1, i))
        result, start, end = [], 0, k-1
        while end < len(nums):
            x, idx = self.get_next_max(heap, start) # start is to ensure idx in curr window
            result.append(x)
            h.heappush(heap, (x*-1, idx)) 
            start, end = start + 1, end + 1
            if end < len(nums):
                h.heappush(heap, (nums[end]*-1, end))
        return result


"""
############################################################################
扫描线算法（Sweep Line）
############################################################################
一个很巧妙的解决时间安排冲突的算法
"""

"""[LeetCode] 253. Meeting Rooms II 会议室之二
Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), 
find the minimum number of conference rooms required.

Example 1:
Input: 
intervals = [[0, 30],[5, 10],[15, 20]]
Output: 2

Example 2:
Input: [[7,10],[2,4]]
Output: 1
NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.

Solution:
遍历时间区间，可以按照每个区间的起点排序,对于起始时间，映射值自增1，对于结束时间，映射值自减1，然后定义结果变量 res，和房间数 rooms
NlogN
"""
from collections import defaultdict
def minMeetingRooms(intervals):
    m = defaultdict(lambda :0)
    for interval in intervals:
        m[interval[0]] = m[interval[0]] + 1
        m[interval[1]] = m[interval[1]] - 1
    
    curr_rooms = 0
    res = 0
    for k, v in sorted(m.items(), key=lambda x: x[0]):
        # sort by time
        curr_rooms += v
        res = max(res, curr_rooms)
    
    return res


"""218. The Skyline Problem
A city's skyline is the outer contour of the silhouette formed by all the buildings 
in that city when viewed from a distance. Given the locations and heights of all the buildings, 
return the skyline formed by these buildings collectively.

The geometric information of each building is given in the array buildings where buildings[i] = [lefti, righti, heighti]:

lefti is the x coordinate of the left edge of the ith building.
righti is the x coordinate of the right edge of the ith building.
heighti is the height of the ith building.

You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height 0.

There must be no consecutive horizontal lines of equal height in the output skyline. 
For instance, [...,[2 3],[4 5],[7 5],[11 5],[12 7],...] is not acceptable; 
the three lines of height 5 should be merged into one in the final output as such: [...,[2 3],[4 5],[12 7],...]

Input: buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
Output: [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
Explanation:
Figure A shows the buildings of the input.
Figure B shows the skyline formed by those buildings. The red points in figure B represent the key points in the output list.

Input: buildings = [[0,2,3],[2,5,3]]
Output: [[0,3],[5,0]]

Pseudo code:
events = {{x: left, h: heights, type:enter}, {x:right, h:heights, type:leaving}}
events.sorted(by=(x, type, h*type))
ds = DS()  # provide 1) max, 2) remove a given element
for e in events:
    if e.type == enter:
        if e.h > ds.max():
            res.append([e.x, e.h])
        ds.add(e.h)
    if e.type == leaving:
        ds.remove(e.h)
        if e.h > ds.max():
            # current e.h is the max before removing
            res.append([x, ds.max()])
"""
# https://zxi.mytechroad.com/blog/tree/leetcode-218-the-skyline-problem/
# https://leetcode.com/problems/the-skyline-problem/discuss/954585/Python-O(n-log-n)-solution-using-heap-explained
# O(nlogn) time. O(n) space
from heapq import heappush, heappop
class Solution:
    def getSkyline(self, buildings):
        left_points = [(l, h, -1, i) for i, (l, _, h) in enumerate(buildings)] # -1 means enter 
        right_points = [(r, h, 1, i) for i, (_, r, h) in enumerate(buildings)]  # 1 means leave
        points = left_points + right_points
        # x first, then 1) enter always before leave 2) if both enter, higher first, 3) if both leave, smaller first
        points = sorted(points, key=lambda x: (x[0], x[2], x[1]*x[2]))  # sort asce
        heap = [(0, -1)]  # use -height as max height heap
        active = set([-1])  # record idx of building that in current location,
        # (0, -1) will nevre be poped so height 0 can be added when all buildings are poped
        res = []
        for x, h, status, idx in points:
            if status == -1:
                active.add(idx)
            else:
                active.remove(idx)

            if status == -1:
                # enter
                if h > -1 * heap[0][0]:
                    # new enter is max, add into res
                    res.append([x, h])
                # always add new h into heap
                heappush(heap, (-h, idx))
            else:
                # leave the max, need to rm inactive buildings from heap
                while len(heap) > 0 and heap[0][1] not in active:
                    heappop(heap)
                if heap[0][0] * -1 != res[-1][1]:
                    # res[-1][1] stores the highest before removing
                    res.append([x, heap[0][0] * -1])

        return res


"""[LeetCode] Employee Free Time 职员的空闲时间
We are given a list schedule of employees, which represents the working time for each employee.
Each employee has a list of non-overlapping Intervals, and these intervals are in sorted order.
Return the list of finite intervals representing common, positive-length free time for all employees, 
also in sorted order.

Example 1:
Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]
Explanation:
There are a total of three employees, and all common
free time intervals would be [-inf, 1], [3, 4], [10, inf].
We discard any intervals that contain inf as they aren't finite.

Example 2:
Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
Output: [[5,6],[7,9]]

(Even though we are representing Intervals in the form [x, y], 
the objects inside are Intervals, not lists or arrays. For example, 
schedule[0][0].start = 1, schedule[0][0].end = 2, and schedule[0][0][0] is not defined.)

Also, we wouldn't include intervals like [5, 5] in our answer, as they have zero length.
"""
from collections import defaultdict
class Solution:
    def employeeFreeTime(self, schedule):
            m = defaultdict(lambda :0)
            for intervals in schedule:
                for interval in intervals:
                    m[interval[0]] += 1
                    m[interval[1]] -= 1
            cnt = 0
            res = []
            free_start = False
            for k,v in sorted(m.items(), key=lambda x:x[0]):
                cnt += v
                if cnt == 0:
                    # res.append([k, float('Inf')])
                    start = k
                    free_start = True
                if cnt > 0 and free_start: # and len(res) > 0:
                    end = k
                    res.append([start, end])
                    free_start = False
                    # res[-1][1] = k
            # last will always be [end, inf]
            return res
            # return res[:-1] if len(res) > 0 else []

# solution 2
schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
class Solution:
    def employeeFreeTime(self, schedule):
        merged_schedule = []
        for intervals in schedule:
            merged_schedule += intervals
        
        merged_schedule = sorted(merged_schedule, key=lambda x:x[0])
        # print(merged_schedule)
        res = []
        start, end = merged_schedule[0]
        for interval in merged_schedule[1:]:
            if interval[0] >= end:
                res.append([end, interval[0]])
                start, end = interval
            else:
                end = max(interval[1], end)

        return res

sol=Solution()
sol.employeeFreeTime([[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]])


"""
############################################################################
TreeMap 
############################################################################
https://grantjenks.com/docs/sortedcontainers/sorteddict.html
SortedDict: (SortedList is a sorted list)
d.popitem(k) will pop the kth smallest item  - logN
d.bisect_left(val) # return the index of val (from left) if inserted - logN


基于红黑树（平衡二叉搜索树）的一种树状 hashmap，增删查改、找求大最小均为logN复杂度，Python当中可以使用SortedDict替代
SortedDict继承了普通的dict全部的方法，除此之外还可以peekitem(k)来找key里面第k大的元素，
popitem(k)来删除掉第k大的元素，弥补了Python自带的heapq没法logN时间复杂度内删除某个元素的缺陷
"""

"""729. My Calendar I
You are implementing a program to use as your calendar. We can add a new event if adding the event 
will not cause a double booking.

A double booking happens when two events have some non-empty intersection (i.e., some moment is 
common to both events.).

The event can be represented as a pair of integers start and end that represents a booking on the 
half-open interval [start, end), the range of real numbers x such that start <= x < end.

Implement the MyCalendar class:

MyCalendar() Initializes the calendar object.
boolean book(int start, int end) Returns true if the event can be added to the calendar 
successfully without causing a double booking. Otherwise, return false and 
do not add the event to the calendar.

Example 1:

Input
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]
Output
[null, true, false, true]

0 <= start < end <= 10**9
At most 1000 calls will be made to book.
"""

# method 1: for each new interval, compare with each existing interval and see whether overlap O(n)
# method 2: use binary search, O(logn), but insertion still takes O(n)

""" Note on bisect
bisect.bisect_left
The returned insertion point i partitions the array a into two halves so that 
all(val < x for val in a[lo : i]) for the left side and 
all(val >= x for val in a[i : hi]) for the right side.

bisect.bisect
The returned insertion point i partitions the array a into two halves so that 
all(val <= x for val in a[lo : i]) for the left side and 
all(val > x for val in a[i : hi]) for the right side.
"""
# binary search tree sol in Python
# https://leetcode.com/problems/my-calendar-i/discuss/109476/Binary-Search-Tree-python
import bisect
intervals = [[10, 20], [15, 25]]
start, end  = 20, 30
class MyCalendar:
    def __init__(self):
        self.d = dict()  # map from end to start
        #self.starts = []
        self.ends = []  # save end for binary search

    def book(self, start: int, end: int) -> bool:
        position = bisect.bisect_left(self.ends, start)
        if position < len(self.ends) and self.d[self.ends[position]] <= end-1:
            # becasue [start, end)  -> [start, end-1]
            return False
        
        else:
            self.d[end-1] = start
            bisect.insort(self.ends, end-1)  # insert [start, end-1] 
            return True


sol=MyCalendar()
sol.book(10, 20)
sol.book(15, 25)
sol.book(20, 30)

"""Tree map solution
class MyCalendar {
public:
    MyCalendar() {}
    
    bool book(int start, int end) {
        auto it = cal.lower_bound(start);
        if (it != cal.end() && it->first < end) return false;
        if (it != cal.begin() && prev(it)->second > start) return false;
        cal[start] = end;
        return true;
    }

private:
    map<int, int> cal;
};
"""
from sortedcontainers import SortedDict
import bisect
# https://stackoverflow.com/questions/51800639/complexity-of-deleting-a-key-from-python-ordered-dict
class MyCalendar:
    def __init__(self):
        self.treemap = SortedDict()        

    def book(self, start: int, end: int) -> bool:
        if start in self.treemap:
            return False
        keys = self.treemap.keys() # sorted, O(1)
        floor = bisect.bisect(keys, start) - 1  # keys[i] < start for i <= floor
        ceiling = floor + 1
        if floor >= 0:
            # [s, e) (floor), [start, end), [s, e) (ceiling)
            s, e = keys[floor], self.treemap[keys[floor]]
            if e > start: return False
        
        if ceiling < len(self.treemap):
            s, e = keys[ceiling], self.treemap[keys[ceiling]]
            if s < end: return False
        
        self.treemap[start] = end # O(1)
        return True


"""981. Time Based Key-Value Store
Design a time-based key-value data structure that can store multiple values for the 
same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

TimeMap() Initializes the object of the data structure.
void set(String key, String value, int timestamp) Stores the key "key" with the value "value"
 at the given time timestamp.
String get(String key, int timestamp) Returns a value such that set was called previously,
 with timestamp_prev <= timestamp. If there are multiple such values, 
 it returns the value associated with the largest timestamp_prev. 
 If there are no values, it returns "".

Example 1:
Input
["TimeMap", "set", "get", "get", "set", "get", "get"]
[[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
Output
[null, null, "bar", "bar", null, "bar2", "bar2"]
"""
from collections import defaultdict
class TimeMap:
    def __init__(self):
        self.dic = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append([timestamp, value])

    def get(self, key: str, timestamp: int) -> str:
        arr = self.dic[key]
        n = len(arr)
        l, r = 0, n
        # find first after timestamp, then l-1
        while l < r:
            mid = l + (r-l) // 2
            if arr[mid][0] <= timestamp:
                l = mid + 1
            else:
                r = mid
        
        return "" if l == 0 else arr[l-1][1]


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

"""846. Hand of Straights
Alice has some number of cards and she wants to rearrange the cards into 
groups so that each group is of size groupSize, and consists of groupSize consecutive cards.

Given an integer array hand where hand[i] is the value written on the ith 
card and an integer groupSize, return true if she can rearrange the cards, or false otherwise.

Example 1:
Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]

Example 2:
Input: hand = [1,2,3,4,5], groupSize = 4
Output: false
Explanation: Alice's hand can not be rearranged into groups of 4.
"""
# O(MlogM + MW) where M is the num of unique card, W is groupSize
from collections import Counter
class Solution:
    def isNStraightHand(self, hand, groupSize: int) -> bool:
        n = len(hand)
        if n % groupSize != 0:
            return False
        m = Counter(hand)
        for i in sorted(m):
            if m[i] > 0:
                cur = m[i]  # need to copy first! 
                for j in range(groupSize):
                    m[i+j] = m[i+j] - cur
                    if m[i+j] < 0:
                        return False
        return True

# Follow Up: We just got lucky AC solution. Because groupSize <= 10000.
# What if groupSize is huge, should we cut off card on by one? ???
from collections import Counter, deque
class Solution:
    def isNStraightHand(self, hand, W):
        c = Counter(hand)
        print(sorted(c.items()))
        q = deque() # stores number of start cards need to be handled
        last_checked, opened = -1, 0 
        # opened: previous head card
        # last_checked: previous card number
        for i in sorted(c.keys()):
            print(q)
            if opened > c[i] or (opened > 0 and i > last_checked + 1): 
                # previous is 1->5, if current is 2->4 or next card is 3
                return False
            q.append((i, c[i] - opened))
            # subtract open card number, e.g., prev 1->5 and curr 2->6, 
            # then only 2->1 needs to be handled later
            last_checked, opened = i, c[i]
            if len(q) == W: 
                # len of q will never exceed W
                opened -= q.popleft()[1]
        return opened == 0

sol=Solution()
sol.isNStraightHand([1,2,3,6,2,3,4,7,8], 3)


""" 480. Sliding Window Median
The median is the middle value in an ordered integer list. If the size of the list is even, 
there is no middle value. So the median is the mean of the two middle values.

For examples, if arr = [2,3,4], the median is 3.
For examples, if arr = [1,2,3,4], the median is (2 + 3) / 2 = 2.5.
You are given an integer array nums and an integer k. There is a sliding window of size k 
which is moving from the very left of the array to the very right. You can only see 
the k numbers in the window. Each time the sliding window moves right by one position.

Return the median array for each window in the original array. Answers within 10-5 of 
the actual value will be accepted.

Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [1.00000,-1.00000,-1.00000,3.00000,5.00000,6.00000]
Explanation: 
Window position                Median
---------------                -----
[1  3  -1] -3  5  3  6  7        1
 1 [3  -1  -3] 5  3  6  7       -1
 1  3 [-1  -3  5] 3  6  7       -1
 1  3  -1 [-3  5  3] 6  7        3
 1  3  -1  -3 [5  3  6] 7        5
 1  3  -1  -3  5 [3  6  7]       6

Example 2:
Input: nums = [1,2,3,4,2,3,1,4,2], k = 3
Output: [2.00000,3.00000,3.00000,3.00000,2.00000,3.00000,2.00000]
"""
# https://zxi.mytechroad.com/blog/difficulty/hard/leetcode-480-sliding-window-median/
# brutal force: O(n*klogk), need to sort window every time
# insertion sort :  O(k*logk+n*k), keep a sorted window, takes O(k) to insert and maintain
# BST, multiset:  O(k*logk + nlogk)
import bisect 
class Solution:
    def medianSlidingWindow(self, nums, k: int):
        res = []
        n = len(nums)
        window = sorted(nums[:k])
        median = (window[(k-1)//2] + window[k//2]) / 2.0
        res.append(median)
        for i in range(k, n):
            old_position = bisect.bisect_left(window, nums[i-k]) # left <= x < right
            window.pop(old_position)
            bisect.insort_left(window, nums[i])
            median = (window[(k-1)//2] + window[k//2]) / 2.0
            res.append(median)
        return res

nums = [1,2,3,4,2,3,1,4,2]; k = 3
sol=Solution()
sol.medianSlidingWindow(nums, k)
sol.medianSlidingWindow([1,3,-1,-3,5,3,6,7], 3)

"""Solution 2:
use a SortedList structure, which was implemented using self-balanced tree.
SortedList enables O(logk) add and O(logk) remove. So the total time complexity is O(nlogk).
"""
from sortedcontainers import SortedList
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        lst = SortedList()    # maintain a sorted list
        res = []
        for i in range(len(nums)):
            lst.add(nums[i])            # O(logk)
            if len(lst) > k:
                lst.remove(nums[i-k])   # if we use heapq here, it takes O(k) here, but for sortedList, it takes O(logk)
            if len(lst) == k:
                median = lst[k//2] if k%2 == 1 else (lst[k//2-1] + lst[k//2]) / 2
                res.append(median)
        return res

# write the binary insertion yourself
class Solution:
    def medianSlidingWindow(self, nums, k: int):
        res = []
        n = len(nums)
        window = sorted(nums[:k])
        median = (window[(k-1)//2] + window[k//2]) / 2.0
        res.append(median)
        for i in range(k, n):
            old_position = self.binary_search(window, nums[i-k]) # find where old x is
            window.pop(old_position)
            window = self.binary_insert(window, nums[i]) #  left <= x < right
            median = (window[(k-1)//2] + window[k//2]) / 2.0 
            res.append(median)
        return res
    
    def binary_search(self, window, x):
        # find location in nums such that left<=x<right (loc+1 is the first < x)
        l, r = 0, len(window)
        while l < r:
            mid = l + (r-l) // 2
            if window[mid] == x:
                return mid
            elif window[mid] < x:
                l = mid+1
            else:
                r = mid
        return None
    
    def binary_insert(self, window, x):
        l, r = 0, len(window)
        find = False
        while l < r:
            mid = l + (r-l) // 2
            if window[mid] == x:
                find=True
                break
            elif window[mid] < x:
                l = mid+1
            else:
                r = mid
        mid = mid if find else l  # insert x at mid
        window.append(0)
        j = len(window) - 1
        while j > mid:
            window[j] = window[j-1]
            j -= 1
        window[j] = x
        return window
            

"""315. Count of Smaller Numbers After Self  ???
You are given an integer array nums and you have to return a new counts array. 
The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

Example 1:
Input: nums = [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.

Idea: 用二分法插入到一个新的数组，这样新数组就是有序的，那么此时该数字在新数组中的坐标就是原数组中其右边所有较小数字的个数
"""
# Solution 1: insertion sort (O(n^2))
class Solution:
    def countSmaller(self, nums):
        sorted_values, out = [], []
        for x in nums[::-1]:
            idx = bisect.bisect_left(sorted_values, x)
            sorted_values.insert(idx, x)
            out.append(idx)
        return out[::-1]
        

# Solution 2: merge sort O(nlogn)
# https://leetcode.com/problems/count-of-smaller-numbers-after-self/discuss/76607/C%2B%2B-O(nlogn)-Time-O(n)-Space-MergeSort-Solution-with-Detail-Explanation
# sort (index, value) pairs. The value is used for the sorting and the index is used for tracking the jumps.
def countSmaller(self, nums):
    def mergesort(enum):
        half = len(enum) // 2
        if half > 0:
            # count smaller element for left and right
            left, right = mergesort(enum[:half]), mergesort(enum[half:])  
            # left[1] = [1, 2, 5, 6]; right[1] = [3, 4, 8] 
            # when updating value of 5 (see how many elems in right smaller than 5), 
            # count how manny still in right after poping all elems > 5
            # only index on left will be updated

            # option 1
            # for i in range(len(enum))[::-1]:
            #     if len(right) == 0 or (len(left) > 0 and left[-1][1] > right[-1][1]):
            #         # left last element > right
            #         smaller[left[-1][0]] += len(right) # all smaller in right should be counted
            #         enum[i] = left.pop()  # last element
            #     else:
            #         enum[i] = right.pop()  # last element

            # option 2
            ileft, iright = 0, 0 # start with beginning index of left and right
            for i in range(len(enum)):
                if ileft <= iright:
                    enum[i] = left[ileft][1] # value
                    smaller[left[ileft][0]] += iright # smaller update based on index
                    ileft += 1
                else:
                    enum[i] = right[iright][1] # right side, no smaller update needed
                    iright += 1

        return enum
    
    smaller = [0] * len(nums)  # smaller count, based on index
    mergesort(list(enumerate(nums)))  # index, value
    return smaller


"""
############################################################################
动态规划（Dynamic Programming）
############################################################################
常见的题目包括找最大最小，找可行性，找总方案数等
"""

"""674. Longest Continuous Increasing Subsequence
Given an unsorted array of integers nums, return the length of the 
longest continuous increasing subsequence (i.e. subarray). The subsequence must be strictly increasing.

A continuous increasing subsequence is defined by two indices l and r (l < r) such that it is 
[nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] and for each l <= i < r, nums[i] < nums[i + 1].

Example 1:
Input: nums = [1,3,5,4,7]
Output: 3
Explanation: The longest continuous increasing subsequence is [1,3,5] with length 3.
Even though [1,3,5,7] is an increasing subsequence, it is not continuous as elements 5 and 7 are separated by element
4.

Example 2:
Input: nums = [2,2,2,2,2]
Output: 1
Explanation: The longest continuous increasing subsequence is [2] with length 1. Note that it must be strictly
increasing.
"""
# Idea: 使用一个计数器，如果遇到大的数字，计数器自增1；如果是一个小的数字，则计数器重置为1。
# 用一个变量 cur 来表示前一个数字，初始化为整型最大值，当前遍历到的数字 num 就和 cur 比较就行了，
# 每次用 cnt 来更新结果 res
"""
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int res = 0, cnt = 0, cur = INT_MAX;
        for (int num : nums) {
            if (num > cur) ++cnt;
            else cnt = 1;
            res = max(res, cnt);
            cur = num;
        }
        return res;
    }
};
"""
class Solution(object):
    def findLengthOfLCIS(self, nums):
        if not nums:
            return 0
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                dp[i] = dp[i - 1] + 1
        return max(dp)


"""300 Longest Increasing Subsequence (接龙型dp)
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements 
without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence 
of the array [0,3,1,6,2,2,7].

Example 1:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

Example 2:
Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:
Input: nums = [7,7,7,7,7,7,7]
Output: 1
"""
# Solution 1: n^2
# dp[i]: length of LIS ends with nums[i]
# dp[i] = max(dp[i], dp[j] + 1) for 0<=j<i and nums[j] < nums[i]
# if we need to print the LIS, then dp[i] needs to save the seq instead of a number
"""
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        if (nums.empty()) return 0;
        int n = nums.size();
        auto f = vector<int>(n, 1);
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < i; ++j)
                if (nums[i] > nums[j])
                    f[i] = max(f[i], f[j] + 1);
        return *max_element(f.begin(), f.end());
    }
};
"""

# Solution 2: patience sort: nlogn. 
# if need to return the LIS, then a single dp array is not enough
# https://www.youtube.com/watch?v=l2rCz7skAlk

# dp[i]: the smallest ending number of a subseq that has length i+1
# dp is increasing , use binary search
# nums = [3,4,1,2,8,5,6]
# num: 3,     4,  1,      2,     8,      5,       6,
# dp: [3], [3,4], [1, 4],[1,2], [1,2,8],[1,2,5], [1,2,5,6]
# res = len(dp)
# https://www.youtube.com/watch?v=l2rCz7skAlk
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        d = []
        for x in nums:
            if len(d) == 0 or x > d[-1]:
                d.append(x)
            else:
                i = bisect_left(d, x)
                d[i] = x
        return len(d)


"""673. Number of Longest Increasing Subsequence
Given an integer array nums, return the number of longest increasing subsequences.
Notice that the sequence has to be strictly increasing.

Example 1:
Input: nums = [1,3,5,4,7]
Output: 2
Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].

Example 2:
Input: nums = [2,2,2,2,2]
Output: 5
Explanation: The length of longest continuous increasing subsequence is 1, 
and there are 5 subsequences' length is 1, so output 5.

** Follow up: what if you need to print all the LIS? 

Idea 1: n^2
将 dp[i] 定义为以 nums[i] 为结尾的递推序列的个数
用 len[i] 表示以 nums[i] 为结尾的递推序列的长度

In the Longest Increasing Subsequence problem, the DP array simply had to store the longest length. 
In this variant, each element in the DP array needs to store two things: 
(1) Length of longest subsequence ending at this index and 
(2) Number of longest subsequences that end at this index. 
I use a two element list for this purpose.
In each loop as we build up the DP array, find the longest length for this index and then 
sum up the numbers at these indices that contribute to this longest length.
https://leetcode.com/problems/number-of-longest-increasing-subsequence/discuss/107320/Python-DP-with-explanation-(Beats-88)
"""
class Solution_DP:
    def findNumberOfLIS(self, nums) -> int:
        n = len(nums)
        if n == 0:
            return 0
        dp = [1]*n  # cnt
        lens = [1]*n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if lens[i] == lens[j] + 1:
                        dp[i] += dp[j]
                    if lens[i] < lens[j] + 1:
                        lens[i] = lens[j] + 1
                        dp[i] = dp[j]

        max_lens = max(lens)
        max_idx = [i for i,l in enumerate(lens) if l==max_lens]
        return sum(dp[i] for i in max_idx)

# follow up
def PrintLIS(nums):
    n = len(nums)
    if n == 0:
        return 0
    dp = [1]*n  # cnt
    lens = [1]*n
    lis_list = [[[num]] for num in nums]  # all LIS ends with num[i]
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lens[i] == lens[j] + 1:
                    dp[i] += dp[j]
                    lis_list[i] += [lis+[nums[i]] for lis in lis_list[j]]
                if lens[i] < lens[j] + 1:
                    lens[i] = lens[j] + 1
                    dp[i] = dp[j]
                    lis_list[i] = [lis+[nums[i]] for lis in lis_list[j]]

    max_lens = max(lens)
    max_idx = [i for i,l in enumerate(lens) if l==max_lens]
    res = []
    nested_res = [lis_list[i] for i in max_idx]
    for list_list in nested_res:
        for curr_lis in list_list:
            res.append(curr_lis)
    return res


nums = [1,3,5,4,7, 3, 10]
PrintLIS(nums)

"""Idea 2: Patience sort nlogn ???
https://leetcode.com/problems/number-of-longest-increasing-subsequence/discuss/916196/Python-Short-O(n-log-n)-solution-beats-100-explained

The idea is to keep several decks, where numbers in each deck are decreasing. 
Also when we see new card, we need to put it to the end of the leftest possible deck. 
Also we have paths: corresponing number of LIS, ending with given num. 
That is in paths[0] we keep number of LIS with length 1, in paths[k] we keep number 
of LIS with length k+1. (we keep cumulative sums) Also we keep ends_decks list to 
have quick access to end of our decks.
"""
# https://leetcode.com/problems/number-of-longest-increasing-subsequence/discuss/916196/Python-Short-O(n-log-n)-solution-beats-100-explained 
import bisect
class Solution:
    def findNumberOfLIS(self, nums):
        if not nums: return 0
        decks, ends_decks, paths = [], [], []
        for num in nums:
            deck_idx = bisect.bisect_left(ends_decks, num) # idx to insert current card
            n_paths = 1
            if deck_idx > 0:
                l = bisect.bisect(decks[deck_idx-1], -num)
                n_paths = paths[deck_idx-1][-1] - paths[deck_idx-1][l]
                
            if deck_idx == len(decks):
                decks.append([-num])
                ends_decks.append(num)
                paths.append([0,n_paths])
            else:
                decks[deck_idx].append(-num)
                ends_decks[deck_idx] = num
                paths[deck_idx].append(n_paths + paths[deck_idx][-1])
        print(decks)
        print(ends_decks)
        print(paths)
        return paths[-1][-1]

sol=Solution()
sol.findNumberOfLIS( [1,3,0,5,4,9, 7])
sol.findNumberOfLIS( [1,3,5,4])

"""63. Unique Paths II
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
The robot can only move either down or right at any point in time. The robot is trying to 
reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?
An obstacle and space is marked as 1 and 0 respectively in the grid.

Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Input: obstacleGrid = [[0,1],[0,0]]
Output: 1
"""
# DP 
# dp[i][j]: num of unique paths to (i,j)
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        dp = [[0 for _ in range(len(obstacleGrid[0])+1)] for _ in range(len(obstacleGrid)+1)]
        m, n = len(dp), len(dp[0])
        dp[0][1] = 1  # initialize to make sure dp[1][1] = 1!
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i-1][j-1] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]

# you may also use BFS+dp[][], but slower than dp
# DFS + memo also works 


"""368. Largest Divisible Subset
Given a set of distinct positive integers nums, return the largest 
subset answer such that every pair (answer[i], answer[j]) of elements in this subset satisfies:

answer[i] % answer[j] == 0, or
answer[j] % answer[i] == 0
If there are multiple solutions, return any of them.

Example 1:
Input: nums = [1,2,3]
Output: [1,2]
Explanation: [1,3] is also accepted.

Example 2:
Input: nums = [1,2,4,8]
Output: [1,2,4,8]
"""
class Solution:
    def largestDivisibleSubset(self, nums):
        nums = sorted(nums)
        n = len(nums)
        lds = [[num] for num in nums] # lds ends with nums[i]
        lens = [1] * n  # lens ens with nums[i]
        for i in range(1, n):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    # nums[i]>nums[j] so nums[j]%nums[i]=0
                    # if lens[i] == lens[j] + 1:
                    #     continue
                    if lens[i] < lens[j] + 1:
                        lens[i] = lens[j] + 1
                        lds[i] = lds[j] + [nums[i]]
        
        return max(lds, key=len)

# print all the 
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums = sorted(nums)
        n = len(nums)
        lds = [[[num]] for num in nums] # lds ends with nums[i]
        lens = [1] * n  # lens ens with nums[i]
        for i in range(1, n):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    # nums[i]>nums[j] so nums[j]%nums[i]=0
                    if lens[i] == lens[j] + 1:
                        lds[i] += [lds_j + [nums[i]] for lds_j in lds[j]]
                    if lens[i] < lens[j] + 1:
                        lens[i] = lens[j] + 1
                        lds[i] = [lds_j + [nums[i]] for lds_j in lds[j]]
        
        max_lens = max(lens)
        max_idx = [i for i,l in enumerate(lens) if l==max_lens]
        res = []
        nested_res = [lds[i] for i in max_idx]
        for list_list in nested_res:
            for curr_lis in list_list:
                res.append(curr_lis)
        return res


"""354. Russian Doll Envelopes
You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the 
width and the height of an envelope.

One envelope can fit into another if and only if both the width and height of one envelope 
are greater than the other envelope's width and height.

Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).

Note: You cannot rotate an envelope.

Example 1:
Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).

Example 2:
Input: envelopes = [[1,1],[1,1],[1,1]]
Output: 1
"""
# Solution 1: n^2
# sort decreasing, and dp[i]: max number if env[i] is the smallest envelop
# dp = [1]^n
# dp[i] = dp[j] + 1 if envelops[i] can fit into env[j], for 0<=j<i
# return max(dp)
# skip

# Solution 2: nlogn, similar to length of LIS
# 信封的宽度还是从小到大排，但是宽度相等时，我们让高度大的在前面 (w,h)
# In this way, if envs[i+1][height] > envs[i][height] then envs[i+1][width] must bigger
# use dp[i] to save the last ("largest") envelop of length i
# keep dp[i] be as small as possible 
envelopes = [[5,4],[5,5],[6,7],[2,3],[6,3],[7,5]]
# [2,3], [5,5], [5,4], [6,7], [6,3], [7,5]
# [2,3] -> [2,3], [5, 5] -> [2,3], [5,4], [6,7] -> [6,3], [5,4], [6,7] -> [6,3], [5,4], [7,5]
# core is to keep width as small as possible because weight already sorted
class Solution:
    def maxEnvelopes(self, envelopes) -> int:
        dp = []
        # only possible for early fit into latter
        envelopes = sorted(envelopes, key=lambda x: (x[0], -1*x[1])) 
        for env in envelopes:
            self.insert_env(dp, env) # smallest env can hold curr
        return len(dp)

    def insert_env(self, dp, env):
        n = len(dp)
        # find idx of last env whose height is smaller (cannot equal) than env, put it at idx+1
        # smallest env has a larger or equal h
        # if equal h, then w must be smaller due to sort
        l, r = 0, n
        while l < r:
            mid = l + (r-l) // 2
            if dp[mid][1] < env[1]:
                l = mid + 1
            else:
                r = mid
        if r == n:
            dp.append(env)
        else:
            dp[l] = env
        return None


"""[LeetCode] 256. Paint House 粉刷房子
There are a row of n houses, each house can be painted with one of the three colors: red, blue or green. 
The cost of painting each house with a certain color is different. You have to paint all the houses such 
that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by a n x 3 cost matrix. 
For example, costs[0][0] is the cost of painting house 0 with color red; costs[1][2] is the cost 
of painting house 1 with color green, and so on... Find the minimum cost to paint all houses.

Note:
All costs are positive integers.

Example:
Input: costs = [[17,2,17],[16,16,5],[14,3,19]]
Output: 10
Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 into blue. 
             Minimum cost: 2 + 5 + 3 = 10.
"""
# dp_red[i]: min cost of painint first i, using red for ith
# dp_blue[i]
# dp_green[i]
# may use dp[i][j] as a general (j is color idx)

# initial: dp[0][j] = costs[0][j]
# iteration formular: 
# dp[i+1][j] = min(dp[i][k] + costs[i+1][j]), for k in list(range(num_colr))-[k])


"""45 Jump Game II
Given an array of non-negative integers nums, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Your goal is to reach the last index in the minimum number of jumps.
You can assume that you can always reach the last index.

Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [2,3,0,1,4]
Output: 2
"""
# solution 1: dp[i]: denotes minimum jumps required from current index to reach till the end.
# initial: dp = [inf]*n, dp[0]=1
# iteration: dp[i] = min(dp[j] + 1) if nums[i] >= i-j for j in range(i,n)
# O(n^2)

# solution 2: Gready: O(n)
# We can iterate over all indices maintaining the furthest reachable position from current index - maxReachable 
#  and currently furthest reached position - lastJumpedPos. 
# Everytime we will try to update lastJumpedPos to furthest possible reachable index - maxReachable.
class Solution:
    def jump(self, nums) -> int:
        if len(nums) == 1:
            # can del this by initialize res, lastJumpedPos, maxReachable with 0
            return 0
        lastJumpedPos = nums[0]
        maxReachable = nums[0]
        res = 1
        i = 0
        for i in range(len(nums)):
            if i > lastJumpedPos:
                # if i goes beyond current longest step, res+1
                res += 1
                lastJumpedPos = maxReachable
            maxReachable = max(maxReachable, i+nums[i])
            if lastJumpedPos >= len(nums) - 1:
                return res
            # if i>maxReachable:
            #     return -1  # if not reachable
        
        return -1


"""132. Palindrome Partitioning II
Given a string s, partition s such that every substring of the partition is a palindrome.
Return the minimum cuts needed for a palindrome partitioning of s.

Example 1:
Input: s = "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.

Example 2:
Input: s = "a"
Output: 0

Example 3:
Input: s = "ab"
Output: 1
"""
# This can be solved by two points:
# cut[j]: minimum cuts needed for a palindrome partitioning of s[:j+1]
# cut[j] is the minimum of cut[i - 1] + 1 (i <= j), if [i, j] is palindrome.
# If [j, i] is palindrome, [j + 1, i - 1] is palindrome, and c[j] == c[i].
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        dp = list(range(n))
        pal = [[True for _ in range(n)] for _ in range(n)]  # pal[i][j] whether s[i:j+1] is pal
        # first calculate pal matrix
        # or may use lru_cache trick
        for j in range(n):
            for i in range(j+1):
                # i = 0,...,j-1
                pal[i][j] = pal[i+1][j-1] and (s[i]==s[j]) if i<j else True
        
        for j in range(n):
            if pal[0][j]:
                # s[:j+1]
                dp[j] = 0
                continue
            for i in range(j+1):
                # i = 0, ..., j
                if pal[i][j]:
                    # if s[i:j+1] is palindrome
                    # 1) i==0: s[:j+1] is palindrome, min cut is 0 
                    # 2) i!=0, then min of cut for s[:i+1] + 1, because s[i:j+1] is palindrome
                    dp[j] = min(dp[j], dp[i-1] + 1) 
        
        return dp[n-1]

sol=Solution()
sol.minCut(s)

# solution 2 use cache trick, TLE
from functools import lru_cache
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

# Solution 3: DFS+memo, TLE
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
# may not need such pal table
"""
Use two variables in two loops to represent a palindrome:
The external loop variable 'i' represents the center of the palindrome. 
The internal loop variable 'j' represents the 'radius' of the palindrome. Apparently, j <= i is a must.
This palindrome can then be represented as s[i-j, i+j]. If this string is indeed a palindrome, 
then one possible value of cut[i+j] is cut[i-j] + 1, where cut[i-j] corresponds to s[0, i-j-1] and 
1 correspond to the palindrome s[i-j, i+j];
class Solution {
public:
    int minCut(string s) {
        int n = s.size();
        vector<int> cut(n+1, 0);  // number of cuts for the first k characters
        for (int i = 0; i <= n; i++) cut[i] = i-1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; i-j >= 0 && i+j < n && s[i-j]==s[i+j] ; j++) // odd length palindrome
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j]);

            for (int j = 1; i-j+1 >= 0 && i+j < n && s[i-j+1] == s[i+j]; j++) // even length palindrome
                cut[i+j+1] = min(cut[i+j+1],1+cut[i-j+1]);
        }
        return cut[n];
    }
};
"""

"""1278. Palindrome Partitioning III
You are given a string s containing lowercase letters and an integer k. You need to:

First, change some characters of s to other lowercase English letters.
Then divide s into k non-empty disjoint substrings such that each substring is a palindrome.
Return the minimal number of characters that you need to change to divide the string.

Example 1
Input: s = "abc", k = 2
Output: 1
Explanation: You can split the string into "ab" and "c", and change 1 character in "ab" to make it palindrome.

Example 2:
Input: s = "aabbc", k = 3
Output: 0
Explanation: You can split the string into "aa", "bb" and "c", all of them are palindrome.

Example 3:
Input: s = "leetcode", k = 8
Output: 0
"""
# dp[i][k] : minimal number of char to change to divide s[:i+1] into k+1 strings
# dp[i][0] = cost(s[:i+1]) # cost of change s[:i+1] into palindrome
# dp[i][k] = min(dp[j][k-1] + cost(s[j+1:i+1])) for j=0,...i-1
# to save time, create costs[i][j]: cost of changing s[i:j+1] into palindrome
# n^2 * k
class Solution:
    def palindromePartition(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[float('Inf') for _ in range(k)] for _ in range(n)]
        costs = [[0 for _ in range(n)] for _ in range(n)]

        for j in range(n):
            for i in range(j)[::-1]:
                # cost=0 when i=j
                # s[i:j+1]
                costs[i][j] = costs[i+1][j-1] if s[i]==s[j] else costs[i+1][j-1]+1

        for i in range(n):
            dp[i][0] = costs[0][i]  # divide into 1 string
            for ki in range(1, k):
                for j in range(i):
                    # dp[j][k-1] + cost(s[j+1:i])
                    dp[i][ki] = min(dp[i][ki], dp[j][ki-1] + costs[j+1][i])
        
        return dp[n-1][k-1]


"""813. Largest Sum of Averages
You are given an integer array nums and an integer k. You can partition the array into at most k non-empty adjacent subarrays. 
The score of a partition is the sum of the averages of each subarray.
Note that the partition must use every integer in nums, and that the score is not necessarily an integer.

Return the maximum score you can achieve of all the possible partitions. Answers within 10-6 of the actual answer will be accepted.

Example 1:
Input: nums = [9,1,2,3,9], k = 3
Output: 20.00000
Explanation: 
The best choice is to partition nums into [9], [1, 2, 3], [9]. The answer is 9 + (1 + 2 + 3) / 3 + 9 = 20.
We could have also partitioned nums into [9, 1], [2], [3, 9], for example.
That partition would lead to a score of 5 + 2 + 6 = 13, which is worse.
"""
# dp[i][k] : LSA of nums[:i] for k subarrays
# dp[i][0] = 0
# dp[i][k] = max(dp[j][k-1] + avg(nums[j+1:i+1])), k-1<=j<i
# nums[j+1:i+1] = cumsums[i] - cumsums[j]
# can also use DFS+memo


"""312. Burst Balloons
You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on 
it represented by an array nums. You are asked to burst all the balloons.

If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins. 
If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.

Example 1:
Input: nums = [3,1,5,8]
Output: 167
Explanation:
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
"""
# DFS+memo: TLE time ? 
from collections import defaultdict
class Solution:
    def maxCoins(self, nums) -> int:
        memo = defaultdict(list)
        adj_nums = [1] + nums + [1]
        return self.dfs(tuple(adj_nums), memo)

    def dfs(self, nums, memo):
        if len(nums) < 3:
            # head and tail are fake numes
            return 0
        if nums in memo:
            return memo[nums]
        res = 0
        for i in range(1, len(nums)-1):
            # cannot burst head and tail
            res = max(res, self.dfs(nums[:i] + nums[i+1:], memo) + nums[i]*nums[i-1]*nums[i+1])
        memo[nums] = res
        return res


sol=Solution()
sol.maxCoins(nums=[3,1,5,8])


# DP: dp[i][j] 表示打爆区间 [i,j] 中的所有气球能得到的最多金币
# if k is the last balloon in nums[i:j+1] to burst, then
# dp[i][j] = max(dp[i][j], nums[i - 1] * nums[k] * nums[j + 1] + dp[i][k - 1] + dp[k + 1][j]),  i ≤ k ≤ j 
# 遍历顺序: [3] -> [1] -> [5] -> [8] -> [3, 1] -> [1, 5] -> [5, 8] -> [3, 1, 5] -> [1, 5, 8] -> [3, 1, 5, 8]
# 所以两个loop是: len, i (len=j-i+1)
"""
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        nums.insert(nums.begin(), 1);
        nums.push_back(1);
        vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
        for (int len = 1; len <= n; ++len) {
            for (int i = 1; i <= n - len + 1; ++i) {
                int j = i + len - 1;
                for (int k = i; k <= j; ++k) {
                    dp[i][j] = max(dp[i][j], nums[i - 1] * nums[k] * nums[j + 1] + dp[i][k - 1] + dp[k + 1][j]);
                }
            }
        }
        return dp[1][n];
    }
};
"""


"""1143. Longest Common Subsequence
Given two strings text1 and text2, return the length of their longest common subsequence. 
If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters 
(can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.

Example 1:
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.

Example 2:
Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
"""
# similar to  Delete Operation for Two Strings
# dp[i][j] 表示 text1 的前i个字符和 text2 的前j个字符的最长相同的子序列的字符个数
# 
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1)+1, len(text2)+1
        dp = [[0 for _ in range(n)] for _ in range(m)]
        # dp[i][j]: text1[:i] and text2[:j]
        for i in range(1,m):
            for j in range(1,n):
                if text1[i-1] == text2[j-1]: 
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m-1][n-1]

sol=Solution()
sol.longestCommonSubsequence(text1 = "abcde", text2 = "ace")


"""Maximum Length of Repeated Subarray
Given two integer arrays nums1 and nums2, return the maximum length of a subarray that appears in both arrays.

Example 1:
Input: nums1 = [1,2,3,2,1], nums2 = [3,2,1,4,7]
Output: 3
Explanation: The repeated subarray with maximum length is [3,2,1].

Example 2:
Input: nums1 = [0,0,0,0,0], nums2 = [0,0,0,0,0]
Output: 5

# Use dynamic programming. dp[i][j] will be the answer for inputs A[i:], B[j:].
其中 dp[i][j] 表示数组A的前i个数字和数组B的前j个数字在尾部匹配的最长子数组的长度
如果 dp[i][j] 不为0，则A中第i个数组和B中第j个数字必须相等，且 dp[i][j] 的值就是往前推分别相等的个数
每次算出一个 dp 值，都要用来更新结果 res
  3 1 2
1 0 1 0
2 0 0 2
2 0 0 1
"""
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        m, n = len(nums1)+1, len(nums2)+1
        dp = [[0 for _ in range(n)] for _ in range(m)]  
        res = 0
        for i in range(1, m):
            for j in range(1, n):
                # nums1[:i], nums2[:j]
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    res = max(res, dp[i][j])
                else:
                    dp[i][j] = 0
        
        return res


"""1044. Longest Duplicate Substring
Given a string s, consider all duplicated substrings: (contiguous) substrings of s that occur 2 or more times. The occurrences may overlap.
Return any duplicated substring that has the longest possible length. If s does not have a duplicated substring, the answer is "".

Example 1:
Input: s = "banana"
Output: "ana"

Example 2:
Input: s = "abcd"
Output: ""

s consists of lowercase English letters.
"""
# Solution 1: dp[i][j]: longest length of overlap for strings endwith with s[i] and s[j]
# dp[i][j] = 0 if s[i]!=s[j] else 1+dp[i-1][j-1]. for 0<=i<j<=n-1
# res = max(dp)
# TLE


# Solution 2: nlogn: 
# for a sliding window with a fixed length K, it takes O(n) to find whether duplicate (hash set)
# we use binary search to find such max K
# Pay attention that it is non trivial to push a very long string into hash!
# need to use  Rabin-Karp algorithm to convert string into an integer
# 编码的方法是用 26 进制，因为限制了都是小写字母，为了防止整型溢出，需要对一个超大的质数取余
# abce -> (1*26^1 + 2*26^2 + 3*26^3 + 5*26^4) % m, m=1e7+7
# May use double hash to further decrease the chance
s="ttwphlndxvcruhoaapgcfovcqopxbyzcidwhbwmpbdaiyanfhotksdvamvtpzvvugyr"
# https://www.youtube.com/watch?v=N7EE0VamNqc
# https://leetcode.com/problems/longest-duplicate-substring/discuss/290871/Python-Binary-Search
# TLE
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        n = len(s)
        l, r = 0, n
        while l < r:
            mid = l + (r-l) // 2
            dup_str = find_duplicate(s, mid, n)
            if len(dup_str) != 0:
                l = mid + 1
            else:
                r = mid
        return find_duplicate(s, l-1, n)

def find_duplicate(s, window_size, n):
    # return "" if no duplicate at window_size
    if window_size<=0:
        return ""
    m = 2**63 - 1
    set0 = set()
    curr_s = s[:window_size]
    set0.add(s2i(curr_s, m, window_size))
    for i in range(window_size, n):
        curr_s = curr_s[1:] + s[i]
        value = s2i(curr_s, m, window_size)
        if value in set0:
            return curr_s
            # print(curr_s)
        else:
            set0.add(value)
    return ""

def s2i(s, m, n):
    res = 0
    for i in range(n):
        res = (res * 26 + (ord(s[i]) - ord('a'))) % m
    return res % m



"""174. Dungeon Game
The demons had captured the princess and imprisoned her in the bottom-right corner of a dungeon. 
The dungeon consists of m x n rooms laid out in a 2D grid. Our valiant knight was initially positioned 
in the top-left room and must fight his way through dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health 
point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health 
upon entering these rooms; other rooms are either empty (represented as 0) or contain magic orbs that 
increase the knight's health (represented by positive integers).

To reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.

Return the knight's minimum initial health so that he can rescue the princess.

Note that any room can contain threats or power-ups, even the first room the knight enters and the bottom-right 
room where the princess is imprisoned.

Input: dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
Output: 7
Explanation: The initial health of the knight must be at least 7 if he follows the optimal path: RIGHT-> RIGHT -> DOWN -> DOWN.

Greedy does not work - can find counter example [[1,-3,3],[0,-2,0],[-3,-3,-3]]

逆着往回推，骑士逆向进入房间后 PK 后所剩的血量就是骑士正向进入房间时 pk 前的起始血量。
所以用当前房间的右边和下边房间中骑士的较小血量减去当前房间的数字
dp[i][j]: min health needed when entering room[i][j]
dp[i][j] = min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j], go backward
create additional fake row and col to handle edge case (i=m-1 or j=n-1)
Pay attention that dp can never be lower than 1: for some room having positive num
"""
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m, n = len(dungeon), len(dungeon[0])
        dp = [[float('Inf') for _ in range(n+1)] for _ in range(m+1)]
        dp[m-1][n] = dp[m][n-1] = 1 # at least 1 health when leaving dp[m-1][n-1]
        for i in range(m)[::-1]:
            for j in range(n)[::-1]:
                # do not include the last row and column (fake)
                dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
        
        return dp[0][0]
        

"""115. Distinct Subsequences
Given two strings s and t, return the number of distinct subsequences of s which equals t.

A string's subsequence is a new string formed from the original string by deleting some 
(can be none) of the characters without disturbing the remaining characters' relative positions. 
(i.e., "ACE" is a subsequence of "ABCDE" while "AEC" is not).

The test cases are generated so that the answer fits on a 32-bit signed integer.

Example 1:
Input: s = "rabbbit", t = "rabbit"
Output: 3
Explanation:
As shown below, there are 3 ways you can generate "rabbit" from S.
rabbbit
rabbbit
rabbbit

Solution: DP
dp[i][j] 表示s中范围是 s[:i] 的子串中能组成t中范围是t[:j] 的子串的子序列的个数

transpose matrix: 
  Ø r a b b b i t (S)
Ø 1 1 1 1 1 1 1 1
r 0 1 1 1 1 1 1 1
a 0 0 1 1 1 1 1 1
b 0 0 0 1 2 3 3 3
b 0 0 0 0 1 3 3 3
i 0 0 0 0 0 0 3 3
t 0 0 0 0 0 0 0 3 
(t)

dp[i][j] = dp[i-1][j] + dp[i - 1][j - 1] if s[i-1] == t[j-1] else dp[i-1][j]
initialize with 0 if s is empty (t not empty), 1 else
"""
s = "rabbbit"; t = "rabbit"
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        if len(s) == 0:
            return 1 if len(t) == 0 else 0
        if len(t) == 0:
            return 1
        m, n = len(s), len(t)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0] = 1
        for j in range(1, n+1):
            dp[0][j] = 0
        for i in range(1, m+1):
            # len.t > len.s
            dp[i][0] = 1
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                # note the matrix above (in comment) should be transposed
                dp[i][j] = dp[i-1][j] + dp[i - 1][j - 1] if s[i-1] == t[j-1] else dp[i-1][j]
        
        return dp[m][n]

# DFS + cache trick, pass but 10%
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        return self.dfs(s,t,0,0)
    
    @lru_cache(None)
    def dfs(self,s, t, i, j):
        # ans for s[i:] and t[j:]
        if j == len(t):
            return 1
        if i == len(s):
            return 0
        if s[i] == t[j]:
            return self.dfs(s,t,i+1,j+1)+self.dfs(s,t,i+1,j)
        else:
            return self.dfs(s,t,i+1,j)

"""91. Decode Ways
A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode an encoded message, all the digits must be grouped then mapped back into letters 
using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a 32-bit integer.

Example 2:
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

Example 3:
Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is different from "06").
"""
class Solution:
    @lru_cache(None)
    def numDecodings(self, s: str) -> int:
        if len(s) > 0 and s[0] == '0':
            return 0
        if len(s) <= 1:
            return 1
        if s[:2] <= '26':
            return self.numDecodings(s[1:]) + self.numDecodings(s[2:])
        else:
            return self.numDecodings(s[1:])

sol=Solution()
sol.numDecodings("226")
sol.numDecodings("1023412342")  # 6
sol.numDecodings("301")
sol.numDecodings("12")

# dp[i]: s[:i+1]
# may simplify by add a fake start idx with dp[0]=1
class Solution:
    def numDecodings(self, s: str) -> int:
        if len(s) == 0:
            return 1
        if s[0] == '0':
            return 0
        n = len(s)
        if n == 1:
            return 1
        # initialize, 0 and 1
        dp = [0] * n
        dp[0] = 1
        if s[1] == '0':
            dp[1] = 1 if int(s[:2]) <= 26 else 0 # "30"
        else:
            dp[1] = 2 if int(s[:2]) <= 26 else 1 # "XXXX20"
        
        for i in range(2, n):
            if s[i] == '0':
                if int(s[i-1:i+1]) > 26 or s[i-1] == '0':
                    # "XXXX30" or "XXXX00"
                    dp[i] = 0
                else:
                    # "XXXX20"
                    dp[i] = dp[i-2]
            else:
                if int(s[i-1:i+1]) > 26 or s[i-1] == '0':
                    # "XXXX03" or "XXXX33"
                    dp[i] = dp[i-1]
                else:
                    dp[i] = dp[i-1] + dp[i-2]
        
        return dp[-1]


"""639. Decode Ways II
A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters 
using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:
"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

In addition to the mapping above, an encoded message may contain the '*' character, 
which can represent any digit from '1' to '9' ('0' is excluded). 
For example, the encoded message "1*" may represent any of the encoded messages 
"11", "12", "13", "14", "15", "16", "17", "18", or "19". Decoding "1*" is equivalent to decoding 
any of the encoded messages it can represent.

Given a string s consisting of digits and '*' characters, return the number of ways to decode it.

Since the answer may be very large, return it modulo 10**9 + 7.

Example 1:
Input: s = "*"
Output: 9

Example 2:
Input: s = "1*"
Output: 18
Explanation: The encoded message can represent any of the encoded messages "11", "12", "13", "14", "15", "16", "17", "18", or "19".
Each of these encoded messages have 2 ways to be decoded (e.g. "11" can be decoded to "AA" or "K").
Hence, there are a total of 9 * 2 = 18 ways to decode "1*".

Example 3:
Input: s = "2*"
Output: 15
Explanation: The encoded message can represent any of the encoded messages "21", "22", "23", "24", "25", "26", "27", "28", or "29".
"21", "22", "23", "24", "25", and "26" have 2 ways of being decoded, but "27", "28", and "29" only have 1 way.
Hence, there are a total of (6 * 2) + (3 * 1) = 12 + 3 = 15 ways to decode "2*".
"""
# hint: cannot list all possible due to large answeer. 
# dp[i]: num of ways for s[:i]  # leave dp[0] as a fake
# dp[0] = 0 if s[0]==0, =1 if s[0] is digit, =9 if s[0]=='*'
# dp[i+1] = dp[i] * c(s[i]) + dp[i-1] * c(s[i-1]+s[i])
# need to discuss c() based on different inputs:
# c(A) = 1 if 1<=A<=26, =0 else  ---  no metter one or two digits
# c('*') = 9; c('**') = 15; c('A*')=9 if A==1, =6 if A==2 else 0
# c('*A')=2 if 0<=A<=6 else 1
class Solution:
    def numDecodings(self, s: str) -> int:
        if len(s) == 0:
            return 1
        if s[0] == '0':
            return 0
        n = len(s)
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 9 if s[0]=='*' else (1 if int(s[0])>0 else 0)
        for i in range(2, n+1):
            dp[i] = dp[i-1] * helper(s[i-1]) + dp[i-2] * helper(s[i-2:i])
            dp[i] = dp[i] % (10**9 + 7)
        return dp[n]

def helper(s):
    if s[0] == '0':
        return 0
    if len(s) == 1:
        return 9 if s=='*' else 1
    if len(s) == 2:
        if s[0] == '*': # "*A" or "**"
            return 15 if s[1]=='*' else (2 if int(s[1]) <= 6 else 1)
        else: # "A*" or "AB"
            if s[1] == '*': 
                return 9 if s[0] == '1' else (6 if s[0] == '2' else 0)
            else:
                return 0 if int(s)>26 else 1

sol=Solution()
sol.numDecodings("226")
sol.numDecodings("1023412342")
sol.numDecodings("301")
sol.numDecodings("12")
# solution 2: dfs
from functools import lru_cache
class Solution:
    def numDecodings(self, s: str) -> int:
        @lru_cache(None)
        def dfs(s, idx):
            if idx == len(s):
                return 1
            if s[idx] == '0':
                return 0
            res = 0
            if s[idx] == '*':
                res += 9*dfs(s, idx+1)
                if idx < len(s) - 1:
                    if s[idx+1] == '*':
                        res += 15 * dfs(s, idx+2) # * cannot be 0!! 11~26 \ 20
                    elif s[idx+1] in ['7', '8', '9']:
                        # *7 - *9
                        res += dfs(s, idx+2)
                    else:
                        # *0 - *6
                        res += 2 * dfs(s, idx+2)
            else: # 1-9
                res += dfs(s, idx+1)
                if idx < len(s) - 1:
                    if s[idx+1] == '*':
                        if s[idx] == '1': # 1*
                            res += 9 * dfs(s, idx+2)
                        elif s[idx] == '2': # 2*
                            res += 6 * dfs(s, idx+2)
                    else:
                        if int(s[idx:idx+2]) <= 26:
                            res += dfs(s, idx+2)
            return res % (10**9 + 7)
        return dfs(s, 0)

sol=Solution()
sol.numDecodings("********")

# DFS + cache trick 2
# MLE (memory exceed limits)
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


"""House Robber II 打家劫舍之二
Note: This is an extension of House Robber.

After robbing those houses on that street, the thief has found himself a new place for his thievery so that he will not 
get too much attention. This time, all houses at this place are arranged in a circle. That means the first house is the 
neighbor of the last one. Meanwhile, the security system for these houses remain the same as for those in the previous street.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of 
money you can rob tonight without alerting the police.

Example 1:
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
"""
# hint: 这道题是之前那道House Robber 打家劫舍的拓展，现在房子排成了一个圆圈，则如果抢了第一家，就不能抢最后一家，因为首尾相连了，
# 所以第一家和最后一家只能抢其中的一家，或者都不抢，那我们这里变通一下，如果我们把第一家和最后一家分别去掉，各算一遍能抢的最大值，
# 然后比较两个值取其中较大的一个即为所求。
class Solution:
    def rob(self, nums) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        gain1 = self.helper(nums[:n-1])
        gain2 = self.helper(nums[1:])
        return max(gain1, gain2)
    
    def helper(self, A):
        if len(A) == 1: 
            return A[0]
        dp = [0] * len(A)
        # dp[i] will denote maximum loot that we can get by considering till ith index
        dp[0] = A[0]
        dp[1] = max(A[0], A[1])
        for i in range(2, len(A)):
            dp[i] = max(dp[i-1], A[i] + dp[i-2])
        return dp[-1]

# dfs + memo
from functools import lru_cache
class Solution:
    def rob(self, nums) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        @lru_cache(None)
        def dfs(i, j):
            # nums[:idx+1]
            if j < i:
                return 0
            return max(dfs(i, j-1), dfs(i, j-2) + nums[j])
        res1 = dfs(0, n-2)
        res2 = dfs(1, n-1)
        return max(res1, res2)


sol=Solution()
sol.rob(nums)


""" 712. Minimum ASCII Delete Sum for Two Strings
Given two strings s1 and s2, return the lowest ASCII sum of deleted characters to make two strings equal.

Example 1:
Input: s1 = "sea", s2 = "eat"
Output: 231
Explanation: Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.
Deleting "t" from "eat" adds 116 to the sum.
At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.

Example 2:
Input: s1 = "delete", s2 = "leet"
Output: 403
Explanation: Deleting "dee" from "delete" to turn the string into "let",
adds 100[d] + 101[e] + 101[e] to the sum.
Deleting "e" from "leet" adds 101[e] to the sum.
At the end, both strings are equal to "let", and the answer is 100+101+101+101 = 403.
If instead we turned both strings into "lee" or "eet", we would get answers of 433 or 417, which are higher.
"""
# similar to Delete Operation for Two Strings
# dp[i][j]: min delete for s1[:i+1] and s2[:j+1]

# dp[0][0] = 0
# dp[i][0] = dp[i-1][0] + ord(s1[i])
# dp[0][j] = dp[0][j-1] + ord(s2[j])

s1 = "delete"; s2 = "leet"
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        dp = [[0 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
        m, n = len(dp), len(dp[0])
        dp[0][0] = 0
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + ord(s1[i-1])

        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + ord(s2[j-1])
        
        for i in range(1, m):
            for j in range(1, n):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    v1 = dp[i][j-1] + ord(s2[j-1])
                    v2 = dp[i-1][j] + ord(s1[i-1])
                    dp[i][j] = min(v1, v2)
        
        return dp[m-1][n-1]
        

"""[LeetCode] Maximal Square 最大正方形
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing all 1's and return its area.
For example, given the following matrix:
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4
"""
# Solution 1: cumsum + DP, O(m*n*min(m,n)) -> O(m * n * log(min(m,n))) if using binary search
# use dp save cumsum of all elements in window of matrix[0][0]:matrix[i][j]
# loop through m*n as matrix[i][j] left top element, 
# for matrix[i][j], check the largest size such that: (can use binary seach to speed up)
# dp[i+size][j+size] - dp[i+size][j] - dp[i][j+size] + dp[i][j] == size*size
# note: dp size m+1 * n+1 where matrix is at size m*n, first row and col are set to be zero
# dp[i][j] is matrix ends with matrix[i-1][j-1]

# Solution 2: O(m*n)
# define dp[i][j] as max side size can achieve at (i,j) as bottom right
# dp[i][j] = 0 if matrix[i][j]==0
# dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1 if matrix[i][j]==1
class Solution:
    def maximalSquare(self, matrix) -> int:
        m, n = len(matrix), len(matrix[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        # dp[i][j] -> ends with matrix[i][j]
        res = 0
        for i in range(m):
            for j in range(n):
                curr_num = 1 if matrix[i][j] == '1' else 0
                if curr_num == 1:
                    if i==0 or j==0: # first row or col
                        dp[i][j] = curr_num
                    else:
                        dp[i][j] = min(min(dp[i-1][j-1], dp[i-1][j]), dp[i][j-1]) + 1
                    res = max(res, dp[i][j])
        return res*res   # area=size*size


"""1277. Count Square Submatrices with All Ones
Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.
Example 1:
Input: matrix = [
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]
Output: 15
Explanation: 
There are 10 squares of side 1.
There are 4 squares of side 2.
There is  1 square of side 3.
Total number of squares = 10 + 4 + 1 = 15.
"""
# the same as above, the size can also represent the number of sqs
# that ends with matrix[i-1][j-1]
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        # dp[i][j] -> ends with matrix[i][j]
        res = 0
        for i in range(m):
            for j in range(n):
                curr_num = matrix[i][j]
                if curr_num == 1:
                    if i==0 or j==0: # first row or col
                        dp[i][j] = curr_num
                    else:
                        dp[i][j] = min(min(dp[i-1][j-1], dp[i-1][j]), dp[i][j-1]) + 1
                    res = res + dp[i][j]
        return res


"""740. Delete and Earn
You are given an integer array nums. You want to maximize the number of points you get by 
performing the following operation any number of times:

Pick any nums[i] and delete it to earn nums[i] points. Afterwards, you must delete every 
element equal to nums[i] - 1 and every element equal to nums[i] + 1.
Return the maximum number of points you can earn by applying the above operation some number of times.

Example 1:
Input: nums = [3,4,2]
Output: 6
Explanation: You can perform the following operations:
- Delete 4 to earn 4 points. Consequently, 3 is also deleted. nums = [2].
- Delete 2 to earn 2 points. nums = [].
You earn a total of 6 points.

Example 2:
Input: nums = [2,2,3,3,3,4]
Output: 9
Explanation: You can perform the following operations:
- Delete a 3 to earn 3 points. All 2's and 4's are also deleted. nums = [3,3].
- Delete a 3 again to earn 3 points. nums = [3].
- Delete a 3 once more to earn 3 points. nums = [].
You earn a total of 9 points.
"""
nums = [1,3]
from collections import Counter
class Solution:
    def deleteAndEarn(self, nums) -> int:
        m = Counter(nums)
        dp = [0]*(len(m)+1)
        sorted_keys = sorted(m.keys())
        dp[1] = sorted_keys[0] * m[sorted_keys[0]]  # max of up to first
        for i in range(1, len(sorted_keys)):
            if sorted_keys[i] -sorted_keys[i-1] == 1:
                # need to delete when next key is neighboring
                # max of taking and not taking nums[idx+2]
                dp[i+1] = max(dp[i-1]+sorted_keys[i]*m[sorted_keys[i]], dp[i])
            else:
                dp[i+1] = dp[i] + sorted_keys[i]*m[sorted_keys[i]]
        return dp[-1]

# another solution that do not need discuss sorted_keys[i] -sorted_keys[i-1] == 1
# need to record min and max of m, then the loop is:
"""
dp = [0] * (maximum+1)
for i in range(minimum, maximum+1):
    dp[i+1] = max(dp[i-1]+i*m[i], dp[i])
"""

"""87. Scramble String
We can scramble a string s to get a string t using the following algorithm:

If the length of the string is 1, stop.
If the length of the string is > 1, do the following:
Split the string into two non-empty substrings at a random index, i.e., if the string is s, divide it to x and y where s = x + y.
Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, s may become s = x + y or s = y + x.
Apply step 1 recursively on each of the two substrings x and y.
Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false.

Example 1:
Input: s1 = "great", s2 = "rgeat"
Output: true
Explanation: One possible scenario applied on s1 is:
"great" --> "gr/eat" // divide at random index.
"gr/eat" --> "gr/eat" // random decision is not to swap the two substrings and keep them in order.
"gr/eat" --> "g/r / e/at" // apply the same algorithm recursively on both substrings. divide at ranom index each of them.
continue but keep all the orders after that

Example 2:
Input: s1 = "abcde", s2 = "caebd"
Output: false
"""
# recursion with memo
# first five mins in: 
# https://www.youtube.com/watch?v=sETxfdHwxc0
from functools import lru_cache
class Solution:
    @lru_cache(maxsize=None)
    def isScramble(self, s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        if len(s1) != len(s2):
            return False
        if sorted(s1) != sorted(s2):
            return False
        n = len(s1)
        for i in range(1, n):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True
            if self.isScramble(s1[:i], s2[n-i:]) and self.isScramble(s1[i:], s2[:n-i]):
                return True
        return False

s1 = "abcde"; s2 = "caebd"
s1 = "great"; s2 = "rgeat"
sol=Solution()
sol.isScramble(s1, s2)

# A formal way of recursive + memo
class Solution:
    def __init__(self):
        self.dic = {}
        
    def isScramble(self, s1, s2):
        if (s1, s2) in self.dic:
            return self.dic[(s1, s2)]
        if len(s1) != len(s2) or sorted(s1) != sorted(s2): # prunning
            self.dic[(s1, s2)] = False
            return False
        if s1 == s2:
            self.dic[(s1, s2)] = True
            return True
        for i in xrange(1, len(s1)):
            if (self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:])) or \
            (self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:-i])):
                self.dic[(s1, s2)] = True
                return True
        self.dic[(s1, s2)] = False
        return False


""" 1140. Stone Game II
Alice and Bob continue their games with piles of stones.  There are a number of piles 
arranged in a row, and each pile has a positive integer number of stones piles[i].  
The objective of the game is to end with the most stones. 

Alice and Bob take turns, with Alice starting first.  Initially, M = 1.

On each player's turn, that player can take all the stones in the first X remaining piles, 
where 1 <= X <= 2M.  Then, we set M = max(M, X).

The game continues until all the stones have been taken.

Assuming Alice and Bob play optimally, return the maximum number of stones Alice can get.

Example 1:
Input: piles = [2,7,9,4,4]
Output: 10
Explanation:  If Alice takes one pile at the beginning, Bob takes two piles, 
then Alice takes 2 piles again. Alice can get 2 + 4 + 4 = 10 piles in total. 
If Alice takes two piles at the beginning, then Bob can take all three piles left. 
In this case, Alice get 2 + 7 = 9 piles in total. So we return 10 since it's larger. 

Example 2:
Input: piles = [1,2,3,4,5,100]
Output: 104

hint: Use dynamic programming: the states are (i, m) for the answer of piles[i:] and that given m.
Or use DFS+memo, the same idea

dfs(piles[i:], m) = max(sum(piles[i:i+x]) + sum(piles[i+x:]) - dfs(piles[i+x:], max(m, x))) for 1<=x<=2m and x+i<=n-1
"""
class Solution:
    def stoneGameII(self, piles) -> int:
        n = len(piles)
        cumsums = [0] * n  # save sum(piles[i:]) !!
        cumsums[n-1] = piles[n-1]
        for i in range(n-1)[::-1]:
            cumsums[i] = cumsums[i+1] + piles[i]
        
        memo = dict()
        return self.dfs(1, 0, memo, cumsums)
    
    def dfs(self, m, i, memo, cumsums):
        n = len(cumsums)
        if n - i <= 2 * m:
            # can take all piles[i:]
            return cumsums[i]
        if (m, i) in memo:
            return memo[(m, i)]
        res = 0
        for x in range(1, min(2*m+1, n-i)):
            cur_sum = cumsums[i] - cumsums[i+x] # sum(piles[i:x+i])
            res = max(res, cur_sum + cumsums[i+x] - self.dfs(max(m, x), i+x, memo, cumsums))
        memo[(m, i)] = res
        # print(memo)
        return res

sol=Solution()
sol.stoneGameII([2,7,9,4,4])
sol.stoneGameII([1,2,3,4,5,100])


"""322. Coin Change
You are given an integer array coins representing coins of different denominations 
and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of 
money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

Example 1:
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:
Input: coins = [2], amount = 3
Output: -1

Example 3:
Input: coins = [1], amount = 0
Output: 0

# dp[i] 表示钱数为i时的最小硬币数的找零，
# can also use dfs+memo, key is i, value is 钱数为i时的最小硬币数的找零
"""
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount == 0:
            return 0
        n = amount
        dp = [float('Inf')] * (n+1)
        dp[0] = 0
        for i in range(1, n+1):
            for coin in coins:
                if i-coin >= 0:
                    dp[i] = min(dp[i], dp[i-coin]+1)
        
        return dp[-1] if dp[-1] < float('Inf') else -1


""" [LeetCode] 518. Coin Change 2 硬币找零之二
You are given an integer array coins representing coins of different denominations and 
an integer amount representing a total amount of money.
Return the number of combinations that make up that amount. If that amount of money cannot 
be made up by any combination of the coins, return 0.
You may assume that you have an infinite number of each kind of coin.
The answer is guaranteed to fit into a signed 32-bit integer.

Example 1:
Input: amount = 5, coins = [1,2,5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

Example 2:
Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.

Example 3:
Input: amount = 10, coins = [10]
Output: 1

Solution: 
makeChange(11) = makeChange(11 using 0-5) + makeChange(11 using 1-5) + makeChange(11 using 2-5) + ...

Each child will continue using 
makeChange(11 using 1-5) = makeChange(6 using 0-2) + makeChange(6 using 1-2) + ... 

Continue will be
makeChange(6 using 0-2) = makeChange(6-0 using 1's) = 1 (because 6%1==0)
makeChange(6 using 1-2) = makeChange(6-2 using 1's) = 1
"""
# 用到最后一个硬币时，判断当前还剩的钱数是否能整除这个硬币，不能的话就返回0，否则返回1
# memo: key is (remain_amount, curr_coin_idx)
class Solution:
    def change(self, amount: int, coins) -> int:
        memo = dict()
        sorted_coins = sorted(coins, reverse=True)
        return self.dfs(sorted_coins, amount, memo, 0)

    def dfs(self, coins, rem_amount, memo, coin_idx):
        if coin_idx == len(coins) - 1:
            # last coin
            # rem=0 will also return 1
            return 1 if rem_amount % coins[coin_idx] == 0 else 0
        if (rem_amount, coin_idx) in memo:
            return memo[(rem_amount, coin_idx)]
        res = 0
        for num_curr_coin in range(1+rem_amount//coins[coin_idx]):
            next_rem_amount = rem_amount - num_curr_coin*coins[coin_idx]
            res += self.dfs(coins, next_rem_amount, memo, coin_idx+1)
        memo[(rem_amount, coin_idx)] = res
        # print(memo)
        return res

sol=Solution()
sol.change(5, [1, 2, 5])

# DFS+cache trick
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

# DP: Unbounded Knapsack
# dp[i][j]: number of comb to achieve j amount, if only using first i kinds of coins
class Solution:
    def change(self, amount: int, coins) -> int:
        n = len(coins)
        dp = [[0 for _ in range(amount+1)] for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            for j in range(amount+1):
                if j == 0: # note if i=0 but j>0, should be 0
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j] # first include all sols that only use first i-1 kinds
                    if j >= coins[i-1]:
                        # add coins[i-1]: note that not i-1 due to unbounded! 
                        # we can use multiple coins[i] till j-coins[i]<0
                        dp[i][j] += dp[i][j-coins[i-1]]
        return dp[n][amount]

# may reduce space complexity by using 1-d array
"""
def change(self, amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for i in coins:
        for j in range(1, amount + 1):
            if j >= i:
                dp[j] += dp[j - i]
    return dp[amount]
"""

"""[LeetCode] 1048. Longest String Chain 
Given a list of words, each word consists of English lowercase letters.

Let's say word1 is a predecessor of word2 if and only if we can add exactly one letter anywhere in word1 to make it equal to word2.  
For example, "abc" is a predecessor of "abac".

A *word chain *is a sequence of words [word_1, word_2, ..., word_k] with k >= 1, 
where word_1 is a predecessor of word_2, word_2 is a predecessor of word_3, and so on.

Return the longest possible length of a word chain with words chosen from the given list of words.
You do not need to keep the order

Example 1:
Input: words = ["a","b","ba","bca","bda","bdca"]
Output: 4
Explanation: One of the longest word chain is "a","ba","bda","bdca".

Example 2:
Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
Output: 5
Explanation: All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].

Idea: 
m = dict() # save the length of LSC ended with key word
sort words by len, start with the shortest, check whether curr_word minus a char is in the m, 
if yes, then m[curr_word] = max(m[curr_word], m[shorter_word]+1)
"""
from collections import defaultdict
class Solution:
    def longestStrChain(self, words) -> int:
        words = sorted(words, key=lambda x:len(x))
        m = defaultdict(list)
        for word in words:
            m[word] = 1
            for i in range(len(word)):
                word_rm_1_char = word[:i] + word[i+1:]
                if word_rm_1_char in m:
                    m[word] = max(m[word_rm_1_char]+1, m[word])
        return max([v for k,v in m.items()])

sol=Solution()
sol.longestStrChain(["xbc","pcxbcf","xb","cxbc","pcxbc"])


"""[LeetCode] 44. Wildcard Matching 外卡匹配
Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.
'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).
Note:
s could be empty and contains only lowercase letters a-z.
p could be empty and contains only lowercase letters a-z, and characters like ? or *.

Example 1:
Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

Example 2:
Input:
s = "aa"
p = "*"
Output: true
Explanation: '*' matches any sequence.

Example 3:
Input:
s = "cb"
p = "?a"
Output: false
Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.

Example 4:
Input:
s = "adceb"
p = "*a*b"
Output: true
Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".

Example 5:
Input:
s = "acdcb"
p = "a*c?b"
Output: false
"""
# Solution 1 DP
# dp[i][j] 表示 s[:i] and p[:j] 匹配
# https://leetcode.com/problems/wildcard-matching/discuss/256025/Python-DP-with-illustration
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0] = True  # "" match ""
        for i in range(1, m+1):
            dp[i][0] = False
        
        for j in range(1, n+1):
            # only when all *
            dp[0][j] = True if dp[0][j-1] and p[j-1]=='*' else False

        for i in range(1, m+1):
            for j in range(1, n+1):
                if p[j-1] == "*":
                    # s: XXX,  p: XXX*
                    dp[i][j] = dp[i][j-1] or dp[i-1][j]  # do not need i-2, j
                elif p[j-1] == "?":
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = dp[i-1][j-1] and (s[i-1] == p[j-1])
        
        return dp[m][n]

# DFS + memo
# need to trim to pass 
# https://leetcode.com/problems/wildcard-matching/discuss/17839/C%2B%2B-recursive-solution-16-ms
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        memo = dict()
        return self.dfs(s, p, 0, 0, memo)

    def dfs(self, s, p, i, j, memo):
        # s[i:] vs p[j:]
        if (i,j) in memo:
            return memo[(i, j)]
        if i == len(s) or j == len(p):
            if i == len(s) and j == len(p):
                return True
            elif j == len(p):
                return False
            else:
                # i == len(s)
                # whether p[j:] only has *
                memo[(i,j)] = (len([p_rem for p_rem in p[j:] if p_rem!='*']) == 0)
                return memo[(i,j)]
        res = False
        if p[j] == '*':
            for idx in range(i, len(s)+1):
                res = res or self.dfs(s, p, idx, j+1, memo)
                if res:
                    break
        elif p[j] == '?':
            res = self.dfs(s, p, i+1, j+1, memo)
        else:
            return (s[i] == p[j]) and self.dfs(s, p, i+1, j+1, memo)
        memo[(i,j)] = res
        # print(memo)
        return res


sol=Solution()
sol.isMatch(s, p)
sol.isMatch("aa", "*")
sol.isMatch("adceb", "*a*b")
sol.isMatch(s="cdcb", p="*c")
sol.isMatch(s="cdcb", p="c")


"""[LeetCode] 10. Regular Expression Matching 正则表达式匹配
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
Note:
s could be empty and contains only lowercase letters a-z.
p could be empty and contains only lowercase letters a-z, and characters like . or *.

Example 1:
Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".

Example 2:
Input:
s = "aa"
p = "a*"
Output: true
Explanation: '*' means zero or more of the precedeng element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".

Example 3:
Input:
s = "ab"
p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".

Example 4:
Input:
s = "aab"
p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore it matches "aab".

Example 5:
Input:
s = "mississippi"
p = "mis*is*p*."
Output: false
"""
# Solution 1: DP
# https://leetcode.com/problems/regular-expression-matching/discuss/5684/c-on-space-dp
# 其中 dp[i][j] 表示 s[:i] 和 p[:j] 是否 match
# 1.  dp[i][j] = dp[i - 1][j - 1], if p[j - 1] != '*' && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
# 2.  dp[i][j] = dp[i][j - 2], if p[j - 1] == '*' and the pattern repeats for 0 times;
# 3.  dp[i][j] = dp[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.'), if p[j - 1] == '*' and the pattern repeats for at least 1 times.
"""
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        for (int j = 1; j <= n; ++j) {
            if (p[j-1] == '*' && j > 1) {
                dp[0][j] = dp[0][j - 2];
            }
            else{
                dp[0][j] = false;
            }
        }
        # for (int i = 1; i <= n; ++j) {
        #     dp[i][0] = false;
        # }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (j > 1 && p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 2] || (i > 0 && (s[i - 1] == p[j - 2] || p[j - 2] == '.') && dp[i - 1][j]);
                } else {
                    dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
                }
            }
        }
        return dp[m][n];
    }
};
"""
# DFS+cache trick solution
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        return self.isMatchhelper(s, p, 0, 0)

    @lru_cache(None)
    def isMatchhelper(self, s, p, i, j):
        if i == len(s) and j == len(p):
            return True
        if j == len(p):
            return False
        if j == len(p) - 1 or p[j+1] != '*': # p[j+1] != '*', more clear
            if i == len(s):
                return False
            if p[j] == '.' or s[i] == p[j]:
                return self.isMatchhelper(s, p, i+1, j+1)
            else: # s[i] != p[j] and p[j] != '.'
                return False
        else: # j < len(p) - 1 and p[j+1] == '*'
            if p[j] == '.':
                for idx in range(i, len(s)+1):
                    # .* can match any string
                    if self.isMatchhelper(s, p, idx, j+2):
                            return True
                return False
            else:
                if i == len(s) or s[i] != p[j]:
                    # ignore the previous char before *
                    return self.isMatchhelper(s, p, i, j+2)
                else: # s[i] == p[j]
                    for idx in range(i, len(s)):
                        # a* can only match either empty or aaa...aa
                        if idx == i or (idx > i and s[idx] == s[idx-1]):
                            # idx == i -> * matches empty
                            # s[idx] == s[idx-1] -> * matches up to idx
                            if self.isMatchhelper(s, p, idx, j+2):
                                return True
                        else: # s[idx] != s[idx-1]
                            return self.isMatchhelper(s, p, idx, j+2)
                    # if s[idx] == s[idx-1] all the way to the last. e.g., (aa, a*)
                    return self.isMatchhelper(s, p, len(s), j+2)

sol=Solution()
sol.isMatch("aa", "a")
sol.isMatch("aa", "a*")
sol.isMatch("ab", ".*")
sol.isMatch("ab", ".*c")
sol.isMatch("a", "ab*")
sol.isMatch("aab", "c*a*b")
sol.isMatch("mississippi", "mis*is*p*.")

# solution 2: DFS with memo
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # assume no consecutive *'s, otherwise use a loop to remove it
        # new_p = p[0]
        # for i in range(1, len(p)):
        #     if p[i-1] != '*' or p[i] != '*':
        #         new_p += p[i]
        # p = new_p
        memo = dict()
        return self.dfs(s, p, 0, 0, memo)
        
    def dfs(self, s, p, i, j, memo):
        # s[i:] matches p[j:]
        # never let p[j]='*'!
        if (i,j) in memo:
            return memo[(i, j)]
        if i==len(s) and j==len(p):
            return True
        if i<len(s) and j==len(p):
            return False
        if len(p[j:]) == 1:
            # if p[j:] longer than 1, include afterwards
            # again, p[j:] will never starts with *
            return True if len(s[i:]) == 1 and (p[j]=='.' or s[i]==p[j]) else False
        if p[j+1] != "*":
            if i < len(s) and (p[j] == '.' or s[i] == p[j]):
                res = self.dfs(s, p, i+1, j+1, memo)
                memo[(i, j)] = res
                return res
            else:
                memo[(i,j)] = False
                return False
        else:
            if i < len(s) and s[i] != p[j] and p[j] != '.':
                # i cannot exceed !
                res = self.dfs(s, p, i, j+2, memo)
                memo[(i, j)] = res
                return res
            else:
                idx = i
                while idx < len(s) and (s[idx] == p[j] or p[j] == '.'):
                    res = self.dfs(s, p, idx, j+2, memo)
                    if res:
                        memo[(i, j)] = res
                        return res
                    idx += 1
                res = self.dfs(s, p, idx, j+2, memo)
                memo[(i, j)] = res
                # print(memo)
                return res


"""32. Longest Valid Parentheses
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

Example 1:
Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".

Example 2:
Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".

Example 3:
Input: s = ""
Output: 0
"""
# Solution 1: Use a stack to track the index of all unmatched open parentheses.
# First, we iterate through the string, keeping track of the indices of open parentheses in a stack. 
# Every time we hit a close parenthesis, we know that the last open and the current close are both valid, 
# so we mark both of these as being valid by overwriting them in the original string with 1.
# 
# Once we have all of the valid individual parentheses marked, we iterate through the string 
# once more and identify the longest sequence of 1, which is our answer.
# Lets take a example , s = ")(()())))", so array will be like this 011111100
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = []
        s_list = list(s)
        for i in range(len(s_list)):
            if s_list[i] == '(':
                stack.append(i)
            else:
                if len(stack) == 0:
                    continue
                else:
                    # find a pair
                    s_list[i] = '1'
                    s_list[stack.pop()] = '1'
        res = 0
        curr_len = 0
        for i in range(len(s_list)):
            if s_list[i] != '1':
                res = max(curr_len, res)
                curr_len = 0
            else:
                curr_len += 1
        
        res = max(curr_len, res) # if last element is )
        return res


s = "()(()"
s="(()"
# can only iterate for once 
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        res = 0
        stack = [-1]  # can still calculate len when all '(' are poped
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if len(stack) > 0:
                    res = max(res, i-stack[-1])
                else:
                    # when stack is empty, means not valid so do not update res
                    # current ) will be the new start
                    stack.append(i)
        return res

# solution2: DP
"""
My solution uses DP. The main idea is as follows: I construct a array longest[], for any longest[i], 
it stores the longest length of valid parentheses which is end at i.
And the DP idea is :

If s[i] is '(', set longest[i] to 0,because any string end with '(' cannot be a valid one.
Else if s[i] is ')'
     If s[i-1] is '(', longest[i] = longest[i-2] + 2
     Else if s[i-1] is ')' and s[i-longest[i-1]-1] == '(', longest[i] = longest[i-1] + 2 + longest[i-longest[i-1]-2]

For example, input "()(())", at i = 5, longest array is [0,2,0,0,2,0], longest[5] = longest[4] + 2 + longest[1] = 6.
"""


"""1043. Partition Array for Maximum Sum
Given an integer array arr, partition the array into (contiguous) subarrays of length at most k. 
After partitioning, each subarray has their values changed to become the maximum value of that subarray.
Return the largest sum of the given array after partitioning. Test cases are generated so that the answer fits in a 32-bit integer.

Example 1:
Input: arr = [1,15,7,9,2,5,10], k = 3
Output: 84
Explanation: arr becomes [15,15,15,9,10,10,10]

Example 2:
Input: arr = [1,4,1,5,7,3,6,1,9,9,3], k = 4
Output: 83

Example 3:
Input: arr = [1], k = 1
Output: 1

hint: dp[i] will be the answer for array A[0], ..., A[i-1].
For j = 1 .. k that keeps everything in bounds, dp[i] is the maximum of dp[i-j] + max(A[i-1], ..., A[i-j]) * j .
"""
class Solution:
    def maxSumAfterPartitioning(self, arr, k: int) -> int:
        n = len(arr)
        if n == 0:
            return 0
        dp = [0] * (n+1) # dp[i] : arr[:i]
        dp[0] = 0
        for i in range(1, n+1):
            curr_max = 0
            for j in range(1, k+1):
                # j=1,...,k; another idea is to use j as location, starts at i-1 to max(0, i-k)
                if i - j >= 0:
                    curr_max = max(curr_max, arr[i-j])
                    dp[i] = max(dp[i], dp[i-j] + curr_max * j)
                    # dp[i] = max(dp[i], dp[i-j] + max(arr[i-j:i]) * j)

        return dp[n]

sol=Solution()
sol.maxSumAfterPartitioning([1,4,1,5,7,3,6,1,9,9,3], 4)

# DFS + cache
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
                break
            # print(self.arr[idx:idx+i])
            res = max(res, self.get_maxsum(arr[idx:idx+i]) + self.maxSumhelper(arr, idx+i, k))
        return res

    def get_maxsum(self, arr):
        return max(arr) * len(arr)


""" 01 Knapsack Problem 
一共有N件物品，第i（i从1开始）件物品的重量为weights[i]，价值为values[i]。在总重量不超过背包承载上限W的情况下，能够装入背包的最大价值是多少？

动态规划的核心思想避免重复计算在01背包问题中体现得淋漓尽致。第i件物品装入或者不装入而获得的最大价值
完全可以由前面i-1件物品的最大价值决定，暴力枚举忽略了这个事实。
""" 
def Knapsack01(values, weights, W):
    # dp[i][j] : max value by taking from first i items, with weight at most j
    n = len(values)
    dp = [[0 for _ in range(W+1)] for _ range(n+1)]
    # for j in range(W+1):
    #     dp[0][j] = 0 # value is 0 if no item
    for i in range(1, n+1):
        for j in range(1, W):
            # if j == 0: # first 0 items
            #     dp[i][j] = 0
            # else: # either take i or not
            # i row results only depend on i-1 row
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]] + values[i-1]) if j>=weights[i] else dp[i-1][j]
    return dp[n][W]

"""Simplify the space time complexity
dp[j] : max value by weights up to j
dp[0,...,W] = 0
for i = 1,...,N
    for j = W,...,weights[i] // 必须逆向枚举!!! 否则dp[i-1][j-weights[i]]已经被overwrite成dp[i][j-weights[i]]
        dp[j] = max(dp[j], dp[j−weights[i]]+v[i])
"""

"""Unbounded Knapsack problem
完全背包（unbounded knapsack problem）与01背包不同就是每种物品可以有无限多个：一共有N种物品，每种物品有无限多个，
第i（i从1开始）种物品的重量为weights[i]，价值为v[i]。在总重量不超过背包承载上限W的情况下，能够装入背包的最大价值是多少？

dp[i][j]表示将前i种物品装进限重为j的背包可以获得的最大价值, 0<=i<=N, 0<=j<=W
这个状态转移方程与01背包问题唯一不同就是max第二项不是dp[i-1]而是dp[i]。
i.e., 
dp[i][j] = max(dp[i-1][j], dp[i][j-weights[i]] + values[i-1])

if reduce space complexity using 1-d
// 完全背包问题思路一伪代码(空间优化版)
dp[0,...,W] = 0
for i = 1,...,N
    for j = weights[i],...,W // 必须正向枚举!!! 因为dp[i][j]依赖于前面的dp[i][j−weights[i]]
        dp[j] = max(dp[j], dp[j−weights[i]]+v[i])
"""


"""[LeetCode] Ones and Zeroes 一和零
In the computer world, use restricted resource you have to generate maximum benefit is what we always want to pursue.
For now, suppose you are a dominator of m 0s and n 1s respectively. On the other hand, there is an array with strings consisting of only 0s and 1s.
Now your task is to find the maximum number of strings that you can form with given m 0s and n 1s. Each 0 and 1 can be used at most once.

Note:
The given numbers of 0s and 1s will both not exceed 100
The size of given string array won't exceed 600.

Input: strs = ["10","0001","111001","1","0"]; m = 5; n = 3 
Output: 4
"""
# 其中dp[i][j][k]表示用如果使用strs[:i]，有j个0和k个1时能组成的最多字符串的个数
from collections import Counter
strs = ["10","0001","111001","1","0"]; m = 5; n = 3 
strs = ["10","0","1"]; m=1; n=1
class Solution:
    def findMaxForm(self, strs, m: int, n: int) -> int:
        n_strs = len(strs)
        dp = [[[0 for _ in range(n+1)] for _ in range(m+1)] for _ in range(n_strs+1)]
        for i in range(1, n_strs+1):
            n_zeros, n_ones = get_ones_zeros(strs[i-1])
            for j in range(m+1):
                for k in range(n+1):
                    dp[i][j][k] = dp[i-1][j][k]
                    if j >= n_zeros and k >= n_ones:
                        dp[i][j][k] = max(dp[i-1][j][k], 1+dp[i-1][j-n_zeros][k-n_ones])
        
        return dp[n_strs][m][n]

def get_ones_zeros(s):
    d = Counter(s)
    return d['0'], d['1']

# optimize by reducing to 2-d array 
class Solution:
    def findMaxForm(self, strs, m: int, n: int) -> int:
        n_strs = len(strs)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(1, n_strs+1):
            n_zeros, n_ones = get_ones_zeros(strs[i-1])
            for j in range(n_zeros, m+1)[::-1]:
                for k in range(n_ones, n+1)[::-1]:
                    # Because need to use previous dp[prev_j][prev_k] (prev_j<=j and prev_k<=k)
                    dp[j][k] = max(dp[j][k], 1+dp[j-n_zeros][k-n_ones])
        
        return dp[m][n]


