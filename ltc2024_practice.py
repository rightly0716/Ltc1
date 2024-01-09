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

arr=[3,2,3,1,2,4,5,5,6] 
k=4
solution=kthlargest()
solution.findKthLargest(arr, k)


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
