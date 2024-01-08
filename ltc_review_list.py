"""
In Ltc, some questions need hard memory
This file contains such questions
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