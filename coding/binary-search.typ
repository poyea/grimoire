= Binary Search

== Find Minimum in Rotated Sorted Array

*Problem:* Find the minimum element in a rotated sorted array (no duplicates).

*Approach - Binary Search:* $O(log n)$ time, $O(1)$ space
- Initialize `result = nums[0]`, `left = 0`, `right = len(nums) - 1`
- While `left <= right`:
  + If `nums[left] < nums[right]`: array is sorted, update result and break
  + Calculate `mid = (left + right) // 2`
  + Update `result = min(result, nums[mid])`
  + If `nums[mid] >= nums[left]`: minimum is in right half
    - `left = mid + 1`
  + Else: minimum is in left half
    - `right = mid - 1`
- Return `result`

*Key insight:* Determine which half is sorted, then search the unsorted half for minimum.

*Key Python concepts:*
- Integer division: `//` operator

== Search in Rotated Sorted Array

*Problem:* Search for target in rotated sorted array, return index or -1.

*Approach - Modified Binary Search:* $O(log n)$ time, $O(1)$ space
- Initialize `left = 0`, `right = len(nums) - 1`
- While `left <= right`:
  + Calculate `mid = (left + right) // 2`
  + If `nums[mid] == target`: return mid
  + Check which half is sorted:
    * If `nums[mid] >= nums[left]`: left half is sorted
      - If `nums[left] <= target < nums[mid]`: search left (`right = mid - 1`)
      - Else: search right (`left = mid + 1`)
    * Else: right half is sorted
      - If `nums[mid] < target <= nums[right]`: search right (`left = mid + 1`)
      - Else: search left (`right = mid - 1`)
- Return -1

*Key insight:* At least one half of rotated array is always sorted - use this to determine search direction.
