# LeetCode Hot 100 记忆卡片

共 105 张卡片

---

## 数组

### 二分查找

#### 卡片 1

**问题**：LeetCode 4. 寻找两个正序数组的中位数

给定两个大小分别为m和n的正序（从小到大）数组nums1和nums2。请你找出并返回这两个正序数组的中位数。

算法的时间复杂度应该为O(log(m+n))。

**答案**：

思路1：归并排序
- 时间复杂度：O(m+n)
- 空间复杂度：O(1)
- 思路：合并两个有序数组，找到中位数

```cpp
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        int total = m + n;
        int i = 0, j = 0;
        int prev = 0, curr = 0;
        
        for (int k = 0; k <= total / 2; k++) {
            prev = curr;
            if (i < m && (j >= n || nums1[i] < nums2[j])) {
                curr = nums1[i++];
            } else {
                curr = nums2[j++];
            }
        }
        
        return total % 2 == 0 ? (prev + curr) / 2.0 : curr;
    }
};
```

思路2：二分查找（最优）
- 时间复杂度：O(log(min(m,n)))
- 空间复杂度：O(1)
- 思路：在较短的数组上二分查找分割点，使得左右两部分元素个数相等或差1

```cpp
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        if (nums1.size() > nums2.size()) {
            return findMedianSortedArrays(nums2, nums1);
        }
        
        int m = nums1.size(), n = nums2.size();
        int left = 0, right = m;
        
        while (left <= right) {
            int partition1 = (left + right) / 2;
            int partition2 = (m + n + 1) / 2 - partition1;
            
            int maxLeft1 = (partition1 == 0) ? INT_MIN : nums1[partition1 - 1];
            int minRight1 = (partition1 == m) ? INT_MAX : nums1[partition1];
            int maxLeft2 = (partition2 == 0) ? INT_MIN : nums2[partition2 - 1];
            int minRight2 = (partition2 == n) ? INT_MAX : nums2[partition2];
            
            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                if ((m + n) % 2 == 0) {
                    return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2.0;
                } else {
                    return max(maxLeft1, maxLeft2);
                }
            } else if (maxLeft1 > minRight2) {
                right = partition1 - 1;
            } else {
                left = partition1 + 1;
            }
        }
        return 0.0;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 35. 搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

**答案**：

思路1：二分查找
- 时间复杂度：O(logn)
- 空间复杂度：O(1)
- 思路：使用二分查找找到目标值或插入位置

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return left;
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 33. 搜索旋转排序数组

整数数组nums按升序排列，数组中的值互不相同。

在传递给函数之前，nums在预先未知的某个下标k（0<=k<nums.length）上进行了旋转。给你旋转后的数组nums和一个整数target，如果nums中存在这个目标值target，则返回它的下标，否则返回-1。

**答案**：

思路：二分查找
- 时间复杂度：O(logn)
- 空间复杂度：O(1)
- 思路：根据mid位置判断左半部分还是右半部分是有序的，然后决定搜索方向

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            
            // 左半部分有序
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else { // 右半部分有序
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return -1;
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 34. 在排序数组中查找元素的第一个和最后一个位置

给你一个按照非递减顺序排列的整数数组nums，和一个目标值target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值target，返回[-1,-1]。

**答案**：

思路：二分查找
- 时间复杂度：O(logn)
- 空间复杂度：O(1)
- 思路：使用两次二分查找，分别找到第一个和最后一个位置

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int first = findFirst(nums, target);
        if (first == -1) return {-1, -1};
        int last = findLast(nums, target);
        return {first, last};
    }
    
private:
    int findFirst(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                if (mid == 0 || nums[mid - 1] != target) return mid;
                right = mid - 1;
            }
        }
        return -1;
    }
    
    int findLast(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                if (mid == nums.size() - 1 || nums[mid + 1] != target) return mid;
                left = mid + 1;
            }
        }
        return -1;
    }
};
```

---

#### 卡片 5

**问题**：LeetCode 74. 搜索二维矩阵

编写一个高效的算法来判断m×n矩阵中，是否存在一个目标值。该矩阵具有如下特性：每行中的整数从左到右按升序排列；每行的第一个整数大于前一行的最后一个整数。

**答案**：

思路：二分查找
- 时间复杂度：O(log(mn))
- 空间复杂度：O(1)
- 思路：将二维矩阵视为一维数组进行二分查找

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int left = 0, right = m * n - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int row = mid / n, col = mid % n;
            
            if (matrix[row][col] == target) {
                return true;
            } else if (matrix[row][col] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return false;
    }
};
```

---

#### 卡片 6

**问题**：LeetCode 81. 搜索旋转排序数组 II

已知存在一个按非降序排列的整数数组nums，数组中的值不必互不相同。

在传递给函数之前，nums在预先未知的某个下标k（0<=k<nums.length）上进行了旋转。给你旋转后的数组nums和一个整数target，请你编写一个函数来判断给定的目标值是否存在于数组中。

**答案**：

思路：二分查找
- 时间复杂度：O(n)最坏情况
- 空间复杂度：O(1)
- 思路：处理重复元素的情况，当nums[left]==nums[mid]时，left++跳过重复

```cpp
class Solution {
public:
    bool search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return true;
            
            if (nums[left] == nums[mid] && nums[mid] == nums[right]) {
                left++;
                right--;
            } else if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return false;
    }
};
```

---

### 双指针

#### 卡片 1

**问题**：LeetCode 11. 盛最多水的容器

给定一个长度为n的整数数组height。有n条垂线，第i条线的两个端点是(i,0)和(i,height[i])。

找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。

**答案**：

思路：双指针
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：从两端开始，每次移动较短的那一边，因为移动较长边不会增加面积

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int maxArea = 0;
        
        while (left < right) {
            int area = min(height[left], height[right]) * (right - left);
            maxArea = max(maxArea, area);
            
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        
        return maxArea;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 15. 三数之和

给你一个整数数组nums，判断是否存在三元组[nums[i], nums[j], nums[k]]满足i!=j、i!=k且j!=k，同时还满足nums[i]+nums[j]+nums[k]==0。请你返回所有和为0且不重复的三元组。

**答案**：

思路：排序+双指针
- 时间复杂度：O(n²)
- 空间复杂度：O(1)（不考虑结果数组）
- 思路：先排序，固定第一个数，用双指针找另外两个数

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        int n = nums.size();
        if (n < 3) return result;
        
        sort(nums.begin(), nums.end());
        
        for (int i = 0; i < n - 2; i++) {
            // 跳过重复元素
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            
            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == 0) {
                    result.push_back({nums[i], nums[left], nums[right]});
                    // 跳过重复元素
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    left++;
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        
        return result;
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 26. 删除有序数组中的重复项

给你一个非严格递增排列的数组nums，请你原地删除重复出现的元素，使每个元素只出现一次，返回删除后数组的新长度。

**答案**：

思路：双指针
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用快慢指针，快指针遍历数组，慢指针指向不重复元素的位置

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.empty()) return 0;
        
        int slow = 0;
        for (int fast = 1; fast < nums.size(); fast++) {
            if (nums[fast] != nums[slow]) {
                slow++;
                nums[slow] = nums[fast];
            }
        }
        return slow + 1;
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 27. 移除元素

给你一个数组nums和一个值val，你需要原地移除所有数值等于val的元素，并返回移除后数组的新长度。

**答案**：

思路：双指针
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用快慢指针，快指针遍历数组，慢指针指向不等于val的元素位置

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slow = 0;
        for (int fast = 0; fast < nums.size(); fast++) {
            if (nums[fast] != val) {
                nums[slow] = nums[fast];
                slow++;
            }
        }
        return slow;
    }
};
```

---

#### 卡片 5

**问题**：LeetCode 75. 颜色分类

给定一个包含红色、白色和蓝色、共n个元素的数组nums，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数0、1和2分别表示红色、白色和蓝色。

**答案**：

思路1：计数排序
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：统计0、1、2的个数，然后重新填充数组

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int count0 = 0, count1 = 0, count2 = 0;
        for (int num : nums) {
            if (num == 0) count0++;
            else if (num == 1) count1++;
            else count2++;
        }
        
        int i = 0;
        while (count0--) nums[i++] = 0;
        while (count1--) nums[i++] = 1;
        while (count2--) nums[i++] = 2;
    }
};
```

思路2：双指针（荷兰国旗问题）
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用三个指针，将0移到左边，2移到右边

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        int curr = 0;
        
        while (curr <= right) {
            if (nums[curr] == 0) {
                swap(nums[left++], nums[curr++]);
            } else if (nums[curr] == 2) {
                swap(nums[curr], nums[right--]);
            } else {
                curr++;
            }
        }
    }
};
```

---

#### 卡片 6

**问题**：LeetCode 88. 合并两个有序数组

给你两个按非递减顺序排列的整数数组nums1和nums2，另有两个整数m和n，分别表示nums1和nums2中元素的数目。

请你合并nums2到nums1中，使合并后的数组同样按非递减顺序排列。

**答案**：

思路：从后往前合并
- 时间复杂度：O(m+n)
- 空间复杂度：O(1)
- 思路：从后往前填充，避免覆盖nums1中的元素

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        
        while (i >= 0 && j >= 0) {
            if (nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
        
        while (j >= 0) {
            nums1[k--] = nums2[j--];
        }
    }
};
```

---

#### 卡片 7

**问题**：LeetCode 16. 最接近的三数之和

给你一个长度为n的整数数组nums和一个目标值target。请你从nums中选出三个整数，使它们的和与target最接近。

**答案**：

思路：排序+双指针
- 时间复杂度：O(n²)
- 空间复杂度：O(1)
- 思路：先排序，固定第一个数，用双指针找另外两个数

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int closestSum = nums[0] + nums[1] + nums[2];
        
        for (int i = 0; i < nums.size() - 2; i++) {
            int left = i + 1, right = nums.size() - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (abs(sum - target) < abs(closestSum - target)) {
                    closestSum = sum;
                }
                if (sum < target) left++;
                else if (sum > target) right--;
                else return sum;
            }
        }
        
        return closestSum;
    }
};
```

---

#### 卡片 8

**问题**：LeetCode 18. 四数之和

给你一个由n个整数组成的数组nums，和一个目标值target。请你找出并返回满足下述全部条件且不重复的四元组[nums[a],nums[b],nums[c],nums[d]]。

**答案**：

思路：排序+双指针
- 时间复杂度：O(n³)
- 空间复杂度：O(1)
- 思路：先排序，固定前两个数，用双指针找后两个数

```cpp
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> result;
        if (nums.size() < 4) return result;
        
        sort(nums.begin(), nums.end());
        
        for (int i = 0; i < nums.size() - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            
            for (int j = i + 1; j < nums.size() - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                
                int left = j + 1, right = nums.size() - 1;
                while (left < right) {
                    long long sum = (long long)nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        result.push_back({nums[i], nums[j], nums[left], nums[right]});
                        while (left < right && nums[left] == nums[left + 1]) left++;
                        while (left < right && nums[right] == nums[right - 1]) right--;
                        left++;
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        
        return result;
    }
};
```

---

#### 卡片 9

**问题**：LeetCode 80. 删除有序数组中的重复项 II

给你一个有序数组nums，请你原地删除重复出现的元素，使得出现次数超过两次的元素只出现两次，返回删除后数组的新长度。

**答案**：

思路：双指针
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用快慢指针，允许每个元素最多出现两次

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.size() <= 2) return nums.size();
        
        int slow = 2;
        for (int fast = 2; fast < nums.size(); fast++) {
            if (nums[fast] != nums[slow - 2]) {
                nums[slow++] = nums[fast];
            }
        }
        
        return slow;
    }
};
```

---

### 哈希表

#### 卡片 1

**问题**：LeetCode 1. 两数之和

给定一个整数数组nums和一个整数目标值target，请你在该数组中找出和为目标值target的那两个整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

**答案**：

思路1：暴力法
- 时间复杂度：O(n²)
- 空间复杂度：O(1)
- 思路：双重循环遍历所有可能的组合

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (nums[i] + nums[j] == target) {
                    return {i, j};
                }
            }
        }
        return {};
    }
};
```

思路2：哈希表（一次遍历）
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用哈希表存储已访问的元素值及其索引，在遍历时查找target-nums[i]是否在哈希表中

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> map;
        for (int i = 0; i < nums.size(); i++) {
            int complement = target - nums[i];
            if (map.find(complement) != map.end()) {
                return {map[complement], i};
            }
            map[nums[i]] = i;
        }
        return {};
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 41. 缺失的第一个正数

给你一个未排序的整数数组nums，请你找出其中没有出现的最小的正整数。

**答案**：

思路：原地哈希
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：将数组本身作为哈希表，将数字i放到位置i-1

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        
        // 将数字放到正确位置
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums[i], nums[nums[i] - 1]);
            }
        }
        
        // 找到第一个缺失的正数
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        
        return n + 1;
    }
};
```

---

### 数学

#### 卡片 1

**问题**：LeetCode 66. 加一

给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位，数组中每个元素只存储单个数字。

**答案**：

思路：模拟加法
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：从后往前遍历，处理进位

```cpp
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        for (int i = digits.size() - 1; i >= 0; i--) {
            digits[i]++;
            if (digits[i] < 10) {
                return digits;
            }
            digits[i] = 0;
        }
        
        // 如果所有位都是9，需要在前面加1
        digits.insert(digits.begin(), 1);
        return digits;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 31. 下一个排列

整数数组的一个排列就是将其所有成员以序列或线性顺序排列。

整数数组的下一个排列是指其整数的下一个字典序更大的排列。

**答案**：

思路：两遍扫描
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：从右往左找到第一个降序位置，然后交换并反转

```cpp
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i = nums.size() - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--;
        
        if (i >= 0) {
            int j = nums.size() - 1;
            while (j > i && nums[j] <= nums[i]) j--;
            swap(nums[i], nums[j]);
        }
        
        reverse(nums.begin() + i + 1, nums.end());
    }
};
```

---

### 矩阵

#### 卡片 1

**问题**：LeetCode 73. 矩阵置零

给定一个m×n的矩阵，如果一个元素为0，则将其所在行和列的所有元素都设为0。请使用原地算法。

**答案**：

思路1：使用标记数组
- 时间复杂度：O(mn)
- 空间复杂度：O(m+n)
- 思路：使用两个数组记录哪些行和列需要置零

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<bool> rowZero(m, false);
        vector<bool> colZero(n, false);
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    rowZero[i] = true;
                    colZero[j] = true;
                }
            }
        }
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rowZero[i] || colZero[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
};
```

思路2：使用第一行和第一列作为标记（空间优化）
- 时间复杂度：O(mn)
- 空间复杂度：O(1)
- 思路：使用矩阵的第一行和第一列来标记需要置零的行和列

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        bool firstRowZero = false, firstColZero = false;
        
        // 检查第一行和第一列是否有0
        for (int j = 0; j < n; j++) {
            if (matrix[0][j] == 0) firstRowZero = true;
        }
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) firstColZero = true;
        }
        
        // 使用第一行和第一列作为标记
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        
        // 根据标记置零
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // 处理第一行和第一列
        if (firstRowZero) {
            for (int j = 0; j < n; j++) matrix[0][j] = 0;
        }
        if (firstColZero) {
            for (int i = 0; i < m; i++) matrix[i][0] = 0;
        }
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 48. 旋转图像

给定一个n×n的二维矩阵matrix表示一个图像。请你将图像顺时针旋转90度。

**答案**：

思路1：转置+反转
- 时间复杂度：O(n²)
- 空间复杂度：O(1)
- 思路：先转置矩阵，然后反转每一行

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        
        // 转置
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
        
        // 反转每一行
        for (int i = 0; i < n; i++) {
            reverse(matrix[i].begin(), matrix[i].end());
        }
    }
};
```

思路2：原地旋转
- 时间复杂度：O(n²)
- 空间复杂度：O(1)
- 思路：一次旋转四个位置

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < (n + 1) / 2; i++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = temp;
            }
        }
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 54. 螺旋矩阵

给你一个m行n列的矩阵matrix，请按照顺时针螺旋顺序，返回矩阵中的所有元素。

**答案**：

思路：模拟
- 时间复杂度：O(mn)
- 空间复杂度：O(1)
- 思路：按照右、下、左、上的顺序遍历，使用边界控制

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> result;
        int top = 0, bottom = matrix.size() - 1;
        int left = 0, right = matrix[0].size() - 1;
        
        while (top <= bottom && left <= right) {
            // 右
            for (int j = left; j <= right; j++) {
                result.push_back(matrix[top][j]);
            }
            top++;
            
            // 下
            for (int i = top; i <= bottom; i++) {
                result.push_back(matrix[i][right]);
            }
            right--;
            
            // 左
            if (top <= bottom) {
                for (int j = right; j >= left; j--) {
                    result.push_back(matrix[bottom][j]);
                }
                bottom--;
            }
            
            // 上
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    result.push_back(matrix[i][left]);
                }
                left++;
            }
        }
        
        return result;
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 59. 螺旋矩阵 II

给你一个正整数n，生成一个包含1到n²所有元素，且元素按顺时针顺序螺旋排列的n×n正方形矩阵matrix。

**答案**：

思路：模拟
- 时间复杂度：O(n²)
- 空间复杂度：O(1)
- 思路：按照右、下、左、上的顺序填充数字

```cpp
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> matrix(n, vector<int>(n, 0));
        int num = 1;
        int top = 0, bottom = n - 1, left = 0, right = n - 1;
        
        while (num <= n * n) {
            // 右
            for (int j = left; j <= right; j++) {
                matrix[top][j] = num++;
            }
            top++;
            
            // 下
            for (int i = top; i <= bottom; i++) {
                matrix[i][right] = num++;
            }
            right--;
            
            // 左
            for (int j = right; j >= left; j--) {
                matrix[bottom][j] = num++;
            }
            bottom--;
            
            // 上
            for (int i = bottom; i >= top; i--) {
                matrix[i][left] = num++;
            }
            left++;
        }
        
        return matrix;
    }
};
```

---

### 贪心

#### 卡片 1

**问题**：LeetCode 56. 合并区间

以数组intervals表示若干个区间的集合，其中单个区间为intervals[i]=[starti,endi]。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

**答案**：

思路：排序+合并
- 时间复杂度：O(nlogn)
- 空间复杂度：O(1)（不考虑结果数组）
- 思路：先按起始位置排序，然后合并重叠的区间

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        
        vector<vector<int>> result;
        for (auto& interval : intervals) {
            if (result.empty() || result.back()[1] < interval[0]) {
                result.push_back(interval);
            } else {
                result.back()[1] = max(result.back()[1], interval[1]);
            }
        }
        
        return result;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 57. 插入区间

给你一个无重叠的，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

**答案**：

思路：一次遍历
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：找到插入位置，合并重叠区间

```cpp
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        vector<vector<int>> result;
        int i = 0;
        
        // 添加所有在新区间之前的区间
        while (i < intervals.size() && intervals[i][1] < newInterval[0]) {
            result.push_back(intervals[i++]);
        }
        
        // 合并重叠区间
        while (i < intervals.size() && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = min(newInterval[0], intervals[i][0]);
            newInterval[1] = max(newInterval[1], intervals[i][1]);
            i++;
        }
        result.push_back(newInterval);
        
        // 添加剩余的区间
        while (i < intervals.size()) {
            result.push_back(intervals[i++]);
        }
        
        return result;
    }
};
```

---


## 链表

#### 卡片 1

**问题**：LeetCode 206. 反转链表

给你单链表的头节点head，请你反转链表，并返回反转后的链表。

**答案**：

思路1：迭代
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用三个指针，逐个反转节点

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;
        
        while (curr) {
            ListNode* next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        
        return prev;
    }
};
```

思路2：递归
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：递归反转后面的链表，然后反转当前节点

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) return head;
        
        ListNode* newHead = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        
        return newHead;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 24. 两两交换链表中的节点

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

**答案**：

思路1：递归
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：递归交换每两个节点

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) return head;
        
        ListNode* first = head;
        ListNode* second = head->next;
        
        first->next = swapPairs(second->next);
        second->next = first;
        
        return second;
    }
};
```

思路2：迭代
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用虚拟头节点，逐个交换相邻节点

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* prev = dummy;
        
        while (prev->next && prev->next->next) {
            ListNode* first = prev->next;
            ListNode* second = prev->next->next;
            
            prev->next = second;
            first->next = second->next;
            second->next = first;
            
            prev = first;
        }
        
        return dummy->next;
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 25. K个一组翻转链表

给你链表的头节点head，每k个节点一组进行翻转，请你返回修改后的链表。

**答案**：

思路：递归+迭代
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：先检查是否有k个节点，然后翻转这k个节点，递归处理后续节点

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* curr = head;
        int count = 0;
        
        // 检查是否有k个节点
        while (curr && count < k) {
            curr = curr->next;
            count++;
        }
        
        if (count == k) {
            // 递归处理后续节点
            curr = reverseKGroup(curr, k);
            
            // 翻转当前k个节点
            while (count > 0) {
                ListNode* next = head->next;
                head->next = curr;
                curr = head;
                head = next;
                count--;
            }
            head = curr;
        }
        
        return head;
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 61. 旋转链表

给你一个链表的头节点head，旋转链表，将链表每个节点向右移动k个位置。

**答案**：

思路：闭合为环
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：先找到链表长度，将链表闭合为环，然后找到新的头节点

```cpp
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (!head || !head->next || k == 0) return head;
        
        // 计算链表长度并找到尾节点
        int n = 1;
        ListNode* tail = head;
        while (tail->next) {
            tail = tail->next;
            n++;
        }
        
        // 闭合为环
        tail->next = head;
        
        // 找到新的尾节点
        k = k % n;
        for (int i = 0; i < n - k; i++) {
            tail = tail->next;
        }
        
        // 断开环
        ListNode* newHead = tail->next;
        tail->next = nullptr;
        
        return newHead;
    }
};
```

---

#### 卡片 5

**问题**：LeetCode 83. 删除排序链表中的重复元素

给定一个已排序的链表的头head，删除所有重复的元素，使每个元素只出现一次。返回已排序的链表。

**答案**：

思路：一次遍历
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：遍历链表，如果当前节点和下一个节点值相同，则删除下一个节点

```cpp
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* curr = head;
        while (curr && curr->next) {
            if (curr->val == curr->next->val) {
                curr->next = curr->next->next;
            } else {
                curr = curr->next;
            }
        }
        return head;
    }
};
```

---

#### 卡片 6

**问题**：LeetCode 92. 反转链表 II

给你单链表的头指针head和两个整数left和right，其中left<=right。请你反转从位置left到位置right的链表节点，返回反转后的链表。

**答案**：

思路：一次遍历
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：找到需要反转的区间，然后反转这部分链表

```cpp
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* prev = dummy;
        
        // 找到left位置的前一个节点
        for (int i = 0; i < left - 1; i++) {
            prev = prev->next;
        }
        
        // 反转从left到right的节点
        ListNode* curr = prev->next;
        for (int i = 0; i < right - left; i++) {
            ListNode* next = curr->next;
            curr->next = next->next;
            next->next = prev->next;
            prev->next = next;
        }
        
        return dummy->next;
    }
};
```

---

#### 卡片 7

**问题**：LeetCode 82. 删除排序链表中的重复元素 II

给定一个已排序的链表的头head，删除所有重复的元素，使每个元素只出现一次。返回已排序的链表。

**答案**：

思路：一次遍历
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用虚拟头节点，跳过所有重复的节点

```cpp
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* prev = dummy;
        
        while (head) {
            if (head->next && head->val == head->next->val) {
                while (head->next && head->val == head->next->val) {
                    head = head->next;
                }
                prev->next = head->next;
            } else {
                prev = prev->next;
            }
            head = head->next;
        }
        
        return dummy->next;
    }
};
```

---

#### 卡片 8

**问题**：LeetCode 86. 分隔链表

给你一个链表的头节点head和一个特定值x，请你对链表进行分隔，使得所有小于x的节点都出现在大于或等于x的节点之前。

**答案**：

思路：双链表
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：创建两个链表，分别存储小于x和大于等于x的节点

```cpp
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* small = new ListNode(0);
        ListNode* large = new ListNode(0);
        ListNode* smallHead = small;
        ListNode* largeHead = large;
        
        while (head) {
            if (head->val < x) {
                small->next = head;
                small = small->next;
            } else {
                large->next = head;
                large = large->next;
            }
            head = head->next;
        }
        
        large->next = nullptr;
        small->next = largeHead->next;
        
        return smallHead->next;
    }
};
```

---

### 分治

#### 卡片 1

**问题**：LeetCode 23. 合并K个升序链表

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

**答案**：

思路1：顺序合并
- 时间复杂度：O(k²n)，k为链表数量，n为平均长度
- 空间复杂度：O(1)
- 思路：依次合并每个链表

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        
        ListNode* result = lists[0];
        for (int i = 1; i < lists.size(); i++) {
            result = mergeTwoLists(result, lists[i]);
        }
        return result;
    }
    
private:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                curr->next = l1;
                l1 = l1->next;
            } else {
                curr->next = l2;
                l2 = l2->next;
            }
            curr = curr->next;
        }
        curr->next = l1 ? l1 : l2;
        return dummy->next;
    }
};
```

思路2：分治合并
- 时间复杂度：O(kn×logk)
- 空间复杂度：O(logk)（递归栈）
- 思路：将k个链表两两合并，递归进行

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return merge(lists, 0, lists.size() - 1);
    }
    
private:
    ListNode* merge(vector<ListNode*>& lists, int left, int right) {
        if (left > right) return nullptr;
        if (left == right) return lists[left];
        
        int mid = (left + right) / 2;
        return mergeTwoLists(merge(lists, left, mid), merge(lists, mid + 1, right));
    }
    
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                curr->next = l1;
                l1 = l1->next;
            } else {
                curr->next = l2;
                l2 = l2->next;
            }
            curr = curr->next;
        }
        curr->next = l1 ? l1 : l2;
        return dummy->next;
    }
};
```

思路3：优先队列（堆）
- 时间复杂度：O(kn×logk)
- 空间复杂度：O(k)
- 思路：使用最小堆维护每个链表的当前节点

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
        
        for (ListNode* list : lists) {
            if (list) pq.push(list);
        }
        
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        
        while (!pq.empty()) {
            ListNode* node = pq.top();
            pq.pop();
            curr->next = node;
            curr = curr->next;
            if (node->next) {
                pq.push(node->next);
            }
        }
        
        return dummy->next;
    }
};
```

---

### 双指针

#### 卡片 1

**问题**：LeetCode 19. 删除链表的倒数第N个结点

给你一个链表，删除链表的倒数第n个结点，并且返回链表的头结点。

**答案**：

思路1：两次遍历
- 时间复杂度：O(L)，L为链表长度
- 空间复杂度：O(1)
- 思路：第一次遍历得到链表长度，第二次遍历删除倒数第n个节点

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        int length = 0;
        ListNode* curr = head;
        while (curr) {
            length++;
            curr = curr->next;
        }
        
        if (n == length) {
            return head->next;
        }
        
        curr = head;
        for (int i = 0; i < length - n - 1; i++) {
            curr = curr->next;
        }
        curr->next = curr->next->next;
        
        return head;
    }
};
```

思路2：一次遍历（双指针）
- 时间复杂度：O(L)
- 空间复杂度：O(1)
- 思路：使用快慢指针，快指针先走n步，然后两个指针同时移动

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* fast = dummy;
        ListNode* slow = dummy;
        
        // 快指针先走n+1步
        for (int i = 0; i <= n; i++) {
            fast = fast->next;
        }
        
        // 快慢指针同时移动
        while (fast) {
            fast = fast->next;
            slow = slow->next;
        }
        
        slow->next = slow->next->next;
        return dummy->next;
    }
};
```

---

### 数学

#### 卡片 1

**问题**：LeetCode 2. 两数相加

给你两个非空的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

**答案**：

思路：模拟加法
- 时间复杂度：O(max(m,n))
- 空间复杂度：O(1)（不考虑结果链表）
- 思路：同时遍历两个链表，逐位相加，处理进位

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        int carry = 0;
        
        while (l1 || l2 || carry) {
            int sum = carry;
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }
            carry = sum / 10;
            curr->next = new ListNode(sum % 10);
            curr = curr->next;
        }
        
        return dummy->next;
    }
};
```

---

### 递归

#### 卡片 1

**问题**：LeetCode 21. 合并两个有序链表

将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**答案**：

思路1：迭代
- 时间复杂度：O(m+n)
- 空间复杂度：O(1)
- 思路：使用虚拟头节点，比较两个链表的节点值，逐个合并

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* dummy = new ListNode(0);
        ListNode* curr = dummy;
        
        while (list1 && list2) {
            if (list1->val < list2->val) {
                curr->next = list1;
                list1 = list1->next;
            } else {
                curr->next = list2;
                list2 = list2->next;
            }
            curr = curr->next;
        }
        
        curr->next = list1 ? list1 : list2;
        return dummy->next;
    }
};
```

思路2：递归
- 时间复杂度：O(m+n)
- 空间复杂度：O(m+n)（递归栈）
- 思路：递归地合并链表

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if (!list1) return list2;
        if (!list2) return list1;
        
        if (list1->val < list2->val) {
            list1->next = mergeTwoLists(list1->next, list2);
            return list1;
        } else {
            list2->next = mergeTwoLists(list1, list2->next);
            return list2;
        }
    }
};
```

---


## 字符串

#### 卡片 1

**问题**：LeetCode 14. 最长公共前缀

编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串""。

**答案**：

思路1：横向扫描
- 时间复杂度：O(mn)，m为字符串平均长度，n为字符串数量
- 空间复杂度：O(1)
- 思路：依次比较每个字符串与第一个字符串的公共前缀

```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (strs.empty()) return "";
        
        string prefix = strs[0];
        for (int i = 1; i < strs.size(); i++) {
            while (strs[i].find(prefix) != 0) {
                prefix = prefix.substr(0, prefix.length() - 1);
                if (prefix.empty()) return "";
            }
        }
        return prefix;
    }
};
```

思路2：纵向扫描
- 时间复杂度：O(mn)
- 空间复杂度：O(1)
- 思路：从第一个字符开始，逐列比较所有字符串

```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (strs.empty()) return "";
        
        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0][i];
            for (int j = 1; j < strs.size(); j++) {
                if (i >= strs[j].length() || strs[j][i] != c) {
                    return strs[0].substr(0, i);
                }
            }
        }
        return strs[0];
    }
};
```

思路3：分治法
- 时间复杂度：O(mn)
- 空间复杂度：O(mlogn)
- 思路：将字符串数组分成两部分，分别求公共前缀，然后合并

```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if (strs.empty()) return "";
        return longestCommonPrefix(strs, 0, strs.size() - 1);
    }
    
private:
    string longestCommonPrefix(vector<string>& strs, int left, int right) {
        if (left == right) return strs[left];
        
        int mid = (left + right) / 2;
        string leftPrefix = longestCommonPrefix(strs, left, mid);
        string rightPrefix = longestCommonPrefix(strs, mid + 1, right);
        return commonPrefix(leftPrefix, rightPrefix);
    }
    
    string commonPrefix(string left, string right) {
        int minLen = min(left.length(), right.length());
        for (int i = 0; i < minLen; i++) {
            if (left[i] != right[i]) {
                return left.substr(0, i);
            }
        }
        return left.substr(0, minLen);
    }
};
```

---

### KMP

#### 卡片 1

**问题**：LeetCode 28. 实现 strStr()

给你两个字符串haystack和needle，请你在haystack字符串中找出needle字符串的第一个匹配项的下标（下标从0开始）。

**答案**：

思路1：暴力匹配
- 时间复杂度：O(mn)
- 空间复杂度：O(1)
- 思路：逐个比较每个位置

```cpp
class Solution {
public:
    int strStr(string haystack, string needle) {
        int m = haystack.length(), n = needle.length();
        for (int i = 0; i <= m - n; i++) {
            int j = 0;
            while (j < n && haystack[i + j] == needle[j]) j++;
            if (j == n) return i;
        }
        return -1;
    }
};
```

思路2：KMP算法
- 时间复杂度：O(m+n)
- 空间复杂度：O(n)
- 思路：使用KMP算法的next数组优化匹配过程

```cpp
class Solution {
public:
    int strStr(string haystack, string needle) {
        if (needle.empty()) return 0;
        
        vector<int> next = getNext(needle);
        int i = 0, j = 0;
        
        while (i < haystack.length() && j < needle.length()) {
            if (j == -1 || haystack[i] == needle[j]) {
                i++;
                j++;
            } else {
                j = next[j];
            }
        }
        
        return j == needle.length() ? i - j : -1;
    }
    
private:
    vector<int> getNext(string& pattern) {
        int n = pattern.length();
        vector<int> next(n, -1);
        int i = 0, j = -1;
        
        while (i < n - 1) {
            if (j == -1 || pattern[i] == pattern[j]) {
                i++;
                j++;
                next[i] = j;
            } else {
                j = next[j];
            }
        }
        
        return next;
    }
};
```

---

### 动态规划

#### 卡片 1

**问题**：LeetCode 5. 最长回文子串

给你一个字符串s，找到s中最长的回文子串。

**答案**：

思路1：中心扩展法
- 时间复杂度：O(n²)
- 空间复杂度：O(1)
- 思路：以每个字符或每两个字符为中心，向两边扩展寻找回文串

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        int start = 0, maxLen = 1;
        
        for (int i = 0; i < s.length(); i++) {
            // 奇数长度回文串
            int len1 = expandAroundCenter(s, i, i);
            // 偶数长度回文串
            int len2 = expandAroundCenter(s, i, i + 1);
            
            int len = max(len1, len2);
            if (len > maxLen) {
                maxLen = len;
                start = i - (len - 1) / 2;
            }
        }
        
        return s.substr(start, maxLen);
    }
    
private:
    int expandAroundCenter(string& s, int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }
};
```

思路2：动态规划
- 时间复杂度：O(n²)
- 空间复杂度：O(n²)
- 思路：dp[i][j]表示s[i...j]是否为回文串

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.length();
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        int start = 0, maxLen = 1;
        
        // 单个字符都是回文
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        
        // 两个字符
        for (int i = 0; i < n - 1; i++) {
            if (s[i] == s[i + 1]) {
                dp[i][i + 1] = true;
                start = i;
                maxLen = 2;
            }
        }
        
        // 长度大于2的子串
        for (int len = 3; len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                int j = i + len - 1;
                if (s[i] == s[j] && dp[i + 1][j - 1]) {
                    dp[i][j] = true;
                    start = i;
                    maxLen = len;
                }
            }
        }
        
        return s.substr(start, maxLen);
    }
};
```

思路3：Manacher算法（最优）
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：利用回文串的对称性，避免重复计算

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        string t = "#";
        for (char c : s) {
            t += c;
            t += "#";
        }
        
        int n = t.length();
        vector<int> p(n, 0);
        int center = 0, right = 0;
        int maxLen = 0, centerIndex = 0;
        
        for (int i = 0; i < n; i++) {
            if (i < right) {
                p[i] = min(right - i, p[2 * center - i]);
            }
            
            int left = i - (1 + p[i]);
            int r = i + (1 + p[i]);
            while (left >= 0 && r < n && t[left] == t[r]) {
                p[i]++;
                left--;
                r++;
            }
            
            if (i + p[i] > right) {
                center = i;
                right = i + p[i];
            }
            
            if (p[i] > maxLen) {
                maxLen = p[i];
                centerIndex = i;
            }
        }
        
        int start = (centerIndex - maxLen) / 2;
        return s.substr(start, maxLen);
    }
};
```

---

### 哈希表

#### 卡片 1

**问题**：LeetCode 49. 字母异位词分组

给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。

**答案**：

思路1：排序+哈希表
- 时间复杂度：O(nklogk)，k为字符串平均长度
- 空间复杂度：O(nk)
- 思路：将每个字符串排序后作为key，相同key的字符串为一组

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> map;
        
        for (string& str : strs) {
            string key = str;
            sort(key.begin(), key.end());
            map[key].push_back(str);
        }
        
        vector<vector<string>> result;
        for (auto& pair : map) {
            result.push_back(pair.second);
        }
        
        return result;
    }
};
```

思路2：计数+哈希表
- 时间复杂度：O(nk)
- 空间复杂度：O(nk)
- 思路：使用字符计数作为key，避免排序

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> map;
        
        for (string& str : strs) {
            string key = getKey(str);
            map[key].push_back(str);
        }
        
        vector<vector<string>> result;
        for (auto& pair : map) {
            result.push_back(pair.second);
        }
        
        return result;
    }
    
private:
    string getKey(string& str) {
        vector<int> count(26, 0);
        for (char c : str) {
            count[c - 'a']++;
        }
        string key;
        for (int i = 0; i < 26; i++) {
            key += to_string(count[i]) + '#';
        }
        return key;
    }
};
```

---

### 数学

#### 卡片 1

**问题**：LeetCode 67. 二进制求和

给你两个二进制字符串a和b，返回它们的和，用二进制字符串表示。

**答案**：

思路：模拟加法
- 时间复杂度：O(max(m,n))
- 空间复杂度：O(1)（不考虑结果字符串）
- 思路：从后往前逐位相加，处理进位

```cpp
class Solution {
public:
    string addBinary(string a, string b) {
        string result;
        int i = a.length() - 1, j = b.length() - 1;
        int carry = 0;
        
        while (i >= 0 || j >= 0 || carry) {
            int sum = carry;
            if (i >= 0) sum += a[i--] - '0';
            if (j >= 0) sum += b[j--] - '0';
            
            result = char(sum % 2 + '0') + result;
            carry = sum / 2;
        }
        
        return result;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 43. 字符串相乘

给定两个以字符串形式表示的非负整数num1和num2，返回num1和num2的乘积，它们的乘积也表示为字符串形式。

**答案**：

思路：模拟乘法
- 时间复杂度：O(mn)
- 空间复杂度：O(m+n)
- 思路：模拟竖式乘法，逐位相乘并处理进位

```cpp
class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") return "0";
        
        int m = num1.length(), n = num2.length();
        vector<int> result(m + n, 0);
        
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int mul = (num1[i] - '0') * (num2[j] - '0');
                int p1 = i + j, p2 = i + j + 1;
                int sum = mul + result[p2];
                
                result[p2] = sum % 10;
                result[p1] += sum / 10;
            }
        }
        
        string str;
        int i = 0;
        while (i < result.size() && result[i] == 0) i++;
        for (; i < result.size(); i++) {
            str += to_string(result[i]);
        }
        
        return str;
    }
};
```

---

### 模拟

#### 卡片 1

**问题**：LeetCode 6. Z字形变换

将一个给定字符串s根据给定的行数numRows，以从上往下、从左到右进行Z字形排列。

**答案**：

思路：按行模拟
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用字符串数组存储每一行的字符，按Z字形顺序填充

```cpp
class Solution {
public:
    string convert(string s, int numRows) {
        if (numRows == 1) return s;
        
        vector<string> rows(min(numRows, int(s.length())));
        int curRow = 0;
        bool goingDown = false;
        
        for (char c : s) {
            rows[curRow] += c;
            if (curRow == 0 || curRow == numRows - 1) {
                goingDown = !goingDown;
            }
            curRow += goingDown ? 1 : -1;
        }
        
        string result;
        for (string row : rows) {
            result += row;
        }
        return result;
    }
};
```

---

### 滑动窗口

#### 卡片 1

**问题**：LeetCode 3. 无重复字符的最长子串

给定一个字符串s，请你找出其中不含有重复字符的最长子串的长度。

**答案**：

思路1：滑动窗口（哈希表）
- 时间复杂度：O(n)
- 空间复杂度：O(min(m,n))，m为字符集大小
- 思路：使用哈希表记录字符最后出现的位置，维护一个滑动窗口

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> map;
        int maxLen = 0;
        int start = 0;
        
        for (int end = 0; end < s.length(); end++) {
            if (map.find(s[end]) != map.end() && map[s[end]] >= start) {
                start = map[s[end]] + 1;
            }
            map[s[end]] = end;
            maxLen = max(maxLen, end - start + 1);
        }
        
        return maxLen;
    }
};
```

思路2：滑动窗口（数组优化）
- 时间复杂度：O(n)
- 空间复杂度：O(1)（固定128或256大小的数组）
- 思路：使用数组代替哈希表，适用于ASCII字符

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> charIndex(128, -1);
        int maxLen = 0;
        int start = 0;
        
        for (int end = 0; end < s.length(); end++) {
            if (charIndex[s[end]] >= start) {
                start = charIndex[s[end]] + 1;
            }
            charIndex[s[end]] = end;
            maxLen = max(maxLen, end - start + 1);
        }
        
        return maxLen;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 76. 最小覆盖子串

给你一个字符串s、一个字符串t。返回s中涵盖t所有字符的最小子串。如果s中不存在涵盖t所有字符的子串，则返回空字符串""。

**答案**：

思路：滑动窗口
- 时间复杂度：O(m+n)
- 空间复杂度：O(m)
- 思路：使用滑动窗口，维护一个包含t中所有字符的窗口

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> need, window;
        for (char c : t) need[c]++;
        
        int left = 0, right = 0;
        int valid = 0;
        int start = 0, len = INT_MAX;
        
        while (right < s.length()) {
            char c = s[right++];
            if (need.count(c)) {
                window[c]++;
                if (window[c] == need[c]) valid++;
            }
            
            while (valid == need.size()) {
                if (right - left < len) {
                    start = left;
                    len = right - left;
                }
                
                char d = s[left++];
                if (need.count(d)) {
                    if (window[d] == need[d]) valid--;
                    window[d]--;
                }
            }
        }
        
        return len == INT_MAX ? "" : s.substr(start, len);
    }
};
```

---


## 树

### BFS

#### 卡片 1

**问题**：LeetCode 102. 二叉树的层序遍历

给你二叉树的根节点root，返回其节点值的层序遍历。（即逐层地，从左到右访问所有节点）。

**答案**：

思路：BFS（队列）
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用队列进行广度优先搜索，按层遍历

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> result;
        if (!root) return result;
        
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            int size = q.size();
            vector<int> level;
            
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                level.push_back(node->val);
                
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            
            result.push_back(level);
        }
        
        return result;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 111. 二叉树的最小深度

给定一个二叉树，找出其最小深度。最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**答案**：

思路1：递归（DFS）
- 时间复杂度：O(n)
- 空间复杂度：O(h)
- 思路：递归计算左右子树的最小深度

```cpp
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) return 0;
        if (!root->left && !root->right) return 1;
        
        int minDepth = INT_MAX;
        if (root->left) {
            minDepth = min(minDepth, minDepth(root->left));
        }
        if (root->right) {
            minDepth = min(minDepth, minDepth(root->right));
        }
        
        return minDepth + 1;
    }
};
```

思路2：迭代（BFS）
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用队列进行层序遍历，找到第一个叶子节点

```cpp
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) return 0;
        
        queue<TreeNode*> q;
        q.push(root);
        int depth = 1;
        
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                
                if (!node->left && !node->right) return depth;
                
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            depth++;
        }
        
        return depth;
    }
};
```

---

### DFS

#### 卡片 1

**问题**：LeetCode 104. 二叉树的最大深度

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**答案**：

思路1：递归（DFS）
- 时间复杂度：O(n)
- 空间复杂度：O(h)
- 思路：递归计算左右子树的最大深度

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return 1 + max(maxDepth(root->left), maxDepth(root->right));
    }
};
```

思路2：迭代（BFS）
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用队列进行层序遍历，记录层数

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        
        queue<TreeNode*> q;
        q.push(root);
        int depth = 0;
        
        while (!q.empty()) {
            int size = q.size();
            depth++;
            
            for (int i = 0; i < size; i++) {
                TreeNode* node = q.front();
                q.pop();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
        }
        
        return depth;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 112. 路径总和

给你二叉树的根节点root和一个表示目标和的整数targetSum。判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和targetSum。

**答案**：

思路：递归（DFS）
- 时间复杂度：O(n)
- 空间复杂度：O(h)
- 思路：递归遍历每个节点，检查是否存在路径和等于targetSum

```cpp
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root) return false;
        
        if (!root->left && !root->right) {
            return root->val == targetSum;
        }
        
        return hasPathSum(root->left, targetSum - root->val) ||
               hasPathSum(root->right, targetSum - root->val);
    }
};
```

---

### 二叉搜索树

#### 卡片 1

**问题**：LeetCode 108. 将有序数组转换为二叉搜索树

给你一个整数数组nums，其中元素已经按升序排列，请你将其转换为一棵高度平衡二叉搜索树。

**答案**：

思路：递归（分治）
- 时间复杂度：O(n)
- 空间复杂度：O(logn)
- 思路：每次选择中间元素作为根节点，递归构建左右子树

```cpp
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return buildBST(nums, 0, nums.size() - 1);
    }
    
private:
    TreeNode* buildBST(vector<int>& nums, int left, int right) {
        if (left > right) return nullptr;
        
        int mid = left + (right - left) / 2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = buildBST(nums, left, mid - 1);
        root->right = buildBST(nums, mid + 1, right);
        
        return root;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 98. 验证二叉搜索树

给你一个二叉树的根节点root，判断其是否是一个有效的二叉搜索树。

**答案**：

思路1：递归（上下界）
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：递归检查每个节点是否在有效范围内

```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return isValidBST(root, LONG_MIN, LONG_MAX);
    }
    
private:
    bool isValidBST(TreeNode* root, long minVal, long maxVal) {
        if (!root) return true;
        
        if (root->val <= minVal || root->val >= maxVal) return false;
        
        return isValidBST(root->left, minVal, root->val) &&
               isValidBST(root->right, root->val, maxVal);
    }
};
```

思路2：中序遍历
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：BST的中序遍历是递增的

```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* curr = root;
        long prev = LONG_MIN;
        
        while (curr || !st.empty()) {
            while (curr) {
                st.push(curr);
                curr = curr->left;
            }
            curr = st.top();
            st.pop();
            
            if (curr->val <= prev) return false;
            prev = curr->val;
            
            curr = curr->right;
        }
        
        return true;
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 99. 恢复二叉搜索树

给你二叉搜索树的根节点root，该树中的恰好两个节点的值被错误地交换。请在不改变其结构的情况下，恢复这棵树。

**答案**：

思路：中序遍历
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：中序遍历BST应该是递增的，找到两个位置错误的节点并交换

```cpp
class Solution {
private:
    TreeNode* first = nullptr;
    TreeNode* second = nullptr;
    TreeNode* prev = new TreeNode(INT_MIN);
    
public:
    void recoverTree(TreeNode* root) {
        inorder(root);
        swap(first->val, second->val);
    }
    
private:
    void inorder(TreeNode* root) {
        if (!root) return;
        
        inorder(root->left);
        
        if (prev->val > root->val) {
            if (!first) first = prev;
            second = root;
        }
        prev = root;
        
        inorder(root->right);
    }
};
```

---

### 二叉树

#### 卡片 1

**问题**：LeetCode 94. 二叉树的中序遍历

给定一个二叉树的根节点root，返回它的中序遍历。

**答案**：

思路1：递归
- 时间复杂度：O(n)
- 空间复杂度：O(h)，h为树的高度
- 思路：左-根-右的顺序遍历

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        inorder(root, result);
        return result;
    }
    
private:
    void inorder(TreeNode* root, vector<int>& result) {
        if (!root) return;
        inorder(root->left, result);
        result.push_back(root->val);
        inorder(root->right, result);
    }
};
```

思路2：迭代（栈）
- 时间复杂度：O(n)
- 空间复杂度：O(h)
- 思路：使用栈模拟递归过程

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        stack<TreeNode*> st;
        TreeNode* curr = root;
        
        while (curr || !st.empty()) {
            while (curr) {
                st.push(curr);
                curr = curr->left;
            }
            curr = st.top();
            st.pop();
            result.push_back(curr->val);
            curr = curr->right;
        }
        
        return result;
    }
};
```

思路3：Morris遍历
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：利用树中的空指针，不需要栈

```cpp
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> result;
        TreeNode* curr = root;
        
        while (curr) {
            if (!curr->left) {
                result.push_back(curr->val);
                curr = curr->right;
            } else {
                TreeNode* prev = curr->left;
                while (prev->right && prev->right != curr) {
                    prev = prev->right;
                }
                if (!prev->right) {
                    prev->right = curr;
                    curr = curr->left;
                } else {
                    prev->right = nullptr;
                    result.push_back(curr->val);
                    curr = curr->right;
                }
            }
        }
        
        return result;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 101. 对称二叉树

给你一个二叉树的根节点root，检查它是否轴对称。

**答案**：

思路1：递归
- 时间复杂度：O(n)
- 空间复杂度：O(h)
- 思路：比较左右子树是否镜像对称

```cpp
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        return isMirror(root, root);
    }
    
private:
    bool isMirror(TreeNode* t1, TreeNode* t2) {
        if (!t1 && !t2) return true;
        if (!t1 || !t2) return false;
        return (t1->val == t2->val) &&
               isMirror(t1->left, t2->right) &&
               isMirror(t1->right, t2->left);
    }
};
```

思路2：迭代（队列）
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用队列进行层序遍历，每次比较两个节点

```cpp
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* t1 = q.front(); q.pop();
            TreeNode* t2 = q.front(); q.pop();
            
            if (!t1 && !t2) continue;
            if (!t1 || !t2) return false;
            if (t1->val != t2->val) return false;
            
            q.push(t1->left);
            q.push(t2->right);
            q.push(t1->right);
            q.push(t2->left);
        }
        
        return true;
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 100. 相同的树

给你两棵二叉树的根节点p和q，编写一个函数来检验这两棵树是否相同。

**答案**：

思路1：递归
- 时间复杂度：O(min(m,n))
- 空间复杂度：O(min(m,n))
- 思路：递归比较两棵树的每个节点

```cpp
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (!p && !q) return true;
        if (!p || !q) return false;
        if (p->val != q->val) return false;
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};
```

思路2：迭代（队列）
- 时间复杂度：O(min(m,n))
- 空间复杂度：O(min(m,n))
- 思路：使用队列进行层序遍历，同时比较两棵树

```cpp
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        queue<TreeNode*> queue;
        queue.push(p);
        queue.push(q);
        
        while (!queue.empty()) {
            TreeNode* node1 = queue.front(); queue.pop();
            TreeNode* node2 = queue.front(); queue.pop();
            
            if (!node1 && !node2) continue;
            if (!node1 || !node2) return false;
            if (node1->val != node2->val) return false;
            
            queue.push(node1->left);
            queue.push(node2->left);
            queue.push(node1->right);
            queue.push(node2->right);
        }
        
        return true;
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 110. 平衡二叉树

给定一个二叉树，判断它是否是高度平衡的二叉树。

**答案**：

思路：自底向上递归
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：计算每个节点的高度，如果左右子树高度差大于1，返回-1表示不平衡

```cpp
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        return height(root) != -1;
    }
    
private:
    int height(TreeNode* root) {
        if (!root) return 0;
        
        int leftHeight = height(root->left);
        if (leftHeight == -1) return -1;
        
        int rightHeight = height(root->right);
        if (rightHeight == -1) return -1;
        
        if (abs(leftHeight - rightHeight) > 1) return -1;
        
        return max(leftHeight, rightHeight) + 1;
    }
};
```

---

#### 卡片 5

**问题**：LeetCode 105. 从前序与中序遍历序列构造二叉树

给定两个整数数组preorder和inorder，其中preorder是二叉树的先序遍历，inorder是同一棵树的中序遍历，请构造二叉树并返回其根节点。

**答案**：

思路：递归
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：前序遍历的第一个元素是根节点，在中序遍历中找到根节点，递归构建左右子树

```cpp
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        unordered_map<int, int> map;
        for (int i = 0; i < inorder.size(); i++) {
            map[inorder[i]] = i;
        }
        
        return build(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1, map);
    }
    
private:
    TreeNode* build(vector<int>& preorder, int preStart, int preEnd,
                   vector<int>& inorder, int inStart, int inEnd,
                   unordered_map<int, int>& map) {
        if (preStart > preEnd || inStart > inEnd) return nullptr;
        
        int rootVal = preorder[preStart];
        TreeNode* root = new TreeNode(rootVal);
        
        int rootIndex = map[rootVal];
        int leftSize = rootIndex - inStart;
        
        root->left = build(preorder, preStart + 1, preStart + leftSize,
                          inorder, inStart, rootIndex - 1, map);
        root->right = build(preorder, preStart + leftSize + 1, preEnd,
                           inorder, rootIndex + 1, inEnd, map);
        
        return root;
    }
};
```

---

#### 卡片 6

**问题**：LeetCode 106. 从中序与后序遍历序列构造二叉树

给定两个整数数组inorder和postorder，其中inorder是二叉树的中序遍历，postorder是同一棵树的后序遍历，请你构造并返回这颗二叉树。

**答案**：

思路：递归
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：后序遍历的最后一个元素是根节点，在中序遍历中找到根节点，递归构建左右子树

```cpp
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        unordered_map<int, int> map;
        for (int i = 0; i < inorder.size(); i++) {
            map[inorder[i]] = i;
        }
        
        return build(inorder, 0, inorder.size() - 1, 
                     postorder, 0, postorder.size() - 1, map);
    }
    
private:
    TreeNode* build(vector<int>& inorder, int inStart, int inEnd,
                   vector<int>& postorder, int postStart, int postEnd,
                   unordered_map<int, int>& map) {
        if (inStart > inEnd || postStart > postEnd) return nullptr;
        
        int rootVal = postorder[postEnd];
        TreeNode* root = new TreeNode(rootVal);
        
        int rootIndex = map[rootVal];
        int leftSize = rootIndex - inStart;
        
        root->left = build(inorder, inStart, rootIndex - 1,
                          postorder, postStart, postStart + leftSize - 1, map);
        root->right = build(inorder, rootIndex + 1, inEnd,
                           postorder, postStart + leftSize, postEnd - 1, map);
        
        return root;
    }
};
```

---

### 回溯

#### 卡片 1

**问题**：LeetCode 113. 路径总和 II

给你二叉树的根节点root和一个整数目标和targetSum，找出所有从根节点到叶子节点路径总和等于给定目标和的路径。

**答案**：

思路：回溯+DFS
- 时间复杂度：O(n²)
- 空间复杂度：O(n)
- 思路：使用DFS遍历所有路径，使用回溯记录当前路径

```cpp
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<vector<int>> result;
        vector<int> path;
        backtrack(root, targetSum, path, result);
        return result;
    }
    
private:
    void backtrack(TreeNode* root, int targetSum, vector<int>& path, vector<vector<int>>& result) {
        if (!root) return;
        
        path.push_back(root->val);
        
        if (!root->left && !root->right && root->val == targetSum) {
            result.push_back(path);
        }
        
        backtrack(root->left, targetSum - root->val, path, result);
        backtrack(root->right, targetSum - root->val, path, result);
        
        path.pop_back();
    }
};
```

---

### 递归

#### 卡片 1

**问题**：LeetCode 95. 不同的二叉搜索树 II

给你一个整数n，请你生成并返回所有由n个节点组成且节点值从1到n互不相同的不同二叉搜索树。

**答案**：

思路：递归（分治）
- 时间复杂度：O(4^n/√n)
- 空间复杂度：O(4^n/√n)
- 思路：对于每个根节点，递归生成左右子树的所有可能组合

```cpp
class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        return generate(1, n);
    }
    
private:
    vector<TreeNode*> generate(int start, int end) {
        vector<TreeNode*> result;
        if (start > end) {
            result.push_back(nullptr);
            return result;
        }
        
        for (int i = start; i <= end; i++) {
            vector<TreeNode*> leftTrees = generate(start, i - 1);
            vector<TreeNode*> rightTrees = generate(i + 1, end);
            
            for (TreeNode* left : leftTrees) {
                for (TreeNode* right : rightTrees) {
                    TreeNode* root = new TreeNode(i);
                    root->left = left;
                    root->right = right;
                    result.push_back(root);
                }
            }
        }
        
        return result;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 124. 二叉树中的最大路径和

路径被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中至多出现一次。该路径至少包含一个节点，且不一定经过根节点。

路径和是路径中各节点值的总和。给你一个二叉树的根节点root，返回其最大路径和。

**答案**：

思路：递归
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：对于每个节点，计算经过该节点的最大路径和，同时返回以该节点为端点的最大路径和

```cpp
class Solution {
private:
    int maxSum = INT_MIN;
    
public:
    int maxPathSum(TreeNode* root) {
        maxGain(root);
        return maxSum;
    }
    
private:
    int maxGain(TreeNode* root) {
        if (!root) return 0;
        
        int leftGain = max(maxGain(root->left), 0);
        int rightGain = max(maxGain(root->right), 0);
        
        int pathSum = root->val + leftGain + rightGain;
        maxSum = max(maxSum, pathSum);
        
        return root->val + max(leftGain, rightGain);
    }
};
```

---

### 链表

#### 卡片 1

**问题**：LeetCode 109. 有序链表转换二叉搜索树

给定一个单链表的头节点head，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

**答案**：

思路：快慢指针+递归
- 时间复杂度：O(nlogn)
- 空间复杂度：O(logn)
- 思路：使用快慢指针找到链表中点作为根节点，递归构建左右子树

```cpp
class Solution {
public:
    TreeNode* sortedListToBST(ListNode* head) {
        if (!head) return nullptr;
        if (!head->next) return new TreeNode(head->val);
        
        ListNode* slow = head, *fast = head, *prev = nullptr;
        while (fast && fast->next) {
            prev = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        
        prev->next = nullptr;
        TreeNode* root = new TreeNode(slow->val);
        root->left = sortedListToBST(head);
        root->right = sortedListToBST(slow->next);
        
        return root;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 114. 二叉树展开为链表

给你二叉树的根结点root，请你将它展开为一个单链表。展开后的单链表应该同样使用TreeNode，其中right子指针指向链表中下一个结点，而左子指针始终为null。

**答案**：

思路：递归
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：递归处理左右子树，然后将左子树接到右子树位置，原右子树接到左子树最右节点

```cpp
class Solution {
public:
    void flatten(TreeNode* root) {
        if (!root) return;
        
        flatten(root->left);
        flatten(root->right);
        
        TreeNode* left = root->left;
        TreeNode* right = root->right;
        
        root->left = nullptr;
        root->right = left;
        
        TreeNode* curr = root;
        while (curr->right) {
            curr = curr->right;
        }
        curr->right = right;
    }
};
```

---


## 动态规划

#### 卡片 1

**问题**：LeetCode 70. 爬楼梯

假设你正在爬楼梯。需要n阶你才能到达楼顶。

每次你可以爬1或2个台阶。你有多少种不同的方法可以爬到楼顶？

**答案**：

思路1：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：dp[i]表示到达第i阶的方法数，dp[i]=dp[i-1]+dp[i-2]

```cpp
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 2) return n;
        vector<int> dp(n + 1);
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
};
```

思路2：空间优化
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：只保存前两个状态

```cpp
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 2) return n;
        int prev = 1, curr = 2;
        for (int i = 3; i <= n; i++) {
            int next = prev + curr;
            prev = curr;
            curr = next;
        }
        return curr;
    }
};
```

思路3：矩阵快速幂
- 时间复杂度：O(logn)
- 空间复杂度：O(1)
- 思路：将递推关系转换为矩阵幂运算

```cpp
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 2) return n;
        vector<vector<long>> base = {{1, 1}, {1, 0}};
        vector<vector<long>> result = matrixPower(base, n - 1);
        return result[0][0] + result[0][1];
    }
    
private:
    vector<vector<long>> matrixPower(vector<vector<long>>& base, int n) {
        vector<vector<long>> result = {{1, 0}, {0, 1}};
        while (n > 0) {
            if (n % 2 == 1) {
                result = matrixMultiply(result, base);
            }
            base = matrixMultiply(base, base);
            n /= 2;
        }
        return result;
    }
    
    vector<vector<long>> matrixMultiply(vector<vector<long>>& a, vector<vector<long>>& b) {
        vector<vector<long>> c(2, vector<long>(2, 0));
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 62. 不同路径

一个机器人位于一个m x n网格的左上角。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。

问总共有多少条不同的路径？

**答案**：

思路1：动态规划
- 时间复杂度：O(mn)
- 空间复杂度：O(mn)
- 思路：dp[i][j]表示到达(i,j)的路径数，dp[i][j]=dp[i-1][j]+dp[i][j-1]

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 1));
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        
        return dp[m - 1][n - 1];
    }
};
```

思路2：空间优化
- 时间复杂度：O(mn)
- 空间复杂度：O(n)
- 思路：只保存一行的状态

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> dp(n, 1);
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }
        
        return dp[n - 1];
    }
};
```

思路3：组合数学
- 时间复杂度：O(min(m,n))
- 空间复杂度：O(1)
- 思路：总路径数等于C(m+n-2, m-1)

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        long long result = 1;
        for (int i = 1; i < min(m, n); i++) {
            result = result * (m + n - 1 - i) / i;
        }
        return result;
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 63. 不同路径 II

一个机器人位于一个m x n网格的左上角。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

**答案**：

思路：动态规划
- 时间复杂度：O(mn)
- 空间复杂度：O(mn)
- 思路：dp[i][j]表示到达(i,j)的路径数，如果(i,j)有障碍物，则dp[i][j]=0

```cpp
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        
        vector<vector<int>> dp(m, vector<int>(n, 0));
        
        // 初始化第一行和第一列
        for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++) {
            dp[0][j] = 1;
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        
        return dp[m - 1][n - 1];
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 64. 最小路径和

给定一个包含非负整数的m x n网格grid，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**答案**：

思路：动态规划
- 时间复杂度：O(mn)
- 空间复杂度：O(mn)
- 思路：dp[i][j]表示到达(i,j)的最小路径和，dp[i][j]=min(dp[i-1][j], dp[i][j-1])+grid[i][j]

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        
        vector<vector<int>> dp(m, vector<int>(n, 0));
        dp[0][0] = grid[0][0];
        
        // 初始化第一行和第一列
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        
        return dp[m - 1][n - 1];
    }
};
```

---

#### 卡片 5

**问题**：LeetCode 198. 打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下，一夜之内能够偷窃到的最高金额。

**答案**：

思路：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：dp[i]表示前i间房屋能偷窃到的最高金额，dp[i]=max(dp[i-1], dp[i-2]+nums[i])

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int prev = 0, curr = 0;
        
        for (int num : nums) {
            int temp = curr;
            curr = max(curr, prev + num);
            prev = temp;
        }
        
        return curr;
    }
};
```

---

#### 卡片 6

**问题**：LeetCode 213. 打家劫舍 II

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋围成一圈，这意味着第一个房屋和最后一个房屋是紧挨着的。

**答案**：

思路：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：将问题分解为两种情况：不偷第一间或不偷最后一间

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.size() == 1) return nums[0];
        return max(robRange(nums, 0, nums.size() - 2), 
                   robRange(nums, 1, nums.size() - 1));
    }
    
private:
    int robRange(vector<int>& nums, int start, int end) {
        int prev = 0, curr = 0;
        for (int i = start; i <= end; i++) {
            int temp = curr;
            curr = max(curr, prev + nums[i]);
            prev = temp;
        }
        return curr;
    }
};
```

---

### 字符串

#### 卡片 1

**问题**：LeetCode 139. 单词拆分

给你一个字符串s和一个字符串列表wordDict作为字典。请你判断是否可以利用字典中出现的单词拼接出s。

**答案**：

思路：动态规划
- 时间复杂度：O(n²)
- 空间复杂度：O(n)
- 思路：dp[i]表示s的前i个字符能否被拆分，dp[i]=dp[j]&&wordDict包含s[j:i]

```cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
        vector<bool> dp(s.length() + 1, false);
        dp[0] = true;
        
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordSet.count(s.substr(j, i - j))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        return dp[s.length()];
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 10. 正则表达式匹配

给你一个字符串s和一个字符规律p，请你来实现一个支持'.'和'*'的正则表达式匹配。

**答案**：

思路：动态规划
- 时间复杂度：O(mn)
- 空间复杂度：O(mn)
- 思路：dp[i][j]表示s的前i个字符和p的前j个字符是否匹配

```cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.length(), n = p.length();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        
        for (int j = 2; j <= n; j++) {
            if (p[j - 1] == '*') dp[0][j] = dp[0][j - 2];
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 2] || 
                               (dp[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.'));
                } else {
                    dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
                }
            }
        }
        
        return dp[m][n];
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 72. 编辑距离

给你两个单词word1和word2，请返回将word1转换成word2所使用的最少操作数。

**答案**：

思路：动态规划
- 时间复杂度：O(mn)
- 空间复杂度：O(mn)
- 思路：dp[i][j]表示word1的前i个字符转换为word2的前j个字符的最少操作数

```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.length(), n = word2.length();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                }
            }
        }
        
        return dp[m][n];
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 91. 解码方法

一条包含字母A-Z的消息通过以下映射进行了编码。给你一个只含数字的非空字符串s，请计算并返回解码方法的总数。

**答案**：

思路：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：dp[i]表示前i个字符的解码方法数，考虑单个字符和两个字符的情况

```cpp
class Solution {
public:
    int numDecodings(string s) {
        if (s[0] == '0') return 0;
        
        int prev = 1, curr = 1;
        
        for (int i = 1; i < s.length(); i++) {
            int temp = curr;
            if (s[i] == '0') {
                if (s[i - 1] == '1' || s[i - 1] == '2') {
                    curr = prev;
                } else {
                    return 0;
                }
            } else if (s[i - 1] == '1' || (s[i - 1] == '2' && s[i] <= '6')) {
                curr = prev + curr;
            }
            prev = temp;
        }
        
        return curr;
    }
};
```

---

#### 卡片 5

**问题**：LeetCode 97. 交错字符串

给定三个字符串s1、s2、s3，请你帮忙验证s3是否是由s1和s2交错组成的。

**答案**：

思路：动态规划
- 时间复杂度：O(mn)
- 空间复杂度：O(mn)
- 思路：dp[i][j]表示s1的前i个字符和s2的前j个字符能否组成s3的前i+j个字符

```cpp
class Solution {
public:
    bool isInterleave(string s1, string s2, string s3) {
        int m = s1.length(), n = s2.length();
        if (m + n != s3.length()) return false;
        
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i > 0) {
                    dp[i][j] = dp[i][j] || (dp[i - 1][j] && s1[i - 1] == s3[i + j - 1]);
                }
                if (j > 0) {
                    dp[i][j] = dp[i][j] || (dp[i][j - 1] && s2[j - 1] == s3[i + j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
};
```

---

### 数学

#### 卡片 1

**问题**：LeetCode 279. 完全平方数

给你一个整数n，返回和为n的完全平方数的最少数量。

**答案**：

思路1：动态规划
- 时间复杂度：O(n√n)
- 空间复杂度：O(n)
- 思路：dp[i]表示和为i的完全平方数的最少数量

```cpp
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j * j <= i; j++) {
                dp[i] = min(dp[i], dp[i - j * j] + 1);
            }
        }
        
        return dp[n];
    }
};
```

思路2：数学（四平方定理）
- 时间复杂度：O(√n)
- 空间复杂度：O(1)
- 思路：根据四平方定理，任何正整数都可以表示为4个整数的平方和

```cpp
class Solution {
public:
    int numSquares(int n) {
        // 检查是否为完全平方数
        if (isSquare(n)) return 1;
        
        // 检查是否满足4^a(8b+7)的形式
        int temp = n;
        while (temp % 4 == 0) temp /= 4;
        if (temp % 8 == 7) return 4;
        
        // 检查是否可以表示为两个平方数的和
        for (int i = 1; i * i <= n; i++) {
            if (isSquare(n - i * i)) return 2;
        }
        
        return 3;
    }
    
private:
    bool isSquare(int n) {
        int root = sqrt(n);
        return root * root == n;
    }
};
```

---

### 数组

#### 卡片 1

**问题**：LeetCode 53. 最大子数组和

给你一个整数数组nums，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**答案**：

思路1：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：dp[i]表示以nums[i]结尾的最大子数组和

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int maxSum = nums[0];
        int currentSum = nums[0];
        
        for (int i = 1; i < nums.size(); i++) {
            currentSum = max(nums[i], currentSum + nums[i]);
            maxSum = max(maxSum, currentSum);
        }
        
        return maxSum;
    }
};
```

思路2：分治法
- 时间复杂度：O(nlogn)
- 空间复杂度：O(logn)
- 思路：将数组分为左右两部分，最大子数组和要么在左边，要么在右边，要么跨越中间

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        return maxSubArrayHelper(nums, 0, nums.size() - 1);
    }
    
private:
    int maxSubArrayHelper(vector<int>& nums, int left, int right) {
        if (left == right) return nums[left];
        
        int mid = (left + right) / 2;
        int leftMax = maxSubArrayHelper(nums, left, mid);
        int rightMax = maxSubArrayHelper(nums, mid + 1, right);
        
        // 跨越中间的最大子数组和
        int leftSum = INT_MIN, rightSum = INT_MIN, sum = 0;
        for (int i = mid; i >= left; i--) {
            sum += nums[i];
            leftSum = max(leftSum, sum);
        }
        sum = 0;
        for (int i = mid + 1; i <= right; i++) {
            sum += nums[i];
            rightSum = max(rightSum, sum);
        }
        
        return max({leftMax, rightMax, leftSum + rightSum});
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 152. 乘积最大子数组

给你一个整数数组nums，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

**答案**：

思路：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：由于负数相乘会变正，需要同时记录最大值和最小值

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int maxProd = nums[0];
        int minProd = nums[0];
        int result = nums[0];
        
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] < 0) {
                swap(maxProd, minProd);
            }
            
            maxProd = max(nums[i], maxProd * nums[i]);
            minProd = min(nums[i], minProd * nums[i]);
            
            result = max(result, maxProd);
        }
        
        return result;
    }
};
```

---

### 栈

#### 卡片 1

**问题**：LeetCode 32. 最长有效括号

给你一个只包含'('和')'的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**答案**：

思路1：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：dp[i]表示以i结尾的最长有效括号长度

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.length();
        vector<int> dp(n, 0);
        int maxLen = 0;
        
        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + (i - dp[i - 1] >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxLen = max(maxLen, dp[i]);
            }
        }
        
        return maxLen;
    }
};
```

思路2：栈
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用栈存储左括号的索引

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        stack<int> st;
        st.push(-1);
        int maxLen = 0;
        
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                st.push(i);
            } else {
                st.pop();
                if (st.empty()) {
                    st.push(i);
                } else {
                    maxLen = max(maxLen, i - st.top());
                }
            }
        }
        
        return maxLen;
    }
};
```

---

### 树

#### 卡片 1

**问题**：LeetCode 96. 不同的二叉搜索树

给你一个整数n，求恰由n个节点组成且节点值从1到n互不相同的二叉搜索树有多少种？返回满足题意的二叉搜索树的种数。

**答案**：

思路：动态规划（卡特兰数）
- 时间复杂度：O(n²)
- 空间复杂度：O(n)
- 思路：dp[i]表示i个节点能组成的BST数量，dp[i]=Σ(dp[j-1]*dp[i-j])，j从1到i

```cpp
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;
        
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        
        return dp[n];
    }
};
```

---

### 矩阵

#### 卡片 1

**问题**：LeetCode 221. 最大正方形

在一个由'0'和'1'组成的二维矩阵内，找到只包含'1'的最大正方形，并返回其面积。

**答案**：

思路：动态规划
- 时间复杂度：O(mn)
- 空间复杂度：O(mn)
- 思路：dp[i][j]表示以(i,j)为右下角的最大正方形边长

```cpp
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> dp(m, vector<int>(n, 0));
        int maxSide = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                    }
                    maxSide = max(maxSide, dp[i][j]);
                }
            }
        }
        
        return maxSide * maxSide;
    }
};
```

---

### 股票

#### 卡片 1

**问题**：LeetCode 121. 买卖股票的最佳时机

给定一个数组prices，它的第i个元素prices[i]表示一支给定股票第i天的价格。

你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。设计一个算法来计算你所能获取的最大利润。

**答案**：

思路：一次遍历
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：记录最低价格，计算每天卖出能获得的最大利润

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minPrice = INT_MAX;
        int maxProfit = 0;
        
        for (int price : prices) {
            if (price < minPrice) {
                minPrice = price;
            } else if (price - minPrice > maxProfit) {
                maxProfit = price - minPrice;
            }
        }
        
        return maxProfit;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 123. 买卖股票的最佳时机 III

给定一个数组，它的第i个元素是一支给定的股票在第i天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成两笔交易。

**答案**：

思路：动态规划
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：使用四个状态变量表示第一次买入、第一次卖出、第二次买入、第二次卖出的最大利润

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int buy1 = INT_MIN, sell1 = 0;
        int buy2 = INT_MIN, sell2 = 0;
        
        for (int price : prices) {
            buy1 = max(buy1, -price);
            sell1 = max(sell1, buy1 + price);
            buy2 = max(buy2, sell1 - price);
            sell2 = max(sell2, buy2 + price);
        }
        
        return sell2;
    }
};
```

---


## 回溯

#### 卡片 1

**问题**：LeetCode 17. 电话号码的字母组合

给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。答案可以按任意顺序返回。

**答案**：

思路：回溯
- 时间复杂度：O(4^n×n)，n为数字个数
- 空间复杂度：O(n)
- 思路：使用回溯算法生成所有可能的字母组合

```cpp
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};
        
        vector<string> result;
        string current;
        backtrack(digits, 0, current, result);
        return result;
    }
    
private:
    vector<string> phoneMap = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    
    void backtrack(string& digits, int index, string& current, vector<string>& result) {
        if (index == digits.length()) {
            result.push_back(current);
            return;
        }
        
        string letters = phoneMap[digits[index] - '0'];
        for (char letter : letters) {
            current.push_back(letter);
            backtrack(digits, index + 1, current, result);
            current.pop_back();
        }
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 22. 括号生成

数字n代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且有效的括号组合。

**答案**：

思路：回溯
- 时间复杂度：O(4^n/√n)
- 空间复杂度：O(n)
- 思路：使用回溯算法，确保左括号数量始终大于等于右括号数量

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        string current;
        backtrack(n, 0, 0, current, result);
        return result;
    }
    
private:
    void backtrack(int n, int left, int right, string& current, vector<string>& result) {
        if (current.length() == 2 * n) {
            result.push_back(current);
            return;
        }
        
        if (left < n) {
            current += '(';
            backtrack(n, left + 1, right, current, result);
            current.pop_back();
        }
        
        if (right < left) {
            current += ')';
            backtrack(n, left, right + 1, current, result);
            current.pop_back();
        }
    }
};
```

---

#### 卡片 3

**问题**：LeetCode 46. 全排列

给定一个不含重复数字的数组nums，返回其所有可能的全排列。你可以按任意顺序返回答案。

**答案**：

思路：回溯
- 时间复杂度：O(n×n!)
- 空间复杂度：O(n)
- 思路：使用回溯算法生成所有排列，使用visited数组标记已使用的元素

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> current;
        vector<bool> visited(nums.size(), false);
        backtrack(nums, current, visited, result);
        return result;
    }
    
private:
    void backtrack(vector<int>& nums, vector<int>& current, vector<bool>& visited, vector<vector<int>>& result) {
        if (current.size() == nums.size()) {
            result.push_back(current);
            return;
        }
        
        for (int i = 0; i < nums.size(); i++) {
            if (!visited[i]) {
                visited[i] = true;
                current.push_back(nums[i]);
                backtrack(nums, current, visited, result);
                current.pop_back();
                visited[i] = false;
            }
        }
    }
};
```

---

#### 卡片 4

**问题**：LeetCode 39. 组合总和

给你一个无重复元素的整数数组candidates和一个目标整数target，找出candidates中可以使数字和为目标数target的所有不同组合，并以列表形式返回。

**答案**：

思路：回溯
- 时间复杂度：O(2^n)
- 空间复杂度：O(target)
- 思路：使用回溯算法，可以重复使用同一个数字

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> result;
        vector<int> current;
        backtrack(candidates, target, 0, current, result);
        return result;
    }
    
private:
    void backtrack(vector<int>& candidates, int target, int start, vector<int>& current, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(current);
            return;
        }
        
        if (target < 0) return;
        
        for (int i = start; i < candidates.size(); i++) {
            current.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i, current, result);
            current.pop_back();
        }
    }
};
```

---

#### 卡片 5

**问题**：LeetCode 40. 组合总和 II

给定一个候选人编号的集合candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。

candidates中的每个数字在每个组合中只能使用一次。

**答案**：

思路：回溯+去重
- 时间复杂度：O(2^n)
- 空间复杂度：O(target)
- 思路：使用回溯算法，跳过重复元素避免重复组合

```cpp
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> result;
        vector<int> current;
        backtrack(candidates, target, 0, current, result);
        return result;
    }
    
private:
    void backtrack(vector<int>& candidates, int target, int start, vector<int>& current, vector<vector<int>>& result) {
        if (target == 0) {
            result.push_back(current);
            return;
        }
        
        for (int i = start; i < candidates.size(); i++) {
            if (i > start && candidates[i] == candidates[i - 1]) continue;
            if (candidates[i] > target) break;
            
            current.push_back(candidates[i]);
            backtrack(candidates, target - candidates[i], i + 1, current, result);
            current.pop_back();
        }
    }
};
```

---

#### 卡片 6

**问题**：LeetCode 47. 全排列 II

给定一个可包含重复数字的序列nums，按任意顺序返回所有不重复的全排列。

**答案**：

思路：回溯+去重
- 时间复杂度：O(n×n!)
- 空间复杂度：O(n)
- 思路：使用回溯算法，通过排序和剪枝避免重复排列

```cpp
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result;
        vector<int> current;
        vector<bool> visited(nums.size(), false);
        backtrack(nums, current, visited, result);
        return result;
    }
    
private:
    void backtrack(vector<int>& nums, vector<int>& current, vector<bool>& visited, vector<vector<int>>& result) {
        if (current.size() == nums.size()) {
            result.push_back(current);
            return;
        }
        
        for (int i = 0; i < nums.size(); i++) {
            if (visited[i] || (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])) {
                continue;
            }
            
            visited[i] = true;
            current.push_back(nums[i]);
            backtrack(nums, current, visited, result);
            current.pop_back();
            visited[i] = false;
        }
    }
};
```

---

#### 卡片 7

**问题**：LeetCode 90. 子集 II

给你一个整数数组nums，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

**答案**：

思路：回溯+去重
- 时间复杂度：O(n×2^n)
- 空间复杂度：O(n)
- 思路：使用回溯算法，通过排序和剪枝避免重复子集

```cpp
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result;
        vector<int> current;
        backtrack(nums, 0, current, result);
        return result;
    }
    
private:
    void backtrack(vector<int>& nums, int start, vector<int>& current, vector<vector<int>>& result) {
        result.push_back(current);
        
        for (int i = start; i < nums.size(); i++) {
            if (i > start && nums[i] == nums[i - 1]) continue;
            current.push_back(nums[i]);
            backtrack(nums, i + 1, current, result);
            current.pop_back();
        }
    }
};
```

---

### DFS

#### 卡片 1

**问题**：LeetCode 79. 单词搜索

给定一个m×n二维字符网格board和一个字符串单词word。如果word存在于网格中，返回true；否则，返回false。

**答案**：

思路：回溯+DFS
- 时间复杂度：O(mn×4^L)，L为单词长度
- 空间复杂度：O(L)
- 思路：从每个位置开始，使用DFS搜索单词，使用回溯避免重复访问

```cpp
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size();
        int n = board[0].size();
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (backtrack(board, word, i, j, 0)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
private:
    bool backtrack(vector<vector<char>>& board, string& word, int i, int j, int index) {
        if (index == word.length()) return true;
        
        if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || 
            board[i][j] != word[index]) {
            return false;
        }
        
        char temp = board[i][j];
        board[i][j] = '#'; // 标记为已访问
        
        bool found = backtrack(board, word, i + 1, j, index + 1) ||
                     backtrack(board, word, i - 1, j, index + 1) ||
                     backtrack(board, word, i, j + 1, index + 1) ||
                     backtrack(board, word, i, j - 1, index + 1);
        
        board[i][j] = temp; // 恢复
        
        return found;
    }
};
```

---

### 位运算

#### 卡片 1

**问题**：LeetCode 78. 子集

给你一个整数数组nums，数组中的元素互不相同。返回该数组所有可能的子集（幂集）。

**答案**：

思路1：回溯
- 时间复杂度：O(n×2^n)
- 空间复杂度：O(n)
- 思路：使用回溯算法生成所有子集

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> current;
        backtrack(nums, 0, current, result);
        return result;
    }
    
private:
    void backtrack(vector<int>& nums, int start, vector<int>& current, vector<vector<int>>& result) {
        result.push_back(current);
        
        for (int i = start; i < nums.size(); i++) {
            current.push_back(nums[i]);
            backtrack(nums, i + 1, current, result);
            current.pop_back();
        }
    }
};
```

思路2：位运算
- 时间复杂度：O(n×2^n)
- 空间复杂度：O(1)（不考虑结果数组）
- 思路：使用位掩码表示每个元素是否在子集中

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        int n = nums.size();
        
        for (int mask = 0; mask < (1 << n); mask++) {
            vector<int> subset;
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) {
                    subset.push_back(nums[i]);
                }
            }
            result.push_back(subset);
        }
        
        return result;
    }
};
```

---

### 字符串

#### 卡片 1

**问题**：LeetCode 93. 复原IP地址

有效IP地址正好由四个整数（每个整数位于0到255之间组成，且不能含有前导0），整数之间用'.'分隔。

给定一个只包含数字的字符串s，用以表示一个IP地址，返回所有可能的有效IP地址。

**答案**：

思路：回溯
- 时间复杂度：O(1)
- 空间复杂度：O(1)
- 思路：使用回溯算法，将字符串分成四段，每段验证是否有效

```cpp
class Solution {
public:
    vector<string> restoreIpAddresses(string s) {
        vector<string> result;
        vector<string> current;
        backtrack(s, 0, current, result);
        return result;
    }
    
private:
    void backtrack(string& s, int start, vector<string>& current, vector<string>& result) {
        if (current.size() == 4) {
            if (start == s.length()) {
                result.push_back(current[0] + "." + current[1] + "." + current[2] + "." + current[3]);
            }
            return;
        }
        
        for (int len = 1; len <= 3 && start + len <= s.length(); len++) {
            string segment = s.substr(start, len);
            if (isValid(segment)) {
                current.push_back(segment);
                backtrack(s, start + len, current, result);
                current.pop_back();
            }
        }
    }
    
    bool isValid(string& segment) {
        if (segment.length() > 1 && segment[0] == '0') return false;
        int num = stoi(segment);
        return num >= 0 && num <= 255;
    }
};
```

---


## 贪心

#### 卡片 1

**问题**：LeetCode 55. 跳跃游戏

给你一个非负整数数组nums，你最初位于数组的第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

**答案**：

思路：贪心
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：维护能到达的最远位置，如果当前位置超过最远位置，则无法到达

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int maxReach = 0;
        
        for (int i = 0; i < nums.size(); i++) {
            if (i > maxReach) return false;
            maxReach = max(maxReach, i + nums[i]);
            if (maxReach >= nums.size() - 1) return true;
        }
        
        return true;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 45. 跳跃游戏 II

给你一个非负整数数组nums，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

**答案**：

思路：贪心
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：每次选择能跳得最远的位置

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        int jumps = 0, end = 0, farthest = 0;
        
        for (int i = 0; i < nums.size() - 1; i++) {
            farthest = max(farthest, i + nums[i]);
            if (i == end) {
                jumps++;
                end = farthest;
            }
        }
        
        return jumps;
    }
};
```

---

### 股票

#### 卡片 1

**问题**：LeetCode 122. 买卖股票的最佳时机 II

给你一个整数数组prices，其中prices[i]表示某支股票第i天的价格。

在每一天，你可以决定是否购买和/或出售股票。你在任何时候最多只能持有一股股票。你也可以先购买，然后在同一天出售。

返回你能获得的最大利润。

**答案**：

思路：贪心
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：只要后一天价格高于前一天，就进行交易

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int profit = 0;
        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] > prices[i - 1]) {
                profit += prices[i] - prices[i - 1];
            }
        }
        return profit;
    }
};
```

---


## 栈

#### 卡片 1

**问题**：LeetCode 20. 有效的括号

给定一个只包括'('，')'，'{'，'}'，'['，']'的字符串s，判断字符串是否有效。

**答案**：

思路：栈
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：遇到左括号入栈，遇到右括号检查是否与栈顶匹配

```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        for (char c : s) {
            if (c == '(' || c == '[' || c == '{') {
                st.push(c);
            } else {
                if (st.empty()) return false;
                char top = st.top();
                st.pop();
                if ((c == ')' && top != '(') ||
                    (c == ']' && top != '[') ||
                    (c == '}' && top != '{')) {
                    return false;
                }
            }
        }
        return st.empty();
    }
};
```

---

### 动态规划

#### 卡片 1

**问题**：LeetCode 85. 最大矩形

给定一个仅包含0和1、大小为rows×cols的二维二进制矩阵，找出只包含1的最大矩形，并返回其面积。

**答案**：

思路：单调栈
- 时间复杂度：O(mn)
- 空间复杂度：O(n)
- 思路：将问题转化为多个柱状图最大矩形问题

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        if (matrix.empty()) return 0;
        
        int m = matrix.size(), n = matrix[0].size();
        vector<int> heights(n, 0);
        int maxArea = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                heights[j] = (matrix[i][j] == '1') ? heights[j] + 1 : 0;
            }
            maxArea = max(maxArea, largestRectangleArea(heights));
        }
        
        return maxArea;
    }
    
private:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        int maxArea = 0;
        
        for (int i = 0; i <= heights.size(); i++) {
            int h = (i == heights.size()) ? 0 : heights[i];
            while (!st.empty() && heights[st.top()] > h) {
                int height = heights[st.top()];
                st.pop();
                int width = st.empty() ? i : i - st.top() - 1;
                maxArea = max(maxArea, height * width);
            }
            st.push(i);
        }
        
        return maxArea;
    }
};
```

---

### 单调栈

#### 卡片 1

**问题**：LeetCode 84. 柱状图中最大的矩形

给定n个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为1。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

**答案**：

思路：单调栈
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用单调递增栈，找到每个柱子左右两边第一个比它矮的柱子

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> st;
        int maxArea = 0;
        
        for (int i = 0; i <= heights.size(); i++) {
            int h = (i == heights.size()) ? 0 : heights[i];
            
            while (!st.empty() && heights[st.top()] > h) {
                int height = heights[st.top()];
                st.pop();
                int width = st.empty() ? i : i - st.top() - 1;
                maxArea = max(maxArea, height * width);
            }
            
            st.push(i);
        }
        
        return maxArea;
    }
};
```

---

### 双指针

#### 卡片 1

**问题**：LeetCode 42. 接雨水

给定n个非负整数表示每个宽度为1的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**答案**：

思路1：双指针
- 时间复杂度：O(n)
- 空间复杂度：O(1)
- 思路：从两端向中间遍历，维护左右最大高度

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        int result = 0;
        
        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    result += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    result += rightMax - height[right];
                }
                right--;
            }
        }
        
        return result;
    }
};
```

思路2：栈
- 时间复杂度：O(n)
- 空间复杂度：O(n)
- 思路：使用栈存储递减的柱子索引，遇到更高的柱子时计算雨水

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        stack<int> st;
        int result = 0;
        
        for (int i = 0; i < height.size(); i++) {
            while (!st.empty() && height[i] > height[st.top()]) {
                int top = st.top();
                st.pop();
                if (st.empty()) break;
                
                int distance = i - st.top() - 1;
                int boundedHeight = min(height[i], height[st.top()]) - height[top];
                result += distance * boundedHeight;
            }
            st.push(i);
        }
        
        return result;
    }
};
```

---

### 设计

#### 卡片 1

**问题**：LeetCode 155. 最小栈

设计一个支持push，pop，top操作，并能在常数时间内检索到最小元素的栈。

**答案**：

思路：辅助栈
- 时间复杂度：O(1)所有操作
- 空间复杂度：O(n)
- 思路：使用两个栈，一个存储元素，一个存储最小值

```cpp
class MinStack {
private:
    stack<int> dataStack;
    stack<int> minStack;
    
public:
    MinStack() {
        minStack.push(INT_MAX);
    }
    
    void push(int val) {
        dataStack.push(val);
        minStack.push(min(val, minStack.top()));
    }
    
    void pop() {
        dataStack.pop();
        minStack.pop();
    }
    
    int top() {
        return dataStack.top();
    }
    
    int getMin() {
        return minStack.top();
    }
};
```

---


## 数学

#### 卡片 1

**问题**：LeetCode 7. 整数反转

给你一个32位的有符号整数x，返回将x中的数字部分反转后的结果。

如果反转后整数超过32位的有符号整数的范围[−2³¹, 2³¹−1]，就返回0。

**答案**：

思路：数学方法
- 时间复杂度：O(log|x|)
- 空间复杂度：O(1)
- 思路：每次取最后一位数字，检查是否溢出

```cpp
class Solution {
public:
    int reverse(int x) {
        int result = 0;
        while (x != 0) {
            int digit = x % 10;
            x /= 10;
            
            // 检查溢出
            if (result > INT_MAX / 10 || (result == INT_MAX / 10 && digit > 7)) return 0;
            if (result < INT_MIN / 10 || (result == INT_MIN / 10 && digit < -8)) return 0;
            
            result = result * 10 + digit;
        }
        return result;
    }
};
```

---

#### 卡片 2

**问题**：LeetCode 9. 回文数

给你一个整数x，如果x是一个回文整数，返回true；否则，返回false。

**答案**：

思路1：转换为字符串
- 时间复杂度：O(log|x|)
- 空间复杂度：O(log|x|)
- 思路：将整数转换为字符串，然后判断是否为回文

```cpp
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;
        string s = to_string(x);
        int left = 0, right = s.length() - 1;
        while (left < right) {
            if (s[left] != s[right]) return false;
            left++;
            right--;
        }
        return true;
    }
};
```

思路2：反转一半数字（最优）
- 时间复杂度：O(log|x|)
- 空间复杂度：O(1)
- 思路：只反转数字的后一半，与前一半比较

```cpp
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0 || (x % 10 == 0 && x != 0)) return false;
        
        int revertedNumber = 0;
        while (x > revertedNumber) {
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }
        
        return x == revertedNumber || x == revertedNumber / 10;
    }
};
```

---

### 二分查找

#### 卡片 1

**问题**：LeetCode 69. x的平方根

给你一个非负整数x，计算并返回x的算术平方根。

由于返回类型是整数，结果只保留整数部分，小数部分将被舍去。

**答案**：

思路1：二分查找
- 时间复杂度：O(logx)
- 空间复杂度：O(1)
- 思路：在[0,x]范围内二分查找平方根

```cpp
class Solution {
public:
    int mySqrt(int x) {
        if (x < 2) return x;
        
        int left = 2, right = x / 2;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            long long square = (long long)mid * mid;
            if (square == x) return mid;
            else if (square < x) left = mid + 1;
            else right = mid - 1;
        }
        
        return right;
    }
};
```

思路2：牛顿法
- 时间复杂度：O(logx)
- 空间复杂度：O(1)
- 思路：使用牛顿迭代法求平方根

```cpp
class Solution {
public:
    int mySqrt(int x) {
        if (x < 2) return x;
        
        double x0 = x;
        double x1 = (x0 + x / x0) / 2.0;
        
        while (abs(x0 - x1) >= 1) {
            x0 = x1;
            x1 = (x0 + x / x0) / 2.0;
        }
        
        return (int)x1;
    }
};
```

---

### 快速幂

#### 卡片 1

**问题**：LeetCode 50. Pow(x, n)

实现pow(x,n)，即计算x的n次幂函数（即xⁿ）。

**答案**：

思路：快速幂
- 时间复杂度：O(logn)
- 空间复杂度：O(1)
- 思路：将指数n转换为二进制，利用x^(2^k)的性质快速计算

```cpp
class Solution {
public:
    double myPow(double x, int n) {
        long long N = n;
        if (N < 0) {
            x = 1 / x;
            N = -N;
        }
        
        double result = 1.0;
        double current = x;
        
        while (N > 0) {
            if (N % 2 == 1) {
                result *= current;
            }
            current *= current;
            N /= 2;
        }
        
        return result;
    }
};
```

---


