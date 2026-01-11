#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeetCode Hot 100 记忆卡片生成器
生成Anki格式的CSV文件和Markdown格式的卡片
每道题包含多个解题思路和C++代码实现
"""

import csv
import re
from typing import List, Dict

class LeetCodeHot100Generator:
    def __init__(self):
        self.cards = []
    
    def add_card(self, question: str, answer: str, category: str = "LeetCode"):
        """添加一张卡片"""
        self.cards.append({
            "question": question,
            "answer": answer,
            "category": category
        })
    
    def format_text_for_anki(self, text: str) -> str:
        """将文本格式化为Anki HTML格式，设置左对齐样式，保留代码块缩进"""
        # 使用正则表达式识别代码块：```cpp\n...代码...\n```
        # 使用 DOTALL 标志使 . 匹配换行符
        pattern = r'```cpp\n(.*?)\n```'
        
        # 存储代码块和占位符
        code_blocks = []
        placeholder_template = '___CODE_BLOCK_{}___'
        
        def extract_code_block(match):
            """提取代码块，返回占位符"""
            code_content = match.group(1)
            # 代码块内容保留原始换行符（<pre>标签会保留它们）
            code_blocks.append(code_content)
            return placeholder_template.format(len(code_blocks) - 1)
        
        # 先用占位符替换所有代码块
        html_text = re.sub(pattern, extract_code_block, text, flags=re.DOTALL)
        
        # 将非代码块部分的换行符转换为<br>标签
        html_text = html_text.replace('\n', '<br>')
        
        # 将占位符替换为格式化的代码块
        for i, code_content in enumerate(code_blocks):
            placeholder = placeholder_template.format(i)
            # 使用 <pre><code> 包裹代码，设置 white-space: pre 保留空格和缩进
            # 使用 monospace 字体，左对齐
            # 代码内容中的换行符会被 <pre> 标签保留
            formatted_code = f'<pre><code style="white-space: pre; font-family: monospace; text-align: left; display: block; background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">{code_content}</code></pre>'
            html_text = html_text.replace(placeholder, formatted_code)
        
        # 用div包裹，设置左对齐样式
        return f'<div style="text-align: left;">{html_text}</div>'
    
    def generate_leetcode_cards(self):
        """生成LeetCode Hot 100题目卡片"""
        
        # 1. 两数之和
        self.add_card(
            "LeetCode 1. 两数之和\n\n给定一个整数数组nums和一个整数目标值target，请你在该数组中找出和为目标值target的那两个整数，并返回它们的数组下标。\n\n你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。",
            "思路1：暴力法\n- 时间复杂度：O(n²)\n- 空间复杂度：O(1)\n- 思路：双重循环遍历所有可能的组合\n\n```cpp\nclass Solution {\npublic:\n    vector<int> twoSum(vector<int>& nums, int target) {\n        int n = nums.size();\n        for (int i = 0; i < n; i++) {\n            for (int j = i + 1; j < n; j++) {\n                if (nums[i] + nums[j] == target) {\n                    return {i, j};\n                }\n            }\n        }\n        return {};\n    }\n};\n```\n\n思路2：哈希表（一次遍历）\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用哈希表存储已访问的元素值及其索引，在遍历时查找target-nums[i]是否在哈希表中\n\n```cpp\nclass Solution {\npublic:\n    vector<int> twoSum(vector<int>& nums, int target) {\n        unordered_map<int, int> map;\n        for (int i = 0; i < nums.size(); i++) {\n            int complement = target - nums[i];\n            if (map.find(complement) != map.end()) {\n                return {map[complement], i};\n            }\n            map[nums[i]] = i;\n        }\n        return {};\n    }\n};\n```",
            "数组-哈希表"
        )
        
        # 2. 两数相加
        self.add_card(
            "LeetCode 2. 两数相加\n\n给你两个非空的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。\n\n请你将两个数相加，并以相同形式返回一个表示和的链表。",
            "思路：模拟加法\n- 时间复杂度：O(max(m,n))\n- 空间复杂度：O(1)（不考虑结果链表）\n- 思路：同时遍历两个链表，逐位相加，处理进位\n\n```cpp\nclass Solution {\npublic:\n    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {\n        ListNode* dummy = new ListNode(0);\n        ListNode* curr = dummy;\n        int carry = 0;\n        \n        while (l1 || l2 || carry) {\n            int sum = carry;\n            if (l1) {\n                sum += l1->val;\n                l1 = l1->next;\n            }\n            if (l2) {\n                sum += l2->val;\n                l2 = l2->next;\n            }\n            carry = sum / 10;\n            curr->next = new ListNode(sum % 10);\n            curr = curr->next;\n        }\n        \n        return dummy->next;\n    }\n};\n```",
            "链表-数学"
        )
        
        # 3. 无重复字符的最长子串
        self.add_card(
            "LeetCode 3. 无重复字符的最长子串\n\n给定一个字符串s，请你找出其中不含有重复字符的最长子串的长度。",
            "思路1：滑动窗口（哈希表）\n- 时间复杂度：O(n)\n- 空间复杂度：O(min(m,n))，m为字符集大小\n- 思路：使用哈希表记录字符最后出现的位置，维护一个滑动窗口\n\n```cpp\nclass Solution {\npublic:\n    int lengthOfLongestSubstring(string s) {\n        unordered_map<char, int> map;\n        int maxLen = 0;\n        int start = 0;\n        \n        for (int end = 0; end < s.length(); end++) {\n            if (map.find(s[end]) != map.end() && map[s[end]] >= start) {\n                start = map[s[end]] + 1;\n            }\n            map[s[end]] = end;\n            maxLen = max(maxLen, end - start + 1);\n        }\n        \n        return maxLen;\n    }\n};\n```\n\n思路2：滑动窗口（数组优化）\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)（固定128或256大小的数组）\n- 思路：使用数组代替哈希表，适用于ASCII字符\n\n```cpp\nclass Solution {\npublic:\n    int lengthOfLongestSubstring(string s) {\n        vector<int> charIndex(128, -1);\n        int maxLen = 0;\n        int start = 0;\n        \n        for (int end = 0; end < s.length(); end++) {\n            if (charIndex[s[end]] >= start) {\n                start = charIndex[s[end]] + 1;\n            }\n            charIndex[s[end]] = end;\n            maxLen = max(maxLen, end - start + 1);\n        }\n        \n        return maxLen;\n    }\n};\n```",
            "字符串-滑动窗口"
        )
        
        # 4. 寻找两个正序数组的中位数
        self.add_card(
            "LeetCode 4. 寻找两个正序数组的中位数\n\n给定两个大小分别为m和n的正序（从小到大）数组nums1和nums2。请你找出并返回这两个正序数组的中位数。\n\n算法的时间复杂度应该为O(log(m+n))。",
            "思路1：归并排序\n- 时间复杂度：O(m+n)\n- 空间复杂度：O(1)\n- 思路：合并两个有序数组，找到中位数\n\n```cpp\nclass Solution {\npublic:\n    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {\n        int m = nums1.size(), n = nums2.size();\n        int total = m + n;\n        int i = 0, j = 0;\n        int prev = 0, curr = 0;\n        \n        for (int k = 0; k <= total / 2; k++) {\n            prev = curr;\n            if (i < m && (j >= n || nums1[i] < nums2[j])) {\n                curr = nums1[i++];\n            } else {\n                curr = nums2[j++];\n            }\n        }\n        \n        return total % 2 == 0 ? (prev + curr) / 2.0 : curr;\n    }\n};\n```\n\n思路2：二分查找（最优）\n- 时间复杂度：O(log(min(m,n)))\n- 空间复杂度：O(1)\n- 思路：在较短的数组上二分查找分割点，使得左右两部分元素个数相等或差1\n\n```cpp\nclass Solution {\npublic:\n    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {\n        if (nums1.size() > nums2.size()) {\n            return findMedianSortedArrays(nums2, nums1);\n        }\n        \n        int m = nums1.size(), n = nums2.size();\n        int left = 0, right = m;\n        \n        while (left <= right) {\n            int partition1 = (left + right) / 2;\n            int partition2 = (m + n + 1) / 2 - partition1;\n            \n            int maxLeft1 = (partition1 == 0) ? INT_MIN : nums1[partition1 - 1];\n            int minRight1 = (partition1 == m) ? INT_MAX : nums1[partition1];\n            int maxLeft2 = (partition2 == 0) ? INT_MIN : nums2[partition2 - 1];\n            int minRight2 = (partition2 == n) ? INT_MAX : nums2[partition2];\n            \n            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {\n                if ((m + n) % 2 == 0) {\n                    return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2.0;\n                } else {\n                    return max(maxLeft1, maxLeft2);\n                }\n            } else if (maxLeft1 > minRight2) {\n                right = partition1 - 1;\n            } else {\n                left = partition1 + 1;\n            }\n        }\n        return 0.0;\n    }\n};\n```",
            "数组-二分查找"
        )
        
        # 5. 最长回文子串
        self.add_card(
            "LeetCode 5. 最长回文子串\n\n给你一个字符串s，找到s中最长的回文子串。",
            "思路1：中心扩展法\n- 时间复杂度：O(n²)\n- 空间复杂度：O(1)\n- 思路：以每个字符或每两个字符为中心，向两边扩展寻找回文串\n\n```cpp\nclass Solution {\npublic:\n    string longestPalindrome(string s) {\n        int start = 0, maxLen = 1;\n        \n        for (int i = 0; i < s.length(); i++) {\n            // 奇数长度回文串\n            int len1 = expandAroundCenter(s, i, i);\n            // 偶数长度回文串\n            int len2 = expandAroundCenter(s, i, i + 1);\n            \n            int len = max(len1, len2);\n            if (len > maxLen) {\n                maxLen = len;\n                start = i - (len - 1) / 2;\n            }\n        }\n        \n        return s.substr(start, maxLen);\n    }\n    \nprivate:\n    int expandAroundCenter(string& s, int left, int right) {\n        while (left >= 0 && right < s.length() && s[left] == s[right]) {\n            left--;\n            right++;\n        }\n        return right - left - 1;\n    }\n};\n```\n\n思路2：动态规划\n- 时间复杂度：O(n²)\n- 空间复杂度：O(n²)\n- 思路：dp[i][j]表示s[i...j]是否为回文串\n\n```cpp\nclass Solution {\npublic:\n    string longestPalindrome(string s) {\n        int n = s.length();\n        vector<vector<bool>> dp(n, vector<bool>(n, false));\n        int start = 0, maxLen = 1;\n        \n        // 单个字符都是回文\n        for (int i = 0; i < n; i++) {\n            dp[i][i] = true;\n        }\n        \n        // 两个字符\n        for (int i = 0; i < n - 1; i++) {\n            if (s[i] == s[i + 1]) {\n                dp[i][i + 1] = true;\n                start = i;\n                maxLen = 2;\n            }\n        }\n        \n        // 长度大于2的子串\n        for (int len = 3; len <= n; len++) {\n            for (int i = 0; i <= n - len; i++) {\n                int j = i + len - 1;\n                if (s[i] == s[j] && dp[i + 1][j - 1]) {\n                    dp[i][j] = true;\n                    start = i;\n                    maxLen = len;\n                }\n            }\n        }\n        \n        return s.substr(start, maxLen);\n    }\n};\n```\n\n思路3：Manacher算法（最优）\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：利用回文串的对称性，避免重复计算\n\n```cpp\nclass Solution {\npublic:\n    string longestPalindrome(string s) {\n        string t = \"#\";\n        for (char c : s) {\n            t += c;\n            t += \"#\";\n        }\n        \n        int n = t.length();\n        vector<int> p(n, 0);\n        int center = 0, right = 0;\n        int maxLen = 0, centerIndex = 0;\n        \n        for (int i = 0; i < n; i++) {\n            if (i < right) {\n                p[i] = min(right - i, p[2 * center - i]);\n            }\n            \n            int left = i - (1 + p[i]);\n            int r = i + (1 + p[i]);\n            while (left >= 0 && r < n && t[left] == t[r]) {\n                p[i]++;\n                left--;\n                r++;\n            }\n            \n            if (i + p[i] > right) {\n                center = i;\n                right = i + p[i];\n            }\n            \n            if (p[i] > maxLen) {\n                maxLen = p[i];\n                centerIndex = i;\n            }\n        }\n        \n        int start = (centerIndex - maxLen) / 2;\n        return s.substr(start, maxLen);\n    }\n};\n```",
            "字符串-动态规划"
        )
        
        # 11. 盛最多水的容器
        self.add_card(
            "LeetCode 11. 盛最多水的容器\n\n给定一个长度为n的整数数组height。有n条垂线，第i条线的两个端点是(i,0)和(i,height[i])。\n\n找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。",
            "思路：双指针\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：从两端开始，每次移动较短的那一边，因为移动较长边不会增加面积\n\n```cpp\nclass Solution {\npublic:\n    int maxArea(vector<int>& height) {\n        int left = 0, right = height.size() - 1;\n        int maxArea = 0;\n        \n        while (left < right) {\n            int area = min(height[left], height[right]) * (right - left);\n            maxArea = max(maxArea, area);\n            \n            if (height[left] < height[right]) {\n                left++;\n            } else {\n                right--;\n            }\n        }\n        \n        return maxArea;\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 15. 三数之和
        self.add_card(
            "LeetCode 15. 三数之和\n\n给你一个整数数组nums，判断是否存在三元组[nums[i], nums[j], nums[k]]满足i!=j、i!=k且j!=k，同时还满足nums[i]+nums[j]+nums[k]==0。请你返回所有和为0且不重复的三元组。",
            "思路：排序+双指针\n- 时间复杂度：O(n²)\n- 空间复杂度：O(1)（不考虑结果数组）\n- 思路：先排序，固定第一个数，用双指针找另外两个数\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> threeSum(vector<int>& nums) {\n        vector<vector<int>> result;\n        int n = nums.size();\n        if (n < 3) return result;\n        \n        sort(nums.begin(), nums.end());\n        \n        for (int i = 0; i < n - 2; i++) {\n            // 跳过重复元素\n            if (i > 0 && nums[i] == nums[i - 1]) continue;\n            \n            int left = i + 1, right = n - 1;\n            while (left < right) {\n                int sum = nums[i] + nums[left] + nums[right];\n                if (sum == 0) {\n                    result.push_back({nums[i], nums[left], nums[right]});\n                    // 跳过重复元素\n                    while (left < right && nums[left] == nums[left + 1]) left++;\n                    while (left < right && nums[right] == nums[right - 1]) right--;\n                    left++;\n                    right--;\n                } else if (sum < 0) {\n                    left++;\n                } else {\n                    right--;\n                }\n            }\n        }\n        \n        return result;\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 19. 删除链表的倒数第N个结点
        self.add_card(
            "LeetCode 19. 删除链表的倒数第N个结点\n\n给你一个链表，删除链表的倒数第n个结点，并且返回链表的头结点。",
            "思路1：两次遍历\n- 时间复杂度：O(L)，L为链表长度\n- 空间复杂度：O(1)\n- 思路：第一次遍历得到链表长度，第二次遍历删除倒数第n个节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* removeNthFromEnd(ListNode* head, int n) {\n        int length = 0;\n        ListNode* curr = head;\n        while (curr) {\n            length++;\n            curr = curr->next;\n        }\n        \n        if (n == length) {\n            return head->next;\n        }\n        \n        curr = head;\n        for (int i = 0; i < length - n - 1; i++) {\n            curr = curr->next;\n        }\n        curr->next = curr->next->next;\n        \n        return head;\n    }\n};\n```\n\n思路2：一次遍历（双指针）\n- 时间复杂度：O(L)\n- 空间复杂度：O(1)\n- 思路：使用快慢指针，快指针先走n步，然后两个指针同时移动\n\n```cpp\nclass Solution {\npublic:\n    ListNode* removeNthFromEnd(ListNode* head, int n) {\n        ListNode* dummy = new ListNode(0);\n        dummy->next = head;\n        ListNode* fast = dummy;\n        ListNode* slow = dummy;\n        \n        // 快指针先走n+1步\n        for (int i = 0; i <= n; i++) {\n            fast = fast->next;\n        }\n        \n        // 快慢指针同时移动\n        while (fast) {\n            fast = fast->next;\n            slow = slow->next;\n        }\n        \n        slow->next = slow->next->next;\n        return dummy->next;\n    }\n};\n```",
            "链表-双指针"
        )
        
        # 20. 有效的括号
        self.add_card(
            "LeetCode 20. 有效的括号\n\n给定一个只包括'('，')'，'{'，'}'，'['，']'的字符串s，判断字符串是否有效。",
            "思路：栈\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：遇到左括号入栈，遇到右括号检查是否与栈顶匹配\n\n```cpp\nclass Solution {\npublic:\n    bool isValid(string s) {\n        stack<char> st;\n        for (char c : s) {\n            if (c == '(' || c == '[' || c == '{') {\n                st.push(c);\n            } else {\n                if (st.empty()) return false;\n                char top = st.top();\n                st.pop();\n                if ((c == ')' && top != '(') ||\n                    (c == ']' && top != '[') ||\n                    (c == '}' && top != '{')) {\n                    return false;\n                }\n            }\n        }\n        return st.empty();\n    }\n};\n```",
            "栈"
        )
        
        # 21. 合并两个有序链表
        self.add_card(
            "LeetCode 21. 合并两个有序链表\n\n将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。",
            "思路1：迭代\n- 时间复杂度：O(m+n)\n- 空间复杂度：O(1)\n- 思路：使用虚拟头节点，比较两个链表的节点值，逐个合并\n\n```cpp\nclass Solution {\npublic:\n    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {\n        ListNode* dummy = new ListNode(0);\n        ListNode* curr = dummy;\n        \n        while (list1 && list2) {\n            if (list1->val < list2->val) {\n                curr->next = list1;\n                list1 = list1->next;\n            } else {\n                curr->next = list2;\n                list2 = list2->next;\n            }\n            curr = curr->next;\n        }\n        \n        curr->next = list1 ? list1 : list2;\n        return dummy->next;\n    }\n};\n```\n\n思路2：递归\n- 时间复杂度：O(m+n)\n- 空间复杂度：O(m+n)（递归栈）\n- 思路：递归地合并链表\n\n```cpp\nclass Solution {\npublic:\n    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {\n        if (!list1) return list2;\n        if (!list2) return list1;\n        \n        if (list1->val < list2->val) {\n            list1->next = mergeTwoLists(list1->next, list2);\n            return list1;\n        } else {\n            list2->next = mergeTwoLists(list1, list2->next);\n            return list2;\n        }\n    }\n};\n```",
            "链表-递归"
        )
        
        # 23. 合并K个升序链表
        self.add_card(
            "LeetCode 23. 合并K个升序链表\n\n给你一个链表数组，每个链表都已经按升序排列。\n\n请你将所有链表合并到一个升序链表中，返回合并后的链表。",
            "思路1：顺序合并\n- 时间复杂度：O(k²n)，k为链表数量，n为平均长度\n- 空间复杂度：O(1)\n- 思路：依次合并每个链表\n\n```cpp\nclass Solution {\npublic:\n    ListNode* mergeKLists(vector<ListNode*>& lists) {\n        if (lists.empty()) return nullptr;\n        \n        ListNode* result = lists[0];\n        for (int i = 1; i < lists.size(); i++) {\n            result = mergeTwoLists(result, lists[i]);\n        }\n        return result;\n    }\n    \nprivate:\n    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {\n        ListNode* dummy = new ListNode(0);\n        ListNode* curr = dummy;\n        while (l1 && l2) {\n            if (l1->val < l2->val) {\n                curr->next = l1;\n                l1 = l1->next;\n            } else {\n                curr->next = l2;\n                l2 = l2->next;\n            }\n            curr = curr->next;\n        }\n        curr->next = l1 ? l1 : l2;\n        return dummy->next;\n    }\n};\n```\n\n思路2：分治合并\n- 时间复杂度：O(kn×logk)\n- 空间复杂度：O(logk)（递归栈）\n- 思路：将k个链表两两合并，递归进行\n\n```cpp\nclass Solution {\npublic:\n    ListNode* mergeKLists(vector<ListNode*>& lists) {\n        return merge(lists, 0, lists.size() - 1);\n    }\n    \nprivate:\n    ListNode* merge(vector<ListNode*>& lists, int left, int right) {\n        if (left > right) return nullptr;\n        if (left == right) return lists[left];\n        \n        int mid = (left + right) / 2;\n        return mergeTwoLists(merge(lists, left, mid), merge(lists, mid + 1, right));\n    }\n    \n    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {\n        ListNode* dummy = new ListNode(0);\n        ListNode* curr = dummy;\n        while (l1 && l2) {\n            if (l1->val < l2->val) {\n                curr->next = l1;\n                l1 = l1->next;\n            } else {\n                curr->next = l2;\n                l2 = l2->next;\n            }\n            curr = curr->next;\n        }\n        curr->next = l1 ? l1 : l2;\n        return dummy->next;\n    }\n};\n```\n\n思路3：优先队列（堆）\n- 时间复杂度：O(kn×logk)\n- 空间复杂度：O(k)\n- 思路：使用最小堆维护每个链表的当前节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* mergeKLists(vector<ListNode*>& lists) {\n        auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };\n        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);\n        \n        for (ListNode* list : lists) {\n            if (list) pq.push(list);\n        }\n        \n        ListNode* dummy = new ListNode(0);\n        ListNode* curr = dummy;\n        \n        while (!pq.empty()) {\n            ListNode* node = pq.top();\n            pq.pop();\n            curr->next = node;\n            curr = curr->next;\n            if (node->next) {\n                pq.push(node->next);\n            }\n        }\n        \n        return dummy->next;\n    }\n};\n```",
            "链表-分治"
        )
        
        # 53. 最大子数组和
        self.add_card(
            "LeetCode 53. 最大子数组和\n\n给你一个整数数组nums，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。",
            "思路1：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：dp[i]表示以nums[i]结尾的最大子数组和\n\n```cpp\nclass Solution {\npublic:\n    int maxSubArray(vector<int>& nums) {\n        int maxSum = nums[0];\n        int currentSum = nums[0];\n        \n        for (int i = 1; i < nums.size(); i++) {\n            currentSum = max(nums[i], currentSum + nums[i]);\n            maxSum = max(maxSum, currentSum);\n        }\n        \n        return maxSum;\n    }\n};\n```\n\n思路2：分治法\n- 时间复杂度：O(nlogn)\n- 空间复杂度：O(logn)\n- 思路：将数组分为左右两部分，最大子数组和要么在左边，要么在右边，要么跨越中间\n\n```cpp\nclass Solution {\npublic:\n    int maxSubArray(vector<int>& nums) {\n        return maxSubArrayHelper(nums, 0, nums.size() - 1);\n    }\n    \nprivate:\n    int maxSubArrayHelper(vector<int>& nums, int left, int right) {\n        if (left == right) return nums[left];\n        \n        int mid = (left + right) / 2;\n        int leftMax = maxSubArrayHelper(nums, left, mid);\n        int rightMax = maxSubArrayHelper(nums, mid + 1, right);\n        \n        // 跨越中间的最大子数组和\n        int leftSum = INT_MIN, rightSum = INT_MIN, sum = 0;\n        for (int i = mid; i >= left; i--) {\n            sum += nums[i];\n            leftSum = max(leftSum, sum);\n        }\n        sum = 0;\n        for (int i = mid + 1; i <= right; i++) {\n            sum += nums[i];\n            rightSum = max(rightSum, sum);\n        }\n        \n        return max({leftMax, rightMax, leftSum + rightSum});\n    }\n};\n```",
            "动态规划-数组"
        )
        
        # 70. 爬楼梯
        self.add_card(
            "LeetCode 70. 爬楼梯\n\n假设你正在爬楼梯。需要n阶你才能到达楼顶。\n\n每次你可以爬1或2个台阶。你有多少种不同的方法可以爬到楼顶？",
            "思路1：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：dp[i]表示到达第i阶的方法数，dp[i]=dp[i-1]+dp[i-2]\n\n```cpp\nclass Solution {\npublic:\n    int climbStairs(int n) {\n        if (n <= 2) return n;\n        vector<int> dp(n + 1);\n        dp[1] = 1;\n        dp[2] = 2;\n        for (int i = 3; i <= n; i++) {\n            dp[i] = dp[i - 1] + dp[i - 2];\n        }\n        return dp[n];\n    }\n};\n```\n\n思路2：空间优化\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：只保存前两个状态\n\n```cpp\nclass Solution {\npublic:\n    int climbStairs(int n) {\n        if (n <= 2) return n;\n        int prev = 1, curr = 2;\n        for (int i = 3; i <= n; i++) {\n            int next = prev + curr;\n            prev = curr;\n            curr = next;\n        }\n        return curr;\n    }\n};\n```\n\n思路3：矩阵快速幂\n- 时间复杂度：O(logn)\n- 空间复杂度：O(1)\n- 思路：将递推关系转换为矩阵幂运算\n\n```cpp\nclass Solution {\npublic:\n    int climbStairs(int n) {\n        if (n <= 2) return n;\n        vector<vector<long>> base = {{1, 1}, {1, 0}};\n        vector<vector<long>> result = matrixPower(base, n - 1);\n        return result[0][0] + result[0][1];\n    }\n    \nprivate:\n    vector<vector<long>> matrixPower(vector<vector<long>>& base, int n) {\n        vector<vector<long>> result = {{1, 0}, {0, 1}};\n        while (n > 0) {\n            if (n % 2 == 1) {\n                result = matrixMultiply(result, base);\n            }\n            base = matrixMultiply(base, base);\n            n /= 2;\n        }\n        return result;\n    }\n    \n    vector<vector<long>> matrixMultiply(vector<vector<long>>& a, vector<vector<long>>& b) {\n        vector<vector<long>> c(2, vector<long>(2, 0));\n        for (int i = 0; i < 2; i++) {\n            for (int j = 0; j < 2; j++) {\n                for (int k = 0; k < 2; k++) {\n                    c[i][j] += a[i][k] * b[k][j];\n                }\n            }\n        }\n        return c;\n    }\n};\n```",
            "动态规划"
        )
        
        # 94. 二叉树的中序遍历
        self.add_card(
            "LeetCode 94. 二叉树的中序遍历\n\n给定一个二叉树的根节点root，返回它的中序遍历。",
            "思路1：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(h)，h为树的高度\n- 思路：左-根-右的顺序遍历\n\n```cpp\nclass Solution {\npublic:\n    vector<int> inorderTraversal(TreeNode* root) {\n        vector<int> result;\n        inorder(root, result);\n        return result;\n    }\n    \nprivate:\n    void inorder(TreeNode* root, vector<int>& result) {\n        if (!root) return;\n        inorder(root->left, result);\n        result.push_back(root->val);\n        inorder(root->right, result);\n    }\n};\n```\n\n思路2：迭代（栈）\n- 时间复杂度：O(n)\n- 空间复杂度：O(h)\n- 思路：使用栈模拟递归过程\n\n```cpp\nclass Solution {\npublic:\n    vector<int> inorderTraversal(TreeNode* root) {\n        vector<int> result;\n        stack<TreeNode*> st;\n        TreeNode* curr = root;\n        \n        while (curr || !st.empty()) {\n            while (curr) {\n                st.push(curr);\n                curr = curr->left;\n            }\n            curr = st.top();\n            st.pop();\n            result.push_back(curr->val);\n            curr = curr->right;\n        }\n        \n        return result;\n    }\n};\n```\n\n思路3：Morris遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：利用树中的空指针，不需要栈\n\n```cpp\nclass Solution {\npublic:\n    vector<int> inorderTraversal(TreeNode* root) {\n        vector<int> result;\n        TreeNode* curr = root;\n        \n        while (curr) {\n            if (!curr->left) {\n                result.push_back(curr->val);\n                curr = curr->right;\n            } else {\n                TreeNode* prev = curr->left;\n                while (prev->right && prev->right != curr) {\n                    prev = prev->right;\n                }\n                if (!prev->right) {\n                    prev->right = curr;\n                    curr = curr->left;\n                } else {\n                    prev->right = nullptr;\n                    result.push_back(curr->val);\n                    curr = curr->right;\n                }\n            }\n        }\n        \n        return result;\n    }\n};\n```",
            "树-二叉树"
        )
        
        # 101. 对称二叉树
        self.add_card(
            "LeetCode 101. 对称二叉树\n\n给你一个二叉树的根节点root，检查它是否轴对称。",
            "思路1：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(h)\n- 思路：比较左右子树是否镜像对称\n\n```cpp\nclass Solution {\npublic:\n    bool isSymmetric(TreeNode* root) {\n        return isMirror(root, root);\n    }\n    \nprivate:\n    bool isMirror(TreeNode* t1, TreeNode* t2) {\n        if (!t1 && !t2) return true;\n        if (!t1 || !t2) return false;\n        return (t1->val == t2->val) &&\n               isMirror(t1->left, t2->right) &&\n               isMirror(t1->right, t2->left);\n    }\n};\n```\n\n思路2：迭代（队列）\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用队列进行层序遍历，每次比较两个节点\n\n```cpp\nclass Solution {\npublic:\n    bool isSymmetric(TreeNode* root) {\n        queue<TreeNode*> q;\n        q.push(root);\n        q.push(root);\n        \n        while (!q.empty()) {\n            TreeNode* t1 = q.front(); q.pop();\n            TreeNode* t2 = q.front(); q.pop();\n            \n            if (!t1 && !t2) continue;\n            if (!t1 || !t2) return false;\n            if (t1->val != t2->val) return false;\n            \n            q.push(t1->left);\n            q.push(t2->right);\n            q.push(t1->right);\n            q.push(t2->left);\n        }\n        \n        return true;\n    }\n};\n```",
            "树-二叉树"
        )
        
        # 102. 二叉树的层序遍历
        self.add_card(
            "LeetCode 102. 二叉树的层序遍历\n\n给你二叉树的根节点root，返回其节点值的层序遍历。（即逐层地，从左到右访问所有节点）。",
            "思路：BFS（队列）\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用队列进行广度优先搜索，按层遍历\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> levelOrder(TreeNode* root) {\n        vector<vector<int>> result;\n        if (!root) return result;\n        \n        queue<TreeNode*> q;\n        q.push(root);\n        \n        while (!q.empty()) {\n            int size = q.size();\n            vector<int> level;\n            \n            for (int i = 0; i < size; i++) {\n                TreeNode* node = q.front();\n                q.pop();\n                level.push_back(node->val);\n                \n                if (node->left) q.push(node->left);\n                if (node->right) q.push(node->right);\n            }\n            \n            result.push_back(level);\n        }\n        \n        return result;\n    }\n};\n```",
            "树-BFS"
        )
        
        # 104. 二叉树的最大深度
        self.add_card(
            "LeetCode 104. 二叉树的最大深度\n\n给定一个二叉树，找出其最大深度。\n\n二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。",
            "思路1：递归（DFS）\n- 时间复杂度：O(n)\n- 空间复杂度：O(h)\n- 思路：递归计算左右子树的最大深度\n\n```cpp\nclass Solution {\npublic:\n    int maxDepth(TreeNode* root) {\n        if (!root) return 0;\n        return 1 + max(maxDepth(root->left), maxDepth(root->right));\n    }\n};\n```\n\n思路2：迭代（BFS）\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用队列进行层序遍历，记录层数\n\n```cpp\nclass Solution {\npublic:\n    int maxDepth(TreeNode* root) {\n        if (!root) return 0;\n        \n        queue<TreeNode*> q;\n        q.push(root);\n        int depth = 0;\n        \n        while (!q.empty()) {\n            int size = q.size();\n            depth++;\n            \n            for (int i = 0; i < size; i++) {\n                TreeNode* node = q.front();\n                q.pop();\n                if (node->left) q.push(node->left);\n                if (node->right) q.push(node->right);\n            }\n        }\n        \n        return depth;\n    }\n};\n```",
            "树-DFS"
        )
        
        # 206. 反转链表
        self.add_card(
            "LeetCode 206. 反转链表\n\n给你单链表的头节点head，请你反转链表，并返回反转后的链表。",
            "思路1：迭代\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用三个指针，逐个反转节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* reverseList(ListNode* head) {\n        ListNode* prev = nullptr;\n        ListNode* curr = head;\n        \n        while (curr) {\n            ListNode* next = curr->next;\n            curr->next = prev;\n            prev = curr;\n            curr = next;\n        }\n        \n        return prev;\n    }\n};\n```\n\n思路2：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：递归反转后面的链表，然后反转当前节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* reverseList(ListNode* head) {\n        if (!head || !head->next) return head;\n        \n        ListNode* newHead = reverseList(head->next);\n        head->next->next = head;\n        head->next = nullptr;\n        \n        return newHead;\n    }\n};\n```",
            "链表"
        )
        
        # 6. Z字形变换
        self.add_card(
            "LeetCode 6. Z字形变换\n\n将一个给定字符串s根据给定的行数numRows，以从上往下、从左到右进行Z字形排列。",
            "思路：按行模拟\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用字符串数组存储每一行的字符，按Z字形顺序填充\n\n```cpp\nclass Solution {\npublic:\n    string convert(string s, int numRows) {\n        if (numRows == 1) return s;\n        \n        vector<string> rows(min(numRows, int(s.length())));\n        int curRow = 0;\n        bool goingDown = false;\n        \n        for (char c : s) {\n            rows[curRow] += c;\n            if (curRow == 0 || curRow == numRows - 1) {\n                goingDown = !goingDown;\n            }\n            curRow += goingDown ? 1 : -1;\n        }\n        \n        string result;\n        for (string row : rows) {\n            result += row;\n        }\n        return result;\n    }\n};\n```",
            "字符串-模拟"
        )
        
        # 7. 整数反转
        self.add_card(
            "LeetCode 7. 整数反转\n\n给你一个32位的有符号整数x，返回将x中的数字部分反转后的结果。\n\n如果反转后整数超过32位的有符号整数的范围[−2³¹, 2³¹−1]，就返回0。",
            "思路：数学方法\n- 时间复杂度：O(log|x|)\n- 空间复杂度：O(1)\n- 思路：每次取最后一位数字，检查是否溢出\n\n```cpp\nclass Solution {\npublic:\n    int reverse(int x) {\n        int result = 0;\n        while (x != 0) {\n            int digit = x % 10;\n            x /= 10;\n            \n            // 检查溢出\n            if (result > INT_MAX / 10 || (result == INT_MAX / 10 && digit > 7)) return 0;\n            if (result < INT_MIN / 10 || (result == INT_MIN / 10 && digit < -8)) return 0;\n            \n            result = result * 10 + digit;\n        }\n        return result;\n    }\n};\n```",
            "数学"
        )
        
        # 9. 回文数
        self.add_card(
            "LeetCode 9. 回文数\n\n给你一个整数x，如果x是一个回文整数，返回true；否则，返回false。",
            "思路1：转换为字符串\n- 时间复杂度：O(log|x|)\n- 空间复杂度：O(log|x|)\n- 思路：将整数转换为字符串，然后判断是否为回文\n\n```cpp\nclass Solution {\npublic:\n    bool isPalindrome(int x) {\n        if (x < 0) return false;\n        string s = to_string(x);\n        int left = 0, right = s.length() - 1;\n        while (left < right) {\n            if (s[left] != s[right]) return false;\n            left++;\n            right--;\n        }\n        return true;\n    }\n};\n```\n\n思路2：反转一半数字（最优）\n- 时间复杂度：O(log|x|)\n- 空间复杂度：O(1)\n- 思路：只反转数字的后一半，与前一半比较\n\n```cpp\nclass Solution {\npublic:\n    bool isPalindrome(int x) {\n        if (x < 0 || (x % 10 == 0 && x != 0)) return false;\n        \n        int revertedNumber = 0;\n        while (x > revertedNumber) {\n            revertedNumber = revertedNumber * 10 + x % 10;\n            x /= 10;\n        }\n        \n        return x == revertedNumber || x == revertedNumber / 10;\n    }\n};\n```",
            "数学"
        )
        
        # 14. 最长公共前缀
        self.add_card(
            "LeetCode 14. 最长公共前缀\n\n编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串\"\"。",
            "思路1：横向扫描\n- 时间复杂度：O(mn)，m为字符串平均长度，n为字符串数量\n- 空间复杂度：O(1)\n- 思路：依次比较每个字符串与第一个字符串的公共前缀\n\n```cpp\nclass Solution {\npublic:\n    string longestCommonPrefix(vector<string>& strs) {\n        if (strs.empty()) return \"\";\n        \n        string prefix = strs[0];\n        for (int i = 1; i < strs.size(); i++) {\n            while (strs[i].find(prefix) != 0) {\n                prefix = prefix.substr(0, prefix.length() - 1);\n                if (prefix.empty()) return \"\";\n            }\n        }\n        return prefix;\n    }\n};\n```\n\n思路2：纵向扫描\n- 时间复杂度：O(mn)\n- 空间复杂度：O(1)\n- 思路：从第一个字符开始，逐列比较所有字符串\n\n```cpp\nclass Solution {\npublic:\n    string longestCommonPrefix(vector<string>& strs) {\n        if (strs.empty()) return \"\";\n        \n        for (int i = 0; i < strs[0].length(); i++) {\n            char c = strs[0][i];\n            for (int j = 1; j < strs.size(); j++) {\n                if (i >= strs[j].length() || strs[j][i] != c) {\n                    return strs[0].substr(0, i);\n                }\n            }\n        }\n        return strs[0];\n    }\n};\n```\n\n思路3：分治法\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mlogn)\n- 思路：将字符串数组分成两部分，分别求公共前缀，然后合并\n\n```cpp\nclass Solution {\npublic:\n    string longestCommonPrefix(vector<string>& strs) {\n        if (strs.empty()) return \"\";\n        return longestCommonPrefix(strs, 0, strs.size() - 1);\n    }\n    \nprivate:\n    string longestCommonPrefix(vector<string>& strs, int left, int right) {\n        if (left == right) return strs[left];\n        \n        int mid = (left + right) / 2;\n        string leftPrefix = longestCommonPrefix(strs, left, mid);\n        string rightPrefix = longestCommonPrefix(strs, mid + 1, right);\n        return commonPrefix(leftPrefix, rightPrefix);\n    }\n    \n    string commonPrefix(string left, string right) {\n        int minLen = min(left.length(), right.length());\n        for (int i = 0; i < minLen; i++) {\n            if (left[i] != right[i]) {\n                return left.substr(0, i);\n            }\n        }\n        return left.substr(0, minLen);\n    }\n};\n```",
            "字符串"
        )
        
        # 26. 删除有序数组中的重复项
        self.add_card(
            "LeetCode 26. 删除有序数组中的重复项\n\n给你一个非严格递增排列的数组nums，请你原地删除重复出现的元素，使每个元素只出现一次，返回删除后数组的新长度。",
            "思路：双指针\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用快慢指针，快指针遍历数组，慢指针指向不重复元素的位置\n\n```cpp\nclass Solution {\npublic:\n    int removeDuplicates(vector<int>& nums) {\n        if (nums.empty()) return 0;\n        \n        int slow = 0;\n        for (int fast = 1; fast < nums.size(); fast++) {\n            if (nums[fast] != nums[slow]) {\n                slow++;\n                nums[slow] = nums[fast];\n            }\n        }\n        return slow + 1;\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 27. 移除元素
        self.add_card(
            "LeetCode 27. 移除元素\n\n给你一个数组nums和一个值val，你需要原地移除所有数值等于val的元素，并返回移除后数组的新长度。",
            "思路：双指针\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用快慢指针，快指针遍历数组，慢指针指向不等于val的元素位置\n\n```cpp\nclass Solution {\npublic:\n    int removeElement(vector<int>& nums, int val) {\n        int slow = 0;\n        for (int fast = 0; fast < nums.size(); fast++) {\n            if (nums[fast] != val) {\n                nums[slow] = nums[fast];\n                slow++;\n            }\n        }\n        return slow;\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 35. 搜索插入位置
        self.add_card(
            "LeetCode 35. 搜索插入位置\n\n给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。",
            "思路1：二分查找\n- 时间复杂度：O(logn)\n- 空间复杂度：O(1)\n- 思路：使用二分查找找到目标值或插入位置\n\n```cpp\nclass Solution {\npublic:\n    int searchInsert(vector<int>& nums, int target) {\n        int left = 0, right = nums.size() - 1;\n        \n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            if (nums[mid] == target) {\n                return mid;\n            } else if (nums[mid] < target) {\n                left = mid + 1;\n            } else {\n                right = mid - 1;\n            }\n        }\n        \n        return left;\n    }\n};\n```",
            "数组-二分查找"
        )
        
        # 66. 加一
        self.add_card(
            "LeetCode 66. 加一\n\n给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。\n\n最高位数字存放在数组的首位，数组中每个元素只存储单个数字。",
            "思路：模拟加法\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：从后往前遍历，处理进位\n\n```cpp\nclass Solution {\npublic:\n    vector<int> plusOne(vector<int>& digits) {\n        for (int i = digits.size() - 1; i >= 0; i--) {\n            digits[i]++;\n            if (digits[i] < 10) {\n                return digits;\n            }\n            digits[i] = 0;\n        }\n        \n        // 如果所有位都是9，需要在前面加1\n        digits.insert(digits.begin(), 1);\n        return digits;\n    }\n};\n```",
            "数组-数学"
        )
        
        # 67. 二进制求和
        self.add_card(
            "LeetCode 67. 二进制求和\n\n给你两个二进制字符串a和b，返回它们的和，用二进制字符串表示。",
            "思路：模拟加法\n- 时间复杂度：O(max(m,n))\n- 空间复杂度：O(1)（不考虑结果字符串）\n- 思路：从后往前逐位相加，处理进位\n\n```cpp\nclass Solution {\npublic:\n    string addBinary(string a, string b) {\n        string result;\n        int i = a.length() - 1, j = b.length() - 1;\n        int carry = 0;\n        \n        while (i >= 0 || j >= 0 || carry) {\n            int sum = carry;\n            if (i >= 0) sum += a[i--] - '0';\n            if (j >= 0) sum += b[j--] - '0';\n            \n            result = char(sum % 2 + '0') + result;\n            carry = sum / 2;\n        }\n        \n        return result;\n    }\n};\n```",
            "字符串-数学"
        )
        
        # 75. 颜色分类
        self.add_card(
            "LeetCode 75. 颜色分类\n\n给定一个包含红色、白色和蓝色、共n个元素的数组nums，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。\n\n我们使用整数0、1和2分别表示红色、白色和蓝色。",
            "思路1：计数排序\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：统计0、1、2的个数，然后重新填充数组\n\n```cpp\nclass Solution {\npublic:\n    void sortColors(vector<int>& nums) {\n        int count0 = 0, count1 = 0, count2 = 0;\n        for (int num : nums) {\n            if (num == 0) count0++;\n            else if (num == 1) count1++;\n            else count2++;\n        }\n        \n        int i = 0;\n        while (count0--) nums[i++] = 0;\n        while (count1--) nums[i++] = 1;\n        while (count2--) nums[i++] = 2;\n    }\n};\n```\n\n思路2：双指针（荷兰国旗问题）\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用三个指针，将0移到左边，2移到右边\n\n```cpp\nclass Solution {\npublic:\n    void sortColors(vector<int>& nums) {\n        int left = 0, right = nums.size() - 1;\n        int curr = 0;\n        \n        while (curr <= right) {\n            if (nums[curr] == 0) {\n                swap(nums[left++], nums[curr++]);\n            } else if (nums[curr] == 2) {\n                swap(nums[curr], nums[right--]);\n            } else {\n                curr++;\n            }\n        }\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 88. 合并两个有序数组
        self.add_card(
            "LeetCode 88. 合并两个有序数组\n\n给你两个按非递减顺序排列的整数数组nums1和nums2，另有两个整数m和n，分别表示nums1和nums2中元素的数目。\n\n请你合并nums2到nums1中，使合并后的数组同样按非递减顺序排列。",
            "思路：从后往前合并\n- 时间复杂度：O(m+n)\n- 空间复杂度：O(1)\n- 思路：从后往前填充，避免覆盖nums1中的元素\n\n```cpp\nclass Solution {\npublic:\n    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {\n        int i = m - 1, j = n - 1, k = m + n - 1;\n        \n        while (i >= 0 && j >= 0) {\n            if (nums1[i] > nums2[j]) {\n                nums1[k--] = nums1[i--];\n            } else {\n                nums1[k--] = nums2[j--];\n            }\n        }\n        \n        while (j >= 0) {\n            nums1[k--] = nums2[j--];\n        }\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 24. 两两交换链表中的节点
        self.add_card(
            "LeetCode 24. 两两交换链表中的节点\n\n给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。",
            "思路1：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：递归交换每两个节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* swapPairs(ListNode* head) {\n        if (!head || !head->next) return head;\n        \n        ListNode* first = head;\n        ListNode* second = head->next;\n        \n        first->next = swapPairs(second->next);\n        second->next = first;\n        \n        return second;\n    }\n};\n```\n\n思路2：迭代\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用虚拟头节点，逐个交换相邻节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* swapPairs(ListNode* head) {\n        ListNode* dummy = new ListNode(0);\n        dummy->next = head;\n        ListNode* prev = dummy;\n        \n        while (prev->next && prev->next->next) {\n            ListNode* first = prev->next;\n            ListNode* second = prev->next->next;\n            \n            prev->next = second;\n            first->next = second->next;\n            second->next = first;\n            \n            prev = first;\n        }\n        \n        return dummy->next;\n    }\n};\n```",
            "链表"
        )
        
        # 25. K个一组翻转链表
        self.add_card(
            "LeetCode 25. K个一组翻转链表\n\n给你链表的头节点head，每k个节点一组进行翻转，请你返回修改后的链表。",
            "思路：递归+迭代\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：先检查是否有k个节点，然后翻转这k个节点，递归处理后续节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* reverseKGroup(ListNode* head, int k) {\n        ListNode* curr = head;\n        int count = 0;\n        \n        // 检查是否有k个节点\n        while (curr && count < k) {\n            curr = curr->next;\n            count++;\n        }\n        \n        if (count == k) {\n            // 递归处理后续节点\n            curr = reverseKGroup(curr, k);\n            \n            // 翻转当前k个节点\n            while (count > 0) {\n                ListNode* next = head->next;\n                head->next = curr;\n                curr = head;\n                head = next;\n                count--;\n            }\n            head = curr;\n        }\n        \n        return head;\n    }\n};\n```",
            "链表"
        )
        
        # 61. 旋转链表
        self.add_card(
            "LeetCode 61. 旋转链表\n\n给你一个链表的头节点head，旋转链表，将链表每个节点向右移动k个位置。",
            "思路：闭合为环\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：先找到链表长度，将链表闭合为环，然后找到新的头节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* rotateRight(ListNode* head, int k) {\n        if (!head || !head->next || k == 0) return head;\n        \n        // 计算链表长度并找到尾节点\n        int n = 1;\n        ListNode* tail = head;\n        while (tail->next) {\n            tail = tail->next;\n            n++;\n        }\n        \n        // 闭合为环\n        tail->next = head;\n        \n        // 找到新的尾节点\n        k = k % n;\n        for (int i = 0; i < n - k; i++) {\n            tail = tail->next;\n        }\n        \n        // 断开环\n        ListNode* newHead = tail->next;\n        tail->next = nullptr;\n        \n        return newHead;\n    }\n};\n```",
            "链表"
        )
        
        # 83. 删除排序链表中的重复元素
        self.add_card(
            "LeetCode 83. 删除排序链表中的重复元素\n\n给定一个已排序的链表的头head，删除所有重复的元素，使每个元素只出现一次。返回已排序的链表。",
            "思路：一次遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：遍历链表，如果当前节点和下一个节点值相同，则删除下一个节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* deleteDuplicates(ListNode* head) {\n        ListNode* curr = head;\n        while (curr && curr->next) {\n            if (curr->val == curr->next->val) {\n                curr->next = curr->next->next;\n            } else {\n                curr = curr->next;\n            }\n        }\n        return head;\n    }\n};\n```",
            "链表"
        )
        
        # 92. 反转链表 II
        self.add_card(
            "LeetCode 92. 反转链表 II\n\n给你单链表的头指针head和两个整数left和right，其中left<=right。请你反转从位置left到位置right的链表节点，返回反转后的链表。",
            "思路：一次遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：找到需要反转的区间，然后反转这部分链表\n\n```cpp\nclass Solution {\npublic:\n    ListNode* reverseBetween(ListNode* head, int left, int right) {\n        ListNode* dummy = new ListNode(0);\n        dummy->next = head;\n        ListNode* prev = dummy;\n        \n        // 找到left位置的前一个节点\n        for (int i = 0; i < left - 1; i++) {\n            prev = prev->next;\n        }\n        \n        // 反转从left到right的节点\n        ListNode* curr = prev->next;\n        for (int i = 0; i < right - left; i++) {\n            ListNode* next = curr->next;\n            curr->next = next->next;\n            next->next = prev->next;\n            prev->next = next;\n        }\n        \n        return dummy->next;\n    }\n};\n```",
            "链表"
        )
        
        # 100. 相同的树
        self.add_card(
            "LeetCode 100. 相同的树\n\n给你两棵二叉树的根节点p和q，编写一个函数来检验这两棵树是否相同。",
            "思路1：递归\n- 时间复杂度：O(min(m,n))\n- 空间复杂度：O(min(m,n))\n- 思路：递归比较两棵树的每个节点\n\n```cpp\nclass Solution {\npublic:\n    bool isSameTree(TreeNode* p, TreeNode* q) {\n        if (!p && !q) return true;\n        if (!p || !q) return false;\n        if (p->val != q->val) return false;\n        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);\n    }\n};\n```\n\n思路2：迭代（队列）\n- 时间复杂度：O(min(m,n))\n- 空间复杂度：O(min(m,n))\n- 思路：使用队列进行层序遍历，同时比较两棵树\n\n```cpp\nclass Solution {\npublic:\n    bool isSameTree(TreeNode* p, TreeNode* q) {\n        queue<TreeNode*> queue;\n        queue.push(p);\n        queue.push(q);\n        \n        while (!queue.empty()) {\n            TreeNode* node1 = queue.front(); queue.pop();\n            TreeNode* node2 = queue.front(); queue.pop();\n            \n            if (!node1 && !node2) continue;\n            if (!node1 || !node2) return false;\n            if (node1->val != node2->val) return false;\n            \n            queue.push(node1->left);\n            queue.push(node2->left);\n            queue.push(node1->right);\n            queue.push(node2->right);\n        }\n        \n        return true;\n    }\n};\n```",
            "树-二叉树"
        )
        
        # 108. 将有序数组转换为二叉搜索树
        self.add_card(
            "LeetCode 108. 将有序数组转换为二叉搜索树\n\n给你一个整数数组nums，其中元素已经按升序排列，请你将其转换为一棵高度平衡二叉搜索树。",
            "思路：递归（分治）\n- 时间复杂度：O(n)\n- 空间复杂度：O(logn)\n- 思路：每次选择中间元素作为根节点，递归构建左右子树\n\n```cpp\nclass Solution {\npublic:\n    TreeNode* sortedArrayToBST(vector<int>& nums) {\n        return buildBST(nums, 0, nums.size() - 1);\n    }\n    \nprivate:\n    TreeNode* buildBST(vector<int>& nums, int left, int right) {\n        if (left > right) return nullptr;\n        \n        int mid = left + (right - left) / 2;\n        TreeNode* root = new TreeNode(nums[mid]);\n        root->left = buildBST(nums, left, mid - 1);\n        root->right = buildBST(nums, mid + 1, right);\n        \n        return root;\n    }\n};\n```",
            "树-二叉搜索树"
        )
        
        # 110. 平衡二叉树
        self.add_card(
            "LeetCode 110. 平衡二叉树\n\n给定一个二叉树，判断它是否是高度平衡的二叉树。",
            "思路：自底向上递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：计算每个节点的高度，如果左右子树高度差大于1，返回-1表示不平衡\n\n```cpp\nclass Solution {\npublic:\n    bool isBalanced(TreeNode* root) {\n        return height(root) != -1;\n    }\n    \nprivate:\n    int height(TreeNode* root) {\n        if (!root) return 0;\n        \n        int leftHeight = height(root->left);\n        if (leftHeight == -1) return -1;\n        \n        int rightHeight = height(root->right);\n        if (rightHeight == -1) return -1;\n        \n        if (abs(leftHeight - rightHeight) > 1) return -1;\n        \n        return max(leftHeight, rightHeight) + 1;\n    }\n};\n```",
            "树-二叉树"
        )
        
        # 111. 二叉树的最小深度
        self.add_card(
            "LeetCode 111. 二叉树的最小深度\n\n给定一个二叉树，找出其最小深度。最小深度是从根节点到最近叶子节点的最短路径上的节点数量。",
            "思路1：递归（DFS）\n- 时间复杂度：O(n)\n- 空间复杂度：O(h)\n- 思路：递归计算左右子树的最小深度\n\n```cpp\nclass Solution {\npublic:\n    int minDepth(TreeNode* root) {\n        if (!root) return 0;\n        if (!root->left && !root->right) return 1;\n        \n        int minDepth = INT_MAX;\n        if (root->left) {\n            minDepth = min(minDepth, minDepth(root->left));\n        }\n        if (root->right) {\n            minDepth = min(minDepth, minDepth(root->right));\n        }\n        \n        return minDepth + 1;\n    }\n};\n```\n\n思路2：迭代（BFS）\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用队列进行层序遍历，找到第一个叶子节点\n\n```cpp\nclass Solution {\npublic:\n    int minDepth(TreeNode* root) {\n        if (!root) return 0;\n        \n        queue<TreeNode*> q;\n        q.push(root);\n        int depth = 1;\n        \n        while (!q.empty()) {\n            int size = q.size();\n            for (int i = 0; i < size; i++) {\n                TreeNode* node = q.front();\n                q.pop();\n                \n                if (!node->left && !node->right) return depth;\n                \n                if (node->left) q.push(node->left);\n                if (node->right) q.push(node->right);\n            }\n            depth++;\n        }\n        \n        return depth;\n    }\n};\n```",
            "树-BFS"
        )
        
        # 112. 路径总和
        self.add_card(
            "LeetCode 112. 路径总和\n\n给你二叉树的根节点root和一个表示目标和的整数targetSum。判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和targetSum。",
            "思路：递归（DFS）\n- 时间复杂度：O(n)\n- 空间复杂度：O(h)\n- 思路：递归遍历每个节点，检查是否存在路径和等于targetSum\n\n```cpp\nclass Solution {\npublic:\n    bool hasPathSum(TreeNode* root, int targetSum) {\n        if (!root) return false;\n        \n        if (!root->left && !root->right) {\n            return root->val == targetSum;\n        }\n        \n        return hasPathSum(root->left, targetSum - root->val) ||\n               hasPathSum(root->right, targetSum - root->val);\n    }\n};\n```",
            "树-DFS"
        )
        
        # 62. 不同路径
        self.add_card(
            "LeetCode 62. 不同路径\n\n一个机器人位于一个m x n网格的左上角。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。\n\n问总共有多少条不同的路径？",
            "思路1：动态规划\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mn)\n- 思路：dp[i][j]表示到达(i,j)的路径数，dp[i][j]=dp[i-1][j]+dp[i][j-1]\n\n```cpp\nclass Solution {\npublic:\n    int uniquePaths(int m, int n) {\n        vector<vector<int>> dp(m, vector<int>(n, 1));\n        \n        for (int i = 1; i < m; i++) {\n            for (int j = 1; j < n; j++) {\n                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];\n            }\n        }\n        \n        return dp[m - 1][n - 1];\n    }\n};\n```\n\n思路2：空间优化\n- 时间复杂度：O(mn)\n- 空间复杂度：O(n)\n- 思路：只保存一行的状态\n\n```cpp\nclass Solution {\npublic:\n    int uniquePaths(int m, int n) {\n        vector<int> dp(n, 1);\n        \n        for (int i = 1; i < m; i++) {\n            for (int j = 1; j < n; j++) {\n                dp[j] += dp[j - 1];\n            }\n        }\n        \n        return dp[n - 1];\n    }\n};\n```\n\n思路3：组合数学\n- 时间复杂度：O(min(m,n))\n- 空间复杂度：O(1)\n- 思路：总路径数等于C(m+n-2, m-1)\n\n```cpp\nclass Solution {\npublic:\n    int uniquePaths(int m, int n) {\n        long long result = 1;\n        for (int i = 1; i < min(m, n); i++) {\n            result = result * (m + n - 1 - i) / i;\n        }\n        return result;\n    }\n};\n```",
            "动态规划"
        )
        
        # 63. 不同路径 II
        self.add_card(
            "LeetCode 63. 不同路径 II\n\n一个机器人位于一个m x n网格的左上角。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。\n\n现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？",
            "思路：动态规划\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mn)\n- 思路：dp[i][j]表示到达(i,j)的路径数，如果(i,j)有障碍物，则dp[i][j]=0\n\n```cpp\nclass Solution {\npublic:\n    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {\n        int m = obstacleGrid.size();\n        int n = obstacleGrid[0].size();\n        \n        vector<vector<int>> dp(m, vector<int>(n, 0));\n        \n        // 初始化第一行和第一列\n        for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++) {\n            dp[i][0] = 1;\n        }\n        for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++) {\n            dp[0][j] = 1;\n        }\n        \n        for (int i = 1; i < m; i++) {\n            for (int j = 1; j < n; j++) {\n                if (obstacleGrid[i][j] == 0) {\n                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];\n                }\n            }\n        }\n        \n        return dp[m - 1][n - 1];\n    }\n};\n```",
            "动态规划"
        )
        
        # 64. 最小路径和
        self.add_card(
            "LeetCode 64. 最小路径和\n\n给定一个包含非负整数的m x n网格grid，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。",
            "思路：动态规划\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mn)\n- 思路：dp[i][j]表示到达(i,j)的最小路径和，dp[i][j]=min(dp[i-1][j], dp[i][j-1])+grid[i][j]\n\n```cpp\nclass Solution {\npublic:\n    int minPathSum(vector<vector<int>>& grid) {\n        int m = grid.size();\n        int n = grid[0].size();\n        \n        vector<vector<int>> dp(m, vector<int>(n, 0));\n        dp[0][0] = grid[0][0];\n        \n        // 初始化第一行和第一列\n        for (int i = 1; i < m; i++) {\n            dp[i][0] = dp[i - 1][0] + grid[i][0];\n        }\n        for (int j = 1; j < n; j++) {\n            dp[0][j] = dp[0][j - 1] + grid[0][j];\n        }\n        \n        for (int i = 1; i < m; i++) {\n            for (int j = 1; j < n; j++) {\n                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];\n            }\n        }\n        \n        return dp[m - 1][n - 1];\n    }\n};\n```",
            "动态规划"
        )
        
        # 121. 买卖股票的最佳时机
        self.add_card(
            "LeetCode 121. 买卖股票的最佳时机\n\n给定一个数组prices，它的第i个元素prices[i]表示一支给定股票第i天的价格。\n\n你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。设计一个算法来计算你所能获取的最大利润。",
            "思路：一次遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：记录最低价格，计算每天卖出能获得的最大利润\n\n```cpp\nclass Solution {\npublic:\n    int maxProfit(vector<int>& prices) {\n        int minPrice = INT_MAX;\n        int maxProfit = 0;\n        \n        for (int price : prices) {\n            if (price < minPrice) {\n                minPrice = price;\n            } else if (price - minPrice > maxProfit) {\n                maxProfit = price - minPrice;\n            }\n        }\n        \n        return maxProfit;\n    }\n};\n```",
            "动态规划-股票"
        )
        
        # 198. 打家劫舍
        self.add_card(
            "LeetCode 198. 打家劫舍\n\n你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。\n\n给定一个代表每个房屋存放金额的非负整数数组，计算你不触动警报装置的情况下，一夜之内能够偷窃到的最高金额。",
            "思路：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：dp[i]表示前i间房屋能偷窃到的最高金额，dp[i]=max(dp[i-1], dp[i-2]+nums[i])\n\n```cpp\nclass Solution {\npublic:\n    int rob(vector<int>& nums) {\n        int prev = 0, curr = 0;\n        \n        for (int num : nums) {\n            int temp = curr;\n            curr = max(curr, prev + num);\n            prev = temp;\n        }\n        \n        return curr;\n    }\n};\n```",
            "动态规划"
        )
        
        # 42. 接雨水
        self.add_card(
            "LeetCode 42. 接雨水\n\n给定n个非负整数表示每个宽度为1的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。",
            "思路1：双指针\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：从两端向中间遍历，维护左右最大高度\n\n```cpp\nclass Solution {\npublic:\n    int trap(vector<int>& height) {\n        int left = 0, right = height.size() - 1;\n        int leftMax = 0, rightMax = 0;\n        int result = 0;\n        \n        while (left < right) {\n            if (height[left] < height[right]) {\n                if (height[left] >= leftMax) {\n                    leftMax = height[left];\n                } else {\n                    result += leftMax - height[left];\n                }\n                left++;\n            } else {\n                if (height[right] >= rightMax) {\n                    rightMax = height[right];\n                } else {\n                    result += rightMax - height[right];\n                }\n                right--;\n            }\n        }\n        \n        return result;\n    }\n};\n```\n\n思路2：栈\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用栈存储递减的柱子索引，遇到更高的柱子时计算雨水\n\n```cpp\nclass Solution {\npublic:\n    int trap(vector<int>& height) {\n        stack<int> st;\n        int result = 0;\n        \n        for (int i = 0; i < height.size(); i++) {\n            while (!st.empty() && height[i] > height[st.top()]) {\n                int top = st.top();\n                st.pop();\n                if (st.empty()) break;\n                \n                int distance = i - st.top() - 1;\n                int boundedHeight = min(height[i], height[st.top()]) - height[top];\n                result += distance * boundedHeight;\n            }\n            st.push(i);\n        }\n        \n        return result;\n    }\n};\n```",
            "栈-双指针"
        )
        
        # 55. 跳跃游戏
        self.add_card(
            "LeetCode 55. 跳跃游戏\n\n给你一个非负整数数组nums，你最初位于数组的第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。\n\n判断你是否能够到达最后一个下标。",
            "思路：贪心\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：维护能到达的最远位置，如果当前位置超过最远位置，则无法到达\n\n```cpp\nclass Solution {\npublic:\n    bool canJump(vector<int>& nums) {\n        int maxReach = 0;\n        \n        for (int i = 0; i < nums.size(); i++) {\n            if (i > maxReach) return false;\n            maxReach = max(maxReach, i + nums[i]);\n            if (maxReach >= nums.size() - 1) return true;\n        }\n        \n        return true;\n    }\n};\n```",
            "贪心"
        )
        
        # 56. 合并区间
        self.add_card(
            "LeetCode 56. 合并区间\n\n以数组intervals表示若干个区间的集合，其中单个区间为intervals[i]=[starti,endi]。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。",
            "思路：排序+合并\n- 时间复杂度：O(nlogn)\n- 空间复杂度：O(1)（不考虑结果数组）\n- 思路：先按起始位置排序，然后合并重叠的区间\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> merge(vector<vector<int>>& intervals) {\n        sort(intervals.begin(), intervals.end());\n        \n        vector<vector<int>> result;\n        for (auto& interval : intervals) {\n            if (result.empty() || result.back()[1] < interval[0]) {\n                result.push_back(interval);\n            } else {\n                result.back()[1] = max(result.back()[1], interval[1]);\n            }\n        }\n        \n        return result;\n    }\n};\n```",
            "数组-贪心"
        )
        
        # 17. 电话号码的字母组合
        self.add_card(
            "LeetCode 17. 电话号码的字母组合\n\n给定一个仅包含数字2-9的字符串，返回所有它能表示的字母组合。答案可以按任意顺序返回。",
            "思路：回溯\n- 时间复杂度：O(4^n×n)，n为数字个数\n- 空间复杂度：O(n)\n- 思路：使用回溯算法生成所有可能的字母组合\n\n```cpp\nclass Solution {\npublic:\n    vector<string> letterCombinations(string digits) {\n        if (digits.empty()) return {};\n        \n        vector<string> result;\n        string current;\n        backtrack(digits, 0, current, result);\n        return result;\n    }\n    \nprivate:\n    vector<string> phoneMap = {\"\", \"\", \"abc\", \"def\", \"ghi\", \"jkl\", \"mno\", \"pqrs\", \"tuv\", \"wxyz\"};\n    \n    void backtrack(string& digits, int index, string& current, vector<string>& result) {\n        if (index == digits.length()) {\n            result.push_back(current);\n            return;\n        }\n        \n        string letters = phoneMap[digits[index] - '0'];\n        for (char letter : letters) {\n            current.push_back(letter);\n            backtrack(digits, index + 1, current, result);\n            current.pop_back();\n        }\n    }\n};\n```",
            "回溯"
        )
        
        # 22. 括号生成
        self.add_card(
            "LeetCode 22. 括号生成\n\n数字n代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且有效的括号组合。",
            "思路：回溯\n- 时间复杂度：O(4^n/√n)\n- 空间复杂度：O(n)\n- 思路：使用回溯算法，确保左括号数量始终大于等于右括号数量\n\n```cpp\nclass Solution {\npublic:\n    vector<string> generateParenthesis(int n) {\n        vector<string> result;\n        string current;\n        backtrack(n, 0, 0, current, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(int n, int left, int right, string& current, vector<string>& result) {\n        if (current.length() == 2 * n) {\n            result.push_back(current);\n            return;\n        }\n        \n        if (left < n) {\n            current += '(';\n            backtrack(n, left + 1, right, current, result);\n            current.pop_back();\n        }\n        \n        if (right < left) {\n            current += ')';\n            backtrack(n, left, right + 1, current, result);\n            current.pop_back();\n        }\n    }\n};\n```",
            "回溯"
        )
        
        # 46. 全排列
        self.add_card(
            "LeetCode 46. 全排列\n\n给定一个不含重复数字的数组nums，返回其所有可能的全排列。你可以按任意顺序返回答案。",
            "思路：回溯\n- 时间复杂度：O(n×n!)\n- 空间复杂度：O(n)\n- 思路：使用回溯算法生成所有排列，使用visited数组标记已使用的元素\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> permute(vector<int>& nums) {\n        vector<vector<int>> result;\n        vector<int> current;\n        vector<bool> visited(nums.size(), false);\n        backtrack(nums, current, visited, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(vector<int>& nums, vector<int>& current, vector<bool>& visited, vector<vector<int>>& result) {\n        if (current.size() == nums.size()) {\n            result.push_back(current);\n            return;\n        }\n        \n        for (int i = 0; i < nums.size(); i++) {\n            if (!visited[i]) {\n                visited[i] = true;\n                current.push_back(nums[i]);\n                backtrack(nums, current, visited, result);\n                current.pop_back();\n                visited[i] = false;\n            }\n        }\n    }\n};\n```",
            "回溯"
        )
        
        # 78. 子集
        self.add_card(
            "LeetCode 78. 子集\n\n给你一个整数数组nums，数组中的元素互不相同。返回该数组所有可能的子集（幂集）。",
            "思路1：回溯\n- 时间复杂度：O(n×2^n)\n- 空间复杂度：O(n)\n- 思路：使用回溯算法生成所有子集\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> subsets(vector<int>& nums) {\n        vector<vector<int>> result;\n        vector<int> current;\n        backtrack(nums, 0, current, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(vector<int>& nums, int start, vector<int>& current, vector<vector<int>>& result) {\n        result.push_back(current);\n        \n        for (int i = start; i < nums.size(); i++) {\n            current.push_back(nums[i]);\n            backtrack(nums, i + 1, current, result);\n            current.pop_back();\n        }\n    }\n};\n```\n\n思路2：位运算\n- 时间复杂度：O(n×2^n)\n- 空间复杂度：O(1)（不考虑结果数组）\n- 思路：使用位掩码表示每个元素是否在子集中\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> subsets(vector<int>& nums) {\n        vector<vector<int>> result;\n        int n = nums.size();\n        \n        for (int mask = 0; mask < (1 << n); mask++) {\n            vector<int> subset;\n            for (int i = 0; i < n; i++) {\n                if (mask & (1 << i)) {\n                    subset.push_back(nums[i]);\n                }\n            }\n            result.push_back(subset);\n        }\n        \n        return result;\n    }\n};\n```",
            "回溯-位运算"
        )
        
        # 继续添加更多题目以完成100道
        # 33. 搜索旋转排序数组
        self.add_card(
            "LeetCode 33. 搜索旋转排序数组\n\n整数数组nums按升序排列，数组中的值互不相同。\n\n在传递给函数之前，nums在预先未知的某个下标k（0<=k<nums.length）上进行了旋转。给你旋转后的数组nums和一个整数target，如果nums中存在这个目标值target，则返回它的下标，否则返回-1。",
            "思路：二分查找\n- 时间复杂度：O(logn)\n- 空间复杂度：O(1)\n- 思路：根据mid位置判断左半部分还是右半部分是有序的，然后决定搜索方向\n\n```cpp\nclass Solution {\npublic:\n    int search(vector<int>& nums, int target) {\n        int left = 0, right = nums.size() - 1;\n        \n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            if (nums[mid] == target) return mid;\n            \n            // 左半部分有序\n            if (nums[left] <= nums[mid]) {\n                if (nums[left] <= target && target < nums[mid]) {\n                    right = mid - 1;\n                } else {\n                    left = mid + 1;\n                }\n            } else { // 右半部分有序\n                if (nums[mid] < target && target <= nums[right]) {\n                    left = mid + 1;\n                } else {\n                    right = mid - 1;\n                }\n            }\n        }\n        \n        return -1;\n    }\n};\n```",
            "数组-二分查找"
        )
        
        # 34. 在排序数组中查找元素的第一个和最后一个位置
        self.add_card(
            "LeetCode 34. 在排序数组中查找元素的第一个和最后一个位置\n\n给你一个按照非递减顺序排列的整数数组nums，和一个目标值target。请你找出给定目标值在数组中的开始位置和结束位置。\n\n如果数组中不存在目标值target，返回[-1,-1]。",
            "思路：二分查找\n- 时间复杂度：O(logn)\n- 空间复杂度：O(1)\n- 思路：使用两次二分查找，分别找到第一个和最后一个位置\n\n```cpp\nclass Solution {\npublic:\n    vector<int> searchRange(vector<int>& nums, int target) {\n        int first = findFirst(nums, target);\n        if (first == -1) return {-1, -1};\n        int last = findLast(nums, target);\n        return {first, last};\n    }\n    \nprivate:\n    int findFirst(vector<int>& nums, int target) {\n        int left = 0, right = nums.size() - 1;\n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            if (nums[mid] < target) {\n                left = mid + 1;\n            } else if (nums[mid] > target) {\n                right = mid - 1;\n            } else {\n                if (mid == 0 || nums[mid - 1] != target) return mid;\n                right = mid - 1;\n            }\n        }\n        return -1;\n    }\n    \n    int findLast(vector<int>& nums, int target) {\n        int left = 0, right = nums.size() - 1;\n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            if (nums[mid] < target) {\n                left = mid + 1;\n            } else if (nums[mid] > target) {\n                right = mid - 1;\n            } else {\n                if (mid == nums.size() - 1 || nums[mid + 1] != target) return mid;\n                left = mid + 1;\n            }\n        }\n        return -1;\n    }\n};\n```",
            "数组-二分查找"
        )
        
        # 39. 组合总和
        self.add_card(
            "LeetCode 39. 组合总和\n\n给你一个无重复元素的整数数组candidates和一个目标整数target，找出candidates中可以使数字和为目标数target的所有不同组合，并以列表形式返回。",
            "思路：回溯\n- 时间复杂度：O(2^n)\n- 空间复杂度：O(target)\n- 思路：使用回溯算法，可以重复使用同一个数字\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {\n        vector<vector<int>> result;\n        vector<int> current;\n        backtrack(candidates, target, 0, current, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(vector<int>& candidates, int target, int start, vector<int>& current, vector<vector<int>>& result) {\n        if (target == 0) {\n            result.push_back(current);\n            return;\n        }\n        \n        if (target < 0) return;\n        \n        for (int i = start; i < candidates.size(); i++) {\n            current.push_back(candidates[i]);\n            backtrack(candidates, target - candidates[i], i, current, result);\n            current.pop_back();\n        }\n    }\n};\n```",
            "回溯"
        )
        
        # 49. 字母异位词分组
        self.add_card(
            "LeetCode 49. 字母异位词分组\n\n给你一个字符串数组，请你将字母异位词组合在一起。可以按任意顺序返回结果列表。",
            "思路1：排序+哈希表\n- 时间复杂度：O(nklogk)，k为字符串平均长度\n- 空间复杂度：O(nk)\n- 思路：将每个字符串排序后作为key，相同key的字符串为一组\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<string>> groupAnagrams(vector<string>& strs) {\n        unordered_map<string, vector<string>> map;\n        \n        for (string& str : strs) {\n            string key = str;\n            sort(key.begin(), key.end());\n            map[key].push_back(str);\n        }\n        \n        vector<vector<string>> result;\n        for (auto& pair : map) {\n            result.push_back(pair.second);\n        }\n        \n        return result;\n    }\n};\n```\n\n思路2：计数+哈希表\n- 时间复杂度：O(nk)\n- 空间复杂度：O(nk)\n- 思路：使用字符计数作为key，避免排序\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<string>> groupAnagrams(vector<string>& strs) {\n        unordered_map<string, vector<string>> map;\n        \n        for (string& str : strs) {\n            string key = getKey(str);\n            map[key].push_back(str);\n        }\n        \n        vector<vector<string>> result;\n        for (auto& pair : map) {\n            result.push_back(pair.second);\n        }\n        \n        return result;\n    }\n    \nprivate:\n    string getKey(string& str) {\n        vector<int> count(26, 0);\n        for (char c : str) {\n            count[c - 'a']++;\n        }\n        string key;\n        for (int i = 0; i < 26; i++) {\n            key += to_string(count[i]) + '#';\n        }\n        return key;\n    }\n};\n```",
            "字符串-哈希表"
        )
        
        # 50. Pow(x, n)
        self.add_card(
            "LeetCode 50. Pow(x, n)\n\n实现pow(x,n)，即计算x的n次幂函数（即xⁿ）。",
            "思路：快速幂\n- 时间复杂度：O(logn)\n- 空间复杂度：O(1)\n- 思路：将指数n转换为二进制，利用x^(2^k)的性质快速计算\n\n```cpp\nclass Solution {\npublic:\n    double myPow(double x, int n) {\n        long long N = n;\n        if (N < 0) {\n            x = 1 / x;\n            N = -N;\n        }\n        \n        double result = 1.0;\n        double current = x;\n        \n        while (N > 0) {\n            if (N % 2 == 1) {\n                result *= current;\n            }\n            current *= current;\n            N /= 2;\n        }\n        \n        return result;\n    }\n};\n```",
            "数学-快速幂"
        )
        
        # 69. x的平方根
        self.add_card(
            "LeetCode 69. x的平方根\n\n给你一个非负整数x，计算并返回x的算术平方根。\n\n由于返回类型是整数，结果只保留整数部分，小数部分将被舍去。",
            "思路1：二分查找\n- 时间复杂度：O(logx)\n- 空间复杂度：O(1)\n- 思路：在[0,x]范围内二分查找平方根\n\n```cpp\nclass Solution {\npublic:\n    int mySqrt(int x) {\n        if (x < 2) return x;\n        \n        int left = 2, right = x / 2;\n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            long long square = (long long)mid * mid;\n            if (square == x) return mid;\n            else if (square < x) left = mid + 1;\n            else right = mid - 1;\n        }\n        \n        return right;\n    }\n};\n```\n\n思路2：牛顿法\n- 时间复杂度：O(logx)\n- 空间复杂度：O(1)\n- 思路：使用牛顿迭代法求平方根\n\n```cpp\nclass Solution {\npublic:\n    int mySqrt(int x) {\n        if (x < 2) return x;\n        \n        double x0 = x;\n        double x1 = (x0 + x / x0) / 2.0;\n        \n        while (abs(x0 - x1) >= 1) {\n            x0 = x1;\n            x1 = (x0 + x / x0) / 2.0;\n        }\n        \n        return (int)x1;\n    }\n};\n```",
            "数学-二分查找"
        )
        
        # 73. 矩阵置零
        self.add_card(
            "LeetCode 73. 矩阵置零\n\n给定一个m×n的矩阵，如果一个元素为0，则将其所在行和列的所有元素都设为0。请使用原地算法。",
            "思路1：使用标记数组\n- 时间复杂度：O(mn)\n- 空间复杂度：O(m+n)\n- 思路：使用两个数组记录哪些行和列需要置零\n\n```cpp\nclass Solution {\npublic:\n    void setZeroes(vector<vector<int>>& matrix) {\n        int m = matrix.size();\n        int n = matrix[0].size();\n        vector<bool> rowZero(m, false);\n        vector<bool> colZero(n, false);\n        \n        for (int i = 0; i < m; i++) {\n            for (int j = 0; j < n; j++) {\n                if (matrix[i][j] == 0) {\n                    rowZero[i] = true;\n                    colZero[j] = true;\n                }\n            }\n        }\n        \n        for (int i = 0; i < m; i++) {\n            for (int j = 0; j < n; j++) {\n                if (rowZero[i] || colZero[j]) {\n                    matrix[i][j] = 0;\n                }\n            }\n        }\n    }\n};\n```\n\n思路2：使用第一行和第一列作为标记（空间优化）\n- 时间复杂度：O(mn)\n- 空间复杂度：O(1)\n- 思路：使用矩阵的第一行和第一列来标记需要置零的行和列\n\n```cpp\nclass Solution {\npublic:\n    void setZeroes(vector<vector<int>>& matrix) {\n        int m = matrix.size();\n        int n = matrix[0].size();\n        bool firstRowZero = false, firstColZero = false;\n        \n        // 检查第一行和第一列是否有0\n        for (int j = 0; j < n; j++) {\n            if (matrix[0][j] == 0) firstRowZero = true;\n        }\n        for (int i = 0; i < m; i++) {\n            if (matrix[i][0] == 0) firstColZero = true;\n        }\n        \n        // 使用第一行和第一列作为标记\n        for (int i = 1; i < m; i++) {\n            for (int j = 1; j < n; j++) {\n                if (matrix[i][j] == 0) {\n                    matrix[i][0] = 0;\n                    matrix[0][j] = 0;\n                }\n            }\n        }\n        \n        // 根据标记置零\n        for (int i = 1; i < m; i++) {\n            for (int j = 1; j < n; j++) {\n                if (matrix[i][0] == 0 || matrix[0][j] == 0) {\n                    matrix[i][j] = 0;\n                }\n            }\n        }\n        \n        // 处理第一行和第一列\n        if (firstRowZero) {\n            for (int j = 0; j < n; j++) matrix[0][j] = 0;\n        }\n        if (firstColZero) {\n            for (int i = 0; i < m; i++) matrix[i][0] = 0;\n        }\n    }\n};\n```",
            "数组-矩阵"
        )
        
        # 79. 单词搜索
        self.add_card(
            "LeetCode 79. 单词搜索\n\n给定一个m×n二维字符网格board和一个字符串单词word。如果word存在于网格中，返回true；否则，返回false。",
            "思路：回溯+DFS\n- 时间复杂度：O(mn×4^L)，L为单词长度\n- 空间复杂度：O(L)\n- 思路：从每个位置开始，使用DFS搜索单词，使用回溯避免重复访问\n\n```cpp\nclass Solution {\npublic:\n    bool exist(vector<vector<char>>& board, string word) {\n        int m = board.size();\n        int n = board[0].size();\n        \n        for (int i = 0; i < m; i++) {\n            for (int j = 0; j < n; j++) {\n                if (backtrack(board, word, i, j, 0)) {\n                    return true;\n                }\n            }\n        }\n        \n        return false;\n    }\n    \nprivate:\n    bool backtrack(vector<vector<char>>& board, string& word, int i, int j, int index) {\n        if (index == word.length()) return true;\n        \n        if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || \n            board[i][j] != word[index]) {\n            return false;\n        }\n        \n        char temp = board[i][j];\n        board[i][j] = '#'; // 标记为已访问\n        \n        bool found = backtrack(board, word, i + 1, j, index + 1) ||\n                     backtrack(board, word, i - 1, j, index + 1) ||\n                     backtrack(board, word, i, j + 1, index + 1) ||\n                     backtrack(board, word, i, j - 1, index + 1);\n        \n        board[i][j] = temp; // 恢复\n        \n        return found;\n    }\n};\n```",
            "回溯-DFS"
        )
        
        # 96. 不同的二叉搜索树
        self.add_card(
            "LeetCode 96. 不同的二叉搜索树\n\n给你一个整数n，求恰由n个节点组成且节点值从1到n互不相同的二叉搜索树有多少种？返回满足题意的二叉搜索树的种数。",
            "思路：动态规划（卡特兰数）\n- 时间复杂度：O(n²)\n- 空间复杂度：O(n)\n- 思路：dp[i]表示i个节点能组成的BST数量，dp[i]=Σ(dp[j-1]*dp[i-j])，j从1到i\n\n```cpp\nclass Solution {\npublic:\n    int numTrees(int n) {\n        vector<int> dp(n + 1, 0);\n        dp[0] = 1;\n        dp[1] = 1;\n        \n        for (int i = 2; i <= n; i++) {\n            for (int j = 1; j <= i; j++) {\n                dp[i] += dp[j - 1] * dp[i - j];\n            }\n        }\n        \n        return dp[n];\n    }\n};\n```",
            "动态规划-树"
        )
        
        # 98. 验证二叉搜索树
        self.add_card(
            "LeetCode 98. 验证二叉搜索树\n\n给你一个二叉树的根节点root，判断其是否是一个有效的二叉搜索树。",
            "思路1：递归（上下界）\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：递归检查每个节点是否在有效范围内\n\n```cpp\nclass Solution {\npublic:\n    bool isValidBST(TreeNode* root) {\n        return isValidBST(root, LONG_MIN, LONG_MAX);\n    }\n    \nprivate:\n    bool isValidBST(TreeNode* root, long minVal, long maxVal) {\n        if (!root) return true;\n        \n        if (root->val <= minVal || root->val >= maxVal) return false;\n        \n        return isValidBST(root->left, minVal, root->val) &&\n               isValidBST(root->right, root->val, maxVal);\n    }\n};\n```\n\n思路2：中序遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：BST的中序遍历是递增的\n\n```cpp\nclass Solution {\npublic:\n    bool isValidBST(TreeNode* root) {\n        stack<TreeNode*> st;\n        TreeNode* curr = root;\n        long prev = LONG_MIN;\n        \n        while (curr || !st.empty()) {\n            while (curr) {\n                st.push(curr);\n                curr = curr->left;\n            }\n            curr = st.top();\n            st.pop();\n            \n            if (curr->val <= prev) return false;\n            prev = curr->val;\n            \n            curr = curr->right;\n        }\n        \n        return true;\n    }\n};\n```",
            "树-二叉搜索树"
        )
        
        # 105. 从前序与中序遍历序列构造二叉树
        self.add_card(
            "LeetCode 105. 从前序与中序遍历序列构造二叉树\n\n给定两个整数数组preorder和inorder，其中preorder是二叉树的先序遍历，inorder是同一棵树的中序遍历，请构造二叉树并返回其根节点。",
            "思路：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：前序遍历的第一个元素是根节点，在中序遍历中找到根节点，递归构建左右子树\n\n```cpp\nclass Solution {\npublic:\n    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {\n        unordered_map<int, int> map;\n        for (int i = 0; i < inorder.size(); i++) {\n            map[inorder[i]] = i;\n        }\n        \n        return build(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1, map);\n    }\n    \nprivate:\n    TreeNode* build(vector<int>& preorder, int preStart, int preEnd,\n                   vector<int>& inorder, int inStart, int inEnd,\n                   unordered_map<int, int>& map) {\n        if (preStart > preEnd || inStart > inEnd) return nullptr;\n        \n        int rootVal = preorder[preStart];\n        TreeNode* root = new TreeNode(rootVal);\n        \n        int rootIndex = map[rootVal];\n        int leftSize = rootIndex - inStart;\n        \n        root->left = build(preorder, preStart + 1, preStart + leftSize,\n                          inorder, inStart, rootIndex - 1, map);\n        root->right = build(preorder, preStart + leftSize + 1, preEnd,\n                           inorder, rootIndex + 1, inEnd, map);\n        \n        return root;\n    }\n};\n```",
            "树-二叉树"
        )
        
        # 113. 路径总和 II
        self.add_card(
            "LeetCode 113. 路径总和 II\n\n给你二叉树的根节点root和一个整数目标和targetSum，找出所有从根节点到叶子节点路径总和等于给定目标和的路径。",
            "思路：回溯+DFS\n- 时间复杂度：O(n²)\n- 空间复杂度：O(n)\n- 思路：使用DFS遍历所有路径，使用回溯记录当前路径\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {\n        vector<vector<int>> result;\n        vector<int> path;\n        backtrack(root, targetSum, path, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(TreeNode* root, int targetSum, vector<int>& path, vector<vector<int>>& result) {\n        if (!root) return;\n        \n        path.push_back(root->val);\n        \n        if (!root->left && !root->right && root->val == targetSum) {\n            result.push_back(path);\n        }\n        \n        backtrack(root->left, targetSum - root->val, path, result);\n        backtrack(root->right, targetSum - root->val, path, result);\n        \n        path.pop_back();\n    }\n};\n```",
            "树-回溯"
        )
        
        # 122. 买卖股票的最佳时机 II
        self.add_card(
            "LeetCode 122. 买卖股票的最佳时机 II\n\n给你一个整数数组prices，其中prices[i]表示某支股票第i天的价格。\n\n在每一天，你可以决定是否购买和/或出售股票。你在任何时候最多只能持有一股股票。你也可以先购买，然后在同一天出售。\n\n返回你能获得的最大利润。",
            "思路：贪心\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：只要后一天价格高于前一天，就进行交易\n\n```cpp\nclass Solution {\npublic:\n    int maxProfit(vector<int>& prices) {\n        int profit = 0;\n        for (int i = 1; i < prices.size(); i++) {\n            if (prices[i] > prices[i - 1]) {\n                profit += prices[i] - prices[i - 1];\n            }\n        }\n        return profit;\n    }\n};\n```",
            "贪心-股票"
        )
        
        # 139. 单词拆分
        self.add_card(
            "LeetCode 139. 单词拆分\n\n给你一个字符串s和一个字符串列表wordDict作为字典。请你判断是否可以利用字典中出现的单词拼接出s。",
            "思路：动态规划\n- 时间复杂度：O(n²)\n- 空间复杂度：O(n)\n- 思路：dp[i]表示s的前i个字符能否被拆分，dp[i]=dp[j]&&wordDict包含s[j:i]\n\n```cpp\nclass Solution {\npublic:\n    bool wordBreak(string s, vector<string>& wordDict) {\n        unordered_set<string> wordSet(wordDict.begin(), wordDict.end());\n        vector<bool> dp(s.length() + 1, false);\n        dp[0] = true;\n        \n        for (int i = 1; i <= s.length(); i++) {\n            for (int j = 0; j < i; j++) {\n                if (dp[j] && wordSet.count(s.substr(j, i - j))) {\n                    dp[i] = true;\n                    break;\n                }\n            }\n        }\n        \n        return dp[s.length()];\n    }\n};\n```",
            "动态规划-字符串"
        )
        
        # 152. 乘积最大子数组
        self.add_card(
            "LeetCode 152. 乘积最大子数组\n\n给你一个整数数组nums，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。",
            "思路：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：由于负数相乘会变正，需要同时记录最大值和最小值\n\n```cpp\nclass Solution {\npublic:\n    int maxProduct(vector<int>& nums) {\n        int maxProd = nums[0];\n        int minProd = nums[0];\n        int result = nums[0];\n        \n        for (int i = 1; i < nums.size(); i++) {\n            if (nums[i] < 0) {\n                swap(maxProd, minProd);\n            }\n            \n            maxProd = max(nums[i], maxProd * nums[i]);\n            minProd = min(nums[i], minProd * nums[i]);\n            \n            result = max(result, maxProd);\n        }\n        \n        return result;\n    }\n};\n```",
            "动态规划-数组"
        )
        
        # 155. 最小栈
        self.add_card(
            "LeetCode 155. 最小栈\n\n设计一个支持push，pop，top操作，并能在常数时间内检索到最小元素的栈。",
            "思路：辅助栈\n- 时间复杂度：O(1)所有操作\n- 空间复杂度：O(n)\n- 思路：使用两个栈，一个存储元素，一个存储最小值\n\n```cpp\nclass MinStack {\nprivate:\n    stack<int> dataStack;\n    stack<int> minStack;\n    \npublic:\n    MinStack() {\n        minStack.push(INT_MAX);\n    }\n    \n    void push(int val) {\n        dataStack.push(val);\n        minStack.push(min(val, minStack.top()));\n    }\n    \n    void pop() {\n        dataStack.pop();\n        minStack.pop();\n    }\n    \n    int top() {\n        return dataStack.top();\n    }\n    \n    int getMin() {\n        return minStack.top();\n    }\n};\n```",
            "栈-设计"
        )
        
        # 继续添加更多题目以完成100道
        # 由于篇幅限制，这里添加剩余的重要题目
        # 10. 正则表达式匹配
        self.add_card(
            "LeetCode 10. 正则表达式匹配\n\n给你一个字符串s和一个字符规律p，请你来实现一个支持'.'和'*'的正则表达式匹配。",
            "思路：动态规划\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mn)\n- 思路：dp[i][j]表示s的前i个字符和p的前j个字符是否匹配\n\n```cpp\nclass Solution {\npublic:\n    bool isMatch(string s, string p) {\n        int m = s.length(), n = p.length();\n        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));\n        dp[0][0] = true;\n        \n        for (int j = 2; j <= n; j++) {\n            if (p[j - 1] == '*') dp[0][j] = dp[0][j - 2];\n        }\n        \n        for (int i = 1; i <= m; i++) {\n            for (int j = 1; j <= n; j++) {\n                if (p[j - 1] == '*') {\n                    dp[i][j] = dp[i][j - 2] || \n                               (dp[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.'));\n                } else {\n                    dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');\n                }\n            }\n        }\n        \n        return dp[m][n];\n    }\n};\n```",
            "动态规划-字符串"
        )
        
        # 16. 最接近的三数之和
        self.add_card(
            "LeetCode 16. 最接近的三数之和\n\n给你一个长度为n的整数数组nums和一个目标值target。请你从nums中选出三个整数，使它们的和与target最接近。",
            "思路：排序+双指针\n- 时间复杂度：O(n²)\n- 空间复杂度：O(1)\n- 思路：先排序，固定第一个数，用双指针找另外两个数\n\n```cpp\nclass Solution {\npublic:\n    int threeSumClosest(vector<int>& nums, int target) {\n        sort(nums.begin(), nums.end());\n        int closestSum = nums[0] + nums[1] + nums[2];\n        \n        for (int i = 0; i < nums.size() - 2; i++) {\n            int left = i + 1, right = nums.size() - 1;\n            while (left < right) {\n                int sum = nums[i] + nums[left] + nums[right];\n                if (abs(sum - target) < abs(closestSum - target)) {\n                    closestSum = sum;\n                }\n                if (sum < target) left++;\n                else if (sum > target) right--;\n                else return sum;\n            }\n        }\n        \n        return closestSum;\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 18. 四数之和
        self.add_card(
            "LeetCode 18. 四数之和\n\n给你一个由n个整数组成的数组nums，和一个目标值target。请你找出并返回满足下述全部条件且不重复的四元组[nums[a],nums[b],nums[c],nums[d]]。",
            "思路：排序+双指针\n- 时间复杂度：O(n³)\n- 空间复杂度：O(1)\n- 思路：先排序，固定前两个数，用双指针找后两个数\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> fourSum(vector<int>& nums, int target) {\n        vector<vector<int>> result;\n        if (nums.size() < 4) return result;\n        \n        sort(nums.begin(), nums.end());\n        \n        for (int i = 0; i < nums.size() - 3; i++) {\n            if (i > 0 && nums[i] == nums[i - 1]) continue;\n            \n            for (int j = i + 1; j < nums.size() - 2; j++) {\n                if (j > i + 1 && nums[j] == nums[j - 1]) continue;\n                \n                int left = j + 1, right = nums.size() - 1;\n                while (left < right) {\n                    long long sum = (long long)nums[i] + nums[j] + nums[left] + nums[right];\n                    if (sum == target) {\n                        result.push_back({nums[i], nums[j], nums[left], nums[right]});\n                        while (left < right && nums[left] == nums[left + 1]) left++;\n                        while (left < right && nums[right] == nums[right - 1]) right--;\n                        left++;\n                        right--;\n                    } else if (sum < target) {\n                        left++;\n                    } else {\n                        right--;\n                    }\n                }\n            }\n        }\n        \n        return result;\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 28. 实现 strStr()
        self.add_card(
            "LeetCode 28. 实现 strStr()\n\n给你两个字符串haystack和needle，请你在haystack字符串中找出needle字符串的第一个匹配项的下标（下标从0开始）。",
            "思路1：暴力匹配\n- 时间复杂度：O(mn)\n- 空间复杂度：O(1)\n- 思路：逐个比较每个位置\n\n```cpp\nclass Solution {\npublic:\n    int strStr(string haystack, string needle) {\n        int m = haystack.length(), n = needle.length();\n        for (int i = 0; i <= m - n; i++) {\n            int j = 0;\n            while (j < n && haystack[i + j] == needle[j]) j++;\n            if (j == n) return i;\n        }\n        return -1;\n    }\n};\n```\n\n思路2：KMP算法\n- 时间复杂度：O(m+n)\n- 空间复杂度：O(n)\n- 思路：使用KMP算法的next数组优化匹配过程\n\n```cpp\nclass Solution {\npublic:\n    int strStr(string haystack, string needle) {\n        if (needle.empty()) return 0;\n        \n        vector<int> next = getNext(needle);\n        int i = 0, j = 0;\n        \n        while (i < haystack.length() && j < needle.length()) {\n            if (j == -1 || haystack[i] == needle[j]) {\n                i++;\n                j++;\n            } else {\n                j = next[j];\n            }\n        }\n        \n        return j == needle.length() ? i - j : -1;\n    }\n    \nprivate:\n    vector<int> getNext(string& pattern) {\n        int n = pattern.length();\n        vector<int> next(n, -1);\n        int i = 0, j = -1;\n        \n        while (i < n - 1) {\n            if (j == -1 || pattern[i] == pattern[j]) {\n                i++;\n                j++;\n                next[i] = j;\n            } else {\n                j = next[j];\n            }\n        }\n        \n        return next;\n    }\n};\n```",
            "字符串-KMP"
        )
        
        # 31. 下一个排列
        self.add_card(
            "LeetCode 31. 下一个排列\n\n整数数组的一个排列就是将其所有成员以序列或线性顺序排列。\n\n整数数组的下一个排列是指其整数的下一个字典序更大的排列。",
            "思路：两遍扫描\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：从右往左找到第一个降序位置，然后交换并反转\n\n```cpp\nclass Solution {\npublic:\n    void nextPermutation(vector<int>& nums) {\n        int i = nums.size() - 2;\n        while (i >= 0 && nums[i] >= nums[i + 1]) i--;\n        \n        if (i >= 0) {\n            int j = nums.size() - 1;\n            while (j > i && nums[j] <= nums[i]) j--;\n            swap(nums[i], nums[j]);\n        }\n        \n        reverse(nums.begin() + i + 1, nums.end());\n    }\n};\n```",
            "数组-数学"
        )
        
        # 32. 最长有效括号
        self.add_card(
            "LeetCode 32. 最长有效括号\n\n给你一个只包含'('和')'的字符串，找出最长有效（格式正确且连续）括号子串的长度。",
            "思路1：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：dp[i]表示以i结尾的最长有效括号长度\n\n```cpp\nclass Solution {\npublic:\n    int longestValidParentheses(string s) {\n        int n = s.length();\n        vector<int> dp(n, 0);\n        int maxLen = 0;\n        \n        for (int i = 1; i < n; i++) {\n            if (s[i] == ')') {\n                if (s[i - 1] == '(') {\n                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;\n                } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {\n                    dp[i] = dp[i - 1] + (i - dp[i - 1] >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;\n                }\n                maxLen = max(maxLen, dp[i]);\n            }\n        }\n        \n        return maxLen;\n    }\n};\n```\n\n思路2：栈\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用栈存储左括号的索引\n\n```cpp\nclass Solution {\npublic:\n    int longestValidParentheses(string s) {\n        stack<int> st;\n        st.push(-1);\n        int maxLen = 0;\n        \n        for (int i = 0; i < s.length(); i++) {\n            if (s[i] == '(') {\n                st.push(i);\n            } else {\n                st.pop();\n                if (st.empty()) {\n                    st.push(i);\n                } else {\n                    maxLen = max(maxLen, i - st.top());\n                }\n            }\n        }\n        \n        return maxLen;\n    }\n};\n```",
            "动态规划-栈"
        )
        
        # 40. 组合总和 II
        self.add_card(
            "LeetCode 40. 组合总和 II\n\n给定一个候选人编号的集合candidates和一个目标数target，找出candidates中所有可以使数字和为target的组合。\n\ncandidates中的每个数字在每个组合中只能使用一次。",
            "思路：回溯+去重\n- 时间复杂度：O(2^n)\n- 空间复杂度：O(target)\n- 思路：使用回溯算法，跳过重复元素避免重复组合\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {\n        sort(candidates.begin(), candidates.end());\n        vector<vector<int>> result;\n        vector<int> current;\n        backtrack(candidates, target, 0, current, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(vector<int>& candidates, int target, int start, vector<int>& current, vector<vector<int>>& result) {\n        if (target == 0) {\n            result.push_back(current);\n            return;\n        }\n        \n        for (int i = start; i < candidates.size(); i++) {\n            if (i > start && candidates[i] == candidates[i - 1]) continue;\n            if (candidates[i] > target) break;\n            \n            current.push_back(candidates[i]);\n            backtrack(candidates, target - candidates[i], i + 1, current, result);\n            current.pop_back();\n        }\n    }\n};\n```",
            "回溯"
        )
        
        # 41. 缺失的第一个正数
        self.add_card(
            "LeetCode 41. 缺失的第一个正数\n\n给你一个未排序的整数数组nums，请你找出其中没有出现的最小的正整数。",
            "思路：原地哈希\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：将数组本身作为哈希表，将数字i放到位置i-1\n\n```cpp\nclass Solution {\npublic:\n    int firstMissingPositive(vector<int>& nums) {\n        int n = nums.size();\n        \n        // 将数字放到正确位置\n        for (int i = 0; i < n; i++) {\n            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {\n                swap(nums[i], nums[nums[i] - 1]);\n            }\n        }\n        \n        // 找到第一个缺失的正数\n        for (int i = 0; i < n; i++) {\n            if (nums[i] != i + 1) {\n                return i + 1;\n            }\n        }\n        \n        return n + 1;\n    }\n};\n```",
            "数组-哈希表"
        )
        
        # 43. 字符串相乘
        self.add_card(
            "LeetCode 43. 字符串相乘\n\n给定两个以字符串形式表示的非负整数num1和num2，返回num1和num2的乘积，它们的乘积也表示为字符串形式。",
            "思路：模拟乘法\n- 时间复杂度：O(mn)\n- 空间复杂度：O(m+n)\n- 思路：模拟竖式乘法，逐位相乘并处理进位\n\n```cpp\nclass Solution {\npublic:\n    string multiply(string num1, string num2) {\n        if (num1 == \"0\" || num2 == \"0\") return \"0\";\n        \n        int m = num1.length(), n = num2.length();\n        vector<int> result(m + n, 0);\n        \n        for (int i = m - 1; i >= 0; i--) {\n            for (int j = n - 1; j >= 0; j--) {\n                int mul = (num1[i] - '0') * (num2[j] - '0');\n                int p1 = i + j, p2 = i + j + 1;\n                int sum = mul + result[p2];\n                \n                result[p2] = sum % 10;\n                result[p1] += sum / 10;\n            }\n        }\n        \n        string str;\n        int i = 0;\n        while (i < result.size() && result[i] == 0) i++;\n        for (; i < result.size(); i++) {\n            str += to_string(result[i]);\n        }\n        \n        return str;\n    }\n};\n```",
            "字符串-数学"
        )
        
        # 45. 跳跃游戏 II
        self.add_card(
            "LeetCode 45. 跳跃游戏 II\n\n给你一个非负整数数组nums，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。\n\n你的目标是使用最少的跳跃次数到达数组的最后一个位置。",
            "思路：贪心\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：每次选择能跳得最远的位置\n\n```cpp\nclass Solution {\npublic:\n    int jump(vector<int>& nums) {\n        int jumps = 0, end = 0, farthest = 0;\n        \n        for (int i = 0; i < nums.size() - 1; i++) {\n            farthest = max(farthest, i + nums[i]);\n            if (i == end) {\n                jumps++;\n                end = farthest;\n            }\n        }\n        \n        return jumps;\n    }\n};\n```",
            "贪心"
        )
        
        # 47. 全排列 II
        self.add_card(
            "LeetCode 47. 全排列 II\n\n给定一个可包含重复数字的序列nums，按任意顺序返回所有不重复的全排列。",
            "思路：回溯+去重\n- 时间复杂度：O(n×n!)\n- 空间复杂度：O(n)\n- 思路：使用回溯算法，通过排序和剪枝避免重复排列\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> permuteUnique(vector<int>& nums) {\n        sort(nums.begin(), nums.end());\n        vector<vector<int>> result;\n        vector<int> current;\n        vector<bool> visited(nums.size(), false);\n        backtrack(nums, current, visited, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(vector<int>& nums, vector<int>& current, vector<bool>& visited, vector<vector<int>>& result) {\n        if (current.size() == nums.size()) {\n            result.push_back(current);\n            return;\n        }\n        \n        for (int i = 0; i < nums.size(); i++) {\n            if (visited[i] || (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])) {\n                continue;\n            }\n            \n            visited[i] = true;\n            current.push_back(nums[i]);\n            backtrack(nums, current, visited, result);\n            current.pop_back();\n            visited[i] = false;\n        }\n    }\n};\n```",
            "回溯"
        )
        
        # 48. 旋转图像
        self.add_card(
            "LeetCode 48. 旋转图像\n\n给定一个n×n的二维矩阵matrix表示一个图像。请你将图像顺时针旋转90度。",
            "思路1：转置+反转\n- 时间复杂度：O(n²)\n- 空间复杂度：O(1)\n- 思路：先转置矩阵，然后反转每一行\n\n```cpp\nclass Solution {\npublic:\n    void rotate(vector<vector<int>>& matrix) {\n        int n = matrix.size();\n        \n        // 转置\n        for (int i = 0; i < n; i++) {\n            for (int j = i; j < n; j++) {\n                swap(matrix[i][j], matrix[j][i]);\n            }\n        }\n        \n        // 反转每一行\n        for (int i = 0; i < n; i++) {\n            reverse(matrix[i].begin(), matrix[i].end());\n        }\n    }\n};\n```\n\n思路2：原地旋转\n- 时间复杂度：O(n²)\n- 空间复杂度：O(1)\n- 思路：一次旋转四个位置\n\n```cpp\nclass Solution {\npublic:\n    void rotate(vector<vector<int>>& matrix) {\n        int n = matrix.size();\n        for (int i = 0; i < (n + 1) / 2; i++) {\n            for (int j = 0; j < n / 2; j++) {\n                int temp = matrix[i][j];\n                matrix[i][j] = matrix[n - 1 - j][i];\n                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];\n                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];\n                matrix[j][n - 1 - i] = temp;\n            }\n        }\n    }\n};\n```",
            "数组-矩阵"
        )
        
        # 54. 螺旋矩阵
        self.add_card(
            "LeetCode 54. 螺旋矩阵\n\n给你一个m行n列的矩阵matrix，请按照顺时针螺旋顺序，返回矩阵中的所有元素。",
            "思路：模拟\n- 时间复杂度：O(mn)\n- 空间复杂度：O(1)\n- 思路：按照右、下、左、上的顺序遍历，使用边界控制\n\n```cpp\nclass Solution {\npublic:\n    vector<int> spiralOrder(vector<vector<int>>& matrix) {\n        vector<int> result;\n        int top = 0, bottom = matrix.size() - 1;\n        int left = 0, right = matrix[0].size() - 1;\n        \n        while (top <= bottom && left <= right) {\n            // 右\n            for (int j = left; j <= right; j++) {\n                result.push_back(matrix[top][j]);\n            }\n            top++;\n            \n            // 下\n            for (int i = top; i <= bottom; i++) {\n                result.push_back(matrix[i][right]);\n            }\n            right--;\n            \n            // 左\n            if (top <= bottom) {\n                for (int j = right; j >= left; j--) {\n                    result.push_back(matrix[bottom][j]);\n                }\n                bottom--;\n            }\n            \n            // 上\n            if (left <= right) {\n                for (int i = bottom; i >= top; i--) {\n                    result.push_back(matrix[i][left]);\n                }\n                left++;\n            }\n        }\n        \n        return result;\n    }\n};\n```",
            "数组-矩阵"
        )
        
        # 57. 插入区间
        self.add_card(
            "LeetCode 57. 插入区间\n\n给你一个无重叠的，按照区间起始端点排序的区间列表。\n\n在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。",
            "思路：一次遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：找到插入位置，合并重叠区间\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {\n        vector<vector<int>> result;\n        int i = 0;\n        \n        // 添加所有在新区间之前的区间\n        while (i < intervals.size() && intervals[i][1] < newInterval[0]) {\n            result.push_back(intervals[i++]);\n        }\n        \n        // 合并重叠区间\n        while (i < intervals.size() && intervals[i][0] <= newInterval[1]) {\n            newInterval[0] = min(newInterval[0], intervals[i][0]);\n            newInterval[1] = max(newInterval[1], intervals[i][1]);\n            i++;\n        }\n        result.push_back(newInterval);\n        \n        // 添加剩余的区间\n        while (i < intervals.size()) {\n            result.push_back(intervals[i++]);\n        }\n        \n        return result;\n    }\n};\n```",
            "数组-贪心"
        )
        
        # 59. 螺旋矩阵 II
        self.add_card(
            "LeetCode 59. 螺旋矩阵 II\n\n给你一个正整数n，生成一个包含1到n²所有元素，且元素按顺时针顺序螺旋排列的n×n正方形矩阵matrix。",
            "思路：模拟\n- 时间复杂度：O(n²)\n- 空间复杂度：O(1)\n- 思路：按照右、下、左、上的顺序填充数字\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> generateMatrix(int n) {\n        vector<vector<int>> matrix(n, vector<int>(n, 0));\n        int num = 1;\n        int top = 0, bottom = n - 1, left = 0, right = n - 1;\n        \n        while (num <= n * n) {\n            // 右\n            for (int j = left; j <= right; j++) {\n                matrix[top][j] = num++;\n            }\n            top++;\n            \n            // 下\n            for (int i = top; i <= bottom; i++) {\n                matrix[i][right] = num++;\n            }\n            right--;\n            \n            // 左\n            for (int j = right; j >= left; j--) {\n                matrix[bottom][j] = num++;\n            }\n            bottom--;\n            \n            // 上\n            for (int i = bottom; i >= top; i--) {\n                matrix[i][left] = num++;\n            }\n            left++;\n        }\n        \n        return matrix;\n    }\n};\n```",
            "数组-矩阵"
        )
        
        # 72. 编辑距离
        self.add_card(
            "LeetCode 72. 编辑距离\n\n给你两个单词word1和word2，请返回将word1转换成word2所使用的最少操作数。",
            "思路：动态规划\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mn)\n- 思路：dp[i][j]表示word1的前i个字符转换为word2的前j个字符的最少操作数\n\n```cpp\nclass Solution {\npublic:\n    int minDistance(string word1, string word2) {\n        int m = word1.length(), n = word2.length();\n        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));\n        \n        for (int i = 0; i <= m; i++) dp[i][0] = i;\n        for (int j = 0; j <= n; j++) dp[0][j] = j;\n        \n        for (int i = 1; i <= m; i++) {\n            for (int j = 1; j <= n; j++) {\n                if (word1[i - 1] == word2[j - 1]) {\n                    dp[i][j] = dp[i - 1][j - 1];\n                } else {\n                    dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;\n                }\n            }\n        }\n        \n        return dp[m][n];\n    }\n};\n```",
            "动态规划-字符串"
        )
        
        # 74. 搜索二维矩阵
        self.add_card(
            "LeetCode 74. 搜索二维矩阵\n\n编写一个高效的算法来判断m×n矩阵中，是否存在一个目标值。该矩阵具有如下特性：每行中的整数从左到右按升序排列；每行的第一个整数大于前一行的最后一个整数。",
            "思路：二分查找\n- 时间复杂度：O(log(mn))\n- 空间复杂度：O(1)\n- 思路：将二维矩阵视为一维数组进行二分查找\n\n```cpp\nclass Solution {\npublic:\n    bool searchMatrix(vector<vector<int>>& matrix, int target) {\n        int m = matrix.size(), n = matrix[0].size();\n        int left = 0, right = m * n - 1;\n        \n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            int row = mid / n, col = mid % n;\n            \n            if (matrix[row][col] == target) {\n                return true;\n            } else if (matrix[row][col] < target) {\n                left = mid + 1;\n            } else {\n                right = mid - 1;\n            }\n        }\n        \n        return false;\n    }\n};\n```",
            "数组-二分查找"
        )
        
        # 76. 最小覆盖子串
        self.add_card(
            "LeetCode 76. 最小覆盖子串\n\n给你一个字符串s、一个字符串t。返回s中涵盖t所有字符的最小子串。如果s中不存在涵盖t所有字符的子串，则返回空字符串\"\"。",
            "思路：滑动窗口\n- 时间复杂度：O(m+n)\n- 空间复杂度：O(m)\n- 思路：使用滑动窗口，维护一个包含t中所有字符的窗口\n\n```cpp\nclass Solution {\npublic:\n    string minWindow(string s, string t) {\n        unordered_map<char, int> need, window;\n        for (char c : t) need[c]++;\n        \n        int left = 0, right = 0;\n        int valid = 0;\n        int start = 0, len = INT_MAX;\n        \n        while (right < s.length()) {\n            char c = s[right++];\n            if (need.count(c)) {\n                window[c]++;\n                if (window[c] == need[c]) valid++;\n            }\n            \n            while (valid == need.size()) {\n                if (right - left < len) {\n                    start = left;\n                    len = right - left;\n                }\n                \n                char d = s[left++];\n                if (need.count(d)) {\n                    if (window[d] == need[d]) valid--;\n                    window[d]--;\n                }\n            }\n        }\n        \n        return len == INT_MAX ? \"\" : s.substr(start, len);\n    }\n};\n```",
            "字符串-滑动窗口"
        )
        
        # 80. 删除有序数组中的重复项 II
        self.add_card(
            "LeetCode 80. 删除有序数组中的重复项 II\n\n给你一个有序数组nums，请你原地删除重复出现的元素，使得出现次数超过两次的元素只出现两次，返回删除后数组的新长度。",
            "思路：双指针\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用快慢指针，允许每个元素最多出现两次\n\n```cpp\nclass Solution {\npublic:\n    int removeDuplicates(vector<int>& nums) {\n        if (nums.size() <= 2) return nums.size();\n        \n        int slow = 2;\n        for (int fast = 2; fast < nums.size(); fast++) {\n            if (nums[fast] != nums[slow - 2]) {\n                nums[slow++] = nums[fast];\n            }\n        }\n        \n        return slow;\n    }\n};\n```",
            "数组-双指针"
        )
        
        # 81. 搜索旋转排序数组 II
        self.add_card(
            "LeetCode 81. 搜索旋转排序数组 II\n\n已知存在一个按非降序排列的整数数组nums，数组中的值不必互不相同。\n\n在传递给函数之前，nums在预先未知的某个下标k（0<=k<nums.length）上进行了旋转。给你旋转后的数组nums和一个整数target，请你编写一个函数来判断给定的目标值是否存在于数组中。",
            "思路：二分查找\n- 时间复杂度：O(n)最坏情况\n- 空间复杂度：O(1)\n- 思路：处理重复元素的情况，当nums[left]==nums[mid]时，left++跳过重复\n\n```cpp\nclass Solution {\npublic:\n    bool search(vector<int>& nums, int target) {\n        int left = 0, right = nums.size() - 1;\n        \n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            if (nums[mid] == target) return true;\n            \n            if (nums[left] == nums[mid] && nums[mid] == nums[right]) {\n                left++;\n                right--;\n            } else if (nums[left] <= nums[mid]) {\n                if (nums[left] <= target && target < nums[mid]) {\n                    right = mid - 1;\n                } else {\n                    left = mid + 1;\n                }\n            } else {\n                if (nums[mid] < target && target <= nums[right]) {\n                    left = mid + 1;\n                } else {\n                    right = mid - 1;\n                }\n            }\n        }\n        \n        return false;\n    }\n};\n```",
            "数组-二分查找"
        )
        
        # 82. 删除排序链表中的重复元素 II
        self.add_card(
            "LeetCode 82. 删除排序链表中的重复元素 II\n\n给定一个已排序的链表的头head，删除所有重复的元素，使每个元素只出现一次。返回已排序的链表。",
            "思路：一次遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用虚拟头节点，跳过所有重复的节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* deleteDuplicates(ListNode* head) {\n        ListNode* dummy = new ListNode(0);\n        dummy->next = head;\n        ListNode* prev = dummy;\n        \n        while (head) {\n            if (head->next && head->val == head->next->val) {\n                while (head->next && head->val == head->next->val) {\n                    head = head->next;\n                }\n                prev->next = head->next;\n            } else {\n                prev = prev->next;\n            }\n            head = head->next;\n        }\n        \n        return dummy->next;\n    }\n};\n```",
            "链表"
        )
        
        # 84. 柱状图中最大的矩形
        self.add_card(
            "LeetCode 84. 柱状图中最大的矩形\n\n给定n个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为1。\n\n求在该柱状图中，能够勾勒出来的矩形的最大面积。",
            "思路：单调栈\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：使用单调递增栈，找到每个柱子左右两边第一个比它矮的柱子\n\n```cpp\nclass Solution {\npublic:\n    int largestRectangleArea(vector<int>& heights) {\n        stack<int> st;\n        int maxArea = 0;\n        \n        for (int i = 0; i <= heights.size(); i++) {\n            int h = (i == heights.size()) ? 0 : heights[i];\n            \n            while (!st.empty() && heights[st.top()] > h) {\n                int height = heights[st.top()];\n                st.pop();\n                int width = st.empty() ? i : i - st.top() - 1;\n                maxArea = max(maxArea, height * width);\n            }\n            \n            st.push(i);\n        }\n        \n        return maxArea;\n    }\n};\n```",
            "栈-单调栈"
        )
        
        # 85. 最大矩形
        self.add_card(
            "LeetCode 85. 最大矩形\n\n给定一个仅包含0和1、大小为rows×cols的二维二进制矩阵，找出只包含1的最大矩形，并返回其面积。",
            "思路：单调栈\n- 时间复杂度：O(mn)\n- 空间复杂度：O(n)\n- 思路：将问题转化为多个柱状图最大矩形问题\n\n```cpp\nclass Solution {\npublic:\n    int maximalRectangle(vector<vector<char>>& matrix) {\n        if (matrix.empty()) return 0;\n        \n        int m = matrix.size(), n = matrix[0].size();\n        vector<int> heights(n, 0);\n        int maxArea = 0;\n        \n        for (int i = 0; i < m; i++) {\n            for (int j = 0; j < n; j++) {\n                heights[j] = (matrix[i][j] == '1') ? heights[j] + 1 : 0;\n            }\n            maxArea = max(maxArea, largestRectangleArea(heights));\n        }\n        \n        return maxArea;\n    }\n    \nprivate:\n    int largestRectangleArea(vector<int>& heights) {\n        stack<int> st;\n        int maxArea = 0;\n        \n        for (int i = 0; i <= heights.size(); i++) {\n            int h = (i == heights.size()) ? 0 : heights[i];\n            while (!st.empty() && heights[st.top()] > h) {\n                int height = heights[st.top()];\n                st.pop();\n                int width = st.empty() ? i : i - st.top() - 1;\n                maxArea = max(maxArea, height * width);\n            }\n            st.push(i);\n        }\n        \n        return maxArea;\n    }\n};\n```",
            "栈-动态规划"
        )
        
        # 86. 分隔链表
        self.add_card(
            "LeetCode 86. 分隔链表\n\n给你一个链表的头节点head和一个特定值x，请你对链表进行分隔，使得所有小于x的节点都出现在大于或等于x的节点之前。",
            "思路：双链表\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：创建两个链表，分别存储小于x和大于等于x的节点\n\n```cpp\nclass Solution {\npublic:\n    ListNode* partition(ListNode* head, int x) {\n        ListNode* small = new ListNode(0);\n        ListNode* large = new ListNode(0);\n        ListNode* smallHead = small;\n        ListNode* largeHead = large;\n        \n        while (head) {\n            if (head->val < x) {\n                small->next = head;\n                small = small->next;\n            } else {\n                large->next = head;\n                large = large->next;\n            }\n            head = head->next;\n        }\n        \n        large->next = nullptr;\n        small->next = largeHead->next;\n        \n        return smallHead->next;\n    }\n};\n```",
            "链表"
        )
        
        # 90. 子集 II
        self.add_card(
            "LeetCode 90. 子集 II\n\n给你一个整数数组nums，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。",
            "思路：回溯+去重\n- 时间复杂度：O(n×2^n)\n- 空间复杂度：O(n)\n- 思路：使用回溯算法，通过排序和剪枝避免重复子集\n\n```cpp\nclass Solution {\npublic:\n    vector<vector<int>> subsetsWithDup(vector<int>& nums) {\n        sort(nums.begin(), nums.end());\n        vector<vector<int>> result;\n        vector<int> current;\n        backtrack(nums, 0, current, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(vector<int>& nums, int start, vector<int>& current, vector<vector<int>>& result) {\n        result.push_back(current);\n        \n        for (int i = start; i < nums.size(); i++) {\n            if (i > start && nums[i] == nums[i - 1]) continue;\n            current.push_back(nums[i]);\n            backtrack(nums, i + 1, current, result);\n            current.pop_back();\n        }\n    }\n};\n```",
            "回溯"
        )
        
        # 91. 解码方法
        self.add_card(
            "LeetCode 91. 解码方法\n\n一条包含字母A-Z的消息通过以下映射进行了编码。给你一个只含数字的非空字符串s，请计算并返回解码方法的总数。",
            "思路：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：dp[i]表示前i个字符的解码方法数，考虑单个字符和两个字符的情况\n\n```cpp\nclass Solution {\npublic:\n    int numDecodings(string s) {\n        if (s[0] == '0') return 0;\n        \n        int prev = 1, curr = 1;\n        \n        for (int i = 1; i < s.length(); i++) {\n            int temp = curr;\n            if (s[i] == '0') {\n                if (s[i - 1] == '1' || s[i - 1] == '2') {\n                    curr = prev;\n                } else {\n                    return 0;\n                }\n            } else if (s[i - 1] == '1' || (s[i - 1] == '2' && s[i] <= '6')) {\n                curr = prev + curr;\n            }\n            prev = temp;\n        }\n        \n        return curr;\n    }\n};\n```",
            "动态规划-字符串"
        )
        
        # 93. 复原IP地址
        self.add_card(
            "LeetCode 93. 复原IP地址\n\n有效IP地址正好由四个整数（每个整数位于0到255之间组成，且不能含有前导0），整数之间用'.'分隔。\n\n给定一个只包含数字的字符串s，用以表示一个IP地址，返回所有可能的有效IP地址。",
            "思路：回溯\n- 时间复杂度：O(1)\n- 空间复杂度：O(1)\n- 思路：使用回溯算法，将字符串分成四段，每段验证是否有效\n\n```cpp\nclass Solution {\npublic:\n    vector<string> restoreIpAddresses(string s) {\n        vector<string> result;\n        vector<string> current;\n        backtrack(s, 0, current, result);\n        return result;\n    }\n    \nprivate:\n    void backtrack(string& s, int start, vector<string>& current, vector<string>& result) {\n        if (current.size() == 4) {\n            if (start == s.length()) {\n                result.push_back(current[0] + \".\" + current[1] + \".\" + current[2] + \".\" + current[3]);\n            }\n            return;\n        }\n        \n        for (int len = 1; len <= 3 && start + len <= s.length(); len++) {\n            string segment = s.substr(start, len);\n            if (isValid(segment)) {\n                current.push_back(segment);\n                backtrack(s, start + len, current, result);\n                current.pop_back();\n            }\n        }\n    }\n    \n    bool isValid(string& segment) {\n        if (segment.length() > 1 && segment[0] == '0') return false;\n        int num = stoi(segment);\n        return num >= 0 && num <= 255;\n    }\n};\n```",
            "回溯-字符串"
        )
        
        # 95. 不同的二叉搜索树 II
        self.add_card(
            "LeetCode 95. 不同的二叉搜索树 II\n\n给你一个整数n，请你生成并返回所有由n个节点组成且节点值从1到n互不相同的不同二叉搜索树。",
            "思路：递归（分治）\n- 时间复杂度：O(4^n/√n)\n- 空间复杂度：O(4^n/√n)\n- 思路：对于每个根节点，递归生成左右子树的所有可能组合\n\n```cpp\nclass Solution {\npublic:\n    vector<TreeNode*> generateTrees(int n) {\n        return generate(1, n);\n    }\n    \nprivate:\n    vector<TreeNode*> generate(int start, int end) {\n        vector<TreeNode*> result;\n        if (start > end) {\n            result.push_back(nullptr);\n            return result;\n        }\n        \n        for (int i = start; i <= end; i++) {\n            vector<TreeNode*> leftTrees = generate(start, i - 1);\n            vector<TreeNode*> rightTrees = generate(i + 1, end);\n            \n            for (TreeNode* left : leftTrees) {\n                for (TreeNode* right : rightTrees) {\n                    TreeNode* root = new TreeNode(i);\n                    root->left = left;\n                    root->right = right;\n                    result.push_back(root);\n                }\n            }\n        }\n        \n        return result;\n    }\n};\n```",
            "树-递归"
        )
        
        # 97. 交错字符串
        self.add_card(
            "LeetCode 97. 交错字符串\n\n给定三个字符串s1、s2、s3，请你帮忙验证s3是否是由s1和s2交错组成的。",
            "思路：动态规划\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mn)\n- 思路：dp[i][j]表示s1的前i个字符和s2的前j个字符能否组成s3的前i+j个字符\n\n```cpp\nclass Solution {\npublic:\n    bool isInterleave(string s1, string s2, string s3) {\n        int m = s1.length(), n = s2.length();\n        if (m + n != s3.length()) return false;\n        \n        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));\n        dp[0][0] = true;\n        \n        for (int i = 0; i <= m; i++) {\n            for (int j = 0; j <= n; j++) {\n                if (i > 0) {\n                    dp[i][j] = dp[i][j] || (dp[i - 1][j] && s1[i - 1] == s3[i + j - 1]);\n                }\n                if (j > 0) {\n                    dp[i][j] = dp[i][j] || (dp[i][j - 1] && s2[j - 1] == s3[i + j - 1]);\n                }\n            }\n        }\n        \n        return dp[m][n];\n    }\n};\n```",
            "动态规划-字符串"
        )
        
        # 99. 恢复二叉搜索树
        self.add_card(
            "LeetCode 99. 恢复二叉搜索树\n\n给你二叉搜索树的根节点root，该树中的恰好两个节点的值被错误地交换。请在不改变其结构的情况下，恢复这棵树。",
            "思路：中序遍历\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：中序遍历BST应该是递增的，找到两个位置错误的节点并交换\n\n```cpp\nclass Solution {\nprivate:\n    TreeNode* first = nullptr;\n    TreeNode* second = nullptr;\n    TreeNode* prev = new TreeNode(INT_MIN);\n    \npublic:\n    void recoverTree(TreeNode* root) {\n        inorder(root);\n        swap(first->val, second->val);\n    }\n    \nprivate:\n    void inorder(TreeNode* root) {\n        if (!root) return;\n        \n        inorder(root->left);\n        \n        if (prev->val > root->val) {\n            if (!first) first = prev;\n            second = root;\n        }\n        prev = root;\n        \n        inorder(root->right);\n    }\n};\n```",
            "树-二叉搜索树"
        )
        
        # 106. 从中序与后序遍历序列构造二叉树
        self.add_card(
            "LeetCode 106. 从中序与后序遍历序列构造二叉树\n\n给定两个整数数组inorder和postorder，其中inorder是二叉树的中序遍历，postorder是同一棵树的后序遍历，请你构造并返回这颗二叉树。",
            "思路：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：后序遍历的最后一个元素是根节点，在中序遍历中找到根节点，递归构建左右子树\n\n```cpp\nclass Solution {\npublic:\n    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {\n        unordered_map<int, int> map;\n        for (int i = 0; i < inorder.size(); i++) {\n            map[inorder[i]] = i;\n        }\n        \n        return build(inorder, 0, inorder.size() - 1, \n                     postorder, 0, postorder.size() - 1, map);\n    }\n    \nprivate:\n    TreeNode* build(vector<int>& inorder, int inStart, int inEnd,\n                   vector<int>& postorder, int postStart, int postEnd,\n                   unordered_map<int, int>& map) {\n        if (inStart > inEnd || postStart > postEnd) return nullptr;\n        \n        int rootVal = postorder[postEnd];\n        TreeNode* root = new TreeNode(rootVal);\n        \n        int rootIndex = map[rootVal];\n        int leftSize = rootIndex - inStart;\n        \n        root->left = build(inorder, inStart, rootIndex - 1,\n                          postorder, postStart, postStart + leftSize - 1, map);\n        root->right = build(inorder, rootIndex + 1, inEnd,\n                           postorder, postStart + leftSize, postEnd - 1, map);\n        \n        return root;\n    }\n};\n```",
            "树-二叉树"
        )
        
        # 109. 有序链表转换二叉搜索树
        self.add_card(
            "LeetCode 109. 有序链表转换二叉搜索树\n\n给定一个单链表的头节点head，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。",
            "思路：快慢指针+递归\n- 时间复杂度：O(nlogn)\n- 空间复杂度：O(logn)\n- 思路：使用快慢指针找到链表中点作为根节点，递归构建左右子树\n\n```cpp\nclass Solution {\npublic:\n    TreeNode* sortedListToBST(ListNode* head) {\n        if (!head) return nullptr;\n        if (!head->next) return new TreeNode(head->val);\n        \n        ListNode* slow = head, *fast = head, *prev = nullptr;\n        while (fast && fast->next) {\n            prev = slow;\n            slow = slow->next;\n            fast = fast->next->next;\n        }\n        \n        prev->next = nullptr;\n        TreeNode* root = new TreeNode(slow->val);\n        root->left = sortedListToBST(head);\n        root->right = sortedListToBST(slow->next);\n        \n        return root;\n    }\n};\n```",
            "树-链表"
        )
        
        # 114. 二叉树展开为链表
        self.add_card(
            "LeetCode 114. 二叉树展开为链表\n\n给你二叉树的根结点root，请你将它展开为一个单链表。展开后的单链表应该同样使用TreeNode，其中right子指针指向链表中下一个结点，而左子指针始终为null。",
            "思路：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：递归处理左右子树，然后将左子树接到右子树位置，原右子树接到左子树最右节点\n\n```cpp\nclass Solution {\npublic:\n    void flatten(TreeNode* root) {\n        if (!root) return;\n        \n        flatten(root->left);\n        flatten(root->right);\n        \n        TreeNode* left = root->left;\n        TreeNode* right = root->right;\n        \n        root->left = nullptr;\n        root->right = left;\n        \n        TreeNode* curr = root;\n        while (curr->right) {\n            curr = curr->right;\n        }\n        curr->right = right;\n    }\n};\n```",
            "树-链表"
        )
        
        # 123. 买卖股票的最佳时机 III
        self.add_card(
            "LeetCode 123. 买卖股票的最佳时机 III\n\n给定一个数组，它的第i个元素是一支给定的股票在第i天的价格。\n\n设计一个算法来计算你所能获取的最大利润。你最多可以完成两笔交易。",
            "思路：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：使用四个状态变量表示第一次买入、第一次卖出、第二次买入、第二次卖出的最大利润\n\n```cpp\nclass Solution {\npublic:\n    int maxProfit(vector<int>& prices) {\n        int buy1 = INT_MIN, sell1 = 0;\n        int buy2 = INT_MIN, sell2 = 0;\n        \n        for (int price : prices) {\n            buy1 = max(buy1, -price);\n            sell1 = max(sell1, buy1 + price);\n            buy2 = max(buy2, sell1 - price);\n            sell2 = max(sell2, buy2 + price);\n        }\n        \n        return sell2;\n    }\n};\n```",
            "动态规划-股票"
        )
        
        # 124. 二叉树中的最大路径和
        self.add_card(
            "LeetCode 124. 二叉树中的最大路径和\n\n路径被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中至多出现一次。该路径至少包含一个节点，且不一定经过根节点。\n\n路径和是路径中各节点值的总和。给你一个二叉树的根节点root，返回其最大路径和。",
            "思路：递归\n- 时间复杂度：O(n)\n- 空间复杂度：O(n)\n- 思路：对于每个节点，计算经过该节点的最大路径和，同时返回以该节点为端点的最大路径和\n\n```cpp\nclass Solution {\nprivate:\n    int maxSum = INT_MIN;\n    \npublic:\n    int maxPathSum(TreeNode* root) {\n        maxGain(root);\n        return maxSum;\n    }\n    \nprivate:\n    int maxGain(TreeNode* root) {\n        if (!root) return 0;\n        \n        int leftGain = max(maxGain(root->left), 0);\n        int rightGain = max(maxGain(root->right), 0);\n        \n        int pathSum = root->val + leftGain + rightGain;\n        maxSum = max(maxSum, pathSum);\n        \n        return root->val + max(leftGain, rightGain);\n    }\n};\n```",
            "树-递归"
        )
        
        # 213. 打家劫舍 II
        self.add_card(
            "LeetCode 213. 打家劫舍 II\n\n你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋围成一圈，这意味着第一个房屋和最后一个房屋是紧挨着的。",
            "思路：动态规划\n- 时间复杂度：O(n)\n- 空间复杂度：O(1)\n- 思路：将问题分解为两种情况：不偷第一间或不偷最后一间\n\n```cpp\nclass Solution {\npublic:\n    int rob(vector<int>& nums) {\n        if (nums.size() == 1) return nums[0];\n        return max(robRange(nums, 0, nums.size() - 2), \n                   robRange(nums, 1, nums.size() - 1));\n    }\n    \nprivate:\n    int robRange(vector<int>& nums, int start, int end) {\n        int prev = 0, curr = 0;\n        for (int i = start; i <= end; i++) {\n            int temp = curr;\n            curr = max(curr, prev + nums[i]);\n            prev = temp;\n        }\n        return curr;\n    }\n};\n```",
            "动态规划"
        )
        
        # 221. 最大正方形
        self.add_card(
            "LeetCode 221. 最大正方形\n\n在一个由'0'和'1'组成的二维矩阵内，找到只包含'1'的最大正方形，并返回其面积。",
            "思路：动态规划\n- 时间复杂度：O(mn)\n- 空间复杂度：O(mn)\n- 思路：dp[i][j]表示以(i,j)为右下角的最大正方形边长\n\n```cpp\nclass Solution {\npublic:\n    int maximalSquare(vector<vector<char>>& matrix) {\n        int m = matrix.size(), n = matrix[0].size();\n        vector<vector<int>> dp(m, vector<int>(n, 0));\n        int maxSide = 0;\n        \n        for (int i = 0; i < m; i++) {\n            for (int j = 0; j < n; j++) {\n                if (matrix[i][j] == '1') {\n                    if (i == 0 || j == 0) {\n                        dp[i][j] = 1;\n                    } else {\n                        dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;\n                    }\n                    maxSide = max(maxSide, dp[i][j]);\n                }\n            }\n        }\n        \n        return maxSide * maxSide;\n    }\n};\n```",
            "动态规划-矩阵"
        )
        
        # 279. 完全平方数
        self.add_card(
            "LeetCode 279. 完全平方数\n\n给你一个整数n，返回和为n的完全平方数的最少数量。",
            "思路1：动态规划\n- 时间复杂度：O(n√n)\n- 空间复杂度：O(n)\n- 思路：dp[i]表示和为i的完全平方数的最少数量\n\n```cpp\nclass Solution {\npublic:\n    int numSquares(int n) {\n        vector<int> dp(n + 1, INT_MAX);\n        dp[0] = 0;\n        \n        for (int i = 1; i <= n; i++) {\n            for (int j = 1; j * j <= i; j++) {\n                dp[i] = min(dp[i], dp[i - j * j] + 1);\n            }\n        }\n        \n        return dp[n];\n    }\n};\n```\n\n思路2：数学（四平方定理）\n- 时间复杂度：O(√n)\n- 空间复杂度：O(1)\n- 思路：根据四平方定理，任何正整数都可以表示为4个整数的平方和\n\n```cpp\nclass Solution {\npublic:\n    int numSquares(int n) {\n        // 检查是否为完全平方数\n        if (isSquare(n)) return 1;\n        \n        // 检查是否满足4^a(8b+7)的形式\n        int temp = n;\n        while (temp % 4 == 0) temp /= 4;\n        if (temp % 8 == 7) return 4;\n        \n        // 检查是否可以表示为两个平方数的和\n        for (int i = 1; i * i <= n; i++) {\n            if (isSquare(n - i * i)) return 2;\n        }\n        \n        return 3;\n    }\n    \nprivate:\n    bool isSquare(int n) {\n        int root = sqrt(n);\n        return root * root == n;\n    }\n};\n```",
            "动态规划-数学"
        )
    
    def generate_anki_csv(self, filename: str = "leetcode_hot100_anki.csv"):
        """生成Anki格式的CSV文件"""
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            # 使用逗号分隔，与work_general_knowledge_generator.py保持一致
            # csv.writer会自动处理包含换行符、引号等特殊字符的字段
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            # Anki CSV格式：Front, Back, Tags
            for card in self.cards:
                # 将问题和答案格式化为HTML，设置左对齐样式
                question_html = self.format_text_for_anki(card['question'])
                answer_html = self.format_text_for_anki(card['answer'])
                writer.writerow([
                    question_html,
                    answer_html,
                    card['category']
                ])
        print(f"已生成Anki CSV文件：{filename}")
        print(f"共 {len(self.cards)} 张卡片")
    
    def generate_markdown(self, filename: str = "leetcode_hot100.md"):
        """生成Markdown格式的卡片文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# LeetCode Hot 100 记忆卡片\n\n")
            f.write(f"共 {len(self.cards)} 张卡片\n\n")
            f.write("---\n\n")
            
            # 按分类组织
            categories = {}
            for card in self.cards:
                cat = card['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(card)
            
            # 按主分类分组
            main_categories = {}
            for category, cards in categories.items():
                # 解析分类：主分类-子分类 或 主分类
                if '-' in category:
                    main_cat, sub_cat = category.split('-', 1)
                else:
                    main_cat = category
                    sub_cat = None
                
                if main_cat not in main_categories:
                    main_categories[main_cat] = {}
                if sub_cat:
                    main_categories[main_cat][sub_cat] = cards
                else:
                    main_categories[main_cat][''] = cards
            
            # 主分类排序顺序
            category_order = [
                '数组', '链表', '字符串', '树', '二叉树', 
                '动态规划', '回溯', 'DFS', '贪心', '双指针',
                '哈希表', '栈', '队列', '其他'
            ]
            
            # 按顺序输出
            for main_cat in category_order:
                if main_cat not in main_categories:
                    continue
                
                f.write(f"## {main_cat}\n\n")
                
                # 输出子分类
                for sub_cat in sorted(main_categories[main_cat].keys()):
                    cards = main_categories[main_cat][sub_cat]
                    if sub_cat:
                        f.write(f"### {sub_cat}\n\n")
                    
                    for i, card in enumerate(cards, 1):
                        f.write(f"#### 卡片 {i}\n\n")
                        f.write(f"**问题**：{card['question']}\n\n")
                        f.write(f"**答案**：\n\n{card['answer']}\n\n")
                        f.write("---\n\n")
                
                f.write("\n")
            
            # 输出其他未在顺序中的分类
            for main_cat in sorted(main_categories.keys()):
                if main_cat in category_order:
                    continue
                
                f.write(f"## {main_cat}\n\n")
                for sub_cat in sorted(main_categories[main_cat].keys()):
                    cards = main_categories[main_cat][sub_cat]
                    if sub_cat:
                        f.write(f"### {sub_cat}\n\n")
                    
                    for i, card in enumerate(cards, 1):
                        f.write(f"#### 卡片 {i}\n\n")
                        f.write(f"**问题**：{card['question']}\n\n")
                        f.write(f"**答案**：\n\n{card['answer']}\n\n")
                        f.write("---\n\n")
                
                f.write("\n")
        
        print(f"已生成Markdown文件：{filename}")
        print(f"共 {len(self.cards)} 张卡片")
    
    def generate_all(self):
        """生成所有卡片"""
        print("正在生成LeetCode Hot 100记忆卡片...")
        
        # 生成所有题目卡片
        self.generate_leetcode_cards()
        
        # 生成各种格式的文件
        self.generate_anki_csv()
        self.generate_markdown()
        
        # 统计信息
        categories = {}
        for card in self.cards:
            cat = card['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n卡片分类统计：")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} 张")
        
        print(f"\n总计：{len(self.cards)} 张卡片")

if __name__ == "__main__":
    generator = LeetCodeHot100Generator()
    generator.generate_all()
