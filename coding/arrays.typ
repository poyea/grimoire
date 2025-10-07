= Arrays

== Product of Array Except Self

*Problem:* Return array where `result[i] = product of all elements except nums[i]`. No division.

*Optimal - Prefix/Suffix:* $O(n)$ time, $O(1)$ space

```cpp
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, 1);
    // Build prefix products
    for (int i = 1; i < n; i++) {
        res[i] = res[i-1] * nums[i-1];
    }
    // Build suffix products and multiply
    int suffix = 1;
    for (int i = n-1; i >= 0; i--) {
        res[i] *= suffix;
        suffix *= nums[i];
    }
    return res;
}
```

*Key insight:* `res[i] = prefix[i] * suffix[i]`. Reuse output array for prefix, track suffix with scalar.

== String Encode and Decode

*Problem:* Serialize/deserialize `vector<string>` to single string. Handle delimiters in strings.

*Length Prefix:* $O(n)$ time, $O(1)$ extra space

```cpp
class Codec {
public:
    string encode(vector<string>& strs) {
        string result;
        for (const auto& s : strs) {
            result += to_string(s.length()) + "#" + s;
        }
        return result;
    }
    vector<string> decode(string s) {
        vector<string> result;
        size_t i = 0;

        while (i < s.length()) {
            size_t hash_pos = s.find('#', i);
            int len = stoi(s.substr(i, hash_pos - i));
            result.push_back(s.substr(hash_pos + 1, len));
            i = hash_pos + 1 + len;
        }
        return result;
    }
};
```

*Why it works:* Length prefix avoids delimiter conflicts. Format: `"4#word5#hello"`
