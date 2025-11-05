= Math & Number Theory

*Hardware foundation:* Integer arithmetic (ADD, MUL) = 1-3 cycles latency [Intel Opt. Manual 2023]. Division (DIV) = 10-40 cycles depending on operands. Modular reduction dominates many algorithms.

== Sieve of Eratosthenes

*Problem:* Find all primes up to n.

*Basic Sieve:* $O(n log log n)$ time, $O(n)$ space

```cpp
vector<bool> sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime[j] = false;
            }
        }
    }

    return is_prime;
}
```

*Cache optimization - Segmented Sieve:*

```cpp
const int L1_CACHE_SIZE = 32768;  // 32KB typical
const int SEG_SIZE = L1_CACHE_SIZE * 8;  // 256K bits = 32KB

vector<bool> segmented_sieve(int n) {
    int limit = sqrt(n);
    vector<bool> small_primes = sieve(limit);
    vector<int> primes;

    for (int i = 2; i <= limit; i++) {
        if (small_primes[i]) primes.push_back(i);
    }

    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (int low = 0; low <= n; low += SEG_SIZE) {
        fill(segment.begin(), segment.end(), true);
        int high = min(low + SEG_SIZE - 1, n);

        for (int p : primes) {
            int start = max(p * p, (low + p - 1) / p * p);
            for (int j = start; j <= high; j += p) {
                is_prime[j] = false;
            }
        }
    }

    return is_prime;
}
```

*Performance:* Segment fits in L1 cache = 3-5x speedup for large n (n > 10^7).

*Bit-packed version:*

```cpp
// Store 8 bools per byte
vector<uint8_t> sieve_packed(int n) {
    vector<uint8_t> is_prime((n + 7) / 8, 0xFF);  // All 1s initially

    auto set_composite = [&](int k) {
        is_prime[k >> 3] &= ~(1 << (k & 7));
    };

    auto is_prime_check = [&](int k) {
        return (is_prime[k >> 3] >> (k & 7)) & 1;
    };

    set_composite(0);
    set_composite(1);

    for (int i = 2; i * i <= n; i++) {
        if (is_prime_check(i)) {
            for (int j = i * i; j <= n; j += i) {
                set_composite(j);
            }
        }
    }

    return is_prime;
}
```

8x memory reduction = better cache utilization.

*Wheel factorization:* Skip multiples of 2, 3, 5 to reduce iterations by 77%.

== GCD (Greatest Common Divisor)

*Euclidean Algorithm:* $O(log min(a, b))$

```cpp
int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}
```

*Modulo cost:* DIV instruction = 10-40 cycles [Intel Opt. Manual 2023, IDIV].

*Binary GCD (Stein's Algorithm):* Avoids division, uses shifts.

```cpp
int binary_gcd(int a, int b) {
    if (a == 0) return b;
    if (b == 0) return a;

    // Count common factors of 2
    int shift = __builtin_ctz(a | b);  // Trailing zeros in a|b

    a >>= __builtin_ctz(a);  // Remove factors of 2 from a

    while (b != 0) {
        b >>= __builtin_ctz(b);  // Remove factors of 2 from b

        if (a > b) swap(a, b);

        b -= a;
    }

    return a << shift;
}
```

*Performance:* 2-3x faster than Euclidean for large numbers (64-bit). Shift and subtract = 1 cycle each.

*C++17 builtin:* `std::gcd(a, b)` - compiler may optimize to binary GCD.

== Modular Arithmetic

*Problem:* Compute $(a times b) mod m$ without overflow.

*Naive:* `(a * b) % m` - overflows if $a times b > 2^(64)$.

*Montgomery Multiplication:* Efficient modular reduction for repeated operations.

```cpp
class Montgomery {
    uint64_t mod, r, r2, mask, inv;

    uint64_t reduce(uint128_t t) {
        uint64_t m = (uint64_t(t) * inv) & mask;
        uint64_t u = (t + uint128_t(m) * mod) >> 64;
        return u >= mod ? u - mod : u;
    }

public:
    Montgomery(uint64_t m) : mod(m) {
        // Precompute constants
        r = 1ULL << 64;  // Conceptual
        mask = ~0ULL;
        inv = modInverse(mod);
        r2 = (uint128_t(r) % mod * r) % mod;
    }

    uint64_t mul(uint64_t a, uint64_t b) {
        return reduce(uint128_t(a) * b);
    }
};
```

*Speedup:* 3-5x faster for modular exponentiation (many multiplications).

*Barrett Reduction:* Precompute $mu = floor(2^(2k) / m)$ for fast division approximation.

```cpp
uint64_t barrett_reduce(uint64_t a, uint64_t mod, uint64_t mu) {
    uint128_t q = (uint128_t(a) * mu) >> 64;
    uint64_t r = a - q * mod;
    return r >= mod ? r - mod : r;
}
```

== Fast Exponentiation

*Problem:* Compute $a^b mod m$ efficiently.

*Binary Exponentiation:* $O(log b)$

```cpp
int64_t power_mod(int64_t a, int64_t b, int64_t m) {
    int64_t res = 1;
    a %= m;

    while (b > 0) {
        if (b & 1) res = (res * a) % m;
        a = (a * a) % m;
        b >>= 1;
    }

    return res;
}
```

*Branch-free version:*

```cpp
int64_t power_mod_branchless(int64_t a, int64_t b, int64_t m) {
    int64_t res = 1;
    a %= m;

    while (b > 0) {
        int64_t mask = -(b & 1);  // -1 if bit set, 0 otherwise
        res = (res * ((a & mask) | (1 & ~mask))) % m;

        a = (a * a) % m;
        b >>= 1;
    }

    return res;
}
```

Eliminates branch on `b & 1` but adds extra multiplication. Profile-dependent.

== Prime Testing

*Trial Division:* $O(sqrt(n))$

```cpp
bool isPrime(int n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }

    return true;
}
```

*Miller-Rabin Primality Test:* Probabilistic, $O(k log^3 n)$ for k rounds.

```cpp
bool miller_rabin(int64_t n, int iterations = 5) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    // Write n-1 as 2^r * d
    int64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }

    // Witness loop
    for (int i = 0; i < iterations; i++) {
        int64_t a = 2 + rand() % (n - 3);
        int64_t x = power_mod(a, d, n);

        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = (x * x) % n;
            if (x == n - 1) {
                composite = false;
                break;
            }
        }

        if (composite) return false;
    }

    return true;
}
```

*Deterministic for n < $2^(64)$:* Use specific witnesses [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37].

== Matrix Exponentiation

*Problem:* Compute $A^n$ for matrix A.

*Application:* Fibonacci in $O(log n)$.

```cpp
using Matrix = array<array<int64_t, 2>, 2>;

Matrix multiply(const Matrix& A, const Matrix& B, int64_t mod) {
    Matrix C = {0};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod;
            }
        }
    }

    return C;
}

Matrix matrix_power(Matrix A, int64_t n, int64_t mod) {
    Matrix res = {{{1, 0}, {0, 1}}};  // Identity

    while (n > 0) {
        if (n & 1) res = multiply(res, A, mod);
        A = multiply(A, A, mod);
        n >>= 1;
    }

    return res;
}

int64_t fibonacci(int n, int64_t mod) {
    if (n == 0) return 0;

    Matrix A = {{{1, 1}, {1, 0}}};
    Matrix result = matrix_power(A, n - 1, mod);

    return result[0][0];
}
```

*Cache blocking for large matrices (n > 32):* Block into tiles to fit in cache.

== References

*Algorithms:*

*Eratosthenes (240 BC)*. Ancient Greek mathematician, sieve algorithm.

*Euclid (300 BC)*. Elements, Book VII: GCD algorithm.

*Stein, J. (1967)*. Computational Problems Associated with Racah Algebra. Journal of Computational Physics 1(3): 397-405.

*Miller, G.L. (1976)*. Riemann's Hypothesis and Tests for Primality. Journal of Computer and System Sciences 13(3): 300-317.

*Rabin, M.O. (1980)*. Probabilistic Algorithm for Testing Primality. Journal of Number Theory 12(1): 128-138.

*Montgomery, P.L. (1985)*. Modular Multiplication Without Trial Division. Mathematics of Computation 44(170): 519-521.

*Crandall, R. & Pomerance, C. (2005)*. Prime Numbers: A Computational Perspective (2nd ed.). Springer. ISBN 978-0387252827.
