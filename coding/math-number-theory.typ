= Math & Number Theory <math-number-theory>

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
    // sqrt() can round down by one in floating-point; bump until correct.
    int limit = (int)sqrt((double)n);
    while ((long long)(limit + 1) * (limit + 1) <= n) limit++;
    while ((long long)limit * limit > n) limit--;
    vector<bool> small_primes = sieve(limit);
    vector<int> primes;

    for (int i = 2; i <= limit; i++) {
        if (small_primes[i]) primes.push_back(i);
    }

    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    vector<bool> segment(SEG_SIZE);

    for (int low = 0; low <= n; low += SEG_SIZE) {
        fill(segment.begin(), segment.end(), true);
        int high = min(low + SEG_SIZE - 1, n);

        for (int p : primes) {
            int start = max(p * p, (low + p - 1) / p * p);
            for (int j = start; j <= high; j += p) {
                segment[j - low] = false;
            }
        }

        for (int i = low; i <= high; i++) {
            if (i < 2) continue;
            is_prime[i] = segment[i - low];
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
// Schematic — `neg_mod_inverse_2_64` computes -mod^{-1} mod 2^64 via Hensel
// lifting (start with -mod mod 4, double-precision Newton step five times).
class Montgomery {
    uint64_t mod, r2, inv;  // inv = -mod^{-1} mod 2^64

    uint64_t reduce(__uint128_t t) {
        // R = 2^64 conceptually; mask-by-64-bits is implicit in uint64_t cast.
        // Since inv ≡ -mod^{-1} (mod 2^64), t + m*mod ≡ 0 (mod 2^64), so the
        // shift below is exact.
        uint64_t m = (uint64_t)t * inv;
        uint64_t u = (uint64_t)((t + (__uint128_t)m * mod) >> 64);
        return u >= mod ? u - mod : u;
    }

public:
    Montgomery(uint64_t m) : mod(m) {
        inv = neg_mod_inverse_2_64(mod);
        // r2 = R^2 mod m = (2^64)^2 mod m, computed without literal 2^64:
        __uint128_t r_mod = ((__uint128_t)1 << 64) % mod;
        r2 = (uint64_t)((r_mod * r_mod) % mod);
    }

    uint64_t mul(uint64_t a, uint64_t b) {
        return reduce((__uint128_t)a * b);
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
    // Use __int128 for the multiplications: when m approaches 2^63, the
    // product (res * a) overflows signed int64 (UB). __int128 is enough
    // because the operands are bounded by m < 2^63.
    __int128 res = 1;
    a %= m;

    while (b > 0) {
        if (b & 1) res = (res * a) % m;
        a = (int64_t)(((__int128)a * a) % m);
        b >>= 1;
    }

    return (int64_t)res;
}
```

*Branch-free version:*

```cpp
int64_t power_mod_branchless(int64_t a, int64_t b, int64_t m) {
    int64_t res = 1;
    a %= m;

    while (b > 0) {
        int64_t mask = -(b & 1);  // -1 if bit set, 0 otherwise
        int64_t mul = (a & mask) | (1 & ~mask);
        // __int128 widen: int64 product overflows when m > 2^32.
        res = (int64_t)(((__int128)res * mul) % m);
        a = (int64_t)(((__int128)a * a) % m);
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

    // Witness loop. Use mt19937_64 so the witness range covers full n;
    // rand() is bounded by RAND_MAX (often 2^15) and samples only a tiny prefix.
    static std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<int64_t> dist(2, n - 2);
    for (int i = 0; i < iterations; i++) {
        int64_t a = dist(rng);
        int64_t x = power_mod(a, d, n);

        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (int j = 0; j < r - 1; j++) {
            x = (int64_t)(((__int128)x * x) % n);  // avoid signed overflow when n > 2^31
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
                // __int128: A[i][k] * B[k][j] overflows int64 when mod ≈ 2^63.
                C[i][j] = (int64_t)((C[i][j] + (__int128)A[i][k] * B[k][j]) % mod);
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

== Vieta Jumping

*Technique:* Infinite descent on quadratic Diophantine equations using Vieta's formulas. Given one solution, "jump" to a smaller one; since positive integers cannot descend forever, the minimal solution must be degenerate, which forces the desired conclusion.

*Canonical example (IMO 1988, Problem 6):* If $a, b$ are positive integers with $(a b + 1) | (a^2 + b^2)$, then $k = (a^2 + b^2) \/ (a b + 1)$ is a perfect square.

*The jump:* Fix $k$ and $b$, and view the relation as a quadratic in $x$:

$ x^2 - k b x + (b^2 - k) = 0. $

If $a$ is one root, Vieta's formulas give the other root

$ a' = k b - a = (b^2 - k) / a. $

The first form shows $a'$ is an integer; the second bounds it: if $a > b$ then $a' = (b^2 - k)\/a < b < a$, so $(a', b)$ is a strictly smaller solution with the same $k$. Take the solution minimizing $a + b$ with $a >= b$. If $a' > 0$, the pair $(b, a')$ contradicts minimality; $a' < 0$ is impossible: with $a' <= -1$, $a'^2 - k b a' + b^2 - k >= a'^2 + k b + b^2 - k >= a'^2 + b^2 > 0$, but a root makes it zero. Hence $a' = 0$, so $k = b^2$.

*Algorithmic view:* the jump map generates all solutions for a fixed $k$ from the base pair $(b, 0)$ — the solution tree of the Markov-style equation. Running it forward enumerates solutions; running it backward (descent) proves structure.

```cpp
// Enumerate (a, b) with (a^2 + b^2) = k * (a*b + 1), k = t^2,
// by jumping up from the root (t, 0): (a, b) -> (k*a - b, a).
#include <cstdint>
#include <utility>
#include <vector>

std::vector<std::pair<int64_t, int64_t>> vieta_solutions(int64_t t, int64_t limit) {
    int64_t k = t * t;
    std::vector<std::pair<int64_t, int64_t>> out;
    int64_t b = 0, a = t;          // degenerate base solution
    while (a <= limit) {
        out.push_back({a, b});
        int64_t next = k * a - b;  // Vieta: the other root of x^2 - k*a*x + (a^2 - k)
        if (next <= a) break;      // t = 1 degenerates: (1,0) and (1,1) only
        b = a;
        a = next;
    }
    return out;
}

// Descent direction: verify a non-degenerate pair reduces to (t, 0).
bool vieta_descend(int64_t a, int64_t b) {  // requires (a*b+1) | (a^2+b^2)
    if (a < b) std::swap(a, b);
    int64_t k = (a * a + b * b) / (a * b + 1);
    while (b > 0) {
        int64_t a2 = k * b - a;    // jump to the smaller root
        a = b;
        b = a2;
        if (a < b) return false;   // descent must be monotone
    }
    return k == a * a;             // minimal solution is (sqrt(k), 0)
}
```

For $t = 2$ ($k = 4$): $(2,0), (8,2), (30,8), (112,30), dots$ — each pair satisfies $(a^2+b^2)\/(a b+1) = 4$. Values grow geometrically (ratio $approx k$), so use `__int128` or stop early when `k * a - b` would overflow. For $t = 1$ the equation $a^2 - a b + b^2 = 1$ has only $(1,0)$ and $(1,1)$.

*When to reach for it:* divisibility conditions symmetric in two variables, where the expression is quadratic in each variable separately. Related to descent arguments on Markov triples ($x^2+y^2+z^2 = 3 x y z$) and Apollonian-style recursions.

== References

*Algorithms:*

*Eratosthenes (240 BC)*. Ancient Greek mathematician, sieve algorithm.

*Euclid (300 BC)*. Elements, Book VII: GCD algorithm.

*Stein, J. (1967)*. Computational Problems Associated with Racah Algebra. Journal of Computational Physics 1(3): 397-405.

*Miller, G.L. (1976)*. Riemann's Hypothesis and Tests for Primality. Journal of Computer and System Sciences 13(3): 300-317.

*Rabin, M.O. (1980)*. Probabilistic Algorithm for Testing Primality. Journal of Number Theory 12(1): 128-138.

*Montgomery, P.L. (1985)*. Modular Multiplication Without Trial Division. Mathematics of Computation 44(170): 519-521.

*Crandall, R. & Pomerance, C. (2005)*. Prime Numbers: A Computational Perspective (2nd ed.). Springer. ISBN 978-0387252827.

== Further Reading

Knuth, D. E. (1969). _The Art of Computer Programming_, Vol. 2: Seminumerical Algorithms. Addison-Wesley. (Comprehensive treatment of arithmetic algorithms, modular arithmetic, GCD, and primality.)

Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). _Introduction to Algorithms_, 4th ed. MIT Press. (Chapter 31 on number-theoretic algorithms: GCD, modular exponentiation, RSA, and Miller-Rabin primality.)

Hardy, G. H., & Wright, E. M. (2008). _An Introduction to the Theory of Numbers_, 6th ed. Oxford University Press. (Classical mathematical foundation for number theory underlying algorithmic applications.)

Shoup, V. (2009). _A Computational Introduction to Number Theory and Algebra_, 2nd ed. Cambridge University Press. (Bridges pure number theory and algorithmic implementation; freely available online.)

Crandall, R., & Pomerance, C. (2005). _Prime Numbers: A Computational Perspective_, 2nd ed. Springer. (Deep coverage of primality testing, integer factorization, and cryptographic applications.)
