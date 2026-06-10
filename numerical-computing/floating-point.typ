= Floating-Point Arithmetic

Every numerical computation rests on a finite approximation of the reals, and most numerical disasters trace back to forgetting that. IEEE 754 floating point is a remarkably well-designed system — deterministic, portable, with carefully specified rounding — yet it still allows $0.1 + 0.2 != 0.3$, sums that depend on summation order, and silent loss of fifteen digits in a single subtraction. This chapter covers the IEEE 754 formats and their semantics, machine epsilon and ulp, catastrophic cancellation, compensated summation, fused multiply-add, the reduced-precision formats driving modern ML hardware, and reproducibility.

*See also:* _Error Analysis_ (how rounding errors propagate through algorithms), _Linear Systems_ (where conditioning amplifies rounding), _Optimization Algorithms_ (mixed-precision training pitfalls).

== IEEE 754 Formats

A binary floating-point number is

$ x = (-1)^s dot 1.f dot 2^(e - "bias"), $

where $s$ is the sign bit, $f$ is the fraction (mantissa without the implicit leading 1), and $e$ is the biased exponent. The standard formats:

#table(
  columns: 5,
  [*Format*], [*Total bits*], [*Exponent*], [*Fraction*], [*Decimal digits*],
  [binary16 (fp16)], [16], [5], [10], [$approx 3.3$],
  [binary32 (fp32)], [32], [8], [23], [$approx 7.2$],
  [binary64 (fp64)], [64], [11], [52], [$approx 15.9$],
  [binary128], [128], [15], [112], [$approx 34.0$],
)

Special values: two zeros ($+0$, $-0$), two infinities, and NaNs (quiet and signaling). NaN compares unequal to everything including itself — the canonical NaN test is `x != x`. The exponent field of all ones encodes infinity (fraction zero) or NaN (fraction nonzero); the all-zeros exponent encodes zero and subnormals.

=== Subnormals

When the exponent underflows below its minimum, IEEE 754 does not jump to zero. *Subnormal* (denormal) numbers drop the implicit leading 1 and fill the gap between zero and the smallest normal number $2^(e_min)$ with evenly spaced values. This guarantees *gradual underflow*: $x - y = 0$ implies $x = y$, a property that simplifies error analysis enormously. The cost is hardware: subnormal arithmetic is often microcoded and can be 10-100$times$ slower, which is why `FTZ` (flush-to-zero) and `DAZ` (denormals-are-zero) flags exist — and why enabling them silently breaks the $x - y = 0 arrow.r.double x = y$ guarantee.

=== Rounding Modes

IEEE 754 specifies that elementary operations ($+, -, times, \/$, `sqrt`) be *correctly rounded*: the result is the exact mathematical value rounded by the active mode. The four modes are round-to-nearest-even (default; ties go to the value with an even last bit, avoiding statistical drift), round toward $+infinity$, round toward $-infinity$, and round toward zero. The directed modes underlie interval arithmetic (see _Error Analysis_). Round-to-nearest gives the fundamental model

$ "fl"(x compose y) = (x compose y)(1 + delta), quad |delta| <= u, $

where $u$ is the unit roundoff and $compose$ is any basic operation.

== Machine Epsilon and ulp

*Machine epsilon* $epsilon$ is the spacing between 1 and the next representable number: $epsilon = 2^(-52) approx 2.22 times 10^(-16)$ for fp64, $2^(-23) approx 1.19 times 10^(-7)$ for fp32. The *unit roundoff* is $u = epsilon \/ 2$ under round-to-nearest.

A *ulp* (unit in the last place) of $x$ is the spacing of representable numbers at $x$'s magnitude: for $x in [2^e, 2^(e+1))$ in fp64, one ulp is $2^(e - 52)$. Correctly rounded operations have error at most $0.5$ ulp; good library functions (`exp`, `log`, `sin`) typically guarantee $<= 1$ ulp. Comparing floats by ulp distance, rather than by an absolute tolerance, is the robust way to write numerical tests:

```python
import math

def ulp_diff(a: float, b: float) -> float:
    """Distance between a and b measured in ulps of a."""
    if a == b:
        return 0.0
    return abs(a - b) / math.ulp(max(abs(a), abs(b)))
```

== Catastrophic Cancellation

Subtracting two nearly equal numbers cancels their leading digits and *promotes* prior rounding errors to relative prominence. The subtraction itself is exact (Sterbenz: if $y \/ 2 <= x <= 2 y$ then $"fl"(x - y) = x - y$); the catastrophe is that errors already present in $x$ and $y$ now dominate.

The classic example is the quadratic formula. For $x^2 - 10^8 x + 1 = 0$, computing $(-b - sqrt(b^2 - 4 a c)) \/ (2 a)$ subtracts two numbers agreeing to 15 digits. The fix uses the algebraically equivalent form for the small root:

$ x_1 = (-b - "sign"(b) sqrt(b^2 - 4 a c)) / (2 a), quad x_2 = c / (a x_1). $

Other canonical rewrites: $1 - cos(x) = 2 sin^2 (x \/ 2)$, `log1p` and `expm1` for arguments near zero, and Welford's algorithm instead of $EE[X^2] - EE[X]^2$ for variance.

== Compensated Summation

Naive left-to-right summation of $n$ terms has worst-case relative error $O(n u)$. *Kahan summation* tracks the rounding error of each addition in a compensation variable and reinjects it, reducing the bound to $O(u) + O(n u^2)$ — effectively independent of $n$:

```python
def kahan_sum(xs):
    s, c = 0.0, 0.0          # sum and running compensation
    for x in xs:
        y = x - c            # corrected next term
        t = s + y            # low-order digits of y are lost here...
        c = (t - s) - y      # ...and recovered here (algebraically zero)
        s = t
    return s
```

The expression `(t - s) - y` is algebraically zero but computes exactly the rounding error of `s + y` — a *compiler must not* simplify it away, which is why aggressive flags like `-ffast-math` break Kahan summation. Pairwise (cascade) summation achieves $O(u log n)$ with no extra state and is what NumPy's `sum` uses. For exact results, Neumaier's variant handles the case $|x| > |s|$, and double-double arithmetic or Shewchuk's expansion arithmetic gives arbitrary-precision accumulation from ordinary floats.

== Fused Multiply-Add

The *fma* instruction computes $a dot b + c$ with a single rounding:

$ "fma"(a, b, c) = "round"(a dot b + c). $

The product is kept exact internally. This halves the error of inner-product accumulation and enables algorithms impossible with separate operations: exact products via $e = "fma"(a, b, -"fl"(a b))$ (TwoProduct), correctly rounded division and square root via Newton iterations, and Kahan's accurate $2 times 2$ determinant. One caution: compilers that contract $a times b + c$ into fma *change results* between builds, another reproducibility hazard. GPUs perform essentially all multiply-accumulate work through fma units.

== Decimal Pitfalls

$0.1$ is not representable in binary: it is the infinite repeating fraction $0.0001100110011..._2$, stored as $0.1000000000000000055511...$ in fp64. Consequences:

- $0.1 + 0.2 = 0.30000000000000004$, so never compare currency with `==`.
- Round-trip printing requires 17 significant digits for fp64 (9 for fp32); modern languages use shortest-round-trip algorithms (Ryū, Grisu) so `print(0.1)` shows `0.1`.
- Financial code should use decimal arithmetic (IEEE 754-2008 decimal128, Python's `decimal`, or integer cents).
- Repeated addition of $0.1$ drifts: the Patriot missile failure (1991, 28 dead) traced to accumulating $0.1 "s"$ in 24-bit fixed point over 100 hours, a clock skew of $0.34 "s"$.

== Reduced-Precision Formats for ML

Deep learning tolerates low precision in a way classical numerics does not, and hardware has chased that tolerance:

#table(
  columns: 5,
  [*Format*], [*Bits*], [*Exponent*], [*Fraction*], [*Notes*],
  [fp32], [32], [8], [23], [Reference; master weights],
  [TF32], [19 used], [8], [10], [NVIDIA tensor cores; fp32 range, fp16 precision],
  [bfloat16], [16], [8], [7], [fp32 range; truncate-friendly; TPU native],
  [fp16], [16], [5], [10], [Needs loss scaling; overflows at 65504],
  [FP8 E4M3], [8], [4], [3], [Forward activations and weights],
  [FP8 E5M2], [8], [5], [2], [Gradients (need range, not precision)],
)

The design logic: *gradients need dynamic range, not precision*, so bfloat16 keeps fp32's 8 exponent bits and sacrifices mantissa — conversion from fp32 is a simple truncation of the low 16 bits. fp16 has more precision but a tiny range, requiring *loss scaling* (multiply the loss by $2^k$ before backward, unscale before the update) to keep gradients out of the subnormal range. FP8 training (Micikevicius et al. 2022) splits the format by role: E4M3 where values are bounded, E5M2 where they are not, with per-tensor scaling factors maintained dynamically. In all schemes, accumulation inside the matrix-multiply happens in fp32 (or fp22-ish on tensor cores), and the optimizer state stays in fp32 — precision is shaved only where the computation is provably tolerant.

== Reproducibility

Floating-point addition is not associative: $(a + b) + c != a + (b + c)$ in general. Any change in evaluation order changes results, which means bitwise reproducibility fights against parallelism:

- *Parallel reductions* (OpenMP, CUDA atomics, all-reduce) sum in nondeterministic order. `torch.use_deterministic_algorithms(True)` forces deterministic kernels at a performance cost.
- *Compiler flags*: `-ffast-math` licenses reassociation, reciprocal approximation, and FTZ; results differ across optimization levels and vector widths (SSE vs AVX changes how loop remainders are handled).
- *Library versions*: cuBLAS and MKL choose algorithms by problem shape and hardware; the same GEMM can differ between GPU models.
- *fma contraction*: whether $a b + c$ is fused is implementation-defined unless pinned with `#pragma STDC FP_CONTRACT`.

Strategies: fix the reduction tree (deterministic all-reduce orders), use integer or fixed-point accumulation for the final reduction (reproducible BLAS, ExBLAS), or relax the requirement to *statistical* reproducibility with documented tolerances. Cross-platform bitwise reproducibility is achievable — IEEE 754 basic operations are fully specified — but every transcendental function, parallel sum, and compiler decision must be controlled.

== Worked Example

Solve $x^2 - 10^8 x + 1 = 0$ in fp64, end to end. The exact roots are $x_1 approx 10^8$ and $x_2 approx 10^(-8)$ (their product is $c \/ a = 1$, their sum $10^8$).

*Step 1: the discriminant.* $b^2 - 4 a c = 10^16 - 4$. Near $10^16$ the fp64 spacing is one ulp $= 2$ (we are in the binade $[2^53, 2^54)$), so $10^16 - 4$ is exactly representable; no error yet.

*Step 2: the square root.* The true value is

$ sqrt(10^16 - 4) = 10^8 sqrt(1 - 4 times 10^(-16)) approx 10^8 - 2 times 10^(-8). $

Near $10^8$ the spacing is one ulp $= 2^(-26) approx 1.49 times 10^(-8)$. The two neighbors of the true value are $10^8$ (distance $2 times 10^(-8)$) and $10^8 - 2^(-26)$ (distance $0.51 times 10^(-8)$), so the correctly rounded result is

$ "fl"(sqrt(10^16 - 4)) = 10^8 - 2^(-26), $

an error of about $0.34$ ulp — well within the $0.5$ ulp guarantee. The square root is nearly perfect.

*Step 3: the naive small root.* Compute $(-b - sqrt(b^2 - 4 a c)) \/ (2 a)$:

$ (10^8 - (10^8 - 2^(-26))) / 2 = 2^(-26) / 2 = 2^(-27) approx 7.4506 times 10^(-9). $

The subtraction is exact (Sterbenz), but it cancels all fifteen matching leading digits and leaves only the square root's rounding error. The true root is $1.0000000000000000 times 10^(-8)$; the computed one is $0.74506 times 10^(-8)$ — a relative error of 25%, i.e. zero correct significant digits, from a single half-ulp-quality input.

*Step 4: the stable form.* Following the rewrite above with $"sign"(b) = -1$:

$ x_1 = (10^8 + (10^8 - 2^(-26))) / 2 = 10^8 - 2^(-27), $

which rounds (ties-to-even) to exactly $10^8$ — an addition of like-signed numbers, so no cancellation. Then

$ x_2 = c / (a x_1) = 1 / 10^8 = 10^(-8), $

correct to full fp64 precision (the true root is $10^(-8)(1 + 10^(-16) + dots)$, indistinguishable at $u approx 1.1 times 10^(-16)$). Same algebra, same inputs: one arrangement loses 16 digits, the other loses none. The half ulp of error was unavoidable; handing it to a cancelling subtraction was not.

== Further Reading

Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." ACM Computing Surveys.

Higham, N. (2002). _Accuracy and Stability of Numerical Algorithms_, 2nd ed. SIAM.

Muller, J.-M. et al. (2018). _Handbook of Floating-Point Arithmetic_, 2nd ed. Birkhäuser.

IEEE Computer Society (2019). "IEEE Standard for Floating-Point Arithmetic." IEEE 754-2019.

Kahan, W. (1965). "Further Remarks on Reducing Truncation Errors." CACM.

Micikevicius, P. et al. (2018). "Mixed Precision Training." ICLR.

Micikevicius, P. et al. (2022). "FP8 Formats for Deep Learning." arXiv.

Shewchuk, J. (1997). "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates." Discrete and Computational Geometry.
