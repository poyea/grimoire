#import "../template.typ": theorem, xref

= Monte Carlo Methods and Random Numbers <monte-carlo-methods>

Monte Carlo methods estimate quantities by averaging over random samples. Their error shrinks only as $N^(-1\/2)$, painfully slowly by the standards of quadrature, but that rate does not depend on dimension, on smoothness, or on the shape of the domain, which makes Monte Carlo the only general method for integrals over dozens or thousands of dimensions: option prices driven by many risk factors, particle transport, rendering, Bayesian posteriors, and statistical physics. Everything rests on random numbers that are not random at all, so the generator matters as much as the estimator. This chapter covers the basic estimator and its error, pseudorandom generators and their failures, sampling from distributions, variance reduction, quasi-Monte Carlo with low-discrepancy sequences, and Markov chain Monte Carlo.

*See also:* #xref("numerical-computing", "numerical-integration", label: "Numerical Integration"), #xref("numerical-computing", "floating-point", label: "Floating Point") (reproducibility), #xref("machine-learning-foundations", "probability-and-information", label: "Probability and Information"), #xref("graphics-and-rendering", "gi-techniques", label: "Global Illumination Techniques") (path tracing).

== The Estimator

To estimate $I = EE[f(X)] = integral f(x) p(x) d x$, draw independent samples $X_1, dots, X_N$ from the density $p$ and average:

$ hat(I)_N = 1 / N sum_(i = 1)^N f(X_i). $

The estimator is unbiased, and if $sigma^2 = "Var"(f(X))$ is finite, the central limit theorem gives

$ hat(I)_N approx I + sigma / sqrt(N) Z, quad Z tilde cal(N)(0, 1). $

Three consequences shape all practice. The error is *statistical*: report $hat(I)_N plus.minus 1.96 hat(sigma) \/ sqrt(N)$ as a 95 percent confidence interval, with $hat(sigma)$ the sample standard deviation. Each additional decimal digit costs 100 times more samples. And the constant $sigma$ is the only lever besides $N$: halving $sigma$ is worth four times the samples, which is why variance reduction is the heart of Monte Carlo engineering.

The sample variance should itself be computed stably, with Welford's one-pass update, rather than as $overline(x^2) - overline(x)^2$, which cancels catastrophically when the mean is large relative to the spread.

== Pseudorandom Number Generators

A *pseudorandom number generator* (PRNG) is a deterministic state machine whose outputs pass statistical tests for randomness. Its quality has several dimensions: period, equidistribution in many dimensions, results on test batteries (TestU01's BigCrush, PractRand), speed, state size, and how well it supports parallel streams.

#table(
  columns: 3,
  [*Generator*], [*Properties*], [*Verdict*],
  [Linear congruential, $x_(k+1) = (a x_k + c) mod m$], [Tiny state; consecutive $d$-tuples lie on few hyperplanes; low bits have short periods], [Avoid for simulation. IBM's RANDU put all triples on 15 planes],
  [Mersenne Twister (MT19937)], [Period $2^19937 - 1$, 2.5 KB state, equidistributed in 623 dimensions], [Fails linear-complexity tests in BigCrush; slow to recover from poor seeding; acceptable but dated],
  [xoshiro256\*\*, xoroshiro], [256-bit state, very fast, passes BigCrush], [Good general default; has jump functions for parallel streams],
  [PCG family], [LCG state with a permuted output function], [Good statistics, small state, multiple streams],
  [Counter-based (Philox, Threefry)], [Output is a keyed bijection of a counter], [Ideal for parallel and GPU work: any stream, any position, no state to share],
  [Cryptographic (ChaCha20, AES-CTR)], [Unpredictable], [Required for security; also fine, if slower, for simulation],
)

*Parallel streams* are the most common practical trap. Giving each thread a generator seeded with `seed + thread_id` can produce correlated streams for some generators. Safe options are counter-based generators (each thread uses its own key or counter range), generators with jump-ahead functions that split one long period into disjoint blocks, or a seeding procedure that hashes seeds well (NumPy's `SeedSequence`). Counter-based generators also give *reproducibility independent of scheduling*: sample $i$ always uses counter $i$, whichever thread computes it.

```cpp
#include <bit>
#include <cstdint>

// xoshiro256** (Blackman and Vigna). Seed the state from a hash such as splitmix64,
// never with all zeros.
class xoshiro256ss {
    std::uint64_t s_[4];

    static std::uint64_t splitmix64(std::uint64_t& x) {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

public:
    explicit xoshiro256ss(std::uint64_t seed) {
        for (auto& s : s_) s = splitmix64(seed);
    }

    std::uint64_t next() {
        const std::uint64_t result = std::rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);
        return result;
    }

    // Uniform double in [0, 1) from the top 53 bits.
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};
```

Converting integers to floating point needs care too: dividing a 32-bit integer by $2^32$ leaves gaps much coarser than the double grid near zero, and some conversions can return exactly 1.0. Taking the top 53 bits and scaling by $2^(-53)$ gives evenly spaced values in $[0, 1)$.

For reproducibility across platforms, note that the C++ standard specifies the engines (`std::mt19937_64` produces the same sequence everywhere) but not the algorithms of the distributions, so `std::normal_distribution` gives different values with libstdc++, libc++ and MSVC. Simulations that must reproduce bit for bit implement their own transforms.

== Sampling Distributions

Uniform samples are transformed into other distributions:

- *Inverse transform*: if $U$ is uniform on $(0, 1)$ and $F$ is a continuous CDF, then $F^(-1)(U)$ has distribution $F$. The exponential distribution is $-ln(1 - U) \/ lambda$. It is exact and monotone, which preserves the structure of stratified and quasi-random inputs, but requires a computable inverse.
- *Box-Muller*: two uniforms give two independent standard normals, $sqrt(-2 ln U_1) cos(2 pi U_2)$ and $sqrt(-2 ln U_1) sin(2 pi U_2)$.
- *Rejection sampling*: sample $x$ from an easy density $g$ with $p <= c g$, accept with probability $p(x) \/ (c g(x))$. The expected number of tries per sample is $c$. The *ziggurat* method (Marsaglia and Tsang) covers the density with horizontal strips so that about 99 percent of samples need only one comparison and one multiplication, making it the fastest normal and exponential sampler in practice.
- *Alias method*: sample from a discrete distribution with $n$ outcomes in $O(1)$ time after $O(n)$ setup.

== Variance Reduction

Consider estimating $integral_0^1 e^x d x = e - 1$ with $f(U) = e^U$. Plain Monte Carlo has $sigma^2 approx 0.242$.

- *Antithetic variates*: use each uniform twice, as $U$ and $1 - U$, averaging $(f(U) + f(1 - U)) \/ 2$. For monotone $f$ the two are negatively correlated. Here the variance of the pair average is about 0.0039, roughly a 31-fold improvement per function evaluation.
- *Control variates*: if $g(X)$ has known mean $mu_g$ and is correlated with $f(X)$, estimate with $f(X) - b (g(X) - mu_g)$. The optimal coefficient is $b = "Cov"(f, g) \/ "Var"(g)$, and the variance falls by the factor $1 - rho^2$. With $g(U) = U$ (mean 1/2), $rho approx 0.992$, $b approx 1.69$, and the variance drops to about 0.0039, a 60-fold improvement at the cost of one subtraction.
- *Importance sampling*: sample from a density $q$ that puts more mass where $|f| p$ is large, and weight each sample by $p \/ q$: $hat(I) = 1 \/ N sum f(X_i) p(X_i) \/ q(X_i)$. The ideal $q$ is proportional to $|f| p$. It is essential for rare events (a 1-in-$10^9$ failure probability is invisible to plain sampling), and dangerous when $q$ has lighter tails than $f p$: the weights then have infinite variance, and the estimator can look converged while being badly wrong.
- *Stratified sampling*: split the domain into strata and sample each separately in proportion to its size (or to its variance). It never increases variance, and with one sample per stratum in one dimension the error improves to $O(N^(-3\/2))$ for smooth integrands.

```cpp
#include <cmath>
#include <cstddef>

struct estimate {
    double mean;
    double std_error;
};

// Welford's algorithm for mean and variance.
class running_stats {
    std::size_t n_ = 0;
    double mean_ = 0.0, m2_ = 0.0;

public:
    void add(double x) {
        ++n_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        m2_ += delta * (x - mean_);
    }
    estimate result() const {
        const double var = n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0;
        return {mean_, std::sqrt(var / static_cast<double>(n_))};
    }
};

estimate exp_control_variate(xoshiro256ss& rng, std::size_t n) {
    constexpr double b = 1.6903;            // Cov(e^U, U) / Var(U)
    running_stats stats;
    for (std::size_t i = 0; i < n; ++i) {
        const double u = rng.uniform();
        stats.add(std::exp(u) - b * (u - 0.5));
    }
    return stats.result();
}
```

In practice $b$ is estimated from a pilot run or from the same samples (which introduces a small bias of order $1 \/ N$).

== Quasi-Monte Carlo

Random points clump and leave gaps. *Low-discrepancy sequences* fill space more evenly by design. Their quality is measured by the *star discrepancy* $D_N^*$, the largest difference, over all boxes anchored at the origin, between the fraction of points in the box and its volume.

#theorem(name: "Koksma-Hlawka inequality")[
For $f$ of bounded variation $V(f)$ in the sense of Hardy and Krause on $[0, 1]^d$, and any points $x_1, dots, x_N$,
$ |1 / N sum_(i = 1)^N f(x_i) - integral_([0, 1]^d) f| <= V(f) D_N^*. $
]

Good sequences achieve $D_N^* = O((log N)^d \/ N)$, so for moderate $d$ quasi-Monte Carlo (QMC) converges almost like $N^(-1)$ instead of $N^(-1\/2)$.

- The *van der Corput* sequence in base $b$ reflects the base-$b$ digits of $i$ about the radix point: in base 2, $1, 2, 3, 4 -> 0.5, 0.25, 0.75, 0.125$. *Halton* uses van der Corput in a different prime base per dimension, which works well in a few dimensions but develops strong correlations between high dimensions with large prime bases.
- *Sobol* sequences use base 2 in every dimension with carefully chosen direction numbers (Joe and Kuo's are standard). They are the default in finance and are generated by XOR operations, as fast as a PRNG.
- *Lattice rules* use points $\{i z \/ N\}$ for a generating vector $z$, constructed component by component for a given class of integrands; they are especially good for periodic integrands.

```cpp
// Radical inverse of i in base b (van der Corput); Halton uses one prime base per dimension.
double radical_inverse(std::uint64_t i, unsigned base) {
    double result = 0.0, scale = 1.0 / base;
    while (i > 0) {
        result += static_cast<double>(i % base) * scale;
        i /= base;
        scale /= base;
    }
    return result;
}
```

In an experiment estimating $integral_([0,1]^4) product_k (pi \/ 2) sin(pi x_k) d x = 1$ with 4,096 points, plain Monte Carlo has a standard error near 0.018, while the Halton points in bases 2, 3, 5 and 7 give an error under 0.0005.

QMC loses two things that Monte Carlo provides: an error estimate (the Koksma-Hlawka bound is not computable in practice) and robustness for discontinuous integrands. *Randomised QMC* restores the first: randomly scramble or shift the point set (Owen scrambling for Sobol, random shifts for lattices), compute several independent replicates, and use their spread as a confidence interval. Scrambled nets also improve the convergence rate for smooth integrands to about $N^(-3\/2)$. QMC also explains why it often works in high nominal dimension: many practical integrands have low *effective dimension*, depending mostly on a few variables or on low-order interactions, and reordering variables (for example with Brownian bridge construction of paths in finance) puts the important ones in the best-distributed coordinates.

== Markov Chain Monte Carlo

When samples from $p$ cannot be drawn directly, as with a Bayesian posterior known only up to its normalising constant, *Markov chain Monte Carlo* (MCMC) builds a Markov chain whose stationary distribution is $p$ and averages along the chain.

The *Metropolis-Hastings* algorithm proposes $y$ from a density $q(y | x)$ and accepts it with probability

$ alpha(x, y) = min(1, (p(y) q(x | y)) / (p(x) q(y | x))), $

otherwise staying at $x$. Only ratios of $p$ appear, so the normalising constant cancels. The acceptance rule makes the chain satisfy *detailed balance*, $p(x) P(x -> y) = p(y) P(y -> x)$, which implies that $p$ is stationary.

MCMC samples are correlated, which changes the error analysis. The variance of the chain average is $sigma^2 tau \/ N$, where $tau = 1 + 2 sum_(k >= 1) rho_k$ is the *integrated autocorrelation time*, so the *effective sample size* is $N \/ tau$. A chain of a million steps with $tau = 1000$ carries the information of only a thousand independent samples. Practitioners discard an initial *burn-in*, run several chains from dispersed starting points, and check that they agree (the $hat(R)$ diagnostic) before trusting results.

Random-walk proposals scale badly with dimension: the step size must shrink as $d^(-1\/2)$ to keep a reasonable acceptance rate. *Hamiltonian Monte Carlo* uses the gradient of $log p$ to propose distant points along simulated Hamiltonian trajectories (integrated with the symplectic leapfrog method), which scales far better. Its adaptive variant, the *No-U-Turn Sampler* (NUTS), chooses trajectory lengths automatically and is the default in Stan and PyMC. *Gibbs sampling* updates one variable or block at a time from its conditional distribution, and works well when those conditionals are easy to sample.

== Pitfalls

- Using `rand()` or a poor LCG, or seeding parallel streams with consecutive integers.
- Reporting Monte Carlo results without a confidence interval.
- Importance sampling with a proposal whose tails are lighter than the integrand's, giving infinite-variance weights.
- Using quasi-Monte Carlo without randomisation, and so without any error estimate.
- Treating MCMC samples as independent; ignoring autocorrelation, burn-in and convergence diagnostics.
- Expecting bit-identical results across standard libraries from `std::` distributions.

== Further Reading

Owen, A. B. (2013). _Monte Carlo Theory, Methods and Examples_. Self-published book, statweb.stanford.edu.

Glasserman, P. (2003). _Monte Carlo Methods in Financial Engineering_. Springer.

L'Ecuyer, P. and Simard, R. (2007). "TestU01: A C Library for Empirical Testing of Random Number Generators." ACM Transactions on Mathematical Software.

Matsumoto, M. and Nishimura, T. (1998). "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform Pseudo-Random Number Generator." ACM Transactions on Modeling and Computer Simulation.

Blackman, D. and Vigna, S. (2021). "Scrambled Linear Pseudorandom Number Generators." ACM Transactions on Mathematical Software.

Salmon, J. K. et al. (2011). "Parallel Random Numbers: As Easy as 1, 2, 3." SC.

Marsaglia, G. and Tsang, W. W. (2000). "The Ziggurat Method for Generating Random Variables." Journal of Statistical Software.

Niederreiter, H. (1992). _Random Number Generation and Quasi-Monte Carlo Methods_. SIAM.

Joe, S. and Kuo, F. Y. (2008). "Constructing Sobol Sequences with Better Two-Dimensional Projections." SIAM Journal on Scientific Computing.

Metropolis, N. et al. (1953). "Equation of State Calculations by Fast Computing Machines." Journal of Chemical Physics.

Hastings, W. K. (1970). "Monte Carlo Sampling Methods Using Markov Chains and Their Applications." Biometrika.

Hoffman, M. D. and Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." Journal of Machine Learning Research.
