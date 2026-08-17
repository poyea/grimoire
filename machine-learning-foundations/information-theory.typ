#import "../template.typ": xref

= Information Theory

Information theory, founded by Claude Shannon in 1948, provides the mathematical language for quantifying uncertainty, compression, and the fundamental limits of communication. Its core quantities — entropy, mutual information, and channel capacity — appear throughout machine learning, statistics, compression, and communications. This chapter develops the classical theory rigorously and connects it to the tools used daily in learning systems.

*See also:* #xref("machine-learning-foundations", "probability-and-information", label: "Probability and Information") (probability foundations), #xref("machine-learning-foundations", "loss-functions", label: "Loss Functions") (cross-entropy as a training objective), #xref("machine-learning-foundations", "generalization-theory", label: "Generalization Theory") (MDL principle, information-theoretic bounds), #xref("machine-learning-foundations", "network-information-theory", label: "Network Information Theory") (multi-user channels, capacity regions).

== Entropy

=== Shannon Entropy

For a discrete random variable $X$ with alphabet $cal(X)$ and PMF $p$, the *Shannon entropy* is

$ H(X) = -sum_(x in cal(X)) p(x) log p(x). $

The base of the logarithm sets the unit: base 2 gives *bits*, base $e$ gives *nats*. We adopt the convention $0 log 0 = 0$ throughout.

*Interpretation.* $H(X)$ measures the expected number of bits needed to describe a draw from $X$, and equivalently the average surprise $-log p(x)$ of an outcome. It is maximised by the uniform distribution ($H = log |cal(X)|$) and equals zero when $X$ is deterministic.

*Key properties:*
- Non-negativity: $H(X) >= 0$.
- Concavity in $p$: entropy is a concave function of the distribution.
- Chain rule: $H(X, Y) = H(X) + H(Y | X)$.
- Data processing: $H(f(X)) <= H(X)$ for any function $f$.

=== Differential Entropy

For a continuous random variable with density $f$,

$ h(X) = -integral f(x) log f(x) d x. $

Unlike discrete entropy, $h$ can be negative. The Gaussian $cal(N)(mu, sigma^2)$ achieves $h = (1/2) log(2 pi e sigma^2)$ nats, the maximum over all distributions with fixed variance.

=== Joint, Conditional, and Chain Entropy

$ H(X, Y) = -sum_(x, y) p(x, y) log p(x, y), $
$ H(Y | X) = sum_x p(x) H(Y | X = x) = H(X, Y) - H(X). $

The *chain rule* generalises to $n$ variables:

$ H(X_1, ..., X_n) = sum_(i=1)^n H(X_i | X_1, ..., X_(i-1)). $

== Mutual Information

=== Definition

The *mutual information* between $X$ and $Y$ is

$ I(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X) = H(X) + H(Y) - H(X, Y). $

It is symmetric ($I(X;Y) = I(Y;X)$), non-negative ($I(X;Y) >= 0$), and equals zero if and only if $X$ and $Y$ are independent.

*Operational meaning.* $I(X;Y)$ is the reduction in uncertainty about $X$ upon observing $Y$, and symmetrically, the amount of information $X$ and $Y$ share.

=== KL Divergence

The Kullback–Leibler divergence from $P$ to $Q$ is

$ D_"KL"(P || Q) = sum_x p(x) log (p(x)) / (q(x)) = EE_P [log (p(X)) / (q(X))]. $

It is non-negative (Gibbs' inequality) and equals zero if and only if $P = Q$. It is not symmetric: $D_"KL"(P||Q) != D_"KL"(Q||P)$ in general.

The mutual information can be expressed as

$ I(X; Y) = D_"KL"(p(x, y) || p(x) p(y)). $

=== Conditional Mutual Information

$ I(X; Y | Z) = H(X | Z) - H(X | Y, Z) = D_"KL"(p(x, y | z) || p(x|z) p(y|z)). $

The *chain rule for MI*: $I(X_1, ..., X_n; Y) = sum_i I(X_i; Y | X_1, ..., X_(i-1))$.

== The Information Diagram

The following diagram relates the four fundamental quantities for two variables:

#align(center)[
#table(
  columns: 3,
  align: center,
  [], [*Marginal*], [*Conditional*],
  [$H(X)$], [$H(X | Y) + I(X;Y)$], [$H(X | Y)$],
  [$H(Y)$], [$H(Y | X) + I(X;Y)$], [$H(Y | X)$],
  [$H(X,Y)$], [$H(X) + H(Y) - I(X;Y)$], [],
)
]

== Data Compression

=== Source Coding Theorem

Shannon's first theorem establishes the fundamental limits of lossless compression.

*Theorem (Shannon, 1948).* For an i.i.d. source with entropy $H(X)$, any lossless code requires at least $H(X)$ bits per symbol on average. A sequence of $n$ symbols can be compressed to $n(H(X) + epsilon)$ bits with probability of error going to zero as $n -> infinity$, for any $epsilon > 0$.

The theorem relies on *typicality*: almost all long sequences drawn from $p$ have empirical entropy close to $H(X)$.

=== Typical Sequences

The *typical set* $A_epsilon^((n))$ is the set of sequences $x^n$ satisfying

$ |-1/n log p(x^n) - H(X)| <= epsilon. $

The typical set has probability close to 1, contains $approx 2^(n H(X))$ sequences, and each has probability $approx 2^(-n H(X))$.

=== Huffman and Arithmetic Coding

*Huffman coding* constructs an optimal prefix-free code whose expected length satisfies $H(X) <= L < H(X) + 1$ bits per symbol. *Arithmetic coding* approaches $H(X)$ exactly in the limit and handles adaptive models efficiently. Modern compressors (Zstandard, brotli) layer learned models atop these entropy coders.

=== Lempel–Ziv Compression

The LZ77/LZ78 family of algorithms is asymptotically optimal for any stationary ergodic source without knowing the source distribution. This universality makes LZ the foundation of gzip, DEFLATE, and most practical compressors.

== Channel Capacity

=== The Noisy-Channel Theorem

A discrete memoryless channel (DMC) is specified by a conditional distribution $p(y|x)$ over input alphabet $cal(X)$ and output alphabet $cal(Y)$. The *channel capacity* is

$ C = max_(p(x)) I(X; Y) quad "bits per channel use." $

*Theorem (Shannon, 1948).* For any rate $R < C$ and any $epsilon > 0$, there exists a code of rate $R$ with block error probability $< epsilon$ for sufficiently large block length $n$. Conversely, any sequence of codes with rate $R > C$ has error probability bounded away from zero.

=== Gaussian Channel

For the additive white Gaussian noise (AWGN) channel $Y = X + Z$ with $Z tilde cal(N)(0, N)$ and power constraint $EE[X^2] <= P$,

$ C = (1/2) log_2(1 + P/N) quad "bits per channel use" $

where $P/N$ is the signal-to-noise ratio. This is Shannon's famous *capacity formula*, which underpins all of wireless and wireline communications.

=== Binary Symmetric Channel

The BSC with crossover probability $p$ has capacity $C = 1 - H_b(p)$, where $H_b(p) = -p log p - (1-p) log(1-p)$ is the binary entropy function.

=== Water-Filling

For parallel Gaussian channels with $n$ sub-channels of noise variance $N_i$ and total power $P$, the optimal power allocation is *water-filling*:

$ P_i = (mu - N_i)^+ $

where $mu$ (the "water level") is chosen so that $sum_i P_i = P$. Channels with higher noise receive less power; channels below the water level receive none.

== Source-Channel Separation

Shannon's *separation theorem* states that, for point-to-point channels, source and channel coding can be designed independently without loss of optimality. This justifies the layered design of communication systems: compress first, then add redundancy for the channel.

The separation result does not generalise to multi-user settings, where joint source-channel coding can be strictly better.

== Differential Entropy and Gaussian Extremality

The Gaussian distribution maximises differential entropy over all distributions with fixed mean and variance:

$ h(X) <= (1/2) log(2 pi e sigma^2) $

with equality if and only if $X tilde cal(N)(mu, sigma^2)$. This *maximum entropy principle* provides the information-theoretic justification for Gaussian models as the least-informative choice consistent with second-order statistics.

== Rate-Distortion Theory

When some distortion is tolerable, perfect reconstruction may be unnecessary. Given a distortion measure $d(x, hat(x))$, the *rate-distortion function* is

$ R(D) = min_(p(hat(x) | x) : EE[d(X, hat(X))] <= D) I(X; hat(X)). $

For a Gaussian source with variance $sigma^2$ and squared-error distortion,

$ R(D) = (1/2) log_2 (sigma^2 / D), quad 0 <= D <= sigma^2. $

Rate-distortion theory underlies lossy compression (JPEG, MP3) and informs the design of quantized neural networks and variational autoencoders.

== Information Theory in Machine Learning

=== Cross-Entropy Loss

The cross-entropy of a model $q$ relative to the true distribution $p$ is

$ H(p, q) = -sum_x p(x) log q(x) = H(p) + D_"KL"(p || q). $

Minimising cross-entropy over a fixed training set is equivalent to maximum likelihood estimation and to minimising the KL divergence from the model to the data distribution.

=== Minimum Description Length

The *MDL principle* selects the model that minimises the total description length: code for the data given the model plus the code for the model itself. MDL operationalises Occam's razor and connects Bayesian model selection to information theory.

=== Mutual Information in Representation Learning

Methods like InfoNCE (used in contrastive learning) and the Information Bottleneck principle use MI as a training objective. The *information bottleneck* seeks a representation $Z$ of $X$ that maximises $I(Z; Y)$ (predictive) while minimising $I(Z; X)$ (compression):

$ min_(p(z|x)) I(Z; X) - beta dot I(Z; Y). $

This trade-off appears in variational autoencoders via the ELBO decomposition and in the analysis of deep network generalisation.

=== Fisher Information

The Fisher information matrix measures the expected curvature of the log-likelihood:

$ cal(I)(theta)_(i j) = EE_theta [partial/(partial theta_i) log p(X; theta) dot partial/(partial theta_j) log p(X; theta)]. $

The Cramér–Rao bound states $"Var"(hat(theta)) >= cal(I)(theta)^(-1)$ for any unbiased estimator, making Fisher information the fundamental limit on estimation precision. Natural gradient descent uses the inverse Fisher metric to achieve reparameterisation-invariant updates.

== Further Reading

- Shannon, C. E. (1948). _A Mathematical Theory of Communication_. Bell System Technical Journal.
- Cover, T. M., & Thomas, J. A. (2006). _Elements of Information Theory_, 2nd ed. Wiley.
- Blahut, R. E. (2010). _Principles and Practice of Information Theory_. Addison-Wesley.
- MacKay, D. J. C. (2003). _Information Theory, Inference, and Learning Algorithms_. Cambridge University Press. (freely available online)
- Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. _Proc. 37th Allerton Conference_.
