#import "../template.typ": overbar, xref

= The Fast Fourier Transform

The FFT computes the discrete Fourier transform of $n$ points in $O(n log n)$ operations instead of $O(n^2)$ — a speedup that turned spectral analysis, fast convolution, and signal processing from theoretical curiosities into everyday tools. Gilbert Strang called it "the most important numerical algorithm of our lifetime"; it ships in every phone, modem, and MRI scanner. This chapter covers the DFT and its algebra, the Cooley-Tukey recursion, real-input and trigonometric transforms, fast convolution, accuracy, and the non-obvious traps of using FFTs correctly.

*See also:* #xref("numerical-computing", "floating-point", label: "Floating-Point Arithmetic") (why FFT rounding errors are benign), #xref("numerical-computing", "error-analysis", label: "Error Analysis") ($O(log n)$ error growth), #xref("numerical-computing", "iterative-methods", label: "Iterative Methods") (circulant preconditioners apply via FFT), #xref("numerical-computing", "ode-integration", label: "ODE Integration") (spectral methods differentiate by FFT).

== The Discrete Fourier Transform

For $x in CC^n$, the DFT is

$ X_k = sum_(j=0)^(n-1) x_j omega_n^(j k), quad omega_n = e^(-2 pi i \/ n), $

with inverse $x_j = 1/n sum_k X_k omega_n^(-j k)$. As a matrix, $F_n$ has entries $omega_n^(j k)$; the scaled matrix $F_n \/ sqrt(n)$ is unitary, so the DFT preserves energy (Parseval) and has condition number 1 — the transform itself amplifies nothing. Three algebraic facts do most of the work in applications:

- *Convolution theorem*: the DFT diagonalizes circular convolution, $"DFT"(x convolve y) = "DFT"(x) dot.o "DFT"(y)$ elementwise.
- *Circulant diagonalization*: every circulant matrix is $F_n^(-1) D F_n$ for a diagonal $D$ holding the DFT of its first column — circulant systems solve in $O(n log n)$, the basis of circulant and Toeplitz preconditioners.
- *Shift theorem*: translation in time is modulation in frequency, the basis of phase correlation and frequency estimation.

Beware convention skew across libraries: the sign of the exponent, and where the $1\/n$ lives (forward, inverse, or split as $1\/sqrt(n)$), differ between NumPy, FFTW, and MATLAB. Round-tripping `ifft(fft(x))` is safe; mixing libraries mid-pipeline is not.

== Cooley-Tukey

The 1965 Cooley-Tukey paper (anticipated by Gauss in 1805, unpublished) splits a length-$n$ DFT, $n = n_1 n_2$, into $n_1$ DFTs of length $n_2$, a multiplication by *twiddle factors* $omega_n^(j k)$, and $n_2$ DFTs of length $n_1$. For $n = 2^m$, the radix-2 decimation-in-time recursion is

$ X_k = E_k + omega_n^k O_k, quad X_(k + n\/2) = E_k - omega_n^k O_k, $

where $E$ and $O$ are the half-length DFTs of the even- and odd-indexed samples. The recurrence $T(n) = 2 T(n\/2) + O(n)$ gives $O(n log n)$; radix-2 uses about $5 n log_2 n$ real flops, and split-radix lowers the constant to $4 n log_2 n$ (further reduced by Van Buskirk and Johnson-Frigo's tangent FFT in 2007, the current record).

Beyond powers of two: mixed-radix Cooley-Tukey handles any smooth $n$; *Rader's algorithm* converts a prime-length DFT into a convolution of length $n - 1$; *Bluestein's chirp-z* re-expresses any length (and fractional frequencies) as a convolution that can be zero-padded to a convenient size. Consequence: padding to a power of two is never *required* for $O(n log n)$ — but performance still varies a lot with factorization, so choose smooth sizes (products of 2, 3, 5, 7) when you control them; `scipy.fft.next_fast_len` does this lookup.

In-place radix-2 implementations consume input in *bit-reversed* order — the index permutation that makes the butterflies contiguous. Modern performance is memory-bound, not flop-bound: FFTW autotunes among recursion strategies per machine ("plans"), and cuFFT/MKL follow the same codelets-plus-planner design. Reuse plans; planning can cost more than the transform.

== Real Input and Trigonometric Transforms

Real input $x$ gives conjugate-symmetric output, $X_(n-k) = overbar(X_k)$, so half the spectrum is redundant: `rfft` computes $n\/2 + 1$ complex outputs in roughly half the time and memory of a complex FFT. Use it whenever the signal is real, which in practice is almost always.

The DCT (discrete cosine transform) and DST are DFTs of symmetrically extended sequences. The even-symmetric extension of DCT-II avoids the artificial jump a periodic extension creates at the boundary, concentrating energy in few coefficients — the reason DCT-II underlies JPEG, and why Chebyshev spectral methods (a DCT in disguise, via $x = cos theta$) achieve exponential convergence for smooth non-periodic functions. Multidimensional transforms factor into 1-D transforms along each axis: an $n times n$ image costs $O(n^2 log n)$.

== Fast Convolution and Spectral Differentiation

Linear convolution of lengths $m$ and $n$ via FFT: zero-pad both to at least $m + n - 1$, transform, multiply, invert — $O((m+n) log(m+n))$ versus $O(m n)$ direct. Without the padding you get *circular* convolution: the tail wraps around and corrupts the head, the single most common FFT bug. For streaming signals against a fixed filter, overlap-add and overlap-save process blocks with bounded latency. The crossover versus direct convolution sits around filter lengths of 64-128 in practice (`scipy.signal.fftconvolve` picks automatically); the same trick multiplies big integers (Schönhage-Strassen) and polynomials.

Spectral differentiation: for a smooth periodic function sampled on a uniform grid, differentiate by transforming, multiplying coefficient $k$ by $i k$ (with the $k = n\/2$ Nyquist mode zeroed for odd derivatives), and inverting. Accuracy is limited only by the smoothness of the function — "spectral accuracy" — which is why pseudospectral PDE solvers resolve turbulence with far fewer points than finite differences.

== Accuracy

The FFT is not just faster than the naive DFT — it is *more accurate*. Each output of an $O(n^2)$ summation accumulates $O(n u)$ error; the FFT's $log_2 n$ stages of unitary-flavored butterflies give

$ (parallel hat(X) - X parallel_2) / (parallel X parallel_2) = O(u log n) $

(Gentleman-Sande, 1966), with $O(u sqrt(log n))$ observed on average. Caveats: the bound assumes accurate twiddle factors — compute them by direct calls to `cos`/`sin` or stable recurrences, never by repeated multiplication of $omega_n$, which loses a digit every few thousand steps; and the bound is *normwise*, so a spectral coefficient near the noise floor $u parallel x parallel$ carries no relative accuracy. A 120 dB dynamic range claim from an fp32 FFT deserves suspicion.

== Pitfalls

- *Circular wraparound.* Forgetting to zero-pad before convolution aliases the result. Pad to $m + n - 1$ (then to a fast length).
- *Spectral leakage.* The DFT assumes the window is one exact period. A sinusoid not landing on a bin center smears across the spectrum as the window's transform (a sinc, for the implicit rectangular window). Apply a window function — Hann for general use, flat-top for amplitude accuracy, Kaiser to dial the tradeoff — and accept the resolution-versus-leakage compromise; windows also change the effective noise bandwidth, which matters when calibrating power spectra.
- *Aliasing at sampling time.* Frequencies above the Nyquist rate $f_s \/ 2$ fold back irreversibly; no post-hoc processing recovers them. Anti-alias filtering must happen *before* sampling. The same phenomenon appears inside pseudospectral solvers as quadratic-nonlinearity aliasing, controlled by the 3/2 dealiasing rule.
- *fftshift confusion.* Library output orders frequencies $0, 1, ..., n\/2, -n\/2+1, ..., -1$; plotting without `fftshift`, or applying a frequency-domain filter built in shifted order to unshifted data, silently scrambles results.
- *Nonuniform samples.* The FFT requires a uniform grid. For irregular sampling use NUFFT (gridding with carefully chosen kernels, e.g. FINUFFT) or the Lomb-Scargle periodogram — interpolating onto a uniform grid first biases the spectrum.

== Further Reading

Cooley, J., Tukey, J. (1965). "An Algorithm for the Machine Calculation of Complex Fourier Series." Math. Comp.

Van Loan, C. (1992). _Computational Frameworks for the Fast Fourier Transform_. SIAM.

Brigham, E. O. (1988). _The Fast Fourier Transform and Its Applications_. Prentice-Hall.

Frigo, M., Johnson, S. (2005). "The Design and Implementation of FFTW3." Proc. IEEE.

Higham, N. (2002). _Accuracy and Stability of Numerical Algorithms_, 2nd ed. SIAM. Chapter 24.

Trefethen, L. (2000). _Spectral Methods in MATLAB_. SIAM.

Harris, F. (1978). "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform." Proc. IEEE.

Heideman, M., Johnson, D., Burrus, C. (1984). "Gauss and the History of the Fast Fourier Transform." IEEE ASSP Magazine.
