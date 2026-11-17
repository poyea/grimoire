#import "../template.typ": theorem, xref

= Automatic Differentiation <automatic-differentiation>

Optimisation, sensitivity analysis, implicit ODE solvers, Newton's method and machine learning all need derivatives of functions defined by programs. Finite differences are easy but inaccurate (at best about half the digits of the arithmetic) and cost one evaluation per input. Symbolic differentiation produces exact formulas but suffers expression swell and cannot handle loops and branches. *Automatic differentiation* (AD) computes derivatives exact to rounding error by applying the chain rule to the elementary operations a program actually executes. Its reverse mode computes the gradient of a scalar function of millions of variables for a small constant multiple of the cost of evaluating the function, which is what makes training neural networks and adjoint-based design optimisation possible. This chapter covers the computational graph, forward mode with dual numbers, reverse mode with a tape, both in C++, the cost theory, Jacobians and Hessians, memory and checkpointing, differentiating through solvers, and the pitfalls.

*See also:* #xref("numerical-computing", "optimization-algorithms", label: "Optimization Algorithms"), #xref("numerical-computing", "ode-integration", label: "ODE Integration") (adjoint sensitivities), #xref("numerical-computing", "floating-point", label: "Floating Point"), #xref("machine-learning-foundations", "linear-algebra-for-ml", label: "Linear Algebra for ML") (Jacobian-vector products).

== Programs as Computational Graphs

Every evaluation of a program is a sequence of elementary operations, each with a known derivative. For $f(x_1, x_2) = x_1 x_2 + sin(x_1)$:

#table(
  columns: 3,
  [*Step*], [*Value*], [*Local partial derivatives*],
  [$v_1 = x_1$], [input], [],
  [$v_2 = x_2$], [input], [],
  [$v_3 = v_1 v_2$], [], [$partial v_3 \/ partial v_1 = v_2$, $partial v_3 \/ partial v_2 = v_1$],
  [$v_4 = sin v_1$], [], [$partial v_4 \/ partial v_1 = cos v_1$],
  [$v_5 = v_3 + v_4$], [output], [$partial v_5 \/ partial v_3 = 1$, $partial v_5 \/ partial v_4 = 1$],
)

The sequence (a *Wengert list*) forms a directed acyclic graph. The derivative of the output with respect to an input is the sum over all paths between them of the products of local partials along each path. AD organises that sum in one of two orders. Control flow is not a problem: the graph is the trace of the operations executed for the given input, so loops are unrolled and only the taken branch of each conditional appears.

== Forward Mode

Forward mode propagates, alongside each value $v_i$, its *tangent* $dot(v)_i = partial v_i \/ partial x$ along a chosen input direction. Each operation updates the tangent with the chain rule: $dot(v)_3 = dot(v)_1 v_2 + v_1 dot(v)_2$, $dot(v)_4 = cos(v_1) dot(v)_1$, and so on. Seeding $dot(x) = e_j$ gives the $j$-th column of the Jacobian; seeding an arbitrary vector $dot(x) = s$ gives the Jacobian-vector product $J s$ in one pass.

The natural implementation is *dual numbers*: values $a + b epsilon$ with $epsilon^2 = 0$, for which $f(a + b epsilon) = f(a) + f'(a) b epsilon$ for any differentiable $f$. Operator overloading makes ordinary code differentiable without changes:

```cpp
#include <cmath>

template <class T>
struct dual {
    T value;
    T tangent;
};

template <class T> dual<T> operator+(dual<T> a, dual<T> b) { return {a.value + b.value, a.tangent + b.tangent}; }
template <class T> dual<T> operator-(dual<T> a, dual<T> b) { return {a.value - b.value, a.tangent - b.tangent}; }
template <class T> dual<T> operator*(dual<T> a, dual<T> b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}
template <class T> dual<T> operator/(dual<T> a, dual<T> b) {
    return {a.value / b.value, (a.tangent * b.value - a.value * b.tangent) / (b.value * b.value)};
}
template <class T> dual<T> sin(dual<T> a) { using std::sin, std::cos; return {sin(a.value), cos(a.value) * a.tangent}; }
template <class T> dual<T> cos(dual<T> a) { using std::sin, std::cos; return {cos(a.value), -sin(a.value) * a.tangent}; }
template <class T> dual<T> exp(dual<T> a) { using std::exp; const T e = exp(a.value); return {e, e * a.tangent}; }

template <class T>
T f(T x1, T x2) {
    using std::sin;
    return x1 * x2 + sin(x1);
}

// df/dx1 at (2, 3): seed x1 with tangent 1, x2 with tangent 0.
// f(dual<double>{2, 1}, dual<double>{3, 0}).tangent == 3 + cos(2)
```

Because `dual<T>` is generic, `dual<dual<double>>` gives second derivatives by nesting. Forward mode costs a small constant times one evaluation per direction, with no extra memory, so it is the right choice when there are few inputs and many outputs: a Jacobian of $f: RR^n -> RR^m$ costs $n$ forward passes.

== Reverse Mode

Reverse mode propagates *adjoints* $overline(v)_i = partial y \/ partial v_i$ of one output $y$ with respect to every intermediate value, backwards from the output. Starting with $overline(y) = 1$, each operation distributes its adjoint to its inputs: for $v_3 = v_1 v_2$, add $overline(v)_3 v_2$ to $overline(v)_1$ and $overline(v)_3 v_1$ to $overline(v)_2$. When the sweep reaches the inputs, $overline(x)$ is the entire gradient. Seeding $overline(y) = w$ for a vector output gives the vector-Jacobian product $w^T J$.

The backward sweep needs the values computed in the forward sweep (here $v_1$ and $v_2$), so reverse mode records the operations and their local partials on a *tape* and then replays it in reverse:

```cpp
#include <cmath>
#include <cstddef>
#include <vector>

class tape {
public:
    struct node {
        std::size_t parent[2];
        double partial[2];          // d(this) / d(parent)
    };

    std::size_t push(std::size_t p0, double d0, std::size_t p1, double d1) {
        nodes_.push_back({{p0, p1}, {d0, d1}});
        return nodes_.size() - 1;
    }

    std::size_t variable() { return push(0, 0.0, 0, 0.0); }

    // Adjoints of every node with respect to `output`.
    std::vector<double> gradient(std::size_t output) const {
        std::vector<double> adjoint(nodes_.size(), 0.0);
        adjoint[output] = 1.0;
        for (std::size_t i = output + 1; i-- > 0;) {
            const node& n = nodes_[i];
            adjoint[n.parent[0]] += n.partial[0] * adjoint[i];
            adjoint[n.parent[1]] += n.partial[1] * adjoint[i];
        }
        return adjoint;
    }

private:
    std::vector<node> nodes_;
};

struct var {
    tape* t;
    std::size_t index;
    double value;
};

inline var make_input(tape& t, double v) { return {&t, t.variable(), v}; }

inline var operator+(var a, var b) {
    return {a.t, a.t->push(a.index, 1.0, b.index, 1.0), a.value + b.value};
}
inline var operator*(var a, var b) {
    return {a.t, a.t->push(a.index, b.value, b.index, a.value), a.value * b.value};
}
inline var sin(var a) {
    return {a.t, a.t->push(a.index, std::cos(a.value), a.index, 0.0), std::sin(a.value)};
}

// tape t; var x1 = make_input(t, 2), x2 = make_input(t, 3);
// var y = f(x1, x2);                       // same template f as above
// auto g = t.gradient(y.index);            // g[x1.index] == 3 + cos(2), g[x2.index] == 2
```

Input nodes point at themselves (or any node) with zero partials, so they contribute nothing in the sweep. Production tapes are far more elaborate (vectorised operations, expression templates to shrink the tape, checkpoints), but the structure is the same.

== The Cost of Gradients

#theorem(name: "Cheap gradient principle")[
For a function $f: RR^n -> RR$ evaluated by a program with operation count $W$, reverse-mode AD computes the full gradient with an operation count of at most $omega W$, where $omega$ is a small constant (at most about 4 for the usual elementary operations), independent of $n$.
]

Forward mode computes $n$ directional derivatives at cost about $n W$; reverse mode computes the whole gradient for a constant multiple of $W$. For a neural network with $10^9$ parameters, the difference is between impossible and routine. The price is memory: the tape stores every intermediate value needed in the backward sweep, so memory grows with the *length* of the computation, not with its working set. A time-stepping simulation of a million steps with a state of a million variables would need terabytes.

The general rule for $f: RR^n -> RR^m$:

#table(
  columns: 3,
  [*Quantity*], [*Best mode*], [*Cost*],
  [Gradient of a scalar ($m = 1$)], [Reverse], [About 3 to 4 evaluations],
  [Jacobian-vector product $J s$], [Forward], [About 2 to 3 evaluations],
  [Vector-Jacobian product $w^T J$], [Reverse], [About 3 to 4 evaluations],
  [Full Jacobian], [Forward if $n < m$, reverse if $m < n$], [$min(n, m)$ passes, fewer with sparsity],
  [Hessian-vector product $H s$], [Forward over reverse], [A small constant times a gradient],
)

== Sparse Jacobians and Hessians

Many Jacobians are sparse: in a discretised PDE each residual depends on a few neighbouring unknowns. If columns $j$ and $k$ have no nonzero rows in common, a single forward pass seeded with $e_j + e_k$ computes both columns, because their contributions land in different rows. Grouping structurally orthogonal columns is a *graph colouring* problem (Curtis, Powell and Reid), and for a banded or stencil Jacobian the number of colours, and so the number of passes, is a small constant independent of $n$. The same idea, with more elaborate colourings, compresses Hessians.

Hessian-vector products are computed without forming the Hessian by differentiating the gradient in a direction: apply forward mode to the reverse-mode gradient code ("forward over reverse"). Newton-Krylov optimisers and many second-order methods need only these products.

== Memory and Checkpointing

When the tape does not fit in memory, *checkpointing* trades recomputation for storage. Save the state only at certain points during the forward sweep; in the backward sweep, recompute each segment's intermediate values from its checkpoint just before they are needed. For a computation of $T$ steps:

- Storing a checkpoint every $sqrt(T)$ steps reduces memory to $O(sqrt(T))$ states at the cost of one extra forward sweep.
- *Binomial checkpointing* (Griewank's `revolve` algorithm) with $c$ checkpoints and $r$ recomputation passes covers $binom(c + r, c)$ steps, which is optimal. Memory and recomputation both grow only logarithmically with $T$.

Deep-learning frameworks expose the same idea as *gradient checkpointing* or *activation recomputation*, keeping only some layers' activations and recomputing the others in the backward pass.

== Differentiating Through Solvers

Programs often contain iterative solvers: Newton's method, a linear solve, an ODE integrator, an optimiser. Differentiating through every iteration works but is wasteful and can be inaccurate, because the derivative of an approximate iterate need not approximate the derivative of the solution. Better is to differentiate the *mathematical* problem.

If $x(p)$ solves $F(x, p) = 0$, the *implicit function theorem* gives $(partial F \/ partial x) (d x \/ d p) = -partial F \/ partial p$. In reverse mode, a scalar objective $J(x(p))$ then has gradient

$ (d J) / (d p) = -lambda^T (partial F) / (partial p), quad "where" quad (partial F / partial x)^T lambda = (partial J / partial x)^T. $

One extra linear solve with the transposed Jacobian (the *adjoint equation*) gives the gradient with respect to any number of parameters. This *adjoint method* is how aerodynamic shapes with thousands of design variables are optimised and how sensitivities of PDE-constrained problems are computed. For linear systems $A x = b$, the rule is $overline(b) = A^(-T) overline(x)$ and $overline(A) = -overline(b) x^T$. For ODEs, the continuous adjoint integrates a backward ODE for $lambda(t)$, the approach popularised in machine learning as *neural ODEs*, though its accuracy depends on how the forward trajectory is reconstructed, and discrete adjoints of the actual integrator are often more reliable.

== Pitfalls

- *Non-differentiable points.* AD differentiates the code, not the mathematics. `abs(x)`, `max` and branches give one-sided or arbitrary values at a kink. Two codes for the same function can also give different derivatives: `sqrt(x * x)` and `abs(x)` are equal, but at 0 AD gives NaN for the first (an infinite slope of $sqrt$ times a zero factor) and 0 or $plus.minus 1$ for the second, depending on the implementation.
- *Piecewise-constant code.* Rounding, table lookups, `floor`, and comparisons have zero derivative almost everywhere, so AD reports gradient 0 where the user expected a useful slope.
- *Approximations differentiated as if exact.* A function computed by a series truncated by a tolerance test, or a solver stopped after a fixed number of iterations, has derivatives that can be inaccurate even when the values are accurate.
- *Perturbation confusion.* Nested forward mode with one shared $epsilon$ silently mixes derivatives of inner and outer functions; tagging each nesting level with its own infinitesimal fixes it.
- *Numerical stability of derivative code.* The derivative of a stable formula can be unstable: the gradient of `log(sum(exp(x)))` evaluated naively overflows just like the function; the derivative of `softmax` needs the same max-subtraction trick.
- *Mutation and aliasing.* Reverse mode needs old values that in-place updates overwrite; tools either copy them to the tape or forbid mutation (as JAX does).

== Tools

#table(
  columns: 3,
  [*Tool*], [*Approach*], [*Typical use*],
  [PyTorch, TensorFlow], [Reverse mode on tensor operations, dynamic graphs], [Machine learning],
  [JAX], [Function transformations (`grad`, `jvp`, `vjp`, `vmap`) on traced pure functions, compiled with XLA], [ML research, scientific computing],
  [CppAD, ADOL-C, CoDiPack, Stan Math], [Operator overloading in C++], [Scientific and statistical codes],
  [Tapenade], [Source transformation of Fortran and C], [Large simulation codes, adjoints],
  [Enzyme], [AD of LLVM IR after optimisation], [Differentiating existing C, C++, Rust, Julia and GPU code],
)

Operator overloading is easy to adopt but adds run-time overhead per operation. Source transformation and compiler-based tools such as Enzyme generate derivative code that the compiler can optimise like any other, which often gives much faster gradients; Enzyme showed that differentiating after optimisation, rather than before, yields substantially faster derivative code.

== Further Reading

Griewank, A. and Walther, A. (2008). _Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation_ (2nd edition). SIAM.

Baydin, A. G. et al. (2018). "Automatic Differentiation in Machine Learning: A Survey." Journal of Machine Learning Research.

Wengert, R. E. (1964). "A Simple Automatic Derivative Evaluation Program." Communications of the ACM.

Linnainmaa, S. (1976). "Taylor Expansion of the Accumulated Rounding Error." BIT Numerical Mathematics.

Griewank, A. (1992). "Achieving Logarithmic Growth of Temporal and Spatial Complexity in Reverse Automatic Differentiation." Optimization Methods and Software.

Curtis, A. R., Powell, M. J. D. and Reid, J. K. (1974). "On the Estimation of Sparse Jacobian Matrices." IMA Journal of Applied Mathematics.

Pearlmutter, B. A. (1994). "Fast Exact Multiplication by the Hessian." Neural Computation.

Siskind, J. M. and Pearlmutter, B. A. (2005). "Perturbation Confusion and Referential Transparency." IFL.

Giles, M. B. and Pierce, N. A. (2000). "An Introduction to the Adjoint Approach to Design." Flow, Turbulence and Combustion.

Moses, W. S. and Churavy, V. (2020). "Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients." NeurIPS.

Chen, R. T. Q. et al. (2018). "Neural Ordinary Differential Equations." NeurIPS.
