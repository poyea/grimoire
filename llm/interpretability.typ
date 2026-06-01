= Interpretability

Interpretability tries to answer: what is *inside* a trained LLM, and how does it compute? The field has matured rapidly from "attention is interpretation" (largely wrong) to *mechanistic interpretability* — a research program that aims to reverse-engineer specific circuits implementing specific behaviors. This chapter covers the toolbox: probing classifiers, the residual stream view, activation patching and causal tracing, sparse autoencoders, dictionary learning, monosemantic features, attention head analysis, and the open problems that limit interpretability as a safety tool.

*See also:* _Transformer Architecture_ (mechanistic concepts assume familiarity with QKV, residual stream), _Safety and Alignment_ (interpretability as a possible lever).

== Why It Is Hard

A modern LLM has $10^(10)$ parameters and runs $10^(12)$ activations per query. Three structural challenges:

1. *Polysemanticity:* a single neuron typically responds to many unrelated concepts. Looking at one neuron tells you little.
2. *Superposition* (Elhage et al. 2022): the model represents *more* features than it has dimensions by encoding them in overlapping (non-orthogonal) directions. Features are not aligned with neurons.
3. *Distributed computation:* most behaviors emerge from interactions of many components — single-head, single-layer, single-MLP explanations are rare.

Linear probes and attention visualizations remain useful as exploratory tools, but the modern field treats them as starting points, not explanations.

== The Residual Stream View

A transformer block reads from and writes to a *residual stream*: a vector per position that is the running sum of all previous components' outputs.

$ x_(l+1) = x_l + "Attn"_l (x_l) + "MLP"_l (x_l) $

Properties (Elhage et al. 2021):

- *Linearity:* the residual stream is a *sum* of contributions; any block's output adds to it without overwriting.
- *Decomposition:* output logits are a linear projection of the final residual stream; backpropagation through the residual decomposes contributions per layer / head.
- *No mixing within a block:* attention and MLP each *read* from the input residual and *add* to it; they do not modify each other's contributions.

This view turns the network into an additive program of small "reads and writes," each of which can be studied in isolation. Most modern interpretability work is built on it.

== Probes

A *probe* is a small classifier (typically logistic regression) trained on activations to predict an external label.

```python
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.lin = nn.Linear(d_in, n_classes)
    def forward(self, h):  # h: [N, d_in]
        return self.lin(h)
```

Train on (activation, label) pairs from a held-out dataset. If the probe's accuracy is high, the information is *linearly decodable* from the activation — necessary but not sufficient for "the model uses this representation."

Probes are useful for:

- Locating *where* a concept first becomes linearly available (across layers).
- Cheap regression analyses (sentiment, syntax, factuality).

They do *not* establish causal use; the model might encode something the probe finds without ever reading it during computation.

== Causal Tracing and Activation Patching

To establish *causal* relevance, intervene on activations and measure the change in output.

=== Activation Patching

Given two inputs $x^"clean"$ (model produces the correct answer) and $x^"corrupt"$ (model produces a wrong answer), run both. Then re-run $x^"corrupt"$ but *patch* in $x^"clean"$'s activation at a specific (layer, position, component). If the output flips toward the clean answer, that activation *causally* contributes.

```python
def patch_and_measure(layer, position, h_clean, h_corrupt, model):
    def hook(module, inp, out):
        out[:, position] = h_clean[:, position]
    handle = model.layers[layer].register_forward_hook(hook)
    out = model(x_corrupt)
    handle.remove()
    return logit_diff(out, target_clean, target_corrupt)
```

Sweep over all (layer, position) pairs to produce a *causal map* — a heatmap of which activations matter. Originally introduced by Meng et al. (2022 — "ROME") to localize factual knowledge in GPT models.

=== Path Patching

Activation patching tells you *what* matters but not *through which path*. Path patching (Wang et al. 2022) restricts the patch to flow only through a specific edge in the computation graph — e.g., "patch attention head L5H2's output, but only into the input of MLP at L7." Refines circuit-level claims.

== Sparse Autoencoders (SAEs)

The polysemanticity + superposition problem motivates *sparse autoencoders*: learn an overcomplete dictionary $W in RR^(D times d)$ ($D >> d$) such that activations decompose into a sparse combination of dictionary directions ("features").

$ h approx W f, quad f in RR^D, quad ||f||_0 "small" $

Training objective:

$ cal(L)_"SAE" = ||h - W f||^2 + lambda ||f||_1, quad f = "ReLU"(W^top h - b) $

The L1 penalty enforces sparsity. After training on a corpus of activations, each dictionary feature ideally fires on a single semantic concept.

Anthropic's "Scaling Monosemanticity" (Templeton et al. 2024) and "Towards Monosemanticity" (Bricken et al. 2023) trained SAEs on Claude 3 and a small toy model respectively; they identified features for "Golden Gate Bridge," "code unsafe to execute," "deception," etc. Steering by amplifying a feature provably modulated the relevant behavior.

=== Variants

- *Top-K SAE* (Gao et al. — OpenAI 2024): force exactly $k$ features active per token; replaces L1 with hard sparsity. More stable training.
- *Gated SAE* (Rajamanoharan et al. — DeepMind 2024): separate the *magnitude* and *gate* heads; reduces the L1-induced shrinkage bias.
- *Jump-ReLU SAE* (Rajamanoharan et al. 2024): learned per-feature threshold.

=== Scaling

SAEs trained on frontier model activations require massive dictionaries (millions of features per layer) to achieve faithful reconstruction. Anthropic, OpenAI, and DeepMind have published SAEs with $D > 10^7$ on flagship models. Cost: a non-trivial fraction of model-training compute.

== Circuits

A *circuit* is a small subgraph of the network — a specific subset of heads and MLPs across layers — that implements a specific behavior.

Examples:

- *Indirect object identification* (Wang et al. 2022): a ~10-component circuit in GPT-2 small that solves "John gave the ball to \_\_\_" → "Mary."
- *Induction heads* (Olsson et al. 2022): pairs of heads that pattern-match A-B...A → B. Underlie in-context learning. Appear during training in a sharp phase transition.
- *Successor heads* (Gould et al. 2023): heads that compute "next item in sequence" across many domains (numbers, months, days).
- *Refusal direction* (Arditi et al. 2024): a single direction in residual stream gates refusal vs. compliance in chat models. Ablating it removes refusals; amplifying it makes the model refuse benign requests.

Circuit-level claims are typically established with a *progressive narrative*: probes locate, patching causally confirms, ablation tests show the component is necessary, and ideally a hand-implementable algorithm matches the model's behavior at that level.

== Attention Head Taxonomy

Heads found across many models:

- *Previous-token heads:* attend to position $i - 1$. Foundational for sequence modeling.
- *Induction heads:* match recent (A, B) pairs and predict the next "B" after the next "A."
- *Successor heads:* sequence-completion.
- *Copy heads:* copy a token from earlier in context to current position.
- *Suppression heads:* down-weight a token to prevent its repetition.
- *Name-mover heads:* implement IOI by attending to the relevant name.

Tools: `TransformerLens` (Neel Nanda) makes it cheap to enumerate and ablate heads on small to medium models.

== Steering and Editing

Once a feature or direction is identified, *steering vectors* can modulate behavior at inference:

```
h ← h + α · v_feature
```

added at a specific layer. Examples:

- *Activation addition / ActAdd* (Turner et al. 2023): "love − hate" applied at layer L5 makes the model more affectionate.
- *Refusal direction ablation* (Arditi et al. 2024): subtract the refusal direction from all residual streams. Bypasses safety training entirely on small models. (A safety concern — interpretability findings transfer to attack.)
- *Feature steering with SAEs* (Templeton 2024): "Golden Gate Claude" amplified the SF Bay feature and made the model claim to be the Golden Gate Bridge.

Model editing:

- *ROME* (Meng et al. 2022): rank-one edit to a single MLP to change a fact. Targeted but brittle.
- *MEMIT* (Meng et al. 2023): batch many ROME-style edits.
- *Linear concept erasure* (Belrose et al. 2023): orthogonally project out a direction to remove a concept from representations.

These methods *demonstrate* that the model uses the identified directions causally. They are not yet a reliable mechanism for safety patches — collateral effects on other behaviors are usually present.

== Tools and Workflow

- `TransformerLens` — load any HF model with hook points exposed on every component.
- `nnsight` — same idea, supports remote inference on hosted models.
- `circuitsvis`, `pyvene` — visualization.
- *Neuronpedia* — community catalog of SAE features for several open models.

A typical mech-interp workflow:

1. Pick a narrow behavior with clear examples (a benchmark with prompt pairs).
2. *Localize* with activation patching: which (layer, position, component) matters?
3. *Decompose* with path patching: through which edges?
4. *Identify* the components — name them, characterize when they fire.
5. *Hypothesize* an algorithm and reproduce model behavior with a hand-written implementation.
6. *Test generalization*: does the circuit explain held-out examples?

== Limits

- *Scale:* most mech-interp results are on GPT-2-small (1.5B) and below. Scaling SAEs to frontier models is feasible but costly; circuit-level work on frontier models has produced fragmentary results, not full reverse-engineering.
- *Faithfulness vs. completeness:* a circuit may be *faithful* (everything it does is real) without being *complete* (lots of relevant computation outside it). Establishing completeness is open.
- *Robustness:* circuits identified at one fine-tune may not persist after another. Interpretability is not yet a stable interface.
- *Adversarial use:* the same techniques that find a refusal direction can ablate it. Interpretability is dual-use.

Despite the limits, mech-interp is currently the most credible *non-behavioral* method to gain insight into what LLMs do internally — and its inclusion in safety roadmaps (Anthropic's Responsible Scaling Policy, OpenAI's preparedness framework) signals it as a serious research bet.

== Further Reading

Elhage, N. et al. (2021). "A Mathematical Framework for Transformer Circuits." Anthropic.

Elhage, N. et al. (2022). "Toy Models of Superposition." Anthropic.

Olsson, C. et al. (2022). "In-context Learning and Induction Heads." Anthropic.

Wang, K. et al. (2022). "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small." ICLR 2023.

Meng, K. et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS. (ROME)

Bricken, T. et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic.

Templeton, A. et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic.

Gao, L. et al. (2024). "Scaling and evaluating sparse autoencoders." OpenAI.

Rajamanoharan, S. et al. (2024). "Improving Dictionary Learning with Gated Sparse Autoencoders." DeepMind.

Arditi, A. et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS.

Turner, A. M. et al. (2023). "Activation Addition: Steering Language Models Without Optimization."

Burns, C. et al. (2023). "Discovering Latent Knowledge in Language Models Without Supervision." ICLR.
