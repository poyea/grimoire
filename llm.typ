#import "template.typ": project

#project("LLM")[
  #align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Large Language Models: Internals, Training & Serving]
      ]
    ]
  )
]

#outline(
  title: "Table of Contents",
  depth: 2,
)

#emph[Enjoy.]

#pagebreak()

#include "llm/introduction.typ"
#pagebreak()

#include "llm/transformer-architecture.typ"
#pagebreak()

#include "llm/pretraining.typ"
#pagebreak()

#include "llm/finetuning.typ"
#pagebreak()

#include "llm/rlhf.typ"
#pagebreak()

#include "llm/inference-optimization.typ"
#pagebreak()

#include "llm/quantization.typ"
#pagebreak()

#include "llm/serving-systems.typ"
#pagebreak()

#include "llm/evaluation.typ"
#pagebreak()

= Conclusion

Modern LLMs combine architectural simplicity (transformer blocks) with engineering complexity at every level — data pipelines, distributed training, alignment, inference optimization, and serving. The chapters in this book cover each layer precisely so you can build, tune, and deploy models with full understanding of the tradeoffs.

*Key synthesis:*

- The transformer's power comes from attention's ability to route information globally, trained via the simple next-token prediction objective at scale.
- Chinchilla scaling laws determine compute-optimal (N, D) pairs; modern models over-train relative to compute-optimal for inference efficiency.
- Fine-tuning via LoRA/QLoRA makes alignment and specialization accessible without full-parameter updates.
- RLHF (PPO) and DPO align models to human preferences; GRPO enables reasoning without a critic.
- Inference is memory-bandwidth-bound at decode; KV cache management (GQA, PagedAttention, quantization) is the critical engineering lever.
- Quantization (GPTQ, AWQ, FP8) achieves near-lossless compression; INT4 is the practical floor for most use cases.
- Production serving requires continuous batching, disaggregated prefill/decode, and careful SLA-aware scheduling.

== Further Reading

Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.

Brown, T. et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.

Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS.

Touvron, H. et al. (2023). "LLaMA 2." arXiv:2307.09288.

Hu, E. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

Rafailov, R. et al. (2023). "Direct Preference Optimization." NeurIPS 2023.

Dao, T. (2023). "FlashAttention-2." ICLR 2024.

Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.

Frantar, E. et al. (2022). "GPTQ: Accurate Post-Training Quantization." ICLR 2023.

DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948.
]
