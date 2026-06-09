= Reinforcement Learning

Reinforcement learning (RL) formalises the problem of an agent learning to act in an environment by trial and error. Unlike supervised learning, there is no labelled target: the agent receives a scalar reward signal and must discover, through interaction, a policy that maximises cumulative reward. This chapter develops the mathematical foundations — Markov decision processes, dynamic programming, temporal-difference learning, policy gradients — through to the modern deep RL methods that underpin current large-scale systems.

*See also:* _Optimization_ (gradient methods used in policy gradient and actor-critic), _Probability and Information_ (Bellman expectation, importance sampling), _Loss Functions_ (value regression objectives), _Reasoning Models_ (RLHF and GRPO).

== Markov Decision Processes

A *Markov Decision Process* (MDP) is a tuple $(cal(S), cal(A), P, R, gamma)$:
- $cal(S)$: state space.
- $cal(A)$: action space.
- $P(s' | s, a)$: transition kernel.
- $R(s, a)$: expected immediate reward.
- $gamma in [0, 1)$: discount factor.

The *Markov property* states that $P(s_{t+1} | s_t, a_t, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)$: the future is independent of the past given the present state.

A *policy* $pi(a | s)$ maps states to distributions over actions. The agent's objective is to find $pi^*$ maximising the expected discounted return:

$ G_t = sum_(k=0)^infinity gamma^k R_(t+k+1). $

=== Value Functions

The *state-value function* under policy $pi$ is

$ V^pi (s) = EE_pi [G_t | S_t = s] = EE_pi [R_(t+1) + gamma V^pi (S_(t+1)) | S_t = s]. $

The *action-value function* (Q-function) is

$ Q^pi (s, a) = EE_pi [G_t | S_t = s, A_t = a] = R(s, a) + gamma sum_(s') P(s'|s,a) V^pi(s'). $

The Bellman equation expresses $V^pi$ as a fixed point:

$ V^pi (s) = sum_a pi(a|s) [R(s,a) + gamma sum_(s') P(s'|s,a) V^pi (s')]. $

=== Optimal Value Functions

The *optimal value functions* $V^* (s) = max_pi V^pi (s)$ and $Q^* (s, a) = max_pi Q^pi (s, a)$ satisfy the *Bellman optimality equations*:

$ V^*(s) &= max_a [R(s,a) + gamma sum_(s') P(s'|s,a) V^*(s')], $
$ Q^*(s,a) &= R(s,a) + gamma sum_(s') P(s'|s,a) max_(a') Q^*(s', a'). $

The optimal policy is $pi^*(a|s) = arg max_a Q^*(s, a)$.

== Dynamic Programming

When the model $(P, R)$ is known, dynamic programming solves the MDP exactly.

=== Policy Evaluation

Iterative policy evaluation applies the Bellman operator $cal(T)^pi$ repeatedly:

$ V_(k+1)(s) = sum_a pi(a|s) [R(s,a) + gamma sum_(s') P(s'|s,a) V_k(s')]. $

This converges to $V^pi$ since $cal(T)^pi$ is a $gamma$-contraction in the $sup$-norm.

=== Policy Iteration

*Policy iteration* alternates:
1. *Evaluation*: compute $V^{pi_k}$ to convergence.
2. *Improvement*: $pi_{k+1}(s) = arg max_a Q^{pi_k}(s, a)$.

Policy improvement is monotone; the algorithm converges in finite steps for finite MDPs.

=== Value Iteration

*Value iteration* applies the Bellman optimality operator directly:

$ V_(k+1)(s) = max_a [R(s,a) + gamma sum_(s') P(s'|s,a) V_k(s')]. $

Convergence is geometric: $||V_k - V^*||_infinity <= gamma^k ||V_0 - V^*||_infinity$.

== Temporal-Difference Learning

When the model is unknown, the agent learns from sampled transitions $(s, a, r, s')$.

=== TD(0)

TD(0) updates the value estimate after each step:

$ V(S_t) <- V(S_t) + alpha [R_(t+1) + gamma V(S_(t+1)) - V(S_t)]. $

The bracketed term $delta_t = R_(t+1) + gamma V(S_(t+1)) - V(S_t)$ is the *TD error* — the surprise relative to the current prediction. TD(0) converges to $V^pi$ under standard step-size conditions.

=== SARSA and Q-Learning

*SARSA* (on-policy):
$ Q(S_t, A_t) <- Q(S_t, A_t) + alpha [R_(t+1) + gamma Q(S_(t+1), A_(t+1)) - Q(S_t, A_t)]. $

*Q-learning* (off-policy):
$ Q(S_t, A_t) <- Q(S_t, A_t) + alpha [R_(t+1) + gamma max_(a') Q(S_(t+1), a') - Q(S_t, A_t)]. $

Q-learning converges to $Q^*$ regardless of the behaviour policy (provided all state-action pairs are visited sufficiently).

=== $n$-step Returns and $"TD"(lambda)$

The $n$-step return $G_t^((n)) = R_(t+1) + gamma R_(t+2) + ... + gamma^(n-1) R_(t+n) + gamma^n V(S_(t+n))$ interpolates between TD(0) ($n=1$) and Monte Carlo ($n -> infinity$). $"TD"(lambda)$ uses the $lambda$-return, a geometric mixture of $n$-step returns, implemented efficiently with eligibility traces.

== Deep Q-Networks

DQN (Mnih et al., 2015) represents $Q^*(s, a)$ with a neural network $Q(s, a; theta)$ and trains it with the loss:

$ cal(L)(theta) = EE [(r + gamma max_(a') Q(s', a'; theta^-) - Q(s, a; theta))^2] $

where $theta^-$ are the parameters of a periodically updated *target network*. Two key stabilisation tricks:
- *Experience replay*: store transitions in a buffer, sample mini-batches to break correlations.
- *Target network*: separate parameters $theta^-$ for the TD target, updated every $C$ steps.

Extensions: Double DQN (decoupled action selection and evaluation), Dueling DQN (separate value and advantage streams), Prioritised Experience Replay, Rainbow (combines six improvements).

== Policy Gradient Methods

Policy gradient methods directly optimise $J(theta) = EE_{tau tilde pi_theta} [G_0]$ by gradient ascent.

=== REINFORCE

The *policy gradient theorem* gives:

$ nabla_theta J(theta) = EE_pi [G_t nabla_theta log pi_theta (A_t | S_t)]. $

REINFORCE estimates this with Monte Carlo rollouts:

$ theta <- theta + alpha G_t nabla_theta log pi_theta (A_t | S_t). $

Variance is high; subtracting a *baseline* $b(s)$ reduces variance without adding bias:

$ nabla_theta J(theta) = EE_pi [(G_t - b(S_t)) nabla_theta log pi_theta (A_t | S_t)]. $

The optimal baseline is approximately $V^pi(s)$.

=== Actor-Critic Methods

Actor-critic separates the policy (actor) and value function (critic). The actor updates using the advantage $A(s,a) = Q(s,a) - V(s)$:

$ theta_"actor" <- theta_"actor" + alpha A(S_t, A_t) nabla_theta log pi_theta (A_t | S_t). $

The critic updates by minimising the TD error. A3C (Asynchronous Advantage Actor-Critic) runs parallel workers; A2C is the synchronous variant. GAE (Generalised Advantage Estimation) uses $lambda$-returns to trade off bias and variance in the advantage estimate:

$ hat(A)_t = sum_(l=0)^(T-t-1) (gamma lambda)^l delta_(t+l), quad delta_t = r_t + gamma V(s_(t+1)) - V(s_t). $

== Proximal Policy Optimization

PPO (Schulman et al., 2017) constrains policy updates to avoid destructive large steps. The clipped objective is:

$ cal(L)^"CLIP"(theta) = EE_t [min(r_t(theta) hat(A)_t, "clip"(r_t(theta), 1-epsilon, 1+epsilon) hat(A)_t)] $

where $r_t(theta) = pi_theta(a_t|s_t) / pi_(theta_"old")(a_t|s_t)$ is the probability ratio and $epsilon approx 0.2$. The clip prevents the ratio from deviating too far from 1, keeping updates conservative. PPO is the dominant algorithm for RLHF fine-tuning of language models (see _Reasoning Models_).

== Soft Actor-Critic

SAC (Haarnoja et al., 2018) augments the reward with an entropy bonus:

$ J(pi) = EE [sum_t gamma^t (R(s_t, a_t) + alpha H(pi(dot|s_t)))]. $

Maximum-entropy RL encourages exploration and robustness. SAC maintains two Q-networks (for variance reduction) and a separate temperature parameter $alpha$ tuned automatically. It is the state-of-the-art model-free algorithm for continuous control.

== Model-Based RL

Model-based RL learns a transition model $hat(P)(s'|s,a)$ and uses it for planning. Key methods:
- *Dyna-Q*: interleave real experience with simulated transitions from the learned model.
- *World models* (Dreamer, DreamerV3): learn a compact latent dynamics model and plan entirely in latent space. DreamerV3 achieves strong results on Atari, continuous control, and Minecraft from pixels.
- *MuZero* (Schrittwieser et al., 2020): learns a latent model supporting MCTS planning without a reward model, achieving superhuman performance on Atari and board games.

== Exploration

Exploration is the core challenge in RL. Strategies range from:
- $epsilon$-greedy: take random action with probability $epsilon$.
- UCB (Upper Confidence Bound): $a = arg max_a [Q(a) + c sqrt(log t / N(a))]$.
- Thompson Sampling: sample $Q$ from its posterior and act greedily.
- Intrinsic motivation: add a curiosity bonus (RND, ICM) for visiting novel states.
- Count-based exploration: reward states proportional to $N(s)^(-1/2)$.

== Multi-Agent RL

When multiple agents interact, the environment is non-stationary from each agent's perspective. Key settings:
- *Cooperative*: agents share reward (CTDE: centralised training, decentralised execution — QMIX, MAPPO).
- *Competitive*: zero-sum games; Nash equilibrium policies (AlphaGo, AlphaStar).
- *Mixed*: partial cooperation; general-sum games.

Self-play (training against past versions of itself) underpins AlphaZero and OpenAI Five.

== RL from Human Feedback

RLHF (Christiano et al., 2017) trains LLMs to follow instructions using human preferences. The pipeline: (1) supervised fine-tuning on demonstrations, (2) train a reward model from pairwise preference data, (3) run PPO against the reward model with a KL penalty against the SFT policy. GRPO (Group Relative Policy Optimisation) and DPO eliminate the reward model by directly optimising preference likelihoods. See _Reasoning Models_ for details.

== Further Reading

- Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_, 2nd ed. MIT Press. (freely available online)
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Haarnoja, T. et al. (2018). Soft actor-critic: off-policy maximum entropy deep RL. _ICML_.
- Schrittwieser, J. et al. (2020). Mastering Atari, Go, Chess and Shogi by planning with a learned model. _Nature_, 588.
- Dreamer V3: Hafner, D. et al. (2023). Mastering diverse domains through world models. arXiv:2301.04104.
