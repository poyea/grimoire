#import "../template.typ": xref

= Network Information Theory <network-information-theory>

Classical information theory analyses a single sender communicating with a single receiver. Network information theory generalises this to systems with multiple senders, receivers, and shared resources: broadcast channels, multiple-access channels, interference channels, relay networks, and distributed source coding. The field reveals that separation of source and channel coding is no longer optimal in multi-user settings, and that cooperation between terminals can dramatically increase capacity.

*See also:* #xref("machine-learning-foundations", "information-theory", label: "Information Theory") (entropy, channel capacity, rate-distortion), _Cryptography_ (secrecy capacity, wiretap channels).

== Multi-User Channel Models

The five canonical channel models are:

#table(
  columns: 3,
  [*Model*], [*Senders*], [*Receivers*],
  [Multiple Access (MAC)], [Multiple ($K$)], [One],
  [Broadcast (BC)], [One], [Multiple ($K$)],
  [Interference (IC)], [Multiple], [Multiple (paired)],
  [Relay], [One + Relay], [One],
  [Wiretap], [One], [Legitimate + Eavesdropper],
)

For each model the central question is: what rate tuples $(R_1, ..., R_K)$ are simultaneously achievable with vanishing error probability?

== Multiple-Access Channel

=== Definition and Capacity Region

In the two-user MAC, senders $X_1$ and $X_2$ transmit independently to a single receiver $Y$ through channel $p(y | x_1, x_2)$. The *capacity region* is the closure of rate pairs $(R_1, R_2)$ satisfying

$ R_1 &<= I(X_1; Y | X_2), \
  R_2 &<= I(X_2; Y | X_1), \
  R_1 + R_2 &<= I(X_1, X_2; Y) $

for some product distribution $p(x_1) p(x_2)$.

*Sum capacity* $C_"sum" = max I(X_1, X_2; Y)$ is achieved by successive interference cancellation (SIC): decode one user treating the other as noise, subtract, then decode the second.

=== Gaussian MAC

For the two-user AWGN MAC with power constraints $P_1, P_2$ and noise $sigma^2$,

$ C_"sum" = (1/2) log_2(1 + (P_1 + P_2) / sigma^2). $

The individual rate bounds depend on the corner points of the pentagon capacity region.

== Broadcast Channel

=== Degraded Broadcast Channel

The BC has one sender transmitting to $K$ receivers. With degraded channels $X -> Y_1 -> Y_2$ (receiver 1 is stronger), the capacity region is

$ R_1 &<= I(X; Y_1 | U), $
$ R_2 &<= I(U; Y_2), $

for some auxiliary $U -> X -> (Y_1, Y_2)$, where $U$ carries the message for the weaker receiver and $X$ adds the stronger receiver's message on top. This is *superposition coding*.

=== Non-Degraded BC

For general BCs the capacity region is given by Marton's inner bound (1979), which uses *coding with auxiliary random variables*. Whether Marton's bound is tight for all BCs remains open. For Gaussian BCs, superposition coding is optimal.

== Interference Channel

The two-user interference channel has pairs $(X_1, Y_1)$ and $(X_2, Y_2)$ where each receiver observes both transmitted signals. The general capacity region is unknown, making the IC one of the central open problems in network information theory.

*Known results:*
- *Very strong interference* ($I(X_1; Y_2) >= I(X_1; Y_1)$): treating interference as noise is suboptimal; decode and cancel.
- *Strong interference*: the Han-Kobayashi scheme achieves capacity.
- *Weak interference*: treating interference as noise is approximately optimal within a constant gap (Etkin-Tse-Wang, 2008).
- *Degrees of freedom*: at high SNR, the IC achieves $K/2$ DoF (half the interference-free capacity), a result proved via interference alignment.

=== Interference Alignment

*Interference alignment* (Cadambe-Jafar, 2008) shows that $K$ users on a fully connected interference channel can each achieve $1/2$ degree of freedom, giving total DoF $= K/2$.

The key insight: design transmit directions so that interference from all unintended users collapses into a low-dimensional subspace at each receiver, leaving a complementary subspace free for the desired signal.

== Relay Channel

In the relay channel, a relay helps the source communicate to the destination. The best-known strategies are:

- *Decode-and-Forward (DF)*: relay fully decodes the source message and retransmits. Achieves capacity for degraded relay channels.
- *Compress-and-Forward (CF)*: relay compresses its observation and forwards the compressed version. Useful when the relay-destination link is strong.
- *Amplify-and-Forward (AF)*: relay amplifies and retransmits; low latency but noise is also amplified.

The *max-flow min-cut bound* upper-bounds the capacity of the relay channel; DF achieves this bound for the degraded case.

== Distributed Source Coding

=== Slepian-Wolf Theorem

Two correlated sources $X$ and $Y$ encode separately but the decoder receives both. *Slepian-Wolf* (1973) states the achievable rate region is

$ R_X &>= H(X | Y), $
$ R_Y &>= H(Y | X), $
$ R_X + R_Y &>= H(X, Y). $

*Remarkably*, the sum rate equals $H(X, Y)$, the same as if the sources could jointly encode. Distributed compression is as efficient as joint compression, even without communication between encoders.

=== Wyner-Ziv: Lossy Compression with Side Information

When the decoder has side information $Y$ and compression of $X$ is lossy, the *Wyner-Ziv rate-distortion function* equals the rate-distortion function with the decoder having access to $Y$ at the encoder:

$ R_"WZ"(D) = R(D | Y) = min_(p(hat(x)|x,y) : EE[d(X,hat(X))] <= D) I(X; hat(X) | Y). $

This *no rate loss theorem* from having the side information only at the decoder, not the encoder, is a cornerstone of distributed video coding.

== Capacity with Feedback

For discrete memoryless channels, feedback does not increase capacity (Shannon, 1956). However, feedback:
- Simplifies coding schemes (Schalkwijk-Kailath for AWGN achieves doubly-exponential error exponent).
- Increases capacity of multi-user channels (feedback can enlarge the MAC capacity region).
- Enables interactive protocols.

== Secrecy Capacity

=== Wiretap Channel

In Wyner's wiretap channel (1975), a legitimate receiver $Y$ and eavesdropper $Z$ observe degraded versions of the input $X$. The *secrecy capacity* is

$ C_s = max_(p(x)) [I(X; Y) - I(X; Z)]^+. $

Secure communication is possible at non-zero rate whenever the eavesdropper's channel is noisier than the legitimate receiver's, without any secret key. This is *information-theoretic security*, holding against computationally unbounded adversaries.

=== Semantic Security

A code is *semantically secure* if no (computationally unbounded) eavesdropper can distinguish ciphertexts. Semantic security over wiretap channels with Gaussian noise requires capacity-approaching codes with randomised encoding.

== Network Coding

In a multi-hop network, traditional routing forwards packets unchanged. *Network coding* (Ahlswede et al., 2000) allows intermediate nodes to transmit linear combinations of received packets. For the butterfly network, network coding achieves the multicast capacity $min$-cut bound, which routing cannot.

For general multicast networks, the multicast capacity is the minimum cut between source and any sink. Linear network coding over finite fields achieves this capacity. Random linear coding is practical and achieves capacity with high probability for large field sizes.

== Degrees of Freedom and Capacity at High SNR

The *degrees of freedom* (DoF) or *multiplexing gain* of a network is

$ "DoF" = lim_(P -> infinity) C(P) / ((1/2) log P). $

DoF counts how many independent streams the network can support at high SNR. Key results:
- AWGN point-to-point: 1 DoF.
- $K times K$ MIMO: $K$ DoF (spatial multiplexing).
- $K$-user interference channel: $K/2$ DoF (interference alignment).
- $K$-user MAC: 1 DoF (bottleneck at receiver).

== Information-Theoretic Limits of Learning

=== Fano's Inequality

For a Markov chain $theta -> X^n -> hat(theta)$,

$ P_e >= 1 - (I(theta; X^n) + 1) / (log |Theta|) $

where $P_e$ is the probability of estimation error. Fano's inequality lower-bounds sample complexity: to distinguish $|Theta|$ hypotheses, one needs $n >= c dot log |Theta| / I(theta; X^n)$ samples. This underlies minimax lower bounds in estimation and learning theory.

=== Channel Simulation and Common Information

*Wyner's common information* $C(X; Y)$ is the minimum rate of a common source from which both $X$ and $Y$ can be generated by independent local operations. It characterises the coordination capacity between two terminals and appears in distributed simulation of correlated sources.

== Further Reading

- El Gamal, A., & Kim, Y.-H. (2011). _Network Information Theory_. Cambridge University Press.
- Cover, T. M., & Thomas, J. A. (2006). _Elements of Information Theory_, 2nd ed. Wiley. Chapter 14 onward.
- Cadambe, V. R., & Jafar, S. A. (2008). Interference alignment and degrees of freedom of the $K$-user interference channel. _IEEE Transactions on Information Theory_, 54(8).
- Slepian, D., & Wolf, J. K. (1973). Noiseless coding of correlated information sources. _IEEE Transactions on Information Theory_, 19(4).
- Ahlswede, R., Cai, N., Li, S.-Y. R., & Yeung, R. W. (2000). Network information flow. _IEEE Transactions on Information Theory_, 46(4).
- Tse, D., & Viswanath, P. (2005). _Fundamentals of Wireless Communication_. Cambridge University Press.
