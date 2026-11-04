#import "../template.typ": xref

= Quantum Cryptography and Networks <quantum-communication>

Quantum mechanics threatens today's public-key cryptography through Shor's algorithm, and it also offers a different kind of security: key distribution whose secrecy rests on physics rather than on the hardness of a mathematical problem. Beyond key distribution, networks that share entanglement between distant nodes would link quantum computers, enable new sensing and timing schemes, and allow computation on a remote quantum server without revealing the input. This chapter covers quantum key distribution from BB84 to device-independent protocols, the attacks on real implementations and their countermeasures, the fundamental rate-loss limit of optical channels, quantum repeaters and memories, the state of deployed networks, and why most security agencies currently recommend post-quantum cryptography over quantum key distribution.

*See also:* #xref("quantum-computing", "quantum-information", label: "Density Matrices, Channels, and Quantum Information") (no-cloning, teleportation, Bell inequalities), #xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography"), #xref("cryptography-and-security", "key-exchange-and-pki", label: "Key Exchange and PKI"), #xref("networking", "link-layer", label: "Link Layer")

== BB84

The first quantum key distribution (QKD) protocol, by Bennett and Brassard (1984), uses single photons (or other two-level systems) and two conjugate bases:

1. Alice sends a random bit encoded in a randomly chosen basis: rectilinear ($|0 chevron.r$, $|1 chevron.r$) or diagonal ($|+ chevron.r$, $|- chevron.r$).
2. Bob measures each photon in a basis he chooses at random.
3. Over an *authenticated* classical channel, they announce their bases (not the bits) and keep only the positions where the bases matched, about half. This is *sifting*.
4. They compare a random sample of sifted bits to estimate the *quantum bit error rate* (QBER).
5. If the QBER is low enough, they run *error correction* (Cascade or LDPC codes), leaking some information, then *privacy amplification* (hashing the key with a randomly chosen universal hash function to a shorter key) to remove whatever an eavesdropper could know.

Security comes from no-cloning and from the disturbance caused by measurement. An eavesdropper who measures in the wrong basis randomises the result, and she cannot know the basis in advance. The simplest attack, *intercept-resend*, measures every photon in a random basis and resends what she saw; it introduces a 25 percent error rate in the sifted key:

```cpp
#include <cmath>
#include <random>

struct bb84_result {
    int sifted_bits = 0;
    int errors = 0;
    double qber() const { return sifted_bits ? static_cast<double>(errors) / sifted_bits : 0.0; }
};

// Ideal single-photon BB84 with an optional intercept-resend eavesdropper.
bb84_result simulate_bb84(int photons, bool eavesdrop, std::mt19937_64& rng) {
    std::bernoulli_distribution coin(0.5);
    bb84_result r;
    for (int i = 0; i < photons; ++i) {
        const bool bit = coin(rng);
        const bool alice_basis = coin(rng);
        bool photon_bit = bit;
        bool photon_basis = alice_basis;
        if (eavesdrop) {
            const bool eve_basis = coin(rng);
            // Measuring in the wrong basis yields a uniformly random result.
            const bool eve_bit = (eve_basis == photon_basis) ? photon_bit : coin(rng);
            photon_bit = eve_bit;          // Eve resends in her own basis
            photon_basis = eve_basis;
        }
        const bool bob_basis = coin(rng);
        const bool bob_bit = (bob_basis == photon_basis) ? photon_bit : coin(rng);
        if (bob_basis == alice_basis) {    // sifting
            ++r.sifted_bits;
            if (bob_bit != bit) ++r.errors;
        }
    }
    return r;
}

double binary_entropy(double p) {
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
}

// Asymptotic secret fraction of the sifted key for BB84 (Shor-Preskill).
double bb84_secret_fraction(double qber) {
    return std::fmax(0.0, 1.0 - 2.0 * binary_entropy(qber));
}
```

The *Shor-Preskill* security proof (2000) shows that, in the limit of long keys, a fraction $1 - 2 h(Q)$ of the sifted key is secret against *any* attack allowed by quantum mechanics, where $h$ is the binary entropy and $Q$ the QBER. The fraction reaches zero at $Q approx 11$ percent, so above that error rate (with one-way post-processing) no key can be extracted. Real systems include finite-key corrections, which require large blocks of raw key before a key can be produced at all.

The authenticated classical channel is essential and often overlooked. Without it, a man-in-the-middle simply runs QKD separately with each party. Authentication needs a pre-shared secret key (QKD then works as key *expansion*) or a digital signature scheme, which reintroduces computational assumptions.

== From Ideal Protocols to Real Devices

The security proofs assume idealised devices. Real ones deviate, and most practical attacks exploit the gap:

- *Photon-number splitting*: practical sources are attenuated lasers emitting weak coherent pulses, and some pulses contain two or more photons. Eve can split off one photon, store it, and measure it after the bases are announced, learning the bit without disturbance. The countermeasure, *decoy states* (Hwang, 2003; Lo, Ma and Chen, 2005), sends pulses of several randomly chosen intensities; Eve's attack affects them differently, revealing her presence. Decoy states are standard in all deployed systems.
- *Detector attacks*: single-photon detectors can be *blinded* with bright light so that they behave as classical detectors controlled by Eve (Lydersen et al., 2010), which broke commercial systems completely. *Measurement-device-independent* QKD (MDI-QKD; Lo, Curty and Qi, 2012) removes all detector side channels by having both Alice and Bob send states to an untrusted relay that performs a Bell measurement.
- *Trojan-horse and side-channel attacks* probe the transmitter with injected light, or exploit timing and spectral differences between states. Isolators, filters and careful engineering mitigate them, and certification schemes test for them.

At the far end of the trust spectrum, *device-independent* QKD derives security from an observed violation of a Bell inequality alone (#xref("quantum-computing", "quantum-information", label: "Density Matrices, Channels, and Quantum Information")), with no model of the devices at all. Ekert's 1991 protocol (E91) introduced the idea of entanglement-based key distribution. Device-independent QKD was demonstrated in 2022 between trapped ions a couple of metres apart, at very low rates; it is the strongest form of security but far from practical deployment.

== Loss and the Rate-Distance Limit

Optical fibre loses about 0.2 dB per kilometre at telecom wavelengths, so transmission $eta$ falls by a factor of 10 every 50 km. Photons cannot be amplified, because amplification would be cloning. Pirandola, Laurenza, Ottaviani and Banchi (2017) showed that for *any* point-to-point protocol over a lossy channel, the secret-key capacity is

$ K <= -log_2(1 - eta) approx 1.44 eta "bits per channel use" $

the *PLOB bound*. Key rates therefore decrease exponentially with distance: practical point-to-point QKD reaches megabits per second over metropolitan distances, but only bits per second or less beyond a few hundred kilometres. *Twin-field QKD* (Lucamarini et al., 2018) has both parties send to a central node that detects single-photon interference, achieving rates scaling as $sqrt(eta)$ and beating the PLOB bound for a direct link; experiments have exceeded 1000 km of fibre at very low rates.

To go further without trusting intermediate nodes requires a *quantum repeater*.

== Quantum Repeaters

A repeater chain divides a long link into short segments and builds entanglement across the whole distance:

1. *Entanglement generation*: create entangled pairs across each short segment, retrying until success, and store each half in a *quantum memory*.
2. *Entanglement swapping*: a Bell measurement on the two memories at an intermediate node teleports the entanglement, joining two short links into one longer link (Briegel et al., 1998).
3. *Entanglement purification* (or, in later generations, error correction): consume several noisy pairs to distil fewer, higher-fidelity pairs, because each swap degrades fidelity.

With memories that hold their states long enough for classical signals to confirm successes on neighbouring segments, the rate falls only polynomially with distance. The requirements are demanding: memories with coherence times of seconds or longer, efficient interfaces between memories and telecom photons, and high-fidelity local operations. Leading platforms include atomic ensembles, trapped ions, and colour centres in diamond (nitrogen-vacancy and silicon-vacancy centres). Milestones include a three-node network with entanglement swapping between NV-centre nodes in Delft (Pompili et al., 2021) and entanglement of silicon-vacancy memories over a 35 km deployed fibre loop (Knaut et al., 2024). No repeater has yet beaten direct transmission over long distances in a deployed setting.

== Deployed Networks

Absent repeaters, deployed networks use *trusted nodes*: QKD runs over each link, and keys are relayed through intermediate stations where they exist in plaintext and must be physically protected. China's Beijing-Shanghai backbone spans about 2000 km this way, and similar networks exist or are being built in Europe (EuroQCI), South Korea, the UK and elsewhere, mostly for government and financial users.

*Satellites* avoid fibre loss over the long part of the path, because loss in vacuum and thin atmosphere is mostly diffraction. China's *Micius* satellite demonstrated satellite-to-ground QKD over more than 1200 km in 2017 and relayed keys between ground stations in China and Austria, though as a trusted node itself. Satellite links work only at night (daylight swamps the detectors) and in clear weather, which limits availability.

== Should You Use QKD?

National security agencies, including the US NSA, the UK NCSC, France's ANSSI and Germany's BSI, have published positions that are cautious about QKD for general use and recommend post-quantum cryptography (#xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography")) as the primary defence against quantum attacks. Their reasons:

- QKD provides only key distribution, not authentication; it still needs pre-shared keys or classical signatures.
- Security depends on implementations that have repeatedly been broken by side-channel attacks, and certification is immature.
- It requires dedicated hardware and often dedicated fibre, with limited distance, and trusted nodes reintroduce exactly the insiders that end-to-end cryptography removes.
- It is easy to deny service: an attacker who disturbs the channel stops key generation.

QKD's advantage is that its security does not rest on computational assumptions that a future algorithm might break, and that keys exchanged today cannot be recorded and decrypted later by a better computer. For most systems, PQC delivers adequate protection with existing infrastructure, and hybrid deployments (QKD keys combined with PQC key exchange) are used where both are available.

== Beyond Key Distribution

A full *quantum internet* (Wehner, Elkouss and Hanson, 2018) would distribute entanglement between arbitrary nodes on demand, enabling more than cryptography:

- *Distributed quantum computing*: linking modules of a quantum computer through photonic interconnects, which most hardware roadmaps already rely on to scale beyond one chip or one trap.
- *Blind quantum computing*: a client with minimal quantum capability delegates a computation to a remote quantum server, which learns nothing about the input, the algorithm or the output.
- *Entanglement-enhanced sensing and timing*: networks of entangled clocks and sensors, and very long baseline telescopes that interfere starlight collected at distant sites through shared entanglement.
- *Position verification, secure voting and other protocols* that are impossible classically under some assumptions.

The near-term network stages are modest: prepare-and-measure networks (QKD), then networks distributing entanglement between a few nodes with memories, then networks with fault-tolerant repeaters. The infrastructure layers being developed (link-layer protocols for heralded entanglement, network stacks, scheduling of memory resources) mirror classical networking but must contend with the fact that the payload cannot be copied or buffered indefinitely.

== Further Reading

Bennett, C., Brassard, G. (1984). "Quantum Cryptography: Public Key Distribution and Coin Tossing." Proceedings of IEEE International Conference on Computers, Systems and Signal Processing.

Ekert, A. (1991). "Quantum Cryptography Based on Bell's Theorem." Physical Review Letters.

Shor, P., Preskill, J. (2000). "Simple Proof of Security of the BB84 Quantum Key Distribution Protocol." Physical Review Letters.

Lo, H.-K., Ma, X., Chen, K. (2005). "Decoy State Quantum Key Distribution." Physical Review Letters.

Lydersen, L. et al. (2010). "Hacking Commercial Quantum Cryptography Systems by Tailored Bright Illumination." Nature Photonics.

Lo, H.-K., Curty, M., Qi, B. (2012). "Measurement-Device-Independent Quantum Key Distribution." Physical Review Letters.

Pirandola, S., Laurenza, R., Ottaviani, C., Banchi, L. (2017). "Fundamental Limits of Repeaterless Quantum Communications." Nature Communications.

Lucamarini, M., Yuan, Z., Dynes, J., Shields, A. (2018). "Overcoming the Rate-Distance Limit of Quantum Key Distribution without Quantum Repeaters." Nature.

Briegel, H.-J., Dür, W., Cirac, J. I., Zoller, P. (1998). "Quantum Repeaters: The Role of Imperfect Local Operations in Quantum Communication." Physical Review Letters.

Wehner, S., Elkouss, D., Hanson, R. (2018). "Quantum Internet: A Vision for the Road Ahead." Science.

Liao, S.-K. et al. (2017). "Satellite-to-Ground Quantum Key Distribution." Nature.
