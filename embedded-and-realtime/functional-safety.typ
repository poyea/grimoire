#import "../template.typ": xref

= Functional Safety and Certification <functional-safety>

When an embedded system can hurt people (a brake controller, an insulin pump, a flight computer, a robot arm), "it passed our tests" is not an acceptable argument. *Functional safety* is the discipline of showing, with evidence an independent assessor will accept, that the risk from the system's malfunctions has been reduced to a tolerable level. It is part engineering (redundancy, diagnostics, safe states) and part process (requirements traceability, verification at defined rigour, configuration control). This chapter covers the standards and their integrity levels, the distinction between systematic and random faults, the hardware and software safety mechanisms that recur across industries, coverage criteria such as MC/DC, and the evidence a certification case is built from.

*See also:* #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET"), #xref("embedded-and-realtime", "embedded-cpp", label: "C++ for Embedded Systems"), #xref("embedded-and-realtime", "debugging-and-testing", label: "Debugging and Testing Embedded Software"), #xref("formal-methods", "model-checking", label: "Model Checking") (formal methods), #xref("operating-systems", "real-time-scheduling", label: "Real-Time Scheduling") (mixed criticality)

== The Standards Landscape

IEC 61508, *Functional safety of electrical/electronic/programmable electronic safety-related systems*, is the generic parent standard. Most industries have a derived standard that adapts its ideas:

#table(
  columns: 4,
  [*Domain*], [*Standard*], [*Integrity levels (lowest to highest)*], [*Basis of the level*],
  [Generic, industrial], [IEC 61508], [SIL 1 to SIL 4], [Required risk reduction; target failure rate],
  [Road vehicles], [ISO 26262], [QM, ASIL A to ASIL D], [Severity, exposure, controllability],
  [Airborne software], [DO-178C (with DO-330, DO-331, DO-332, DO-333 supplements)], [DAL E to DAL A], [Effect of failure on the aircraft],
  [Airborne hardware], [DO-254], [DAL E to DAL A], [Same as software],
  [Medical device software], [IEC 62304], [Class A, B, C], [Possible injury from a software failure],
  [Railway], [EN 50128 / EN 50716], [SIL 0 to SIL 4], [As IEC 61508],
  [Machinery], [ISO 13849, IEC 62061], [PL a to e, SIL 1 to 3], [Risk graph],
)

The integrity level is decided *before* design, from an analysis of what can go wrong, and it then dictates how much rigour every later activity needs: which coding rules apply, which coverage criterion testing must reach, how independent the reviewers must be, and what failure rate the hardware must achieve.

=== ISO 26262 in Brief

ISO 26262 starts with a *hazard analysis and risk assessment* (HARA). For each hazardous event (for example "unintended full braking at motorway speed") the team rates:

- *Severity* S0 to S3 (no injuries to life-threatening or fatal injuries),
- *Exposure* E0 to E4 (how often the driving situation occurs),
- *Controllability* C0 to C3 (whether a typical driver can avoid harm).

A table maps the combination to an ASIL. S3, E4, C3 gives ASIL D, the most demanding. Each hazard yields a *safety goal* ("avoid unintended braking above 0.3 g"), with its ASIL and a *safe state* (here, braking disabled with a warning to the driver). Safety goals are refined into functional and then technical safety requirements, allocated to hardware and software elements, and traced through design, implementation and test. *ASIL decomposition* lets a requirement at ASIL D be met by two sufficiently independent elements, for example ASIL B(D) plus ASIL B(D), if their independence is demonstrated.

== Two Kinds of Faults

The most useful idea in functional safety is that failures have two fundamentally different origins, and each is handled differently.

*Systematic faults* are design errors: a wrong requirement, a software bug, a misconfigured clock. They are deterministic (the same conditions trigger them every time) and every copy of the product contains them. Redundancy with identical copies does not help, because both copies fail together. They are handled by *process*: rigorous requirements, reviews, coding standards, structural coverage, static analysis, and tool qualification. All software faults are systematic.

*Random hardware faults* are physical: a transistor wears out, a cosmic-ray neutron flips an SRAM bit, a solder joint cracks. They occur at a statistical rate and are handled by *safety mechanisms* that detect or tolerate them, and by quantitative analysis showing that the residual rate of undetected dangerous faults is low enough. ISO 26262 sets hardware metric targets per ASIL:

#table(
  columns: 4,
  [*ASIL*], [*Single-point fault metric*], [*Latent fault metric*], [*Residual failure rate (PMHF)*],
  [B], [at least 90 percent], [at least 60 percent], [below 100 FIT],
  [C], [at least 97 percent], [at least 80 percent], [below 100 FIT],
  [D], [at least 99 percent], [at least 90 percent], [below 10 FIT],
)

One FIT is one failure per $10^9$ device-hours. A PMHF below 10 FIT for ASIL D means less than one undetected dangerous failure per $10^8$ hours of operation. The single-point fault metric is, roughly, the fraction of the hardware's dangerous failure rate that is detected or made safe by safety mechanisms. Achieving 99 percent on an $"MCU"$ requires diagnostics on almost everything: CPU, memories, clocks, power, buses and peripherals.

== Hardware Safety Mechanisms

Safety $"MCU"$s (Infineon AURIX, NXP S32K3 and S32E, ST SPC58 and Stellar, TI Hercules and Sitara, Renesas RH850) are built around these mechanisms, and their safety manuals specify the software needed to complete them:

- *Lockstep cores*: two identical cores execute the same instructions, typically offset by one or two cycles and with physically separated layouts so that one particle strike or local voltage dip cannot corrupt both identically. A comparator checks their outputs every cycle, and a mismatch raises a fault to a safety controller.
- *ECC on SRAM, flash, caches and buses*: single-error correction and double-error detection (SECDED). Correctable errors are counted and reported, because a rising count is a warning sign.
- *Built-in self-test*: logic BIST and memory BIST run at start-up (and sometimes periodically) to detect latent faults in the logic and memories that the runtime mechanisms rely on.
- *Clock and voltage monitors*: an independent oscillator checks the main clock's frequency, and window comparators check every supply rail.
- *Windowed and independent watchdogs*: a watchdog clocked from its own oscillator that resets the device if it is serviced too late *or too early*, which catches a runaway loop that services it continuously.
- *Memory protection*: an $"MPU"$ (and on larger devices bus-level firewalls) that stops a fault in low-integrity software corrupting high-integrity data.
- *Error signalling to the outside*: a dedicated error pin that an external safety supervisor (often a separate small $"MCU"$ or a system basis chip with its own watchdog) can use to force the safe state, independent of the main processor's software.

== Software Safety Mechanisms

Software adds diagnostics where hardware has none, and must guarantee that its own failures are detected.

=== Watchdog with Program Flow Monitoring

A watchdog that is serviced from a timer interrupt proves only that the timer interrupt runs. *Program flow monitoring* (called alive, deadline and logical supervision in AUTOSAR's watchdog manager) checks that the right code ran in the right order within the right time. Each checkpoint folds its identifier into a running signature, and the watchdog is serviced only if the signature at the end of the cycle matches the value computed offline for the correct sequence:

```cpp
#include <cstdint>

class flow_monitor {
public:
    void checkpoint(std::uint32_t id) {
        // Order-sensitive: different sequences give different signatures.
        signature_ = rotate_left(signature_, 5) ^ id;
    }

    // Called once per control cycle from the lowest-priority safety task.
    bool end_of_cycle(std::uint32_t expected) {
        const bool ok = (signature_ == expected);
        signature_ = seed;
        return ok;
    }

private:
    static constexpr std::uint32_t seed = 0x5AFE'0001u;
    static constexpr std::uint32_t rotate_left(std::uint32_t v, unsigned s) {
        return (v << s) | (v >> (32 - s));
    }
    std::uint32_t signature_ = seed;
};

// The expected signature is computed at compile time from the intended order.
consteval std::uint32_t expected_signature() {
    std::uint32_t s = 0x5AFE'0001u;
    for (std::uint32_t id : {0x11u, 0x22u, 0x33u}) {   // read, compute, actuate
        s = ((s << 5) | (s >> 27)) ^ id;
    }
    return s;
}

flow_monitor g_flow;

void control_cycle() {
    g_flow.checkpoint(0x11);  read_inputs();
    g_flow.checkpoint(0x22);  compute_outputs();
    g_flow.checkpoint(0x33);  write_outputs();

    if (g_flow.end_of_cycle(expected_signature())) {
        service_windowed_watchdog();          // only on a correct sequence
    } else {
        enter_safe_state();                   // do not wait for the timeout
    }
}
```

A skipped step, a repeated step, or steps out of order all produce a wrong signature, the watchdog is not serviced, and the device reaches its safe state within one watchdog window.

=== Protecting Critical Data

A bit flip in an uncorrected memory, or a stray write from a bug elsewhere, can silently change a safety-critical variable. Storing a value together with its bitwise complement detects both:

```cpp
#include <concepts>
#include <optional>

template <std::unsigned_integral T>
class redundant_value {
public:
    void set(T v) {
        value_ = v;
        complement_ = static_cast<T>(~v);
    }
    [[nodiscard]] std::optional<T> get() const {
        if (value_ != static_cast<T>(~complement_)) return std::nullopt;  // corrupted
        return value_;
    }

private:
    volatile T value_{};
    volatile T complement_{static_cast<T>(~T{})};
};

redundant_value<std::uint16_t> g_max_torque_nm;
```

`volatile` keeps the compiler from deducing that the two copies always agree and deleting the check. Placing the two copies in different memory banks protects further against a single fault corrupting both.

=== Plausibility and Freedom from Interference

- *Plausibility checks* compare redundant or physically related signals: two pedal position sensors with different transfer functions, a wheel-speed signal against vehicle acceleration, a motor current against commanded torque. A disagreement beyond tolerance means a fault in one of them.
- *End-to-end communication protection* (the AUTOSAR E2E profiles) adds a CRC, a sequence counter and a data identifier to safety-relevant messages, so that corruption, loss, repetition, reordering, delay and masquerading on any bus (#xref("embedded-and-realtime", "fieldbuses", label: "Fieldbuses: CAN, LIN, and Deterministic Ethernet")) are detected regardless of the bus's own integrity.
- *Freedom from interference*: when software of different integrity levels shares an $"MCU"$, the higher-integrity part must be protected *spatially* (the $"MPU"$ prevents lower-integrity code writing its memory), *temporally* (a timing-protected $"RTOS"$ such as AUTOSAR OS with timing protection, or time partitioning, prevents lower-integrity code consuming its CPU time), and in *communication*. Without that, the whole $"MCU"$ must be developed to the highest level present.

== Process Evidence: Requirements, Coverage, Tools

=== Traceability

Every safety requirement must trace down to design elements, code and tests, and every piece of code must trace up to a requirement. Code with no requirement is either unnecessary or evidence of a missing requirement. Tools such as Polarion, codebeamer and DOORS hold the trace links, and an auditor will sample them.

=== Coding Standards

MISRA C:2023 and MISRA C++:2023 are the common baseline. Their rules remove undefined and implementation-defined behaviour, restrict features whose behaviour is hard to analyse (dynamic memory, recursion, unrestricted pointer arithmetic), and require consistent, checkable style. Compliance is checked by tools, and every deviation needs a recorded justification. #xref("embedded-and-realtime", "embedded-cpp", label: "C++ for Embedded Systems") describes a compatible C++ baseline.

=== Structural Coverage and MC/DC

Testing must show not only that requirements are met but that the tests exercised the code. The required criterion rises with the integrity level. DO-178C requires statement coverage at DAL C, decision coverage at DAL B, and *modified condition/decision coverage* (MC/DC) at DAL A. ISO 26262 highly recommends MC/DC at ASIL D.

MC/DC requires that each condition in a decision is shown to *independently affect* the decision's outcome: for each condition there is a pair of tests that differ only in that condition and produce different outcomes. For `if (a && (b || c))`:

#table(
  columns: 5,
  [*Test*], [*a*], [*b*], [*c*], [*Outcome*],
  [1], [T], [T], [F], [T],
  [2], [F], [T], [F], [F],
  [3], [T], [F], [F], [F],
  [4], [T], [F], [T], [T],
)

Tests 1 and 2 show `a` independently matters, tests 1 and 3 show `b`, and tests 3 and 4 show `c`. MC/DC needs at least $n + 1$ tests for $n$ conditions, against $2^n$ for exhaustive combinations, yet it catches the classic error of a condition that has no effect (a wrong operator, a copy-paste duplicate). Coverage is measured on the object code or with qualified instrumentation, and the compiler must not have introduced untested branches, which is one reason DAL A requires analysis down to object code.

=== Tool Qualification

If a tool's output is not independently verified, a bug in the tool could introduce or hide an error. DO-330 and ISO 26262-8 classify tools by their impact and the likelihood that their errors would be detected (ISO 26262's tool confidence levels TCL1 to TCL3), and require qualification of those that could silently cause a safety failure. That is why certified compilers (for example qualified versions of GCC- or Clang-based toolchains), certified $"RTOS"$ kernels (SafeRTOS, QNX OS for Safety, PikeOS, VxWorks Cert, INTEGRITY) and qualified static analysers carry a premium: they arrive with the evidence the project would otherwise have to produce.

=== Formal Methods

DO-333 allows formal verification to replace some testing objectives. Proving properties of a model or of code (absence of run-time errors with Astrée, functional properties with SPARK Ada or Frama-C, protocol and state-machine properties with a model checker) gives guarantees that no finite test campaign can. Airbus has used abstract interpretation to prove the absence of run-time errors in flight-control software for its aircraft. #xref("formal-methods", "model-checking", label: "Model Checking") and #xref("formal-methods", "separation-logic", label: "Separation Logic") cover the techniques.

== Beyond Malfunctions: SOTIF and Security

ISO 26262 addresses hazards caused by *malfunctions*. A perception system that works exactly as designed but misclassifies a pedestrian at dusk has not malfunctioned; its specification was insufficient for that situation. ISO 21448, *Safety of the Intended Functionality* (SOTIF), covers this class of hazard, central to driver assistance and automated driving, by systematically searching for triggering conditions and reducing the unknown unsafe scenarios. Machine-learned components add further questions about data coverage and robustness that standards such as ISO/PAS 8800 (2024) for AI in road vehicles are beginning to answer.

Security and safety also interact: an attacker who can send forged brake commands creates a safety hazard. ISO/SAE 21434 defines a cybersecurity engineering process parallel to ISO 26262, and the two analyses (HARA and threat analysis and risk assessment, TARA) must be coordinated, because a security countermeasure (such as locking out a bus after failed authentication) can itself create a safety hazard.

== The Safety Case

The end product of functional safety work is a *safety case*: a structured argument, supported by evidence, that the system is acceptably safe for its intended use in its intended environment. Notations such as Goal Structuring Notation (GSN) make the argument explicit: a top-level claim ("the brake controller is acceptably safe"), decomposed through strategies ("argument over each identified hazard") into sub-claims, each supported by evidence (analyses, test reports, review records, tool qualification data). An independent assessor evaluates the argument and samples the evidence. The safety case is maintained for the product's whole life, and every change in the field (including every firmware update, #xref("embedded-and-realtime", "bootloaders-and-updates", label: "Bootloaders, Secure Boot, and Firmware Updates")) requires an impact analysis against it.

== Further Reading

IEC (2010). "IEC 61508: Functional safety of electrical/electronic/programmable electronic safety-related systems." Standard, 2nd ed.

ISO (2018). "ISO 26262: Road vehicles: Functional safety." Standard, 2nd ed.

RTCA (2011). "DO-178C: Software Considerations in Airborne Systems and Equipment Certification." Standard.

ISO (2022). "ISO 21448: Road vehicles: Safety of the intended functionality." Standard.

Hayhurst, K., Veerhusen, D., Chilenski, J., Rierson, L. (2001). "A Practical Tutorial on Modified Condition/Decision Coverage." NASA/TM-2001-210876.

MISRA (2023). "MISRA C:2023 Guidelines for the use of the C language in critical systems." Standard.

Smith, D., Simpson, K. (2020). "The Safety Critical Systems Handbook." Butterworth-Heinemann, 5th ed.

Kelly, T., Weaver, R. (2004). "The Goal Structuring Notation: A Safety Argument Notation." DSN Workshop on Assurance Cases.

Souyris, J., Wiels, V., Delmas, D., Delseny, H. (2009). "Formal Verification of Avionics Software Products." FM 2009, LNCS 5850.
