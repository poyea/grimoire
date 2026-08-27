#import "../template.typ": xref

= Architecture Evaluation <architecture-evaluation>

An architecture is good only relative to the qualities a system must exhibit, and those qualities are worthless as adjectives ("scalable", "secure") until they are made concrete enough to test a design against. Architecture evaluation is the discipline of doing exactly that: eliciting the quality attributes that matter, expressing them as scenarios, and analysing whether the architecture's decisions support or undermine them, before the expensive parts are built, and continuously afterwards. This chapter covers quality-attribute scenarios, utility trees, ATAM and its lightweight descendants, quantitative approaches, and how to document the results so they survive.

*See also:* #xref("software-architecture", "architectural-styles", label: "Architectural Styles") (the trade-off vocabulary being evaluated), #xref("software-architecture", "evolutionary-architecture", label: "Evolutionary Architecture") (fitness functions as continuous, automated evaluation), _Resilience Patterns_ (availability tactics that scenarios probe).

== Quality Attributes and Why "-ilities" Fail

Functional requirements determine what the system does; *quality attributes* (performance, availability, modifiability, security, usability, testability, deployability...) determine whether the architecture is fit. Two standard observations (Bass, Clements & Kazman, _Software Architecture in Practice_):

- Architecture largely does not determine functionality, almost any structure can compute anything, but it decisively determines quality attributes, and they are where architectures fail.
- Bare attribute names are unanalysable. "The system shall be modifiable" is vacuous: modifiable *by whom*, *in response to what*, *within what cost*? Every system is modifiable with enough money and immodifiable for some change.

The fix is the *quality attribute scenario*, a six-part, testable statement: *source* of stimulus, *stimulus*, *artifact*, *environment*, *response*, and *response measure*. Examples:

- Availability: "A hardware failure (stimulus) of one node (artifact) during peak load (environment) is detected and traffic rerouted (response) with no failed user requests and recovery within 30 seconds (measure)."
- Modifiability: "A developer (source) adds a new payment provider (stimulus) to the payments service (artifact) at design time (environment); the change is isolated to one module (response) and ships within two weeks with no regression elsewhere (measure)."
- Performance: "1,000 concurrent users (source) submit search requests (stimulus) under normal operation (environment); the system serves results (response) with p99 latency under 300 ms (measure)."

The response *measure* is the load-bearing part: it is what makes the scenario falsifiable, and what later becomes a fitness function or an $"SLO"$.

== Utility Trees and Prioritisation

Stakeholders always want every attribute; evaluation requires forcing rank. The *utility tree* (from ATAM) does this: root node "utility", second level the quality attributes, third level refinements, leaves the concrete scenarios, each leaf rated on two dimensions: *business importance* and *architectural risk/difficulty* (typically High/Medium/Low each). The (H, H) leaves, critical to the business and hard for the architecture, are where evaluation effort goes. In practice a system has three to five genuinely *driving* scenarios; an architecture is chosen to serve those, and the rest are checked for "not catastrophically harmed".

Complementary technique: the *Quality Attribute Workshop* (QAW, SEI), a facilitated session run *before* there is an architecture, generating and prioritising scenarios from business goals, so the drivers exist when design starts rather than being reverse-engineered at review time.

== ATAM

The *Architecture Tradeoff Analysis Method* (Kazman, Klein & Clements, SEI, 1998–2000) is the most thoroughly documented evaluation method. Its conceptual outputs matter more than its ceremony:

- *Sensitivity points*: decisions where a quality response is strongly affected by one parameter (replication factor strongly affects availability).
- *Tradeoff points*: decisions that are sensitivity points for multiple attributes in opposite directions (synchronous replication helps consistency, hurts write latency), the places where architecture is actually decided.
- *Risks* and *non-risks*: decisions whose consequences are problematic (or explicitly fine) relative to the driving scenarios; risks aggregate into *risk themes* tied back to business goals.

The method itself runs in two phases over roughly nine steps: present business drivers and the architecture, catalogue the architectural approaches used, build the utility tree, then analyse each high-priority scenario against the approaches, walking the architecture and recording how each decision supports or obstructs the scenario, first with the architects, then again with the broader stakeholder group, finishing with brainstormed and prioritised scenarios from stakeholders. A full ATAM takes several days and a trained evaluation team, which is exactly why most organisations do not run it, and why its ideas usually arrive via lighter vehicles.

== Lightweight Methods

- *Lightweight ATAM / mini-ATAM*: half a day, internal evaluators, top five scenarios only, walk the design against each and record risks and tradeoffs. Captures most of the value for projects that cannot fund the full method.
- *SAAM* (Software Architecture Analysis Method, Kazman et al., 1994): ATAM's predecessor, focused on modifiability, evaluate by mapping change scenarios onto the structure and counting which components each change touches; still a perfectly good five-line technique: "list the ten most likely changes; for each, which modules are touched?" High scatter is the answer you were afraid of.
- *CBAM* (Cost Benefit Analysis Method, Kazman, Asundi & Klein, 2001): extends ATAM with economics, estimate each candidate decision's benefit (utility gain across scenarios) per unit cost, so the portfolio of architectural improvements is chosen by ROI rather than taste.
- *Architecture reviews as risk hunts*: Fairbanks' "risk-driven model" (_Just Enough Software Architecture_, 2010), spend design and evaluation effort proportional to risk; identify the highest risks, apply just enough technique to mitigate them, stop.
- *Decision-centric review*: review ADRs rather than diagrams (see #xref("software-architecture", "evolutionary-architecture", label: "Evolutionary Architecture")), asking for each significant decision: what alternatives were considered, what scenario justifies it, what new risks does it introduce?

A practitioner heuristic worth keeping from all of these: the evaluation's most valuable output is usually the *list of questions the architects could not answer*, missing scenarios, unstated assumptions, and undocumented constraints surface faster in a structured walk-through than in months of construction.

== Quantitative Evaluation

Scenario walk-throughs are qualitative; some attributes admit numbers early:

- *Performance modelling*: back-of-the-envelope capacity maths (arrival rates, service times, Little's law, $L = lambda W$), queueing models for saturation behaviour, and load tests against a walking skeleton. The cheap, high-value version: compute the dominant resource per request (disk seeks, fan-out count, bytes moved) and multiply by peak rate before committing to a design.
- *Availability modelling*: combine component availabilities along the request path, series components multiply ($0.999 times 0.999$), redundant parallel components compound failure probabilities ($1 - (1 - 0.999)^2$ for two independent replicas where one suffices). The model's value is exposing single points of failure and the chain-depth penalty, not predicting four decimal places, since real failures correlate.
- *Static structure metrics*: cyclomatic dependencies, component coupling and cohesion, Martin's instability and abstractness metrics, change-coupling mined from version control (files that always change together but live in different components indicate a misplaced boundary, Tornhill's _Your Code as a Crime Scene_, 2015, operationalises this).
- *Prototypes and spikes*: for the one or two decisions with the highest cost of being wrong (can the database sustain the write load? does the latency budget survive the extra hop?), a focused throwaway experiment beats any amount of analysis.

== Continuous Evaluation

A point-in-time review decays the day it ends. The modern synthesis ties evaluation into delivery:

- The driving scenarios' response measures become *fitness functions* in the pipeline and $"SLO"$s in production (see #xref("software-architecture", "evolutionary-architecture", label: "Evolutionary Architecture")), so the architecture is re-evaluated on every commit and every minute, respectively.
- Error budgets (SRE practice) operationalise the availability scenario: when the budget burns, feature work yields to reliability work, an evaluation outcome with teeth.
- Periodic lightweight reviews (quarterly mini-ATAM on the highest-churn area) catch drift that automation cannot express, especially erosion of intent and accumulation of unrecorded decisions.

== Documenting the Evaluation

Outputs worth writing down, in roughly descending value: the prioritised scenario list with measures; the risks and tradeoff points with owners; ADRs for decisions made or reaffirmed during the review; and an updated architecture description. For the last, *views* are the organising idea (ISO/IEC/IEEE 42010; Kruchten's 4+1 view model, 1995; Rozanski & Woods' viewpoints, 2005): no single diagram serves all stakeholders, so document the structures that answer real questions, module/dependency structure for developers, component-and-connector for runtime reasoning, deployment for operations. The C4 model (Brown, 2011) is the pragmatic mainstream choice: context, container, component, code, with strict rules about what each level may show. A diagram that cannot be checked against the code is decoration; pair each documented structure with the fitness function or test that keeps it honest.

== Pitfalls

- Evaluating against adjectives instead of scenarios; the review becomes opinion exchange.
- Letting the loudest stakeholder's attribute dominate; utility trees exist to force explicit, joint prioritisation.
- Review theatre: a board that approves slide decks but never traces a scenario through the actual design, and is staffed by people too far from the system to find the bodies.
- One-shot evaluation: a clean ATAM in 2023 says nothing about the system shipped in 2026 unless the measures became automated checks.
- Confusing the model with reality: availability arithmetic that assumes independent failures, load models that ignore the real traffic mix, metrics gamed once they become targets (Goodhart's law applies to fitness functions too).
- Skipping the economics: every identified risk costs something to fix; without CBAM-style prioritisation, reviews produce a list of forty findings and zero changes.

== Further Reading

- Bass, L., Clements, P., & Kazman, R. (2021). _Software Architecture in Practice_, 4th ed. Addison-Wesley.
- Clements, P., Kazman, R., & Klein, M. (2001). _Evaluating Software Architectures: Methods and Case Studies_. Addison-Wesley.
- Kazman, R., Klein, M., & Clements, P. (2000). ATAM: Method for architecture evaluation. SEI Technical Report CMU/SEI-2000-TR-004.
- Fairbanks, G. (2010). _Just Enough Software Architecture_. Marshall & Brainerd.
- Kruchten, P. (1995). The 4+1 view model of architecture. _IEEE Software_, 12(6).
- Brown, S. The C4 model for visualising software architecture. c4model.com.
