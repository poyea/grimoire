#import "template.typ": project

#project("Databases")[
  #align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Databases: Theory, Internals & Modern Systems]
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

#include "database/foundations.typ"
#pagebreak()

#include "database/storage-engines.typ"
#pagebreak()

#include "database/recovery-and-logging.typ"
#pagebreak()

#include "database/buffer-pool-and-io.typ"
#pagebreak()

#include "database/concurrency-control.typ"
#pagebreak()

#include "database/isolation-and-consistency-models.typ"
#pagebreak()

#include "database/query-compilation.typ"
#pagebreak()

#include "database/query-optimization.typ"
#pagebreak()

#include "database/joins-and-aggregation.typ"
#pagebreak()

#include "database/vector-and-similarity-search.typ"
#pagebreak()

#include "database/transactions-distributed.typ"
#pagebreak()

#include "database/consensus-and-replication.typ"
#pagebreak()

#include "database/partitioning-and-elasticity.typ"
#pagebreak()

#include "database/weakly-consistent-systems.typ"
#pagebreak()

#include "database/column-stores-and-vectorized-execution.typ"
#pagebreak()

#include "database/lakehouses-and-open-formats.typ"
#pagebreak()

#include "database/streaming-and-incremental-computation.typ"
#pagebreak()

#include "database/time-series-and-graph.typ"
#pagebreak()

#include "database/hardware-aware-design.typ"
#pagebreak()

#include "database/security-and-privacy.typ"
#pagebreak()

#include "database/observability-and-self-driving.typ"
#pagebreak()

#include "database/benchmarking-and-research-methods.typ"
#pagebreak()

= Conclusion

Databases sit at the intersection of theory and systems: relational algebra and AGM bounds dictate what query plans are achievable; consensus impossibility results bound what distributed transactions can guarantee; hardware trends — NVMe, RDMA, CXL, GPUs — keep redrawing the cost model. The chapters in this book cover each layer at research depth so you can reason about, design, and critique modern data systems with full understanding of the tradeoffs.

*Key synthesis:*

- Worst-case-optimal joins, learned indexes, and factorized representations have rewritten what "optimal" means in query processing — Yannakakis, AGM, and Generic Join are now baseline.
- MVCC dominates OLTP because locks force a total order on conflicting operations even when versions expose a satisfiable partial order; Silo, Hekaton, and Cicada illustrate the design space.
- Adya's cycle-based isolation framework subsumes ANSI levels and exposes write skew and long-fork anomalies that SQL standards under-specify.
- FLP impossibility, Spanner's TrueTime, EPaxos, and Flexible Paxos quorums together define the achievable region for distributed transactions and consensus.
- CRDTs, causal+ consistency, and HLCs provide the formal underpinnings for geo-distributed AP systems.
- Vectorized execution, codegen, and morsel-driven parallelism close the gap to hand-written C while preserving compositionality.
- Lakehouses (Iceberg/Delta/Hudi) prove that ACID over object storage is achievable with the right snapshot protocol, separating storage from compute permanently.
- Hardware-aware design (RDMA KV, GPU OLAP, CXL pooling) and learned tuning (Bao, NEO, OtterTune) are the active research frontiers.

== Further Reading

Hellerstein, J., Stonebraker, M., Hamilton, J. (2007). "Architecture of a Database System." Foundations and Trends in Databases.

Bernstein, P., Hadzilacos, V., Goodman, N. (1987). "Concurrency Control and Recovery in Database Systems." Addison-Wesley.

Adya, A. (1999). "Weak Consistency: A Generalized Theory and Optimistic Implementations for Distributed Transactions." PhD thesis, MIT.

Mohan, C. et al. (1992). "ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging." TODS.

Cahill, M., Röhm, U., Fekete, A. (2008). "Serializable Isolation for Snapshot Databases." SIGMOD.

Tu, S. et al. (2013). "Speedy Transactions in Multicore In-Memory Databases." SOSP.

Corbett, J. et al. (2012). "Spanner: Google's Globally-Distributed Database." OSDI.

Thomson, A. et al. (2012). "Calvin: Fast Distributed Transactions for Partitioned Database Systems." SIGMOD.

Lamport, L. (1998). "The Part-Time Parliament." TOCS (Paxos).

Ongaro, D., Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC (Raft).

Moraru, I., Andersen, D., Kaminsky, M. (2013). "There Is More Consensus in Egalitarian Parliaments." SOSP (EPaxos).

Howard, H. et al. (2017). "Flexible Paxos: Quorum Intersection Revisited." OPODIS.

Atul Adya, B. Liskov (1999); Bailis, P. et al. (2014). "Highly Available Transactions: Virtues and Limitations." VLDB.

Shapiro, M. et al. (2011). "Conflict-Free Replicated Data Types." SSS.

Hellerstein, J., Alvaro, P. (2020). "Keeping CALM: When Distributed Consistency Is Easy." CACM.

Ngo, H., Porat, E., Ré, C., Rudra, A. (2018). "Worst-Case Optimal Join Algorithms." JACM.

Neumann, T. (2011). "Efficiently Compiling Efficient Query Plans for Modern Hardware." VLDB.

Leis, V. et al. (2014). "Morsel-Driven Parallelism." SIGMOD.

Kraska, T. et al. (2018). "The Case for Learned Index Structures." SIGMOD.

Malkov, Y., Yashunin, D. (2018). "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW." TPAMI.

Subramanya, S. et al. (2019). "DiskANN." NeurIPS.

McSherry, F., Murray, D., Isaacs, R., Isard, M. (2013). "Differential Dataflow." CIDR.

Budiu, M. et al. (2023). "DBSP: Automatic Incremental View Maintenance for Rich Query Languages." VLDB.

Marcus, R. et al. (2021). "Bao: Making Learned Query Optimization Practical." SIGMOD.

Van Aken, D. et al. (2017). "Automatic Database Management System Tuning Through Large-Scale Machine Learning." SIGMOD (OtterTune).
]
