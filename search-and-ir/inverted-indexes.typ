#import "../template.typ": xref

= Inverted Indexes

The inverted index is the central data structure of full-text search: a mapping from each term in the vocabulary to the list of documents that contain it. Virtually every search engine — from Lucene and Elasticsearch to the web-scale indexes at Google and Bing — is built on this structure, refined over five decades with compression schemes, skip pointers, and segment-based architectures. This chapter covers index anatomy, construction algorithms, posting compression, and the Lucene segment model.

*See also:* _Query Processing_ (traversing postings at query time), #xref("search-and-ir", "ranking-classical", label: "Ranking: Classical Models") (statistics stored in the index), #xref("search-and-ir", "vector-search", label: "Vector Search") (the analogous structure for dense embeddings), and the _Databases_ volume's chapters on B-trees and LSM-trees (Lucene segments are an LSM variant).

== Anatomy of an Inverted Index

An inverted index has two parts:

- The *dictionary* (lexicon): the set of distinct terms, each with its document frequency $"df"(t)$ and a pointer to its postings list.
- The *postings lists*: for each term $t$, a sorted list of document identifiers containing $t$, optionally with per-document payloads.

A posting may carry, in increasing order of cost:
1. The document ID alone (boolean retrieval).
2. The term frequency $"tf"(t, d)$ (ranked retrieval).
3. Positions of each occurrence (phrase and proximity queries).
4. Per-field frequencies and arbitrary payloads (field-weighted ranking).

Heaps' law estimates vocabulary size as $V approx k N^beta$ with $k in (10, 100)$ and $beta approx 0.5$: a 1-billion-token collection has on the order of a few million distinct terms. Zipf's law states that the $i$-th most frequent term has collection frequency proportional to $1 / i$, so a handful of terms own enormous postings lists while the long tail is short — a skew that drives every compression and traversal decision below.

=== Dictionary Structures

The dictionary must support exact lookup, and often prefix lookup (for wildcard and autocomplete). Common choices:

- *Sorted string array* with binary search: compact, cache-friendly, supports prefix ranges.
- *Hash table*: $O(1)$ exact lookup, no prefix support.
- *Finite state transducer (FST)*: a minimal acyclic automaton mapping terms to ordinals or offsets. Shared prefixes _and_ suffixes are stored once, so an FST is often smaller than the raw concatenated terms. Lucene stores its term dictionary as an FST index over on-disk term blocks, keeping memory use per segment to a few kilobytes per field.
- *Trie / burst trie*: fast prefix traversal; used during in-memory index construction (SPIMI variants).

== Index Construction

Building an index over a collection larger than memory requires external-memory algorithms.

=== BSBI: Blocked Sort-Based Indexing

1. Read documents until a memory budget fills, accumulating (termID, docID) pairs.
2. Sort the block by (termID, docID) and write it to disk as a run.
3. After all blocks, perform a multi-way merge of the runs into the final index.

With block size $M$ pairs and $T$ total pairs, the cost is dominated by sorting, $O(T log M)$ in memory plus a linear merge. BSBI requires a global term-to-termID mapping, which itself may not fit in memory.

=== SPIMI: Single-Pass In-Memory Indexing

SPIMI (Heinz & Zobel, 2003) drops the global mapping: each block builds its own dictionary (a hash of term strings to growable postings buffers), writes a complete mini-index sorted by term, and the final merge matches terms by string. SPIMI is faster than BSBI in practice (no global sort of pairs, postings appended directly) and is essentially what Lucene does when it flushes an in-memory segment.

=== Distributed Construction with MapReduce

Web-scale indexing was the original motivating example for MapReduce (Dean & Ghemawat, 2004):

- *Map*: parse a document partition, emit (term, (docID, tf, positions)) pairs.
- *Shuffle*: group by term (term-partitioned index) or by document range (document-partitioned index).
- *Reduce*: concatenate and compress each term's postings, write index shards.

Production engines almost always choose *document partitioning* — each shard indexes a disjoint subset of documents and every query fans out to all shards — because it isolates failures, balances load under term skew, and keeps per-document updates local. Term partitioning gives lower fan-out but suffers from Zipf-skewed hot shards.

== Postings Compression

Postings dominate index size. Document IDs in a sorted list are stored as *deltas* (d-gaps): $d_1, d_2 - d_1, d_3 - d_2, ...$, which are small for frequent terms — precisely the lists that matter most.

=== Variable-Byte Encoding

Each integer is split into 7-bit groups; the high bit of each byte flags continuation. The value 137 becomes two bytes. Variable-byte is byte-aligned, simple, and fast to decode, but wastes space on small gaps (minimum one byte) and decodes one value per branch. It was the workhorse of early Lucene and remains common for positions.

=== PForDelta

PForDelta (Zukowski et al., 2006) compresses blocks of 128 integers at once. Choose a bit width $b$ such that, say, 90% of the values fit in $b$ bits; pack those values into a tight bit array, and store the *exceptions* (values needing more than $b$ bits) separately as patches. Decoding is branch-free SIMD unpacking plus a short patch loop — hundreds of millions of integers per second. Lucene's current default postings format uses a PForDelta variant for doc-ID blocks.

=== Elias-Fano

Elias-Fano encodes a monotone sequence of $n$ values bounded by $u$ in at most $n (2 + ceil(log_2 (u / n)))$ bits, within a fraction of a bit of the information-theoretic optimum. Split each value into low bits (stored verbatim) and high bits (stored as a unary-coded bucket histogram). Crucially it supports $O(1)$ random access and efficient `nextGEQ(x)` — skip to the first posting $>= x$ — without decompressing the block, which is exactly the operation conjunctive query processing needs. Partitioned Elias-Fano (Ottaviano & Venturini, 2014) adapts to local density and is used in production at Meta and in the PISA research engine.

=== Skip Lists

For long postings lists, a conjunction such as `rare AND common` should not scan the common term's list linearly. *Skip pointers* embed a sparse secondary index every $approx sqrt(L)$ postings (or one pointer per compressed block), allowing `nextGEQ` to jump over blocks whose maximum doc ID is too small. Multi-level skip lists generalize this to logarithmic search. Block-aligned skips compose naturally with PForDelta: skip data stores each block's last doc ID, and only blocks that may contain a match are decoded. Skip entries also carry *block-max impact scores* used by Block-Max WAND (next chapter).

== Positional Indexes

A positional posting stores the ordinal positions of each term occurrence: $(d, "tf", (p_1, ..., p_"tf"))$. Positions typically multiply index size by 2–4$times$ but enable:

- *Phrase queries*: intersect postings, then check that positions satisfy $p_j^((2)) = p_i^((1)) + 1$.
- *Proximity scoring*: reward documents where query terms appear within a small window.

A cheaper alternative for common phrases is indexing word *n-grams* as terms; engines like Google historically combined a positional index with phrase-optimized auxiliary indexes for frequent two-word phrases.

== Lucene Segment Architecture

Lucene — the engine inside Elasticsearch, OpenSearch, and Solr — never updates an index file in place. The design is a log-structured merge over immutable *segments*:

1. Incoming documents accumulate in an in-memory buffer (a SPIMI-style hash of postings builders).
2. On flush (memory threshold or commit), the buffer is written as a new immutable segment: term dictionary FST, postings, stored fields, norms, doc values.
3. Deletes are recorded as a per-segment bitset ("live docs"); an update is a delete plus a reinsert. Deleted documents are skipped at read time and physically reclaimed only at merge.
4. A background *merge policy* (tiered merging by default) selects segments of similar size and rewrites them into one, amortizing to $O(log N)$ rewrite passes per document over its lifetime.

Searches execute per-segment and combine results, so a query's cost grows with segment count — the reason merge tuning matters operationally. A *near-real-time refresh* makes a flushed segment searchable within about a second without an expensive fsync-backed commit. The same immutable-segment pattern appears in LSM-tree storage engines (see the _Databases_ volume); Lucene applies it to an inverted index instead of a key-value store.

== Index Size in Practice

#table(
  columns: 3,
  [*Component*], [*Typical share*], [*Notes*],
  [Doc-ID postings], [10–25% of text], [delta + PForDelta],
  [Frequencies], [5–10%], [small integers],
  [Positions], [30–60%], [largest component when enabled],
  [Term dictionary], [1–2%], [FST-compressed],
  [Stored fields / doc values], [varies], [for retrieval and sorting, not matching],
)

A well-compressed non-positional index is commonly 10–20% of the raw text size; with positions, 40–70%.

== Further Reading

- Manning, C., Raghavan, P., & Schütze, H. (2008). _Introduction to Information Retrieval_, chs. 1–5. Cambridge University Press.
- Zobel, J., & Moffat, A. (2006). Inverted files for text search engines. _ACM Computing Surveys_.
- Heinz, S., & Zobel, J. (2003). Efficient single-pass index construction for text databases. _JASIST_.
- Dean, J., & Ghemawat, S. (2004). MapReduce: simplified data processing on large clusters. _OSDI_.
- Ottaviano, G., & Venturini, R. (2014). Partitioned Elias-Fano indexes. _SIGIR_.
- McCandless, M., Hatcher, E., & Gospodnetić, O. (2010). _Lucene in Action_, 2nd ed. Manning.
