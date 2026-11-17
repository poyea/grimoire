#import "../template.typ": xref

= Query Understanding <query-understanding>

Queries are short, misspelled, ambiguous and written in the user's vocabulary rather than the documents'. Median web queries are two or three words; a product search for `red nike runing shoes 10` mixes a colour, a brand, a misspelled category and a size. *Query understanding* is everything a search system does to the query before retrieval: correcting it, completing it while it is typed, segmenting it into concepts, classifying its intent, extracting structure, and rewriting or expanding it so that it matches the right documents. Small improvements here move quality metrics more than large improvements in ranking, because no ranker can rank documents that retrieval never found. This chapter covers spelling correction with the noisy channel model and a C++ symmetric-delete candidate generator, query auto-completion, segmentation, intent and entity recognition, query expansion and relevance feedback, rewriting with logs and with language models, and how to evaluate it all.

*See also:* #xref("search-and-ir", "text-analysis", label: "Text Analysis and Tokenisation"), #xref("search-and-ir", "query-processing", label: "Query Processing"), #xref("search-and-ir", "learning-to-rank", label: "Learning to Rank") (click data), #xref("search-and-ir", "rag-and-search-systems", label: "RAG and Search Systems"), #xref("coding", "tries", label: "Tries").

== The Pipeline

A typical query understanding pipeline runs in a few milliseconds before retrieval:

+ *Normalisation*: the same analysis as indexing (#xref("search-and-ir", "text-analysis", label: "Text Analysis and Tokenisation")).
+ *Spelling correction*: detect and fix misspellings, deciding whether to correct silently, suggest, or leave the query alone.
+ *Segmentation*: group words into concepts (`new york` `times square`).
+ *Classification*: intent (navigational, informational, transactional), vertical or category, sensitivity, language.
+ *Entity and attribute extraction*: brands, places, people, sizes, prices, dates, turned into structured filters or boosts.
+ *Rewriting and expansion*: synonyms, abbreviations, relaxation when results are scarce, alternative formulations.

Each stage can be rules, statistics from query logs, or learned models, and production systems mix all three.

== Spelling Correction

Ten to fifteen percent of web queries contain a misspelling. The standard framework is the *noisy channel* model: the user intended a correct query $c$, and a noisy process produced the observed query $q$. The best correction is

$ hat(c) = op("arg max")_c P(c | q) = op("arg max")_c P(q | c) P(c). $

- The *language model* $P(c)$ comes from query logs and the indexed text: frequent queries and words are likely intentions. Word-level n-grams supply context, so that `pain killer` is preferred over `pain kilter` and `apple pie` corrects differently from `apple pi`.
- The *error model* $P(q | c)$ scores how likely the typo is: edit distance as a baseline, refined with confusion statistics learned from logs (which letters are substituted for which, keyboard adjacency, phonetic confusions such as `ph` for `f`), and the observation that most misspellings are within edit distance two and rarely in the first letter.

The decision is as important as the candidate. Correcting a rare but valid word (a surname, a product code, a new slang term) frustrates users more than leaving a typo. Systems compare the corrected and original queries' likelihoods and result quality, auto-correct only with high confidence, show "did you mean" in between, and always let the user search for the original.

=== Generating Candidates

Comparing the query against every dictionary word is too slow. Two efficient approaches:

- *Levenshtein automata* (Schulz and Mihov): build an automaton accepting all strings within distance $k$ of the query and intersect it with the dictionary stored as a finite-state transducer. Lucene's fuzzy queries work this way.
- *Symmetric delete* (SymSpell): precompute, for every dictionary word, all strings obtained by deleting up to $k$ characters, and index them. At query time, generate the query's own deletions and look them up. Two strings within edit distance $k$ share a common deletion string, so the lookup finds every candidate; a final distance check removes false matches. It uses only deletions, so the number of variants per word grows as $O(L^k)$ rather than with the alphabet size.

```cpp
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Optimal string alignment distance (Levenshtein plus adjacent transposition).
int osa_distance(const std::string& a, const std::string& b) {
    const std::size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> d(n + 1, std::vector<int>(m + 1));
    for (std::size_t i = 0; i <= n; ++i) d[i][0] = static_cast<int>(i);
    for (std::size_t j = 0; j <= m; ++j) d[0][j] = static_cast<int>(j);
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j) {
            const int cost = a[i - 1] == b[j - 1] ? 0 : 1;
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1])
                d[i][j] = std::min(d[i][j], d[i - 2][j - 2] + 1);
        }
    return d[n][m];
}

class symmetric_delete_index {
    int max_distance_;
    std::unordered_map<std::string, std::vector<std::string>> deletes_;   // variant -> words
    std::unordered_map<std::string, long> frequency_;

    static void deletions(const std::string& w, int depth, std::unordered_set<std::string>& out) {
        if (depth == 0 || w.size() <= 1) return;
        for (std::size_t i = 0; i < w.size(); ++i) {
            std::string shorter = w.substr(0, i) + w.substr(i + 1);
            if (out.insert(shorter).second) deletions(shorter, depth - 1, out);
        }
    }

public:
    explicit symmetric_delete_index(int max_distance) : max_distance_(max_distance) {}

    void add(const std::string& word, long count) {
        frequency_[word] += count;
        std::unordered_set<std::string> variants{word};
        deletions(word, max_distance_, variants);
        for (const auto& v : variants) deletes_[v].push_back(word);
    }

    // Candidates within max_distance, closest first, then most frequent.
    std::vector<std::pair<std::string, int>> lookup(const std::string& query) const {
        std::unordered_set<std::string> variants{query};
        deletions(query, max_distance_, variants);
        std::unordered_set<std::string> seen;
        std::vector<std::pair<std::string, int>> hits;
        for (const auto& v : variants) {
            const auto it = deletes_.find(v);
            if (it == deletes_.end()) continue;
            for (const auto& word : it->second) {
                if (!seen.insert(word).second) continue;
                const int d = osa_distance(query, word);
                if (d <= max_distance_) hits.emplace_back(word, d);
            }
        }
        std::sort(hits.begin(), hits.end(), [&](const auto& x, const auto& y) {
            if (x.second != y.second) return x.second < y.second;
            return frequency_.at(x.first) > frequency_.at(y.first);
        });
        return hits;
    }
};
```

With a dictionary of `running` (frequent), `ruining` and `turning`, the query `runing` returns `running` and `ruining` at distance 1 (ordered by frequency) and `turning` at distance 2. Real systems then score the top candidates with the full noisy channel model, including the context of neighbouring words, and handle word-boundary errors (`newyork`, `new yor k`) by also considering splits and joins.

== Query Auto-Completion

Suggesting completions as the user types shapes the queries that are issued, often more than any later stage. The basic problem: given a prefix, return the top $k$ completions by score, within a few milliseconds per keystroke.

- A *trie* (or a compressed variant, or a finite-state transducer as in Lucene's suggesters) stores the candidate queries. Storing at each node the top-$k$ completions in its subtree, precomputed offline, makes lookup $O(|"prefix"| + k)$. Alternatively, store each subtree's maximum score and run a best-first search, which supports larger $k$ and score updates.
- *Scores* start from popularity in the query log, decayed over time so that trending queries rise, and are adjusted for context: the user's location, language, previous queries in the session, and personal history.
- *Fuzzy completion* tolerates typos in the prefix by traversing the trie with a bounded edit-distance automaton.
- *Filtering* removes offensive, dangerous, defamatory and spammy suggestions, a policy problem as much as a technical one, since suggestions appear to come from the service itself.

Completions are only as good as the candidate set, which is typically past queries with a minimum frequency (to protect privacy and remove noise) plus generated candidates from entity names, product titles, or a language model for long-tail prefixes.

== Segmentation and Structure

Segmentation groups query words into units that should be matched together: `new york times square` is `new york` + `times square`, not `new york times` + `square`. Unsupervised methods score candidate segments by how much more often the words co-occur than chance (pointwise mutual information over query logs or n-gram counts), and dynamic programming finds the best segmentation. Segments become phrase constraints or proximity boosts at retrieval time.

In domain search the more valuable structure is *attributes*. A product query `red nike running shoes size 10 under 100` contains a colour, brand, category, size and price constraint. *Query tagging*, a sequence-labelling problem solved with CRFs and now with fine-tuned transformers, assigns each token a slot, and the result becomes structured filters and boosts against catalogue fields. Precision matters: turning a soft preference into a hard filter on a mis-tagged token returns nothing.

== Intent and Classification

Broder's taxonomy divides web queries into *navigational* (reach a specific site: `facebook login`), *informational* (learn something), and *transactional* (do something: buy, download). The class changes the answer: navigational queries want one result at the top, informational ones want diverse authoritative sources, transactional ones want product or service pages. Search engines also classify queries by *vertical* (news, images, local, shopping, video), deciding which specialised indexes to query and which result blocks to show, by freshness need (`election results` wants today's documents), and by sensitivity (medical, financial, adult content). Classifiers are trained on query logs with labels from clicks, human annotation, or weak supervision, and they must be fast because they run on every query.

== Expansion and Relevance Feedback

Vocabulary mismatch, where the user's words differ from the documents' (`heart attack` versus `myocardial infarction`), limits lexical retrieval. Expansion adds related terms to the query.

- *Rocchio* feedback (1971) moves the query vector toward relevant documents and away from non-relevant ones: $q' = alpha q + beta overline(d)_R - gamma overline(d)_N$.
- *Pseudo-relevance feedback* (PRF) assumes the top $k$ results of the initial query are relevant without asking the user, extracts their most distinctive terms, and reissues the expanded query. The *RM3* relevance model is the strong standard baseline: it estimates a term distribution from the top documents weighted by their query likelihood, and interpolates it with the original query. PRF improves average effectiveness, but it hurts queries whose initial results are poor (*query drift*), and it doubles retrieval cost.
- *Document expansion* moves the work to index time: doc2query generates likely queries for each document with a sequence-to-sequence model and appends them to the document text, improving lexical retrieval without any query-time cost.

Dense retrieval (#xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval")) addresses vocabulary mismatch differently, by matching in an embedding space, but expansion remains valuable for lexical and hybrid systems.

== Rewriting from Logs and with Language Models

Query logs are the richest source of rewrites. Users reformulate unsuccessful queries within a session (`cheap flights nyc` followed by `cheap flights new york`), and when the reformulation is followed by satisfied clicks, the pair is evidence for a rewrite. Mining such pairs, plus queries that lead to clicks on the same documents (*click graphs*), yields synonyms, abbreviations and spelling variants specific to the domain, which hand-written lists never cover.

Large language models offer rewriting without logs:

- *Query rewriting* by prompting: expand abbreviations, fix grammar, decompose multi-part questions, or generate several alternative formulations whose results are merged.
- *HyDE* (hypothetical document embeddings) asks a model to write a hypothetical answer document and embeds that instead of the query, since a document-like text lands closer to real documents in embedding space.
- *query2doc* appends a generated pseudo-document to the query for lexical or dense retrieval.

These help most for long, conversational or underspecified queries, and are costly in latency (a model call before retrieval) and risk (hallucinated content steering retrieval toward wrong documents). Systems often apply them selectively, for queries a classifier predicts will benefit, and cache rewrites for frequent queries.

== Evaluating Query Understanding

Each component can be evaluated alone (precision and recall of spelling corrections, accuracy of intent labels, tagging F1), but the measure that matters is end-to-end: does the search result improve? Useful signals:

- *Null and low result rate*: the fraction of queries returning nothing or almost nothing, a direct symptom of mismatch.
- *Reformulation rate*: how often users immediately rewrite their query, a proxy for failure.
- Offline relevance metrics on a judged query set, with and without the component.
- Online A/B tests on click-through, abandonment and downstream conversions.

A component that improves its own accuracy can still hurt end-to-end quality, for example a more aggressive spelling corrector that is right more often but damages the rare queries where it is wrong. Evaluating per query segment (head, torso, tail; navigational versus informational) exposes such trade-offs.

== Pitfalls

- Auto-correcting rare but valid words, names and codes.
- Hard filters from mis-tagged attributes, returning zero results.
- Pseudo-relevance feedback applied to every query, including those with poor initial results.
- Suggestions and rewrites learned from logs without filtering offensive or manipulated content.
- LLM rewriting on the critical path for every query without measuring the latency and quality impact.
- Evaluating components in isolation and missing end-to-end regressions.

== Further Reading

Kernighan, M. D., Church, K. W. and Gale, W. A. (1990). "A Spelling Correction Program Based on a Noisy Channel Model." COLING.

Brill, E. and Moore, R. C. (2000). "An Improved Error Model for Noisy Channel Spelling Correction." ACL.

Cucerzan, S. and Brill, E. (2004). "Spelling Correction as an Iterative Process that Exploits the Collective Knowledge of Web Users." EMNLP.

Schulz, K. U. and Mihov, S. (2002). "Fast String Correction with Levenshtein Automata." International Journal on Document Analysis and Recognition.

Cai, F. and de Rijke, M. (2016). "A Survey of Query Auto Completion in Information Retrieval." Foundations and Trends in Information Retrieval.

Broder, A. (2002). "A Taxonomy of Web Search." ACM SIGIR Forum.

Rocchio, J. J. (1971). "Relevance Feedback in Information Retrieval." In _The SMART Retrieval System_, Prentice Hall.

Lavrenko, V. and Croft, W. B. (2001). "Relevance-Based Language Models." SIGIR.

Nogueira, R. et al. (2019). "Document Expansion by Query Prediction." arXiv:1904.08375.

Gao, L., Ma, X., Lin, J. and Callan, J. (2023). "Precise Zero-Shot Dense Retrieval without Relevance Labels." ACL.

Wang, L., Yang, N. and Wei, F. (2023). "Query2doc: Query Expansion with Large Language Models." EMNLP.
