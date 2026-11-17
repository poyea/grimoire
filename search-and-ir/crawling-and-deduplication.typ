#import "../template.typ": xref

= Web Crawling and Near-Duplicate Detection <crawling-and-deduplication>

A web search engine can only rank what it has fetched, and the web is effectively infinite: calendars that generate a page for every future date, session identifiers that make every URL unique, sites that change by the minute, and a large fraction of pages that are copies or near-copies of others. A crawler must decide what to fetch, in what order, how often to revisit it, and how fast it may hit each site without harming it, all with a finite budget. After fetching, it must recognise duplicates so that the index does not fill with copies. This chapter covers crawler architecture and the URL frontier, politeness and the robots exclusion protocol, URL normalisation and traps, prioritisation, recrawl policies for freshness, exact and near-duplicate detection with shingling, MinHash, locality-sensitive hashing and SimHash with C++ implementations, and the adversarial side of crawling.

*See also:* #xref("search-and-ir", "ranking-classical", label: "Ranking: Classical Models") (PageRank and HITS), #xref("search-and-ir", "inverted-indexes", label: "Inverted Indexes"), #xref("search-and-ir", "distributed-search", label: "Distributed Search Architecture"), #xref("search-and-ir", "vector-search", label: "Vector Search") (locality-sensitive hashing), #xref("networking", "application-protocols", label: "Application Protocols") (HTTP).

== Crawler Architecture

A crawler is a loop: take a URL from the *frontier*, fetch it, parse it, extract links, add new links to the frontier. At scale, each step is a distributed system of its own:

```text
            ┌──────────────── URL frontier ────────────────┐
seeds ────► │ priority queues ─► per-host queues (politeness)│ ──► fetchers ──► content store
            └───────────────────────▲──────────────────────┘        │
                                    │                               ▼
                       seen-URL filter ◄── normalise ◄── link extraction ◄── parser
                                                                    │
                                                         duplicate detection ──► indexer
```

- *Fetchers* make many concurrent HTTP requests (thousands per machine with asynchronous I/O), with a local DNS cache, because DNS resolution would otherwise dominate latency.
- *Parsers* extract links and text, increasingly after rendering JavaScript in a headless browser for pages that build their content client-side, at a cost of one to two orders of magnitude more CPU per page.
- The *seen-URL filter* answers "have we already discovered this URL?" for billions of URLs. Large crawlers use disk-based sorted sets, partitioned by host, with an in-memory cache; Bloom filters save memory at the cost of occasionally skipping a new URL (a false positive means a URL is wrongly treated as seen).
- The work is partitioned across machines by *host*, so that each host's politeness state and seen URLs live on one machine and most links (which point to the same host) stay local.

== Politeness

A crawler with thousands of connections could take down a small site. Polite crawlers:

- obey `robots.txt`, standardised as the Robots Exclusion Protocol in RFC 9309 (2022): `User-agent` groups, `Allow` and `Disallow` rules with longest-match precedence, fetched and cached per host (for up to 24 hours by the RFC's recommendation);
- keep at most one connection to a host at a time and wait between requests, typically several times the previous response time, and honour `Crawl-delay` where supported;
- identify themselves with a descriptive `User-Agent` and a contact URL;
- back off on errors, `429 Too Many Requests` and `503`, and respect `Retry-After`.

Mercator's frontier design separates the two concerns. *Front queues* order URLs by priority; *back queues*, one per host, enforce politeness. A heap of hosts keyed by the earliest time each may be contacted again hands the next eligible host to a fetcher, which takes a URL from that host's back queue and refills it from the front queues. Priority decides which URLs are wanted; politeness decides when each host can be visited.

== URLs, Normalisation, and Traps

Many URL strings name the same resource. Normalisation (following RFC 3986) lowercases the scheme and host, removes default ports, resolves `.` and `..` segments, decodes percent-encoded unreserved characters, and removes fragments. Site-specific rules go further: dropping session identifiers and tracking parameters (`utm_source`), sorting query parameters, and honouring the page's `rel="canonical"` link, which names the URL the site prefers to be indexed.

*Crawler traps* generate unbounded URL spaces: calendars with "next month" links forever, faceted navigation with every combination of filters, relative links that keep appending path segments (`/a/b/a/b/...`), and deliberately adversarial sites. Defences are budgets rather than detection: limit pages per host and per path pattern, limit URL length and path depth, and lower the priority of URLs whose content turns out to duplicate pages already fetched.

== What to Crawl First

With a budget of $N$ fetches out of an effectively infinite web, order matters. Breadth-first crawling from good seeds is a strong baseline: Najork and Wiener found that it discovers high-PageRank pages early, because important pages have many inbound links and are reached in few hops. Better orders use estimates of importance computed online, such as partial PageRank on the graph discovered so far or the OPIC (online page importance) algorithm, which distributes "cash" along links as pages are fetched. *Focused crawlers* prioritise links by a classifier's estimate that the target page is on a given topic, for vertical search engines.

Sites can help: *sitemaps* list URLs with last-modified dates and priorities, and push protocols (such as IndexNow) notify search engines of changes directly, reducing the need to discover and re-check pages.

== Freshness and Recrawling

Pages change: news front pages every few minutes, product pages when prices change, most pages rarely or never. Each refetch costs bandwidth for the crawler and the site, so the crawler must allocate a fixed recrawl budget across pages to keep the index fresh.

A standard model treats changes to page $i$ as a Poisson process with rate $lambda_i$. If page $i$ is refetched at regular intervals with frequency $f_i$, its expected freshness (the fraction of time the stored copy is up to date) is

$ F(lambda_i, f_i) = f_i / lambda_i (1 - e^(-lambda_i \/ f_i)). $

Cho and Garcia-Molina used this model to show a counterintuitive result: allocating refetches *proportionally* to change rates is worse than allocating them *uniformly*, and the optimal policy gives pages that change extremely often *fewer* refetches than moderate changers, because no affordable rate keeps them fresh, so effort is better spent where it buys freshness. In practice change rates are estimated from the history of fetches (how often the content hash changed), weighted by page importance, and combined with signals such as sitemaps and HTTP validators (`ETag`, `Last-Modified`, `304 Not Modified`) that make a refetch of an unchanged page cheap.

== Exact Duplicates

A large fraction of fetched pages are exact copies under different URLs: mirrors, `http` and `https` variants, `www` and bare domains, parameter variations. Hashing the normalised content (for example, after removing markup and whitespace) with a 64- or 128-bit fingerprint detects them, and the index keeps one canonical URL per fingerprint, chosen by signals such as the canonical link, URL shortness and PageRank.

== Near Duplicates

Many more pages are *near* duplicates: the same article with different ads, navigation, timestamps or comment counts; syndicated content; scraped copies. Exact hashes miss them, so near-duplicate detection compares *sets of features*.

*Shingling* represents a document as the set $S$ of its overlapping word $k$-grams (shingles), typically with $k$ from 4 to 9. The similarity of two documents is the Jaccard coefficient of their shingle sets:

$ J(A, B) = (|A inter B|) / (|A union B|). $

Comparing every pair of documents is impossible at web scale. *MinHash* (Broder, 1997) compresses each set into a short signature. For a random hash function $h$, the probability that the minimum hash value over $A$ equals that over $B$ is exactly $J(A, B)$, because the minimum over $A union B$ is equally likely to come from any element of the union, and the two minima agree exactly when it lies in $A inter B$. With $m$ independent hash functions, the fraction of agreeing minima estimates $J$ with standard deviation about $sqrt(J (1 - J) \/ m)$.

```cpp
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

std::uint64_t mix64(std::uint64_t x) {                // splitmix64 finaliser
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::uint64_t hash_text(std::string_view s, std::uint64_t seed) {   // FNV-1a, then mixed
    std::uint64_t h = 0xcbf29ce484222325ull ^ mix64(seed);
    for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
    return mix64(h);
}

std::unordered_set<std::string> word_shingles(const std::vector<std::string>& words, std::size_t k) {
    std::unordered_set<std::string> out;
    for (std::size_t i = 0; i + k <= words.size(); ++i) {
        std::string s = words[i];
        for (std::size_t j = 1; j < k; ++j) s += ' ' + words[i + j];
        out.insert(std::move(s));
    }
    return out;
}

std::vector<std::uint64_t> minhash_signature(const std::unordered_set<std::string>& shingles,
                                             std::size_t num_hashes) {
    std::vector<std::uint64_t> sig(num_hashes, std::numeric_limits<std::uint64_t>::max());
    for (const auto& s : shingles)
        for (std::size_t i = 0; i < num_hashes; ++i)
            sig[i] = std::min(sig[i], hash_text(s, i));
    return sig;
}

double estimated_jaccard(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b) {
    std::size_t equal = 0;
    for (std::size_t i = 0; i < a.size(); ++i) equal += a[i] == b[i];
    return static_cast<double>(equal) / static_cast<double>(a.size());
}
```

For two 300-word documents that differ in 15 words, the true Jaccard similarity of their 4-shingle sets is 0.68, and a 128-hash signature estimates it within about 0.04. Production systems avoid computing $m$ full hashes per shingle by using one hash and $m$ cheap permutations, or *one-permutation hashing*, which splits a single hash's range into $m$ bins.

=== Finding Candidate Pairs with LSH

Signatures make each comparison cheap but do not avoid comparing all pairs. *Locality-sensitive hashing* with *banding* does: split each signature of $m = b r$ values into $b$ bands of $r$ rows, hash each band, and treat documents that collide in any band as candidates. Two documents with Jaccard similarity $s$ become candidates with probability

$ 1 - (1 - s^r)^b, $

an S-shaped curve. With $b = 16$ bands of $r = 8$ rows, pairs with $s = 0.8$ become candidates with probability about 0.95, pairs with $s = 0.5$ about 0.06, and pairs with $s = 0.3$ about 0.001. Choosing $b$ and $r$ places the threshold of the curve; candidates are then verified by comparing full signatures.

=== SimHash

Charikar's *SimHash* produces a single 64-bit fingerprint whose Hamming distance reflects cosine similarity of feature vectors. Each feature (word or shingle, with a weight) is hashed to 64 bits; for each bit position, add the weight if the feature's hash has a 1 there and subtract it otherwise; the fingerprint's bit is 1 where the sum is positive.

```cpp
#include <array>
#include <bit>

std::uint64_t simhash(const std::unordered_set<std::string>& features) {
    std::array<long, 64> votes{};
    for (const auto& f : features) {
        const std::uint64_t h = hash_text(f, 0);
        for (int bit = 0; bit < 64; ++bit) votes[bit] += ((h >> bit) & 1) ? 1 : -1;
    }
    std::uint64_t fingerprint = 0;
    for (int bit = 0; bit < 64; ++bit)
        if (votes[bit] > 0) fingerprint |= std::uint64_t{1} << bit;
    return fingerprint;
}

int hamming(std::uint64_t a, std::uint64_t b) { return std::popcount(a ^ b); }
```

The probability that a bit differs between two fingerprints is $theta \/ pi$, where $theta$ is the angle between the documents' feature vectors, so similar documents have fingerprints that differ in few bits. Google's crawler used 64-bit SimHash over billions of pages and treated fingerprints within Hamming distance 3 as near-duplicates. That threshold catches pages that differ in boilerplate and small edits, not substantially edited copies: the two documents above, with Jaccard similarity 0.68, differ in about a dozen bits, while unrelated documents differ in around 32. Finding all fingerprints within distance $k$ of a query among billions uses the pigeonhole principle: split the 64 bits into $k + 1$ blocks; any fingerprint within distance $k$ matches the query exactly in at least one block, so tables sorted by each block (or permuted copies) find candidates with exact lookups.

MinHash is the better estimator of set overlap and supports adjustable thresholds; SimHash is more compact (8 bytes per document) and suits finding very close duplicates at enormous scale.

== Content Extraction

The useful content of a page is often a small fraction of its HTML: navigation, footers, ads and related-article lists make up the rest. *Boilerplate removal* (text density and link density heuristics, as in Boilerpipe, or learned models over the DOM) extracts the main content before indexing and duplicate detection. Without it, pages on the same site look similar because they share templates, and copies on different sites look different because their templates differ.

== The Adversarial Web

Crawling and indexing are adversarial. *Link farms* and paid links inflate link-based importance; TrustRank and related methods propagate trust from a manually vetted seed set and demote pages far from it. *Cloaking* serves different content to crawlers and to users, detected by fetching pages with different user agents and from different addresses and comparing. *Content spam* stuffs keywords or generates text automatically, a problem that large language models have made cheaper. Crawlers must also protect themselves: malformed HTML, enormous responses, and slow-drip servers that hold connections open require strict limits on size and time per fetch.

The other side of the relationship has changed as well. Sites increasingly restrict crawlers that collect data for training AI models, through `robots.txt` rules for specific user agents and network-level blocking, and the economics of crawling (who benefits, who pays the bandwidth) are once again an open question.

== Pitfalls

- Ignoring `robots.txt` or overloading hosts, which gets the crawler blocked and harms sites.
- No per-host or per-pattern budgets, so traps consume the crawl.
- Recrawling by change rate alone, wasting budget on pages that can never be kept fresh.
- Near-duplicate detection on raw HTML, where shared templates dominate the features.
- Bloom-filter seen sets sized too small, silently dropping new URLs.

== Further Reading

Heydon, A. and Najork, M. (1999). "Mercator: A Scalable, Extensible Web Crawler." World Wide Web.

Najork, M. and Wiener, J. L. (2001). "Breadth-First Crawling Yields High-Quality Pages." WWW.

Cho, J. and Garcia-Molina, H. (2003). "Effective Page Refresh Policies for Web Crawlers." ACM Transactions on Database Systems.

Olston, C. and Najork, M. (2010). "Web Crawling." Foundations and Trends in Information Retrieval.

Koster, M., Illyes, G., Zeller, H. and Sassman, L. (2022). "Robots Exclusion Protocol." RFC 9309.

Broder, A. Z. (1997). "On the Resemblance and Containment of Documents." Compression and Complexity of Sequences.

Broder, A. Z., Glassman, S. C., Manasse, M. S. and Zweig, G. (1997). "Syntactic Clustering of the Web." WWW.

Charikar, M. S. (2002). "Similarity Estimation Techniques from Rounding Algorithms." STOC.

Manku, G. S., Jain, A. and Das Sarma, A. (2007). "Detecting Near-Duplicates for Web Crawling." WWW.

Leskovec, J., Rajaraman, A. and Ullman, J. D. (2020). _Mining of Massive Datasets_ (3rd edition). Cambridge University Press.

Gyöngyi, Z., Garcia-Molina, H. and Pedersen, J. (2004). "Combating Web Spam with TrustRank." VLDB.

Kohlschütter, C., Fankhauser, P. and Nejdl, W. (2010). "Boilerplate Detection Using Shallow Text Features." WSDM.
