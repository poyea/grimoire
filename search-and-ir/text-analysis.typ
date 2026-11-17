#import "../template.typ": xref

= Text Analysis and Tokenisation <text-analysis>

Before a search engine can index a document or match a query, it must turn text into units it can compare: terms for an inverted index, token ids for a neural model. That step decides what can ever match. If the index stores `naïve` and the query says `naive`, or the document says `running` and the query `run`, or a Japanese sentence is split at the wrong place, no ranking function can recover the lost match. Text analysis is unglamorous and language-specific, and it causes a large share of real-world search quality bugs. This chapter covers Unicode and normalisation with a C++ UTF-8 decoder, word segmentation across scripts, case folding, stemming and lemmatisation, stop words, compounds, n-grams and phonetic matching, language identification, and the subword tokenisers used by neural models, with a C++ byte-pair encoder.

*See also:* #xref("search-and-ir", "query-processing", label: "Query Processing") (analysis chains at query time), #xref("search-and-ir", "inverted-indexes", label: "Inverted Indexes"), #xref("search-and-ir", "query-understanding", label: "Query Understanding"), #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval").

== Characters, Bytes, and Unicode

Text arrives as bytes in some encoding, nearly always UTF-8 today. UTF-8 encodes each Unicode *code point* (up to U+10FFFF) in one to four bytes: ASCII as itself, other characters with a lead byte that states the length and continuation bytes of the form `10xxxxxx`. A robust decoder rejects malformed input rather than passing garbage into the index: truncated sequences, stray continuation bytes, *overlong* encodings (such as `C0 80` for U+0000, a classic way to smuggle characters past filters), UTF-16 surrogate code points (U+D800 to U+DFFF), and values above U+10FFFF.

```cpp
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

// Decode UTF-8 into code points, rejecting malformed, overlong and surrogate sequences.
std::optional<std::vector<char32_t>> decode_utf8(std::string_view bytes) {
    std::vector<char32_t> out;
    std::size_t i = 0;
    while (i < bytes.size()) {
        const auto b0 = static_cast<unsigned char>(bytes[i]);
        int length;
        char32_t cp;
        char32_t min_value;
        if (b0 < 0x80)              { length = 1; cp = b0;        min_value = 0; }
        else if ((b0 >> 5) == 0x6)  { length = 2; cp = b0 & 0x1F; min_value = 0x80; }
        else if ((b0 >> 4) == 0xE)  { length = 3; cp = b0 & 0x0F; min_value = 0x800; }
        else if ((b0 >> 3) == 0x1E) { length = 4; cp = b0 & 0x07; min_value = 0x10000; }
        else return std::nullopt;                         // stray continuation or invalid lead
        if (i + length > bytes.size()) return std::nullopt;
        for (int k = 1; k < length; ++k) {
            const auto b = static_cast<unsigned char>(bytes[i + k]);
            if ((b >> 6) != 0x2) return std::nullopt;     // not a continuation byte
            cp = (cp << 6) | (b & 0x3F);
        }
        if (cp < min_value || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            return std::nullopt;                          // overlong, out of range, surrogate
        out.push_back(cp);
        i += length;
    }
    return out;
}
```

Code points are not characters as users see them. `é` can be one code point (U+00E9) or two (`e` followed by the combining acute accent U+0301), and an emoji with a skin-tone modifier or a family emoji is several code points joined together. Unicode defines *grapheme clusters* for user-perceived characters (UAX \#29), and any processing that splits, truncates or counts "characters" should use them.

=== Normalisation

Because the same text has several encodings, analysis begins with *Unicode normalisation*:

#table(
  columns: 3,
  [*Form*], [*Effect*], [*Example*],
  [NFC], [Canonical composition: precomposed characters where they exist], [`e` + U+0301 becomes `é`],
  [NFD], [Canonical decomposition], [`é` becomes `e` + U+0301],
  [NFKC], [Compatibility composition: also folds presentational variants], [`ﬁ` ligature becomes `fi`, full-width `Ａ` becomes `A`, `²` becomes `2`],
  [NFKD], [Compatibility decomposition], [As NFKC, decomposed],
)

Search engines usually apply NFKC (or NFKC with case folding, `NFKC_Casefold`), so that full-width Latin letters common in East Asian text match ordinary ones. *Accent folding* goes further, removing diacritics (decompose with NFD, drop combining marks), so that `naïve` matches `naive`. Whether that is right depends on the language: in English accents rarely matter, while in Spanish or German they can distinguish words (`año` "year" and `ano` "anus"; `schön` and `schon`). Engines often index both folded and unfolded forms in separate fields and score exact matches higher.

=== Case Folding

Lowercasing is not a byte operation. *Case folding*, defined by Unicode for case-insensitive comparison, maps German `ß` to `ss` (full folding), Greek final sigma `ς` to `σ`, and handles characters whose uppercase and lowercase forms differ in length. Some rules are language-specific: in Turkish and Azerbaijani, uppercase `I` lowercases to dotless `ı`, and `i` uppercases to dotted `İ`, so applying the default mapping to Turkish text breaks matching for some of the most common words. ICU implements all of this, and production analysers use it rather than hand-written tables.

== Segmentation

Splitting text into words is easy for languages that separate words with spaces and hard for those that do not.

- *Space-delimited scripts* still have hard cases: apostrophes (`don't`, `O'Brien`, `l'homme`), hyphens (`state-of-the-art`, `e-mail`), numbers with separators (`3.14`, `1,000,000`), URLs, email addresses, product codes (`RTX-4090`), and code identifiers (`getUserName`, `user_name`). UAX \#29 defines default word boundaries; search analysers add rules for the domain, such as splitting camel case for code search while keeping the original token too.
- *Chinese and Japanese* have no spaces. Dictionary-based or statistical segmenters (MeCab, Sudachi and Kuromoji for Japanese; jieba and ICU's dictionary for Chinese) split text into words, but segmentation is ambiguous and errors cause missed matches. A robust alternative indexes overlapping character *bigrams*, so that any substring query of length two or more can match, at the cost of a larger index and less precise ranking. Many systems combine both.
- *Thai, Lao, Khmer and Burmese* also lack spaces and need dictionary-based segmentation.
- *Korean* separates phrases with spaces but attaches particles to nouns, so morphological analysis (such as Nori in Lucene) strips them.

== Stemming and Lemmatisation

Inflected forms of a word (`connect`, `connected`, `connecting`, `connection`) usually express the same need. Two approaches reduce them to a common form:

- *Stemming* strips suffixes by rule, without a dictionary. The *Porter* stemmer (1980) applies five ordered steps of suffix rules conditioned on a measure of the remaining stem, turning `connection` and `connected` into `connect` and `generalization` into `gener`. Stems need not be words. *Snowball* provides stemmers for dozens of languages; *Krovetz* is a lighter, dictionary-checked stemmer for English that produces real words and makes fewer aggressive errors.
- *Lemmatisation* maps each word to its dictionary form using a lexicon and part-of-speech information: `better` to `good`, `mice` to `mouse`, `saw` to `see` or `saw` depending on context. It is more precise and more expensive, and it matters more for morphologically rich languages (Finnish, Turkish, Arabic) than for English.

Stemming trades precision for recall. Aggressive stemmers *overstem* (`university` and `universe` both become `univers`; `organization` and `organ` both become `organ`) and *understem* (irregular forms such as `ran` and `run` stay apart). Measured effects on English retrieval quality are modest, positive on average but negative for some queries; effects are much larger for highly inflected languages. Most production systems index both a stemmed field and an exact field, so that exact matches can be rewarded.

== Stop Words, Compounds, Synonyms

*Stop words* (`the`, `of`, `and`) were removed to shrink indexes and speed up queries. With compressed postings and dynamic pruning, they are cheap to keep, and removing them breaks phrase queries (`to be or not to be`, `the who`, `vitamin a`). Modern engines usually keep them and let ranking discount them.

*Compound words* in German, Dutch, Swedish and Finnish join nouns without spaces: `Donaudampfschifffahrt`, `Fußballweltmeisterschaft`. A query for `Weltmeisterschaft` should match the compound. *Decompounding* splits words into parts using a dictionary (and handling linking elements such as `s` in `Arbeitsplatz`), and indexes the parts alongside the whole.

*Synonyms* (`tv`, `television`) can be expanded at index time (larger index, requires reindexing to change) or query time (flexible, but multi-word synonyms interact awkwardly with phrase queries and with scoring statistics). Query-time expansion with graph-aware token streams is the more common choice now, and learned query rewriting has replaced much hand-maintained synonym work (#xref("search-and-ir", "query-understanding", label: "Query Understanding")).

== N-grams and Phonetic Matching

Some matching needs units below or beside the word:

- *Character n-grams* (overlapping substrings of length $n$, typically 3) support substring search, partial matches and robustness to misspellings. Indexing trigrams lets a query for any substring of length three or more find candidates, which is how code search engines and PostgreSQL's `pg_trgm` support fast regular-expression and `LIKE '%term%'` search.
- *Edge n-grams* index prefixes (`s`, `se`, `sea`, `sear`, `searc`, `search`) for search-as-you-type.
- *Phonetic* algorithms (Soundex, Double Metaphone, Beider-Morse) map words to codes that sound alike, used for names in genealogy, customer and law-enforcement databases.

== Language Identification

Analysis depends on the language, so mixed-language collections first identify it. Character n-gram statistics are remarkably effective: Cavnar and Trenkle's 1994 method compared ranked n-gram frequency profiles, and modern classifiers (fastText's language identification model, CLD3) use n-gram features with a small neural network, reaching high accuracy on sentences. Short texts such as queries are hard (`chat` is English and French), so query-time analysis often uses the user's locale and the languages of the indexed documents as priors, or searches several language-specific fields at once.

== Subword Tokenisation for Neural Models

Neural retrieval and language models need a fixed vocabulary of a few tens of thousands of units that can represent any text without unknown tokens. *Subword* tokenisers achieve this by learning frequent fragments: common words become single tokens, rare words split into pieces.

- *Byte-pair encoding* (BPE), adapted from compression by Sennrich and colleagues (2016), starts from characters (or, in *byte-level* BPE as in GPT-2, from the 256 byte values, so that any input is representable) and repeatedly merges the most frequent adjacent pair in the training corpus, recording each merge. Encoding new text applies the learned merges in order of priority.
- *WordPiece* (used by BERT) chooses merges that most increase the likelihood of the training data under a unigram model, and marks word-internal pieces with `##`.
- The *Unigram* language model tokeniser (Kudo, 2018), implemented in SentencePiece, starts from a large vocabulary and prunes it, and can sample alternative segmentations during training. SentencePiece also treats whitespace as an ordinary symbol, so it works on languages without spaces and makes tokenisation reversible.

```cpp
#include <climits>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

using merge_ranks = std::map<std::pair<std::string, std::string>, int>;

// Apply learned BPE merges to one pre-tokenised word, highest-priority (lowest rank) first.
std::vector<std::string> bpe_encode(const std::string& word, const merge_ranks& ranks) {
    std::vector<std::string> symbols;
    for (char c : word) symbols.emplace_back(1, c);   // byte-level start
    while (symbols.size() > 1) {
        int best_rank = INT_MAX;
        std::size_t best = 0;
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = ranks.find({symbols[i], symbols[i + 1]});
            if (it != ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best_rank == INT_MAX) break;               // no applicable merge left
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }
    return symbols;
}

// With ranks {("l","o"):0, ("lo","w"):1, ("e","r"):2, ("low","er"):3},
// bpe_encode("lower") -> {"lower"}, bpe_encode("lowest") -> {"low", "e", "s", "t"}.
```

This loop is quadratic in word length, which is acceptable for words; production tokenisers use a priority queue over adjacent pairs and cache results for frequent words. Tokenisation interacts with retrieval in practical ways: the same word can tokenise differently with and without a leading space, numbers split into arbitrary pieces, and languages underrepresented in the tokeniser's training data need several times more tokens per word, which costs context length and money.

== Index and Query Must Agree

Every choice in this chapter must be applied identically at index time and query time, or terms will not match. The common bugs: an analyser change deployed to queries before documents are reindexed; a synonym or stop-word list edited without reindexing; a language detected differently for a short query than for the documents; a tokeniser version mismatch between the model that embedded the corpus and the one embedding queries. Treat the analysis chain as part of the index format, versioned and changed only with a reindex.

== Pitfalls

- Indexing unnormalised Unicode, so visually identical strings never match.
- Default case mappings applied to Turkish or Azerbaijani text.
- Space-based tokenisation of Chinese, Japanese or Thai.
- Aggressive stemming with no exact-match field.
- Removing stop words and breaking phrase queries.
- Changing the analysis chain without reindexing.

== Further Reading

The Unicode Consortium (2024). "Unicode Standard Annex \#15: Unicode Normalization Forms." Unicode specification.

The Unicode Consortium (2024). "Unicode Standard Annex \#29: Unicode Text Segmentation." Unicode specification.

Porter, M. F. (1980). "An Algorithm for Suffix Stripping." Program.

Krovetz, R. (1993). "Viewing Morphology as an Inference Process." SIGIR.

Hollink, V., Kamps, J., Monz, C. and de Rijke, M. (2004). "Monolingual Document Retrieval for European Languages." Information Retrieval.

Cavnar, W. B. and Trenkle, J. M. (1994). "N-Gram-Based Text Categorization." SDAIR.

Sennrich, R., Haddow, B. and Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." ACL.

Kudo, T. (2018). "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." ACL.

Kudo, T. and Richardson, J. (2018). "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." EMNLP.

Manning, C. D., Raghavan, P. and Schütze, H. (2008). _Introduction to Information Retrieval_. Cambridge University Press.
