# Vendored pronunciation reference sources

The source snapshots are stored as deterministic gzip streams (`gzip -n -9`).
The uncompressed SHA-256 values in `manifest.json` pin the exact downloaded
inputs and are verified before every database build.

`large_pinyin`, `zdic_cibs`, `zdic_cybs`, and CC-CEDICT are imported into the
runtime SQLite database as pronunciation sources. The jieba corpus dictionary
is imported into the same database for offline word-frequency and
part-of-speech signals. The upstream `pinyin.txt` snapshot is also vendored and
checksummed, but is not imported as a fifth pronunciation provenance: it is an
input of the upstream `large_pinyin` aggregate, whose per-record provenance is
the one used by this application.

The mozillazg phrase datasets are MIT licensed; see
`LICENSE.phrase-pinyin-data`. CC-CEDICT is CC BY-SA 4.0; see
`LICENSE.CC-CEDICT` and the preserved source header. The jieba corpus
dictionary is MIT licensed; see `LICENSE.jieba`.
The general standard character table (国发〔2013〕23号) uses the MIT-licensed jaywcjlove dataset (`LICENSE.tgscc`).
The HSK vocabulary standard (GF0025-2021) uses the MIT-licensed drkameleon dataset (`LICENSE.hsk`).

`excluded_words.txt` records owner-governed lookup exclusions without changing
the original source files. It participates in the build fingerprint.
