# Legacy baseline

The original osumapper project was created by
[kotritrona](https://github.com/kotritrona) and published at
[github.com/kotritrona/osumapper](https://github.com/kotritrona/osumapper). The
legacy source and trained assets preserved here come from that upstream work.

The historical implementations remain in their original repository locations:

- [`v6.2/`](../v6.2/)
- [`v7.0/`](../v7.0/)

They are preserved byte-for-byte by Git commit `d86ec04` (`chore: preserve legacy
osumapper baseline`). Modern application code must not be added to or edited in
these directories. Use Git to compare them against the baseline if a compatibility
adapter appears to change legacy behavior.

The unsigned `TimingAnlyz.exe` and `bass.dll` are retained for archaeology only.
The modern application does not execute them.
