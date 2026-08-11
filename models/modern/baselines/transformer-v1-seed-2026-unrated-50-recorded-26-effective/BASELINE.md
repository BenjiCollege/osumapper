# Transformer-v1 baseline

This directory is an immutable copy of the first complete modern rhythm-model run.

- Architecture: `conv_audio_encoder_transformer` (Transformer-v1)
- Seed: `2026`
- Data policy: unrated maps were included in the split
- Recorded epochs: `50`
- Effective non-empty epochs: `26`
- Known issue: the finite training dataset was exhausted on alternating epochs
- Test split: `42` songs, `185` maps, `950771` grid candidates
- Threshold: `0.5`
- Precision: `0.5704779762670331`
- Recall: `0.9185886815486571`
- F1: `0.7038430540650016`
- PR-AUC: `0.7898290544601929`
- Mean timing error: `0.7034305637712079 ms`
- Mean density error: `1.8351805733640114 objects/second`

These metrics demonstrate that the model learned the training task. They do not
demonstrate production mapping quality because the training split included unrated
maps. Do not resume this snapshot. New training should start from a clean output
directory after the empty-epoch fix.

The copied Keras model SHA-256 values are:

- `best.keras`: `DC66D67DCCB46BED7678837D612CAE5F602C2EF6EC4B88EC20DA826E9A94078F`
- `last.keras`: `DC66D67DCCB46BED7678837D612CAE5F602C2EF6EC4B88EC20DA826E9A94078F`
- `model.keras`: `A26D8C052F3EAF8B57B0306DB5EB6D78D85E6568EB9E81AF7AD4D6B9A43065CE`
