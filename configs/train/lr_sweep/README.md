# xLSTM LR Sweep Configs

These configs hold batch size fixed at 8 so LR schedule comparisons are not
confounded by different optimizer step counts.

The step counts are based on the tokenizer chunk reports currently used for
xLSTM experiments:

| tokenizer | train chunks | batch | steps/epoch | 3 epoch steps |
| --- | ---: | ---: | ---: | ---: |
| `11` | 33879 | 8 | 4235 | 12705 |
| `11_chords` | 34039 | 8 | 4255 | 12765 |

For batch 8, the schedule endpoints mean:

| endpoint | tokenizer `11` | tokenizer `11_chords` |
| ---: | ---: | ---: |
| 200 | 1.6% | 1.6% |
| 4000 | 31.5% | 31.3% |
| 8000 | 63.0% | 62.7% |
| 12000 | 94.5% | 94.0% |

The `peak_*` and `warmup_*` configs keep the old experiment decay endpoint
of 4000 steps. The `decay_*` configs test longer cosine decays over roughly
two thirds and almost all of the 3 epoch run.
