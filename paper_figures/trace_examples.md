# Sample reasoning traces (Appendix H source material)

Hand-curated from the multi-seed re-evaluation JSONs at
`results/extracted/results_multiseed/<condition>/gsm8k_results.json`,
which evaluate every condition on the same 1,319-example GSM8K test split
in identical order. Trace text is reproduced verbatim, with byte-level
encoding glitches in the original log files (e.g. ■ → �) reproduced as
they appear, since the grader scores on the extracted numeric answer.

---

## Example: GSM8K test problem #17 — "Jill's annual salary"

> Jill gets paid \$20 per hour to teach and \$30 to be a cheerleading coach.
> If she works 50 weeks a year, 35 hours a week as a teacher and 15 hours a
> week as a coach, what's her annual salary?
>
> **Ground truth: \$57,500**

This problem requires the model to compute two products
(35 × 50 × \$20 = \$35,000 and 15 × 50 × \$30 = \$22,500) and add them.
Baseline (A) emits "35" — picking up a single number from the problem with no
arithmetic. Three of the four trained recipes (B, D, G) get the right answer;
the RL-on-inserted-layers recipe (C) makes a different reasoning error.

### B — LoRA + GRPO  (predicted: \$57,500 ✓)

> We need to find the annual salary of Jill. We are given that Jill gets paid
> \$20 per hour to teach and \$30 to be a cheerleading coach. We are also
> given that she works 50 weeks a year, 35 hours a week as a teacher and
> 15 hours a week as a coach.
>
> We need to find the annual salary of Jill. We can find the annual salary
> by calculating the total amount she earns in a year.
>
> [...]
>
> – Jill works 35 hours a week as a teacher. Therefore, the total number of
>   hours she works as a teacher in a year is 35 × 50 = 1750 hours.
> – Jill gets paid \$20 per hour to teach. Therefore, the total amount she
>   earns as a teacher in a year is 1750 × \$20 = \$35,000.
> – Jill works 15 hours a week as a coach. Therefore, the total number of
>   hours she works as a coach in a year is 15 × 50 = 750 hours.
> – Jill gets paid \$30 per hour to be a coach. Therefore, the total amount
>   she earns as a coach in a year is 750 × \$30 = \$22,500.
>
> Therefore, the total amount Jill earns in a year is \$35,000 + \$22,500
> = \$57,500. Therefore, the annual salary of Jill is \$57,500. Therefore,
> the annual salary of Jill is \$57,500.

Note the verbose, repetitive style and the doubled final-sentence pattern.
This is a documented artifact of GRPO-trained models: repeating the answer
maximises the likelihood of getting the format-bonus and answer-detection
signal triggered for the reward. The GSM8K reward function in `rewards.py`
gives 1.0 for any correct numeric answer and 0.1 for showing reasoning
patterns; B has clearly been shaped to maximise both.

### C — Inserted layers + GRPO  (predicted: \$72,500 ✗)

> We know that Jill gets paid \$20 per hour to teach and \$30 to be a
> cheerleading coach. If she works 50 weeks a year, 35 hours a week as a
> teacher and 15 hours a week as a coach, we can calculate her annual
> salary.
>
> To calculate her annual salary, we need to calculate the number of hours
> she works in a year. We know that she works 35 hours a week as a teacher
> and 15 hours a week as a coach. So, we can calculate the total number of
> hours she works in a year as follows:
>
> 50 · 35 = 1750 hours as a teacher
> 50 · 15 = 750 hours as a coach
> 1750 + 750 = 2500 total hours
>
> So, we can calculate her annual salary as follows:
>
> 20 · 2500 = 50,000 dollars as a teacher
> 30 · 750 = 22,500 dollars as a coach
> 50,000 + 22,500 = 72,500 dollars total
>
> Therefore, Jill's annual salary is \$72,500.

A genuine reasoning error: C correctly computes the per-role hours
(1,750 and 750), but then multiplies the teacher's hourly rate by the
**total** hours (\$20 × 2,500 = \$50,000) instead of by the teacher's
hours (\$20 × 1,750 = \$35,000). The trace looks confident and structured,
but the arithmetic glues to the wrong totals. This is the kind of failure
mode the inserted-layer-RL recipe produces consistently — the model has
learned to format chain-of-thought but has not been shaped tightly enough
on the answer-correctness signal to catch its own mis-allocations.

### D — Inserted layers + SFT  (predicted: \$57,500 ✓)

> First find the total number of hours Jill works as a teacher: 50 weeks/year
> * 35 hours/week = <<50*35=1750>>1750 hours
> Then find the total number of hours Jill works as a coach: 50 weeks/year
> * 15 hours/week = <<50*15=750>>750 hours
> Then multiply the number of hours Jill works as a teacher by her hourly
> rate to find her annual salary as a teacher: \$20/hour * 1750 hours
> = \$<<20*1750=35000>>35000
> Then multiply the number of hours Jill works as a coach by her hourly rate
> to find her annual salary as a coach: \$30/hour * 750 hours
> = \$<<30*750=22500>>22500
> Then add her salary as a teacher to her salary as a coach to find her
> total annual salary: \$35000 + \$22500 = \$<<35000+22500=57500>>57500
> #### 57500

D has clearly memorised the GSM8K-train solution-trace style — note the
`<<...=...>>` calculator annotations and the `####` answer marker, which
are exactly the format of the gold solutions in `train_dataset["answer"]`.
Whether this is "real reasoning" or "stylistic mimicry that happens to
include arithmetic" is debatable, but the answer is right and the working
is structured.

### G — LoRA distillation  (predicted: \$57,500 ✓)

> We need to find the annual salary of Jill. We are given that she gets
> paid \$20 per hour to teach and \$30 to be a cheerleading coach. We are
> also given that she works 50 weeks a year, 35 hours a week as a teacher
> and 15 hours a week as a coach. So, we need to find the total number of
> hours she works in a year. We can find the total number of hours she
> works in a year by multiplying the number of weeks she works in a year
> with the number of hours she works in a week. So, the total number of
> hours she works in a year is 50 × 35 = 1750 hours. Now, we need to find
> the total amount she gets paid in a year. We can find the total amount
> she gets paid in a year by multiplying the number of hours she works in
> a year with the amount she gets paid per hour. So, the total amount she
> gets paid in a year is 1750 × \$20 = \$35,000. We are also given that
> she gets paid \$30 to be a cheerleading coach. So, the total amount she
> gets paid in a year is \$35,000 + 50 × 15 × \$30 = \$35,000 + 750 ×
> \$30 = \$35,000 + \$22,500 = \$57,500. So, the annual salary of Jill
> is \$57,500. Therefore, the answer is \$57,500.

G's prose style is much closer to B's than to D's — long, redundant, no
calculator annotations, no `####` — which makes sense: G's training signal
is the LoRA-trained model B's logits, so G has learned to imitate B's
output distribution rather than the GSM8K solution-trace format. D and G
arrive at the same final answer via stylistically very different routes,
exactly because their training signals come from different sources.

---

## Stylistic summary across recipes (qualitative)

- **B (LoRA + GRPO):** verbose, repetitive, often doubles the final-answer
  sentence; format reflects what the GRPO reward shaped.
- **C (Inserted + GRPO):** structurally confident chain-of-thought, but
  arithmetic mis-allocations are common; the recipe never quite locked
  onto the answer-correctness signal as tightly as B did.
- **D (Inserted + SFT):** terse, structured GSM8K-style traces with
  `<<calculator>>` annotations and `####` answer markers — direct
  stylistic mimicry of the supervised gold traces.
- **G (Distillation):** verbose like B (its teacher), no `####` markers.
  Style is inherited from the teacher; arithmetic accuracy is teacher-
  bounded.
- **E (Two-stage):** mixed style (LoRA-merged base preserves B-like
  verbosity; inserted-layer SFT pushes toward D-like terseness in some
  prompts).

These are observations, not measurements — Section 6 of the paper notes
that no human evaluation of trace quality was performed.
