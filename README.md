# Small fine-tuned models vs general LLMs for scientific text classification

**Question:** to label the role of each sentence in a medical research abstract, should you fine-tune a small model or prompt a general large language model? And how much labelled data does the small model really need?

**Short answer:** a fine-tuned 110-million-parameter biomedical model (BiomedBERT) was the most accurate model tested (macro-F1 **0.870** on 30,135 test sentences). The best general LLM (Qwen2.5-14B-Instruct (4-bit): zero shot position), using **no labelled data at all**, reached **0.775**: as good as small models trained on thousands of examples, but significantly behind the fine-tuned model and about **495 times slower**. With only **640 labelled sentences**, a SetFit model matched a TF-IDF model trained on 180,040. And a manual review found that about **half of the best model's "errors" were labelling errors in the benchmark itself**.

## The task and data

**PubMed 20k RCT** ([Dernoncourt & Lee, 2017](https://github.com/Franck-Dernoncourt/pubmed-rct)): 20,000 abstracts of randomised controlled trials, about 240,000 sentences, each labelled **BACKGROUND, OBJECTIVE, METHODS, RESULTS or CONCLUSIONS**.

- Splits: 180,040 training, 30,212 development and 30,135 test sentences.
- Classes are imbalanced (OBJECTIVE is 7.7% of test sentences), so the main metric is **macro-F1**, which weights every class equally.
- Every model is compared on the **same balanced set of 1,000 test sentences** (200 per class, fixed seed), so paired significance tests are possible. Models that run locally are also scored on the full test set.
- The dataset carries no explicit licence and abstracts may be under copyright, so it is **downloaded at runtime and not redistributed** here.

## Models compared

| Family | Model | Labelled training data |
|---|---|---|
| Classic baseline | TF-IDF + logistic regression, with and without sentence position | 180,040 |
| Few-shot fine-tuning | SetFit on a biomedical sentence-transformer (S-PubMedBert-MS-MARCO), 8 to 512 examples per class | 40 to 2,560 |
| Full fine-tuning | BiomedBERT (110M parameters), with and without class weights | 180,040 |
| General LLM | Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct, zero-shot and few-shot in 4-bit, run locally | 0 or 10 examples in the prompt |

"Position" means the sentence's place in its abstract (for example "[3 of 9]"), given to every model that uses it in the same way.

## Results

All scores on the same 1,000 balanced test sentences, with 95% bootstrap confidence intervals. Speeds are measured on a free Google Colab T4 GPU.

| Model | Labelled examples | Macro-F1 (1,000 set) | 95% CI | Full test macro-F1 | OBJECTIVE F1 | Invalid outputs | ms/sentence |
|---|---|---|---|---|---|---|---|
| BiomedBERT fine-tuned + position + class weights | 180,040 | 0.852 | 0.829 to 0.873 | 0.870 | 0.703 | 0 | 1.9 |
| BiomedBERT fine-tuned + position | 180,040 | 0.844 | 0.822 to 0.866 | 0.868 | 0.669 | 0 | 1.8 |
| SetFit 512/class + position | 2,560 | 0.825 | 0.802 to 0.845 | 0.817 | 0.723 | 0 | 3.6 |
| TF-IDF + position | 180,040 | 0.806 | 0.783 to 0.831 | 0.833 | 0.695 | 0 |  |
| Qwen2.5-14B-Instruct (4-bit): zero shot position | 0 | 0.775 | 0.749 to 0.798 |  | 0.678 | 9 | 947.3 |
| SetFit 512/class | 2,560 | 0.767 | 0.740 to 0.791 | 0.764 | 0.672 | 0 | 3.4 |
| TF-IDF (text) | 180,040 | 0.746 | 0.719 to 0.773 | 0.763 | 0.669 | 0 |  |
| Qwen2.5-7B-Instruct (4-bit): zero shot position | 0 | 0.703 | 0.673 to 0.730 |  | 0.678 | 7 | 488.1 |
| Qwen2.5-7B-Instruct (4-bit): few shot position | 10 | 0.672 | 0.643 to 0.699 |  | 0.665 | 0 | 1046.3 |
| Qwen2.5-7B-Instruct (4-bit): zero shot | 0 | 0.624 | 0.593 to 0.653 |  | 0.652 | 11 | 468.8 |


### How much labelled data does a small model need?

| Examples per class | Total labelled sentences | SetFit macro-F1 (1,000 set) | Runs |
|---|---|---|---|
| 8 | 40 | 0.388 ± 0.019 | 3 |
| 32 | 160 | 0.678 ± 0.044 | 3 |
| 128 | 640 | 0.748 | 1 |
| 512 | 2,560 | 0.767 | 1 |
| (TF-IDF, text only) | 180,040 | 0.746 | 1 |


![Learning curve](results/chart_learning_curve.png)

### Accuracy against speed

![Accuracy against speed](results/chart_accuracy_vs_speed.png)

## Key findings

1. **Fine-tuning a small domain model wins.** BiomedBERT fine-tuned + position + class weights reached macro-F1 **0.852** on the balanced set and **0.870** on the full test set, at **1.9 ms per sentence**. It was significantly better than the best baseline (exact McNemar test, p = 8e-6).
2. **Labels are expensive; you need far fewer than you might think.** SetFit with **128 examples per class (640 sentences)** scored **0.748**, the same as TF-IDF trained on **180,040** sentences (0.746). Gains flatten quickly after that.
3. **LLMs are the right tool without labels, not once you have them.** Doubling the LLM from 7B to 14B raised its zero-shot (with position) macro-F1 from **0.703** to **0.775**, mostly by fixing CONCLUSIONS (F1 0.624 to 0.864). With no labelled data, the 14B matched SetFit trained on 2,560 sentences without position (0.767) and beat TF-IDF on text (0.746). But it stayed behind every model that combined training data with position, and significantly behind the best (exact McNemar p = 4e-12), at **947 ms per sentence** against **1.9 ms**. OBJECTIVE stayed the hardest class for every model (F1 about 0.67 to 0.72). Adding 10 examples to the 7B's prompt lowered its score (0.672 against 0.703) in this setup.
4. **Class weights fix part of the rare-class problem, at no overall cost.** Square-root inverse-frequency weights raised OBJECTIVE F1 from **0.669** to **0.703**. The overall difference was not significant (p = 0.36), so this is a targeted improvement, not a better model overall. Interestingly, SetFit trained on balanced classes still had the best OBJECTIVE F1 (0.723).
5. **The benchmark's labels are a ceiling.** I reviewed 30 randomly sampled sentences where the best unweighted model disagreed with the gold label:

| Judgement | Count |
|---|---|
| Model right, dataset label wrong | 15 |
| Dataset right, model wrong | 9 |
| Both wrong (another label fits better) | 3 |
| Genuinely ambiguous | 3 |

Only about 40% of the model's disagreements were genuine errors. Extrapolated to all disagreements, the model's true accuracy is likely around **91 to 96%**, against a reported 85%. With 30 sentences the uncertainty is wide (the "model right" share has a 95% interval of roughly 33% to 67%), and the review was done by one reviewer who could see both labels, so this is an indicative finding rather than a precise estimate. A blind, two-reviewer relabelling would firm it up.

## Statistical tests

Exact McNemar tests, paired on the same 1,000 sentences, comparing the best model with each of the others:

- vs **BiomedBERT fine-tuned + position**: best right and other wrong on 18 sentences, the reverse on 12; p = 0.36
- vs **SetFit 512/class + position**: best right and other wrong on 79 sentences, the reverse on 48; p = 0.0075
- vs **TF-IDF + position**: best right and other wrong on 78 sentences, the reverse on 31; p = 8e-6
- vs **Qwen2.5-14B-Instruct (4-bit): zero shot position**: best right and other wrong on 113 sentences, the reverse on 31; p = 4e-12
- vs **SetFit 512/class**: best right and other wrong on 133 sentences, the reverse on 44; p = 1e-11
- vs **TF-IDF (text)**: best right and other wrong on 137 sentences, the reverse on 31; p = 4e-17
- vs **Qwen2.5-7B-Instruct (4-bit): zero shot position**: best right and other wrong on 183 sentences, the reverse on 34; p = 7e-26
- vs **Qwen2.5-7B-Instruct (4-bit): few shot position**: best right and other wrong on 206 sentences, the reverse on 30; p = 2e-33
- vs **Qwen2.5-7B-Instruct (4-bit): zero shot**: best right and other wrong on 243 sentences, the reverse on 31; p = 6e-42


## Limitations

- **No frontier LLM result.** A Gemini comparison is included as notebook 05, but it could not be completed on the free API tier because of quota and availability limits. The LLMs tested here are open models (7B and 14B) in 4-bit, so the result shows the gap for general models of that size, not for the largest commercial models.
- **LLM prompting was kept simple.** Each sentence was classified in its own prompt, with the same label definitions and, for few-shot, the same 10 examples; outputs that did not name a valid label were counted as wrong. An earlier version sent 20 sentences per prompt; the models sometimes skipped or repeated a label in the list, which misaligned answers and sentences and made those results unreliable in both directions. They were discarded and the evaluation was rerun one sentence at a time.
- **Single runs.** Fine-tuning ran once per setting; GPU training is not fully deterministic, and small run-to-run differences (around 0.003 macro-F1) were observed.
- **One dataset, one task.** Conclusions may not transfer to other text types or label schemes.

## Reproduce

Everything runs on free Google Colab. Open the notebooks in order:

| Notebook | What it does | Runtime |
|---|---|---|
| `01_data` | Download data, fix the evaluation set | CPU |
| `02_baseline` | TF-IDF baselines | CPU, about 1 min |
| `03_setfit_learning_curve` | SetFit at 8 to 512 examples per class | T4 GPU, 20 to 35 min |
| `04_biomedbert_finetune` | Full fine-tune | T4 GPU, about 15 min |
| `04b_biomedbert_class_weights` | Full fine-tune with class weights | T4 GPU, about 15 min |
| `05_gemini_optional` | Frontier LLM via API (needs a key) | CPU |
| `06_open_llm_local` | Qwen2.5-7B and 14B-Instruct, 4-bit, one sentence per prompt | T4 GPU, 30 to 60 min per model |
| `07_final_comparison` | Tables, tests, charts, error sample | CPU |

Data and intermediate results are saved to Google Drive under `sci-classify/`. Long-running notebooks save as they go and can be resumed.

## Author

Dolapo Peter Abikoye, MSc Artificial Intelligence and Data Science (Keele University), medically trained. [LinkedIn](https://www.linkedin.com/in/abikoye-dolapo-371079347) · [GitHub](https://github.com/PeterAI-del)
