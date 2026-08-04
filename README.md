# MCQ Answer Predictor

Project T22026 — Deep Learning and Generative AI
Roll number: 24f2005363
Competition: Smart MCQ Solver Challenge (Kaggle)

---

## 1. What this project does

Given a multiple choice question and its five options (A, B, C, D, E), the model
predicts which option is the correct answer.

It does not answer the question the way a person would. It never "reads and
understands" the question in a real sense. What it does is turn the question text
into numbers, and then learn from thousands of past examples which kinds of
number patterns tend to go with answer A, which tend to go with answer B, and so
on. That is all a classifier is.

Input:

```
prompt : "Which planet is closest to the Sun?"
A      : "Venus"
B      : "Mercury"
C      : "Mars"
D      : "Earth"
E      : "Jupiter"
```

Output: a ranked guess, best first.

```
B A D
```

Three letters are given instead of one because of how the competition is scored
(explained in section 4).

---

## 2. The main idea in three steps

Every one of the three models in this project follows the same three steps. Only
step 1 changes between them.

**Step 1 — turn text into numbers.**
A neural network cannot take a sentence as input. It only takes numbers. So the
question and its five options are first glued into one long string, and then that
string is converted into a list of numbers called a *vector*. This is the step
where the three models differ.

**Step 2 — feed those numbers to a small neural network.**
The network is a few `Linear` layers stacked on top of each other. It takes the
vector in and produces five numbers out, one per answer letter.

**Step 3 — convert the five numbers into probabilities.**
A `softmax` turns the five raw numbers into five probabilities that add up to 1,
for example `[0.05, 0.70, 0.10, 0.10, 0.05]`. Here the model is most confident
about B. The three highest get written into the submission file.

---

## 3. How the text becomes numbers

The row is joined into one string first:

```
Which planet is closest to the Sun? A) Venus B) Mercury C) Mars D) Earth E) Jupiter
```

Then it gets converted to a vector in one of two ways.

### Way 1 — TF-IDF (used by model A)

TF-IDF counts words. Imagine building a list of the 8000 most common words and
word-pairs across the whole training set. Each question then becomes a list of
8000 numbers, where each number says how important that particular word is in
that particular question.

- A word that shows up a lot in *this* question gets a higher number.
- A word that shows up in *every* question (like "the" or "which") gets pushed
  down, because it tells you nothing useful.

TF-IDF is fast and needs no pretrained model. Its weakness is that it has no idea
what words mean. To TF-IDF, "car" and "automobile" are two completely unrelated
things, because they are spelled differently.

### Way 2 — sentence embeddings (used by models B and C)

Here a pretrained model that has already read a large amount of text does the
conversion. It reads the sentence and produces a vector (384 or 768 numbers) that
represents the *meaning* of the sentence, not just which words appeared.

The useful property is that sentences with similar meanings end up with similar
vectors. "car" and "automobile" land close together. This is why embeddings
usually beat TF-IDF on text tasks.

Two different pretrained models are used, so their results can be compared:

- `paraphrase-MiniLM-L6-v2` — small and fast, gives 384 numbers per sentence
- `all-mpnet-base-v2` — bigger and slower, gives 768 numbers per sentence

Both of these are frozen. They are only used to convert text to vectors. They are
never trained here. The only thing being trained is the small network sitting on
top.

---

## 4. The scoring metric: MAP@3

The competition does not simply ask "was your top answer right or wrong". It lets
you submit three guesses per question, and gives partial credit based on where the
correct answer sits in your list.

| Correct answer is your... | You score |
| --- | --- |
| 1st guess | 1.00 |
| 2nd guess | 0.50 |
| 3rd guess | 0.33 |
| not in your top 3 | 0.00 |

MAP@3 is the average of that score over every question.

This is why the submission file has three letters per row, and why it is always
worth filling all three slots. A second or third guess can only help; it can never
lose you points.

The code for this is `get_map3()` in `utils.py`. It sorts the five probabilities
from high to low, takes the top three, and checks where the true answer landed.

---

## 5. The three models

Three models are built so their results can be compared fairly, which is what the
project asks for.

### Model A — TF-IDF + a plain network

```
8000 TF-IDF numbers -> 400 -> 128 -> 5
```

with ReLU activations and dropout of 0.3.

This is the from-scratch baseline. No pretrained model anywhere. It exists to show
what a simple approach gets you, so the embedding models have something to be
measured against.

### Model B — MiniLM embeddings + a small head

```
384 embedding numbers -> 200 -> 5
```

with Tanh activation and dropout of 0.2.

The head is deliberately small. The embeddings are already carrying most of the
meaning, so a large network on top would mostly just memorise the training set.
**This is the model that gets uploaded to Hugging Face**, as required by the
project brief.

### Model C — MPNet embeddings + a wider head

```
768 embedding numbers -> 384 -> 96 -> 5
```

with ReLU and dropout of 0.25, and a slightly lower learning rate of 0.0007.

Same idea as B but with the larger embedding model, to see whether the bigger
pretrained model is actually worth the extra time it takes.

### Side by side

| | Model A | Model B | Model C |
| --- | --- | --- | --- |
| Text to numbers | TF-IDF | MiniLM | MPNet |
| Input size | 8000 | 384 | 768 |
| Hidden layers | 400, 128 | 200 | 384, 96 |
| Activation | ReLU | Tanh | ReLU |
| Dropout | 0.3 | 0.2 | 0.25 |
| Learning rate | 0.001 | 0.001 | 0.0007 |
| Batch size | 64 | 32 | 32 |
| Epochs | 22 | 18 | 18 |
| Best epoch picked on | val f1 | val map3 | val f1 |

---

## 6. The train/validation split, and why it is done carefully

The training data is split into two parts: about 85 percent to learn from, and
about 15 percent held back to test on. The held-back part is called the
*validation set*. It is the only honest way to check whether the model has
actually learned something or has just memorised the answers.

There is a trap here. This dataset contains **repeated questions** — the same
question appears more than once. If a plain random split is used, the same
question can end up in both halves. The model then sees it during training, sees
it again during validation, recognises it, and gets it right. The validation score
comes out looking great and means nothing. This is called **data leakage**.

The fix is `GroupShuffleSplit`, grouping on the `prompt` column. It keeps every
copy of the same question together on one side of the split, so a question is
either fully in training or fully in validation, never both.

The code prints a check so you can confirm it worked:

```
train rows 4250 val rows 750 | prompt overlap (want 0): 0
```

If that last number is anything other than 0, the validation score cannot be
trusted.

---

## 7. Overfitting and dropout

Overfitting is when the model memorises the training data instead of learning a
general pattern. You spot it when the training loss keeps falling but the
validation score stops improving or gets worse.

Two things guard against it here:

**Dropout** randomly switches off a fraction of the neurons during each training
step. Because no single neuron can be relied on, the network is pushed to spread
what it learns across many of them. Dropout is only active during training; it
turns itself off automatically at evaluation time when `model.eval()` is called.

**Keeping the best checkpoint.** After every epoch the validation score is
computed, and the model weights are saved to disk only when that score improves.
At the end, the saved file is loaded back. So even if the last few epochs made the
model worse, the version kept is the best one it ever reached, not the final one.

---

## 8. Experiment tracking with Weights & Biases

Every epoch of every model logs its numbers (train loss, validation loss,
accuracy, f1, map3) to Weights & Biases. This draws live curves in the browser.

The reason for this is practical. With three models and around twenty epochs each,
that is roughly sixty lines of printed numbers, and comparing them by eye is
painful and error-prone. On a graph you can see immediately where a model stopped
improving or started overfitting.

All runs go to project `24f2005363-t22026` and are grouped under the Hugging Face
username, so the three training runs and the final inference run sit together.

---

## 9. Uploading to Hugging Face

After training, model B is uploaded to a Hugging Face repository named
`<your-username>/mcq-answer-predictor-t22026`.

Files uploaded:

| File | What it is |
| --- | --- |
| `net_b.pt` | the trained weights — the important one |
| `model_b_meta.json` | which embedding model to use, and the input size |
| `README.md` | model card with the validation scores |
| `score_table.csv` | the full A / B / C comparison |
| `winner.json` | which model scored highest on map3 |
| `compare_plot.png` | the comparison bar chart |

Two points that commonly cause confusion:

**Model B is uploaded even if it did not win.** The project brief specifies model
B, so it goes up regardless. If `winner.json` says `model_c`, that is expected
behaviour and not a bug. The README notes the difference honestly.

**`net_b.pt` is only the small head**, not the sentence transformer. It is about
half a megabyte, not hundreds. Anyone using it also needs to download
`paraphrase-MiniLM-L6-v2` separately to produce the embeddings first. That is
exactly why `model_b_meta.json` records the embedding model name.

---

## 10. The inference step, and why it exists

The last stage downloads `net_b.pt` back from Hugging Face and runs predictions
with it.

At first this looks pointless, since the model is already sitting in memory. It is
not. The point is to prove the upload actually works. It is very easy to upload a
file that is corrupted, empty, or has the wrong shape, and never find out because
you kept using the in-memory copy the whole time.

So the inference stage deliberately starts fresh:

- weights are downloaded from the hub, not read from the local folder
- the text is embedded again from scratch, not reused from training
- a brand new empty `NetB` is created and the downloaded weights are loaded into it

If the score it produces matches the score from training, the upload is confirmed
good. It then draws a confusion matrix, writes the final submission file, and logs
one more run to Weights & Biases.

### Reading the confusion matrix

The confusion matrix is a 5x5 grid. Rows are the true answers, columns are what
the model predicted. Correct predictions sit on the diagonal from top-left to
bottom-right. Anything off the diagonal is a mistake, and its position tells you
which letters the model is mixing up. A bright column means the model is biased
towards guessing that one letter too often.

---

## 11. Project structure

```
.
├── src/
│   ├── utils.py       shared code: data, models, metrics, logins
│   ├── train.py       trains all 3 models, compares, uploads model B
│   └── inference.py   downloads model B and runs predictions
├── outputs/           everything the scripts produce
└── README.md
```

### What is in each file

**`utils.py`** holds everything the other two files both need, so they cannot
drift apart. If the text-joining function lived in two places and only one was
edited, training and inference would silently disagree. Inside it:

- `load_data()` finds and reads the CSV files
- `join_row()` glues prompt and options into one string
- `make_split()` does the group-aware train/validation split
- `get_map3()` computes the competition metric
- `write_submission()` writes the top-3 letters in the required format
- `NetA`, `NetB`, `NetC` are the three network definitions
- `run_epoch()` is one pass over the data, training or evaluating
- `embed_texts()` runs a sentence transformer over the text
- `Run` wraps Weights & Biases so a missing key does not crash the script

**`train.py`** runs the whole training pipeline in order: load, split, train A,
train B, train C, compare, upload.

**`inference.py`** is the reload-and-verify step described in section 10.

---

## 12. How to run it

### Install

```bash
pip install torch transformers sentence-transformers scikit-learn \
            huggingface_hub wandb pandas matplotlib
```

### Keys

Two keys are needed: one for Hugging Face, one for Weights & Biases. On Kaggle,
add them under Add-ons > Secrets as `HF_TOKEN` and `WANDB_API_KEY`. Running
anywhere else, set them as environment variables:

```bash
export HF_TOKEN=hf_xxxxxxxx
export WANDB_API_KEY=xxxxxxxx
```

If a key is missing the script prints a note and carries on without that feature,
rather than crashing halfway through.

### Train

```bash
# on Kaggle, the data path is found automatically
python src/train.py

# running locally
python src/train.py --data-dir data --out-dir outputs

# quick check that everything works, 2 epochs, no uploads
python src/train.py --data-dir data --quick --no-wandb --no-push --skip-c
```

Run the quick version first. It finishes in under a minute and tells you whether
the pipeline works before you commit to a full run.

### Predict

```bash
# repo is read from outputs/hub_repo.json automatically
python src/inference.py

# or name the repo yourself
python src/inference.py --repo-id yourname/mcq-answer-predictor-t22026

# skip the hub entirely and use local weights
python src/inference.py --weights outputs/net_b.pt --no-wandb
```

### The flags

| Flag | What it does |
| --- | --- |
| `--data-dir` | folder containing train.csv and test.csv |
| `--out-dir` | where results get written, default `outputs` |
| `--quick` | 2 epochs per model, for testing the pipeline |
| `--skip-c` | skip MPNet, which is the slow one |
| `--no-wandb` | do not log to Weights & Biases |
| `--no-push` | do not upload to Hugging Face |
| `--seed` | random seed, default 7, controls the split |
| `--epochs-a/b/c` | epochs per model |
| `--max-features` | TF-IDF vocabulary size, default 8000 |

---

## 13. What gets written to `outputs/`

| File | What it is |
| --- | --- |
| `net_a.pt`, `net_b.pt`, `net_c.pt` | trained weights, best epoch of each |
| `score_table.csv` | the comparison table, sorted by map3 |
| `winner.json` | which model scored highest |
| `compare_plot.png` | bar chart of map3 by model |
| `answer_count.png` | how many of each answer letter is in the data |
| `inference_confusion.png` | confusion matrix and per-class accuracy |
| `sub_model_a.csv` and similar | one submission per model |
| `submission_hub_model_b.csv` | the final submission, from the hub model |
| `model_b_meta.json` | embedding model name and input size |
| `label_meta.json` | which letter maps to which class number |
| `val_prob_*.npy`, `test_prob_*.npy` | saved probabilities, useful for blending later |

---

## 14. Things that went wrong, and how they were handled

**Repeated questions leaking across the split.** Solved with `GroupShuffleSplit`
on the prompt column. The overlap count is printed every run as a check.

**Validation score looking better than the leaderboard score.** Normally a sign of
leakage or of overfitting to the validation set through too much tuning. The group
split is the first line of defence; keeping the best checkpoint by validation
score is the second.

**A missing API key stopping the whole run.** Everything to do with Weights &
Biases and Hugging Face is wrapped so that a missing key prints a message and the
script continues. Losing the logging is annoying; losing an hour of training is
much worse.

**The wrong input size at inference time.** MiniLM gives 384 numbers and MPNet
gives 768. Loading model B's weights into a network built for 768 fails
immediately. `model_b_meta.json` records the correct size so the network is always
rebuilt to match.

---

## 15. Ideas for improving the score

- Blend the three models by averaging their probabilities. The saved `.npy` files
  are there for exactly this. Models that make different mistakes often combine
  well.
- Fine-tune the sentence transformer instead of freezing it. Slower and more
  likely to overfit, but the ceiling is higher.
- Use cross-validation instead of a single split, so every row gets used for
  validation once. This gives a much steadier estimate of the true score.
- Encode each option separately against the question, rather than gluing all five
  into one string. This gives the model a clearer view of which option is which.
- Search the hyperparameters with Optuna rather than picking learning rates and
  layer sizes by hand.

---

## 16. Glossary

**Epoch** — one full pass through all the training data.

**Batch** — a small group of rows processed together, here 32 or 64. Faster than
one row at a time, and lighter on memory than all rows at once.

**Learning rate** — how big a step the model takes when correcting itself. Too
large and it overshoots; too small and it takes forever.

**Loss** — a number measuring how wrong the model is. Training means pushing this
down. Cross entropy is the loss used here, standard for classification.

**Softmax** — turns raw output numbers into probabilities that add up to 1.

**Dropout** — randomly switching off neurons during training to discourage
memorising.

**Embedding** — a list of numbers representing the meaning of a piece of text.

**Accuracy** — the fraction of questions where the top guess was correct.

**F1 (macro)** — an average that treats every answer letter as equally important.
Useful when some letters appear more often than others, because a model that
always guesses the most common letter would score decent accuracy but poor macro
f1.

**Checkpoint** — a saved copy of the model weights at a particular point.

**Data leakage** — when information from the validation set reaches the model
during training, making scores look better than they truly are.
