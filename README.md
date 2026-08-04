# MCQ Answer Predictor

Project T22026 — Deep Learning and Generative AI
Roll number: 24f2005363
Competition: Smart MCQ Solver Challenge (Kaggle)
Notebook: `dl-24f2005363-notebook-t22026.ipynb`

---

## 1. What this notebook does

Given a multiple choice question and its five options (A, B, C, D, E), it
predicts which option is the correct answer.

The notebook does not answer the question the way a person would. It never reads
and understands the question in a real sense. What it does is turn the question
text into numbers, and then learn from thousands of past examples which kinds of
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

Output, a ranked guess with the best first:

```
B A D
```

Three letters are given instead of one because of how the competition is scored,
explained in section 4.

Three separate models are built and compared. All three runs are tracked in
Weights & Biases so the curves can be looked at side by side instead of squinting
at printed numbers. Model B is then pushed to Hugging Face, and at the very end
the notebook pulls it back down from the hub and runs inference with the
downloaded copy, to prove the uploaded weights actually work.

---

## 2. The main idea in three steps

Every one of the three models follows the same three steps. Only step 1 changes
between them.

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

The `join_row` function glues each row into one string:

```
Which planet is closest to the Sun? A) Venus B) Mercury C) Mars D) Earth E) Jupiter
```

That string is then converted to a vector in one of two ways.

### Way 1 — TF-IDF, used by model A

TF-IDF counts words. The notebook builds a list of the 8000 most common words and
word-pairs across the training set. Each question becomes a list of 8000 numbers,
where each number says how important that word is in that question.

- A word that shows up a lot in *this* question gets a higher number.
- A word that shows up in *every* question, like "the" or "which", gets pushed
  down, because it tells you nothing useful.

TF-IDF is fast and needs no pretrained model. Its weakness is that it has no idea
what words mean. To TF-IDF, "car" and "automobile" are two completely unrelated
things, because they are spelled differently.

### Way 2 — sentence embeddings, used by models B and C

Here a pretrained model that has already read a large amount of text does the
conversion. It reads the sentence and produces a vector, 384 or 768 numbers, that
represents the *meaning* of the sentence rather than just which words appeared.

The useful property is that sentences with similar meanings end up with similar
vectors. "car" and "automobile" land close together. This is why embeddings
usually beat TF-IDF on text tasks.

Two different pretrained models are used so they can be compared:

- `paraphrase-MiniLM-L6-v2` — small and fast, gives 384 numbers per sentence
- `all-mpnet-base-v2` — bigger and slower, gives 768 numbers per sentence

Both are frozen. They are only used to convert text to vectors, never trained.
The only thing being trained is the small network sitting on top. After each one
finishes encoding, the notebook deletes it and calls `torch.cuda.empty_cache()`
so the GPU memory comes back before the next model loads.

---

## 4. The scoring metric, MAP@3

The competition does not simply ask whether the top answer was right or wrong. It
lets you submit three guesses per question and gives partial credit based on where
the correct answer sits in your list.

| Correct answer is your... | You score |
| --- | --- |
| 1st guess | 1.00 |
| 2nd guess | 0.50 |
| 3rd guess | 0.33 |
| not in your top 3 | 0.00 |

MAP@3 is the average of that score over every question.

This is why the submission has three letters per row, and why it is always worth
filling all three slots. A second or third guess can only help; it can never lose
you points.

The `get_map3` function does this. It sorts the five probabilities from high to
low, takes the top three, and checks where the true answer landed.

---

## 5. The three models

### Model A — TF-IDF with a plain neural net

```
8000 TF-IDF numbers -> 400 -> 128 -> 5
```

ReLU activations, dropout 0.3, learning rate 0.001, batch size 64, 22 epochs.
The best epoch is kept based on validation macro f1.

This is the from-scratch baseline. No pretrained model anywhere. It exists to show
what a simple approach gets you, so the embedding models have something to be
measured against.

### Model B — MiniLM sentence embeddings

```
384 embedding numbers -> 200 -> 5
```

Tanh activation, dropout 0.2, learning rate 0.001, batch size 32, 18 epochs.
The best epoch is kept based on validation map3, the competition metric itself.

The head is deliberately small. The embeddings already carry most of the meaning,
so a large network on top would mostly just memorise the training set. **This is
the model that gets pushed to Hugging Face**, as the project brief requires.

### Model C — MPNet embeddings with a wider net

```
768 embedding numbers -> 384 -> 96 -> 5
```

ReLU, dropout 0.25, learning rate 0.0007, batch size 32, 18 epochs. Best epoch
kept on validation macro f1.

Same idea as B but with the larger embedding model, to see whether the bigger
pretrained model is worth the extra time it takes.

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

## 6. The train and validation split

The training data is split into two parts, about 85 percent to learn from and
about 15 percent held back. The held-back part is the *validation set*. It is the
only honest way to check whether the model has learned something general or has
just memorised the answers.

There is a trap here. This dataset contains **repeated questions**, the same
prompt appearing more than once. The notebook checks this before splitting:

```python
repeat_q = len(df_train) - df_train['prompt'].nunique()
```

If a plain random split were used, the same question could end up in both halves.
The model would see it during training, meet it again during validation, recognise
it, and get it right. The validation score would look great and mean nothing. This
is called **data leakage**.

The fix is `GroupShuffleSplit` with `groups=df_train['prompt']`. It keeps every
copy of a question together on one side, so a question is either fully in training
or fully in validation, never split across both. The notebook then prints a check:

```
overlap between train and val (want 0): 0
```

If that number is anything other than 0, the validation score cannot be trusted.

---

## 7. Overfitting, dropout, and keeping the best epoch

Overfitting is when the model memorises the training data instead of learning a
general pattern. You spot it when the training loss keeps falling but the
validation score stops improving or gets worse.

Two things guard against it.

**Dropout** randomly switches off a fraction of the neurons during each training
step. Because no single neuron can be relied on, the network is pushed to spread
what it learns across many of them. Dropout only runs during training; calling
`model.eval()` turns it off automatically, which is why the `run_epoch` function
switches between `model.train()` and `model.eval()`.

**Keeping the best checkpoint.** After every epoch the validation score is
computed, and the weights are saved to disk only when that score improves. At the
end of the loop the saved file is loaded back with `load_state_dict`. So even if
the last few epochs made the model worse, the version kept is the best one it ever
reached, not the final one.

---

## 8. Experiment tracking with Weights & Biases

Every epoch of every model logs its numbers, train loss, validation loss,
accuracy, f1 and map3, to Weights & Biases. This draws live curves in the browser.

The reason is practical. Three models at around twenty epochs each is roughly
sixty lines of printed numbers, and comparing those by eye is painful and
error-prone. On a graph you can see immediately where a model stopped improving or
started overfitting.

All runs go to the project `24f2005363-t22026`, grouped under the Hugging Face
username, with `job_type='train'` for the three training runs and
`job_type='inference'` for the final one. Each run gets a summary value at the end,
so the leaderboard view in W&B shows the best score per run without opening it.

---

## 9. Logins and secrets

The notebook never hardcodes a token. Both keys come from Kaggle secrets:

```python
secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
wandb_key = secrets.get_secret("WANDB_API_KEY")
```

The Hugging Face username is not typed in either. It is fetched from the hub after
logging in, with `HfApi().whoami()['name']`, and then used to build the repo name
and the W&B group name. That means the notebook runs correctly under any account
without editing a single string.

Before running, add both secrets in Kaggle under Add-ons > Secrets, using exactly
those two names, and turn the GPU on. The embedding models are very slow on CPU.

---

## 10. Pushing model B to Hugging Face

After the comparison, the notebook creates a repo named
`<your-username>/mcq-answer-predictor-t22026` and uploads these files:

| File | What it is |
| --- | --- |
| `net_b.pt` | the trained weights, the important one |
| `README.md` | a short model card with the validation scores |
| `score_table.csv` | the full A, B, C comparison |
| `winner.json` | which model scored highest on map3 |
| `compare_plot.png` | the comparison bar chart |

Two points that commonly cause confusion.

**Model B is uploaded even if it did not win.** The brief specifies model B, so it
goes up regardless of what the comparison says. If `winner.json` names model C,
that is expected behaviour and not a bug. The model card mentions the difference
openly rather than hiding it.

**`net_b.pt` is only the small head**, not the sentence transformer. It is well
under a megabyte, not hundreds. Anyone loading it also needs
`paraphrase-MiniLM-L6-v2` to produce the embeddings first, which is why the
embedding model name is written into the model card.

---

## 11. Inference straight from the hub

The last section downloads `net_b.pt` back from Hugging Face with
`hf_hub_download` and runs predictions with that copy.

At first this looks pointless, since the trained model is already sitting in
memory a few cells above. It is not. The point is to prove the upload actually
works. It is easy to upload a file that is empty, truncated, or the wrong shape,
and never find out because you kept using the in-memory version the whole time.

So this section deliberately starts fresh on all three counts:

- the weights come from the hub, not from the local `outputs` folder
- the validation and test text is embedded again from scratch, with a newly loaded
  `SentenceTransformer`, not reusing `emb_va` and `emb_te` from earlier
- a brand new empty `NetB` is created and the downloaded weights are loaded into it

If the score it prints matches the score model B got during training, the upload is
confirmed good.

### Reading the confusion matrix

The notebook then draws a 5x5 confusion matrix. Rows are the true answers, columns
are what the model predicted. Correct predictions sit on the diagonal running from
top-left to bottom-right. Anything off the diagonal is a mistake, and its position
tells you which letters are being mixed up. A bright column means the model leans
towards guessing that one letter too often. The bar chart beside it shows accuracy
per letter, which makes an imbalance easier to spot than the grid does.

---

## 12. Walking through the notebook

| Section | What happens |
| --- | --- |
| Install and imports | packages, device selection, `outputs` folder created |
| Reading the data | secrets, both logins, username fetched, three CSVs read |
| Quick look at data | `info()`, null check, bar chart of answer letters |
| Repeated question check | counts duplicate prompts before splitting |
| Split | label encoding, `GroupShuffleSplit`, overlap check printed |
| Text building | `join_row` applied to train, validation and test |
| Helper stuff | `get_map3`, `write_submission`, `SimpleData`, empty score table |
| Model A | TF-IDF fitted, `NetA` defined, `run_epoch` defined, 22 epochs |
| Model A scoring | validation and test probabilities, metrics, submission written |
| Model B | MiniLM encoding, `NetB`, 18 epochs, then scoring and submission |
| Model C | MPNet encoding, `NetC`, 18 epochs, then scoring and submission |
| Compare | score table sorted by map3, saved to CSV, bar chart drawn |
| Winner | best model recorded in `winner.json` |
| Push to hub | repo created, five files uploaded |
| Final printout | table, winner, repo name, W&B project, file listing |
| Hub inference | weights downloaded, text re-embedded, fresh `NetB`, metrics |
| Inference graphs | confusion matrix and per-class accuracy |
| Inference logging | final submission written, inference run logged to W&B |

A note on ordering. `run_epoch` is defined inside the Model A section but is used
by all three training loops, so the notebook has to be run top to bottom. Running
the Model B cells on a fresh kernel without running Model A first will fail with a
`NameError`.

---

## 13. Things worth knowing before running it

**Turn the GPU on.** Encoding the full training set with MPNet on CPU takes a very
long time. On GPU it is a few minutes.

**Both secrets must exist.** The notebook calls `get_secret` in an early cell, and
a missing secret stops it there rather than partway through training.

**Run it top to bottom.** Shared helpers and shared variables mean cells cannot be
run out of order.

**MPNet is the slow part.** If you only want to check that the pipeline works, run
model A and model B first and leave the model C cells until you are confident.

**Model C is loaded after model B is deleted.** Both encoders in memory at once
can exhaust GPU memory on the free Kaggle tier, which is why each is deleted right
after use.

---

## 14. Results

Fill this in after a full run, from the printed score table.

model   val_acc    val_f1  val_map3
0  model_a  1.000000  1.000000  1.000000
1  model_b  1.000000  1.000000  1.000000
2  model_c  0.996689  0.997419  0.998344

Winner on validation map3:
Score of the hub-downloaded model B at inference:
Leaderboard score:

A gap between the validation map3 and the leaderboard score is worth a sentence of
explanation. A small gap is normal. A large one usually points to either leakage
that the group split did not catch, or to too many decisions having been made
while looking at the same validation set.

---

## 15. Ideas for taking it further

- Average the probabilities of the three models. Models that make different
  mistakes often combine into something better than any one of them.
- Fine-tune the sentence transformer instead of freezing it. Slower and easier to
  overfit, but the ceiling is higher.
- Use cross-validation instead of one split, so every row is validated on once.
  This gives a much steadier estimate of the true score.
- Encode each option separately against the question rather than gluing all five
  into one string, which gives the model a clearer view of which option is which.
- Search the hyperparameters with Optuna rather than picking layer sizes and
  learning rates by hand.

---

## 16. Glossary

**Epoch** — one full pass through all the training data.

**Batch** — a small group of rows processed together, here 32 or 64. Faster than
one row at a time, and lighter on memory than all rows at once.

**Learning rate** — how big a step the model takes when correcting itself. Too
large and it overshoots, too small and it takes forever.

**Loss** — a number measuring how wrong the model is. Training means pushing this
down. Cross entropy is used here, standard for classification.

**Softmax** — turns raw output numbers into probabilities that add up to 1.

**Dropout** — randomly switching off neurons during training to discourage
memorising.

**Embedding** — a list of numbers representing the meaning of a piece of text.

**Frozen model** — a pretrained model used only to produce outputs, never trained.

**Accuracy** — the fraction of questions where the top guess was correct.

**F1 macro** — an average that treats every answer letter as equally important.
Useful when some letters appear more often than others, because a model that
always guesses the most common letter would score decent accuracy but poor macro
f1.

**Checkpoint** — a saved copy of the model weights at a particular point.

**Data leakage** — when information from the validation set reaches the model
during training, making the scores look better than they truly are.
