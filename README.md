# SignSync — Text/Voice to Sign (starter)

A working, from-scratch prototype of the text/voice → sign pipeline. No build
tools, no server — just open `index.html` in Chrome.

## How it works right now

```
[typed text]  ──┐
                ├──> gloss.js (reorder into sign order) ──> app.js (dictionary lookup) ──> on-screen tiles
[spoken voice] ─┘         │                                        │
                    rule-based today                unknown words fall back to fingerspelling
```

- **dictionary.json** — the sign "vocabulary." Right now each word maps to a
  placeholder emoji. This is the first thing to upgrade: replace `asset` with
  a URL to a real ASL video clip, GIF, or image once you have (or license)
  actual footage. The lookup code doesn't care what `asset` points to.
- **gloss.js** — reorders English into rough sign order and drops filler
  words (articles, "to be" verbs). It's a handful of hand-written rules —
  good enough to ship, not linguistically accurate.
- **app.js** — wires it together, handles the Web Speech API for voice input,
  and fingerspells anything missing from the dictionary.

Try it: type "Where is the nearest pharmacy?" and watch it reorder to
`NEAREST PHARMACY WHERE` before rendering tiles.

## Growing the dictionary

This is the highest-leverage, lowest-tech-risk thing you can do next. Every
word you add makes the whole app better, with zero ML involved. Aim for the
~500 most common words for your use case (greetings, emergency terms, daily
needs) before worrying about the model below.

## Step 3 for real: training a text-to-gloss model

The rule-based `toGloss()` function is a stand-in for an actual model. Here's
the real path:

1. **Get a parallel dataset.** [ASLG-PC12](https://achrafothman.net/aslsmt/)
   pairs English sentences with their ASL gloss. There are others depending
   on the sign language you're targeting.
2. **Fine-tune a small seq2seq model** — T5-small or a similar encoder-decoder
   is plenty for this; you don't need a huge model. Rough shape in Python
   with Hugging Face `transformers`:

   ```python
   from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

   model = T5ForConditionalGeneration.from_pretrained("t5-small")
   tokenizer = T5Tokenizer.from_pretrained("t5-small")

   # dataset: list of {"input": "Where is the pharmacy?", "target": "PHARMACY WHERE"}
   # tokenize both sides, then:
   trainer = Trainer(model=model, args=TrainingArguments(output_dir="./gloss-model", num_train_epochs=5))
   trainer.train()
   ```

3. **Serve it** behind a small API endpoint (FastAPI/Flask is enough).
4. **Swap it in** — replace the body of `toGloss()` in `gloss.js` with a
   `fetch()` call to your endpoint. Nothing else in the app changes, because
   the dictionary-lookup and rendering code only cares about the gloss array
   it receives.

## Where this gets genuinely hard (and how to avoid it)

Generating brand-new sign *animation* from gloss (instead of playing a
pre-recorded clip) is an open research problem — datasets like
[How2Sign](https://how2sign.github.io/) and
[RWTH-PHOENIX-Weather-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
exist for exactly this, but it's a serious undertaking. Clip-based lookup
(what this prototype does) is what most production sign-language apps
actually ship, because it's reliable and the quality is as good as your
source footage — no model in the loop to get things wrong.

## Suggested order of work

1. Grow the dictionary with real assets (biggest visible improvement, no ML).
2. Improve `gloss.js` rules for your target sign language's grammar.
3. Fine-tune the T5 gloss model once rules stop being "good enough."
4. Only chase animation generation if clip-based signing is a genuine
   dead end for your use case.
