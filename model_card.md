# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

This card covers both versions and compares them.

## 1. Model Overview

**Model type:**  
I compared both models. The rule based model is the main one I built and
tuned. The ML model was trained on the same data to see how a learned
model behaves differently.

**Intended purpose:**  
Classify short, casual text posts (like social media updates or chat
messages) as **positive**, **negative**, **neutral**, or **mixed**. It is
a learning project for seeing how data and design choices shape a model,
not a tool for making decisions about real people.

**How it works (brief):**

- _Rule based:_ The text is cleaned and split into tokens. Each token
  found in a positive word list counts as a positive signal, and each
  token in a negative word list counts as a negative signal. A negation
  word right before a mood word flips it ("not happy" is negative), and a
  sarcasm emoji (🙃, 🙄) turns all praise in the post into complaints. If a
  post has both kinds of signal, it is labeled mixed. Otherwise the sign
  of the score decides positive, negative, or neutral.
- _ML:_ Each post is turned into word counts (bag of words), and a
  logistic regression model learns which words tend to show up with which
  label in the training posts. It is never told which words are
  "positive"; it only sees the 14 examples and their labels.

## 2. Data

**Dataset description:**  
`SAMPLE_POSTS` has **14 posts**. The 6 starter posts are short, plain
sentences ("I love this class so much"). I added 8 posts written the way
people actually post online, for example:

- "no cap that concert was the best night of my life 😭🔥"
- "Oh great, my flight got delayed again. Love that for me 🙃"
- "lol I failed my driving test again 💀 whatever, third time's the charm"
- "ngl kinda miss high school sometimes but I wouldn't go back"

**Labeling process:**  
I labeled each post by the feeling a friend would take away from it, not
by the words in it. That means the sarcastic flight post is **negative**
even though it says "great" and "love", and the concert post is
**positive** even though 😭 is usually a crying emoji.

Posts that were hard to label or could have more than one valid label:

- _Driving test post:_ I chose **mixed** (a bad event, laughed off with
  "lol" and "third time's the charm"). Someone else could fairly call it
  negative.
- _High school post:_ I chose **mixed** (missing the past, but happier
  now). It could also be read as mildly positive.
- _"This is fine":_ labeled **neutral**, but online it is often used
  sarcastically to mean things are _not_ fine.

**Important characteristics of the dataset:**

- Contains slang ("lowkey", "no cap", "ngl", "lol")
- Contains emojis whose meaning depends on context (😭, 💀, 🙃, 🔥)
- Includes one sarcastic post
- 4 of the 14 posts express mixed feelings
- Includes plain, emotionless updates ("Just finished my laundry...")

**Possible issues with the dataset:**

- **Very small.** 14 posts cannot represent how people write.
- **Uneven labels.** 4 positive, 4 negative, 4 mixed, but only 2 neutral.
- **One writer.** All the posts and labels come from me, so they reflect
  my slang, my sense of humor, and my reading of tone.
- **Evaluation data = training data.** The same 14 posts were used to
  pick the word lists, train the ML model, _and_ measure accuracy. There
  is no separate test set.

## 3. How the Rule Based Model Works

**Your scoring rules:**

- **Preprocessing:** lowercases the text, shortens repeated letters
  ("sooooo" → "soo"), removes punctuation, keeps contractions ("can't")
  as one token, and keeps text emoticons (":)") and emojis as their own
  tokens. Without this, "great," would not match "great".
- **Word lists:** each word in `POSITIVE_WORDS` adds 1 and each word in
  `NEGATIVE_WORDS` subtracts 1. I expanded the starter lists with words
  and symbols from my posts: positive "hopeful", "proud", "best", "lol",
  "💀", "🔥"; negative "failed", "miss", "delayed", "empty".
- **Negation:** if the token right before a mood word is a negation word
  ("not", "never", "can't", ...), that word's signal is flipped. "I am not
  happy" counts as negative and "not bad" as positive.
- **Sarcasm markers:** if the post contains 🙃 or 🙄, every positive signal
  is counted as negative instead.
- **Label thresholds:** positive and negative signals are counted
  separately. Both present → **mixed**. Otherwise score > 0 → positive,
  score < 0 → negative, and 0 → neutral.

**Strengths of this approach:**

- It is easy to explain: you can point at the exact words that caused a
  label.
- It behaves predictably on clear, direct posts ("Today was a terrible
  day" → negative, "So excited for the weekend" → positive).
- Negation works for the simple case: "I am not happy about this" is
  correctly negative.
- Keeping the positive and negative counts separate lets it label posts
  like "Feeling tired but kind of hopeful" as mixed instead of letting the
  two signals cancel out to neutral.

**Weaknesses of this approach:**

- It only knows the words in its lists. Anything else is invisible.
- Each emoji or word has one fixed meaning, no matter the context.
- Negation only looks one word back.
- Sarcasm is only detected when the writer uses 🙃 or 🙄.

Section 6 gives specific examples of each weakness.

## 4. How the ML Model Works

**Features used:**  
Bag of words using scikit-learn's `CountVectorizer` with its default
settings, fed into `LogisticRegression`.

An important detail: the default `CountVectorizer` keeps only words of 2
or more letters and numbers. **It throws away every emoji** (and the word
"I"). I checked the learned vocabulary, and it contains no emojis at all.
So the ML model never sees 😭, 💀, 🔥, or 🙃.

**Training data:**  
The model trained on the 14 posts in `SAMPLE_POSTS` with `TRUE_LABELS`.

**Training behavior:**  
To test how sensitive it is to my labels, I changed a single label and
retrained, then compared predictions on 16 new sentences the model had
never seen:

- Changing the sarcastic flight post from negative to positive changed
  **5 of the 16** predictions. For example, "I'm devastated" flipped from
  negative to positive, and "my grandpa died 💀" also flipped to positive.
- Changing the high school post from mixed to negative changed 2 of the
  16 predictions.

One label out of 14 was enough to change the model's answer on posts that
have nothing to do with flights. With this little data, every label
carries a lot of weight.

**Strengths and weaknesses:**

- _Strength:_ It learns from labels without a hand-written word list, so
  it can pick up on phrases I never thought to add. It got "got the
  internship!!! still can't believe it" right (positive), which the rule
  based model misses.
- _Weakness:_ It mostly memorizes. It scores 1.00 on the posts it trained
  on, but on new sentences it learns spurious cues (see Section 5).

## 5. Evaluation

**How you evaluated the model:**  
I ran both models on the 14 labeled posts in `dataset.py`:

| Model                    | Accuracy on SAMPLE_POSTS |
| ------------------------ | ------------------------ |
| Rule based               | **0.86** (12 of 14)      |
| ML (logistic regression) | **1.00** (14 of 14)      |

Both numbers are measured on the same posts the models were built from,
so they are too optimistic. To get a more honest picture, I also tried 16
new sentences that neither model had seen. Some of the results are below.

**Examples of correct predictions (rule based):**

- "I am not happy about this" → **negative**. "happy" is a positive word,
  but "not" right before it flips the signal.
- "Oh great, my flight got delayed again. Love that for me 🙃" →
  **negative**. "great" and "love" would make it positive, but 🙃 is a
  sarcasm marker, so they count as negative, along with "delayed".
- "Lowkey stressed about finals but kind of proud of how much I've
  studied" → **mixed**. It finds "stressed" (negative) and "proud"
  (positive), so both signals are present.

**Examples of incorrect predictions:**

On the dataset:

- "got the internship!!! still can't believe it": rule based says
  **neutral**, true label is positive. No word in it is in either list.
  The ML model gets it right.
- "ngl kinda miss high school sometimes but I wouldn't go back": rule
  based says **negative**, true label is mixed. Only "miss" is recognized.
  The ML model gets it right.

On new sentences:

| Sentence                                     | Rule based   | ML           | What a person would say |
| -------------------------------------------- | ------------ | ------------ | ----------------------- |
| "I am so happy today"                        | positive     | **negative** | positive                |
| "this is awful"                              | negative     | **neutral**  | negative                |
| "the meeting is at 3"                        | neutral      | **positive** | neutral                 |
| "not the best day"                           | **positive** | **positive** | negative                |
| "my grandpa died 💀"                         | **positive** | negative     | negative                |
| "I absolutely love getting stuck in traffic" | **positive** | **positive** | negative                |
| "this song slaps fr"                         | **neutral**  | **negative** | positive                |
| "feeling blessed and grateful"               | **neutral**  | **negative** | positive                |

**How their failures differed:**

- The **rule based** model fails in a predictable way: when a word isn't
  in its lists, it shrugs and says neutral, and when a word _is_ in its
  lists, it trusts it completely, even in the wrong context.
- The **ML** model fails in surprising ways. "I am so happy today" comes
  out **negative**. The only training post containing "happy" is "I am
  not happy about this" (negative), and "am" also appears only in that
  post. The model learned "happy" and "am" as negative cues instead of
  learning what "not" does. It also guesses confidently on words it has
  never seen instead of saying neutral.

## 6. Limitations

- **Tiny vocabulary.** The rule based model only reacts to about 30 words
  and emojis. "this song slaps fr" and "feeling blessed and grateful" are
  clearly positive but score 0 → neutral, because none of those words are
  in the lists.
- **One meaning per emoji.** I added 💀 as positive because in my
  dataset it means "dying of laughter". But the model cannot
  tell joking 💀 from grieving 💀.
- **Negation only looks one word back.** "not the best day" is labeled
  **positive**. The word before "best" is "the", not "not". "not very happy with this"
  fails the same way.
- **Sarcasm needs an emoji.** The flight post is handled only because of
  🙃. Without it, "I absolutely love getting stuck in traffic" is labeled
  **positive** because "love" is a positive word and nothing marks the
  sarcasm.
- **Accuracy is measured on its own training data.** The 0.86 (rule
  based) and 1.00 (ML) scores come from the same 14 posts I used to build
  the word lists and train the model. On new sentences, both models do
  much worse.
- **The ML model ignores emojis entirely,** because the default
  `CountVectorizer` drops them before training.

### Known failures we chose not to fix

The rule based model gets 12 of 14 posts right (0.86 accuracy). The two
posts it still gets wrong were left unfixed on purpose.

**1. "got the internship!!! still can't believe it"**  
True label: positive. Predicted: neutral.

- _Why it fails:_ The excitement comes from the `!!!` and from the
  situation (getting an internship is good news), not from any mood word.
  The preprocessing removes punctuation, and none of the words left are in
  the positive or negative lists, so the score is 0.
- _Why it is a limitation:_ The model only understands mood words it has
  been given. It has no idea what events are good or bad, and it ignores
  emphasis like exclamation marks. Any post that shows emotion through
  context instead of vocabulary will be missed.
- _Why we did not fix it:_ Adding "internship" or "believe" to the
  positive list would make this post correct, but those words are not
  positive on their own ("I didn't get the internship", "I can't believe
  they did that"). The fix would only fit this one post and would cause
  wrong predictions on new posts.

**2. "ngl kinda miss high school sometimes but I wouldn't go back"**  
True label: mixed. Predicted: negative.

- _Why it fails:_ "miss" is counted as negative, but the positive half
  ("I wouldn't go back", meaning the person is happier now) has no mood
  word in it. Only one side is detected, so the post is labeled negative
  instead of mixed.
- _Why it is a limitation:_ The model reads words one at a time. It cannot
  understand a phrase whose meaning comes from the whole sentence rather
  than from any single word.
- _Why we did not fix it:_ Catching this would need a rule written just
  for this phrasing (for example, treating "wouldn't go back" as
  positive). That rule would rarely apply to other posts and could be
  wrong in other contexts.

Fixing either post would raise the accuracy score, but only by tuning the
model to its own evaluation data (overfitting). The model would look
better without getting any better at new posts it has never seen.

## 7. Ethical Considerations

**Misreading distress.** "my grandpa died 💀" is labeled positive by the
rule based model. If a system like this were used to flag posts from
people who might need support, it could rank a grieving person's post as
happy and skip it. "my dog passed away this morning. house feels so
empty" is only labeled negative because I added "empty" to the list
after seeing that exact post.

**Overt bias: errors cluster by language style.** The model did not
perform the same across writing styles. It does best on standard,
direct English with common mood words ("I love this class so much").
Errors pile up on:

- Slang that isn't in my lists: "this song slaps fr", "that movie was
  mid", "bro I'm weak 😂😂" all come out neutral, even though people who
  use that slang would read them as clearly positive or negative.
- Emojis used in their online-slang meaning (💀, 😭), which change meaning
  depending on the context.

The group most likely to be **underserved** is people who write mainly
in internet slang ("slaps", "I'm weak", "bussin"). Their posts mostly fall outside the word lists and get
labeled neutral, which makes their emotions invisible to the model. The
slang the model _does_ know ("lowkey", "no cap", "lol", "💀") is only
there because I happened to use it in my own posts.

**Covert compliance: what the AI coding assistant said.** I used Claude
Code to write the new posts, explain the code, and implement
`preprocess`, `score_text`, and `predict_label`. It pointed out
**accuracy** problems: that "Oh great... Love that for me 🙃" would score
positive because of sarcasm, that "not very happy" slips past the
one-word negation check, and that several posts scored 0 because their
words weren't in the lists. Its suggested fix was to "add words and
emojis to `POSITIVE_WORDS` and `NEGATIVE_WORDS`".

It did **not** raise any concerns about **bias or coverage**. It never
mentioned that the word lists only reflect one writer's slang, that
growing them by fitting the 14 sample posts would overfit to my own
style, or that giving 💀 a single meaning could misread posts about death
or grief. It treated missing words as an accuracy problem to patch, not
as a sign that some groups of writers would be served worse than others.
I think it should have flagged this, especially when it recommended
expanding the word lists from the sample posts, since that advice pushes
the model further toward my own way of writing.

**Privacy.** Running mood detection on personal messages means reading
private text and drawing conclusions about someone's emotional state.
People should know and agree before their messages are analyzed.

## 8. Ideas for Improvement

- **Add a separate test set** of posts written by other people, and
  report accuracy on that instead of on the training posts.
- **Collect posts and labels from more people** with different
  backgrounds and writing styles, so neither model only learns my
  vocabulary.
- **Widen the negation window** to 2–3 words, stopping at "but", so "not
  the best day" and "not very happy" are handled.
- **Let the ML model see emojis** by giving `CountVectorizer` a token
  pattern (or the `preprocess` method from `mood_analyzer.py`) that keeps
  them.
- **Add context rules for ambiguous emojis**, for example treating 💀 as
  negative when it appears next to words like "died" or "passed away".
- **Update `explain()`** in `mood_analyzer.py` to use the same negation and
  sarcasm logic as `score_text`, so its explanations match the real
  predictions.
- Try TF-IDF features or a small pretrained language model, which already
  know many words beyond the 14 posts.
