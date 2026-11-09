# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the meaning of the mood word right after them.
NEGATION_WORDS = {
    "not", "no", "never", "isn't", "wasn't", "don't", "doesn't",
    "didn't", "can't", "won't", "aren't", "ain't",
}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Normalizes repeated characters ("soooo" -> "soo", "!!!" -> "!!")
          - Keeps text emoticons (":)", ":-(", ":D") as single tokens
          - Splits emojis ("🥲", "😂") into their own tokens
          - Removes punctuation, but keeps apostrophes inside words ("can't")
        """
        cleaned = text.strip().lower()
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        tokens = re.findall(
            r"[:;]-?[()dp]"             # text emoticons
            r"|[a-z0-9]+(?:'[a-z]+)*"   # words and contractions
            r"|[^\x00-\x7f\s️‍]",  # emojis and other symbols
            cleaned,
        )

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        positive_count, negative_count = self.count_signals(text)
        return positive_count - negative_count

    def count_signals(self, text: str) -> Tuple[int, int]:
        """
        Count positive and negative signals in the text separately.

        Returns (positive_count, negative_count). Keeping the two counts
        apart lets predict_label spot "mixed" posts, where both kinds of
        signal show up and would otherwise cancel out in the total score.
        """
        # Enhancement: negation handling. A negation word directly before a
        # mood word flips its effect, so "not happy" counts as negative and
        # "not bad" counts as positive.
        tokens = self.preprocess(text)
        positive_count = 0
        negative_count = 0

        for i, token in enumerate(tokens):
            if token in self.positive_words:
                value = 1
            elif token in self.negative_words:
                value = -1
            else:
                continue

            if i > 0 and tokens[i - 1] in NEGATION_WORDS:
                value = -value

            if value > 0:
                positive_count += 1
            else:
                negative_count += 1

        return positive_count, negative_count

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        positive_count, negative_count = self.count_signals(text)

        # Both kinds of signal present means the post has mixed feelings,
        # even if the total score happens to lean one way.
        if positive_count > 0 and negative_count > 0:
            return "mixed"

        score = positive_count - negative_count
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
