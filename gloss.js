/**
 * gloss.js — turns an English sentence into a rough sign-language "gloss":
 * the stripped-down, reordered word sequence that sign languages actually use.
 *
 * This version uses simple hand-written rules. It's intentionally basic —
 * it gets you a working pipeline TODAY. When you're ready for the real
 * ML step, replace the body of `toGloss()` with a call to your fine-tuned
 * text-to-gloss model (see README.md, Step 3) without changing anything
 * else in the app.
 */

// Words that most sign languages drop entirely (articles, most "to be" verbs).
const DROP_WORDS = new Set([
  "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
  "do", "does", "did", "to", "of", "for"
]);

// Question words move to the END of the sentence in ASL-style ordering.
const QUESTION_WORDS = new Set(["what", "where", "when", "who", "why", "how"]);

function tokenize(sentence) {
  return sentence
    .toLowerCase()
    .replace(/[^\w\s]/g, "")   // strip punctuation
    .split(/\s+/)
    .filter(Boolean);
}

/**
 * toGloss(sentence) -> array of gloss tokens, e.g.
 *   "Where is the nearest pharmacy?" -> ["nearest", "pharmacy", "where"]
 *
 * REPLACE THIS with a model call when you have one, e.g.:
 *   const gloss = await fetch("/api/gloss", { method: "POST", body: sentence }).then(r => r.json());
 */
function toGloss(sentence) {
  const tokens = tokenize(sentence);

  const question = tokens.filter(t => QUESTION_WORDS.has(t));
  const rest = tokens.filter(t => !QUESTION_WORDS.has(t) && !DROP_WORDS.has(t));

  return [...rest, ...question];
}

// Exposed for app.js (works as a plain <script> include, no bundler needed)
window.SignSyncGloss = { toGloss, tokenize };
