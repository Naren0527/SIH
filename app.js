/**
 * app.js — glues everything together:
 *   1. Get text (typed or spoken)
 *   2. Reorder into gloss (gloss.js)
 *   3. Look up each gloss word in the sign dictionary (now with smart matching)
 *   4. Render matches, or fall back to fingerspelling for unknown words
 */

let DICTIONARY = {};

function loadDictionary() {
  // No fetch needed anymore — dictionary.js sets window.SIGN_DICTIONARY directly,
  // which works fine even when the page is opened via double-click (file://).
  DICTIONARY = window.SIGN_DICTIONARY || {};
}

// --- 1. Aliases and Phrase Dictionaries ---
const WORD_ALIASES = {
  "ate": "eat", "eaten": "eat", "eating": "eat",
  "ran": "run", "running": "run",
  "went": "go", "gone": "go",
  "saw": "see", "seen": "see",
  "spoke": "speak", "spoken": "speak",
  "bought": "buy", "buying": "buy",
  "said": "tell", "told": "tell",
  "felt": "feel", "feeling": "feel",
  "children": "child", "men": "man", "women": "woman",
  "feet": "walk", "teeth": "clean",
  "physician": "doctor", "nurse": "doctor", "clinic": "hospital",
  "automobile": "car", "vehicle": "car", "taxi": "car", "cab": "car",
  "urgent": "emergency", "crisis": "emergency",
  "rapid": "fast", "quick": "fast",
  "hi": "hello", "hey": "hello", "greetings": "hello",
  "farewell": "goodbye",
  "assistance": "help", "assist": "help",
  "beverage": "drink", "meal": "food", "snack": "food"
};

const PHRASE_DICTIONARY = {
  "thank you": { type: "emoji", asset: "🙏" },
  "good morning": { type: "emoji", asset: "🌅" },
  "good afternoon": { type: "emoji", asset: "🌤️" },
  "good evening": { type: "emoji", asset: "🌆" },
  "good night": { type: "emoji", asset: "🌙" },
  "fire brigade": { type: "emoji", asset: "🔥" }
};

/**
 * fingerspell(word) -> array of single-letter "tiles" as a fallback
 * when a word isn't in the dictionary.
 */
function fingerspell(word) {
  return word.split("").map(letter => ({ type: "letter", asset: letter.toUpperCase() }));
}

// --- 2. Smart Root-Word Resolver (Stemmer) ---
function resolveBaseWord(word) {
  // Check exact match
  if (DICTIONARY[word]) return word;
  
  // Check aliases
  if (WORD_ALIASES[word] && DICTIONARY[WORD_ALIASES[word]]) return WORD_ALIASES[word];

  // If Compromise.js is added to index.html, use it for perfect grammar mapping
  if (window.nlp) {
    const doc = window.nlp(word);
    const singular = doc.nouns().toSingular().text();
    if (DICTIONARY[singular]) return singular;
    const infinitive = doc.verbs().toInfinitive().text();
    if (DICTIONARY[infinitive]) return infinitive;
  }

  // Manual fallback for suffixes
  const suffixes = [
    { rule: /ing$/, replace: "" },
    { rule: /ed$/, replace: "" },
    { rule: /es$/, replace: "" },
    { rule: /s$/, replace: "" },
    { rule: /ly$/, replace: "" }
  ];

  for (const { rule, replace } of suffixes) {
    if (rule.test(word)) {
      const candidate = word.replace(rule, replace);
      if (DICTIONARY[candidate]) return candidate;
      if (WORD_ALIASES[candidate]) return WORD_ALIASES[candidate];

      // Handle double consonants (e.g., "stopping" -> "stop")
      if (candidate.length > 2 && candidate[candidate.length - 1] === candidate[candidate.length - 2]) {
        const singleConsonant = candidate.slice(0, -1);
        if (DICTIONARY[singleConsonant]) return singleConsonant;
      }
    }
  }
  return null;
}

// --- 3. Greedy Matcher ---
function translateToSigns(sentence) {
  const gloss = window.SignSyncGloss.toGloss(sentence);
  const signs = [];
  let i = 0;

  while (i < gloss.length) {
    // Check 2-word phrases first
    if (i + 1 < gloss.length) {
      const twoWord = gloss[i] + " " + gloss[i + 1];
      if (PHRASE_DICTIONARY[twoWord]) {
        signs.push({ word: twoWord, ...PHRASE_DICTIONARY[twoWord] });
        i += 2;
        continue;
      }
    }

    const currentWord = gloss[i];
    const resolvedWord = resolveBaseWord(currentWord);

    if (resolvedWord && DICTIONARY[resolvedWord]) {
      signs.push({ word: currentWord, ...DICTIONARY[resolvedWord] });
    } else {
      fingerspell(currentWord).forEach(f => signs.push({ word: currentWord, ...f }));
    }
    i++;
  }
  
  return { gloss, signs };
}

// --- 4. Render Logic ---
function renderSigns(signs) {
  const grid = document.getElementById("signGrid");
  grid.innerHTML = "";

  if (signs.length === 0) {
    grid.innerHTML = `<p class="empty">Say or type something to see it in sign.</p>`;
    return;
  }

  signs.forEach(sign => {
    const tile = document.createElement("div");
    tile.className = "tile" + (sign.type === "letter" ? " tile-letter" : "");
    tile.innerHTML = `<span class="glyph">${sign.asset}</span><span class="label">${sign.word}</span>`;
    grid.appendChild(tile);
  });
}

function renderGloss(gloss) {
  document.getElementById("glossLine").textContent = gloss.length
    ? "Gloss: " + gloss.join(" · ").toUpperCase()
    : "";
}

function runTranslation(sentence) {
  if (!sentence.trim()) return;
  const { gloss, signs } = translateToSigns(sentence);
  renderGloss(gloss);
  renderSigns(signs);
}

/* ---------- Wire up UI ---------- */

window.addEventListener("DOMContentLoaded", () => {
  loadDictionary();

  const textInput = document.getElementById("textInput");
  const translateBtn = document.getElementById("translateBtn");
  const micBtn = document.getElementById("micBtn");
  const micStatus = document.getElementById("micStatus");

  translateBtn.addEventListener("click", () => runTranslation(textInput.value));
  textInput.addEventListener("keydown", e => {
    if (e.key === "Enter") runTranslation(textInput.value);
  });

  /* --- Voice input via the browser's built-in Web Speech API ---
     No ML training needed here: Chrome/Edge ship a free speech recognizer.
     For production/cross-browser support, swap this for a server call to
     OpenAI Whisper or another ASR API — the rest of the pipeline is identical. */
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    micBtn.disabled = true;
    micStatus.textContent = "Voice input isn't supported in this browser — try Chrome.";
  } else {
    const recognizer = new SpeechRecognition();
    recognizer.lang = "en-US";
    recognizer.interimResults = false;

    let listening = false;

    micBtn.addEventListener("click", () => {
      if (listening) {
        recognizer.stop();
        return;
      }
      recognizer.start();
    });

    recognizer.addEventListener("start", () => {
      listening = true;
      micBtn.classList.add("listening");
      micStatus.textContent = "Listening…";
    });

    recognizer.addEventListener("result", event => {
      const transcript = event.results[0][0].transcript;
      textInput.value = transcript;
      runTranslation(transcript);
    });

    recognizer.addEventListener("end", () => {
      listening = false;
      micBtn.classList.remove("listening");
      micStatus.textContent = "Tap the mic to speak";
    });

    recognizer.addEventListener("error", event => {
      micStatus.textContent = "Mic error: " + event.error;
    });
  }
});