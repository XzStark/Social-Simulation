/* UI translation only: scenario briefs, user input, and model output retain their language. */
(() => {
  const normalize = (text) => text.replace(/\s+/g, " ").trim();
  const dictionary = window.PEOPLE_TRANSLATIONS || {};
  const english = new Map(
    Object.entries(dictionary).map(([key, value]) => [normalize(key), value]),
  );
  const chinese = new Map(
    Object.entries(dictionary).map(([key, value]) => [normalize(value), key]),
  );
  const originalText = new WeakMap();
  const originalAttributes = new WeakMap();
  const attrs = ["placeholder", "title", "aria-label", "alt"];
  let language = "en";
  try {
    language = localStorage.getItem("people_language") === "zh" ? "zh" : "en";
  } catch (_) {}
  function translated(text) {
    const key = normalize(text);
    if (/^（用时 [\d.]+s）$/.test(key) && !english.has(key)) {
      const value = key.replace(/^（用时 ([\d.]+)s）$/, "(elapsed $1 s)");
      english.set(key, value);
      chinese.set(value, key);
    }
    const zh = english.has(key) ? key : chinese.get(key);
    if (!zh) return text;
    const value = language === "zh" ? zh : english.get(zh);
    const whitespace = text.match(/^(\s*)[\s\S]*?(\s*)$/);
    return (whitespace?.[1] || "") + value + (whitespace?.[2] || "");
  }
  function translateNode(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      if (
        node.parentElement?.closest("script,style,textarea,[data-no-translate]")
      )
        return;
      const current = normalize(node.nodeValue);
      if (!current) return;
      translated(node.nodeValue); // Register supported dynamic UI messages before matching.
      if (
        !originalText.has(node) &&
        (english.has(current) || chinese.has(current))
      )
        originalText.set(node, node.nodeValue);
      let original = originalText.get(node);
      if (original !== undefined) {
        // A simulation update may reuse a text node; never overwrite fresh output.
        if (
          ![normalize(original), normalize(translated(original))].includes(
            current,
          )
        ) {
          if (!english.has(current) && !chinese.has(current)) {
            originalText.delete(node);
            return;
          }
          original = node.nodeValue;
          originalText.set(node, original);
        }
        const next = spaced(translated(original), node);
        if (node.nodeValue !== next) node.nodeValue = next;
      } else if (language === "en") {
        const next = spaced(node.nodeValue, node);
        if (node.nodeValue !== next) node.nodeValue = next;
      }
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;
    if (node.matches("script,style")) return;
    let originals = originalAttributes.get(node);
    if (!originals) {
      originals = {};
      originalAttributes.set(node, originals);
    }
    attrs.forEach((attr) => {
      const current = node.getAttribute(attr);
      if (!current) return;
      if (!originals[attr]) originals[attr] = current;
      const next = translated(originals[attr]);
      if (next !== current) node.setAttribute(attr, next);
    });
    if (!node.matches("textarea,[data-no-translate]"))
      [...node.childNodes].forEach(translateNode);
  }
  function spaced(text, node) {
    if (language !== "en") return text;
    // Inline emphasis and code do not carry word spacing in the Chinese source.
    const previous = node.previousSibling,
      following = node.nextSibling;
    if (
      previous?.nodeType === Node.ELEMENT_NODE &&
      /[A-Za-z0-9.)]$/.test(previous.textContent) &&
      /^[A-Za-z]/.test(text)
    )
      text = " " + text;
    if (
      following?.nodeType === Node.ELEMENT_NODE &&
      /[A-Za-z0-9.)]$/.test(text) &&
      /^[A-Za-z]/.test(following.textContent)
    )
      text += " ";
    return text;
  }
  function applyLanguage(next) {
    language = next === "zh" ? "zh" : "en";
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.documentElement.dataset.language = language;
    translateNode(document.head);
    translateNode(document.body);
    document
      .querySelectorAll("button[data-language]")
      .forEach((button) =>
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.language === language),
        ),
      );
    try {
      localStorage.setItem("people_language", language);
    } catch (_) {}
    window.dispatchEvent(
      new CustomEvent("people-language-change", { detail: { language } }),
    );
  }
  document
    .querySelectorAll("button[data-language]")
    .forEach((button) =>
      button.addEventListener("click", () =>
        applyLanguage(button.dataset.language),
      ),
    );
  const observer = new MutationObserver((records) => {
    records.forEach((record) => {
      if (record.type === "childList") record.addedNodes.forEach(translateNode);
      else translateNode(record.target);
    });
  });
  applyLanguage(language);
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
    attributes: true,
    attributeFilter: attrs,
  });
  window.peopleI18n = {
    get language() {
      return language;
    },
    translate: translated,
    setLanguage: applyLanguage,
  };
  window.addEventListener("pagehide", () => observer.disconnect(), {
    once: true,
  });
})();
