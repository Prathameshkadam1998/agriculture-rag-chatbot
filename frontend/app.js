/* AgriBot frontend. No framework, no build step. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    threadWrap: $("thread-wrap"),
    welcome: $("welcome"),
    examples: $("examples"),
    messages: $("messages"),
    form: $("composer"),
    input: $("question"),
    send: $("send"),
    count: $("count"),
    status: $("status"),
    statusText: $("status-text"),
    statRows: $("stat-rows"),
    statChunks: $("stat-chunks"),
    sidebar: $("sidebar"),
    scrim: $("scrim"),
    menuBtn: $("menu-btn"),
    themeBtn: $("theme-btn"),
    newChat: $("new-chat"),
  };

  const ICONS = {
    bug: '<path d="M10 6.2a3 3 0 0 1 3 3v2.6a3 3 0 0 1-6 0V9.2a3 3 0 0 1 3-3ZM7.6 6.4 6 4.6M12.4 6.4 14 4.6M7 10H3.8M16.2 10H13M7.2 12.8l-2.6 1.8M12.8 12.8l2.6 1.8"/>',
    leaf: '<path d="M4.5 15.5c0-6.5 4-10.5 11-11 0 7-4 11-11 11ZM4.5 15.5 10 10"/>',
    layers: '<path d="m10 3.5 7 3.6-7 3.6-7-3.6 7-3.6ZM3.6 10.6 10 13.9l6.4-3.3M3.6 13.6 10 16.9l6.4-3.3"/>',
    flask: '<path d="M8 3.5h4M8.6 3.5v4.3l-4 6.6a1.5 1.5 0 0 0 1.3 2.3h8.2a1.5 1.5 0 0 0 1.3-2.3l-4-6.6V3.5M6.6 12.5h6.8"/>',
    sprout: '<path d="M10 17v-7M10 11c0-3.2-2.2-5-5.8-5 0 3.4 2.3 5 5.8 5ZM10 13c0-2.6 2-4.2 5.8-4.2 0 2.8-2.2 4.2-5.8 4.2Z"/>',
    drop: '<path d="M10 3.2c2.8 3.2 4.8 5.8 4.8 8.3a4.8 4.8 0 0 1-9.6 0c0-2.5 2-5.1 4.8-8.3Z"/>',
  };

  const THINKING_STEPS = [
    "Searching the knowledge base by meaning and by keyword",
    "Fusing both result lists",
    "Reranking candidates with a cross-encoder",
    "Writing the answer from the top sources",
  ];

  const state = { busy: false };

  /* ---------- tiny DOM helpers ---------- */

  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of [].concat(children)) {
      if (child != null) node.append(child);
    }
    return node;
  }

  function svg(inner, size = 16) {
    const wrap = document.createElement("span");
    wrap.innerHTML =
      `<svg viewBox="0 0 20 20" width="${size}" height="${size}" fill="none" stroke="currentColor" ` +
      `stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`;
    return wrap.firstChild;
  }

  function scrollToEnd() {
    requestAnimationFrame(() => {
      els.threadWrap.scrollTop = els.threadWrap.scrollHeight;
    });
  }

  /* ---------- safe, minimal markdown ---------- */

  // Turns **bold** and [1]-style citations into nodes. Everything else is text,
  // so nothing from the server is ever interpreted as HTML.
  function inline(text, onCite, sourceCount) {
    const fragment = document.createDocumentFragment();
    const pattern = /\*\*([^*]+)\*\*|\[(\d{1,2})\]/g;
    let last = 0;
    let match;
    while ((match = pattern.exec(text))) {
      if (match.index > last) fragment.append(text.slice(last, match.index));
      if (match[1]) {
        fragment.append(el("strong", { text: match[1] }));
      } else {
        const number = Number(match[2]);
        if (onCite && number >= 1 && number <= sourceCount) {
          fragment.append(
            el("button", {
              class: "cite",
              type: "button",
              text: String(number),
              "aria-label": `Show source ${number}`,
              onclick: () => onCite(number),
            })
          );
        } else {
          fragment.append(match[0]);
        }
      }
      last = pattern.lastIndex;
    }
    if (last < text.length) fragment.append(text.slice(last));
    return fragment;
  }

  function renderAnswer(text, onCite, sourceCount) {
    const root = el("div", { class: "answer" });
    let list = null;
    let paragraph = [];

    const flush = () => {
      if (paragraph.length) {
        root.append(el("p", {}, inline(paragraph.join(" "), onCite, sourceCount)));
        paragraph = [];
      }
    };

    for (const raw of text.replace(/\r/g, "").split("\n")) {
      const line = raw.trim();
      const bullet = line.match(/^(?:[-*•]|(\d+)[.)])\s+(.*)$/);
      if (bullet) {
        flush();
        const type = bullet[1] ? "ol" : "ul";
        if (!list || list.tagName.toLowerCase() !== type) {
          list = el(type);
          root.append(list);
        }
        list.append(el("li", {}, inline(bullet[2], onCite, sourceCount)));
      } else if (!line) {
        flush();
        list = null;
      } else {
        list = null;
        paragraph.push(line.replace(/^#{1,6}\s+/, ""));
      }
    }
    flush();
    return root;
  }

  /* ---------- messages ---------- */

  function addUserMessage(text) {
    const message = el("div", { class: "msg msg-user" }, el("div", { class: "bubble", text }));
    els.messages.append(message);
    return message;
  }

  // Put the question at the top of the view so the answer reads from its start,
  // instead of jumping to the bottom of a long answer card.
  function scrollToTurn(node) {
    if (!node) return scrollToEnd();
    requestAnimationFrame(() => {
      const top = node.getBoundingClientRect().top - els.threadWrap.getBoundingClientRect().top;
      els.threadWrap.scrollTop += top - 12;
    });
  }

  function botShell(cardClass = "") {
    const card = el("div", { class: `card ${cardClass}`.trim() });
    const avatar = el("div", { class: "avatar", "aria-hidden": "true" }, svg(ICONS.sprout, 18));
    const message = el("div", { class: "msg msg-bot" }, [avatar, card]);
    els.messages.append(message);
    return card;
  }

  function addThinking() {
    const card = botShell();
    const steps = THINKING_STEPS.map((label) =>
      el("div", { class: "thinking-step" }, [el("i"), el("span", { text: label })])
    );
    card.append(el("div", { class: "thinking", role: "status", "aria-label": "Working on your answer" }, steps));

    let current = 0;
    const advance = () => {
      steps.forEach((step, index) => {
        step.classList.toggle("done", index < current);
        step.classList.toggle("active", index === current);
      });
    };
    advance();
    const timer = setInterval(() => {
      if (current < steps.length - 1) {
        current += 1;
        advance();
      }
    }, 650);

    return {
      card,
      stop() {
        clearInterval(timer);
        card.replaceChildren();
      },
    };
  }

  function generatorLabel(generator) {
    if (!generator) return "";
    const fallback = generator.includes("(fallback)");
    if (generator.startsWith("extractive")) {
      return fallback ? "Quoted from the knowledge base (writer unavailable)" : "Quoted from the knowledge base";
    }
    if (generator.startsWith("llm:")) return `Written by ${generator.slice(4)} from the sources`;
    return generator;
  }

  function sourceCard(source, number) {
    const item = el("li", { class: "source", id: `src-${source.uid}` });
    const answer = el("p", { class: "source-a", text: source.answer });
    const percent = Math.round(source.relevance * 100);

    const foot = el("div", { class: "source-foot" }, [
      el("span", { class: "bar", "aria-hidden": "true" }, el("span", { style: `width:${Math.max(percent, 3)}%` })),
      el("span", { text: `${percent}% relevant` }),
    ]);

    if (source.answer.length > 190) {
      const more = el("button", {
        class: "more",
        type: "button",
        text: "Show all",
        "aria-expanded": "false",
        onclick: () => {
          const open = item.classList.toggle("expanded");
          more.textContent = open ? "Show less" : "Show all";
          more.setAttribute("aria-expanded", String(open));
        },
      });
      foot.append(more);
    }

    item.append(
      el("span", { class: "source-num", text: String(number) }),
      el("div", {}, [el("p", { class: "source-q", text: source.question }), answer, foot])
    );
    return item;
  }

  function relatedBlock(label, questions) {
    if (!questions.length) return null;
    return el("div", { class: "related" }, [
      el("p", { class: "related-label", text: label }),
      el(
        "div",
        { class: "related-list" },
        questions.map((question) =>
          el("button", { class: "pill", type: "button", text: question, onclick: () => ask(question) })
        )
      ),
    ]);
  }

  function renderResponse(card, data) {
    if (data.kind !== "answer") {
      card.classList.add("notice");
      card.append(renderAnswer(data.answer));
      const related = relatedBlock("Closest topics I do know", data.related || []);
      if (related) card.append(related);
      return;
    }

    const uid = Math.random().toString(36).slice(2, 8);
    const sources = (data.sources || []).map((source, index) => ({ ...source, uid: `${uid}-${index + 1}` }));
    let details = null;

    const showSource = (number) => {
      const target = document.getElementById(`src-${uid}-${number}`);
      if (!target || !details) return;
      details.open = true;
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 1400);
    };

    const quoted = (data.generator || "").startsWith("extractive");
    if (quoted && sources[0]) {
      card.append(
        el("p", { class: "matched" }, ["Closest question in the knowledge base: ", el("q", { text: sources[0].question })])
      );
    }

    card.append(renderAnswer(data.answer, showSource, sources.length));

    const meta = el("div", { class: "meta" });
    if (data.confidence) {
      meta.append(
        el(
          "span",
          {
            class: "chip",
            dataset: { level: data.confidence.label },
            title: `Best source relevance: ${Math.round(data.confidence.score * 100)}%`,
          },
          [el("i"), `${data.confidence.label} confidence`]
        )
      );
    }
    meta.append(el("span", { text: generatorLabel(data.generator) }));
    if (data.timings_ms) {
      meta.append(el("span", { class: "dot", text: "·" }), el("span", { text: `${(data.timings_ms.total / 1000).toFixed(1)} s` }));
    }
    const copy = el("button", {
      class: "copy",
      type: "button",
      text: "Copy",
      onclick: async () => {
        try {
          await navigator.clipboard.writeText(data.answer);
          copy.textContent = "Copied";
        } catch {
          copy.textContent = "Copy failed";
        }
        setTimeout(() => (copy.textContent = "Copy"), 1600);
      },
    });
    meta.append(copy);
    card.append(meta);

    if (sources.length) {
      details = el("details", { class: "sources" }, [
        el("summary", {}, [
          svg('<path d="m7.5 5 5 5-5 5"/>', 14),
          sources.length === 1 ? "1 source" : `${sources.length} sources`,
        ]),
        el("ol", { class: "source-list" }, sources.map((source, index) => sourceCard(source, index + 1))),
      ]);
      card.append(details);
    }

    const related = relatedBlock("Related questions", data.related || []);
    if (related) card.append(related);
  }

  function renderError(card, message, question) {
    card.classList.add("error");
    card.append(
      el("div", { class: "answer" }, el("p", { text: message })),
      el("button", {
        class: "btn retry",
        type: "button",
        text: "Try again",
        onclick: () => {
          if (state.busy) return;
          card.closest(".msg").remove();
          ask(question, { repeat: true });
        },
      })
    );
  }

  /* ---------- asking ---------- */

  async function ask(question, { repeat = false } = {}) {
    question = question.trim();
    if (!question || state.busy) return;

    state.busy = true;
    updateSend();
    els.welcome.hidden = true;
    const turn = repeat ? els.messages.lastElementChild : addUserMessage(question);
    const thinking = addThinking();
    scrollToEnd();

    try {
      const response = await fetch("api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      let data = null;
      try {
        data = await response.json();
      } catch {
        /* non-JSON error page */
      }

      if (!response.ok) {
        let detail = data && data.detail;
        if (Array.isArray(detail)) detail = detail.map((item) => String(item.msg).replace(/^Value error, /, "")).join(" ");
        throw new Error(
          response.status === 429
            ? detail || "Too many questions in a short time. Please wait a minute."
            : detail || `The server returned an error (${response.status}).`
        );
      }

      thinking.stop();
      renderResponse(thinking.card, data);
    } catch (error) {
      thinking.stop();
      const offline = error instanceof TypeError;
      renderError(
        thinking.card,
        offline
          ? "I couldn't reach the server. On free hosting it sleeps when idle and takes about a minute to wake up."
          : error.message,
        question
      );
    } finally {
      state.busy = false;
      updateSend();
      scrollToTurn(turn);
      if (window.matchMedia("(min-width: 901px)").matches) els.input.focus();
    }
  }

  /* ---------- composer ---------- */

  function updateSend() {
    els.send.disabled = state.busy || !els.input.value.trim();
  }

  function autosize() {
    // An empty box falls back to its natural one-row height from the markup.
    els.input.style.height = "";
    if (els.input.value) {
      els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
    }
    const length = els.input.value.length;
    const max = els.input.maxLength;
    els.count.hidden = length < max * 0.8;
    els.count.textContent = `${length} / ${max}`;
    els.count.classList.toggle("near", length >= max);
  }

  els.input.addEventListener("input", () => {
    autosize();
    updateSend();
  });

  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      els.form.requestSubmit();
    }
  });

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = els.input.value;
    if (!question.trim() || state.busy) return;
    els.input.value = "";
    autosize();
    ask(question);
  });

  /* ---------- sidebar, theme, reset ---------- */

  function setSidebar(open) {
    els.sidebar.classList.toggle("open", open);
    els.scrim.hidden = !open;
    els.menuBtn.setAttribute("aria-expanded", String(open));
  }
  els.menuBtn.addEventListener("click", () => setSidebar(!els.sidebar.classList.contains("open")));
  els.scrim.addEventListener("click", () => setSidebar(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setSidebar(false);
  });

  els.themeBtn.addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {
      localStorage.setItem("agribot-theme", root.dataset.theme);
    } catch {
      /* private mode: the choice just won't persist */
    }
  });

  els.newChat.addEventListener("click", () => {
    if (state.busy) return;
    els.messages.replaceChildren();
    els.welcome.hidden = false;
    els.threadWrap.scrollTop = 0;
    setSidebar(false);
    els.input.focus();
  });

  /* ---------- start-up ---------- */

  function renderExamples(examples) {
    els.examples.replaceChildren(
      ...examples.map((example) =>
        el("button", { class: "example", type: "button", onclick: () => ask(example.question) }, [
          el("span", { class: "example-topic" }, [svg(ICONS[example.icon] || ICONS.leaf), example.topic]),
          el("span", { class: "example-q", text: example.question }),
        ])
      )
    );
  }

  async function loadInfo(attempt = 0) {
    try {
      const response = await fetch("api/info");
      if (!response.ok) throw new Error(String(response.status));
      const info = await response.json();

      const number = new Intl.NumberFormat();
      els.statRows.textContent = info.dataset_rows ? number.format(info.dataset_rows) : "—";
      els.statChunks.textContent = info.chunks ? number.format(info.chunks) : "—";
      els.status.dataset.state = "ready";
      els.statusText.replaceChildren(
        "Ready",
        el("span", { class: "status-detail", text: ` · ${number.format(info.chunks || 0)} passages indexed` })
      );

      if (info.max_question_chars) els.input.maxLength = info.max_question_chars;
      renderExamples(info.examples || []);
    } catch {
      if (attempt < 8) {
        els.status.dataset.state = "loading";
        els.statusText.textContent = "Waking the server…";
        setTimeout(() => loadInfo(attempt + 1), 4000);
      } else {
        els.status.dataset.state = "error";
        els.statusText.textContent = "Server unreachable";
      }
    }
  }

  loadInfo();
  autosize();
  updateSend();
  if (window.matchMedia("(min-width: 901px)").matches) els.input.focus();
})();
