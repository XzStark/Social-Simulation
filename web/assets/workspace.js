/* Presentation enhancements only. Simulation requests remain in index.html. */
(() => {
  const get = (id) => document.getElementById(id);
  const modeButtons = [...document.querySelectorAll("[data-mode]")];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const backdrops = [
    ...document.querySelectorAll(
      ".modal-backdrop,.wizard-backdrop,.mode-backdrop",
    ),
  ];
  backdrops.forEach((backdrop) => {
    backdrop.setAttribute("role", "dialog");
    backdrop.setAttribute("aria-modal", "true");
    const title = backdrop.querySelector("h2,h3");
    if (title) {
      if (!title.id) title.id = `${backdrop.id}-title`;
      backdrop.setAttribute("aria-labelledby", title.id);
    }
  });
  function syncMode(mode) {
    modeButtons.forEach((button) =>
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode)),
    );
    try {
      localStorage.setItem("people_ui_mode", mode);
    } catch (_) {
      /* private storage */
    }
  }
  modeButtons.forEach((button) =>
    button.addEventListener("click", () => {
      if (typeof setUiMode === "function") setUiMode(button.dataset.mode);
      syncMode(button.dataset.mode);
    }),
  );
  get("pickProfessionalMode")?.addEventListener("click", () =>
    syncMode("professional"),
  );
  get("pickPublicMode")?.addEventListener("click", () => syncMode("public"));
  get("headerWizardBtn")?.addEventListener("click", () =>
    get("openWizardBtn")?.click(),
  );
  const status = get("status");
  status?.setAttribute("role", "status");
  status?.setAttribute("aria-live", "polite");
  get("errBox")?.setAttribute("role", "alert");
  function syncStatus() {
    const started =
      /运行中|完成|暂停|tick|第\s*\d|已中止|推演中|流式|progress|complete|paused|stopped|round|stream/i.test(
        status?.textContent || "",
      ) || !!get("finalMetrics")?.children.length;
    get("status-panel")?.classList.toggle("is-empty", !started);
    if (get("status-empty")) get("status-empty").hidden = started;
    if (started)
      document.querySelectorAll(".workspace canvas").forEach((canvas) => {
        if (window.Chart) window.Chart.getChart(canvas)?.resize();
      });
  }
  const stateObserver = new MutationObserver(syncStatus);
  if (status)
    stateObserver.observe(status, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  syncStatus();
  const links = [...document.querySelectorAll(".sidebar-nav a")];
  const visibleSections = new Set();
  const navObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) visibleSections.add(entry.target);
        else visibleSections.delete(entry.target);
      });
      const nearest = [...visibleSections].sort(
        (a, b) =>
          Math.abs(a.getBoundingClientRect().top - 110) -
          Math.abs(b.getBoundingClientRect().top - 110),
      )[0];
      if (nearest)
        links.forEach((link) =>
          link.setAttribute(
            "aria-current",
            String(link.hash === `#${nearest.id}`),
          ),
        );
    },
    { rootMargin: "-100px 0px -55% 0px", threshold: 0 },
  );
  links.forEach((link) => {
    const section = document.querySelector(link.hash);
    if (section) navObserver.observe(section);
    link.addEventListener("click", () => {
      links.forEach((item) =>
        item.setAttribute("aria-current", String(item === link)),
      );
      if (section && !reduce) {
        section.classList.remove("flash-focus");
        requestAnimationFrame(() => section.classList.add("flash-focus"));
      }
    });
  });
  const triggers = new WeakMap();
  const closeControls = {
    wizardBackdrop: "wizardCloseBtn",
    publicHistoryBackdrop: "publicHistoryCloseBtn",
    publicGuideBackdrop: "publicGuideCloseBtn",
    swanManualBackdrop: "swanManualCancelBtn",
    modePickerBackdrop: "pickProfessionalMode",
  };
  const dialogObserver = new MutationObserver((records) => {
    records.forEach(({ target }) => {
      const shown = target.classList.contains("show");
      target.setAttribute("aria-hidden", String(!shown));
      if (shown) {
        triggers.set(target, document.activeElement);
        requestAnimationFrame(() =>
          target
            .querySelector('button,input,select,textarea,[tabindex="0"]')
            ?.focus(),
        );
      } else {
        const trigger = triggers.get(target);
        if (trigger?.isConnected && trigger !== document.body) trigger.focus();
      }
    });
    const anyOpen = backdrops.some((el) => el.classList.contains("show"));
    document.body.style.overflow = anyOpen ? "hidden" : "";
    document
      .querySelector(".workspace-shell")
      ?.toggleAttribute("inert", anyOpen);
    document.querySelector(".site-header")?.toggleAttribute("inert", anyOpen);
  });
  backdrops.forEach((el) =>
    dialogObserver.observe(el, {
      attributes: true,
      attributeFilter: ["class"],
    }),
  );
  document.addEventListener("keydown", (event) => {
    const active = backdrops
      .filter((el) => el.classList.contains("show"))
      .sort(
        (a, b) =>
          Number(getComputedStyle(b).zIndex) -
          Number(getComputedStyle(a).zIndex),
      )[0];
    if (!active) return;
    if (event.key === "Escape" && closeControls[active.id]) {
      event.preventDefault();
      get(closeControls[active.id])?.click();
    }
    if (event.key === "Tab") {
      const focusable = [
        ...active.querySelectorAll(
          'button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),a[href],summary,[tabindex="0"]',
        ),
      ].filter((el) => el.getClientRects().length);
      const first = focusable[0],
        last = focusable.at(-1);
      if (
        event.shiftKey &&
        (document.activeElement === first ||
          !active.contains(document.activeElement))
      ) {
        event.preventDefault();
        last?.focus();
      } else if (
        !event.shiftKey &&
        (document.activeElement === last ||
          !active.contains(document.activeElement))
      ) {
        event.preventDefault();
        first?.focus();
      }
    }
  });
  window.addEventListener(
    "pagehide",
    () => {
      stateObserver.disconnect();
      navObserver.disconnect();
      dialogObserver.disconnect();
    },
    { once: true },
  );
})();
