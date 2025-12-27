// Panelizer preview (desktop)
const state = {
  book: null,
  pageIndex: 0,
  page: null,
  panelIndex: 0,
  mode: "panel", // "panel" | "page"
  overlay: true,
  // Free pan/zoom state
  freeMode: false,
  transform: { tx: 0, ty: 0, scale: 1 },
};

const els = {
  meta: document.getElementById("meta"),
  status: document.getElementById("status"),
  error: document.getElementById("error"),
  viewport: document.getElementById("viewport"),
  img: document.getElementById("pageImage"),
  overlay: document.getElementById("overlay"),
  prevPage: document.getElementById("prevPage"),
  nextPage: document.getElementById("nextPage"),
  prevPanel: document.getElementById("prevPanel"),
  nextPanel: document.getElementById("nextPanel"),
  toggleMode: document.getElementById("toggleMode"),
  toggleOverlay: document.getElementById("toggleOverlay"),
  pageJump: document.getElementById("pageJump"),
  goPage: document.getElementById("goPage"),
};

function setError(message) {
  if (!message) {
    els.error.hidden = true;
    els.error.textContent = "";
    return;
  }
  els.error.hidden = false;
  els.error.textContent = message;
}

async function fetchJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}\n${text}`.trim());
  }
  return res.json();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function pageCount() {
  return state.book?.page_count ?? 0;
}

function currentPanelBBox() {
  if (!state.page?.order?.length) return null;
  const id = state.page.order[clamp(state.panelIndex, 0, state.page.order.length - 1)];
  return state.page.panels.find((p) => p.id === id)?.bbox ?? null;
}

function updateStatus() {
  const pageTotal = pageCount();
  const pageNum = pageTotal ? state.pageIndex + 1 : 0;

  let panelPart = "No panels";
  let panelConfText = null;
  if (state.page?.order?.length) {
    panelPart = `Panel ${state.panelIndex + 1}/${state.page.order.length}`;
    const id = state.page.order[clamp(state.panelIndex, 0, state.page.order.length - 1)];
    const conf = state.page.panels?.find((p) => p.id === id)?.confidence;
    panelConfText = typeof conf === "number" ? `Panel conf ${conf.toFixed(2)}` : "Panel conf ?";
  }

  const cvConf = state.page?.cv_confidence;
  const cvConfText = typeof cvConf === "number" ? `CV conf ${cvConf.toFixed(2)}` : "CV conf ?";
  const viewText = state.freeMode ? "free view" : `${state.mode} view`;
  const parts = [`Page ${pageNum}/${pageTotal}`, panelPart];
  if (panelConfText) parts.push(panelConfText);
  parts.push(cvConfText, viewText);
  els.status.textContent = parts.join(" · ");

  const dir = state.book?.reading_direction ?? "?";
  els.meta.textContent = `${dir.toUpperCase()} · ${pageTotal} pages`;
}

function clearOverlay() {
  els.overlay.replaceChildren();
}

function drawOverlay() {
  clearOverlay();
  if (!state.overlay) return;
  if (!state.page?.panels?.length) return;
  if (!els.img.naturalWidth || !els.img.naturalHeight) return;

  const currentId = state.page.order?.[state.panelIndex] ?? null;
  const panelsToDraw =
    state.mode === "panel"
      ? state.page.panels.filter((p) => p.id === currentId)
      : state.page.panels;
  if (!panelsToDraw.length) return;

  const orderIndexById = new Map();
  if (Array.isArray(state.page.order)) {
    state.page.order.forEach((id, i) => orderIndexById.set(id, i + 1));
  }
  for (const panel of panelsToDraw) {
    const [x, y, w, h] = panel.bbox;
    const box = document.createElement("div");
    box.className = "box" + (currentId === panel.id ? " current" : "");
    // Boxes are specified in natural image pixels; the overlay is transformed like the image.
    box.style.left = `${x}px`;
    box.style.top = `${y}px`;
    box.style.width = `${w}px`;
    box.style.height = `${h}px`;

    const label = document.createElement("div");
    label.className = "label";
    const n = orderIndexById.get(panel.id) ?? "?";
    const conf = typeof panel.confidence === "number" ? panel.confidence.toFixed(2) : "?";
    label.textContent = `#${n} cv:${conf}`;
    box.appendChild(label);

    els.overlay.appendChild(box);
  }
}

function syncOverlayToImage() {
  if (!els.img.naturalWidth || !els.img.naturalHeight) return;
  // Match the overlay's coordinate space to the untransformed image (natural pixels),
  // then apply the same transform as the image so bboxes align.
  els.overlay.style.width = `${els.img.naturalWidth}px`;
  els.overlay.style.height = `${els.img.naturalHeight}px`;
  els.overlay.style.transform = els.img.style.transform || "";
}

function applyTransform() {
  const { tx, ty, scale } = state.transform;
  els.img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  syncOverlayToImage();
  drawOverlay();
}

function computeFitTransform() {
  if (!els.img.naturalWidth || !els.img.naturalHeight) return null;

  const vw = els.viewport.clientWidth;
  const vh = els.viewport.clientHeight;
  if (!vw || !vh) return null;

  const iw = els.img.naturalWidth;
  const ih = els.img.naturalHeight;

  if (state.mode === "page") {
    const scale = Math.min(vw / iw, vh / ih);
    const tx = (vw - iw * scale) / 2;
    const ty = (vh - ih * scale) / 2;
    return { tx, ty, scale };
  }

  const bbox = currentPanelBBox();
  if (!bbox) return null;

  const [x, y, w, h] = bbox;
  const scale = Math.min(vw / w, vh / h);
  const tx = -x * scale + (vw - w * scale) / 2;
  const ty = -y * scale + (vh - h * scale) / 2;
  return { tx, ty, scale };
}

function fitImageToViewport() {
  if (!els.img.naturalWidth || !els.img.naturalHeight) return;

  // In free mode, just apply the current transform
  if (state.freeMode) {
    applyTransform();
    return;
  }

  const fit = computeFitTransform();
  if (!fit) {
    // No valid panel bbox in panel mode - fall back to page mode
    if (state.mode === "panel") {
      state.mode = "page";
      els.toggleMode.textContent = "Panel view";
      fitImageToViewport();
    }
    return;
  }

  state.transform = fit;
  applyTransform();
}

async function loadBook() {
  state.book = await fetchJSON("/api/book");
  state.pageIndex = clamp(state.pageIndex, 0, Math.max(0, pageCount() - 1));
  els.pageJump.value = String(state.pageIndex + 1);
  updateStatus();
}

async function loadPage() {
  setError("");
  const idx = state.pageIndex;

  state.page = await fetchJSON(`/api/page/${idx}.json`);
  state.panelIndex = 0;

  els.img.src = `/api/page/${idx}.png`;
  els.pageJump.value = String(idx + 1);
  updateStatus();
}

function setMode(mode) {
  state.mode = mode;
  state.freeMode = false; // Snap back to fitted view
  els.toggleMode.textContent = state.mode === "page" ? "Panel view" : "Page view";
  fitImageToViewport();
  updateStatus();
}

function setOverlay(on) {
  state.overlay = on;
  els.toggleOverlay.textContent = state.overlay ? "Overlay off" : "Overlay on";
  syncOverlayToImage();
  drawOverlay();
}

async function nextPanel() {
  const n = state.page?.order?.length ?? 0;
  if (!n) {
    if (state.pageIndex < pageCount() - 1) await nextPage();
    return;
  }

  if (state.panelIndex < n - 1) {
    state.panelIndex = clamp(state.panelIndex + 1, 0, n - 1);
    updateStatus();
    fitImageToViewport();
    return;
  }

  if (state.pageIndex < pageCount() - 1) await nextPage();
}

async function prevPanel() {
  const n = state.page?.order?.length ?? 0;
  if (!n) {
    if (state.pageIndex > 0) await prevPage();
    return;
  }

  if (state.panelIndex > 0) {
    state.panelIndex = clamp(state.panelIndex - 1, 0, n - 1);
    updateStatus();
    fitImageToViewport();
    return;
  }

  if (state.pageIndex <= 0) return;
  await prevPage();

  const newN = state.page?.order?.length ?? 0;
  state.panelIndex = newN ? newN - 1 : 0;
  updateStatus();
  fitImageToViewport();
}

async function nextPage() {
  const n = pageCount();
  if (!n) return;
  state.pageIndex = clamp(state.pageIndex + 1, 0, n - 1);
  await loadPage();
}

async function prevPage() {
  const n = pageCount();
  if (!n) return;
  state.pageIndex = clamp(state.pageIndex - 1, 0, n - 1);
  await loadPage();
}

function attachEvents() {
  els.prevPage.addEventListener("click", () => prevPage().catch((e) => setError(String(e))));
  els.nextPage.addEventListener("click", () => nextPage().catch((e) => setError(String(e))));
  els.prevPanel.addEventListener("click", () => prevPanel().catch((e) => setError(String(e))));
  els.nextPanel.addEventListener("click", () => nextPanel().catch((e) => setError(String(e))));
  els.toggleMode.addEventListener("click", () => setMode(state.mode === "page" ? "panel" : "page"));
  els.toggleOverlay.addEventListener("click", () => setOverlay(!state.overlay));
  function goToPage() {
    const wanted = Number.parseInt(els.pageJump.value, 10);
    if (!Number.isFinite(wanted)) return;
    state.pageIndex = clamp(wanted - 1, 0, Math.max(0, pageCount() - 1));
    loadPage().catch((e) => setError(String(e)));
  }
  els.goPage.addEventListener("click", goToPage);
  els.pageJump.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      goToPage();
    }
  });

  window.addEventListener("resize", () => fitImageToViewport());
  els.img.addEventListener("load", () => {
    syncOverlayToImage();
    fitImageToViewport();
  });

  // Keyboard navigation (viewport focusable)
  els.viewport.addEventListener("keydown", (e) => {
    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      setMode(state.mode === "page" ? "panel" : "page");
      return;
    }

    const isShift = e.shiftKey;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      if (isShift) nextPage().catch((err) => setError(String(err)));
      else nextPanel().catch((err) => setError(String(err)));
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (isShift) prevPage().catch((err) => setError(String(err)));
      else prevPanel().catch((err) => setError(String(err)));
    }
  });

  // Mouse wheel zoom
  els.viewport.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      if (!els.img.naturalWidth) return;

      const rect = els.viewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      // Zoom factor (smaller = finer control)
      const zoomFactor = 1.1;
      const delta = e.deltaY < 0 ? zoomFactor : 1 / zoomFactor;

      const oldScale = state.transform.scale;
      const newScale = clamp(oldScale * delta, 0.1, 20);

      // Zoom centered on mouse position
      const { tx, ty } = state.transform;
      const newTx = mouseX - (mouseX - tx) * (newScale / oldScale);
      const newTy = mouseY - (mouseY - ty) * (newScale / oldScale);

      state.transform = { tx: newTx, ty: newTy, scale: newScale };
      state.freeMode = true;
      applyTransform();
      updateStatus();
    },
    { passive: false },
  );

  // Mouse drag panning
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartTx = 0;
  let dragStartTy = 0;

  els.viewport.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return; // Left button only
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragStartTx = state.transform.tx;
    dragStartTy = state.transform.ty;
    els.viewport.style.cursor = "grabbing";
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStartX;
    const dy = e.clientY - dragStartY;
    state.transform.tx = dragStartTx + dx;
    state.transform.ty = dragStartTy + dy;
    state.freeMode = true;
    applyTransform();
    updateStatus();
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      els.viewport.style.cursor = "";
    }
  });

  // Touch swipe (basic)
  let touchStartX = null;
  let touchStartY = null;
  els.viewport.addEventListener(
    "touchstart",
    (e) => {
      if (e.touches.length !== 1) return;
      touchStartX = e.touches[0].clientX;
      touchStartY = e.touches[0].clientY;
    },
    { passive: true },
  );

  els.viewport.addEventListener(
    "touchend",
    (e) => {
      if (touchStartX == null || touchStartY == null) return;
      const t = e.changedTouches[0];
      const dx = t.clientX - touchStartX;
      const dy = t.clientY - touchStartY;
      touchStartX = null;
      touchStartY = null;

      if (Math.abs(dx) < 30 || Math.abs(dx) < Math.abs(dy)) return;
      if (dx < 0) nextPanel().catch(() => {});
      else prevPanel().catch(() => {});
    },
    { passive: true },
  );
}

async function main() {
  setError("");
  attachEvents();
  setMode("panel");
  setOverlay(true);

  try {
    await loadBook();
    await loadPage();
    els.viewport.focus();
  } catch (e) {
    setError(String(e));
  }
}

main();
