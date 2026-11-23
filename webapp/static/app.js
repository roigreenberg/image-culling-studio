/* ImageCulling Studio — Fast Lightroom-style Frontend */
(() => {
  "use strict";

  const PAGE_SIZE = 350;
  const COLORS = [
    { key: "", name: "None", css: "cp-none", shortcut: "0" },
    { key: "Red", name: "Red", css: "cf-Red", shortcut: "6" },
    { key: "Yellow", name: "Yellow", css: "cf-Yellow", shortcut: "7" },
    { key: "Green", name: "Green", css: "cf-Green", shortcut: "8" },
    { key: "Blue", name: "Blue", css: "cf-Blue", shortcut: "9" },
    { key: "Purple", name: "Purple", css: "cf-Purple", shortcut: "P" },
  ];

  const LAYER_LABELS = {
    layer2_to_edit: "L2 · Edit",
    layer1_keep: "L1 · Keep",
    rejected: "Not Selected",
    to_delete: "To Delete",
    unculled: "Unculled",
  };

  const state = {
    directories: [],
    currentDirId: "",
    currentDir: null,
    settings: {},
    filters: {
      subfolder: "__all__",
      ai_layer: "",
      use_ai_layer: false,
      rating_val: "",   // "" = All, "0".."5"
      rating_op: "gte", // "gte" | "eq" | "lte"
      labels: [],       // multi-select array e.g. ["Green", "Yellow", "__none__"]
      speed_tier: "",
      camera: "",
      lens: "",
      track: "",
      species: "",
      scenario_type: "",
      hybrid_only: false,
      search: "",
      only_burst_best: false,
      sort_by: "overall_rank",
      sort_dir: "asc",
    },
    view: "grid", // "grid" | "loupe" | "compare"
    items: [],
    total: 0,
    offset: 0,
    loading: false,
    selectedIds: new Set(),
    activeIndex: 0,
    burstSiblings: [],
    filmstripMode: "list", // "list" | "burst"
    filmstripWindowStart: 0,
    filmstripWindowEnd: 0,
    zoomMode: "fit", // "fit" | 50 | 100 | 200 | 300 | "custom"
    zoomScale: 1.0,  // CSS scale relative to fit (1.0 = fit)
    focusX: 0.5,     // Normalized [0..1] image X centered in viewport
    focusY: 0.5,     // Normalized [0..1] image Y centered in viewport
    showAfPoints: true,
    showHighlights: true,
    showBbox: false,
    highlightMaskCanvas: null,
    highlightMaskKey: "",
    highlightClipPct: 0,
    isDragging: false,
    dragMoved: false,
    dragStartX: 0,
    dragStartY: 0,
    dragStartFocusX: 0.5,
    dragStartFocusY: 0.5,
    compareShowCrop: false,
    compareItems: [],
    ws: null,
    preloadCache: new Map(),
    fullPreloadCache: new Map(),
    loupeLoadedFullId: "",
    isHoldingFull: false,
    preHoldSrc: "",
    perfQueue: [],
    perfFlushTimer: null,
    saveSessionTimer: null,
  };

  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try {
        const j = await res.json();
        if (j.detail) msg = j.detail;
      } catch (_) {}
      throw new Error(msg);
    }
    return res.json();
  }

  let toastTimer = null;
  function toast(msg, ms = 2600) {
    const el = $("toast");
    el.textContent = msg;
    el.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), ms);
  }

  async function copyToClipboard(text, label) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    toast(`Copied ${label}: ${text}`, 2000);
  }

  function recordPerfEvent(ev) {
    state.perfQueue.push(ev);
    if (state.perfQueue.length >= 6) {
      flushPerfEvents();
    } else if (!state.perfFlushTimer) {
      state.perfFlushTimer = setTimeout(flushPerfEvents, 2500);
    }
  }

  function flushPerfEvents() {
    clearTimeout(state.perfFlushTimer);
    state.perfFlushTimer = null;
    if (!state.perfQueue.length) return;
    const batch = state.perfQueue.splice(0, 30);
    api("/api/perf/log", {
      method: "POST",
      body: JSON.stringify({ events: batch }),
    }).catch(() => {});
  }

  function getParentDirPath(sourcePath) {
    if (!sourcePath) return "";
    const idx = Math.max(sourcePath.lastIndexOf("\\"), sourcePath.lastIndexOf("/"));
    return idx > 0 ? sourcePath.slice(0, idx) : sourcePath;
  }

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function getSpeedInfo(expSec, expStr) {
    if (!expSec || expSec <= 0) {
      return { tier: "", icon: "⏱", label: "Unknown", cls: "", text: expStr || "—" };
    }
    if (expSec <= 0.000625) {
      return { tier: "super_fast", icon: "⚡", label: "Super Fast (Action/BIF)", cls: "spd-super_fast", text: expStr };
    }
    if (expSec <= 0.002) {
      return { tier: "fast", icon: "🚀", label: "Fast (Wildlife)", cls: "spd-fast", text: expStr };
    }
    if (expSec <= 0.008) {
      return { tier: "medium", icon: "⏱", label: "Medium (Perched)", cls: "spd-medium", text: expStr };
    }
    return { tier: "slow", icon: "🐢", label: "Slow (Motion Risk)", cls: "spd-slow", text: expStr };
  }

  function renderStarsText(rating) {
    const r = Number(rating || 0);
    let html = "";
    for (let i = 1; i <= 5; i++) {
      html += i <= r ? "★" : '<span class="star-off">★</span>';
    }
    return html;
  }

  function formatBytes(b) {
    if (!b) return "—";
    if (b > 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + " MB";
    if (b > 1024) return (b / 1024).toFixed(0) + " KB";
    return b + " B";
  }

  function formatEta(sec) {
    if (!sec || sec <= 0) return "";
    const s = Math.round(sec);
    if (s < 60) return `${s}s left`;
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m}m ${rem}s left`;
  }

  function saveLocalScrollState() {
    try {
      const gridEl = $("gridView");
      if (!state.currentDirId) return;
      const prev = getSavedLocalScrollState() || {};
      const activeImg = state.items[state.activeIndex];
      // Only update scrollTop when Grid view is actually visible (never overwrite with 0 from hidden Loupe!)
      const scrollTop = (state.view === "grid" && gridEl && gridEl.scrollHeight > 0)
        ? Math.round(gridEl.scrollTop || 0)
        : (prev.dir_id === state.currentDirId && prev.subfolder === state.filters.subfolder ? (prev.scrollTop || 0) : 0);
      sessionStorage.setItem("cull_grid_pos", JSON.stringify({
        dir_id: state.currentDirId,
        subfolder: state.filters.subfolder,
        filters: state.filters,
        view: state.view === "compare" ? "grid" : state.view,
        active_image_id: activeImg ? activeImg.image_id : (prev.active_image_id || ""),
        active_index: state.activeIndex || 0,
        scrollTop,
        loadedCount: Math.max(state.items.length, prev.loadedCount || 0),
      }));
    } catch (_) {}
  }

  function getSavedLocalScrollState() {
    try {
      const raw = sessionStorage.getItem("cull_grid_pos");
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  function scheduleSessionSave() {
    saveLocalScrollState();
    clearTimeout(state.saveSessionTimer);
    state.saveSessionTimer = setTimeout(() => {
      const local = getSavedLocalScrollState() || {};
      const activeImg = state.items[state.activeIndex];
      api("/api/session", {
        method: "PUT",
        body: JSON.stringify({
          state: {
            dir_id: state.currentDirId,
            filters: state.filters,
            view: state.view === "compare" ? "grid" : state.view,
            active_image_id: activeImg ? activeImg.image_id : "",
            active_index: state.activeIndex || 0,
            grid_scroll_top: local.scrollTop || 0,
            loaded_count: state.items.length,
          },
        }),
      }).catch(() => {});
    }, 300);
  }

  // ------------------------------------------------------------------
  // WebSocket Live Events
  // ------------------------------------------------------------------
  function connectWs() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/events`);
    state.ws = ws;
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "task_progress") {
          updateProgressPill(msg.data);
        } else if (msg.type === "log") {
          prependLiveLog(msg.data);
        } else if (msg.type === "refresh_directory") {
          if (msg.dir_id === state.currentDirId) {
            refreshDirectoryAndImages();
          }
        }
      } catch (_) {}
    };
    ws.onclose = () => {
      setTimeout(connectWs, 2500);
    };
  }

  function updateProgressPill(st) {
    const pill = $("progressPill");
    const text = $("progressText");
    const pct = $("progressPct");
    const fill = $("progressBarFill");
    const cancelBtn = $("btnCancelTask");

    pill.classList.remove("idle", "active", "error");
    if (st.active) {
      pill.classList.add("active");
      const eta = st.eta_seconds ? ` · ${formatEta(st.eta_seconds)}` : "";
      text.textContent = `${st.message}${eta}`;
      pct.textContent = `${Math.round(st.percent || 0)}%`;
      fill.style.width = `${Math.min(100, st.percent || 0)}%`;
      cancelBtn.classList.remove("hidden");
    } else {
      pill.classList.add(st.stage === "error" ? "error" : "idle");
      text.textContent = st.message || "Ready";
      pct.textContent = st.stage === "done" ? "100%" : "";
      fill.style.width = st.stage === "done" ? "100%" : "0%";
      cancelBtn.classList.add("hidden");
    }
  }

  function prependLiveLog(entry) {
    const list = $("eventLogList");
    if (!list) return;
    const div = document.createElement("div");
    div.className = "log-line";
    div.innerHTML = `
      <span class="log-ts">${escapeHtml(entry.timestamp?.slice(11, 19) || "")}</span>
      <span class="log-lvl lvl-${escapeHtml(entry.level)}">[${escapeHtml(entry.level)}]</span>
      <span class="log-cat">${escapeHtml(entry.category)}</span>
      <span>${escapeHtml(entry.message)}</span>
    `;
    list.prepend(div);
    while (list.children.length > 300) list.removeChild(list.lastChild);
  }

  // ------------------------------------------------------------------
  // Init & Directory Loading
  // ------------------------------------------------------------------
  async function init() {
    bindUI();
    connectWs();

    try {
      state.settings = await api("/api/settings");
      $("chkAutoAdvance").checked = !!state.settings.auto_advance_on_rate;
      state.showBbox = !!state.settings.show_bbox_overlay;
      if (state.settings.grid_card_size) {
        $("cardSizeSlider").value = state.settings.grid_card_size;
        document.documentElement.style.setProperty("--card-size", state.settings.grid_card_size + "px");
      }

      const data = await api("/api/directories");
      state.directories = data.directories || [];
      renderDirectorySelector();

      if (state.directories.length > 0) {
        const sess = data.session || {};
        const localPos = getSavedLocalScrollState();
        const merged = { ...sess, ...(localPos || {}) };
        const savedDir = state.directories.find((d) => d.dir_id === merged.dir_id);
        const initialDir = savedDir || state.directories[0];
        if (merged.filters) {
          Object.assign(state.filters, merged.filters);
        }
        const resumeScroll = (merged.dir_id === initialDir.dir_id) ? (merged.scrollTop ?? merged.grid_scroll_top ?? 0) : 0;
        const resumeLoaded = (merged.dir_id === initialDir.dir_id) ? Math.max(merged.loadedCount || 0, merged.loaded_count || 0, (merged.active_index || 0) + 60) : 0;
        const resumeImgId = (merged.dir_id === initialDir.dir_id) ? (merged.active_image_id || "") : "";
        const resumeView = (merged.dir_id === initialDir.dir_id) ? (merged.view || "grid") : "grid";
        await selectDirectory(initialDir.dir_id, resumeImgId, resumeView, resumeScroll, resumeLoaded);
      } else {
        toast("Click 'Open Folder' to add your first photo directory.");
      }
    } catch (err) {
      toast("Startup error: " + err.message, 5000);
    }
  }

  function renderDirectorySelector() {
    const sel = $("dirSelect");
    sel.innerHTML = "";
    for (const d of state.directories) {
      const opt = document.createElement("option");
      opt.value = d.dir_id;
      opt.textContent = `${d.name} (${d.stats?.total || 0})`;
      sel.appendChild(opt);
    }
    if (state.currentDirId) sel.value = state.currentDirId;
  }

  async function selectDirectory(dirId, resumeImageId = "", resumeView = "grid", resumeScrollTop = 0, resumeLoadedCount = 0) {
    state.currentDirId = dirId;
    $("dirSelect").value = dirId;
    state.selectedIds.clear();
    updateBatchBar();

    const dir = await api(`/api/directories/${dirId}`);
    state.currentDir = dir;
    $("dirPathLabel").textContent = dir.path;
    renderFolders(dir);
    updateLayerBadges(dir);
    await loadFacets();
    syncFilterControlsFromState();

    // Load enough images to restore exact previous scroll position or active picture cleanly
    const targetCount = Math.min(2000, Math.max(PAGE_SIZE, resumeLoadedCount || 0));
    await loadImages(true, targetCount);
    while ((resumeScrollTop > 0 || resumeImageId) && state.items.length < resumeLoadedCount && state.items.length < state.total) {
      if (resumeImageId && state.items.some((x) => x.image_id === resumeImageId)) break;
      await loadImages(false, Math.min(2000, resumeLoadedCount - state.items.length));
    }

    if (resumeImageId) {
      const idx = state.items.findIndex((x) => x.image_id === resumeImageId);
      if (idx >= 0) state.activeIndex = idx;
    }
    if (resumeView === "loupe" && state.items.length > 0) {
      setView("loupe");
    } else {
      setView("grid");
      if (resumeScrollTop > 0) {
        const gridEl = $("gridView");
        requestAnimationFrame(() => {
          gridEl.scrollTop = resumeScrollTop;
          setTimeout(() => { gridEl.scrollTop = resumeScrollTop; }, 120);
          setTimeout(() => { gridEl.scrollTop = resumeScrollTop; }, 350);
        });
      }
    }
  }

  async function refreshDirectoryAndImages() {
    if (!state.currentDirId) return;
    const gridEl = $("gridView");
    const prevScroll = gridEl ? gridEl.scrollTop : 0;
    const prevLoaded = Math.max(PAGE_SIZE, state.items.length);
    const data = await api("/api/directories");
    state.directories = data.directories || [];
    renderDirectorySelector();
    const dir = await api(`/api/directories/${state.currentDirId}`);
    state.currentDir = dir;
    renderFolders(dir);
    updateLayerBadges(dir);
    await loadFacets();
    await loadImages(true, Math.min(2000, prevLoaded));
    while (prevScroll > 0 && state.items.length < prevLoaded && state.items.length < state.total) {
      await loadImages(false, Math.min(2000, prevLoaded - state.items.length));
    }
    if (prevScroll > 0 && state.view === "grid" && gridEl) {
      requestAnimationFrame(() => { gridEl.scrollTop = prevScroll; });
    }
  }

  function renderFolders(dir) {
    const list = $("folderList");
    list.innerHTML = "";

    const st = dir.stats || {};
    const allBtn = document.createElement("button");
    allBtn.className = "folder-item" + (state.filters.subfolder === "__all__" ? " active" : "");
    allBtn.innerHTML = `
      <span>📁 All Folders</span>
      <span class="folder-sub-badges">
        <span class="badge badge-l2" title="Layer 2 To Edit">${st.layer2 || 0}</span>
        <span class="badge">${st.total || 0}</span>
      </span>
    `;
    allBtn.onclick = () => {
      state.filters.subfolder = "__all__";
      renderFolders(dir);
      updateLayerBadges(dir);
      loadFacets();
      loadImages(true);
    };
    list.appendChild(allBtn);

    for (const sf of dir.subfolders || []) {
      const btn = document.createElement("button");
      btn.className = "folder-item" + (state.filters.subfolder === sf.subfolder ? " active" : "");
      btn.innerHTML = `
        <span>📂 ${escapeHtml(sf.subfolder)}</span>
        <span class="folder-sub-badges">
          ${sf.layer2 ? `<span class="badge badge-l2" title="Layer 2 To Edit">${sf.layer2}</span>` : ""}
          ${sf.unculled ? `<span class="badge" style="color:#c4b5fd" title="Unculled">${sf.unculled} new</span>` : ""}
          <span class="badge">${sf.total}</span>
        </span>
      `;
      btn.onclick = () => {
        state.filters.subfolder = sf.subfolder;
        renderFolders(dir);
        updateLayerBadges(dir);
        loadFacets();
        loadImages(true);
      };
      list.appendChild(btn);
    }
  }

  function updateLayerBadges(dir) {
    let s = dir.stats || {};
    if (state.filters.subfolder && state.filters.subfolder !== "__all__") {
      const found = (dir.subfolders || []).find((x) => x.subfolder === state.filters.subfolder);
      if (found) s = found;
    }
    const useAi = !!state.filters.use_ai_layer;
    const l2 = useAi ? (s.layer2 || 0) : (s.color_layer2 ?? s.layer2 ?? 0);
    const l1Only = useAi ? (s.layer1 || 0) : (s.color_layer1 ?? s.layer1 ?? 0);
    const rej = useAi ? (s.rejected || 0) : (s.color_rejected ?? s.rejected ?? 0);
    const del = useAi ? (s.to_delete || 0) : (s.color_to_delete ?? s.to_delete ?? 0);
    $("cntAll").textContent = s.total || 0;
    $("cntL2").textContent = l2;
    $("cntL1").textContent = l1Only + l2;
    $("cntRej").textContent = rej;
    if ($("cntDel")) $("cntDel").textContent = del;
    $("cntUnc").textContent = s.unculled || 0;
  }

  async function refreshSidebarCounts() {
    if (!state.currentDirId) return;
    try {
      const dir = await api(`/api/directories/${state.currentDirId}`);
      state.currentDir = dir;
      renderFolders(dir);
      updateLayerBadges(dir);
    } catch (_) {}
  }

  async function loadFacets() {
    if (!state.currentDirId) return;
    const sub = state.filters.subfolder === "__all__" ? "" : state.filters.subfolder;
    const facets = await api(`/api/directories/${state.currentDirId}/facets?subfolder=${encodeURIComponent(sub)}`);
    const camSel = $("cameraFilter");
    const lensSel = $("lensFilter");
    camSel.innerHTML = '<option value="">All Cameras</option>' +
      (facets.cameras || []).map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
    lensSel.innerHTML = '<option value="">All Lenses</option>' +
      (facets.lenses || []).map((l) => `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`).join("");
    camSel.value = state.filters.camera || "";
    lensSel.value = state.filters.lens || "";

    const sc = facets.speed_counts || {};
    $("cntSpdSuper").textContent = sc.super_fast || 0;
    $("cntSpdFast").textContent = sc.fast || 0;
    $("cntSpdMed").textContent = sc.medium || 0;
    $("cntSpdSlow").textContent = sc.slow || 0;

    // Project Kestrel facets: Track, Species (BioCLIP 2), Hybrid Light
    const trk = facets.tracks || {};
    if ($("cntTrkWild")) $("cntTrkWild").textContent = trk.Wildlife || 0;
    if ($("cntTrkLand")) $("cntTrkLand").textContent = trk.Landscape || 0;
    if ($("cntHybrid")) $("cntHybrid").textContent = facets.hybrid_count || 0;
    const spSel = $("speciesFilter");
    if (spSel) {
      spSel.innerHTML = '<option value="">All Species (BioCLIP 2)</option>' +
        (facets.species || []).map((s) => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)} (${s.count})</option>`).join("");
      spSel.value = state.filters.species || "";
    }
  }

  function syncFilterControlsFromState() {
    document.querySelectorAll("#layerFilters .filter-row").forEach((b) => {
      b.classList.toggle("active", b.dataset.layer === state.filters.ai_layer);
    });
    if ($("chkUseAiLayer")) $("chkUseAiLayer").checked = !!state.filters.use_ai_layer;
    document.querySelectorAll("#ratingOpBar .rop-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.op === (state.filters.rating_op || "gte"));
    });
    document.querySelectorAll("#ratingFilterBar .rf-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.rating === String(state.filters.rating_val ?? ""));
    });
    const selLabels = Array.isArray(state.filters.labels) ? state.filters.labels : [];
    document.querySelectorAll("#colorFilterBar .cf-dot").forEach((b) => {
      const lbl = b.dataset.label;
      if (lbl === "__all__") {
        b.classList.toggle("active", selLabels.length === 0);
      } else {
        b.classList.toggle("active", selLabels.includes(lbl));
      }
    });
    document.querySelectorAll("#speedFilters .filter-row").forEach((b) => {
      b.classList.toggle("active", b.dataset.speed === state.filters.speed_tier);
    });
    document.querySelectorAll("#trackFilters .rf-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.track === (state.filters.track || ""));
    });
    if ($("speciesFilter")) $("speciesFilter").value = state.filters.species || "";
    if ($("scenarioFilter")) $("scenarioFilter").value = state.filters.scenario_type || "";
    if ($("chkHybridOnly")) $("chkHybridOnly").checked = !!state.filters.hybrid_only;
    $("chkBurstBest").checked = !!state.filters.only_burst_best;
    $("searchInput").value = state.filters.search || "";
    $("sortSelect").value = `${state.filters.sort_by}:${state.filters.sort_dir}`;
  }

  // ------------------------------------------------------------------
  // Load Images (Paginated)
  // ------------------------------------------------------------------
  async function loadImages(reset = false, customLimit = null) {
    if (!state.currentDirId || state.loading) return;
    state.loading = true;
    const batchLimit = customLimit ? Math.min(2000, Math.max(1, customLimit)) : PAGE_SIZE;
    if (reset) {
      state.offset = 0;
      state.items = [];
      state.filmstripWindowStart = 0;
      state.filmstripWindowEnd = 0;
      $("photoGrid").innerHTML = "";
    }
    const loadBtn = $("btnLoadMore");
    if (loadBtn) loadBtn.textContent = "Loading more...";

    const f = state.filters;
    const params = new URLSearchParams({
      dir_id: state.currentDirId,
      subfolder: f.subfolder === "__all__" ? "" : f.subfolder,
      ai_layer: f.ai_layer,
      use_ai_layer: String(!!f.use_ai_layer),
      label: Array.isArray(f.labels) ? f.labels.join(",") : "",
      speed_tier: f.speed_tier,
      camera: f.camera,
      lens: f.lens,
      track: f.track || "",
      species: f.species || "",
      scenario_type: f.scenario_type || "",
      hybrid_only: String(!!f.hybrid_only),
      search: f.search,
      only_burst_best: String(!!f.only_burst_best),
      sort_by: f.sort_by,
      sort_dir: f.sort_dir,
      offset: String(state.offset),
      limit: String(batchLimit),
    });
    if (f.rating_val !== "" && f.rating_val !== null && f.rating_val !== undefined) {
      params.set("rating_val", String(f.rating_val));
      params.set("rating_op", f.rating_op || "gte");
    }

    const tGrid0 = performance.now();
    try {
      const res = await api(`/api/images?${params.toString()}`);
      const tApi = performance.now();
      state.total = res.total;
      const startIdx = state.items.length;
      state.items.push(...res.items);
      state.offset += res.items.length;
      $("resultCount").textContent = `${state.total.toLocaleString()} images`;
      appendGridCards(res.items, startIdx);
      const tDone = performance.now();
      recordPerfEvent({
        event: "grid_batch",
        count: res.items.length,
        api_ms: Math.round(tApi - tGrid0),
        render_ms: Math.round(tDone - tApi),
        total_ms: Math.round(tDone - tGrid0),
        message: `Grid batch (${res.items.length} imgs): api ${Math.round(tApi - tGrid0)}ms, dom ${Math.round(tDone - tApi)}ms, total ${Math.round(tDone - tGrid0)}ms`,
      });
      $("loadMoreWrap").classList.toggle("hidden", state.items.length >= state.total);
      if (state.view === "loupe" && reset && state.items.length > 0) {
        state.activeIndex = 0;
        renderLoupe();
      }
      scheduleSessionSave();
    } catch (err) {
      toast("Failed to load images: " + err.message);
    } finally {
      state.loading = false;
      const loadBtn = $("btnLoadMore");
      if (loadBtn) loadBtn.textContent = "Load More Images";
    }
    // Automatically warm the next batch early ONLY when appropriate (never when #gridView is hidden!)
    if (state.items.length > 0 && state.items.length < state.total) {
      const gridEl = $("gridView");
      const gridVisible = state.view === "grid" && gridEl && gridEl.scrollHeight > gridEl.clientHeight + 200;
      const nearGridBottom = gridVisible && (
        gridEl.scrollTop + gridEl.clientHeight >= gridEl.scrollHeight - 4000 ||
        (gridEl.scrollTop + gridEl.clientHeight) / gridEl.scrollHeight >= 0.65
      );
      const nearLoupeEnd = state.view === "loupe" && state.activeIndex >= state.items.length - 80;
      if (reset || nearGridBottom || nearLoupeEnd) {
        setTimeout(() => {
          if (!state.loading && state.items.length < state.total) loadImages(false);
        }, 80);
      }
      // If filmstrip is open in list mode and near right edge, append newly loaded items
      if (state.view === "loupe" && state.filmstripMode === "list") {
        maybeExtendFilmstripRight();
      }
    }
  }

  function appendGridCards(items, startIdx) {
    const frag = document.createDocumentFragment();
    items.forEach((img, i) => {
      const idx = startIdx + i;
      frag.appendChild(createCardEl(img, idx));
    });
    $("photoGrid").appendChild(frag);
  }

  function createCardEl(img, idx) {
    const card = document.createElement("div");
    card.className = "photo-card" +
      (state.selectedIds.has(img.image_id) ? " selected" : "") +
      (img.effective_label ? ` label-${img.effective_label}` : "");
    card.dataset.id = img.image_id;
    card.dataset.idx = String(idx);

    const spd = getSpeedInfo(img.exposure_seconds, img.exposure_time);
    const scoreTxt = img.is_culled ? Math.round(img.overall_score || img.sharpness_score || 0) : "—";
    const layerLbl = LAYER_LABELS[img.ai_layer] || img.ai_layer;

    card.innerHTML = `
      <div class="card-thumb-wrap">
        <input type="checkbox" class="card-chk" ${state.selectedIds.has(img.image_id) ? "checked" : ""} />
        <img class="card-thumb" loading="lazy" decoding="async" src="/api/media/thumb/${encodeURIComponent(img.image_id)}" alt="${escapeHtml(img.filename)}" />
        <span class="card-layer-pill pill-${escapeHtml(img.ai_layer)}">${escapeHtml(layerLbl)}</span>
        ${img.burst_size > 1 ? `<span class="card-burst-badge">${escapeHtml(img.burst_id)} #${img.burst_rank}/${img.burst_size}</span>` : ""}
        <span class="card-score-badge" title="AI Score (Sharpness: ${Math.round(img.sharpness_score || 0)})">${scoreTxt}</span>
      </div>
      <div class="card-meta">
        <div class="card-row-1">
          <span class="card-filename" title="${escapeHtml(img.filename)}">${escapeHtml(img.filename)}</span>
          <span class="spd-badge ${spd.cls}" title="Shutter: ${escapeHtml(spd.text)} — ${escapeHtml(spd.label)}">${spd.icon} ${escapeHtml(spd.text)}</span>
        </div>
        <div class="card-row-2">
          <span class="card-stars">${renderStarsText(img.effective_rating)}</span>
          <span>${img.focal_length ? Math.round(img.focal_length) + "mm" : ""} ${img.iso ? "ISO " + img.iso : ""}</span>
        </div>
      </div>
    `;

    const chk = card.querySelector(".card-chk");
    chk.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSelect(img.image_id, chk.checked);
      card.classList.toggle("selected", chk.checked);
    });

    card.addEventListener("click", (e) => {
      if (e.ctrlKey || e.metaKey) {
        const next = !state.selectedIds.has(img.image_id);
        toggleSelect(img.image_id, next);
        chk.checked = next;
        card.classList.toggle("selected", next);
        return;
      }
      state.activeIndex = idx;
      // Always open a newly selected image from the gallery in Fit (zoomed out)
      state.zoomMode = "fit";
      state.zoomScale = 1.0;
      state.focusX = 0.5;
      state.focusY = 0.5;
      setView("loupe");
    });

    return card;
  }

  function refreshCardInDom(img) {
    const card = document.querySelector(`.photo-card[data-id="${CSS.escape(img.image_id)}"]`);
    if (!card) return;
    card.className = "photo-card" +
      (state.selectedIds.has(img.image_id) ? " selected" : "") +
      (img.effective_label ? ` label-${img.effective_label}` : "");
    const stars = card.querySelector(".card-stars");
    if (stars) stars.innerHTML = renderStarsText(img.effective_rating);
  }

  function toggleSelect(imageId, selected) {
    if (selected) state.selectedIds.add(imageId);
    else state.selectedIds.delete(imageId);
    updateBatchBar();
  }

  function updateBatchBar() {
    const n = state.selectedIds.size;
    $("ddSelCount").textContent = String(n);
    $("batchCount").textContent = `${n} selected`;
    $("batchBar").classList.toggle("hidden", n === 0);
  }

  // ------------------------------------------------------------------
  // View Switcher (Grid / Loupe / Compare)
  // ------------------------------------------------------------------
  function setView(viewName) {
    const prevView = state.view;
    state.view = viewName;
    if (viewName === "loupe" && prevView !== "loupe") {
      // Returning from Gallery/Compare into Loupe always starts in Fit
      state.zoomMode = "fit";
      state.zoomScale = 1.0;
      state.focusX = 0.5;
      state.focusY = 0.5;
    }
    document.querySelectorAll(".view-switcher .btn-seg").forEach((b) => {
      b.classList.toggle("active", b.dataset.view === viewName);
    });
    $("gridView").classList.toggle("hidden", viewName !== "grid");
    $("loupeView").classList.toggle("hidden", viewName !== "loupe");
    $("compareView").classList.toggle("hidden", viewName !== "compare");

    if (viewName === "loupe") {
      if (state.items.length === 0) {
        toast("No images match current filters.");
        setView("grid");
        return;
      }
      if (state.activeIndex >= state.items.length) state.activeIndex = 0;
      renderLoupe();
    } else if (viewName === "compare") {
      openCompareView();
    } else if (viewName === "grid" && prevView === "loupe") {
      // Returning from Loupe (Esc / G) -> scroll straight to the exact image you just left!
      const activeImg = state.items[state.activeIndex];
      if (activeImg) {
        requestAnimationFrame(() => {
          const card = document.querySelector(`.photo-card[data-id="${CSS.escape(activeImg.image_id)}"]`);
          if (card) {
            card.scrollIntoView({ behavior: "instant", block: "center" });
            saveLocalScrollState();
          }
        });
      }
    }
    scheduleSessionSave();
  }

  // ------------------------------------------------------------------
  // Super-Fast Loupe View + ±5 HQ Preloading + Zoom + BBox Overlay
  // ------------------------------------------------------------------
  async function renderLoupe() {
    const img = state.items[state.activeIndex];
    if (!img) return;

    // Auto-load next page MUCH earlier (100 images before end) so Loupe never stalls
    if (state.activeIndex >= state.items.length - 100 && state.items.length < state.total) {
      loadImages(false);
    }

    const tNav0 = performance.now();
    const hqUrl = `/api/media/hq/${encodeURIComponent(img.image_id)}`;
    const fullUrl = getZoomMediaUrl(img.image_id);
    const loupeImg = $("loupeImg");
    const isZoomed = state.zoomScale > 1.01 || state.zoomMode !== "fit";
    const hasCachedFull = state.fullPreloadCache.has(img.image_id);
    const hasCachedHq = state.preloadCache.has(img.image_id);
    const navImgId = img.image_id;

    if (isZoomed && hasCachedFull) {
      state.loupeLoadedFullId = img.image_id;
      loupeImg.src = fullUrl;
      updateResQualityBadge("full", state.fullPreloadCache.get(img.image_id));
    } else {
      state.loupeLoadedFullId = "";
      loupeImg.src = hqUrl;
      updateResQualityBadge("hq");
    }

    loupeImg.onload = () => {
      const tImgLoaded = performance.now();
      // Never change zoomScale during Hold-Q 45MP peek/release so framing stays 100% locked!
      if (!state.isHoldingFull && !state.skipNextZoomRecompute && typeof state.zoomMode === "number") {
        state.zoomScale = computeCssScaleForPercent(state.zoomMode);
      }
      state.skipNextZoomRecompute = false;
      applyLoupeZoomTransform();
      const tOv0 = performance.now();
      drawLoupeOverlay(img);
      const tOv1 = performance.now();
      updateZoomPresetUI();
      if (state.items[state.activeIndex]?.image_id === navImgId) {
        const imgMs = Math.round(tImgLoaded - tNav0);
        const ovMs = Math.round(tOv1 - tOv0);
        const totMs = Math.round(tOv1 - tNav0);
        recordPerfEvent({
          event: "loupe_nav",
          filename: img.filename,
          zoomed: isZoomed,
          cached: isZoomed ? hasCachedFull : hasCachedHq,
          res: `${loupeImg.naturalWidth}x${loupeImg.naturalHeight}`,
          img_ms: imgMs,
          overlay_ms: ovMs,
          total_ms: totMs,
          message: `Loupe → ${img.filename} (${loupeImg.naturalWidth}x${loupeImg.naturalHeight}, ${isZoomed ? (hasCachedFull ? "zoom-cached" : "zoom-hq") : (hasCachedHq ? "preloaded" : "fetch")}): img ${imgMs}ms, overlay ${ovMs}ms, total ${totMs}ms`,
        });
      }
    };

    // If zoomed in and not yet showing full-res, seamlessly upgrade to full 45MP!
    if (isZoomed) {
      if (!hasCachedFull) upgradeLoupeToFullRes(img);
      preloadFullNeighbors(state.activeIndex);
    }

    // Aggressive preload next 4 & prev 2 HQ JPGs in browser cache for 0ms flipping!
    preloadNeighbors(state.activeIndex);

    // Top HUD — clickable folder (copies full folder path) & filename (copies filename)
    const folderFullPath = getParentDirPath(img.source_path);
    const fnEl = $("loupeFilename");
    fnEl.innerHTML =
      `<span class="copy-part" id="copyFolderPart" title="Click to copy full folder path:\n${escapeHtml(folderFullPath)}">${escapeHtml(img.subfolder)}</span>` +
      `<span class="copy-sep">/</span>` +
      `<span class="copy-part" id="copyFilePart" title="Click to copy filename">${escapeHtml(img.filename)}</span>`;
    $("copyFolderPart").onclick = (e) => {
      e.stopPropagation();
      copyToClipboard(folderFullPath, "folder path");
    };
    $("copyFilePart").onclick = (e) => {
      e.stopPropagation();
      copyToClipboard(img.filename, "filename");
    };
    const lp = $("loupeLayerPill");
    lp.textContent = LAYER_LABELS[img.ai_layer] || img.ai_layer;
    lp.className = `hud-chip pill-${img.ai_layer}`;

    const spd = getSpeedInfo(img.exposure_seconds, img.exposure_time);
    const sp = $("loupeSpeedPill");
    sp.className = `hud-chip spd-badge ${spd.cls}`;
    sp.textContent = `${spd.icon} ${spd.text}`;
    sp.title = spd.label;

    $("btnToggleAf").classList.toggle("active", state.showAfPoints);
    $("btnToggleHighlights").classList.toggle("active", state.showHighlights);
    $("btnToggleBbox").classList.toggle("active", state.showBbox);
    updateZoomPresetUI();

    // Bottom rating & color picker
    renderLoupeControls(img);

    // Right Inspector
    $("inspCropImg").src = `/api/media/crop/${encodeURIComponent(img.image_id)}`;
    renderInspector(img);

    // Always fetch image detail (pulls burst siblings + on-the-fly Canon AF focus points!)
    $("fsPosition").textContent = `${state.activeIndex + 1} / ${state.total}`;
    const reqImageId = img.image_id;
    api(`/api/images/${encodeURIComponent(reqImageId)}`)
      .then((res) => {
        if (res.image) {
          img.af_points = res.image.af_points || [];
        }
        state.burstSiblings = res.burst_siblings && res.burst_siblings.length ? res.burst_siblings : [img];
        $("fsBurstCount").textContent = String(state.burstSiblings.length);
        // Only redraw if user is still viewing this same image
        if (state.items[state.activeIndex]?.image_id === reqImageId) {
          drawLoupeOverlay(img);
          renderFilmstrip();
        }
      })
      .catch(() => {
        state.burstSiblings = [img];
        $("fsBurstCount").textContent = "1";
        renderFilmstrip();
      });
    scheduleSessionSave();
  }

  function preloadNeighbors(centerIdx) {
    const offsets = [1, 2, 3, 4, -1, -2];
    for (const off of offsets) {
      const idx = centerIdx + off;
      if (idx >= 0 && idx < state.items.length) {
        const id = state.items[idx].image_id;
        if (!state.preloadCache.has(id)) {
          const im = new Image();
          im.src = `/api/media/hq/${encodeURIComponent(id)}`;
          state.preloadCache.set(id, im);
          if (state.preloadCache.size > 40) {
            const firstKey = state.preloadCache.keys().next().value;
            state.preloadCache.delete(firstKey);
          }
        }
      }
    }
  }

  function getZoomMediaUrl(imageId, forceRaw = false) {
    const dim = forceRaw ? 8192 : (state.settings.zoom_preview_max_dim || 8192);
    return `/api/media/full/${encodeURIComponent(imageId)}?raw=${forceRaw || dim >= 8000 ? 1 : 0}&dim=${dim}`;
  }

  function updateResQualityBadge(mode, loadedImg = null) {
    const el = $("resQualityBadge");
    if (!el) return;
    el.classList.remove("full", "loading");
    if (mode === "full") {
      el.classList.add("full");
      if (loadedImg && loadedImg.naturalWidth && loadedImg.naturalHeight) {
        const mp = Math.round((loadedImg.naturalWidth * loadedImg.naturalHeight) / 1e6);
        el.textContent = mp >= 5 ? `${mp}MP` : "Full";
        el.title = `Full resolution (${loadedImg.naturalWidth} × ${loadedImg.naturalHeight})`;
      } else {
        el.textContent = "Full";
        el.title = "Full resolution image";
      }
    } else if (mode === "loading") {
      el.classList.add("loading");
      el.textContent = "Full…";
      el.title = "Upgrading to full resolution...";
    } else {
      el.textContent = "HQ";
      el.title = "Fast HQ preview (zoom in to auto-load full resolution)";
    }
  }

  function upgradeLoupeToFullRes(img) {
    if (!img || state.loupeLoadedFullId === img.image_id) return;
    const reqId = img.image_id;
    const fullUrl = getZoomMediaUrl(reqId);

    if (state.fullPreloadCache.has(reqId)) {
      const cached = state.fullPreloadCache.get(reqId);
      state.loupeLoadedFullId = reqId;
      $("loupeImg").src = fullUrl;
      updateResQualityBadge("full", cached);
      return;
    }

    updateResQualityBadge("loading");
    const tUp0 = performance.now();
    const im = new Image();
    im.src = fullUrl;
    const finish = () => {
      const upMs = Math.round(performance.now() - tUp0);
      state.fullPreloadCache.set(reqId, im);
      if (state.fullPreloadCache.size > 16) {
        const firstKey = state.fullPreloadCache.keys().next().value;
        state.fullPreloadCache.delete(firstKey);
      }
      recordPerfEvent({
        event: "zoom_upgrade",
        filename: img.filename,
        res: `${im.naturalWidth}x${im.naturalHeight}`,
        total_ms: upMs,
        message: `Zoom upgrade ${img.filename} (${im.naturalWidth}x${im.naturalHeight}): ${upMs}ms`,
      });
      // Swap seamlessly with ZERO flicker if user is still on this image
      if (state.view === "loupe" && state.items[state.activeIndex]?.image_id === reqId) {
        state.loupeLoadedFullId = reqId;
        $("loupeImg").src = fullUrl;
        updateResQualityBadge("full", im);
      }
    };
    if (im.decode) {
      im.decode().then(finish).catch(() => {
        if (im.complete && im.naturalWidth > 0) finish();
      });
    } else {
      im.onload = finish;
    }
  }

  function startHoldFullRes() {
    if (state.view !== "loupe" || state.isHoldingFull) return;
    const img = state.items[state.activeIndex];
    if (!img) return;
    state.isHoldingFull = true;
    const loupeImg = $("loupeImg");
    state.preHoldSrc = loupeImg.src;
    const btn = $("btnHoldFull");
    if (btn) btn.classList.add("active");
    updateResQualityBadge("loading");

    const reqId = img.image_id;
    const rawUrl = `/api/media/full/${encodeURIComponent(reqId)}?raw=1`;
    const t0 = performance.now();
    const im = new Image();
    im.src = rawUrl;
    const apply = () => {
      if (!state.isHoldingFull || state.items[state.activeIndex]?.image_id !== reqId) return;
      loupeImg.src = rawUrl;
      updateResQualityBadge("full", im);
      recordPerfEvent({
        event: "hold_full_45mp",
        filename: img.filename,
        res: `${im.naturalWidth}x${im.naturalHeight}`,
        total_ms: Math.round(performance.now() - t0),
        message: `Hold-Q 45MP peek ${img.filename} (${im.naturalWidth}x${im.naturalHeight}): ${Math.round(performance.now() - t0)}ms`,
      });
    };
    if (im.decode) {
      im.decode().then(apply).catch(() => { if (im.complete && im.naturalWidth) apply(); });
    } else {
      im.onload = apply;
    }
  }

  function stopHoldFullRes() {
    if (!state.isHoldingFull) return;
    state.isHoldingFull = false;
    state.skipNextZoomRecompute = true;
    const btn = $("btnHoldFull");
    if (btn) btn.classList.remove("active");
    const img = state.items[state.activeIndex];
    if (state.view === "loupe" && img && state.preHoldSrc) {
      $("loupeImg").src = state.preHoldSrc;
      const isZoomed = state.zoomScale > 1.01 || state.zoomMode !== "fit";
      updateResQualityBadge(isZoomed && state.loupeLoadedFullId === img.image_id ? "full" : "hq", state.fullPreloadCache.get(img.image_id));
    }
    state.preHoldSrc = "";
  }

  function preloadFullNeighbors(centerIdx) {
    for (const off of [1, -1]) {
      const idx = centerIdx + off;
      if (idx >= 0 && idx < state.items.length) {
        const id = state.items[idx].image_id;
        if (!state.fullPreloadCache.has(id)) {
          const im = new Image();
          im.src = getZoomMediaUrl(id);
          im.onload = () => {
            state.fullPreloadCache.set(id, im);
            if (state.fullPreloadCache.size > 10) {
              const firstKey = state.fullPreloadCache.keys().next().value;
              state.fullPreloadCache.delete(firstKey);
            }
          };
        }
      }
    }
  }

  function getCameraFocusCenter(img) {
    const pts = img?.af_points;
    if (Array.isArray(pts) && pts.length > 0) {
      const primary = pts.find((p) => p.in_focus) || pts[0];
      return { x: primary.cx, y: primary.cy, hasAf: true };
    }
    return { x: 0.5, y: 0.5, hasAf: false };
  }

  function getLoupeBaseSize() {
    const el = $("loupeImg");
    const baseW = el.offsetWidth || el.clientWidth || 1;
    const baseH = el.offsetHeight || el.clientHeight || 1;
    const natW = el.naturalWidth || baseW;
    const natH = el.naturalHeight || baseH;
    return { baseW, baseH, natW, natH };
  }

  function getFullRefWidth() {
    const img = state.items[state.activeIndex];
    const { baseW, natW } = getLoupeBaseSize();
    // Always use consistent full sensor width (e.g. 8192 for R5m2 landscape, 5464 for portrait)
    // so 25%/50%/100%/200% zoom scale NEVER jumps when HQ swaps to 45MP!
    if (img && img.preview_width >= 2000) return img.preview_width * 2;
    return Math.max(natW, baseW * 2);
  }

  function computeCssScaleForPercent(pct) {
    const { baseW } = getLoupeBaseSize();
    const oneToOneCss = Math.max(1.6, getFullRefWidth() / Math.max(1, baseW));
    return Math.max(1.0, oneToOneCss * (pct / 100));
  }

  function getCurrentZoomPercent() {
    if (state.zoomMode === "fit" || state.zoomScale <= 1.01) return null;
    const { baseW } = getLoupeBaseSize();
    const oneToOneCss = Math.max(1.6, getFullRefWidth() / Math.max(1, baseW));
    return Math.round((state.zoomScale / oneToOneCss) * 100);
  }

  function updateZoomPresetUI() {
    document.querySelectorAll("#zoomPresets .zp-btn").forEach((b) => {
      const z = b.dataset.zoom;
      const isAct = (z === "fit" && state.zoomMode === "fit") ||
                    (z !== "fit" && Number(z) === state.zoomMode);
      b.classList.toggle("active", isAct);
    });
    const pct = getCurrentZoomPercent();
    $("zoomPctLabel").textContent = pct === null ? "Fit" : `${pct}%`;
  }

  function setZoomPreset(preset, centerOnFocusIfFromFit = true) {
    const img = state.items[state.activeIndex];
    if (preset === "fit") {
      state.zoomMode = "fit";
      state.zoomScale = 1.0;
      state.focusX = 0.5;
      state.focusY = 0.5;
    } else {
      const wasFit = state.zoomMode === "fit" || state.zoomScale <= 1.01;
      state.zoomMode = Number(preset);
      state.zoomScale = computeCssScaleForPercent(state.zoomMode);
      if (wasFit && centerOnFocusIfFromFit && img) {
        const fc = getCameraFocusCenter(img);
        state.focusX = fc.x;
        state.focusY = fc.y;
      }
    }
    applyLoupeZoomTransform();
    updateZoomPresetUI();
    if (preset !== "fit" && img) {
      upgradeLoupeToFullRes(img);
      preloadFullNeighbors(state.activeIndex);
    }
  }

  function stepZoomByFactor(factor, cursorClientX = null, cursorClientY = null) {
    const img = state.items[state.activeIndex];
    const wasFit = state.zoomScale <= 1.01;
    const oldScale = state.zoomScale;
    const newScale = Math.max(1.0, Math.min(10.0, oldScale * factor));

    if (newScale <= 1.02) {
      setZoomPreset("fit");
      return;
    }

    const { baseW, baseH } = getLoupeBaseSize();
    if (cursorClientX !== null && cursorClientY !== null) {
      // Zoom smoothly toward the mouse cursor position
      const areaRect = $("loupeCanvasArea").getBoundingClientRect();
      const dxScreen = cursorClientX - (areaRect.left + areaRect.width / 2);
      const dyScreen = cursorClientY - (areaRect.top + areaRect.height / 2);
      const imgXAtCursor = state.focusX + dxScreen / (baseW * oldScale);
      const imgYAtCursor = state.focusY + dyScreen / (baseH * oldScale);
      state.focusX = Math.max(0, Math.min(1, imgXAtCursor - dxScreen / (baseW * newScale)));
      state.focusY = Math.max(0, Math.min(1, imgYAtCursor - dyScreen / (baseH * newScale)));
    } else if (wasFit && img) {
      const fc = getCameraFocusCenter(img);
      state.focusX = fc.x;
      state.focusY = fc.y;
    }

    state.zoomScale = newScale;
    state.zoomMode = "custom";
    applyLoupeZoomTransform();
    updateZoomPresetUI();
    if (img) {
      upgradeLoupeToFullRes(img);
      preloadFullNeighbors(state.activeIndex);
    }
  }

  function jumpZoomToFocus() {
    const img = state.items[state.activeIndex];
    if (!img) return;
    const fc = getCameraFocusCenter(img);
    state.focusX = fc.x;
    state.focusY = fc.y;
    if (state.zoomMode === "fit" || state.zoomScale <= 1.01) {
      state.zoomMode = 100;
      state.zoomScale = computeCssScaleForPercent(100);
    }
    applyLoupeZoomTransform();
    updateZoomPresetUI();
    upgradeLoupeToFullRes(img);
    preloadFullNeighbors(state.activeIndex);
    toast(fc.hasAf ? "Centered on camera AF focus point" : "No AF point in file — centered on middle", 1500);
  }

  function applyLoupeZoomTransform() {
    const wrap = $("loupeImgWrap");
    const zoomed = state.zoomScale > 1.01;
    wrap.classList.toggle("zoomed", zoomed);
    if (!zoomed) {
      wrap.style.transform = "translate(0px, 0px) scale(1)";
      return;
    }
    const { baseW, baseH } = getLoupeBaseSize();
    state.focusX = Math.max(0, Math.min(1, state.focusX));
    state.focusY = Math.max(0, Math.min(1, state.focusY));
    const tx = (0.5 - state.focusX) * baseW * state.zoomScale;
    const ty = (0.5 - state.focusY) * baseH * state.zoomScale;
    wrap.style.transform = `translate(${tx.toFixed(1)}px, ${ty.toFixed(1)}px) scale(${state.zoomScale.toFixed(4)})`;
  }

  function computeGentleHighlightMask(el, cacheKey) {
    if (!el || !el.naturalWidth || !el.naturalHeight) return null;
    if (state.highlightMaskKey === cacheKey && state.highlightMaskCanvas) {
      return state.highlightMaskCanvas;
    }
    try {
      const maxDim = 900;
      const scale = Math.min(1, maxDim / Math.max(el.naturalWidth, el.naturalHeight));
      const mw = Math.max(1, Math.round(el.naturalWidth * scale));
      const mh = Math.max(1, Math.round(el.naturalHeight * scale));
      const off = document.createElement("canvas");
      off.width = mw;
      off.height = mh;
      const octx = off.getContext("2d", { willReadFrequently: true });
      octx.drawImage(el, 0, 0, mw, mh);
      const imgData = octx.getImageData(0, 0, mw, mh);
      const d = imgData.data;
      const out = octx.createImageData(mw, mh);
      const od = out.data;

      let clipped = 0;
      const total = mw * mh;
      // Gentle diagonal zebra on blown/burned highlights (R,G,B >= 252)
      for (let y = 0; y < mh; y++) {
        const rowOff = y * mw;
        for (let x = 0; x < mw; x++) {
          const i = (rowOff + x) * 4;
          const r = d[i], g = d[i + 1], b = d[i + 2];
          if (r >= 252 && g >= 252 && b >= 252) {
            clipped++;
            // Soft diagonal zebra stripe (every 6px, 2px stripe) + ultra-subtle tint
            const stripe = ((x + y) % 6) < 2;
            od[i] = 255;
            od[i + 1] = stripe ? 65 : 95;
            od[i + 2] = stripe ? 65 : 95;
            od[i + 3] = stripe ? 92 : 28; // very gentle alpha so image stays natural
          }
        }
      }
      octx.putImageData(out, 0, 0);
      state.highlightMaskCanvas = off;
      state.highlightMaskKey = cacheKey;
      state.highlightClipPct = (clipped / Math.max(1, total)) * 100;

      const btn = $("btnToggleHighlights");
      if (btn) {
        const pctStr = state.highlightClipPct >= 0.05 ? ` ${state.highlightClipPct.toFixed(1)}%` : "";
        btn.textContent = `☀ Highlights${pctStr} (H)`;
      }
      return off;
    } catch (_) {
      return null;
    }
  }

  function drawLoupeOverlay(img) {
    const canvas = $("loupeOverlay");
    const el = $("loupeImg");
    const w = el.offsetWidth || el.clientWidth;
    const h = el.offsetHeight || el.clientHeight;
    if (!w || !h) return;
    // Use high-DPI internal resolution so AF boxes stay razor-sharp even at 300% zoom!
    const dpr = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // 0. Gentle Burned Highlights (Zebra) Overlay — cached strictly per image_id so zoom upgrades don't re-scan!
    if (state.showHighlights && el.complete && el.naturalWidth > 0) {
      const mask = computeGentleHighlightMask(el, img.image_id);
      if (mask) {
        ctx.drawImage(mask, 0, 0, w, h);
      }
    }

    // 1. Camera AF Focus Area (Canon-style red brackets)
    if (state.showAfPoints && Array.isArray(img.af_points) && img.af_points.length > 0) {
      for (const box of img.af_points) {
        const bx = box.x * w;
        const by = box.y * h;
        const bw = box.w * w;
        const bh = box.h * h;
        if (box.in_focus) {
          ctx.strokeStyle = "#ef4444";
          ctx.lineWidth = 1.4;
          ctx.strokeRect(bx, by, bw, bh);
          // Crisp corner brackets like Canon EVF / DPP (100% transparent inside)
          const corner = Math.max(4, Math.min(bw, bh) * 0.22);
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          // Top-left
          ctx.moveTo(bx, by + corner); ctx.lineTo(bx, by); ctx.lineTo(bx + corner, by);
          // Top-right
          ctx.moveTo(bx + bw - corner, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + corner);
          // Bottom-right
          ctx.moveTo(bx + bw, by + bh - corner); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw - corner, by + bh);
          // Bottom-left
          ctx.moveTo(bx + corner, by + bh); ctx.lineTo(bx, by + bh); ctx.lineTo(bx, by + bh - corner);
          ctx.stroke();
        } else {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
          ctx.lineWidth = 1.1;
          ctx.setLineDash([4, 3]);
          ctx.strokeRect(bx, by, bw, bh);
          ctx.setLineDash([]);
        }
      }
    }

    // 2. Optional YOLO Subject & Head Boxes
    if (state.showBbox && img.preview_width && img.preview_height) {
      const sx = w / img.preview_width;
      const sy = h / img.preview_height;
      if (img.bbox && img.bbox.length === 4) {
        const [x1, y1, x2, y2] = img.bbox;
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 1.8;
        ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      }
      if (img.head_bbox && img.head_bbox.length === 4) {
        const [x1, y1, x2, y2] = img.head_bbox;
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 1.8;
        ctx.setLineDash([5, 3]);
        ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
        ctx.setLineDash([]);
      }
    }
  }

  function renderLoupeControls(img) {
    const starsEl = $("loupeStars");
    starsEl.innerHTML = "";
    for (let i = 1; i <= 5; i++) {
      const b = document.createElement("button");
      b.className = "star-btn" + (i <= (img.effective_rating || 0) ? " on" : "");
      b.textContent = "★";
      b.title = `Set ${i} star${i > 1 ? "s" : ""} (Key ${i})`;
      b.onclick = () => setRating(img, img.effective_rating === i ? 0 : i);
      starsEl.appendChild(b);
    }

    const colEl = $("loupeColors");
    colEl.innerHTML = "";
    for (const c of COLORS) {
      const b = document.createElement("button");
      b.className = `cp-dot ${c.css}` + ((img.effective_label || "") === c.key ? " active" : "");
      b.title = `${c.name} (${c.shortcut})`;
      if (!c.key) b.textContent = "∅";
      b.onclick = () => setColorLabel(img, c.key);
      colEl.appendChild(b);
    }

    $("btnFlagPick").classList.toggle("active", img.user_flag === 1);
    $("btnFlagReject").classList.toggle("active", img.user_flag === -1);

    const xmpEl = $("loupeXmpStatus");
    if (img.xmp_exists) {
      xmpEl.textContent = `XMP ✓ (${img.rating_source})`;
      xmpEl.style.color = "#34d399";
    } else if (img.camera_rating > 0) {
      xmpEl.textContent = `Camera ${img.camera_rating}★`;
      xmpEl.style.color = "#fbbf24";
    } else {
      xmpEl.textContent = "No XMP yet";
      xmpEl.style.color = "#94a3b8";
    }
  }

  function renderInspector(img) {
    const bars = [
      { name: "Overall AI Score", val: img.overall_score || 0, color: "#10b981" },
      { name: "Head/Eye Sharpness", val: img.sharpness_score || 0, color: "#3b82f6" },
      { name: "Eye Contact", val: img.eye_contact_score || 0, color: "#a855f7" },
      { name: "Action / Pose", val: img.action_score || 0, color: "#f59e0b" },
    ];
    if (img.scenic_score > 0 || img.track === "Landscape") {
      bars.push({ name: "Scenic / Light", val: img.scenic_score || 0, color: "#ec4899" });
    }
    if (img.personal_score !== null && img.personal_score !== undefined) {
      bars.push({ name: "Personal Ranker", val: img.personal_score, color: "#38bdf8" });
    }
    $("inspScores").innerHTML = bars.map((b) => `
      <div class="sbar-row">
        <div class="sbar-head"><span>${b.name}</span><strong>${b.val ? b.val.toFixed(1) : "—"}</strong></div>
        <div class="sbar-track"><div class="sbar-fill" style="width:${Math.min(100, b.val)}%;background:${b.color}"></div></div>
      </div>
    `).join("");

    const speciesDisp = img.species
      ? `${img.species} (${Math.round((img.species_conf || 0) * 100)}%)`
      : (img.detected_class ? `${img.detected_class} (${Math.round((img.detection_conf || 0) * 100)}%)` : "—");
    const scenMap = { action_sequence: "🎬 Action Sequence", static_group: "📸 Static Group", single: "1️⃣ Single Shot" };
    const trackDisp = (img.track || "—") + (img.hybrid_bonus ? " · ✨ +15% Light" : "");

    $("inspAiMeta").innerHTML = `
      <span class="meta-k">AI Layer</span><span class="meta-v">${escapeHtml(LAYER_LABELS[img.ai_layer] || img.ai_layer)}</span>
      <span class="meta-k">Track</span><span class="meta-v">${escapeHtml(trackDisp)}</span>
      <span class="meta-k">Species</span><span class="meta-v">${escapeHtml(speciesDisp)}</span>
      <span class="meta-k">Scenario</span><span class="meta-v">${escapeHtml(scenMap[img.scenario_type] || img.scenario_type || "—")}</span>
      <span class="meta-k">Overall Rank</span><span class="meta-v">${img.overall_rank < 999999 ? "#" + img.overall_rank : "—"}</span>
      <span class="meta-k">Burst</span><span class="meta-v">${img.burst_id ? `${img.burst_id} (Rank #${img.burst_rank} of ${img.burst_size})` : "Standalone"}</span>
    `;

    const spd = getSpeedInfo(img.exposure_seconds, img.exposure_time);
    const sb = $("inspSpeedBanner");
    sb.className = `speed-banner ${spd.cls}`;
    sb.innerHTML = `<span>${spd.icon} ${escapeHtml(spd.text)}</span><span>${escapeHtml(spd.label)}</span>`;

    const loupeEl = $("loupeImg");
    let dispW = img.preview_width || 0;
    let dispH = img.preview_height || 0;
    if (state.loupeLoadedFullId === img.image_id && loupeEl && loupeEl.naturalWidth) {
      dispW = loupeEl.naturalWidth;
      dispH = loupeEl.naturalHeight;
    } else if (dispW >= 3000 && (img.file_ext === ".cr3" || (img.filename || "").toLowerCase().endsWith(".cr3"))) {
      // Pipeline decoded 1/2 DCT preview of 45MP CR3; actual RAW/embedded full-res is 2x
      dispW *= 2;
      dispH *= 2;
    }
    const mpStr = (dispW && dispH) ? ` (${((dispW * dispH) / 1e6).toFixed(1)} MP)` : "";

    $("inspExif").innerHTML = `
      <span class="meta-k">Date / Time</span><span class="meta-v">${escapeHtml(img.timestamp_str || "—")}</span>
      <span class="meta-k">Camera</span><span class="meta-v">${escapeHtml(img.camera_model || "—")}</span>
      <span class="meta-k">Lens</span><span class="meta-v">${escapeHtml(img.lens_model || "Run Refresh EXIF")}</span>
      <span class="meta-k">Zoom / Focal</span><span class="meta-v">${img.focal_length ? img.focal_length + " mm" : "—"}</span>
      <span class="meta-k">Aperture</span><span class="meta-v">${img.f_number ? "f/" + img.f_number : "—"}</span>
      <span class="meta-k">ISO</span><span class="meta-v">${img.iso || "—"}</span>
      <span class="meta-k">Resolution</span><span class="meta-v">${dispW ? `${dispW} × ${dispH}${mpStr}` : "—"}</span>
      <span class="meta-k">File Size</span><span class="meta-v">${formatBytes(img.file_size)}</span>
    `;

    $("inspXmpMeta").innerHTML = `
      <span class="meta-k">Current Rating</span><span class="meta-v">${img.effective_rating || 0}★ (${escapeHtml(img.rating_source)})</span>
      <span class="meta-k">Color Label</span><span class="meta-v">${escapeHtml(img.effective_label || "None")}</span>
      <span class="meta-k">In-Camera</span><span class="meta-v">${img.camera_rating ? img.camera_rating + "★" : "None"} ${escapeHtml(img.camera_label || "")}</span>
      <span class="meta-k">XMP Sidecar</span><span class="meta-v">${img.xmp_exists ? "Present (Lightroom Synced)" : "Created on first edit"}</span>
    `;
  }

  function createFilmstripItemEl(item, currentId) {
    const div = document.createElement("div");
    div.className = "fs-item" + (item.image_id === currentId ? " active" : "");
    div.dataset.id = item.image_id;
    div.title = `${item.filename} · Sharp: ${Math.round(item.sharpness_score || 0)}`;
    div.innerHTML = `
      <img loading="lazy" decoding="async" src="/api/media/thumb/${encodeURIComponent(item.image_id)}" alt="" />
      <div class="fs-sharp-bar" style="width:${Math.min(100, item.sharpness_score || 0)}%"></div>
      ${item.effective_rating ? `<span class="fs-star-mini">${"★".repeat(item.effective_rating)}</span>` : ""}
    `;
    div.onclick = () => {
      const idx = state.items.findIndex((x) => x.image_id === item.image_id);
      if (idx >= 0) {
        state.activeIndex = idx;
        renderLoupe();
      } else {
        state.items.splice(state.activeIndex + 1, 0, item);
        state.activeIndex += 1;
        renderLoupe();
      }
    };
    return div;
  }

  function maybeExtendFilmstripRight() {
    if (state.filmstripMode !== "list") return;
    const strip = $("filmstripScroll");
    if (!strip) return;
    const current = state.items[state.activeIndex];
    const currentId = current ? current.image_id : "";
    const nearRight = strip.scrollLeft + strip.clientWidth >= strip.scrollWidth - 1000;
    if (!nearRight) return;

    // 1. Append more already-loaded items from state.items into the filmstrip DOM
    if (state.filmstripWindowEnd < state.items.length) {
      const nextEnd = Math.min(state.items.length, state.filmstripWindowEnd + 150);
      const frag = document.createDocumentFragment();
      for (let i = state.filmstripWindowEnd; i < nextEnd; i++) {
        frag.appendChild(createFilmstripItemEl(state.items[i], currentId));
      }
      strip.appendChild(frag);
      state.filmstripWindowEnd = nextEnd;
    }

    // 2. If approaching the end of loaded items, fetch the next batch from backend early!
    if (state.filmstripWindowEnd >= state.items.length - 150 && state.items.length < state.total && !state.loading) {
      loadImages(false);
    }
  }

  function renderFilmstrip() {
    const strip = $("filmstripScroll");
    strip.innerHTML = "";
    const current = state.items[state.activeIndex];
    if (!current) return;

    const frag = document.createDocumentFragment();
    if (state.filmstripMode === "burst" && state.burstSiblings.length > 0) {
      for (const item of state.burstSiblings) {
        frag.appendChild(createFilmstripItemEl(item, current.image_id));
      }
    } else {
      const start = Math.max(0, state.activeIndex - 40);
      const end = Math.min(state.items.length, Math.max(state.activeIndex + 140, state.filmstripWindowEnd));
      state.filmstripWindowStart = start;
      state.filmstripWindowEnd = end;
      for (let i = start; i < end; i++) {
        frag.appendChild(createFilmstripItemEl(state.items[i], current.image_id));
      }
    }
    strip.appendChild(frag);
    const activeEl = strip.querySelector(".fs-item.active");
    if (activeEl) activeEl.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
  }

  // ------------------------------------------------------------------
  // Compare View (Selected images OR Top 4 of current burst)
  // ------------------------------------------------------------------
  async function openCompareView() {
    let items = [];
    if (state.selectedIds.size >= 2) {
      items = state.items.filter((x) => state.selectedIds.has(x.image_id)).slice(0, 6);
      $("compareTitle").textContent = `Comparing ${items.length} Selected Images`;
    } else if (state.items[state.activeIndex]) {
      const cur = state.items[state.activeIndex];
      if (cur.burst_id) {
        const res = await api(`/api/images/${encodeURIComponent(cur.image_id)}`);
        items = (res.burst_siblings || [cur])
          .sort((a, b) => (b.sharpness_score || 0) - (a.sharpness_score || 0))
          .slice(0, 4);
        $("compareTitle").textContent = `Burst ${cur.burst_id} — Top ${items.length} Sharpest Frames`;
      } else {
        items = state.items.slice(state.activeIndex, state.activeIndex + 2);
        $("compareTitle").textContent = `Comparing ${items.length} Adjacent Images`;
      }
    }
    state.compareItems = items;
    renderCompareGrid();
  }

  function renderCompareGrid() {
    const grid = $("compareGrid");
    grid.innerHTML = "";
    for (const img of state.compareItems) {
      const spd = getSpeedInfo(img.exposure_seconds, img.exposure_time);
      const url = state.compareShowCrop
        ? `/api/media/crop/${encodeURIComponent(img.image_id)}`
        : `/api/media/hq/${encodeURIComponent(img.image_id)}`;
      const card = document.createElement("div");
      card.className = "compare-card";
      card.innerHTML = `
        <div class="compare-img-wrap"><img src="${url}" alt="${escapeHtml(img.filename)}" /></div>
        <div class="compare-footer">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <strong>${escapeHtml(img.filename)}</strong>
            <span class="spd-badge ${spd.cls}">${spd.icon} ${escapeHtml(spd.text)}</span>
          </div>
          <div class="muted">
            Score: <strong>${Math.round(img.overall_score || 0)}</strong> ·
            Sharp: <strong>${Math.round(img.sharpness_score || 0)}</strong> ·
            Eye: <strong>${Math.round(img.eye_contact_score || 0)}</strong>
          </div>
          <div style="display:flex;gap:6px;align-items:center">
            ${[1, 2, 3, 4, 5].map((r) => `<button class="btn btn-xs ${img.effective_rating === r ? "btn-primary" : ""}" data-cr="${r}">${r}★</button>`).join("")}
            <button class="btn btn-xs" data-cl="Green">🟢</button>
            <button class="btn btn-xs" data-cl="Yellow">🟡</button>
            <button class="btn btn-xs" data-cl="Red">🔴</button>
          </div>
        </div>
      `;
      card.querySelectorAll("[data-cr]").forEach((b) => {
        b.onclick = async () => {
          await setRating(img, Number(b.dataset.cr), false);
          renderCompareGrid();
        };
      });
      card.querySelectorAll("[data-cl]").forEach((b) => {
        b.onclick = async () => {
          await setColorLabel(img, b.dataset.cl, false);
          renderCompareGrid();
        };
      });
      grid.appendChild(card);
    }
  }

  // ------------------------------------------------------------------
  // Real-Time Rating, Color Label & Undo
  // ------------------------------------------------------------------
  async function setRating(img, rating, allowAdvance = true) {
    if (!img) return;
    img.effective_rating = rating;
    refreshCardInDom(img);
    if (state.view === "loupe") renderLoupeControls(img);

    try {
      const res = await api(`/api/images/${encodeURIComponent(img.image_id)}/metadata`, {
        method: "POST",
        body: JSON.stringify({ rating }),
      });
      Object.assign(img, res.image);
      refreshCardInDom(img);
      refreshSidebarCounts();
      if (state.view === "loupe" && state.items[state.activeIndex]?.image_id === img.image_id) {
        renderLoupeControls(img);
        renderInspector(img);
      }
      if (res.xmp_warning) toast(`Saved in DB (XMP warning: ${res.xmp_warning})`, 3500);
      if (allowAdvance && state.view === "loupe" && $("chkAutoAdvance").checked) {
        stepLoupe(1);
      }
    } catch (err) {
      toast("Rating error: " + err.message);
    }
  }

  async function setColorLabel(img, label, allowAdvance = true) {
    if (!img) return;
    img.effective_label = label;
    refreshCardInDom(img);
    if (state.view === "loupe") renderLoupeControls(img);

    try {
      const res = await api(`/api/images/${encodeURIComponent(img.image_id)}/metadata`, {
        method: "POST",
        body: JSON.stringify({ label }),
      });
      Object.assign(img, res.image);
      refreshCardInDom(img);
      refreshSidebarCounts();
      if (state.view === "loupe" && state.items[state.activeIndex]?.image_id === img.image_id) {
        renderLoupeControls(img);
        renderInspector(img);
      }
      if (allowAdvance && state.view === "loupe" && $("chkAutoAdvance").checked) {
        stepLoupe(1);
      }
    } catch (err) {
      toast("Label error: " + err.message);
    }
  }

  async function setUserFlag(img, flag) {
    if (!img) return;
    img.user_flag = flag;
    try {
      const res = await api(`/api/images/${encodeURIComponent(img.image_id)}/metadata`, {
        method: "POST",
        body: JSON.stringify({ user_flag: flag }),
      });
      Object.assign(img, res.image);
      if (state.view === "loupe") renderLoupeControls(img);
      if (state.view === "loupe" && $("chkAutoAdvance").checked && flag !== 0) {
        stepLoupe(1);
      }
    } catch (err) {
      toast("Flag error: " + err.message);
    }
  }

  async function batchUpdate(payload) {
    const ids = Array.from(state.selectedIds);
    if (!ids.length) return;
    try {
      const res = await api("/api/images/batch-metadata", {
        method: "POST",
        body: JSON.stringify({ image_ids: ids, ...payload }),
      });
      toast(`Updated ${res.updated} images (XMP synced: ${res.xmp_synced})`);
      await loadImages(true);
    } catch (err) {
      toast("Batch update failed: " + err.message);
    }
  }

  async function undoLast() {
    try {
      const res = await api(`/api/changes/undo-last?dir_id=${encodeURIComponent(state.currentDirId)}`, {
        method: "POST",
      });
      toast(res.message || "Undone");
      if (res.ok) {
        await loadImages(true);
        if (state.view === "loupe") renderLoupe();
      }
    } catch (err) {
      toast("Undo failed: " + err.message);
    }
  }

  function stepLoupe(delta) {
    const next = state.activeIndex + delta;
    if (next >= 0 && next < state.items.length) {
      state.activeIndex = next;
      renderLoupe();
    }
  }

  // ------------------------------------------------------------------
  // Pipeline, Previews & Scanning Triggers
  // ------------------------------------------------------------------
  async function startCulling(mode) {
    if (!state.currentDirId) return;
    const sub = state.filters.subfolder === "__all__" ? "" : state.filters.subfolder;
    const selIds = Array.from(state.selectedIds);
    if (mode === "selected" && selIds.length === 0) {
      toast("Select at least one image first (use checkboxes or Ctrl+Click).");
      return;
    }
    try {
      await api("/api/pipeline/start", {
        method: "POST",
        body: JSON.stringify({
          dir_id: state.currentDirId,
          mode,
          subfolder: sub,
          selected_ids: selIds,
        }),
      });
      toast(`Started AI Culling (${mode})!`);
    } catch (err) {
      toast(err.message, 4000);
    }
  }

  async function startHqPreviewGen() {
    if (!state.currentDirId) return;
    const sub = state.filters.subfolder === "__all__" ? "" : state.filters.subfolder;
    try {
      await api("/api/previews/generate", {
        method: "POST",
        body: JSON.stringify({ dir_id: state.currentDirId, subfolder: sub }),
      });
      toast("Converting folder to fast HQ JPGs in background...");
    } catch (err) {
      toast(err.message, 4000);
    }
  }

  async function startMetadataScan() {
    if (!state.currentDirId) return;
    const sub = state.filters.subfolder === "__all__" ? "" : state.filters.subfolder;
    try {
      await api(`/api/directories/${state.currentDirId}/scan?subfolder=${encodeURIComponent(sub)}`, {
        method: "POST",
      });
      toast("Scanning EXIF, Lens, Camera Ratings & Lightroom XMPs...");
    } catch (err) {
      toast(err.message, 4000);
    }
  }

  // ------------------------------------------------------------------
  // Modals: Open Dir, Settings, Logs/Restore, Learning
  // ------------------------------------------------------------------
  async function openSettingsModal() {
    state.settings = await api("/api/settings");
    const s = state.settings;
    $("setRealtimeXmp").checked = !!s.realtime_xmp_write;
    $("setAutoAdvance").checked = !!s.auto_advance_on_rate;
    $("setAiAutoApply").checked = !!s.ai_xmp_auto_apply;
    $("setAiOnlyUnrated").checked = !!s.ai_xmp_only_if_unrated;
    const m = s.ai_xmp_mapping || {};
    const l2 = m.layer2_to_edit || { enabled: true, rating: null, label: "Green" };
    const l1 = m.layer1_keep || { enabled: true, rating: null, label: "Yellow" };
    const rj = m.rejected || { enabled: false, rating: null, label: "" };
    const dl = m.to_delete || { enabled: true, rating: null, label: "Red" };
    $("mapL2Enabled").checked = !!l2.enabled;
    $("mapL2Rating").value = l2.rating != null ? String(l2.rating) : "0";
    $("mapL2Label").value = l2.label ?? "Green";
    $("mapL1Enabled").checked = !!l1.enabled;
    $("mapL1Rating").value = l1.rating != null ? String(l1.rating) : "0";
    $("mapL1Label").value = l1.label ?? "Yellow";
    $("mapRejEnabled").checked = !!rj.enabled;
    $("mapRejRating").value = rj.rating != null ? String(rj.rating) : "";
    $("mapRejLabel").value = rj.label ?? "";
    if ($("mapDelEnabled")) {
      $("mapDelEnabled").checked = !!dl.enabled;
      $("mapDelRating").value = dl.rating != null ? String(dl.rating) : "";
      $("mapDelLabel").value = dl.label ?? "Red";
    }
    $("setHqDim").value = s.hq_preview_max_dim || 1600;
    $("setHqQuality").value = s.hq_preview_quality || 86;
    if ($("setZoomDim")) $("setZoomDim").value = String(s.zoom_preview_max_dim || 8192);
    if ($("setZoomQuality")) $("setZoomQuality").value = s.zoom_preview_quality || 94;
    $("aiXmpStatusMsg").textContent = "";
    $("modalSettings").classList.remove("hidden");
  }

  function gatherSettingsForm() {
    return {
      realtime_xmp_write: $("setRealtimeXmp").checked,
      auto_advance_on_rate: $("setAutoAdvance").checked,
      ai_xmp_auto_apply: $("setAiAutoApply").checked,
      ai_xmp_only_if_unrated: $("setAiOnlyUnrated").checked,
      ai_xmp_mapping: {
        layer2_to_edit: {
          enabled: $("mapL2Enabled").checked,
          rating: Number($("mapL2Rating").value),
          label: $("mapL2Label").value,
        },
        layer1_keep: {
          enabled: $("mapL1Enabled").checked,
          rating: Number($("mapL1Rating").value),
          label: $("mapL1Label").value,
        },
        rejected: {
          enabled: $("mapRejEnabled").checked,
          rating: $("mapRejRating").value === "" ? null : Number($("mapRejRating").value),
          label: $("mapRejLabel").value,
        },
        to_delete: {
          enabled: $("mapDelEnabled") ? $("mapDelEnabled").checked : true,
          rating: ($("mapDelRating") && $("mapDelRating").value !== "") ? Number($("mapDelRating").value) : null,
          label: $("mapDelLabel") ? $("mapDelLabel").value : "Red",
        },
      },
      hq_preview_max_dim: Number($("setHqDim").value) || 1600,
      hq_preview_quality: Number($("setHqQuality").value) || 86,
      zoom_preview_max_dim: Number($("setZoomDim")?.value) || 3072,
      zoom_preview_quality: Number($("setZoomQuality")?.value) || 88,
    };
  }

  async function openLogsModal(tab = "events") {
    $("modalLogs").classList.remove("hidden");
    switchLogTab(tab);
  }

  async function switchLogTab(tab) {
    document.querySelectorAll("#modalLogs .m-tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.logTab === tab);
    });
    $("tabEvents").classList.toggle("hidden", tab !== "events");
    $("tabChanges").classList.toggle("hidden", tab !== "changes");
    $("tabRuns").classList.toggle("hidden", tab !== "runs");

    if (tab === "events") await loadEventLogs();
    if (tab === "changes") await loadChangeLogs();
    if (tab === "runs") await loadPipelineRuns();
  }

  async function loadEventLogs() {
    const lvl = $("logLevelFilter").value;
    const cat = $("logCatFilter").value;
    const res = await api(`/api/logs?limit=200&level=${encodeURIComponent(lvl)}&category=${encodeURIComponent(cat)}`);
    const list = $("eventLogList");
    list.innerHTML = (res.logs || []).map((l) => `
      <div class="log-line">
        <span class="log-ts">${escapeHtml((l.timestamp || "").replace("T", " ").slice(0, 19))}</span>
        <span class="log-lvl lvl-${escapeHtml(l.level)}">[${escapeHtml(l.level)}]</span>
        <span class="log-cat">${escapeHtml(l.category)}</span>
        <span>${escapeHtml(l.message)}</span>
      </div>
    `).join("") || '<div class="muted">No logs yet.</div>';
  }

  async function loadChangeLogs() {
    const res = await api(`/api/changes?limit=150&dir_id=${encodeURIComponent(state.currentDirId)}`);
    const tbody = $("changeLogBody");
    tbody.innerHTML = "";
    for (const c of res.changes || []) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml((c.timestamp || "").replace("T", " ").slice(0, 19))}</td>
        <td>${escapeHtml(c.filename)}</td>
        <td>${escapeHtml(c.action_type)}</td>
        <td><strong>${escapeHtml(c.field)}</strong>: ${escapeHtml(c.old_value || "∅")} → ${escapeHtml(c.new_value || "∅")}</td>
        <td>${c.xmp_synced ? "✓ XMP" : "DB"}</td>
        <td>${c.reverted ? '<span class="muted">Restored</span>' : `<button class="btn btn-xs" data-restore="${c.id}">↩ Restore</button>`}</td>
      `;
      const btn = tr.querySelector("[data-restore]");
      if (btn) {
        btn.onclick = async () => {
          await api(`/api/changes/${c.id}/restore`, { method: "POST" });
          toast(`Restored ${c.filename}`);
          loadChangeLogs();
          loadImages(true);
        };
      }
      tbody.appendChild(tr);
    }
  }

  async function loadPipelineRuns() {
    const res = await api("/api/pipeline/runs");
    $("runsLogBody").innerHTML = (res.runs || []).map((r) => `
      <tr>
        <td>${escapeHtml((r.started_at || "").replace("T", " ").slice(0, 19))}</td>
        <td>${escapeHtml(r.mode)}</td>
        <td>${escapeHtml(r.status)}</td>
        <td>${r.processed_images} / ${r.total_images}</td>
        <td>${r.skipped_errors}</td>
        <td class="muted">${escapeHtml(JSON.stringify(r.summary || {}))}</td>
      </tr>
    `).join("") || '<tr><td colspan="6" class="muted">No runs recorded from Web UI yet (previous CLI session was auto-imported).</td></tr>';
  }

  async function openLearningModal() {
    $("modalLearning").classList.remove("hidden");
    const body = $("learningBody");
    body.innerHTML = '<p class="muted">Analyzing your ratings vs AI predictions...</p>';
    try {
      const stats = await api(`/api/learning/${state.currentDirId}`);
      const byR = stats.by_rating || [];
      const mat = stats.matrix || [];
      const ovs = stats.notable_overrides || [];
      const tr = stats.training_ready || { total_picks: 0, keep_edit: 0, reject: 0, can_train: false, model_exists: false };

      body.innerHTML = `
        <div class="settings-group">
          <h4>🤖 Personalized LightGBM Ranker (${tr.model_exists ? "Model Active ✓" : "Not Trained Yet"})</h4>
          <p class="muted">
            Export your ratings &amp; flags from this folder (<strong>${tr.keep_edit}</strong> Keep/Edit, <strong>${tr.reject}</strong> Reject)
            to train <code>models/personal_ranker.pkl</code> on SigLIP 2 + BioCLIP 2 embeddings and automatically re-rank!
          </p>
          <div class="sync-action-row">
            <button id="btnTrainPersonalRanker" class="btn btn-primary btn-sm" ${tr.can_train ? "" : "disabled"}>
              🚀 Train Personal Ranker &amp; Reselect
            </button>
            <span id="trainStatusMsg" class="muted">${tr.can_train ? "Ready to train!" : "Rate at least 3 Keep/Edit (2–5★/Green/Yellow) and 3 Reject (1★/Red/X) images first."}</span>
          </div>
        </div>
        <div class="settings-group">
          <h4>Average AI Scores by Your Star Rating</h4>
          ${byR.length ? `
            <table class="data-table">
              <thead><tr><th>Your Rating</th><th>Count</th><th>Avg Overall</th><th>Avg Sharpness</th><th>Avg Eye</th><th>Avg Action</th></tr></thead>
              <tbody>
                ${byR.map((r) => `<tr><td>${"★".repeat(r.rating)}</td><td>${r.count}</td><td>${r.avg_overall}</td><td>${r.avg_sharpness}</td><td>${r.avg_eye}</td><td>${r.avg_action}</td></tr>`).join("")}
              </tbody>
            </table>
          ` : '<p class="muted">Rate some images (1–5★) to see how AI scores correlate with your taste!</p>'}
        </div>
        <div class="settings-group">
          <h4>AI Layer vs Your Ratings Matrix</h4>
          ${mat.length ? `
            <table class="data-table">
              <thead><tr><th>AI Layer</th><th>4–5★ (Keepers)</th><th>2–3★ (Mid)</th><th>1★ / Rejected</th><th>Total Rated</th></tr></thead>
              <tbody>
                ${mat.map((m) => `<tr><td>${escapeHtml(LAYER_LABELS[m.ai_layer] || m.ai_layer)}</td><td>${m.high_4_5}</td><td>${m.mid_2_3}</td><td>${m.low_1_rej}</td><td>${m.total}</td></tr>`).join("")}
              </tbody>
            </table>
          ` : '<p class="muted">No rated images yet.</p>'}
        </div>
        <div class="settings-group">
          <h4>Notable Overrides (Where You Disagreed With AI)</h4>
          ${ovs.length ? `
            <table class="data-table">
              <thead><tr><th>File</th><th>Folder</th><th>AI Layer</th><th>AI Score</th><th>Your Rating</th></tr></thead>
              <tbody>
                ${ovs.map((o) => `<tr><td>${escapeHtml(o.filename)}</td><td>${escapeHtml(o.subfolder)}</td><td>${escapeHtml(o.ai_layer)}</td><td>${Math.round(o.overall_score || 0)}</td><td>${o.effective_rating}★</td></tr>`).join("")}
              </tbody>
            </table>
          ` : '<p class="muted">No major disagreements recorded yet.</p>'}
        </div>
      `;
      const trainBtn = $("btnTrainPersonalRanker");
      if (trainBtn) {
        trainBtn.onclick = async () => {
          trainBtn.disabled = true;
          $("trainStatusMsg").textContent = "Training LightGBM on BioCLIP 2 + SigLIP 2 embeddings...";
          try {
            const res = await api(`/api/learning/${state.currentDirId}/train`, {
              method: "POST",
              body: JSON.stringify({ auto_reselect: true }),
            });
            $("trainStatusMsg").textContent = `${res.message} — Re-ranking folder now...`;
            toast("Personal LightGBM ranker trained! Re-ranking folder...", 3500);
          } catch (e) {
            $("trainStatusMsg").textContent = `Error: ${e.message}`;
            trainBtn.disabled = false;
          }
        };
      }
    } catch (err) {
      body.innerHTML = `<p class="muted">Error: ${escapeHtml(err.message)}</p>`;
    }
  }

  // ------------------------------------------------------------------
  // Event Bindings & Keyboard Shortcuts
  // ------------------------------------------------------------------
  function bindUI() {
    // Open Dir modal
    $("btnOpenDir").onclick = () => $("modalOpenDir").classList.remove("hidden");
    $("btnNativeBrowse").onclick = async () => {
      try {
        const res = await api("/api/directories/browse", { method: "POST" });
        if (res.path) $("inputDirPath").value = res.path;
      } catch (err) {
        toast("Folder picker unavailable: " + err.message);
      }
    };
    $("btnConfirmOpenDir").onclick = async () => {
      const path = $("inputDirPath").value.trim();
      if (!path) return toast("Please enter or browse a directory path.");
      try {
        const res = await api("/api/directories/open", {
          method: "POST",
          body: JSON.stringify({ path, scan_now: $("chkScanOnOpen").checked }),
        });
        $("modalOpenDir").classList.add("hidden");
        const dirs = await api("/api/directories");
        state.directories = dirs.directories || [];
        renderDirectorySelector();
        await selectDirectory(res.directory.dir_id);
        toast("Directory opened!");
      } catch (err) {
        toast(err.message, 4500);
      }
    };

    // Directory switcher
    $("dirSelect").onchange = (e) => {
      state.filters.subfolder = "__all__";
      selectDirectory(e.target.value);
    };

    // Cull dropdown menu
    $("btnRunCullMenu").onclick = (e) => {
      e.stopPropagation();
      $("cullDropdown").classList.toggle("hidden");
    };
    document.addEventListener("click", () => $("cullDropdown").classList.add("hidden"));
    document.querySelectorAll("[data-cull-mode]").forEach((btn) => {
      btn.onclick = () => startCulling(btn.dataset.cullMode);
    });
    $("btnFastReselect").onclick = async () => {
      if (!state.currentDirId) return;
      try {
        await api("/api/pipeline/reselect", {
          method: "POST",
          body: JSON.stringify({ dir_id: state.currentDirId }),
        });
        toast("Running fast reselect (~2s) from cached scores & embeddings.npz...");
      } catch (err) {
        toast(err.message, 4000);
      }
    };
    $("btnSyncResultsJson").onclick = async () => {
      if (!state.currentDirId) return;
      try {
        const res = await api(`/api/directories/${state.currentDirId}/sync-results`, { method: "POST" });
        toast(`Synced ${res.imported} images from results.json (embeddings.npz: ${res.has_embeddings ? "OK" : "missing"})`);
        await refreshDirectoryAndImages();
      } catch (err) {
        toast(err.message, 4000);
      }
    };
    $("btnGenHqJpgs").onclick = () => startHqPreviewGen();
    $("btnScanMeta").onclick = () => startMetadataScan();
    $("btnCancelTask").onclick = (e) => {
      e.stopPropagation();
      api("/api/pipeline/cancel", { method: "POST" });
    };
    $("progressPill").onclick = () => openLogsModal("events");

    // View switchers
    document.querySelectorAll(".view-switcher .btn-seg").forEach((b) => {
      b.onclick = () => setView(b.dataset.view);
    });

    // Sidebar filters
    document.querySelectorAll("#layerFilters .filter-row").forEach((b) => {
      b.onclick = () => {
        state.filters.ai_layer = b.dataset.layer;
        syncFilterControlsFromState();
        loadImages(true);
      };
    });
    if ($("chkUseAiLayer")) {
      $("chkUseAiLayer").onchange = (e) => {
        state.filters.use_ai_layer = e.target.checked;
        if (state.currentDir) updateLayerBadges(state.currentDir);
        loadImages(true);
      };
    }
    document.querySelectorAll("#ratingOpBar .rop-btn").forEach((b) => {
      b.onclick = () => {
        state.filters.rating_op = b.dataset.op;
        syncFilterControlsFromState();
        if (state.filters.rating_val !== "") loadImages(true);
      };
    });
    document.querySelectorAll("#ratingFilterBar .rf-btn").forEach((b) => {
      b.onclick = () => {
        state.filters.rating_val = b.dataset.rating;
        syncFilterControlsFromState();
        loadImages(true);
      };
    });
    document.querySelectorAll("#colorFilterBar .cf-dot").forEach((b) => {
      b.onclick = () => {
        const lbl = b.dataset.label;
        if (!Array.isArray(state.filters.labels)) state.filters.labels = [];
        if (lbl === "__all__") {
          state.filters.labels = [];
        } else {
          const idx = state.filters.labels.indexOf(lbl);
          if (idx >= 0) state.filters.labels.splice(idx, 1);
          else state.filters.labels.push(lbl);
        }
        syncFilterControlsFromState();
        loadImages(true);
      };
    });
    document.querySelectorAll("#speedFilters .filter-row").forEach((b) => {
      b.onclick = () => {
        state.filters.speed_tier = b.dataset.speed;
        syncFilterControlsFromState();
        loadImages(true);
      };
    });
    $("cameraFilter").onchange = (e) => { state.filters.camera = e.target.value; loadImages(true); };
    $("lensFilter").onchange = (e) => { state.filters.lens = e.target.value; loadImages(true); };
    document.querySelectorAll("#trackFilters .rf-btn").forEach((b) => {
      b.onclick = () => {
        state.filters.track = b.dataset.track;
        syncFilterControlsFromState();
        loadImages(true);
      };
    });
    if ($("speciesFilter")) $("speciesFilter").onchange = (e) => { state.filters.species = e.target.value; loadImages(true); };
    if ($("scenarioFilter")) $("scenarioFilter").onchange = (e) => { state.filters.scenario_type = e.target.value; loadImages(true); };
    if ($("chkHybridOnly")) $("chkHybridOnly").onchange = (e) => { state.filters.hybrid_only = e.target.checked; loadImages(true); };
    $("chkBurstBest").onchange = (e) => { state.filters.only_burst_best = e.target.checked; loadImages(true); };

    // Search, Sort, Card Size
    let searchT = null;
    $("searchInput").oninput = (e) => {
      clearTimeout(searchT);
      searchT = setTimeout(() => {
        state.filters.search = e.target.value;
        loadImages(true);
      }, 220);
    };
    $("sortSelect").onchange = (e) => {
      const [by, dir] = e.target.value.split(":");
      state.filters.sort_by = by;
      state.filters.sort_dir = dir;
      loadImages(true);
    };
    $("cardSizeSlider").oninput = (e) => {
      document.documentElement.style.setProperty("--card-size", e.target.value + "px");
    };

    // Infinite scroll — trigger VERY early (5000px or 60% scrolled) + save scroll position for refresh!
    $("gridView").addEventListener("scroll", () => {
      const el = $("gridView");
      saveLocalScrollState();
      scheduleSessionSave();
      const nearBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 5000 ||
                         (el.scrollHeight > 0 && (el.scrollTop + el.clientHeight) / el.scrollHeight >= 0.6);
      if (nearBottom && state.items.length < state.total) {
        loadImages(false);
      }
    }, { passive: true });
    window.addEventListener("beforeunload", () => saveLocalScrollState());
    $("btnLoadMore").onclick = () => loadImages(false);

    // Filmstrip infinite horizontal scroll + mouse-wheel horizontal scrolling
    const fsScroll = $("filmstripScroll");
    fsScroll.addEventListener("scroll", () => {
      maybeExtendFilmstripRight();
    }, { passive: true });
    fsScroll.addEventListener("wheel", (e) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        fsScroll.scrollLeft += e.deltaY * 1.6;
        maybeExtendFilmstripRight();
      }
    }, { passive: false });

    // Batch actions
    document.querySelectorAll("[data-batch-rate]").forEach((b) => {
      b.onclick = () => batchUpdate({ rating: Number(b.dataset.batchRate) });
    });
    document.querySelectorAll("[data-batch-label]").forEach((b) => {
      b.onclick = () => batchUpdate({ label: b.dataset.batchLabel });
    });
    $("btnBatchCull").onclick = () => startCulling("selected");
    $("btnClearSel").onclick = () => {
      state.selectedIds.clear();
      document.querySelectorAll(".photo-card.selected").forEach((c) => c.classList.remove("selected"));
      document.querySelectorAll(".card-chk").forEach((c) => (c.checked = false));
      updateBatchBar();
    };

    // Loupe buttons & Zoom/Pan/AF controls
    $("btnLoupePrev").onclick = () => stepLoupe(-1);
    $("btnLoupeNext").onclick = () => stepLoupe(1);

    document.querySelectorAll("#zoomPresets .zp-btn").forEach((b) => {
      b.onclick = () => setZoomPreset(b.dataset.zoom === "fit" ? "fit" : Number(b.dataset.zoom));
    });
    $("btnJumpFocus").onclick = () => jumpZoomToFocus();
    const holdBtn = $("btnHoldFull");
    if (holdBtn) {
      holdBtn.addEventListener("mousedown", (e) => { e.preventDefault(); startHoldFullRes(); });
      holdBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startHoldFullRes(); }, { passive: false });
      holdBtn.addEventListener("mouseleave", () => stopHoldFullRes());
    }
    window.addEventListener("mouseup", () => stopHoldFullRes());
    window.addEventListener("touchend", () => stopHoldFullRes());
    window.addEventListener("blur", () => stopHoldFullRes());
    window.addEventListener("keyup", (e) => {
      if (e.code === "KeyQ" || e.key.toLowerCase() === "q") stopHoldFullRes();
    });
    $("btnToggleAf").onclick = () => {
      state.showAfPoints = !state.showAfPoints;
      $("btnToggleAf").classList.toggle("active", state.showAfPoints);
      if (state.items[state.activeIndex]) drawLoupeOverlay(state.items[state.activeIndex]);
    };
    $("btnToggleHighlights").onclick = () => {
      state.showHighlights = !state.showHighlights;
      $("btnToggleHighlights").classList.toggle("active", state.showHighlights);
      if (state.items[state.activeIndex]) drawLoupeOverlay(state.items[state.activeIndex]);
    };
    $("btnToggleBbox").onclick = () => {
      state.showBbox = !state.showBbox;
      $("btnToggleBbox").classList.toggle("active", state.showBbox);
      if (state.items[state.activeIndex]) drawLoupeOverlay(state.items[state.activeIndex]);
    };
    $("inspCropBox").onclick = () => {
      if (state.zoomMode === "fit" || state.zoomScale <= 1.01) setZoomPreset(100, true);
      else setZoomPreset("fit");
    };

    // Mouse wheel zoom on Loupe canvas
    const canvasArea = $("loupeCanvasArea");
    canvasArea.addEventListener("wheel", (e) => {
      if (state.view !== "loupe") return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.24 : 1 / 1.24;
      stepZoomByFactor(factor, e.clientX, e.clientY);
    }, { passive: false });

    // Mouse drag panning + click-to-toggle zoom (when not dragged)
    const imgWrap = $("loupeImgWrap");
    imgWrap.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      state.isDragging = true;
      state.dragMoved = false;
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.dragStartFocusX = state.focusX;
      state.dragStartFocusY = state.focusY;
      if (state.zoomScale > 1.01) imgWrap.classList.add("dragging");
    });
    window.addEventListener("mousemove", (e) => {
      if (!state.isDragging) return;
      const dx = e.clientX - state.dragStartX;
      const dy = e.clientY - state.dragStartY;
      if (Math.abs(dx) > 4 || Math.abs(dy) > 4) state.dragMoved = true;
      if (state.zoomScale > 1.01 && state.dragMoved) {
        const { baseW, baseH } = getLoupeBaseSize();
        state.focusX = Math.max(0, Math.min(1, state.dragStartFocusX - dx / (baseW * state.zoomScale)));
        state.focusY = Math.max(0, Math.min(1, state.dragStartFocusY - dy / (baseH * state.zoomScale)));
        applyLoupeZoomTransform();
      }
    });
    window.addEventListener("mouseup", (e) => {
      if (!state.isDragging) return;
      state.isDragging = false;
      imgWrap.classList.remove("dragging");
      if (!state.dragMoved && state.view === "loupe") {
        // Quick click without dragging toggles Fit <-> 100% centered on the EXACT clicked pixel!
        if (state.zoomMode === "fit" || state.zoomScale <= 1.01) {
          const rect = $("loupeImg").getBoundingClientRect();
          const cx = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
          const cy = Math.max(0, Math.min(1, (e.clientY - rect.top) / Math.max(1, rect.height)));
          state.focusX = cx;
          state.focusY = cy;
          setZoomPreset(100, false);
        } else {
          setZoomPreset("fit");
        }
      }
    });
    window.addEventListener("resize", () => {
      if (state.view === "loupe" && state.items[state.activeIndex]) {
        applyLoupeZoomTransform();
        drawLoupeOverlay(state.items[state.activeIndex]);
      }
    });
    $("btnLoupeCompareBurst").onclick = () => setView("compare");
    $("btnFlagPick").onclick = () => setUserFlag(state.items[state.activeIndex], 1);
    $("btnFlagReject").onclick = () => setUserFlag(state.items[state.activeIndex], -1);

    document.querySelectorAll(".fs-tab").forEach((b) => {
      b.onclick = () => {
        state.filmstripMode = b.dataset.fsMode;
        document.querySelectorAll(".fs-tab").forEach((x) => x.classList.toggle("active", x === b));
        renderFilmstrip();
      };
    });

    // Compare buttons
    $("btnCompareToggleCrop").onclick = () => {
      state.compareShowCrop = !state.compareShowCrop;
      renderCompareGrid();
    };
    $("btnCompareExit").onclick = () => setView("grid");

    // Topbar utility buttons
    $("btnUndo").onclick = () => undoLast();
    $("btnLogs").onclick = () => openLogsModal("events");
    $("btnLearning").onclick = () => openLearningModal();
    $("btnSettings").onclick = () => openSettingsModal();
    $("btnShortcuts").onclick = () => $("modalShortcuts").classList.remove("hidden");

    // Settings modal save & AI XMP sync
    $("btnSaveSettings").onclick = async () => {
      const prevZoomDim = state.settings.zoom_preview_max_dim;
      const payload = gatherSettingsForm();
      state.settings = await api("/api/settings", { method: "PUT", body: JSON.stringify(payload) });
      if (payload.zoom_preview_max_dim !== prevZoomDim) {
        state.fullPreloadCache.clear();
        state.loupeLoadedFullId = "";
      }
      $("chkAutoAdvance").checked = !!state.settings.auto_advance_on_rate;
      $("modalSettings").classList.add("hidden");
      toast("Settings saved!");
    };
    $("btnDryRunAiXmp").onclick = async () => {
      await api("/api/settings", { method: "PUT", body: JSON.stringify(gatherSettingsForm()) });
      const sub = state.filters.subfolder === "__all__" ? "" : state.filters.subfolder;
      const res = await api("/api/xmp/apply-ai-mapping", {
        method: "POST",
        body: JSON.stringify({ dir_id: state.currentDirId, subfolder: sub, dry_run: true }),
      });
      $("aiXmpStatusMsg").textContent = `Preview: would update ${res.would_update} XMP files (${res.skipped_rated} skipped because already rated).`;
    };
    $("btnApplyAiXmp").onclick = async () => {
      await api("/api/settings", { method: "PUT", body: JSON.stringify(gatherSettingsForm()) });
      const sub = state.filters.subfolder === "__all__" ? "" : state.filters.subfolder;
      $("aiXmpStatusMsg").textContent = "Writing XMP sidecars...";
      const res = await api("/api/xmp/apply-ai-mapping", {
        method: "POST",
        body: JSON.stringify({ dir_id: state.currentDirId, subfolder: sub, dry_run: false }),
      });
      $("aiXmpStatusMsg").textContent = `Done! Synced ${res.synced} XMPs (${res.skipped_rated} skipped, ${res.failed} failed). You can Ctrl+Z to undo.`;
      await loadImages(true);
    };

    // Logs modal tabs & filters
    document.querySelectorAll("#modalLogs .m-tab").forEach((b) => {
      b.onclick = () => switchLogTab(b.dataset.logTab);
    });
    $("logLevelFilter").onchange = () => loadEventLogs();
    $("logCatFilter").onchange = () => loadEventLogs();
    $("btnRefreshLogs").onclick = () => loadEventLogs();

    // Modal close buttons
    document.querySelectorAll("[data-close]").forEach((b) => {
      b.onclick = () => $(b.dataset.close).classList.add("hidden");
    });

    // Auto-blur dropdowns after selecting so keyboard shortcuts (0-9, Z, A, Q) keep working immediately
    document.querySelectorAll("select").forEach((s) => {
      s.addEventListener("change", () => s.blur());
    });

    // Global Keyboard Shortcuts (Lightroom / Photo Mechanic style)
    window.addEventListener("keydown", (e) => {
      const target = e.target;
      const tag = (target?.tagName || "").toLowerCase();
      const inputType = (target?.type || "").toLowerCase();
      // Only block shortcuts when typing inside text/search/number inputs or textareas (NOT checkboxes or buttons!)
      if (tag === "textarea" || (tag === "input" && !["checkbox", "radio", "range", "button"].includes(inputType))) return;
      if (tag === "select" && e.key !== "Escape") return;

      if ((e.ctrlKey || e.metaKey) && (e.code === "KeyZ" || e.key.toLowerCase() === "z")) {
        e.preventDefault();
        undoLast();
        return;
      }

      // Ctrl + / Ctrl - / Ctrl 0 zoom controls in Loupe view
      if ((e.ctrlKey || e.metaKey) && state.view === "loupe") {
        if (e.key === "=" || e.key === "+" || e.code === "Equal" || e.code === "NumpadAdd") {
          e.preventDefault();
          stepZoomByFactor(1.3);
          return;
        }
        if (e.key === "-" || e.key === "_" || e.code === "Minus" || e.code === "NumpadSubtract") {
          e.preventDefault();
          stepZoomByFactor(1 / 1.3);
          return;
        }
        if (e.key === "0" || e.code === "Digit0" || e.code === "Numpad0") {
          e.preventDefault();
          setZoomPreset("fit");
          return;
        }
      }

      if (e.key === "Escape") {
        document.querySelectorAll(".modal-backdrop").forEach((m) => m.classList.add("hidden"));
        if (state.view !== "grid") setView("grid");
        return;
      }

      // Use physical e.code so all shortcuts work identically in lowercase, uppercase, CapsLock, and Hebrew layout!
      const code = e.code;
      const k = e.key.toLowerCase();
      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        if (code === "KeyG" || k === "g") { setView("grid"); return; }
        if (code === "KeyE" || k === "e") { setView("loupe"); return; }
        if (code === "KeyC" || k === "c") { setView("compare"); return; }
        if (e.key === "?") { $("modalShortcuts").classList.toggle("hidden"); return; }
      }

      const activeImg = state.items[state.activeIndex];
      if (!activeImg) return;

      if (e.key === "ArrowRight") {
        e.preventDefault();
        if (state.view === "loupe") stepLoupe(1);
        else { state.activeIndex = Math.min(state.items.length - 1, state.activeIndex + 1); }
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (state.view === "loupe") stepLoupe(-1);
        else { state.activeIndex = Math.max(0, state.activeIndex - 1); }
      } else if (code === "KeyZ" || k === "z" || code === "Space" || e.key === " ") {
        if (state.view === "loupe") {
          e.preventDefault();
          if (state.zoomMode === "fit" || state.zoomScale <= 1.01) setZoomPreset(100, true);
          else setZoomPreset("fit");
        }
      } else if (code === "KeyA" || k === "a") {
        if (state.view === "loupe") {
          e.preventDefault();
          jumpZoomToFocus();
        }
      } else if (code === "KeyQ" || k === "q") {
        if (state.view === "loupe" && !e.repeat) {
          e.preventDefault();
          startHoldFullRes();
        }
      } else if (code === "KeyF" || k === "f") {
        if (state.view === "loupe") {
          e.preventDefault();
          $("btnToggleAf").click();
        }
      } else if (code === "KeyH" || k === "h") {
        if (state.view === "loupe") {
          e.preventDefault();
          $("btnToggleHighlights").click();
        }
      } else if (code === "KeyB" || k === "b") {
        if (state.view === "loupe") $("btnToggleBbox").click();
      } else if (
        ["Digit0", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Numpad0", "Numpad1", "Numpad2", "Numpad3", "Numpad4", "Numpad5"].includes(code) ||
        ["0", "1", "2", "3", "4", "5"].includes(e.key)
      ) {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const digit = ["0", "1", "2", "3", "4", "5"].includes(e.key) ? Number(e.key) : Number(code.slice(-1));
        if (state.view === "grid" && state.selectedIds.size > 0) batchUpdate({ rating: digit });
        else setRating(activeImg, digit);
      } else if (code === "Digit6" || code === "Numpad6" || e.key === "6") {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const nextLbl = activeImg.effective_label === "Red" ? "" : "Red";
        if (state.view === "grid" && state.selectedIds.size > 0) batchUpdate({ label: "Red" });
        else setColorLabel(activeImg, nextLbl);
      } else if (code === "Digit7" || code === "Numpad7" || e.key === "7") {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const nextLbl = activeImg.effective_label === "Yellow" ? "" : "Yellow";
        if (state.view === "grid" && state.selectedIds.size > 0) batchUpdate({ label: "Yellow" });
        else setColorLabel(activeImg, nextLbl);
      } else if (code === "Digit8" || code === "Numpad8" || e.key === "8") {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const nextLbl = activeImg.effective_label === "Green" ? "" : "Green";
        if (state.view === "grid" && state.selectedIds.size > 0) batchUpdate({ label: "Green" });
        else setColorLabel(activeImg, nextLbl);
      } else if (code === "Digit9" || code === "Numpad9" || e.key === "9") {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const nextLbl = activeImg.effective_label === "Blue" ? "" : "Blue";
        if (state.view === "grid" && state.selectedIds.size > 0) batchUpdate({ label: "Blue" });
        else setColorLabel(activeImg, nextLbl);
      } else if (code === "KeyP" || k === "p") {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const nextLbl = activeImg.effective_label === "Purple" ? "" : "Purple";
        if (state.view === "grid" && state.selectedIds.size > 0) batchUpdate({ label: "Purple" });
        else setColorLabel(activeImg, nextLbl);
      } else if (code === "KeyX" || k === "x") {
        setUserFlag(activeImg, -1);
      } else if (code === "KeyU" || k === "u") {
        setUserFlag(activeImg, 0);
      }
    });
  }

  window.addEventListener("DOMContentLoaded", init);
})();
