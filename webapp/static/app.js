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
    lastSelectedIdx: -1,
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
    highlightMaskCache: new Map(),
    highlightClipPct: 0,
    isDragging: false,
    dragMoved: false,
    dragStartX: 0,
    dragStartY: 0,
    dragStartFocusX: 0.5,
    dragStartFocusY: 0.5,
    compareShowCrop: false,
    compareItems: [],
    // #1 Burst-by-Burst Gallery state
    bursts: [],
    burstsTotal: 0,
    burstsOffset: 0,
    burstsLoading: false,
    burstFilter: "",
    burstSort: "timestamp",
    activeBurstIdx: 0,
    expandedBursts: new Map(), // burst_id -> full items array
    manualSplits: new Set(),   // image_ids where user manually split a burst
    // #2 Winner vs Challenger Split-Screen state
    wvcBurstId: "",
    wvcFrames: [],
    wvcWinnerIdx: 0,
    wvcChalIdx: 0,
    wvcSelectedIds: new Set(),
    wvcAnchorIdx: 0,
    wvcBurstPos: 0,
    wvcTotalBursts: 0,
    wvcPrevBurstId: "",
    wvcNextBurstId: "",
    wvcFirstImgPos: 0,
    wvcTotalImgs: 0,
    wvcZoomed: false,
    wvcScale: 3.2,
    wvcPanX: 0,
    wvcPanY: 0,
    wvcDragging: false,
    wvcDragMoved: false,
    wvcDragStartX: 0,
    wvcDragStartY: 0,
    wvcDragStartPanX: 0,
    wvcDragStartPanY: 0,
    wvcReturnView: "grid",
    ws: null,
    bookmarks: [],
    versionInfo: null,
    updateInfo: null,
    updatingApp: false,
    cropEditMode: false,
    cropDraft: null,
    cropHoverHorizonIdx: null,
    cropDrag: null,
    preloadCache: new Map(),
    fullPreloadCache: new Map(),
    loupeLoadedFullId: "",
    isHoldingFull: false,
    preHoldSrc: "",
    perfQueue: [],
    perfFlushTimer: null,
    saveSessionTimer: null,
    isRestoring: true,
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
    if (state.isRestoring) return;
    try {
      const gridEl = $("gridView");
      if (!state.currentDirId) return;
      const prev = getSavedLocalScrollState() || {};
      const activeImg = state.items[state.activeIndex];
      // Only update scrollTop when Grid view is actually visible (never overwrite with 0 from hidden Loupe!)
      const scrollTop = (state.view === "grid" && gridEl && gridEl.scrollHeight > 0)
        ? Math.round(gridEl.scrollTop || 0)
        : (prev.dir_id === state.currentDirId && prev.subfolder === state.filters.subfolder ? (prev.scrollTop || 0) : 0);
      const wvcWin = state.wvcFrames[state.wvcWinnerIdx];
      const wvcChal = state.wvcFrames[state.wvcChalIdx];
      sessionStorage.setItem("cull_grid_pos", JSON.stringify({
        dir_id: state.currentDirId,
        subfolder: state.filters.subfolder,
        filters: state.filters,
        view: state.view,
        active_image_id: activeImg ? activeImg.image_id : (prev.active_image_id || ""),
        active_index: state.activeIndex || 0,
        scrollTop,
        loadedCount: Math.max(state.items.length, prev.loadedCount || 0),
        wvc_burst_id: state.wvcBurstId || prev.wvc_burst_id || "",
        wvc_winner_id: wvcWin ? wvcWin.image_id : (prev.wvc_winner_id || ""),
        wvc_chal_id: wvcChal ? wvcChal.image_id : (prev.wvc_chal_id || ""),
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
    if (state.isRestoring) return;
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
            view: state.view,
            active_image_id: activeImg ? activeImg.image_id : "",
            active_index: state.activeIndex || 0,
            grid_scroll_top: local.scrollTop || 0,
            loaded_count: state.items.length,
            wvc_burst_id: local.wvc_burst_id || "",
            wvc_winner_id: local.wvc_winner_id || "",
            wvc_chal_id: local.wvc_chal_id || "",
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
        } else if (msg.type === "smart_crop_ready") {
          if (msg.dir_id === state.currentDirId) {
            handleSmartCropReadyWs(msg.image_ids || []);
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
        const resumeWvc = (merged.dir_id === initialDir.dir_id && merged.wvc_burst_id)
          ? { burst_id: merged.wvc_burst_id, winner_id: merged.wvc_winner_id || "", chal_id: merged.wvc_chal_id || "" }
          : null;
        await selectDirectory(initialDir.dir_id, resumeImgId, resumeView, resumeScroll, resumeLoaded, resumeWvc);
      } else {
        if ($("welcomeCard")) $("welcomeCard").classList.remove("hidden");
      }
    } catch (err) {
      toast("Startup error: " + err.message, 5000);
    } finally {
      state.isRestoring = false;
      saveLocalScrollState();
    }
    // Non-blocking background version & GitHub Releases update check (fails silently if offline)
    loadVersionInfo().catch(() => {});
    checkAppUpdates(false).catch(() => {});
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

  async function selectDirectory(dirId, resumeImageId = "", resumeView = "grid", resumeScrollTop = 0, resumeLoadedCount = 0, resumeWvc = null) {
    if ($("welcomeCard")) $("welcomeCard").classList.add("hidden");
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
    loadManualSplits(dirId);
    loadBookmarks(dirId);
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
    if (resumeView === "compare" && resumeWvc && resumeWvc.burst_id) {
      await openWinnerVsChallengerForBurst(resumeWvc.burst_id, resumeWvc.chal_id, resumeWvc.winner_id);
    } else if (resumeView === "bursts") {
      setView("bursts");
    } else if (resumeView === "loupe" && state.items.length > 0) {
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

  function hasVisibleCrop(crop) {
    if (!crop) return false;
    const l = Number(crop.left ?? 0);
    const t = Number(crop.top ?? 0);
    const r = Number(crop.right ?? 1);
    const b = Number(crop.bottom ?? 1);
    const ang = Number(crop.angle ?? 0);
    return (l > 0.002 || t > 0.002 || r < 0.998 || b < 0.998 || Math.abs(ang) >= 0.05);
  }

  function renderCardCropOverlayHtml(img) {
    const c = img && img.crop;
    if (!hasVisibleCrop(c)) return "";
    const l = (Math.max(0, Math.min(1, Number(c.left ?? 0))) * 100).toFixed(2);
    const t = (Math.max(0, Math.min(1, Number(c.top ?? 0))) * 100).toFixed(2);
    const w = (Math.max(0.02, Math.min(1, Number(c.right ?? 1) - Number(c.left ?? 0))) * 100).toFixed(2);
    const h = (Math.max(0.02, Math.min(1, Number(c.bottom ?? 1) - Number(c.top ?? 0))) * 100).toFixed(2);
    const ang = Number(c.angle ?? 0);
    const applied = !!(c.applied || img.crop_applied);
    const cls = applied ? "card-crop-rect applied" : "card-crop-rect";
    const badge = applied
      ? `<span class="card-crop-badge" title="Crop applied (${Math.round(c.crop_width_px || 0)}×${Math.round(c.crop_height_px || 0)} px · ${ang.toFixed(1)}°)">✂ ${Math.abs(ang) >= 0.05 ? `${ang.toFixed(1)}°` : "Crop"}</span>`
      : "";
    return `<div class="${cls}" style="left:${l}%;top:${t}%;width:${w}%;height:${h}%;transform:rotate(${ang.toFixed(2)}deg)"></div>${badge}`;
  }

  function createCardEl(img, idx) {
    const card = document.createElement("div");
    card.className = "photo-card" +
      (state.selectedIds.has(img.image_id) ? " selected" : "") +
      (idx === state.activeIndex && state.selectedIds.size > 0 ? " main-selected" : "") +
      (img.effective_label ? ` label-${img.effective_label}` : "");
    card.dataset.id = img.image_id;
    card.dataset.idx = String(idx);

    const spd = getSpeedInfo(img.exposure_seconds, img.exposure_time);
    const scoreTxt = img.is_culled ? Math.round(img.overall_score || img.sharpness_score || 0) : "—";
    const dispLayer = getDisplayLayerForImage(img);
    const layerLbl = LAYER_LABELS[dispLayer] || dispLayer;

    const canSplit = img.burst_size > 1 && img.burst_index > 1;
    const canUnsplit = state.manualSplits.has(img.image_id);

    card.innerHTML = `
      <div class="card-thumb-wrap">
        <input type="checkbox" class="card-chk" ${state.selectedIds.has(img.image_id) ? "checked" : ""} />
        <img class="card-thumb" loading="lazy" decoding="async" src="/api/media/thumb/${encodeURIComponent(img.image_id)}" alt="${escapeHtml(img.filename)}" />
        ${renderCardCropOverlayHtml(img)}
        <span class="card-layer-pill pill-${escapeHtml(dispLayer)}">${escapeHtml(layerLbl)}</span>
        ${canSplit ? `<button class="card-split-btn" style="top:28px" data-split-id="${escapeHtml(img.image_id)}" title="Split burst at #${img.burst_index}/${img.burst_size}">✂ Split</button>` : ""}
        ${canUnsplit ? `<button class="card-split-btn merge-btn" style="top:28px" data-unsplit-id="${escapeHtml(img.image_id)}" title="Merge split burst back">🔗 Merge</button>` : ""}
        ${img.burst_size > 1 ? `<span class="card-burst-badge">${escapeHtml(img.burst_id)} #${img.burst_index}/${img.burst_size}</span>` : ""}
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
      if (e.shiftKey) {
        selectGridRange(idx, true);
        return;
      }
      toggleSelect(img.image_id, chk.checked);
      if (chk.checked && state.selectedIds.size === 1) state.activeIndex = idx;
      state.lastSelectedIdx = idx;
      syncGridSelectionDom();
    });
    const spBtn = card.querySelector("[data-split-id]");
    if (spBtn) spBtn.onclick = (e) => { e.stopPropagation(); splitBurstAt(img.image_id); };
    const unspBtn = card.querySelector("[data-unsplit-id]");
    if (unspBtn) unspBtn.onclick = (e) => { e.stopPropagation(); unsplitBurstAt(img.image_id); };

    card.addEventListener("click", (e) => {
      if (e.shiftKey) {
        e.preventDefault();
        selectGridRange(idx, true);
        return;
      }
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        // Keep current main select included when starting a Ctrl multi-select
        if (state.selectedIds.size === 0 && state.items[state.activeIndex]) {
          state.selectedIds.add(state.items[state.activeIndex].image_id);
        }
        const next = !state.selectedIds.has(img.image_id);
        toggleSelect(img.image_id, next);
        if (next && state.selectedIds.size === 1) state.activeIndex = idx;
        state.lastSelectedIdx = idx;
        syncGridSelectionDom();
        return;
      }
      if (state.selectedIds.size > 0 && state.selectedIds.has(img.image_id)) {
        state.activeIndex = idx;
        state.lastSelectedIdx = idx;
        syncGridSelectionDom();
        return;
      }
      state.selectedIds.clear();
      updateBatchBar();
      state.activeIndex = idx;
      state.lastSelectedIdx = idx;
      syncGridSelectionDom();
      // Always open a newly selected image from the gallery in Fit (zoomed out)
      state.zoomMode = "fit";
      state.zoomScale = 1.0;
      state.focusX = 0.5;
      state.focusY = 0.5;
      setView("loupe");
    });

    return card;
  }

  function selectGridRange(targetIdx, preserveMain = true) {
    const anchor = state.lastSelectedIdx >= 0 ? state.lastSelectedIdx : Math.max(0, state.activeIndex);
    const lo = Math.max(0, Math.min(anchor, targetIdx));
    const hi = Math.min(state.items.length - 1, Math.max(anchor, targetIdx));
    for (let i = lo; i <= hi; i++) {
      if (state.items[i]) state.selectedIds.add(state.items[i].image_id);
    }
    if (!preserveMain || state.activeIndex < 0 || state.activeIndex >= state.items.length) {
      state.activeIndex = targetIdx;
    }
    state.lastSelectedIdx = targetIdx;
    updateBatchBar();
    syncGridSelectionDom();
  }

  function syncGridSelectionDom() {
    const activeId = state.items[state.activeIndex]?.image_id || "";
    const hasMulti = state.selectedIds.size > 0;
    document.querySelectorAll(".photo-card").forEach((c) => {
      const id = c.dataset.id;
      const isSel = state.selectedIds.has(id);
      c.classList.toggle("selected", isSel);
      c.classList.toggle("main-selected", hasMulti && id === activeId);
      const chk = c.querySelector(".card-chk");
      if (chk) chk.checked = isSel;
    });
    document.querySelectorAll("#filmstripScroll .fs-item").forEach((fs) => {
      fs.classList.toggle("fs-multi-selected", state.selectedIds.has(fs.dataset.id));
    });
  }

  function getTargetSelectionIds(mainImg, isWvc = false) {
    const set = isWvc ? state.wvcSelectedIds : state.selectedIds;
    if (!set || set.size === 0) return mainImg ? [mainImg.image_id] : [];
    const ids = new Set(set);
    if (mainImg && mainImg.image_id) ids.add(mainImg.image_id);
    return Array.from(ids);
  }

  function getDisplayLayerForImage(img) {
    if (state.filters.use_ai_layer) return img.ai_layer || "rejected";
    const lbl = img.effective_label || "";
    if (lbl === "Green") return "layer2_to_edit";
    if (lbl === "Yellow") return "layer1_keep";
    if (lbl === "Red") return "to_delete";
    return img.is_culled ? "rejected" : "unculled";
  }

  function applyImageColorFrame(imgEl, label) {
    if (!imgEl) return;
    imgEl.classList.remove("img-label-Green", "img-label-Yellow", "img-label-Red", "img-label-Blue", "img-label-Purple");
    if (label) imgEl.classList.add(`img-label-${label}`);
    if (imgEl.id === "loupeImg") {
      const ca = $("loupeCanvasArea");
      if (ca) {
        ca.classList.remove("canvas-label-Green", "canvas-label-Yellow", "canvas-label-Red", "canvas-label-Blue", "canvas-label-Purple");
        if (label) ca.classList.add(`canvas-label-${label}`);
      }
    }
  }

  function refreshCardInDom(img) {
    const dispLayer = getDisplayLayerForImage(img);
    const layerLbl = LAYER_LABELS[dispLayer] || dispLayer;

    // 1. Update Grid card (border, pill, stars)
    const card = document.querySelector(`.photo-card[data-id="${CSS.escape(img.image_id)}"]`);
    if (card) {
      card.className = "photo-card" +
        (state.selectedIds.has(img.image_id) ? " selected" : "") +
        (img.effective_label ? ` label-${img.effective_label}` : "");
      const stars = card.querySelector(".card-stars");
      if (stars) stars.innerHTML = renderStarsText(img.effective_rating);
      const pill = card.querySelector(".card-layer-pill");
      if (pill) {
        pill.className = `card-layer-pill pill-${dispLayer}`;
        pill.textContent = layerLbl;
      }
      const thumbWrap = card.querySelector(".card-thumb-wrap");
      if (thumbWrap) {
        thumbWrap.querySelectorAll(".card-crop-rect, .card-crop-badge").forEach((n) => n.remove());
        const cropHtml = renderCardCropOverlayHtml(img);
        if (cropHtml) {
          const imgNode = thumbWrap.querySelector(".card-thumb");
          if (imgNode) imgNode.insertAdjacentHTML("afterend", cropHtml);
        }
      }
    }

    // 2. Update Filmstrip thumbnail (in Loupe & Winner vs Challenger)
    document.querySelectorAll(`.fs-item[data-id="${CSS.escape(img.image_id)}"]`).forEach((fs) => {
      fs.classList.remove("fs-label-Green", "fs-label-Yellow", "fs-label-Red", "fs-label-Blue", "fs-label-Purple");
      if (img.effective_label) fs.classList.add(`fs-label-${img.effective_label}`);
      let st = fs.querySelector(".fs-star-mini");
      if (img.effective_rating) {
        if (!st) { st = document.createElement("span"); st.className = "fs-star-mini"; fs.appendChild(st); }
        st.textContent = "★".repeat(img.effective_rating);
      } else if (st) {
        st.remove();
      }
    });

    // 3. Update Burst-by-Burst card if visible
    document.querySelectorAll(`.burst-card[data-id="${CSS.escape(img.image_id)}"]`).forEach((bc) => {
      const lbl = img.effective_label || "";
      bc.className = "burst-card" + (lbl ? ` b-label-${lbl}` : " is-unselected");
      let tag = bc.querySelector(".burst-pick-tag");
      const tagTxt = lbl === "Green" ? "✓ EDIT" : (lbl === "Yellow" ? "✓ KEEP" : (lbl === "Red" ? "✕ DEL" : ""));
      if (tagTxt) {
        if (!tag) { tag = document.createElement("span"); bc.appendChild(tag); }
        tag.className = `burst-pick-tag bpt-${lbl}`;
        tag.textContent = tagTxt;
      } else if (tag) {
        tag.remove();
      }
    });
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
    if ($("burstsView")) $("burstsView").classList.toggle("hidden", viewName !== "bursts");
    $("loupeView").classList.toggle("hidden", viewName !== "loupe");
    $("compareView").classList.toggle("hidden", viewName !== "compare");

    if (viewName === "bursts") {
      loadBursts(true);
    } else if (viewName === "loupe") {
      if (state.items.length === 0) {
        toast("No images match current filters.");
        setView("grid");
        return;
      }
      if (state.activeIndex >= state.items.length) state.activeIndex = 0;
      renderLoupe();
    } else if (viewName === "compare") {
      state.wvcReturnView = (prevView === "bursts" || prevView === "loupe") ? prevView : "grid";
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

    // Top HUD — image counter + clickable folder (copies full folder path) & filename (copies filename)
    const loupeCtr = $("loupeCounter");
    if (loupeCtr) {
      const tot = state.total || state.items.length || 0;
      loupeCtr.textContent = `Image ${(state.activeIndex + 1).toLocaleString()} / ${tot.toLocaleString()}`;
    }
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
    const dispLayer = getDisplayLayerForImage(img);
    lp.textContent = LAYER_LABELS[dispLayer] || dispLayer;
    lp.className = `hud-chip pill-${dispLayer}`;
    applyImageColorFrame($("loupeImg"), img.effective_label);

    const spd = getSpeedInfo(img.exposure_seconds, img.exposure_time);
    const sp = $("loupeSpeedPill");
    sp.className = `hud-chip spd-badge ${spd.cls}`;
    sp.textContent = `${spd.icon} ${spd.text}`;
    sp.title = spd.label;

    $("btnToggleAf").classList.toggle("active", state.showAfPoints);
    $("btnToggleHighlights").classList.toggle("active", state.showHighlights);
    $("btnToggleBbox").classList.toggle("active", state.showBbox);
    if ($("btnLoupeSplitBurst")) {
      $("btnLoupeSplitBurst").classList.toggle("hidden", !(img.burst_size > 1 && img.burst_index > 1));
    }
    if ($("btnLoupeUnsplitBurst")) {
      $("btnLoupeUnsplitBurst").classList.toggle("hidden", !state.manualSplits.has(img.image_id));
    }
    updateZoomPresetUI();

    // Bottom rating & color picker
    renderLoupeControls(img);

    // Right Inspector
    $("inspCropImg").src = `/api/media/crop/${encodeURIComponent(img.image_id)}`;
    renderInspector(img);
    state.cropDraft = null;
    state.cropHoverHorizonIdx = null;
    renderLoupeCropPanel(img);
    positionLoupeCropBox();

    // Always fetch image detail (pulls burst siblings + on-the-fly Canon AF focus points!)
    $("fsPosition").textContent = `${state.activeIndex + 1} / ${state.total}`;
    const reqImageId = img.image_id;
    api(`/api/images/${encodeURIComponent(reqImageId)}`)
      .then((res) => {
        if (res.image) {
          img.af_points = res.image.af_points || [];
          img.crop = res.image.crop ?? img.crop ?? null;
          img.crop_applied = !!res.image.crop_applied;
        }
        state.burstSiblings = res.burst_siblings && res.burst_siblings.length ? res.burst_siblings : [img];
        $("fsBurstCount").textContent = String(state.burstSiblings.length);
        // Only redraw if user is still viewing this same image
        if (state.items[state.activeIndex]?.image_id === reqImageId) {
          renderLoupeCropPanel(img);
          positionLoupeCropBox();
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

  function computeGentleHighlightMask(el, cacheKey, updateLoupeBtn = true) {
    if (!el || !el.naturalWidth || !el.naturalHeight) return null;
    if (state.highlightMaskCache.has(cacheKey)) {
      const cached = state.highlightMaskCache.get(cacheKey);
      if (updateLoupeBtn) {
        const btn = $("btnToggleHighlights");
        if (btn) {
          const pctStr = cached.clipPct >= 0.05 ? ` ${cached.clipPct.toFixed(1)}%` : "";
          btn.textContent = `☀ Highlights${pctStr} (H)`;
        }
      }
      return cached.canvas;
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
      const clipPct = (clipped / Math.max(1, total)) * 100;
      state.highlightMaskCanvas = off;
      state.highlightMaskKey = cacheKey;
      state.highlightClipPct = clipPct;
      state.highlightMaskCache.set(cacheKey, { canvas: off, clipPct });
      if (state.highlightMaskCache.size > 16) {
        const oldestKey = state.highlightMaskCache.keys().next().value;
        state.highlightMaskCache.delete(oldestKey);
      }

      if (updateLoupeBtn) {
        const btn = $("btnToggleHighlights");
        if (btn) {
          const pctStr = clipPct >= 0.05 ? ` ${clipPct.toFixed(1)}%` : "";
          btn.textContent = `☀ Highlights${pctStr} (H)`;
        }
      }
      return off;
    } catch (_) {
      return null;
    }
  }

  function drawWvcPaneOverlay(canvas, el, img) {
    if (!canvas || !el || !img) return;
    const w = el.offsetWidth || el.clientWidth;
    const h = el.offsetHeight || el.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(3, Math.max(2, window.devicePixelRatio || 1));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    if (state.showHighlights && el.complete && el.naturalWidth > 0) {
      const mask = computeGentleHighlightMask(el, img.image_id, false);
      if (mask) ctx.drawImage(mask, 0, 0, w, h);
    }

    if (state.showAfPoints && Array.isArray(img.af_points) && img.af_points.length > 0) {
      for (const box of img.af_points) {
        const bx = box.x * w;
        const by = box.y * h;
        const bw = box.w * w;
        const bh = box.h * h;
        if (box.in_focus) {
          ctx.strokeStyle = "#ef4444";
          ctx.fillStyle = "rgba(239, 68, 68, 0.14)";
          ctx.lineWidth = 2.0;
          ctx.fillRect(bx, by, bw, bh);
          ctx.strokeRect(bx, by, bw, bh);
          const cx = bx + bw * 0.5;
          const cy = by + bh * 0.5;
          const tick = Math.min(bw, bh) * 0.25;
          ctx.beginPath();
          ctx.moveTo(cx - tick, cy); ctx.lineTo(cx + tick, cy);
          ctx.moveTo(cx, cy - tick); ctx.lineTo(cx, cy + tick);
          ctx.stroke();
        } else {
          ctx.strokeStyle = "rgba(239, 68, 68, 0.45)";
          ctx.lineWidth = 1.2;
          ctx.strokeRect(bx, by, bw, bh);
        }
      }
    }
  }

  function drawWvcOverlays() {
    if (state.view !== "compare") return;
    const win = state.wvcFrames[state.wvcWinnerIdx];
    const chal = state.wvcFrames[state.wvcChalIdx];
    $("btnWvcToggleAf")?.classList.toggle("active", state.showAfPoints);
    $("btnWvcToggleHighlights")?.classList.toggle("active", state.showHighlights);
    if (win) drawWvcPaneOverlay($("wvcWinOverlay"), $("wvcWinImg"), win);
    if (chal) drawWvcPaneOverlay($("wvcChalOverlay"), $("wvcChalImg"), chal);
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

    // 3. Smart Straighten & Crop preview (when NOT in DOM edit mode) + Detected Horizon Guide Line
    const activeCrop = (state.cropEditMode && state.cropDraft) ? state.cropDraft : img.crop;
    if (activeCrop) {
      const horizons = Array.isArray(activeCrop.horizons) ? activeCrop.horizons : [];
      const hIdx = state.cropHoverHorizonIdx !== null
        ? state.cropHoverHorizonIdx
        : (state.cropEditMode ? activeCrop.selected_horizon_idx : null);
      if (hIdx !== null && hIdx !== undefined && hIdx >= 0) {
        const hz = horizons.find((item) => item.id === hIdx) || horizons[hIdx];
        if (hz) {
          ctx.save();
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 2;
          ctx.setLineDash([7, 5]);
          ctx.beginPath();
          ctx.moveTo((hz.x1 ?? 0) * w, (hz.y1 ?? hz.y_norm ?? 0.5) * h);
          ctx.lineTo((hz.x2 ?? 1) * w, (hz.y2 ?? hz.y_norm ?? 0.5) * h);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
        }
      }

      if (!state.cropEditMode && hasVisibleCrop(activeCrop)) {
        const cx = ((activeCrop.left + activeCrop.right) * 0.5) * w;
        const cy = ((activeCrop.top + activeCrop.bottom) * 0.5) * h;
        const rw = (activeCrop.right - activeCrop.left) * w;
        const rh = (activeCrop.bottom - activeCrop.top) * h;
        const rad = ((activeCrop.angle || 0) * Math.PI) / 180;
        const applied = !!(activeCrop.applied || img.crop_applied);

        ctx.save();
        if (applied) {
          // Dim outside area using even-odd clip path around the rotated crop rect
          ctx.fillStyle = "rgba(0, 0, 0, 0.48)";
          ctx.beginPath();
          ctx.rect(0, 0, w, h);
          ctx.translate(cx, cy);
          ctx.rotate(rad);
          ctx.rect(-rw / 2, -rh / 2, rw, rh);
          ctx.fill("evenodd");
          ctx.strokeStyle = "#facc15";
          ctx.lineWidth = 2;
          ctx.strokeRect(-rw / 2, -rh / 2, rw, rh);
        } else {
          // Suggested (unapplied) crop preview: faint gray dashed border
          ctx.translate(cx, cy);
          ctx.rotate(rad);
          ctx.strokeStyle = "rgba(200, 210, 225, 0.68)";
          ctx.lineWidth = 1.6;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(-rw / 2, -rh / 2, rw, rh);
          ctx.setLineDash([]);
        }
        ctx.restore();
      }
    }
  }

  // ------------------------------------------------------------------
  // Auto Smart Straighten & Smart Crop Controller
  // ------------------------------------------------------------------
  function getOrigDimsForCrop(img, crop) {
    const el = $("loupeImg");
    const ow = Number(crop?.orig_width || img?.width || el?.naturalWidth || 6000);
    const oh = Number(crop?.orig_height || img?.height || el?.naturalHeight || 4000);
    return { ow: Math.max(1, ow), oh: Math.max(1, oh) };
  }

  function ensureCropDraft(img) {
    if (!img) return null;
    if (state.cropDraft && state.cropDraft._imageId === img.image_id) return state.cropDraft;
    const base = img.crop || {};
    const { ow, oh } = getOrigDimsForCrop(img, base);
    state.cropDraft = {
      _imageId: img.image_id,
      left: Number(base.left ?? 0),
      top: Number(base.top ?? 0),
      right: Number(base.right ?? 1),
      bottom: Number(base.bottom ?? 1),
      angle: Number(base.angle ?? 0),
      aspect_locked: base.aspect_locked !== undefined ? !!base.aspect_locked : true,
      aspect_ratio: Number(base.aspect_ratio || (ow / oh)),
      orig_width: ow,
      orig_height: oh,
      min_long_px: Number(base.min_long_px || 2700),
      selected_horizon_idx: base.selected_horizon_idx ?? null,
      horizons: Array.isArray(base.horizons) ? base.horizons : [],
      source: base.source || "manual",
      applied: !!(base.applied || img.crop_applied),
    };
    return state.cropDraft;
  }

  function updateCropSizeReadout(img, c) {
    const sizeEl = $("cropSizeReadout");
    const clampEl = $("cropMinClampBadge");
    if (!sizeEl) return;
    if (!c) {
      const { ow, oh } = getOrigDimsForCrop(img, null);
      sizeEl.textContent = `${Math.round(ow)} × ${Math.round(oh)} px (Full)`;
      if (clampEl) clampEl.classList.add("hidden");
      return;
    }
    const { ow, oh } = getOrigDimsForCrop(img, c);
    const wPx = Math.round(Math.max(0, (c.right - c.left) * ow));
    const hPx = Math.round(Math.max(0, (c.bottom - c.top) * oh));
    const minLong = Number(c.min_long_px || 2700);
    const isFull = !hasVisibleCrop(c);
    sizeEl.textContent = `${wPx.toLocaleString()} × ${hPx.toLocaleString()} px${isFull ? " (Full)" : ""}`;
    if (clampEl) {
      const clamped = !isFull && Math.max(ow, oh) > minLong && Math.max(wPx, hPx) <= minLong + 3;
      clampEl.classList.toggle("hidden", !clamped);
    }
  }

  function renderLoupeCropPanel(img) {
    if (!img) return;
    const c = (state.cropEditMode && state.cropDraft && state.cropDraft._imageId === img.image_id)
      ? state.cropDraft
      : img.crop;

    const badge = $("cropStatusBadge");
    if (badge) {
      badge.className = "crop-status-pill";
      if (c && (c.applied || img.crop_applied)) {
        badge.textContent = "Applied ✓";
        badge.classList.add("applied");
      } else if (c && hasVisibleCrop(c)) {
        badge.textContent = "Suggested";
        badge.classList.add("suggested");
      } else {
        badge.textContent = "None";
      }
    }

    $("btnToggleCropEdit")?.classList.toggle("active", state.cropEditMode);
    $("btnCropEditBtn")?.classList.toggle("active", state.cropEditMode);

    updateCropSizeReadout(img, c);
    const ruleEl = $("cropRuleBadge");
    if (ruleEl) {
      const ruleTxt = c && c.composition_rule ? String(c.composition_rule) : "";
      ruleEl.textContent = ruleTxt;
      ruleEl.title = ruleTxt ? `Composition: ${ruleTxt}` : "";
      ruleEl.classList.toggle("hidden", !ruleTxt);
    }

    const ang = Number(c?.angle ?? 0);
    if ($("cropAngleSlider")) $("cropAngleSlider").value = String(ang.toFixed(1));
    if ($("cropAngleInput")) $("cropAngleInput").value = ang.toFixed(1);

    const locked = c ? (c.aspect_locked !== undefined ? !!c.aspect_locked : true) : true;
    const lockBtn = $("btnCropLockRatio");
    if (lockBtn) {
      lockBtn.classList.toggle("active", locked);
      lockBtn.textContent = locked ? "🔒 Locked" : "🔓 Free";
    }

    // Highlight matching ratio preset
    const { ow, oh } = getOrigDimsForCrop(img, c);
    const origRatio = ow / oh;
    const curRatio = Number(c?.aspect_ratio || origRatio);
    document.querySelectorAll("#cropRatioPresets .crp-btn").forEach((b) => {
      const target = b.dataset.ratio === "orig" ? origRatio : Number(b.dataset.ratio);
      b.classList.toggle("active", locked && Math.abs(curRatio - target) < 0.03);
    });

    // Detected Horizons selector
    const hzWrap = $("cropHorizonsWrap");
    const hzList = $("cropHorizonsList");
    const horizons = Array.isArray(c?.horizons) ? c.horizons : [];
    if (hzWrap && hzList) {
      if (horizons.length > 0) {
        hzWrap.classList.remove("hidden");
        hzList.innerHTML = "";
        const selIdx = c?.selected_horizon_idx;
        const noneBtn = document.createElement("button");
        noneBtn.className = "horizon-pill" + ((selIdx === null || selIdx === undefined || selIdx < 0) && Math.abs(ang) < 0.05 ? " active" : "");
        noneBtn.textContent = "0.0° Level / None";
        noneBtn.onclick = () => selectCropHorizon(img, -1, 0.0);
        hzList.appendChild(noneBtn);

        horizons.forEach((hz, i) => {
          const hid = hz.id !== undefined ? hz.id : i;
          const b = document.createElement("button");
          b.className = "horizon-pill" + (selIdx === hid ? " active" : "");
          b.textContent = hz.label || `Horizon ${i + 1} (${Number(hz.angle || 0).toFixed(1)}°)`;
          b.title = `Confidence: ${Math.round((hz.confidence || 0) * 100)}% · Click to align to this horizon`;
          b.onmouseenter = () => { state.cropHoverHorizonIdx = hid; drawLoupeOverlay(img); };
          b.onmouseleave = () => { state.cropHoverHorizonIdx = null; drawLoupeOverlay(img); };
          b.onclick = () => selectCropHorizon(img, hid, Number(hz.angle || 0));
          hzList.appendChild(b);
        });
      } else {
        hzWrap.classList.add("hidden");
      }
    }
  }

  function positionLoupeCropBox() {
    const layer = $("loupeCropLayer");
    const box = $("loupeCropBox");
    const img = state.items[state.activeIndex];
    if (!layer || !box) return;
    if (!state.cropEditMode || state.view !== "loupe" || !img) {
      layer.classList.add("hidden");
      return;
    }
    const c = ensureCropDraft(img);
    layer.classList.remove("hidden");
    box.classList.toggle("suggested", !c.applied);
    box.style.left = `${(c.left * 100).toFixed(3)}%`;
    box.style.top = `${(c.top * 100).toFixed(3)}%`;
    box.style.width = `${((c.right - c.left) * 100).toFixed(3)}%`;
    box.style.height = `${((c.bottom - c.top) * 100).toFixed(3)}%`;
    box.style.transform = `rotate(${Number(c.angle || 0).toFixed(2)}deg)`;
  }

  async function setCropEditMode(enabled) {
    const img = state.items[state.activeIndex];
    state.cropEditMode = !!enabled;
    if (state.cropEditMode && img) {
      if (state.zoomScale > 1.01) setZoomPreset("fit");
      if (!img.crop) {
        await runSmartCropAuto(img, false, false);
      }
      ensureCropDraft(img);
    } else {
      state.cropHoverHorizonIdx = null;
    }
    if (img) {
      renderLoupeCropPanel(img);
      positionLoupeCropBox();
      drawLoupeOverlay(img);
    }
  }

  function applyCropResponseToState(res) {
    if (!res || !res.image) return null;
    const updated = res.image;
    const item = state.items.find((x) => x.image_id === updated.image_id);
    if (item) {
      Object.assign(item, updated);
      refreshCardInDom(item);
    }
    const wvcItem = state.wvcFrames.find((x) => x.image_id === updated.image_id);
    if (wvcItem) Object.assign(wvcItem, updated);
    const active = state.items[state.activeIndex];
    if (active && active.image_id === updated.image_id) {
      state.cropDraft = null;
      if (state.cropEditMode) ensureCropDraft(active);
      renderLoupeCropPanel(active);
      positionLoupeCropBox();
      drawLoupeOverlay(active);
    }
    return item || updated;
  }

  async function runSmartCropAuto(img, applyNow = false, force = false) {
    if (!img) return;
    toast(applyNow ? "Calculating & applying Smart Crop..." : "Calculating Smart Straighten & Crop...", 1400);
    try {
      const res = await api(`/api/images/${encodeURIComponent(img.image_id)}/smart-crop`, {
        method: "POST",
        body: JSON.stringify({ apply: !!applyNow, force: !!force, min_long_px: 2700 }),
      });
      applyCropResponseToState(res);
      if (res.crop) {
        const w = Math.round(res.crop.crop_width_px || 0);
        const h = Math.round(res.crop.crop_height_px || 0);
        const ang = Number(res.crop.angle || 0).toFixed(1);
        toast(`${applyNow ? "Applied" : "Suggested"} crop: ${w}×${h} px (${ang}°)`, 2000);
      }
    } catch (err) {
      toast("Smart Crop error: " + err.message, 3500);
    }
  }

  async function saveCropDraft(img, overrideApplied = null) {
    if (!img) return;
    const d = ensureCropDraft(img);
    const applied = overrideApplied !== null ? !!overrideApplied : !!d.applied;
    d.applied = applied;
    try {
      const res = await api(`/api/images/${encodeURIComponent(img.image_id)}/crop`, {
        method: "PUT",
        body: JSON.stringify({
          left: d.left,
          top: d.top,
          right: d.right,
          bottom: d.bottom,
          angle: d.angle,
          aspect_locked: d.aspect_locked,
          aspect_ratio: d.aspect_ratio,
          selected_horizon_idx: d.selected_horizon_idx,
          applied,
        }),
      });
      applyCropResponseToState(res);
      if (overrideApplied === true) {
        toast(`Crop applied${res.xmp_synced ? " & synced to XMP ✓" : " ✓"}`, 1800);
      }
    } catch (err) {
      toast("Save crop failed: " + err.message, 3500);
    }
  }

  async function selectCropHorizon(img, horizonIdx, angleDeg) {
    const d = ensureCropDraft(img);
    d.selected_horizon_idx = horizonIdx >= 0 ? horizonIdx : null;
    d.angle = Number(angleDeg || 0);
    renderLoupeCropPanel(img);
    positionLoupeCropBox();
    drawLoupeOverlay(img);
    await saveCropDraft(img, null);
  }

  async function resetCropForImage(img, mode = "clear") {
    if (!img) return;
    try {
      const res = await api(`/api/images/${encodeURIComponent(img.image_id)}/crop/reset`, {
        method: "POST",
        body: JSON.stringify({ mode }),
      });
      applyCropResponseToState(res);
      toast(mode === "auto" ? "Recomputed Auto Smart Crop" : "Crop & straighten reset to full frame", 1600);
    } catch (err) {
      toast("Reset crop failed: " + err.message, 3500);
    }
  }

  async function batchSmartCrop(reset = false) {
    const ids = Array.from(state.selectedIds);
    if (!ids.length) return;
    toast(reset ? `Resetting crop on ${ids.length} images...` : `Smart cropping ${ids.length} images...`, 2000);
    try {
      const res = await api("/api/images/batch-smart-crop", {
        method: "POST",
        body: JSON.stringify({ image_ids: ids, apply: !reset, reset: !!reset, force: false }),
      });
      const updatedList = res.images || [];
      for (const u of updatedList) {
        const item = state.items.find((x) => x.image_id === u.image_id);
        if (item) {
          Object.assign(item, u);
          refreshCardInDom(item);
        }
      }
      toast(reset ? `Reset crop on ${res.updated || ids.length} images` : `Applied Smart Crop to ${res.updated || ids.length} images`, 2600);
    } catch (err) {
      toast("Batch Smart Crop error: " + err.message, 4000);
    }
  }

  async function handleSmartCropReadyWs(imageIds) {
    if (!Array.isArray(imageIds) || !imageIds.length) return;
    const idSet = new Set(imageIds);
    const affected = state.items.filter((x) => idSet.has(x.image_id));
    if (!affected.length) return;
    // Refresh up to 24 visible/loaded items smoothly in place without jarring full reload
    for (const item of affected.slice(0, 24)) {
      try {
        const res = await api(`/api/images/${encodeURIComponent(item.image_id)}`);
        if (res.image) {
          item.crop = res.image.crop ?? null;
          item.crop_applied = !!res.image.crop_applied;
          refreshCardInDom(item);
          if (state.view === "loupe" && state.items[state.activeIndex]?.image_id === item.image_id) {
            renderLoupeCropPanel(item);
            positionLoupeCropBox();
            drawLoupeOverlay(item);
          }
        }
      } catch (_) {}
    }
  }

  function applyCropRatioPreset(img, ratioStr) {
    const d = ensureCropDraft(img);
    const { ow, oh } = getOrigDimsForCrop(img, d);
    const targetRatio = ratioStr === "orig" ? (ow / oh) : Number(ratioStr);
    if (!targetRatio || targetRatio <= 0) return;
    d.aspect_locked = true;
    d.aspect_ratio = targetRatio;
    // Re-fit box around current center preserving normalized area
    const cx = (d.left + d.right) * 0.5;
    const cy = (d.top + d.bottom) * 0.5;
    const normRatio = targetRatio / (ow / oh); // (w_norm / h_norm)
    let nw = d.right - d.left;
    let nh = nw / normRatio;
    if (nh > 1) { nh = 1; nw = nh * normRatio; }
    if (nw > 1) { nw = 1; nh = nw / normRatio; }
    let l = cx - nw / 2;
    let t = cy - nh / 2;
    if (l < 0) l = 0;
    if (t < 0) t = 0;
    if (l + nw > 1) l = 1 - nw;
    if (t + nh > 1) t = 1 - nh;
    d.left = Math.max(0, l);
    d.top = Math.max(0, t);
    d.right = Math.min(1, l + nw);
    d.bottom = Math.min(1, t + nh);
    renderLoupeCropPanel(img);
    positionLoupeCropBox();
    drawLoupeOverlay(img);
    saveCropDraft(img, null);
  }

  function renderLoupeControls(img) {
    const starsEl = $("loupeStars");
    starsEl.innerHTML = "";
    for (let i = 1; i <= 5; i++) {
      const b = document.createElement("button");
      b.className = "star-btn" + (i <= (img.effective_rating || 0) ? " on" : "");
      b.textContent = "★";
      b.title = `Set ${i} star${i > 1 ? "s" : ""} (Key ${i})`;
      b.onclick = () => {
        const nextR = img.effective_rating === i ? 0 : i;
        const targets = getTargetSelectionIds(img, false);
        if (targets.length > 1) batchUpdate({ rating: nextR }, targets);
        else setRating(img, nextR);
      };
      starsEl.appendChild(b);
    }

    const colEl = $("loupeColors");
    colEl.innerHTML = "";
    for (const c of COLORS) {
      const b = document.createElement("button");
      b.className = `cp-dot ${c.css}` + ((img.effective_label || "") === c.key ? " active" : "");
      b.title = `${c.name} (${c.shortcut})`;
      if (!c.key) b.textContent = "∅";
      b.onclick = () => {
        const nextLbl = c.key ? (img.effective_label === c.key ? "" : c.key) : "";
        const targets = getTargetSelectionIds(img, false);
        if (targets.length > 1) batchUpdate({ label: nextLbl }, targets);
        else setColorLabel(img, nextLbl);
      };
      colEl.appendChild(b);
    }

    $("btnFlagPick").classList.toggle("active", img.user_flag === 1);
    $("btnFlagReject").classList.toggle("active", img.user_flag === -1);

    // Immediately reflect current color directly on the Loupe image border & top HUD pill
    applyImageColorFrame($("loupeImg"), img.effective_label);
    const lp = $("loupeLayerPill");
    if (lp) {
      const dispLayer = getDisplayLayerForImage(img);
      lp.textContent = LAYER_LABELS[dispLayer] || dispLayer;
      lp.className = `hud-chip pill-${dispLayer}`;
    }

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
    const lblCls = item.effective_label ? ` fs-label-${item.effective_label}` : "";
    const multiCls = state.selectedIds.has(item.image_id) ? " fs-multi-selected" : "";
    div.className = "fs-item" + lblCls + multiCls + (item.image_id === currentId ? " active" : "");
    div.dataset.id = item.image_id;
    div.title = `${item.filename} · ${item.effective_label || "Unselected"} · Sharp: ${Math.round(item.sharpness_score || 0)} (Shift+Click range · Ctrl+Click multi-select)`;
    div.innerHTML = `
      <img loading="lazy" decoding="async" src="/api/media/thumb/${encodeURIComponent(item.image_id)}" alt="" />
      <div class="fs-sharp-bar" style="width:${Math.min(100, item.sharpness_score || 0)}%"></div>
      ${item.effective_rating ? `<span class="fs-star-mini">${"★".repeat(item.effective_rating)}</span>` : ""}
    `;
    div.onclick = (e) => {
      const idx = state.items.findIndex((x) => x.image_id === item.image_id);
      if (e.shiftKey && idx >= 0) {
        e.preventDefault();
        selectGridRange(idx, true);
        return;
      }
      if ((e.ctrlKey || e.metaKey) && idx >= 0) {
        e.preventDefault();
        if (state.selectedIds.size === 0 && state.items[state.activeIndex]) {
          state.selectedIds.add(state.items[state.activeIndex].image_id);
        }
        toggleSelect(item.image_id, !state.selectedIds.has(item.image_id));
        state.lastSelectedIdx = idx;
        syncGridSelectionDom();
        return;
      }
      const keepSelection = state.selectedIds.size > 0 && state.selectedIds.has(item.image_id);
      if (!keepSelection) {
        state.selectedIds.clear();
        updateBatchBar();
      }
      if (idx >= 0) {
        state.activeIndex = idx;
        state.lastSelectedIdx = idx;
        renderLoupe();
      } else {
        state.items.splice(state.activeIndex + 1, 0, item);
        state.activeIndex += 1;
        state.lastSelectedIdx = state.activeIndex;
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
  // #1 BURST-BY-BURST GALLERY MODE
  // ------------------------------------------------------------------
  async function loadBursts(reset = false) {
    if (!state.currentDirId || state.burstsLoading) return;
    state.burstsLoading = true;
    if (reset) {
      state.burstsOffset = 0;
      state.bursts = [];
      state.activeBurstIdx = 0;
      if ($("burstsList")) $("burstsList").innerHTML = "";
    }
    const f = state.filters;
    const params = new URLSearchParams({
      dir_id: state.currentDirId,
      subfolder: f.subfolder === "__all__" ? "" : f.subfolder,
      track: f.track || "",
      species: f.species || "",
      scenario_type: f.scenario_type || "",
      search: f.search || "",
      burst_filter: state.burstFilter || "",
      use_ai_layer: String(!!f.use_ai_layer),
      sort_by: state.burstSort || "timestamp",
      offset: String(state.burstsOffset),
      limit: "40",
    });
    try {
      const res = await api(`/api/bursts?${params.toString()}`);
      state.burstsTotal = res.total;
      const startIdx = state.bursts.length;
      state.bursts.push(...res.bursts);
      state.burstsOffset += res.bursts.length;
      if ($("burstsCountLabel")) {
        $("burstsCountLabel").textContent = `${state.burstsTotal.toLocaleString()} bursts / scenarios`;
      }
      const frag = document.createDocumentFragment();
      res.bursts.forEach((b, i) => frag.appendChild(createBurstRowEl(b, startIdx + i)));
      $("burstsList").appendChild(frag);
      $("burstsLoadMoreWrap").classList.toggle("hidden", state.bursts.length >= state.burstsTotal);
    } catch (err) {
      toast("Failed to load bursts: " + err.message);
    } finally {
      state.burstsLoading = false;
    }
  }

  async function loadManualSplits(dirId) {
    if (!dirId) return;
    try {
      const res = await api(`/api/bursts/splits/${dirId}`);
      state.manualSplits = new Set(res.image_ids || []);
    } catch (_) {}
  }

  async function splitBurstAt(imageId) {
    if (!imageId) return;
    try {
      const res = await api("/api/bursts/split", {
        method: "POST",
        body: JSON.stringify({ image_id: imageId }),
      });
      state.manualSplits.add(imageId);
      state.expandedBursts.clear();
      toast(`✂ Split ${res.orig_burst_id} → ${res.orig_burst_id} (${res.left_count}) + ${res.new_burst_id} (${res.right_count})`, 3000);
      if (state.view === "bursts") {
        await loadBursts(true);
      } else if (state.view === "compare") {
        await loadWvcBurst(res.new_burst_id, imageId);
      } else if (state.view === "loupe") {
        const cur = state.items[state.activeIndex];
        if (cur && cur.image_id === imageId) {
          cur.burst_id = res.new_burst_id;
          cur.burst_index = 1;
          cur.burst_size = res.right_count;
        }
        renderLoupe();
      } else {
        await loadImages(true);
      }
    } catch (err) {
      toast("Split failed: " + err.message, 3500);
    }
  }

  async function unsplitBurstAt(imageId) {
    if (!imageId) return;
    try {
      const res = await api("/api/bursts/unsplit", {
        method: "POST",
        body: JSON.stringify({ image_id: imageId }),
      });
      state.manualSplits.delete(imageId);
      state.expandedBursts.clear();
      toast(`🔗 Merged back into ${res.merged_burst_id} (${res.total_count} frames)`, 2800);
      if (state.view === "bursts") {
        await loadBursts(true);
      } else if (state.view === "compare") {
        await loadWvcBurst(res.merged_burst_id, imageId);
      } else if (state.view === "loupe") {
        const cur = state.items[state.activeIndex];
        if (cur && cur.image_id === imageId) {
          cur.burst_id = res.merged_burst_id;
          cur.burst_size = res.total_count;
        }
        renderLoupe();
      } else {
        await loadImages(true);
      }
    } catch (err) {
      toast("Merge failed: " + err.message, 3500);
    }
  }

  function createBurstCardEl(img, burstObj) {
    const lbl = img.effective_label || "";
    const card = document.createElement("div");
    card.className = "burst-card" + (lbl ? ` b-label-${lbl}` : " is-unselected");
    card.dataset.id = img.image_id;
    const tagTxt = lbl === "Green" ? "✓ EDIT" : (lbl === "Yellow" ? "✓ KEEP" : (lbl === "Red" ? "✕ DEL" : ""));
    const canSplit = img.burst_size > 1 && img.burst_index > 1;
    const canUnsplit = state.manualSplits.has(img.image_id);
    card.innerHTML = `
      <img class="burst-card-thumb" loading="lazy" decoding="async" src="/api/media/thumb/${encodeURIComponent(img.image_id)}" alt="${escapeHtml(img.filename)}" />
      ${tagTxt ? `<span class="burst-pick-tag bpt-${lbl}">${tagTxt}</span>` : ""}
      ${canSplit ? `<button class="card-split-btn" data-split-id="${escapeHtml(img.image_id)}" title="Split burst starting at #${img.burst_index}/${img.burst_size}">✂ Split</button>` : ""}
      ${canUnsplit ? `<button class="card-split-btn merge-btn" data-unsplit-id="${escapeHtml(img.image_id)}" title="Merge this split burst back into previous burst">🔗 Merge</button>` : ""}
      <div class="burst-card-info">
        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">#${img.burst_index} ${escapeHtml(img.filename)}</span>
        <span class="muted">S:${Math.round(img.sharpness_score || 0)}</span>
      </div>
    `;
    const spBtn = card.querySelector("[data-split-id]");
    if (spBtn) {
      spBtn.onclick = (e) => { e.stopPropagation(); splitBurstAt(img.image_id); };
    }
    const unspBtn = card.querySelector("[data-unsplit-id]");
    if (unspBtn) {
      unspBtn.onclick = (e) => { e.stopPropagation(); unsplitBurstAt(img.image_id); };
    }
    card.onclick = (e) => {
      e.stopPropagation();
      // Open directly in Winner vs Challenger focused on this exact frame!
      openWinnerVsChallengerForBurst(burstObj.burst_id, img.image_id);
    };
    return card;
  }

  function createBurstRowEl(b, idx) {
    const row = document.createElement("div");
    row.className = "burst-row" + (idx === state.activeBurstIdx ? " active-burst" : "");
    row.dataset.burstId = b.burst_id;
    row.dataset.burstIdx = String(idx);

    const scenMap = { action_sequence: "🎬 Action", static_group: "📸 Static", single: "1️⃣ Single" };
    const isExpanded = state.expandedBursts.has(b.burst_id);
    const expandedItems = isExpanded ? state.expandedBursts.get(b.burst_id) : null;

    row.innerHTML = `
      <div class="burst-row-head">
        <div class="burst-row-title">
          <strong>${escapeHtml(b.burst_id)}</strong>
          <span>${escapeHtml(b.subfolder)}</span>
          ${b.species ? `<span class="badge">${escapeHtml(b.species)}</span>` : ""}
          ${b.scenario_type ? `<span class="badge">${scenMap[b.scenario_type] || b.scenario_type}</span>` : ""}
          <span class="muted">${b.count} frames · Best ${Math.round(b.max_score)}</span>
          ${b.edit_count ? `<span class="badge badge-l2">🟢 ${b.edit_count} Edit</span>` : ""}
          ${b.keep_count ? `<span class="badge badge-l1">🟡 ${b.keep_count} Keep</span>` : ""}
          ${b.delete_count ? `<span class="badge badge-del">🔴 ${b.delete_count} Del</span>` : ""}
          ${(!b.edit_count && !b.keep_count) ? `<span class="badge">0 selected</span>` : ""}
        </div>
        <div class="burst-row-actions">
          <button class="btn btn-xs btn-primary" data-wvc-open="${escapeHtml(b.burst_id)}">◫ Winner vs Challenger</button>
          ${b.count > 1 ? `<button class="btn btn-xs btn-ghost" data-burst-toggle="${escapeHtml(b.burst_id)}">${isExpanded ? "▲ Collapse" : `▼ Show All ${b.count}`}</button>` : ""}
        </div>
      </div>
      <div class="burst-strip-grid"></div>
    `;

    const grid = row.querySelector(".burst-strip-grid");
    if (isExpanded && expandedItems) {
      expandedItems.forEach((img) => grid.appendChild(createBurstCardEl(img, b)));
    } else {
      const shown = b.selected_items.length > 0 ? b.selected_items : b.preview_others;
      shown.forEach((img) => grid.appendChild(createBurstCardEl(img, b)));
      const remaining = b.count - shown.length;
      if (remaining > 0) {
        const expTile = document.createElement("button");
        expTile.className = "burst-expand-tile";
        expTile.innerHTML = `<span>+ ${remaining} more frames</span><span class="muted">Click or Space to inspect all</span>`;
        expTile.onclick = (e) => {
          e.stopPropagation();
          toggleExpandBurst(b, row);
        };
        grid.appendChild(expTile);
      }
    }

    row.onclick = () => setActiveBurstIdx(idx);
    row.querySelector("[data-wvc-open]").onclick = (e) => {
      e.stopPropagation();
      setActiveBurstIdx(idx);
      openWinnerVsChallengerForBurst(b.burst_id);
    };
    const togBtn = row.querySelector("[data-burst-toggle]");
    if (togBtn) {
      togBtn.onclick = (e) => {
        e.stopPropagation();
        setActiveBurstIdx(idx);
        toggleExpandBurst(b, row);
      };
    }
    return row;
  }

  function setActiveBurstIdx(idx) {
    if (idx < 0 || idx >= state.bursts.length) return;
    state.activeBurstIdx = idx;
    document.querySelectorAll(".burst-row.active-burst").forEach((el) => el.classList.remove("active-burst"));
    const el = document.querySelector(`.burst-row[data-burst-idx="${idx}"]`);
    if (el) {
      el.classList.add("active-burst");
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    if (idx >= state.bursts.length - 8 && state.bursts.length < state.burstsTotal && !state.burstsLoading) {
      loadBursts(false);
    }
  }

  async function toggleExpandBurst(b, rowEl) {
    if (state.expandedBursts.has(b.burst_id)) {
      state.expandedBursts.delete(b.burst_id);
    } else {
      try {
        const res = await api(`/api/bursts/${state.currentDirId}/${encodeURIComponent(b.burst_id)}`);
        state.expandedBursts.set(b.burst_id, res.items || []);
      } catch (err) {
        toast("Failed to expand burst: " + err.message);
        return;
      }
    }
    const idx = Number(rowEl.dataset.burstIdx);
    const fresh = createBurstRowEl(b, idx);
    rowEl.replaceWith(fresh);
  }

  // ------------------------------------------------------------------
  // #2 WINNER VS. CHALLENGER SPLIT-SCREEN COMPARE
  // ------------------------------------------------------------------
  async function openWinnerVsChallengerForBurst(burstId, focusImageId = "", pinnedWinnerId = "") {
    if (!burstId) return;
    state.wvcBurstId = burstId;
    state.wvcReturnView = state.view === "compare" ? state.wvcReturnView : state.view;
    state.view = "compare";
    document.querySelectorAll(".view-switcher .btn-seg").forEach((b) => {
      b.classList.toggle("active", b.dataset.view === "compare");
    });
    $("gridView").classList.add("hidden");
    if ($("burstsView")) $("burstsView").classList.add("hidden");
    $("loupeView").classList.add("hidden");
    $("compareView").classList.remove("hidden");
    await loadWvcBurst(burstId, focusImageId, pinnedWinnerId);
  }

  async function openCompareView() {
    let targetBurstId = state.wvcBurstId;
    let focusImgId = "";
    if (state.wvcReturnView === "bursts" && state.bursts[state.activeBurstIdx]) {
      targetBurstId = state.bursts[state.activeBurstIdx].burst_id;
    } else if (state.items[state.activeIndex]?.burst_id) {
      targetBurstId = state.items[state.activeIndex].burst_id;
      focusImgId = state.items[state.activeIndex].image_id;
    }
    if (!targetBurstId && state.bursts.length > 0) {
      targetBurstId = state.bursts[0].burst_id;
    }
    if (!targetBurstId) {
      toast("Select a photo or burst to open Winner vs. Challenger.");
      setView("grid");
      return;
    }
    await loadWvcBurst(targetBurstId, focusImgId);
  }

  async function loadWvcBurst(burstId, focusImageId = "", pinnedWinnerId = "") {
    try {
      const sub = state.filters.subfolder === "__all__" ? "" : (state.filters.subfolder || "");
      const q = sub ? `?subfolder=${encodeURIComponent(sub)}` : "";
      const res = await api(`/api/bursts/${state.currentDirId}/${encodeURIComponent(burstId)}${q}`);
      const frames = res.items || [];
      if (!frames.length) {
        toast("Burst has no frames.");
        return;
      }
      state.wvcBurstId = burstId;
      state.wvcFrames = frames;
      state.wvcBurstPos = res.burst_pos || 0;
      state.wvcTotalBursts = res.total_bursts || 0;
      state.wvcPrevBurstId = res.prev_burst_id || "";
      state.wvcNextBurstId = res.next_burst_id || "";
      state.wvcFirstImgPos = res.first_image_pos || 0;
      state.wvcTotalImgs = res.total_images || 0;

      // Restore exact pinned winner if specified, else pick best Green -> Yellow -> highest score
      let winIdx = pinnedWinnerId ? frames.findIndex((f) => f.image_id === pinnedWinnerId) : -1;
      if (winIdx < 0) {
        let bestScore = -1;
        for (const tier of ["Green", "Yellow", ""]) {
          frames.forEach((f, i) => {
            const match = tier === "" || f.effective_label === tier;
            if (match && (f.overall_score || 0) > bestScore) {
              bestScore = f.overall_score || 0;
              winIdx = i;
            }
          });
          if (winIdx >= 0) break;
        }
      }
      state.wvcWinnerIdx = Math.max(0, winIdx);

      // Pick Challenger: clicked image if specified, else always start from the FIRST image (index 0)
      let chalIdx = 0;
      if (focusImageId) {
        const fi = frames.findIndex((f) => f.image_id === focusImageId);
        if (fi >= 0) chalIdx = fi;
      }
      state.wvcChalIdx = chalIdx;
      state.wvcSelectedIds.clear();
      state.wvcAnchorIdx = chalIdx;
      state.wvcZoomed = false;
      state.wvcScale = 3.2;
      state.wvcPanX = 0;
      state.wvcPanY = 0;
      renderWvc();
    } catch (err) {
      toast("Failed to load burst: " + err.message);
    }
  }

  function applyWvcZoomTransforms() {
    const win = state.wvcFrames[state.wvcWinnerIdx];
    const chal = state.wvcFrames[state.wvcChalIdx];
    const pairs = [
      { wrap: $("wvcLeftWrap"), imgEl: $("wvcWinImg"), rec: win },
      { wrap: $("wvcRightWrap"), imgEl: $("wvcChalImg"), rec: chal },
    ];
    $("btnWvcZoomToggle").classList.toggle("active", state.wvcZoomed);
    for (const p of pairs) {
      if (!p.wrap || !p.imgEl) continue;
      p.wrap.classList.toggle("zoomed", state.wvcZoomed);
      if (!state.wvcZoomed || !p.rec) {
        p.wrap.style.transform = "translate(0px, 0px) scale(1)";
        continue;
      }
      const fc = getCameraFocusCenter(p.rec);
      const fx = Math.max(0, Math.min(1, fc.x + state.wvcPanX));
      const fy = Math.max(0, Math.min(1, fc.y + state.wvcPanY));
      const bw = p.imgEl.offsetWidth || 1;
      const bh = p.imgEl.offsetHeight || 1;
      const sc = state.wvcScale || 3.2;
      const tx = (0.5 - fx) * bw * sc;
      const ty = (0.5 - fy) * bh * sc;
      p.wrap.style.transform = `translate(${tx.toFixed(1)}px, ${ty.toFixed(1)}px) scale(${sc.toFixed(3)})`;
    }
  }

  function renderWvcPaneControls(container, img) {
    if (!container || !img) return;
    container.innerHTML = `
      <div style="display:flex;align-items:center;gap:6px">
        <div class="star-picker">
          ${[1, 2, 3, 4, 5].map((r) => `<button class="star-btn ${r <= (img.effective_rating || 0) ? "on" : ""}" data-wr="${r}">★</button>`).join("")}
        </div>
        <div class="color-picker">
          ${COLORS.map((c) => `<button class="cp-dot ${c.css} ${(img.effective_label || "") === c.key ? "active" : ""}" data-wl="${c.key}" title="${c.name}">${c.key ? "" : "∅"}</button>`).join("")}
        </div>
      </div>
      <span class="muted">Overall: <strong>${Math.round(img.overall_score || 0)}</strong> · Sharp: <strong>${Math.round(img.sharpness_score || 0)}</strong> · Eye: <strong>${Math.round(img.eye_contact_score || 0)}</strong></span>
    `;
    container.querySelectorAll("[data-wr]").forEach((b) => {
      b.onclick = async () => {
        const r = Number(b.dataset.wr);
        const nextR = img.effective_rating === r ? 0 : r;
        const isChalPane = container.id === "wvcChalControls";
        const targets = isChalPane ? getTargetSelectionIds(img, true) : [img.image_id];
        if (targets.length > 1) {
          await batchUpdate({ rating: nextR }, targets);
        } else {
          await setRating(img, nextR, false);
          renderWvc();
        }
      };
    });
    container.querySelectorAll("[data-wl]").forEach((b) => {
      b.onclick = async () => {
        const nextLbl = b.dataset.wl ? (img.effective_label === b.dataset.wl ? "" : b.dataset.wl) : "";
        const isChalPane = container.id === "wvcChalControls";
        const targets = isChalPane ? getTargetSelectionIds(img, true) : [img.image_id];
        if (targets.length > 1) {
          await batchUpdate({ label: nextLbl }, targets);
        } else {
          await setColorLabel(img, nextLbl, false);
          renderWvc();
        }
      };
    });
  }

  function renderWvc() {
    const frames = state.wvcFrames;
    if (!frames.length) return;
    const win = frames[state.wvcWinnerIdx] || frames[0];
    const chal = frames[state.wvcChalIdx] || frames[0];

    const greenCnt = frames.filter((f) => f.effective_label === "Green").length;
    const yellowCnt = frames.filter((f) => f.effective_label === "Yellow").length;
    $("compareTitle").textContent = `${state.wvcBurstId} · ${frames.length} frames (${greenCnt} Green, ${yellowCnt} Yellow)`;

    // Burst counter (current / total bursts)
    const bCtr = $("compareBurstCounter");
    if (bCtr) {
      const bIdx = state.bursts.findIndex((b) => b.burst_id === state.wvcBurstId);
      const bCur = bIdx >= 0 ? (bIdx + 1) : state.wvcBurstPos;
      const bTot = (bIdx >= 0 && state.burstsTotal > 0) ? state.burstsTotal : state.wvcTotalBursts;
      bCtr.textContent = bTot > 0 ? `Burst ${bCur.toLocaleString()} / ${bTot.toLocaleString()}` : "Burst —";
    }

    // Image counter (current Challenger / total filtered or directory images)
    const iCtr = $("compareImageCounter");
    if (iCtr) {
      const filteredIdx = state.items.findIndex((it) => it.image_id === chal.image_id);
      if (filteredIdx >= 0) {
        const tot = state.total || state.items.length;
        iCtr.textContent = `Image ${(filteredIdx + 1).toLocaleString()} / ${tot.toLocaleString()}`;
      } else if (state.wvcFirstImgPos > 0 && state.wvcTotalImgs > 0) {
        iCtr.textContent = `Image ${(state.wvcFirstImgPos + state.wvcChalIdx).toLocaleString()} / ${state.wvcTotalImgs.toLocaleString()}`;
      } else {
        iCtr.textContent = `Frame ${state.wvcChalIdx + 1} / ${frames.length}`;
      }
    }

    const deltaSharp = (chal.sharpness_score || 0) - (win.sharpness_score || 0);
    const db = $("compareDeltaBadge");
    if (win.image_id === chal.image_id) {
      db.textContent = "Same frame";
      db.style.color = "#cbd5e1";
    } else if (deltaSharp >= 0) {
      db.textContent = `Challenger +${deltaSharp.toFixed(1)} sharper`;
      db.style.color = "#4ade80";
    } else {
      db.textContent = `Winner +${Math.abs(deltaSharp).toFixed(1)} sharper`;
      db.style.color = "#fbbf24";
    }

    // Pane border highlights
    const lp = $("wvcLeftPane");
    const rp = $("wvcRightPane");
    lp.className = "wvc-pane" + (win.effective_label ? ` wvc-${win.effective_label.toLowerCase()}` : "");
    rp.className = "wvc-pane" + (chal.effective_label ? ` wvc-${chal.effective_label.toLowerCase()}` : "");

    $("wvcWinName").textContent = win.filename;
    $("wvcWinMeta").textContent = `${win.effective_label || "None"} · ${win.exposure_time || ""}`;
    $("wvcChalName").textContent = chal.filename;
    $("wvcChalMeta").textContent = `${chal.effective_label || "None"} · ${chal.exposure_time || ""}`;
    if ($("btnWvcSplitHere")) {
      $("btnWvcSplitHere").classList.toggle("hidden", !(frames.length > 1 && state.wvcChalIdx > 0));
    }
    if ($("btnWvcUnsplitHere")) {
      $("btnWvcUnsplitHere").classList.toggle("hidden", !state.manualSplits.has(chal.image_id));
    }

    const winUrl = state.wvcZoomed ? getZoomMediaUrl(win.image_id) : `/api/media/hq/${encodeURIComponent(win.image_id)}`;
    const chalUrl = state.wvcZoomed ? getZoomMediaUrl(chal.image_id) : `/api/media/hq/${encodeURIComponent(chal.image_id)}`;
    const wi = $("wvcWinImg");
    const ci = $("wvcChalImg");
    wi.onload = () => { applyWvcZoomTransforms(); drawWvcOverlays(); };
    ci.onload = () => { applyWvcZoomTransforms(); drawWvcOverlays(); };
    applyImageColorFrame(wi, win.effective_label);
    applyImageColorFrame(ci, chal.effective_label);
    if (wi.dataset.loadedUrl !== winUrl) {
      wi.dataset.loadedUrl = winUrl;
      wi.src = winUrl;
    }
    if (ci.dataset.loadedUrl !== chalUrl) {
      ci.dataset.loadedUrl = chalUrl;
      ci.src = chalUrl;
    }
    applyWvcZoomTransforms();
    drawWvcOverlays();

    renderWvcPaneControls($("wvcWinControls"), win);
    renderWvcPaneControls($("wvcChalControls"), chal);

    // Fetch AF points for both frames if missing so synced focus zoom & AF box overlay work on both panes!
    for (const f of [win, chal]) {
      if (f.af_points === null || f.af_points === undefined) {
        api(`/api/images/${encodeURIComponent(f.image_id)}`).then((r) => {
          if (r.image) f.af_points = r.image.af_points || [];
          if (state.wvcZoomed) applyWvcZoomTransforms();
          drawWvcOverlays();
        }).catch(() => {});
      }
    }

    // Show/hide prev & next arrows at the burst boundaries (no endless wrap-around)
    if ($("btnWvcChalPrev")) $("btnWvcChalPrev").classList.toggle("hidden", state.wvcChalIdx <= 0);
    if ($("btnWvcChalNext")) $("btnWvcChalNext").classList.toggle("hidden", state.wvcChalIdx >= frames.length - 1);

    // Render Color-Coded Burst Filmstrip
    const wvcSelCount = state.wvcSelectedIds.size;
    $("wvcStripPos").textContent = wvcSelCount > 1
      ? `Frame ${state.wvcChalIdx + 1} / ${frames.length} · ${wvcSelCount} selected (Esc to clear)`
      : `Frame ${state.wvcChalIdx + 1} / ${frames.length}`;
    const strip = $("wvcFilmstrip");
    strip.innerHTML = "";
    const frag = document.createDocumentFragment();
    frames.forEach((f, i) => {
      const div = document.createElement("div");
      const lblCls = f.effective_label ? ` fs-label-${f.effective_label}` : "";
      const pinCls = i === state.wvcWinnerIdx ? " fs-pinned" : "";
      const multiCls = state.wvcSelectedIds.has(f.image_id) ? " fs-multi-selected" : "";
      div.className = "fs-item" + lblCls + pinCls + multiCls + (i === state.wvcChalIdx ? " active" : "");
      div.dataset.id = f.image_id;
      div.title = `${f.filename} · ${f.effective_label || "Unselected"} · Sharp: ${Math.round(f.sharpness_score || 0)} (Shift+Click range · Ctrl+Click multi-select)`;
      div.innerHTML = `
        <img loading="lazy" decoding="async" src="/api/media/thumb/${encodeURIComponent(f.image_id)}" alt="" />
        <div class="fs-sharp-bar" style="width:${Math.min(100, f.sharpness_score || 0)}%"></div>
        ${f.effective_rating ? `<span class="fs-star-mini">${"★".repeat(f.effective_rating)}</span>` : ""}
      `;
      div.onclick = (e) => {
        if (e.shiftKey) {
          e.preventDefault();
          const anchor = state.wvcAnchorIdx >= 0 ? state.wvcAnchorIdx : state.wvcChalIdx;
          const lo = Math.max(0, Math.min(anchor, i));
          const hi = Math.min(frames.length - 1, Math.max(anchor, i));
          for (let k = lo; k <= hi; k++) {
            state.wvcSelectedIds.add(frames[k].image_id);
          }
          state.wvcAnchorIdx = i;
          renderWvc();
          return;
        }
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          if (state.wvcSelectedIds.size === 0 && frames[state.wvcChalIdx]) {
            state.wvcSelectedIds.add(frames[state.wvcChalIdx].image_id);
          }
          if (state.wvcSelectedIds.has(f.image_id)) state.wvcSelectedIds.delete(f.image_id);
          else state.wvcSelectedIds.add(f.image_id);
          state.wvcAnchorIdx = i;
          renderWvc();
          return;
        }
        if (!(state.wvcSelectedIds.size > 0 && state.wvcSelectedIds.has(f.image_id))) {
          state.wvcSelectedIds.clear();
        }
        state.wvcChalIdx = i;
        state.wvcAnchorIdx = i;
        renderWvc();
      };
      frag.appendChild(div);
    });
    strip.appendChild(frag);
    const actEl = strip.querySelector(".fs-item.active");
    if (actEl) actEl.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    scheduleSessionSave();
  }

  function stepWvcChallenger(delta) {
    const n = state.wvcFrames.length;
    if (n <= 1) return;
    const next = state.wvcChalIdx + delta;
    if (next < 0 || next >= n) return;
    state.wvcChalIdx = next;
    renderWvc();
  }

  async function wvcSwapWinner() {
    const win = state.wvcFrames[state.wvcWinnerIdx];
    const chal = state.wvcFrames[state.wvcChalIdx];
    if (!win || !chal || win.image_id === chal.image_id) return;
    const winColor = (win.effective_label === "Green" || win.effective_label === "Yellow") ? win.effective_label : "Green";
    const demoteColor = winColor === "Green" ? "Yellow" : "";
    await setColorLabel(chal, winColor, false);
    await setColorLabel(win, demoteColor, false);
    state.wvcWinnerIdx = state.wvcChalIdx;
    renderWvc();
    toast(`Swapped Winner! ${chal.filename} → ${winColor}, ${win.filename} → ${demoteColor || "None"}`);
  }

  async function wvcKeepBoth() {
    const win = state.wvcFrames[state.wvcWinnerIdx];
    const chal = state.wvcFrames[state.wvcChalIdx];
    if (!win || !chal) return;
    const winColor = (win.effective_label === "Green" || win.effective_label === "Yellow") ? win.effective_label : "Green";
    await setColorLabel(chal, winColor, false);
    renderWvc();
    toast(`Kept Both! ${chal.filename} → ${winColor}`);
  }

  async function wvcStepBurst(delta) {
    // If current burst is in the loaded burst list (e.g. from Burst Rows with filters/sort), step within that list
    let idx = state.bursts.findIndex((b) => b.burst_id === state.wvcBurstId);
    if (idx >= 0) {
      const nextIdx = idx + delta;
      if (nextIdx >= state.bursts.length && state.bursts.length < state.burstsTotal) {
        await loadBursts(false);
      }
      if (nextIdx < 0 || nextIdx >= state.bursts.length) {
        toast(delta > 0 ? "Reached last burst" : "Already at first burst");
        return;
      }
      state.activeBurstIdx = nextIdx;
      if (nextIdx >= state.bursts.length - 5 && state.bursts.length < state.burstsTotal) {
        loadBursts(false);
      }
      await loadWvcBurst(state.bursts[nextIdx].burst_id);
      return;
    }

    // Otherwise (e.g. right after a page refresh deep in the shoot), use exact server-provided prev/next burst IDs!
    const targetBurstId = delta > 0 ? state.wvcNextBurstId : state.wvcPrevBurstId;
    if (!targetBurstId) {
      toast(delta > 0 ? "Reached last burst" : "Already at first burst");
      return;
    }
    await loadWvcBurst(targetBurstId);
  }

  // ------------------------------------------------------------------
  // Editable Jump-to-Position Counters & Persistent Bookmarks
  // ------------------------------------------------------------------
  function openInlineJumpEditor(chipEl, currentVal, maxVal, prefixLabel, onJump) {
    if (!chipEl || chipEl.querySelector("input")) return;
    const origText = chipEl.textContent;
    const input = document.createElement("input");
    input.type = "number";
    input.className = "jump-inline-input";
    input.min = "1";
    if (maxVal > 0) input.max = String(maxVal);
    input.value = String(Math.max(1, currentVal || 1));

    let finished = false;
    const cancel = () => {
      if (finished) return;
      finished = true;
      chipEl.textContent = origText;
    };
    const commit = async () => {
      if (finished) return;
      finished = true;
      const n = parseInt(input.value, 10);
      chipEl.textContent = origText;
      if (!Number.isFinite(n) || n < 1) return;
      const target = maxVal > 0 ? Math.max(1, Math.min(maxVal, n)) : n;
      await onJump(target);
    };

    chipEl.textContent = "";
    if (prefixLabel) {
      const sp = document.createElement("span");
      sp.textContent = prefixLabel + " ";
      chipEl.appendChild(sp);
    }
    chipEl.appendChild(input);
    if (maxVal > 0) {
      const sp2 = document.createElement("span");
      sp2.textContent = ` / ${maxVal.toLocaleString()}`;
      chipEl.appendChild(sp2);
    }
    input.focus();
    input.select();

    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") {
        e.preventDefault();
        commit();
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancel();
      }
    });
    input.addEventListener("blur", () => cancel());
  }

  async function jumpToBurstPosition(pos) {
    if (!state.currentDirId) return;
    // If Burst Rows list is already loaded and covers `pos`, use it so active burst filter/sort is honored
    if (state.bursts.length >= pos && state.bursts[pos - 1]) {
      state.activeBurstIdx = pos - 1;
      await loadWvcBurst(state.bursts[pos - 1].burst_id);
      return;
    }
    try {
      const sub = state.filters.subfolder === "__all__" ? "" : (state.filters.subfolder || "");
      const q = sub ? `&subfolder=${encodeURIComponent(sub)}` : "";
      const res = await api(`/api/bursts/${state.currentDirId}/at-position?pos=${encodeURIComponent(pos)}${q}`);
      if (res.burst_id) {
        await loadWvcBurst(res.burst_id);
        toast(`Jumped to Burst ${res.pos} / ${res.total_bursts}`, 1400);
      }
    } catch (err) {
      toast("Jump to burst failed: " + err.message, 3000);
    }
  }

  async function jumpToImagePositionInWvc(pos) {
    if (!state.currentDirId) return;
    const chal = state.wvcFrames[state.wvcChalIdx];
    const inFiltered = chal && state.items.some((it) => it.image_id === chal.image_id);
    if (inFiltered && pos <= state.items.length && state.items[pos - 1]?.burst_id) {
      const target = state.items[pos - 1];
      await loadWvcBurst(target.burst_id, target.image_id);
      toast(`Jumped to Image ${pos.toLocaleString()} (${target.filename})`, 1400);
      return;
    }
    try {
      const sub = state.filters.subfolder === "__all__" ? "" : (state.filters.subfolder || "");
      const q = sub ? `&subfolder=${encodeURIComponent(sub)}` : "";
      const res = await api(`/api/images/${state.currentDirId}/at-position?pos=${encodeURIComponent(pos)}${q}`);
      if (res.burst_id) {
        await loadWvcBurst(res.burst_id, res.image_id);
        toast(`Jumped to Image ${res.pos.toLocaleString()} (${res.filename})`, 1400);
      } else {
        toast("That image is not assigned to a burst yet.", 2500);
      }
    } catch (err) {
      toast("Jump to image failed: " + err.message, 3000);
    }
  }

  async function jumpToImagePositionInLoupe(pos) {
    const tot = state.total || state.items.length;
    if (!tot) return;
    const targetIdx = Math.max(0, Math.min(tot - 1, pos - 1));
    while (state.items.length <= targetIdx && state.items.length < state.total) {
      await loadImages(false);
    }
    if (targetIdx < state.items.length) {
      state.activeIndex = targetIdx;
      renderLoupe();
      toast(`Jumped to Image ${(targetIdx + 1).toLocaleString()} / ${tot.toLocaleString()}`, 1400);
    }
  }

  async function loadBookmarks(dirId) {
    if (!dirId) return;
    try {
      const res = await api(`/api/bookmarks?dir_id=${encodeURIComponent(dirId)}`);
      state.bookmarks = res.bookmarks || [];
      renderBookmarksList();
    } catch (_) {}
  }

  function renderBookmarksList() {
    const badge = $("bookmarkCountBadge");
    if (badge) {
      badge.textContent = String(state.bookmarks.length);
      badge.classList.toggle("hidden", state.bookmarks.length === 0);
    }
    const list = $("bookmarksList");
    if (!list) return;
    if (!state.bookmarks.length) {
      list.innerHTML = `<div class="bm-empty muted">No bookmarks yet. Click "+ Bookmark Here" to save your spot!</div>`;
      return;
    }
    list.innerHTML = "";
    const viewNames = { compare: "Winner vs", loupe: "Loupe", grid: "Grid", bursts: "Bursts" };
    for (const bm of state.bookmarks) {
      const row = document.createElement("div");
      row.className = "bm-item";
      const vCls = `v-${bm.view || "compare"}`;
      const vLabel = viewNames[bm.view] || bm.view;
      const titleParts = [];
      if (bm.label) titleParts.push(bm.label);
      else if (bm.view === "compare" && bm.burst_pos) titleParts.push(`Burst ${bm.burst_pos}`);
      else if (bm.filename) titleParts.push(bm.filename);
      else titleParts.push(vLabel);

      const subParts = [];
      if (bm.burst_id) subParts.push(bm.burst_pos ? `Burst ${bm.burst_pos} (${bm.burst_id})` : bm.burst_id);
      if (bm.image_pos) subParts.push(`Image ${Number(bm.image_pos).toLocaleString()}`);
      if (bm.filename) subParts.push(bm.filename);
      if (bm.subfolder) subParts.push(bm.subfolder);

      row.innerHTML = `
        <div class="bm-item-main">
          <div class="bm-item-top">
            <span class="bm-view-pill ${vCls}">${escapeHtml(vLabel)}</span>
            <span>${escapeHtml(titleParts.join(" · "))}</span>
          </div>
          <div class="bm-item-sub">${escapeHtml(subParts.join(" · ") || "Saved position")}</div>
        </div>
        <button class="bm-del-btn" title="Delete bookmark">✕</button>
      `;
      row.onclick = (e) => {
        if (e.target.closest(".bm-del-btn")) return;
        $("bookmarksDropdown")?.classList.add("hidden");
        jumpToBookmark(bm);
      };
      row.querySelector(".bm-del-btn").onclick = (e) => {
        e.stopPropagation();
        deleteBookmark(bm.id);
      };
      list.appendChild(row);
    }
  }

  async function saveCurrentBookmark(customLabel = "") {
    if (!state.currentDirId) return;
    const sub = state.filters.subfolder === "__all__" ? "" : (state.filters.subfolder || "");
    const payload = {
      dir_id: state.currentDirId,
      label: (customLabel || "").trim(),
      view: state.view,
      subfolder: sub,
      burst_id: "",
      winner_id: "",
      chal_id: "",
      image_id: "",
      filename: "",
      burst_pos: 0,
      image_pos: 0,
      scroll_top: Math.round($("gridView")?.scrollTop || 0),
    };
    if (state.view === "compare") {
      const win = state.wvcFrames[state.wvcWinnerIdx];
      const chal = state.wvcFrames[state.wvcChalIdx];
      const bIdx = state.bursts.findIndex((b) => b.burst_id === state.wvcBurstId);
      const filteredIdx = chal ? state.items.findIndex((it) => it.image_id === chal.image_id) : -1;
      payload.burst_id = state.wvcBurstId || "";
      payload.winner_id = win?.image_id || "";
      payload.chal_id = chal?.image_id || "";
      payload.image_id = chal?.image_id || win?.image_id || "";
      payload.filename = chal?.filename || win?.filename || "";
      payload.burst_pos = bIdx >= 0 ? (bIdx + 1) : (state.wvcBurstPos || 0);
      payload.image_pos = filteredIdx >= 0 ? (filteredIdx + 1) : (state.wvcFirstImgPos ? (state.wvcFirstImgPos + state.wvcChalIdx) : 0);
    } else if (state.view === "loupe") {
      const img = state.items[state.activeIndex];
      payload.image_id = img?.image_id || "";
      payload.filename = img?.filename || "";
      payload.burst_id = img?.burst_id || "";
      payload.image_pos = state.activeIndex + 1;
    } else if (state.view === "bursts") {
      const b = state.bursts[state.activeBurstIdx];
      payload.burst_id = b?.burst_id || "";
      payload.burst_pos = state.activeBurstIdx + 1;
    } else {
      const img = state.items[state.activeIndex];
      payload.image_id = img?.image_id || "";
      payload.filename = img?.filename || "";
      payload.image_pos = state.activeIndex + 1;
    }

    try {
      const res = await api("/api/bookmarks", { method: "POST", body: JSON.stringify(payload) });
      if (res.bookmark) {
        state.bookmarks.unshift(res.bookmark);
        renderBookmarksList();
        toast(`🔖 Bookmarked ${payload.label || payload.filename || payload.burst_id || payload.view}`, 1800);
      }
    } catch (err) {
      toast("Bookmark error: " + err.message, 3000);
    }
  }

  async function jumpToBookmark(bm) {
    if (!bm) return;
    // Restore subfolder if it differs
    const targetSub = bm.subfolder || "__all__";
    if (state.filters.subfolder !== targetSub) {
      state.filters.subfolder = targetSub;
      syncFilterControlsFromState();
      await loadImages(true);
    }
    if (bm.view === "compare" && bm.burst_id) {
      await openWinnerVsChallengerForBurst(bm.burst_id, bm.chal_id || "", bm.winner_id || "");
      toast(`Jumped to bookmark (${bm.label || bm.filename || bm.burst_id})`, 1600);
      return;
    }
    if (bm.view === "loupe") {
      if (bm.image_id) {
        let idx = state.items.findIndex((it) => it.image_id === bm.image_id);
        while (idx < 0 && state.items.length < state.total) {
          await loadImages(false);
          idx = state.items.findIndex((it) => it.image_id === bm.image_id);
        }
        if (idx >= 0) state.activeIndex = idx;
      } else if (bm.image_pos > 0) {
        await jumpToImagePositionInLoupe(bm.image_pos);
      }
      setView("loupe");
      toast(`Jumped to bookmark (${bm.label || bm.filename || "Loupe"})`, 1600);
      return;
    }
    if (bm.view === "bursts") {
      setView("bursts");
      if (bm.burst_pos > 0) {
        state.activeBurstIdx = Math.max(0, bm.burst_pos - 1);
      }
      toast(`Jumped to bookmark (${bm.label || bm.burst_id || "Bursts"})`, 1600);
      return;
    }
    setView("grid");
    if (bm.scroll_top > 0) {
      requestAnimationFrame(() => { $("gridView").scrollTop = bm.scroll_top; });
    }
    toast(`Jumped to bookmark (${bm.label || bm.filename || "Grid"})`, 1600);
  }

  async function deleteBookmark(bookmarkId) {
    try {
      await api(`/api/bookmarks/${bookmarkId}`, { method: "DELETE" });
      state.bookmarks = state.bookmarks.filter((b) => b.id !== bookmarkId);
      renderBookmarksList();
    } catch (err) {
      toast("Delete bookmark error: " + err.message, 3000);
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

  async function batchUpdate(payload, explicitIds = null) {
    const ids = Array.isArray(explicitIds) && explicitIds.length
      ? explicitIds
      : Array.from(state.selectedIds);
    if (!ids.length) return;
    // Optimistically update in-memory items & DOM immediately
    const idSet = new Set(ids);
    for (const arr of [state.items, state.wvcFrames]) {
      for (const it of arr) {
        if (!idSet.has(it.image_id)) continue;
        if (payload.rating !== undefined) it.effective_rating = payload.rating;
        if (payload.label !== undefined) it.effective_label = payload.label;
        if (payload.user_flag !== undefined) it.user_flag = payload.user_flag;
        refreshCardInDom(it);
      }
    }
    if (state.view === "loupe" && state.items[state.activeIndex]) {
      renderLoupeControls(state.items[state.activeIndex]);
      renderInspector(state.items[state.activeIndex]);
    } else if (state.view === "compare") {
      renderWvc();
    }
    try {
      const res = await api("/api/images/batch-metadata", {
        method: "POST",
        body: JSON.stringify({ image_ids: ids, ...payload }),
      });
      refreshSidebarCounts();
      const desc = payload.rating !== undefined
        ? `${payload.rating}★`
        : (payload.label !== undefined ? (payload.label || "No Color") : "Updated");
      toast(`${res.updated} images → ${desc}`, 1400);
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
        await refreshSidebarCounts();
        if (state.view === "compare" && state.wvcBurstId) {
          const curWinId = state.wvcFrames[state.wvcWinnerIdx]?.image_id || "";
          const curChalId = state.wvcFrames[state.wvcChalIdx]?.image_id || "";
          await loadWvcBurst(state.wvcBurstId, curChalId, curWinId);
        } else {
          await loadImages(true);
          if (state.view === "loupe") renderLoupe();
        }
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
    $("mapL2Rating").value = l2.rating != null ? String(l2.rating) : "";
    $("mapL2Label").value = l2.label ?? "Green";
    $("mapL1Enabled").checked = !!l1.enabled;
    $("mapL1Rating").value = l1.rating != null ? String(l1.rating) : "";
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
    loadVersionInfo().catch(() => {});
    renderUpdateUi();
  }

  async function loadVersionInfo() {
    try {
      state.versionInfo = await api("/api/system/version");
      renderUpdateUi();
    } catch (_) {
      // Fail silently
    }
  }

  function renderUpdateUi() {
    const vi = state.versionInfo;
    const ui = state.updateInfo;
    const curVer = (ui && ui.current_version) || (vi && vi.current_version) || "";
    if ($("setCurrentVersion")) {
      $("setCurrentVersion").textContent = curVer ? `v${curVer}` : "v—";
    }
    const relUrl = (ui && ui.html_url) || (vi && vi.releases_url);
    if ($("setReleasesLink") && relUrl) {
      $("setReleasesLink").href = relUrl;
    }
    const installBtn = $("btnInstallUpdate");
    if (installBtn) {
      const avail = !!(ui && ui.update_available && ui.download_url);
      installBtn.disabled = !avail || state.updatingApp;
      installBtn.textContent = state.updatingApp
        ? "⏳ Updating..."
        : (avail ? `⬇ Install v${ui.latest_version}` : "⬇ Install Update");
    }
    if ($("setChangelogSummary")) {
      $("setChangelogSummary").textContent = (ui && ui.update_available)
        ? `What's New in v${ui.latest_version}`
        : "What's New / Changelog";
    }
    if ($("setChangelogBody")) {
      const notes = (ui && ui.update_available && ui.changelog)
        ? ui.changelog
        : ((vi && vi.changelog) || "No changelog available.");
      $("setChangelogBody").textContent = notes;
    }
  }

  async function checkAppUpdates(manual = false) {
    const statusEl = $("updateStatusMsg");
    if (manual && statusEl) statusEl.textContent = "Checking GitHub Releases...";
    try {
      const res = await api("/api/system/update-check");
      state.updateInfo = res;
      renderUpdateUi();
      if (res.ok && res.update_available) {
        const banner = $("updateBanner");
        if (banner) {
          const label = res.release_name && res.release_name !== `v${res.latest_version}`
            ? `Update v${res.latest_version} available — ${res.release_name}`
            : `Update v${res.latest_version} available (current: v${res.current_version})`;
          if ($("updateBannerText")) $("updateBannerText").textContent = label;
          banner.classList.remove("hidden");
        }
        if (statusEl) statusEl.textContent = `New version v${res.latest_version} is available!`;
      } else if (manual && statusEl) {
        statusEl.textContent = res.ok
          ? (res.error || `You're up to date (v${res.current_version}).`)
          : (res.error || "Could not check for updates.");
      }
    } catch (err) {
      if (manual && statusEl) statusEl.textContent = `Update check failed: ${err.message}`;
    }
  }

  async function installAppUpdate() {
    if (state.updatingApp) return;
    const ui = state.updateInfo;
    const verLbl = ui?.latest_version ? `v${ui.latest_version}` : "the latest version";
    if (!confirm(`Install ${verLbl}?\n\nYour photos, studio_data/ database, cull_output/, models/, and .env are strictly protected. You will need to restart ImageCulling Studio after the update completes.`)) {
      return;
    }
    state.updatingApp = true;
    renderUpdateUi();
    const statusEl = $("updateStatusMsg");
    if (statusEl) statusEl.textContent = "Downloading and applying update...";
    const bannerBtn = $("btnBannerUpdateNow");
    if (bannerBtn) { bannerBtn.disabled = true; bannerBtn.textContent = "⏳ Updating..."; }
    try {
      const res = await api("/api/system/update-apply", {
        method: "POST",
        body: JSON.stringify({ download_url: ui?.download_url || null }),
      });
      if (res.ok && res.updated) {
        const msg = res.message || `Updated to v${res.new_version}. Please restart ImageCulling Studio.`;
        if (statusEl) statusEl.textContent = msg;
        toast(msg, 8000);
        if ($("updateBannerText")) $("updateBannerText").textContent = `✅ ${msg}`;
        if (bannerBtn) bannerBtn.classList.add("hidden");
        await loadVersionInfo();
      } else if (res.ok && !res.updated) {
        const msg = res.message || "Already up to date.";
        if (statusEl) statusEl.textContent = msg;
        toast(msg, 3500);
      } else {
        const errMsg = res.error || "Update failed.";
        if (statusEl) statusEl.textContent = errMsg;
        toast("Update failed: " + errMsg, 6000);
      }
    } catch (err) {
      if (statusEl) statusEl.textContent = "Update failed: " + err.message;
      toast("Update failed: " + err.message, 6000);
    } finally {
      state.updatingApp = false;
      if (bannerBtn && !bannerBtn.classList.contains("hidden")) {
        bannerBtn.disabled = false;
        bannerBtn.textContent = "Update Now";
      }
      renderUpdateUi();
    }
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
          rating: $("mapL2Rating").value === "" ? null : Number($("mapL2Rating").value),
          label: $("mapL2Label").value,
        },
        layer1_keep: {
          enabled: $("mapL1Enabled").checked,
          rating: $("mapL1Rating").value === "" ? null : Number($("mapL1Rating").value),
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
      zoom_preview_max_dim: Number($("setZoomDim")?.value) || 8192,
      zoom_preview_quality: Number($("setZoomQuality")?.value) || 94,
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

    // Welcome card Open Folder CTA
    if ($("btnWelcomeOpenDir")) {
      $("btnWelcomeOpenDir").onclick = () => $("btnOpenDir").click();
    }

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
    document.addEventListener("click", (e) => {
      $("cullDropdown").classList.add("hidden");
      if (!e.target.closest(".bookmark-menu-wrap")) {
        $("bookmarksDropdown")?.classList.add("hidden");
      }
    });
    if ($("btnBookmarksMenu")) {
      $("btnBookmarksMenu").onclick = (e) => {
        e.stopPropagation();
        $("bookmarksDropdown")?.classList.toggle("hidden");
      };
    }
    if ($("btnAddBookmark")) {
      $("btnAddBookmark").onclick = async () => {
        const inp = $("inputBookmarkLabel");
        const lbl = inp ? inp.value : "";
        await saveCurrentBookmark(lbl);
        if (inp) inp.value = "";
      };
    }
    if ($("inputBookmarkLabel")) {
      $("inputBookmarkLabel").addEventListener("keydown", async (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          const lbl = $("inputBookmarkLabel").value;
          await saveCurrentBookmark(lbl);
          $("inputBookmarkLabel").value = "";
        }
      });
    }
    if ($("btnWvcBookmark")) {
      $("btnWvcBookmark").onclick = () => saveCurrentBookmark("");
    }
    // Editable Jump-to-Position Counters (Burst X/Y & Image X/Y)
    if ($("compareBurstCounter")) {
      $("compareBurstCounter").onclick = () => {
        const bIdx = state.bursts.findIndex((b) => b.burst_id === state.wvcBurstId);
        const cur = bIdx >= 0 ? (bIdx + 1) : (state.wvcBurstPos || 1);
        const tot = (bIdx >= 0 && state.burstsTotal > 0) ? state.burstsTotal : (state.wvcTotalBursts || 0);
        openInlineJumpEditor($("compareBurstCounter"), cur, tot, "Burst", (n) => jumpToBurstPosition(n));
      };
    }
    if ($("compareImageCounter")) {
      $("compareImageCounter").onclick = () => {
        const chal = state.wvcFrames[state.wvcChalIdx];
        const filteredIdx = chal ? state.items.findIndex((it) => it.image_id === chal.image_id) : -1;
        const cur = filteredIdx >= 0 ? (filteredIdx + 1) : (state.wvcFirstImgPos ? (state.wvcFirstImgPos + state.wvcChalIdx) : 1);
        const tot = filteredIdx >= 0 ? (state.total || state.items.length) : (state.wvcTotalImgs || 0);
        openInlineJumpEditor($("compareImageCounter"), cur, tot, "Image", (n) => jumpToImagePositionInWvc(n));
      };
    }
    if ($("loupeCounter")) {
      $("loupeCounter").onclick = () => {
        const tot = state.total || state.items.length || 0;
        openInlineJumpEditor($("loupeCounter"), state.activeIndex + 1, tot, "Image", (n) => jumpToImagePositionInLoupe(n));
      };
    }
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
    if ($("btnBatchSmartCrop")) $("btnBatchSmartCrop").onclick = () => batchSmartCrop(false);
    if ($("btnBatchResetCrop")) $("btnBatchResetCrop").onclick = () => batchSmartCrop(true);
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
    if ($("btnToggleCropEdit")) $("btnToggleCropEdit").onclick = () => setCropEditMode(!state.cropEditMode);
    if ($("btnCropEditBtn")) $("btnCropEditBtn").onclick = () => setCropEditMode(!state.cropEditMode);
    if ($("btnCropAuto")) {
      $("btnCropAuto").onclick = () => {
        const img = state.items[state.activeIndex];
        if (img) runSmartCropAuto(img, false, !!img.crop);
      };
    }
    if ($("btnCropApply")) {
      $("btnCropApply").onclick = () => {
        const img = state.items[state.activeIndex];
        if (!img) return;
        if (!img.crop && !state.cropDraft) runSmartCropAuto(img, true, false);
        else saveCropDraft(img, true);
      };
    }
    if ($("btnCropReset")) {
      $("btnCropReset").onclick = () => {
        const img = state.items[state.activeIndex];
        if (img) resetCropForImage(img, "clear");
      };
    }
    if ($("btnCropLockRatio")) {
      $("btnCropLockRatio").onclick = () => {
        const img = state.items[state.activeIndex];
        if (!img) return;
        const d = ensureCropDraft(img);
        d.aspect_locked = !d.aspect_locked;
        if (d.aspect_locked) {
          const { ow, oh } = getOrigDimsForCrop(img, d);
          d.aspect_ratio = ((d.right - d.left) * ow) / Math.max(1, (d.bottom - d.top) * oh);
        }
        renderLoupeCropPanel(img);
        saveCropDraft(img, null);
      };
    }
    document.querySelectorAll("#cropRatioPresets .crp-btn").forEach((b) => {
      b.onclick = () => {
        const img = state.items[state.activeIndex];
        if (img) applyCropRatioPreset(img, b.dataset.ratio);
      };
    });
    const angleSlider = $("cropAngleSlider");
    const angleInput = $("cropAngleInput");
    const onAngleLive = (val) => {
      const img = state.items[state.activeIndex];
      if (!img) return;
      const d = ensureCropDraft(img);
      const clamped = Math.max(-15, Math.min(15, Number(val) || 0));
      d.angle = clamped;
      d.selected_horizon_idx = null;
      if (angleSlider) angleSlider.value = clamped.toFixed(1);
      if (angleInput) angleInput.value = clamped.toFixed(1);
      positionLoupeCropBox();
      drawLoupeOverlay(img);
    };
    const onAngleReset = () => {
      const img = state.items[state.activeIndex];
      if (!img) return;
      onAngleLive(0);
      saveCropDraft(img, null);
      toast("Angle reset to 0.0°", 1100);
    };
    if (angleSlider) {
      angleSlider.addEventListener("input", () => onAngleLive(angleSlider.value));
      angleSlider.addEventListener("change", () => {
        const img = state.items[state.activeIndex];
        if (img) saveCropDraft(img, null);
      });
      angleSlider.addEventListener("dblclick", onAngleReset);
    }
    if (angleInput) {
      angleInput.addEventListener("change", () => {
        onAngleLive(angleInput.value);
        const img = state.items[state.activeIndex];
        if (img) saveCropDraft(img, null);
      });
      angleInput.addEventListener("dblclick", onAngleReset);
    }
    if ($("cropAngleLabel")) $("cropAngleLabel").addEventListener("dblclick", onAngleReset);

    // Interactive Crop Box drag, handle resize, on-frame rotation & double-click reset
    const cropLayerEl = $("loupeCropLayer");
    const cropBoxEl = $("loupeCropBox");
    const startCropRotateDrag = (e, img) => {
      const d = ensureCropDraft(img);
      const rect = $("loupeImg").getBoundingClientRect();
      const cx = rect.left + ((d.left + d.right) * 0.5) * rect.width;
      const cy = rect.top + ((d.top + d.bottom) * 0.5) * rect.height;
      const startPointerAngle = Math.atan2(e.clientY - cy, e.clientX - cx) * (180 / Math.PI);
      state.cropDrag = {
        handle: "rotate",
        cx,
        cy,
        startPointerAngle,
        startAngle: Number(d.angle || 0),
        moved: false,
      };
      cropLayerEl?.classList.add("rotating");
    };
    if (cropLayerEl) {
      // Lightroom-style: dragging anywhere outside the crop box rotates the frame directly!
      cropLayerEl.addEventListener("mousedown", (e) => {
        if (e.button !== 0 || e.target.closest("#loupeCropBox")) return;
        const img = state.items[state.activeIndex];
        if (!img || !state.cropEditMode) return;
        e.preventDefault();
        e.stopPropagation();
        startCropRotateDrag(e, img);
      });
    }
    if (cropBoxEl) {
      cropBoxEl.addEventListener("dblclick", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.target.closest(".crop-rotate-handle")) {
          onAngleReset();
          return;
        }
        const img = state.items[state.activeIndex];
        if (img) resetCropForImage(img, "clear");
      });
      cropBoxEl.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        const img = state.items[state.activeIndex];
        if (!img) return;
        e.preventDefault();
        e.stopPropagation();
        const rotHandle = e.target.closest(".crop-rotate-handle");
        if (rotHandle) {
          startCropRotateDrag(e, img);
          return;
        }
        const d = ensureCropDraft(img);
        const handle = e.target.closest(".crop-handle")?.dataset.handle || "move";
        const rect = $("loupeImg").getBoundingClientRect();
        state.cropDrag = {
          handle,
          startX: e.clientX,
          startY: e.clientY,
          imgW: Math.max(1, rect.width),
          imgH: Math.max(1, rect.height),
          startL: d.left,
          startT: d.top,
          startR: d.right,
          startB: d.bottom,
          moved: false,
        };
      });
      window.addEventListener("mousemove", (e) => {
        if (!state.cropDrag) return;
        const img = state.items[state.activeIndex];
        if (!img) return;
        const d = ensureCropDraft(img);
        const dr = state.cropDrag;
        if (dr.handle === "rotate") {
          const curPointerAngle = Math.atan2(e.clientY - dr.cy, e.clientX - dr.cx) * (180 / Math.PI);
          let delta = curPointerAngle - dr.startPointerAngle;
          if (delta > 180) delta -= 360;
          if (delta < -180) delta += 360;
          if (Math.abs(delta) > 0.05) dr.moved = true;
          const nextAngle = Math.max(-15, Math.min(15, Math.round((dr.startAngle + delta) * 10) / 10));
          onAngleLive(nextAngle);
          return;
        }
        const dx = (e.clientX - dr.startX) / dr.imgW;
        const dy = (e.clientY - dr.startY) / dr.imgH;
        if (Math.abs(dx) > 0.001 || Math.abs(dy) > 0.001) dr.moved = true;

        const minFrac = 0.06;
        let l = dr.startL, t = dr.startT, r = dr.startR, b = dr.startB;
        const h = dr.handle;

        if (h === "move") {
          const bw = r - l, bh = b - t;
          l = Math.max(0, Math.min(1 - bw, dr.startL + dx));
          t = Math.max(0, Math.min(1 - bh, dr.startT + dy));
          r = l + bw;
          b = t + bh;
        } else {
          if (h.includes("w")) l = Math.max(0, Math.min(r - minFrac, dr.startL + dx));
          if (h.includes("e")) r = Math.min(1, Math.max(l + minFrac, dr.startR + dx));
          if (h.includes("n")) t = Math.max(0, Math.min(b - minFrac, dr.startT + dy));
          if (h.includes("s")) b = Math.min(1, Math.max(t + minFrac, dr.startB + dy));

          if (d.aspect_locked && d.aspect_ratio > 0) {
            const { ow, oh } = getOrigDimsForCrop(img, d);
            const normRatio = d.aspect_ratio / (ow / oh);
            if (h === "n" || h === "s") {
              const nh = b - t;
              const nw = Math.min(1, nh * normRatio);
              const cx = (dr.startL + dr.startR) * 0.5;
              l = Math.max(0, Math.min(1 - nw, cx - nw / 2));
              r = l + nw;
            } else {
              const nw = r - l;
              const nh = nw / normRatio;
              if (h.includes("n")) {
                t = Math.max(0, b - nh);
                const actualH = b - t;
                const actualW = actualH * normRatio;
                if (h.includes("w")) l = r - actualW; else r = l + actualW;
              } else {
                b = Math.min(1, t + nh);
                const actualH = b - t;
                const actualW = actualH * normRatio;
                if (h.includes("w")) l = r - actualW; else r = l + actualW;
              }
            }
          }
        }
        d.left = Math.max(0, Math.min(1, l));
        d.top = Math.max(0, Math.min(1, t));
        d.right = Math.max(d.left + minFrac, Math.min(1, r));
        d.bottom = Math.max(d.top + minFrac, Math.min(1, b));
        positionLoupeCropBox();
        updateCropSizeReadout(img, d);
      });
      window.addEventListener("mouseup", () => {
        if (!state.cropDrag) return;
        const moved = state.cropDrag.moved;
        state.cropDrag = null;
        cropLayerEl?.classList.remove("rotating");
        const img = state.items[state.activeIndex];
        if (moved && img) saveCropDraft(img, null);
      });
    }
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
      if (state.cropEditMode && e.target.closest("#loupeCropLayer")) return;
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
    if ($("btnLoupeSplitBurst")) {
      $("btnLoupeSplitBurst").onclick = () => {
        const cur = state.items[state.activeIndex];
        if (cur) splitBurstAt(cur.image_id);
      };
    }
    if ($("btnLoupeUnsplitBurst")) {
      $("btnLoupeUnsplitBurst").onclick = () => {
        const cur = state.items[state.activeIndex];
        if (cur) unsplitBurstAt(cur.image_id);
      };
    }
    $("btnFlagPick").onclick = () => setUserFlag(state.items[state.activeIndex], 1);
    $("btnFlagReject").onclick = () => setUserFlag(state.items[state.activeIndex], -1);

    document.querySelectorAll(".fs-tab").forEach((b) => {
      b.onclick = () => {
        state.filmstripMode = b.dataset.fsMode;
        document.querySelectorAll(".fs-tab").forEach((x) => x.classList.toggle("active", x === b));
        renderFilmstrip();
      };
    });

    // #1 Burst-by-Burst view controls
    document.querySelectorAll("#burstFilterBar .rop-btn").forEach((b) => {
      b.onclick = () => {
        state.burstFilter = b.dataset.bf;
        document.querySelectorAll("#burstFilterBar .rop-btn").forEach((x) => x.classList.toggle("active", x === b));
        loadBursts(true);
      };
    });
    if ($("burstSortSelect")) {
      $("burstSortSelect").onchange = (e) => {
        state.burstSort = e.target.value;
        loadBursts(true);
      };
    }
    if ($("btnLoadMoreBursts")) $("btnLoadMoreBursts").onclick = () => loadBursts(false);
    if ($("burstsView")) {
      $("burstsView").addEventListener("scroll", () => {
        const el = $("burstsView");
        if (state.view === "bursts" && el.scrollTop + el.clientHeight >= el.scrollHeight - 1200 && state.bursts.length < state.burstsTotal) {
          loadBursts(false);
        }
      }, { passive: true });
    }

    // #2 Winner vs Challenger buttons
    if ($("btnWvcPrevBurst")) $("btnWvcPrevBurst").onclick = () => wvcStepBurst(-1);
    if ($("btnWvcNextBurst")) $("btnWvcNextBurst").onclick = () => wvcStepBurst(1);
    if ($("btnWvcChalPrev")) $("btnWvcChalPrev").onclick = () => stepWvcChallenger(-1);
    if ($("btnWvcChalNext")) $("btnWvcChalNext").onclick = () => stepWvcChallenger(1);
    if ($("btnWvcZoomToggle")) {
      $("btnWvcZoomToggle").onclick = () => {
        state.wvcZoomed = !state.wvcZoomed;
        state.wvcPanX = 0;
        state.wvcPanY = 0;
        if (state.wvcZoomed) state.wvcScale = 3.2;
        renderWvc();
      };
    }
    // Synced mouse-drag panning & mouse-wheel zoom across BOTH Winner & Challenger panes
    for (const stageId of ["wvcLeftStage", "wvcRightStage"]) {
      const stEl = $(stageId);
      if (!stEl) continue;
      stEl.addEventListener("mousedown", (e) => {
        if (e.button !== 0 || e.target.closest("button")) return;
        e.preventDefault();
        state.wvcDragging = true;
        state.wvcDragMoved = false;
        state.wvcDragStartX = e.clientX;
        state.wvcDragStartY = e.clientY;
        state.wvcDragStartPanX = state.wvcPanX;
        state.wvcDragStartPanY = state.wvcPanY;
        if (state.wvcZoomed) {
          $("wvcLeftWrap")?.classList.add("dragging");
          $("wvcRightWrap")?.classList.add("dragging");
        }
      });
      stEl.addEventListener("wheel", (e) => {
        if (state.view !== "compare") return;
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.22 : 1 / 1.22;
        const wasZoomed = state.wvcZoomed;
        const nextScale = Math.max(1.0, Math.min(8.0, (wasZoomed ? state.wvcScale : 1.0) * factor));
        if (nextScale <= 1.05) {
          state.wvcZoomed = false;
          state.wvcScale = 3.2;
          state.wvcPanX = 0;
          state.wvcPanY = 0;
          renderWvc();
        } else {
          state.wvcScale = nextScale;
          if (!wasZoomed) {
            state.wvcZoomed = true;
            renderWvc();
          } else {
            applyWvcZoomTransforms();
          }
        }
      }, { passive: false });
    }
    window.addEventListener("mousemove", (e) => {
      if (!state.wvcDragging) return;
      const dx = e.clientX - state.wvcDragStartX;
      const dy = e.clientY - state.wvcDragStartY;
      if (Math.abs(dx) > 4 || Math.abs(dy) > 4) state.wvcDragMoved = true;
      if (state.wvcZoomed && state.wvcDragMoved) {
        const refImg = $("wvcWinImg") || $("wvcChalImg");
        const bw = (refImg && refImg.offsetWidth) || 500;
        const bh = (refImg && refImg.offsetHeight) || 350;
        const sc = state.wvcScale || 3.2;
        state.wvcPanX = Math.max(-0.5, Math.min(0.5, state.wvcDragStartPanX - dx / (bw * sc)));
        state.wvcPanY = Math.max(-0.5, Math.min(0.5, state.wvcDragStartPanY - dy / (bh * sc)));
        applyWvcZoomTransforms();
      }
    });
    window.addEventListener("mouseup", () => {
      if (!state.wvcDragging) return;
      state.wvcDragging = false;
      $("wvcLeftWrap")?.classList.remove("dragging");
      $("wvcRightWrap")?.classList.remove("dragging");
      if (!state.wvcDragMoved && state.view === "compare") {
        $("btnWvcZoomToggle").click();
      }
    });
    if ($("btnWvcSwap")) $("btnWvcSwap").onclick = () => wvcSwapWinner();
    if ($("btnWvcKeepBoth")) $("btnWvcKeepBoth").onclick = () => wvcKeepBoth();
    if ($("btnWvcToggleAf")) {
      $("btnWvcToggleAf").onclick = () => {
        state.showAfPoints = !state.showAfPoints;
        $("btnToggleAf")?.classList.toggle("active", state.showAfPoints);
        drawWvcOverlays();
      };
    }
    if ($("btnWvcToggleHighlights")) {
      $("btnWvcToggleHighlights").onclick = () => {
        state.showHighlights = !state.showHighlights;
        $("btnToggleHighlights")?.classList.toggle("active", state.showHighlights);
        drawWvcOverlays();
      };
    }
    if ($("btnWvcPinChallenger")) {
      $("btnWvcPinChallenger").onclick = () => {
        state.wvcWinnerIdx = state.wvcChalIdx;
        renderWvc();
        toast("Pinned Challenger as Winner on Left");
      };
    }
    if ($("btnWvcSplitHere")) {
      $("btnWvcSplitHere").onclick = () => {
        const chal = state.wvcFrames[state.wvcChalIdx];
        if (chal) splitBurstAt(chal.image_id);
      };
    }
    if ($("btnWvcUnsplitHere")) {
      $("btnWvcUnsplitHere").onclick = () => {
        const chal = state.wvcFrames[state.wvcChalIdx];
        if (chal) unsplitBurstAt(chal.image_id);
      };
    }
    $("btnCompareExit").onclick = () => setView(state.wvcReturnView || "grid");

    // Topbar utility buttons
    $("btnUndo").onclick = () => undoLast();
    $("btnLogs").onclick = () => openLogsModal("events");
    $("btnLearning").onclick = () => openLearningModal();
    $("btnSettings").onclick = () => openSettingsModal();
    $("btnShortcuts").onclick = () => $("modalShortcuts").classList.remove("hidden");
    document.querySelectorAll("#helpModalTabs .m-tab").forEach((b) => {
      b.onclick = () => {
        const tab = b.dataset.helpTab;
        document.querySelectorAll("#helpModalTabs .m-tab").forEach((x) => x.classList.toggle("active", x === b));
        $("helpTabGuide")?.classList.toggle("hidden", tab !== "guide");
        $("helpTabShortcuts")?.classList.toggle("hidden", tab !== "shortcuts");
        $("helpTabAi")?.classList.toggle("hidden", tab !== "ai");
      };
    });

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

    // Update banner & Settings "Version & Updates" buttons
    if ($("btnBannerDismiss")) $("btnBannerDismiss").onclick = () => $("updateBanner")?.classList.add("hidden");
    if ($("btnBannerWhatsNew")) {
      $("btnBannerWhatsNew").onclick = async () => {
        await openSettingsModal();
        const det = $("setChangelogDetails");
        if (det) {
          det.open = true;
          det.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      };
    }
    if ($("btnBannerUpdateNow")) $("btnBannerUpdateNow").onclick = () => installAppUpdate();
    if ($("btnCheckUpdates")) $("btnCheckUpdates").onclick = () => checkAppUpdates(true);
    if ($("btnInstallUpdate")) $("btnInstallUpdate").onclick = () => installAppUpdate();

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

    function getPressedDigit(e) {
      if (e.code && /^Digit[0-9]$/.test(e.code)) return Number(e.code.slice(5));
      if (e.code && /^Numpad[0-9]$/.test(e.code)) return Number(e.code.slice(6));
      if (typeof e.key === "string" && /^[0-9]$/.test(e.key)) return Number(e.key);
      const kc = e.keyCode || e.which || 0;
      if (kc >= 48 && kc <= 57) return kc - 48;
      if (kc >= 96 && kc <= 105) return kc - 96;
      return -1;
    }

    // Global Keyboard Shortcuts (Lightroom / Photo Mechanic style)
    window.addEventListener("keydown", (e) => {
      const target = e.target;
      const tag = (target?.tagName || "").toLowerCase();
      const inputType = (target?.type || "").toLowerCase();
      const pressedDigit = getPressedDigit(e);

      // In Loupe (Crop Edit mode, or when a crop suggestion/draft exists), Enter ALWAYS applies the crop and does nothing else!
      if (e.key === "Enter" && state.view === "loupe" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const activeModal = document.querySelector(".modal-backdrop:not(.hidden)");
        const curImg = state.items[state.activeIndex];
        if (!activeModal && curImg && (state.cropEditMode || curImg.crop || state.cropDraft) && !target?.closest(".jump-chip") && target?.id !== "searchInput" && target?.id !== "inputBookmarkLabel") {
          e.preventDefault();
          e.stopPropagation();
          if (target && typeof target.blur === "function") target.blur();
          if (target?.id === "cropAngleInput") onAngleLive(target.value);
          saveCropDraft(curImg, true);
          return;
        }
      }

      // Only block shortcuts when typing inside text/search/number inputs or textareas (NOT checkboxes or buttons!)
      if (tag === "textarea" || (tag === "input" && !["checkbox", "radio", "range", "button"].includes(inputType))) return;
      if (tag === "select" && e.key !== "Escape") {
        if (pressedDigit >= 0 && !e.ctrlKey && !e.metaKey && !e.altKey) {
          target.blur();
        } else {
          return;
        }
      }

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
        const openModals = Array.from(document.querySelectorAll(".modal-backdrop")).filter((m) => !m.classList.contains("hidden"));
        if (openModals.length) {
          openModals.forEach((m) => m.classList.add("hidden"));
          return;
        }
        if (state.view === "compare" && state.wvcSelectedIds.size > 0) {
          state.wvcSelectedIds.clear();
          renderWvc();
          return;
        }
        if (state.selectedIds.size > 0) {
          state.selectedIds.clear();
          updateBatchBar();
          syncGridSelectionDom();
          return;
        }
        if (state.view === "compare") setView(state.wvcReturnView || "grid");
        else if (state.view !== "grid") setView("grid");
        return;
      }

      // Use physical e.code so all shortcuts work identically in lowercase, uppercase, CapsLock, and Hebrew layout!
      const code = e.code;
      const k = e.key.toLowerCase();
      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        if (code === "KeyG" || k === "g") { setView("grid"); return; }
        if (code === "KeyM" || k === "m") { setView("bursts"); return; }
        if (code === "KeyE" || k === "e") { setView("loupe"); return; }
        if (code === "KeyC" || k === "c") { setView("compare"); return; }
        if (e.key === "?") { $("modalShortcuts").classList.toggle("hidden"); return; }
      }

      // Burst-by-Burst keyboard navigation (↑/↓ jump bursts, Space expand, Enter Winner vs Challenger)
      if (state.view === "bursts") {
        if (e.key === "ArrowDown") { e.preventDefault(); setActiveBurstIdx(state.activeBurstIdx + 1); return; }
        if (e.key === "ArrowUp") { e.preventDefault(); setActiveBurstIdx(state.activeBurstIdx - 1); return; }
        if (code === "Space" || e.key === " ") {
          e.preventDefault();
          const b = state.bursts[state.activeBurstIdx];
          const row = document.querySelector(`.burst-row[data-burst-idx="${state.activeBurstIdx}"]`);
          if (b && row) toggleExpandBurst(b, row);
          return;
        }
        if (e.key === "Enter") {
          e.preventDefault();
          const b = state.bursts[state.activeBurstIdx];
          if (b) openWinnerVsChallengerForBurst(b.burst_id);
          return;
        }
      }

      // Winner vs Challenger keyboard shortcuts (←/→ challengers, ↑/↓ bursts, Z zoom, S swap, K keep both, 0-9 rate/color Challenger)
      if (state.view === "compare") {
        const chal = state.wvcFrames[state.wvcChalIdx];
        if (e.key === "ArrowRight") { e.preventDefault(); stepWvcChallenger(1); return; }
        if (e.key === "ArrowLeft") { e.preventDefault(); stepWvcChallenger(-1); return; }
        if (e.key === "ArrowDown") { e.preventDefault(); wvcStepBurst(1); return; }
        if (e.key === "ArrowUp") { e.preventDefault(); wvcStepBurst(-1); return; }
        if (code === "KeyZ" || k === "z" || code === "Space" || e.key === " ") {
          e.preventDefault();
          $("btnWvcZoomToggle").click();
          return;
        }
        if (code === "KeyS" || k === "s") { e.preventDefault(); wvcSwapWinner(); return; }
        if (code === "KeyK" || k === "k") { e.preventDefault(); wvcKeepBoth(); return; }
        if (code === "KeyF" || k === "f") { e.preventDefault(); $("btnWvcToggleAf")?.click(); return; }
        if (code === "KeyH" || k === "h") { e.preventDefault(); $("btnWvcToggleHighlights")?.click(); return; }
        if (chal && !e.ctrlKey && !e.metaKey && !e.altKey) {
          const wvcTargets = getTargetSelectionIds(chal, true);
          if (pressedDigit >= 0 && pressedDigit <= 5) {
            e.preventDefault();
            if (wvcTargets.length > 1) {
              batchUpdate({ rating: pressedDigit }, wvcTargets);
            } else {
              toast(`${chal.filename} → ${pressedDigit}★`, 1100);
              setRating(chal, pressedDigit, false).then(() => renderWvc());
            }
            return;
          }
          const colorByDigit = { 6: "Red", 7: "Yellow", 8: "Green", 9: "Blue" };
          const cTarget = colorByDigit[pressedDigit] || ((code === "KeyP" || k === "p") ? "Purple" : "");
          if (cTarget) {
            e.preventDefault();
            // Always toggle based on the MAIN selected image (chal)
            const nextLbl = chal.effective_label === cTarget ? "" : cTarget;
            if (wvcTargets.length > 1) {
              batchUpdate({ label: nextLbl }, wvcTargets);
            } else {
              toast(`${chal.filename} → ${nextLbl || "No Color"}`, 1100);
              setColorLabel(chal, nextLbl, false).then(() => renderWvc());
            }
            return;
          }
        }
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
      } else if (code === "KeyR" || k === "r") {
        if (state.view === "loupe" && !e.ctrlKey && !e.metaKey && !e.altKey) {
          e.preventDefault();
          setCropEditMode(!state.cropEditMode);
        }
      } else if (code === "KeyS" || k === "s") {
        if (state.view === "loupe" && !e.ctrlKey && !e.metaKey && !e.altKey) {
          e.preventDefault();
          if (activeImg.burst_size > 1 && activeImg.burst_index > 1) {
            splitBurstAt(activeImg.image_id);
          } else if (state.manualSplits.has(activeImg.image_id)) {
            unsplitBurstAt(activeImg.image_id);
          } else {
            toast("Already frame 1 of this burst", 1400);
          }
        }
      } else if (pressedDigit >= 0 && pressedDigit <= 5) {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const targets = getTargetSelectionIds(activeImg, false);
        if (targets.length > 1) batchUpdate({ rating: pressedDigit }, targets);
        else { toast(`${activeImg.filename} → ${pressedDigit}★`, 1100); setRating(activeImg, pressedDigit); }
      } else if (pressedDigit === 6 || pressedDigit === 7 || pressedDigit === 8 || pressedDigit === 9 || code === "KeyP" || k === "p") {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        const colorByDigit = { 6: "Red", 7: "Yellow", 8: "Green", 9: "Blue" };
        const cTarget = colorByDigit[pressedDigit] || "Purple";
        // Always decide toggle direction from the MAIN selected image (activeImg)
        const nextLbl = activeImg.effective_label === cTarget ? "" : cTarget;
        const targets = getTargetSelectionIds(activeImg, false);
        if (targets.length > 1) batchUpdate({ label: nextLbl }, targets);
        else { toast(`${activeImg.filename} → ${nextLbl || "No Color"}`, 1100); setColorLabel(activeImg, nextLbl); }
      } else if (code === "KeyX" || k === "x") {
        setUserFlag(activeImg, -1);
      } else if (code === "KeyU" || k === "u") {
        setUserFlag(activeImg, 0);
      }
    }, true);
  }

  window.addEventListener("DOMContentLoaded", init);
})();
