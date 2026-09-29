"use strict";

(() => {
  const VERSION = "abv-review-v2";
  const data = JSON.parse(document.getElementById("review-data").textContent);
  if (data.version !== VERSION || !Array.isArray(data.fields) || !Array.isArray(data.bundles)) {
    throw new Error("审核包数据版本或字段不匹配");
  }
  const bundles = data.bundles;
  const tasks = bundles.flatMap(bundle => bundle.tasks.map(task => ({task, bundle})));
  const taskById = new Map(tasks.map(item => [item.task.task_id, item]));
  const bundleById = new Map(bundles.map(bundle => [bundle.id, bundle]));
  const packageKey = `aic-abv-review-v2:${bundles.length}:${bundles[0]?.id || "empty"}:${bundles.at(-1)?.id || "empty"}`;
  const $ = id => document.getElementById(id);
  const sourceSnapshot = Object.fromEntries(tasks.map(({task, bundle}) => [task.task_id, {
    task_id: task.task_id, bundle_id: task.bundle_id || bundle.id, category: task.category || bundle.category,
    split: task.split || bundle.split, scene_id: task.scene_id || bundle.scene,
    source_id: task.source_id, proposed_query: task.proposed_query, original_query: task.original_query,
    bbox: task.bbox, target_object_id: task.target_object_id, images: task.images,
  }]));
  const state = {version: VERSION, packageKey, reviewer: "", currentBundleId: bundles[0]?.id || null,
    currentTaskId: bundles[0]?.tasks[0]?.task_id || null, records: {}, exposure: {}, blindRecords: {}, sourceSnapshot};
  const filters = {search: "", category: "", status: ""};
  const CATEGORY = {
    depth_relation: "深度关系", diag_depth: "深度诊断", ir_complement: "红外互补",
    diag_ir: "红外诊断", reliability: "RGB 稳健性", diag_aux: "辅助诊断",
    robo_competition: "同类目标竞争", reserve_ir: "红外候补",
  };
  let showBox = true;

  function node(tag, className = "", content) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (content !== undefined) element.textContent = String(content);
    return element;
  }
  function feedback(message, error = false) {
    const box = $("feedback");
    box.textContent = message;
    box.hidden = !message;
    box.classList.toggle("error", error);
  }
  function save() {
    try {
      localStorage.setItem(packageKey, JSON.stringify(state));
      $("save-status").textContent = `已自动保存 · ${new Date().toLocaleTimeString()}`;
    } catch (error) {
      $("save-status").textContent = "自动保存失败";
      feedback(`本地保存失败：${error.message}`, true);
    }
  }
  function restore() {
    try {
      const raw = localStorage.getItem(packageKey);
      if (!raw) {
        const prior = data.revision?.source_package_key;
        const previous = prior && localStorage.getItem(prior);
        if (previous) {
          const count = mergePrevious(JSON.parse(previous));
          save(); feedback(`已沿用 ${count} 条未改题目的旧记录；新版深度题从待看开始。`);
        } else $("save-status").textContent = "自动草稿已就绪";
        return;
      }
      const saved = JSON.parse(raw);
      if (saved.version !== VERSION || saved.packageKey !== packageKey) throw new Error("记录版本或审核包不匹配");
      Object.assign(state, saved);
      state.records ||= {};
      state.exposure ||= {};
      state.blindRecords ||= {};
      const changed = tasks.filter(({task}) => saved.sourceSnapshot?.[task.task_id] &&
        JSON.stringify(saved.sourceSnapshot[task.task_id]) !== JSON.stringify(sourceSnapshot[task.task_id]));
      for (const {task} of changed) if (state.records[task.task_id]?.decision) state.records[task.task_id].decision = "";
      state.sourceSnapshot = sourceSnapshot;
      if (!bundleById.has(state.currentBundleId)) state.currentBundleId = bundles[0]?.id || null;
      if (!bundleById.get(state.currentBundleId)?.tasks.some(task => task.task_id === state.currentTaskId)) {
        state.currentTaskId = bundleById.get(state.currentBundleId)?.tasks[0]?.task_id || null;
      }
      $("save-status").textContent = "已恢复本地草稿";
      if (changed.length) feedback(`${changed.length} 题来源已变化，原决定已撤回，请重新查看。`, true);
    } catch (error) {
      $("save-status").textContent = "草稿读取失败";
      feedback(`草稿读取失败：${error.message}`, true);
    }
  }
  function record(id) { return state.records[id] || (state.records[id] = {}); }
  function mergePrevious(previous) {
    const unchanged = new Set(data.revision.unchanged_task_ids);
    let count = 0;
    for (const [id, value] of Object.entries(previous.records || {})) {
      if (!Object.keys(value).length) continue;
      if (unchanged.has(id) && JSON.stringify(previous.sourceSnapshot?.[id]) === JSON.stringify(sourceSnapshot[id])) {
        if (!Object.keys(state.records[id] || {}).length) { state.records[id] = value; count++; }
      } else if (data.revision.retired_depth_task_ids.includes(id)) {
        state.previousDepthReviews ||= {};
        state.previousDepthReviews[id] = {record: value, source: previous.sourceSnapshot?.[id]};
      }
    }
    return count;
  }
  function currentBundle() { return bundleById.get(state.currentBundleId); }
  function currentTask() { return currentBundle()?.tasks.find(task => task.task_id === state.currentTaskId); }
  function actualQuery(task) { return (record(task.task_id).approved_query || task.proposed_query || "").trim(); }
  function translation(query) { return data.translations?.[(query || "").trim()] || ""; }
  function category(bundle) { return bundle.category || bundle.tasks[0]?.category || ""; }
  function categoryName(bundle) { return CATEGORY[category(bundle)] || category(bundle); }
  function decision(id) { return record(id).decision || ""; }
  function bundleStatus(bundle) {
    const values = bundle.tasks.map(task => decision(task.task_id));
    if (values.some(value => !value)) return "pending";
    return values.every(value => value === "approve") ? "approve" : "reject";
  }
  function matchesBundle(bundle, withStatus = true) {
    if (filters.category && category(bundle) !== filters.category) return false;
    if (withStatus && filters.status && bundleStatus(bundle) !== filters.status) return false;
    const query = filters.search.trim().toLocaleLowerCase();
    return !query || bundle.id.toLocaleLowerCase().includes(query) || bundle.tasks.some(task => [
      task.task_id, task.proposed_query, task.original_query, actualQuery(task),
      translation(task.proposed_query), translation(task.original_query), translation(actualQuery(task)),
    ].some(text => (text || "").toLocaleLowerCase().includes(query)));
  }
  function renderProgress() {
    const approved = tasks.filter(({task}) => decision(task.task_id) === "approve").length;
    const rejected = tasks.filter(({task}) => decision(task.task_id) === "reject").length;
    $("progress-text").textContent = `${approved + rejected}/${tasks.length} 已看 · 保留 ${approved} · 剔除 ${rejected}`;
    $("progress-fill").style.width = `${tasks.length ? (approved + rejected) / tasks.length * 100 : 0}%`;
  }
  function renderList() {
    const visible = bundles.filter(bundle => matchesBundle(bundle));
    $("list-count").textContent = `${visible.length}/${bundles.length} 组`;
    const list = $("bundle-list"); list.replaceChildren();
    for (const bundle of visible) {
      const button = node("button", "bundle-item"); button.type = "button";
      button.classList.toggle("active", bundle.id === state.currentBundleId);
      button.setAttribute("aria-current", bundle.id === state.currentBundleId ? "true" : "false");
      const index = bundles.indexOf(bundle) + 1;
      const pending = bundle.tasks.filter(task => !decision(task.task_id)).length;
      const status = bundleStatus(bundle);
      const top = node("span", "bundle-top");
      top.append(node("strong", "", `第 ${index} 组 · ${categoryName(bundle)}`),
        node("span", `state-chip ${status}`, pending ? `${pending} 待看` : status === "approve" ? "已保留" : "含剔除"));
      const firstQuery = bundle.tasks[0]?.proposed_query || "";
      button.append(top, node("span", "bundle-sub", translation(firstQuery) || firstQuery));
      button.addEventListener("click", () => selectBundle(bundle));
      list.append(button);
    }
    if (!visible.length) list.append(node("p", "bundle-sub", "当前筛选没有题组"));
  }
  function selectTask(taskId) {
    const item = taskById.get(taskId);
    state.currentBundleId = item.bundle.id;
    state.currentTaskId = taskId;
    render(); save();
    $("main-content").scrollTop = 0;
  }
  function selectBundle(bundle) {
    const first = bundle.tasks.find(task => !decision(task.task_id)) || bundle.tasks[0];
    selectTask(first.task_id);
  }
  function navigate(direction) {
    const index = tasks.findIndex(({task}) => task.task_id === state.currentTaskId);
    const target = tasks[index + direction];
    if (target) selectTask(target.task.task_id);
    else feedback("已到题目边界");
  }
  function nextPending(fromId, includeSameBundle = true) {
    const currentIndex = tasks.findIndex(({task}) => task.task_id === fromId);
    const bundle = currentBundle();
    if (includeSameBundle && matchesBundle(bundle, false)) {
      const sibling = bundle.tasks.find(task => task.task_id !== fromId && !decision(task.task_id));
      if (sibling) return sibling.task_id;
    }
    const ordered = [...tasks.slice(currentIndex + 1), ...tasks.slice(0, currentIndex)];
    return ordered.find(({task, bundle: itemBundle}) => !decision(task.task_id) && matchesBundle(itemBundle, false))?.task.task_id || null;
  }
  function nextIncomplete() {
    const next = nextPending(state.currentTaskId);
    if (next) selectTask(next);
    else feedback("当前范围没有待看题");
  }
  function renderQuery(task) {
    const query = actualQuery(task);
    const chinese = translation(query);
    $("proposed-query").textContent = query || "英文 Query 缺失";
    $("proposed-query-zh").textContent = chinese || (query !== (task.proposed_query || "").trim()
      ? "英文已修订，中文译文待更新" : "中文译文待补充");
    $("proposed-query-zh").classList.toggle("translation-pending", !chinese);
  }
  function openImage(src, title) {
    $("dialog-title").textContent = title;
    const image = $("dialog-image");
    image.src = src; image.alt = title; image.className = "fit"; image.style.width = "";
    $("image-dialog").showModal();
  }
  function mediaFrame(src, label) {
    const frame = node("figure", "media-frame");
    frame.append(node("figcaption", "", label));
    if (!src) { frame.append(node("div", "missing-image", `${label}暂缺`)); return frame; }
    const button = node("button", "image-button"); button.type = "button";
    button.setAttribute("aria-label", `放大查看${label}`);
    const image = node("img"); image.src = src; image.alt = label; image.loading = "lazy";
    image.addEventListener("error", () => { button.replaceWith(node("div", "missing-image", `${label}加载失败`)); });
    button.append(image); button.addEventListener("click", () => openImage(src, label));
    frame.append(button); return frame;
  }
  function renderMedia(bundle, task) {
    const area = $("media-area"); area.replaceChildren();
    area.append(mediaFrame(showBox ? task.answer_image : bundle.views?.rgb, showBox ? "RGB · 目标框" : "RGB · 原图"));
    const auxiliary = [["infrared", "红外"], ["depth", "深度"]].filter(([key]) => bundle.views?.[key]);
    for (const [key, label] of auxiliary) area.append(mediaFrame(bundle.views[key], label));
    area.classList.toggle("three", auxiliary.length === 2);
  }
  function renderDecision(task) {
    const value = decision(task.task_id);
    const stateLabel = {approve: "已保留", reject: "已剔除"}[value] || "待看";
    $("decision-state").textContent = stateLabel;
    $("clear-decision").hidden = !value;
    $("approve").setAttribute("aria-pressed", value === "approve" ? "true" : "false");
    $("reject").setAttribute("aria-pressed", value === "reject" ? "true" : "false");
  }
  function render() {
    renderProgress(); renderList();
    const bundle = currentBundle(); const task = currentTask();
    if (!bundle || !task) { $("bundle-title").textContent = "没有题目"; return; }
    const position = tasks.findIndex(item => item.task.task_id === task.task_id) + 1;
    $("bundle-position").textContent = `第 ${position} / ${tasks.length} 题`;
    $("bundle-title").textContent = categoryName(bundle);
    $("bundle-meta").textContent = `${bundle.split === "diagnostic" ? "诊断" : "训练"} · 第 ${bundles.indexOf(bundle) + 1} 组 · 本组 ${bundle.tasks.length} 题`;
    $("previous-task").disabled = position <= 1;
    $("next-task").disabled = position >= tasks.length;
    const tabs = $("task-tabs"); tabs.replaceChildren();
    bundle.tasks.forEach((item, index) => {
      const button = node("button", "task-tab"); button.type = "button";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", item.task_id === task.task_id ? "true" : "false");
      button.append(node("strong", "", `本组第 ${index + 1} 题 · ${{approve: "已保留", reject: "已剔除"}[decision(item.task_id)] || "待看"}`),
        node("small", "", translation(actualQuery(item)) || actualQuery(item)));
      button.addEventListener("click", () => selectTask(item.task_id)); tabs.append(button);
    });
    renderQuery(task); renderMedia(bundle, task); renderDecision(task);
    const repair = task.repair_info;
    $("repair-card").hidden = !repair;
    $("repair-feedback").textContent = repair?.user_note || "";
    $("repair-change").textContent = repair?.reason_zh || "";
    $("previous-box").hidden = !repair?.previous_answer_image;
    $("previous-box").onclick = repair?.previous_answer_image
      ? () => openImage(repair.previous_answer_image, "原框（修正前）；主页面显示修正后目标框") : null;
    const values = record(task.task_id);
    $("modality-judgment").value = values.modality_judgment || "";
    $("approved-query").value = values.approved_query || "";
    $("note").value = values.note || "";
    $("original-query").textContent = task.original_query || "未提供";
    $("original-query-zh").textContent = task.original_query ? translation(task.original_query) || "中文译文待补充" : "未提供";
    $("source-meta").textContent = `题号 ${task.task_id} · 目标 ${task.target_object_id || "—"} · 框 ${JSON.stringify(task.bbox || [])}`;
    const priorNotes = (task.predecessor_task_ids || []).map(id => state.previousDepthReviews?.[id]?.record.note).filter(Boolean);
    if (priorNotes.length) $("source-meta").textContent += ` · 同对象旧题反馈：${priorNotes.join("；")}`;
  }
  function decide(value) {
    const task = currentTask(); if (!task) return;
    const values = record(task.task_id);
    values.decision = value;
    values.review_mode = "quick";
    save();
    const next = nextPending(task.task_id);
    if (next) selectTask(next);
    else { render(); feedback("待看题已处理完，可回看或导出 CSV"); }
  }
  function skip() {
    const next = nextPending(state.currentTaskId);
    if (next) selectTask(next);
    else navigate(1);
  }
  function clearDecision() {
    const task = currentTask(); if (!task) return;
    record(task.task_id).decision = "";
    save(); render();
  }
  function csvCell(value) { return `"${String(value ?? "").replaceAll('"', '""')}"`; }
  function download(name, body, type) {
    const url = URL.createObjectURL(new Blob([body], {type}));
    const anchor = node("a"); anchor.href = url; anchor.download = name; document.body.append(anchor);
    anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function exportCsv() {
    const lines = [data.fields.map(csvCell).join(",")];
    for (const {task, bundle} of tasks) {
      const staticValues = {task_id: task.task_id, category: task.category || bundle.category,
        split: task.split || bundle.split, bundle_id: task.bundle_id || bundle.id,
        scene_id: task.scene_id || bundle.scene, source_id: task.source_id,
        original_query: task.original_query, proposed_query: task.proposed_query};
      const values = record(task.task_id);
      lines.push(data.fields.map(field => csvCell(Object.hasOwn(staticValues, field) ? staticValues[field] : values[field] ?? "")).join(","));
    }
    download("abv_review_decisions.csv", "\uFEFF" + lines.join("\r\n") + "\r\n", "text/csv;charset=utf-8");
    feedback(`已导出 ${tasks.length} 题`);
  }
  function exportJson() {
    download("abv_review_backup.json", JSON.stringify({version: VERSION, packageKey,
      exportedAt: new Date().toISOString(), state}, null, 2) + "\n", "application/json;charset=utf-8");
    feedback("备份已导出");
  }
  async function importJson(file) {
    const backup = JSON.parse(await file.text());
    if (backup.version === VERSION && data.revision?.source_package_key === backup.packageKey) {
      const count = mergePrevious(backup.state);
      save(); render(); feedback(`已沿用 ${count} 条旧记录；旧深度反馈单独保留，新题待看。`);
      return;
    }
    if (backup.version !== VERSION || backup.packageKey !== packageKey || backup.state?.version !== VERSION) {
      throw new Error("备份版本或审核包不匹配");
    }
    for (const id of Object.keys(backup.state.records || {})) if (!taskById.has(id)) throw new Error(`未知题号：${id}`);
    const changed = Object.keys(backup.state.records || {}).filter(id =>
      JSON.stringify(backup.state.sourceSnapshot?.[id]) !== JSON.stringify(sourceSnapshot[id]));
    if (changed.length) throw new Error(`${changed.length} 题来源与当前包不符`);
    Object.assign(state.records, backup.state.records || {});
    Object.assign(state.blindRecords, backup.state.blindRecords || {});
    Object.assign(state.exposure, backup.state.exposure || {});
    if (backup.state.previousDepthReviews) {
      state.previousDepthReviews ||= {};
      Object.assign(state.previousDepthReviews, backup.state.previousDepthReviews);
    }
    if (taskById.has(backup.state.currentTaskId)) {
      state.currentTaskId = backup.state.currentTaskId;
      state.currentBundleId = taskById.get(state.currentTaskId).bundle.id;
    }
    save(); render(); feedback("备份已导入");
  }

  for (const value of [...new Set(bundles.map(category))].sort()) {
    const option = node("option", "", CATEGORY[value] || value); option.value = value;
    $("filter-category").append(option);
  }
  $("search").addEventListener("input", event => { filters.search = event.target.value; renderList(); });
  $("filter-category").addEventListener("change", event => { filters.category = event.target.value; renderList(); });
  $("filter-status").addEventListener("change", event => { filters.status = event.target.value; renderList(); });
  $("next-incomplete").addEventListener("click", nextIncomplete);
  $("previous-task").addEventListener("click", () => navigate(-1));
  $("next-task").addEventListener("click", () => navigate(1));
  $("show-box").addEventListener("change", event => {
    showBox = event.target.checked; renderMedia(currentBundle(), currentTask());
  });
  $("modality-judgment").addEventListener("change", event => {
    record(currentTask().task_id).modality_judgment = event.target.value; save();
  });
  $("approved-query").addEventListener("input", event => {
    const task = currentTask(); const values = record(task.task_id);
    if (values.approved_query === event.target.value) return;
    values.approved_query = event.target.value;
    if (values.decision) values.decision = "";
    renderQuery(task); renderDecision(task); renderProgress(); renderList(); save();
  });
  $("note").addEventListener("input", event => { record(currentTask().task_id).note = event.target.value; save(); });
  $("approve").addEventListener("click", () => decide("approve"));
  $("reject").addEventListener("click", () => decide("reject"));
  $("skip").addEventListener("click", skip);
  $("clear-decision").addEventListener("click", clearDecision);
  $("export-csv").addEventListener("click", exportCsv);
  $("export-json").addEventListener("click", exportJson);
  $("import-button").addEventListener("click", () => $("import-json").click());
  $("import-json").addEventListener("change", async event => {
    const file = event.target.files[0]; if (!file) return;
    try { await importJson(file); } catch (error) { feedback(`导入失败：${error.message}`, true); }
    event.target.value = "";
  });
  $("close-dialog").addEventListener("click", () => $("image-dialog").close());
  document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
    const image = $("dialog-image"); const fit = button.dataset.zoom === "fit";
    image.classList.toggle("fit", fit);
    image.style.width = fit ? "" : `${image.naturalWidth * Number(button.dataset.zoom) / 100}px`;
    $("zoom-stage").scrollTo(0, 0);
  }));
  document.addEventListener("keydown", event => {
    if ($("image-dialog").open || event.altKey || event.ctrlKey || event.metaKey || event.repeat) return;
    const focused = document.activeElement;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(focused?.tagName) || focused?.isContentEditable) return;
    const action = {"1": () => decide("approve"), "2": () => decide("reject"), "3": skip,
      ArrowLeft: () => navigate(-1), ArrowRight: () => navigate(1)}[event.key];
    if (action) { event.preventDefault(); action(); }
  });
  restore(); render();
})();
