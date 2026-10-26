(() => {
  const $ = (id) => document.getElementById(id);
  let selectedSide = new URLSearchParams(location.search).get("side");
  const withSide = (url) => selectedSide ? `${url}${url.includes("?") ? "&" : "?"}side=${encodeURIComponent(selectedSide)}` : url;
  const msg = $("msg");
  const canvas = $("handCanvas");
  const poseList = $("poseList");
  let ws = null;
  let jointNames = [];
  let handSide = "left";
  let handType = "o20";
  let productHandType = "o20";
  let panelMode = "";
  let panelState = "";
  let dragging = false;
  let editingValue = false; // 数值框输入中，避免 snapshot 冲掉半成品
  let selected = "";
  let editingName = "";
  let showReserved = false;
  let layoutKey = "";
  let posesKey = ""; // 避免 20Hz snapshot 整表重绘打断列表点击
  let poseSummaries = []; // [{name, keyframe_count, keyframes:[{name,duration}]}]
  let selectedKeyframe = "";
  let progressDragging = false;
  let kfPlayToken = 0;
  let kfPlayRaf = null;
  let lastScrubAt = 0;
  let pendingScrub = null;
  let expandedPoses = new Set(); // 姿态库展开的手势名
  let poseFilter = "";
  let coupledJoints = [];
  let coupleByTip = {}; // tipIndex -> sourceIndex
  let displayUnit = "hw255"; // hw255 | rad | deg — 仅显示；控制仍为 0–255
  let jointScales = []; // [{lower,upper,inverted}] 与 jointNames 对齐
  let displayAvailable = ["hw255", "rad", "deg"];
  let displayApprox = false;
  let lastDisplayMetaKey = "";
  const DISPLAY_STORAGE_KEY = "teach_panel_display_unit";
  const JOINT_MIN = 0;
  const JOINT_MAX = 255;
  const STEP_FINE = 1;
  const STEP_COARSE = 5;
  const RAD_STEP_FINE = 0.01;
  const RAD_STEP_COARSE = 0.05;
  const DEG_STEP_FINE = 0.5;
  const DEG_STEP_COARSE = 2.5;
  const KF_NAME_PRESETS = ["pre", "pre2", "grasp", "hold", "release"];
  const SCRUB_MIN_MS = 33;

  let sharedState = null, builtinCatalog = null, sharedFresh = false, sharedPolling = false;
  let builtinGroup = "default", builtinBusy = false, builtinStopping = false, builtinSignature = "";
  let sideOptionsKey = "", sharedReadSequence = 0, pendingBuiltinMotion = null, sharedUnavailable = false;
  const pendingPanelActions = new Set();

  function builtinMessage(text, error = false) {
    const el = $("builtinMessage");
    el.textContent = text; el.hidden = !text;
    el.className = `builtin-notice${error ? " error" : ""}`;
  }
  async function sharedRequest(path, body, method = "POST") {
    const response = await fetch(`/api/workbench/${path}`, {
      method: body === undefined ? "GET" : method,
      headers: { "Content-Type": "application/json" },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const value = await response.json();
    if (!response.ok) {
      const error = new Error(value.detail?.reason || value.message || `请求失败 (${response.status})`);
      error.status = response.status; throw error;
    }
    return value;
  }
  function sideSwitchReason() {
    if (!sharedFresh || !ws || ws.readyState !== WebSocket.OPEN) return "等待侧别与连接确认";
    if (pendingBuiltinMotion) return "动作请求尚未确认，请等待或停止当前手势后切换";
    if (builtinBusy || builtinStopping || pendingPanelActions.size) return "正在处理请求，请稍候";
    if (["TEACH_EDIT", "PLAYBACK"].includes(panelState)) return "先停止播放并退出本页示教，再切换机械手或页面";
    // A passive observer cannot stop another controller or close its editor.
    // Navigating the idle page neither acquires a lease nor sends motion.
    if (sharedState?.observe_only) return "";
    if (sharedState?.input_pause?.editors) return "其他页面仍在编辑，请先关闭对应编辑器后切换";
    const hand = sharedState?.hands?.[selectedSide];
    if (hand?.state_confirmed === false || hand?.execution_kind === "unknown") return "等待执行状态确认";
    if (hand?.command_pending || (hand?.active && !["hold", "idle"].includes(hand.active))) return "先停止当前手势，再切换机械手或页面";
    return "";
  }
  function updatePanelControls() {
    const reason = sideSwitchReason();
    $("panelSide").disabled = !!reason;
    $("sideSwitchHint").textContent = reason;
    $("workbenchLink").href = withSide("/workbench");
    $("workbenchLink").setAttribute("aria-disabled", String(!!reason));
    const readonly = !!sharedState?.observe_only;
    $("builtinReadonly").hidden = !readonly;
    $("panelSpeed").disabled = !sharedFresh || readonly || builtinBusy || builtinCatalog?.motion_controls !== 1;
    $("panelStop").disabled = !sharedFresh || readonly || builtinStopping;
    $("builtinPause").hidden = !sharedState?.input_pause?.paused;
    $("builtinPauseText").textContent = sharedState?.input_pause?.editors
      ? "示教或编辑期间，内置手势执行已暂停。请先停止播放并退出示教，或关闭其他编辑器。"
      : "编辑已结束，动作输入仍暂停。确认后可恢复控制。";
    $("panelResume").disabled = !sharedFresh || builtinBusy || !!sharedState?.input_pause?.editors || !sharedState?.input_pause?.ready;
    for (const id of ["btnPlay", "btnStopPlay", "btnReinit"]) $(id).disabled = readonly;
    document.querySelectorAll('[data-pose-motion]').forEach(el => { el.disabled = readonly; });
  }
  function renderBuiltins() {
    const all = (builtinCatalog?.primitives || []).filter(item => !item.template && ["default", "perception"].includes(item.group));
    const query = $("builtinSearch").value.trim().toLowerCase();
    const group = all.filter(item => item.group === builtinGroup);
    const visible = group.filter(item => `${item.name} ${item.label} ${item.description || ""}`.toLowerCase().includes(query));
    const groupName = builtinGroup === "default" ? "默认原语" : "感知原语";
    document.querySelectorAll('[data-builtin-group]').forEach(el => {
      const selected = el.dataset.builtinGroup === builtinGroup;
      const entries = all.filter(item => item.group === el.dataset.builtinGroup);
      el.setAttribute("aria-selected", String(selected));
      el.textContent = `${el.dataset.builtinGroup === "default" ? "默认原语" : "感知原语"} (${entries.length})`;
    });
    $("clearBuiltinSearch").hidden = !query;
    $("builtinFilterState").textContent = !builtinCatalog ? "正在读取内置手势…"
      : `${selectedSide === "left" ? "左手" : "右手"} · ${groupName}：显示 ${visible.length}/${group.length}${query ? ` · 筛选“${query}”` : " · 包含当前不可用项"}`;
    const commonReason = !sharedFresh || !ws || ws.readyState !== WebSocket.OPEN ? "等待连接确认"
      : sharedState?.observe_only ? "只读观察模式不发送动作"
      : sharedState?.input_pause?.paused ? "编辑期间动作暂停，请先退出示教或恢复控制" : "";
    const signature = JSON.stringify([visible, commonReason, builtinBusy]);
    if (signature === builtinSignature) return;
    builtinSignature = signature;
    $("builtinList").replaceChildren(...visible.map(item => {
      const row = document.createElement("div"); row.className = "builtin-item";
      const info = document.createElement("div"), label = document.createElement("strong"), name = document.createElement("small"), reason = document.createElement("small");
      label.textContent = item.label; name.textContent = item.name;
      reason.textContent = item.reason || commonReason || "可提交执行";
      info.append(label, name, reason);
      const execute = document.createElement("button"); execute.type = "button"; execute.textContent = "执行";
      execute.setAttribute("aria-label", `执行内置手势 ${item.label}`);
      execute.disabled = !item.available || !!commonReason || builtinBusy;
      execute.title = item.reason || commonReason || item.label;
      execute.onclick = () => void runBuiltin(async () => {
        const motion = pendingBuiltinMotion = { side: selectedSide, afterRead: sharedReadSequence, stopping: false };
        updatePanelControls();
        try { await sharedRequest("select", { side: selectedSide, name: item.name }); }
        catch (error) {
          if (error.status >= 400 && error.status < 500 && pendingBuiltinMotion === motion) pendingBuiltinMotion = null;
          throw error;
        }
        builtinMessage(`已提交“${item.label}”，可随时停止。`);
      });
      row.append(info, execute); return row;
    }));
    if (!visible.length && builtinCatalog) {
      const empty = document.createElement("p"); empty.className = "muted builtin-empty";
      empty.textContent = query ? "此分类没有匹配项，请清除筛选或切换分类。" : "当前分类没有原语。";
      $("builtinList").append(empty);
    }
  }
  async function refreshBuiltins() {
    if (sharedPolling || sharedUnavailable) return;
    sharedPolling = true;
    const readSequence = ++sharedReadSequence;
    try {
      const snapshot = await sharedRequest("state");
      const hands = snapshot.hands || {};
      if (!hands[selectedSide]) selectedSide = hands[snapshot.default_side] ? snapshot.default_side : Object.keys(hands)[0];
      sharedState = snapshot;
      if (pendingBuiltinMotion && readSequence > pendingBuiltinMotion.afterRead) {
        const hand = hands[pendingBuiltinMotion.side];
        const observed = pendingBuiltinMotion.stopping
          ? hand && !hand.command_pending && hand.state_confirmed !== false && hand.execution_kind === "idle" && !hand.active
          : hand && (hand.command_pending || (hand.active && !["hold", "idle"].includes(hand.active)));
        if (observed) pendingBuiltinMotion = null;
      }
      const optionsKey = JSON.stringify(Object.entries(hands).map(([side, value]) => [side, value.model]));
      if (sideOptionsKey !== optionsKey) {
        sideOptionsKey = optionsKey;
        $("panelSide").replaceChildren(...Object.entries(hands).map(([side, value]) => new Option(`${side === "left" ? "左手" : "右手"} · ${value.model.toUpperCase()}`, side)));
      }
      $("panelSide").value = selectedSide || "";
      if (document.activeElement !== $("panelSpeed") && !builtinBusy) $("panelSpeed").value = snapshot.primitive_speed;
      builtinCatalog = await sharedRequest(`catalog?side=${encodeURIComponent(selectedSide)}`);
      sharedFresh = true;
    } catch (error) {
      sharedFresh = false;
      if (error.status === 404) {
        // TeachPanelApp without a workbench still serves the original WS editor.
        sharedUnavailable = true;
        document.querySelector('.builtin-panel').hidden = true;
        document.querySelector('.hand-navigation').hidden = true;
      } else builtinMessage(`内置手势暂不可用：${error.message}`, true);
    } finally {
      sharedPolling = false; updatePanelControls(); renderBuiltins();
    }
  }
  async function runBuiltin(action) {
    if (builtinBusy) return;
    builtinBusy = true; updatePanelControls(); renderBuiltins();
    try { await action(); } catch (error) { builtinMessage(error.message, true); }
    finally { builtinBusy = false; await refreshBuiltins(); updatePanelControls(); renderBuiltins(); }
  }
  $("panelSide").onchange = () => {
    const next = $("panelSide").value;
    if (sideSwitchReason() || !sharedState?.hands?.[next]) { $("panelSide").value = selectedSide; return; }
    location.assign(`${location.pathname}?side=${encodeURIComponent(next)}`);
  };
  $("workbenchLink").onclick = event => { if (sideSwitchReason()) { event.preventDefault(); builtinMessage(sideSwitchReason(), true); } };
  $("panelSpeed").onchange = () => void runBuiltin(async () => {
    const result = await sharedRequest("speed", { speed: Number($("panelSpeed").value) }, "PUT");
    $("panelSpeed").value = result.speed; builtinMessage("手势速度已生效，仅用于本次运行。");
  });
  $("panelStop").onclick = async () => {
    if (builtinStopping || !sharedFresh || sharedState?.observe_only) return;
    builtinStopping = true;
    pendingBuiltinMotion = { side: selectedSide, afterRead: sharedReadSequence, stopping: true };
    updatePanelControls(); stopKeyframeAutoPlay(); pendingScrub = null;
    try { await sharedRequest("stop", { side: selectedSide }); builtinMessage("已提交停止手势。"); }
    catch (error) { builtinMessage(error.message, true); }
    finally { builtinStopping = false; await refreshBuiltins(); updatePanelControls(); }
  };
  $("panelResume").onclick = () => void runBuiltin(async () => { await sharedRequest("resume", {}); builtinMessage("控制已恢复，请重新操作。"); });
  document.querySelectorAll('[data-builtin-group]').forEach(el => { el.onclick = () => { builtinGroup = el.dataset.builtinGroup; renderBuiltins(); }; });
  $("builtinSearch").oninput = renderBuiltins;
  $("clearBuiltinSearch").onclick = () => { $("builtinSearch").value = ""; renderBuiltins(); };

  function setMsg(text, ok) {
    msg.textContent = text || "";
    msg.style.color = ok ? "#0d6e56" : "#9a3412";
  }

  function loadStoredDisplayUnit() {
    try {
      const v = String(localStorage.getItem(DISPLAY_STORAGE_KEY) || "").trim();
      if (v === "hw255" || v === "rad" || v === "deg") return v;
    } catch (_) {
      /* ignore */
    }
    return "hw255";
  }

  function storeDisplayUnit(unit) {
    try {
      localStorage.setItem(DISPLAY_STORAGE_KEY, unit);
    } catch (_) {
      /* ignore */
    }
  }

  function scaleAt(idx) {
    const s = jointScales[idx];
    if (!s) return { lower: 0, upper: 0, inverted: false };
    return {
      lower: Number(s.lower) || 0,
      upper: Number(s.upper) || 0,
      inverted: Boolean(s.inverted),
    };
  }

  function hwToRad(idx, hw) {
    const { lower, upper, inverted } = scaleAt(idx);
    if (!(upper > lower)) return 0;
    let ratio = Math.max(0, Math.min(255, Number(hw))) / 255;
    if (inverted) ratio = 1 - ratio;
    return lower + ratio * (upper - lower);
  }

  function radToHw(idx, rad) {
    const { lower, upper, inverted } = scaleAt(idx);
    if (!(upper > lower)) return 0;
    let ratio = (Number(rad) - lower) / (upper - lower);
    ratio = Math.max(0, Math.min(1, ratio));
    if (inverted) ratio = 1 - ratio;
    return Math.round(Math.max(0, Math.min(255, ratio * 255)));
  }

  function hwToDisplay(idx, hw) {
    if (displayUnit === "rad") return hwToRad(idx, hw);
    if (displayUnit === "deg") return (hwToRad(idx, hw) * 180) / Math.PI;
    return Math.round(Number(hw));
  }

  function displayToHw(idx, displayVal) {
    const n = Number(displayVal);
    if (!Number.isFinite(n)) return null;
    if (displayUnit === "rad") return radToHw(idx, n);
    if (displayUnit === "deg") return radToHw(idx, (n * Math.PI) / 180);
    return clampJoint(n);
  }

  function formatDisplay(idx, hw) {
    const v = hwToDisplay(idx, hw);
    if (displayUnit === "rad") return Number(v).toFixed(3);
    if (displayUnit === "deg") return Number(v).toFixed(1);
    return String(Math.round(Number(v)));
  }

  function displayStep(fine) {
    if (displayUnit === "rad") return fine ? RAD_STEP_FINE : RAD_STEP_COARSE;
    if (displayUnit === "deg") return fine ? DEG_STEP_FINE : DEG_STEP_COARSE;
    return fine ? STEP_FINE : STEP_COARSE;
  }

  function numInputAttrs() {
    if (displayUnit === "rad") {
      return { min: null, max: null, step: String(RAD_STEP_FINE), title: "输入弧度；↑↓±0.01，Shift ±0.05" };
    }
    if (displayUnit === "deg") {
      return { min: null, max: null, step: String(DEG_STEP_FINE), title: "输入角度°；↑↓±0.5，Shift ±2.5" };
    }
    return {
      min: String(JOINT_MIN),
      max: String(JOINT_MAX),
      step: String(STEP_FINE),
      title: "键盘输入 0–255；↑↓±1，Shift+↑↓±5",
    };
  }

  function refreshNumDisplays() {
    jointNames.forEach((_, idx) => {
      const slider = $(`j${idx}`);
      const num = $(`jn${idx}`);
      if (!slider || !num) return;
      if (document.activeElement === num) return;
      num.value = formatDisplay(idx, Number(slider.value));
      const attrs = numInputAttrs();
      if (attrs.min == null) {
        num.removeAttribute("min");
        num.removeAttribute("max");
      } else {
        num.min = attrs.min;
        num.max = attrs.max;
      }
      num.step = attrs.step;
      if (!num.disabled) num.title = attrs.title;
    });
  }

  function syncDisplayUnitSelect() {
    const sel = $("displayUnit");
    if (!sel) return;
    // 下拉打开时勿改 options/value，否则会强制收起并闪烁
    if (document.activeElement === sel) return;
    const opts = Array.from(sel.options);
    opts.forEach((opt) => {
      const ok = displayAvailable.includes(opt.value);
      opt.disabled = !ok;
      // 不用 hidden：部分浏览器对 select option.hidden 重绘会闪
      opt.hidden = false;
      if (!ok) {
        opt.disabled = true;
      }
    });
    if (!displayAvailable.includes(displayUnit)) {
      displayUnit = "hw255";
      storeDisplayUnit(displayUnit);
    }
    if (sel.value !== displayUnit) {
      sel.value = displayUnit;
    }
    sel.title = displayApprox
      ? "无 URDF 限位时为 0~π/2 线性近似；有 Converter 后为真实关节限位"
      : "仅改变数值显示；滑条/下发/保存仍为 0–255";
  }

  function applyJointDisplayMeta(meta) {
    if (!meta || typeof meta !== "object") return;
    const key = JSON.stringify({
      available: meta.available || [],
      scales: meta.scales || [],
      approx: Boolean(meta.approx),
    });
    if (key === lastDisplayMetaKey) {
      return; // 20Hz snapshot 勿重复改 DOM
    }
    lastDisplayMetaKey = key;
    const prevUnit = displayUnit;
    if (Array.isArray(meta.available) && meta.available.length) {
      displayAvailable = meta.available.map(String);
    }
    if (Array.isArray(meta.scales)) {
      jointScales = meta.scales;
    }
    displayApprox = Boolean(meta.approx);
    syncDisplayUnitSelect();
    if (prevUnit !== displayUnit || displayUnit !== "hw255") {
      refreshNumDisplays();
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}${withSide("/ws")}`);
    ws.onopen = () => { setMsg("已连接", true); updatePanelControls(); renderBuiltins(); };
    ws.onclose = () => {
      setMsg("连接断开（已 Mute）");
      pendingPanelActions.clear(); updatePanelControls(); renderBuiltins();
      setTimeout(connect, 1500);
    };
    ws.onerror = () => setMsg("WebSocket 错误");
    ws.onmessage = (ev) => {
      let data;
      try {
        data = JSON.parse(ev.data);
      } catch (_) {
        return;
      }
      if (data.type === "error") {
        pendingPanelActions.clear(); updatePanelControls();
        const m = data.message || "error";
        if (m === "muted_or_not_teach") {
          setMsg("未进 TEACH 或已 Mute：滑条仅本地预览，不下发真机");
        } else {
          setMsg(m);
        }
        return;
      }
      if (data.type === "ok") {
        if (["enter_teach", "load_pose"].includes(data.action)) panelState = "TEACH_EDIT";
        if (["play_pose", "scrub_pose"].includes(data.action)) panelState = "PLAYBACK";
        pendingPanelActions.delete(data.action); updatePanelControls();
        if (data.action === "load_pose") {
          const n = normalizePoseName(data.name || selected);
          selected = n;
          if (n) expandedPoses.add(n);
          $("poseName").value = n || "";
          if (data.keyframe) {
            selectKeyframe(data.keyframe, { fillDur: false });
          } else {
            resetKeyframeContext({ suggestFor: findPoseSummary(n) });
          }
          if (data.duration != null) $("poseDur").value = data.duration;
          if (data.description != null) $("poseDesc").value = data.description;
          $("chkUseTargets").checked = true;
          renderPoseList(poseSummaries);
          updateKeyframeEditor(findPoseSummary(n));
          const kfHint = data.keyframe ? ` 帧「${data.keyframe}」` : "";
          setMsg(`已载入「${data.name}」${kfHint}可二次修改`, true);
        } else if (data.action === "set_prefs") {
          if (typeof data.return_to_teach_after_playback === "boolean") {
            $("chkReturnTeach").checked = data.return_to_teach_after_playback;
          }
          setMsg("偏好已更新", true);
        } else if (data.action === "play_pose") {
          const kf = data.keyframe ? ` / ${data.keyframe}` : "";
          const p =
            data.progress != null && data.progress !== ""
              ? ` @${Number(data.progress).toFixed(2)}`
              : "";
          setMsg(`播放「${data.name || ""}${kf}${p}」…`, true);
        } else if (data.action === "set_progress" || data.action === "scrub_pose") {
          // 拖动 / 单帧 rAF 高频，避免刷屏；状态看时间轴
          if (!progressDragging && kfPlayRaf == null) {
            const kf = data.keyframe ? ` / ${data.keyframe}` : "";
            setMsg(`细控${kf} p=${Number(data.progress || 0).toFixed(3)}`, true);
          }
        } else if (data.action === "save_pose") {
          send({ type: "list_poses" });
          const kf = data.keyframe ? ` 帧「${data.keyframe}」` : "";
          setMsg(`已保存「${data.name || ""}」${kf}`, true);
          if (data.keyframe) {
            const poseName = normalizePoseName(data.name || selected);
            selectPose(poseName, { resetKeyframe: false, fillMeta: false, announce: false });
            if (data.advance_keyframe) {
              // 「＋添加」：预填下一建议帧名，方便连续录中间态
              const summary = findPoseSummary(poseName);
              const kfs = summary && Array.isArray(summary.keyframes)
                ? summary.keyframes.slice()
                : [];
              const has = kfs.some(
                (k) => normalizePoseName(k.name || k) === normalizePoseName(data.keyframe)
              );
              if (!has) {
                kfs.push({
                  name: data.keyframe,
                  duration: Number($("poseDur").value || 0.6),
                });
              }
              const next = suggestNextKfName({ name: poseName, keyframes: kfs });
              if ($("poseKeyframe") && next) $("poseKeyframe").value = next;
              selectKeyframe(next || data.keyframe, { fillDur: false });
            } else {
              // 「保存」=覆盖当前帧：保持帧名，避免二次编辑写到建议的下一帧
              selectKeyframe(data.keyframe);
              if ($("poseKeyframe")) $("poseKeyframe").value = data.keyframe;
            }
          }
        } else {
          setMsg(`${data.action || "ok"} 成功`, true);
          if (
            data.action === "delete_pose" ||
            data.action === "delete_keyframe" ||
            data.action === "move_keyframe" ||
            data.action === "rename_keyframe"
          ) {
            if (
              data.action === "delete_keyframe" &&
              normalizePoseName(data.keyframe) === selectedKeyframe
            ) {
              selectedKeyframe = "";
              resetKeyframeContext({
                suggestFor: findPoseSummary(data.name || selected),
              });
            }
            if (data.action === "rename_keyframe") {
              const oldKf = normalizePoseName(data.keyframe);
              const newKf = normalizePoseName(data.new_keyframe);
              if (oldKf && oldKf === selectedKeyframe && newKf) {
                selectKeyframe(newKf);
              }
            }
            if (
              data.action === "delete_pose" &&
              normalizePoseName(data.name) === selected
            ) {
              beginNewGesture();
            }
            send({ type: "list_poses" });
          }
        }
      }
      if (data.type === "snapshot" || data.panel_state) {
        applySnapshot(data);
      }
    };
  }

  setInterval(() => { if (ws && ws.readyState === WebSocket.OPEN) send({ type: "heartbeat" }); }, 2000);
  setInterval(() => { if (!document.hidden && ws && ws.readyState === WebSocket.OPEN) send({ type: "live_heartbeat" }); }, 150);
  document.addEventListener("visibilitychange", () => { if (document.hidden) send({ type: "live_stop" }); });

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (sharedState?.observe_only && ["play_pose", "scrub_pose", "set_progress", "reinit", "stop_playback"].includes(obj.type)) {
        setMsg("只读观察模式不发送动作；仍可编辑和保存姿态"); return;
      }
      if (["enter_teach", "exit_teach", "load_pose", "play_pose", "scrub_pose", "stop_playback", "reinit"].includes(obj.type)) {
        pendingPanelActions.add(obj.type); updatePanelControls();
      }
      ws.send(JSON.stringify(obj));
    }
  }

  function clampJoint(v) {
    const n = Math.round(Number(v));
    if (!Number.isFinite(n)) return null;
    return Math.max(JOINT_MIN, Math.min(JOINT_MAX, n));
  }

  function isJointUiBusy() {
    return dragging || editingValue;
  }

  function syncCoupledTipsFromDom() {
    Object.keys(coupleByTip).forEach((tipStr) => {
      const tip = Number(tipStr);
      const mid = coupleByTip[tip];
      const midEl = $(`j${mid}`);
      const tipEl = $(`j${tip}`);
      const tipNum = $(`jn${tip}`);
      if (!midEl || !tipEl) return;
      tipEl.value = midEl.value;
      if (tipNum && document.activeElement !== tipNum) {
        tipNum.value = formatDisplay(tip, Number(midEl.value));
      }
    });
  }

  function readPositions() {
    syncCoupledTipsFromDom();
    return jointNames.map((_, idx) => {
      const el = $(`j${idx}`);
      return el ? Number(el.value) : 128;
    });
  }

  function setJointLocal(index, value, { push = true, flush = false } = {}) {
    const v = clampJoint(value);
    if (v == null) return null;
    const slider = $(`j${index}`);
    const num = $(`jn${index}`);
    if (slider) slider.value = String(v);
    if (num && document.activeElement !== num) num.value = formatDisplay(index, v);
    else if (num && document.activeElement === num && flush) {
      num.value = formatDisplay(index, v);
    }
    syncCoupledTipsFromDom();
    if (push) send({ type: "set_joints", positions: readPositions() });
    if (flush) send({ type: "flush_joints" });
    return v;
  }

  function applySnapshot(s) {
    handSide = s.hand_side || handSide;
    handType = s.hand_type || handType;
    productHandType = s.product_hand_type || handType;
    if (Array.isArray(s.coupled_joints)) {
      coupledJoints = s.coupled_joints;
      coupleByTip = {};
      coupledJoints.forEach((c) => {
        if (c && typeof c.index === "number" && typeof c.source_index === "number") {
          coupleByTip[c.index] = c.source_index;
        }
      });
    }
    $("meta").textContent =
      `${handType} / ${handSide} / joints=${s.num_joints || jointNames.length}` +
      (coupledJoints.length ? ` / tip耦合×${coupledJoints.length}` : "") +
      ` / 显示=${displayUnit === "rad" ? "rad" : displayUnit === "deg" ? "deg" : "0–255"}`;
    const mode = s.mode || "?";
    panelMode = mode;
    panelState = s.panel_state || "";
    updatePanelControls();
    const modeEl = $("modeBadge");
    modeEl.textContent = `mode=${mode}`;
    modeEl.className = "badge" + (mode === "teach" ? " teach" : "");
    $("stateBadge").textContent = s.panel_state || "?";
    $("stateBadge").className =
      "badge" + (s.muted ? " warn" : s.panel_state === "TEACH_EDIT" ? " teach" : "");
    $("primBadge").textContent = s.active_primitive || "hold";

    editingName = s.editing_pose_name || "";
    const editBadge = $("editBadge");
    const editHint = $("editHint");
    if (editingName) {
      editBadge.classList.remove("hidden");
      editBadge.textContent = `编辑:${editingName}`;
      editHint.classList.remove("hidden");
      $("editName").textContent = editingName;
      if (!$("poseName").value) $("poseName").value = editingName;
    } else {
      editBadge.classList.add("hidden");
      editHint.classList.add("hidden");
    }

    if (Array.isArray(s.joint_names) && s.joint_names.length) {
      const coupleKey = coupledJoints.map((c) => `${c.index}:${c.source_index}`).join(",");
      const key = `${handType}|${handSide}|${s.joint_names.join(",")}|${showReserved}|${coupleKey}`;
      if (key !== layoutKey) {
        jointNames = s.joint_names;
        layoutKey = key;
        buildHand(s.targets || s.feedback || []);
      }
    }
    if (s.joint_display) {
      applyJointDisplayMeta(s.joint_display);
    }
    if (s.coupling_hint) {
      // layout.hint 已含耦合说明；此处仅作兜底提示
      const hint = $("handHint");
      if (hint && hint.textContent && !hint.textContent.includes("耦合")) {
        hint.textContent = hint.textContent + " · " + s.coupling_hint;
      }
    }
    if (!isJointUiBusy() && Array.isArray(s.targets)) {
      s.targets.forEach((v, i) => {
        const el = $(`j${i}`);
        const num = $(`jn${i}`);
        if (el) el.value = v;
        if (num) num.value = formatDisplay(i, v);
      });
      syncCoupledTipsFromDom();
    }
    if (Array.isArray(s.poses)) {
      const key = JSON.stringify(s.poses);
      if (key !== posesKey) {
        posesKey = key;
        poseSummaries = s.poses.map(normalizePoseSummary).filter((p) => p.name);
        renderPoseList(poseSummaries);
        renderKeyframeList(findPoseSummary(selected));
      } else {
        refreshPoseListActive();
      }
    }
    if (typeof s.return_to_teach_after_playback === "boolean") {
      $("chkReturnTeach").checked = s.return_to_teach_after_playback;
    }
    if (s.editing_pose_name) {
      editingName = s.editing_pose_name;
    }
    updateMotionBadge(s);
    if (!progressDragging && typeof s.motion_progress === "number") {
      syncTimelineFromMotion(s);
    }
    // hint / error：拖动中不抢焦点文案；error 优先于 hint
    if (!isJointUiBusy() && !progressDragging) {
      if (s.last_error) setMsg(s.last_error, false);
      else if (s.hint) setMsg(s.hint, !String(s.hint).includes("失败") && !String(s.hint).includes("断开"));
      else if (s.muted && s.panel_state === "TEACH_EDIT") {
        setMsg("示教中已 Mute：等待 mode=teach 或处理被拒指令后再拖滑条");
      } else if (s.panel_state === "PRIMITIVE_IDLE" && s.mode === "primitive") {
        if (!msg.textContent) {
          setMsg("可直接点列表「播放」；调滑条请先「开始示教」", true);
        }
      }
    }
  }

  function updateMotionBadge(s) {
    const el = $("motionBadge");
    if (!el) return;
    const drive = s.motion_drive || "-";
    const kf = s.motion_keyframe || s.playback_keyframe || "-";
    const p =
      typeof s.motion_progress === "number"
        ? Number(s.motion_progress).toFixed(3)
        : "-";
    el.textContent = `drive=${drive} | keyframe=${kf} | progress=${p}`;
  }

  function normalizePoseName(name) {
    return String(name || "").trim().toLowerCase();
  }

  function normalizePoseSummary(raw) {
    if (raw && typeof raw === "object") {
      return {
        name: normalizePoseName(raw.name),
        keyframe_count: Number(raw.keyframe_count || 0),
        keyframes: Array.isArray(raw.keyframes) ? raw.keyframes : [],
        duration: raw.duration,
        description: raw.description || "",
        space: raw.space,
        unit: raw.unit,
      };
    }
    return {
      name: normalizePoseName(raw),
      keyframe_count: 0,
      keyframes: [],
    };
  }

  function findPoseSummary(name) {
    const n = normalizePoseName(name);
    return poseSummaries.find((p) => p.name === n) || null;
  }

  function selectPose(
    name,
    { fillInput = true, announce = false, resetKeyframe = true, fillMeta = true } = {}
  ) {
    const n = normalizePoseName(name);
    const prev = selected;
    if (!n) {
      selected = "";
      if (resetKeyframe) resetKeyframeContext({ suggestFor: null });
      if (fillMeta) applyEditorMetaForPose("");
      refreshPoseListActive();
      updateKeyframeEditor(null);
      setTimelineEnabled(false);
      return "";
    }
    selected = n;
    if (fillInput && $("poseName")) $("poseName").value = n;
    let needFull = false;
    if (!expandedPoses.has(n)) {
      expandedPoses.add(n);
      needFull = true;
    }
    // 换手势：关键清空上一手势的帧名/选中帧，避免「带入」
    if (resetKeyframe && prev !== n) {
      const summary = findPoseSummary(n);
      resetKeyframeContext({ suggestFor: summary });
      if (fillMeta) applyEditorMetaForPose(n);
      if (editingName && editingName !== n) {
        send({ type: "clear_edit" });
      }
    } else if (fillMeta && prev !== n) {
      applyEditorMetaForPose(n);
    }
    const summary = findPoseSummary(n);
    updateKeyframeEditor(summary);
    setTimelineEnabled(keyframeSegments(summary).length > 0);
    const hasNode =
      poseList &&
      Array.from(poseList.querySelectorAll("li.pose-node")).some(
        (li) => normalizePoseName(li.dataset.name) === n
      );
    if (needFull || !hasNode) {
      renderPoseList(poseSummaries);
    } else {
      refreshPoseListActive();
    }
    if (announce) setMsg(`已选手势「${n}」`, true);
    return n;
  }

  function resetKeyframeContext({ suggestFor } = {}) {
    selectedKeyframe = "";
    const next = suggestNextKfName(suggestFor || null);
    if ($("poseKeyframe")) $("poseKeyframe").value = next;
  }

  function applyEditorMetaForPose(name) {
    const n = normalizePoseName(name);
    const summary = findPoseSummary(n);
    if (!summary) {
      if ($("poseDesc")) $("poseDesc").value = "";
      if ($("poseDur")) $("poseDur").value = "0.6";
      return;
    }
    if ($("poseDesc")) $("poseDesc").value = summary.description || "";
    const segs = keyframeSegments(summary);
    const seg = segs.find((s) => s.name === selectedKeyframe);
    const dur =
      (seg && seg.duration) ||
      Number(summary.duration) ||
      0.6;
    if ($("poseDur")) $("poseDur").value = String(dur);
  }

  function isInTeach() {
    return panelMode === "teach" || panelState === "TEACH_EDIT";
  }

  function setTimelineEnabled(on) {
    const slider = $("progressSlider");
    if (!slider) return;
    slider.disabled = !on;
    slider.title = on
      ? "按各关键帧 duration 加权，拖动跨段细控"
      : "当前手势无关键帧，无法拖进度（请先＋添加）";
  }

  /** 新建手势：清编辑态与关键帧上下文；滑条保留（可从当前示教姿起步） */
  function beginNewGesture() {
    stopKeyframeAutoPlay();
    progressDragging = false;
    selected = "";
    selectedKeyframe = "";
    editingName = "";
    if ($("poseName")) $("poseName").value = "";
    if ($("poseDesc")) $("poseDesc").value = "";
    if ($("poseDur")) $("poseDur").value = "0.6";
    if ($("poseKeyframe")) $("poseKeyframe").value = "pre";
    send({ type: "clear_edit" });
    refreshPoseListActive();
    updateKeyframeEditor(null);
    setTimelineEnabled(false);
    // 不必整树重建，仅去高亮
    if (poseList && !poseList.querySelector("li.pose-node")) {
      renderPoseList(poseSummaries);
    } else {
      refreshPoseListActive();
    }
    setMsg("已开新手势：填写手势名 → 示教 →「＋ 添加」中间态", true);
    if ($("poseName")) $("poseName").focus();
  }

  /**
   * 手势名输入变更：切到新手势上下文。
   * 存在库中 → 展开其关键帧；不存在 → 空关键帧列表 + 从 pre 起建议。
   */
  function onPoseNameChanged() {
    const n = normalizePoseName($("poseName") && $("poseName").value);
    if (n === selected) return;
    if (!n) {
      beginNewGesture();
      return;
    }
    selectPose(n, {
      fillInput: false,
      announce: true,
      resetKeyframe: true,
      fillMeta: true,
    });
  }

  function selectKeyframe(name, { fillInput = true, fillDur = true } = {}) {
    selectedKeyframe = normalizePoseName(name);
    if (fillInput && $("poseKeyframe")) $("poseKeyframe").value = selectedKeyframe;
    if (fillDur) {
      const summary = findPoseSummary(selected);
      const seg = keyframeSegments(summary).find((s) => s.name === selectedKeyframe);
      if (seg && $("poseDur")) $("poseDur").value = String(seg.duration);
    }
    refreshPoseListActive();
    return selectedKeyframe;
  }

  /** 播放名：选中 > 列表高亮行 > 编辑中 > 名称框 */
  function resolvePlayName() {
    const sel = normalizePoseName(selected);
    if (sel) return sel;
    const active = poseList && poseList.querySelector("li.pose-node.active .pose-name");
    const fromDom = normalizePoseName(
      active && (active.dataset.name || active.textContent)
    );
    if (fromDom) return fromDom;
    const edit = normalizePoseName(editingName);
    if (edit) return edit;
    return normalizePoseName($("poseName") && $("poseName").value);
  }

  function resolveKeyframeName() {
    const sel = normalizePoseName(selectedKeyframe);
    if (sel) return sel;
    return normalizePoseName($("poseKeyframe") && $("poseKeyframe").value);
  }

  function playPoseByName(name, opts = {}) {
    const n = selectPose(name || resolvePlayName(), {
      fillInput: true,
      resetKeyframe: false,
    });
    if (!n) {
      setMsg("请先点击列表中的手势选中，再点「播放全部」；或直接点条目右侧「播放」");
      return;
    }
    const payload = { type: "play_pose", name: n };
    if (opts.keyframe) payload.keyframe = opts.keyframe;
    if (opts.progress != null && opts.progress !== "") {
      payload.progress = opts.progress;
    }
    setMsg(
      `播放「${n}${opts.keyframe ? " / " + opts.keyframe : ""}」…`,
      true
    );
    send(payload);
  }

  function refreshPoseListActive() {
    if (!poseList) return;
    poseList.querySelectorAll("li.pose-node").forEach((li) => {
      const n = normalizePoseName(li.dataset.name);
      li.classList.toggle(
        "active",
        Boolean(n && (n === selected || n === normalizePoseName(editingName)))
      );
    });
    poseList.querySelectorAll("li.kf-node").forEach((li) => {
      const poseLi = li.closest("li.pose-node");
      const poseName = normalizePoseName(poseLi && poseLi.dataset.name);
      const kf = normalizePoseName(li.dataset.keyframe);
      li.classList.toggle(
        "active",
        Boolean(poseName === selected && kf && kf === selectedKeyframe)
      );
    });
  }

  function keyframeSegments(summary) {
    const raw = summary && Array.isArray(summary.keyframes) ? summary.keyframes : [];
    return raw
      .map((kf, index) => ({
        name: normalizePoseName(kf.name || kf),
        duration: Math.max(0.05, Number(kf.duration) || 0.6),
        index,
      }))
      .filter((s) => s.name);
  }

  function totalKeyframeDuration(segments) {
    return segments.reduce((acc, s) => acc + s.duration, 0);
  }

  /** 全局 g∈[0,1] → 段内 keyframe + local progress（按时长加权） */
  function mapGlobalProgress(summary, g) {
    const segs = keyframeSegments(summary);
    if (!segs.length) return null;
    const gClamped = Math.max(0, Math.min(1, Number(g) || 0));
    const total = totalKeyframeDuration(segs);
    if (total <= 0) {
      return { keyframe: segs[0].name, local: gClamped, index: 0 };
    }
    let t = gClamped * total;
    for (let i = 0; i < segs.length; i++) {
      const d = segs[i].duration;
      const last = i === segs.length - 1;
      if (t <= d || last) {
        const local = d > 0 ? Math.max(0, Math.min(1, t / d)) : 1;
        return { keyframe: segs[i].name, local, index: i };
      }
      t -= d;
    }
    const last = segs[segs.length - 1];
    return { keyframe: last.name, local: 1, index: segs.length - 1 };
  }

  /** 段内 local → 全局 g */
  function mapLocalToGlobal(summary, keyframe, local) {
    const segs = keyframeSegments(summary);
    if (!segs.length) return Math.max(0, Math.min(1, Number(local) || 0));
    const kf = normalizePoseName(keyframe);
    const p = Math.max(0, Math.min(1, Number(local) || 0));
    const total = totalKeyframeDuration(segs);
    if (total <= 0) return p;
    let acc = 0;
    for (let i = 0; i < segs.length; i++) {
      if (segs[i].name === kf) {
        return (acc + p * segs[i].duration) / total;
      }
      acc += segs[i].duration;
    }
    return 0;
  }

  function suggestNextKfName(summary) {
    const used = new Set(keyframeSegments(summary).map((s) => s.name));
    for (let i = 0; i < KF_NAME_PRESETS.length; i++) {
      if (!used.has(KF_NAME_PRESETS[i])) return KF_NAME_PRESETS[i];
    }
    let n = 1;
    while (used.has(`kf_${n}`)) n += 1;
    return `kf_${n}`;
  }

  function stopKeyframeAutoPlay() {
    kfPlayToken += 1;
    if (kfPlayRaf != null) {
      cancelAnimationFrame(kfPlayRaf);
      kfPlayRaf = null;
    }
  }

  function scrubPose(name, keyframe, progress, { force = false } = {}) {
    const n = normalizePoseName(name);
    const kf = normalizePoseName(keyframe);
    if (!n || !kf) return false;
    const p = Math.max(0, Math.min(1, Number(progress) || 0));
    const payload = { type: "scrub_pose", name: n, keyframe: kf, progress: p };
    const now = performance.now();
    const edge = p <= 0.001 || p >= 0.999;
    if (!force && !edge && now - lastScrubAt < SCRUB_MIN_MS) {
      pendingScrub = payload;
      return true;
    }
    lastScrubAt = now;
    pendingScrub = null;
    send(payload);
    return true;
  }

  function flushPendingScrub() {
    if (!pendingScrub) return;
    lastScrubAt = performance.now();
    send(pendingScrub);
    pendingScrub = null;
  }

  function updateTimelineChrome(summary, mapped) {
    const label = $("timelineLabel");
    const marks = $("timelineMarks");
    const segs = keyframeSegments(summary);
    const total = totalKeyframeDuration(segs);
    if (label) {
      if (!segs.length) {
        label.textContent = "—";
      } else if (mapped) {
        label.textContent =
          `${mapped.keyframe} · local ${Number(mapped.local).toFixed(2)} · Σ${total.toFixed(1)}s`;
      } else {
        label.textContent = `${segs.length} 段 · Σ${total.toFixed(1)}s`;
      }
    }
    if (!marks) return;
    marks.innerHTML = "";
    if (!segs.length || total <= 0) return;
    let acc = 0;
    segs.forEach((seg) => {
      const pct = (acc / total) * 100;
      const el = document.createElement("span");
      el.className = "timeline-mark";
      el.style.left = `${pct}%`;
      el.textContent = seg.name;
      el.title = `${seg.name} · ${seg.duration.toFixed(1)}s`;
      marks.appendChild(el);
      acc += seg.duration;
    });
  }

  function setGlobalSlider(g, { localLabel } = {}) {
    const slider = $("progressSlider");
    const label = $("progressVal");
    const gClamped = Math.max(0, Math.min(1, Number(g) || 0));
    if (slider && (document.activeElement !== slider || progressDragging)) {
      slider.value = String(Math.round(gClamped * 1000));
    }
    if (label) {
      const pct = `${Math.round(gClamped * 100)}%`;
      label.textContent =
        localLabel != null ? `${pct} · L${Number(localLabel).toFixed(2)}` : pct;
    }
  }

  function syncTimelineFromMotion(s) {
    const poseName = normalizePoseName(
      s.playback_name || selected || ($("poseName") && $("poseName").value)
    );
    const summary = findPoseSummary(poseName);
    const kf = normalizePoseName(
      s.motion_keyframe || s.playback_keyframe || selectedKeyframe
    );
    const local =
      typeof s.motion_progress === "number" ? Number(s.motion_progress) : 0;
    let g = local;
    let mapped = null;
    if (summary && keyframeSegments(summary).length && kf) {
      g = mapLocalToGlobal(summary, kf, local);
      mapped = { keyframe: kf, local, index: -1 };
      if (kf !== selectedKeyframe) selectKeyframe(kf, { fillInput: false });
    }
    setGlobalSlider(g, { localLabel: local });
    updateTimelineChrome(summary, mapped);
  }

  function emitGlobalScrub(g) {
    const n = resolvePlayName();
    const summary = findPoseSummary(n);
    const segs = keyframeSegments(summary);
    if (!n) {
      setMsg("请先选中姿态");
      return;
    }
    if (!segs.length) {
      setMsg("该姿态尚无关键帧，无法细控进度");
      return;
    }
    const mapped = mapGlobalProgress(summary, g);
    if (!mapped) return;
    selectKeyframe(mapped.keyframe, { fillInput: true });
    setGlobalSlider(g, { localLabel: mapped.local });
    updateTimelineChrome(summary, mapped);
    scrubPose(n, mapped.keyframe, mapped.local);
  }

  function playKeyframeIndependent(name, keyframe) {
    const n = selectPose(name || resolvePlayName(), {
      fillInput: true,
      resetKeyframe: false,
      fillMeta: false,
    });
    const kf = selectKeyframe(keyframe || resolveKeyframeName());
    if (!n || !kf) {
      setMsg("请先选中姿态与关键帧");
      return;
    }
    const summary = findPoseSummary(n);
    const seg = keyframeSegments(summary).find((s) => s.name === kf);
    if (!seg) {
      setMsg(`姿态「${n}」无关键帧「${kf}」`);
      return;
    }
    stopKeyframeAutoPlay();
    progressDragging = false;
    setMsg(`引擎单帧 AUTO「${n}/${kf}」(${seg.duration.toFixed(1)}s)…`, true);
    send({
      type: "play_pose",
      name: n,
      keyframe: kf,
      auto_segment: true,
    });
  }

  function addKeyframeFromSliders() {
    onPoseNameChanged();
    const name = normalizePoseName($("poseName").value);
    if (!name) {
      setMsg("请先填写新手势名（勿沿用上一手势；可点「新建手势」）");
      return;
    }
    if (!isInTeach()) {
      setMsg("请先点「开始示教」，再＋添加中间态（否则可能保存到非示教目标）");
      return;
    }
    let kf = normalizePoseName($("poseKeyframe") && $("poseKeyframe").value);
    if (!kf) {
      kf = suggestNextKfName(findPoseSummary(name));
      if ($("poseKeyframe")) $("poseKeyframe").value = kf;
    }
    selectPose(name, { resetKeyframe: false, fillMeta: false });
    selectKeyframe(kf);
    expandedPoses.add(name);
    setMsg(`追加「${name} / ${kf}」…`, true);
    send({
      type: "save_pose",
      name,
      keyframe: kf,
      duration: Number($("poseDur").value || 0.6),
      description: $("poseDesc").value || "",
      use_targets: $("chkUseTargets").checked,
      advance_keyframe: true,
    });
  }

  function updateKeyframeEditor(summary) {
    const empty = $("keyframeEmpty");
    const segs = keyframeSegments(summary);
    updateTimelineChrome(summary, null);
    setTimelineEnabled(segs.length > 0);
    if (empty) {
      if (!summary || !summary.name) {
        empty.classList.remove("hidden");
        empty.textContent = "先填写/选中手势名，再示教添加中间态。关键帧见下方姿态库展开。";
      } else if (!segs.length) {
        empty.classList.remove("hidden");
        empty.textContent = `手势「${summary.name}」尚无关键帧。示教摆好姿 →「＋ 添加」。`;
      } else {
        empty.classList.add("hidden");
      }
    }
  }

  /** @deprecated 兼容旧调用名 */
  function renderKeyframeList(summary) {
    updateKeyframeEditor(summary);
  }

  function bindJointControls(slider, numEl, stepMinus, stepPlus, opts) {
    const readonly = Boolean(opts && opts.readonly);
    const idx = opts.index;
    if (readonly) {
      slider.disabled = true;
      numEl.disabled = true;
      stepMinus.disabled = true;
      stepPlus.disabled = true;
      const title = (opts && opts.title) || "耦合/只读：随指中";
      slider.title = title;
      numEl.title = title;
      return;
    }

    const endDrag = (ev) => {
      if (ev && slider.hasPointerCapture && slider.hasPointerCapture(ev.pointerId)) {
        try {
          slider.releasePointerCapture(ev.pointerId);
        } catch (_) {
          /* ignore */
        }
      }
      if (!dragging) return;
      dragging = false;
      setJointLocal(idx, slider.value, { push: true, flush: true });
    };

    slider.addEventListener("pointerdown", (ev) => {
      dragging = true;
      try {
        slider.setPointerCapture(ev.pointerId);
      } catch (_) {
        /* ignore */
      }
    });
    slider.addEventListener("pointerup", endDrag);
    slider.addEventListener("pointercancel", endDrag);
    slider.addEventListener("lostpointercapture", () => {
      if (dragging) {
        dragging = false;
        setJointLocal(idx, slider.value, { push: true, flush: true });
      }
    });
    slider.addEventListener("input", () => {
      dragging = true;
      if (numEl) numEl.value = formatDisplay(idx, Number(slider.value));
      syncCoupledTipsFromDom();
      send({ type: "set_joints", positions: readPositions() });
    });
    // 键盘：←/→ ±1，Shift+←/→ ±5（range 原生为 step，这里放大粗调）
    slider.addEventListener("keydown", (ev) => {
      let delta = 0;
      if (ev.key === "ArrowLeft" || ev.key === "ArrowDown") delta = -STEP_FINE;
      else if (ev.key === "ArrowRight" || ev.key === "ArrowUp") delta = STEP_FINE;
      else if (ev.key === "PageDown") delta = -STEP_COARSE;
      else if (ev.key === "PageUp") delta = STEP_COARSE;
      else if (ev.key === "Home") {
        ev.preventDefault();
        setJointLocal(idx, JOINT_MIN, { flush: true });
        return;
      } else if (ev.key === "End") {
        ev.preventDefault();
        setJointLocal(idx, JOINT_MAX, { flush: true });
        return;
      } else return;
      if (ev.shiftKey) delta = delta < 0 ? -STEP_COARSE : STEP_COARSE;
      ev.preventDefault();
      setJointLocal(idx, Number(slider.value) + delta, { flush: true });
    });
    slider.addEventListener("wheel", (ev) => {
      // 仅焦点在滑条时滚轮微调，避免刷页面时误触
      if (document.activeElement !== slider) return;
      ev.preventDefault();
      const dir = ev.deltaY > 0 ? -1 : 1;
      const step = ev.shiftKey ? STEP_COARSE : STEP_FINE;
      setJointLocal(idx, Number(slider.value) + dir * step, { flush: true });
    }, { passive: false });

    const commitNum = ({ flush = true } = {}) => {
      const hw = displayToHw(idx, numEl.value);
      if (hw == null) {
        numEl.value = formatDisplay(idx, Number(slider.value));
        return;
      }
      setJointLocal(idx, hw, { push: true, flush });
      numEl.value = formatDisplay(idx, hw);
    };
    numEl.addEventListener("focus", () => {
      editingValue = true;
      numEl.select();
    });
    numEl.addEventListener("blur", () => {
      editingValue = false;
      commitNum({ flush: true });
    });
    numEl.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        commitNum({ flush: true });
        numEl.blur();
        return;
      }
      if (ev.key === "Escape") {
        ev.preventDefault();
        numEl.value = formatDisplay(idx, Number(slider.value));
        editingValue = false;
        numEl.blur();
        return;
      }
      let delta = 0;
      if (ev.key === "ArrowUp") delta = displayStep(!ev.shiftKey);
      else if (ev.key === "ArrowDown") delta = -displayStep(!ev.shiftKey);
      else return;
      ev.preventDefault();
      // 显示单位步进：先转当前显示值再加减，再写回 hw
      const curDisp = Number(numEl.value);
      const baseDisp = Number.isFinite(curDisp)
        ? curDisp
        : hwToDisplay(idx, Number(slider.value));
      const hw = displayToHw(idx, baseDisp + delta);
      if (hw == null) return;
      const next = setJointLocal(idx, hw, { flush: true });
      if (next != null) numEl.value = formatDisplay(idx, next);
    });
    numEl.addEventListener("change", () => commitNum({ flush: true }));
    numEl.addEventListener("wheel", (ev) => {
      if (document.activeElement !== numEl) return;
      ev.preventDefault();
      const dir = ev.deltaY > 0 ? -1 : 1;
      const step = displayStep(!ev.shiftKey);
      const curDisp = Number(numEl.value);
      const baseDisp = Number.isFinite(curDisp)
        ? curDisp
        : hwToDisplay(idx, Number(slider.value));
      const hw = displayToHw(idx, baseDisp + step * dir);
      if (hw == null) return;
      const next = setJointLocal(idx, hw, { flush: true });
      if (next != null) numEl.value = formatDisplay(idx, next);
    }, { passive: false });

    stepMinus.addEventListener("click", () => {
      setJointLocal(idx, Number(slider.value) - STEP_FINE, { flush: true });
    });
    stepPlus.addEventListener("click", () => {
      setJointLocal(idx, Number(slider.value) + STEP_FINE, { flush: true });
    });
  }

  function makeJointCard(j, values) {
    const card = document.createElement("div");
    card.className =
      "joint-card role-" +
      (j.role || "joint") +
      (j.reserved ? " reserved" : "") +
      (j.coupled ? " coupled" : "");
    if (j.reserved && !showReserved) {
      card.classList.add("hidden");
    }
    // 显式 flex order，避免排序/缓存导致指尖跑到底
    const orderMap = { tip: 1, mid: 2, base: 3, rot: 4, abd: 5, joint: 6, rsv: 9 };
    card.style.order = String(orderMap[j.role] || 6);
    const head = document.createElement("div");
    head.className = "jname";
    const tag = j.coupled
      ? `<em class="couple-tag" title="随指中 · ros2_control 不可独立控制">耦合/只读</em>`
      : "";
    head.innerHTML = `<b>${j.short}</b><span>${j.name}</span>${tag}`;
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(JOINT_MIN);
    input.max = String(JOINT_MAX);
    input.step = String(STEP_FINE);
    input.id = `j${j.index}`;
    input.setAttribute("aria-label", `${j.short} ${j.name}`);
    const initial = values[j.index] != null ? values[j.index] : 128;
    input.value = initial;

    const ctrl = document.createElement("div");
    ctrl.className = "jctrl";
    const minus = document.createElement("button");
    minus.type = "button";
    minus.className = "jstep";
    minus.textContent = "−";
    minus.title = "−1（Shift 点滑条/键盘为 ±5）";
    minus.setAttribute("aria-label", `${j.short} 减 1`);
    const num = document.createElement("input");
    num.type = "number";
    num.className = "jnum";
    num.id = `jn${j.index}`;
    const attrs = numInputAttrs();
    if (attrs.min != null) {
      num.min = attrs.min;
      num.max = attrs.max;
    }
    num.step = attrs.step;
    num.value = formatDisplay(j.index, initial);
    num.inputMode = "decimal";
    num.title = attrs.title;
    num.setAttribute("aria-label", `${j.short} 关节值`);
    const plus = document.createElement("button");
    plus.type = "button";
    plus.className = "jstep";
    plus.textContent = "+";
    plus.title = "+1 (0–255)";
    plus.setAttribute("aria-label", `${j.short} 加 1`);
    ctrl.append(minus, num, plus);

    bindJointControls(input, num, minus, plus, {
      index: j.index,
      readonly: j.coupled,
      title: j.coupledLabel || "耦合/只读：随指中",
    });
    card.append(head, input, ctrl);
    return card;
  }

  function buildHand(values) {
    const layout = window.HandLayout.buildHandLayout(
      jointNames,
      handSide,
      handType,
      coupledJoints
    );
    $("handHint").textContent = layout.hint;
    $("handHint").dataset.baseHint = layout.hint;
    canvas.innerHTML = "";
    canvas.className =
      "hand-canvas side-" + layout.side + (layout.compact ? " compact" : "");

    layout.columns.forEach((col) => {
      const colEl = document.createElement("div");
      colEl.className = "finger-col " + col.finger;
      if (!col.joints.length) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.style.fontSize = "0.75rem";
        empty.style.textAlign = "center";
        empty.textContent = "—";
        colEl.appendChild(empty);
      } else {
        col.joints.forEach((j) => colEl.appendChild(makeJointCard(j, values)));
      }
      const title = document.createElement("div");
      title.className = "finger-title";
      title.textContent = col.title;
      colEl.appendChild(title);
      canvas.appendChild(colEl);
    });

    if (layout.other.length) {
      const row = document.createElement("div");
      row.className = "other-row";
      layout.other.forEach((j) => row.appendChild(makeJointCard(j, values)));
      canvas.appendChild(row);
    }
    syncCoupledTipsFromDom();
  }

  function renderPoseList(poses) {
    poseList.innerHTML = "";
    const q = String(poseFilter || "").trim().toLowerCase();
    const visible = q
      ? poses.filter((raw) => normalizePoseSummary(raw).name.includes(q))
      : poses;
    visible.forEach((raw) => {
      const summary = normalizePoseSummary(raw);
      const name = summary.name;
      if (!name) return;
      const segs = keyframeSegments(summary);
      const expanded = expandedPoses.has(name);
      const li = document.createElement("li");
      li.className = "pose-node";
      li.dataset.name = name;
      if (name === selected || name === normalizePoseName(editingName)) {
        li.classList.add("active");
      }

      const row = document.createElement("div");
      row.className = "pose-row";

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "pose-toggle";
      toggle.textContent = segs.length ? (expanded ? "▼" : "▶") : "·";
      toggle.disabled = !segs.length;
      toggle.title = segs.length ? (expanded ? "收起关键帧" : "展开关键帧") : "无关键帧";
      toggle.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!segs.length) return;
        if (expandedPoses.has(name)) expandedPoses.delete(name);
        else expandedPoses.add(name);
        renderPoseList(poseSummaries);
      });

      const label = document.createElement("span");
      label.className = "pose-name";
      label.dataset.name = name;
      label.textContent = name;
      if (summary.keyframe_count > 0) {
        const badge = document.createElement("span");
        badge.className = "pose-badge";
        badge.textContent = `${summary.keyframe_count}帧`;
        badge.title = "关键帧数量（不是名称后缀）";
        label.appendChild(badge);
      }
      label.title = "点击选中该手势；展开查看关键帧";

      const acts = document.createElement("div");
      acts.className = "acts";
      const bPlay = document.createElement("button");
      bPlay.type = "button";
      bPlay.textContent = "播放";
      bPlay.title = "整段 AUTO 回放";
      bPlay.dataset.poseMotion = "true"; bPlay.disabled = !!sharedState?.observe_only;
      const bEdit = document.createElement("button");
      bEdit.type = "button";
      bEdit.textContent = "编辑";
      bEdit.title = "载入到滑条并进入示教";

      row.addEventListener("click", (e) => {
        if (e.target.closest("button")) return;
        selectPose(name, { announce: true, resetKeyframe: true });
      });
      bPlay.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectPose(name, { resetKeyframe: false });
        playPoseByName(name);
      });
      bEdit.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectPose(name, { resetKeyframe: true });
        send({ type: "load_pose", name });
      });
      bPlay.addEventListener("click", (e) => e.stopPropagation());
      bEdit.addEventListener("click", (e) => e.stopPropagation());

      acts.append(bPlay, bEdit);
      row.append(toggle, label, acts);
      li.appendChild(row);

      if (expanded && segs.length) {
        const child = document.createElement("ul");
        child.className = "kf-children";
        segs.forEach((seg) => {
          const kfLi = document.createElement("li");
          kfLi.className = "kf-node";
          kfLi.dataset.keyframe = seg.name;
          if (name === selected && seg.name === selectedKeyframe) {
            kfLi.classList.add("active");
          }
          const main = document.createElement("div");
          main.className = "kf-main";
          const idx = document.createElement("span");
          idx.className = "kf-idx";
          idx.textContent = String(seg.index + 1);
          const kfLabel = document.createElement("span");
          kfLabel.textContent = `${seg.name} · ${seg.duration.toFixed(1)}s`;
          main.append(idx, kfLabel);
          const kfActs = document.createElement("div");
          kfActs.className = "kf-acts";
          const bKfPlay = document.createElement("button");
          bKfPlay.type = "button";
          bKfPlay.textContent = "回放";
          bKfPlay.title = "本帧独立 0→1";
          bKfPlay.dataset.poseMotion = "true"; bKfPlay.disabled = !!sharedState?.observe_only;
          const bKfLoad = document.createElement("button");
          bKfLoad.type = "button";
          bKfLoad.textContent = "载入";
          const bKfHold = document.createElement("button");
          bKfHold.type = "button";
          bKfHold.textContent = "定点";
          bKfHold.title = "停在本帧 progress=0";
          bKfHold.dataset.poseMotion = "true"; bKfHold.disabled = !!sharedState?.observe_only;
          const bKfDel = document.createElement("button");
          bKfDel.type = "button";
          bKfDel.className = "danger";
          bKfDel.textContent = "删";
          const bUp = document.createElement("button");
          bUp.type = "button";
          bUp.textContent = "↑";
          bUp.title = "上移";
          bUp.disabled = seg.index <= 0;
          const bDown = document.createElement("button");
          bDown.type = "button";
          bDown.textContent = "↓";
          bDown.title = "下移";
          bDown.disabled = seg.index >= segs.length - 1;
          const bRename = document.createElement("button");
          bRename.type = "button";
          bRename.textContent = "改名";
          bKfPlay.addEventListener("click", (e) => {
            e.stopPropagation();
            selectPose(name, { resetKeyframe: false, fillMeta: false });
            selectKeyframe(seg.name);
            playKeyframeIndependent(name, seg.name);
          });
          bKfLoad.addEventListener("click", (e) => {
            e.stopPropagation();
            selectPose(name, { resetKeyframe: false, fillMeta: false });
            selectKeyframe(seg.name);
            send({ type: "load_pose", name, keyframe: seg.name });
          });
          bKfHold.addEventListener("click", (e) => {
            e.stopPropagation();
            selectPose(name, { resetKeyframe: false, fillMeta: false });
            selectKeyframe(seg.name);
            scrubPose(name, seg.name, 0, { force: true });
            setGlobalSlider(mapLocalToGlobal(summary, seg.name, 0), {
              localLabel: 0,
            });
            setMsg(`停在「${name}/${seg.name}」起点`, true);
          });
          bKfDel.addEventListener("click", (e) => {
            e.stopPropagation();
            if (!confirm(`删除「${name} / ${seg.name}」？`)) return;
            selectPose(name, { resetKeyframe: false, fillMeta: false });
            send({ type: "delete_keyframe", name, keyframe: seg.name });
          });
          bUp.addEventListener("click", (e) => {
            e.stopPropagation();
            send({
              type: "move_keyframe",
              name,
              keyframe: seg.name,
              delta: -1,
            });
          });
          bDown.addEventListener("click", (e) => {
            e.stopPropagation();
            send({
              type: "move_keyframe",
              name,
              keyframe: seg.name,
              delta: 1,
            });
          });
          bRename.addEventListener("click", (e) => {
            e.stopPropagation();
            const next = prompt(`重命名关键帧「${seg.name}」为：`, seg.name);
            if (next == null) return;
            const nn = normalizePoseName(next);
            if (!nn || nn === seg.name) return;
            send({
              type: "rename_keyframe",
              name,
              keyframe: seg.name,
              new_keyframe: nn,
            });
          });
          kfActs.append(bKfPlay, bKfHold, bKfLoad, bUp, bDown, bRename, bKfDel);
          kfLi.append(main, kfActs);
          kfLi.addEventListener("click", (e) => {
            if (e.target.closest("button")) return;
            selectPose(name, {
              resetKeyframe: false,
              announce: false,
              fillMeta: false,
            });
            selectKeyframe(seg.name);
            setMsg(`已选「${name} / ${seg.name}」`, true);
          });
          child.appendChild(kfLi);
        });
        li.appendChild(child);
      }

      poseList.appendChild(li);
    });
    updateKeyframeEditor(findPoseSummary(selected));
  }

  $("btnTeach").onclick = () => send({ type: "enter_teach" });
  $("btnExit").onclick = () => send({ type: "exit_teach" });
  $("btnReinit").onclick = () => {
    if (!confirm("执行重新初始化（init）？将退出示教模式。")) return;
    send({ type: "reinit" });
  };
  $("btnSync").onclick = () => send({ type: "sync_feedback" });
  displayUnit = loadStoredDisplayUnit();
  if ($("displayUnit")) {
    syncDisplayUnitSelect();
    $("displayUnit").addEventListener("change", () => {
      const next = String($("displayUnit").value || "hw255");
      if (!displayAvailable.includes(next)) {
        setMsg("当前无 URDF Converter，仅支持 0–255 显示", false);
        $("displayUnit").value = "hw255";
        displayUnit = "hw255";
        storeDisplayUnit(displayUnit);
        refreshNumDisplays();
        return;
      }
      displayUnit = next;
      storeDisplayUnit(displayUnit);
      refreshNumDisplays();
      const label =
        displayUnit === "rad" ? "弧度 rad" : displayUnit === "deg" ? "角度 °" : "0–255";
      setMsg(`显示单位：${label}（控制仍为 0–255）`, true);
    });
  }
  if ($("btnNewPose")) {
    $("btnNewPose").onclick = () => beginNewGesture();
  }
  if ($("poseFilter")) {
    $("poseFilter").addEventListener("input", (e) => {
      poseFilter = String(e.target.value || "");
      renderPoseList(poseSummaries);
    });
  }
  if ($("poseName")) {
    $("poseName").addEventListener("change", () => onPoseNameChanged());
    $("poseName").addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        onPoseNameChanged();
        $("poseName").blur();
      }
    });
  }
  $("btnSave").onclick = () => {
    onPoseNameChanged();
    const name = normalizePoseName($("poseName").value);
    if (!name) {
      setMsg("请先填写手势名（不回落到上一编辑名）");
      return;
    }
    send({
      type: "save_pose",
      name,
      keyframe: ($("poseKeyframe") && $("poseKeyframe").value.trim()) || "",
      duration: Number($("poseDur").value || 0.6),
      description: $("poseDesc").value || "",
      use_targets: $("chkUseTargets").checked,
    });
  };
  $("btnPlay").onclick = () => {
    stopKeyframeAutoPlay();
    playPoseByName(resolvePlayName());
  };
  if ($("btnAddKf")) {
    $("btnAddKf").onclick = () => addKeyframeFromSliders();
  }
  if ($("btnSuggestKf")) {
    $("btnSuggestKf").onclick = () => {
      const n = resolvePlayName() || normalizePoseName($("poseName").value);
      const next = suggestNextKfName(findPoseSummary(n));
      if ($("poseKeyframe")) $("poseKeyframe").value = next;
      selectKeyframe(next, { fillDur: false });
      setMsg(`建议帧名「${next}」`, true);
    };
  }
  if ($("btnStopPlay")) {
    $("btnStopPlay").onclick = () => {
      stopKeyframeAutoPlay();
      progressDragging = false;
      send({ type: "stop_playback" });
    };
  }
  if ($("progressSlider")) {
    const slider = $("progressSlider");
    const emit = () => emitGlobalScrub(Number(slider.value) / 1000);
    slider.addEventListener("pointerdown", () => {
      if (slider.disabled) return;
      stopKeyframeAutoPlay();
      progressDragging = true;
    });
    slider.addEventListener("pointerup", () => {
      if (slider.disabled) return;
      progressDragging = false;
      emit();
      flushPendingScrub();
    });
    slider.addEventListener("input", () => {
      if (slider.disabled) return;
      const g = Number(slider.value) / 1000;
      const n = resolvePlayName();
      const summary = findPoseSummary(n);
      const mapped = mapGlobalProgress(summary, g);
      if (mapped) {
        setGlobalSlider(g, { localLabel: mapped.local });
        updateTimelineChrome(summary, mapped);
      } else if ($("progressVal")) {
        $("progressVal").textContent = `${Math.round(g * 100)}%`;
      }
      if (progressDragging) emit();
    });
  }
  $("btnDelete").onclick = () => {
    const name = normalizePoseName($("poseName").value || selected);
    if (!name) return;
    if (!confirm(`删除手势 ${name}?`)) return;
    send({ type: "delete_pose", name });
  };
  $("btnClearEdit").onclick = () => {
    send({ type: "clear_edit" });
    editingName = "";
    const editBadge = $("editBadge");
    const editHint = $("editHint");
    if (editBadge) editBadge.classList.add("hidden");
    if (editHint) editHint.classList.add("hidden");
    refreshPoseListActive();
    setMsg("已取消编辑标记（手势选中与滑条保留）", true);
  };
  $("chkShowRsv").onchange = (e) => {
    showReserved = Boolean(e.target.checked);
    layoutKey = "";
    buildHand(readPositions());
  };
  $("chkReturnTeach").onchange = (e) => {
    send({
      type: "set_prefs",
      return_to_teach_after_playback: Boolean(e.target.checked),
    });
  };

  async function exportBundleDownload() {
    try {
      setMsg("正在导出…", true);
      const res = await fetch(withSide("/api/pose_bundle/export"));
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setMsg(err.message || `导出失败 HTTP ${res.status}`);
        return;
      }
      const count = res.headers.get("X-Pose-Count") || "?";
      if (count === "0" || res.headers.get("X-Pose-Empty") === "1") {
        if (!confirm("当前库为空，仍要下载空 bundle 吗？")) {
          setMsg("已取消空库导出", true);
          return;
        }
      }
      const blob = await res.blob();
      let filename = `${productHandType}_${handSide}_poses.pose_bundle.json`;
      const cd = res.headers.get("Content-Disposition") || "";
      const m = /filename=\"([^\"]+)\"/.exec(cd);
      if (m) filename = m[1];
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setMsg(`已下载 ${filename}（${count} 条）`, true);
    } catch (err) {
      setMsg(`导出失败: ${err && err.message ? err.message : err}`);
    }
  }

  async function importBundleFile(file) {
    if (!file) return;
    const strategy = ($("importStrategy") && $("importStrategy").value) || "skip";
    const verify =
      !$("chkVerifyChecksum") || Boolean($("chkVerifyChecksum").checked);
    try {
      setMsg(`正在导入（${strategy}）…`, true);
      const text = await file.text();
      const qs = new URLSearchParams({
        strategy,
        verify_checksum: verify ? "true" : "false",
      });
      const res = await fetch(withSide(`/api/pose_bundle/import?${qs}`), {
        method: "POST",
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: text,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.success === false) {
        const hint =
          data.filter_hand_type || data.filter_hand_side
            ? `（面板过滤 ${data.filter_hand_type || "?"}/${data.filter_hand_side || "?"}）`
            : "";
        setMsg((data.message || `导入失败 HTTP ${res.status}`) + hint);
        return;
      }
      if (Array.isArray(data.poses)) {
        posesKey = "";
        renderPoseList(data.poses);
      }
      send({ type: "list_poses" });
      setMsg(data.message || "导入完成", true);
    } catch (err) {
      setMsg(`导入失败: ${err && err.message ? err.message : err}`);
    }
  }

  async function importFactoryPresets() {
    let strategy = ($("importStrategy") && $("importStrategy").value) || "skip";
    if (strategy === "rename") {
      setMsg("出厂预设不支持「重命名」，已按「跳过已有」处理", true);
      strategy = "skip";
    }
    const overwrite = strategy === "overwrite";
    if (
      !confirm(
        overwrite
          ? "将用出厂预设覆盖同名 preset_*（标定/配置更新后应选此项），继续？"
          : "导入可规划静态原语终态为 preset_*（已存在则跳过；若姿态不对请改选「覆盖」），继续？"
      )
    ) {
      return;
    }
    try {
      setMsg("正在写入出厂预设…", true);
      const res = await fetch(
        withSide(`/api/pose_bundle/import_factory?overwrite=${overwrite ? "true" : "false"}`),
        { method: "POST" }
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.success === false) {
        setMsg(data.message || `出厂预设失败 HTTP ${res.status}`);
        return;
      }
      if (Array.isArray(data.poses)) {
        posesKey = "";
        renderPoseList(data.poses);
      }
      send({ type: "list_poses" });
      setMsg(data.message || `已写入 ${data.count || 0} 条`, true);
    } catch (err) {
      setMsg(`出厂预设失败: ${err && err.message ? err.message : err}`);
    }
  }

  $("btnExportBundle").onclick = () => exportBundleDownload();
  $("btnImportBundle").onclick = () => {
    const input = $("importBundleFile");
    if (input) input.click();
  };
  $("importBundleFile").onchange = (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    importBundleFile(file);
  };
  $("btnImportFactory").onclick = () => importFactoryPresets();

  void refreshBuiltins().finally(connect);
  setInterval(() => { if (!document.hidden) void refreshBuiltins(); }, 750);
})();
