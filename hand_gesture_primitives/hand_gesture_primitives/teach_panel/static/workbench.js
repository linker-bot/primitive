/* Standalone host UI. Motion authority, revisions and timeouts live in Python. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const copy = value => structuredClone(value);
  const requestedSide = new URLSearchParams(location.search).get('side');
  const state = { snapshot: null, side: '', catalog: null, library: null, menu: [], menuDirty: false, group: 'default', fresh: false };
  const clientId = globalThis.crypto?.randomUUID?.() || `browser_${Date.now()}_${Math.random()}`;
  let generation = 0, loading = false, busy = false, stopping = false, signature = '', pollBusy = false;
  let draft = null, frame = 0, baseline = '', token = '', editingSide = '', editorLocked = false;
  let sequence = 0, pending = {}, liveActive = false, liveSending = false, liveGeneration = 0, lastLive = 0, needsStop = false;
  let stopBarrier = Promise.resolve(), held = null, heldOwner = null, holdSequence = 0, holdSending = false;

  function message(text, error = false, editor = false) {
    const el = $(editor ? 'editor-message' : 'notice');
    el.textContent = text; el.hidden = !text; el.className = `notice${error ? ' error' : ''}`;
  }
  async function api(path, body, method = 'POST') {
    const result = await fetch(`/api/workbench/${path}`, { method: body === undefined ? 'GET' : method, headers: { 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
    let value; try { value = await result.json(); } catch { throw new Error(`服务响应无效 (${result.status})`); }
    if (!result.ok) throw new Error(value.detail?.reason || value.detail?.message || value.detail || value.message || `请求失败 (${result.status})`);
    return value;
  }
  const canMove = () => state.fresh && !state.snapshot?.observe_only;
  const moveLocked = () => !canMove() || !!state.snapshot?.input_pause?.paused;
  const sideBody = body => ({ side: state.side, ...body });
  function button(text, handler, disabled = false, aria = '') {
    const el = document.createElement('button'); el.type = 'button'; el.textContent = text; el.disabled = disabled;
    if (aria) el.setAttribute('aria-label', aria);
    el.addEventListener('click', handler); return el;
  }
  async function run(fn, editor = false) {
    if (busy) return;
    busy = true; updateDisabled();
    try { await fn(); } catch (error) { message(error.message, true, editor); }
    finally { busy = false; updateDisabled(); void refreshState(); }
  }
  function updateDisabled() {
    $('side').disabled = busy || !!token;
    $('stop').disabled = stopping || !canMove(); $('speed').disabled = busy || !canMove() || state.catalog?.motion_controls !== 1;
    $('new-pose').disabled = busy || !state.fresh;
    $('resume').disabled = busy || !state.fresh || !!state.snapshot?.input_pause?.editors || !state.snapshot?.input_pause?.ready;
    $('save-menu').disabled = busy || !state.fresh || !state.menuDirty;
    if (!draft) return;
    $('edit-fields').disabled = busy || editorLocked || !state.fresh;
    $('follow').disabled = editorLocked || !canMove() || state.catalog?.motion_controls !== 1;
    for (const id of ['save-pose', 'save-as', 'delete-pose']) $(id).disabled = busy || editorLocked || !state.fresh;
    for (const id of ['preview-frame', 'preview-all', 'scrub']) $(id).disabled = busy || editorLocked || !canMove() || !!invalidDraft();
    $('editor-stop').disabled = stopping || !canMove(); $('save-pose').disabled ||= !!invalidDraft();
    $('delete-pose').hidden = !draft.revision;
  }
  async function loadLibrary() {
    const selected = state.side, requestGeneration = generation;
    const data = await api(`library?side=${encodeURIComponent(selected)}`);
    if (requestGeneration === generation && selected === state.side) state.library = data;
    return data;
  }
  async function refreshCatalog(full = false) {
    if (!state.side || loading) return;
    loading = true; const selected = state.side, requestGeneration = generation;
    try {
      if (full) await loadLibrary();
      const data = await api(`catalog?side=${encodeURIComponent(selected)}`);
      if (selected !== state.side || requestGeneration !== generation) return;
      state.catalog = data;
      if (!state.menuDirty) { state.menu = [...data.menu]; renderMenu(); }
      renderCatalog(); updateDisabled();
    } catch (error) { message(error.message, true); }
    finally { loading = false; }
  }
  async function refreshState() {
    if (pollBusy) return; pollBusy = true;
    try {
      const snapshot = await api('state');
      state.snapshot = snapshot; state.fresh = true;
      $('connection').textContent = '已连接';
      const hands = Object.entries(snapshot.hands);
      if ($('side').options.length !== hands.length) {
        $('side').replaceChildren(...hands.map(([side, value]) => new Option(`${side === 'left' ? '左手' : '右手'} · ${value.model.toUpperCase()}`, side)));
      }
      if (!state.side || !snapshot.hands[state.side]) { state.side = snapshot.hands[requestedSide] ? requestedSide : snapshot.hands[snapshot.default_side] ? snapshot.default_side : hands[0]?.[0] || ''; $('side').value = state.side; void refreshCatalog(true); }
      const hand = snapshot.hands[state.side];
      $('legacy-link').href = `/?side=${encodeURIComponent(state.side)}`;
      $('backend').textContent = snapshot.backend === 'mock' ? 'Mock 模拟反馈' : 'ROS 原语节点';
      if (document.activeElement !== $('speed')) $('speed').value = snapshot.primitive_speed;
      $('pause').hidden = !snapshot.input_pause?.paused; $('readonly').hidden = !snapshot.observe_only;
      if (hand) {
        $('active').textContent = hand.active || '手势待命';
        const labels = { idle: '待命', running: '执行中', paused: '已暂停，续按继续', completed: '轨迹到达终点', live: '编辑实时跟随' };
        $('runtime').textContent = `${labels[hand.playback_state] || hand.playback_state || ''}${hand.keyframe ? ` · ${hand.keyframe}` : ''}`;
        $('progress').value = Math.max(0, hand.overall_progress ?? 0);
        $('settle').textContent = ({ settling: '等待机械手到位', settled: '反馈已确认到位', settle_timeout: '到位超时，请检查反馈', failed: '执行失败' })[hand.settle_status] || hand.hold_reason || '';
        document.querySelectorAll('.gesture').forEach(el => el.classList.toggle('active', el.dataset.name === hand.active));
      }
      updateDisabled();
    } catch (error) {
      state.fresh = false; $('connection').textContent = '连接中断';
      if (draft) { $('follow').checked = false; void haltLive().catch(() => {}); }
      void releaseHold(); updateDisabled();
    } finally { pollBusy = false; }
  }
  function renderCatalog() {
    const rows = (state.catalog?.primitives || []).filter(item => !item.template);
    const search = $('search').value.toLowerCase();
    const items = rows.filter(item => item.group === state.group && (!$('available').checked || item.available) && `${item.name} ${item.label} ${item.description || ''}`.toLowerCase().includes(search));
    document.querySelectorAll('[data-group]').forEach(el => {
      el.setAttribute('aria-selected', String(el.dataset.group === state.group));
      const names = { default: '默认原语', perception: '带感知原语', custom: '自定义原语' };
      const group = rows.filter(item => item.group === el.dataset.group);
      el.textContent = `${names[el.dataset.group]} ${group.filter(item => item.available).length}/${group.length}`;
    });
    const groupName = { default: '默认原语', perception: '带感知原语', custom: '自定义原语' }[state.group];
    const total = rows.filter(item => item.group === state.group).length;
    $('filter-state').textContent = `${groupName}：显示 ${items.length}/${total}${search ? ` · 筛选“${search}”` : ''}${$('available').checked ? ' · 仅可用' : ''}`;
    $('clear-filter').hidden = !search && !$('available').checked;
    const newSignature = JSON.stringify([items, moveLocked(), state.group]);
    if (signature === newSignature) return; signature = newSignature;
    $('catalog').replaceChildren(...items.map(item => {
      const card = document.createElement('div'); card.className = 'gesture'; card.dataset.name = item.name;
      const title = document.createElement('strong'); title.textContent = item.label;
      const badge = document.createElement('span'); badge.className = `badge${item.available ? '' : ' unavailable'}`; badge.textContent = item.available ? '可用' : '不可用'; title.append(badge);
      const name = document.createElement('span'); name.className = 'name'; name.textContent = item.name;
      const description = document.createElement('span'); description.className = 'reason'; description.textContent = item.reason || item.description || '';
      const controls = document.createElement('div'); controls.className = 'row';
      controls.append(button('执行', () => item.available ? run(async () => { await api('select', sideBody({ name: item.name })); }) : message(`${item.label} 当前不可用：${item.reason}`, true), item.available && moveLocked(), `${item.label} · ${item.available ? '可用' : '不可用'}`));
      if (item.supports_hold) {
        const hold = button('按住推进', () => {}, !item.available || moveLocked(), `按住推进 ${item.label}`); hold.className = 'hold';
        hold.addEventListener('pointerdown', event => { event.preventDefault(); hold.setPointerCapture(event.pointerId); void beginHold(item.name, hold, 'pointer'); });
        hold.addEventListener('lostpointercapture', () => { if (heldOwner?.element === hold) void releaseHold(); });
        hold.addEventListener('keydown', event => { if ([' ', 'Enter'].includes(event.key)) { event.preventDefault(); if (!event.repeat) void beginHold(item.name, hold, 'keyboard'); } });
        hold.addEventListener('blur', () => { if (heldOwner?.element === hold && heldOwner.input === 'keyboard') void releaseHold(); });
        hold.addEventListener('keyup', event => { if ([' ', 'Enter'].includes(event.key)) { event.preventDefault(); void releaseHold(); } });
        controls.append(hold);
      }
      if (item.group === 'custom') controls.append(button('编辑', () => run(() => openEditor(item.name.slice(11))), !state.fresh, `编辑 ${item.label}`));
      card.append(title, name, description, controls); return card;
    }));
    if (!items.length) { const p = document.createElement('p'); p.className = 'muted'; p.textContent = search || $('available').checked ? '当前筛选下没有匹配的原语，请清除筛选或切换分类。' : '当前分类还没有原语。'; $('catalog').append(p); }
  }
  async function beginHold(name, element, input) {
    if (moveLocked() || held) return;
    const current = held = { name, side: state.side };
    heldOwner = { element, input };
    try { await sendHold(true); } catch (error) { message(error.message, true); if (held === current) await releaseHold(); }
  }
  async function sendHold(running) {
    if (!held || holdSending) return;
    holdSending = true;
    try { await api('held', { ...held, running, client_id: clientId, sequence: ++holdSequence }); }
    finally { holdSending = false; }
  }
  async function releaseHold() {
    const previous = held; held = null; heldOwner = null;
    if (!previous) return;
    try { await api('held', { ...previous, running: false, client_id: clientId, sequence: ++holdSequence }); }
    catch (error) { message(error.message, true); }
  }
  window.addEventListener('pointerup', () => void releaseHold());
  window.addEventListener('pointercancel', () => void releaseHold());
  function suspendContinuousInput() {
    void releaseHold(); $('follow').checked = false; void haltLive().catch(() => {});
  }
  window.addEventListener('blur', suspendContinuousInput);
  setInterval(() => {
    if (!held) return;
    // Removing a focused button need not emit blur. Check authority before every
    // renewal so catalog rerenders cannot retain a detached input owner.
    if (!heldOwner?.element.isConnected || heldOwner.element.disabled ||
        (heldOwner.input === 'keyboard' && document.activeElement !== heldOwner.element)) {
      void releaseHold(); return;
    }
    const current = held;
    if (!holdSending) void sendHold(true).catch(error => { message(error.message, true); if (held === current) void releaseHold(); });
  }, 100);

  function renderMenu() {
    const rows = state.catalog?.primitives || [], label = name => rows.find(item => item.name === name)?.label || name;
    $('menu').replaceChildren(...state.menu.map((name, index) => {
      const li = document.createElement('li'), row = document.createElement('div'), span = document.createElement('span'); row.className = 'row'; span.textContent = label(name); row.append(span);
      const change = fn => { fn(); state.menuDirty = true; renderMenu(); updateDisabled(); };
      row.append(button('↑', () => change(() => [state.menu[index - 1], state.menu[index]] = [state.menu[index], state.menu[index - 1]]), index === 0, `前移 ${label(name)}`), button('↓', () => change(() => [state.menu[index + 1], state.menu[index]] = [state.menu[index], state.menu[index + 1]]), index === state.menu.length - 1, `后移 ${label(name)}`), button('×', () => change(() => state.menu.splice(index, 1)), false, `移除 ${label(name)}`)); li.append(row); return li;
    }));
    $('menu-add').replaceChildren(new Option('选择原语…', ''), ...rows.filter(item => item.supported && !item.template && !state.menu.includes(item.name)).map(item => new Option(label(item.name), item.name)));
  }
  $('add-menu').onclick = () => { const value = $('menu-add').value; if (value) { state.menu.push(value); state.menuDirty = true; renderMenu(); updateDisabled(); } };
  $('save-menu').onclick = () => run(async () => { await api('menu', sideBody({ names: state.menu }), 'PUT'); state.menuDirty = false; await refreshCatalog(); message('原语菜单已保存。'); });
  document.addEventListener('keydown', event => { if (event.repeat || token || moveLocked() || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return; const index = Number(event.key) - 1; if (/^[1-9]$/.test(event.key) && state.menu[index]) void run(() => api('select', sideBody({ name: state.menu[index] }))); });

  function syncDraft() {
    if (!draft) return;
    draft.name = $('pose-name').value.trim(); draft.description = $('pose-description').value;
    draft.keyframes[frame].name = $('frame-name').value.trim(); draft.keyframes[frame].duration = Number($('duration').value);
  }
  function invalidDraft() {
    if (!draft) return '';
    if (!/^[a-z0-9_]{1,64}$/.test(draft.name)) return '名称只能包含小写字母、数字和下划线，最长64字符。';
    if (new Set(draft.keyframes.map(value => value.name)).size !== draft.keyframes.length) return '关键帧名称不能重复。';
    for (const value of draft.keyframes) {
      if (!/^[a-z_][a-z0-9_]{0,63}$/.test(value.name)) return '关键帧名称需以小写字母或下划线开头。';
      if (!Number.isFinite(value.duration) || value.duration < .2 || value.duration > 60) return '关键帧时长范围为0.2–60秒。';
      if (value.positions.some((q, index) => !Number.isFinite(q) || q < state.library.joints[index].lower - 1e-7 || q > state.library.joints[index].upper + 1e-7)) return '关节目标无效或超出限位。';
    }
    return '';
  }
  function validateDraft() { syncDraft(); const reason = invalidDraft(); $('draft-warning').textContent = reason; $('draft-warning').hidden = !reason; updateDisabled(); }
  async function openEditor(name) {
    await releaseHold(); await loadLibrary();
    const pose = name ? state.library.poses.find(value => value.name === name) : null;
    if (name && !pose) throw new Error('姿态已删除或内容无效，请刷新。');
    const lease = await api('edit/begin', {}); token = lease.token; editorLocked = false; editingSide = state.side;
    let confirmed = lease;
    for (let count = 0; !confirmed.ready && count < 25; count++) { await new Promise(resolve => setTimeout(resolve, 100)); confirmed = await api('edit/heartbeat', { token }); }
    if (!confirmed.ready) { await api('edit/release', { token }); token = ''; throw new Error('停止尚未确认；控制保持暂停。'); }
    let current;
    try { current = await api(`capture?side=${encodeURIComponent(editingSide)}`); }
    catch (error) { const failedToken = token; token = ''; await api('edit/release', { token: failedToken }).catch(() => {}); throw error; }
    draft = pose ? copy(pose) : { name: '', description: '', revision: '', keyframes: [{ name: 'frame_1', duration: .6, positions: current.positions }] };
    frame = 0; baseline = JSON.stringify(draft); pending = {}; liveActive = false; sequence = 0;
    $('editor-title').textContent = pose ? '编辑自定义原语' : '新建自定义原语';
    $('editor-hand').textContent = `${state.library.model.toUpperCase()} · ${editingSide === 'left' ? '左手' : '右手'}`;
    $('pose-name').value = draft.name; $('pose-name').readOnly = !!draft.revision; $('pose-description').value = draft.description;
    $('follow').checked = canMove() && state.catalog?.motion_controls === 1; message('', false, true); renderFrame(); $('editor').showModal(); validateDraft();
  }
  function renderFrame() {
    $('frames').replaceChildren(...draft.keyframes.map((value, index) => {
      const el = button(`${index + 1} · ${value.name}`, () => run(async () => { syncDraft(); await haltLive(); frame = index; renderFrame(); }, true)); el.setAttribute('aria-pressed', String(index === frame)); return el;
    }));
    $('frame-name').value = draft.keyframes[frame].name; $('duration').value = draft.keyframes[frame].duration;
    $('frame-up').disabled = frame === 0; $('frame-down').disabled = frame === draft.keyframes.length - 1; $('frame-delete').disabled = draft.keyframes.length === 1;
    $('add-frame').disabled = $('copy-frame').disabled = draft.keyframes.length >= 100;
    $('joints').replaceChildren(...state.library.joints.map((joint, index) => {
      const row = document.createElement('div'); row.className = 'joint';
      const label = document.createElement('label'); label.htmlFor = `joint-${index}`; label.textContent = joint.name;
      const bounds = document.createElement('small'); bounds.textContent = `${(joint.lower * 180 / Math.PI).toFixed(1)}° — ${(joint.upper * 180 / Math.PI).toFixed(1)}°`; label.append(bounds);
      const slider = document.createElement('input'); slider.type = 'range'; slider.min = joint.lower * 180 / Math.PI; slider.max = joint.upper * 180 / Math.PI; slider.step = .1; slider.setAttribute('aria-label', `${joint.name} 滑条`);
      const number = document.createElement('input'); number.type = 'number'; number.step = .1; number.id = `joint-${index}`; number.setAttribute('aria-label', joint.name);
      slider.value = number.value = Number((draft.keyframes[frame].positions[index] * 180 / Math.PI).toFixed(3));
      function update(source, other) {
        const q = source.value.trim() ? Number(source.value) * Math.PI / 180 : NaN;
        draft.keyframes[frame].positions[index] = q; other.value = source.value;
        if (Number.isFinite(q) && q >= joint.lower - 1e-7 && q <= joint.upper + 1e-7) { if ($('follow').checked && !editorLocked) pending[joint.name] = q; }
        else { $('follow').checked = false; void haltLive().catch(() => {}); }
        validateDraft();
      }
      slider.oninput = () => update(slider, number); number.onchange = () => update(number, slider);
      const unit = document.createElement('span'); unit.textContent = '°'; row.append(label, slider, number, unit); return row;
    }));
    validateDraft();
  }
  async function haltLive() {
    liveGeneration++; pending = {}; liveActive = false;
    if (!token || !needsStop) return stopBarrier;
    needsStop = false;
    const request = api('live/stop', { side: editingSide, token, sequence: ++sequence });
    stopBarrier = request.catch(() => {});
    try { await request; } catch (error) { needsStop = true; throw error; }
  }
  setInterval(async () => {
    if (!token || !draft || !$('follow').checked || !canMove() || editorLocked || liveSending) return;
    if (!Object.keys(pending).length && (!liveActive || performance.now() - lastLive < 150)) return;
    liveSending = true; const version = liveGeneration;
    try {
      await stopBarrier; if (version !== liveGeneration || !token || !$('follow').checked) return;
      const updates = pending; pending = {}; lastLive = performance.now(); needsStop = true;
      await api('live', { side: editingSide, token, sequence: ++sequence, updates });
      if (version === liveGeneration) liveActive = true;
    } catch (error) { if (version === liveGeneration) { $('follow').checked = false; message(error.message, true, true); void haltLive().catch(() => {}); } }
    finally { liveSending = false; }
  }, 50);
  setInterval(async () => {
    if (!token) return;
    try { await api('edit/heartbeat', { token }); }
    catch (error) { editorLocked = true; $('follow').checked = false; message(error.message, true, true); void haltLive().catch(() => {}); updateDisabled(); }
  }, 2000);
  $('follow').onchange = () => { void haltLive().catch(error => message(error.message, true, true)); };
  for (const id of ['pose-name', 'pose-description', 'frame-name', 'duration']) $(id).addEventListener('input', validateDraft);
  async function mutateFrames(fn) { syncDraft(); await haltLive(); fn(); renderFrame(); }
  $('add-frame').onclick = () => run(() => mutateFrames(() => { draft.keyframes.push({ name: `frame_${draft.keyframes.length + 1}`, duration: .6, positions: state.library.joints.map(joint => Math.max(joint.lower, Math.min(joint.upper, 0))) }); frame = draft.keyframes.length - 1; }), true);
  $('copy-frame').onclick = () => run(() => mutateFrames(() => { const value = copy(draft.keyframes[frame]); value.name = `frame_${draft.keyframes.length + 1}`; draft.keyframes.push(value); frame = draft.keyframes.length - 1; }), true);
  $('frame-up').onclick = () => run(() => mutateFrames(() => { [draft.keyframes[frame - 1], draft.keyframes[frame]] = [draft.keyframes[frame], draft.keyframes[frame - 1]]; frame--; }), true);
  $('frame-down').onclick = () => run(() => mutateFrames(() => { [draft.keyframes[frame + 1], draft.keyframes[frame]] = [draft.keyframes[frame], draft.keyframes[frame + 1]]; frame++; }), true);
  $('frame-delete').onclick = () => run(() => mutateFrames(() => { draft.keyframes.splice(frame, 1); frame = Math.min(frame, draft.keyframes.length - 1); }), true);
  $('capture').onclick = () => run(async () => { const value = await api(`capture?side=${editingSide}`); draft.keyframes[frame].positions = value.positions; renderFrame(); message('已读取当前手姿，尚未保存。', false, true); }, true);
  $('save-as').onclick = () => { syncDraft(); draft.name = `${draft.name || 'pose'}_copy`; draft.revision = ''; $('pose-name').value = draft.name; $('pose-name').readOnly = false; validateDraft(); };
  $('save-pose').onclick = () => run(async () => { syncDraft(); draft = await api('poses', { side: editingSide, draft }, 'PUT'); baseline = JSON.stringify(draft); $('pose-name').readOnly = true; message('已保存原语。', false, true); await refreshCatalog(true); }, true);
  async function preview(single) { syncDraft(); await haltLive(); const value = copy(draft); if (single) value.keyframes = [value.keyframes[frame]]; await api('preview', { side: editingSide, token, draft: value }); message('已提交试播。', false, true); }
  $('preview-frame').onclick = () => run(() => preview(true), true); $('preview-all').onclick = () => run(() => preview(false), true);
  $('scrub').oninput = () => { void api('progress', { side: editingSide, token, progress: Number($('scrub').value) }).catch(error => message(error.message, true, true)); };
  async function stopHand(side, inEditor = false) {
    if (stopping || !canMove()) return;
    stopping = true; updateDisabled();
    $('follow').checked = false;
    // Send stop immediately even while a save, preview or live request is waiting.
    const barriers = [releaseHold(), haltLive()];
    try { await api('stop', { side }); message('已停止手势。', false, inEditor); }
    catch (error) { message(error.message, true, inEditor); }
    finally { await Promise.allSettled(barriers); stopping = false; updateDisabled(); void refreshState(); }
  }
  $('editor-stop').onclick = () => void stopHand(editingSide, true);
  async function closeEditor(force = false) {
    if (!draft) return; syncDraft();
    if (!force && JSON.stringify(draft) !== baseline && !confirm('草稿尚未保存，是否放弃修改？')) return;
    await haltLive(); const oldToken = token; token = ''; draft = null; $('editor').close();
    if (oldToken) await api('edit/release', { token: oldToken });
    await refreshCatalog(true); await refreshState();
  }
  $('close-editor').onclick = () => run(() => closeEditor(), true);
  $('editor').addEventListener('cancel', event => { event.preventDefault(); void run(() => closeEditor(), true); });
  $('delete-pose').onclick = () => run(async () => { if (!confirm(`删除原语 ${draft.name}？`)) return; await haltLive(); await api('poses/delete', { side: editingSide, name: draft.name, revision: draft.revision }); await closeEditor(true); }, true);
  $('new-pose').onclick = () => run(() => openEditor());
  $('stop').onclick = () => void stopHand(state.side);
  $('speed').onchange = () => run(async () => { const value = await api('speed', { speed: Number($('speed').value) }, 'PUT'); $('speed').value = value.speed; message('手势速度已生效，仅用于本次运行。'); });
  $('resume').onclick = () => run(async () => { await api('resume', {}); message('控制已恢复，请重新操作。'); });
  $('refresh').onclick = () => run(() => refreshCatalog(true));
  $('side').onchange = () => { generation++; void releaseHold(); state.side = $('side').value; const url = new URL(location.href); url.searchParams.set('side', state.side); history.replaceState(null, '', url); state.catalog = null; state.library = null; state.menuDirty = false; signature = ''; void refreshCatalog(true); };
  document.querySelectorAll('[data-group]').forEach(el => el.onclick = () => { state.group = el.dataset.group; renderCatalog(); });
  $('search').oninput = renderCatalog; $('available').onchange = renderCatalog;
  $('clear-filter').onclick = () => { $('search').value = ''; $('available').checked = false; renderCatalog(); };
  function bundleUrl(path) { return `/api/pose_bundle/${path}${path.includes('?') ? '&' : '?'}side=${state.side}`; }
  $('export').onclick = () => { location.href = bundleUrl('export'); };
  $('import').onclick = () => $('bundle-file').click();
  $('bundle-file').onchange = () => run(async () => {
    const file = $('bundle-file').files[0]; if (!file) return;
    const response = await fetch(bundleUrl(`import?strategy=${$('import-strategy').value}&verify_checksum=true`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: await file.text() });
    const value = await response.json(); if (!response.ok) throw new Error(value.detail || value.message || JSON.stringify(value));
    $('bundle-file').value = ''; await refreshCatalog(true); message(`姿态包已导入：${value.imported || 0} 项。`);
  });
  $('factory').onclick = () => run(async () => { const response = await fetch(bundleUrl(`import_factory?overwrite=${$('import-strategy').value === 'overwrite'}`), { method: 'POST' }); const value = await response.json(); if (!response.ok) throw new Error(value.message || value.detail || JSON.stringify(value)); await refreshCatalog(true); message(`已导入 ${value.count} 个出厂姿态。`); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) suspendContinuousInput(); else void refreshState(); });
  window.addEventListener('pagehide', () => { if (token) fetch('/api/workbench/edit/release', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token }), keepalive: true }).catch(() => {}); });
  setInterval(() => { if (!document.hidden) { void refreshState(); if (!busy) void refreshCatalog(); } }, 500);
  void refreshState();
})();
