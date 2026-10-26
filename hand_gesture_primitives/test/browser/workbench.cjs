/* Run against an isolated --mock workbench; PLAYWRIGHT_MODULE may point at an existing install. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.WORKBENCH_URL || 'http://127.0.0.1:8001';
const name = `browser_${Date.now()}`;
async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1360, height: 950 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => dialog.accept());
  async function request(path, data, method = 'POST') {
    const response = await page.request.fetch(`${base}/api/workbench/${path}`, { method: data === undefined ? 'GET' : method, ...(data === undefined ? {} : { data }) });
    assert.equal(response.ok(), true, await response.text());
    return response.json();
  }
  async function until(callback, label) {
    const deadline = Date.now() + 8000;
    while (Date.now() < deadline) { if (await callback()) return; await new Promise(resolve => setTimeout(resolve, 50)); }
    throw new Error(`Timed out: ${label}`);
  }
  try {
    await page.goto(`${base}/workbench`);
    await page.getByText('已连接', { exact: true }).waitFor();
    await page.getByRole('button', { name: '准备抓取 · 不可用', exact: true }).click();
    await page.getByRole('status').filter({ hasText: '当前不可用' }).waitFor();
    const speed = page.getByRole('spinbutton', { name: '手势速度倍率' });
    await speed.fill('1.5'); await speed.press('Tab');
    await until(async () => (await request('state')).primitive_speed === 1.5, 'speed confirmation');
    await page.getByRole('tab', { name: /自定义原语/ }).click();
    await page.getByRole('button', { name: '新建原语', exact: true }).click();
    const editor = page.getByRole('dialog');
    await editor.waitFor();
    assert.equal(await editor.getByRole('checkbox', { name: '实时跟随', exact: true }).isChecked(), true);
    const before = (await request('capture?side=right')).positions;
    await editor.getByRole('slider', { name: 'thumb_cmc_pitch 滑条', exact: true }).fill('12');
    await until(async () => Math.abs((await request('capture?side=right')).positions[0] - 12 * Math.PI / 180) < 1e-6, 'live joint update');
    const after = (await request('capture?side=right')).positions;
    assert.deepEqual(after.slice(1), before.slice(1));
    await editor.getByRole('button', { name: '复制当前帧', exact: true }).click();
    await until(async () => (await request('state')).hands.right.execution_kind === 'idle', 'frame switch stops live');
    await editor.getByRole('checkbox', { name: '实时跟随', exact: true }).uncheck();
    await editor.getByRole('slider', { name: 'thumb_cmc_pitch 滑条', exact: true }).fill('20');
    assert.ok(Math.abs((await request('capture?side=right')).positions[0] - 12 * Math.PI / 180) < 1e-6);
    await editor.getByLabel('原语名称', { exact: true }).fill(name);
    await editor.getByLabel('说明', { exact: true }).fill('browser keyframes');
    let finishSave, savePending = false;
    const saveBarrier = new Promise(resolve => { finishSave = resolve; });
    await page.route('**/api/workbench/poses', async route => { savePending = true; await saveBarrier; await route.continue(); });
    await editor.getByRole('button', { name: '保存原语', exact: true }).click();
    await until(async () => savePending, 'save is waiting');
    assert.equal(await editor.getByRole('button', { name: '停止', exact: true }).isEnabled(), true);
    let stopArrived = false;
    const observeStop = request => { if (request.url().endsWith('/api/workbench/stop')) stopArrived = true; };
    page.on('request', observeStop);
    await editor.getByRole('button', { name: '停止', exact: true }).click();
    await until(async () => stopArrived, 'stop sent while save remains pending');
    page.off('request', observeStop);
    const savedResponse = page.waitForResponse(response => response.url().endsWith('/api/workbench/poses'));
    finishSave(); await savedResponse; await page.unroute('**/api/workbench/poses');
    await editor.getByRole('status').filter({ hasText: '已保存原语' }).waitFor();
    const saved = (await request('library?side=right')).poses.find(value => value.name === name);
    assert.equal(saved.keyframes.length, 2);
    await editor.getByRole('button', { name: '试播当前帧', exact: true }).click();
    await until(async () => Math.abs((await request('capture?side=right')).positions[0] - 20 * Math.PI / 180) < .01, 'preview reaches frame');
    await editor.getByRole('button', { name: '停止', exact: true }).click();
    await editor.getByRole('button', { name: '关闭', exact: true }).click();
    await page.getByRole('button', { name: '恢复控制', exact: true }).click();
    await page.getByLabel('加入菜单的原语').selectOption(`named_pose:${name}`);
    await page.getByRole('button', { name: '加入', exact: true }).click();
    await page.getByRole('button', { name: '应用菜单', exact: true }).click();
    await until(async () => (await request('catalog?side=right')).menu.includes(`named_pose:${name}`), 'saved menu');
    const heldDraft = (await request('library?side=right')).poses.find(value => value.name === name);
    heldDraft.keyframes.forEach(value => { value.duration = 3; });
    await request('poses', { side: 'right', draft: heldDraft }, 'PUT');
    await page.reload();
    await page.getByRole('tab', { name: /自定义原语/ }).click();
    await page.getByRole('button', { name: `${name} · 可用`, exact: true }).waitFor();
    const hold = page.getByRole('button', { name: `按住推进 ${name}`, exact: true });
    await hold.hover(); await page.mouse.down();
    await until(async () => (await request('state')).hands.right.playback_state === 'running', 'held playback');
    await page.mouse.up();
    await until(async () => (await request('state')).hands.right.playback_state === 'paused', 'release pauses playback');
    await hold.focus(); await page.keyboard.down('Space');
    await until(async () => (await request('state')).hands.right.playback_state === 'running', 'keyboard held playback');
    await page.evaluate(() => window.dispatchEvent(new Event('blur')));
    await until(async () => (await request('state')).hands.right.playback_state === 'paused', 'window blur pauses held playback');
    const blurred = (await request('state')).hands.right.overall_progress;
    await page.waitForTimeout(650);
    assert.equal((await request('state')).hands.right.overall_progress, blurred, 'blur stops watchdog renewal');
    await page.keyboard.up('Space');
    await hold.focus(); await page.keyboard.down('Space');
    await until(async () => (await request('state')).hands.right.playback_state === 'running', 'keyboard hold before Tab');
    await page.keyboard.press('Tab');
    assert.equal(await hold.evaluate(el => document.activeElement === el), false, 'Tab transfers focus inside the page');
    await page.keyboard.up('Space');
    await until(async () => (await request('state')).hands.right.playback_state === 'paused', 'Tab owner loss pauses held playback');
    const tabProgress = (await request('state')).hands.right.overall_progress;
    let renewedAfterTab = 0;
    const observeHeld = request => { if (request.url().endsWith('/api/workbench/held') && request.postDataJSON()?.running) renewedAfterTab++; };
    page.on('request', observeHeld);
    await page.waitForTimeout(650);
    page.off('request', observeHeld);
    assert.equal(renewedAfterTab, 0, 'focus loss sends no watchdog renewals');
    assert.equal((await request('state')).hands.right.overall_progress, tabProgress);
    await hold.focus(); await page.keyboard.down('Space');
    await until(async () => { const hand = (await request('state')).hands.right; return hand.playback_state === 'running' && hand.overall_progress > tabProgress; }, 'fresh press resumes after Tab');
    await page.keyboard.up('Space');
    await until(async () => (await request('state')).hands.right.playback_state === 'paused', 'fresh press releases normally');
    await hold.focus(); await page.keyboard.down('Space');
    await until(async () => (await request('state')).hands.right.playback_state === 'running', 'keyboard hold before owner removal');
    await hold.evaluate(el => el.remove());
    await until(async () => (await request('state')).hands.right.playback_state === 'paused', 'removed owner stops renewal');
    const removedProgress = (await request('state')).hands.right.overall_progress;
    await page.waitForTimeout(650);
    assert.equal((await request('state')).hands.right.overall_progress, removedProgress);
    await page.keyboard.up('Space');
    await page.locator('#search').fill('__missing_owner__'); await page.locator('#search').fill('');
    await page.getByRole('button', { name: '停止手势', exact: true }).click();
    await page.getByLabel('手势侧别').selectOption('left');
    await until(async () => (await page.getByRole('button', { name: `${name} · 可用`, exact: true }).count()) === 0, 'side isolation');
    assert.equal((await request('library?side=left')).poses.some(value => value.name === name), false);
    await page.getByLabel('手势侧别').selectOption('right');
    await page.getByRole('button', { name: `编辑 ${name}`, exact: true }).click();
    await editor.waitFor();
    await editor.getByRole('checkbox', { name: '实时跟随', exact: true }).check();
    await editor.getByRole('slider', { name: 'thumb_cmc_pitch 滑条', exact: true }).fill('10');
    await until(async () => (await request('state')).hands.right.execution_kind === 'live', 'live after reopen');
    await page.evaluate(() => { Object.defineProperty(document, 'hidden', { configurable: true, value: true }); document.dispatchEvent(new Event('visibilitychange')); });
    await until(async () => (await request('state')).hands.right.execution_kind === 'idle', 'backgrounding stops live');
    assert.equal(await editor.getByRole('checkbox', { name: '实时跟随', exact: true }).isChecked(), false);
    await page.evaluate(() => { Object.defineProperty(document, 'hidden', { configurable: true, value: false }); document.dispatchEvent(new Event('visibilitychange')); });
    await editor.getByRole('button', { name: '关闭', exact: true }).click();
    await page.getByRole('button', { name: '恢复控制', exact: true }).click();
    const artifacts = process.env.WORKBENCH_ARTIFACT_DIR;
    if (artifacts) { fs.mkdirSync(artifacts, { recursive: true }); await page.screenshot({ path: path.join(artifacts, 'standalone-desktop.png'), fullPage: true }); }
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'mobile layout fits');
    if (artifacts) await page.screenshot({ path: path.join(artifacts, 'standalone-mobile.png'), fullPage: true });
    assert.deepEqual(errors, []);
    console.log('PASS: catalog, dual hands, keyframes, CAS-backed save, live, background stop, menu, speed, held playback, window/Tab/DOM owner loss, fresh press recovery, stop during pending save, responsive layout');
  } finally {
    await request('stop', { side: 'right' }).catch(() => {});
    const library = await request('library?side=right').catch(() => ({ poses: [] }));
    const pose = library.poses.find(value => value.name === name);
    if (pose) await request('poses/delete', { side: 'right', name, revision: pose.revision }).catch(() => {});
    await browser.close();
  }
}
async function readonly() {
  const url = process.env.WORKBENCH_READONLY_URL;
  if (!url) return;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.on('dialog', dialog => dialog.accept());
  const name = `readonly_${Date.now()}`;
  try {
    await page.goto(`${url}/workbench`);
    await page.getByText('已连接', { exact: true }).waitFor();
    assert.equal(await page.getByRole('spinbutton', { name: '手势速度倍率' }).isDisabled(), true);
    await page.getByRole('button', { name: '新建原语', exact: true }).click();
    const editor = page.getByRole('dialog'); await editor.waitFor();
    assert.equal(await editor.getByRole('checkbox', { name: '实时跟随', exact: true }).isDisabled(), true);
    assert.equal(await editor.getByRole('button', { name: '试播整个动作', exact: true }).isDisabled(), true);
    await editor.getByLabel('原语名称', { exact: true }).fill(name);
    await editor.getByRole('button', { name: '保存原语', exact: true }).click();
    await editor.getByRole('status').filter({ hasText: '已保存原语' }).waitFor();
    await editor.getByLabel('说明', { exact: true }).fill('discard this draft change');
    await editor.getByRole('button', { name: '关闭', exact: true }).click();
    await page.getByRole('tab', { name: /自定义原语/ }).click();
    await page.getByRole('button', { name: `${name} · 不可用`, exact: true }).waitFor();
    const capture = await (await page.request.get(`${url}/api/workbench/capture?side=right`)).json();
    assert.deepEqual(capture.positions, [0, 0, 0, 0, 0, 0]);
    const library = await (await page.request.get(`${url}/api/workbench/library?side=right`)).json();
    const saved = library.poses.find(value => value.name === name);
    assert.equal(saved.description, '');
    await page.request.post(`${url}/api/workbench/poses/delete`, { data: { side: 'right', name, revision: saved.revision } });
    console.log('PASS: read-only editor saves without motion, preview/live/speed disabled, dirty draft close guarded');
  } finally { await browser.close(); }
}
async function legacyLayout() {
  const url = process.env.WORKBENCH_LEGACY_LAYOUT_URL;
  if (!url) return;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  const capture = async () => (await (await page.request.get(`${url}/api/workbench/capture?side=left`)).json()).positions;
  const post = async (path, data) => {
    const response = await page.request.post(`${url}/api/workbench/${path}`, { data });
    assert.equal(response.ok(), true, await response.text()); return response.json();
  };
  try {
    const library = await (await page.request.get(`${url}/api/workbench/library?side=left`)).json();
    assert.equal(library.model, 'l25');
    const token = (await post('edit/begin', {})).token;
    const updates = Object.fromEntries(library.joints.map((joint, index) => [joint.name, .12 + index * .015]));
    await post('live', { side: 'left', token, sequence: 1, updates });
    await page.waitForTimeout(100);
    await post('live/stop', { side: 'left', token, sequence: 2 });
    await post('edit/release', { token }); await post('resume', {});
    await page.goto(`${url}/legacy?side=left`);
    await page.locator('#j24').waitFor({ state: 'attached' });
    await page.locator('#btnTeach').click();
    await page.waitForFunction(() => document.getElementById('stateBadge').textContent === 'TEACH_EDIT');
    const before = await capture();
    const slider = page.locator('#j6');
    await slider.fill(String(Number(await slider.inputValue()) + 1));
    const deadline = Date.now() + 5000;
    let after = before;
    while (Date.now() < deadline) {
      after = await capture();
      if (Math.abs(after[4] - before[4]) > 1e-6) break;
      await page.waitForTimeout(40);
    }
    assert.ok(Math.abs(after[4] - before[4]) > 1e-6, 'L25 native slider changes its active controller joint');
    assert.deepEqual(after.filter((_, index) => index !== 4), before.filter((_, index) => index !== 4), 'untouched L25 joints retain exact radians');
    assert.deepEqual(errors, []);
    console.log('PASS: legacy L25 native25 browser mapping, single-joint edit preserves other active16 joints');
  } finally {
    await browser.close();
  }
}
main().then(readonly).then(legacyLayout).catch(error => { console.error(error); process.exitCode = 1; });
