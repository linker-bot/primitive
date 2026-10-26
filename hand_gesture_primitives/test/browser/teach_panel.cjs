/* Run only against isolated Mock services. Never point this at an active robot. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.WORKBENCH_URL || 'http://127.0.0.1:8001';
const readonlyBase = process.env.WORKBENCH_READONLY_URL;
const name = `restored_${Date.now()}`;

async function until(check, label) {
  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    if (await check()) return;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(`Timed out: ${label}`);
}
async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => dialog.accept());
  const acknowledgedSaves = new Set();
  page.on('websocket', socket => socket.on('framereceived', frame => {
    const message = JSON.parse(frame.payload.toString());
    if (message.type === 'ok' && message.action === 'save_pose') {
      acknowledgedSaves.add(`${new URL(socket.url()).host}/${message.name}`);
    }
  }));
  async function api(endpoint, data, method = 'POST', url = base) {
    const response = await page.request.fetch(`${url}/api/workbench/${endpoint}`, { method: data === undefined ? 'GET' : method, ...(data === undefined ? {} : { data }) });
    assert.equal(response.ok(), true, await response.text()); return response.json();
  }
  try {
    const initial = await api('state');
    assert.equal(initial.backend, 'mock', 'this test requires an isolated Mock service');
    assert.equal(initial.default_side, 'right');
    await page.goto(base);
    await page.locator('#handCanvas .joint-card').first().waitFor();
    await until(async () => await page.locator('#panelSide').inputValue() === 'right', 'default primary side');
    assert.equal(await page.locator('#poseList').count(), 1);
    assert.equal(await page.locator('#progressSlider').count(), 1);
    assert.equal(await page.locator('#editor').count(), 0, 'the original editor remains inline');
    const open = page.getByRole('button', { name: '执行内置手势 张开', exact: true });
    await until(() => open.isEnabled(), 'built-in catalog ready');
    assert.equal(await page.getByRole('button', { name: '执行内置手势 准备抓取', exact: true }).isDisabled(), true);
    await page.locator('#builtinSearch').fill('no_such_gesture');
    await page.getByRole('tab', { name: /^感知原语/ }).click();
    assert.match(await page.locator('#builtinFilterState').textContent(), /显示 0\/.*no_such_gesture/);
    await page.getByRole('button', { name: '清除筛选', exact: true }).click();
    assert.ok(await page.locator('#builtinList .builtin-item').count() > 0);
    await page.getByRole('tab', { name: /^默认原语/ }).click();
    // A catalog poll may have captured idle before select and arrive after it.
    // It must not unlock navigation while the new action is running.
    const oldIdleState = await api('state');
    assert.equal(oldIdleState.hands.right.active, null);
    let releaseOldState, finishOldState, oldStateCaptured = false, captureOldState = true;
    const oldStateBarrier = new Promise(resolve => { releaseOldState = resolve; });
    const oldStateFulfilled = new Promise(resolve => { finishOldState = resolve; });
    await page.route('**/api/workbench/state', async route => {
      if (!captureOldState) { await route.continue(); return; }
      captureOldState = false;
      oldStateCaptured = true;
      await oldStateBarrier;
      try {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(oldIdleState) });
      } finally { finishOldState(); }
    });
    try {
      await until(async () => oldStateCaptured, 'idle state response delayed');
      await page.getByRole('button', { name: '执行内置手势 握拳', exact: true }).click();
      await until(async () => (await api('state')).hands.right.active === 'fist', 'motion accepted while old poll waits');
      assert.equal(await page.locator('#panelSide').isDisabled(), true, 'pending motion retains side guard');
      assert.equal(await page.locator('#workbenchLink').getAttribute('aria-disabled'), 'true');
      const previousUrl = page.url();
      await page.locator('#panelSide').evaluate(el => { el.value = 'left'; el.dispatchEvent(new Event('change')); });
      assert.equal(page.url(), previousUrl, 'programmatic side change cannot bypass pending guard');
    } finally { releaseOldState(); if (oldStateCaptured) await oldStateFulfilled; }
    // Keep the pass-through handler until browser.close. Removing routing while
    // the released state starts its catalog fetch can strand that request.
    await page.locator('#panelStop').click();
    await until(() => page.locator('#panelSide').isEnabled(), 'confirmed stop clears pending guard');
    await page.locator('#panelSpeed').fill('1.25'); await page.locator('#panelSpeed').press('Tab');
    await until(async () => (await api('state')).primitive_speed === 1.25, 'speed acknowledged');
    await open.click();
    await until(async () => (await api('state')).hands.right.active === 'open', 'builtin execution');
    await until(() => page.locator('#panelSide').isDisabled(), 'motion locks side navigation');
    await page.locator('#panelStop').click();
    await until(() => page.locator('#panelSide').isEnabled(), 'stop permits navigation');
    await page.locator('#btnTeach').click();
    await until(async () => (await page.locator('#stateBadge').textContent()) === 'TEACH_EDIT', 'legacy teach entered');
    assert.equal(await page.locator('#panelSide').isDisabled(), true);
    assert.equal(await page.locator('#workbenchLink').getAttribute('aria-disabled'), 'true');
    await until(() => open.isDisabled(), 'lease disables builtin execution');
    await page.locator('#poseName').fill(name);
    await page.locator('#poseKeyframe').fill('prepare');
    await page.locator('#btnSave').click();
    // A library read performs IO too; let the WebSocket save finish before polling it.
    await until(() => acknowledgedSaves.has(`${new URL(base).host}/${name}`), 'legacy save acknowledged');
    await until(async () => (await api('library?side=right')).poses.some(pose => pose.name === name), 'legacy save remains available');
    await page.locator('#btnExit').click();
    await until(async () => (await page.locator('#stateBadge').textContent()) === 'PRIMITIVE_IDLE', 'legacy teach exited');
    // Releasing another editor is deliberately sticky. The original page must
    // offer an explicit resume action, not send resume during refresh/navigation.
    const lease = await api('edit/begin', {});
    await api('edit/release', { token: lease.token });
    await until(() => page.locator('#panelResume').isEnabled(), 'explicit resume available');
    assert.equal(await open.isDisabled(), true);
    await page.locator('#panelResume').click();
    await until(() => open.isEnabled(), 'explicit resume restores builtin execution');
    await open.click(); await page.locator('#panelStop').click();
    await until(() => page.locator('#panelSide').isEnabled(), 'second stop permits navigation');
    let navigationWrites = 0;
    const observe = request => { if (request.method() !== 'GET') navigationWrites++; };
    page.on('request', observe);
    await page.locator('#panelSide').selectOption('left');
    await page.waitForURL('**/?side=left');
    await until(async () => (await page.locator('#meta').textContent()).includes('left'), 'left websocket selected');
    assert.match(await page.locator('#workbenchLink').getAttribute('href'), /side=left/);
    await page.goto(`${base}/workbench?side=right`);
    await until(async () => await page.locator('#side').inputValue() === 'right', 'workbench URL side wins');
    assert.equal(await page.locator('#legacy-link').getAttribute('href'), '/?side=right');
    await page.goto(`${base}/workbench?side=unknown`);
    await until(async () => await page.locator('#side').inputValue() === 'right', 'invalid URL side uses backend default');
    await page.locator('#search').fill('no_such_gesture');
    await page.getByRole('tab', { name: /带感知原语/ }).click();
    assert.match(await page.locator('#filter-state').textContent(), /显示 0\/.*no_such_gesture/);
    await page.locator('#clear-filter').click();
    await until(async () => await page.locator('#catalog .gesture').count() > 0, 'workbench filter clears');
    await page.goto(`${base}/?side=right`);
    await page.locator('#handCanvas .joint-card').first().waitFor();
    page.off('request', observe);
    assert.equal(navigationWrites, 0, 'navigation and initialization never send actions');
    const artifacts = process.env.RESTORED_PANEL_ARTIFACT_DIR;
    if (artifacts) {
      fs.mkdirSync(artifacts, { recursive: true });
      await page.screenshot({ path: path.join(artifacts, 'teach-panel-desktop.png'), fullPage: true });
      await page.setViewportSize({ width: 390, height: 844 });
      await page.screenshot({ path: path.join(artifacts, 'teach-panel-mobile.png'), fullPage: true });
    }
    if (readonlyBase) {
      assert.equal((await api('state', undefined, 'GET', readonlyBase)).backend, 'mock');
      await page.goto(readonlyBase);
      await page.locator('#handCanvas .joint-card').first().waitFor();
      await page.locator('#builtinReadonly').waitFor();
      assert.equal(await page.locator('#panelSpeed').isDisabled(), true);
      assert.equal(await page.locator('#panelStop').isDisabled(), true);
      assert.equal(await open.isDisabled(), true);
      await page.locator('#btnTeach').click();
      await until(async () => (await page.locator('#stateBadge').textContent()) === 'TEACH_EDIT', 'readonly editor entered');
      await page.locator('#poseName').fill(name); await page.locator('#poseKeyframe').fill('prepare');
      await page.locator('#btnSave').click();
      await until(() => acknowledgedSaves.has(`${new URL(readonlyBase).host}/${name}`), 'readonly save acknowledged');
      await until(async () => (await api('library?side=right', undefined, 'GET', readonlyBase)).poses.some(pose => pose.name === name), 'readonly draft saved');
      assert.equal(await page.locator('#btnPlay').isDisabled(), true);
      assert.deepEqual((await api('capture?side=right', undefined, 'GET', readonlyBase)).positions, [0, 0, 0, 0, 0, 0]);
      await page.locator('#btnExit').click();
    }
    assert.deepEqual(errors, []);
    console.log('PASS: restored hand/tree layout, discoverable builtins, filters, speed/stop, lease/resume, side navigation, no navigation writes, readonly save');
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
