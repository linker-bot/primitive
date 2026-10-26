/* Isolated Mock UI fixture: foreign motion is injected into GET state only. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const base = process.env.WORKBENCH_READONLY_URL;
if (!base) throw new Error('WORKBENCH_READONLY_URL must point to an isolated readonly Mock service');
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
  const otherEditor = await browser.newPage();
  const observer = await browser.newPage();
  const errors = [], writes = [], commands = [];
  observer.on('pageerror', error => errors.push(error.message));
  otherEditor.on('pageerror', error => errors.push(error.message));
  observer.on('request', request => { if (request.method() !== 'GET') writes.push(request.url()); });
  observer.on('websocket', socket => socket.on('framesent', frame => {
    const message = JSON.parse(frame.payload.toString());
    if (!['heartbeat', 'live_heartbeat'].includes(message.type)) commands.push(message.type);
  }));
  try {
    const state = await (await observer.request.get(`${base}/api/workbench/state`)).json();
    assert.equal(state.backend, 'mock'); assert.equal(state.observe_only, true);
    assert.ok(state.hands.left && state.hands.right, 'both hands are required');
    await otherEditor.goto(`${base}/?side=right`);
    await otherEditor.locator('#handCanvas .joint-card').first().waitFor();
    await otherEditor.locator('#btnTeach').click();
    await until(async () => await otherEditor.locator('#stateBadge').textContent() === 'TEACH_EDIT', 'other page owns a real editor lease');
    await observer.route('**/api/workbench/state', async route => {
      const response = await route.fetch();
      const snapshot = await response.json();
      snapshot.hands.right = { ...snapshot.hands.right, active: 'ring', execution_kind: 'primitive',
        playback_state: 'running', command_pending: false, state_confirmed: true };
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(snapshot) });
    });
    await observer.goto(`${base}/?side=right`);
    await observer.locator('#handCanvas .joint-card').first().waitFor();
    await until(() => observer.locator('#panelSide').isEnabled(), 'passive observer may change sides despite foreign motion/editor');
    assert.equal(await observer.locator('#panelStop').isDisabled(), true);
    assert.equal(await observer.locator('#workbenchLink').getAttribute('aria-disabled'), 'false');
    await observer.locator('#panelSide').selectOption('left');
    await observer.waitForURL('**/?side=left');
    await until(() => observer.locator('#panelSide').isEnabled(), 'foreign editor does not lock left observation');
    await observer.getByRole('link', { name: '按住推进与菜单', exact: true }).click();
    await observer.waitForURL('**/workbench?side=left');
    await until(async () => await observer.locator('#side').inputValue() === 'left', 'workbench preserves observer side');
    assert.deepEqual(writes, [], 'passive navigation sends no POST/PUT/DELETE');
    assert.deepEqual(commands, [], 'passive navigation sends no WS actions');

    await observer.goto(`${base}/?side=right`);
    await observer.locator('#handCanvas .joint-card').first().waitFor();
    await observer.locator('#btnTeach').click();
    await until(async () => await observer.locator('#stateBadge').textContent() === 'TEACH_EDIT', 'own readonly editor entered');
    assert.equal(await observer.locator('#panelSide').isDisabled(), true, 'own editing still prevents navigation');
    assert.equal(await observer.locator('#workbenchLink').getAttribute('aria-disabled'), 'true');
    await observer.locator('#btnExit').click();
    await until(() => observer.locator('#panelSide').isEnabled(), 'exiting only our editor restores passive navigation');
    const after = await (await observer.request.get(`${base}/api/workbench/state`)).json();
    assert.equal(after.input_pause.editors, 1, 'foreign lease remains intact');
    assert.equal(after.input_pause.paused, true, 'navigation never resumes the shared input');
    assert.deepEqual(writes, []);
    assert.deepEqual(commands, ['enter_teach', 'exit_teach']);
    assert.deepEqual(errors, []);
    console.log('PASS: readonly foreign motion/editor permits pure navigation, own editor remains protected, no motion POST or automatic resume');
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
