"""The independent ROS/Mock workbench contract, with no LeRobot imports."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hand_gesture_primitives.teach_panel.server import create_workbench_app
from hand_gesture_primitives.teach_panel.workbench import WorkbenchHost


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path / 'poses'))
    host = WorkbenchHost.mock({'right': 'o6', 'left': 'o6'})
    with TestClient(create_workbench_app(host)) as value:
        value.workbench = host
        yield value


def wait_for(check, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := check():
            return value
        time.sleep(.02)
    raise AssertionError('condition did not become true')


def draft(name='test_grasp', q=.15):
    return {'name': name, 'description': '独立多帧动作', 'revision': '', 'keyframes': [
        {'name': 'prepare', 'positions': [q] * 6, 'duration': .6},
        {'name': 'grasp', 'positions': [q + .1] * 6, 'duration': .6},
    ]}


def save(client, value, side='right'):
    result = client.put('/api/workbench/poses', json={'side': side, 'draft': value})
    assert result.status_code == 200, result.text
    return result.json()


def lease(client):
    result = client.post('/api/workbench/edit/begin')
    assert result.status_code == 200, result.text
    return result.json()['token']


def test_mock_imports_and_runs_without_ros_or_lerobot(tmp_path):
    code = '''
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'rclpy', 'lerobot', 'linker_lerobot'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Reject())
from hand_gesture_primitives.teach_panel.main import main
from hand_gesture_primitives.teach_panel.workbench import WorkbenchHost
from hand_gesture_primitives.teach_panel.server import create_workbench_app
from fastapi.testclient import TestClient
host = WorkbenchHost.mock()
with TestClient(create_workbench_app(host)) as client:
    assert client.get('/api/workbench/state').status_code == 200
    assert client.post('/api/workbench/select', json={'side':'right','name':'open'}).status_code == 200
'''
    environment = {**os.environ, 'HAND_GESTURE_POSE_DIR': str(tmp_path / 'isolated'),
                   'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, '-c', code], env=environment, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('default_side', ['left', 'right'])
def test_default_page_and_websocket_use_configured_side_independent_of_order(tmp_path, monkeypatch, default_side):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path / 'poses'))
    other = 'left' if default_side == 'right' else 'right'
    host = WorkbenchHost.mock({other: 'o6', default_side: 'o6'}, default_side=default_side)
    with TestClient(create_workbench_app(host)) as client:
        state = client.get('/api/workbench/state').json()
        assert state['default_side'] == default_side
        assert list(state['hands'])[0] == other
        assert 'id="handCanvas"' in client.get('/').text
        assert 'id="handCanvas"' in client.get('/legacy').text
        assert 'id="catalog"' in client.get('/workbench').text
        with client.websocket_connect('/ws') as ws:
            assert ws.receive_json()['hand_side'] == default_side
        with client.websocket_connect('/ws?side=' + other) as ws:
            assert ws.receive_json()['hand_side'] == other
        # Connecting or viewing either page must not start any motion.
        assert all(hand.status()['active'] in {None, 'hold', 'idle'} for hand in host.hands.values())


def test_catalog_crud_revision_menu_and_dual_side_isolation(client):
    assert client.get('/').status_code == 200
    assert client.get('/legacy').status_code == 200
    catalog = client.get('/api/workbench/catalog?side=right').json()
    assert not next(item for item in catalog['primitives'] if item['name'] == 'grasp_ready')['available']
    stored = save(client, draft())
    assert stored['revision']
    assert client.get('/api/workbench/library?side=left').json()['poses'] == []
    assert client.put('/api/workbench/poses', json={'side': 'right', 'draft': draft()}).status_code == 409
    menu = ['open', 'named_pose:test_grasp', 'fist']
    assert client.put('/api/workbench/menu', json={'side': 'right', 'names': menu}).json()['menu'] == menu
    stored['keyframes'][0]['positions'][0] = .3
    changed = save(client, stored)
    assert changed['revision'] != stored['revision']
    assert client.post('/api/workbench/poses/delete', json={'side': 'right', 'name': stored['name'], 'revision': stored['revision']}).status_code == 409
    assert client.post('/api/workbench/select', json={'side': 'right', 'name': 'named_pose:test_grasp'}).status_code == 200
    assert client.post('/api/workbench/poses/delete', json={'side': 'right', 'name': changed['name'], 'revision': changed['revision']}).status_code == 400
    assert client.post('/api/workbench/stop', json={'side': 'right'}).status_code == 200
    assert client.post('/api/workbench/poses/delete', json={'side': 'right', 'name': changed['name'], 'revision': changed['revision']}).status_code == 200
    previous_id = client.get('/api/workbench/catalog?side=right').json()['id_map'].index('named_pose:test_grasp')
    save(client, draft('next_grasp'))
    assert client.get('/api/workbench/catalog?side=right').json()['id_map'].index('named_pose:next_grasp') > previous_id
    assert client.get('/api/workbench/library?side=unknown').status_code == 422


def test_live_only_changes_requested_joint_stops_late_requests_and_requires_resume(client):
    token = lease(client)
    before = client.get('/api/workbench/capture?side=right').json()['positions']
    request = {'side': 'right', 'token': token, 'sequence': 1, 'updates': {'thumb_cmc_pitch': .2}}
    assert client.post('/api/workbench/live', json=request).status_code == 200
    wait_for(lambda: abs(client.get('/api/workbench/capture?side=right').json()['positions'][0] - .2) < 1e-6)
    after = client.get('/api/workbench/capture?side=right').json()['positions']
    np.testing.assert_allclose(after[1:], before[1:])
    np.testing.assert_allclose(client.get('/api/workbench/capture?side=left').json()['positions'], [0] * 6)
    assert client.post('/api/workbench/live/stop', json={**request, 'sequence': 3, 'updates': {}}).status_code == 200
    late = client.post('/api/workbench/live', json={**request, 'sequence': 2})
    assert late.status_code == 400 and late.json()['detail']['code'] == 'live_stale'
    assert client.post('/api/workbench/edit/release', json={'token': token}).status_code == 200
    assert client.get('/api/workbench/state').json()['input_pause']['paused']
    assert client.post('/api/workbench/select', json={'side': 'right', 'name': 'open'}).status_code == 400
    assert client.post('/api/workbench/resume').status_code == 200
    assert client.post('/api/workbench/select', json={'side': 'right', 'name': 'open'}).status_code == 200


def test_live_expiry_multi_page_and_invalid_values(client):
    token, other = lease(client), lease(client)
    request = {'side': 'right', 'token': token, 'sequence': 1, 'updates': {'thumb_cmc_pitch': .2}}
    assert client.post('/api/workbench/live', json=request).status_code == 200
    assert client.post('/api/workbench/live', json={**request, 'token': other}).json()['detail']['code'] == 'live_busy'
    wait_for(lambda: not client.workbench.live.sessions['right']['active'])
    assert client.post('/api/workbench/live', json={**request, 'sequence': 2}).json()['detail']['code'] == 'live_expired'
    assert client.post('/api/workbench/live/stop', json={**request, 'sequence': 3, 'updates': {}}).status_code == 200
    assert client.post('/api/workbench/live', json={**request, 'sequence': 4, 'updates': {'thumb_cmc_pitch': 99.0}}).status_code == 400
    assert client.post('/api/workbench/live', json={**request, 'sequence': 5, 'updates': {'unknown': .1}}).status_code == 400
    assert client.post('/api/workbench/edit/release', json={'token': token}).status_code == 200
    assert client.post('/api/workbench/resume').status_code == 400
    client.workbench.edit.leases[other] = time.monotonic() - 1
    wait_for(lambda: not client.workbench.edit.leases)
    assert client.get('/api/workbench/state').json()['input_pause']['paused']
    assert client.post('/api/workbench/resume').status_code == 200


def test_held_pause_speed_restart_and_watchdog(client):
    stored = save(client, draft())
    request = {'side': 'right', 'name': 'named_pose:' + stored['name'], 'client_id': 'browser', 'sequence': 1, 'running': True}
    assert client.post('/api/workbench/held', json=request).status_code == 200
    wait_for(lambda: client.workbench.hands['right'].status()['overall_progress'] > .05)
    assert client.post('/api/workbench/held', json={**request, 'sequence': 2, 'running': False}).status_code == 200
    progress = client.workbench.hands['right'].status()['overall_progress']
    time.sleep(.1)
    assert client.workbench.hands['right'].status()['overall_progress'] == progress
    assert client.put('/api/workbench/speed', json={'speed': 1.5}).json() == {'speed': 1.5}
    assert client.post('/api/workbench/held', json={**request, 'sequence': 3}).status_code == 200
    wait_for(lambda: client.workbench.hands['right'].status()['overall_progress'] > progress)
    wait_for(lambda: client.workbench.held['right'].get('expired'))
    assert client.post('/api/workbench/held', json={**request, 'sequence': 4}).json()['detail']['code'] == 'held_release_required'
    assert client.post('/api/workbench/held', json={**request, 'sequence': 5, 'running': False}).status_code == 200
    assert client.post('/api/workbench/held', json={**request, 'sequence': 6}).status_code == 200
    assert client.post('/api/workbench/select', json={'side': 'right', 'name': 'open'}).status_code == 200
    assert client.post('/api/workbench/held', json={**request, 'sequence': 7}).status_code == 400
    for speed in [0.0, 4.0, 'NaN']:
        assert client.put('/api/workbench/speed', json={'speed': speed}).status_code == 422


def test_readonly_can_save_but_never_moves(tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path / 'readonly'))
    host = WorkbenchHost.mock(observe_only=True)
    with TestClient(create_workbench_app(host)) as client:
        token = lease(client)
        saved = save(client, draft())
        assert client.post('/api/workbench/preview', json={'side': 'right', 'token': token, 'draft': saved}).status_code == 400
        assert client.post('/api/workbench/live', json={'side': 'right', 'token': token, 'sequence': 1, 'updates': {'thumb_cmc_pitch': .2}}).status_code == 400
        assert client.put('/api/workbench/speed', json={'speed': 2.0}).status_code == 400
        np.testing.assert_allclose(host.feedback('right'), [0] * 6)


def test_io_lock_does_not_block_stop_or_control_ticks(client):
    from hand_gesture_primitives.pose_library import _pose_path
    path = _pose_path('o6', 'right')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a+') as locked:
        fcntl.flock(locked, fcntl.LOCK_EX)
        with ThreadPoolExecutor(1) as pool:
            request = pool.submit(client.put, '/api/workbench/poses', json={'side': 'right', 'draft': draft()})
            wait_for(lambda: client.workbench.pending_io)
            before = client.workbench.hands['right'].feedback_id
            assert client.post('/api/workbench/stop', json={'side': 'right'}).status_code == 200
            wait_for(lambda: client.workbench.hands['right'].feedback_id > before + 2)
            fcntl.flock(locked, fcntl.LOCK_UN)
            assert request.result(timeout=3).status_code == 200


def test_bundle_roundtrip_and_selected_side(client):
    stored = save(client, draft())
    export = client.get('/api/pose_bundle/export?side=right')
    assert export.status_code == 200 and len(export.json()['entries']) == 1
    assert export.headers['X-Pose-Count'] == '1'
    assert export.headers['X-Hand-Type'] == 'o6'
    assert export.headers['X-Hand-Side'] == 'right'
    assert 'X-Pose-Empty' not in export.headers
    empty = client.get('/api/pose_bundle/export?side=left')
    assert empty.json()['entries'] == []
    assert empty.headers['X-Pose-Count'] == '0'
    assert empty.headers['X-Pose-Empty'] == '1'
    assert empty.headers['X-Hand-Type'] == 'o6'
    assert empty.headers['X-Hand-Side'] == 'left'
    assert client.post('/api/workbench/poses/delete', json={'side': 'right', 'name': stored['name'], 'revision': stored['revision']}).status_code == 200
    result = client.post('/api/pose_bundle/import?side=right', content=export.content)
    assert result.status_code == 200, result.text
    assert client.get('/api/workbench/library?side=right').json()['poses'][0]['name'] == stored['name']
    assert client.get('/api/workbench/library?side=left').json()['poses'] == []
    assert client.post('/api/pose_bundle/import_factory?side=left&overwrite=true').status_code == 200


def test_legacy_websocket_edits_use_shared_lease_and_revision(client):
    def receive_until(ws, predicate):
        for _ in range(30):
            message = ws.receive_json()
            if predicate(message):
                return message
        raise AssertionError('websocket message not found')

    stored = save(client, draft())
    with client.websocket_connect('/ws?side=right') as ws:
        receive_until(ws, lambda value: value.get('type') == 'snapshot')
        ws.send_json({'type': 'load_pose', 'name': stored['name'], 'keyframe': 'prepare'})
        assert receive_until(ws, lambda value: value.get('action') == 'load_pose')['type'] == 'ok'
        assert client.post('/api/workbench/select', json={'side': 'right', 'name': 'open'}).status_code == 400
        stored['keyframes'][0]['positions'][0] = .3
        save(client, stored)
        ws.send_json({'type': 'save_pose', 'name': stored['name'], 'keyframe': 'prepare', 'duration': .6})
        rejected = receive_until(ws, lambda value: value.get('type') == 'error')
        assert 'revision_conflict' in rejected['message']
        ws.send_json({'type': 'set_joints', 'positions': [130.0] * 6})
        wait_for(lambda: client.workbench.live.sessions.get('right', {}).get('active'))
    wait_for(lambda: not client.workbench.live.sessions['right']['active'])
    assert client.get('/api/workbench/state').json()['input_pause']['paused']


@pytest.mark.parametrize('field,value', [('positions', [99.0] * 6), ('positions', [0.1]), ('duration', 0)])
def test_invalid_pose_does_not_save_or_move(client, field, value):
    bad = draft()
    bad['keyframes'][0][field] = value
    assert client.put('/api/workbench/poses', json={'side': 'right', 'draft': bad}).status_code in {400, 422}
    assert client.get('/api/workbench/library?side=right').json()['poses'] == []
    np.testing.assert_allclose(client.workbench.feedback('right'), [0] * 6)


def test_primary_side_and_model_are_resolved_before_topics():
    from hand_gesture_primitives.teach_panel.configuration import resolve_hands
    side, models, topics = resolve_hands('left', 'l25', ['right:o6'])
    assert side == 'right' and models == {'right': 'o6'}
    for key in ['gesture_cmd', 'gesture_status', 'gesture_result', 'motion_progress']:
        assert topics[key].startswith('/right_hand/')
    side, models, topics = resolve_hands('left', 'l25', ['left:o6', 'right:l6'], {'gesture_cmd': '/custom/left/cmd'})
    assert side == 'left' and models[side] == 'o6'
    assert topics['gesture_cmd'] == '/custom/left/cmd'
    assert topics['gesture_status'] == '/left_hand/gesture_status'
    assert resolve_hands('left', 'o6', [''])[1] == {'left': 'o6'}


def test_held_name_cannot_change_until_release(client):
    body = {'side': 'right', 'name': 'fist', 'client_id': 'held-owner', 'sequence': 1, 'running': True}
    assert client.post('/api/workbench/held', json=body).status_code == 200
    response = client.post('/api/workbench/held', json={**body, 'name': 'open', 'sequence': 2})
    assert response.status_code == 400
    assert response.json()['detail']['code'] == 'held_release_required'
    assert client.workbench.held['right']['name'] == 'fist'
    assert client.workbench.hands['right'].status()['active'] == 'fist'
    assert client.post('/api/workbench/held', json={**body, 'sequence': 3, 'running': False}).status_code == 200
    assert client.post('/api/workbench/held', json={**body, 'name': 'open', 'sequence': 4}).status_code == 200
    assert client.workbench.held['right']['name'] == 'open'
    assert client.workbench.hands['right'].status()['active'] == 'open'


def test_replaced_browser_cannot_reclaim_held_motion_after_another_page_releases(client):
    first = {'side': 'right', 'name': 'fist', 'client_id': 'first-page', 'sequence': 1, 'running': True}
    second = {'side': 'right', 'name': 'open', 'client_id': 'second-page', 'sequence': 1, 'running': True}
    assert client.post('/api/workbench/held', json=first).status_code == 200
    assert client.post('/api/workbench/stop', json={'side': 'right'}).status_code == 200
    assert client.post('/api/workbench/held', json=second).status_code == 200
    assert client.post('/api/workbench/held', json={**second, 'sequence': 2, 'running': False}).status_code == 200
    rejected = client.post('/api/workbench/held', json={**first, 'sequence': 2})
    assert rejected.status_code == 400 and rejected.json()['detail']['code'] == 'held_release_required'
    assert client.workbench.hands['right'].status()['active'] == 'open'
    assert client.post('/api/workbench/held', json={**first, 'sequence': 3, 'running': False}).status_code == 200
    assert client.post('/api/workbench/held', json={**first, 'sequence': 4}).status_code == 200


def test_ros_runtime_rejects_wrong_model_and_legacy_waits_for_confirmed_preview(tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path / 'fake-ros'))
    mock = WorkbenchHost.mock()
    hand = mock.hands['right']
    hand.service.save(draft())
    hand.install(hand.service.load_library())

    class Client:
        def __init__(self):
            self.model = 'o6'
            self.state = {'active': None, 'execution_kind': 'idle', 'command_pending': False, 'state_confirmed': True}
            self.commands = []

        def snapshot(self):
            return dict(self.state)

        def capabilities(self):
            return {'version': 2, 'model': self.model, 'side': 'right', 'mode': 'primitive'}

        def synchronize(self):
            return self.capabilities()

        def supports_motion(self):
            return True

        def stop(self):
            self.commands.append(('stop', None))
            self.state['command_pending'] = True

        def require_progress(self):
            pass

        def preview(self, entry, *, initial_progress=None):
            self.initial_progress = initial_progress
            self.commands.append(('preview', entry.name))
            self.state['command_pending'] = True
            return 'preview-one'

        def progress(self, value, *, control_id=None):
            assert control_id == "preview-one"
            self.commands.append(('progress', value))

        def live(self, q):
            self.commands.append(('live', q))

        def wait_stopped(self):
            pass

        def close(self):
            pass

    transport = Client()
    hand.client = transport
    host = WorkbenchHost({'right': hand}, backend='ros', feedback_timeout=20)
    try:
        host.start()
        pending = host.submit('legacy', client_id='legacy', side='right', layout='o6',
                              message={'type': 'play_pose', 'name': 'test_grasp', 'keyframe': 'prepare', 'progress': .35})
        wait_for(lambda: bool(transport.commands))
        assert transport.commands == [('stop', None)] and not pending.done()
        transport.state['command_pending'] = False
        assert pending.result(2)['action'] == 'play_pose'
        assert transport.commands[-1][0] == 'preview'
        assert not any(kind == 'progress' for kind, _ in transport.commands)
        assert transport.initial_progress == .35
        host.submit('legacy', client_id='legacy', side='right', layout='o6',
                    message={'type': 'scrub_pose', 'name': 'test_grasp', 'keyframe': 'prepare', 'progress': .45}).result(2)
        assert sum(kind == 'preview' for kind, _ in transport.commands) == 1
        transport.state.update(command_pending=False, execution_kind='preview', active='named_pose:draft', control_id='preview-one')
        wait_for(lambda: transport.commands[-1] == ('progress', .45))
        transport.model = 'l6'
        token = host.legacy_sessions['legacy']['token']
        before = len(transport.commands)
        with pytest.raises(ValueError, match='匹配手型'):
            host.submit('live', side='right', token=token, sequence=1,
                        updates={'thumb_cmc_pitch': .2}).result(2)
        assert len(transport.commands) == before
        with pytest.raises(ValueError, match='匹配手型'):
            host.submit('preview', side='right', token=token, draft=draft()).result(2)
        assert len(transport.commands) == before
    finally:
        host.close()
        mock.close()


def test_readonly_ros_editor_does_not_wait_for_or_stop_an_active_node(tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path / 'readonly-active-ros'))
    mock = WorkbenchHost.mock()
    hand = mock.hands['right']
    actions = []

    class ActiveClient:
        def snapshot(self):
            return {'active': 'fist', 'execution_kind': 'primitive', 'command_pending': False, 'state_confirmed': True}

        def synchronize(self):
            return None

        def capabilities(self):
            return {'version': 2, 'model': 'o6', 'side': 'right', 'mode': 'primitive'}

        def supports_motion(self):
            return True

        def stop(self):
            actions.append('stop')

        def set_speed(self, speed):
            actions.append('speed')

        def preview(self, entry):
            actions.append('preview')

        def live(self, positions):
            actions.append('live')

        def close(self):
            pass

    hand.client = ActiveClient()
    host = WorkbenchHost({'right': hand}, backend='ros', observe_only=True, feedback_timeout=20)
    try:
        with TestClient(create_workbench_app(host)) as client:
            response = client.post('/api/workbench/edit/begin')
            assert response.status_code == 200 and response.json()['ready']
            token = response.json()['token']
            assert client.get('/api/workbench/capture?side=right').status_code == 200
            saved = save(client, draft())
            assert client.post('/api/workbench/preview', json={'side': 'right', 'token': token, 'draft': saved}).status_code == 400
            assert client.post('/api/workbench/live', json={'side': 'right', 'token': token, 'sequence': 1, 'updates': {'thumb_cmc_pitch': .2}}).status_code == 400
            assert client.put('/api/workbench/speed', json={'speed': 2.0}).status_code == 400
            assert client.post('/api/workbench/stop', json={'side': 'right'}).status_code == 400
        assert actions == []
    finally:
        mock.observe_only = True
        mock.close()
