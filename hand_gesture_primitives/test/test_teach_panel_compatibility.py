"""Legacy browser contracts exercised without ROS or hardware."""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hand_gesture_primitives.hand_config import HandConfig, semantic_legacy255_to_hardware
from hand_gesture_primitives.teach_panel.server import create_workbench_app
from hand_gesture_primitives.teach_panel.workbench import HandRuntime, WorkbenchHost
from hand_gesture_primitives.workbench_core import JointAdapter
from hand_gesture_primitives.workbench_runner import HandPrimitiveRunner


def legacy(host, message, *, client='page', side='right', layout='o6'):
    return host.submit('legacy', client_id=client, side=side, layout=layout, message=message).result(3)


def wait_for(check):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError('condition did not become true')


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path))
    value = WorkbenchHost.mock({'right': 'o6', 'left': 'o6'})
    value.submit('save', side='right', draft={
        'name': 'trajectory', 'description': '', 'revision': '',
        'keyframes': [{'name': 'end', 'positions': [.4] * 6, 'duration': .2}],
    }).result(3)
    try:
        yield value
    finally:
        value.close()


def test_legacy_scrub_keeps_anchor_and_rewinds(host):
    message = {'type': 'scrub_pose', 'name': 'trajectory', 'keyframe': 'end'}
    for progress, target in [(.5, .2), (.5, .2), (0., 0.), (1., .4)]:
        legacy(host, {**message, 'progress': progress})
        wait_for(lambda: np.allclose(host.feedback('right'), target))
    assert len(host.hand('right').status()['history']) == 1
    snapshot = legacy(host, {'type': 'snapshot'})
    assert snapshot['playback_progress_mode']
    assert snapshot['motion_drive'] == 'progress'
    assert snapshot['panel_state'] == 'PLAYBACK'


def test_legacy_progress_requires_own_preview_and_stop_invalidates(host):
    legacy(host, {'type': 'scrub_pose', 'name': 'trajectory', 'keyframe': 'end', 'progress': .5})
    first = host.legacy_sessions['page']['token']
    second = host.submit('edit_begin').result(3)['token']
    with pytest.raises(ValueError):
        host.submit('progress', side='right', token=second, progress=.9).result(3)
    host.submit('stop', side='right').result(3)
    with pytest.raises(ValueError):
        host.submit('progress', side='right', token=first, progress=.9).result(3)
    legacy(host, {'type': 'scrub_pose', 'name': 'trajectory', 'keyframe': 'end', 'progress': 0.})
    assert len(host.hand('right').status()['history']) == 2


@pytest.mark.parametrize('product,layout', [('l25', 'l25'), ('g20', 'g20'), ('g20', 'l25'), ('o6', 'o6')])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_legacy_layout_roundtrip_and_single_joint(product, layout, side, tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path))
    seed = WorkbenchHost.mock({side: layout})
    original = seed.hand(side).adapter
    adapter = JointAdapter(product, side, original.joint_names, original.lower, original.upper, original.converter)
    q = np.linspace(.12, .38, len(adapter.joint_names))
    hand = HandRuntime(adapter, runner=HandPrimitiveRunner(product, side, 30, adapter=adapter),
                       feedback=q.copy(), feedback_at=time.monotonic())
    hand.layout_hand_type = layout
    value = WorkbenchHost({side: hand})
    try:
        legacy(value, {'type': 'enter_teach'}, side=side, layout=layout)
        snapshot = legacy(value, {'type': 'snapshot'}, side=side, layout=layout)
        cfg = HandConfig(layout)
        expected = semantic_legacy255_to_hardware(cfg, adapter.converter.radians_to_semantic(q))
        assert len(snapshot['targets']) == cfg.num_joints
        np.testing.assert_allclose(snapshot['targets'], expected)
        # A browser rounds its range inputs to whole hardware units. An unchanged
        # slider must not quantize unrelated controller joints when another moves.
        positions = np.floor(np.asarray(expected) + .5)
        baseline = np.asarray(expected)
        for index in range(cfg.num_joints):
            if not 1 < positions[index] < 254:
                continue
            changed = baseline.copy()
            changed[index] = positions[index] + 1
            desired = adapter.converter.semantic_to_radians(cfg.to_semantic(changed))
            if np.count_nonzero(np.abs(np.asarray(desired) - q) > 1e-8) == 1:
                break
        else:
            raise AssertionError('no independently controlled layout joint')
        positions[index] += 1
        legacy(value, {'type': 'set_joints', 'positions': positions.tolist()}, side=side, layout=layout)
        wait_for(lambda: np.allclose(value.feedback(side), desired))
        changed_joints = np.flatnonzero(np.abs(np.asarray(desired) - q) > 1e-8)
        assert len(changed_joints) == 1
        np.testing.assert_allclose(value.feedback(side), desired)
    finally:
        value.close()
        seed.close()


def test_secondary_g20_uses_its_resolved_layout(tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path))
    value = WorkbenchHost.mock({'left': 'o6', 'right': 'l25'})
    value.hand('right').adapter.model = 'g20'
    value.hand('right').layout_hand_type = 'l25'
    with TestClient(create_workbench_app(value)) as client:
        with client.websocket_connect('/ws?side=right') as ws:
            snapshot = ws.receive_json()
            assert snapshot['hand_type'] == 'l25'
            assert snapshot['product_hand_type'] == 'g20'
            assert len(snapshot['joint_names']) == len(snapshot['feedback']) == 25
            assert len(snapshot['joint_display']['scales']) == 25


def test_scrub_revision_and_keyframe_change_reanchor_once(host):
    message = {'type': 'scrub_pose', 'name': 'trajectory', 'keyframe': 'end', 'progress': .5}
    legacy(host, message)
    wait_for(lambda: np.allclose(host.feedback('right'), .2))
    stored = host.submit('library', side='right').result(3)['poses'][0]
    stored['keyframes'][0]['positions'] = [.5] * 6
    stored['keyframes'].append({'name': 'other', 'positions': [.1] * 6, 'duration': .3})
    host.submit('save', side='right', draft=stored).result(3)
    legacy(host, message)
    wait_for(lambda: np.allclose(host.feedback('right'), .35))
    legacy(host, message)
    assert len(host.hand('right').status()['history']) == 2
    legacy(host, {**message, 'keyframe': 'other', 'progress': 0.})
    time.sleep(.1)
    np.testing.assert_allclose(host.feedback('right'), .35)
    legacy(host, {**message, 'keyframe': 'other', 'progress': 1.})
    wait_for(lambda: np.allclose(host.feedback('right'), .1))
    assert len(host.hand('right').status()['history']) == 3
    np.testing.assert_allclose(host.feedback('left'), 0.)


def test_initial_zero_progress_and_expired_lease_never_resume(host):
    legacy(host, {'type': 'play_pose', 'name': 'trajectory', 'keyframe': 'end'})
    time.sleep(.15)
    np.testing.assert_allclose(host.feedback('right'), 0.)
    token = host.legacy_sessions['page']['token']
    host.submit('edit_release', token=token).result(3)
    assert 'right' not in host.previews
    with pytest.raises(ValueError):
        host.submit('progress', side='right', token=token, progress=1.).result(3)
    np.testing.assert_allclose(host.feedback('right'), 0.)


@pytest.mark.parametrize('was_teach,pref_start,pref_end,override,expected', [
    (True, True, True, None, 'TEACH_EDIT'),
    (True, True, True, False, 'PRIMITIVE_IDLE'),
    (True, True, False, True, 'PRIMITIVE_IDLE'),
    (True, False, True, None, 'PRIMITIVE_IDLE'),
    (True, False, True, True, 'TEACH_EDIT'),
    (False, True, True, None, 'PRIMITIVE_IDLE'),
    (False, True, True, True, 'TEACH_EDIT'),
    (False, False, False, True, 'PRIMITIVE_IDLE'),
])
def test_legacy_stop_preserves_original_preference_precedence(host, was_teach, pref_start, pref_end, override, expected):
    if was_teach:
        legacy(host, {'type': 'enter_teach'})
    legacy(host, {'type': 'set_prefs', 'return_to_teach_after_playback': pref_start})
    legacy(host, {'type': 'play_pose', 'name': 'trajectory'})
    legacy(host, {'type': 'set_prefs', 'return_to_teach_after_playback': pref_end})
    legacy(host, {'type': 'stop_playback', 'resume_teach': override})
    snapshot = legacy(host, {'type': 'snapshot'})
    assert snapshot['panel_state'] == expected
    assert snapshot['return_to_teach_after_playback'] is pref_end
    assert snapshot['playback_name'] == ''
    assert snapshot['muted'] is (expected != 'TEACH_EDIT')
    assert bool(host.legacy_sessions['page']['token']) is (expected == 'TEACH_EDIT')
    assert host.snapshot()['input_pause']['paused']


def test_legacy_auto_returns_only_after_feedback_settles(host):
    legacy(host, {'type': 'enter_teach'})
    legacy(host, {'type': 'play_pose', 'name': 'trajectory'})
    wait_for(lambda: host.hand('right').status()['done'])
    wait_for(lambda: legacy(host, {'type': 'snapshot'})['panel_state'] == 'TEACH_EDIT')
    snapshot = legacy(host, {'type': 'snapshot'})
    np.testing.assert_allclose(snapshot['targets'], snapshot['feedback'])
    assert snapshot['playback_name'] == ''
    assert not snapshot['muted']


def test_legacy_progress_never_finishes_and_auto_default_survives_switch(host):
    legacy(host, {'type': 'enter_teach'})
    legacy(host, {'type': 'play_pose', 'name': 'trajectory'})
    legacy(host, {'type': 'set_progress', 'progress': 1.})
    time.sleep(.3)
    snapshot = legacy(host, {'type': 'snapshot'})
    assert snapshot['panel_state'] == 'PLAYBACK'
    assert snapshot['playback_progress_mode']
    assert snapshot['muted']
    legacy(host, {'type': 'stop_playback'})
    assert legacy(host, {'type': 'snapshot'})['panel_state'] == 'TEACH_EDIT'
    legacy(host, {'type': 'scrub_pose', 'name': 'trajectory', 'keyframe': 'end', 'progress': 1.})
    time.sleep(.3)
    assert legacy(host, {'type': 'snapshot'})['panel_state'] == 'PLAYBACK'
    legacy(host, {'type': 'stop_playback'})
    assert legacy(host, {'type': 'snapshot'})['panel_state'] == 'PRIMITIVE_IDLE'


def test_other_page_replacing_preview_cannot_resume_original_teach(host):
    legacy(host, {'type': 'enter_teach'})
    legacy(host, {'type': 'play_pose', 'name': 'trajectory'})
    other = host.submit('edit_begin').result(3)['token']
    stored = host.submit('library', side='right').result(3)['poses'][0]
    host.submit('preview', side='right', token=other, draft=stored).result(3)
    wait_for(lambda: host.hand('right').status()['done'])
    snapshot = legacy(host, {'type': 'snapshot'})
    assert snapshot['panel_state'] == 'PRIMITIVE_IDLE'
    assert snapshot['muted']
    assert other in host.edit.leases


@pytest.fixture
def ros_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path))
    seed = WorkbenchHost.mock()
    hand = seed.hand('right')
    hand.service.save({'name': 'trajectory', 'keyframes': [
        {'name': 'end', 'duration': .3, 'positions': [.3] * 6},
    ]})
    hand.install(hand.service.load_library())

    class Transport:
        def __init__(self):
            self.state = {'node_id': 'node', 'control_id': None, 'active': None,
                          'execution_kind': 'idle', 'state_confirmed': True,
                          'command_pending': False, 'done': False, 'settle_status': 'idle'}

        def snapshot(self):
            return dict(self.state)

        def synchronize(self):
            return self.capabilities()

        def capabilities(self):
            return {'version': 2, 'node_id': 'node', 'model': 'o6', 'side': 'right',
                    'mode': 'primitive', 'progress_controls': 1}

        def stop(self):
            self.state['command_pending'] = True

        def preview(self, entry, *, initial_progress=None):
            self.state['command_pending'] = True
            return 'preview'

        def require_progress(self):
            pass

        def progress(self, value, *, control_id):
            assert control_id == 'preview'

        def wait_stopped(self):
            pass

        def close(self):
            pass

    transport = Transport()
    hand.client = transport
    value = WorkbenchHost({'right': hand}, backend='ros', feedback_timeout=20)
    try:
        legacy(value, {'type': 'enter_teach'})
        transport.state['command_pending'] = False
        wait_for(lambda: value.edit.state['ready'])
        legacy(value, {'type': 'play_pose', 'name': 'trajectory'})
        transport.state.update(control_id='preview', active='named_pose:draft',
                               execution_kind='preview', command_pending=False)
        yield value, transport
    finally:
        value.close()
        seed.close()


def test_legacy_clock_completion_and_timeout_do_not_return_teach(ros_legacy):
    value, transport = ros_legacy
    transport.state.update(playback_state='completed', overall_progress=1., settle_status='settling')
    time.sleep(.1)
    assert legacy(value, {'type': 'snapshot'})['panel_state'] == 'PLAYBACK'
    transport.state['settle_status'] = 'settle_timeout'
    time.sleep(.1)
    assert legacy(value, {'type': 'snapshot'})['panel_state'] == 'PLAYBACK'
    transport.state.update(done=True, settle_status='settled')
    wait_for(lambda: legacy(value, {'type': 'snapshot'})['panel_state'] == 'TEACH_EDIT')


@pytest.mark.parametrize('protected', [False, True])
def test_legacy_ros_idle_terminal_needs_execution_proof(ros_legacy, protected):
    value, transport = ros_legacy
    transport.state.update(active=None, execution_kind='idle', done=True, settle_status='settled')
    if protected:
        transport.state.update(completed_control_id='preview', completed_kind='preview')
    expected = 'TEACH_EDIT' if protected else 'PRIMITIVE_IDLE'
    wait_for(lambda: legacy(value, {'type': 'snapshot'})['panel_state'] == expected)


def test_legacy_stop_waits_for_ack_before_unmuting(ros_legacy):
    value, transport = ros_legacy
    legacy(value, {'type': 'stop_playback', 'resume_teach': True})
    pending = legacy(value, {'type': 'snapshot'})
    assert pending['panel_state'] == 'PLAYBACK' and pending['muted']
    transport.state.update(active=None, execution_kind='idle', control_id=None, command_pending=False)
    wait_for(lambda: legacy(value, {'type': 'snapshot'})['panel_state'] == 'TEACH_EDIT')
    assert not legacy(value, {'type': 'snapshot'})['muted']
