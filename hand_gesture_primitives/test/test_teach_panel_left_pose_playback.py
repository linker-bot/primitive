"""Mock regression for the real-machine left-side, single-pre-frame workflow."""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hand_gesture_primitives.teach_panel.server import create_workbench_app
from hand_gesture_primitives.teach_panel.workbench import WorkbenchHost


def receive_action(socket, action):
    for _ in range(40):
        message = socket.receive_json()
        assert message.get('type') != 'error', message
        if message.get('action') == action:
            assert message['type'] == 'ok'
            return message
    raise AssertionError(f'No websocket acknowledgement for {action}')


@pytest.mark.parametrize('play_options', [{}, {'keyframe': 'pre', 'auto_segment': True}])
def test_left_pre_pose_plays_at_quarter_speed_without_moving_right(tmp_path, monkeypatch, play_options):
    """Whole-pose and one-frame play must work even when the default hand is right."""
    monkeypatch.setenv('HAND_GESTURE_POSE_DIR', str(tmp_path / 'isolated-poses'))
    host = WorkbenchHost.mock({'left': 'o6', 'right': 'o6'}, speed=.25, default_side='right')
    target = [.2501960784, 1.0654901961, .9976470588, .5333333333, .5898039216, 1.0541176471]
    draft = {'name': 'test_01', 'description': '', 'revision': '', 'keyframes': [
        {'name': 'pre', 'positions': target, 'duration': .6},
    ]}
    with TestClient(create_workbench_app(host)) as client:
        saved = client.put('/api/workbench/poses', json={'side': 'left', 'draft': draft})
        assert saved.status_code == 200, saved.text
        assert client.get('/api/workbench/library?side=right').json()['poses'] == []
        assert client.get('/api/workbench/state').json()['default_side'] == 'right'
        with client.websocket_connect('/ws?side=left') as socket:
            initial = socket.receive_json()
            assert initial['hand_side'] == 'left'
            started = time.monotonic()
            socket.send_json({'type': 'play_pose', 'name': 'test_01', **play_options})
            receive_action(socket, 'play_pose')
            observed_intermediate = False
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                snapshot = client.get('/api/workbench/state').json()
                left = snapshot['hands']['left']
                np.testing.assert_allclose(host.feedback('right'), np.zeros(6), atol=1e-12)
                assert snapshot['hands']['right']['active'] in {None, 'hold', 'idle'}
                assert snapshot['hands']['right']['execution_kind'] == 'idle'
                if left['playback_state'] == 'running' and .1 < left['overall_progress'] < .8:
                    observed_intermediate = True
                    assert snapshot['primitive_speed'] == .25
                    assert np.max(np.abs(host.feedback('left') - target)) > .05
                if left['done'] and left['settle_status'] == 'settled':
                    break
                time.sleep(.02)
            else:
                raise AssertionError(f'Left playback did not settle: {left}')
            assert observed_intermediate
            assert time.monotonic() - started >= 2.0  # .6 s / .25, allowing scheduler boundaries.
            np.testing.assert_allclose(host.feedback('left'), target, atol=1e-6)
            np.testing.assert_allclose(host.feedback('right'), np.zeros(6), atol=1e-12)
