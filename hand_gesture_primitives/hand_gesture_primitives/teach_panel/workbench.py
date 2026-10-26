"""Standalone workbench host: one control thread, optional ROS, no LeRobot dependency.

Only the in-process mock runtime produces simulated joint feedback. For ROS, all
motion goes through the shared v2 client; gesture_node remains the sole writer.
Library I/O runs on a separate worker so a file lock cannot block motion/stop.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import queue
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..hand_profile import load_hand_profile
from ..workbench_catalog import PrimitiveMenu
from ..workbench_core import JointAdapter, PoseDraft, PrimitiveRejectedError, WorkbenchService
from ..workbench_edit import EditLeaseManager, LiveController
from ..workbench_motion import validate_speed
from ..workbench_runner import HandPrimitiveRunner
from ..workbench_support import pose_revision, primitive_reason

logger = logging.getLogger(__name__)
Side = Literal['left', 'right']


@dataclass
class HandRuntime:
    adapter: JointAdapter
    runner: object = None
    client: object = None
    feedback: np.ndarray | None = None
    feedback_at: float = 0.0
    feedback_id: int = 0
    library: dict = field(default_factory=dict)
    menu: list = field(default_factory=list)
    order: list = field(default_factory=list)
    layout_hand_type: str = ""

    def __post_init__(self):
        self.layout_hand_type = self.layout_hand_type or self.adapter.model
        self.service = WorkbenchService(self.adapter)
        self.profile = load_hand_profile(self.adapter.model)
        self.install(self.service.load_library())

    def install(self, library):
        self.library = library
        if not hasattr(self, 'primitive_menu'):
            self.primitive_menu = PrimitiveMenu(self.adapter.model)
        requested = library['preferences']
        self.primitive_menu.configure(
            requested if requested is not None else self.primitive_menu.menu,
            [f'named_pose:{name}' for name in library['entries']],
        )
        self.menu, self.order = self.primitive_menu.menu, self.primitive_menu.order

    def status(self):
        if self.client is not None:
            return self.client.snapshot()
        return asdict(self.runner.status)


class WorkbenchHost:
    def __init__(self, hands: dict[str, HandRuntime], *, backend='mock', observe_only=False,
                 speed=1.0, control_rate=30.0, feedback_timeout=1.0, default_side=None):
        if not hands or set(hands) - {'left', 'right'}:
            raise ValueError('至少配置一只 left/right 机械手')
        for side, hand in hands.items():
            if hand.adapter.side != side:
                raise ValueError('机械手侧别不匹配')
        if default_side is None:
            default_side = 'right' if 'right' in hands else next(iter(hands))
        if default_side not in hands:
            raise ValueError('default_side must refer to a configured hand')
        self.default_side = default_side
        self.hands = hands
        self.backend, self.observe_only = backend, bool(observe_only)
        self.speed, self.control_rate = validate_speed(speed), float(control_rate)
        self.feedback_timeout = float(feedback_timeout)
        if not 1 <= self.control_rate <= 200:
            raise ValueError('control_rate must be between 1 and 200 Hz')
        self.commands = queue.Queue(maxsize=512)
        self.io = ThreadPoolExecutor(1, thread_name_prefix='gesture-library')
        self.pending_io = None
        self.pending_speed = None
        self.pending_progress = {}
        self.previews = {}
        self.pending_legacy = []
        self.held = {}
        self.held_sequences = {}
        self.held_release_required = set()
        self.legacy_sessions = {}
        self._feedback_lock = threading.Lock()
        self._incoming_feedback = {}
        self._snapshot_lock = threading.Lock()
        self._snapshot = {}
        self._halt = threading.Event()
        self._thread = None
        self.closed = False
        self.fault = ''
        self.edit = EditLeaseManager(on_pause=self._pause, ready=self._ready,
                                     neutral=lambda: not any(value.get('running') for value in self.held.values()))
        self.live = LiveController(self.edit, adapter=lambda side: self.hand(side).adapter,
                                   feedback=self.feedback, send=self._live_target, stop=self._stop,
                                   controllable=self.controllable)
        self._publish_snapshot()

    @classmethod
    def mock(cls, models=None, *, observe_only=False, speed=1.0, control_rate=30.0, default_side=None):
        from ..joint_state_mapping import ros2_control_joint_names

        hands = {}
        for side, model in (models or {'right': 'o6'}).items():
            names = [name.split('_hand_', 1)[-1] for name in ros2_control_joint_names(model, side)]
            upper = [.58, 1.3, 1.6, 1.6, 1.6, 1.6] if model in {'o6', 'l6'} else [np.pi / 2] * len(names)
            adapter = JointAdapter.from_limits(model, side, names, [0.0] * len(names), upper)
            runner = HandPrimitiveRunner(model, side, control_rate, adapter=adapter, speed_scale=speed)
            hands[side] = HandRuntime(adapter, runner=runner, feedback=np.zeros(len(names)), feedback_at=time.monotonic())
        return cls(hands, backend='mock', observe_only=observe_only, speed=speed,
                   control_rate=control_rate, default_side=default_side)

    def hand(self, side):
        if side not in self.hands:
            raise PrimitiveRejectedError('当前工作台未配置此侧机械手', 'invalid_side')
        return self.hands[side]

    def controllable(self):
        if self.observe_only:
            raise PrimitiveRejectedError('只读观察模式不接受动作指令', 'observe_only')

    def feedback(self, side):
        hand = self.hand(side)
        if hand.feedback is None or time.monotonic() - hand.feedback_at > self.feedback_timeout:
            return None
        try:
            return hand.adapter.validate_q(hand.feedback).copy()
        except ValueError:
            return None

    def update_feedback(self, side, positions):
        """ROS callbacks coalesce feedback; they never mutate execution state."""
        self.hand(side)
        with self._feedback_lock:
            self._incoming_feedback[side] = (np.asarray(positions, dtype=float).copy(), time.monotonic())

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='gesture-workbench', daemon=True)
        self._thread.start()

    def submit(self, operation, **values):
        reply = Future()
        if self.closed:
            reply.set_exception(PrimitiveRejectedError('工作台已关闭'))
            return reply
        self.start()
        try:
            self.commands.put_nowait((operation, values, reply))
        except queue.Full:
            reply.set_exception(PrimitiveRejectedError('控制请求过多，请稍后重试', 'busy'))
        return reply

    async def request(self, operation, **values):
        try:
            return await asyncio.wrap_future(self.submit(operation, **values))
        except PrimitiveRejectedError as exc:
            raise HTTPException(409 if exc.code == 'revision_conflict' else 400,
                                detail={'code': exc.code, 'reason': exc.reason}) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, detail={'code': 'invalid_request', 'reason': str(exc)}) from exc
        except Exception as exc:
            logger.exception('workbench request failed')
            raise HTTPException(500, detail={'code': 'workbench_error', 'reason': str(exc)}) from exc

    def snapshot(self):
        with self._snapshot_lock:
            return copy.deepcopy(self._snapshot)

    def _publish_snapshot(self):
        snapshot = {'backend': self.backend, 'observe_only': self.observe_only,
                    'default_side': self.default_side,
                    'primitive_speed': self.speed, 'control_rate': self.control_rate,
                    'input_pause': dict(self.edit.state), 'library_busy': bool(self.pending_io),
                    'fault': self.fault, 'hands': {}}
        for side, hand in self.hands.items():
            try:
                status = hand.status()
            except Exception as exc:
                status = {'active': None, 'hold_reason': str(exc), 'state_confirmed': False}
            snapshot['hands'][side] = {'model': hand.adapter.model, **status,
                                      'feedback_valid': self.feedback(side) is not None}
        with self._snapshot_lock:
            self._snapshot = snapshot

    def _run(self):
        period = 1.0 / self.control_rate
        while not self._halt.is_set():
            started = time.monotonic()
            try:
                with self._feedback_lock:
                    received, self._incoming_feedback = self._incoming_feedback, {}
                for side, (q, stamp) in received.items():
                    hand = self.hand(side)
                    hand.feedback, hand.feedback_at = q, stamp
                    hand.feedback_id += 1
                self._poll_io()
                self._poll_speed()
                for _ in range(32):
                    try:
                        operation, values, reply = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if reply.cancelled():
                        continue
                    try:
                        result = self._dispatch(operation, values, reply)
                        if result is not _DEFERRED and not reply.done():
                            reply.set_result(result)
                    except Exception as exc:
                        if not reply.done():
                            reply.set_exception(exc)
                self.live.poll()
                self._poll_legacy()
                for side, held in self.held.items():
                    if held.get('running') and time.monotonic() > held['deadline']:
                        held.update(running=False, expired=True)
                        self.held_release_required.add((held['client_id'], side))
                        self._running(side, False)
                if self.backend == 'mock':
                    for hand in self.hands.values():
                        hand.feedback_at = time.monotonic()
                        hand.feedback_id += 1
                        if not self.observe_only:
                            target = hand.runner.tick(hand.feedback.copy(), feedback_id=hand.feedback_id)
                            if target is not None:
                                hand.feedback = hand.adapter.validate_q(target).copy()
                else:
                    for hand in self.hands.values():
                        hand.client.synchronize()
                self._poll_progress()
                self._poll_legacy_playback()
                self.fault = ''
            except Exception as exc:
                logger.exception('workbench tick failed')
                self.fault = str(exc)
                for side in self.hands:
                    try:
                        self._stop(side)
                    except Exception:
                        logger.exception('could not stop %s after workbench fault', side)
            self._publish_snapshot()
            self._halt.wait(max(0.0, period - (time.monotonic() - started)))

    def _pause(self):
        for side in self.hands:
            if not self.observe_only:
                self._stop(side)

    def _ready(self):
        if self.observe_only:
            return True
        return all(not (status := hand.status()).get('command_pending')
                   and status.get('state_confirmed', True)
                   and status.get('active') in {None, 'hold', 'idle'} for hand in self.hands.values())

    def _revoke_held(self, side):
        if side in self.held:
            current = self.held[side]
            if not current.get('released'):
                self.held_release_required.add((current['client_id'], side))
            current.update(running=False, revoked=True)

    def _forget_preview(self, side):
        self.previews.pop(side, None)
        self.pending_progress.pop(side, None)
        for session in self.legacy_sessions.values():
            if session.get("preview_side") == side:
                session.pop("preview_key", None)
                session.pop("preview_id", None)

    def _stop(self, side):
        hand = self.hand(side)
        self._forget_preview(side)
        self._revoke_held(side)
        self.pending_progress.pop(side, None)
        keep = []
        for pending in self.pending_legacy:
            if pending[2]['side'] == side:
                if not pending[3].done():
                    pending[3].set_exception(PrimitiveRejectedError('试播已被停止请求取消'))
            else:
                keep.append(pending)
        self.pending_legacy = keep
        if hand.client is not None:
            hand.client.stop()
        else:
            q = self.feedback(side)
            if q is None:
                hand.runner.stop()
            else:
                hand.runner.hold(q)
        if side in self.live.sessions:
            self.live.sessions[side].update(active=False)

    def _live_target(self, side, q):
        hand = self.hand(side)
        self._revoke_held(side)
        if hand.client is not None:
            self._require_ros_hand(hand)
            hand.client.live(list(q))
        else:
            hand.runner.live_target(q)
        self._forget_preview(side)

    def _running(self, side, running):
        hand = self.hand(side)
        if hand.client is not None:
            hand.client.playback(running)
        else:
            hand.runner.set_running(running)

    def _require_ros_hand(self, hand):
        report = hand.client.capabilities()
        if (not report or report.get('version') != 2
                or report.get('side') != hand.adapter.side
                or report.get('model') != hand.profile.hand_type):
            raise PrimitiveRejectedError('外部节点未提供匹配手型和侧别的 v2 能力状态')
        if report.get('mode') != 'primitive':
            raise PrimitiveRejectedError('外部节点未处于原语模式')
        return report

    def catalog(self, side):
        hand = self.hand(side)
        report = hand.client.capabilities() if hand.client is not None else None
        signals = {'position': self.feedback(side) is not None, 'current': False, 'torque': False,
                   'tactile': False, 'vision': False, 'gate': False}
        rows = hand.service.catalog()
        for name, entry in hand.library['entries'].items():
            rows.append({'name': f'named_pose:{name}', 'label': name, 'supported': True,
                         'category': 'custom', 'group': 'custom', 'description': entry.description,
                         'revision': pose_revision(entry), 'keyframe_count': len(entry.effective_keyframes()),
                         'supports_hold': True, 'hold_reason': ''})
        motion = hand.client is None or hand.client.supports_motion()
        for item in rows:
            name = item['name']
            if name.startswith('named_pose:'):
                pose_name = name.split(':', 1)[1]
                entry = hand.library['entries'][pose_name]
                reason = hand.library['errors'].get(pose_name, '')
                if not reason and not signals['position']:
                    reason = '等待有效、未过期的关节位置反馈'
                if not reason and hand.client is not None:
                    if not report or report.get('version') != 2 or report.get('model') != hand.profile.hand_type or report.get('side') != side:
                        reason = '外部节点尚未提供匹配的 v2 能力状态'
                    elif report.get('mode') != 'primitive':
                        reason = '外部节点未处于原语模式'
                    elif entry.unit == 'rad' and not report.get('pose_capabilities', {}).get('radian_converter'):
                        reason = '外部节点未配置弧度姿态换算器'
                    elif report.get('library_error'):
                        reason = report['library_error']
                    elif report.get('poses', {}).get(pose_name) != pose_revision(entry):
                        reason = '外部节点未加载同一版本姿态，请共享 HAND_GESTURE_POSE_DIR'
            elif hand.client is not None:
                try:
                    self._require_ros_hand(hand)
                    reason = report.get('reasons', {}).get(name, '等待外部原语节点的能力状态')
                except PrimitiveRejectedError as exc:
                    reason = exc.reason
                if not signals['position']:
                    reason = '等待有效、未过期的关节位置反馈'
            else:
                reason = primitive_reason(hand.profile, name, signals)
            if not reason and self.observe_only:
                reason = '只读观察模式不接受动作指令'
            item.update(available=not bool(reason), reason=reason)
            if not motion:
                item.update(supports_hold=False, hold_reason='外部节点需升级以支持按住推进')
        return {'model': hand.adapter.model, 'side': side, 'primitives': rows,
                'menu': list(hand.menu), 'id_map': list(hand.order), 'motion_controls': int(motion)}

    def _select(self, side, name, *, held=False, entry=None, preview=False, initial_progress=None):
        self.controllable()
        hand = self.hand(side)
        q = self.feedback(side)
        if q is None:
            raise PrimitiveRejectedError('当前关节反馈缺失或过期')
        if not preview:
            item = next((item for item in self.catalog(side)['primitives'] if item['name'] == name), None)
            if item is None or not item['available']:
                raise PrimitiveRejectedError(item['reason'] if item else '未知原语')
            if held and not item.get('supports_hold'):
                raise PrimitiveRejectedError(item.get('hold_reason') or '此原语不支持按住推进')
            if name.startswith('named_pose:'):
                entry = hand.library['entries'][name.split(':', 1)[1]]
        if hand.client is not None:
            self._require_ros_hand(hand)
            if preview:
                control_id = hand.client.preview(entry, initial_progress=initial_progress)
            else:
                hand.client.send_primitive(name, entry, held=held)
        else:
            hand.runner.select(name, q, entry=entry, execution_kind='preview' if preview else None, held=held)
            control_id = uuid.uuid4().hex
            if initial_progress is not None:
                hand.runner.set_progress(initial_progress)
        self._forget_preview(side)
        return control_id if preview else None

    def _held(self, values):
        self.controllable()
        side, client, sequence = values['side'], values['client_id'], values['sequence']
        hand = self.hand(side)
        key = (client, side)
        if sequence <= self.held_sequences.get(key, -1):
            raise PrimitiveRejectedError('按住推进请求已过期', 'held_stale')
        self.held_sequences[key] = sequence
        current = self.held.get(side)
        if not values['running']:
            self.held_release_required.discard(key)
            if current and current['client_id'] == client:
                if current.get('running') and not current.get('revoked'):
                    self._running(side, False)
                current.update(running=False, released=True, expired=False)
            return {'ok': True}
        if self.edit.paused:
            raise PrimitiveRejectedError('请关闭编辑器并显式恢复控制')
        if key in self.held_release_required:
            raise PrimitiveRejectedError('控制权已变化或输入超时，请先松开再按', 'held_release_required')
        if current and current['client_id'] == client and not current.get('released') and current['name'] != values['name']:
            raise PrimitiveRejectedError('按住期间不能切换原语，请先松开再按', 'held_release_required')
        if current and current['client_id'] == client and not current.get('released') and (current.get('revoked') or current.get('expired')):
            raise PrimitiveRejectedError('动作已切换或输入超时，请先松开再按', 'held_release_required')
        if current and current.get('running') and current['client_id'] != client:
            raise PrimitiveRejectedError('另一页面正在按住推进此手', 'held_busy')
        resume = (current and current['name'] == values['name'] and not current.get('revoked')
                  and hand.status().get('playback_state') not in {'completed', 'idle'}
                  and hand.status().get('active') == values['name'])
        if not current or current.get('released') or current['client_id'] != client:
            if resume:
                self._running(side, True)
            else:
                self._select(side, values['name'], held=True)
        else:
            self._running(side, True)
        self.held[side] = {**values, 'deadline': time.monotonic() + .5, 'released': False,
                           'revoked': False, 'expired': False}
        return {'ok': True}

    def _io_operation(self, kind, values, reply, transform=None):
        side = values['side']
        hand = self.hand(side)
        if self.pending_io:
            if kind == 'library':
                return copy.deepcopy(hand.library['public'])
            raise PrimitiveRejectedError('姿态库正在更新，请稍后重试', 'library_busy')
        if kind == 'delete' and hand.status().get('active') == f"named_pose:{values['name']}":
            raise PrimitiveRejectedError('请先停止正在执行的姿态，再删除')
        if kind == 'menu':
            known = {item['name'] for item in self.catalog(side)['primitives'] if item['supported'] and not item.get('template')}
            if len(values['names']) != len(set(values['names'])) or set(values['names']) - known:
                raise PrimitiveRejectedError('菜单包含重复、未知或当前手型不支持的原语')

        def work():
            if kind == 'save':
                result = hand.service.save(values['draft'])
            elif kind == 'delete':
                hand.service.delete(values['name'], values['revision'])
                result = {'ok': True}
            elif kind == 'menu':
                hand.service.menu_preferences(values['names'])
                result = {'menu': values['names']}
            elif kind.startswith('bundle_'):
                result = self._bundle(kind, hand, values)
            else:
                result = None
            library = hand.service.load_library()
            if kind == 'library':
                result = library['public']
            return result, library

        self.pending_io = self.io.submit(work), reply, side, transform
        return _DEFERRED

    def _poll_io(self):
        if not self.pending_io or not self.pending_io[0].done():
            return
        worker, reply, side, transform = self.pending_io
        self.pending_io = None
        try:
            value, library = worker.result()
            self.hand(side).install(library)
            if transform is not None:
                value = transform(value)
            if not reply.done():
                reply.set_result(value)
        except Exception as exc:
            if not isinstance(exc, (PrimitiveRejectedError, ValueError, ValidationError)):
                exc = PrimitiveRejectedError(f'姿态库操作失败：{exc}', 'library_error')
            if not reply.done():
                reply.set_exception(exc)

    def _poll_speed(self):
        if not self.pending_speed:
            return
        reply, speed, requests = self.pending_speed
        results = [self.hand(side).client.motion_result(request) for side, request in requests.items()]
        if any(value is None for value in results):
            return
        self.pending_speed = None
        errors = [value['reason'] for value in results if not value['accepted']]
        if not reply.done():
            if errors:
                reply.set_exception(PrimitiveRejectedError('；'.join(errors)))
            else:
                self.speed = speed
                reply.set_result({'speed': speed})

    def _dispatch(self, kind, values, reply):
        if kind in {'library', 'save', 'delete', 'menu', 'bundle_export', 'bundle_import', 'bundle_factory'}:
            return self._io_operation(kind, values, reply)
        if kind == 'catalog':
            return self.catalog(values['side'])
        if kind == 'capture':
            q = self.feedback(values['side'])
            if q is None:
                raise PrimitiveRejectedError('当前关节反馈缺失或过期，不能读取手姿')
            return {'positions': q.tolist()}
        if kind == 'edit_begin':
            return self.edit.begin()
        if kind == 'edit_heartbeat':
            return self.edit.heartbeat(values['token'])
        if kind == 'edit_release':
            result = self.live.release(values['token'])
            for side, preview in list(self.previews.items()):
                if preview['token'] == values['token']:
                    self._forget_preview(side)
            return result
        if kind == 'resume':
            return self.edit.resume()
        if kind in {'live', 'live_stop'}:
            self.hand(values['side'])
            return self.live.request(**values, stop=kind == 'live_stop')
        if kind == 'held':
            return self._held(values)
        if kind == 'legacy':
            return self._legacy(values, reply)
        self.controllable()
        if kind == 'stop':
            self._stop(values['side'])
            if values['side'] in self.live.sessions:
                self.live.sessions[values['side']]['expired'] = True
            return {'ok': True}
        if kind == 'speed':
            speed = validate_speed(values['speed'])
            if self.pending_speed:
                raise PrimitiveRejectedError('上一条速度修改尚未确认')
            if self.backend == 'mock':
                for hand in self.hands.values():
                    hand.runner.set_speed(speed)
                self.speed = speed
                return {'speed': speed}
            for hand in self.hands.values():
                self._require_ros_hand(hand)
                hand.client.require_motion()
            requests = {side: hand.client.set_speed(speed) for side, hand in self.hands.items()}
            self.pending_speed = reply, speed, requests
            return _DEFERRED
        if kind == 'select':
            if self.edit.paused:
                raise PrimitiveRejectedError('请关闭编辑器并显式恢复控制')
            self._select(values['side'], values['name'])
            self._revoke_held(values['side'])
            return {'ok': True}
        if kind in {'preview', 'progress'}:
            self.edit.poll()
            if values.get('token') not in self.edit.leases:
                raise PrimitiveRejectedError('编辑占用已过期，请重新打开编辑器', 'edit_expired')
            if not self.edit.state['ready']:
                raise PrimitiveRejectedError('停止手势尚未确认')
            side = values['side']
            hand = self.hand(side)
            live = self.live.sessions.get(side)
            if live and live['active'] and live['token'] != values['token']:
                raise PrimitiveRejectedError('另一页面正在实时控制此手', 'live_busy')
            self.live.stop(side)
            if kind == 'preview':
                entry = hand.service.draft_entry(values['draft'])
                entry = replace(entry, name='draft_' + values['token'])
                initial_progress = values.get('initial_progress')
                if initial_progress is not None:
                    self._validate_progress(initial_progress)
                    if hand.client is not None:
                        hand.client.require_progress()
                control_id = self._select(side, 'named_pose:' + entry.name, entry=entry, preview=True,
                                          initial_progress=initial_progress)
                status = hand.status()
                self.previews[side] = {
                    'token': values['token'], 'control_id': control_id,
                    'node_id': status.get('node_id'), 'confirmed': hand.client is None,
                    'deadline': time.monotonic() + 2.0,
                }
                self._revoke_held(side)
                return {'ok': True, 'control_id': control_id}
            self._validate_progress(values['progress'])
            preview = self.previews.get(side)
            if not preview or preview['token'] != values['token']:
                raise PrimitiveRejectedError('请先试播此页面的当前帧或整个动作', 'preview_changed')
            ready = self._preview_ready(side, preview)
            if ready is None:
                raise PrimitiveRejectedError('试播已改变或尚未确认，请重新试播', 'preview_changed')
            if not ready:
                self.pending_progress[side] = (dict(preview), values['progress'])
            else:
                self._apply_preview_progress(side, preview, values['progress'])
            return {'ok': True}
        raise ValueError(f'未知工作台操作：{kind}')

    @staticmethod
    def _validate_progress(progress):
        if isinstance(progress, bool) or not np.isfinite(progress) or not 0 <= progress <= 1:
            raise PrimitiveRejectedError('progress 必须是 0 到 1 的有限数值')

    def _preview_ready(self, side, preview):
        """False means awaiting this preview's acknowledgement; None means revoked."""
        if (self.previews.get(side, {}).get('control_id') != preview['control_id']
                or preview['token'] not in self.edit.leases):
            return None
        hand = self.hand(side)
        status = hand.status()
        if hand.client is None:
            return True if status.get('execution_kind') == 'preview' else None
        if status.get('node_id') != preview['node_id']:
            return None
        if (status.get('control_id') == preview['control_id']
                and status.get('execution_kind') == 'preview'
                and status.get('state_confirmed')):
            self.previews[side]['confirmed'] = True
            return True
        if not preview['confirmed'] and status.get('command_pending') and time.monotonic() < preview['deadline']:
            return False
        return None

    def _apply_preview_progress(self, side, preview, progress):
        hand = self.hand(side)
        if hand.client is not None:
            hand.client.progress(progress, control_id=preview['control_id'])
        else:
            hand.runner.set_progress(progress)

    def _poll_progress(self):
        for side, preview in list(self.previews.items()):
            if self._preview_ready(side, preview) is None:
                self._forget_preview(side)
        for side, (preview, progress) in list(self.pending_progress.items()):
            ready = self._preview_ready(side, preview)
            if ready is not False:
                self.pending_progress.pop(side, None)
            if ready:
                self._apply_preview_progress(side, preview, progress)

    def _poll_legacy(self):
        pending, self.pending_legacy = self.pending_legacy, []
        for deadline, token, values, reply in pending:
            if reply.done():
                continue
            if token not in self.edit.leases or values['client_id'] not in self.legacy_sessions:
                reply.set_exception(PrimitiveRejectedError('编辑占用已结束', 'edit_expired'))
            elif time.monotonic() > deadline:
                reply.set_exception(PrimitiveRejectedError('停止手势尚未确认，请重试'))
            elif self.edit.state['ready']:
                try:
                    result = self._legacy(values, reply)
                    if result is not _DEFERRED:
                        reply.set_result(result)
                except Exception as exc:
                    reply.set_exception(exc)
            else:
                self.pending_legacy.append((deadline, token, values, reply))

    def _bundle(self, kind, hand, values):
        from ..pose_bundle import build_bundle, collect_library_entries, import_bundle, read_bundle
        from ..pose_library import list_pose_summaries

        model, side = hand.adapter.model, hand.adapter.side
        if kind == 'bundle_export':
            entries = collect_library_entries(hand_type=model, hand_side=side)
            return build_bundle(entries, exporter='teach_panel').to_dict(with_checksum=True)
        if kind == 'bundle_factory':
            from ..export_primitive_poses import export_primitive_poses

            count, _, names = export_primitive_poses(model, side, set_name='planable', write_library=True, overwrite=values['overwrite'])
            return {'success': count > 0, 'count': count, 'names': names,
                    'message': f'factory imported={count}', 'poses': list_pose_summaries(model, hand_side=side)}
        with tempfile.TemporaryDirectory(prefix='gesture-import-') as directory:
            path = Path(directory) / 'poses.json'
            path.write_bytes(values['raw'])
            bundle = read_bundle(path, verify_checksum=values['verify_checksum'])
            result = import_bundle(bundle, strategy=values['strategy'], hand_type=model, hand_side=side, default_side=side)
        return {'success': result.success, 'message': result.summary, 'imported': result.imported,
                'skipped': result.skipped, 'failed': result.failed,
                'poses': list_pose_summaries(model, hand_side=side),
                'filter_hand_type': model, 'filter_hand_side': side}

    def _finish_legacy_playback(self, session, *, resume_teach=None, success=True):
        playback = session.get('playback')
        if playback is None:
            return
        # Match PanelController: an explicit override still cannot bypass the
        # persistent preference, and changing the preference does not rewrite
        # the default captured when playback began.
        resume = playback['resume_default'] if resume_teach is None else bool(resume_teach)
        resume = bool(success and resume and session['return_to_teach_after_playback']
                      and session['token'] in self.edit.leases)
        side = playback['side']
        if self.previews.get(side, {}).get('control_id') == playback['control_id']:
            self._forget_preview(side)
        session.update(playback=None, progress_mode=False,
                       panel_state='TEACH_EDIT' if resume else 'PRIMITIVE_IDLE')
        if resume:
            q = self.feedback(side)
            if q is not None:
                session['targets'] = q.copy()
        else:
            self.live.release(session['token'])
            session['token'] = ''

    def _poll_legacy_playback(self):
        for session in self.legacy_sessions.values():
            playback = session.get('playback')
            if playback is None:
                if session.get('panel_state') == 'TEACH_EDIT' and session['token'] not in self.edit.leases:
                    session['panel_state'] = 'PRIMITIVE_IDLE'
                continue
            side = playback['side']
            hand = self.hand(side)
            status = hand.status()
            if session['token'] not in self.edit.leases:
                self._finish_legacy_playback(session, success=False)
                continue
            if playback.get('stopping'):
                stopped = (not status.get('command_pending') and status.get('state_confirmed', True)
                           and status.get('execution_kind') == 'idle'
                           and status.get('active') in {None, 'hold', 'idle'})
                if stopped:
                    self._finish_legacy_playback(session, resume_teach=playback['resume_override'])
                elif time.monotonic() > playback['deadline']:
                    # Keep the stop action available and remain muted until a
                    # correlated idle acknowledgement or a lease expiry arrives.
                    session['error'] = '停止手势尚未确认，请重试停止'
                continue
            if hand.client is None:
                owned = self.previews.get(side, {}).get('control_id') == playback['control_id']
            else:
                owned = (status.get('node_id') == playback['node_id']
                         and status.get('control_id') == playback['control_id']
                         and status.get('execution_kind') in {'preview', 'idle'})
                if status.get('execution_kind') == 'idle':
                    owned = (owned and status.get('completed_control_id') == playback['control_id']
                             and status.get('completed_kind') == 'preview')
            if not owned:
                if (hand.client is not None and not playback['confirmed']
                        and status.get('command_pending') and time.monotonic() < playback['deadline']):
                    continue
                self._finish_legacy_playback(session, success=False)
                continue
            playback['confirmed'] = True
            if (not playback['progress_mode'] and not status.get('command_pending')
                    and status.get('state_confirmed', True) and status.get('done')
                    and status.get('settle_status') == 'settled'):
                self._finish_legacy_playback(session)

    def _legacy(self, values, reply):
        """Translate the original websocket vocabulary, including per-page CAS."""
        from ..hand_config import HandConfig, semantic_legacy255_to_hardware

        client, side, msg = values['client_id'], values['side'], values['message']
        hand = self.hand(side)
        layout = values.get('layout') or hand.adapter.model
        session = self.legacy_sessions.setdefault(client, {
            'token': '', 'sequence': 0, 'name': '', 'keyframe': '', 'revisions': {},
            'targets': None, 'return_to_teach_after_playback': True,
            'panel_state': 'PRIMITIVE_IDLE', 'playback': None, 'error': '',
        })
        kind = str(msg.get('type', ''))
        token = session['token']
        poses = hand.library['public']['poses']
        summaries = [{**pose, 'keyframe_count': len(pose['keyframes'])} for pose in poses]
        for pose in poses:
            session['revisions'].setdefault(pose['name'], pose['revision'])

        def current_pose(name):
            pose = next((pose for pose in poses if pose['name'] == name), None)
            if pose is None:
                raise PrimitiveRejectedError(f'姿态不存在：{name}')
            value = copy.deepcopy(pose)
            value['revision'] = session['revisions'].get(name, '')
            return value

        def begin():
            if session['token'] not in self.edit.leases:
                session['token'] = self.edit.begin()['token']
            return session['token']

        def ui(q):
            if q is None:
                return None
            semantic = hand.adapter.converter.radians_to_semantic(list(q))
            return semantic_legacy255_to_hardware(HandConfig(layout), semantic)

        def response(action=kind, **extra):
            return {'type': 'ok', 'action': action, **extra}

        def saved(result):
            session['name'] = result['name']
            session['revisions'][result['name']] = result['revision']
            return response(name=result['name'], keyframe=session['keyframe'],
                            revision=result['revision'], advance_keyframe=bool(msg.get('advance_keyframe')))

        if kind == 'snapshot':
            q = self.feedback(side)
            targets = session['targets'] if session['targets'] is not None else q
            status = hand.status()
            playback = session.get('playback') or {}
            editing = session['panel_state'] == 'TEACH_EDIT' and token in self.edit.leases
            ready = (editing and self.edit.state['ready'] and q is not None
                     and not status.get('command_pending') and status.get('state_confirmed', True))
            progress_mode = playback.get('progress_mode', False)
            return {
                'type': 'snapshot', 'panel_state': session['panel_state'],
                'mode': 'teach' if editing else 'primitive', 'muted': not ready or self.observe_only,
                'active_primitive': status.get('active') or 'hold', 'targets': ui(targets), 'feedback': ui(q),
                'last_error': session['error'], 'hint': '', 'playback_name': playback.get('name', ''),
                'playback_keyframe': playback.get('keyframe', ''), 'playback_progress_mode': progress_mode,
                'editing_pose_name': session['name'], 'editing_keyframe': session['keyframe'],
                'motion_drive': 'progress' if progress_mode else 'auto', 'motion_keyframe': status.get('keyframe', ''),
                'motion_progress': (playback.get('progress', 0.0) if progress_mode else
                                    max(0.0, status.get('progress', status.get('overall_progress', 0.0)))),
                'num_joints': HandConfig(layout).num_joints, 'poses': summaries,
                'return_to_teach_after_playback': session['return_to_teach_after_playback'],
            }
        if kind in {'heartbeat', 'live_heartbeat'}:
            if token:
                self.edit.heartbeat(token)
                live = self.live.sessions.get(side)
                if kind == 'live_heartbeat' and live and live['active'] and live['token'] == token:
                    session['sequence'] += 1
                    self.live.request(token, side, session['sequence'], {})
            return None
        if kind == 'disconnect':
            if token:
                self.live.release(token)
            self.legacy_sessions.pop(client, None)
            return None
        if kind == 'enter_teach':
            begin()
            if session.get('playback') and self.previews.get(side, {}).get('control_id') == session['playback']['control_id']:
                self._stop(side)
            session.update(panel_state='TEACH_EDIT', playback=None, error='')
            if session['targets'] is None:
                session['targets'] = self.feedback(side)
            return response()
        if kind == 'exit_teach':
            session.update(panel_state='PRIMITIVE_IDLE', playback=None)
            if token:
                self.live.release(token)
            session['token'] = ''
            if not self.edit.leases:
                self.edit.resume()
            return response()
        if kind == 'set_prefs':
            session['return_to_teach_after_playback'] = bool(msg.get('return_to_teach_after_playback', True))
            return response(return_to_teach_after_playback=session['return_to_teach_after_playback'])
        if kind == 'list_poses':
            return {'type': 'poses', 'poses': summaries}
        if kind == 'sync_feedback':
            q = self.feedback(side)
            if q is None:
                raise PrimitiveRejectedError('当前关节反馈缺失或过期')
            session['targets'] = q
            return response()
        if kind in {'clear_edit', 'live_stop'}:
            if token:
                session['sequence'] += 1
                self.live.request(token, side, session['sequence'], {}, stop=True)
            if kind == 'clear_edit':
                session.update(name='', keyframe='')
            return response()
        if kind in {'set_joints', 'flush_joints'}:
            if kind == 'flush_joints':
                return None
            positions = np.asarray(msg.get('positions', []), dtype=float)
            if positions.shape != (HandConfig(layout).num_joints,) or not np.isfinite(positions).all() or np.any(positions < 0) or np.any(positions > 255):
                raise PrimitiveRejectedError('无效的关节目标')
            previous = session['targets'] if session['targets'] is not None else self.feedback(side)
            if previous is None:
                raise PrimitiveRejectedError('当前关节反馈缺失或过期')
            baseline = np.asarray(ui(previous))
            # Old range inputs round to whole hardware units. Preserve exact radian
            # targets for all untouched sliders rather than treating that rounding
            # as a request to move every joint.
            changed = ~np.isclose(positions, baseline, atol=1e-8, rtol=0)
            changed &= ~np.isclose(positions, np.floor(baseline + .5), atol=1e-8, rtol=0)
            merged = np.where(changed, positions, baseline)
            semantic = HandConfig(layout).to_semantic(merged.tolist())
            q = hand.adapter.validate_q(hand.adapter.converter.semantic_to_radians(semantic))
            q = np.where(np.isclose(q, previous, atol=1e-8, rtol=0), previous, q)
            session['targets'] = q
            status = hand.status()
            if (not token or token not in self.edit.leases or session['panel_state'] != 'TEACH_EDIT'
                    or status.get('command_pending') or not status.get('state_confirmed', True)):
                raise PrimitiveRejectedError('muted_or_not_teach')
            updates = {name: float(value) for index, (name, value) in enumerate(zip(hand.adapter.joint_names, q))
                       if previous is None or abs(float(previous[index]) - value) > 1e-9}
            if updates:
                session['sequence'] += 1
                self.live.request(token, side, session['sequence'], updates)
            return None
        if kind == 'load_pose':
            name = str(msg.get('name', '')).strip().lower()
            pose = current_pose(name)
            keyframe = str(msg.get('keyframe', '')).strip().lower()
            target = next((value for value in pose['keyframes'] if value['name'] == keyframe), None) if keyframe else pose['keyframes'][-1]
            if target is None:
                raise PrimitiveRejectedError('关键帧不存在')
            begin()
            self.live.stop(side)
            if session.get('playback') and self.previews.get(side, {}).get('control_id') == session['playback']['control_id']:
                self._stop(side)
            session.update(name=name, keyframe=keyframe, targets=np.asarray(target['positions']),
                           panel_state='TEACH_EDIT', playback=None, error='')
            session['revisions'][name] = next(value['revision'] for value in poses if value['name'] == name)
            return response(name=name, keyframe=keyframe, duration=target['duration'],
                            description=pose['description'], keyframes=pose['keyframes'], revision=session['revisions'][name])
        if kind == 'save_pose':
            name = str(msg.get('name') or session['name']).strip().lower()
            keyframe = str(msg.get('keyframe') or session['keyframe']).strip().lower()
            existing = next((value for value in poses if value['name'] == name), None)
            if existing and len(existing['keyframes']) > 1 and not keyframe:
                raise PrimitiveRejectedError('请指定需要覆盖的关键帧')
            q = session['targets'] if msg.get('use_targets', True) else self.feedback(side)
            if q is None:
                raise PrimitiveRejectedError('无可用关节值')
            draft = current_pose(name) if existing else {'name': name, 'description': '', 'revision': '', 'keyframes': []}
            keyframe = keyframe or (draft['keyframes'][0]['name'] if draft['keyframes'] else 'frame_1')
            value = {'name': keyframe, 'positions': list(q), 'duration': float(msg.get('duration', .6))}
            index = next((index for index, old in enumerate(draft['keyframes']) if old['name'] == keyframe), None)
            if index is None:
                draft['keyframes'].append(value)
            else:
                draft['keyframes'][index] = value
            draft['description'] = str(msg.get('description', draft['description']))
            if 'revision' in msg:
                draft['revision'] = msg['revision']
            session['keyframe'] = keyframe
            return self._io_operation('save', {'side': side, 'draft': draft}, reply, saved)
        if kind == 'delete_pose':
            name = str(msg.get('name', '')).strip().lower()
            revision = msg.get('revision', session['revisions'].get(name, ''))
            return self._io_operation('delete', {'side': side, 'name': name, 'revision': revision}, reply,
                                      lambda _: response(name=name))
        if kind in {'delete_keyframe', 'move_keyframe', 'rename_keyframe'}:
            name, keyframe = str(msg.get('name', '')).strip().lower(), str(msg.get('keyframe', '')).strip().lower()
            draft = current_pose(name)
            index = next((index for index, value in enumerate(draft['keyframes']) if value['name'] == keyframe), None)
            if index is None:
                raise PrimitiveRejectedError('关键帧不存在')
            if kind == 'delete_keyframe':
                if len(draft['keyframes']) == 1:
                    raise PrimitiveRejectedError('不能删除最后一帧，请删除整个原语')
                draft['keyframes'].pop(index)
            elif kind == 'move_keyframe':
                target = index + int(msg.get('delta', 0))
                if not 0 <= target < len(draft['keyframes']):
                    raise PrimitiveRejectedError('关键帧位置超出范围')
                draft['keyframes'][index], draft['keyframes'][target] = draft['keyframes'][target], draft['keyframes'][index]
            else:
                draft['keyframes'][index]['name'] = str(msg.get('new_keyframe', '')).strip().lower()
            def changed(result):
                session['revisions'][name] = result['revision']
                return response(name=name, keyframe=keyframe, new_keyframe=msg.get('new_keyframe', ''))
            return self._io_operation('save', {'side': side, 'draft': draft}, reply, changed)
        if kind in {'play_pose', 'scrub_pose'}:
            name = str(msg.get('name', '')).strip().lower()
            draft = current_pose(name)
            keyframe = str(msg.get('keyframe', '')).strip().lower()
            if keyframe:
                draft['keyframes'] = [value for value in draft['keyframes'] if value['name'] == keyframe]
                if not draft['keyframes']:
                    raise PrimitiveRejectedError('关键帧不存在')
            was_teach = session['panel_state'] == 'TEACH_EDIT'
            edit_token = begin()
            self.edit.poll()
            if not self.edit.state['ready']:
                self.pending_legacy.append((time.monotonic() + 2.5, edit_token, values, reply))
                return _DEFERRED
            if kind == 'scrub_pose' and not keyframe:
                raise PrimitiveRejectedError('关键帧名为空')
            progress = msg.get('progress')
            auto_segment = bool(msg.get('auto_segment')) and bool(keyframe) and progress is None
            progress_mode = (bool(keyframe) or progress is not None) and not auto_segment
            if progress_mode:
                progress = float(progress) if progress is not None else 0.0
                if not np.isfinite(progress):
                    raise PrimitiveRejectedError('progress 无效')
                progress = max(0.0, min(1.0, progress))
            revision = next(value['revision'] for value in poses if value['name'] == name)
            preview_key = (side, client, edit_token, name, revision, keyframe)
            current = self.previews.get(side)
            reuse = (progress_mode and session.get('preview_key') == preview_key
                     and current and session.get('preview_id') == current['control_id']
                     and self._preview_ready(side, current) is not None)
            if reuse:
                self._dispatch('progress', {'side': side, 'token': edit_token, 'progress': progress}, reply)
            else:
                result = self._dispatch('preview', {'side': side, 'token': edit_token, 'draft': draft,
                                                    'initial_progress': progress if progress_mode else None}, reply)
                session.update(preview_key=preview_key, preview_id=result['control_id'], preview_side=side)
            if not reuse or kind == 'play_pose' or session.get('playback') is None:
                preview = self.previews[side]
                session['playback'] = {
                    **preview, 'side': side, 'name': name, 'keyframe': keyframe,
                    'resume_default': bool(session['return_to_teach_after_playback'] and was_teach and not progress_mode),
                }
            session['playback'].update(progress_mode=progress_mode, progress=progress)
            session.update(panel_state='PLAYBACK', progress_mode=progress_mode, error='')
            return response(name=name, keyframe=keyframe, progress=progress)
        if kind == 'set_progress':
            if session.get('playback') is None:
                raise PrimitiveRejectedError('未在试播/细控中')
            progress = float(msg.get('progress', 0))
            if not np.isfinite(progress):
                raise PrimitiveRejectedError('progress 无效')
            progress = max(0.0, min(1.0, progress))
            self._dispatch('progress', {'side': side, 'token': token, 'progress': progress}, reply)
            session['playback'].update(progress_mode=True, progress=progress)
            session['progress_mode'] = True
            return response(progress=progress)
        if kind == 'stop_playback':
            if session.get('playback') is None:
                return response()
            self.controllable()
            self._stop(side)
            session['playback'].update(stopping=True, resume_override=msg.get('resume_teach'),
                                       deadline=time.monotonic() + 2.5)
            self._poll_legacy_playback()
            return response()
        if kind == 'reinit':
            session.update(panel_state='PRIMITIVE_IDLE', playback=None, name='', keyframe='', error='')
            if token:
                self.live.release(token)
                session['token'] = ''
            self.edit.resume()
            self._select(side, 'init')
            return response()
        raise PrimitiveRejectedError(f'unknown_type:{kind}')

    def close(self):
        if self.closed:
            return
        if self._thread is not None and not self.observe_only:
            for side in self.hands:
                try:
                    self.submit('stop', side=side).result(timeout=2)
                except Exception:
                    logger.exception('workbench shutdown stop failed for %s', side)
        self.closed = True
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        for hand in self.hands.values():
            if hand.client is not None:
                try:
                    if not self.observe_only:
                        hand.client.wait_stopped()
                except Exception:
                    logger.exception('ROS stop was not confirmed during shutdown')
                finally:
                    hand.client.close()
        self.io.shutdown(wait=True, cancel_futures=True)
        if self.pending_io:
            self._poll_io()
        while not self.commands.empty():
            _, _, reply = self.commands.get_nowait()
            if not reply.done():
                reply.set_exception(PrimitiveRejectedError('工作台已关闭'))
        for _, _, _, reply in self.pending_legacy:
            if not reply.done():
                reply.set_exception(PrimitiveRejectedError('工作台已关闭'))
        self.pending_legacy.clear()
        if self.pending_speed and not self.pending_speed[0].done():
            self.pending_speed[0].set_exception(PrimitiveRejectedError('工作台已关闭'))


_DEFERRED = object()


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)


class SideRequest(StrictRequest):
    side: Side


class PoseRequest(SideRequest):
    draft: PoseDraft


class PreviewRequest(PoseRequest):
    token: str = Field(min_length=1, max_length=64)


class DeleteRequest(SideRequest):
    name: str = Field(min_length=1, max_length=64)
    revision: str = Field(min_length=1)


class MenuRequest(SideRequest):
    names: list[str] = Field(max_length=200)


class SelectRequest(SideRequest):
    name: str = Field(min_length=1, max_length=128)


class TokenRequest(StrictRequest):
    token: str = Field(min_length=1, max_length=64)


class LiveRequest(TokenRequest):
    side: Side
    sequence: int = Field(ge=0)
    updates: dict[str, float] = Field(default_factory=dict, max_length=64)


class SpeedRequest(StrictRequest):
    speed: float = Field(ge=.25, le=3.0)


class HeldRequest(SelectRequest):
    client_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    running: bool


class ProgressRequest(TokenRequest):
    side: Side
    progress: float = Field(ge=0, le=1)


def install_api(app: FastAPI, host: WorkbenchHost):
    @app.get('/api/workbench/state')
    async def state():
        return host.snapshot()

    @app.get('/api/workbench/catalog')
    async def catalog(side: Side = 'right'):
        return await host.request('catalog', side=side)

    @app.get('/api/workbench/library')
    async def library(side: Side = 'right'):
        return await host.request('library', side=side)

    @app.get('/api/workbench/capture')
    async def capture(side: Side = 'right'):
        return await host.request('capture', side=side)

    @app.put('/api/workbench/poses')
    async def save(body: PoseRequest):
        return await host.request('save', **body.model_dump())

    @app.post('/api/workbench/poses/delete')
    async def delete(body: DeleteRequest):
        return await host.request('delete', **body.model_dump())

    @app.put('/api/workbench/menu')
    async def menu(body: MenuRequest):
        return await host.request('menu', **body.model_dump())

    @app.post('/api/workbench/select')
    async def select(body: SelectRequest):
        return await host.request('select', **body.model_dump())

    @app.post('/api/workbench/preview')
    async def preview(body: PreviewRequest):
        return await host.request('preview', **body.model_dump())

    @app.post('/api/workbench/stop')
    async def stop(body: SideRequest):
        return await host.request('stop', **body.model_dump())

    @app.put('/api/workbench/speed')
    async def speed(body: SpeedRequest):
        return await host.request('speed', **body.model_dump())

    @app.post('/api/workbench/held')
    async def held(body: HeldRequest):
        return await host.request('held', **body.model_dump())

    @app.post('/api/workbench/progress')
    async def progress(body: ProgressRequest):
        return await host.request('progress', **body.model_dump())

    @app.post('/api/workbench/live')
    async def live(body: LiveRequest):
        return await host.request('live', **body.model_dump())

    @app.post('/api/workbench/live/stop')
    async def live_stop(body: LiveRequest):
        return await host.request('live_stop', **body.model_dump())

    @app.post('/api/workbench/edit/begin')
    async def edit_begin():
        return await host.request('edit_begin')

    @app.post('/api/workbench/edit/heartbeat')
    async def edit_heartbeat(body: TokenRequest):
        return await host.request('edit_heartbeat', **body.model_dump())

    @app.post('/api/workbench/edit/release')
    async def edit_release(body: TokenRequest):
        return await host.request('edit_release', **body.model_dump())

    @app.post('/api/workbench/resume')
    async def resume():
        return await host.request('resume')
