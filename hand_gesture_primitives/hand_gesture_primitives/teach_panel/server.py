"""FastAPI + WebSocket Teach 面板服务。"""

from __future__ import annotations

import asyncio
import json
import socket
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..export_primitive_poses import export_primitive_poses
from ..hand_config import HandConfig
from ..joint_coupling import couple_meta, coupling_hint
from ..pose_bundle import (
    MERGE_STRATEGIES,
    build_bundle,
    collect_library_entries,
    import_bundle,
    read_bundle,
)
from ..pose_library import (
    DEFAULT_SAVE_SPACE,
    commit_saved_pose,
    delete_pose,
    delete_pose_keyframe,
    list_pose_summaries,
    load_pose,
    move_pose_keyframe,
    pose_entry_to_hardware,
    prepare_pose_for_save,
    rename_pose_keyframe,
)
from .controller import PanelController, PanelState
from .layout_convert import product_hardware_to_ui, ui_to_product_hardware

STATIC_DIR = Path(__file__).resolve().parent / "static"


class TeachPanelApp:
    def __init__(
        self,
        controller: PanelController,
        *,
        hand_type: str,
        hand_side: str,
        joint_names: List[str],
        lock: Optional[threading.Lock] = None,
        layout_hand_type: str = "",
        radian_converter=None,
        workbench=None,
    ) -> None:
        self.controller = controller
        self.workbench = workbench
        # 姿态库 / gesture 执行手型（产品）
        self.hand_type = hand_type
        # 滑条布局手型（整机 G20 → l25）
        self.layout_hand_type = (layout_hand_type or hand_type).strip().lower()
        self.hand_side = hand_side
        self.joint_names = list(joint_names)
        self._radian_converter = radian_converter
        self._lock = lock or threading.Lock()
        self._clients: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._cached_poses: List[Dict[str, Any]] = list_pose_summaries(
            hand_type, hand_side=hand_side
        )
        self._cached_joint_display: Dict[str, Any] = self._joint_display_meta()
        @asynccontextmanager
        async def lifespan(_app):
            if workbench is not None:
                workbench.start()
            try:
                yield
            finally:
                if workbench is not None:
                    await asyncio.to_thread(workbench.close)

        self.app = FastAPI(title="hand_gesture_primitives teach_panel", lifespan=lifespan)
        self.app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        self.app.get("/")(self._index)
        self.app.get("/legacy")(self._legacy_index)
        self.app.get("/workbench")(self._workbench_index)
        self.app.websocket("/ws")(self._ws)
        self.app.get("/api/poses")(self._api_poses)
        self.app.get("/api/pose_bundle/export")(self._api_export_bundle)
        self.app.post("/api/pose_bundle/import")(self._api_import_bundle)
        self.app.post("/api/pose_bundle/import_factory")(self._api_import_factory)

        if workbench is not None:
            from .workbench import install_api
            self.app.state.workbench = workbench
            install_api(self.app, workbench)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def notify_snapshot(self) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(), self._loop)

    async def _index(self):
        return FileResponse(STATIC_DIR / "index.html")

    async def _workbench_index(self):
        if self.workbench is None:
            raise HTTPException(404, detail="Workbench is not configured")
        return FileResponse(STATIC_DIR / "workbench.html")

    async def _legacy_index(self):
        return FileResponse(STATIC_DIR / "index.html")

    async def _api_poses(self, side: str | None = None):
        if self.workbench is not None:
            library = await self.workbench.request("library", side=side or self.hand_side)
            return {"poses": [{**pose, "keyframe_count": len(pose["keyframes"])} for pose in library["poses"]]}
        return {"poses": self._cached_poses}

    def _bundle_filename(self) -> str:
        return f"{self.hand_type}_{self.hand_side}_poses.pose_bundle.json"

    def _machine_hint(self) -> str:
        try:
            return socket.gethostname()
        except OSError:
            return ""

    async def _api_export_bundle(self, side: str | None = None):
        """下载当前手型/侧 Pose Library 为 pose_bundle JSON。"""
        if self.workbench is not None:
            selected = side or self.hand_side
            payload = await self.workbench.request("bundle_export", side=selected)
            model = self.workbench.hand(selected).adapter.model
            entries = payload.get("entries", [])
            headers = {
                "Content-Disposition": f'attachment; filename="{model}_{selected}_poses.pose_bundle.json"',
                "X-Pose-Count": str(len(entries)),
                "X-Hand-Type": model,
                "X-Hand-Side": selected,
            }
            if not entries:
                headers["X-Pose-Empty"] = "1"
            return Response(json.dumps(payload, ensure_ascii=False, indent=2),
                            media_type="application/json", headers=headers)
        with self._lock:
            entries = collect_library_entries(
                hand_type=self.hand_type,
                hand_side=self.hand_side,
            )
            bundle = build_bundle(
                entries,
                exporter="teach_panel",
                machine_hint=self._machine_hint(),
            )
            payload = bundle.to_dict(with_checksum=True)
        body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        headers = {
            "Content-Disposition": f'attachment; filename="{self._bundle_filename()}"',
            "X-Pose-Count": str(len(entries)),
            "X-Hand-Type": self.hand_type,
            "X-Hand-Side": self.hand_side,
        }
        if not entries:
            headers["X-Pose-Empty"] = "1"
        return Response(
            content=body.encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers=headers,
        )

    async def _api_import_bundle(
        self,
        request: Request,
        strategy: str = Query("skip"),
        verify_checksum: bool = Query(True),
        side: str | None = None,
    ):
        """上传 pose_bundle JSON 正文并合并进当前手型/侧库。"""
        strategy_l = (strategy or "skip").strip().lower()
        if strategy_l not in MERGE_STRATEGIES:
            return JSONResponse(
                {"success": False, "message": f"invalid strategy: {strategy}"},
                status_code=400,
            )
        raw = await request.body()
        if not raw:
            return JSONResponse(
                {"success": False, "message": "empty upload"},
                status_code=400,
            )
        if len(raw) > 8 * 1024 * 1024:
            return JSONResponse(
                {"success": False, "message": "upload too large (max 8MiB)"},
                status_code=413,
            )
        if self.workbench is not None:
            result = await self.workbench.request("bundle_import", side=side or self.hand_side,
                                                  raw=raw, strategy=strategy_l, verify_checksum=bool(verify_checksum))
            return JSONResponse(result, status_code=200 if result["success"] else 409)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pose_bundle.json", delete=False
            ) as tmp:
                tmp.write(raw)
                tmp_path = Path(tmp.name)
            with self._lock:
                bundle = read_bundle(tmp_path, verify_checksum=bool(verify_checksum))
                result = import_bundle(
                    bundle,
                    strategy=strategy_l,
                    hand_type=self.hand_type,
                    hand_side=self.hand_side,
                    default_side=self.hand_side,
                )
                self._refresh_poses()
                poses = list(self._cached_poses)
        except Exception as exc:  # noqa: BLE001 — HTTP 边界
            return JSONResponse(
                {"success": False, "message": str(exc) or type(exc).__name__},
                status_code=400,
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        self.notify_snapshot()
        status = 200 if result.success else 409
        return JSONResponse(
            {
                "success": result.success,
                "message": result.summary,
                "imported": result.imported,
                "skipped": result.skipped,
                "failed": result.failed,
                "poses": poses,
                "filter_hand_type": self.hand_type,
                "filter_hand_side": self.hand_side,
            },
            status_code=status,
        )

    async def _api_import_factory(self, overwrite: bool = Query(False), side: str | None = None):
        """将可规划静态原语终态写入当前库（preset_<prim>），便于本机快速补齐。"""
        if self.workbench is not None:
            result = await self.workbench.request("bundle_factory", side=side or self.hand_side, overwrite=bool(overwrite))
            return JSONResponse(result, status_code=200 if result["success"] else 409)
        with self._lock:
            try:
                count, _, names = export_primitive_poses(
                    self.hand_type,
                    self.hand_side,
                    set_name="planable",
                    write_library=True,
                    overwrite=bool(overwrite),
                )
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(
                    {"success": False, "message": str(exc) or type(exc).__name__},
                    status_code=400,
                )
            self._refresh_poses()
            poses = list(self._cached_poses)
        self.notify_snapshot()
        if count <= 0:
            return JSONResponse(
                {
                    "success": False,
                    "message": (
                        "factory imported=0 "
                        "(all presets already exist or nothing planable; use overwrite)"
                    ),
                    "count": 0,
                    "names": names,
                    "poses": poses,
                },
                status_code=409,
            )
        return {
            "success": True,
            "message": f"factory imported={count}",
            "count": count,
            "names": names,
            "poses": poses,
        }

    def _refresh_poses(self) -> None:
        self._cached_poses = list_pose_summaries(self.hand_type, hand_side=self.hand_side)

    @staticmethod
    def _strip_hand_joint_prefix(name: str) -> str:
        n = str(name or "").strip().lower()
        for prefix in ("l_hand_", "r_hand_"):
            if n.startswith(prefix):
                return n[len(prefix) :]
        return n

    def _joint_display_meta(self, *, joint_names=None, converter=None, hand_type=None, layout_hand_type=None) -> Dict[str, Any]:
        """示教显示层量纲：内部控制始终 hw0~255；可切 rad/deg。

        有 Converter 时用 URDF 限位；否则对未匹配关节用 0~π/2 线性近似（approx=True）。
        """
        joint_names = self.joint_names if joint_names is None else joint_names
        converter = self._radian_converter if converter is None else converter
        hand_type = hand_type or self.hand_type
        layout_hand_type = layout_hand_type or self.layout_hand_type
        n_ui = len(joint_names)
        default_hi = 1.5707963267948966  # π/2
        scales: List[Dict[str, Any]] = [
            {"lower": 0.0, "upper": 0.0, "inverted": False} for _ in range(n_ui)
        ]
        approx = False
        conv = converter

        if conv is not None:
            conv_names = [str(x) for x in (getattr(conv, "joint_names", []) or [])]
            conv_limits = list(getattr(conv, "limits", []) or [])
            conv_sidx = list(getattr(conv, "semantic_indices", []) or [])
            inv_set = set(getattr(conv, "inverted_indices", ()) or ())
            n_conv = min(len(conv_names), len(conv_limits), len(conv_sidx))

            # 1) 短名 / 去 l_hand_|r_hand_ 前缀匹配（O20/O30 等同序机型）
            by_short: Dict[str, Tuple[float, float, bool]] = {}
            for i in range(n_conv):
                lo, hi = conv_limits[i]
                inv = int(conv_sidx[i]) in inv_set
                full = conv_names[i]
                short = self._strip_hand_joint_prefix(full)
                by_short[full.lower()] = (float(lo), float(hi), bool(inv))
                by_short[short] = (float(lo), float(hi), bool(inv))

            matched = [False] * n_ui
            for i, jn in enumerate(joint_names):
                key = str(jn).strip().lower()
                hit = by_short.get(key) or by_short.get(self._strip_hand_joint_prefix(key))
                if hit is None:
                    continue
                lo, hi, inv = hit
                scales[i] = {"lower": lo, "upper": hi, "inverted": inv}
                matched[i] = True

            # 2) 同长度未命中：按序对齐（identity 机型兜底）
            if n_ui == n_conv:
                for i in range(n_ui):
                    if matched[i]:
                        continue
                    lo, hi = conv_limits[i]
                    scales[i] = {
                        "lower": float(lo),
                        "upper": float(hi),
                        "inverted": int(conv_sidx[i]) in inv_set,
                    }
                    matched[i] = True

            # 3) L25/G20 等：layout→semantic 探测，映射到 Converter 槽位
            if not all(matched):
                try:
                    layout_cfg = HandConfig(layout_hand_type or hand_type)
                    product_cfg = HandConfig(hand_type)
                    sidx_to_conv = {int(sidx): ci for ci, sidx in enumerate(conv_sidx[:n_conv])}
                    base = [128.0] * n_ui
                    for i in range(n_ui):
                        if matched[i]:
                            continue
                        lo_ui = list(base)
                        hi_ui = list(base)
                        lo_ui[i] = 0.0
                        hi_ui[i] = 255.0
                        try:
                            lo_prod = ui_to_product_hardware(
                                lo_ui,
                                product_key=hand_type,
                                layout_key=layout_hand_type or hand_type,
                            )
                            hi_prod = ui_to_product_hardware(
                                hi_ui,
                                product_key=hand_type,
                                layout_key=layout_hand_type or hand_type,
                            )
                        except ValueError:
                            continue
                        if product_cfg.has_from_semantic:
                            sa = product_cfg.to_semantic(lo_prod)
                            sb = product_cfg.to_semantic(hi_prod)
                        else:
                            sa = layout_cfg.to_semantic(lo_ui)
                            sb = layout_cfg.to_semantic(hi_ui)
                        best = max(
                            range(min(len(sa), len(sb))),
                            key=lambda k: abs(float(sb[k]) - float(sa[k])),
                            default=-1,
                        )
                        if best < 0 or abs(float(sb[best]) - float(sa[best])) < 1.0:
                            continue
                        ci = sidx_to_conv.get(int(best))
                        if ci is None:
                            continue
                        lo, hi = conv_limits[ci]
                        scales[i] = {
                            "lower": float(lo),
                            "upper": float(hi),
                            "inverted": int(conv_sidx[ci]) in inv_set,
                        }
                        matched[i] = True
                except (FileNotFoundError, ValueError, OSError, TypeError):
                    pass

        # 未匹配 / 无 Converter：0~π/2 线性近似，保证 rad/deg 始终可选
        for i, s in enumerate(scales):
            if float(s["upper"]) > float(s["lower"]):
                continue
            scales[i] = {"lower": 0.0, "upper": default_hi, "inverted": False}
            approx = True

        return {
            "available": ["hw255", "rad", "deg"],
            "scales": scales,
            "default": "hw255",
            "approx": approx,
            "has_converter": conv is not None,
        }

    def _build_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = self.controller.snapshot()
        snap.update(
            {
                "type": "snapshot",
                "hand_type": self.layout_hand_type,
                "product_hand_type": self.hand_type,
                "hand_side": self.hand_side,
                "joint_names": self.joint_names,
                "poses": self._cached_poses,
                "coupled_joints": couple_meta(self.layout_hand_type, self.joint_names),
                "coupling_hint": coupling_hint(self.layout_hand_type),
                "joint_display": self._cached_joint_display,
            }
        )
        return snap

    async def _broadcast(self) -> None:
        if not self._clients:
            return
        payload = json.dumps(self._build_snapshot())
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _ws(self, websocket: WebSocket) -> None:
        if self.workbench is not None:
            await self._workbench_ws(websocket)
            return
        await websocket.accept()
        self._clients.add(websocket)
        with self._lock:
            self.controller.on_ws_connected()
        await websocket.send_text(json.dumps(self._build_snapshot()))
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "invalid_json"})
                    )
                    continue
                resp = self._handle_client_msg(msg)
                if resp is not None:
                    await websocket.send_text(json.dumps(resp))
                await self._broadcast()
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)
            if not self._clients:
                with self._lock:
                    self.controller.on_ws_disconnected()

    async def _workbench_ws(self, websocket: WebSocket) -> None:
        """Old websocket messages share the same leases, CAS and v2 motion writer."""
        await websocket.accept()
        client_id = uuid.uuid4().hex
        selected = websocket.query_params.get("side", self.hand_side)
        try:
            hand = self.workbench.hand(selected)
        except ValueError:
            await websocket.close(code=1008)
            return
        layout = hand.layout_hand_type
        joint_names = list(HandConfig(layout).joint_names)
        display = self._joint_display_meta(joint_names=joint_names, converter=hand.adapter.converter,
                                           hand_type=hand.adapter.model, layout_hand_type=layout)
        last_snapshot = 0.0

        async def command(message):
            return await self.workbench.request("legacy", client_id=client_id, side=selected,
                                                layout=layout, message=message)

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                except asyncio.TimeoutError:
                    raw = None
                if raw is not None:
                    try:
                        message = json.loads(raw)
                        if not isinstance(message, dict):
                            raise ValueError("invalid_json")
                        response = await command(message)
                        if response is not None:
                            await websocket.send_json(response)
                    except (HTTPException, ValueError, TypeError) as exc:
                        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                        text = detail.get("reason", str(detail)) if isinstance(detail, dict) else str(detail)
                        await websocket.send_json({"type": "error", "message": text})
                if time.monotonic() - last_snapshot >= .1:
                    snapshot = await command({"type": "snapshot"})
                    snapshot.update({"hand_type": layout, "product_hand_type": hand.adapter.model,
                                     "hand_side": selected, "joint_names": list(HandConfig(layout).joint_names),
                                     "coupled_joints": couple_meta(layout, list(HandConfig(layout).joint_names)),
                                     "coupling_hint": coupling_hint(layout)})
                    snapshot["joint_display"] = display
                    await websocket.send_json(snapshot)
                    last_snapshot = time.monotonic()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            try:
                await command({"type": "disconnect"})
            except Exception:
                pass

    def _handle_client_msg(self, msg: dict) -> Optional[dict]:
        mtype = str(msg.get("type") or "")
        if mtype in {"heartbeat", "live_heartbeat"}:
            return None
        with self._lock:
            if mtype == "live_stop":
                self.controller.muted = True
                self.controller._pending_targets = None
                return None
            if mtype == "enter_teach":
                self.controller.enter_teach()
                return {"type": "ok", "action": "enter_teach"}
            if mtype == "exit_teach":
                self.controller.exit_teach()
                return {"type": "ok", "action": "exit_teach"}
            if mtype == "set_joints":
                positions = msg.get("positions") or []
                err = self.controller.set_joints(positions)
                if err:
                    return {"type": "error", "message": err}
                return None
            if mtype == "flush_joints":
                self.controller.flush_joints()
                return None
            if mtype == "save_pose":
                return self._save_pose(msg)
            if mtype == "delete_pose":
                name = str(msg.get("name") or "").strip().lower()
                ok = delete_pose(self.hand_type, name, hand_side=self.hand_side)
                if not ok:
                    return {"type": "error", "message": f"pose_missing:{name}"}
                self._refresh_poses()
                return {"type": "ok", "action": "delete_pose", "name": name}
            if mtype == "play_pose":
                name = str(msg.get("name") or "").strip().lower()
                if not name:
                    return {
                        "type": "error",
                        "message": "姿态名为空：请先点选列表中的姿态，或填写名称",
                    }
                keyframe = str(msg.get("keyframe") or "").strip().lower()
                progress = msg.get("progress", None)
                auto_segment = bool(msg.get("auto_segment"))
                prog_f = None
                if progress is not None and str(progress).strip() != "":
                    try:
                        prog_f = float(progress)
                    except (TypeError, ValueError):
                        return {"type": "error", "message": "progress 无效"}
                self.controller.begin_playback(
                    name,
                    keyframe=keyframe or None,
                    progress=prog_f,
                    auto_segment=auto_segment,
                )
                if self.controller.last_error:
                    return {
                        "type": "error",
                        "message": self.controller.last_error,
                    }
                return {
                    "type": "ok",
                    "action": "play_pose",
                    "name": name,
                    "keyframe": keyframe,
                    "progress": prog_f,
                    "auto_segment": auto_segment,
                }
            if mtype == "set_progress":
                err = self.controller.set_playback_progress(msg.get("progress", 0.0))
                if err:
                    return {"type": "error", "message": err}
                return {
                    "type": "ok",
                    "action": "set_progress",
                    "progress": self.controller.motion_progress,
                }
            if mtype == "scrub_pose":
                name = str(msg.get("name") or "").strip().lower()
                keyframe = str(msg.get("keyframe") or "").strip().lower()
                try:
                    prog = float(msg.get("progress", 0.0))
                except (TypeError, ValueError):
                    return {"type": "error", "message": "progress 无效"}
                err = self.controller.scrub_to(name, keyframe=keyframe, progress=prog)
                if err:
                    return {"type": "error", "message": err}
                return {
                    "type": "ok",
                    "action": "scrub_pose",
                    "name": name,
                    "keyframe": keyframe,
                    "progress": self.controller.motion_progress,
                }
            if mtype == "stop_playback":
                resume = msg.get("resume_teach", None)
                self.controller.stop_playback(
                    resume_teach=None if resume is None else bool(resume)
                )
                return {"type": "ok", "action": "stop_playback"}
            if mtype == "delete_keyframe":
                name = str(msg.get("name") or "").strip().lower()
                keyframe = str(msg.get("keyframe") or "").strip().lower()
                err = delete_pose_keyframe(
                    self.hand_type, name, keyframe, hand_side=self.hand_side
                )
                if err:
                    return {"type": "error", "message": err}
                self._refresh_poses()
                return {
                    "type": "ok",
                    "action": "delete_keyframe",
                    "name": name,
                    "keyframe": keyframe,
                }
            if mtype == "move_keyframe":
                name = str(msg.get("name") or "").strip().lower()
                keyframe = str(msg.get("keyframe") or "").strip().lower()
                try:
                    delta = int(msg.get("delta", 0))
                except (TypeError, ValueError):
                    return {"type": "error", "message": "delta 无效"}
                err = move_pose_keyframe(
                    self.hand_type,
                    name,
                    keyframe,
                    delta=delta,
                    hand_side=self.hand_side,
                )
                if err:
                    return {"type": "error", "message": err}
                self._refresh_poses()
                return {
                    "type": "ok",
                    "action": "move_keyframe",
                    "name": name,
                    "keyframe": keyframe,
                    "delta": delta,
                }
            if mtype == "rename_keyframe":
                name = str(msg.get("name") or "").strip().lower()
                keyframe = str(msg.get("keyframe") or "").strip().lower()
                new_keyframe = str(msg.get("new_keyframe") or "").strip().lower()
                err = rename_pose_keyframe(
                    self.hand_type,
                    name,
                    keyframe,
                    new_keyframe,
                    hand_side=self.hand_side,
                )
                if err:
                    return {"type": "error", "message": err}
                self._refresh_poses()
                return {
                    "type": "ok",
                    "action": "rename_keyframe",
                    "name": name,
                    "keyframe": keyframe,
                    "new_keyframe": new_keyframe,
                }
            if mtype == "reinit":
                self.controller.reinit()
                return {"type": "ok", "action": "reinit"}
            if mtype == "load_pose":
                return self._load_pose_for_edit(msg)
            if mtype == "sync_feedback":
                err = self.controller.sync_from_feedback()
                if err:
                    return {"type": "error", "message": err}
                return {"type": "ok", "action": "sync_feedback"}
            if mtype == "clear_edit":
                self.controller.clear_editing_pose()
                return {"type": "ok", "action": "clear_edit"}
            if mtype == "set_prefs":
                if "return_to_teach_after_playback" in msg:
                    self.controller.return_to_teach_after_playback = bool(
                        msg.get("return_to_teach_after_playback")
                    )
                return {
                    "type": "ok",
                    "action": "set_prefs",
                    "return_to_teach_after_playback": (
                        self.controller.return_to_teach_after_playback
                    ),
                }
            if mtype == "list_poses":
                self._refresh_poses()
                return {
                    "type": "poses",
                    "poses": self._cached_poses,
                }
        return {"type": "error", "message": f"unknown_type:{mtype}"}

    def _load_pose_for_edit(self, msg: dict) -> dict:
        name = str(msg.get("name") or "").strip().lower()
        keyframe = str(msg.get("keyframe") or "").strip().lower()
        entry = load_pose(self.hand_type, name, hand_side=self.hand_side)
        if entry is None:
            return {"type": "error", "message": f"pose_missing:{name}"}
        positions = list(entry.positions)
        duration = float(entry.duration)
        if keyframe:
            found = None
            for kf in entry.effective_keyframes():
                if kf.name == keyframe:
                    found = kf
                    break
            if found is None:
                names = [kf.name for kf in entry.effective_keyframes()]
                return {
                    "type": "error",
                    "message": f"keyframe_missing:{keyframe} available:{names}",
                }
            positions = list(found.positions)
            duration = float(found.duration)
        # 构造临时 entry 视图供 hardware 换算
        from dataclasses import replace

        view = replace(entry, positions=positions, duration=duration)
        product_hw = pose_entry_to_hardware(
            view, self.hand_type, radian_converter=self._radian_converter
        )
        if product_hw is None:
            return {
                "type": "error",
                "message": f"pose_unit_rad_requires_converter:{name}",
            }
        try:
            ui = product_hardware_to_ui(
                product_hw,
                product_key=self.hand_type,
                layout_key=self.layout_hand_type,
            )
        except ValueError as exc:
            return {"type": "error", "message": str(exc)}
        err = self.controller.load_pose_for_edit(name, ui, enter_teach=True, keyframe=keyframe)
        if err:
            return {"type": "error", "message": err}
        return {
            "type": "ok",
            "action": "load_pose",
            "name": name,
            "keyframe": keyframe,
            "duration": duration,
            "description": entry.description,
            "keyframes": [
                {"name": kf.name, "duration": float(kf.duration)} for kf in entry.keyframes
            ],
        }

    def _save_pose(self, msg: dict) -> dict:
        name = str(msg.get("name") or "").strip().lower()
        if not name and self.controller.editing_pose_name:
            name = self.controller.editing_pose_name
        keyframe = str(msg.get("keyframe") or "").strip().lower()
        # 二次编辑：帧名空时优先覆盖正在编辑的关键帧，避免误写成「建议的下一帧名」或整姿态被拒
        if not keyframe and self.controller.editing_keyframe:
            keyframe = self.controller.editing_keyframe
        if not keyframe and name:
            existing = load_pose(self.hand_type, name, hand_side=self.hand_side)
            if existing is not None and existing.keyframes:
                if len(existing.keyframes) == 1:
                    keyframe = existing.keyframes[0].name
                else:
                    names = [kf.name for kf in existing.keyframes]
                    return {
                        "type": "error",
                        "message": (
                            f"姿态 '{name}' 有 {len(names)} 个关键帧 {names}；"
                            "请填写要覆盖的帧名，或从姿态库点选该帧再保存"
                        ),
                    }
        duration = float(msg.get("duration") or 0.6)
        description = str(msg.get("description") or "")
        # 示教中默认保存滑条目标（避免未到位反馈把姿态冲成 init）
        in_teach = self.controller.state == PanelState.TEACH_EDIT
        prefer_targets = bool(
            msg.get(
                "use_targets",
                in_teach or bool(self.controller.editing_pose_name),
            )
        )
        if prefer_targets:
            ui_positions = list(self.controller.targets)
        else:
            ui_positions = self.controller.capture_positions()
            if ui_positions is None:
                ui_positions = list(self.controller.targets)
        if not ui_positions:
            return {
                "type": "error",
                "message": "无可用关节值（等待反馈或拖动滑条）",
            }
        try:
            product_hw = ui_to_product_hardware(
                ui_positions,
                product_key=self.hand_type,
                layout_key=self.layout_hand_type,
            )
        except ValueError as exc:
            return {"type": "error", "message": str(exc)}
        space = str(msg.get("space") or DEFAULT_SAVE_SPACE).strip().lower()
        prepared = prepare_pose_for_save(
            self.hand_type,
            product_hw,
            space=space,
            radian_converter=self._radian_converter,
        )
        if isinstance(prepared, str):
            return {"type": "error", "message": prepared}
        positions, space, unit = prepared
        err = commit_saved_pose(
            self.hand_type,
            name,
            positions,
            duration=duration,
            description=description,
            source="teach_panel",
            space=space,
            unit=unit,
            hand_side=self.hand_side,
            keyframe=keyframe,
        )
        if err:
            return {"type": "error", "message": err}
        self.controller.editing_pose_name = name
        self.controller.editing_keyframe = keyframe
        self.controller.mark_targets_clean()
        self._refresh_poses()
        return {
            "type": "ok",
            "action": "save_pose",
            "name": name,
            "keyframe": keyframe or "",
            "space": space,
            "unit": unit,
            "advance_keyframe": bool(msg.get("advance_keyframe")),
        }


def create_workbench_app(workbench):
    """Construct the standalone application without importing ROS or LeRobot."""
    side = workbench.default_side
    hand = workbench.hands[side]
    layout = HandConfig(hand.layout_hand_type)
    panel = TeachPanelApp(
        PanelController(layout.num_joints), hand_type=hand.adapter.model, hand_side=side,
        layout_hand_type=hand.layout_hand_type,
        joint_names=list(layout.joint_names), radian_converter=hand.adapter.converter,
        workbench=workbench,
    )
    return panel.app
