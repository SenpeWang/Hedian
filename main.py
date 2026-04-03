#!/usr/bin/env python3
"""
核电站监护制合规检测系统 — 主流水线
多路流式同步: Voice ‖ MOT → Qwen 报告
"""
import os, sys, json, time, threading, queue, subprocess, re
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ── 路径 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HEDIAN   = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_ROOT = os.path.abspath(os.path.join(HEDIAN, "..", "Hedian_data"))
VIDEO_PATH   = os.path.join(DATA_ROOT, "data", "camFRONT.mpg")
RESULT_ROOT  = os.path.join(DATA_ROOT, "Result")

from MOT import Settings, WORKSTATIONS, DIST_CLOSE, DIST_NEAR, FORWARD_SECONDS
from MOT.detector.detector import ObjectDetector
from MOT.tracker.tracker import DynamicTracker, TrackState
from MOT.utils.visualizer import Visualizer
from VOICE.voice import (
    main as voice_main, apply_corrections, classify_intent,
    build_voice_events, build_key_moments, DEVICE_PATTERN,
)
from Qwen.QwenEvaluate import generate_report
from Qwen.evaluate import evaluate_mini_report, evaluate_supervision_report

# ══════════════════════════════════════════════════════════
#  共享队列 / 事件
# ══════════════════════════════════════════════════════════
frame_queue              = queue.Queue(maxsize=8)
event_queue              = queue.Queue()
pending_voice_queue      = queue.Queue()
supervision_signal_queue = queue.Queue()

experiment_end_event = threading.Event()
mot_done             = threading.Event()
report_generated     = threading.Event()

voice_events_all   = []            # Voice → Qwen
mini_reports_shared = []           # on_seg → Qwen
supervision_reports_shared = []    # 监护制度独立流程评价
_data_lock = threading.Lock()

pipeline_state = "idle"            # idle / running / done / error
_result_dir = ""
_socketio_ref = None               # WebSocket 引用（由 vis.py 注入）

def set_socketio(sio):
    global _socketio_ref
    _socketio_ref = sio

# ── 全局帧索引（Voice 先行后构建） ──
_voice_by_frame = {}               # {frame_id: [event, ...]}  Voice events indexed by frame
_voice_max_frame = 0               # max frame_id in voice events
_global_fps = 30.0                 # 视频 FPS

settings = Settings()


# ══════════════════════════════════════════════════════════
#  辅助
# ══════════════════════════════════════════════════════════
def _ts_str(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _push_event(etype, data):
    ev = {"type": etype, **data}
    event_queue.put(ev)
    # 广播给所有 SSE 客户端
    try:
        from vis import _sse_clients, _sse_clients_lock
        with _sse_clients_lock:
            for client_q in _sse_clients:
                try:
                    client_q.put_nowait(ev)
                except:
                    pass
    except ImportError:
        pass


def _push_status(text):
    _push_event("status", {"text": text})


def _push_progress(label, pct, detail=""):
    """推送进度条事件: label=阶段名, pct=0~100, detail=附加信息"""
    _push_event("progress", {"label": label, "pct": min(100, max(0, int(pct))), "detail": detail})



# ══════════════════════════════════════════════════════════
#  Voice 线程
# ══════════════════════════════════════════════════════════
_flow_state = {"cmd": None, "confirms": [], "last_ts": 0}

# ── 监护请求状态（语音侧记录） ──
_sup_request_state = {"active": False, "request_ev": None, "confirm_ev": None}


def _make_mini_report(cmd_ev, confirms):
    """三段式沟通评价 — 调用 evaluate.py 评分 + 副作用"""
    mr = evaluate_mini_report(cmd_ev, confirms)
    with _data_lock:
        mini_reports_shared.append(mr)
    return mr


def _make_supervision_report(request_ev, confirm_ev=None,
                             hand_raise_info=None, bound_info=None,
                             sup_status_list=None, operator=None, fps=30.0):
    """监护制度评价 — 调用 evaluate.py 评分 + 副作用"""
    with _data_lock:
        report_id = len(supervision_reports_shared) + 1
    sr = evaluate_supervision_report(
        request_ev, confirm_ev, hand_raise_info, bound_info,
        sup_status_list, operator, report_id, fps)
    with _data_lock:
        supervision_reports_shared.append(sr)
    _push_event("supervision_report", sr)
    return sr


def on_seg(words):
    """Voice 每个窗口的回调：分句 → 构建 voice_events_all + mini_reports（帧释放由 _flush_voice_by_frame 负责）"""
    global _flow_state
    if not words:
        return
    # 过滤 None
    words = [w for w in words if w is not None and w.get("word")]
    if not words:
        return
    # 按 >1.5s 停顿分句
    sentences, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > 1.5:
            sentences.append(cur); cur = [w]
        else:
            cur.append(w)
    if cur:
        sentences.append(cur)

    for sent_words in sentences:
        text = "".join(w["word"] for w in sent_words)
        text = apply_corrections(text)
        if not text.strip():
            continue
        ts = round(sent_words[0]["start"], 2)
        # 取上一 intent
        prev_intent = ""
        with _data_lock:
            if voice_events_all:
                prev_intent = voice_events_all[-1].get("intent", "")
        intent = classify_intent(text, prev_intent)
        ev = {"time_sec": ts, "text": text, "intent": intent}

        with _data_lock:
            voice_events_all.append(ev)

        # ── 三段式状态机（构建 mini_reports） ──
        if intent == "操作指令":
            if _flow_state["cmd"] and _flow_state["confirms"]:
                _make_mini_report(_flow_state["cmd"], _flow_state["confirms"])
            _flow_state = {"cmd": ev, "confirms": [], "last_ts": ts}

        elif intent == "监护请求":
            # 监护请求作为独立流程起点
            _sup_request_state["active"] = True
            _sup_request_state["request_ev"] = ev
            _sup_request_state["confirm_ev"] = None
            _push_event("voice", {"time_sec": ts, "text": text, "intent": intent})

        elif intent == "监护确认":
            # 监护确认关联到监护请求
            if _sup_request_state["active"]:
                _sup_request_state["confirm_ev"] = ev
            if _flow_state["cmd"]:
                _flow_state["confirms"].append(ev)
                _flow_state["last_ts"] = ts

        elif intent == "确认":
            if _flow_state["cmd"]:
                _flow_state["confirms"].append(ev)
                _flow_state["last_ts"] = ts

        elif intent == "操作结束":
            if _flow_state["cmd"]:
                _make_mini_report(_flow_state["cmd"], _flow_state["confirms"])
                _flow_state = {"cmd": None, "confirms": [], "last_ts": 0}

        elif intent == "实验结束":
            if _flow_state["cmd"]:
                _make_mini_report(_flow_state["cmd"], _flow_state["confirms"])
                _flow_state = {"cmd": None, "confirms": [], "last_ts": 0}
            experiment_end_event.set()


def run_voice(result_dir):
    """Voice 线程入口 — 每次从视频重新处理，不复用缓存"""
    _push_status("🎤 语音模块启动")
    _push_progress("语音处理", 0, "初始化")

    try:
        # 提取音频（每次都重新提取）
        data_dir = os.path.join(DATA_ROOT, "data")
        raw_wav = os.path.join(data_dir, "_tmp_denoised.wav")
        _push_progress("语音处理", 5, "ffmpeg 提取音频")
        subprocess.run([
            "ffmpeg", "-y", "-i", VIDEO_PATH,
            "-ar", "16000", "-ac", "1", "-vn", raw_wav
        ], capture_output=True)
        _push_progress("语音处理", 10, "音频提取完成")

        # 设置 voice 模块路径
        import VOICE.voice as vm
        vm.RAW_AUDIO = raw_wav
        vm.OUTPUT_DIR = result_dir

        _push_progress("语音处理", 12, "加载 Whisper large-v3 模型")
        _push_status("🎤 加载 Whisper large-v3 模型…")

        def _progress(done, total):
            pct = int(done / total * 100) if total else 0
            mapped = 12 + int(pct * 0.78)
            _push_progress("语音处理", mapped, f"转录中 {pct}%")
            if pct % 10 == 0:
                _push_status(f"🎤 语音转录 {pct}%")

        vm.main(progress_cb=_progress, segment_cb=on_seg)
        _push_progress("语音处理", 95, "保存 JSON 文件")
        _push_status("🎤 语音转录完成，保存 JSON")
        _push_progress("语音处理", 100, "完成")
        _push_status("✅ 语音转录完成")
    except Exception as e:
        _push_status(f"❌ Voice 错误: {e}")
        _push_progress("语音处理", -1, f"错误: {e}")
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════
#  MOT 线程
# ══════════════════════════════════════════════════════════
def _flush_voice_by_frame(fc):
    """按帧号释放语音事件（帧对齐）—— 只释放 frame_id <= fc 的事件"""
    global _voice_by_frame
    released = []
    for fid in sorted(k for k in _voice_by_frame if k <= fc):
        for ev in _voice_by_frame[fid]:
            etype = ev.get("type", "voice")  # 不修改原对象
            data = {k: v for k, v in ev.items() if k != "type"}
            _push_event(etype, data)
            released.append(ev)
        del _voice_by_frame[fid]
    return released


def run_mot(result_dir):
    """MOT 线程入口"""
    global pipeline_state
    _push_status("🎯 MOT 启动")
    _push_progress("目标跟踪", 0, "加载 YOLO 检测模型")

    det = ObjectDetector(
        model_path=settings.detection.model_path,
        pose_model_path=settings.detection.pose_model_path,
        conf_threshold=settings.detection.conf_threshold,
        pose_confidence=settings.detection.pose_confidence,
        nms_threshold=settings.detection.nms_threshold,
        img_size=settings.detection.img_size,
    )
    _push_progress("目标跟踪", 5, "加载 YOLOPose 模型")
    trk = DynamicTracker()
    vis = Visualizer()

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracking_events = []
    kf_dir = os.path.join(result_dir, "key_frames")
    os.makedirs(kf_dir, exist_ok=True)

    fc = 0
    target_interval = 1.0 / fps
    start_wall = time.time()
    role_event_sent = False

    # ── 监护状态机 ──
    # States: idle / requesting / bound
    sup_state = "idle"
    sup_target_role = None
    sup_close_start_ts = -1.0
    sup_far_start_ts = -1.0
    BIND_HOLD_SEC = 3.0
    UNBIND_HOLD_SEC = 20.0

    # ── 当前监护流程信息收集 ──
    _cur_hand_raise_info = None   # 举手事件信息 dict
    _cur_bound_info = None        # 监护绑定事件信息 dict
    _cur_sup_status_list = []     # 监护距离状态列表

    # ── 举手检测确认 ──
    VOTE_WINDOW = 2          # 连续2次检测到即确认
    VOTE_THRESHOLD = 2       # 2次就够
    COOLDOWN_FRAMES = 300    # 举手后冷却10秒不重复检测
    hand_raise_buffer = {}   # {role: deque([True/False, ...])}
    last_hand_raise_frame = {}  # {role: frame_count}

    # 操作帧（从 key_moments.json 文件轮询）
    ops_f = set()
    ope_f = set()
    km_mtime = 0

    _push_event("video_start", {})

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts = fc / fps
            fc += 1

            # ── 检测 + 跟踪 ──
            detections = det.detect(frame)
            tracks = trk.track(frame, detections)

            # ── 角色分配（一次） ──
            if not trk.roles_assigned and trk.initialized and len(tracks) >= 2:
                trk._assign_roles(tracks)

            if trk.roles_assigned and not role_event_sent:
                role_event_sent = True
                ev = {"time_sec": round(ts, 2), "event": "ROLE_ASSIGNED",
                      "details": dict(trk.role_map), "frame_id": fc}
                tracking_events.append(ev)
                _push_event("tracking", ev)
                cv2.imwrite(os.path.join(kf_dir, f"role_assigned_{ts:.1f}s.jpg"), frame)

            # ── Visualize + push frame ──
            vis_frame = vis.draw_tracks(frame, tracks)
            _, jpeg = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            try:
                frame_queue.put_nowait(jpeg.tobytes())
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(jpeg.tobytes())

            # ── 同步释放语音事件 ──
            _flush_voice_by_frame(fc)

            # ═══════════════════════════════════════
            #  监护状态机（YOLOPose 直接触发）
            # ═══════════════════════════════════════

            # (1) idle 状态：每 10 帧用 YOLOPose 检测举手（多帧投票机制）
            if sup_state == "idle" and fc % 10 == 0 and trk.roles_assigned:
                poses = det.detect_pose(frame)
                for rname in ("ROAD1", "ROAD2"):
                    # 冷却期检查
                    if fc - last_hand_raise_frame.get(rname, -9999) < COOLDOWN_FRAMES:
                        continue
                    rt = trk.get_track_by_role(rname)
                    if rt is None:
                        continue
                    rc = rt.get_center()
                    best_pose, best_d = None, float("inf")
                    for p in poses:
                        d = np.linalg.norm(rc - p["center"])
                        if d < best_d:
                            best_d, best_pose = d, p

                    raised = (best_pose is not None and det.check_hand_raised(best_pose["keypoints"]))

                    # 投票缓冲
                    if rname not in hand_raise_buffer:
                        hand_raise_buffer[rname] = deque(maxlen=VOTE_WINDOW)
                    hand_raise_buffer[rname].append(raised)

                    # 只有投票窗口满且超过阈值才触发
                    buf = hand_raise_buffer[rname]
                    if len(buf) == VOTE_WINDOW and sum(buf) >= VOTE_THRESHOLD:
                        last_hand_raise_frame[rname] = fc
                        hand_raise_buffer[rname].clear()

                        kf_name = f"hand_raise_{rname}_{ts:.1f}s.jpg"
                        kf_path = os.path.join(kf_dir, kf_name)
                        cv2.imwrite(kf_path, frame)

                        hev = {"time_sec": round(ts, 2), "event": "HAND_RAISE_SUPERVISION",
                               "operator": rname, "frame_id": fc, "raised": True,
                               "key_frame": kf_name}
                        tracking_events.append(hev)
                        _push_event("tracking", hev)

                        # 记录举手信息到当前监护流程
                        _cur_hand_raise_info = hev
                        _cur_bound_info = None
                        _cur_sup_status_list = []

                        sup_state = "requesting"
                        sup_target_role = rname
                        sup_close_start_ts = -1.0
                        sup_far_start_ts = -1.0
                        _push_status(f"✋ {rname} 举手 — 请求监护")
                        print(f"[监护] ✅ YOLOPose 检测到 {rname} 举手 @{ts:.1f}s (投票确认)")
                        break

            # (2) requesting / bound 状态下：持续检测 LEADER 与 target 距离
            if sup_state in ("requesting", "bound") and sup_target_role and fc % 30 == 0:
                leader = trk.get_track_by_role("LEADER")
                target = trk.get_track_by_role(sup_target_role)
                if leader and target:
                    d = float(np.linalg.norm(leader.get_center() - target.get_center()))

                    if d <= DIST_CLOSE:
                        status_label = "监护中"
                        if sup_close_start_ts < 0:
                            sup_close_start_ts = ts
                        sup_far_start_ts = -1.0
                    elif d <= DIST_NEAR:
                        status_label = "接近中"
                        sup_close_start_ts = -1.0
                        sup_far_start_ts = -1.0
                    else:
                        status_label = "未监护"
                        sup_close_start_ts = -1.0
                        if sup_far_start_ts < 0:
                            sup_far_start_ts = ts

                    sev = {"time_sec": round(ts, 2), "event": "SUPERVISOR_STATUS",
                           "operator": sup_target_role, "distance_px": int(d),
                           "status": status_label, "frame_id": fc}
                    tracking_events.append(sev)
                    _push_event("tracking", sev)
                    _cur_sup_status_list.append(sev)

                    # requesting → bound：≤280px 持续 3 秒
                    if sup_state == "requesting" and sup_close_start_ts >= 0:
                        if ts - sup_close_start_ts >= BIND_HOLD_SEC:
                            sup_state = "bound"
                            bev = {"time_sec": round(ts, 2), "event": "SUPERVISION_BOUND",
                                   "operator": sup_target_role, "frame_id": fc}
                            tracking_events.append(bev)
                            _push_event("tracking", bev)
                            _cur_bound_info = bev
                            _push_status(f"🔗 {sup_target_role} 监护绑定")

                    # bound → idle：>560px 持续 20 秒
                    if sup_state == "bound" and sup_far_start_ts >= 0:
                        if ts - sup_far_start_ts >= UNBIND_HOLD_SEC:
                            uev = {"time_sec": round(ts, 2), "event": "SUPERVISION_END",
                                   "reason": "距离超限持续20s", "frame_id": fc}
                            tracking_events.append(uev)
                            _push_event("tracking", uev)

                            # ── 生成监护制度评价报告 ──
                            if _sup_request_state.get("active") and _sup_request_state.get("request_ev"):
                                _make_supervision_report(
                                    _sup_request_state["request_ev"],
                                    confirm_ev=_sup_request_state.get("confirm_ev"),
                                    hand_raise_info=_cur_hand_raise_info,
                                    bound_info=_cur_bound_info,
                                    sup_status_list=_cur_sup_status_list,
                                    operator=sup_target_role, fps=fps)
                                _sup_request_state["active"] = False

                            sup_state = "idle"
                            sup_target_role = None
                            sup_close_start_ts = -1.0
                            sup_far_start_ts = -1.0
                            _cur_hand_raise_info = None
                            _cur_bound_info = None
                            _cur_sup_status_list = []
                            _push_status("⚠️ 监护解绑")

            # ═══════════════════════════════════════
            #  操作帧辅助（key_moments.json 轮询）
            # ═══════════════════════════════════════
            if fc % 30 == 0:
                km_path = os.path.join(result_dir, "key_moments.json")
                try:
                    if os.path.exists(km_path):
                        mt = os.path.getmtime(km_path)
                        if mt > km_mtime:
                            km_mtime = mt
                            with open(km_path, encoding="utf-8") as f:
                                km = json.load(f)
                            for oc in km.get("operation_commands", []):
                                of = int(oc["timestamp_sec"] * fps)
                                if of not in ops_f:
                                    ops_f.add(of)
                            for oe in km.get("operation_ends", []):
                                ef = int(oe["timestamp_sec"] * fps)
                                if ef not in ope_f:
                                    ope_f.add(ef)
                except Exception:
                    pass

            # ── 操作节点标注 ──
            if fc in ops_f:
                oev = {"time_sec": round(ts, 2), "event": "OPERATION_START"}
                tracking_events.append(oev)
                _push_event("tracking", oev)
                cv2.imwrite(os.path.join(kf_dir, f"op_start_{ts:.1f}s.jpg"), frame)

            if fc in ope_f:
                sv_result = "数据不足"
                recent_sv = [e for e in tracking_events
                             if e.get("event") == "SUPERVISOR_STATUS" and e["time_sec"] > ts - 30]
                if recent_sv:
                    close_count = sum(1 for e in recent_sv if e.get("status") == "监护中")
                    sv_result = "全程在位" if close_count / len(recent_sv) > 0.7 else "部分在位"
                oev = {"time_sec": round(ts, 2), "event": "OPERATION_END",
                       "supervisor_result": sv_result}
                tracking_events.append(oev)
                _push_event("tracking", oev)
                cv2.imwrite(os.path.join(kf_dir, f"op_end_{ts:.1f}s.jpg"), frame)

            # ── 帧率控制 ──
            expected = fc * target_interval
            elapsed = time.time() - start_wall
            if elapsed < expected:
                time.sleep(expected - elapsed)

            # ── 状态更新 ──
            if fc % 90 == 0:
                pct = fc * 100 // total_frames if total_frames else 0
                sup_info = f" | 监护:{sup_state}" if sup_state != "idle" else ""
                _push_progress("目标跟踪", 8 + int(pct * 0.87), f"{_ts_str(ts)} {pct}%{sup_info}")
            if fc % 300 == 0:
                pct = fc * 100 // total_frames if total_frames else 0
                sup_info = f" | 监护:{sup_state}" if sup_state != "idle" else ""
                _push_status(f"🎯 {fc}/{total_frames}帧 {_ts_str(ts)} {pct}%{sup_info}")

    finally:
        cap.release()
        # 释放全部剩余语音事件
        _flush_voice_by_frame(fc + 999999)

        # ── 处理残留的监护流程（视频结束但监护未解绑） ──
        if _sup_request_state.get("active") and _sup_request_state.get("request_ev"):
            _make_supervision_report(
                _sup_request_state["request_ev"],
                confirm_ev=_sup_request_state.get("confirm_ev"),
                hand_raise_info=_cur_hand_raise_info,
                bound_info=_cur_bound_info,
                sup_status_list=_cur_sup_status_list,
                operator=sup_target_role, fps=fps)
            _sup_request_state["active"] = False

        # 写 tracking_events.json
        te_path = os.path.join(result_dir, "tracking_events.json")
        with open(te_path, "w", encoding="utf-8") as f:
            json.dump(tracking_events, f, ensure_ascii=False, indent=2)

        # 写 supervision_reports.json
        with _data_lock:
            sr_list = list(supervision_reports_shared)
        sr_path = os.path.join(result_dir, "supervision_reports.json")
        with open(sr_path, "w", encoding="utf-8") as f:
            json.dump(sr_list, f, ensure_ascii=False, indent=2)

        frame_queue.put(None)
        mot_done.set()
        _push_status("✅ MOT 处理完成")


# ══════════════════════════════════════════════════════════
#  大模型报告
# ══════════════════════════════════════════════════════════
def _gen_report(result_dir):
    """生成 Qwen 合规评估报告 — 从 JSON 文件读取数据"""
    global pipeline_state
    if report_generated.is_set():
        return
    report_generated.set()

    _push_event("report_progress", {"text": "正在聚合事件数据…"})

    # ── 优先从文件读取（Voice / MOT 已各自写入） ──
    ve_path = os.path.join(result_dir, "voice_events.json")
    te_path = os.path.join(result_dir, "tracking_events.json")
    km_path = os.path.join(result_dir, "key_moments.json")

    # voice_events
    if os.path.exists(ve_path):
        with open(ve_path, encoding="utf-8") as f:
            ve = json.load(f)
    else:
        with _data_lock:
            ve = list(voice_events_all)
        with open(ve_path, "w", encoding="utf-8") as f:
            json.dump(ve, f, ensure_ascii=False, indent=2)

    # tracking_events
    if os.path.exists(te_path):
        with open(te_path, encoding="utf-8") as f:
            te = json.load(f)
    else:
        te = []

    # mini_reports（内存中积累的三段式评价）
    with _data_lock:
        mr = list(mini_reports_shared)
    mr_path = os.path.join(result_dir, "mini_reports.json")
    with open(mr_path, "w", encoding="utf-8") as f:
        json.dump(mr, f, ensure_ascii=False, indent=2)

    kf_dir = os.path.join(result_dir, "key_frames")
    kf_files = sorted(os.listdir(kf_dir)) if os.path.isdir(kf_dir) else []

    _push_event("report_progress", {"text": f"数据聚合完成: voice={len(ve)}, tracking={len(te)}, mini={len(mr)}"})

    def _progress(text):
        _push_event("report_progress", {"text": text})

    report, total, grade = generate_report(ve, te, mr, kf_files, progress_cb=_progress)

    _push_event("report", {"text": report})

    # 保存报告
    with open(os.path.join(result_dir, "compliance_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # 运行摘要
    summary = {
        "run_id": os.path.basename(result_dir),
        "video": VIDEO_PATH,
        "voice_events": len(ve), "tracking_events": len(te), "mini_reports": len(mr),
        "supervision_reports": len(supervision_reports_shared),
        "key_frames": kf_files,
        "total_score": total, "grade": grade,
    }
    with open(os.path.join(result_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _push_status(f"✅ 完成 — {result_dir}")
    _push_event("done", {"result_dir": result_dir})
    # 广播 sentinel 给所有 SSE 客户端
    try:
        from vis import _sse_clients, _sse_clients_lock
        with _sse_clients_lock:
            for client_q in _sse_clients:
                try:
                    client_q.put_nowait(None)
                except:
                    pass
    except ImportError:
        pass
    event_queue.put(None)


# ══════════════════════════════════════════════════════════
#  Pipeline 入口
# ══════════════════════════════════════════════════════════
def run_pipeline():
    """
    流水线入口（Voice 先行 + MOT 帧同步）

    架构:
      1. Voice 同步处理 → voice_events.json / key_moments.json
      2. 构建全局帧索引（voice event → frame_id）
      3. MOT 逐帧处理，每帧释放对应 frame_id 的语音事件（帧对齐）
      4. 所有事件通过 SSE 推送前端
      5. 全部完成后 Qwen 生成报告
    """
    global pipeline_state, _result_dir, _flow_state
    global _voice_by_frame, _voice_max_frame, _global_fps
    pipeline_state = "running"
    _flow_state = {"cmd": None, "confirms": [], "last_ts": 0}

    # 清空队列/事件
    for q in (frame_queue, event_queue, pending_voice_queue, supervision_signal_queue):
        while not q.empty():
            try: q.get_nowait()
            except queue.Empty: break
    experiment_end_event.clear()
    mot_done.clear()
    report_generated.clear()
    voice_events_all.clear()
    mini_reports_shared.clear()
    _voice_by_frame.clear()
    _voice_max_frame = 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULT_ROOT, run_id)
    os.makedirs(result_dir, exist_ok=True)
    _result_dir = result_dir

    _push_status("🚀 流水线启动")

    # ═══════════════════════════════════════
    #  阶段1: 获取视频 FPS
    # ═══════════════════════════════════════
    _push_progress("全局", 0, "读取视频信息")
    cap = cv2.VideoCapture(VIDEO_PATH)
    _global_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    _push_status(f"📹 视频 FPS: {_global_fps:.1f}")

    # ═══════════════════════════════════════
    #  阶段2: Voice 同步处理（先行）
    # ═══════════════════════════════════════
    _push_status("🎤 语音模块启动（同步先行）")
    _push_progress("语音处理", 0, "初始化")
    run_voice(result_dir)

    # ═══════════════════════════════════════
    #  阶段3: 构建全局帧索引
    # ═══════════════════════════════════════
    _push_status("📋 构建全局帧索引")
    _push_progress("全局", 5, "构建帧索引")

    # 加载 voice_events
    ve_path = os.path.join(result_dir, "voice_events.json")
    if os.path.exists(ve_path):
        with open(ve_path, encoding="utf-8") as f:
            ve_list = json.load(f)
    else:
        with _data_lock:
            ve_list = list(voice_events_all)

    # 为每个 voice event 计算 frame_id 并建帧索引
    for ev in ve_list:
        ts = ev.get("time_sec", 0)
        fid = int(ts * _global_fps)
        ev["type"] = "voice"   # SSE 事件类型标记
        _voice_by_frame.setdefault(fid, []).append(ev)
        if fid > _voice_max_frame:
            _voice_max_frame = fid

    # 加载 key_moments → 操作信号也加入帧索引
    km_path = os.path.join(result_dir, "key_moments.json")
    if os.path.exists(km_path):
        with open(km_path, encoding="utf-8") as f:
            km = json.load(f)
        for sr in km.get("supervision_requests", []):
            fid = int(sr["timestamp_sec"] * _global_fps)
            _voice_by_frame.setdefault(fid, []).append({
                "type": "voice", "time_sec": sr["timestamp_sec"],
                "text": sr["text"], "intent": "监护请求"
            })

    # mini_reports 也加入帧索引
    with _data_lock:
        mr_list = list(mini_reports_shared)
    for mr in mr_list:
        ts = mr.get("time_sec", 0)
        fid = int(ts * _global_fps)
        _voice_by_frame.setdefault(fid, []).append({"type": "mini_report", **mr})

    total_indexed = len(ve_list) + len(mr_list)
    _push_status(f"📋 帧索引: {total_indexed} 事件 → {len(_voice_by_frame)} 帧")
    _push_progress("全局", 8, f"{total_indexed} 事件已索引")

    # ═══════════════════════════════════════
    #  阶段4: MOT 帧对齐处理
    # ═══════════════════════════════════════
    _push_status("🎯 MOT 帧对齐处理启动")
    # Voice 先行可能已设置 experiment_end_event（检测到"实验结束"），清除它
    experiment_end_event.clear()
    mt = threading.Thread(target=run_mot, args=(result_dir,), daemon=True)
    mt.start()

    # 等待 MOT 完成（只等 mot_done，不等 experiment_end_event）
    mot_done.wait()

    # ═══════════════════════════════════════
    #  阶段5: Qwen 报告
    # ═══════════════════════════════════════
    try:
        _gen_report(result_dir)
    except Exception as e:
        _push_status(f"❌ 报告生成失败: {e}")
        import traceback; traceback.print_exc()
        _push_event("done", {"result_dir": result_dir, "error": str(e)})
        # 广播 sentinel
        try:
            from vis import _sse_clients, _sse_clients_lock
            with _sse_clients_lock:
                for client_q in _sse_clients:
                    try:
                        client_q.put_nowait(None)
                    except:
                        pass
        except ImportError:
            pass
        event_queue.put(None)

    pipeline_state = "done"


# ══════════════════════════════════════════════════════════
#  启动入口
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    sys.modules['main'] = sys.modules['__main__']
    import vis
    vis.set_pipeline_runner(run_pipeline)
    print('[Server] 启动 Flask 服务器 (threaded=True)')
    vis.app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

