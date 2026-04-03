#!/usr/bin/env python3
"""
分段评估模块 — 三段式沟通评分 + 监护制度评分
从 main.py 分离的纯评估逻辑，不含副作用（不推事件、不写共享列表）
"""
import re

DEVICE_PATTERN = re.compile(r'(1ES\w+|T1RPA\w+|LCO\w+|RPA\w+|SM3)')


def _ts_str(sec):
    """秒数 → MM:SS 格式"""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def evaluate_mini_report(cmd_ev, confirms):
    """
    三段式沟通评分（纯评分，无副作用）
    只评估：设备编号 + 确认闭环，不涉及监护制度

    Args:
        cmd_ev: dict, 操作指令事件 {"time_sec", "text", ...}
        confirms: list[dict], 确认事件列表

    Returns: dict 评价结果
    """
    text = cmd_ev.get("text", "")
    has_device = bool(DEVICE_PATTERN.search(text))
    dm = DEVICE_PATTERN.search(text)
    device = dm.group(0) if dm else ""

    score = 10
    details = []

    # 设备编号
    if has_device:
        details.append("✅ 含设备编号")
    else:
        details.append("❌ 无设备编号"); score -= 3

    # 确认
    if confirms:
        details.append(f"✅ {len(confirms)}次确认")
        gap = confirms[0]["ti - me_sec"]cmd_ev["time_sec"] if confirms else 0
        if gap > 5:
            details.append("⚠️ 确认较慢"); score -= 1
        else:
            details.append("✅ 确认及时")
    else:
        details.append("❌ 无确认"); score -= 4

    score = max(0, min(10, score))
    return {
        "time": _ts_str(cmd_ev["time_sec"]),
        "time_sec": round(cmd_ev["time_sec"], 2),
        "device": device,
        "cmd_text": text,
        "score": score,
        "detail": " | ".join(details),
    }


def evaluate_supervision_report(request_ev, confirm_ev=None,
                                hand_raise_info=None, bound_info=None,
                                sup_status_list=None, operator=None,
                                report_id=1, fps=30.0):
    """
    计算一次监护制度流程的评价（纯评分，无副作用）

    Args:
        request_ev: dict, 监护请求事件
        confirm_ev: dict|None, 监护确认事件
        hand_raise_info: dict|None, 举手检测信息
        bound_info: dict|None, 监护绑定信息
        sup_status_list: list[dict]|None, 监护距离状态列表
        operator: str|None, 操作员角色名
        report_id: int, 报告序号
        fps: float, 视频帧率

    Returns: dict 评价结果
    """
    score = 10
    details = []
    req_ts = request_ev.get("time_sec", 0)
    req_frame = int(req_ts * fps)

    # 操作员举手
    if hand_raise_info:
        details.append("✅ 操作员举手")
        hr_ts = hand_raise_info.get("time_sec", 0)
        hr_gap = hr_ts - req_ts
        if hr_gap > 30:
            details.append(f"⚠️ 举手响应较慢({hr_gap:.0f}s)")
            score -= 1
        else:
            details.append(f"✅ 举手响应及时({hr_gap:.0f}s)")
    else:
        details.append("❌ 未检测到举手")
        score -= 3

    # 监护员到位
    if bound_info:
        details.append("✅ 监护员到位")
        arrive_gap = bound_info.get("time_sec", 0) - req_ts
        if arrive_gap > 60:
            details.append(f"⚠️ 到位较慢({arrive_gap:.0f}s)")
            score -= 1
    else:
        details.append("❌ 监护员未到位")
        score -= 3

    # 全程在位比例
    if sup_status_list:
        close_count = sum(1 for s in sup_status_list if s.get("status") == "监护中")
        total = len(sup_status_list)
        ratio = close_count / total if total else 0
        if ratio > 0.8:
            details.append(f"✅ 全程在位({ratio:.0%})")
        elif ratio > 0.5:
            details.append(f"⚠️ 部分在位({ratio:.0%})")
            score -= 1
        else:
            details.append(f"❌ 监护不足({ratio:.0%})")
            score -= 2
    else:
        details.append("— 无监护距离数据")

    # 监护确认
    if confirm_ev:
        details.append("✅ 监护已确认")
    else:
        details.append("⚠️ 无语音确认")
        score -= 1

    score = max(0, min(10, score))

    return {
        "id": report_id,
        "time": _ts_str(req_ts),
        "time_sec": round(req_ts, 2),
        "frame_id": req_frame,
        "request_text": request_ev.get("text", ""),
        "operator": operator or (hand_raise_info.get("operator", "?") if hand_raise_info else "?"),
        "supervisor": "LEADER",
        "hand_raise": bool(hand_raise_info),
        "hand_raise_time_sec": round(hand_raise_info["time_sec"], 2) if hand_raise_info else None,
        "hand_raise_frame_id": hand_raise_info.get("frame_id") if hand_raise_info else None,
        "supervisor_arrived": bool(bound_info),
        "arrive_time_sec": round(bound_info["time_sec"], 2) if bound_info else None,
        "arrive_frame_id": bound_info.get("frame_id", int(bound_info["time_sec"] * fps)) if bound_info else None,
        "bound_duration_sec": round(sup_status_list[-1]["time_sec"] - sup_status_list[0]["time_sec"], 1) if sup_status_list and len(sup_status_list) > 1 else 0,
        "supervisor_status": details[-3] if len(details) >= 3 else "—",
        "key_frame": hand_raise_info.get("key_frame", "") if hand_raise_info else "",
        "score": score,
        "detail": " | ".join(details),
    }
