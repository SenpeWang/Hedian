"""Behavior 结果存储：统一管理与落盘所有行为关键事件.

本模块以面向对象方式统管行为类事件（举手、手指屏幕、手指文件等），
所有事件通过统一入口追加落盘到 behavior_key_moments.json。
"""
import fcntl
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple

from core.base_storage import BaseStorage
from core.event_bus import EventBus, EventTopic
from core.path_manager import PathConfig

logger = logging.getLogger("module.behavior.storage")

BEHAVIOR_KEY_MOMENTS_FILENAME = "behavior_key_moments.json"
VALID_IDENTITIES: Set[str] = {"LEADER", "ROAD1", "ROAD2"}


class BehaviorStorage(BaseStorage):
    """行为事件存储：以面向对象方式统一落盘所有行为关键事件。

    举手与手指类事件均通过统一方法（add_event / report_hand_raise）
    原子追加到 behavior_key_moments.json，避免多进程写入互相覆盖。
    """

    def __init__(self, paths: PathConfig) -> None:
        """初始化行为存储.

        Args:
            paths: 路径管理器，提供结果目录解析能力.
        """
        super().__init__(paths, "behavior")

    def _read_events(self, run_id: str) -> List[Dict[str, Any]]:
        """读取已有行为事件列表.

        Args:
            run_id: 本次运行标识.

        Returns:
            已有事件列表；文件不存在或损坏时返回空列表.
        """
        file_path = self._paths.get_result_path(
            run_id, "behavior", BEHAVIOR_KEY_MOMENTS_FILENAME
        )
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as read_error:
            logger.warning(f"读取 {file_path} 失败: {read_error}")
            return []

    def add_event(self, run_id: str, event: Dict[str, Any]) -> None:
        """跨进程安全地追加单条行为事件.

        tracker(举手) 与 behavior(手指类) 两个进程并发写同一文件,
        读-改-写全程持文件锁; 写回前按 localSec 排序, 保证文件时间轴有序.

        Args:
            run_id: 本次运行标识.
            event: 行为事件字典，约定包含 localSec 与 key_moment 字段.
        """
        file_path = str(
            self._paths.get_result_path(
                run_id, "behavior", BEHAVIOR_KEY_MOMENTS_FILENAME
            )
        )
        with self._locked(file_path):
            events_list = self._read_events(run_id)
            events_list.append(event)
            events_list.sort(key=lambda event: event.get("localSec", 0))
            save_ok = self._save_json_atomic(
                BEHAVIOR_KEY_MOMENTS_FILENAME, run_id, events_list,
                output_path=file_path,
            )
            if not save_ok:
                logger.warning(f"行为事件落盘失败(将依赖收尾合并补写): {event}")

    def merge_events(self, run_id: str, events: List[Dict[str, Any]]) -> None:
        """收尾兜底合并: 仅补写文件中缺失的事件(按完整事件键去重).

        Args:
            run_id: 本次运行标识.
            events: 内存中累积的事件列表.
        """
        file_path = str(
            self._paths.get_result_path(
                run_id, "behavior", BEHAVIOR_KEY_MOMENTS_FILENAME
            )
        )
        with self._locked(file_path):
            merged = self._read_events(run_id)
            seen = {self._event_key(event) for event in merged}
            added = 0
            for event in events:
                key = self._event_key(event)
                if key not in seen:
                    merged.append(event)
                    seen.add(key)
                    added += 1
            if added:
                merged.sort(key=lambda event: event.get("localSec", 0))
                save_ok = self._save_json_atomic(
                    BEHAVIOR_KEY_MOMENTS_FILENAME, run_id, merged,
                    output_path=file_path,
                )
                if not save_ok:
                    logger.warning(f"收尾合并落盘失败: 补写 {added} 条事件丢失")
                    return
                logger.info(f"收尾合并补写 {added} 条缺失事件")

    @staticmethod
    def _event_key(event: Dict[str, Any]) -> Tuple[Any, ...]:
        """构造事件去重键(localSec+文案+来源+track_id).

        Args:
            event: 行为事件字典.

        Returns:
            事件唯一键.
        """
        return (event.get("localSec"), event.get("key_moment"),
                event.get("source"), event.get("track_id"))

    def _locked(self, file_path: str) -> "_FileLock":
        """获取跨进程文件锁上下文.

        Args:
            file_path: 目标数据文件路径, 锁文件为其加 .lock 后缀.

        Returns:
            排他锁上下文管理器.
        """
        return _FileLock(file_path + ".lock")

    def report_hand_raise(
        self,
        event_bus: EventBus,
        run_id: str,
        identity: Optional[str],
        timestamp: float,
        track_id: Optional[int] = None,
        inference_fn: Optional[Any] = None,
    ) -> None:
        """按行为类规则上报一次举手事件.

        完成三件事：
        1. 落盘到 behavior_key_moments.json（与手指类同一文件）
           - 已分配身份：记录为 '{identity}举手'（如 'ROAD1举手'）
           - 未分配身份：记录为 '举手'
        2. 推送推理流供前端展示（署名 behavior）
        3. 推送事件流供 web 规则层判定（EventTopic.BEHAVIOR_HAND_RAISED）

        Args:
            event_bus: 事件总线，用于推送推理流与事件流.
            run_id: 本次运行标识.
            identity: 举手者的合法身份 ('LEADER'|'ROAD1'|'ROAD2')；未分配严格为 None.
            timestamp: 事件发生时间（秒）.
            track_id: 跟踪目标全局唯一 ID，可选.
            inference_fn: 推理流推送回调，为 None 时回退到 event_bus.push_display.
        """
        local_sec = round(timestamp, 2)
        valid_identity = identity if identity in VALID_IDENTITIES else None
        key_moment = f"{valid_identity}举手" if valid_identity else "举手"

        self.add_event(run_id, {
            "localSec": local_sec,
            "key_moment": key_moment,
            "source": "behavior",
        })

        display_data = {
            "state": "举手",
            "operator": valid_identity,
        }
        if track_id is not None:
            display_data["track_id"] = track_id

        if inference_fn is not None:
            inference_fn("behavior", {
                "localSec": local_sec,
                "tag": "HAND_RAISED",
                "data": display_data,
            })
        elif hasattr(event_bus, "push_display"):
            event_bus.push_display("behavior", {
                "localSec": local_sec,
                "tag": "HAND_RAISED",
                "data": display_data,
            })

        event_payload = {
            "localSec": local_sec,
            "operator": valid_identity,
        }
        if track_id is not None:
            event_payload["track_id"] = track_id

        event_bus.publish(
            EventTopic.BEHAVIOR_HAND_RAISED,
            event_payload,
            timestamp=timestamp,
        )

        logger.info(
            f"举手上报: {key_moment}"
            f" {'(track_id=' + str(track_id) + ')' if track_id is not None else ''}"
            f" @{local_sec}s [source=behavior]"
        )


class _FileLock:
    """基于 fcntl.flock 的跨进程排他锁上下文管理器."""

    def __init__(self, lock_path: str) -> None:
        """初始化锁上下文.

        Args:
            lock_path: 锁文件路径.
        """
        self._lock_path = lock_path
        self._handle: Optional[TextIO] = None

    def __enter__(self) -> "_FileLock":
        """打开锁文件并加排他锁, 加锁失败时关闭句柄后重抛.

        Returns:
            自身.
        """
        self._handle = open(self._lock_path, "w")
        try:
            fcntl.flock(self._handle, fcntl.LOCK_EX)
        except BaseException:
            self._handle.close()
            self._handle = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """解锁并关闭句柄, 清理失败不掩盖 with 体的原始异常.

        Args:
            exc_type: with 体抛出的异常类型, 未抛异常时为 None.
            exc_value: 抛出的异常实例, 未抛异常时为 None.
            exc_traceback: 抛出异常的堆栈跟踪, 未抛异常时为 None.
        """
        try:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
        finally:
            try:
                self._handle.close()
            except OSError as close_error:
                logger.warning(f"关闭锁文件失败: {close_error}")
            finally:
                self._handle = None
