"""
流程数据提取模块.

负责从各模块的 JSON 文件中提取指定时间范围的事件数据。
从 Redis 读取模块实时进度，等待所有模块处理完再提取。
"""
import os
import json
import time
import logging
import redis
from typing import Dict, List, Tuple

from core.inference_stream import KEY_PROGRESS, KEY_SOURCE_DONE

logger = logging.getLogger("evaluation.data_extractor")


class FlowDataExtractor:
    """
    流程数据提取器.

    从各模块的 JSON 文件中提取指定时间范围的事件数据。
    从 Redis 读取模块实时进度，等待所有模块处理完再提取。
    """

    def __init__(self, result_dir: str, redis_client=None):
        """
        初始化数据提取器.

        Args:
            result_dir: 结果目录路径
            redis_client: Redis 客户端（用于读取模块进度）
        """
        self._result_dir = result_dir
        self._redis = redis_client or redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )

        # 动态解析 config.yaml 确定启用的模块
        self._enabled_modules = {"voice", "tracker", "gaze", "behavior"}
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                modules_cfg = cfg.get("modules", {})
                self._enabled_modules = set()
                if modules_cfg.get("voice", True):
                    self._enabled_modules.add("voice")
                if modules_cfg.get("tracker", True):
                    self._enabled_modules.add("tracker")
                    self._enabled_modules.add("gaze")  # gaze 伴随 tracker 启用
                if modules_cfg.get("behavior", True):
                    self._enabled_modules.add("behavior")
                logger.info(f"数据提取器初始化成功，当前启用的等待模块: {self._enabled_modules}")
            except Exception as e:
                logger.warning(f"数据提取器加载配置文件失败，默认等待全部模块: {e}")

    def extract(self, start_sec: float, end_sec: float,
                wait: bool = True, timeout: int = 300) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """
        提取指定时间范围的事件.

        Args:
            start_sec: 开始时间（秒）
            end_sec: 结束时间（秒）
            wait: 是否等待所有模块处理完
            timeout: 超时时间（秒），默认5分钟

        Returns:
            (voice_events, tracker_events, gaze_events, behavior_events)
        """
        if wait:
            logger.info(f"等待所有模块处理到 {end_sec}s...")
            self._wait_all_modules(end_sec, timeout)
            # 等 progress 后额外 sleep 等各模块 key_moments 文件写完(progress 是帧处理进度, 事件写盘滞后)
            time.sleep(2.0)

        events = {
            source: self._extract_events_from_json(source, start_sec, end_sec)
            for source in ("voice", "tracker", "gaze", "behavior")
        }

        logger.info(
            f"提取事件完成: voice={len(events['voice'])}条, tracker={len(events['tracker'])}条, "
            f"gaze={len(events['gaze'])}条, behavior={len(events['behavior'])}条, "
            f"时间范围={start_sec:.2f}s ~ {end_sec:.2f}s"
        )

        return tuple(events.values())

    def save_extracted_data(self, flow_data: dict) -> None:
        """保存到 evaluation/extracted_{flow_type}_{flow_id}.json."""
        try:
            eval_dir = os.path.join(self._result_dir, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)

            flow_id = flow_data.get("flow_id", "unknown_flow")
            flow_type = flow_data.get("flow_type", "unknown")
            filename = f"extracted_{flow_type}_{flow_id}.json"
            output_path = os.path.join(eval_dir, filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(flow_data, f, ensure_ascii=False, indent=2)

            logger.info(f"提取拼接好的流程数据已成功保存到: {output_path}")

        except Exception as e:
            logger.error(f"保存提取的流程数据失败: {e}", exc_info=True)

    def _wait_all_modules(self, target_sec: float, timeout: int) -> None:
        """
        等待所有模块都处理到目标时间.

        从 Redis 读取 inference:progress（per-source 粒度，与 InferenceSync
        _compute_global_sec 一致），取所有未结束 source 的最小进度，当最小值
        >= target_sec 时返回。

        注意：进度字段名是 source 名（voice/tracking/gaze/video_front/
        video_pop 等），而非模块名。早期实现用 hget 按模块名
        读取 tracker/behavior，这两个字段永远不存在 → 恒为 0 → 每次都空等
        到 timeout，导致流程结束后评估被延迟数分钟。

        Args:
            target_sec: 目标时间（秒）
            timeout: 超时时间（秒）
        """
        start_time = time.time()

        while True:
            progress = self._get_all_module_progress()

            # 无任何 source 进度：模块可能尚未启动或已全部退出，直接提取避免空等
            if not progress:
                logger.warning("inference:progress 无可用 source 进度，跳过等待直接提取")
                return

            # 所有未结束 source 都已超过目标时间即可返回
            if min(progress.values()) >= target_sec:
                logger.info(f"所有模块都已处理到 {target_sec}s，进度: {progress}")
                return

            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"等待超时 ({timeout}s)，模块进度: {progress}")
                return

            # 等待1秒再检查
            logger.debug(f"等待中... 进度: {progress}, 目标: {target_sec}s")
            time.sleep(1)

    def _get_all_module_progress(self) -> Dict[str, float]:
        """
        从 Redis 获取所有未结束 source 的实时进度.

        读取 inference:progress 全部字段（per-source），剔除已写入
        inference:source_done 的结束 source。

        Returns:
            {source_name: progress_sec}
        """
        try:
            all_progress = self._redis.hgetall(KEY_PROGRESS)
            done = self._redis.hgetall(KEY_SOURCE_DONE)
        except Exception as e:
            logger.error(f"读取模块进度失败: {e}")
            return {}

        progress: Dict[str, float] = {}
        for source, val in all_progress.items():
            if source in done:
                continue
            try:
                progress[source] = float(val)
            except (TypeError, ValueError):
                continue
        return progress

    def _extract_events_from_json(self, source: str, start_sec: float, end_sec: float) -> List[Dict]:
        """
        从 {source}/{source}_key_moments.json 提取时间范围内的事件.

        Args:
            source: 模块名（voice/tracker/gaze/behavior）
            start_sec: 开始时间
            end_sec: 结束时间

        Returns:
            事件列表
        """
        path = os.path.join(self._result_dir, source, f"{source}_key_moments.json")

        if not os.path.exists(path):
            logger.warning(f"{source}事件文件不存在: {path}")
            return []

        try:
            with open(path, encoding="utf-8") as f:
                all_events = json.load(f)

            # 按时间范围过滤
            return [
                ev for ev in all_events
                if start_sec <= (ev.get("localSec") or 0) <= end_sec
            ]

        except Exception as e:
            logger.error(f"加载{source}事件失败: {e}")
            return []

