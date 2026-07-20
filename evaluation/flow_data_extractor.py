"""
流程数据提取模块

负责从各模块的 JSON 文件中提取指定时间范围的事件数据。
从 Redis 读取模块实时进度，等待所有模块处理完再提取。
"""
import os
import json
import time
import logging
import redis
from typing import Dict, List, Tuple

logger = logging.getLogger("evaluation.data_extractor")


class FlowDataExtractor:
    """
    流程数据提取器

    从各模块的 JSON 文件中提取指定时间范围的事件数据。
    从 Redis 读取模块实时进度，等待所有模块处理完再提取。
    """

    def __init__(self, result_dir: str, redis_client=None):
        """
        初始化数据提取器

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
        提取指定时间范围的事件

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

        voice_events = self._extract_voice_events(start_sec, end_sec)
        tracker_events = self._extract_tracker_events(start_sec, end_sec)
        gaze_events = self._extract_gaze_events(start_sec, end_sec)
        behavior_events = self._extract_behavior_events(start_sec, end_sec)

        logger.info(
            f"提取事件完成: voice={len(voice_events)}条, tracker={len(tracker_events)}条, "
            f"gaze={len(gaze_events)}条, behavior={len(behavior_events)}条, "
            f"时间范围={start_sec:.2f}s ~ {end_sec:.2f}s"
        )

        return voice_events, tracker_events, gaze_events, behavior_events

    def save_extracted_data(self, flow_data: dict) -> None:
        """保存到 evaluation/extracted_{flow_type}_{flow_id}.json"""
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
        等待所有模块都处理到目标时间

        从 Redis 读取 inference:progress，检查每个模块的进度。

        Args:
            target_sec: 目标时间（秒）
            timeout: 超时时间（秒）
        """
        start_time = time.time()

        while True:
            # 从 Redis 获取所有模块进度
            progress = self._get_all_module_progress()

            # 检查是否所有模块都过了目标时间
            all_passed = all(p >= target_sec for p in progress.values())

            if all_passed:
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
        从 Redis 获取所有模块的实时进度

        Returns:
            {module_name: progress_sec}
        """
        progress = {}
        for module_name in self._enabled_modules:
            progress[module_name] = self._get_module_progress(module_name)
        return progress

    def _get_module_progress(self, module_name: str) -> float:
        """
        从 Redis 获取单个模块的实时进度

        读取 inference:progress Hash 中的模块进度。

        Args:
            module_name: 模块名称

        Returns:
            模块处理到的时间（秒）
        """
        try:
            progress = self._redis.hget("inference:progress", module_name)
            return float(progress) if progress else 0.0
        except Exception as e:
            logger.error(f"获取 {module_name} 进度失败: {e}")
            return 0.0

    def _extract_voice_events(self, start_sec: float, end_sec: float) -> List[Dict]:
        """
        从 voice_key_moments.json 提取语音事件

        Args:
            start_sec: 开始时间
            end_sec: 结束时间

        Returns:
            语音事件列表
        """
        vkm_path = os.path.join(self._result_dir, "voice", "voice_key_moments.json")

        if not os.path.exists(vkm_path):
            logger.warning(f"语音事件文件不存在: {vkm_path}")
            return []

        try:
            with open(vkm_path, encoding="utf-8") as f:
                all_events = json.load(f)

            # 按时间范围过滤
            filtered_events = []
            for ev in all_events:
                ts = ev.get("localSec") or 0
                if start_sec <= ts <= end_sec:
                    filtered_events.append(ev)

            return filtered_events

        except Exception as e:
            logger.error(f"加载语音事件失败: {e}")
            return []

    def _extract_tracker_events(self, start_sec: float, end_sec: float) -> List[Dict]:
        """
        从 tracker_key_moments.json 提取 Tracker 事件

        Args:
            start_sec: 开始时间
            end_sec: 结束时间

        Returns:
            Tracker 事件列表
        """
        mkm_path = os.path.join(self._result_dir, "tracker", "tracker_key_moments.json")

        if not os.path.exists(mkm_path):
            logger.warning(f"Tracker事件文件不存在: {mkm_path}")
            return []

        try:
            with open(mkm_path, encoding="utf-8") as f:
                all_events = json.load(f)

            # 按时间范围过滤
            filtered_events = []
            for ev in all_events:
                ts = ev.get("localSec") or 0
                if start_sec <= ts <= end_sec:
                    filtered_events.append(ev)

            return filtered_events

        except Exception as e:
            logger.error(f"加载Tracker事件失败: {e}")
            return []

    def _extract_gaze_events(self, start_sec: float, end_sec: float) -> List[Dict]:
        """
        从 gaze_key_moments.json 提取 Gaze 事件

        Args:
            start_sec: 开始时间
            end_sec: 结束时间

        Returns:
            Gaze 事件列表
        """
        gkm_path = os.path.join(self._result_dir, "gaze", "gaze_key_moments.json")

        if not os.path.exists(gkm_path):
            logger.warning(f"Gaze事件文件不存在: {gkm_path}")
            return []

        try:
            with open(gkm_path, encoding="utf-8") as f:
                all_events = json.load(f)

            # 按时间范围过滤
            filtered_events = []
            for ev in all_events:
                ts = ev.get("localSec") or 0
                if start_sec <= ts <= end_sec:
                    filtered_events.append(ev)

            return filtered_events

        except Exception as e:
            logger.error(f"加载Gaze事件失败: {e}")
            return []

    def _extract_behavior_events(self, start_sec: float, end_sec: float) -> List[Dict]:
        """
        从 behavior_key_moments.json 提取 Behavior 事件（举手、手指屏幕等）

        Args:
            start_sec: 开始时间
            end_sec: 结束时间

        Returns:
            Behavior 事件列表
        """
        bkm_path = os.path.join(self._result_dir, "behavior", "behavior_key_moments.json")

        if not os.path.exists(bkm_path):
            logger.warning(f"Behavior事件文件不存在: {bkm_path}")
            return []

        try:
            with open(bkm_path, encoding="utf-8") as f:
                all_events = json.load(f)

            # 按时间范围过滤
            filtered_events = []
            for ev in all_events:
                ts = ev.get("localSec") or 0
                if start_sec <= ts <= end_sec:
                    filtered_events.append(ev)

            return filtered_events

        except Exception as e:
            logger.error(f"加载Behavior事件失败: {e}")
            return []

