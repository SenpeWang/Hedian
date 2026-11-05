"""场景(机室)配置加载.

支持双机室: 每个机室一套工位基准点(anchor)与判定多边形(polygon), 以及一套屏幕关注区
(ROI.json). 全部集中在 config.yaml 的 `scene` 段 + data/workstations.json, 避免换机室改代码.

config.yaml 示例::

    scene:
      room_id: 1                        # 1=机室1, 2=机室2
      workstations: "data/workstations.json"
      roi: "data/ROI_room1.json"        # 可选；缺省时按 ROI_room{id}.json -> ROI.json 回退
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("module.tracker.scene")

IDENTITIES: Tuple[str, ...] = ("LEADER", "ROAD1", "ROAD2")


@dataclass
class SceneConfig:
    """单个机室的场景配置.

    Attributes:
        room_id: 机室编号字符串 ("1"/"2").
        anchors: 身份 -> 工位基准像素点, 供最近距离身份分配使用.
        polygons: 身份 -> 归一化多边形顶点列表, 供在岗/离岗判定使用.
        image_size: 多边形归一化所依据的画面尺寸 (w, h).
        roi_path: 屏幕关注区(ROI) 文件路径; 文件不存在时为 None.
    """

    room_id: str
    anchors: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    polygons: Dict[str, List[List[float]]] = field(default_factory=dict)
    image_size: Tuple[int, int] = (1920, 1080)
    roi_path: Optional[Path] = None

    def polygon_px(self, identity: str) -> Optional[List[Tuple[float, float]]]:
        """把指定身份的归一化多边形换算成像素顶点.

        Args:
            identity: 'LEADER' | 'ROAD1' | 'ROAD2'.

        Returns:
            像素顶点列表; 未配置时返回 None.
        """
        pts = self.polygons.get(identity)
        if not pts:
            return None
        w, h = self.image_size
        return [(float(x) * w, float(y) * h) for x, y in pts]


def _resolve(base_dir: Path, raw: str) -> Path:
    """相对路径按项目根解析, 绝对路径原样返回."""
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p)


def load_scene(config: dict, base_dir: Path) -> SceneConfig:
    """按 config['scene'] 加载当前机室的场景配置.

    缺失文件时返回空 anchors/polygons (调用方应回退到代码内默认值), 不抛异常,
    以保证旧配置与旧数据目录仍可启动.

    Args:
        config: 完整配置字典.
        base_dir: 项目根目录.

    Returns:
        SceneConfig: 当前机室场景配置.
    """
    scene_cfg = config.get("scene", {}) or {}
    room_id = str(scene_cfg.get("room_id", 1))
    ws_file = scene_cfg.get("workstations", "data/workstations.json")
    ws_path = _resolve(base_dir, ws_file)

    scene = SceneConfig(room_id=room_id)

    # ---- 工位定义 ----
    try:
        if ws_path.is_file():
            data = json.loads(ws_path.read_text("utf-8"))
            size = data.get("image_size") or [1920, 1080]
            scene.image_size = (int(size[0]), int(size[1]))
            room = (data.get("rooms") or {}).get(room_id)
            if room is None:
                logger.warning(
                    "workstations.json 中没有机室 %s, 回退到默认工位坐标", room_id
                )
            else:
                for identity in IDENTITIES:
                    item = room.get(identity) or {}
                    anchor = item.get("anchor")
                    if anchor and len(anchor) >= 2:
                        scene.anchors[identity] = (float(anchor[0]), float(anchor[1]))
                    polygon = item.get("polygon")
                    if polygon and len(polygon) >= 3:
                        scene.polygons[identity] = polygon
                logger.info(
                    "机室 %s 工位已加载: anchors=%d polygons=%d",
                    room_id, len(scene.anchors), len(scene.polygons),
                )
        else:
            logger.warning("工位定义文件不存在: %s, 回退到默认工位坐标", ws_path)
    except Exception as exc:  # 配置错误不应中断推流
        logger.error("加载工位定义失败(%s): %s", ws_path, exc, exc_info=True)

    # ---- 屏幕关注区 ROI: 显式配置 > ROI_room{id}.json > ROI.json ----
    try:
        roi_raw = scene_cfg.get("roi")
        candidates = []
        if roi_raw:
            candidates.append(_resolve(base_dir, str(roi_raw)))
        candidates.append(base_dir / "data" / f"ROI_room{room_id}.json")
        candidates.append(base_dir / "data" / "ROI.json")
        for cand in candidates:
            if cand.is_file():
                scene.roi_path = cand
                break
        if scene.roi_path is None:
            logger.warning("机室 %s 未找到 ROI 文件, 关注度判定将退化为默认行为", room_id)
        else:
            logger.info("机室 %s ROI: %s", room_id, scene.roi_path)
    except Exception as exc:
        logger.error("解析 ROI 路径失败: %s", exc, exc_info=True)

    return scene
