# 核电站监护制合规检测系统 — 规格说明 (Specification)

> 本文档定义系统的完整规格，作为开发依据。每个需求均有唯一 ID，可追踪、可验证。
> 接口契约（§6）以当前代码为准，AI 改代码前请同时参阅 [`../CLAUDE.md`](../CLAUDE.md) 的改动禁区。

---

## 1. Overview

### 1.1 问题描述

核电站主控室操作人员须遵守监护制、自唱票、信息通报等核安全规程。当前依赖人工监督，效率低、易遗漏。本系统通过两路视频 + 音频的 AI 分析，自动检测操作行为并评估合规性。

### 1.2 系统范围

本系统接收**两路视频**输入（主视角 front + 操作视角 pop，其中 pop 同时覆盖手指屏幕与手指文件检测），输出实时可视化前端和合规评估报告。系统**不**涉及：
- 门禁/安防控制
- 操作权限管理
- 历史数据长期存储（仅保留当次推理结果）

> ⚠ 系统**只有两路视频**（`camFRONT` / `camPOP`）。历史上曾有的 `camBUP` 第三路视角已删除，任何配置/代码/文档都不得再出现 `videos.bup` 或 `video_bup`。

### 1.3 术语表

| 术语 | 定义 |
|------|------|
| 监护制 | 一人操作、一人监护的核安全规程 |
| 自唱票 | 操作员独立执行时自行唱票确认 |
| 信息通报 | 重要信息变更时全员通报确认 |
| global_sec | 全局对齐时钟，取所有未结束且 expected 的 source 进度最小值 |
| source | 推理流数据来源标识（voice/tracking/video_front/gaze/video_pop/behavior 等） |
| view_id | WebSocket 二进制协议中的视角标识：0=front, 1=pop（硬约束，不可改） |

---

## 2. Goals & Non-Goals

### 2.1 Goals

| ID | Goal | 优先级 |
|----|------|--------|
| G-01 | 实时检测主控室人员位置、角色、动作 | P0 |
| G-02 | 实时转录操作人员语音并识别关键字 | P0 |
| G-03 | 自动识别监护制、自唱票、信息通报流程 | P0 |
| G-04 | 大模型评估每段流程的合规性 | P0 |
| G-05 | 前端实时可视化两路视频（front + pop）+ 标注 + 字幕 + 评估报告 | P0 |
| G-06 | 推理结束后 100% 释放 GPU 显存 | P0 |

### 2.2 Non-Goals

| ID | Non-Goal | 理由 |
|----|----------|------|
| NG-01 | 不保存历史数据 | 仅保留当次推理结果 |
| NG-02 | 不支持多用户并发 | 单次仅处理一组视频 |
| NG-03 | 不提供远程控制接口 | 仅监督辅助，不执行操作 |
| NG-04 | 不要求 100% 准确率 | 辅助判断，最终由人决定 |

---

## 3. Functional Requirements

### 3.1 视频处理

| ID | 需求 | 验证方式 |
|----|------|---------|
| F-01 | 系统须处理两路视频（camFRONT/camPOP，其中 camPOP 同时承担手指屏幕与手指文件检测） | 启动后两路窗口均有画面输出 |
| F-02 | 系统须以 ≥2 FPS 处理视频帧 | 日志记录帧率 |
| F-03 | 视频帧编码为 JPEG quality ≤ 40（front Q35 / pop Q40），统一缩放到 960×540 | 检查编码参数 |

### 3.2 人员检测与跟踪

| ID | 需求 | 验证方式 |
|----|------|---------|
| F-10 | 系统须检测主控室人员位置并分配唯一 ID | 前端显示人员框 + ID |
| F-11 | 系统须识别角色：LEADER/ROAD1/ROAD2/SUPERVISOR | 前端显示角色标签 |
| F-12 | 系统须计算人员间距离，阈值可配置 | config.yaml 验证 |
| F-13 | 系统须检测举手动作（由 Tracker 的 HandRaiser 产出 `behavior.hand_raised`） | 输出 HAND_RAISED 事件 |

### 3.3 语音识别

| ID | 需求 | 验证方式 |
|----|------|---------|
| F-20 | 系统须从 camFRONT 音频轨道提取语音 | 输出文本 |
| F-21 | 系统须识别以下关键字：请求监护、设备码（九字码）、执行、核对、信息通报、通报完毕、收到 | 输出 key_moment 事件 |
| F-22 | 系统须使用拼音匹配处理 ASR 识别误差 | 测试 "请求监护" vs "请结束" |

### 3.4 凝视检测

| ID | 需求 | 验证方式 |
|----|------|---------|
| F-30 | 系统须检测人员头部位置 | 输出头部框 |
| F-31 | 系统须估计注视方向并判断是否在 ROI 内 | 输出 IN_ROI/OUTSIDE_ROI |
| F-32 | 脱盘持续 ≥ 60 秒须触发 GAZE_ALERT 事件 | 输出 GAZE_ALERT |
| F-33 | 凝视异常数据须记录持续时间 | `gaze_key_moments.json` 格式校验 |

### 3.5 行为检测

| ID | 需求 | 验证方式 |
|----|------|---------|
| F-40 | 系统须检测手指指向屏幕（camPOP 操作视角，由 `FingerScreenDetector` 判定，输出 `behavior.finger_screen`） | 输出 FINGER_SCREEN 事件 |
| F-41 | 系统须检测手指指向文件（camPOP，由 `FingerFileDetector` 判定，输出 `behavior.finger_file`） | 输出 FINGER_FILE 事件 |

> 行为检测特殊性：`BehaviorModule` 在 camPOP 上共享一次 YOLO（`behavior_yolo.pt`）推理，每 5 帧 `model.track` 一次，把 `results` 串行分发给 `FingerScreenDetector.detect(...)` 与 `FingerFileDetector.detect(...)` 两个纯判定器。**判定器不持有模型。**

### 3.6 规则检测

| ID | 需求 | 验证方式 |
|----|------|---------|
| F-50 | 监护制：语音"请求监护" + 5秒内举手 → 触发流程 | 输出 FLOW_STARTED |
| F-51 | 监护制：九字码复述 → 执行 → 核对确认 → 三项检查 | 输出检查清单 |
| F-52 | 监护制：监护员与操作员离开 > 10秒 → 结束流程 | 输出 FLOW_ENDED |
| F-53 | 自唱票：设备码重复 → 触发流程 | 输出 FLOW_STARTED |
| F-54 | 信息通报：举手 + "信息通报" → 触发流程 | 输出 FLOW_STARTED |

### 3.7 评估

| ID | 需求 | 验证方式 |
|----|------|---------|
| F-60 | 流程结束后须提取各模块 key_moments | 检查提取的 JSON 文件 |
| F-61 | 流程结束后须调用 Qwen3-8B 生成评估报告 | 输出 qwen_response_*.json |
| F-62 | 评估报告须推送到前端（经 `push_text` 直推，绕过对齐中间件） | 前端显示评估内容 |
| F-63 | 提取 keymoment 前，须 `_wait_all_modules(end_sec, timeout=90)` 等所有推理模块进度 `min >= end_sec` | 日志可见等待完成或 90s 超时放行 |
| F-64 | 推评估 chunk 前，须 `wait_playback_reached(end_sec-0.5, timeout=60)` 等前端可视化播放到流程结束（不剧透） | 流程结束前评估不出现，结束后才逐字 |
| F-65 | 评估报告须逐 token 流式输出（`TextIteratorStreamer`→`segment_report_stream` chunk 累积→前端 typewriter 逐字）；`segment_report` 不覆盖 `streamBuffer`；typewriter 追完 + `reportText` 到达→切完成态显分数 | 前端逐字显示，完成后显分数/进度条 |

### 3.8 交互

| 需求 | 验证方式 |
|------|---------|
| 前端须显示两路视频画面（front + pop） | 页面加载后可见 |
| 前端须在视频上叠加人员框、距离线、标签 | 启动后可见 |
| 前端须实时显示语音识别字幕 | 有语音时显示 |
| 前端须提供"开始"/"停止"按钮 | 页面可见按钮 |
| 前端须展示评估报告 | 流程结束后可见 |

---

## 4. Non-Functional Requirements

| ID | 需求 | 指标 | 验证方式 |
|----|------|------|---------|
| N-01 | 推理延迟 | 视频处理 ≤ 实时 1x 速度 | 日志对比 wall-clock |
| N-02 | 前端同步延迟 | ≤ 3 秒 | 视频帧时间戳对比 |
| N-03 | GPU 显存释放 | 推理结束后 100% 释放 | `nvidia-smi` 检查 |
| N-04 | 模块可配置 | 各模块可独立启用/禁用 | config.yaml 修改验证 |
| N-05 | 断网安全 | 文件修改使用原子操作 | 验证 mv 流程 |
| N-06 | 内存泄漏 | 重复 10 次推理后显存不增长 | 对比首次和末次显存 |

---

## 5. Architecture

### 5.1 进程模型

```
main.py (协调器, FastAPI+uvicorn 常驻, 0 显存)
  ├── Web 进程 (常驻)
  │     ├── FastAPI (端口 5002)
  │     ├── WebSocket (二进制帧 + JSON)
  │     ├── InferenceSync (global_sec 对齐)
  │     ├── Rules (规则状态机)
  │     └── Evaluation (Qwen3-8B 评估, 子进程)
  │
  └── on_pipeline_start 用 multiprocessing.Process(spawn) 拉起:
        ├── VoiceModule 进程
        ├── TrackerModule 进程 (内含 GazeModule)
        └── BehaviorModule 进程
  （启动时会 r.flushdb() 清空 Redis）
```

### 5.2 通信架构

```
                   ┌──────────────┐
                   │  Redis       │
                   │  ─────────   │
                   │  Streams     │
                   │  Hash        │
                   │  Keys        │
                   └──────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
     EventBus       InferenceStream   Progress / Snapshot
  (module:events:*) (inference:     (inference:
                     results:all)    progress / snapshot)
          │               │               │
          │               │               │
     Rules + Eval    InferenceSync    Modules
                     → WebSocket     写入进度
```

### 5.3 数据流

```
双轨设计：

轨道 1 — 推理流（可视化数据）：
  Module → push_display() → InferenceStream → Redis Stream (inference:results:all)
  + InferenceSync 订阅 event_bus FLOW_STARTED/FLOW_ENDED → push_display(flow_start/flow_end)
  → InferenceSync (global_sec 对齐) → WSHandler → 前端

轨道 2 — 事件流（业务通信）：
  Module → publish(EventTopic) → EventBus → Redis Stream (module:events:*)
  → Rules (状态机) → Evaluation (评估) → 前端（评估报告经 push_text 直推）
```

---

## 6. Interface Specifications

### 6.1 推理流接口（`core/inference_stream.py`）

```python
# 写入（唯一入口）
def push_display(source: str, data: dict) -> None

# 数据格式
{
  "source": "voice|tracking|gaze|video_front|video_pop|behavior|progress|flow_start|flow_end",
  "localSec": float,   # 时间对齐类事件必须包含，否则被丢弃
  "tag": str,
  "data": dict
}

# 视频帧数据（data.frame_data 为 latin1 编码的 JPEG 字节串）
{
  "source": "video_pop",
  "localSec": 12.34,
  "tag": "video",
  "data": {"frame_data": "<jpeg bytes decoded as latin1>"}
}
```

### 6.2 事件流接口（`core/event_bus.py`）

```python
# 事件类型（EventTopic 枚举，同时是 Redis Stream key 后缀）
EventTopic.VOICE_KEY_MOMENT      # voice.key_moment
EventTopic.TRACKER_PROXIMITY     # tracker.proximity
EventTopic.TRACKER_HEADCOUNT     # tracker.headcount
EventTopic.BEHAVIOR_HAND_RAISED  # behavior.hand_raised（发布者为 Tracker 的 HandRaiser）
EventTopic.BEHAVIOR_FINGER_SCREEN# behavior.finger_screen（发布者为 BehaviorModule）
EventTopic.BEHAVIOR_FINGER_FILE  # behavior.finger_file（发布者为 BehaviorModule）
EventTopic.GAZE_ATTENTION        # gaze.attention
EventTopic.GAZE_ALERT            # gaze.alert
EventTopic.FLOW_STARTED          # flow.started
EventTopic.FLOW_ENDED            # flow.ended
EventTopic.RULE_KEY_MOMENT       # rule.key_moment
EventTopic.SAVE_KEY_MOMENTS      # save.key_moments（评估层通知各模块落盘）

# 发布
def publish(topic: EventTopic, data: dict, ts: float = 0.0) -> None

# 订阅
def subscribe(topic: EventTopic, callback: Callable) -> None
```

消息包格式：`{"type": str, "data": dict, "ts": float}`（由 `publish` 自动封装）。

### 6.3 HTTP API

| 端点 | 方法 | 请求体 | 响应 |
|------|------|--------|------|
| `/start` | POST | 无 | 触发 `on_pipeline_start`（spawn 拉起子进程 + flushdb） |
| `/stop` | POST | 无 | 终止推理子进程 |
| `/ws/data` | WebSocket | — | 二进制 batch + 文本帧（done / segment_report） |
| `/` 或 `/status` | GET | — | 页面 / 状态 |

### 6.4 WebSocket 协议（`web/ws_handler.py` ↔ `frontend/src/composables/useWS.ts`）

**二进制消息格式（大端）：**

```
Offset  Size  Field
0       1     version (uint8)         # 当前恒为 1
1       4     globalSec (float32)      # 大端
5       1     view_count (uint8)

重复 view_count 次:
  6       1     view_id (uint8: 0=front, 1=pop)   # ⚠ 硬约束，不可改
  7       4     frame_len (uint32)                # 大端
  11      N     JPEG bytes

尾部:
  M       4     json_len (uint32)      # 大端
  M+4     N     JSON UTF-8 bytes（含 meta，如 gaze/voice/tracking/progress/flow_*）
```

文本帧（独立通道，不经对齐）：`{"source": "done"}`（EOF 哨兵）、`{"tag": "segment_report", ...}`、`{"tag": "segment_report_stream", ...}`。

> ⚠ **view_id 红线（与 CLAUDE.md / DEVELOPMENT_GUIDE 呼应）**：`ws_handler.VIEW_MAP` 固定 `video_front→0`、`video_pop→1`；前端 `useWS.ts` 在 `playOneBatch` 里**硬编码** `f[0]`→front、`f[1]`→pop 解包（`if (f[0] !== undefined)` / `if (f[1] !== undefined)`）。任何改动 view_id 取值或顺序的尝试都会导致某路视频黑屏，**严禁修改**。

### 6.5 关键事件 JSON 格式（示例）

```json
[
  {"localSec": 2.72,  "key_moment": "请求护卫"},
  {"localSec": 10.15, "key_moment": "1EAS013VB"},
  {"localSec": 60.15, "key_moment": "没有看盘台持续15.4秒"}
]
```

约束：每条记录仅含 `localSec`（float）和 `key_moment`（str）两个字段。

---

## 7. Configuration（`config.yaml`）

```yaml
app:
  gpu: "0"
  fps: 30.0
  port: 5002

redis:
  host: "localhost"
  port: 6379
  db: 0

paths:
  data_root: "data"
  model_root: "models"
  result_root: "data/results"

videos:                # ⚠ 仅两路，无 bup
  front: "data/videos/camFRONT.mpg"
  pop: "data/videos/camPOP.mpg"

supervision:
  bind_hold_sec: 10.0
  unbind_hold_sec: 10.0
  dist_close_px: 200
  dist_near_px: 560
  consec_raise: 3
  consec_idle: 3
  cooldown_sec: 1.5

tracker:
  detection:
    conf_threshold: 0.65
    pose_confidence: 0.35
    nms_threshold: 0.35
    img_size: 640

bus:
  max_queue_size: 1024

voice:
  sample_rate: 16000
  sentence_gap_sec: 0.6
  asr_engine: "qwen3"
  model_path: "models/voice/qwen/Qwen3-ASR-0.6B"
  aligner_path: "models/voice/qwen/Qwen3-ForcedAligner-0.6B"
  torch_dtype: "bfloat16"

gaze:
  head_conf_th: 0.55
  inout_th: 0.5
  heatmap_th: 0.3
  head_min_size: 20
  head_max_size: 300

behavior:
  finger_screen:
    detect_conf: 0.25
    screen_overlap_threshold: 0.2
    max_dist: 10
    cooldown_sec: 1.5
  finger_file:
    detect_conf: 0.25
    file_iou_threshold: 0.2
    cooldown_sec: 1.5

modules:
  gaze: true
  voice: true
  tracker: true
  personnel_status: true
  info_notice: true
  behavior: true

rules:
  supervision: true
  self_ticket: true
  personnel_status: true
  info_notice: true
```

---

## 8. Deliverables

| ID | 交付物 | 路径 |
|----|--------|------|
| D-01 | 源代码 | `A_DemoSrc/` |
| D-02 | 配置文件 | `config.yaml` |
| D-03 | 前端构建产物 | `frontend/dist/` |
| D-04 | 架构文档 | `docs/ARCHITECTURE.md` |
| D-05 | 开发指南 | `docs/DEVELOPMENT_GUIDE.md` |
| D-06 | AI 导航/红线 | `CLAUDE.md` |
| D-07 | 行为规范标准 | `docs/行为规范标准.md` |

---

## 9. Open Questions

| ID | 问题 | 状态 |
|----|------|------|
| O-01 | 是否支持多路视频同时推理？ | 已实现（两路：front + pop，无 bup） |
| O-02 | 是否支持自定义 ROI 区域？ | 已实现（SCREEN_POLYGONS 预标注） |
| O-03 | 是否支持评估报告导出？ | 待定 |
| O-04 | 是否支持多语言语音识别？ | 当前仅中文 |
