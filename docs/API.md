# API 参考文档（API Reference）

> 本文档是项目的**接口契约（Interface Contract）**，与 `ARCHITECTURE.md`、`SDD.md`、`DEVELOPMENT_GUIDE.md` 同级，集中描述系统对外的接口事实基线（WebSocket 协议、EventTopic 发布-订阅主题、`push_display` 推送接口、`config.yaml` 配置字段）。
>
> 作用：给开发者与 AI 辅助编程工具一份"能调什么、协议/参数/返回值是什么"的权威清单，避免生成代码偏离真实接口。
>
> 本文所有字段名、事件名、payload 结构均**逐项对齐代码**，不引入任何推测性接口。

---

## 0. 文档关系与读者

| 文档 | 层级 | 职责 |
| --- | --- | --- |
| `CLAUDE.md` | 约束层 | 改动禁区、运行/重启约定 |
| `ARCHITECTURE.md` / `architecture.txt` | 规格/设计层 | 系统结构与数据流 |
| `SDD.md` | 规格/设计层 | 模块规格与设计决策 |
| `DEVELOPMENT_GUIDE.md` | 流程层 | 开发/调试/部署流程 |
| **`API.md`（本文件）** | **接口层** | **对外接口契约：协议、主题、推送、配置** |

**读者**：后端开发者、前端开发者、AI 编码助手。

**事实基线来源**：`web/ws_handler.py`、`frontend/src/composables/useWS.ts`、`core/event_bus.py`、`core/inference_stream.py`、`core/base_module.py`、`config.yaml`。

---

## 1. WebSocket 接入协议

前端通过 WebSocket 接收视频帧（二进制批量包）与结构化文本消息（评估报告、停止信号等）。

### 1.1 连接

| 项 | 值 |
| --- | --- |
| 路径（固定） | `/ws/data` |
| 协议 | `https:` 页面 → `wss:`；否则 `ws:` |
| 主机 | `window.location.host`（同源） |
| 二进制类型 | `arraybuffer` |

拼接逻辑（`frontend/src/composables/useWS.ts`）：

```ts
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
socket = new WebSocket(`${protocol}//${window.location.host}/ws/data`)
socket.binaryType = 'arraybuffer'
```

### 1.2 重连机制

**当前无自动重连逻辑**：`onclose` / `onerror` 仅将状态置 `idle` 并停止播放循环（`stopPlaybackLoop()`）。
重连依赖用户在 UI 上重新触发 `startPipeline()` → `connect()`。

> 红线提示：不要在客户端私自加"自动重连死循环"，避免与服务端 `_active_connections` 状态不一致。重连入口统一走 UI 的 `startPipeline()`。

### 1.3 二进制帧格式（视频批量包）

每个二进制包是**单条消息承载多视角**：视频帧（JPEG）+ JSON 元数据**原子到达**，便于前端按 `globalSec` 对齐两路画面。

字节布局（大端 / network byte order）：

```
+--------+-------------------+-----------+--------------------------------------------------+----------------------------------+
| version| global_sec        | viewCount | 每视角: [viewId | frameLen | JPEG字节] × viewCount    | metaJsonLen | meta JSON 字符串       |
| 1 byte | float32 (4 bytes) | 1 byte    | 1 byte + uint32 (4 bytes) + frameLen bytes        | uint32 (4 bytes) | UTF-8       |
+--------+-------------------+-----------+--------------------------------------------------+----------------------------------+
```

- `version`：协议版本号（当前 `_BINARY_VERSION = 1`），被消费但不参与解析逻辑。
- `global_sec`：`float32` 大端，全局时间轴秒数（服务端统一对齐）。
- `viewCount`：`uint8`，后续视角块数量。
- 每视角块：
  - `viewId`：`uint8`，视角标识（见 1.4 红线）。
  - `frameLen`：`uint32` 大端，JPEG 字节数。
  - `JPEG字节`：`frameLen` 字节的 JPEG 图像。
- 尾部：`metaJsonLen`（`uint32` 大端）+ UTF-8 JSON 字符串（结构化元数据）。

后端打包实现（`web/ws_handler.py`）：

```python
struct.pack("!BfB", _BINARY_VERSION, global_sec, len(frames))   # version, global_sec, viewCount
struct.pack("!BI", view_id, len(jpeg_data)) + jpeg_data          # 每视角
struct.pack("!I", len(meta_json_bytes)) + meta_json_bytes        # 尾部 meta
```

前端解包实现（`parseBinaryPacket`）：

```ts
const view = new DataView(buf)
let offset = 0
view.getUint8(offset); offset += 1               // version（消费后丢弃）
const globalSec = view.getFloat32(offset, false) // 大端 float32
offset += 4
const viewCount = view.getUint8(offset); offset += 1
for (let i = 0; i < viewCount; i++) {
  const viewId = view.getUint8(offset); offset += 1
  const frameLen = view.getUint32(offset, false); offset += 4
  const jpegBytes = new Uint8Array(buf, offset, frameLen)
  offset += frameLen
  frames[viewId] = URL.createObjectURL(new Blob([jpegBytes], { type: 'image/jpeg' }))
}
const jsonLen = view.getUint32(offset, false); offset += 4
const jsonStr = new TextDecoder('utf-8').decode(jsonBytes)
meta = JSON.parse(jsonStr)
```

### 1.4 视角映射红线（view_id）

| view_id | 常量 | 视角 | 视频源键 | 含义 |
| --- | --- | --- | --- | --- |
| `0` | `VIEW_ID_FRONT` | front | `video_front` | 前置广角视角 `camFRONT` |
| `1` | `VIEW_ID_POP` | pop | `video_pop` | 行为特写视角 `camPOP` |

后端常量（`web/ws_handler.py`）：

```python
VIEW_ID_FRONT = 0  # camFRONT 前置广角视角
VIEW_ID_POP = 1    # camPOP 行为特写视角
_VIEW_KEY_TO_ID = {"video_front": VIEW_ID_FRONT, "video_pop": VIEW_ID_POP}
```

前端解包后以数组键固化映射（`playOneBatch`）：

```ts
const f = batch.frames
if (f[0] !== undefined) { unifiedBatch.frontUrl = f[0]; unifiedBatch.hasFront = true }  // f[0] = front
if (f[1] !== undefined) { unifiedBatch.popUrl = f[1];  unifiedBatch.hasPop  = true }  // f[1] = pop
```

> ⚠️ **红线（与 `CLAUDE.md` §7 一致）**：`0=front`、`1=pop` 的映射**禁止新增或改动**。系统**仅有两个视角**，不存在第三路（无 `bup` / `V3` / `cambup` / 三路视频残留）。任何新增视角必须先在 `ws_handler.py` 与 `useWS.ts` 两端同步扩展，不得擅自加 `f[2]`。

### 1.5 文本消息分支（非 ArrayBuffer）

服务端通过 `ws_handler.push_text()` 推送纯 JSON 文本（评估报告等），前端按 `tag` 分流：

| `source` / `tag` | 前端处理 | 说明 |
| --- | --- | --- |
| `source === 'done'` 且 `tag === 'stop'` | 置 `status='done'`，停止播放循环 | 处理结束硬停止信号 |
| `source === 'done'`（其他） | 置 `eofReceived = true` | 正常 EOF |
| `tag === 'segment_report_stream'` | `handleDirectSegReportStream(d)` | 逐 token 流式评估报告直推（chunk 累积 streamBuffer，前端 typewriter 逐字） |
| `tag === 'segment_report'` | `handleDirectSegReport(d)` | 整段评估报告直推（不覆盖 streamBuffer；触发前端 typewriter 追完 → 切 streaming=false 完成态显分数） |

> 文本通道与二进制 batch **物理解耦**：文本消息不携带 `globalSec`。视频帧走二进制批量包，评估报告走文本 JSON。

### 1.6 前端 → 后端上报

前端每 ≥ 0.5s 向服务端上报播放进度：

```ts
socket.send(JSON.stringify({ type: 'playback_progress', current_sec: ... }))
```

---

## 2. EventTopic 发布-订阅主题清单

消息总线基于 **Redis Stream**（`core/event_bus.py`）。消息统一格式：

```json
{ "type": "<EventTopic>", "data": { ... }, "ts": <float> }
```

Stream key 规则：`module:events:<EventTopic>`。发布用 `xadd`，订阅用消费组 `XREADGROUP` 阻塞读取（每进程独立消费组实现跨进程广播）。

### 2.1 主题常量（`EventTopic`）

| 常量名 | 主题字符串 | 发布者 | 订阅者 | 触发条件 |
| --- | --- | --- | --- | --- |
| `VOICE_KEY_MOMENT` | `voice.key_moment` | Voice 模块 | Rules / Evaluation | 语音检测到关键语义节点 |
| `TRACKER_PROXIMITY` | `tracker.proximity` | Tracker 模块 | Rules | 人员靠近/离开监护设备 |
| `TRACKER_HEADCOUNT` | `tracker.headcount` | Tracker 模块 | Rules | 画面人数统计变化 |
| `BEHAVIOR_HAND_RAISED` | `behavior.hand_raised` | **Tracker（HandRaiser）** | **Behavior 模块** | 举手动作被检测 |
| `BEHAVIOR_FINGER_SCREEN` | `behavior.finger_screen` | Behavior 模块 | Rules / Evaluation | 手指指向屏幕判定命中 |
| `BEHAVIOR_FINGER_FILE` | `behavior.finger_file` | Behavior 模块 | Rules / Evaluation | 手指指向文件判定命中 |
| `GAZE_ATTENTION` | `gaze.attention` | Gaze 模块 | Rules | 注意力状态变化 |
| `GAZE_ALERT` | `gaze.alert` | Gaze 模块 | Rules | 注意力告警（如长时间离岗） |
| `FLOW_STARTED` | `flow.started` | Rules 模块 | 各模块 | 监护流程开始 |
| `FLOW_ENDED` | `flow.ended` | Rules 模块 | 各模块 | 监护流程结束 |
| `RULE_KEY_MOMENT` | `rule.key_moment` | Rules 模块 | Evaluation | 规则判定产生关键事件 |
| `SAVE_KEY_MOMENTS` | `save.key_moments` | Evaluation 模块 | 各模块 | 通知各模块立即保存 key_moments |

### 2.2 行为检测订阅关系（重点）

- Tracker 的 **HandRaiser** 发布 `behavior.hand_raised`；
- **Behavior 模块仅订阅**，不发布举手事件：

```python
self.event_bus.subscribe(EventTopic.BEHAVIOR_HAND_RAISED, self._on_hand_raised)
```

- Behavior 模块自身**输出**两个事件（见上表 `BEHAVIOR_FINGER_SCREEN` / `BEHAVIOR_FINGER_FILE`），由纯判定器（`FINGER_SCREEN` / `FINGER_FILE`）产生，经 `behavior_module` 的 `_EVENT_TOPIC_MAP` 映射后 `push_event()` 发布：

```python
_EVENT_TOPIC_MAP = {
    "FINGER_SCREEN": EventTopic.BEHAVIOR_FINGER_SCREEN,
    "FINGER_FILE":   EventTopic.BEHAVIOR_FINGER_FILE,
}
```

> 注意职责边界：**举手（hand_raised）由 Tracker 负责检测与发布，Behavior 只消费并据此做后续判定**，不反向发布举手事件。

### 2.3 总线 API 签名

```python
# 初始化
EventStream(redis_host="localhost", redis_port=6379, redis_db=0, max_workers=4)

# 发布
event_bus.publish(msg_type: str, data: dict, ts: float = 0.0) -> None
#   消息体: {"type": msg_type, "data": data, "ts": ts}
#   写入 Redis Stream: module:events:<msg_type>，maxlen=10000

# 订阅
event_bus.subscribe(msg_type: str, callback: Callable) -> None
#   callback 签名: def callback(msg: dict) -> None  （msg 为完整 {"type","data","ts"}）

# 取消订阅
event_bus.unsubscribe(msg_type: str, callback: Callable) -> None

# 生命周期
event_bus.start() -> None
event_bus.stop()  -> None
event_bus.get_stats() -> dict
```

投递语义：消费组读取 → **先执行全部回调成功后 `xack`**（避免"先 ack 后回调"在进程退出/回调失败时事件永久丢失）。`NOGROUP` 自动重建消费组。

---

## 3. `push_display` 推送接口

`push_display` 是各业务模块向**推理流（InferenceStream，Redis 承载）**推送结构化推理事件的统一入口，再由 `web/ws_handler` 取流后组装为 §1 二进制批量包推送给前端。

### 3.1 模块层入口（`core/base_module.py`）

```python
def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
    """推送数据到推理流（非即时类型自动登记为归属 source）."""
    if event_type not in ("progress", "video_start"):
        self._inference_sources.add(event_type)
    self.inference_stream.push_display(event_type, data)
```

- `event_type` 为 `"progress"` / `"video_start"` 时视为即时类型，不登记为归属 source。
- 其他类型会自动加入 `_inference_sources`（用于跨模块进度对齐限速）。

### 3.2 推理流实现（`core/inference_stream.py`）

```python
def push_display(self, event_type: str, data: Dict[str, Any]) -> None:
    ev = {"source": event_type, **data}
    # 写入 Redis Stream（_KEY_EVENT_STREAM），字段: local_sec / counter / payload
```

- 事件体：`{"source": event_type, **data}`。
- 必须包含 `localSec` 字段（时间对齐事件），否则记录错误日志并丢弃（除 `_immediate_types` 外）。
- `payload` 经 `json.dumps(ensure_ascii=False)` 写入。

### 3.3 常用调用示例（Behavior 模块）

| 调用 | `event_type` (source) | payload 字段 | 说明 |
| --- | --- | --- | --- |
| `push_display("progress", {...})` | `progress` | `localSec`(float, 2位)、`tag="progress"`、进度百分比 | 播放进度 |
| `push_display(video_source, {...})` | `video_front` / `video_pop` | `localSec`(float)、`tag="video"`、帧数据 | 视频帧（source 即视角键） |
| `push_display(tag, payload)` | 业务 tag | 业务字段 | 通用业务事件 |

> `video_source` 取值固定为 `"video_front"` / `"video_pop"`，与 §1.4 视角映射一致。

### 3.4 WebSocket 推送（`web/ws_handler.py`）

- `push(event: Optional[Dict]) -> None`：组装二进制批量包（§1.3）推送给 `_active_connections` 中**所有在线客户端**（通过 `asyncio.run_coroutine_threadsafe(connection.send_bytes(...))`）。`event=None` 时发 `{"source": "done"}` 文本哨兵。
- `push_text(event: Dict) -> None`：纯 JSON 文本，走 `send_text`，推给所有在线客户端（评估报告 `push_direct` 直推，**绕过 InferenceSync 对齐中间件**；见 §1.5）。评估的"时钟对齐"由后端 `wait_playback_reached` 阻塞等待前端播到流程结束实现，非 batch 对齐。

---

## 4. `config.yaml` 配置字段表

全局配置，覆盖 `config.py` 默认值。修改后需按 `CLAUDE.md` §8 重启相关进程。

### 4.1 顶层结构

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `app.gpu` | str | `"0"` | 推理使用的 GPU 设备号 |
| `app.fps` | float | `30.0` | 系统目标帧率 |
| `app.port` | int | `5002` | 后端 HTTP/WebSocket 服务端口 |
| `redis.host` | str | `"localhost"` | Redis 主机 |
| `redis.port` | int | `6379` | Redis 端口 |
| `redis.db` | int | `0` | Redis 数据库号 |
| `paths.data_root` | str | `"data"` | 数据根目录 |
| `paths.model_root` | str | `"models"` | 模型根目录 |
| `paths.result_root` | str | `"data/results"` | 结果输出目录 |

### 4.2 视频源（仅两路，红线）

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `videos.front` | str | `"data/videos/camFRONT.mpg"` | 前置广角视角源（view_id=0） |
| `videos.pop` | str | `"data/videos/camPOP.mpg"` | 行为特写视角源（view_id=1） |

> ⚠️ **仅 front / pop 两路**。无第三路（`bup` / `V3` / `cambup` / 三路视频均不存在）。新增视频源必须同步扩展 §1.4 视角映射与 `ws_handler._VIEW_KEY_TO_ID`。

### 4.3 监护与检测参数

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `supervision.bind_hold_sec` | float | `10.0` | 绑定保持时长（秒） |
| `supervision.unbind_hold_sec` | float | `10.0` | 解绑保持时长（秒） |
| `supervision.dist_close_px` | int | `200` | 近距判定像素阈值（close） |
| `supervision.dist_near_px` | int | `560` | 邻近判定像素阈值（near） |
| `supervision.consec_raise` | int | `3` | 连续举手帧数阈值 |
| `supervision.consec_idle` | int | `3` | 连续空闲帧数阈值 |
| `supervision.cooldown_sec` | float | `1.5` | 事件冷却时间（秒） |

### 4.4 Tracker 检测

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `tracker.detection.conf_threshold` | float | `0.65` | 目标检测置信阈值 |
| `tracker.detection.pose_confidence` | float | `0.35` | 姿态估计置信阈值 |
| `tracker.detection.nms_threshold` | float | `0.35` | NMS 非极大抑制阈值 |
| `tracker.detection.img_size` | int | `640` | 推理输入尺寸 |

### 4.5 总线与语音

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `bus.max_queue_size` | int | `1024` | 总线最大队列长度 |
| `voice.sample_rate` | int | `16000` | 音频采样率（Hz） |
| `voice.sentence_gap_sec` | float | `0.6` | 句间停顿阈值（秒） |
| `voice.asr_engine` | str | `"qwen3"` | ASR 引擎标识 |
| `voice.model_path` | str | `"models/voice/qwen/Qwen3-ASR-0.6B"` | ASR 模型路径 |
| `voice.aligner_path` | str | `"models/voice/qwen/Qwen3-ForcedAligner-0.6B"` | 强制对齐模型路径 |
| `voice.torch_dtype` | str | `"bfloat16"` | 模型权重 dtype |

### 4.6 Gaze（视线）

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `gaze.head_conf_th` | float | `0.55` | 头部检测置信阈值 |
| `gaze.inout_th` | float | `0.5` | 视线进出阈值 |
| `gaze.heatmap_th` | float | `0.3` | 注意力热力图阈值 |
| `gaze.head_min_size` | int | `20` | 头部最小尺寸（px） |
| `gaze.head_max_size` | int | `300` | 头部最大尺寸（px） |

### 4.7 Behavior（行为判定）

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `behavior.finger_screen.detect_conf` | float | `0.25` | 指屏检测置信阈值 |
| `behavior.finger_screen.screen_overlap_threshold` | float | `0.2` | 手指-屏幕重叠阈值 |
| `behavior.finger_screen.max_dist` | int | `10` | 最大距离（像素） |
| `behavior.finger_screen.cooldown_sec` | float | `1.5` | 冷却时间（秒） |
| `behavior.finger_file.detect_conf` | float | `0.25` | 指文件检测置信阈值 |
| `behavior.finger_file.file_iou_threshold` | float | `0.2` | 手指-文件 IoU 阈值 |
| `behavior.finger_file.cooldown_sec` | float | `1.5` | 冷却时间（秒） |

### 4.8 模块与规则开关

| 字段 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `modules.gaze` | bool | `true` | 启用 Gaze 模块 |
| `modules.voice` | bool | `true` | 启用 Voice 模块 |
| `modules.tracker` | bool | `true` | 启用 Tracker 模块 |
| `modules.personnel_status` | bool | `true` | 启用人员状态模块 |
| `modules.info_notice` | bool | `true` | 启用信息通知模块 |
| `modules.behavior` | bool | `true` | 启用 Behavior 模块 |
| `rules.supervision` | bool | `true` | 启用监护规则 |
| `rules.self_ticket` | bool | `true` | 启用自检票规则 |
| `rules.personnel_status` | bool | `true` | 启用人员状态规则 |
| `rules.info_notice` | bool | `true` | 启用信息通知规则 |

---

## 5. 版本与变更约定

- **二进制协议版本**：由 `web/ws_handler.py` 的 `_BINARY_VERSION`（当前 `1`）控制。变更帧布局时必须递增该版本，并同步更新前端 `parseBinaryPacket`。
- **EventTopic 新增/改名**：属于破坏性变更，须同时更新：
  1. `core/event_bus.py` 的 `EventTopic` 常量；
  2. 所有 `publish` / `subscribe` 调用方（参见 §2.1 表）；
  3. 本文件 §2 主题表。
- **config.yaml 字段变更**：新增/改名配置项须同步更新 `config.py` 默认值与本文件 §4 字段表；删除字段需确认无运行时代码引用。
- **接口变更记录**：任何对外接口（WebSocket 协议、EventTopic、push_display、config）修改，均需在 `CLAUDE.md` 与本文对应章节同步更新，禁止出现"代码改了、文档没动"的漂移。
- **红线保护**：`view_id` 0=front / 1=pop 映射、仅两路视频、Behavior 只订阅 `behavior.hand_raised` 不发布——这些约束在 `CLAUDE.md` §7 列为改动禁区，本文件与之事实同源。

---

## 6. 接口事实速查（防误用清单）

- ✅ WebSocket 路径固定 `/ws/data`，无自动重连（走 UI `startPipeline()`）。
- ✅ 单条二进制包 = 多视角（front+pop）视频帧 + JSON meta，按 `globalSec` 对齐。
- ✅ `version(1B) | global_sec(float32 BE) | viewCount(1B) | [viewId(1B)|frameLen(u32 BE)|JPEG]×N | metaLen(u32 BE)|meta(JSON)`。
- ✅ view_id 红线：`0=front`（`video_front`）、`1=pop`（`video_pop`），**仅两路**。
- ✅ EventTopic 为 Redis Stream key，格式 `{"type","data","ts"}`。
- ✅ `behavior.hand_raised` 由 **Tracker** 发布、**Behavior** 仅订阅。
- ✅ `push_display(event_type, data)` → 推理流 Redis；`ws_handler.push/push_text` → 客户端。
- ✅ config 仅 `videos.front` / `videos.pop` 两路，无 bup/V3/三路残留。
