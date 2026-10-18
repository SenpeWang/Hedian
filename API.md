# 接口文档

> 本文件描述系统各层之间的**接口契约**，供协作开发遵循。
> 改动任何接口（路由 / 字段名 / 消息类型 / Redis 键）**必须同步更新本文件**。

系统分四层，数据自左向右流动：

```
前端 ──HTTP / WebSocket──► web 服务层 ──EventTopic──► 规则层 ──► 评估层
                               │
                               └──Redis──► 感知模块（各自独立子进程）
```

---

## 一、HTTP 接口

基址 `http://<host>:<port>`。端口取自 `config.yaml` 的 `app.port`（默认 `5002`），
可用 `main.py --port` 覆盖。

| 方法 | 路径 | 请求 | 返回 | 说明 |
|---|---|---|---|---|
| POST | `/start` | 无 body | `{"status":"started"}` 或 `{"status":"already_running"}` | 启动推理；已在 running 时**拒绝** |
| POST | `/stop` | 无 | `{"status":"stopped","terminated_processes":N}` | 停止推理子进程 |
| POST | `/reset` | 无 | `{"status":"reset","terminated_processes":N}` | 停止 + 清空全部缓存 |
| GET | `/status` | 无 | `{"pipeline":"idle"\|"running","ws_clients":N}` | 流水线状态 |
| GET | `/api/config` | 无 | 配置字典 | |
| GET | `/api/modules` | 无 | `{"modules":{...}}` | 各模块开关 |
| GET | `/api/video/front` | 无 | 视频文件流，或 404 JSON | front 视角源视频 |
| GET | `/api/video/pop` | 无 | 视频文件流，或 404 JSON | pop 视角源视频 |
| WS | `/ws/data` | 见第二节 | 见第二节 | 双工：下行视频 + 数据，上行播放进度 |

### 1.1 三个控制接口的副作用差异（**易踩坑**）

| | `/start` | `/stop` | `/reset` |
|---|:---:|:---:|:---:|
| 终止推理子进程 | — | ✅ | ✅ |
| 清 Redis | **`flushdb()` 清空整库** | 按前缀删 `inference:*` / `module:*` / `pipeline:*` | **`flushdb()` 清空整库** |
| 清可视化缓存 | ✅ | — | ✅ |
| 发布 `pipeline.start` | ✅ | — | — |
| 写 `pipeline:start_signal`（`ex=3600`） | ✅ | — | — |
| WebSocket 通知前端 | — | 推 `{source:"done",tag:"stop",data:{reason}}` | **无** |
| 状态置为 | `running` | `idle` | `idle` |

> ⚠️ **`/start` 与 `/reset` 会 `flushdb`**。同机多实例部署时**必须为每个实例分配独立
> Redis 库**（`config.yaml` 的 `redis.db`，或 `main.py --redis-db`），否则会互相清库、
> 抢消费组消息。

---

## 二、WebSocket 协议

### 2.1 路径

```
/ws/data
```

### 2.2 下行 —— 二进制视频帧

```
┌──────────┬────────┬──────────────────┐
│ 1 字节   │ 1 字节 │  N 字节          │
│ channel  │  type  │  fMP4 分片       │
└──────────┴────────┴──────────────────┘
```

| `channel` | 视角 |
|---|---|
| `0` | front（正面） |
| `1` | pop（操作盘） |

| `type` | 段类型 |
|---|---|
| `0` | init（`ftyp` + `moov`，首段） |
| `1` | media（`moof` + `mdat`） |
| `2` | end |

前端用 MSE（Media Source Extensions）拼接播放。帧长不足 2 字节或 channel/type 非法时，
前端解析返回 `null` 并丢弃。

### 2.3 下行 —— JSON 文本

按 `source` 字段区分：

| `source` | 时机 | 载荷字段 |
|---|---|---|
| `status` | 连接建立后补发快照 | `totalDuration` · `globalSec` · `status` |
| `done` | 流水线结束 / 用户停止 | `tag` · `data.reason` |
| `segment_report_stream` | 评估流式输出 | `flow_id` · `chunk` |
| `segment_report` | 评估终态 | `flow_id` · `flow_type` · `score` · `report_text` · `flow_continue_sec` |
| （无 `source`，即 **batch**） | 推理过程中持续推送 | 见下 |

#### batch —— 结构化数据批次

```jsonc
{
  "globalSec": 12.5,          // 推理侧全局时钟（各 source 进度的 min）
  "totalDuration": 644.2,     // 视频总时长（秒）
  "sourceTimes": {            // 各视角整体进度，供前端速率引擎取 min 决定播放倍率
    "front": 13.1, "pop": 12.8, "voice": 95.0
  },

  "flow_start": [ { "localSec": 95.5, "tag": "flow_start",
      "data": { "flow_id": 1, "flow_type": "supervision", "flow_start_sec": 95.5 } } ],

  "flow_end":   [ { "localSec": 130.2, "tag": "flow_end",
      "data": { "flow_id": 1, "flow_type": "supervision",
                "flow_end_sec": 130.2, "flow_continue_sec": 34.7 } } ],

  "voice":     [ { "localSec": 95.5, "tag": "text",
      "data": { "text": "一回路请讲", "keys": ["一回路", "请讲"] } } ],

  "tracking":  [ { "localSec": 60.0, "tag": "PEOPLE_COUNT_UPDATE",
      "data": { "count": 3, "state_alert": "" } } ],

  "gaze":      [ { "localSec": 60.0,
      "data": { "has_heads": true, "heads_count": 3,
                "any_in_roi": true, "away_duration": 0.0 } } ],

  "behavior":  [ { "localSec": 113.4, "tag": "FINGER_FILE", "data": { "iou": 0.31 } } ]
}
```

**各数组的消费方**（前端 `App.vue` 的路由）：

| 数组 | 去向 | 同步语义 |
|---|---|---|
| `flow_start` / `flow_end` | 系统通知栏 —— 事件流 | **跟画面**（`sec ≤ 播放时刻`） |
| `behavior` | 系统通知栏 —— 事件流（与流程事件合并为一条时间线） | **跟画面** |
| `tracking` / `gaze` | 系统通知栏 —— 状态量 | **实时**（收到即显示） |
| `voice` | 实时字幕面板 | 跟画面 |
| （报告两条） | 流程评价面板 | 打字机逐字揭示 |

> ⚠️ **两类语义必须区分**：**状态量**（人数 / 凝视）实时显示，监控者要立刻看到异常；
> **事件流**（流程 / 行为）严格跟画面，保证标记与画面对齐、便于回看定位。

### 2.4 上行

仅一种消息：

```json
{ "type": "playback_progress", "current_sec": 12.5 }
```

除多端同步外，它还参与**报告推送时机判定** —— 评估器会等前端播放到流程结束时刻才推报告。

---

## 三、Redis 结构

Redis 是跨进程通信骨干。**模块之间只经事件主题耦合，不直接互相调用**。

| key | 类型 | 字段 / 格式 | 用途 |
|---|---|---|---|
| `inference:progress` | Hash | field = `大类.细粒度`（如 `tracker.tracking`）<br>value = `str(local_sec)` | 各 source 推理进度 |
| `inference:snapshot` | Hash | field = `module_name`<br>value = JSON | 模块最新状态，供前端刷新补发 |
| `inference:results:all` | Stream | `local_sec` · `counter` · `payload`（JSON，含 `source`） | 结构化事件流 |
| `inference:source_done` | Hash | field = `大类.细粒度`<br>value = 最终 `sec` | 各 source 结束信号 |
| `inference:vis_stream:{front\|pop}` | Stream | `type` ∈ `init`/`media`/`end`，`data`(bytes) | fMP4 视频分片 |
| `pipeline:start_signal` | String | 时间戳，`ex=3600` | 启动信号 |
| `module:events:{topic}` | Stream | `payload`（JSON `{type,data,ts}`） | **事件总线**，见第四节 |

---

## 四、事件总线（跨进程）

主题常量定义于 `core/event_bus.py` 的 `EventTopic`。
**主题值格式：`<发布者>.<语义>`（小写点分）**。

| 常量 | 值 | 发布方 | 订阅方 |
|---|---|---|---|
| `VOICE_KEY_MOMENT` | `voice.key_moment` | voice | supervision · self_ticket · normal_comm · info_notice |
| `TRACKER_PROXIMITY` | `tracker.proximity` | tracker | supervision |
| `TRACKER_HEADCOUNT` | `tracker.headcount` | tracker | personnel_status |
| `BEHAVIOR_HAND_RAISED` | `behavior.hand_raised` | behavior | supervision · info_notice |
| `BEHAVIOR_FINGER_SCREEN` | `behavior.finger_screen` | behavior | supervision |
| `BEHAVIOR_FINGER_FILE` | `behavior.finger_file` | behavior | supervision |
| `GAZE_ALERT` | `gaze.alert` | gaze | personnel_status |
| `FLOW_STARTED` | `flow.started` | 各规则状态机 | inference_sync · flow_recorder · 评估器 · tracker · gaze |
| `FLOW_ENDED` | `flow.ended` | 各规则状态机 | 同上 |
| `RULE_KEY_MOMENT` | `rule.key_moment` | 规则层 | tracker |
| `SAVE_KEY_MOMENTS` | `save.key_moments` | 评估器 | 各模块（经 `BaseModule` 统一订阅） |
| （非 `EventTopic`） | `pipeline.start` | http_server | main |

**新增一类流程的接入成本**：只需新增一个规则文件 + 在 `config.yaml` 的 `rules:` 加一个开关
—— `RuleRegistry.discover()` 用 `pkgutil` 动态扫描，**无需改动 core 基础设施**。

---

## 五、前端类型契约

业务模型见 `frontend/src/types.ts`；帧解析与消息判定见 `frontend/src/media/protocol.ts`。

| 类型 | 关键字段 |
|---|---|
| `FlowType` | `'supervision' \| 'self_ticket' \| 'info_notice' \| 'normal_comm'` |
| `FlowEvent` | `sec` · `flowType` · `name` · `color` · `isStart` · `flowId?` |
| `BehaviorEvent` | `sec` · `tag` · `name` · `color` |
| `NotifyEvent` | `sec` · `text` · `color` · `key` —— 流程与行为合并后的统一渲染形状 |
| `VoiceEntry` | `sec` · `text` · `keys` |
| `SegCard` | `flowId` · `flowType` · `score` · `reportText` · `continueSec` · `collapsed` · `streamBuffer` · `streaming` |
| `GazeState` | `hasHeads` · `headsCount` · `anyInRoi` · `awayDuration` |
| `PeopleState` | `count` · `alert` · `alertColor` |

**消息判定**：

| 函数 | 判据 |
|---|---|
| `isStreamMsg` | `source === 'segment_report_stream'` 或 `type === 'report_stream'` 或 `source === 'stream'` |
| `isTerminalMsg` | `source === 'segment_report'` 或 `type === 'report'` 或 `source === 'report'` |

**字段命名对照**（跨语言约定，非命名不一致）：

| 后端（snake_case） | 前端（camelCase） |
|---|---|
| `flow_type` | `flowType` |
| `flow_id` | `flowId` |
| `heads_count` | `headsCount` |
| `away_duration` | `awayDuration` |

> ⚠️ **协议字段一律不得改名**，包括 `localSec` · `flow_type` · `totalDuration` ·
> `"box"` · `"duration"` 这类**字符串字典键** —— 改名即断链，且不会有编译期报错。
