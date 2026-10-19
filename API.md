# 接口文档

> **文档版本** 1.2.0 ｜ **适用代码** `main` @ `bb12580` ｜ **最后更新** 2026-10-18

本文档描述核电站监护制合规检测系统**各层之间的接口契约**，是前后端、
模块间协作的基准。**任何接口改动必须同步更新本文档**（见 §1.4）。

---

## 目录

- [一、阅读指南](#一阅读指南)
- [二、系统概览](#二系统概览)
- [三、通用约定](#三通用约定)
- [四、HTTP 接口](#四http-接口)
- [五、WebSocket 协议](#五websocket-协议)
- [六、Redis 结构](#六redis-结构)
- [七、事件总线](#七事件总线)
- [八、前端类型契约](#八前端类型契约)
- [九、变更记录](#九变更记录)
- [附录 A：错误码](#附录-a错误码)
- [附录 B：术语表](#附录-b术语表)

---

## 一、阅读指南

### 1.1 适用范围

| 读者 | 关注章节 |
|---|---|
| 前端开发 | §3 通用约定 · §4 HTTP · §5 WebSocket · §8 前端契约 |
| 感知模块开发 | §3 · §6 Redis · §7 事件总线 |
| 规则 / 评估层开发 | §7 事件总线 · §8 |
| 部署运维 | §2.2 部署形态 · §2.3 安全边界 · §4.2 控制接口副作用 |

### 1.2 版本与兼容性策略

本文档版本与系统版本同步（当前 **1.2.0**，对应
[Releases](https://github.com/SenpeWang/Hedian/releases)）。

**破坏性变更**（需升 MINOR 或 MAJOR，且必须走 PR 评审）：

- 删除或重命名已存在的路由、Redis 键、事件主题
- 修改已存在字段的**名称**或**类型**
- 修改已存在字段的**语义**（如把「秒」改成「毫秒」）
- 收紧取值范围导致既有调用方可能越界

**兼容性变更**（可直接合入）：

- 新增**可选**字段
- 新增路由 / 新的事件主题 / 新增消息类型
- 放宽取值范围

> ⚠️ **协议字段一旦发布即冻结**。字符串字典键（如 `"box"`、`"duration"`、
> `"start_time"`）与前端字段名（`localSec`、`totalDuration`）均属协议 ——
> 改名不会触发任何编译期报错，只会在运行时静默断链。

### 1.3 时间基准（**最易混淆，务必先读**）

系统内同时存在**三种时间**，接口字段严格区分：

| 时间 | 含义 | 出现位置 |
|---|---|---|
| **源视频时间轴** | 视频内的秒数（播放轴） | `localSec` · `flow_start_sec` · `sec` |
| **推理侧全局时钟** | 各未结束 source 进度的 `min` | `globalSec` |
| **墙钟** | 真实世界时间戳（`time.time()`） | 仅内部限流用，**不出现在任何接口字段中** |

### 1.4 变更管理

1. 改动接口 → **同一 PR 内更新本文档**
2. 破坏性变更需在 PR 描述中列明影响面与迁移方式
3. 文档与代码不一致时，**以代码为准**并立即修文档
4. 提交信息 scope 取 `web` / `frontend` / `core` 等模块名（见 README 贡献指南）

---

## 二、系统概览

### 2.1 分层与数据流

```
┌────────┐  HTTP / WebSocket   ┌──────────────┐   EventTopic   ┌────────┐   ┌──────────┐
│  前端  │ ◄─────────────────► │  web 服务层  │ ◄────────────► │ 规则层 │──►│  评估层  │
└────────┘                     └──────┬───────┘                └────────┘   └──────────┘
                                      │ Redis
                                      ▼
                        ┌─────────────────────────────┐
                        │ 感知模块（各自独立子进程）   │
                        │ tracker · gaze · voice · behavior │
                        └─────────────────────────────┘
```

**设计要点**：感知模块之间**只经事件主题耦合，互不感知彼此实现** ——
新增一类规程只需新增一个规则文件 + 一个配置开关，无需改动 core 基础设施。

### 2.2 部署形态

| 项 | 说明 |
|---|---|
| 监听 | `0.0.0.0:<port>` |
| 端口来源 | `config.yaml` 的 `app.port`（默认 `5002`），可被 `main.py --port` 覆盖 |
| Redis | `config.yaml` 的 `redis.*`，可被 `main.py --redis-db` 覆盖 |
| 前端产物 | 后端**静态托管** `frontend/dist/`（SPA，`html=True` 兜底），无需独立 Nginx |
| 多实例 | **必须为每个实例分配独立 Redis 库** —— 见 §4.2.4 |

### 2.3 安全边界

> ⚠️ **本服务不含认证与鉴权机制**，设计前提是**部署于受控内网**。

| 项 | 现状 |
|---|---|
| 认证 | 无（无 token / session） |
| CORS | 未开启跨域（前端由同源静态托管提供） |
| 传输加密 | 无（HTTP / WS 明文） |
| 访问控制 | 依赖网络层（内网隔离 / 防火墙） |

**部署要求**：不得直接暴露于公网。如需外网访问，必须在反向代理层补齐
TLS、认证与访问控制。

---

## 三、通用约定

### 3.1 传输与编码

| 项 | 约定 |
|---|---|
| HTTP 方法 | 仅 `GET` / `POST` |
| 请求体 | 控制类接口**不接受**请求体 |
| 响应编码 | JSON，`UTF-8` |
| 内容类型 | `application/json`；视频资源为 `video/mp4` |
| 静态资源 | 无缓存（`NoCacheStaticFiles`），避免调试期拿到旧产物 |

### 3.2 错误响应规范

统一格式：

```json
{ "error": "<人类可读的错误描述>" }
```

| HTTP 状态 | 触发条件 | 示例 |
|---|---|---|
| `200` | 成功 | — |
| `404` | 请求的资源不存在 | `{"error":"front video not found"}` |
| `422` | 请求格式不合法（FastAPI 内建） | — |

> ⚠️ **注意**：控制类接口的**业务状态不以 HTTP 状态码表达**。例如 `/start`
> 在流水线已运行时返回 **`200`** + `{"status":"already_running"}` ——
> **调用方必须检查响应体，不能只看状态码**（见 §4.2.1）。

### 3.3 幂等性

| 接口 | 幂等性 |
|---|---|
| `POST /start` | **幂等** —— 已在运行时拒绝重复启动，不产生副作用 |
| `POST /stop` | **幂等** —— 未运行时返回 `terminated_processes: 0` |
| `POST /reset` | **幂等** —— 重复调用结果一致 |
| `GET` 类 | 天然幂等 |

### 3.4 超时与重试建议

| 场景 | 建议 |
|---|---|
| 控制类接口（`/start` `/stop` `/reset`） | 超时 ≥ 10s（需终止子进程）；**不要自动重试** |
| 查询类接口 | 超时 5s，可重试 |
| 视频资源 | 支持 HTTP Range（206），播放器自行分段拉取 |
| WebSocket | 长连接，前端需实现重连；重连后由服务端**补发状态快照**恢复（见 §5.1） |

---

## 四、HTTP 接口

### 4.1 接口总览

| # | 方法 | 路径 | 用途 | 幂等 |
|---|---|---|---|---|
| 1 | POST | `/start` | 启动流水线 | ✅ |
| 2 | POST | `/stop` | 停止流水线 | ✅ |
| 3 | POST | `/reset` | 停止并清空全部缓存 | ✅ |
| 4 | GET | `/status` | 查询流水线状态 | ✅ |
| 5 | GET | `/api/config` | 读取运行配置 | ✅ |
| 6 | GET | `/api/modules` | 读取模块开关 | ✅ |
| 7 | GET | `/api/video/front` | 拉取 front 视角源视频 | ✅ |
| 8 | GET | `/api/video/pop` | 拉取 pop 视角源视频 | ✅ |
| 9 | WS | `/ws/data` | 双工数据通道 | — |

### 4.2 控制类接口

#### 4.2.1 `POST /start`

启动推理流水线。

**请求**：无参数、无请求体。

**响应**：

| `status` | 含义 | HTTP |
|---|---|---|
| `started` | 本次成功启动 | `200` |
| `already_running` | 流水线已在运行，**本次调用未生效** | `200` |

```json
{ "status": "started" }
```

**成功启动时的副作用**（顺序执行）：

1. `flushdb()` —— **清空整个 Redis 库**
2. 重置 WebSocket 层状态
3. 写 `pipeline:start_signal`（`ex=3600`）
4. 生成 `run_id`（`YYYYmmdd_HHMMSS`）并写入配置
5. 重置推理同步器
6. 发布事件 `pipeline.start`（载荷含 `run_id`）
7. 状态置为 `running`

```bash
curl -X POST http://<host>:<port>/start
```

#### 4.2.2 `POST /stop`

终止推理子进程。**不**清空 Redis 整库，仅按前缀删除推理相关键。

**响应**：

```json
{ "status": "stopped", "terminated_processes": 2 }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | `string` | 恒为 `"stopped"` |
| `terminated_processes` | `integer` | 实际终止的子进程数；未运行时为 `0` |

**副作用**：终止子进程 → 删除 `inference:*` / `module:*` / `pipeline:*` →
重置同步器 → 状态置 `idle` → **通过 WebSocket 推送**
`{"source":"done","tag":"stop","data":{"reason":"user_stopped"}}`

#### 4.2.3 `POST /reset`

停止流水线并清空**全部**缓存（含可视化缓存）。用于页面刷新后从干净状态开始。

**响应**：

```json
{ "status": "reset", "terminated_processes": 0 }
```

**副作用**：终止子进程 → `flushdb()` → 重置同步器 → 清空可视化 init 缓存 →
状态置 `idle` → **不推送任何 WebSocket 消息**（与 `/stop` 的关键区别）

#### 4.2.4 三者差异对照（**部署前必读**）

| | `/start` | `/stop` | `/reset` |
|---|:---:|:---:|:---:|
| 终止子进程 | — | ✅ | ✅ |
| 清 Redis | **`flushdb()` 清整库** | 按前缀删 | **`flushdb()` 清整库** |
| 清可视化缓存 | ✅ | — | ✅ |
| 发布 `pipeline.start` | ✅ | — | — |
| 写 `pipeline:start_signal` | ✅ | — | — |
| WebSocket 通知 | — | ✅ 推 `done` | **无** |
| 状态置为 | `running` | `idle` | `idle` |

> ⚠️ **`/start` 与 `/reset` 会清空整个 Redis 库。同机部署多实例时，
> 必须为每个实例分配独立库**（`config.yaml` 的 `redis.db`，或
> `main.py --redis-db N`），否则实例间会互相清库、互抢消费组消息。

### 4.3 查询类接口

#### 4.3.1 `GET /status`

```json
{ "pipeline": "idle", "ws_clients": 1 }
```

| 字段 | 类型 | 取值 | 说明 |
|---|---|---|---|
| `pipeline` | `string` | `idle` \| `running` | 流水线状态 |
| `ws_clients` | `integer` | ≥ 0 | 当前在线 WebSocket 客户端数 |

#### 4.3.2 `GET /api/config`

返回完整运行配置（即 `config.yaml` 与代码默认值合并后的结果）。

#### 4.3.3 `GET /api/modules`

```json
{ "modules": { "voice": true, "tracker": true, "gaze": true,
               "behavior": true, "evaluation": true,
               "personnel_status": true, "info_notice": true } }
```

> `evaluation: false` 时 web 进程不创建评估编排器，流程结束后不产出评估报告；
> 其余模块为 `false` 时对应推理子进程不启动，且该模块的 source 自动退出
> `expected_sources` 对齐集合，不影响流水线 `done` 判定。

### 4.4 资源类接口

#### 4.4.1 `GET /api/video/front` ｜ `GET /api/video/pop`

流式返回视角源视频，**支持 HTTP Range 请求（206）**，供播放器分段拉取与硬解，
并保留原生音轨。

**文件解析顺序**：`data/videos/<front|pop>.mp4` → `data/videos/<front|pop>.mpg`

| 情况 | 响应 |
|---|---|
| 命中 | `200` / `206`，`Content-Type: video/mp4`，文件流 |
| 未命中 | `404` `{"error":"front video not found"}` |

### 4.5 静态资源兜底

`/` 挂载 `frontend/dist/`（SPA，`html=True`），未匹配的路径回落到静态资源。
**前端产物不入版本库**，部署前需在 `frontend/` 执行 `npm run build`。

---

## 五、WebSocket 协议

### 5.1 连接与生命周期

| 项 | 约定 |
|---|---|
| 路径 | `/ws/data` |
| 建连后 | 服务端**主动补发状态快照**（`source: "status"`），前端据此恢复进度 |
| 心跳 | 前端定期上行播放进度即视为存活 |
| 断开 | 服务端注销连接，`ws_clients` 减一 |
| 重连 | 前端负责重连；重连后仍由快照恢复，无需重放历史消息 |
| 页面刷新 | 前端在加载时调用 `/reset` 并重置本地 store，保证从干净状态开始 |

### 5.2 下行 —— 二进制视频帧

```
┌──────────┬────────┬────────────────────┐
│ 1 字节   │ 1 字节 │  N 字节            │
│ channel  │  type  │  fMP4 分片         │
└──────────┴────────┴────────────────────┘
```

| `channel` | 视角 |
|---|---|
| `0` | front（正面） |
| `1` | pop（操作盘） |

| `type` | 段类型 | 内容 |
|---|---|---|
| `0` | `init` | `ftyp` + `moov`（首段，必须最先喂给 MSE） |
| `1` | `media` | `moof` + `mdat` |
| `2` | `end` | 结束标记 |

**前端处理**：用 MSE（Media Source Extensions）拼接。帧长不足 2 字节、
或 channel / type 非法时，解析返回 `null` 并**丢弃该帧**（不中断连接）。

### 5.3 下行 —— JSON 文本

按 `source` 字段区分类型：

| `source` | 时机 | 载荷 |
|---|---|---|
| `status` | 建连后补发快照 | `totalDuration` · `globalSec` · `status` |
| `done` | 流水线结束或用户停止 | `tag` · `data.reason` |
| `segment_report_stream` | 评估流式输出 | `flow_id` · `chunk` |
| `segment_report` | 评估终态 | `flow_id` · `flow_type` · `score` · `report_text` · `flow_continue_sec` |
| *（无 `source`）* | 推理过程中持续推送 | **batch**，见下 |

#### 5.3.1 batch —— 结构化数据批次

```jsonc
{
  "globalSec": 12.5,          // number  推理侧全局时钟（各 source 进度 min）
  "totalDuration": 644.2,     // number  视频总时长（秒）
  "sourceTimes": {            // object  各视角整体进度，供前端速率引擎取 min
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

  "behavior":  [ { "localSec": 113.4, "tag": "FINGER_FILE",
      "data": { "role": "操作员", "contain_ratio": 0.88 } },
                 { "localSec": 141.3, "tag": "FINGER_SCREEN",
      "data": { "role": "机组长", "overlap_ratio": 0.0, "min_dist": 7.9 } } ]
}
```

**各数组的消费方与同步语义**（前端 `App.vue` 路由）：

| 数组 | 去向 | 同步语义 | 说明 |
|---|---|---|---|
| `flow_start` / `flow_end` | 系统通知栏 · 事件流 | **跟画面**（`sec ≤ 播放时刻`） | 流程标记须与画面对齐，便于回看定位 |
| `behavior` | 系统通知栏 · 事件流 | **跟画面** | 与流程事件合并为一条按时刻排序的时间线 |
| `tracking` / `gaze` | 系统通知栏 · **状态量** | **实时**（收到即显示） | 监控者需立刻看到异常，不被播放进度卡住 |
| `voice` | 实时字幕面板 | 跟画面 | 按秒归并，命中关键词高亮 |
| *（报告两条）* | 流程评价面板 | 打字机逐字揭示 | 见 §5.3.4 |

> ⚠️ **两类语义必须区分**：**状态量**实时；**事件流**跟画面。
> 混用会导致「监控数值滞后」或「流程标记与画面错位」。

**`behavior` 的 `tag` 取值**：

| `tag` | 含义 |
|---|---|
| `HAND_RAISED` | 举手 |
| `FINGER_SCREEN` | 手指指向屏幕 |
| `FINGER_FILE` | 手指指向文件 |

**`behavior` 各 `tag` 的 `data` 字段**：

| 字段 | 出现于 | 含义 |
|---|---|---|
| `role` | `FINGER_SCREEN` / `FINGER_FILE` | 行为人角色：`机组长` / `操作员` / `机组长和操作员`；无法判定时为 `null` |
| `overlap_ratio` | `FINGER_SCREEN` | 指向手框与机箱区域的最大面积重叠比（0–1） |
| `min_dist` | `FINGER_SCREEN` | 指向手框到机箱多边形的最短距离（像素） |
| `contain_ratio` | `FINGER_FILE` | 指向手框被文件框包含的面积比（0–1） |
| `state` / `frame_id` | 全部 | 事件状态文案与触发帧号（`HAND_RAISED` 另有 `operator` / `track_id`） |

**`flow_type` 取值**：`supervision`（监护制）· `self_ticket`（自唱票）·
`info_notice`（信息通报）· `normal_comm`（普通沟通）

#### 5.3.2 报告推送时机（**重要设计**）

评估报告**不是**流程一结束就推送，而是**等前端播放进度到达该流程的结束时刻**才推：

- 每个 token chunk 推送前等待 `播放进度 ≥ 流程结束时刻`
- **超时 60s 强制放行**（避免前端未打开时评估被永久阻塞）
- 因此报告卡片出现的位置，正好是画面走到流程结束那一刻

### 5.4 上行

仅一种消息：

```json
{ "type": "playback_progress", "current_sec": 12.5 }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | `string` | 恒为 `"playback_progress"` |
| `current_sec` | `number` | 前端当前播放位置（源视频秒） |

**用途**：① 多端同步；② **报告推送时机判定**（见 §5.3.2）。

### 5.5 典型交互时序

```
前端                      web 服务端                     感知模块
 │                            │                              │
 │── WS 连接 /ws/data ───────►│                              │
 │◄─ 快照 {source:"status"} ──│                              │
 │                            │                              │
 │── POST /start ────────────►│ flushdb / 发布 pipeline.start│
 │                            │─────────────────────────────►│ 启动子进程
 │◄─ 二进制帧 front/pop ──────│◄───── fMP4 分片 ─────────────│
 │◄─ batch {flow_start,...} ──│◄───── 事件流 ────────────────│
 │                            │                              │
 │── 上行 playback_progress ─►│ （用于报告推送时机判定）      │
 │                            │                              │
 │◄─ segment_report_stream ───│◄───── 评估流式产出 ──────────│
 │◄─ segment_report ──────────│       （终态）                │
```

---

## 六、Redis 结构

Redis 是跨进程通信骨干。**模块间只经事件主题耦合，不直接互相调用**。

| key | 类型 | 字段 / 格式 | 用途 | TTL |
|---|---|---|---|---|
| `inference:progress` | Hash | field = `大类.细粒度`（如 `tracker.tracking`）<br>value = `str(local_sec)` | 各 source 推理进度 | 无 |
| `inference:snapshot` | Hash | field = `module_name`<br>value = JSON | 模块最新状态，供前端刷新补发 | 无 |
| `inference:results:all` | Stream | `local_sec` · `counter` · `payload`（JSON，含 `source`） | 结构化事件流 | 无 |
| `inference:source_done` | Hash | field = `大类.细粒度`<br>value = 最终 `sec` | 各 source 结束信号 | 无 |
| `inference:vis_stream:{front\|pop}` | Stream | `type` ∈ `init`/`media`/`end` · `data`(bytes) | fMP4 视频分片 | 无 |
| `pipeline:start_signal` | String | 时间戳 | 启动信号 | **3600s** |
| `module:events:{topic}` | Stream | `payload`（JSON `{type,data,ts}`） | 事件总线（见 §7） | 无 |

**细粒度命名**：`inference:progress` 的 field 采用 `大类.细粒度`（如
`tracker.tracking`、`behavior.video_pop`、`gaze`、`voice`），便于按模块聚合查询。

---

## 七、事件总线

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
| *（非 `EventTopic`）* | `pipeline.start` | http_server | main |

**消息格式**（写在 `module:events:{topic}` Stream）：

```json
{ "type": "<topic>", "data": { }, "ts": 0.0 }
```

**扩展方式**：新增感知模块或规则，只需定义新的 `EventTopic` 常量并
publish / subscribe —— 发布方与订阅方互不感知彼此实现。

---

## 八、前端类型契约

业务模型见 `frontend/src/types.ts`；帧解析与消息判定见 `frontend/src/media/protocol.ts`。

| 类型 | 字段 | 类型 |
|---|---|---|
| `FlowType` | — | `'supervision' \| 'self_ticket' \| 'info_notice' \| 'normal_comm'` |
| `FlowEvent` | `sec` · `flowType` · `name` · `color` · `isStart` · `flowId?` | `number` · `FlowType` · `string` · `string` · `boolean` · `number?` |
| `BehaviorEvent` | `sec` · `tag` · `name` · `color` | `number` · `string` · `string` · `string` |
| `NotifyEvent` | `sec` · `text` · `color` · `key` | `number` · `string` · `string` · `string` |
| `VoiceEntry` | `sec` · `text` · `keys` | `number` · `string` · `string[]` |
| `SegCard` | `flowId` · `flowType` · `score` · `reportText` · `continueSec` · `collapsed` · `streamBuffer` · `streaming` | `string` · `FlowType` · `number` · `string` · `number\|string` · `boolean` · `string` · `boolean` |
| `GazeState` | `hasHeads` · `headsCount` · `anyInRoi` · `awayDuration` | `boolean` · `number` · `boolean` · `number` |
| `PeopleState` | `count` · `alert` · `alertColor` | `number\|string` · `string` · `string` |

> `NotifyEvent` 是流程事件与行为事件**合并后的统一渲染形状**（见 §5.3.1）。

**消息判定函数**：

| 函数 | 判据 |
|---|---|
| `isStreamMsg` | `source === 'segment_report_stream'` ｜ `type === 'report_stream'` ｜ `source === 'stream'` |
| `isTerminalMsg` | `source === 'segment_report'` ｜ `type === 'report'` ｜ `source === 'report'` |

**字段命名对照**（跨语言约定，**非**命名不一致）：

| 后端（snake_case） | 前端（camelCase） |
|---|---|
| `flow_type` | `flowType` |
| `flow_id` | `flowId` |
| `heads_count` | `headsCount` |
| `away_duration` | `awayDuration` |

---

## 九、当前版本

**文档版本** 1.2.0 ｜ **适用代码** `main` @ `f7585ee`

---

## 附录 A：错误码

| HTTP | `error` 内容 | 触发条件 | 处理建议 |
|---|---|---|---|
| `404` | `front video not found` | `data/videos/front.{mp4,mpg}` 均不存在 | 检查视频文件与 `config.yaml` 的 `videos` 配置 |
| `404` | `pop video not found` | 同上（pop） | 同上 |
| `422` | （FastAPI 内建结构） | 请求参数格式不合法 | 检查请求构造 |

> 控制类接口的业务状态**不通过 HTTP 状态码表达**，详见 §3.2。

---

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| **flow（流程）** | 一次完整的规程执行片段（如一次监护制绑定），由规则层状态机识别边界并分配 `flow_id` |
| **source** | 进度上报单元，细粒度命名如 `tracker.tracking`、`voice`、`behavior.video_pop` |
| **globalSec** | 推理侧全局时钟 = 各未结束 source 进度的 `min` |
| **localSec** | 源视频时间轴上的秒数 |
| **batch** | 一次结构化数据推送批次（含各类事件数组） |
| **快照（snapshot）** | 建连时补发的当前状态，用于前端恢复 |
| **状态量 / 事件流** | 通知栏的两类数据；前者实时、后者跟画面 |
