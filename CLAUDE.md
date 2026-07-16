# 核电站监护制合规检测系统 — 开发规约

本文档定义了本项目的编译、测试、粗粒度目录树规范及底层开发规约，任何后续模型与开发者必须严格遵循。

---

## 1. 编译与校验指令

```bash
# 静态语法编译校验
cd /home/wangshengping/Hedian/A_DemoSrc
/home/wangshengping/myconda/envs/sp_hedian/bin/python -m py_compile \
  core/base_module.py \
  core/display_buffer.py \
  core/event_bus.py \
  modules/tracker/tracker_module.py \
  modules/tracker/storage.py \
  modules/voice/voice_module.py \
  modules/voice/storage.py \
  modules/gaze/gaze_module.py \
  rules/supervision_rule.py \
  rules/self_ticket_rule.py \
  rules/info_notice_rule.py \
  main.py
```

---

## 2. 项目粗粒度目录规范

```
A_DemoSrc/                                # 项目根目录
├── main.py                               # 进程协调器入口
├── config.yaml                           # 全局配置文件
├── core/                                 # 核心通信与框架层（总线与缓冲器）
├── modules/                              # 算法模块层
│   ├── tracker/                          # 人员与视觉追踪
│   ├── gaze/                             # 头部与注视追踪（嵌入运行）
│   └── voice/                            # 语音转录与后处理
├── rules/                                # 规程合规判定状态机层
├── evaluation/                           # 流程终期评估层
└── models/                               # 静态模型资源目录
```

---

## 3. 代码开发规范 (Python)

- **命名风格**: 必须使用 **`camelCase` (驼峰命名法)**。
  - 函数与变量名: `pushDisplay`, `pushEvent`, `localSec`, `globalSec`, `roleDetails`
  - 严禁在 Python 代码中使用蛇形命名法（如 `push_display`, `local_sec`）。
- **异步处理**: 模块间通信必须通过 `EventBus` 异步进行，禁止跨模块直接耦合。
- **路径引用**: 始终使用相对路径 (如基于 `PathConfig` 解析 )，禁止硬编码绝对路径。

---

## 4. 双轨通信与推送统一接口规约

- **展示流（前端渲染）**与**事件流（业务逻辑）**采用彻底解耦的双轨设计。
- 模块必须分别显式调用对应的接口进行推送，严禁混用：
  * **展示流推送**：使用 `self.push_display(channel, data)` 写入对齐缓冲区，向前端展示轨迹或滚动文本。
  * **事件流推送**：使用 `self.push_event(EventTopic.XXX, data, ts=ts)` 发送至 Redis 事件总线，触发后端状态机运转。

---

## 5. 后端核心数据流转逻辑与模型遵循规则

任何后续模型修改、扩展系统时，必须严格遵守以下 4 条底层后端处理逻辑规则，禁止破坏：

### 规则一：基于 Redis Stream 的异步事件发布机制
- 所有子模块进程（语音、视觉追踪等）之间禁止发生同步 imports 依赖或直接函数调用。
- 	必须通过 `EventBus` 的 `publish()` 接口发布事件到 Redis Stream 队列。
- 	所有的合规规则状态机（`rules/` 目录）必须以消费者身份异步订阅这些 Stream 事件以执行状态转移。

### 规则二：基于最小进度（min localSec）的全局时钟对齐机制
- 各推理进程（Voice、Tracker等）在其主循环迭代中，必须实时向 Redis 共享 Hash 键 `inference:progress` 写入当前的推理进度 `localSec`。
- 展示缓冲器（`DisplayBuffer`）必须读取所有运行模块的 `localSec`，计算交集最小值作为 `globalSec = min(localSec)`。
- 写入展示缓冲区的数据只有在 `timestamp <= globalSec` 时，才允许通过 SSE 推送到前端，以保持视频帧、字幕和视线标注物理对齐。

### 规则三：精简 2 字段关键时刻独立落盘机制与格式示例
- 各模块（Voice、Tracker、Gaze）产生的关键报警及动作记录必须保存为独立的 JSON 文件（`Voice_key_moments.json`、`tracker_key_moments.json`、`gaze_key_moments.json`）。
- **数据结构硬约束**：文件内每条记录必须仅包含且严格包含两个字段：
  ```json
  {
    "localSec": float,
    "key_moment": "事件描述字符串"
  }
  ```
- **真实数据格式示例如下（必须完全对齐该格式）**：
  ```json
  [
    { "localSec": 2.72,  "key_moment": "请求监护" },
    { "localSec": 10.15, "key_moment": "1EAS013VB" },
    { "localSec": 12.50, "key_moment": "ROAD1举手" },
    { "localSec": 60.15, "key_moment": "没有看盘台持续15.4秒" },
    { "localSec": 75.40, "key_moment": "没有给予关注" }
  ]
  ```
- **Gaze 异常统计约束**：Gaze 模块仅允许记录异常状态。
  - 对于脱盘，必须在违规结束或视频终止时保存，且在描述中统计持续时间，格式为：`"没有看盘台持续XX.X秒"`。
  - 对于注意力缺失，直接记录为 `"没有给予关注"`（不附加持续时间）。

### 规则四：流程结束进度等待与多模态证据链合并提取机制
- 当判定规则判定流程结束时，数据提取器（`FlowDataExtractor`）必须首先读取 Redis `inference:progress`。
- 提取器必须执行**阻塞等待**，直到所有子模块的处理进度都追平或超过该流程的结束时间 `end_sec`（超时阈值为 300 秒），确保落盘 the JSON 文件写入完整。
- 随后，提取器读取 Voice、Tracker 和 Gaze 模块的三个 JSON 关键时刻文件，过滤并拼接该流程时间段内的所有事件，保存为 `evaluation/extracted_flow_<flow_id>.json`，然后将其传给大模型评估器。
