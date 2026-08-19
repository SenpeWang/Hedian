# 核电站监护制合规检测系统开发手册

## 1. 概述
本手册统一团队的开发标准、命名规范及协作流程，确保系统在高并发、多模态推理场景下具备可维护性与稳定性。系统采用多进程异步推理架构，各业务模块（voice / tracker / gaze / behavior）独立运行，通过 Redis 实现**事件流**与**推理流**双轨通信。

> ⚠ 与 AI 协作时，请先读本目录的 [`../CLAUDE.md`](../CLAUDE.md)（项目导航 + 改动禁区清单），其中第 7 节的"改动禁区"是改代码前必须看的内容。

## 2. 编码规范
- **命名风格**: 严格采用 **snake_case（蛇形命名法）**。
  - 函数名: `on_voice_signal`, `push_event`, `push_display`
  - 变量名: `local_sec`, `global_sec`, `role_details`
  - 严禁使用驼峰命名法（如 `onVoiceSignal`）。
- **跨线 JSON 字段**: `camelCase`（`localSec`、`flowId`、`flowType`），与 Redis 内部 `snake_case`（`local_sec`、`payload`）共存，不要混。
- **类型安全**: 在 Python 代码中优先使用类型注解，提升代码可读性与健壮性。
- **模块结构**: 遵循逻辑解耦，每个功能模块应有明确的入口、事件聚合接口与独立存储实现。

## 3. 架构原则

### 3.1 双轨数据流
系统通过 Redis 维护两条独立的数据流，严禁混用：

- **事件流（Event Stream）**：用于模块间业务通信。
  - 实现：`core/event_bus.py`（基于 Redis Stream + 独立消费组，key 前缀 `module:events:`）。
  - 典型事件（见 `EventTopic` 枚举）：`voice.key_moment`、`tracker.proximity`、`behavior.hand_raised`、`behavior.finger_screen`、`behavior.finger_file`、`flow.started`、`flow.ended`。
  - 特点：低频、关键、必须被可靠消费，驱动规则状态机转移。

- **推理流（Inference Stream）**：用于前端可视化展示。
  - 实现：`core/inference_stream.py`（模块进程写入）+ `core/inference_sync.py`（Web 进程读取并对齐）。
  - 典型数据：`video_front`、`video_pop`、`voice`、`gaze`、`tracking`、`progress`。
  - `flow_start`/`flow_end` 由 `inference_sync` 订阅 `event_bus` 的 `FLOW_STARTED`/`FLOW_ENDED` 后 `push_display` 转推（见 §11.6）。
  - 特点：高频、连续，必须经过 `global_sec` 时间对齐后打包为 batch 推送给前端。

### 3.2 异步推理与结果分发（spawn 子进程）
各后端模块以独立进程异步执行推理，互不阻塞：

- **voice / tracker / gaze / behavior** 模块分别产生原始推理结果。
- 每个模块将结果同时分发到两条流：
  - **事件流**：推送关键业务事件（如 `voice.key_moment`、`behavior.hand_raised`），供规则引擎、评估模块及其他业务模块消费。
  - **推理流**：推送可视化数据（如视频帧、转写文本、注视状态、跟踪框），供 `InferenceSync` 进行时间对齐与前端展示。
- `main.py` 用 `multiprocessing.set_start_method("spawn", force=True)`，点"开始测试"后 `on_pipeline_start` 用 `multiprocessing.Process` 拉起 voice/tracker/behavior 三子进程。**spawn 会让每个子进程重新 import 全部模块**，import 链不完整（如引用已删除的类）会直接在子进程崩溃，表现就是 Web 卡在"推理中"。

### 3.3 时间同步（Temporal Alignment）
- 所有业务事件与推理数据必须包含 `localSec` 字段（相对于视频起始的秒数）。
- `InferenceSync` 基于 **`global_sec = min(未结束且 expected 的 source 进度)`** 计算对齐时钟；已结束的 source 从 min 中剔除，缺失视频帧用 `video_front`/`video_pop` 的最后一帧补全。
- 对齐完即发、**不后端限速**，播放节奏交由前端 30fps 队列（`useWS.ts` 的 rAF 出帧循环）。
- 历史版本曾用"限速 + 5 秒容忍窗口"，当前实现已改为"去限速 + 主动结束信号驱动 done"，不要再沿用旧描述。

### 3.4 状态维护
- 各模块通过 `InferenceStream.update_module_snapshot` 实时更新 `module_snapshots`。
- `InferenceSync` 在打包 batch 时附加 `context`（由 `_get_context` 聚合各模块快照），便于前端获取各模块最新上下文。
- 模块退出时须主动 `mark_sources_done` 写入 `inference:source_done`（最终 local_sec），否则 `global_sec` 永远卡住、done 哨兵永不触发。

## 4. 目录结构规范

```
/home/wangshengping/Hedian/A_DemoSrc/
├── main.py                 # 入口：spawn 拉起子进程 + Web 常驻（端口 5002，0 显存）
├── config.yaml             # 用户配置（仅 videos.front / videos.pop 两路）
├── core/                   # 核心框架
│   ├── base_module.py      # 业务模块基类
│   ├── base_storage.py     # 存储基类
│   ├── config_manager.py   # 配置管理
│   ├── event_bus.py        # 事件流总线（EventTopic 枚举）
│   ├── inference_stream.py # 推理流写入端（push_display）
│   ├── inference_sync.py   # 推理流同步 / 时间对齐（global_sec）
│   ├── path_manager.py     # 路径管理
│   ├── redis_conn.py       # Redis 连接池
│   └── logger.py           # 统一日志
├── modules/                # 业务模块
│   ├── voice/              # 语音转录（Qwen3-ASR）
│   ├── tracker/            # 多目标跟踪与举手检测（HandRaiser）
│   ├── gaze/               # 凝视估计（tracker 进程内）
│   └── behavior/           # 行为检测：behavior_module + screen_detect + file_detector + base_detector
├── rules/                  # 规则状态机
│   ├── rule_base.py        # 规则基类与注册表
│   ├── flow_recorder.py    # 流程事件记录器
│   ├── supervision_rule.py # 监护制
│   ├── self_ticket_rule.py # 自唱票
│   ├── info_notice_rule.py # 信息通报
│   └── personnel_status_rule.py # 人员状态监控
├── evaluation/             # 评估层
│   ├── flow_evaluation_manager.py # 流程评估编排
│   ├── flow_data_extractor.py     # 流程数据提取
│   └── qwen_evaluator.py          # Qwen3-8B 大模型评估
└── web/                    # Web 层
    └── ws_handler.py       # WebSocket 二进制打包推送（VIEW_MAP: 0=front,1=pop）
```

> 前端在 `frontend/`（Vue3+Vite），关键文件 `frontend/src/composables/useWS.ts`。

## 5. 开发流程
1. **研究与规划**: 修改重大逻辑前，先在 `docs/` 下创建设计草案，明确对事件流与推理流的影响，并核对 [`../CLAUDE.md`](../CLAUDE.md) 的改动禁区。
2. **本地测试**: 确保新增事件包含 `localSec`，并验证 `InferenceSync` 对齐逻辑未被破坏；改完跑 `py_compile`（见 CLAUDE.md §1）。
3. **验证**: 检查日志，确保无 `AttributeError` 或 `ValueError`；确认事件推送顺序与 `global_sec` 同步；确认 `segment_report` 在 `done` 哨兵之前已送达前端。

## 6. 架构图
见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)（Mermaid）与 [`architecture.txt`](./architecture.txt)（纯文本）。

## 7. ⚠ 改动红线（与 CLAUDE.md §7 呼应，改前必看）

1. **`ws_handler.VIEW_MAP` 的 `view_id` 映射（0=front, 1=pop）不可改。**
   前端 `useWS.ts` 在 `playOneBatch` 里**硬编码** `f[0]`→front、`f[1]`→pop（`if (f[0] !== undefined)` / `if (f[1] !== undefined)`）。映射写错（历史上曾误把 pop 写成 view_id=2 → pop 黑屏）会让某路视频永远不显示。
2. **`FingerScreenDetector` / `FingerFileDetector` 是纯判定器，须由 `BehaviorModule.process_video` 统一加载 `behavior_yolo.pt`、每 5 帧 `model.track` 后分发 `results` 给 `detect(frame, results, frame_count, fps)`。** 不可改回"判定器自带模型"的旧接口，旧类 `FingerBupDetector`/`FingerPopDetector` 已删，引用即 spawn 崩溃。
3. **`InferenceSync.global_sec` 时钟逻辑与 `expected_sources` 不可破坏。** 改对齐逻辑前先读懂 `global_sec` 计算与 `_last_video_cache`（只缓存 `video_front`/`video_pop` 两路）。
4. **`main.py` 启动 `r.flushdb()` 清空 Redis**：重启会清掉前次推理全部缓存；结果落盘靠 `key_moments.json` 文件，不依赖 Redis 跨重启保留。
5. **视频帧协议字节序与字段顺序不可改**：`[1B version][4B globalSec f32 大端][1B view_count][重复:1B view_id+4B len 大端+JPEG][4B json_len 大端+JSON]`，前后端必须一致。

## 8. 常见故障排查
- **卡在"推理中"**：优先排查 spawn 子进程 import 崩溃（旧 detector 类名、类不存在）、`config.yaml` 视频路径不存在、Redis 未启动。
- **某路视频黑屏**：检查 `ws_handler.VIEW_MAP` 与前端 `f[0]/f[1]` 是否对应；`batch` 里是否真含 `video_front`/`video_pop` 帧。
- **全局时钟冻结**：检查 `inference:progress` 中各 source 进度是否正常更新、是否有 source 忘了 `mark_sources_done` 导致 `global_sec` 永远取最小值。
- **命名错误**：检查方法名与变量名是否使用 `snake_case`，避免残留驼峰命名。
- **路径问题**：始终使用 `PathManager` 或 `BASE_DIR` 计算路径，禁止硬编码绝对路径。
- **事件丢失**：检查事件是否写入正确的 `module:events:<topic>` key；评估结果需经 `push_text` 直推（绕过对齐）。
- **显存占用**：大模型评估子进程若未正常退出，主进程会在超时后强制终止，确保 GPU 显存释放。

## 9. 评估层前后端协作注意事项

### 9.1 推送通道与帧类型
| 通道 | 函数 | 事件 | 帧类型 | 经对齐 |
|------|------|------|--------|--------|
| `sync_fn` | `push`（二进制 batch） | `flow_start`/`flow_end` | 二进制 batch | ✅ |
| `direct_fn` | `push_text` | `segment_report_stream`（流式 chunk） | 文本帧 | ❌ |
| `inference_fn` | `push_text` | `segment_report`（最终报告） | 文本帧 | ❌ |

- 评估报告走**文本帧**（`ws_handler.push_text`），禁止走二进制 `push`：直推事件无 `globalSec`/无视频帧，塞进 `push` 会被打包成 `globalSec=0` 的 batch，前端全局时钟被拉回 0 且 meta 不匹配 → 报告丢失。
- `FlowEvaluationManager` 初始化必须同时传入三个函数，缺 `inference_fn` → 最终报告永不推送，卡片卡在"评估推理中"。
- 评估报告绕过对齐中间件零延迟推送，但卡片解锁仍由前端按画面进度触发（见 9.2）。

### 9.2 前端缓冲与画面进度解锁（后端 wait_playback_reached）
- 后端 `wait_playback_reached()`（`flow_evaluation_manager.py:237-255`）：首个评估 chunk 推送前等前端播放到 `end_sec - 0.5`，60s 超时兜底；`has_playback_waited` 只等一次（后续 chunk 实时推）。评估结果与画面同步涌现，不剧透。
- 前端 `handleReportEvent`（`useWS.ts:348-373`）：收到 `segment_report_stream` chunk 直接 `card.streamBuffer += chunk` + `streaming=true`，**无 buffer / 无 checkUnlock**（旧 `flowReportBuffer`/`checkUnlockReports`/`currentGlobalSec` 已删除）。
### 9.3 评估重置
- `FlowEvaluationManager.reset()` 取消未完成评估任务、清空报告列表，防止旧结果混入新 run。


---

## 11. 各模块开发注意事项（2026-08-17 更新）

### 11.1 tracker 模块（front 视角，GPU0）
- 内部串行链：检测→跟踪→举手→凝视→画标注→编码。整体一个速率，不拆模块取 min。
- gaze 内嵌 tracker（非独立进程），gaze 进度独立写（`_independent_progress_sources={"gaze"}`）。
- hand 稀疏：frame_step=3 + PoseEMAFilter（config supervision.hand_frame_step）。
- `_borrowed_sources={"behavior"}`：tracker 代推举手但归属 behavior，退出不标 behavior done。
- fps 从 CAP_PROP_FPS 读，读不到 raise。
- `_env_setup` del LD_LIBRARY_PATH（torch 库），gaze 靠 ctypes 预加载绕过。**勿删**。

### 11.2 behavior 模块（pop 视角，GPU1）
- 共享一次 YOLO forward（model.track），结果喂 screen+file 检测器复用。
- 稀疏推理：infer_every_n_frames=5。
- 进度字段 `behavior.video_pop`（fine=video_pop），_compute_global_sec 用大类比对计入。**改字段名会破坏 pop 计入**。

### 11.3 gaze 模块（内嵌 tracker，GPU0）
- 异步后台线程（ThreadPoolExecutor），主循环不阻塞。`_gaze_interval` 稀疏提交。
- 缓存绘制：后台未完成用上次结果。
- providers 构造默认 `["CUDAExecutionProvider"]`（删硬编码覆盖，靠构造默认）。

### 11.4 voice 模块（GPU1）
- device 不传，speech_transcriber 构造默认 `device="cuda"`（靠 CUDA_VISIBLE_DEVICES 定卡）。
- 进度字段 `voice.voice`，已计入 globalSec。

### 11.5 评估模块（qwen，GPU1，短命子进程）— ⚠ 最关键注意事项
- **异步事件驱动**：FLOW_ENDED → _on_flow_ended → _process_flow_pipeline。
- **⚠ 提取 keymoment 前必须等所有模块推理到流程结束**：`_wait_all_modules(end_sec, timeout=90)` 等各模块进度到 end_sec 才 extract。**改提取时序必须保留此等待**（防慢模块 keymoment 未产出）。
- **⚠ 评估结果推送时机**：`wait_playback_reached` 等前端播放到 `end_sec-0.5` 才推（前端已可视化到流程结束才推评估，超时60s兜底）。**这是"前端到流程结束才推评估"的耦合点，勿删**。
- **直推不走对齐**：push_direct → ws_handler.push_text（绕过 InferenceSync）。source=`segment_report_stream`/`segment_report`。
- **逐 token**：TextIteratorStreamer 逐段产 → stream_callback → chunk 推送。
- **GPU**：eval_gpu 从 config gpu_map["evaluation"]，_qwen_worker 设 CUDA_VISIBLE_DEVICES（gpu_manager.py 已删）。

### 11.6 InferenceSync（对齐推送）
- globalSec=min(各未结束 expected source 进度)，用 _SOURCE_CATEGORY 大类比对。
- 对齐即发不限速（POLL_INTERVAL_SEC=0.005）。
- batch 带 sourceTimes（各视角进度供前端速率引擎）+ totalDuration（`_build_batch` 带 `self.duration`）。
- **event_bus 桥接**：`__init__` 接收 `event_bus` 参数，订阅 `FLOW_STARTED`/`FLOW_ENDED`，回调 `_on_flow_started`/`_on_flow_ended` 调 `push_display` 把 flow 事件转推进 `results:all`，随 `_build_batch` 按 `source` 分组推前端（供系统通知事件流栏）。
- 视频流不进 results:all（_build_batch 无 video 分支，_last_video_cache 已删）。
- 死锁兜底 600s。

### 11.7 VisEncoder + VisStreamForwarder
- fMP4：ffmpeg rawvideo→libx264 ultrafast baseline +frag_keyframe。front 带音频（不用-shortest），pop 静音。
- PTS=帧序/fps=localSec（视频帧时间戳=推理进度秒）。
- VisStreamForwarder.start() 要 `_stop.clear()`（重启清停止信号）。

### 11.8 ws_handler
- send_vis_chunk：init 段先缓存（_vis_init_cache），新连接补发。
- update_playback_sec：max 单调递增（前端不回退）。
- push_text：评估直推。

### 11.9 前端 useWS
- **progress**（317）：`status==='done' ? 100 : totalDuration>0 ? globalSec/totalDuration*100 : 0`（**推理进度**，非播放进度）。
- **globalSec + totalDuration 来源**：`handleBatchEvent` 从 `raw.globalSec` + `raw.totalDuration` 取（后端 `_build_batch` 带 `self.duration`，~644s 稳定，**不依赖前端 video 加载**）。
- **connect status 补发**（220）：WS 连接时后端补发 `{source:status, totalDuration, globalSec, status}`，前端立即有 totalDuration（不用等首个 batch）。
- **setClockFns**（169）：只注 front/pop `currentTime`（MSE trim 用）；dur 参数已删（totalDuration 不再依赖 video.duration）。
- **MSE**：串行 appendBuffer + maybeTrim（清播放点前 8s 防 QuotaExceededError）。
- **速率引擎**：viewSecs 从 `batch.sourceTimes`，`_updateRate` 算各视角 Δsec/Δwall，`playbackRate=min(v_front,v_pop,v_voice,1.0)` 只降不升下限 0.2。
- **评估 token**：`handleReportEvent` 中 `card.streamBuffer += data.chunk`（勿丢 data.chunk）。
### 11.10 前端 VideoPanel
- **followTo 严格对齐**（每帧调）：检查 front 时刻是否在 pop 已缓冲区间内，在则偏差>0.1 即 seek 到 `front.currentTime`；不在则 `playbackRate=0` 暂停等待（避免 seek 到未缓冲位置乱跳帧）。
- 不再用"平滑调 rate 追随"（旧实现：加速/减速追随，累积偏差导致 pop 后半段超前）。
- preservesPitch=true 防变调。
- 暴露：playVideo/pauseVideo/followTo/currentTime/duration。

### 11.11 前端 ReportPanel
- **sc-head 去"评估中…"**：全过程显示 `{{ cardLabel(flowType) }} #{{ flowId }}`（监护制#1/#2、自唱票#1）；streaming 图标 🤖，完成图标 🛡️ + `[continueSec]s` + 分数。

- **setInterval 60ms typewriter**（54-65）：每 tick `shownLen[flowId]+1`，~16 字/秒，模拟大模型慢速逐字（前端收到完整数据后逐字展示，非 token 实时）。
- **shownText**（42-48）：`streamBuffer.slice(0, shownLen)` → `parseReportContent` 拆 `<think>` 块，think+report 都逐字；`streaming=false` 时返回完整。
- **segment_report 不覆盖**（366-369）：`segment_report`（完整）到达只设 `card.reportText`（兜底），**不覆盖 streamBuffer、不设 streaming=false** → typewriter 自然逐字到末尾再停（之前 bug：覆盖导致一次性全显示）。
- **滚底**（62）：`typewriterTick` 每次调 `scrollToBottom()`（nextTick + scrollTop=scrollHeight，smooth），逐字跟随最新输出（同 VoicePanel）。
- **无光标**，`parseReportContent` 拆 think/report 两段常驻显示。


## 12.1 前端可视化限制与实现思想（2026-08-18 更新）

### 可视化红线（不可改样式）
- **标注画进帧**：draw_* 画进像素 → fMP4，前端只 MSE 播放，不 Canvas 实时画坐标（对齐复杂度归零）。
- **进度条样式**：CSS 进度条（`.pbar`/`.pfill` div+width%），不改字符/颜色/线宽（已替代字符 █░ 定型）。
- **卡片/视频样式**：seg-card/sc-head/sc-bar、VideoPanel CSS 全锁。

### 同步机制（哪些同步、怎么同步）
- **速率同步**：`playbackRate=min(v_front,v_pop,v_voice,1.0)` 只降不升（下限 0.2），preservesPitch 防变调。
- **pop 追随 front**：`followTo` 每帧严格对齐（buffered 内偏差>0.1 seek 到 front.currentTime，未缓冲 rate=0 暂停）。
- **两级对齐**：后端 `globalSec`（min 各视角进度）对齐推事件；前端 `currentPlaybackSec`（video.currentTime）按播放时刻取数显示。
- **progress 不依赖 video**：globalSec+totalDuration 来自后端 batch（self.duration），video 未加载也能更新进度。

### 流程评估可视化
- **不剧透**：`wait_playback_reached` 等前端播到流程结束才推评估 chunk。
- **逐字流式**：setInterval 60ms/字（~16 字/秒，模拟大模型）；segment_report 不覆盖不中断 typewriter。
- **滚底跟随**：typewriterTick + scrollToBottom 逐字滚底（同语音转录面板）。
- **think+report 两段**：parseReportContent 拆思考块与正文，常驻显示。

### 前端可视化 ASCII 效果图

**整体布局**（header + 双视频列 + 底部三面板）：
```
┌───────────────────────────────────────────────────────────────────┐
│ ⚛️ 核电站行为合规检测系统  [👤开始测试][🛑停止]  推理进度 ████████░░ 49.8% │
├───────────────────────────┬───────────────────────────────────────┤
│  camFRONT (1x 主时钟带音)  │  camPOP (静音 followTo 追随 front)     │
│  [带标注 fMP4 流]          │  [带标注 fMP4 流]                     │
├──────────┬────────────────┴───────────────────┬──────────────────┤
│🎤人员对话 │📊系统通知                          │📋流程评价        │
│ [00:12]..│ 👥监控室:3人🟢  👁凝视:就绪        │ 监护2 自唱1      │
│ [00:15]..│ ──────────────                      │ 通报1 均分7.5    │
│ [00:20]..│ [流程]00:30监护制▶ 00:45自唱票▶    │ 总分22           │
│ (自动滚底)│ (状态量持续+事件流只增)            │ ──────────       │
│          │                                    │ 🛡️监护制#3[45s]  │
│          │                                    │ ████████ 8/10    │
│          │                                    │ 🧠思考推理过程    │
│          │                                    │ 请求监护后5秒内  │
│          │                                    │ 举手...▌(逐字)   │
│          │                                    │ (涌现+滚底)      │
└──────────┴────────────────────┴──────────────────────────────────┘
```

**进度条**（CSS div+width% 替代字符）：
```
推理进度 ████████████░░░░░░░░ 49.8%
         └─ pfill width=49.8% ─┘└─ pbar 底色 ─┘
点开始前隐藏(v-show=running/starting)，done 时 100%
```

**流程评估卡片**（逐字流式 → 完成）：
```
┌─ 🤖 监护制 #3 ▼ ──────────────────┐  ← streaming=true, 无分数
│ 🧠 思考推理过程                      │  ← think-block 常驻
│ 请求监护语音发起前后5秒内举手, 合格▌ │  ← shownText 截 streamBuffer 逐字
│ ──────────────────────────────────  │
│ (正式报告正文逐字涌现)               │  ← sc-detail
└────────────────────────────────────┘
  ↓ segment_report 到达(不覆盖) → typewriter 赶满 →
┌─ 🛡️监护制#3[45s] ▼ ───────── 8/10 ─┐  ← streaming=false, 显分数+sc-bar
│ ████████░░ 8/10                     │  ← sc-bar-fill width=80%
│ 🧠 思考推理过程                     │
│ 请求监护语音发起前后5秒内举手, 合格  │  ← 完整
│ 评估:合规度高, 监护员到位, 九字码... │
└────────────────────────────────────┘
  setInterval 60ms/字 + scrollToBottom 滚底跟随
```

**同步可视化**（速率引擎 + followTo）：
```
batch.sourceTimes: front=438  pop=438  voice=438 → v=Δsec/Δwall
playbackRate = min(v_front,v_pop,v_voice,1.0)  只降不升(下限0.2)
  front: 1.0x ━━━━━━━━━━━━━━━━━━ (主时钟, 带音频, preservesPitch)
  pop:   0.9x ━━━━━━━━━━━━━━━━━━ (followTo 平滑追随 front.currentTime)
  推理慢于实时→降速慢放,不超前不卡顿; 快于实时→封顶1.0x
```