# ARCHITECTURE.md — 施工蓝图

## 1. 整体架构分层图

```
┌─────────────────────────────────────────────────────────────────┐
│  前端层 (Vue3 + TS + MSE)                                        │
│  App.vue(主时钟) ─ useWS(速率引擎/时序池/MSE) ─ components       │
│  (VideoPanel/NotifyPanel/ReportPanel/HeaderBar)                  │
└────────────┬──────────────────────────────────────┬─────────────┘
             │ WebSocket /ws/data                    │
             │ (二进制fMP4帧 + JSON文本帧)           │
┌────────────┴──────────────────────────────────────┴─────────────┐
│  Web层 (main.py + uvicorn, GPU不绑)                              │
│  FastAPI ─ InferenceSync(对齐推送) ─ VisStreamForwarder(视频转发)│
│           FlowEvaluationManager(评估编排)                        │
└────┬───────────┬──────────────┬──────────────┬─────────────────┘
     │spawn      │spawn          │spawn         │mp.spawn(短命)
┌────┴────┐ ┌────┴─────┐ ┌──────┴──────┐ ┌────┴────────┐
│voice    │ │tracker   │ │behavior     │ │qwen评估     │
│GPU1     │ │GPU0      │ │GPU1         │ │GPU1         │
│Qwen3-ASR│ │detect+   │ │YOLO+        │ │Qwen3-8B     │
│+Aligner │ │track+hand│ │screen+file │ │(逐token)    │
│         │ │+gaze     │ │             │ │             │
└────┬────┘ └────┬─────┘ └──────┬──────┘ └────┬────────┘
     │           │              │              │
     └───────────┴──────────────┴──────────────┘
                       │ Redis Stream
            ┌──────────┴──────────┐
            │ 事件流 module:events:*│(模块间通信)
            │ 推理流 inference:results:all│(前端可视化)
            │ 视频流 inference:vis_stream:front|pop│(fMP4)
            │ 进度 inference:progress│(Hash)
            └─────────────────────┘
```

## 2. 核心技术选型及选型理由

| 技术 | 选型 | 理由 |
|---|---|---|
| Web框架 | FastAPI + uvicorn | 异步 + WebSocket 原生支持 + 自动文档 |
| 进程模型 | multiprocessing spawn | 模块隔离崩溃 + GPU 独占 + spawn 兼容 CUDA |
| 通信 | Redis Stream | 解耦 + 持久 + 发布订阅 + Stream 天然时序 |
| 视频编码 | ffmpeg libx264 fMP4 | `+frag_keyframe+empty_moov` 支持流式 append，MSE 原生播放 |
| 前端播放 | MSE (MediaSource) | fMP4 流式 appendBuffer，标注画进帧对齐归零 |
| 检测 | ultralytics YOLO11l | 微调过的 MOT 模型，box 精度给跟踪用 |
| 姿态 | YOLO26s-pose | 微调过，举手几何判定 |
| 凝视 | ONNX (yolov8n_head + gazelle) | ORT CUDA，gaze 异步后台线程 |
| 语音 | Qwen3-ASR + ForcedAligner | 转录 + 时间对齐 |
| 评估 | Qwen3-8B (TextIteratorStreamer) | 逐 token 流式，合规评分 |
| 变速 | video.playbackRate + preservesPitch | 最慢视角决定速度，防变调 |
| 锁步 | followTo 平滑调 rate | 避离散 seek 导致前后跳 |

## 3. 各模块详细职责边界

### tracker（front 视角，GPU0）
- 职责：检测人→跟踪(DeepSORT)→举手(pose稀疏)→凝视(ONNX异步)→画标注→编码fMP4
- 边界：gaze 内嵌 tracker（非独立进程）；hand 借用 behavior 署名（_borrowed_sources）；gaze 进度独立写（_independent_progress_sources）
- 串行链：每帧 检测→跟踪→举手→凝视→画→编码，整体一个速率不拆

### behavior（pop 视角，GPU1）
- 职责：YOLO共享推理→ROI→手指屏幕→手指文件→画标注→编码fMP4
- 边界：每5帧推理1次复用；进度字段 behavior.video_pop（大类比对计入 globalSec）

### gaze（内嵌 tracker，GPU0）
- 职责：头部检测+凝视估计，异步后台线程，缓存绘制
- 边界：不依赖 detect/track/hand（用原始 frame 独立）；providers 构造默认 CUDA

### voice（GPU1）
- 职责：ASR 转录 + 时间对齐
- 边界：device 构造默认 cuda；进度 voice.voice

### evaluation（qwen，GPU1 短命子进程）
- 职责：收 flow_end → 等各模块到 end_sec → 提取 keymoment → 大模型评估 → 等前端可视化到流程结束 → 直推评估结果
- 边界：直推不走对齐（push_direct）；逐 token；gpu_manager 已删靠 eval_gpu

### InferenceSync（web 进程）
- 职责：globalSec=min(各视角进度) 对齐推送 + sourceTimes 供速率引擎
- event_bus 桥接：__init__ 接 event_bus，订阅 FLOW_STARTED/FLOW_ENDED，回调 push_display(flow_start/flow_end) 转推进 results:all，随 _build_batch 按 source 分组推前端（供系统通知事件流栏）
- 边界：对齐即发不限速；视频流不进 results:all

### VisEncoder + VisStreamForwarder
- 职责：帧→fMP4→Redis Stream→ws send_bytes
- 边界：PTS=帧序/fps=localSec；front 带音频不用 -shortest

## 4. 关键数据流

### 视频流（fMP4 二进制）
```
子进程 while读帧 → 推理 → draw_*画标注进帧
  → VisEncoder(ffmpeg -r fps, PTS=帧序/fps=localSec, +frag_keyframe)
  → xadd inference:vis_stream:{front|pop}
  → VisStreamForwarder xread → ws_handler.send_bytes([channel][type]+fMP4)
  → WebSocket二进制 → 前端 MSE appendBuffer → <video> 播放(playbackRate变速)
```

### 结构化结果（JSON 文本）
```
各模块 push_display(tracking/gaze/voice/progress)
  + inference_sync 订阅 event_bus FLOW_STARTED/FLOW_ENDED → push_display(flow_start/flow_end)
  → InferenceStream → inference:results:all
  → InferenceSync globalSec=min(各视角进度) 对齐
    → _push_events_up_to(只推 localSec≤globalSec, 快视角压pending)
    → _build_batch(含sourceTimes各视角进度) → ws.push → 前端
```

### 评估（异步事件驱动）
```
flow_end → _on_flow_ended → _process_flow_pipeline:
  1. _wait_all_modules(end_sec, 90)     # 等各模块推理到流程结束
  2. extract keymoment                  # 提取关键事件
  3. qwen.evaluate(TextIteratorStreamer) # spawn子进程GPU1逐token
  4. wait_playback_reached(end_sec-0.5) # 等前端可视化到流程结束
  5. push_direct(segment_report_stream, chunk)  # 直推逐token
  6. push_direct(segment_report, report_text)  # 完整报告
```

### 前端速率引擎
```
batch.sourceTimes → useWS viewSecs{front,pop,voice}
  → _updateRate: v=Δsec/Δwall(各视角整体速度)
  → playbackRate=min(v_front,v_pop,v_voice,1.0) 只降不升
  → front/pop 设 playbackRate + preservesPitch
  → pop followTo(每帧): 按front.currentTime偏差平滑调rate追随(不离散syncTo)
  → progress = globalSec/totalDuration*100 (推理进度, totalDuration from batch self.duration, 不依赖video)
  → CSS进度条(.pbar/.pfill div+width%, 替代字符█░); done时100%
```

## 5. 重要设计决策与权衡（Trade-off）

### 决策1：标注画进帧 vs 推坐标让前端画
- **选**：画进帧（draw_* 画进像素 → fMP4）
- **理由**：标注与视频物理一体，对齐复杂度归零；不用前端 Canvas 实时画坐标
- **权衡**：后端编码开销 +6ms/帧，但消除前端对齐抖动，值

### 决策2：视角级 GPU 分配 vs 全局单卡
- **选**：视角级（front→GPU0, voice/pop/eval→GPU1）
- **理由**：front 被 pop 争抢拖慢（30→21fps），分卡后 front 独占稳 33fps
- **权衡**：需多卡；gpu_manager 自主选卡删除（统一分配）

### 决策3：最慢视角决定播放速度 vs 固定 1x
- **选**：playbackRate=min(各视角速度,1.0) 只降不升
- **理由**：推理慢于实时时，固定1x会 underrun 卡顿；变速慢放不超前不卡
- **权衡**：音频变速需 preservesPitch 防变调；只降不升保证不破坏同步

### 决策4：评估等前端可视化到流程结束 vs 收到 flow_end 立即推
- **选**：wait_playback_reached 等前端播到 end_sec
- **理由**：评估结果要和画面同步涌现，不能超前显示
- **权衡**：前端慢放时评估等待更久（超时60s兜底）

### 决策5：setInterval 60ms 慢速逐字 vs token 实时
- **选**：后端逐 chunk 推 → 前端 `streamBuffer += chunk` 累积 → `setInterval 60ms` 改 `shownLen` 逐字截取展示（~16 字/秒，模拟大模型）
- **理由**：像大模型那样慢速逐字涌现，体验一致；前端收到完整数据后逐字展示
- **权衡**：`segment_report`（完整）到达**不覆盖 streamBuffer、不设 streaming=false**（否则一次性全显示，中断流式）；typewriter 赶满后自然停；`typewriterTick` 调 `scrollToBottom` 滚底跟随；无光标

### 决策6：hand 稀疏 frame_step=3 vs 每帧
- **选**：每3帧推理1次 pose + EMA 平滑
- **理由**：pose 变化慢，v1 背书；省 hand 2/3 开销
- **权衡**：detect 不稀疏（跟踪需要帧帧）

### 决策7：不换 yolo11l/TRT/降 imgsz
- **选**：保留 yolo11l 87GFLOPs + 变速慢放兜底
- **理由**：用户微调过模型不换不合并；不降 imgsz 保精度；不用 TRT
- **权衡**：front 30fps 是上限，靠变速兜底慢于实时

### 决策8：两级对齐（后端 globalSec + 前端 currentPlaybackSec）
- **选**：后端压快视角事件 + 前端按播放时刻取数
- **理由**：后端保证推的事件已就绪；前端保证显示对齐画面
- **权衡**：两级缺一不可（只后端会跳变，只前端乱推未来）
- **progress 不依赖 video**：`progress = globalSec/totalDuration*100`，globalSec + totalDuration 都来自后端 batch（`_build_batch` 带 `self.duration`），connect 时 status 帧补发 totalDuration 兜底；video 未加载也能更新进度
