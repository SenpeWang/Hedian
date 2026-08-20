# REQUIREMENTS.md — 验收标准（按代码模块组织）

## 1. 核心业务目标

核电站主控室三类操作规程的**自动合规检测**：监护制(supervision)、信息通报(info_notice)、自唱票(self_ticket)。用计算机视觉（目标检测/跟踪/姿态/凝视/手指指向）+ 语音识别（关键字/9字码）+ 大模型评估，自动判定操作人/监护人是否按规定执行规程，输出 10 分制合规评分与可追溯报告。

业务规则详见 `docs/行为规范标准.md` + `docs/rule_doc/{监护制,信息通报,自唱票}.md`。

## 2. 用户角色与场景

| 角色 | 场景 |
|---|---|
| 监护人(US/值长) | 监督操作人，下达"可以执行"命令，回答"收到" |
| 操作人(RO1/RO2 一回路/二回路) | 执行规程（请求监护/举手/读9字码/指向程序/执行/核对）|
| 信息发起者 | 信息通报：举手高声喊"信息通报"，传递信息，喊"通报完毕" |
| 系统运维 | 部署推理，查日志/Redis 排查 |
| 合规审查 | 回看 data/results/ 评估报告事后追溯 |

## 3. 功能需求清单（按代码模块）

### 3.1 tracker 模块（`modules/tracker/`，front 视角，GPU0）
- **TR-01**：front 视频检测(yolo11l)+跟踪(DeepSORT)+角色分配(LEADER/ROAD1/ROAD2)，画标注(draw_tracks/距离线)进帧。
- **TR-02**：举手检测(yolo26s-pose 稀疏 frame_step=3+EMA)，产出 `behavior.hand_raised` 事件。
- **TR-03**：监控室人数(PEOPLE_COUNT_UPDATE，<1告警/1提醒/>=3正常)。
- **TR-04**：fps 从 `CAP_PROP_FPS` 读源真实帧率(30)，读不到 raise 不兜底。
- **TR-05**：GPU 分配 tracker→GPU0。

### 3.2 gaze 模块（`modules/gaze/`，凝视，GPU0/ONNX）
- **GZ-01**：头部检测(ONNX head_detector)+注视估计(gazelle，异步后台线程)。
- **GZ-02**：判断视线在 ROI 内(has_heads/any_in_roi/awayDuration，离开>60s 告警)。
- **GZ-03**：gaze ONNX 上 GPU(main.py 设 LD_LIBRARY_PATH 指向 nvidia cu12 pip 库, cufft 错误=0)。
- **GZ-04**：画 head/gaze 标注进帧。

### 3.3 voice 模块（`modules/voice/`，语音转录，GPU1）
- **VC-01**：从 camFRONT 音频提取语音(Qwen3-ASR-0.6B + ForcedAligner 词级对齐)，`push_display("voice", {localSec, text, keys})` 走推理流对齐。
- **VC-02**：识别关键字(请求监护/设备码/执行/核对/信息通报/通报完毕/收到)，拼音匹配容错。
- **VC-03**：9字码格式 `([1-9]?[A-Z]{2,3}\d{3}[A-Z]{2}|1EAS\w+|T\d*RPA\w+|LCO[\w\.]+|RPA\w+|SM3)`。
- **VC-04**：GPU 分配 voice→GPU1。

### 3.4 behavior 模块（`modules/behavior/`，pop 视角，GPU1）
- **BH-01**：pop 共享一次 YOLO(behavior_yolo.pt 每5帧 `model.track`)，结果串行分发给 FingerScreenDetector+FingerFileDetector(纯判定器不持模型)。
- **BH-02**：手指指向屏幕检测(`behavior.finger_screen`)。
- **BH-03**：手指指向文件检测(`behavior.finger_file`)。
- **BH-04**：画 ROI+检测框进帧。
- **BH-05**：GPU 分配 behavior→GPU1。

### 3.5 rules 模块（`rules/`，规程状态机）
- **RL-01**：监护制——"请求监护"+5s内举手启动(FLOW_STARTED)/全程监护/9字码复述+执行+核对/监护人离开结束(FLOW_ENDED)。
- **RL-02**：信息通报——举手+"信息通报"启动/团队关注/信息传递/"通报完毕"即时闭环(不等"收到")/值长"收到"不阻塞。
- **RL-03**：自唱票——**前端 GUI 控件触发启动(红线，不依赖语音)**/9字码唱票/下一流程自动收尾。
- **RL-04**：沟通规范——岗位名称+"请讲"/"收到"/电话 OVER 法/三段式"请复述"。
- **RL-05**：红线——单独举手不启动流程；自唱票不依赖语音启动。

### 3.6 evaluation 模块（`evaluation/`，Qwen3-8B，GPU1）
- **EV-01**：异步事件驱动(FLOW_ENDED→`_process_flow_pipeline`)。
- **EV-02**：`_wait_all_modules(end_sec, timeout=90)` 等所有模块推理到流程结束。
- **EV-03**：Qwen3-8B 子进程评估(TextIteratorStreamer 逐 token)。
- **EV-04**：评估维度(监护制5维/信息通报4维/自唱票3维, 10分制)。
- **EV-05**：`wait_playback_reached(end_sec-0.5, timeout=60)` 等前端播到流程结束才推。
- **EV-06**：`push_direct` 直推绕对齐中间件(segment_report_stream/segment_report)。
- **EV-07**：逐 token 流式 + 完成态切换(streamBuffer 累积/typewriter 60ms/追完+reportText→streaming=false 显分数)。
- **EV-08**：GPU 分配 evaluation→GPU1。

### 3.7 infra 模块（`web/`+`core/`，基础设施）
- **IN-01**：InferenceSync `globalSec=min(各视角进度含停滞0)` 对齐推送(`_push_events_up_to localSec<=globalSec`)。
- **IN-02**：速率引擎前端 `playbackRate=min(v_front,v_pop,v_voice,1.0)` EMA 平滑 下限0.2(慢路拖累主等从)。
- **IN-03**：VisEncoder fMP4 编码(ffmpeg libx264 ultrafast baseline +frag_keyframe, PTS=帧序/fps, front 带音频/pop 静音)。
- **IN-04**：VisStreamForwarder 两路独立消费 vis_stream→ws send_bytes。
- **IN-05**：`/reset` kill 子进程+flushdb 清缓存, `/start` `/stop` 控制。
- **IN-06**：关键帧/评估报告落盘 `data/results/<run_id>/`。
- **IN-07**：600s 死锁兜底(无推进强制收尾)。
- **IN-08**：Redis Stream 短期流(results:all/vis_stream, 不持久; 结构化结果落盘 json)。

### 3.8 frontend 模块（`frontend/`，Vue3 可视化）
- **FE-01**：双路 MSE fMP4 流式播放(front 带音频主时钟/pop 静音)。
- **FE-02**：主时钟 `currentPlaybackSec=front.currentTime`，所有面板按此取数。
- **FE-03**：pop 自驱动 RAF `followTo(currentPlaybackSec)` buffered-seek 对齐(偏差>0.15s seek, 兜底不推末尾)。
- **FE-04**：MSE SourceBuffer `maybeTrim` 按主时钟删前8s(可视化过删, pop 失锁不乱删)。
- **FE-05**：状态量状态栏(人数/凝视 `getLatestAt(currentPlaybackSec)` 持续显示, TimeSeriesPool reactive front 暂停仍更新)。
- **FE-06**：事件流状态栏(语音/流程 `filter sec<=currentPlaybackSec+3` 累积)。
- **FE-07**：评估逐字 typewriter(60ms/字, 完成态切换显分数/进度条/完成图标)。
- **FE-08**：进度条 `globalSec/totalDuration`(推理进度超前播放, stopped 不跳100 done 才100)。
- **FE-09**：刷新 `/reset` 重置, WebSocket 重连指数退避+上报待发队列+生命周期清理(socket/MediaSource/blob URL)。
- **FE-10**：不超前/不卡顿(变速慢放+MSE trim)。

## 4. 非功能需求

| 指标 | 要求 |
|---|---|
| front 推理速度 | ≥30fps(独占GPU0); 慢于实时时变速慢放兜底 |
| pop 推理速度 | ≥25fps(GPU1) |
| 前端播放 | 不超前/不卡顿花屏(变速+MSE trim); front/pop 画面时刻一致 |
| 评估延迟 | 前端播放到流程结束后≤60s 出评估结果 |
| 9字码准确率 | 正则 `([1-9]?[A-Z]{2,3}\d{3}[A-Z]{2}|1EAS\w+|T\d*RPA\w+|LCO[\w\.]+|RPA\w+|SM3)` 匹配 |
| GPU | RTX 4090×2, 视角级分配, gaze 上 GPU(cufft=0) |
| 可追溯 | 关键帧/评估报告落盘 |
| 用户体验 | 评估逐字流式无光标; 进度条和画面同步; 四面板不超前透刷 |

## 5. 不做清单（Out of Scope）

- **不改前端可视化样式**: CSS/颜色/线宽/字体/draw_* 样式参数
- **不降 imgsz**: 检测输入分辨率固定640(精度)
- **不换检测模型权重**: yolo11l/yolo26s-pose 微调过, 不换不合并
- **不用 TensorRT**: 不换推理引擎
- **不跳帧取巧**: detect 不降频(hand frame_step=3 是 v1 背书例外)
- **不前端变速超 1x**: playbackRate 上限1.0, 绝不加速
- **不做历史回放接口**: vis_stream 实时, 清了就没了(关键帧/报告落盘可追溯)
- **不做多用户并发**: 单用户单次推理
- **不改后端评估协议**: _wait_all_modules + wait_playback_reached + push_direct 时序保持
- **自唱票启动不依赖语音**: 前端 GUI 弹窗信号触发(红线)
- **信息通报即时闭环**: 喊"通报完毕"即结束, 不等"收到"

## 6. 时间戳命名规范（前后端统一）

以下时间戳**值同（源视频秒）**，但语义角色不同：

| 命名 | 语义 |
|---|---|
| `localSec` | 模块内推理进度（per-source，frame_count/fps，update_module_time 写） |
| `globalSec` | 结构化数据经 InferenceSync 对齐后推送的全局时间（=对齐闸门 min 各视角，同一概念，不冲突） |
| `PTS` | 视频流 fMP4 帧时间戳（=帧序/fps，ffmpeg -r fps 生成，前端 sequence 模式隐式 ≈ PTS） |
| `currentPlaybackSec` | 前端播放进度（front.currentTime，主时钟，可 seek/变速） |

模块内 `localSec` 经 InferenceSync 对齐推送后即 `globalSec`（值不变，对齐只筛选 `localSec<=闸门 globalSec` 推送）。视频流 `PTS` 走独立 `vis_stream` 通道（不经 InferenceSync）。结构化数据 `globalSec` 与对齐闸门 `globalSec` 是同一概念，不冲突。

