# CLAUDE.md — AI 会话指引

## 1. 项目简介

核电站监护制合规检测系统：分析主控室 front（正面，30fps，带音频）+ pop（操作盘，25fps）两路视频与音频，自动检测操作人/监护人是否按监护制规程操作（请求监护→举手→监护人到位→读设备码→执行→核对），用 Qwen3-8B 大模型评估合规性，前端流式可视化（视频标注画进帧 + 结构化面板按播放时刻对齐 + 评估逐字输出）。

## 2. 核心目录结构与各层职责

```
Hedian/A_DemoSrc/
├── main.py                 # 入口层：spawn 业务子进程(voice/tracker/behavior) + web进程 + InferenceSync + VisStreamForwarder
├── config.yaml             # 配置层：app.gpu_map(视角级GPU) redis paths videos
├── core/                   # 基础设施层
│   ├── inference_sync.py     # 对齐推送：globalSec=min(各视角进度)，batch带sourceTimes
│   ├── inference_stream.py   # 模块写进度/事件(KEY_PROGRESS, KEY_EVENT_STREAM, _SOURCE_CATEGORY)
│   ├── event_bus.py          # 事件流(module:events:*, EventTopic)
│   ├── base_module.py        # 模块基类(update_progress, push_display, _inference_sources)
│   ├── vis_encoder.py       # fMP4编码(ffmpeg rawvideo→libx264, PTS=帧序/fps)
│   ├── path_manager.py / config_manager.py / logger.py
├── modules/                # 推理层（spawn子进程）
│   ├── tracker/             # front视角：object_detector(检测) multi_object_tracker(跟踪) ocsort_bytetrack visualizer
│   ├── gaze/                # 凝视(内嵌tracker)：gaze_module head_detector gaze_estimator
│   ├── behavior/            # pop视角：behavior_module hand_raiser finger_screen finger_file
│   ├── voice/               # 语音：voice_module speech_transcriber
├── evaluation/             # 评估层
│   ├── flow_evaluation_manager.py  # 评估编排(_wait_all_modules+提取keymoment+wait_playback_reached)
│   ├── qwen_evaluator.py           # Qwen3-8B子进程(TextIteratorStreamer逐token)
│   ├── flow_data_extractor.py      # keymoment提取
├── web/                    # Web层
│   ├── http_server.py       # FastAPI路由 + ws endpoint(收playback_progress)
│   ├── ws_handler.py        # WebSocket推送(send_vis_chunk/push/push_text, update_playback_sec)
│   ├── vis_forwarder.py     # 消费vis_stream→send_bytes
├── rules/                  # 规则层（监护制/自唱票/信息通报状态机）
├── frontend/src/           # 前端层(Vue3+TS)
│   ├── composables/useWS.ts # WebSocket+MSE+速率引擎+时序池+评估路由
│   ├── components/           # VideoPanel/NotifyPanel/ReportPanel/HeaderBar
│   ├── App.vue              # 主时钟+锁步
├── data/videos/            # camFRONT.mpg(30fps,641s) camPOP.mpg(25fps,644s)
├── models/                 # detection/gaze/voice/evaluation权重
├── docs/                   # REQUIREMENTS.md ARCHITECTURE.md API.md SDD.md
```

## 3. 常用开发命令

```bash
# 启动（视角级GPU分配）
cd /home/wangshengping/Hedian/A_DemoSrc
setsid nohup /home/wangshengping/myconda/envs/sp_hedian/bin/python main.py --gpu 0 > /tmp/hedian_web.log 2>&1 < /dev/null &

# 前端构建（系统node v12太旧，用node20）
cd frontend && PATH=/home/wangshengping/node20/bin:$PATH npm run build

# 查推理状态
redis-cli XLEN inference:vis_stream:front        # 视频流产出量
redis-cli HGETALL inference:progress              # 各模块进度
grep -aE "module.tracker.*帧|举手|评估" /tmp/hedian_web.log

# 停止推理
curl -X POST http://127.0.0.1:5002/stop
pgrep -f main.py | xargs kill -9

# SSH连接
ssh wangshengping@10.152.88.66   # conda env: sp_hedian, 端口5002, Redis localhost:6379/0
```

## 4. 硬性编码规范

- **命名**：Python 严格 snake_case（`on_voice_signal`、`local_sec`）；跨线 JSON camelCase（`localSec`、`flowId`）；严禁驼峰 Python 命名。
- **类型注解**：Python 优先类型注解（`def track(self, frame: np.ndarray, detections: List[Dict]) -> List[STrack]`）。
- **注释**：状态量状态栏标注 getLatestAt（人数/凝视），事件流状态栏标注 filter（语音/流程）；关键时序约束必须注释（如 _wait_all_modules 必须在 extract 前）。
- **fps**：从 `cv2.CAP_PROP_FPS` 读源视频真实帧率，读不到 `raise` 不兜底（防时间轴错位）。
- **device**：模块内不显式设卡，统一靠 `main.py CUDA_VISIBLE_DEVICES`（gpu_manager.py 已删）。
- **远端改动**：用 `python3 - << 'PYEOF'` heredoc 或 base64 传脚本（避免 SSH 引号嵌套）；单引号 heredoc 防变量展开。
- **错误处理**：推理子进程失败要落日志 + 不拖垮主流程（VisEncoder feed_frame 写失败置 _stopped）。
- **备份**：改动前 copy .bak，验证后清理（勿堆积）。

## 5. 相关文档引用

- **需求验收标准**：见 `docs/REQUIREMENTS.md`（功能需求清单 FR-xx、非功能需求、不做清单）
- **架构施工蓝图**：见 `docs/ARCHITECTURE.md`（分层图、技术选型理由、模块职责边界、数据流、设计权衡）
- **API 接口**：见 `docs/API.md`
- **详细设计**：见 `docs/SDD.md`
- **开发规范**：见 `docs/DEVELOPMENT_GUIDE.md`（各模块注意事项、改动禁区红线）

## 6. 评估异步链路时序红线

流程评价异步运行、**不走 InferenceSync 对齐中间件**（push_direct 直推 ws）。以下 4 条时序不可违反，改动评估层前必须逐条核对：

1. **直推绕对齐**：评估报告（`segment_report`/`segment_report_stream`）走 `push_direct`→`ws_handler.push_text`→`send_text`，**严禁走 InferenceSync/`push_display`/`_build_batch` 对齐**（仅 `flow_start`/`flow_end` 走 `push_sync`→对齐）。评估的"时钟对齐"由后端 `wait_playback_reached` 阻塞等待实现，非中间件 batch。
2. **模块齐到再提取**：`flow_end` 后必须 `_wait_all_modules(end_sec, timeout=90)` 等所有推理模块进度 `min >= end_sec` 才提取 keymoment；读进度用 `hgetall KEY_PROGRESS`（勿按模块名 `hget`，历史 bug 恒 0 空等）。
3. **前端播到再推**：推评估 chunk 前必须 `wait_playback_reached(end_sec-0.5, timeout=60)` 等前端 `currentPlaybackSec`（WS 回传 playback_progress，非 globalSec）到流程结束；`has_playback_waited` 一次性，首个 chunk 前等一次。
4. **逐 token 流式 + 完成态**：Qwen `TextIteratorStreamer` 逐 token → `push_direct("segment_report_stream", chunk)` 多次 → 前端 `streamBuffer` 累积 → ReportPanel `setInterval 60ms` typewriter 逐字；`segment_report` 不覆盖 `streamBuffer`；typewriter 追赶到末尾且 `reportText` 到达 → 置 `streaming=false` 切完成态（显分数/进度条/完成图标）。

## 7. 改动禁区

- **评估异步链路时序**：§6 四条红线不可破坏（`_wait_all_modules` 时序、`wait_playback_reached` 等待、`push_direct` 绕对齐、逐 token 流式）。详见 `docs/DEVELOPMENT_GUIDE.md` §11.5。
- **核电前端可视化样式**：严禁修改前端可视化样式（CSS/布局/配色），只处理逻辑/链路问题。
- **fMP4 PTS**：`PTS=帧序/fps`，`sb.mode='sequence'` 忽略 PTS 按 append 顺序造时戳，勿改编码时戳逻辑。

## 8. 时间戳命名规范（前后端统一）

以下时间戳**值同（源视频秒）**，但语义角色不同，前后端须统一认知：

| 命名 | 语义 | 出处 |
|---|---|---|
| `localSec` | 模块内推理进度（per-source，`frame_count/fps`，`update_module_time` 写） | 各模块循环内 |
| `globalSec` | 结构化数据经 InferenceSync 对齐后推送的全局时间（=对齐闸门 `min(各视角进度)`，**同一概念**，不冲突） | `inference_sync` 对齐推送 + batch 字段 |
| `PTS` | 视频流 fMP4 帧时间戳（`=帧序/fps`，ffmpeg `-r fps` 生成） | `vis_encoder` |
| `currentPlaybackSec` | 前端播放进度（`front.currentTime`，主时钟） | `useWS` + `VideoPanel` |

- **模块内 `localSec`** 经 InferenceSync 对齐推送后即 **`globalSec`**（值不变，对齐只筛选 `localSec <= 闸门 globalSec` 推送，不改值）
- **视频流 `PTS`** 独立通道（`vis_stream`，不经 InferenceSync），前端 MSE `sequence` 模式按 append 顺序造时戳（隐式 ≈ PTS）
- 结构化数据 `globalSec` 与对齐闸门 `globalSec` 本来就是一个意思（对齐后的全局时间），**不冲突**

