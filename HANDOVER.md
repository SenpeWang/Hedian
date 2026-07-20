# 核电规程合规判定项目任务交接文档 (HANDOVER)

本交接文档适用于 `/home/wangshengping/Hedian/A_DemoSrc` 目录下的多模态合规研判系统。

---

## 一、 系统整体架构

```
main.py (协调器)
├── voice 进程    → 语音转文字 + key_moment提取
├── tracker 进程  → 目标检测+跟踪+举手检测(内部调用 GazeModule)
├── behavior 进程 → 行为检测(手指指屏幕)
└── web 进程      → Flask + SSE推送 + ModuleSync对齐 + Evaluation评估
```

> gaze 不是独立进程，作为 `GazeModule` 嵌入 tracker 进程内调用。

**双流机制（严格解耦）：**
- **推理流（前端可视化）**：`push_display(source, data)` → Redis Stream → ModuleSync 按 globalSec 对齐打包 batch → SSE → 前端
- **事件流（模型间通信）**：`push_event(EventTopic.XXX, data, ts)` → EventBus → 规则订阅（监护制/自唱票/信息通报/人员状态）

---

## 二、 推理流统一格式（所有模块统一）

### 数据格式

```json
{
  "source": "voice|tracking|gaze|video|progress|behavior",
  "localSec": 12.5,
  "tag": "text|status|alert|frame|progress",
  "data": { ... 纯展示字段 ... }
}
```

| 字段 | 说明 | 必填 |
|------|------|------|
| `source` | 模块来源，前端 batch 路径按此分组分发 | 是 |
| `localSec` | 对齐时间戳，有=需要对齐，无=立即推 | 对齐数据必填 |
| `tag` | 事件类型，区分同模块的不同事件 | 是 |
| `data` | 纯展示业务数据，**严禁夹带** source/type/event/localSec 等元字段 | 是 |

### 各模块推理流字段

| 模块 | source | tag | data |
|------|--------|-----|------|
| **voice** | `"voice"` | `"text"` | `{"text": "请求监护"}` |
| **tracking** | `"tracking"` | `"SUPERVISOR_STATUS"` | `{"state": "监护中", "operator": "ROAD1", "distance_px": 150}` |
| **behavior** | `"behavior"` | `"HAND_RAISED"` | `{"state": "举手", "operator": "ROAD1"}` |
| **tracking** | `"tracking"` | `"ROLE_ASSIGNED"` | `{"roles": {"LEADER": 3, ...}}` |
| **tracking** | `"tracking"` | `"PEOPLE_COUNT_UPDATE"` | `{"count": 3, "state": "主控室仅有1人", "state_alert": "..."}` |
| **gaze** | `"gaze"` | `"gaze_status"` | `{"has_heads": true, "any_in_roi": true, "heads_count": 2}` |
| **gaze** | `"gaze"` | `"GAZE_ALERT"` | `{"state": "无人注视盘台", "away_duration": 65.2}` |
| **gaze** | `"gaze"` | `"ATTENTION_RESULT"` | `{"has_turned": true, "displacement": 120.5}` |
| **video** | `"video"` | `"frame"` | `{"frame_data": "/9j/4...", "frame_id": 1234}` |
| **progress** | `"progress"` | `"progress"` | `{"label": "gaze", "pct": 15.0}` |

### 代码示例

```python
# voice
self.push_display("voice", {"localSec": ts, "tag": "text", "data": {"text": text}})

# tracker - SUPERVISOR_STATUS
self.push_display("tracking", {
    "localSec": round(ts, 2),
    "tag": "SUPERVISOR_STATUS",
    "data": {"state": state_label, "operator": road_name, "distance_px": int(d)}
})

# gaze
self._display_fn("gaze", {
    "localSec": round(ts, 2),
    "tag": "GAZE_ALERT",
    "data": {"state": "无人注视盘台", "away_duration": away_dur}
})

# video（统一走 source="video" + tag="frame"，不再使用 video_frame 特殊路径）
self.push_display("video", {
    "localSec": round(ts, 2),
    "tag": "frame",
    "data": {
        "frame_data": base64.b64encode(jpeg.tobytes()).decode("utf-8"),
        "frame_id": frame_count,
    }
})

# progress
self.push_display("progress", {
    "localSec": round(ts, 2),
    "tag": "progress",
    "data": {"label": "gaze", "pct": round(gaze_pct, 1)}
})
```

### 评估层单事件（绕过对齐，直接推送）

`FlowEvaluationManager` 通过 `display_buffer.push_display` 推送的事件由 `ModuleSync.push_display` 直接转发到 SSE（不经过 Redis Stream 对齐），前端走单事件路径识别：

| source | 时机 | 主要字段 |
|--------|------|---------|
| `flow_start` | 流程开始 | `flowId, flow_type, flow_start_sec` |
| `flow_end` | 流程结束 | `flowId, flow_type, flow_end_sec, flow_continue_sec` |
| `segment_report_stream` | 大模型流式评估 | `flowId, chunk` |
| `segment_report` | 评估完成 | `flowId, flow_type, score, report_text` |
| `done` | 流水线结束 | （空） |

---

## 三、 batch 对齐与推送（核心对齐逻辑）

### ModuleSync 对齐流程

```
每帧(1/30s):
  1. _compute_global_sec()
     = min(voice进度, tracker进度, gaze进度)
  2. 从 Redis Stream 读取所有事件
  3. 按 localSec 排序
  4. 只取 localSec <= global_sec 的事件
  5. 按 source 分组打包为 batch
  6. SSE → 前端
```

### batch 结构

```json
{
  "globalSec": 12.5,
  "voice": [
    {"localSec": 12.5, "tag": "text", "data": {"text": "请求监护"}}
  ],
  "tracking": [
    {"localSec": 12.5, "tag": "SUPERVISOR_STATUS", "data": {"state": "监护中", "operator": "ROAD1"}}
  ],
  "gaze": [
    {"localSec": 12.5, "tag": "gaze_status", "data": {"has_heads": true}}
  ],
  "video": [
    {"localSec": 12.5, "tag": "frame", "data": {"frame_data": "/9j/4..."}}
  ],
  "progress": [
    {"localSec": 12.5, "tag": "progress", "data": {"label": "gaze", "pct": 15.0}}
  ]
}
```

### 完成检测

```
ModuleSync 自主检测 global_sec 停滞超过30秒 → 视为所有模块推理完成
→ 刷新剩余事件
→ 推送 done 信号
→ 前端收到 done → 关闭 SSE → 显示"重新测试"按钮
```

### 关键代码

```python
# ModuleSync._push_events_up_to
batch = {"globalSec": global_sec}
for ev in events_to_push:
    source = ev.get("type", "unknown")
    if source not in batch:
        batch[source] = []
    batch[source].append({"localSec": ev.get("localSec"), "tag": ev.get("tag"), "data": ev.get("data")})
self._do_push(batch)
```

---

## 四、 前端处理逻辑

### SSE 接收

```javascript
evtSource.onmessage = function(e) {
    const d = JSON.parse(e.data);
    handleEvent(d);
    if (d.source === 'done') { evtSource.close(); }
};
```

### handleEvent 路由

```javascript
function handleEvent(d) {
    if (d.globalSec !== undefined) {
        // 这是 batch → 按 source 分发
        if (d.voice) for (const ev of d.voice) addVoiceItem(ev);
        if (d.tracking) for (const ev of d.tracking) addDetEvent(ev);
        if (d.gaze) for (const ev of d.gaze) addGazeEvent(ev);
        if (d.video) for (const ev of d.video) streamImg.src = 'data:image/jpeg;base64,' + ev.data.frame_data;
        if (d.progress) for (const ev of d.progress) updateProgress(ev);
        return;
    }
    // 单事件（flow_start/flow_end/segment_report/done 等评估层事件，绕过对齐直接推送）
    if (d.source === 'flow_start') addFlowStart(d);
    else if (d.source === 'flow_end') addFlowEnd(d);
    else if (d.source === 'segment_report') addSegCard(d);
    else if (d.source === 'segment_report_stream') addSegStream(d);
    else if (d.source === 'progress') updateProgress(d);
    else if (d.source === 'video_start') { /* 启动音频 */ }
    else if (d.source === 'done') { /* 完成处理 */ }
}
```

### 前端函数

| 函数 | 处理 | 数据来源 |
|------|------|---------|
| `addVoiceItem(d)` | 语音文本显示 | `d.data.text` |
| `addDetEvent(d)` | 跟踪状态显示 | `d.data.state`, `d.data.operator` |
| `addGazeEvent(d)` | 凝视告警显示 | `d.tag === 'GAZE_ALERT'` |
| `updateProgress(d)` | 进度条更新 | `d.data.label`, `d.data.pct` |
| `streamImg.src` | 视频帧更新 | `d.data.frame_data` |

---

## 五、 事件流通信

### EventTopic 定义

```python
class EventTopic:
    # Voice -> Rules
    VOICE_KEY_MOMENT = "voice.key_moment"       # {localSec, key_moment}

    # Tracker -> Rules
    TRACKER_PROXIMITY = "tracker.proximity"     # {localSec, state, operator, distance_px}
    TRACKER_HEADCOUNT = "tracker.headcount"     # {localSec, count}

    # Behavior -> Rules（举手检测嵌入 tracker 调用，但归属 behavior）
    BEHAVIOR_HAND_RAISED = "behavior.hand_raised"  # {localSec, operator}

    # Gaze -> Rules
    GAZE_ALERT = "gaze.alert"                   # {localSec, state, away_duration, ...}
    GAZE_ATTENTION = "gaze.attention"           # {localSec, has_turned, displacement, ...}

    # Rules -> Evaluation
    FLOW_STARTED = "flow.started"               # {flow_id, flow_type, flow_start_sec}
    FLOW_ENDED = "flow.ended"                   # {flow_id, flow_type, flow_end_sec}
```

> 已删除的死代码 EventTopic（无订阅者）：`TRACKER_SUPERVISION_BOUND`、`TRACKER_SUPERVISION_END`、`TRACKER_ROLE_ASSIGNED`、`BEHAVIOR_FINGER_SCREEN`、`PIPELINE_STATUS`、`PIPELINE_PROGRESS`、`TRACKER_HAND_RAISED`（已改名 `BEHAVIOR_HAND_RAISED`）。

### 规则订阅

| 规则 | 订阅事件 | 来源 |
|------|---------|------|
| 监护制 | `VOICE_KEY_MOMENT`, `BEHAVIOR_HAND_RAISED`, `TRACKER_PROXIMITY` | voice, behavior, tracker |
| 自唱票 | `VOICE_KEY_MOMENT` | voice |
| 信息通报 | `VOICE_KEY_MOMENT`, `BEHAVIOR_HAND_RAISED`, `GAZE_ATTENTION` | voice, behavior, gaze |
| 人员状态 | `GAZE_ALERT`, `TRACKER_HEADCOUNT` | gaze, tracker |
| 评价层 | `FLOW_STARTED`, `FLOW_ENDED` | rules |
| Tracker 模块 | `FLOW_STARTED`, `FLOW_ENDED` | rules（用于本地维护 `_supervision_active` 标志） |

> 三大流程（监护/自唱票/信息通报）相互独立，各自拥有独立的开始与结束条件，不存在跨流程联动收尾。

---

## 六、 各模块事件流

### Voice

```python
# 展示流：完整转录文本
self.push_display("voice", {"localSec": local_sec, "tag": "text", "data": {"text": text}})

# 事件流：关键事件(监护/执行/核对/九字码)
self.push_event(EventTopic.VOICE_KEY_MOMENT, {
    "localSec": local_sec,
    "key_moment": key_moment
}, ts=local_sec)
```

### Tracker

```python
# 展示流：跟踪状态
self.push_display("tracking", {
    "localSec": round(ts, 2),
    "tag": "SUPERVISOR_STATUS",
    "data": {"state": state_label, "operator": road_name, "distance_px": int(d)}
})

# 事件流：距离变化
self.push_event(EventTopic.TRACKER_PROXIMITY, {
    "localSec": round(ts, 2),
    "state": state_label, "operator": road_name, "distance_px": int(d)
}, ts=ts)
```

### Gaze（嵌入 Tracker，通过 GazeProcessor 调用）

```python
# 展示流：凝视告警
self._display_fn("gaze", {
    "localSec": round(ts, 2),
    "tag": "GAZE_ALERT",
    "data": {"state": "无人注视盘台", "away_duration": away_dur}
})

# 事件流：凝视告警
self._event_bus.publish(EventTopic.GAZE_ALERT, event, ts=ts)
```

---

## 七、 前后端交互流程

```
用户点击"开始测试"
  → POST /start → 清理 Redis 缓存 → 设置 pipeline:start_signal
  → 各模块收到信号 → 开始推理
  → 每个模块 push_display(带localSec) 到 Redis Stream
  → ModuleSync 每帧:
      1. 计算 global_sec = min(voice, tracker, gaze)
      2. 收集 localSec <= global_sec 的事件
      3. 按 source 分组打包 batch
      4. SSE → 前端
  → 前端收到 batch → 按 source 分发渲染
  → 所有模块推理完成 → global_sec 停滞30秒
  → ModuleSync 推送 done → 前端结束
```

---

## 八、 关键文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 入口，启动所有进程 |
| `core/module_sync.py` | ModuleSync 对齐引擎，global_sec 计算，batch 打包；`push_display` 直接转发评估层单事件 |
| `core/inference_bus.py` | 推理流写入端（模块进程使用），统一格式 `{source, localSec, tag, data}` |
| `core/display_buffer.py` | 包装器，writer_only=True→InferenceBus, False→ModuleSync |
| `core/event_bus.py` | 事件总线，EventTopic 定义（仅保留有订阅者） |
| `core/base_module.py` | 模块基类，push_display/push_event/update_progress（1秒节流） |
| `modules/voice/voice_module.py` | 语音模块 |
| `modules/tracker/tracker_module.py` | 跟踪模块（内部调用 GazeModule + HandRaiser），含可视化帧推送 |
| `modules/gaze/gaze_module.py` | GazeModule 凝视处理器（嵌入 tracker 进程） |
| `modules/behavior/hand_raiser.py` | 举手检测器（供 tracker 调用） |
| `modules/behavior/behavior_module.py` | 行为模块（手指指屏幕检测） |
| `rules/supervision_rule.py` | 监护制状态机 |
| `rules/rule_base.py` | 规则注册表 |
| `evaluation/flow_evaluation_manager.py` | 流程评估管理器，推送 flow_start/flow_end/segment_report |
| `web/http_server.py` | Flask 服务器，SSE 端点（已移除 /stream /stream2 MJPEG 路由） |
| `web/sse_handler.py` | SSE 推送处理器 |
| `web/static/index.html` | 前端页面 |

---

## 九、 运行方式

```bash
# 启动（GPU 1）
cd /home/wangshengping/Hedian/A_DemoSrc
PYTHONUNBUFFERED=1 screen -dmS hedian /home/wangshengping/myconda/envs/sp_hedian/bin/python main.py --gpu 1

# 访问
# http://10.152.57.223:5002
# 点击"开始测试"运行推理

# 停止
pkill -f 'main.py'
fuser -k 5002/tcp
```

---

## 十、 代码更新流程

```bash
# 1. 本地写 Python 脚本
cat > /tmp/fix.py << 'PYEOF'
# 纯 Python，无 shell 转义问题
with open("/path/to/file") as f:
    c = f.read()
c = c.replace("old", "new")
with open("/path/to/file", "w") as f:
    f.write(c)
print("OK")
PYEOF

# 2. scp 到远端
scp -o StrictHostKeyChecking=no /tmp/fix.py wangshengping@10.152.57.223:/tmp/

# 3. ssh 执行
ssh -o StrictHostKeyChecking=no wangshengping@10.152.57.223 "python3 /tmp/fix.py"

# 4. 验证语法
ssh -o StrictHostKeyChecking=no wangshengping@10.152.57.223 \
  "python3 -c 'compile(open(\"/home/wangshengping/Hedian/A_DemoSrc/xxx.py\").read(), \"xxx.py\", \"exec\"); print(\"OK\")'"

# 5. 提交推送
ssh -o StrictHostKeyChecking=no wangshengping@10.152.57.223 \
  "cd /home/wangshengping/Hedian/A_DemoSrc && git add . && git commit -m 'xxx' && git push"
```

**注意：**
- 不要用 `sed` 直接修改远端文件（引号嵌套容易出错）
- 不要用 `python3 -c` 内联脚本（引号冲突问题）
- 必须用 `cat > file << 'PYEOF'`（单引号PYEOF保证bash不解析）
- 改完必须验证语法：`compile()`