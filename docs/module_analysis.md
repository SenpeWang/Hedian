# 模块分析文档

本文档对核电站监护制合规检测系统的所有 Python 模块进行详细分析，包括需求说明、功能描述和实现逻辑问题。

---

## main.py

### 需求

- **职责**：核电站监护制合规检测系统主流程控制器，协调 Voice、MOT、Qwen 三大模块的并行执行
- **输入**：
  - 视频文件（`Hedian_data/data/camFRONT.mpg`）
  - 音频文件（从视频提取）
- **输出**：
  - 检测结果（JSON 格式）
  - 评估报告（Qwen 生成）
  - 实时视频流（通过 WebSocket/SSE 推送到前端）

### 功能

- `run_pipeline()`：启动完整检测流程，包括：
  - 视频 FPS 检测
  - Voice 线程异步启动（Whisper 语音转录）
  - MOT 线程异步启动（目标跟踪与行为检测）
  - Qwen 报告生成
- `run_mot(result_dir)`：MOT 线程入口，处理视频帧：
  - YOLO 目标检测
  - 多目标跟踪
  - 监护状态机管理（idle/requesting/bound）
  - 举手检测（YOLOPose）
  - 距离计算与监护绑定判断
- `run_voice(result_dir)`：Voice 线程入口：
  - ffmpeg 音频提取
  - Whisper large-v3 模型加载与转录
  - 语音事件构建与意图分类
- `on_seg(words)`：Voice 分段回调，处理转录结果：
  - 分句处理
  - 意图分类（监护请求、监护确认、核对确认等）
  - 设备码提取
  - 事件推送到 FlowEngine
- `_push_event(etype, data)`：统一事件推送接口
- `_gen_report(result_dir)`：生成 Qwen 总结报告

### 实现逻辑问题

1. **硬编码路径问题**：
   - `VIDEO_PATH` 和 `DATA_ROOT` 使用硬编码路径，缺乏配置灵活性
   - 建议：使用配置文件或环境变量管理路径

2. **全局状态管理**：
   - 大量使用全局变量（`_voice_supervision_request`, `_last_device_code` 等）
   - 建议：封装为类属性，提高代码可维护性

3. **线程安全问题**：
   - `voice_events_all` 等共享变量使用 `_data_lock` 保护，但部分访问路径可能遗漏
   - 建议：全面审查锁的使用范围

4. **时间同步机制**：
   - Voice 和 MOT 的时间戳同步依赖人工设定的容差（-3s ~ +1s）
   - 建议：使用更精确的帧级同步机制

5. **异常处理不完善**：
   - 部分代码块使用裸 `except Exception`，可能掩盖真正的问题
   - 建议：细化异常类型，增加日志记录

6. **资源释放问题**：
   - `cap.release()` 在 `finally` 块中，但部分临时文件可能未清理
   - 建议：增加临时文件清理逻辑

---

## vis.py

### 需求

- **职责**：Flask Web 可视化层，提供前端界面和实时数据推送
- **输入**：
  - 视频流（MJPEG）
  - 事件流（SSE）
  - 音频文件
- **输出**：
  - HTML 前端页面
  - 实时视频流（`/stream`, `/stream2`）
  - 事件流（`/events`）
  - 状态信息（`/status`）

### 功能

- `index()`：返回主页面 HTML
- `start()`：启动流水线，检查状态避免重复启动
- `events()`：SSE 事件流端点，支持多客户端
- `stream()`：视频流端点（MJPEG 格式）
- `stream2()`：第二路视频流（预留，当前未完全实现）
- `audio()`：音频文件服务
- `status()`：返回当前流水线状态
- `set_pipeline_runner(fn)`：注入主流程启动函数
- `_HTML_PAGE`：内嵌的完整前端页面（HTML + CSS + JavaScript）

### 实现逻辑问题

1. **硬编码 HTML**：
   - 整个前端页面作为 Python 字符串硬编码，难以维护
   - 建议：分离为独立模板文件（如 Jinja2）

2. **SSE 客户端管理**：
   - 使用列表存储客户端队列，删除操作是 O(n)
   - 建议：使用集合或字典优化

3. **视频流异常处理**：
   - `stream2` 路由中异常处理过于简单，可能导致连接挂起
   - 建议：增加更完善的超时和重连机制

4. **缺少认证授权**：
   - 所有端点都是公开的，无访问控制
   - 建议：增加基本的身份验证机制

5. **静态资源服务**：
   - 当前未充分利用 Flask 的 static 文件夹
   - 建议：将 CSS/JS 分离到 static 目录

---

## Aggregator/event_aggregator.py

### 需求

- **职责**：多模块事件的按时间戳排序和统一推送
- **输入**：
  - Voice 模块事件（带 time_sec）
  - MOT 模块事件（带 time_sec）
- **输出**：
  - 按时间排序的事件流（通过 SSE 推送到前端）

### 功能

- `EventAggregator` 类：
  - `__init__(fps)`：初始化优先队列和线程
  - `add_event(module, event)`：接收模块事件并加入优先队列
  - `start()`：启动聚合线程
  - `stop()`：停止聚合线程并刷新剩余事件
  - `_aggregation_loop()`：主循环，按 fps 频率推送事件
  - `_push_events_up_to(current_time)`：推送时间戳 <= 当前时间的事件
  - `_flush_remaining_events()`：推送队列中剩余的所有事件
  - `_do_push(module, event)`：执行实际的事件推送（到 SSE）
  - `get_stats()`：获取聚合器统计信息
  - `clear()`：清空事件队列

### 实现逻辑问题

1. **推送目标硬编码**：
   - `_do_push` 方法直接导入 `vis` 模块的 `_sse_clients`
   - 建议：通过依赖注入解耦

2. **时间计算依赖帧率**：
   - 使用 `frame_count / fps` 计算当前时间，可能与实际视频时间有偏差
   - 建议：使用视频实际播放时间

3. **缺少事件去重**：
   - 相同时间戳的多个事件可能产生排序不确定性
   - 建议：增加事件唯一标识和去重机制

4. **内存使用**：
   - 优先队列可能积累大量事件（如果推送线程阻塞）
   - 建议：增加队列大小限制和溢出处理

5. **线程停止超时**：
   - `stop()` 方法使用 2 秒超时，可能不够充分
   - 建议：增加可配置的超时时间

---

## Flow/flow_engine.py

### 需求

- **职责**：实时流程判断引擎，管理监护制流程和自唱票子流程的生命周期
- **输入**：
  - Voice 信号（SUPERVISION_REQUEST, EXPERIMENT_END 等）
  - MOT 信号（SUPERVISION_START, SUPERVISION_END 等）
- **输出**：
  - 流程开始/结束事件
  - 分段评估报告（JSON）

### 功能

- `FlowEngine` 类：
  - `__init__(result_dir, fps, model_path, progress_cb, push_event_fn)`：初始化流程引擎
  - `on_voice_signal(signal)`：接收 Voice 模块信号
  - `on_mot_signal(signal)`：接收 MOT 模块信号
  - `_start_supervision_flow(start_signal, source)`：启动监护制流程
  - `_close_supervision_flow(end_signal, source)`：关闭监护制流程
  - `_start_self_ticket_flow(start_signal, source)`：启动自唱票子流程
  - `_close_self_ticket_flow(end_signal, source)`：关闭自唱票子流程
  - `_close_all_flows(end_signal, source)`：关闭所有活跃流程
  - `_attach_buffered_events(flow)`：将缓冲的事件附加到流程
  - `_load_events_from_result(start_sec, end_sec)`：从结果文件加载事件
  - `_evaluate_and_save(flow)`：评估流程并保存报告
  - `_do_evaluate(flow_data)`：执行实际评估（在线程池中）
  - `_on_eval_done(future, flow_id)`：评估完成回调
  - `_save_segment_reports()`：保存分段报告
  - `finalize()`：视频结束时的清理操作
  - `get_voice_summary()`：获取语音数据摘要
  - `get_mot_summary()`：获取跟踪数据摘要

### 实现逻辑问题

1. **流程触发条件不完整**：
   - 自唱票流程的触发条件（手指指向、九字码重复）尚未完全实现
   - 建议：完善 POINTING_DETECTED 和 SELF_TICKET_CODE_REPEAT 的信号生成

2. **事件缓冲策略**：
   - 使用简单的列表缓冲，可能积累大量事件
   - 建议：增加缓冲大小限制和时间窗口

3. **评估并发控制**：
   - 使用 `ThreadPoolExecutor(max_workers=2)`，但大模型推理可能占用大量资源
   - 建议：根据硬件资源动态调整 worker 数量

4. **错误处理**：
   - 评估失败时返回默认报告，但缺少详细的错误日志
   - 建议：增加错误追踪和重试机制

5. **父子流程关系**：
   - 当前只记录 sub_flow_ids，缺少更丰富的关系信息
   - 建议：增加流程嵌套深度限制和循环检测

---

## Flow/flow_extractor.py

### 需求

- **职责**：流程关键信息提取器，从结果 JSON 文件中提取流程数据
- **输入**：
  - `Voice/Voice_key_moments.json`
  - `Mot/Mot_key_moments.json`
  - `Qwen/Qwen_segment_reports.json`
- **输出**：
  - 结构化的流程数据（监护制流程、自唱票流程）

### 功能

- `load_result_data(result_dir, fps)`：加载所有结果数据
- `_add_frame_id(event, fps)`：为事件添加 frame_id 字段
- `extract_supervision_flows(voice_data, mot_data, fps)`：提取监护制流程
- `extract_self_ticket_flows(voice_data, mot_data, fps)`：提取自唱票流程（未实现）
- `evaluate_flows(result_dir, fps)`：评估所有流程

### 实现逻辑问题

1. **自唱票流程未实现**：
   - `extract_self_ticket_flows` 函数为空（`pass`）
   - 建议：尽快实现自唱票流程的提取逻辑

2. **流程匹配算法简单**：
   - 使用简单的贪心算法匹配请求和结束事件
   - 建议：考虑更复杂的场景（如重叠流程、嵌套流程）

3. **默认时长硬编码**：
   - 无结束事件时默认使用 60 秒时长
   - 建议：根据实际视频长度或配置调整

4. **缺少流程验证**：
   - 不验证提取的流程是否合理（如开始时间 > 结束时间）
   - 建议：增加数据验证和清洗

---

## Qwen/evaluate.py

### 需求

- **职责**：分段评估模块，使用 Qwen 大模型评估单个流程
- **输入**：
  - 流程数据（voice_events, mot_events, 时间范围等）
  - Qwen 模型路径
- **输出**：
  - 分段评估报告（分数、评级、详细评价）

### 功能

- `evaluate_flow_segment(flow_data, model_path, progress_cb, stream_cb)`：评估单个流程
- `_build_supervision_prompt(flow_data)`：构建监护制流程的评估提示
- `_build_self_ticket_prompt(flow_data)`：构建自唱票流程的评估提示
- `_extract_score(text, max_score)`：从模型输出中提取分数
- `_extract_grade(text)`：从模型输出中提取评级

### 实现逻辑问题

1. **模型重复加载**：
   - 每个流程评估都重新加载模型，效率低下
   - 建议：实现模型单例或缓存机制

2. **提示词硬编码**：
   - 评分标准和提示词作为字符串硬编码
   - 建议：分离到配置文件，支持动态调整

3. **评分解析脆弱**：
   - 依赖正则表达式解析模型输出，可能因格式变化而失败
   - 建议：使用结构化输出（如 JSON）或更健壮的解析逻辑

4. **流式回调异常处理**：
   - `stream_cb` 回调异常可能导致线程挂起
   - 建议：增加异常捕获和超时机制

5. **温度参数固定**：
   - `temperature=0.3` 固定，不适合所有场景
   - 建议：根据评估类型动态调整

---

## Qwen/QwenEvaluate.py

### 需求

- **职责**：大模型总结评估模块，基于分段报告生成总结
- **输入**：
  - 分段评估报告列表
  - 语音数据摘要
  - 跟踪数据摘要
- **输出**：
  - 总结报告（总分、评级、详细评价）

### 功能

- `generate_summary_report(segment_reports, voice_summary, mot_summary, model_path, progress_cb)`：生成总结报告
- `_build_summary_prompt(segment_reports, voice_summary, mot_summary)`：构建总结提示词

### 实现逻辑问题

1. **模型重复加载**：
   - 与 `evaluate.py` 相同，每次调用都重新加载模型
   - 建议：统一模型管理，实现共享实例

2. **评分权重硬编码**：
   - 监护制度 60 分、自唱票 40 分的权重固定
   - 建议：支持配置化权重

3. **缺少报告缓存**：
   - 相同输入每次都重新生成
   - 建议：增加缓存机制，避免重复计算

4. **输出格式依赖模型**：
   - 依赖模型输出特定格式（总分、评级）
   - 建议：使用更结构化的输出格式

---

## Voice/voice.py

### 需求

- **职责**：语音处理模块，将视频音频转录为文本并提取关键事件
- **输入**：
  - 音频文件（WAV 格式，16kHz）
- **输出**：
  - `Voice_key_moments.json`（关键事件）
  - `Voice_full_text.json`（完整转录文本）

### 功能

- `main(progress_cb, segment_cb)`：主函数：
  - 音频加载和去噪（noisereduce）
  - Whisper large-v3 模型加载
  - 滑动窗口分段转录
  - 幻觉移除和后处理
  - 事件构建和保存
- `transcribe_segmented(audio, sr, model, prompt, ...)`：滑动窗口分段转录
- `denoise_nr_stationary(audio, sr)`：稳态去噪
- `_detect_speech_end(audio, sr)`：检测语音结束位置
- `remove_hallucinations(words)`：移除连续重复的短词（Whisper 幻觉）
- `apply_corrections(text)`：应用后处理纠错规则
- `classify_intent(text, prev_intent)`：意图分类（基于关键词规则）
- `build_voice_events(words)`：构建语音事件列表
- `build_key_moments(voice_events)`：提取关键时间点
- `build_merged_key_moments(voice_events)`：合并为统一结构

### 实现逻辑问题

1. **纠错规则维护困难**：
   - 140+ 条纠错规则硬编码，难以维护
   - 建议：分离到外部配置文件或数据库

2. **意图分类规则简单**：
   - 基于关键词匹配，可能误判
   - 建议：引入更复杂的 NLP 模型或规则引擎

3. **窗口参数硬编码**：
   - `window_sec=25`, `overlap_sec=5` 等参数固定
   - 建议：支持配置化，适应不同音频长度

4. **缺少说话人分离**：
   - 无法区分不同说话人
   - 建议：集成说话人分离（diarization）功能

5. **设备码提取局限**：
   - 仅支持预定义的设备码模式
   - 建议：使用更通用的实体识别方法

6. **全局变量使用**：
   - `RAW_AUDIO`, `OUTPUT_DIR` 等作为全局变量
   - 建议：封装为类或使用依赖注入

---

## MOT/detector/detector.py

### 需求

- **职责**：目标检测器，基于 YOLO 进行人员检测和姿态估计
- **输入**：
  - 视频帧（numpy array）
- **输出**：
  - 检测框列表（person 类别）
  - 姿态关键点（YOLOPose）

### 功能

- `DetectionConfig` 数据类：检测器配置
- `ObjectDetector` 类：
  - `__init__(model_path, pose_model_path, ...)`：加载 YOLO 和 YOLOPose 模型
  - `detect(frame)`：执行目标检测，返回 person 检测框
  - `detect_with_pose(frame)`：同 detect（不主动做 Pose）
  - `detect_pose(frame)`：YOLOPose 姿态估计
  - `check_hand_raised(keypoints, conf_thr)`：举手检测（静态方法）

### 实现逻辑问题

1. **模型路径硬编码**：
   - 默认模型路径使用相对路径计算
   - 建议：使用配置文件管理模型路径

2. **姿态模型可选但功能依赖**：
   - 举手检测依赖姿态模型，但模型是可选加载的
   - 建议：明确依赖关系，增加运行时检查

3. **举手检测阈值固定**：
   - `WRIST_MARGIN = 15` 等阈值硬编码
   - 建议：支持配置化，适应不同场景

4. **类别过滤硬编码**：
   - `if class_id == 0` 假设 person 是类别 0
   - 建议：使用配置或模型元数据获取类别信息

5. **缺少检测结果缓存**：
   - 每帧都重新推理，无缓存机制
   - 建议：对于静态场景可考虑缓存优化

---

## MOT/tracker/tracker.py

### 需求

- **职责**：动态多目标跟踪器，实现零 ID 切换的精确跟踪
- **输入**：
  - 检测框列表
  - 视频帧（用于初始化）
- **输出**：
  - 跟踪轨迹列表（STrack）

### 功能

- `TrackerConfig` 数据类：跟踪器配置
- `TrackState` 枚举：跟踪状态（NEW, TRACKED, LOST, REMOVED）
- `STrack` 类：单个跟踪轨迹：
  - 位置历史、速度历史、框历史
  - 稳定性评分
  - 角色分配（LEADER/ROAD1/ROAD2）
  - 相似度计算（`compute_similarity`）
- `DynamicTracker` 类：
  - `__init__()`：初始化跟踪器
  - `_assign_roles(tracks)`：按工位坐标分配角色
  - `detect_target_count()`：动态检测目标数量
  - `initialize_tracker(detections)`：初始化跟踪器
  - `compute_cost_matrix(tracks, detections)`：计算匹配代价矩阵
  - `match_and_update(tracks, detections)`：匹配并更新跟踪
  - `track(frame, detections)`：主跟踪函数
  - `get_statistics()`：获取统计信息
  - `get_track_by_role(role)`：按角色获取跟踪

### 实现逻辑问题

1. **工位坐标依赖**：
   - 角色分配依赖 `WORKSTATIONS` 全局变量
   - 建议：通过配置注入，支持不同场景

2. **初始化逻辑复杂**：
   - 使用 30 帧缓冲和多种条件判断，逻辑复杂
   - 建议：简化初始化逻辑，增加注释

3. **相似度计算权重固定**：
   - 位置、大小、速度等权重硬编码
   - 建议：支持配置化权重

4. **丢失轨迹处理**：
   - `max_lost_frames = 200` 固定
   - 建议：根据帧率动态计算

5. **ID 分配策略**：
   - 使用简单递增 ID，重启后 ID 重新从 1 开始
   - 建议：考虑使用 UUID 或时间戳前缀

6. **角色分配一次性**：
   - 角色只在初始化时分配，不支持动态调整
   - 建议：支持角色重新分配（如人员交换位置）

---

## MOT/utils/visualizer.py

### 需求

- **职责**：可视化模块，绘制跟踪框和角色标签
- **输入**：
  - 视频帧
  - 跟踪轨迹列表
- **输出**：
  - 可视化后的帧

### 功能

- `Visualizer` 类：
  - `__init__()`：初始化
  - `draw_tracks(frame, tracks, show_id, show_conf)`：绘制跟踪框和标签
  - `draw_info(frame, info)`：绘制信息文本

### 实现逻辑问题

1. **功能过于简单**：
   - 仅支持绘制框和标签，缺少更丰富的可视化（如轨迹、热力图）
   - 建议：增加更多可视化选项

2. **颜色配置硬编码**：
   - `ROLE_COLORS` 字典硬编码
   - 建议：支持配置化颜色

3. **字体和大小固定**：
   - 使用固定字体和大小
   - 建议：支持根据帧分辨率自适应

4. **缺少抗锯齿**：
   - OpenCV 默认线条可能有锯齿
   - 建议：使用亚像素精度绘制

---

## MOT/main.py

### 需求

- **职责**：MOT 独立运行入口，处理视频并输出结果
- **输入**：
  - 视频文件（通过配置）
- **输出**：
  - 结果视频（MP4）
  - 结果 JSON

### 功能

- `MultiObjectTracker` 类：
  - `__init__()`：初始化检测器、跟踪器、可视化器
  - `_check_environment()`：检查环境和文件
  - `process_video(max_frames)`：处理视频：
    - 视频读取
    - 逐帧检测和跟踪
    - 结果保存
- `main()`：命令行入口

### 实现逻辑问题

1. **配置依赖外部模块**：
   - 依赖 `config.settings.Settings`，但该模块未在文件列表中
   - 建议：确保配置模块存在或提供默认配置

2. **缺少与主流程的集成**：
   - 作为独立模块运行，与 `main.py` 的集成不够紧密
   - 建议：统一接口，避免代码重复

3. **结果格式不一致**：
   - 输出格式与 `main.py` 中的 MOT 线程不同
   - 建议：统一结果格式

4. **缺少实时输出**：
   - 仅保存文件，无实时流输出
   - 建议：增加实时流推送功能

---

## 总结

### 关键发现

1. **架构设计**：
   - 采用多线程并行处理（Voice + MOT）
   - 使用事件驱动架构（FlowEngine）
   - 前后端分离（Flask + SSE）

2. **主要优点**：
   - 模块化设计，职责清晰
   - 支持实时处理和流式输出
   - 使用大模型进行智能评估

3. **共性问题**：
   - 大量硬编码配置和路径
   - 全局变量使用较多
   - 部分功能未完全实现（自唱票流程）
   - 模型加载缺乏缓存机制

4. **改进建议优先级**：
   - **高**：配置外部化、模型缓存、自唱票流程实现
   - **中**：代码重构（减少全局变量）、异常处理完善
   - **低**：前端模板分离、可视化增强

---

*文档生成时间：2026-04-27*
