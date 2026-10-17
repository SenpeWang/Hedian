# RELEASE.md — 版本发布记录

> 本文件记录每个发布版本相对上一版本的**全部改动**（需求背景 / 改动文件与原因 /
> 行为变化 / 验证方式 / 回退方法），供上传仓库时追溯。
> 更细粒度的逐次修改记录（含日志证据与离线单测结果）见开发目录的 `CHANGELOG.md`。

**版本基线**：`new_Hedian`（原始基线，即本仓库的 1.0.0 版本）。
基线本身已包含用户的手工编辑（`KEYWORD_LABELS` 新增岗位词
`一回路/二回路/机组长/请讲/请监护` 五条、730 行 `("请监护", "请求监护")`），
不计入本版本改动。

**本版本（1.1.0）包含**：变更 1（普通沟通流程）+ 变更 2（"请监护"标红与监护制白名单）
+ 变更 3（GPU 显存分配调整）+ 变更 4（普通沟通超时改用独立计时器，移除 tracker 依赖）
+ 变更 5（语音转录 VAD 分段改造，方案 A）+ 变更 6（15s 窗口改在视频时间轴上判定）
+ 变更 7（普通沟通最长时长 15s → 30s）+ 变更 8（修复 PoseEMAFilter 缺失方法致
tracker 崩溃）+ 变更 9（语音转录改为流式流水线：增量读音频 + 在线切片 + 跨段成句）
+ 变更 10（测试输入视频切换为 segment002.mpg）+ 变更 11（同句应答：修复岗位词/
"请讲"/"收到"同句时确认信号被丢弃）+ 变更 12（岗位词被句边界切开：保护词跨断点拼接
+ 断句阈值去倒挂）+ 变更 13（普通沟通：窗口 30s→50s + 被新沟通顶掉 + 超时收尾不再吞事件）
+ 变更 14（前端流程通知同秒排序）+ 变更 15/16（段间左上下文档位实验 2.0s，已回退）
+ 变更 17（删除批处理流水线，语音转写仅保留流式流水线）
+ 变更 18（全仓 PEP 8 / Google 风格达标改造：black 重排 + 行宽/docstring 清零）。
+ 变更 19（修复 redis.db 配置不生效，多实例 Redis 隔离）。

**不包含**：移除裸"监护"触发（见文末"已知问题"）；vLLM token 级流式推理
（环境未装 vLLM，且流式不支持时间戳，会打断依赖字级时间戳的整条链路）。

**打包清理（相对开发目录）**：本发布包不含 `data/`、`models/`（软链接）、`run.log`、
`.backups/`、开发用 `AGENTS.md`、`CHANGELOG.md` 及本地文档；`config.yaml` 已恢复
仓库相对路径与默认端口 5002（去除本机绝对路径）；`.gitignore` 补上 `frontend/dist/`
与 `!RELEASE.md`；前端补充 ESLint + Prettier 配置与 `lint` / `format` 脚本（对齐
README 开发规范）。

---

## 目录

- [变更 1：新增"普通沟通"流程](#变更-1新增普通沟通流程)
- [变更 2："请监护"标红 + 监护制启动白名单](#变更-2请监护标红--监护制启动白名单)
- [变更 3：GPU 显存分配调整](#变更-3gpu-显存分配调整)
- [变更 4：普通沟通超时改用独立计时器](#变更-4普通沟通超时改用独立计时器)
- [变更 5：语音转录 VAD 分段改造（方案 A）](#变更-5语音转录-vad-分段改造方案-a)
- [变更 6：15s 窗口改在视频时间轴上判定](#变更-615s-窗口改在视频时间轴上判定)
- [变更 7：普通沟通最长持续时长 15s → 30s](#变更-7普通沟通最长持续时长-15s--30s)
- [变更 8：修复 PoseEMAFilter 缺失方法致 tracker 崩溃](#变更-8修复-poseemafilter-缺失方法致-tracker-崩溃)
- [变更 9：语音转录改为流式流水线](#变更-9语音转录改为流式流水线)
- [变更 10：测试输入视频切换为 segment002.mpg](#变更-10测试输入视频切换为-segment002mpg)
- [变更 11：修复"岗位词/请讲/收到"同句时确认信号被丢弃](#变更-11修复岗位词请讲收到同句时确认信号被丢弃)
- [变更 12：修复岗位词被句边界切开](#变更-12修复岗位词被句边界切开)
- [变更 13：普通沟通"被新沟通顶掉" + 超时收尾不再吞事件（窗口 30s → 50s）](#变更-13普通沟通被新沟通顶掉--超时收尾不再吞事件窗口-30s--50s)
- [变更 14：前端流程通知同秒排序](#变更-14前端流程通知同秒排序)
- [变更 15/16：段间左上下文档位实验（2.0s），已回退](#变更-1516段间左上下文档位实验20s已回退)
- [变更 17：删除批处理流水线（辅助链路）](#变更-17删除批处理流水线辅助链路)
- [已知问题（本版本未修复）](#已知问题本版本未修复)

---

## 变更 1：新增"普通沟通"流程

**背景**：语音识别到岗位词（一回路 / 二回路 / 机组长）后 3 秒内识别到"请讲"
→ "普通沟通"流程开始；流程持续一段时间，中途识别到"收到"提前结束；
开始/结束时前端"系统通知"栏显示"普通沟通开始 / 普通沟通结束"。

| # | 文件 | 类型 | 改动内容 |
|---|------|------|----------|
| 1 | `rules/normal_comm_rule.py` | 新建 | 普通沟通规则状态机：订阅 `VOICE_KEY_MOMENT`，"岗位词→3s内请讲→启动 / 收到→提前结束 / 超时自动结束"；发布 `FLOW_STARTED/FLOW_ENDED`（`flow_type="normal_comm"`）；`register()` 供注册表自动发现 |
| 2 | `config.yaml` | 修改 | `rules:` 节新增 `normal_comm: true` |
| 3 | `frontend/src/types.ts` | 修改 | `FlowType` 联合类型增加 `'normal_comm'` |
| 4 | `frontend/src/composables/useNotify.ts` | 修改 | `flowTypeMap` 增加 `normal_comm: ['#00ff88', '普通沟通']` |

复用现有机制：`KEYWORD_LABELS` 关键词事件、`RuleRegistry.discover()` 自动发现、
`core/inference_sync.py` 通用转推链路、`rules/flow_recorder.py` 自动落盘。

---

## 变更 2："请监护"标红 + 监护制启动白名单

**背景**："请监护"三个字要在人员对话记录中标红，且能正常触发监护制启动、不产生副作用。
前端标红是字面子串匹配，`keys` 里放归一化标签 `请求监护`（4 字）在显示文本
"请监护"（3 字）中匹配不到。

| # | 文件 | 改动内容 |
|---|------|----------|
| 1 | `modules/voice/speech_transcriber.py` | `KEYWORD_LABELS` 中 `("请监护", "请求监护")` → `("请监护", "请监护")`，使 `keys` 含"请监护"可命中标红 |
| 2 | `rules/supervision_rule.py` | 启动白名单 `["监护", "请求监护"]` → `["请监护", "请求监护"]`；设备码排除表加入 `"请监护"`（防止污染"九字码复述"检查项） |

行为变化："请监护"：标红 ✅、触发监护制 ✅、checklist 不被污染 ✅；"请求监护"行为不变。

---

## 变更 3：GPU 显存分配调整

**背景**：实测 OOM 事故——evaluation（Qwen3-8B，约 16GB）与 voice/behavior 同在
GPU 1，叠加外部进程后显存耗尽，语音子进程重启时 CUDA OOM、转录无产出。

| 文件 | 改动内容 |
|---|---|
| `config.yaml` | `gpu_map.evaluation: "1"` → `"0"`（16GB 评估模型挪到 GPU 0，GPU 1 只留 voice+behavior） |

实测显存需求（bf16）：evaluation ~16GB、tracker ~2~4GB、behavior ~4GB、
voice ~2~4GB，合计 ~28~30GB，单张 24GB 卡放不下，需双卡。

---

## 变更 4：普通沟通超时改用独立计时器

**背景**：普通沟通的开始/结束判定与前端推送不需要 tracker 事件；事件驱动超时会
"顺延"到下一条语音事件才触发（实测 47.4s 才结束，应为 15s）。

| 文件 | 改动内容 |
|---|---|
| `rules/normal_comm_rule.py` | 移除 `TRACKER_PROXIMITY`/`BEHAVIOR_HAND_RAISED` 订阅；启动 `threading.Timer` 独立计时器，到点以视频时间轴 `flow_start_sec + 15` 结束；`_close_flow`/`reset` 时取消计时器 |

行为变化：流程开始后精确 15s（视频时间轴）自动结束，无人说话也不再延迟；
普通沟通与 tracker 模块完全解耦。

---

## 变更 5：语音转录 VAD 分段改造（方案 A）

**背景**：旧滑窗实现（20s 窗/15s 步长/5s 重叠 + midpoint 去重）对同一段语音的
转录结果与评估链路 run_infer.py 不一致，典型问题："二/回路巡盘"被拆分。

**方案**：ffmpeg silencedetect VAD 按静音边界切自然发言段（无重叠），每段切独立
wav 以文件路径喂模型；保留 ForcedAligner 字级时间戳与逐段 `on_segment` 回调；
解码参数对齐 run_infer.py（`num_beams=1`、`no_repeat_ngram_size=3` 等）。

| 文件 | 改动内容 |
|---|---|
| `modules/voice/speech_transcriber.py` | 删除滑窗 + midpoint 去重；新增 `_detect_speech_intervals`（VAD）、`_probe_duration`、`_extract_segment`；解码参数对齐评估链路 |
| `modules/voice/voice_module.py` | 初始化透传 5 个 VAD 配置项 |
| `config.yaml` | `voice:` 节新增 `vad_noise_db / vad_min_silence_sec / vad_speech_pad_sec / vad_min_speech_sec / vad_max_chunk_sec` |

---

## 变更 6：15s 窗口改在视频时间轴上判定

**背景**：变更 4 的 wall-clock 计时器存在竞态：voice 转录远快于实时，音频里远处的
"收到"会在计时器触发前抵达，把本应超时结束的流程提前按"收到"关闭，结束时间错误。

| 文件 | 改动内容 |
|---|---|
| `rules/normal_comm_rule.py` | 流程活跃时，"收到"仅在视频时间轴窗口内才按收到时刻提前结束；任何事件时间戳超窗 → 以 `flow_start_sec + 15` 超时关闭 |

计时器保留作兜底；`_close_flow` 的 `_active` 检查保证双重触发幂等。

---

## 变更 7：普通沟通最长持续时长 15s → 30s

| 文件 | 改动内容 |
|---|---|
| `rules/normal_comm_rule.py` | `MAX_FLOW_SEC = 15.0` → `30.0`（后续变更 13 再调整为 50.0） |

---

## 变更 8：修复 PoseEMAFilter 缺失方法致 tracker 崩溃

**背景**：长视频推理时 tracker 子进程在 ~50s 处（首次出现人物轨迹过期）崩溃：
`'PoseEMAFilter' object has no attribute 'remove'`（`hand_raiser.py:218`），
导致前端视频黑屏。

| 文件 | 改动内容 |
|---|---|
| `modules/behavior/behavior_utils.py` | `PoseEMAFilter` 类补齐缺失的 `remove(track_id)` 与 `clear()` 两个方法（原始代码 bug） |

---

## 变更 9：语音转录改为流式流水线

**背景**：批处理路径下"收到"被切成"收" / "到"——切片点落在词中间，且段间无词级
衔接、每段独立成句。同时要求切片策略运行时可配置、可切换（持续人工 A/B）。

**实现口径**：块级流式（增量读入 + 在线切片 + 逐段增量解码 + 保留字级时间戳），
不引入新依赖、不改变下游协议。vLLM token 级流式因环境与时间戳约束未启用（见头部说明）。

| 文件 | 类型 | 改动内容 |
|---|---|---|
| `modules/voice/streaming_source.py` | 新建 | `PcmStream`（ffmpeg stdout 管道 → 16k 单声道 float32 增量块，支持文件与直播源）、`PcmRing`（有界环形缓冲）、`is_live_source()`；内存占用与音频总时长无关 |
| `modules/voice/segmenters.py` | 新建 | 切片策略接口 + `SilenceVadSegmenter`（在线能量 VAD，超长时在能量最低点切分）+ `FixedWindowSegmenter` + 注册表；策略运行时可切换 |
| `modules/voice/sentence_assembler.py` | 新建 | `SentenceAssembler`：跨段全局词流 → 按停顿成句；重叠重复词去重、超长强制封句、`tick()` 按游标及时封句（"收"+"到"可拼回"收到"） |
| `modules/voice/speech_transcriber.py` | 修改 | 新增 `transcribe_stream(...)`：增量读 PCM → 在线切片 → 逐段解码 → 段闭合即回调；每段向前多取 `boundary_overlap_sec` 作左上下文 |
| `modules/voice/voice_module.py` | 修改 | 按 `voice.pipeline` 分流；批处理逻辑整体下沉为 `_process_video_legacy`；新增 `_process_video_stream` |
| `config.yaml` | 修改 | `voice` 段新增 `stream_block_sec / segmenter / boundary_overlap_sec / vad_split_search_sec / vad_split_margin_db / max_sentence_sec / fixed_window_sec / fixed_window_overlap_sec` |

**关键约束（勿改）**：流式路径 per-source 进度以已解码段的结束时间上报，不使用
音频读取游标——ffmpeg 预读远快于实时，否则会推快 `global_sec`、语音文字超前于其他模态。

---

## 变更 10：测试输入视频切换为 segment002.mpg

| 文件 | 改动内容 |
|---|---|
| `config.yaml` | `videos.front` / `videos.pop` → `segment002.mpg`（本地测试输入；发布包中已恢复仓库相对路径默认值） |

---

## 变更 11：修复"岗位词/请讲/收到"同句时确认信号被丢弃

**背景**：`process_transcribed_words` 按 `KEYWORD_LABELS` 表序输出事件，"收到"
排在岗位词/"请讲"之前且同句共用句首时间戳，导致事件流为
`收到(被丢弃) → 一回路 → 请讲(启动)`，之后再无"收到"，只能空等超时。

| 文件 | 改动内容 |
|---|---|
| `modules/voice/speech_transcriber.py` | 关键词改为**按句中实际出现位置**排序后输出事件（设备码仍先输出；拼音近似匹配且字面不存在的词排在字面命中之后） |
| `modules/voice/voice_module.py` | `VOICE_KEY_MOMENT` 载荷新增 `keys`（该句全部关键词，已按位置排序），流式与 legacy 两条路径同步 |
| `rules/normal_comm_rule.py` | 新增 `_has_ack_after()`：`请讲` 启动流程后若同句后续含"收到"则立即以 `recv_ack` 收尾 |

行为变化：同句 `岗位词 + 请讲 + 收到` → 启动后立即 `recv_ack` 收尾（持续 0.0s）；
`收到` 在 `请讲` 之前（上轮残留）不误判；`keys` 为增量字段，其他三个规则不受影响。

---

## 变更 12：修复岗位词被句边界切开

**背景**：`二回路` / `一回路` 被**从句边界切开**（"嗯 二 ⟦1.12s⟧ 回路巡盘"）。
根因两条并存：① 阈值倒挂——`sentence_gap_sec(0.6) < vad_min_silence_sec(0.8)`，
句边界比段边界更敏感；② 断句只看时间差，不检查切点是否落在词内部。

| 文件 | 改动内容 |
|---|---|
| `modules/voice/sentence_assembler.py` | 新增保护词机制（`resolve_protected_words` / `tail_breaks_protected_word` / `tail_is_protected_prefix`）；`push()` 与 `tick()` 均不得在保护词内部断句；句尾为保护词未完成前缀时挂起，最多多等 `protected_hold_sec` |
| `modules/voice/speech_transcriber.py` | `process_transcribed_words()` 新增 `protected_words` 参数，停顿切句处加同样门控 |
| `modules/voice/voice_module.py` | 从配置读取 `protected_words` / `protected_hold_sec`，同时传入两处断句逻辑 |
| `config.yaml` | `sentence_gap_sec: 0.6 → 1.0`（去倒挂，≥ `vad_min_silence_sec`）；新增 `protected_words: null`（自动从 `KEYWORD_LABELS` 派生）、`protected_hold_sec: 2.0` |

实测：可疑 `二 回路` / `一 回路` / 句首 `回路` 均降为 0；上屏延迟几乎无感知。

---

## 变更 13：普通沟通"被新沟通顶掉" + 超时收尾不再吞事件（窗口 30s → 50s）

**背景**（run_id=`20260912_163122`）：`824.34s` 的「机组长」抵达时距流程开始已
41.40s ≥ 窗口 → 走超时收尾分支后 `return`，**把岗位词吞掉**；紧随的「请讲」因
`_last_post_word_ts` 已被 `_close_flow` 重置而无法启动。这是"任何阈值 N 都会在
N+ε 处复现"的结构性缺陷，因此必须方案 A（收尾后不 return）+ 顶掉语义配套。

| 文件 | 改动内容 |
|---|---|
| `rules/normal_comm_rule.py` | `MAX_FLOW_SEC: 30.0 → 50.0`；`_on_voice_intent` 重构为固定 5 步顺序：①岗位词**无条件登记** ②窗口内"收到"→结束 ③超时收尾**去掉 return** ④流程活跃时"岗位词 + 3s 内请讲"→ `new_comm` 结束并立即开新流程 ⑤空闲时启动（含变更 11 同句应答）；`_close_flow` 不再重置 `_last_post_word_ts`（改由 `_start_flow` 消费时清空）；结束来源新增 `new_comm` |

**判定顺序（顺序本身是关键，勿随意调整）**：

```
① 岗位词无条件登记
② 流程中 + 窗口内"收到"      → 结束(recv_ack)，return
③ 流程中 + 已超窗口          → 结束(timeout)，【不 return】
④ 流程中 + 岗位词后3s内"请讲" → 结束(new_comm) + 立即开新流程，return
⑤ 空闲 + 岗位词后3s内"请讲"   → 启动（+ 同句应答立即 recv_ack）
```

---

## 变更 14：前端流程通知同秒排序

**背景**：后端时序已正确（旧流程在「请讲」时刻结束并开新流程），但前端把整批
`flow_start` 全部塞进列表再塞 `flow_end`，批次内真实先后被丢弃，同一秒会渲染成
"先开始后结束"的 0 秒流程观感。

**关键**：排序不能只看 `(时间, 是否开始)`——同一秒有两种语义、正确顺序相反：
被顶掉（`flow_id` 不同）应"结束 → 开始"；同句应答（`flow_id` 相同）应"开始 → 结束"。
因此排序键取 `(sec, flowId, isStart ? 0 : 1)`。

| 文件 | 改动内容 |
|---|---|
| `frontend/src/types.ts` | `FlowEvent` 新增 `flowId?: number` |
| `frontend/src/composables/useNotify.ts` | `addFlow` 改为去重后按 `(sec, flowId, isStart?0:1)` **有序插入**（`splice`） |

离线回归 4 场景（被顶掉 / 同句应答 / 乱序到达 / 完整时间线）全部通过。

---

## 变更 15/16：段间左上下文档位实验（2.0s），已回退

**变更 15**：把 `voice.boundary_overlap_sec` 从 0.4s 提到 2.0s 验证幻觉抑制，
并新增 A/B 对比工具 `tools/asr_ab.py`。受控 A/B（58 段完全一致）结论：内容 99.4%
一致、自重复率仅微降、唯一明确改善是 842s 处 42 字复读式乱码消失；但**关键词命中
下降**（二回路 9→6、请讲 8→7、信息通报 2→1、执行 9→8，含新增识别错误），而
这些词正是监护制与信息通报的触发条件。

**变更 16**：据此回退 `boundary_overlap_sec` 2.0s → 0.4s，并删除 A/B 工具与
实验产物（`tools/`、`data/results/ab_context_cut30/`、`run_ab.log`），完整回退。

---

## 变更 17：删除批处理流水线（辅助链路）

**背景**：运行历史核查（45 场 `run.log`）证实流式上线后批处理链路 **0 次执行**；
变更 15/16 已确认无需保留 A/B 对比能力。批处理唯一入口是 `voice.pipeline` 分流。

| 文件 | 改动内容 |
|---|---|
| `modules/voice/voice_module.py` | 移除 `voice.pipeline` 分流，固定调用 `_process_video_stream`；删除 `_process_video_legacy` 及仅被其使用的 `_extract_audio` 与 `subprocess` 导入 |
| `modules/voice/speech_transcriber.py` | 删除批处理核心 `transcribe()` 及专用辅助 `_extract_segment` / `_detect_speech_intervals` / `SILENCE_EVENT_PATTERN`；**保留 `_probe_duration`**（流式路径仍在用） |
| `config.yaml` | 删除 `voice.pipeline` 键（发布包中端口与视频路径另恢复为仓库默认值） |

行为变化：无（删除前批处理链路从未被启用，流式链路代码路径与输出行为不受影响）。

---

## 变更 18：全仓 PEP 8 / Google 风格达标改造

### 需求/背景

README 开发规范明确"Python 基于 PEP 8 并遵循 Google Python Style Guide"。
工具化校验（`pycodestyle --max-line-length=79`、`pydocstyle --convention=google`、
`black --check --line-length 79`，运行于 `~/myconda/envs/qwen3-asr` 环境）初测：
pycodestyle 375 条违规（E501 行宽 360 + 其他 15）、pydocstyle 40 条、
57/58 个文件需 black 重排。

### 改动内容

| # | 改动 | 说明 |
|---|------|------|
| 1 | `black --line-length 79` 全仓重排 | 57 个文件格式规范化（AST 不变） |
| 2 | 手工修复 black 无法自动处理的 E501 | 长日志 f-string 拆分为隐式拼接、长注释/docstring 重排、评估提示词表格行缩短（语义保持） |
| 3 | 修复 E203（8 处） | 重构复杂切片表达式为简单变量切片（`ordered[index+1:]` → `ordered[tail_start:]` 等），消除 black 与 pycodestyle 的切片空格冲突 |
| 4 | 修复 E402（main.py 3 处） | `sys.path` 注入后才能导入 `core` 包属设计意图，按惯例加 `# noqa: E402` 注明 |
| 5 | 修复 docstring 细节 | D212（摘要提到首行，14 处模块/类/方法 docstring）、D415（中文句号改英文句点）、D105（`PcmStream.__enter__/__exit__` 补 docstring）、D205（摘要后补空行）、D202（`run_tracker_process` 处 black 与 pydocstyle 冲突，按惯例 `# noqa: D202` 豁免并注明原因） |
| 6 | README 同步 | 引言去除历史性表述、删除"版本记录"章节、代码风格条目更新为"基于 PEP 8 的 Google 风格"细则 |

### 行为变化

- 纯格式与注释变更，运行行为不变（`py_compile` 全量通过）。
- 唯一文本级变化：`qwen_evaluator.py` 评估提示词中三处表格行文字精简（语义等价）。

### 验证方式（全部于 `~/myconda/envs/qwen3-asr` 环境执行）

```bash
python -m pycodestyle --max-line-length=79 $(find . -name '*.py')     # 0 违规
python -m pydocstyle --convention=google $(find . -name '*.py')       # 0 违规
python -m black --check --line-length 79 $(find . -name '*.py')       # 58 files unchanged
python -m compileall -q main.py core modules rules evaluation web     # 通过
```

### 回退方法

格式类改动无回退必要；如需恢复提示词原文，见开发目录 `CHANGELOG.md` 中
run_infer 对齐条目。

---

## 变更 19：修复 redis.db 配置不生效（多实例 Redis 隔离）

### 背景

端到端实测时发现：`main.py` 的 `REDIS_HOST/PORT/DB` 是写死的模块常量
（默认 `localhost:6379/0`），`config.yaml` 的 `redis.db` 从未被读取。
同一台机器多实例部署时，所有实例的 EventBus / InferenceSync / `on_pipeline_start`
的 flushdb 全部落到 db0：互相 flushdb 清数据、同一消费组（`web_process_group` +
同名消费者）互相抢 `pipeline.start` 消息，导致一方 /start 后另一方模块无法拉起。

### 改动

| 文件 | 改动 |
|---|---|
| `main.py` | `main()` 在读取 `redis` 配置段后同步模块级全局变量（`global REDIS_HOST/PORT/DB`）；`run_web_process` 与 `_run_module_process`（spawn 子进程重新 import 后全局变量回默认值）入口处从 `config_dict` 恢复全局变量 |

`http_server._get_redis` 原本就读 `config["redis_db"]`（main() 已注入），无需改动。
默认值回退链不变（未配置时仍为 localhost:6379/0），单实例用户行为不受影响。

### 验证

`--config` 指定 `db: 5` 启动：db0 不再被写入/清空，`pipeline.start` 事件进入
db5 并被本实例消费，4 个模块子进程正常拉起并完成推理。

---

## 已知问题（本版本未修复）

1. **裸"监护"误触发监护制**：任何含"监护"二字的句子（如"监护人二回路"）经
   `("监护", "请求监护")` 子串匹配产出 `请求监护`，误触发监护制开始。
2. **300s 超时关闭后无法立即重新开始**：模块速度差导致 tracker 的超时关闭事件
   晚于 voice 的新"请监护"到达，启动请求被吞。

---

## 回退方法（整版本）

本版本相对基线（`new_Hedian`）为纯增量改动 + 有限文件修改，完整回退即：
用基线仓库覆盖本包以下文件——`rules/normal_comm_rule.py`（删除，基线无此文件）、
`modules/voice/{streaming_source,segmenters,sentence_assembler}.py`（删除）、
`modules/voice/{speech_transcriber,voice_module}.py`、
`modules/behavior/behavior_utils.py`、`rules/supervision_rule.py`、
`frontend/src/{types.ts,composables/useNotify.ts}`、`config.yaml`、
`frontend/package.json`。逐变更回退方法见各条目及开发目录 `CHANGELOG.md`。
