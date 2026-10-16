"""大模型（Qwen3）评估模块，以子进程方式运行会话评估.

在独立子进程中加载 Qwen3 模型按流程类型构建 prompt 并生成评估报告，
子进程退出即释放显存，评估结果与 prompt 一并落盘供前端展示。
"""
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("evaluation.qwen")


# 设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 三类流程评估 prompt 共用的「思考块」输出指令（红线要求）
_THINK_BLOCK_PROMPT = """
## 重要输出指令（红线要求）
请务必在给出上述任何报告结构之前，先在 `<think>...</think>` 标签内输出你对该流程的详细分析、步步推导和思考推理过程。
格式必须为：
<think>
[这里是你的推理和思考过程，一步步分析每一项评分的扣分/给分依据，字数在 100-300 字左右]
</think>

[接下来是上述要求的报告内容]
"""

# 单次评估的总超时（秒），覆盖「子进程加载模型 + 生成」全过程。
# 背景：子进程若卡在 CUDA 生成阶段——既不退出也不往队列吐数据——外层
# `while worker_process.is_alive() or not msg_queue.empty()` 会一直空转，
# 导致 finally 里的 terminate() 永远执行不到；残留进程会一直占着约 15GB
# 显存，下一轮评估加载 Qwen3-8B 时显存不足，推理子进程静默起不来。
# 因此这里必须给等待设定总上限，超时即强制回收。
_EVAL_TOTAL_TIMEOUT_SEC = 300


def _qwen_worker(
    model_path: str,
    prompt: str,
    message_queue: Any,
    eval_gpu: str = "1",
) -> None:
    """子进程工作函数：在指定 GPU 上加载模型并生成评估报告.

    按 config 分配的 eval_gpu 启动模型（不再自主探测），
    评估完成后子进程退出，显存 100% 自动释放。

    Args:
        model_path: Qwen 模型本地权重目录路径.
        prompt: 输入的大模型提示词.
        message_queue: 跨进程消息通信队列.
        eval_gpu: 评估子进程使用的 GPU 编号.
    """
    # 视角级 GPU 分配: 评估走 config 指定的 GPU(eval_gpu), 不再自主探测
    os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    selected_gpu = eval_gpu

    try:
        import torch
        torch.set_num_threads(1)
        import threading
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

        logging.getLogger("evaluation.qwen").info(
            f"子进程加载 Qwen3 模型: {model_path} (GPU {selected_gpu}, bfloat16)"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            local_files_only=True,
        )

        try:
            allocated = torch.cuda.memory_allocated("cuda:0") / 1024**3
            logging.getLogger("evaluation.qwen").info(
                f"Qwen3 模型加载完成，GPU {selected_gpu} 显存占用: {allocated:.2f} GB"
            )
        except RuntimeError:
            # 显存查询为纯信息性日志，CUDA 不可用或设备无效时静默跳过
            pass

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=False,
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 8192,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            message_queue.put(("chunk", text))

        thread.join()
        message_queue.put(("done", None))

    except Exception as error:
        import traceback
        message_queue.put(("error", f"{error}\n{traceback.format_exc()}"))


def _friendly_failure_text(error: object) -> str:
    """把评估失败的原始异常收敛成一句可读结论.

    原始 OOM 异常带整段 traceback 与显存明细，直接写进报告卡片会让前端
    "流程评价"整块变成一堆堆栈文本，故只保留可行动的结论。

    Args:
        error: 子进程错误载荷或异常对象.

    Returns:
        面向展示的简短失败原因.
    """
    text = str(error)
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return (
            "评估失败：GPU 显存不足，评估卡剩余空间不足以加载 Qwen3-8B。"
            "请把 config 的 gpu_map.evaluation 调整到空闲显卡，"
            "或把 evaluation.max_workers 降为 1 串行评估。（本轮不参与评分）"
        )
    first_line = text.strip().splitlines()[0] if text.strip() else "未知错误"
    return f"评估失败：{first_line[:200]}（本轮不参与评分）"


class QwenEvaluator:
    """Qwen 大模型评估器（子进程模式，走 config 分配的 GPU，评估完自动释放显存）."""

    def __init__(self, model_path: Optional[str] = None, eval_gpu: str = "1") -> None:
        """初始化 Qwen 评估器.

        Args:
            model_path: Qwen 模型本地权重目录路径，为 None 时使用默认路径.
            eval_gpu: 评估子进程使用的 GPU 编号.
        """
        self._model_path = model_path
        self._eval_gpu = eval_gpu
        # 登记当前活跃的评估子进程。评估子进程由本类自行 spawn，不在 main.py
        # 的 active_worker_processes 列表内，流水线停止/重置时扫不到它，
        # 故在此单独登记，供 terminate_all() 统一兜底清理。
        self._active_processes: set = set()
        self._proc_lock = threading.Lock()

    def terminate_all(self, reason: str = "外部回收") -> int:
        """强制终止所有仍在运行的评估子进程，回收其占用的显存.

        用于流水线停止、页面刷新重置、finalize 超时等场景的显存兜底回收。

        Args:
            reason: 触发回收的原因描述，仅用于日志追溯.

        Returns:
            实际被终止的子进程数量.
        """
        with self._proc_lock:
            processes = list(self._active_processes)
        killed = 0
        for proc in processes:
            try:
                if proc.is_alive():
                    logger.warning(
                        f"{reason}：评估子进程 pid={proc.pid} 仍在运行，强制终止以释放显存"
                    )
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        # SIGTERM 对卡在 CUDA 上下文里的进程可能无效，升级为 SIGKILL
                        proc.kill()
                        proc.join(timeout=2)
                    killed += 1
            except Exception as kill_error:
                logger.error(f"终止评估子进程失败: {kill_error}")
        with self._proc_lock:
            self._active_processes.clear()
        if killed:
            logger.info(f"{reason}：共强制终止 {killed} 个评估子进程")
        return killed

    def evaluate(
        self,
        flow_data: Dict[str, Any],
        stream_callback: Optional[Any] = None,
        total_flows: int = 0,
    ) -> Dict[str, Any]:
        """在独立子进程中评估流程，评估完子进程退出释放显存.

        Args:
            flow_data: 提取拼接完成的流程数据字典.
            stream_callback: 逐 token 流式回调，可为 None.
            total_flows: 同类型流程总数，用于生成第几个流程的描述.

        Returns:
            评估结果字典，含 flow_type/score/report_text/prompt.
        """
        worker_process = None
        prompt = ""
        try:
            prompt = self._build_prompt(flow_data, total_flows=total_flows)

            logger.info(
                f"开始 Qwen 评估（子进程，动态空闲 GPU），"
                f"flow_id={flow_data.get('flow_id')} total_flows={total_flows}..."
            )

            ctx = mp.get_context("spawn")
            msg_queue = ctx.Queue()
            worker_process = ctx.Process(
                target=_qwen_worker,
                args=(self._model_path, prompt, msg_queue, self._eval_gpu),
            )
            worker_process.start()
            with self._proc_lock:
                self._active_processes.add(worker_process)

            report_text = ""
            # 总超时兜底：子进程若卡在 CUDA 生成阶段——既不退出也不吐数据——
            # 下面的 while 会一直空转，finally 里的 terminate() 永远执行不到，
            # 残留进程会长期占着约 15GB 显存，下一轮评估便加载不了模型。
            deadline = time.time() + _EVAL_TOTAL_TIMEOUT_SEC
            eval_timed_out = False
            while worker_process.is_alive() or not msg_queue.empty():
                if time.time() > deadline:
                    eval_timed_out = True
                    break
                try:
                    msg_type, payload = msg_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if msg_type == "chunk":
                    report_text += payload
                    if stream_callback:
                        try:
                            stream_callback(payload)
                        except Exception as error:
                            # 回调失败（如前端断开）不影响评估流程，仅记录
                            logger.warning(f"流式回调推送失败（不影响评估）: {error}")
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    logger.error(f"子进程评估失败: {payload}")
                    return {
                        "flow_type": flow_data.get("flow_type", "未知"),
                        "score": 0,
                        "report_text": f"评估失败: {payload}",
                        "prompt": prompt,
                    }

            if eval_timed_out:
                logger.error(
                    f"评估子进程超过 {_EVAL_TOTAL_TIMEOUT_SEC}s 未产出结果，判定为卡死，"
                    f"强制回收（flow_id={flow_data.get('flow_id')}）"
                )
                return {
                    "flow_type": flow_data.get("flow_type", "未知"),
                    "score": 0,
                    "report_text": (
                        f"评估超时：子进程 {_EVAL_TOTAL_TIMEOUT_SEC}s 未返回结果，"
                        "已强制回收以免持续占用显存（本轮不参与评分）"
                    ),
                    "prompt": prompt,
                }

            worker_process.join(timeout=5)
            if worker_process.is_alive():
                logger.warning("子进程未正常退出，强制终止")
                worker_process.terminate()
                worker_process.join(timeout=3)

            score = self._extract_score(report_text)
            logger.info(f"Qwen 评估完成，评分: {score}")

            return {
                "flow_type": flow_data.get("flow_type", "未知"),
                "score": score,
                "report_text": self._clean_report_text(report_text),
                "prompt": prompt,
            }

        except Exception as error:
            logger.error(f"Qwen 评估失败: {error}", exc_info=True)
            return {
                "flow_type": flow_data.get("flow_type", "未知"),
                "score": 0,
                "report_text": _friendly_failure_text(error),
                "prompt": prompt,
            }
        finally:
            if worker_process is not None:
                try:
                    if worker_process.is_alive():
                        logger.warning("清理：评估子进程仍在运行，强制终止")
                        worker_process.terminate()
                        worker_process.join(timeout=3)
                        if worker_process.is_alive():
                            # SIGTERM 可能被 CUDA 上下文吞掉，升级 SIGKILL 确保显存释放
                            worker_process.kill()
                            worker_process.join(timeout=2)
                    else:
                        worker_process.join(timeout=1)
                except Exception as cleanup_error:
                    logger.error(f"清理评估子进程出错: {cleanup_error}")
                with self._proc_lock:
                    self._active_processes.discard(worker_process)
                try:
                    worker_process.close()
                except (OSError, ValueError):
                    # ValueError: 进程仍运行；OSError: 管道/句柄问题，均为清理兜底，可忽略
                    pass

    def _build_prompt(self, flow_data: Dict[str, Any], total_flows: int = 0) -> str:
        """按流程类型构建评估 prompt.

        Args:
            flow_data: 提取拼接完成的流程数据字典.
            total_flows: 同类型流程总数，大于 0 时在 prompt 中标注第几个流程.

        Returns:
            针对该流程类型的完整评估 prompt 文本.
        """
        flow_type = flow_data.get("flow_type", "未知")
        flow_id = flow_data.get("flow_id", 0)

        flow_type_label_map = {
            "supervision": "监护制",
            "info_notice": "信息通报",
        }
        flow_type_label = flow_type_label_map.get(flow_type, flow_type)

        if total_flows > 0:
            flow_data = dict(flow_data)
            flow_data["_total_flows"] = total_flows
            flow_data["_flow_id_desc"] = (
                f"本次共 {total_flows} 个{flow_type_label}流程，"
                f"当前为第 {flow_id} 个{flow_type_label}流程"
            )

        if flow_type == "supervision":
            return self._build_supervision_prompt(flow_data)
        elif flow_type == "info_notice":
            return self._build_info_notice_prompt(flow_data)
        else:
            return f"评估流程类型: {flow_type}"

    def _build_info_notice_prompt(self, flow_data: Dict[str, Any]) -> str:
        """构建信息通报评估 prompt.

        Args:
            flow_data: 提取拼接完成的流程数据字典.

        Returns:
            完整的信息通报评估 prompt 文本.
        """
        voice_events = flow_data.get("voice_events", [])
        gaze_events = flow_data.get("gaze_events", [])
        behavior_events = flow_data.get("behavior_events", [])

        checklist = flow_data.get("content_checklist") or {}
        items_spec = flow_data.get("_team_items_spec") or []
        score = flow_data.get("team_total_score")
        if score is None:
            score = sum(item[2] for item in items_spec if checklist.get(item[0]))
        max_score = flow_data.get("team_max_score") or sum(
            item[2] for item in items_spec) or 10

        parts = []
        parts.append("你是核电站主控室信息通报规程合规检测评审专家。本流程按团队整体判定，"
                     "不区分一回路、二回路，请根据以下流程数据与逐项判定结果撰写评估报告。")
        parts.append("")
        parts.append("## 信息通报规程规则")
        parts.append("1. 信息发起者举手 + 高声喊出\"信息通报\"或\"信息通告\"，流程开始（红线：未举手扣分）")
        parts.append("2. 团队成员听到后应回复\"请讲\"，未回复则扣分")
        parts.append("3. 团队成员停下手中工作接受信息并予以关注，未做到则扣分")
        parts.append("4. 信息传递结束，发起者喊出\"通报完毕\"确认信息结束，未喊出则扣分")
        parts.append("5. 团队成员回答\"收到\"，未回答则扣分")
        parts.append("")
        parts.append("## 流程信息")
        parts.append("- 流程类型: 信息通报")
        if flow_data.get('_flow_id_desc'):
            parts.append(f"- {flow_data.get('_flow_id_desc')}")
        parts.append(f"- 开始时间: {flow_data.get('flow_start_sec', 0)}s")
        parts.append(f"- 结束时间: {flow_data.get('flow_end_sec', 0)}s")
        parts.append(f"- 持续时间: {flow_data.get('flow_continue_sec', 0)}s")
        parts.append(f"- 触发来源: {flow_data.get('start_source', '未知')}")
        parts.append(f"- 结束来源: {flow_data.get('end_source', '未知')}")
        initiator = flow_data.get("initiator_role")
        if initiator:
            parts.append(f"- 信息发起人工位: {initiator}")
        attention_sec = flow_data.get("attention_check_sec")
        if attention_sec is not None and attention_sec >= 0:
            parts.append(f"- 关注度检测时刻: 流程开始后 {attention_sec}s")
        parts.append("")
        parts.append("## 语音事件（按时间排序）")
        for event in voice_events:
            line = f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}"
            parts.append(line)
        if not voice_events:
            parts.append("- 无")
        parts.append("")
        parts.append("## 注视事件（逐工位关注度判定）")
        for event in gaze_events:
            line = f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}"
            operators = event.get("operators")
            if operators:
                detail = "，".join(
                    f"{role}={'已关注' if turned else '未关注'}"
                    for role, turned in operators.items())
                line += f"（{detail}）"
            parts.append(line)
        if not gaze_events:
            parts.append("- 无")
        parts.append("")
        parts.append("## 行为事件（举手等）")
        for event in behavior_events:
            parts.append(f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}")
        if not behavior_events:
            parts.append("- 无")
        parts.append("")
        parts.append(f"## 评估维度（团队整体计分，每项2分，满分{max_score}分）")
        parts.append("| 维度 | 评估要点 |")
        parts.append("|------|---------|")
        parts.append("| 1. 举手并喊出信息通报 | 发起人是否举手 + 喊出\"信息通报\" |")
        parts.append("| 2. 回复请讲 | 团队成员是否回复\"请讲\" |")
        parts.append("| 3. 停下工作中并予以关注 | 团队成员是否停下手中工作接受信息 |")
        parts.append("| 4. 喊出通报完毕 | 发起人是否喊出\"通报完毕\" |")
        parts.append("| 5. 回答收到 | 团队成员是否回答\"收到\" |")
        parts.append("")
        parts.append("## 逐项判定结果（已由规则引擎按上述维度离线判定，请直接引用，不要改动）")
        parts.append("| 评估项 | 判定 | 得分 |")
        parts.append("|------|------|------|")
        for key, label, weight in items_spec:
            ok = bool(checklist.get(key))
            mark = "合规" if ok else "扣分"
            parts.append(
                f"| {label} | {mark} | {weight if ok else 0}分/{weight}分 |")
        parts.append("")
        parts.append("**得分汇总**")
        parts.append(f"- 团队总分: {score}分/{max_score}分")
        parts.append("")
        parts.append("## 输出格式")
        parts.append("请严格按以下格式输出评估报告（各项分数必须与上面的判定结果一致）：")
        parts.append("")
        parts.append("### 信息通报评估报告")
        parts.append("")
        parts.append("#### 逐项评分")
        for key, label, weight in items_spec:
            ok = bool(checklist.get(key))
            mark = "合规" if ok else "扣分"
            parts.append(
                f"- {label}: {weight if ok else 0}分/{weight}分 - [{mark}，说明依据]")
        parts.append("")
        parts.append(f"#### 总分: {score}分/{max_score}分")
        parts.append("")
        parts.append("#### 综合说明")
        parts.append("[对团队成员执行信息通报流程的总体评价，包括合规点与改进建议]")
        parts.append("")
        parts.append("请严格按以上格式输出，不要输出其他内容。")

        prompt = "\n".join(parts)
        prompt += _THINK_BLOCK_PROMPT
        return prompt

    def _build_supervision_prompt(self, flow_data: Dict[str, Any]) -> str:
        """构建监护制评估 prompt.

        Args:
            flow_data: 提取拼接完成的流程数据字典.

        Returns:
            完整的监护制评估 prompt 文本.
        """
        voice_events = flow_data.get("voice_events", [])
        tracker_events = flow_data.get("tracker_events", [])
        behavior_events = flow_data.get("behavior_events", [])

        # ============ 第一部分：完整的评估规程 prompt ============
        prompt = f"""你是核电站主控室监护制规程合规检测评审专家。请根据附带的关键事件（keymoment），评估该流程与监护制规程要求的合规性。

## 监护制规程要求（动作与语音步骤顺序）
1. 流程启动：操作人喊出"请求监护"，且发起前后5秒内有举手动作（双向5秒窗口：举手落在"请求监护"语音±5秒内即为合格；红线规则：单独举手不启动流程，必须语音 + 举手）
2. 监护人到位：监护人移动至操作人身旁（跟踪事件"监护员已到位监护X回路"）
3. 手指指向文件：若有纸质程序计划，操作人左手指向程序待执行指令（有程序分支的关键判断特征）
4. 手指指向屏幕与复述：操作人/监护人手指指向操作控件，操作人读出9字码，监护人核对并复述9字码
5. 执行命令：监护人下达"可以执行"命令（包含"执行"二字即可接受）
6. 核对确认：双方检查设备状态，喊出"核对"
7. 监护结束：监护人离开操作人（跟踪事件"监护员已离开监护X回路"）

## 评估维度（共6项评估要点，满分10分）
| 维度 | 评估要点 | 评分标准 |
|------|---------|---------|
| 1. 流程启动 | "请求监护"语音发起前后5秒内是否有举手(双向5秒) | 2分=语音+5秒内举手(合格)；1分=仅语音(无举手或举手超5秒)；0分=无启动信号 |
| 2. 监护人到位 | 跟踪事件是否记录到位 | 2分=有到位记录；0分=无 |
| 3. 指令复述与设备码 | 操作人读9字码 + 监护人复述9字码 | 2分=双方都复述；1分=单方复述；0分=无复述 |
| 4. 动作规范(指向屏幕/文件) | 是否有手指指向屏幕(操作控件)或指向文件(程序指令)动作 | 2分=双动作/指向规范；1分=有单项指向；0分=无指向动作 |
| 5. 执行与核对 | 是否有含"执行"命令及"核对"确认 | 2分=执行与核对齐全；1分=有单项；0分=均无 |

## 顺序一致性检查
请检查 keymoment 的时间顺序是否符合监护制要求：
- "请求监护"应在流程开始
- 监护人到位应在"请求监护"之后
- 9字码读出应在监护人到位之后
- "执行"应在9字码复述之后
- "核对"应在"执行"之后
- 监护人离开应在流程末尾
- 若顺序错乱，请在综合说明中指出

## 输出格式（严格按此格式输出）
### 监护制评估报告

#### 维度评分
- 流程启动: X分/2分 - [依据与说明]
- 监护人到位: X分/2分 - [依据与说明]
- 指令复述与设备码: X分/2分 - [依据与说明]
- 动作规范(指向屏幕/文件): X分/2分 - [依据与说明]
- 执行与核对: X分/2分 - [依据与说明]

#### 顺序一致性: [符合/部分错乱/严重错乱]
- [若有错乱，列出具体错乱步骤]

#### 总分: X分/10分

#### 综合说明
[对流程执行情况的总体评价，包括合规点、顺序问题与改进建议]

## 流程信息
- 流程类型: 监护制
- {flow_data.get('_flow_id_desc', '')}
- 开始时间: {flow_data.get('flow_start_sec', 0)}s
- 结束时间: {flow_data.get('flow_end_sec', 0)}s
- 持续时间: {flow_data.get('flow_continue_sec', 0)}s
"""

        # ============ 第二部分：附带的 keymoment 数据 ============
        prompt += "\n## 附带的关键事件（keymoment）\n请根据以下 keymoment 进行评估：\n\n"

        prompt += "### 语音事件（按时间顺序）\n"
        for event in voice_events:
            prompt += f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}\n"
        if not voice_events:
            prompt += "- 无\n"

        prompt += "\n### 跟踪事件（监护到位/离开等）\n"
        for event in tracker_events:
            prompt += f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}\n"
        if not tracker_events:
            prompt += "- 无\n"

        prompt += "\n### 行为事件（举手等）\n"
        for event in behavior_events:
            prompt += f"- [{event.get('localSec', 0)}s] {event.get('key_moment', '')}\n"
        if not behavior_events:
            prompt += "- 无\n"

        prompt += _THINK_BLOCK_PROMPT
        return prompt

    @staticmethod
    def _clean_report_text(report_text: str) -> str:
        """剥离思考块与特殊 token，只保留可读的报告正文.

        模型按 `_THINK_BLOCK_PROMPT` 要求先输出 `<think>...</think>`，但常会多补一个
        未配对的 `</think>`；正文末尾也常带 `<|im_end|>` 等特殊 token。落盘与推送的
        报告文本据此清洗（前端另有 think 折叠展示逻辑，按原始流自行解析）。

        Args:
            report_text: 大模型原始输出全文.

        Returns:
            去除思考块与特殊 token 后 strip 的正文.
        """
        import re

        text = re.sub(r"<think>.*?</think>", "", report_text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<\|(?:im_end|im_start|endoftext|eot_id)\|>", "", text)
        return text.strip()

    def _extract_score(self, report_text: str) -> int:
        """从报告文本中提取评分.

        Args:
            report_text: 大模型生成的评估报告全文.

        Returns:
            提取到的评分；无法提取时记录告警并默认返回 5 分.
        """
        import re

        # 清除思考块干扰，防止匹配到大模型在思考推理过程中演算的各项中间分值
        clean_text = self._clean_report_text(report_text)

        patterns = [
            r"总分[：:]\s*(\d+)",
            r"得分[：:]\s*(\d+)",
            r"(\d+)/10",
            r"评分[：:]\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                return int(match.group(1))

        logger.warning("无法从报告中提取评分，默认返回 5 分")
        return 5
