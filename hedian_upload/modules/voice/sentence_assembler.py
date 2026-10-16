"""跨段全局词流组装器：把"每段独立成句"改为"全局词流成句".

旧实现每解码完一段就独立调用 `process_transcribed_words`，段与段之间没有词级
衔接。于是当切片点落在词中间时，词与句会被永久切断——实测 01:04 的 "…我来跟
好收" 与 01:15 的 "到然后" 本应是 "收到"，却因分属两个段而拆成 "收" / "到"，
关键词 "收到" 也随之丢失。

本类维护一条**全局词序列**：

  1. 段与段之间的词先按时间拼接（并丢弃重叠区里被重复解码的词）；
  2. 再按词间停顿切句，只有停顿超过 `sentence_gap_sec`（句子"封口"）才整句交付下游；
  3. 即使切片点落在 "收" 与 "到" 之间，两句仍会被拼成 "收到"，关键词不丢。

同时提供 `tick()`：当音频游标推进到句尾之后超过阈值时立即封句，避免"必须等到
下一个词才吐上一句"造成的人工延迟。

## 保护词（词完整性保护）

仅靠时间阈值无法解决"停顿较长时把词切成两半"的问题：实测
`二`（11:35）｜`回路巡盘`（11:38）之间隔了 1.12s，`一`｜`回路…` 之间也有约 0.8s，
都超过了断句阈值，于是岗位词 `二回路` / `一回路` 被拆开，规则层再也匹配不到。

因此增加**保护词**机制：只要断句点落在某个保护词（岗位词、复合关键词）内部，
就不许在此处断句——

  - `push()`：新词到达时，若"待拼句尾 + 新词开头"仍能拼成某个保护词，则不封句；
  - `tick()`：若待拼句尾是某个保护词的**未完成前缀**（如 "二"、"请"），则挂起
    封句，最多多等 `protected_hold_sec` 秒，避免句子被无限期延迟上屏。

保护词默认为 `KEYWORD_LABELS` 中长度 ≥ 2 的关键词（见 `resolve_protected_words`），
也可通过 `voice.protected_words` 覆盖。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("module.voice.sentence")

# 重叠解码时，判定"重复词"的时间容差（秒）
DEDUP_TOLERANCE_SEC = 0.02

# 保护词兜底表（当无法从 KEYWORD_LABELS 派生时使用）
DEFAULT_PROTECTED_WORDS: Tuple[str, ...] = (
    "一回路", "二回路", "机组长", "请讲", "请监护", "请求监护", "收到",
    "执行", "核对", "核实", "信息通报", "通报完毕",
)

# 句尾是保护词未完成前缀时，额外挂起封句的最长秒数（防止上屏被无限延迟）
DEFAULT_PROTECTED_HOLD_SEC = 2.0


def resolve_protected_words(override: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """解析生效的保护词表.

    优先使用调用方覆盖值；否则从 `speech_transcriber.KEYWORD_LABELS` 派生
    （取长度 ≥ 2 的关键词并去重），保证新增关键词时无需同步维护；派生失败时
    退回 `DEFAULT_PROTECTED_WORDS`。

    Args:
        override: 调用方指定的保护词；为空/None 时自动派生.

    Returns:
        去重后的保护词元组；无保护词时为空元组.
    """
    if override:
        words: List[str] = []
        for item in override:
            text = str(item or "").strip()
            if len(text) >= 2 and text not in words:
                words.append(text)
        return tuple(words)
    if override is not None:  # 显式传入空序列 → 关闭保护
        return ()

    try:
        from modules.voice.speech_transcriber import KEYWORD_LABELS

        derived: List[str] = []
        for keyword, _label in KEYWORD_LABELS:
            keyword = str(keyword or "").strip()
            if len(keyword) >= 2 and keyword not in derived:
                derived.append(keyword)
        if derived:
            return tuple(derived)
    except Exception as derive_error:  # 派生失败不影响主流程
        logger.debug("保护词派生失败，使用内置兜底表: %s", derive_error)

    return DEFAULT_PROTECTED_WORDS


def tail_breaks_protected_word(
    tail_text: str,
    incoming: str,
    protected_words: Sequence[str],
) -> bool:
    """判断在 `tail_text` 与 `incoming` 之间断句是否会切开保护词.

    判据：存在保护词 kw 与切点 k（1 ≤ k < len(kw)），使得待拼句尾以 `kw[:k]`
    结尾，且新词是 `kw[k:]` 的前缀（新词可能只是该保护词的一部分，例如单字
    "回" 对应 `二回路` 的剩余部分 "回路"）。

    Args:
        tail_text: 待拼句子的尾部文本（只需包含最后 len(kw)-1 个字符即可）.
        incoming: 新到达的词.
        protected_words: 保护词表.

    Returns:
        断句会切开保护词返回 True（即不应在此断句）.
    """
    if not tail_text or not incoming or not protected_words:
        return False
    for word in protected_words:
        for split_at in range(1, len(word)):
            head, remainder = word[:split_at], word[split_at:]
            if tail_text.endswith(head) and remainder.startswith(incoming):
                return True
    return False


def tail_is_protected_prefix(tail_text: str, protected_words: Sequence[str]) -> bool:
    """判断待拼句尾是否为某个保护词的**未完成前缀**.

    用于 `tick()`：此时还不知道下一个词是什么，只能先把句子挂起，等新词到达后
    再由 `tail_breaks_protected_word` 决定是否真的拼接。

    Args:
        tail_text: 待拼句子的尾部文本.
        protected_words: 保护词表.

    Returns:
        句尾是保护词的非完整前缀（如 "二"、"请"）返回 True.
    """
    if not tail_text or not protected_words:
        return False
    for word in protected_words:
        # 只取真前缀（不含整个词）：完整保护词结尾已是安全断点
        for split_at in range(1, len(word)):
            if tail_text.endswith(word[:split_at]):
                return True
    return False


class SentenceAssembler:
    """全局词流 → 句子（跨段可拼接，且不会切开保护词）."""

    def __init__(
        self,
        sentence_gap_sec: float = 0.6,
        max_sentence_sec: float = 30.0,
        max_sentence_words: int = 200,
        dedup_tolerance_sec: float = DEDUP_TOLERANCE_SEC,
        protected_words: Optional[Sequence[str]] = None,
        protected_hold_sec: float = DEFAULT_PROTECTED_HOLD_SEC,
    ) -> None:
        """初始化组装器.

        Args:
            sentence_gap_sec: 判定句子边界的词间停顿阈值（秒）.
            max_sentence_sec: 单句最长时长（秒），超长强制封句，防止无限累积.
            max_sentence_words: 单句最多词数，超长强制封句.
            dedup_tolerance_sec: 重复词判定的时间容差（秒）.
            protected_words: 保护词表；None 表示自动从 KEYWORD_LABELS 派生，
                传入空序列表示关闭保护.
            protected_hold_sec: 句尾为保护词前缀时额外挂起封句的最长秒数.
        """
        self.sentence_gap_sec = float(sentence_gap_sec)
        self.max_sentence_sec = float(max_sentence_sec)
        self.max_sentence_words = int(max_sentence_words)
        self.dedup_tolerance_sec = float(dedup_tolerance_sec)
        self.protected_words = tuple(resolve_protected_words(protected_words))
        self.protected_hold_sec = max(0.0, float(protected_hold_sec))
        # 句尾缓冲只需覆盖"保护词长度 - 1"个字符，避免每次拼接整句
        max_word_len = max((len(word) for word in self.protected_words), default=0)
        self._tail_horizon = max(0, max_word_len - 1)
        self._tail_text = ""
        # 已消费词的结束时间上界：用于丢弃重叠区里被重复解码的词
        self._last_end = 0.0
        # 当前尚未封口的句子（词列表）
        self._pending: List[Dict[str, Any]] = []
        self._sealed_sentences = 0

    # ── 对外接口 ────────────────────────────────────────────────

    def push(self, words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """喂入一段解码结果，返回本次"封口"的完整句子.

        Args:
            words: 带 start/end/word 的时间戳词表（某一段的解码结果）.

        Returns:
            已封口的句子列表，每句为按时间排序的词列表.
        """
        if not words:
            return []

        accepted: List[Dict[str, Any]] = []
        for word_item in words:
            if not word_item:
                continue
            text = str(word_item.get("word") or "").strip()
            if not text:
                continue
            start = float(word_item.get("start") or 0.0)
            end = float(word_item.get("end", start) or start)
            if end < start:
                end = start
            # 去重：整词都落在已消费区间内 → 是重叠解码的重复词
            if end <= self._last_end + self.dedup_tolerance_sec:
                continue
            accepted.append({"start": start, "end": end, "word": text})

        if not accepted:
            return []

        accepted.sort(key=lambda item: item["start"])

        sealed: List[List[Dict[str, Any]]] = []
        for word_item in accepted:
            if self._pending and (
                word_item["start"] - self._pending[-1]["end"] > self.sentence_gap_sec
            ):
                # 词完整性保护：切点落在保护词内部时不封句，继续拼接
                if tail_breaks_protected_word(
                    self._tail_text, word_item["word"], self.protected_words
                ):
                    logger.debug(
                        "保护词跨出停顿拼接: 句尾=%r + %r",
                        self._tail_text, word_item["word"],
                    )
                else:
                    sealed.append(self._take_pending())
            self._append_pending(word_item)
            if self._is_overflow():
                sealed.append(self._take_pending())
        return [sentence for sentence in sealed if sentence]

    def tick(self, now_sec: float) -> List[List[Dict[str, Any]]]:
        """按音频游标推进封句：句尾之后停顿已确认时立即交付.

        避免"必须等到下一个词到达才吐上一句"造成的额外等待。

        若句尾仍是某个保护词的未完成前缀（如 "二"、"请"），则挂起封句，最多
        额外等待 `protected_hold_sec` 秒；超过后按普通规则交付，避免无限延迟。

        Args:
            now_sec: 当前已消费到的绝对音频时间（秒）.

        Returns:
            已封口的句子列表（通常 0 或 1 句）.
        """
        if not self._pending:
            return []
        idle_sec = float(now_sec) - self._pending[-1]["end"]
        if idle_sec <= self.sentence_gap_sec:
            return []
        if (
            tail_is_protected_prefix(self._tail_text, self.protected_words)
            and idle_sec <= self.sentence_gap_sec + self.protected_hold_sec
        ):
            return []
        return [self._take_pending()]

    def flush(self) -> List[List[Dict[str, Any]]]:
        """流结束时交付尚未封口的句子.

        Returns:
            剩余句子的列表（0 或 1 句）.
        """
        if not self._pending:
            return []
        return [self._take_pending()]

    def describe(self) -> Dict[str, Any]:
        """返回生效参数，便于日志对照与 A/B 测试.

        Returns:
            参数字典.
        """
        return {
            "sentence_gap_sec": self.sentence_gap_sec,
            "max_sentence_sec": self.max_sentence_sec,
            "max_sentence_words": self.max_sentence_words,
            "protected_words": len(self.protected_words),
            "protected_hold_sec": self.protected_hold_sec,
            "sealed_sentences": self._sealed_sentences,
        }

    # ── 内部 ────────────────────────────────────────────────────

    def _append_pending(self, word_item: Dict[str, Any]) -> None:
        """把词加入待拼缓冲，并维护句尾文本缓冲.

        Args:
            word_item: 待加入的词（含 start/end/word）.
        """
        self._pending.append(word_item)
        self._last_end = max(self._last_end, word_item["end"])
        if self._tail_horizon > 0:
            self._tail_text = (self._tail_text + word_item["word"])[-self._tail_horizon:]

    def _is_overflow(self) -> bool:
        """判断当前句是否超出长度上限，需要强制封口.

        Returns:
            需要强制封口返回 True.
        """
        if not self._pending:
            return False
        if len(self._pending) >= self.max_sentence_words:
            return True
        span = self._pending[-1]["end"] - self._pending[0]["start"]
        return self.max_sentence_sec > 0 and span >= self.max_sentence_sec

    def _take_pending(self) -> List[Dict[str, Any]]:
        """取出当前句并清空待拼缓冲，同时累计封句数.

        Returns:
            已取出的句子词列表；无待拼词时返回空列表.
        """
        if not self._pending:
            return []
        sentence = self._pending
        self._pending = []
        self._tail_text = ""
        self._sealed_sentences += 1
        return sentence
