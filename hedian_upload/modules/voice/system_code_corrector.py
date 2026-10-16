"""九字码（DCS 设备识别码）结构识别与矫正.

从源项目 Nuclear_DCS_Speech_Module_20260810 合并而来（纯标准库实现，无第三方依赖）。

相比 ``extract_nine_char_codes`` 的「恰好 9 位字母数字」硬匹配，本模块按
**DCS 设备码的真实结构**识别，并用法定的系统码/设备类型码表做校验，因此能
处理 ASR 口语化输出带来的各种变形：

    9 位（完整码）： 1 位机组号 + 3 位系统码 + 3 位设备号 + 2 位设备类型码
                     形如 ``1RCP001PO``
    8 位（省略机组号）：3 位系统码 + 3 位设备号 + 2 位设备类型码
                     形如 ``RIS208VP``

核心能力：

    * 中文数字读音还原（零/洞/幺/腰/两 …）与字母音译还原（诶/批/威/艾斯 …）；
    * 字母位上的数字歧义消解（``一``/``1``/``１`` → ``E``），数字位不受影响；
    * 允许口语分隔符（空格、标点、"短横线/横杠"）与机组号后的"号/号机/号机组"；
    * 相邻符号的时间间隔约束（默认 1.5s），避免把句中零散字符误拼成设备码；
    * 系统码必须命中 `system_codes`、设备类型码必须命中 `device_type_codes`，
      两条码表缺失任一条即判定为误拼，不产出结果。

对外主入口为 :func:`correct_system_codes`。
"""
import copy
import json
import os
import re
import unicodedata


DEFAULT_RULES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "system_code_rules.json")
)

# 规则表按路径缓存：语音流水线是逐句调用的，避免每句重复读盘解析码表
_RULES_CACHE = {}

DIGIT_ALIASES = {
    "零": "0",
    "〇": "0",
    "洞": "0",
    "一": "1",
    "幺": "1",
    "腰": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}

# These aliases are only interpreted inside a code-shaped candidate. Ordinary
# Chinese text is never converted globally.
LETTER_ALIASES = {
    "诶": "A",
    "哎": "A",
    "比": "B",
    "毕": "B",
    "西": "C",
    "迪": "D",
    "伊": "E",
    "衣": "E",
    "艾弗": "F",
    "埃弗": "F",
    "吉": "G",
    "艾尺": "H",
    "埃尺": "H",
    "艾": "I",
    "爱": "I",
    "杰": "J",
    "开": "K",
    "凯": "K",
    "艾勒": "L",
    "埃勒": "L",
    "艾姆": "M",
    "埃姆": "M",
    "艾恩": "N",
    "埃恩": "N",
    "恩": "N",
    "欧": "O",
    "皮": "P",
    "批": "P",
    "丘": "Q",
    "秋": "Q",
    "阿尔": "R",
    "啊尔": "R",
    "艾儿": "R",
    "艾尔": "R",
    "艾斯": "S",
    "爱斯": "S",
    "提": "T",
    "梯": "T",
    "优": "U",
    "尤": "U",
    "维": "V",
    "威": "V",
    "微": "V",
    "达不溜": "W",
    "大不留": "W",
    "达布留": "W",
    "艾克斯": "X",
    "埃克斯": "X",
    "歪": "Y",
    "外": "Y",
    "贼德": "Z",
    "泽德": "Z",
    "哲德": "Z",
}

AMBIGUOUS_SYMBOL_ALIASES = {
    "一": [
        {
            "value": "E",
            "kind": "letter",
            "conversion": "ambiguous_one_as_e",
        }
    ],
    "1": [
        {
            "value": "E",
            "kind": "letter",
            "conversion": "ambiguous_1_as_e",
        }
    ],
    "１": [
        {
            "value": "E",
            "kind": "letter",
            "conversion": "ambiguous_1_as_e",
        }
    ],
}

_LETTER_ALIAS_ITEMS = sorted(
    LETTER_ALIASES.items(),
    key=lambda item: (-len(item[0]), item[0]),
)
_GENERAL_GAP_WORDS = ("短横线", "横杠", "横线", "杠")
_SEPARATORS_RE = re.compile(r"[\s,，.。:：;；、\-—_／/\\]+")
_STRONG_BOUNDARY_RE = re.compile(r"[,，.。:：;；、]")


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_system_code_rules(path=None):
    """加载九字码码表（按路径缓存）.

    Args:
        path: 规则表 JSON 路径；None 表示使用 DEFAULT_RULES_PATH.

    Returns:
        含 path / source_document / system_codes / device_type_codes 的字典.

    Raises:
        ValueError: 码表为空或缺少合法代码项时抛出.
    """
    rules_path = path or DEFAULT_RULES_PATH
    cached = _RULES_CACHE.get(rules_path)
    if cached is not None:
        return cached

    with open(rules_path, "r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    systems = {
        str(item).strip().upper()
        for item in payload.get("system_codes", [])
        if re.fullmatch(r"[A-Z]{3}", str(item).strip().upper())
    }
    device_types = {
        str(item).strip().upper()
        for item in payload.get("device_type_codes", [])
        if re.fullmatch(r"[A-Z]{2}", str(item).strip().upper())
    }
    if not systems or not device_types:
        raise ValueError(f"系统代码规则缺少合法代码表: {rules_path}")
    rules = {
        "path": rules_path,
        "source_document": payload.get("source_document", ""),
        "system_codes": systems,
        "device_type_codes": device_types,
    }
    _RULES_CACHE[rules_path] = rules
    return rules


def _time_for_source_span(word_ranges, source_start, source_end, segment):
    hits = [
        item
        for item in word_ranges
        if item["source_end"] > source_start and item["source_start"] < source_end
    ]
    if not hits:
        start = _safe_float(segment.get("start"))
        end = max(start, _safe_float(segment.get("end"), start))
        return start, end
    return (
        min(item["start"] for item in hits),
        max(item["end"] for item in hits),
    )


def _segment_source(segment):
    words = segment.get("words") or []
    if not words:
        return str(segment.get("text") or ""), []

    parts = []
    ranges = []
    cursor = 0
    for index, word in enumerate(words):
        text = str(word.get("word") or word.get("text") or "")
        if not text:
            continue
        start = cursor
        parts.append(text)
        cursor += len(text)
        ranges.append({
            "word_index": index,
            "source_start": start,
            "source_end": cursor,
            "start": _safe_float(word.get("start"), _safe_float(segment.get("start"))),
            "end": max(
                _safe_float(word.get("start"), _safe_float(segment.get("start"))),
                _safe_float(word.get("end"), _safe_float(segment.get("end"))),
            ),
        })
    source = "".join(parts)
    if source:
        return source, ranges
    return str(segment.get("text") or ""), []


def _mapped_symbols(text, word_ranges, segment):
    symbols = []
    cursor = 0
    while cursor < len(text):
        character = text[cursor]
        normalized = unicodedata.normalize("NFKC", character).upper()
        value = ""
        kind = ""
        end = cursor + 1

        if len(normalized) == 1 and normalized in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            value = normalized
            kind = "ascii_letter" if character.isascii() else "fullwidth_letter"
        elif len(normalized) == 1 and normalized in "0123456789":
            value = normalized
            kind = "ascii_digit" if character.isascii() else "fullwidth_digit"
        elif character in DIGIT_ALIASES:
            value = DIGIT_ALIASES[character]
            kind = "chinese_digit"
        else:
            for alias, letter in _LETTER_ALIAS_ITEMS:
                if text.startswith(alias, cursor):
                    value = letter
                    kind = "phonetic_letter"
                    end = cursor + len(alias)
                    break

        if not value:
            cursor += 1
            continue

        start_time, end_time = _time_for_source_span(
            word_ranges,
            cursor,
            end,
            segment,
        )
        symbols.append({
            "value": value,
            "kind": "digit" if value.isdigit() else "letter",
            "conversion": kind,
            "alternatives": copy.deepcopy(
                AMBIGUOUS_SYMBOL_ALIASES.get(character, [])
            ),
            "source_start": cursor,
            "source_end": end,
            "start": start_time,
            "end": end_time,
        })
        cursor = end
    return symbols


def _gap_is_allowed(gap, boundary_index, has_unit):
    normalized = unicodedata.normalize("NFKC", gap)
    has_strong_boundary = bool(_STRONG_BOUNDARY_RE.search(normalized))
    for word in _GENERAL_GAP_WORDS:
        normalized = normalized.replace(word, "")
    normalized = _SEPARATORS_RE.sub("", normalized)
    if not normalized:
        if has_unit and boundary_index == 0 and has_strong_boundary:
            return False
        return True
    return bool(
        has_unit
        and boundary_index == 0
        and normalized in {"号", "号机", "号机组"}
    )


def _candidate_is_contiguous(window, source, has_unit, max_symbol_pause):
    for index, (left, right) in enumerate(zip(window, window[1:])):
        gap = source[left["source_end"]:right["source_start"]]
        if not _gap_is_allowed(gap, index, has_unit):
            return False
        if (
            max_symbol_pause > 0
            and right["start"] - left["end"] > max_symbol_pause
        ):
            return False
    return True


def _candidate_components(window, has_unit):
    values = "".join(item["value"] for item in window)
    if has_unit:
        return {
            "canonical": values,
            "unit_number": values[0],
            "system_code": values[1:4],
            "equipment_number": values[4:7],
            "device_type": values[7:9],
        }
    return {
        "canonical": values,
        "unit_number": "",
        "system_code": values[0:3],
        "equipment_number": values[3:6],
        "device_type": values[6:8],
    }


def _resolve_window_kinds(window, pattern):
    resolved = []
    for symbol, expected in zip(window, pattern):
        expected_kind = "digit" if expected == "d" else "letter"
        if symbol["kind"] == expected_kind:
            resolved.append(symbol)
            continue

        alternative = next(
            (
                item
                for item in symbol.get("alternatives", [])
                if item.get("kind") == expected_kind
            ),
            None,
        )
        if alternative is None:
            return None
        resolved.append({
            **symbol,
            "value": alternative["value"],
            "kind": alternative["kind"],
            "conversion": alternative["conversion"],
        })
    return resolved


def _find_candidates(
    source,
    symbols,
    rules,
    allow_unit_omitted,
    max_symbol_pause,
):
    patterns = [
        (9, "dlll" + "ddd" + "ll", True),
        (8, "lll" + "ddd" + "ll", False),
    ]
    candidates = []
    for symbol_index in range(len(symbols)):
        for length, pattern, has_unit in patterns:
            if not has_unit and not allow_unit_omitted:
                continue
            window = symbols[symbol_index:symbol_index + length]
            if len(window) != length:
                continue
            resolved_window = _resolve_window_kinds(window, pattern)
            if resolved_window is None:
                continue
            if not _candidate_is_contiguous(
                resolved_window,
                source,
                has_unit=has_unit,
                max_symbol_pause=max_symbol_pause,
            ):
                continue

            components = _candidate_components(resolved_window, has_unit)
            if components["system_code"] not in rules["system_codes"]:
                continue
            if components["device_type"] not in rules["device_type_codes"]:
                continue

            source_start = window[0]["source_start"]
            source_end = window[-1]["source_end"]
            conversions = sorted({
                item["conversion"]
                for item in resolved_window
                if item["conversion"] not in {"ascii_letter", "ascii_digit"}
            })
            candidates.append({
                **components,
                "has_unit_number": has_unit,
                "source_start": source_start,
                "source_end": source_end,
                "observed_text": source[source_start:source_end],
                "start": min(item["start"] for item in resolved_window),
                "end": max(item["end"] for item in resolved_window),
                "conversion_types": conversions,
            })

    candidates.sort(key=lambda item: (
        item["source_start"],
        -int(item["has_unit_number"]),
        -(item["source_end"] - item["source_start"]),
    ))
    selected = []
    for candidate in candidates:
        if any(
            candidate["source_start"] < item["source_end"]
            and candidate["source_end"] > item["source_start"]
            for item in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _replace_source(source, replacements):
    result = []
    cursor = 0
    for item in replacements:
        result.append(source[cursor:item["source_start"]])
        result.append(item["canonical"])
        cursor = item["source_end"]
    result.append(source[cursor:])
    return "".join(result)


def _replace_word_texts(words, word_ranges, source, replacements):
    output = copy.deepcopy(words)
    ranges_by_index = {item["word_index"]: item for item in word_ranges}
    for index, word in enumerate(output):
        word_range = ranges_by_index.get(index)
        if word_range is None:
            continue
        word_start = word_range["source_start"]
        word_end = word_range["source_end"]
        original = source[word_start:word_end]
        parts = []
        cursor = word_start
        for replacement in replacements:
            if replacement["source_end"] <= word_start:
                continue
            if replacement["source_start"] >= word_end:
                break
            if replacement["source_start"] >= word_start:
                parts.append(source[cursor:replacement["source_start"]])
                parts.append(replacement["canonical"])
            cursor = max(cursor, min(word_end, replacement["source_end"]))
        parts.append(source[cursor:word_end])
        corrected = "".join(parts)
        if corrected != original:
            word["asr_word"] = original
            word["word"] = corrected
    return output


def correct_system_codes(
    segments,
    *,
    rules_path=None,
    enabled=True,
    allow_unit_omitted=True,
    max_symbol_pause=1.5,
):
    """按 DCS 设备码结构识别并矫正文本中的九字码.

    Args:
        segments: 段列表，每项需含 ``text``，可选含 ``words``（元素带
            ``word``/``start``/``end``）与 ``start``/``end``/``segment_id``.
        rules_path: 码表 JSON 路径；None 表示使用 DEFAULT_RULES_PATH.
        enabled: False 时原样返回，不做任何矫正.
        allow_unit_omitted: 是否允许 8 位（省略机组号）设备码.
        max_symbol_pause: 相邻符号允许的最大停顿（秒）；<=0 表示不校验停顿.

    Returns:
        (矫正后的段列表, 设备码事件列表)；每段被改写时保留 raw_text/asr_text
        并挂上 system_code_event_ids，词表原文存入 asr_word.
    """
    corrected_segments = copy.deepcopy(segments or [])
    if not enabled:
        return corrected_segments, []

    rules = load_system_code_rules(rules_path)
    events = []
    for segment_index, segment in enumerate(corrected_segments):
        source, word_ranges = _segment_source(segment)
        if not source:
            continue
        symbols = _mapped_symbols(source, word_ranges, segment)
        replacements = _find_candidates(
            source,
            symbols,
            rules,
            allow_unit_omitted=allow_unit_omitted,
            max_symbol_pause=max_symbol_pause,
        )
        if not replacements:
            continue

        segment_event_ids = []
        for replacement in replacements:
            event_id = len(events)
            segment_event_ids.append(event_id)
            corrected = replacement["observed_text"] != replacement["canonical"]
            events.append({
                "system_code_event_id": event_id,
                "keyword_id": f"system_code_{replacement['canonical'].lower()}",
                "type": "device_code",
                "canonical_text": replacement["canonical"],
                "canonical_normalized": replacement["canonical"],
                "observed_text": replacement["observed_text"],
                "observed_normalized": replacement["canonical"],
                "matched_alias": replacement["canonical"],
                "match_method": (
                    "system_rule_normalized"
                    if replacement["conversion_types"] or corrected
                    else "system_rule_exact"
                ),
                "match_score": 0.98 if replacement["conversion_types"] else 1.0,
                "corrected": corrected,
                "start": round(replacement["start"], 3),
                "end": round(replacement["end"], 3),
                "first_word_start": round(replacement["start"], 3),
                "last_word_end": round(replacement["end"], 3),
                "source_segment_id": segment.get("segment_id", segment_index),
                "source_text": source,
                "unit_number": replacement["unit_number"],
                "system_code": replacement["system_code"],
                "equipment_number": replacement["equipment_number"],
                "device_type": replacement["device_type"],
                "has_unit_number": replacement["has_unit_number"],
                "conversion_types": replacement["conversion_types"],
                "rule_source": rules["source_document"],
            })

        corrected_text = _replace_source(source, replacements)
        segment.setdefault("raw_text", source)
        segment.setdefault("asr_text", segment["raw_text"])
        segment["text"] = corrected_text
        segment["system_code_event_ids"] = segment_event_ids
        if word_ranges:
            segment["words"] = _replace_word_texts(
                segment.get("words") or [],
                word_ranges,
                source,
                replacements,
            )

    return corrected_segments, events
