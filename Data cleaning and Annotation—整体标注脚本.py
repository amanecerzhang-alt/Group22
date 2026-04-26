from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from volcenginesdkarkruntime import Ark


BASE_DIR = Path("/Users/amamnecerzhang/Desktop/未命名文件夹 2")
OUTPUT_DIR = Path("/Users/amamnecerzhang/Desktop/output_145_preserve")
STATE_PATH = OUTPUT_DIR / "progress.json"
USAGE_PATH = OUTPUT_DIR / "usage.json"
INPUT_FILES = [
    "_合并_01_准备与引流(1).csv",
    "_合并_02_建立信任与诱导(1).csv",
    "_合并_03_交易与收割(1).csv",
    "_合并_04_辅助知识与工具(1).csv",
]

TEXT_SOURCE_COLS = ["子分类", "文件名", "关键词", "提及金额", "话术示例"]

RULE_COLUMNS = [
    "心理策略_贪婪",
    "心理策略_恐惧",
    "心理策略_信任权威",
    "心理策略_情感依赖",
    "心理策略_从众",
    "心理策略_紧迫感",
    "紧迫感等级",
    "最大金额",
]

API_COLUMNS = ["诈骗人设_API", "话术功能_API_1", "话术功能_API_2", "话术功能_API_3"]

TACTIC_KEYWORDS = {
    "心理策略_贪婪": ["收益", "赚", "盈利", "翻倍", "回报", "高回报", "利润", "财富自由", "本金", "扭亏为盈", "资产", "节点", "%"],
    "心理策略_恐惧": ["冻结", "没收", "违法", "洗钱", "监管", "担保金", "保证金", "无法提取", "投诉", "后果自负", "风险", "封号", "账户异常", "处罚", "起诉"],
    "心理策略_信任权威": ["老师", "教授", "分析师", "专家", "官方", "国家", "政府", "监管局", "银行", "平台", "交易所", "牌照", "客服", "企业家", "金融中心", "MSB", "叔叔", "团队"],
    "心理策略_情感依赖": ["亲爱的", "宝贝", "想你", "喜欢你", "爱你", "缘分", "真诚", "陪伴", "见面", "拥抱", "飞吻", "两个人", "未来", "恋爱", "浪漫"],
    "心理策略_从众": ["很多人", "大家都", "粉丝", "会员", "群组", "大量投资者", "都在", "一起赚钱", "留言", "前来咨询", "都想", "别人"],
    "心理策略_紧迫感": ["马上", "立刻", "尽快", "抓紧", "截止", "最后", "错过", "来不及", "24小时", "72小时", "今天", "名额有限", "限时", "稍纵即逝", "规定时间", "5分钟后"],
}

URGENCY_HIGH = ["24小时", "72小时", "今天截止", "截止日期", "规定时间内", "5分钟后", "最后时间", "立刻", "马上", "立即", "尽快完成", "否则"]
URGENCY_MEDIUM = ["错过", "名额有限", "机会", "尽快", "抓紧", "这次机会", "不等人", "稍纵即逝", "限时", "早点"]

MONEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"[ \t]*(?P<unit>亿|万|千|百|[Ww])?"
    r"[ \t]*(?P<currency>美元|美金|刀|元|块|人民币|欧元|澳元|港币|USDT|USD|U)?"
    r"(?![A-Za-z0-9])"
)
YEAR_OR_DATE_PATTERN = re.compile(r"((19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2})|((19|20)\d{2}\s*年)|(\d{1,2}:\d{2})")
NON_MONEY_SUFFIX_PATTERN = re.compile(r"^(岁|周年|号|日|月|点|强|届|条|位|名|次|节|区|楼|期|分钟|小时|天|年|枚|个|笔|倍|%)")
MONEY_CONTEXT_KEYWORDS = [
    "金额", "资金", "本金", "入金", "充值", "存入", "转入", "转账", "汇款", "提现", "提取", "提款", "余额", "账户",
    "收益", "盈利", "利润", "赚", "亏", "佣金", "担保金", "保证金", "投资", "订单", "发货资金", "周转资金", "启动资金",
    "欧元", "美元", "美金", "刀", "元", "块", "人民币", "港币", "澳元", "usdt",
]

PERSONA_LABELS = ["投资导师", "成功人设", "恋人", "官方客服", "内部人", "同伴"]
FUNCTION_LABELS = [
    "人设包装", "破冰搭讪", "情感拉拢", "引流转平台", "知识铺垫", "利益诱导",
    "开户引导", "充值催单", "案例展示", "打消顾虑", "风险威胁", "善后维稳",
]
SUBCATEGORY_TO_FUNCTION = {
    "01_虚假人设打造": "人设包装",
    "02_引流与平台操作": "引流转平台",
    "03_虚假平台工具包装": "知识铺垫",
    "01_破冰与日常聊天": "破冰搭讪",
    "02_情感建设与拉近关系": "情感拉拢",
    "03_话题切入（切客）": "利益诱导",
    "04_打消顾虑与铺垫": "打消顾虑",
    "01_开户与操作指导": "开户引导",
    "02_首充与续充（杀猪）": "充值催单",
    "03_应对质疑与维稳": "善后维稳",
    "01_背景知识": "知识铺垫",
    "02_流程与管理": "开户引导",
    "03_特殊案例与地区": "案例展示",
}
PERSONA_HINTS = {
    "投资导师": ["老师", "教授", "分析师", "带你", "指导", "教你", "投资经验"],
    "成功人设": ["创业", "财富自由", "开公司", "资产", "成功", "企业家", "老板"],
    "恋人": ["亲爱的", "宝贝", "想你", "见面", "恋爱", "喜欢你", "两个人"],
    "官方客服": ["客服", "平台", "交易所", "上传身份证", "审核", "担保", "保证金"],
    "内部人": ["内部", "内幕", "渠道", "消息", "名额", "叔叔", "团队消息"],
    "同伴": ["我也在做", "一起赚钱", "我先带你", "我也是投资者", "一起做"],
}


def split_records(data: bytes) -> list[bytes]:
    records: list[bytes] = []
    start = 0
    in_quotes = False
    i = 0
    while i < len(data):
        b = data[i]
        if b == 34:
            if in_quotes and i + 1 < len(data) and data[i + 1] == 34:
                i += 2
                continue
            in_quotes = not in_quotes
        elif not in_quotes and b in (10, 13):
            if b == 13 and i + 1 < len(data) and data[i + 1] == 10:
                i += 1
            records.append(data[start : i + 1])
            start = i + 1
        i += 1
    if start < len(data):
        records.append(data[start:])
    return records


def decode_record(record: bytes) -> str:
    return record.decode("gbk", errors="surrogateescape")


def parse_csv_record(record: bytes) -> list[str]:
    reader = csv.reader(io.StringIO(decode_record(record)))
    return next(reader)


def csv_encode_field(value: object) -> bytes:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(["" if value is None else str(value)])
    return buf.getvalue().encode("gbk", errors="surrogateescape")


def append_columns_to_record(record: bytes, values: list[object]) -> bytes:
    if record.endswith(b"\r\n"):
        body, newline = record[:-2], b"\r\n"
    elif record.endswith(b"\n"):
        body, newline = record[:-1], b"\n"
    elif record.endswith(b"\r"):
        body, newline = record[:-1], b"\r"
    else:
        body, newline = record, b""
    suffix = b"".join(b"," + csv_encode_field(v) for v in values)
    return body + suffix + newline


def combine_text(row: dict[str, str]) -> str:
    parts: list[str] = []
    for col in TEXT_SOURCE_COLS:
        value = row.get(col, "")
        if value:
            parts.append(value.strip())
    for key, value in row.items():
        if key.startswith("完整内容_") and value:
            parts.append(value.strip())
    return "\n".join(parts)


def match_any_keyword(text: str, keywords: list[str]) -> str:
    return "1" if any(keyword in text for keyword in keywords) else "0"


def urgency_level(text: str) -> str:
    if any(keyword in text for keyword in URGENCY_HIGH):
        return "高"
    if any(keyword in text for keyword in URGENCY_MEDIUM):
        return "中"
    return "低"


def normalize_amount(value: float, unit: str | None) -> float:
    unit = unit or ""
    if unit == "亿":
        return value * 100000000
    if unit in {"万", "W", "w"}:
        return value * 10000
    if unit == "千":
        return value * 1000
    if unit == "百":
        return value * 100
    return value


def has_amount_context(context: str) -> bool:
    lowered = context.lower()
    return any(keyword in lowered for keyword in MONEY_CONTEXT_KEYWORDS)


def extract_max_amount(full_text: str) -> str:
    numbers: list[float] = []
    for match in MONEY_PATTERN.finditer(full_text):
        matched_text = match.group(0).strip()
        if not matched_text:
            continue
        start = match.start()
        end = match.end()
        left = full_text[max(0, start - 10) : start]
        right = full_text[end : min(len(full_text), end + 10)]
        local_context = left + matched_text + right
        if YEAR_OR_DATE_PATTERN.search(local_context):
            continue
        if right.startswith("%"):
            continue
        if NON_MONEY_SUFFIX_PATTERN.match(right.strip()):
            continue
        try:
            value = float(match.group("num").replace(",", ""))
        except ValueError:
            continue
        unit = match.group("unit")
        currency = match.group("currency")
        explicit_money = bool(unit or currency)
        contextual_money = has_amount_context(local_context)
        if not explicit_money and not contextual_money:
            continue
        if not explicit_money and value < 100:
            continue
        if value >= 1900 and value <= 2100 and not explicit_money:
            continue
        numbers.append(normalize_amount(value, unit))
    if not numbers:
        return ""
    max_value = max(numbers)
    return str(int(max_value)) if float(max_value).is_integer() else str(max_value)


def build_rule_annotation(row: dict[str, str]) -> list[object]:
    full_text = combine_text(row)
    values: list[object] = []
    for col in RULE_COLUMNS[:-2]:
        values.append(match_any_keyword(full_text, TACTIC_KEYWORDS[col]))
    values.append(urgency_level(full_text))
    values.append(extract_max_amount(full_text))
    return values


def sanitize_for_api(text: str) -> str:
    return text.encode("gbk", errors="ignore").decode("gbk", errors="ignore")


def coarse_persona_hint(text: str) -> str:
    scores: list[tuple[int, str]] = []
    for label, hints in PERSONA_HINTS.items():
        score = sum(1 for hint in hints if hint in text)
        scores.append((score, label))
    scores.sort(reverse=True)
    return scores[0][1] if scores and scores[0][0] > 0 else ""


def build_prompt(row: dict[str, str]) -> dict[str, object]:
    subcategory = row.get("子分类", "")
    full_text = sanitize_for_api(combine_text(row))
    function_hint = SUBCATEGORY_TO_FUNCTION.get(subcategory, "")
    persona_hint = coarse_persona_hint(full_text)
    instructions = (
        "你是中文诈骗话术标注助手。"
        "请只输出JSON。"
        "维度二“诈骗人设”必须从这些值中选一个：" + "、".join(PERSONA_LABELS) + "。"
        "维度三“话术功能”可以多选，最少1个，最多3个，必须从这些值中选择：" + "、".join(FUNCTION_LABELS) + "。"
        "优先根据正文语义判断，不要被零散关键词误导。"
        "如果子分类提示与正文冲突，以正文为准。"
        "如果功能很多，请只保留最核心的1到3个，按重要性排序。"
    )
    user_text = {
        "子分类": subcategory,
        "文件名": row.get("文件名", ""),
        "关键词": row.get("关键词", ""),
        "提及金额": row.get("提及金额", ""),
        "子分类映射_话术功能提示": function_hint,
        "关键词粗分_诈骗人设提示": persona_hint,
        "正文": full_text[:12000],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "诈骗人设": {"type": "string", "enum": PERSONA_LABELS},
            "话术功能": {
                "type": "array",
                "items": {"type": "string", "enum": FUNCTION_LABELS},
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
            },
        },
        "required": ["诈骗人设", "话术功能"],
    }
    return {"instructions": instructions, "user_text": user_text, "schema": schema}


def extract_output_text(resp: Any) -> str:
    text_value = None
    for item in resp.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text_value = content.text
                    break
        if text_value:
            break
    if text_value:
        return text_value
    status = getattr(resp, "status", None)
    incomplete_details = getattr(resp, "incomplete_details", None)
    raise RuntimeError(f"API 未返回可解析文本。status={status!r}, incomplete_details={incomplete_details!r}, response={resp!r}")


def call_ark_label(row: dict[str, str], client: Ark, model: str) -> list[str]:
    prompt = build_prompt(row)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": prompt["instructions"]}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt["user_text"], ensure_ascii=False)}]},
        ],
        text={"format": {"type": "json_schema", "name": "annotation_result", "schema": prompt["schema"], "strict": True}},
    )
    result = json.loads(extract_output_text(response))
    funcs = list(result["话术功能"])[:3]
    while len(funcs) < 3:
        funcs.append("")
    return [result["诈骗人设"], funcs[0], funcs[1], funcs[2]]


def call_ark_batch_once(rows: list[dict[str, str]], client: Ark, model: str) -> tuple[list[list[str]], dict[str, int]]:
    payload_rows = []
    for i, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        payload_rows.append(
            {
                "id": i,
                "文件名": row.get("文件名", ""),
                "子分类": row.get("子分类", ""),
                "关键词提示_诈骗人设": prompt["user_text"]["关键词粗分_诈骗人设提示"],
                "子分类提示_话术功能": prompt["user_text"]["子分类映射_话术功能提示"],
                "正文": prompt["user_text"]["正文"],
            }
        )
    instructions = (
        "你是中文诈骗话术标注助手。请只输出JSON数组。"
        "每条记录都必须返回 id、诈骗人设、话术功能。"
        "诈骗人设必须从这些值中选一个：投资导师、成功人设、恋人、官方客服、内部人、同伴。"
        "话术功能可以多选，最少1个，最多3个，必须从这些值中选择："
        "人设包装、破冰搭讪、情感拉拢、引流转平台、知识铺垫、利益诱导、开户引导、充值催单、案例展示、打消顾虑、风险威胁、善后维稳。"
        "如果功能很多，只保留最核心的1到3个并按重要性排序。"
    )
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "id": {"type": "integer"},
                "诈骗人设": {"type": "string", "enum": PERSONA_LABELS},
                "话术功能": {
                    "type": "array",
                    "items": {"type": "string", "enum": FUNCTION_LABELS},
                    "minItems": 1,
                    "maxItems": 3,
                    "uniqueItems": True,
                },
            },
            "required": ["id", "诈骗人设", "话术功能"],
        },
    }
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": instructions}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload_rows, ensure_ascii=False)}]},
        ],
        text={"format": {"type": "json_schema", "name": "batch_result", "schema": schema, "strict": True}},
    )
    results = json.loads(extract_output_text(resp))
    by_id = {item["id"]: item for item in results}
    labels: list[list[str]] = []
    for i in range(1, len(rows) + 1):
        result = by_id[i]
        funcs = list(result["话术功能"])[:3]
        while len(funcs) < 3:
            funcs.append("")
        labels.append([result["诈骗人设"], funcs[0], funcs[1], funcs[2]])
    usage = {
        "input_tokens": int(getattr(resp.usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(resp.usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return labels, usage


def empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def add_usage(total: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    total["input_tokens"] += delta["input_tokens"]
    total["output_tokens"] += delta["output_tokens"]
    total["total_tokens"] += delta["total_tokens"]
    return total


def call_ark_batch(rows: list[dict[str, str]], client: Ark, model: str) -> tuple[list[list[str]], dict[str, int], int]:
    try:
        labels, usage = call_ark_batch_once(rows, client=client, model=model)
        return labels, usage, 1
    except Exception as exc:
        if len(rows) == 1:
            return [call_ark_label(rows[0], client=client, model=model)], empty_usage(), 2
        midpoint = max(1, len(rows) // 2)
        left_labels, left_usage, left_requests = call_ark_batch(rows[:midpoint], client=client, model=model)
        right_labels, right_usage, right_requests = call_ark_batch(rows[midpoint:], client=client, model=model)
        merged_usage = add_usage(empty_usage(), left_usage)
        add_usage(merged_usage, right_usage)
        print(f"[降级重试] 原批次 {len(rows)} 条失败，已拆分处理。原因: {exc}")
        return left_labels + right_labels, merged_usage, left_requests + right_requests


def load_state() -> dict[str, object]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"files": {}}


def save_state(state: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_usage() -> dict[str, int]:
    if USAGE_PATH.exists():
        return json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    return {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def save_usage(usage: dict[str, int]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    USAGE_PATH.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")


def write_partial_output(output_path: Path, records: list[bytes]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"".join(records))


def target_file_list(env_name: str) -> list[str]:
    return [name.strip() for name in os.environ.get(env_name, "").split(",") if name.strip()]


def annotate_rules() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_files = target_file_list("RULE_TARGET_FILES")
    for file_name in INPUT_FILES:
        if target_files and file_name not in target_files:
            continue
        input_path = BASE_DIR / file_name
        output_path = OUTPUT_DIR / file_name.replace(".csv", "_仅追加标注145.csv")
        data = input_path.read_bytes()
        records = split_records(data)
        header = parse_csv_record(records[0])
        out_records = [append_columns_to_record(records[0], RULE_COLUMNS)]
        for record in records[1:]:
            if not record.strip():
                out_records.append(record)
                continue
            row_values = parse_csv_record(record)
            row = {header[i]: row_values[i] if i < len(row_values) else "" for i in range(len(header))}
            out_records.append(append_columns_to_record(record, build_rule_annotation(row)))
        output_path.write_bytes(b"".join(out_records))
        print(f"[规则标注] {file_name} -> {output_path}")


def annotate_api() -> None:
    api_key = os.environ.get("ARK_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing ARK_API_KEY")
    model = os.environ.get("ARK_MODEL", "doubao-seed-2-0-lite-260215")
    sleep_s = float(os.environ.get("ARK_SLEEP_SECONDS", "0"))
    save_every = int(os.environ.get("ARK_SAVE_EVERY", "20"))
    batch_size = int(os.environ.get("ARK_BATCH_SIZE", "3"))
    target_files = target_file_list("ARK_TARGET_FILES")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=api_key)
    state = load_state()
    usage_totals = load_usage()
    file_state = state.setdefault("files", {})

    grand_total = 0
    grand_done = 0
    per_file_meta: dict[str, dict[str, object]] = {}
    for file_name in INPUT_FILES:
        if target_files and file_name not in target_files:
            continue
        records = split_records((BASE_DIR / file_name).read_bytes())
        total_rows = len(records) - 1
        done_rows = int(file_state.get(file_name, {}).get("done_rows", 0))
        per_file_meta[file_name] = {"total_rows": total_rows, "records": records}
        grand_total += total_rows
        grand_done += min(done_rows, total_rows)

    print(f"总任务: 已完成 {grand_done}/{grand_total}, 剩余 {grand_total - grand_done}")

    for file_name in INPUT_FILES:
        if target_files and file_name not in target_files:
            continue
        output_path = OUTPUT_DIR / file_name.replace(".csv", "_仅追加标注23_API.csv")
        records = per_file_meta[file_name]["records"]
        total_rows = int(per_file_meta[file_name]["total_rows"])
        header = parse_csv_record(records[0])
        file_progress = file_state.setdefault(file_name, {})
        done_rows = int(file_progress.get("done_rows", 0))

        if output_path.exists() and done_rows > 0:
            out_records = split_records(output_path.read_bytes())
        else:
            out_records = [append_columns_to_record(records[0], API_COLUMNS)]
            done_rows = 0

        print(f"开始文件: {file_name}，已完成 {done_rows}/{total_rows}，剩余 {total_rows - done_rows}")
        idx = done_rows + 1
        while idx <= total_rows:
            batch_records: list[bytes] = []
            batch_rows: list[dict[str, str]] = []
            batch_indices: list[int] = []
            while idx <= total_rows and len(batch_rows) < batch_size:
                record = records[idx]
                if not record.strip():
                    out_records.append(record)
                    file_progress["done_rows"] = idx
                    idx += 1
                    continue
                row_values = parse_csv_record(record)
                row = {header[i]: row_values[i] if i < len(row_values) else "" for i in range(len(header))}
                batch_records.append(record)
                batch_rows.append(row)
                batch_indices.append(idx)
                idx += 1
            if not batch_rows:
                continue

            labels_batch, usage, request_count = call_ark_batch(batch_rows, client=client, model=model)
            usage_totals["requests"] += request_count
            usage_totals["input_tokens"] += usage["input_tokens"]
            usage_totals["output_tokens"] += usage["output_tokens"]
            usage_totals["total_tokens"] += usage["total_tokens"]

            for row_idx, record, labels in zip(batch_indices, batch_records, labels_batch):
                out_records.append(append_columns_to_record(record, labels))
                file_progress["done_rows"] = row_idx
                file_progress["output_path"] = str(output_path)

            current_done = int(file_progress["done_rows"])
            if current_done % save_every == 0 or current_done == total_rows:
                write_partial_output(output_path, out_records)
                save_state(state)
                save_usage(usage_totals)
                completed_all = sum(
                    min(int(file_state.get(name, {}).get("done_rows", 0)), int(meta["total_rows"]))
                    for name, meta in per_file_meta.items()
                )
                remaining_all = grand_total - completed_all
                print(f"[进度] 文件 {file_name}: {current_done}/{total_rows}，总进度 {completed_all}/{grand_total}，剩余 {remaining_all}，累计tokens {usage_totals['total_tokens']}")

            if sleep_s > 0:
                time.sleep(sleep_s)

        write_partial_output(output_path, out_records)
        save_state(state)
        save_usage(usage_totals)
        print(f"完成文件: {file_name}")
    print("全部完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整体标注脚本：支持规则标注、API 标注或全部执行。")
    parser.add_argument("--mode", choices=["rule", "api", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"rule", "all"}:
        annotate_rules()
    if args.mode in {"api", "all"}:
        annotate_api()


if __name__ == "__main__":
    main()
