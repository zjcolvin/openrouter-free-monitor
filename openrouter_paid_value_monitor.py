#!/usr/bin/env python3
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

MODELS_API = "https://openrouter.ai/api/v1/models"
DEEPINFRA_API = "https://api.deepinfra.com/models/list"
DEFAULT_STATE_FILE = "output/openrouter_paid_state.json"
DEFAULT_HISTORY_FILE = "output/openrouter_paid_history.json"
DISCORD_LIMIT = 1900

# Reference Direct API Rates (USD per 1 Million Tokens)
OFFICIAL_PRICING = {
    "anthropic/claude-3-opus": {"prompt": 15.00, "completion": 75.00},
    "anthropic/claude-3.5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "anthropic/claude-3.5-haiku": {"prompt": 0.80, "completion": 4.00},
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "openai/o1": {"prompt": 15.00, "completion": 60.00},
    "openai/o3-mini": {"prompt": 1.10, "completion": 4.40},
    "google/gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
    "google/gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
    "deepseek/deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "deepseek/deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
}

# Reference SiliconFlow Rates (USD per 1 Million Tokens)
SILICONFLOW_PRICING = {
    "deepseek/deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "deepseek/deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
    "qwen/qwen-2.5-72b-instruct": {"prompt": 0.165, "completion": 0.165},
    "qwen/qwen2.5-72b-instruct": {"prompt": 0.165, "completion": 0.165},
    "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.165, "completion": 0.165},
}


@dataclass
class PaidModelRecord:
    model_name: str
    model_id: str
    provider: str
    prompt_price_m: float
    completion_price_m: float
    blended_price: float
    performance_score: Optional[float]
    value_index: Optional[float]
    perf_source: str
    context_length: Optional[int] = None
    output_modalities: Optional[List[str]] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_slug_id(model_id: str) -> str:
    s = model_id.lower().strip()
    s = s.replace(":free", "").replace("-instruct", "").replace("_instruct", "")
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


def fetch_openrouter_models(api_key: Optional[str] = None) -> List[dict]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(MODELS_API, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_deepinfra_models() -> List[dict]:
    try:
        resp = requests.get(DEEPINFRA_API, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching DeepInfra models: {e}")
        return []


def get_provider_breakdown(model_id: str) -> str:
    try:
        url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        model_data = data.get("data") or {}
        endpoints = []
        if isinstance(model_data, dict):
            endpoints = model_data.get("endpoints", [])
        elif isinstance(model_data, list):
            endpoints = model_data

        if not endpoints:
            return ""

        provider_rates = []
        for ep in endpoints:
            if not isinstance(ep, dict):
                continue
            p_name = ep.get("provider_name")
            pricing = ep.get("pricing") or {}
            p_prompt = safe_float(pricing.get("prompt")) or 0.0
            p_completion = safe_float(pricing.get("completion")) or 0.0
            p_blended = (p_prompt * 0.75 + p_completion * 0.25) * 1000000
            provider_rates.append((p_blended, p_name, p_prompt * 1000000, p_completion * 1000000))

        provider_rates.sort()

        top_providers = []
        for rate, name, prompt, completion in provider_rates[:3]:
            top_providers.append(f"{name} (${prompt:.2f}/${completion:.2f})")

        return " / ".join(top_providers)
    except Exception as e:
        print(f"Error fetching provider breakdown for {model_id}: {e}")
        return ""


def get_comparison_text(model_id: str, prompt_price_m: float, completion_price_m: float, normalized_di_map: dict) -> Tuple[str, bool]:
    or_blended = prompt_price_m * 0.75 + completion_price_m * 0.25
    comparisons = []
    has_cheaper = False
    norm_id = normalize_slug_id(model_id)

    # 1. Official Direct API comparison
    off_rate = None
    for pattern, rate in OFFICIAL_PRICING.items():
        if normalize_slug_id(pattern) in norm_id or norm_id in normalize_slug_id(pattern):
            off_rate = rate
            break
    if off_rate:
        off_blended = off_rate["prompt"] * 0.75 + off_rate["completion"] * 0.25
        diff_pct = (1 - (or_blended / off_blended)) * 100
        if diff_pct >= 1.0:
            comparisons.append(f"直連 ${off_blended:.2f} (省 {diff_pct:.0f}%)")
        else:
            comparisons.append(f"直連 ${off_blended:.2f}")

    # 2. DeepInfra comparison
    di_rate = None
    for di_norm_name, di_model in normalized_di_map.items():
        if di_norm_name in norm_id or norm_id in di_norm_name:
            di_rate = di_model
            break
    if di_rate:
        di_prompt = (di_rate['pricing'].get('cents_per_input_token') or 0.0) * 10000
        di_completion = (di_rate['pricing'].get('cents_per_output_token') or 0.0) * 10000
        di_blended = di_prompt * 0.75 + di_completion * 0.25
        if di_blended < or_blended - 0.01:
            comparisons.append(f"DeepInfra ${di_blended:.2f} ⚠️")
            has_cheaper = True
        else:
            comparisons.append(f"DeepInfra ${di_blended:.2f}")

    # 3. SiliconFlow comparison
    sf_rate = None
    for pattern, rate in SILICONFLOW_PRICING.items():
        if normalize_slug_id(pattern) in norm_id or norm_id in normalize_slug_id(pattern):
            sf_rate = rate
            break
    if sf_rate:
        sf_blended = sf_rate["prompt"] * 0.75 + sf_rate["completion"] * 0.25
        if sf_blended < or_blended - 0.01:
            comparisons.append(f"SiliconFlow ${sf_blended:.2f} ⚠️")
            has_cheaper = True
        else:
            comparisons.append(f"SiliconFlow ${sf_blended:.2f}")

    if not comparisons:
        return "", False

    return "比照外網: " + " | ".join(comparisons), has_cheaper


def build_current_snapshot(raw_models: List[dict]) -> Dict[str, PaidModelRecord]:
    current: Dict[str, PaidModelRecord] = {}

    for item in raw_models:
        model_id = item.get("id")
        if not model_id:
            continue
        pricing = item.get("pricing") or {}
        prompt_price_token = safe_float(pricing.get("prompt")) or 0.0
        completion_price_token = safe_float(pricing.get("completion")) or 0.0

        # Skip free models or invalid pricing models
        if prompt_price_token <= 0 and completion_price_token <= 0:
            continue

        prompt_price_m = prompt_price_token * 1000000
        completion_price_m = completion_price_token * 1000000
        blended_price = prompt_price_m * 0.75 + completion_price_m * 0.25

        # Extract benchmarks
        benchmarks = item.get("benchmarks") or {}
        performance_score = None
        perf_source = "N/A"

        if "artificial_analysis" in benchmarks:
            aa = benchmarks["artificial_analysis"] or {}
            idx = safe_float(aa.get("intelligence_index"))
            if idx is not None:
                performance_score = idx
                perf_source = "AA"

        if performance_score is None and "design_arena" in benchmarks:
            arena = benchmarks["design_arena"] or []
            elos = [safe_float(a.get("elo")) for a in arena if safe_float(a.get("elo")) is not None]
            if elos:
                avg_elo = sum(elos) / len(elos)
                performance_score = max(0.0, min(100.0, (avg_elo - 1000.0) / 5.0))
                perf_source = "Arena"

        value_index = None
        if performance_score is not None and blended_price > 0:
            value_index = (performance_score / blended_price) * 100

        provider = model_id.split("/", 1)[0] if "/" in model_id else "unknown"
        model_name = item.get("name") or model_id
        top_provider = item.get("top_provider") or {}
        context_length = top_provider.get("context_length") or item.get("context_length")
        output_modalities = item.get("output_modalities") or top_provider.get("output_modalities") or []

        current[model_id] = PaidModelRecord(
            model_name=model_name,
            model_id=model_id,
            provider=provider,
            prompt_price_m=prompt_price_m,
            completion_price_m=completion_price_m,
            blended_price=blended_price,
            performance_score=performance_score,
            value_index=value_index,
            perf_source=perf_source,
            context_length=context_length,
            output_modalities=output_modalities,
        )

    return current


def load_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"models": {}, "last_run": None}
    return json.loads(p.read_text(encoding="utf-8"))


def split_field_value(lines: List[str], max_len: int = 1000) -> List[str]:
    chunks = []
    current_chunk = []
    current_len = 0
    for line in lines:
        if current_len + len(line) + 1 > max_len:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line) + 1
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def format_model_line(rank: int, r: PaidModelRecord, di_map: dict) -> str:
    vision_emoji = " 👁️" if r.output_modalities and any(m in ["image", "multimodal"] for m in r.output_modalities) else ""
    context_str = ""
    if r.context_length:
        ctx = r.context_length
        context_str = f" `{ctx // 1000}k`🔥" if ctx >= 128000 else f" `{ctx // 1000}k`"

    price_str = f"${r.prompt_price_m:.2f}/${r.completion_price_m:.2f}" if r.prompt_price_m != r.completion_price_m else f"${r.prompt_price_m:.2f}"
    
    line = f"`#{rank}` **[{r.model_name}](https://openrouter.ai/{r.model_id})** | {price_str} (M tokens){context_str}{vision_emoji}\n"
    line += f"> 📊 性能: `{r.performance_score:.1f}` ({r.perf_source}) | 性价比: `{r.value_index:.1f}`"
    
    comp_text, has_cheaper = get_comparison_text(r.model_id, r.prompt_price_m / 1000000, r.completion_price_m / 1000000, di_map)
    if comp_text:
        line += f"\n> 🏷️ {comp_text}"
        
    return line


def build_discord_markdown_embed(
    ranked_models: List[PaidModelRecord], 
    price_drops: List[dict], 
    di_map: dict
) -> dict:
    fields = []

    # Add Price Drops field
    if price_drops:
        drop_lines = []
        for d in price_drops:
            drop_lines.append(f"> • **[{d['model_name']}](https://openrouter.ai/{d['model_id']})**: Blended Price `${d['prev_price']:.2f}` ➡️ `${d['cur_price']:.2f}` (降價 {d['drop_pct']:.0f}%) 📉")
        
        drop_chunks = split_field_value(drop_lines, 1000)
        for idx, chunk in enumerate(drop_chunks):
            name = "📢 每日降價/限時促銷" if idx == 0 else "📢 每日降價/限時促銷 (續)"
            fields.append({"name": name, "value": chunk, "inline": False})
    else:
        fields.append({"name": "📢 每日降價/限時促銷", "value": "*今日無付費模型降價異動*", "inline": False})

    # Filter ranked models into T1, T2, T3
    # T1: Rank 1-5, requires perf >= 70
    t1_models = [m for m in ranked_models if m.performance_score >= 70][:5]
    
    # T2: Rank 6-15, requires perf >= 55, excluding T1 models
    t1_ids = {m.model_id for m in t1_models}
    t2_models = [m for m in ranked_models if m.model_id not in t1_ids and m.performance_score >= 55][:10]
    
    # T3: Rank 16-20, requires perf >= 40, excluding T1 & T2 models
    t2_ids = {m.model_id for m in t2_models}
    t3_models = [m for m in ranked_models if m.model_id not in t1_ids and m.model_id not in t2_ids and m.performance_score >= 40][:5]

    # Add T1 field
    if t1_models:
        t1_lines = []
        for idx, r in enumerate(t1_models, 1):
            t1_lines.append(format_model_line(idx, r, di_map))
            # Fetch provider breakdown for top 3
            if idx <= 3:
                breakdown = get_provider_breakdown(r.model_id)
                if breakdown:
                    t1_lines.append(f"> 🔌 託管商: {breakdown}")
        
        t1_chunks = split_field_value(t1_lines, 1000)
        for idx, chunk in enumerate(t1_chunks):
            name = "🏆 T1 頂尖高性價比榜 (性能 ≥ 70)" if idx == 0 else "🏆 T1 頂尖高性價比榜 (續)"
            fields.append({"name": name, "value": chunk, "inline": False})

    # Add T2 field
    if t2_models:
        t2_lines = []
        for idx, r in enumerate(t2_models, 6):
            t2_lines.append(format_model_line(idx, r, di_map))
            if idx <= 8:
                breakdown = get_provider_breakdown(r.model_id)
                if breakdown:
                    t2_lines.append(f"> 🔌 託管商: {breakdown}")
        
        t2_chunks = split_field_value(t2_lines, 1000)
        for idx, chunk in enumerate(t2_chunks):
            name = "🥈 T2 優質中階性價比榜 (性能 ≥ 55)" if idx == 0 else "🥈 T2 優質中階性價比榜 (續)"
            fields.append({"name": name, "value": chunk, "inline": False})

    # Add T3 field
    if t3_models:
        t3_lines = []
        for idx, r in enumerate(t3_models, 16):
            t3_lines.append(format_model_line(idx, r, di_map))
            
        t3_chunks = split_field_value(t3_lines, 1000)
        for idx, chunk in enumerate(t3_chunks):
            name = "🥉 T3 特色性價比榜 (性能 ≥ 40)" if idx == 0 else "🥉 T3 特色性價比榜 (續)"
            fields.append({"name": name, "value": chunk, "inline": False})

    total_paid_models = len(ranked_models)
    description = f"資料更新時間：`{utc_now()}`\n付費模型總數：**{total_paid_models}** 個 (面板僅展示前 20 個高 CP 值模型)\n*性價比公式 = (性能得分 / Blended Price 1M tokens) * 100*"
    
    embed = {
        "title": "📊 OpenRouter 付費模型性價比天梯日報",
        "description": description,
        "color": 16750848, # 0xff9900 Orange
        "fields": fields,
        "footer": {
            "text": "資料來源: OpenRouter API, DeepInfra API & Artificial Analysis"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return embed


def send_to_discord(webhook_url: str, summary: str, embed: dict) -> None:
    payload = {
        "content": summary[:DISCORD_LIMIT],
        "embeds": [embed]
    }
    resp = requests.post(webhook_url, json=payload, timeout=60)
    resp.raise_for_status()


def save_change_history(price_drops: List[dict], path: str) -> None:
    p = Path(path)
    history = []
    if p.exists():
        try:
            history = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    timestamp = utc_now()
    changes = []
    for d in price_drops:
        changes.append({
            "timestamp": timestamp,
            "model_id": d["model_id"],
            "model_name": d["model_name"],
            "status": "Price Drop",
            "note": f"Blended Price dropped from ${d['prev_price']:.2f} to ${d['cur_price']:.2f} (省 {d['drop_pct']:.0f}%)"
        })
            
    if changes:
        history.extend(changes)
        history = history[-1000:]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # Load environment variables from .env if it exists
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    state_file = os.getenv("STATE_FILE", DEFAULT_STATE_FILE)
    history_file = os.getenv("HISTORY_FILE", DEFAULT_HISTORY_FILE)

    previous_state = load_state(state_file)
    prev_models = previous_state.get("models", {}) or {}

    raw_models = fetch_openrouter_models(api_key=api_key)
    di_raw = fetch_deepinfra_models()

    # Pre-map DeepInfra models by normalized slug
    normalized_di_map = {}
    for item in di_raw:
        name = item.get("model_name")
        if name:
            normalized_di_map[normalize_slug_id(name)] = item

    current = build_current_snapshot(raw_models)

    # Detect price drops
    price_drops = []
    for model_id, record in current.items():
        prev = prev_models.get(model_id)
        if prev:
            prev_blended = prev.get("blended_price")
            cur_blended = record.blended_price
            if prev_blended and cur_blended < prev_blended - 0.01:
                drop_pct = (1 - (cur_blended / prev_blended)) * 100
                price_drops.append({
                    "model_name": record.model_name,
                    "model_id": model_id,
                    "prev_price": prev_blended,
                    "cur_price": cur_blended,
                    "drop_pct": drop_pct
                })

    save_change_history(price_drops, history_file)

    # Save next state
    next_state = {
        "last_run": utc_now(),
        "models": {mid: asdict(rec) for mid, rec in current.items()}
    }
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    Path(state_file).write_text(json.dumps(next_state, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sort paid models for rankings (models with a valid value index)
    ranked_models = [m for m in current.values() if m.value_index is not None]
    ranked_models.sort(key=lambda x: x.value_index, reverse=True)

    # Build summary text
    summary = f"OpenRouter 付費模型性價比監控 | 本次付費模型: {len(ranked_models)} 個"
    if price_drops:
        summary += f" | 偵測到 {len(price_drops)} 個模型降價！"
        
    alert_mention = os.getenv("ALERT_MENTION", "").strip()
    if price_drops and alert_mention:
        summary = f"{alert_mention} {summary}"

    embed = build_discord_markdown_embed(ranked_models, price_drops, normalized_di_map)

    if webhook_url:
        send_to_discord(webhook_url, summary, embed)
        print("Posted to Discord webhook successfully (paid models value report).")
    else:
        print("DISCORD_WEBHOOK_URL not set; report generated locally only.")

    print(f"History file: {history_file}")
    print(f"State file: {state_file}")
    print(f"Paid models with value index: {len(ranked_models)}")


if __name__ == "__main__":
    main()
