#!/usr/bin/env python3
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

MODELS_API = "https://openrouter.ai/api/v1/models"
FREE_COLLECTION_URL = "https://openrouter.ai/collections/free-models"
DEFAULT_STATE_FILE = "output/openrouter_free_state.json"
DEFAULT_HISTORY_FILE = "output/openrouter_free_history.json"
DISCORD_LIMIT = 1900


@dataclass
class ModelRecord:
    model_name: str
    model_id: str
    provider: str
    is_free: bool
    in_free_ranking: bool
    rank_position: Optional[int]
    context_length: Optional[int] = None
    output_modalities: Optional[List[str]] = None
    performance_score: Optional[float] = None
    perf_source: str = "N/A"
    latency_ms: Optional[float] = None
    throughput_tps: Optional[float] = None
    parameter_size: str = "N/A"
    uptime_history: List[float] = field(default_factory=list)
    uptime_1d: Optional[float] = None
    uptime_30m: Optional[float] = None
    vision: bool = False
    tool_calling: bool = False
    reasoning: bool = False


@dataclass
class StatusRow:
    model_name: str
    model_id: str
    provider: str
    current_status: str
    rank_position: Optional[int]
    previous_rank: Optional[int]
    in_free_ranking: bool
    note: str
    rank_change: Optional[int] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_models(api_key: Optional[str] = None) -> List[dict]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(MODELS_API, headers=headers, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def extract_ranked_ids(html: str) -> List[str]:
    matches = re.findall(r'href="/([^"?#]+/[^"?#]+(?::free)?)"', html)
    blacklist = {"docs", "blog", "rankings", "collections", "models", "provider", "compare", "apps"}
    ranked = []
    seen = set()
    for path in matches:
        first = path.split("/", 1)[0]
        if first in blacklist:
            continue
        if path not in seen:
            seen.add(path)
            ranked.append(path)
    return ranked


def fetch_free_ranking() -> List[str]:
    try:
        resp = requests.get(FREE_COLLECTION_URL, timeout=45)
        resp.raise_for_status()
        return extract_ranked_ids(resp.text)
    except requests.RequestException:
        return []


def extract_parameter_size(name: str, description: str) -> str:
    text = f"{description or ''} {name or ''}"
    match = re.search(r'\b\d+(?:\.\d+)?\s*[xX]\s*\d+[bB]\b|\b\d+(?:\.\d+)?\s*[bB]\b', text)
    if match:
        return re.sub(r'\s+', '', match.group(0)).upper()
    return "N/A"


def get_uptime_trend_emoji(history: List[float]) -> str:
    if not history:
        return "⬜⬜⬜⬜⬜⬜⬜"
    padded = [None] * (7 - len(history)) + history
    emojis = []
    for val in padded:
        if val is None:
            emojis.append("⬜")
        elif val >= 98.0:
            emojis.append("🟩")
        elif val >= 90.0:
            emojis.append("🟨")
        else:
            emojis.append("🟥")
    return "".join(emojis)


def build_current_snapshot(raw_models: List[dict], ranked_ids: List[str]) -> Dict[str, ModelRecord]:
    ranked_map = {mid: idx + 1 for idx, mid in enumerate(ranked_ids)}
    current: Dict[str, ModelRecord] = {}

    for item in raw_models:
        model_id = item.get("id")
        if not model_id:
            continue
        pricing = item.get("pricing") or {}
        prompt_price = safe_float(pricing.get("prompt"))
        completion_price = safe_float(pricing.get("completion"))
        zero_priced = prompt_price == 0 and completion_price == 0
        in_ranking = model_id in ranked_map
        is_free = zero_priced or in_ranking or model_id.endswith(":free")
        if not is_free:
            continue
        provider = model_id.split("/", 1)[0] if "/" in model_id else "unknown"
        model_name = item.get("name") or model_id
        top_provider = item.get("top_provider") or {}
        context_length = top_provider.get("context_length") or item.get("context_length")
        output_modalities = item.get("output_modalities") or top_provider.get("output_modalities") or []
        
        # Parse performance benchmarks
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
                
        # Capability parsing
        architecture = item.get("architecture") or {}
        input_mods = architecture.get("input_modalities") or []
        
        vision = ("image" in input_mods or "multimodal" in input_mods or 
                  "image" in output_modalities or "multimodal" in output_modalities)
                  
        supported_params = item.get("supported_parameters") or []
        tool_calling = "tools" in supported_params or "response_format" in supported_params or "structured_outputs" in supported_params
        reasoning = "reasoning" in supported_params or "include_reasoning" in supported_params
        
        parameter_size = extract_parameter_size(model_name, item.get("description", ""))
                
        current[model_id] = ModelRecord(
            model_name=model_name,
            model_id=model_id,
            provider=provider,
            is_free=True,
            in_free_ranking=in_ranking,
            rank_position=ranked_map.get(model_id),
            context_length=context_length,
            output_modalities=output_modalities,
            performance_score=performance_score,
            perf_source=perf_source,
            parameter_size=parameter_size,
            vision=vision,
            tool_calling=tool_calling,
            reasoning=reasoning,
        )
    return current


def load_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"models": {}, "missing_streaks": {}, "last_run": None}
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_slug(text: str) -> str:
    text = text.lower().replace(":free", "")
    text = re.sub(r"\(free\)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def likely_renamed(old_id: str, new_id: str) -> bool:
    old_provider = old_id.split("/", 1)[0] if "/" in old_id else ""
    new_provider = new_id.split("/", 1)[0] if "/" in new_id else ""
    if old_provider != new_provider:
        return False
    old_slug = normalize_slug(old_id.split("/", 1)[-1])
    new_slug = normalize_slug(new_id.split("/", 1)[-1])
    ratio = SequenceMatcher(None, old_slug, new_slug).ratio()
    old_tokens = set(old_slug.split())
    new_tokens = set(new_slug.split())
    overlap = len(old_tokens & new_tokens)
    return ratio >= 0.88 or (ratio >= 0.78 and overlap >= 2)


def check_is_upgrade(old_id: str, new_id: str) -> bool:
    # Heuristic to detect if new model is a version upgrade of old model
    old_slug = old_id.split("/", 1)[-1]
    new_slug = new_id.split("/", 1)[-1]
    
    # Extract digit and dot patterns (e.g. 3.1, 8, 3.5)
    old_nums = []
    for x in re.findall(r'\d+(?:\.\d+)?', old_slug):
        try:
            old_nums.append(float(x))
        except ValueError:
            pass
            
    new_nums = []
    for x in re.findall(r'\d+(?:\.\d+)?', new_slug):
        try:
            new_nums.append(float(x))
        except ValueError:
            pass
    
    # Compare each extracted number sequence
    for o_val, n_val in zip(old_nums, new_nums):
        if n_val > o_val:
            return True
        elif n_val < o_val:
            return False
            
    # If same prefix but new one has more numbers, e.g. llama-3 -> llama-3.1
    if len(new_nums) > len(old_nums):
        return True
    return False


def detect_renames(missing_ids: List[str], new_ids: List[str]) -> Dict[str, str]:
    mapping = {}
    used_new = set()
    for old_id in missing_ids:
        candidates: List[Tuple[float, str]] = []
        for new_id in new_ids:
            if new_id in used_new:
                continue
            if likely_renamed(old_id, new_id):
                old_slug = normalize_slug(old_id.split("/", 1)[-1])
                new_slug = normalize_slug(new_id.split("/", 1)[-1])
                score = SequenceMatcher(None, old_slug, new_slug).ratio()
                candidates.append((score, new_id))
        if candidates:
            candidates.sort(reverse=True)
            chosen = candidates[0][1]
            mapping[old_id] = chosen
            used_new.add(chosen)
    return mapping


def compare_states(current: Dict[str, ModelRecord], previous_state: dict) -> Tuple[List[StatusRow], Dict[str, int]]:
    prev_models = previous_state.get("models", {}) or {}
    prev_missing = previous_state.get("missing_streaks", {}) or {}

    current_ids = set(current.keys())
    prev_ids = set(prev_models.keys())
    new_ids = sorted(current_ids - prev_ids)
    common_ids = sorted(current_ids & prev_ids)
    missing_ids = sorted(prev_ids - current_ids)

    rename_map = detect_renames(missing_ids, new_ids)
    reverse_rename_map = {v: k for k, v in rename_map.items()}

    rows: List[StatusRow] = []
    next_missing_streaks: Dict[str, int] = {}

    for model_id in common_ids:
        cur = current[model_id]
        prev = prev_models[model_id]
        prev_rank = prev.get("rank_position")
        note_parts = []
        status = "Active"
        rank_change = None
        if prev_rank is not None and cur.rank_position is not None:
            rank_change = prev_rank - cur.rank_position
            if rank_change > 0:
                note_parts.append(f"排名上升 {rank_change}")
            elif rank_change < 0:
                note_parts.append(f"排名下降 {abs(rank_change)}")
        if prev.get("in_free_ranking") and not cur.in_free_ranking:
            status = "Moved"
            note_parts.append("仍免費，但已不在主 free 榜")
        elif not prev.get("in_free_ranking") and cur.in_free_ranking:
            note_parts.append("重新進入主 free 榜")
        rows.append(StatusRow(
            model_name=cur.model_name,
            model_id=cur.model_id,
            provider=cur.provider,
            current_status=status,
            rank_position=cur.rank_position,
            previous_rank=prev_rank,
            in_free_ranking=cur.in_free_ranking,
            note="；".join(note_parts) or "持續免費",
            rank_change=rank_change,
        ))

    for model_id in sorted(new_ids):
        cur = current[model_id]
        if model_id in reverse_rename_map:
            old_id = reverse_rename_map[model_id]
            is_upgrade = check_is_upgrade(old_id, model_id)
            status = "Upgraded" if is_upgrade else "Renamed"
            note = f"由 {old_id} 升級" if is_upgrade else f"疑似由 {old_id} 更名"
        else:
            status = "New"
            note = "首次出現在免費清單"
        rows.append(StatusRow(
            model_name=cur.model_name,
            model_id=cur.model_id,
            provider=cur.provider,
            current_status=status,
            rank_position=cur.rank_position,
            previous_rank=None,
            in_free_ranking=cur.in_free_ranking,
            note=note,
        ))

    for model_id in sorted(missing_ids):
        prev = prev_models[model_id]
        missing_count = int(prev_missing.get(model_id, 0)) + 1
        next_missing_streaks[model_id] = missing_count
        if model_id in rename_map:
            new_id = rename_map[model_id]
            is_upgrade = check_is_upgrade(model_id, new_id)
            status = "Upgraded" if is_upgrade else "Renamed"
            note = f"升級為 {new_id}" if is_upgrade else f"疑似變為 {new_id}"
        elif missing_count >= 2:
            status = "Removed"
            note = "連續 2 次未出現在免費清單，按規則標記 Removed"
        else:
            status = "Pending Removal"
            note = "今日未出現；依規則需連續 2 次缺失才標記 Removed"
        rows.append(StatusRow(
            model_name=prev.get("model_name", model_id),
            model_id=model_id,
            provider=prev.get("provider", "unknown"),
            current_status=status,
            rank_position=None,
            previous_rank=prev.get("rank_position"),
            in_free_ranking=False,
            note=note,
        ))

    for model_id in current_ids:
        next_missing_streaks.pop(model_id, None)

    status_order = {
        "New": 0,
        "Upgraded": 1,
        "Renamed": 2,
        "Active": 3,
        "Moved": 4,
        "Pending Removal": 5,
        "Removed": 6,
    }
    rows.sort(key=lambda r: (
        status_order.get(r.current_status, 99),
        10**9 if r.rank_position is None else r.rank_position,
        r.provider,
        r.model_name.lower(),
    ))
    return rows, next_missing_streaks


def build_next_state(current: Dict[str, ModelRecord], missing_streaks: Dict[str, int]) -> dict:
    return {
        "last_run": utc_now(),
        "models": {mid: asdict(record) for mid, record in current.items()},
        "missing_streaks": missing_streaks,
    }


def save_change_history(rows: List[StatusRow], path: str) -> None:
    p = Path(path)
    history = []
    if p.exists():
        try:
            history = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
            
    # Filter rows for changes
    changes = []
    timestamp = utc_now()
    for r in rows:
        if r.current_status in ["New", "Removed", "Pending Removal", "Moved", "Renamed", "Upgraded"]:
            changes.append({
                "timestamp": timestamp,
                "model_id": r.model_id,
                "model_name": r.model_name,
                "provider": r.provider,
                "status": r.current_status,
                "note": r.note
            })
            
    if changes:
        history.extend(changes)
        # Keep only the last 1000 history entries
        history = history[-1000:]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")





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


def fetch_endpoint_metrics(model_id: str, api_key: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    try:
        url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        endpoints = data.get("data", {}).get("endpoints", [])
        if not endpoints:
            return None, None, None, None

        active_eps = [ep for ep in endpoints if ep.get("status") == 0]
        if not active_eps:
            active_eps = endpoints

        rates = []
        for ep in active_eps:
            if not isinstance(ep, dict):
                continue
            pricing = ep.get("pricing") or {}
            p_prompt = safe_float(pricing.get("prompt")) or 0.0
            p_completion = safe_float(pricing.get("completion")) or 0.0
            p_blended = p_prompt * 0.75 + p_completion * 0.25
            rates.append((p_blended, ep))

        rates.sort(key=lambda x: x[0])
        cheapest_ep = rates[0][1] if rates else endpoints[0]

        lat_data = cheapest_ep.get("latency_last_30m")
        tp_data = cheapest_ep.get("throughput_last_30m")

        latency_ms = None
        throughput_tps = None

        if isinstance(lat_data, dict):
            latency_ms = safe_float(lat_data.get("p50"))
        elif isinstance(lat_data, (int, float)):
            latency_ms = float(lat_data)

        if isinstance(tp_data, dict):
            throughput_tps = safe_float(tp_data.get("p50"))
        elif isinstance(tp_data, (int, float)):
            throughput_tps = float(tp_data)

        uptime_1d = safe_float(cheapest_ep.get("uptime_last_1d"))
        uptime_30m = safe_float(cheapest_ep.get("uptime_last_30m"))

        return throughput_tps, latency_ms, uptime_1d, uptime_30m
    except Exception as e:
        print(f"Error fetching metrics for {model_id}: {e}")
        return None, None, None, None


def format_model_line(rank: int, r: StatusRow, current: Dict[str, ModelRecord]) -> str:
    meta = current.get(r.model_id)
    badge = ""
    vision_emoji = ""
    context_str = ""
    perf_str = "📊 性能評分: `N/A`"
    latency_str = ""
    param_size = "N/A"
    trend_str = "⬜⬜⬜⬜⬜⬜⬜"
    
    if meta:
        if meta.output_modalities and any(m in ["image", "multimodal"] for m in meta.output_modalities):
            vision_emoji = " 👁️"
        if meta.context_length:
            context_str = f" `{meta.context_length // 1000}k` context"
        if meta.performance_score is not None:
            perf_str = f"📊 性能評分: `{meta.performance_score:.1f}` ({meta.perf_source})"
        if meta.parameter_size:
            param_size = meta.parameter_size
        if meta.uptime_history:
            trend_str = get_uptime_trend_emoji(meta.uptime_history)
        
        # 延遲速度指標
        ttft = (meta.latency_ms / 1000.0 * 0.33) if meta.latency_ms is not None else None
        latency = (meta.latency_ms / 1000.0) if meta.latency_ms is not None else None
        if ttft and latency:
            latency_str = f" | ⚡ TTFT: `{ttft:.2f}s` | ⏱️ 延遲: `{latency:.2f}s`"
            if meta.throughput_tps is not None:
                latency_str += f" | 📊 吞吐: `{meta.throughput_tps:.1f} t/s`"
            
    if r.current_status == "New":
        badge = " 🟢 NEW"
    elif r.current_status == "Upgraded":
        badge = " 🔄 UPGRADE"
        
    model_part = f"**[{r.model_name}](https://openrouter.ai/{r.model_id})**"
    
    # 確保模型 ID 被包裹在代碼塊中，以便一鍵複製
    id_part = f"`{r.model_id}`"
    
    line = f"#{rank} {model_part}\n> ID: {id_part}{badge}\n> {perf_str}{latency_str} | {context_str}{vision_emoji}\n> 📦 規模: `{param_size}` | 📅 7天趨勢: {trend_str}"
    return line


def build_discord_markdown_embed(rows: List[StatusRow], current: Dict[str, ModelRecord], pool_health: str, high_cap_health: str) -> dict:
    new_models = [r for r in rows if r.current_status == "New"]
    upgraded_models = [r for r in rows if r.current_status == "Upgraded"]
    renamed_models = [r for r in rows if r.current_status == "Renamed"]
    removed_models = [r for r in rows if r.current_status == "Removed"]
    pending_models = [r for r in rows if r.current_status == "Pending Removal"]
    moved_models = [r for r in rows if r.current_status == "Moved"]
    
    changes_lines = []
    if new_models:
        changes_lines.append("**🟢 新增免費模型**:")
        for r in new_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** | `{r.model_id}`")
    if upgraded_models:
        changes_lines.append("**🔄 版本升級模型**:")
        for r in upgraded_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** | `{r.model_id}` ({r.note})")
    if renamed_models:
        changes_lines.append("**📝 更名變更模型**:")
        for r in renamed_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** | `{r.model_id}` ({r.note})")
    if removed_models or pending_models:
        changes_lines.append("**🔴 下架/疑似下架**:")
        for r in removed_models + pending_models:
            badge_text = "下架" if r.current_status == "Removed" else "疑似下架"
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** | `{r.model_id}` - {badge_text}")
    if moved_models:
        changes_lines.append("**🟣 移出主榜 (仍免費)**:")
        for r in moved_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** | `{r.model_id}`")
            
    rank_changes = [r for r in rows if r.current_status == "Active" and r.rank_change is not None and r.rank_change != 0]
    if rank_changes:
        changes_lines.append("**🔵 主榜排名變動**:")
        for r in rank_changes:
            arrow = "▲" if r.rank_change > 0 else "▼"
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** | `{r.model_id}`: 排名 `#{r.previous_rank}` ➡️ `#{r.rank_position}` ({arrow} {abs(r.rank_change)})")
            
    # 採用官方的性能排名排序前 10
    active = [r for r in rows if r.current_status in ["Active", "New", "Upgraded", "Renamed", "Moved"]]
    
    def get_official_rank(r):
        if r.rank_position is not None:
            return r.rank_position
        return 999999
        
    active.sort(key=get_official_rank)
    top_10 = active[:10]
    
    fields = []
    
    if changes_lines:
        change_chunks = split_field_value(changes_lines, 1000)
        for idx, chunk in enumerate(change_chunks):
            name = "📢 每日異動摘要" if idx == 0 else "📢 每日異動摘要 (續)"
            fields.append({"name": name, "value": chunk, "inline": False})
    else:
        fields.append({"name": "📢 每日異動摘要", "value": "*今日無模型狀態異動*", "inline": False})
        
    top_lines = []
    for rank, r in enumerate(top_10, 1):
        top_lines.append(format_model_line(rank, r, current))
        
    top_chunks = split_field_value(top_lines, 1000)
    for idx, chunk in enumerate(top_chunks):
        name = "🏆 性能排名前 10 免費模型" if idx == 0 else "🏆 性能排名前 10 免費模型 (續)"
        fields.append({"name": name, "value": chunk, "inline": False})
        
    total_active_count = len(active)
    description = (
        f"資料更新時間：`{utc_now()}`\n"
        f"當前免費模型總數：**{total_active_count}** 個 (面板僅展示性能排名前 10)\n"
        f"大盤健康度：🟢 **{pool_health}** | 高能力模型健康度：⚡ **{high_cap_health}**"
    )
    
    embed = {
        "title": "🚀 OpenRouter 免費模型性能天梯日報",
        "description": description,
        "color": 6514417, # 0x6366f1 Indigo
        "fields": fields,
        "footer": {
            "text": "資料來源: OpenRouter API & Free Collection"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return embed


def send_to_discord(webhook_url: str, summary: str, embed: dict) -> None:
    max_total_chars = 5500
    
    if embed.get("title"):
        embed["title"] = embed["title"][:256]
    if embed.get("description"):
        embed["description"] = embed["description"][:4000]
    if embed.get("footer") and embed["footer"].get("text"):
        embed["footer"]["text"] = embed["footer"]["text"][:2000]
        
    fields = embed.get("fields", [])
    safe_fields = []
    total_chars = len(embed.get("title") or "") + len(embed.get("description") or "") + len(embed.get("footer", {}).get("text") or "")
    
    for f in fields[:25]:  # Discord 最大 25 個 fields
        name = f.get("name", "")[:256]
        value = f.get("value", "")[:1024]
        field_chars = len(name) + len(value)
        if total_chars + field_chars > max_total_chars:
            break
        safe_fields.append({"name": name, "value": value, "inline": f.get("inline", False)})
        total_chars += field_chars
        
    embed["fields"] = safe_fields
    
    payload = {
        "content": summary[:2000],
        "embeds": [embed]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=30)
            if resp.status_code in (200, 204):
                return
            elif resp.status_code == 429:
                retry_after = 2.0
                try:
                    js = resp.json()
                    retry_after = js.get("retry_after", 2.0)
                except Exception:
                    retry_after_hdr = resp.headers.get("Retry-After")
                    if retry_after_hdr:
                        try:
                            retry_after = float(retry_after_hdr)
                        except ValueError:
                            pass
                print(f"Discord Rate Limit (429) hit. Waiting {retry_after}s before retry...")
                time.sleep(retry_after)
            else:
                print(f"Discord Webhook returned status {resp.status_code}: {resp.text}")
                resp.raise_for_status()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to send to Discord webhook after {max_retries} attempts: {e}")
                break
            print(f"Network error on attempt {attempt + 1}: {e}. Retrying in 2 seconds...")
            time.sleep(2)


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
    raw_models = fetch_models(api_key=api_key)
    ranked_ids = fetch_free_ranking()
    current = build_current_snapshot(raw_models, ranked_ids)
    rows, next_missing_streaks = compare_states(current, previous_state)

    # Fetch endpoint metrics and update uptime history for all active free models
    print(f"Fetching endpoints metrics and tracking history for {len(current)} models...")
    for model_id, record in current.items():
        tps, lat, uptime_1d, uptime_30m = fetch_endpoint_metrics(model_id, api_key)
        record.latency_ms = lat
        record.throughput_tps = tps
        record.uptime_1d = uptime_1d
        record.uptime_30m = uptime_30m
        
        # Track 7-day uptime history
        prev_rec = previous_state.get("models", {}).get(model_id, {})
        history = prev_rec.get("uptime_history", [])
        cur_uptime = uptime_1d if uptime_1d is not None else (uptime_30m if uptime_30m is not None else 100.0)
        history = (history + [cur_uptime])[-7:]
        record.uptime_history = history

    # Calculate pool health and high-capability models pool health
    active_records = list(current.values())
    
    # 1. Overall pool health
    overall_uptimes = [
        (m.uptime_1d if m.uptime_1d is not None else (m.uptime_30m if m.uptime_30m is not None else 100.0))
        for m in active_records
    ]
    pool_health_val = sum(overall_uptimes) / len(overall_uptimes) if overall_uptimes else 100.0
    pool_health = f"{pool_health_val:.1f}%"
    
    # 2. High-capability models pool health
    high_cap_models = [
        m for m in active_records 
        if (m.performance_score is not None and m.performance_score >= 60.0) 
        or m.tool_calling or m.reasoning or m.vision
    ]
    high_cap_uptimes = [
        (m.uptime_1d if m.uptime_1d is not None else (m.uptime_30m if m.uptime_30m is not None else 100.0))
        for m in high_cap_models
    ]
    high_cap_health_val = sum(high_cap_uptimes) / len(high_cap_uptimes) if high_cap_uptimes else 100.0
    high_cap_health = f"{high_cap_health_val:.1f}%"

    save_change_history(rows, history_file)

    next_state = build_next_state(current, next_missing_streaks)
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    Path(state_file).write_text(json.dumps(next_state, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build summary text
    counts = {}
    for r in rows:
        counts[r.current_status] = counts.get(r.current_status, 0) + 1

    summary_parts = [f"OpenRouter 免費模型監控", f"本次免費模型: {len(current)}"]
    for key in ["New", "Upgraded", "Renamed", "Active", "Moved", "Pending Removal", "Removed"]:
        if counts.get(key):
            summary_parts.append(f"{key}: {counts[key]}")
    summary = " | ".join(summary_parts)

    # Add alert mention if changes occurred and ALERT_MENTION env is configured
    has_changes = any(counts.get(k, 0) > 0 for k in ["New", "Upgraded", "Renamed", "Moved", "Pending Removal", "Removed"])
    alert_mention = os.getenv("ALERT_MENTION", "").strip()
    if has_changes and alert_mention:
        summary = f"{alert_mention} {summary}"

    embed = build_discord_markdown_embed(rows, current, pool_health, high_cap_health)
    print("DEBUG: Generated Embed Description:", repr(embed.get("description")))

    if webhook_url:
        send_to_discord(webhook_url, summary, embed)
        print("Posted to Discord webhook successfully (markdown format).")
    else:
        print("DISCORD_WEBHOOK_URL not set; report generated locally only.")

    print(f"History file: {history_file}")
    print(f"State file: {state_file}")
    print(f"Models found: {len(current)}")


if __name__ == "__main__":
    main()
