#!/usr/bin/env python3
import json
import os
import re
from dataclasses import dataclass, asdict
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
        current[model_id] = ModelRecord(
            model_name=model_name,
            model_id=model_id,
            provider=provider,
            is_free=True,
            in_free_ranking=in_ranking,
            rank_position=ranked_map.get(model_id),
            context_length=context_length,
            output_modalities=output_modalities,
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


def format_model_line(r: StatusRow, current: Dict[str, ModelRecord]) -> str:
    meta = current.get(r.model_id)
    badge = ""
    
    vision_emoji = ""
    context_str = ""
    
    if meta:
        if meta.output_modalities and any(m in ["image", "multimodal"] for m in meta.output_modalities):
            vision_emoji = " 👁️"
            
        if meta.context_length:
            ctx = meta.context_length
            if ctx >= 128000:
                context_str = f" `{ctx // 1000}k`🔥"
            elif ctx >= 1000:
                context_str = f" `{ctx // 1000}k`"
            else:
                context_str = f" `{ctx}`"
                
    if r.current_status == "New":
        badge = " 🟢 NEW"
    elif r.current_status == "Upgraded":
        badge = " 🔄 UPGRADE"
    elif r.current_status == "Renamed":
        badge = " 📝 RENAME"
    elif r.current_status == "Moved":
        badge = " 🟣 MOVED"
    elif r.rank_change is not None and r.rank_change != 0:
        if r.rank_change > 0:
            badge = f" ▲{r.rank_change}"
        else:
            badge = f" ▼{abs(r.rank_change)}"
            
    rank_prefix = f"`#{r.rank_position}` " if r.rank_position else "• "
    model_part = f"**[{r.model_name}](https://openrouter.ai/{r.model_id})**"
    id_part = f"`{r.model_id}`"
    
    return f"{rank_prefix}{model_part} | {id_part}{context_str}{vision_emoji}{badge}"


def build_discord_markdown_embed(rows: List[StatusRow], current: Dict[str, ModelRecord]) -> dict:
    new_models = [r for r in rows if r.current_status == "New"]
    upgraded_models = [r for r in rows if r.current_status == "Upgraded"]
    renamed_models = [r for r in rows if r.current_status == "Renamed"]
    removed_models = [r for r in rows if r.current_status == "Removed"]
    pending_models = [r for r in rows if r.current_status == "Pending Removal"]
    moved_models = [r for r in rows if r.current_status == "Moved"]
    rank_changes = [r for r in rows if r.current_status == "Active" and r.rank_change is not None and r.rank_change != 0]
    
    changes_lines = []
    if new_models:
        changes_lines.append("**🟢 新增免費模型**:")
        for r in new_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** (`{r.model_id}`)")
    if upgraded_models:
        changes_lines.append("**🔄 版本升級模型**:")
        for r in upgraded_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** ({r.note})")
    if renamed_models:
        changes_lines.append("**📝 更名變更模型**:")
        for r in renamed_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** ({r.note})")
    if removed_models or pending_models:
        changes_lines.append("**🔴 下架/疑似下架**:")
        for r in removed_models + pending_models:
            badge_text = "下架" if r.current_status == "Removed" else "疑似下架"
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** (`{r.model_id}`) - {badge_text} ({r.note})")
    if moved_models:
        changes_lines.append("**🟣 移出主榜 (仍免費)**:")
        for r in moved_models:
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})** (`{r.model_id}`)")
    if rank_changes:
        changes_lines.append("**🔵 主榜排名變動**:")
        for r in rank_changes:
            arrow = "▲" if r.rank_change > 0 else "▼"
            changes_lines.append(f"> • **[{r.model_name}](https://openrouter.ai/{r.model_id})**: 排名 `#{r.previous_rank}` ➡️ `#{r.rank_position}` ({arrow} {abs(r.rank_change)})")
            
    # Select exactly top 20 active models
    active = [r for r in rows if r.current_status in ["Active", "New", "Upgraded", "Renamed", "Moved"]]
    ranked = [r for r in active if r.rank_position is not None]
    ranked.sort(key=lambda x: x.rank_position)
    unranked = [r for r in active if r.rank_position is None]
    unranked.sort(key=lambda x: x.model_name.lower())
    
    selected_ranked = ranked[:20]
    selected_unranked = unranked[:max(0, 20 - len(selected_ranked))]
    
    t1_models = [r for r in selected_ranked if r.rank_position <= 5]
    t2_models = [r for r in selected_ranked if 5 < r.rank_position <= 15]
    t3_models = [r for r in selected_ranked if r.rank_position > 15]
    t4_models = selected_unranked
    
    fields = []
    
    # Add changes field
    if changes_lines:
        change_chunks = split_field_value(changes_lines, 1000)
        for idx, chunk in enumerate(change_chunks):
            name = "📢 每日異動摘要" if idx == 0 else "📢 每日異動摘要 (續)"
            fields.append({"name": name, "value": chunk, "inline": False})
    else:
        fields.append({"name": "📢 每日異動摘要", "value": "*今日無模型狀態或排名異動*", "inline": False})
        
    # Add T1 field
    if t1_models:
        t1_lines = [format_model_line(r, current) for r in t1_models]
        t1_chunks = split_field_value(t1_lines, 1000)
        for idx, chunk in enumerate(t1_chunks):
            name = "🏆 T1 旗艦性能 (Rank 1 - 5)" if idx == 0 else "🏆 T1 旗艦性能 (Rank 1 - 5) (續)"
            fields.append({"name": name, "value": chunk, "inline": False})
        
    # Add T2 field
    if t2_models:
        t2_lines = [format_model_line(r, current) for r in t2_models]
        t2_chunks = split_field_value(t2_lines, 1000)
        for idx, chunk in enumerate(t2_chunks):
            name = "🥈 T2 主流推薦 (Rank 6 - 15)" if idx == 0 else "🥈 T2 主流推薦 (Rank 6 - 15) (續)"
            fields.append({"name": name, "value": chunk, "inline": False})
        
    # Add T3 field
    if t3_models:
        t3_lines = [format_model_line(r, current) for r in t3_models]
        t3_chunks = split_field_value(t3_lines, 1000)
        for idx, chunk in enumerate(t3_chunks):
            name = "🥉 T3 輕量特色 (Rank 16 - 20)" if idx == 0 else "🥉 T3 輕量特色 (Rank 16 - 20) (續)"
            fields.append({"name": name, "value": chunk, "inline": False})
        
    # Add T4 field
    if t4_models:
        t4_lines = [format_model_line(r, current) for r in t4_models]
        t4_chunks = split_field_value(t4_lines, 1000)
        for idx, chunk in enumerate(t4_chunks):
            name = "📁 T4 補充免費模型 (無排名)" if idx == 0 else "📁 T4 補充免費模型 (續)"
            fields.append({"name": name, "value": chunk, "inline": False})
            
    total_active_count = len(active)
    description = f"資料更新時間：`{utc_now()}`\n當前免費模型總數：**{total_active_count}** 個 (面板僅展示前 20 個)"
    
    embed = {
        "title": "🚀 OpenRouter 免費模型監控日報",
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
    payload = {
        "content": summary[:DISCORD_LIMIT],
        "embeds": [embed]
    }
    resp = requests.post(webhook_url, json=payload, timeout=60)
    resp.raise_for_status()


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

    embed = build_discord_markdown_embed(rows, current)

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
