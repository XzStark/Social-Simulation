"""
地区情境 grounding：模拟前多轮「检索」摘要（维基可选 + LLM 合成），
使政策/商户反应建立在「当地已有政策 + 中央框架 + 基层常态」之上。
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import httpx

from people.config import IssuerArchetype, RegionalGroundingSpec, Scenario
from people.providers.base import LLMClient
from people.web_search_context import fetch_web_grounding_snippets

GROUNDING_SYSTEM = """你是政策与市场研究的「情境检索」助理，输出**合法 JSON 对象**（不要 markdown 围栏）。
硬性规则：
1) **禁止编造**具体法律法规、文号、生效日期；不确定须写「需核实」。
2) 明确这是**演练用合成摘要**，不是官方法律意见或备案检索结果。若输入含开放网页检索摘录，其可能为推广、旧闻或偏题，**不可**当作权威依据，仅作舆情/公开信息的弱线索。
3) bullet_points 用短句，5～10 条；summary_100 用中文≤120字概括。
4) JSON 形态固定为：
{"bullet_points":["…"],"summary_100":"…","caveats":["…"]}
"""


def _wiki_zh_extract(region: str, *, max_chars: int) -> str:
    q = (region or "").strip()
    if not q or max_chars <= 0:
        return ""
    title = q.replace(" ", "_")
    url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "people-regional-grounding/1.0 (simulation)"})
            if r.status_code != 200:
                return ""
            data = r.json()
            ex = (data.get("extract") or "").strip()
            return ex[:max_chars]
    except (OSError, httpx.HTTPError, ValueError, json.JSONDecodeError):
        return ""


def _issuer_for_scale(scale: str) -> IssuerArchetype | None:
    m: dict[str, IssuerArchetype] = {
        "street_shop": "sme",
        "sme_chain": "sme",
        "regional_group": "large_group",
        "national_group": "megacorp",
    }
    return m.get(scale)


def _passes_for_scenario(scenario: Scenario, rg: RegionalGroundingSpec) -> list[dict[str, str]]:
    region = (rg.region_label or scenario.policy_context.jurisdiction_name or "").strip() or "（未填地区）"
    sector = (rg.business_sector_brief or "").strip()
    scale_note = rg.business_scale
    pb = (scenario.player_brief or "").strip()[:800]
    uk_loc = (rg.user_known_local_policy_brief or "").strip()
    uk_cen = (rg.user_known_central_policy_brief or "").strip()

    if scenario.exercise_type == "policy":
        return [
            {
                "id": "local_existing",
                "title": "第1轮·地方既有政策与执行环境",
                "block": f"地区：{region}\n指挥台将讨论的新政策/措施要点（摘要）：{pb or '（未提供）'}\n"
                f"用户已知的当地既有政策/口径（可空）：{uk_loc or '（无）'}\n"
                "请归纳：此地与议题相关的**已常见**地方规定重点、执行部门习惯、处罚/督查敏感点（条目式，勿造文号）。",
            },
            {
                "id": "central_alignment",
                "title": "第2轮·中央上位框架与条块衔接",
                "block": f"地区：{region}\n议题同上。\n用户已知的中央/部委层面要点（可空）：{uk_cen or '（无）'}\n"
                "请归纳：与上述地方事项常见对应的**上位法律/行政法规/部委规章框架**（只写类型与典型关切，勿造文号）。",
            },
            {
                "id": "grassroots_baseline",
                "title": "第3轮·基层与街坊常态（非对新政策的反应）",
                "block": f"地区：{region}\n在**尚未讨论本轮新政策之前**，基层（社区、小商户、网格、县域媒体）对类似议题的"
                "**常态**态度与信息渠道习惯；避免写成已经知道新政策后的反应。",
            },
        ]

    # product / merchant
    scale_line = f"主体体量档位：{scale_note}（street_shop=街边小店… national_group=全国性集团）。"
    return [
        {
            "id": "local_commercial",
            "title": "第1轮·地方商事/行业监管与城管食安等常见关切",
            "block": f"地区：{region}\n{scale_line}\n业态/行业：{sector or '（未指定，请按 player_brief 推断）'}\n"
            f"产品/开店动作摘要：{pb or '（未提供）'}\n"
            "请归纳：证照、消防、环保、食安、城管、市监等**常见地方执行关切**（勿造个案）。",
        },
        {
            "id": "central_sector",
            "title": "第2轮·国家层面对该业态的通用监管与消费者保护框架",
            "block": f"业态：{sector or '（从 player_brief 推断）'}\n"
            "请归纳：国家层面**通用**监管框架与合规要点（类型级描述，勿造文号）。",
        },
        {
            "id": "local_central_stress",
            "title": "第3轮·地方执行差异与中央政策在当地的落地摩擦点",
            "block": f"地区：{region}；体量：{scale_note}\n"
            "请归纳：中央政策到地方时常见的**执行摩擦**（财政、编制、条块、舆情窗口等），仍勿编造具体文件。",
        },
    ]


def _call_pass(
    client: LLMClient,
    *,
    pass_id: str,
    title: str,
    task_block: str,
    wiki_excerpt: str,
    web_excerpt: str,
    temperature: float,
) -> dict[str, Any]:
    wiki_part = f"\n【可选：开放百科摘要片段（可能为空或不准确）】\n{wiki_excerpt}\n" if wiki_excerpt else ""
    web_part = (
        f"\n【可选：开放网页检索摘录（snippet 级，非全文；可能广告/过时/偏题，须交叉核实）】\n{web_excerpt}\n"
        if web_excerpt
        else ""
    )
    user = json.dumps(
        {
            "regional_grounding_pass": True,
            "pass_id": pass_id,
            "pass_title": title,
            "task": task_block + wiki_part + web_part,
        },
        ensure_ascii=False,
    )
    user += "\n输出 JSON：{bullet_points, summary_100, caveats}"
    return client.complete_json(system=GROUNDING_SYSTEM, user=user, temperature=temperature)


def apply_regional_grounding(
    scenario: Scenario, cfg: Any
) -> tuple[Scenario, list[dict[str, Any]], dict[str, Any]]:
    """
    若启用：多轮 fast LLM（及可选维基 / 开放网页检索）→ 合并进 player_brief / policy_context；
    返回 (scenario, trace, web_search_meta)。
    """
    rg = scenario.regional_grounding
    if not rg.enabled:
        return scenario, [], {}

    region = (rg.region_label or scenario.policy_context.jurisdiction_name or "").strip()
    wiki = ""
    if rg.mode == "wikipedia_then_llm" and region:
        wiki = _wiki_zh_extract(region, max_chars=int(rg.max_wiki_chars))

    web_excerpt = ""
    web_meta: dict[str, Any] = {}
    if rg.web_search_enabled:
        web_excerpt, web_meta = fetch_web_grounding_snippets(scenario, rg)

    passes = _passes_for_scenario(scenario, rg)
    trace: list[dict[str, Any]] = []
    client = cfg.fast_client
    local_pass_ids = ("local_existing", "local_commercial")

    for p in passes:
        raw = _call_pass(
            client,
            pass_id=p["id"],
            title=p["title"],
            task_block=p["block"],
            wiki_excerpt=wiki if p["id"] in local_pass_ids else "",
            web_excerpt=web_excerpt if p["id"] in local_pass_ids else "",
            temperature=float(rg.pass_temperature),
        )
        bps = raw.get("bullet_points") if isinstance(raw.get("bullet_points"), list) else []
        bps_s = [str(x).strip() for x in bps if str(x).strip()][:12]
        summ = str(raw.get("summary_100") or "")[:200]
        cvs = raw.get("caveats") if isinstance(raw.get("caveats"), list) else []
        cvs_s = [str(x).strip() for x in cvs if str(x).strip()][:6]
        trace.append(
            {
                "pass_id": p["id"],
                "title": p["title"],
                "bullet_points": bps_s,
                "summary_100": summ,
                "caveats": cvs_s,
                "wikipedia_used": bool(wiki),
                "web_search_used": bool(web_excerpt),
            }
        )

    digest_lines = [
        "【地区情境·模拟前多轮检索摘要 — 仅供推演，非官方法规/非备案检索】",
        f"目标地区：{region or '（见 policy_context.jurisdiction_name）'}",
    ]
    if wiki:
        digest_lines.append("【开放百科摘录（可能不完整）】" + wiki[: min(400, len(wiki))] + ("…" if len(wiki) > 400 else ""))
    if web_excerpt:
        digest_lines.append(
            "【开放网页检索摘录（snippet，非权威；须核实）】"
            + web_excerpt[: min(500, len(web_excerpt))]
            + ("…" if len(web_excerpt) > 500 else "")
        )
    for t in trace:
        digest_lines.append(f"— {t['title']} —")
        digest_lines.append(t.get("summary_100") or "")
        for b in t.get("bullet_points") or []:
            digest_lines.append(f"· {b}")
    digest = "\n".join(x for x in digest_lines if x)

    orig_pb = scenario.player_brief or ""
    new_pb = (
        digest
        + "\n\n【指挥台·本轮新动作 / 新政策（须叠加在上述既有地方与中央情境之上推演）】\n"
        + orig_pb
    )

    pc = scenario.policy_context
    extra_norm = (
        "\n\n【基层反响设定】请在「既有地方政策与民风常态」之上，叠加指挥台新政策带来的**增量**反应；"
        "勿写成对真空的首次反应。"
    )
    new_pc = pc.model_copy(
        update={
            "jurisdiction_name": (pc.jurisdiction_name or region),
            "local_norms_brief": (pc.local_norms_brief or "") + extra_norm,
        }
    )

    updates: dict[str, Any] = {
        "player_brief": new_pb,
        "policy_context": new_pc,
    }

    if rg.sync_issuer_archetype_from_scale and rg.business_scale != "unset":
        arch = _issuer_for_scale(rg.business_scale)
        if arch is not None:
            updates["issuer"] = scenario.issuer.model_copy(update={"archetype": arch})

    scenario2 = scenario.model_copy(update=updates)
    return scenario2, trace, web_meta


def regional_grounding_artifact_dict(
    trace: list[dict[str, Any]],
    *,
    region: str,
    web_search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ws = web_search or {}
    disclaimer = "开放百科与模型合成摘要，非官方法律检索；不得直接作为合规依据。"
    if ws.get("used"):
        disclaimer += " 含开放网页检索摘要，非权威信源，须交叉核实。"
    return {
        "disclaimer": disclaimer,
        "region_label": region,
        "passes": trace,
        "web_search": ws,
    }


def prebake_regional_grounding_for_ensemble(scenario: Scenario, cfg: Any) -> Scenario:
    """
    并行 ensemble 父进程先跑一轮 grounding 并写入 player_brief，再把 regional_grounding.enabled 关掉，
    避免每个 worker 重复 3 次快模型调用。
    """
    if not scenario.regional_grounding.enabled:
        return scenario
    s2, _, _ = apply_regional_grounding(scenario, cfg)
    return s2.model_copy(
        update={"regional_grounding": s2.regional_grounding.model_copy(update={"enabled": False})}
    )
