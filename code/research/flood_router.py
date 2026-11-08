#!/usr/bin/env python3
"""
GE 洪水制图 · 决策表路由（确定性代码，非 LLM）。

输入：LLM 场景分析标签 {scene, density, confidence} + 确定性信号指标
输出：RoutingResult { expectedBest, ranking, matchedRule, missingFields }

规则（按优先级，命中第一条即停；urban 用 S_urban 相干度策略）：
    urban + highDensity   → urban 优先（城区洪水：灾前高相干区去相干）
    urban                 → urban > chg > int
    open                  → int 优先（开阔区洪水 = 暗水）
    vegetated             → chg 优先（植被淹没 = 后向散射变化）
    默认                  → 无预期（四路全跑，取 IoU 最高者）
"""
RULES = [
    {
        "name": "urban+highDensity",
        "needs": ["scene", "scene_density"],
        "match": lambda a: a["scene"] == "urban" and a["scene_density"] == "high",
        "expected": "urban",
        "ranking": ["urban", "chg", "int", "coh"],
    },
    {
        "name": "urban",
        "needs": ["scene"],
        "match": lambda a: a["scene"] == "urban",
        "expected": "urban",
        "ranking": ["urban", "chg", "int", "coh"],
    },
    {
        "name": "open",
        "needs": ["scene"],
        "match": lambda a: a["scene"] == "open",
        "expected": "int",
        "ranking": ["int", "chg", "coh"],
    },
    {
        "name": "vegetated",
        "needs": ["scene"],
        "match": lambda a: a["scene"] in ("vegetated", "rural"),
        "expected": "chg",
        "ranking": ["chg", "int", "coh"],
    },
]


def route(analysis):
    """按优先级匹配决策表。analysis 至少含 scene；可选 scene_density。"""
    missing = set()
    for rule in RULES:
        fields_ok = True
        for key in rule["needs"]:
            val = analysis.get(key)
            if val is None or val == "":
                missing.add(key)
                fields_ok = False
        if not fields_ok:
            continue
        if rule["match"](analysis):
            return {
                "expectedBest": rule["expected"],
                "ranking": list(rule["ranking"]),
                "matchedRule": rule["name"],
                "missingFields": sorted(missing),
            }
    return {
        "expectedBest": None,
        "ranking": [],
        "matchedRule": None,
        "missingFields": sorted(missing),
    }
