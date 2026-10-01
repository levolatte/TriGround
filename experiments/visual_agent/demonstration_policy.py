"""GT-free, scripted choices of real evidence tools for offline demonstrations.

The caller records ``origin=scripted`` and supplies completed tool events. ``None``
ends evidence gathering; the separate offline labeler chooses the final box.
"""

from __future__ import annotations

import re


def _has(query, words):
    return bool(re.search(r"\b(?:" + "|".join(words) + r")\b", query))


def _action(event):
    return event.get("executed_action") or event.get("action") or {}


def _candidate_order(candidate):
    scores = [float(source["score"]) for source in candidate.get("sources", [])
              if source.get("score") is not None]
    box = candidate["bbox"]
    return (-max(scores, default=0.0), (box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
            str(candidate["id"]))


def next_action(row, candidates, events, semantic):
    """Choose one permitted observation from Query, current pool and actual events."""
    semantic = semantic.get("query_info", semantic)
    query = row["query"].casefold()
    targets = [candidate for candidate in candidates if candidate["role"] == "target"]
    keep = next(candidate for candidate in targets if str(candidate["id"]) == "KEEP")
    other_targets = sorted((candidate for candidate in targets if str(candidate["id"]) != "KEEP"
                            and candidate["bbox"] != keep["bbox"]), key=_candidate_order)
    references = [candidate for candidate in candidates if candidate["role"] == "reference"]
    observed = [event for event in events if _action(event).get("action") in
                {"inspect_regions", "measure_depth", "search_candidates"}]
    if len(observed) >= 6:
        return None
    used = {_action(event)["action"] for event in observed}
    last = observed[-1] if observed else None
    last_kind = _action(last).get("action") if last else None
    last_status = (last.get("observation") or {}).get("status") if last else None

    relation = semantic.get("relation_type")
    scope = semantic.get("scope")
    reference = re.search(
        r"\b(?:to\s+the\s+)?(?:left|right)\s+of\b|"
        r"\b(?:next\s+to|beside|between|behind|above|below|in\s+front\s+of)\b|"
        r"\bnear\s+(?!(?:the\s+)?(?:camera|viewer)\b)", query)
    target_query = query[:reference.start()] if reference else query
    explicit_camera = _has(target_query, ("foreground", "background", "midground", "frontmost", "backmost"))
    explicit_camera |= bool(re.search(r"\b(?:front|middle|back)\s+(?:ground|layer|of the scene)\b", target_query))
    explicit_camera |= bool(re.search(r"\b(?:to|from|near|far|toward)\s+(?:the\s+)?(?:camera|viewer)\b", target_query))
    explicit_camera |= any(word in target_query for word in ("前景", "中景", "背景", "镜头距离", "相机距离", "靠近镜头", "远离镜头"))
    bare_comparison = _has(target_query, ("nearer", "farther", "closer", "closest", "nearest", "farthest", "furthest"))
    reference_distance = bool(re.search(r"\b(?:to|from|than)\s+(?:the\s+)?(?!camera\b|viewer\b)[a-z]+", query))
    sideways = bool(re.search(r"\b(?:nearer|farther|closer|further)\s+(?:left|right|up|down)\b", query))
    camera = scope == "single" and (explicit_camera or
                                     relation in {"camera_near", "camera_far"} and
                                     bare_comparison and not reference_distance and not sideways)
    appearance = _has(query, ("red", "blue", "green", "yellow", "white", "black", "brown", "gray",
                              "grey", "orange", "purple", "pink", "silver", "gold", "color", "colour",
                              "striped", "plaid", "shirt", "wearing", "text", "logo", "holding"))
    appearance |= any(word in query for word in ("红色", "蓝色", "绿色", "白色", "黑色", "颜色", "衣服", "文字"))
    visibility = _has(query, ("hidden", "obscured", "occluded", "blurred", "dim", "dark", "thermal",
                              "infrared", "heat", "hot", "cold"))
    visibility |= any(word in query for word in ("红外", "热源", "遮挡", "模糊", "昏暗"))
    spatial = _has(query, ("left", "right", "leftmost", "rightmost", "middle", "center", "centre",
                           "first", "second", "third", "fourth", "last", "top", "bottom", "upper", "lower"))
    spatial |= any(word in query for word in ("左边", "右边", "最左", "最右", "中间", "第二", "第三"))
    ordinal_words = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh",
                     "eighth", "ninth", "tenth")
    ordinal = max([index for index, word in enumerate(ordinal_words, 1) if _has(query, (word,))] +
                  [int(match.group(1)) for match in re.finditer(r"\b(10|[1-9])(?:st|nd|rd|th)\b", query)] +
                  [index for index, word in enumerate("一二三四五六七八九十", 1) if "第" + word in query] +
                  [2 if _has(query, ("other", "another")) else 1])
    images = row["images"]
    rgb, ir = bool(images.get("rgb")), bool(images.get("ir"))

    # An explicit ordinal with too few target hypotheses is a GT-free reason to search.
    missing_ordinal = ordinal > len(targets)
    no_external_target = not other_targets
    search_needed = missing_ordinal or no_external_target
    category = semantic.get("target_category")
    searches = [event for event in observed if _action(event).get("action") == "search_candidates"]
    ir_registered = row.get("depth_encoding") == "city_mm" or row.get("ir_rgb_registration") == "normalized_shared_frame"
    retry_ir = (len(searches) == 1 and _action(searches[0]).get("modality") == "rgb" and
                (searches[0].get("observation") or {}).get("status") == "EMPTY" and
                last is searches[0] and ir and ir_registered)
    if (search_needed and category and scope == "single" and row.get("dino_model_path")
            and (not searches or retry_ir)):
        modality = "ir" if retry_ir or visibility and ir and ir_registered else "rgb"
        if images.get(modality):
            return {"action": "search_candidates", "category": category, "region": "full",
                    "modality": modality, "role": "target",
                    "evidence_note": ("RGB search was EMPTY; check aligned IR for the missing target."
                                      if retry_ir else
                                      "No external target hypothesis exists; search the stated category."
                                      if no_external_target else
                                      "Query ordinal exceeds current target pool; search its stated category.")}

    if (camera and row.get("depth_encoding") == "city_mm" and images.get("depth_raw")
            and "measure_depth" not in used):
        ids = ["KEEP"] + [str(candidate["id"]) for candidate in other_targets[:1]]
        return {"action": "measure_depth", "candidate_ids": ids,
                "evidence_note": "Check camera-depth ordering; only reliable measurements can support it."}

    search_added = any((_action(event).get("action") == "search_candidates" and
                        (event.get("observation") or {}).get("data", {}).get("appended_ids"))
                       for event in observed)
    search_failed = last_kind == "search_candidates" and last_status in {"UNKNOWN", "EMPTY"}
    depth_unknown = last_kind == "measure_depth" and last_status in {"UNKNOWN", "EMPTY"}
    need_image = (appearance or visibility or relation == "other" or
                  (len(targets) > 1 and not (spatial and not camera)) or
                  search_added or search_failed or depth_unknown)
    if "inspect_regions" in used or not need_image or (not rgb and not ir):
        return None
    if appearance and not rgb and not visibility:
        return None
    recent_added = [str(value) for event in observed if _action(event).get("action") == "search_candidates"
                    for value in (event.get("observation") or {}).get("data", {}).get("appended_ids", [])]
    ids = [recent_added[-1]] if recent_added and any(str(c["id"]) == recent_added[-1] for c in targets) else ["KEEP"]
    companion = (references[:1] if relation == "other" and references else other_targets[:1])
    if len(ids) == 1 and companion and str(companion[0]["id"]) not in ids:
        ids.append(str(companion[0]["id"]))
    compare_identity = visibility or depth_unknown or search_added or len(targets) > 1 and not spatial
    modalities = (["rgb", "ir"] if rgb and ir and compare_identity else
                  ["rgb"] if rgb else ["ir"])
    view = "cross" if len(modalities) == 2 else "pair" if len(ids) == 2 else "single"
    reason = ("Depth was UNKNOWN; compare object identity without claiming camera order."
              if depth_unknown else
              "Search was uninformative; inspect the current target without claiming a new one."
              if search_failed else
              "Search added a target; compare its actual RGB/IR appearance."
              if search_added else
              "Compare RGB appearance and IR silhouette at current candidate locations."
              if len(modalities) == 2 else
              "Check RGB appearance and original-scene relations."
              if modalities == ["rgb"] else
              "Check IR silhouette at the current candidate location.")
    return {"action": "inspect_regions", "candidate_ids": ids, "modalities": modalities,
            "view": view, "evidence_note": reason}
