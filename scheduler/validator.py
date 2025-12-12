from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .utils import DAY_MIN

@dataclass
class Violation:
    code: str
    message: str
    task_id: Optional[str] = None
    therapist_id: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[dict] = None


def intervals_overlap(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _merge(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    intervals = sorted([(int(a),int(b)) for a,b in intervals if b> a], key=lambda x: x[0])
    out: List[Tuple[int,int]] = []
    for a,b in intervals:
        if not out or a > out[-1][1]:
            out.append((a,b))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], b))
    return out


def _availability_unavailable(avail: List[Dict[str,int]], horizon: int=DAY_MIN) -> List[Tuple[int,int]]:
    if not avail:
        return [(0,horizon)]
    av = _merge([(w["start"], w["end"]) for w in avail])
    un: List[Tuple[int,int]] = []
    cur = 0
    for a,b in av:
        if a > cur:
            un.append((cur,a))
        cur = max(cur,b)
    if cur < horizon:
        un.append((cur,horizon))
    return un


def validate_schedule(input_data: dict, schedule: dict, *, separation_mode: str = "start_to_start") -> List[Violation]:
    violations: List[Violation] = []

    tasks = {t["task_id"]: t for t in input_data.get("tasks", [])}
    therapists = {th["therapist_id"]: th for th in input_data.get("therapists", [])}
    patients = {p["patient_id"]: p for p in input_data.get("patients", [])}
    resources = {r["resource_id"]: r for r in input_data.get("resources", [])}
    locks = {lk["task_id"]: lk for lk in input_data.get("locks", [])}

    cfg = input_data.get("config", {})
    default_doc = int(cfg.get("default_doc_buffer_min", 10))
    gym_turn = int(cfg.get("gym_turnover_min", 5))
    max_ot = int(cfg.get("max_overtime_min", 120))

    # Resolve resource IDs
    gym_id = next((r["resource_id"] for r in resources.values() if r.get("resource_type") == "GYM_ROOM"), "GYM_ROOM_1")
    assist_id = next((r["resource_id"] for r in resources.values() if r.get("resource_type") == "ASSIST_POOL"), "ASSIST_POOL")
    assist_cap = int(resources.get(assist_id, {}).get("capacity", cfg.get("assist_pool_capacity", 1)))

    assignments = schedule.get("assignments", [])

    # 1) task assignment count
    count_by_task: Dict[str,int] = defaultdict(int)
    for a in assignments:
        count_by_task[a["task_id"]] += 1

    for tid, t in tasks.items():
        must = bool(t.get("must_schedule", True)) and t.get("readiness", "READY") == "READY"
        c = count_by_task.get(tid, 0)
        if must and c != 1:
            violations.append(Violation(
                code="TASK_ASSIGNMENT_COUNT",
                message=f"Task {tid} must be scheduled exactly once but count={c}.",
                task_id=tid,
                details={"count": c, "must": must}
            ))
        if (not must) and c > 1:
            violations.append(Violation(
                code="TASK_ASSIGNMENT_COUNT",
                message=f"Task {tid} optional/not-ready but scheduled multiple times count={c}.",
                task_id=tid,
                details={"count": c}
            ))

    # Build therapist occupied intervals (face + doc_buffer)
    th_occ: Dict[str, List[Tuple[int,int,str]]] = {k: [] for k in therapists.keys()}
    gym_occ: List[Tuple[int,int,str]] = []     # start,end,task_id (includes turnover)
    assist_occ: List[Tuple[int,int,str]] = []  # face only
    eq_occ: Dict[str, List[Tuple[int,int,str]]] = defaultdict(list)

    # gym unavailability
    gym_unavail = _availability_unavailable(resources.get(gym_id, {}).get("availability", []))

    # assist unavailability
    assist_unavail = _availability_unavailable(resources.get(assist_id, {}).get("availability", []))

    # 2) per-assignment checks
    for a in assignments:
        tid = a["task_id"]
        if tid not in tasks:
            violations.append(Violation(code="UNKNOWN_TASK", message=f"Assignment references unknown task {tid}", task_id=tid))
            continue
        t = tasks[tid]
        k = a["therapist_id"]
        if k not in therapists:
            violations.append(Violation(code="UNKNOWN_THERAPIST", message=f"Assignment references unknown therapist {k}", task_id=tid, therapist_id=k))
            continue
        th = therapists[k]
        start, end = int(a["start_min"]), int(a["end_min"])
        face = int(t["face_dur_min"])
        doc = int(t.get("doc_buffer_min", default_doc))
        loc = a.get("location", {}).get("type") or a.get("location_type") or "BEDSIDE"

        # duration
        if end != start + face:
            violations.append(Violation(
                code="DURATION_MISMATCH",
                message=f"Task {tid} end != start + face_dur.",
                task_id=tid,
                details={"start": start, "end": end, "face": face}
            ))

        # location allowed
        if loc not in set(t.get("allowed_locations", ["BEDSIDE"])):
            violations.append(Violation(
                code="LOCATION_NOT_ALLOWED",
                message=f"Task {tid} scheduled at {loc} but allowed={t.get('allowed_locations')}.",
                task_id=tid,
                details={"loc": loc, "allowed": t.get("allowed_locations", [])}
            ))

        # windows
        windows = t.get("allowed_windows", [])
        if windows:
            ok_window = any(start >= int(w["start"]) and end <= int(w["end"]) for w in windows)
            if not ok_window:
                violations.append(Violation(
                    code="WINDOW_VIOLATION",
                    message=f"Task {tid} scheduled outside allowed windows.",
                    task_id=tid,
                    details={"start": start, "end": end, "windows": windows}
                ))

        # skills
        req = set(t.get("required_skills", []))
        if not req.issubset(set(th.get("skills", []))):
            violations.append(Violation(
                code="SKILL_MISMATCH",
                message=f"Therapist {k} lacks skills {sorted(req)} for task {tid}.",
                task_id=tid, therapist_id=k,
                details={"required": sorted(req), "therapist_skills": th.get("skills", [])}
            ))

        # shift bounds (+max overtime)
        occ_end = start + face + doc
        if start < int(th["shift_start"]) or occ_end > int(th["shift_end"]) + max_ot:
            violations.append(Violation(
                code="SHIFT_BOUNDS",
                message=f"Task {tid} violates therapist {k} shift/overtime bounds.",
                task_id=tid, therapist_id=k,
                details={"start": start, "occ_end": occ_end, "shift": [th["shift_start"], th["shift_end"]], "max_ot": max_ot}
            ))

        # breaks
        for br in th.get("breaks", []):
            bs, be = int(br["start"]), int(br["end"])
            if intervals_overlap((start, occ_end), (bs, be)):
                violations.append(Violation(
                    code="BREAK_OVERLAP",
                    message=f"Task {tid} overlaps break for therapist {k}.",
                    task_id=tid, therapist_id=k,
                    details={"task_occ": [start, occ_end], "break": br}
                ))

        # lock integrity
        if tid in locks and locks[tid].get("locked", False):
            lk = locks[tid]
            locked_loc = lk.get("location") or lk.get("location_type")
            if (lk.get("therapist_id") and a["therapist_id"] != lk["therapist_id"]) or \
               (lk.get("start_min") is not None and int(a["start_min"]) != int(lk["start_min"])) or \
               (locked_loc and loc != locked_loc):
                violations.append(Violation(
                    code="LOCK_VIOLATION",
                    message=f"Locked task {tid} was changed.",
                    task_id=tid,
                    details={"locked": lk, "actual": a}
                ))

        # resource reservations
        res_list = set(a.get("resources", []))
        if loc == "GYM":
            if gym_id not in res_list:
                violations.append(Violation(
                    code="MISSING_GYM_RESERVATION",
                    message=f"Task {tid} in gym but missing {gym_id} reservation.",
                    task_id=tid,
                    resource_id=gym_id
                ))
        if bool(t.get("requires_assist", False)):
            if assist_id not in res_list:
                violations.append(Violation(
                    code="MISSING_ASSIST_RESERVATION",
                    message=f"Task {tid} requires assist but missing {assist_id} reservation.",
                    task_id=tid,
                    resource_id=assist_id
                ))

        # build occupancy
        th_occ[k].append((start, occ_end, tid))
        if loc == "GYM":
            gym_occ.append((start, end + gym_turn, tid))
            # gym availability
            for ua, ub in gym_unavail:
                if intervals_overlap((start, end + gym_turn), (ua, ub)):
                    violations.append(Violation(
                        code="GYM_UNAVAILABLE",
                        message=f"Task {tid} overlaps gym unavailable period.",
                        task_id=tid,
                        resource_id=gym_id,
                        details={"task": [start, end+gym_turn], "unavail": [ua, ub]}
                    ))
                    break

        if bool(t.get("requires_assist", False)):
            assist_occ.append((start, end, tid))
            # assist availability
            for ua, ub in assist_unavail:
                if intervals_overlap((start, end), (ua, ub)):
                    violations.append(Violation(
                        code="ASSIST_UNAVAILABLE",
                        message=f"Task {tid} overlaps assist unavailable period.",
                        task_id=tid,
                        resource_id=assist_id,
                        details={"task": [start, end], "unavail": [ua, ub]}
                    ))
                    break

        # equipment occupancy
        for rid in set(t.get("required_resources", [])):
            if rid in resources and resources[rid].get("resource_type") == "EQUIPMENT_SET":
                eq_occ[rid].append((start, end, tid))

    # 3) therapist overlaps (with doc buffer)
    for k, ints in th_occ.items():
        ints_sorted = sorted(ints, key=lambda x: x[0])
        for a, b in zip(ints_sorted, ints_sorted[1:]):
            if intervals_overlap((a[0], a[1]), (b[0], b[1])):
                violations.append(Violation(
                    code="THERAPIST_OVERLAP",
                    message=f"Therapist {k} has overlapping tasks {a[2]} and {b[2]}.",
                    therapist_id=k,
                    details={"a": a, "b": b}
                ))

    # 4) gym overlaps (with turnover)
    gym_sorted = sorted(gym_occ, key=lambda x: x[0])
    for a, b in zip(gym_sorted, gym_sorted[1:]):
        if intervals_overlap((a[0], a[1]), (b[0], b[1])):
            violations.append(Violation(
                code="GYM_OVERLAP",
                message=f"Gym overlap between tasks {a[2]} and {b[2]}.",
                resource_id=gym_id,
                details={"a": a, "b": b}
            ))

    # 5) assist capacity
    if assist_occ:
        events = []
        for st, en, tid in assist_occ:
            events.append((st, +1, tid))
            events.append((en, -1, tid))
        events.sort()
        cur = 0
        for time, delta, tid in events:
            cur += delta
            if cur > assist_cap:
                violations.append(Violation(
                    code="ASSIST_CAPACITY",
                    message=f"Assist capacity exceeded at t={time}.",
                    resource_id=assist_id,
                    details={"time": time, "cur": cur, "cap": assist_cap}
                ))
                break

    # 6) equipment capacity
    for rid, occ in eq_occ.items():
        cap = int(resources[rid].get("capacity", 1))
        events = []
        for st, en, tid in occ:
            events.append((st, +1, tid))
            events.append((en, -1, tid))
        events.sort()
        cur = 0
        for time, delta, tid in events:
            cur += delta
            if cur > cap:
                violations.append(Violation(
                    code="EQUIP_CAPACITY",
                    message=f"Equipment capacity exceeded for {rid} at t={time}.",
                    resource_id=rid,
                    details={"time": time, "cur": cur, "cap": cap}
                ))
                break

    # 7) session separation
    for sep in input_data.get("separations", []):
        a_id, b_id, gap = sep["task_a"], sep["task_b"], int(sep.get("min_gap_min", 0))
        aa = next((x for x in assignments if x["task_id"] == a_id), None)
        bb = next((x for x in assignments if x["task_id"] == b_id), None)
        if aa and bb:
            if separation_mode == "end_to_start":
                ok = int(bb["start_min"]) >= int(aa["end_min"]) + gap
            else:
                ok = int(bb["start_min"]) >= int(aa["start_min"]) + gap
            if not ok:
                violations.append(Violation(
                    code="SESSION_SEPARATION",
                    message=f"Separation violated between {a_id} and {b_id}.",
                    details={"a": aa, "b": bb, "gap": gap, "mode": separation_mode}
                ))

    return violations


def why_infeasible(task: dict, input_data: dict, existing_schedule: dict, *, separation_mode: str = "start_to_start", step: int = 5) -> dict:
    """Best-effort diagnostic: scan candidate starts and record dominant blocker types."""
    cfg = input_data.get("config", {})
    default_doc = int(cfg.get("default_doc_buffer_min", 10))
    gym_turn = int(cfg.get("gym_turnover_min", 5))
    max_ot = int(cfg.get("max_overtime_min", 120))

    therapists = {t["therapist_id"]: t for t in input_data.get("therapists", [])}
    patients = {p["patient_id"]: p for p in input_data.get("patients", [])}
    resources = {r["resource_id"]: r for r in input_data.get("resources", [])}

    gym_id = next((r["resource_id"] for r in resources.values() if r.get("resource_type") == "GYM_ROOM"), "GYM_ROOM_1")
    assist_id = next((r["resource_id"] for r in resources.values() if r.get("resource_type") == "ASSIST_POOL"), "ASSIST_POOL")
    assist_cap = int(resources.get(assist_id, {}).get("capacity", cfg.get("assist_pool_capacity", 1)))

    # Build current calendars from schedule
    th_cal: Dict[str, List[Tuple[int,int,str]]] = defaultdict(list)
    gym_cal: List[Tuple[int,int,str]] = []
    assist_cal: List[Tuple[int,int,str]] = []

    tasks_by_id = {t["task_id"]: t for t in input_data.get("tasks", [])}

    for a in existing_schedule.get("assignments", []):
        tid = a["task_id"]
        t = tasks_by_id.get(tid, {})
        start = int(a["start_min"]); end = int(a["end_min"])
        face = int(t.get("face_dur_min", end-start))
        doc = int(t.get("doc_buffer_min", default_doc))
        loc = a.get("location", {}).get("type") or a.get("location_type") or "BEDSIDE"
        k = a.get("therapist_id")
        if k:
            th_cal[k].append((start, start + face + doc, tid))
        if loc == "GYM":
            gym_cal.append((start, end + gym_turn, tid))
        if bool(t.get("requires_assist", False)):
            assist_cal.append((start, end, tid))

    # add breaks
    for k, th in therapists.items():
        for br in th.get("breaks", []):
            th_cal[k].append((int(br["start"]), int(br["end"]), "BREAK"))

    gym_unavail = _availability_unavailable(resources.get(gym_id, {}).get("availability", []))
    assist_unavail = _availability_unavailable(resources.get(assist_id, {}).get("availability", []))

    req_skills = set(task.get("required_skills", []))
    allowed_locs = list(dict.fromkeys(task.get("allowed_locations", ["BEDSIDE"])))
    face = int(task.get("face_dur_min", 30))
    doc = int(task.get("doc_buffer_min", default_doc))
    pid = task.get("patient_id")
    p = patients.get(pid, {})

    blockers = defaultdict(int)
    examples = []

    eligible_ks = [k for k, th in therapists.items() if req_skills.issubset(set(th.get("skills", [])))]
    if not eligible_ks:
        return {"ok": False, "dominant": "NO_ELIGIBLE_THERAPIST", "details": {"required_skills": sorted(req_skills)}}

    windows = task.get("allowed_windows", [])
    if not windows:
        windows = [{"start": 0, "end": DAY_MIN}]

    # separation constraints
    sep_rules = [(s["task_a"], s["task_b"], int(s.get("min_gap_min", 0))) for s in input_data.get("separations", [])]
    # quick lookup for already scheduled predecessors
    assigned = {a["task_id"]: a for a in existing_schedule.get("assignments", [])}

    for k in eligible_ks:
        th = therapists[k]
        for loc in allowed_locs:
            for w in windows:
                a = int(w["start"]); b = int(w["end"])
                t0 = ((a + step - 1)//step)*step
                while t0 + face <= b:
                    occ = (t0, t0 + face + doc)
                    # shift
                    if t0 < int(th["shift_start"]) or occ[1] > int(th["shift_end"]) + max_ot:
                        blockers["SHIFT_BOUNDS"] += 1
                        t0 += step
                        continue
                    # therapist busy/break
                    if any(intervals_overlap((occ[0], occ[1]), (x[0], x[1])) for x in th_cal[k]):
                        blockers["THERAPIST_BUSY"] += 1
                        t0 += step
                        continue
                    # gym
                    if loc == "GYM":
                        pat = (t0, t0 + face + gym_turn)
                        if any(intervals_overlap(pat, (x[0], x[1])) for x in gym_cal):
                            blockers["GYM_BUSY"] += 1
                            t0 += step
                            continue
                        if any(intervals_overlap(pat, ua) for ua in gym_unavail):
                            blockers["GYM_UNAVAILABLE"] += 1
                            t0 += step
                            continue
                    # assist
                    if bool(task.get("requires_assist", False)):
                        pat = (t0, t0 + face)
                        if any(intervals_overlap(pat, ua) for ua in assist_unavail):
                            blockers["ASSIST_UNAVAILABLE"] += 1
                            t0 += step
                            continue
                        # capacity
                        events = 0
                        for st,en,_ in assist_cal:
                            if intervals_overlap(pat, (st,en)):
                                events += 1
                        if events >= assist_cap:
                            blockers["ASSIST_BUSY"] += 1
                            t0 += step
                            continue
                    # separation
                    violated = False
                    for a_id, b_id, gap in sep_rules:
                        if task.get("task_id") == b_id and a_id in assigned:
                            aa = assigned[a_id]
                            if separation_mode == "end_to_start":
                                if t0 < int(aa["end_min"]) + gap:
                                    blockers["SEPARATION"] += 1
                                    violated = True
                                    break
                            else:
                                if t0 < int(aa["start_min"]) + gap:
                                    blockers["SEPARATION"] += 1
                                    violated = True
                                    break
                    if violated:
                        t0 += step
                        continue
                    # feasible!
                    return {
                        "ok": True,
                        "suggested": {"therapist_id": k, "location": loc, "start_min": t0},
                        "patient": {"unit": p.get("unit"), "room": p.get("room")},
                    }
                # end while

    if not blockers:
        dominant = "UNKNOWN"
    else:
        dominant = max(blockers.items(), key=lambda kv: kv[1])[0]

    return {
        "ok": False,
        "dominant": dominant,
        "blockers": dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:6]),
        "note": "Counts are from scanning candidate start times across eligible therapists/locations/windows.",
    }
