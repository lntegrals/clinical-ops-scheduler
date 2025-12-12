from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from datetime import date

from .utils import DAY_MIN

STEP = 5  # minutes

@dataclass
class SolveResult:
    schedule: Dict[str, Any]
    status: str


def _merge(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    intervals = sorted([(int(a),int(b)) for a,b in intervals if b> a], key=lambda x: x[0])
    out: List[Tuple[int,int]] = []
    for a,b in intervals:
        if not out or a > out[-1][1]:
            out.append((a,b))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], b))
    return out

def _overlaps(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])

def _is_free(start: int, end: int, cal: List[Tuple[int,int]]) -> bool:
    for a,b in cal:
        if _overlaps((start,end),(a,b)):
            return False
    return True

def _add_interval(cal: List[Tuple[int,int]], start: int, end: int) -> None:
    cal.append((start,end))

def _availability_complement(avail: List[Dict[str,int]], horizon: int=DAY_MIN) -> List[Tuple[int,int]]:
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


def solve_greedy(data: Dict[str, Any], *, separation_mode: str = "start_to_start") -> SolveResult:
    config = data.get("config", {})
    default_doc = int(config.get("default_doc_buffer_min", 10))
    gym_turn = int(config.get("gym_turnover_min", 5))

    therapists = {t["therapist_id"]: t for t in data.get("therapists", [])}
    patients = {p["patient_id"]: p for p in data.get("patients", [])}
    resources = {r["resource_id"]: r for r in data.get("resources", [])}
    tasks = [dict(t) for t in data.get("tasks", [])]
    locks = {lk["task_id"]: lk for lk in data.get("locks", [])}

    # calendars
    th_cal: Dict[str, List[Tuple[int,int]]] = {k: [] for k in therapists}
    for k, th in therapists.items():
        # breaks
        for br in th.get("breaks", []):
            _add_interval(th_cal[k], int(br["start"]), int(br["end"]))
    gym_id = next((r["resource_id"] for r in resources.values() if r.get("resource_type") == "GYM_ROOM"), "GYM_ROOM_1")
    gym_cal: List[Tuple[int,int]] = []
    if gym_id in resources:
        un = _availability_complement(resources[gym_id].get("availability", []))
        for a,b in un:
            _add_interval(gym_cal, a, b)
    assist_id = next((r["resource_id"] for r in resources.values() if r.get("resource_type") == "ASSIST_POOL"), "ASSIST_POOL")
    assist_cap = int(resources.get(assist_id, {}).get("capacity", config.get("assist_pool_capacity", 1)))
    assist_cal: List[Tuple[int,int]] = []
    if assist_id in resources:
        un = _availability_complement(resources[assist_id].get("availability", []))
        for a,b in un:
            _add_interval(assist_cal, a, b)

    # separation tracking (for greedy we enforce via constraints)
    sep_rules = [(s["task_a"], s["task_b"], int(s.get("min_gap_min", 0))) for s in data.get("separations", [])]
    start_by_task: Dict[str,int] = {}
    end_by_task: Dict[str,int] = {}

    # scarcity: count therapist eligibility per task skills
    def eligible_therapists_for_task(t: Dict[str,Any]) -> List[str]:
        req = set(t.get("required_skills", []))
        out = []
        for k, th in therapists.items():
            if req.issubset(set(th.get("skills", []))):
                out.append(k)
        return out

    # sorting key
    def sort_key(t: Dict[str,Any]):
        tid = t["task_id"]
        readiness = t.get("readiness", "READY")
        must = bool(t.get("must_schedule", True)) and readiness == "READY"
        pr = int(t.get("priority", 3))
        windows = t.get("allowed_windows", [])
        earliest = min((w["start"] for w in windows), default=0)
        deadline = min((w["end"] for w in windows), default=DAY_MIN)
        # discharge urgency
        pid = t.get("patient_id")
        p = patients.get(pid, {})
        dd = p.get("discharge_target_date")
        urg = 999
        try:
            if dd and t.get("service_date"):
                urg = (date.fromisoformat(str(dd)) - date.fromisoformat(str(t.get("service_date")))).days
        except Exception:
            urg = 999
        # scarcity
        elig = eligible_therapists_for_task(t)
        scarcity = 1 if (len(elig) <= 1 or "GYM" in set(t.get("allowed_locations", [])) or bool(t.get("requires_assist", False))) else 0
        dur = int(t.get("face_dur_min", 30))
        return (
            0 if must else 1,
            -pr,
            -scarcity,
            urg,
            deadline,
            earliest,
            -dur,
            tid,
        )

    tasks_sorted = sorted(tasks, key=sort_key)

    assignments: List[Dict[str,Any]] = []

    def travel_cost(pid: str, k: str) -> int:
        p = patients.get(pid, {})
        home = therapists[k].get("home_unit")
        if home in (None, "", "UNKNOWN"):
            return 2
        if home == "FLOAT":
            return 1
        return 0 if p.get("unit") == home else 1

    def option_cost(t: Dict[str,Any], k: str, loc: str) -> Tuple[int,int]:
        # (travel, workload)
        travel = travel_cost(t.get("patient_id"), k)
        workload = sum(1 for a in assignments if a["therapist_id"] == k)
        return (travel, workload)

    def feasible_start(t: Dict[str,Any], k: str, loc: str) -> Optional[int]:
        th = therapists[k]
        face = int(t["face_dur_min"])
        doc = int(t.get("doc_buffer_min", default_doc))
        occ_end_delta = face + doc
        pat_end_delta = face + (gym_turn if loc == "GYM" else 0)

        windows = t.get("allowed_windows", [{"start": th["shift_start"], "end": th["shift_end"]}])
        for w in windows:
            a = int(w["start"]); b = int(w["end"])
            # iterate in STEP increments
            start = ((a + STEP - 1)//STEP)*STEP
            while start + face <= b:
                occ = (start, start + occ_end_delta)
                if start < int(th["shift_start"]) or occ[1] > int(th["shift_end"]) + int(config.get("max_overtime_min", 120)):
                    start += STEP; continue
                if not _is_free(occ[0], occ[1], th_cal[k]):
                    start += STEP; continue
                if loc == "GYM":
                    pat = (start, start + pat_end_delta)
                    if not _is_free(pat[0], pat[1], gym_cal):
                        start += STEP; continue
                if bool(t.get("requires_assist", False)):
                    # capacity check by counting overlaps in assist_cal (treat capacity>1 by soft)
                    pat = (start, start + face)
                    # If capacity==1, just check free in single calendar
                    if assist_cap == 1:
                        if not _is_free(pat[0], pat[1], assist_cal):
                            start += STEP; continue
                    else:
                        # simple sweep: count overlaps at start time
                        cnt = sum(1 for a2,b2 in assist_cal if _overlaps(pat,(a2,b2)))
                        if cnt >= assist_cap:
                            start += STEP; continue
                # separation rules
                tid = t["task_id"]
                for a_id, b_id, gap in sep_rules:
                    if tid == b_id and a_id in start_by_task:
                        if separation_mode == "end_to_start":
                            if start < end_by_task[a_id] + gap:
                                start += STEP
                                break
                        else:
                            if start < start_by_task[a_id] + gap:
                                start += STEP
                                break
                else:
                    return start
                continue
        return None

    # schedule locks first
    for tid, lk in locks.items():
        if tid not in [t["task_id"] for t in tasks_sorted]:
            continue
        if not lk.get("locked", True):
            continue
        t = next(tt for tt in tasks_sorted if tt["task_id"] == tid)
        k = lk.get("therapist_id")
        loc = lk.get("location") or lk.get("location_type") or "BEDSIDE"
        start = int(lk.get("start_min", 0))
        face = int(t["face_dur_min"])
        doc = int(t.get("doc_buffer_min", default_doc))
        _add_interval(th_cal[k], start, start + face + doc)
        if loc == "GYM":
            _add_interval(gym_cal, start, start + face + gym_turn)
        if bool(t.get("requires_assist", False)):
            _add_interval(assist_cal, start, start + face)
        start_by_task[tid] = start
        end_by_task[tid] = start + face
        assignments.append({
            "task_id": tid,
            "patient_id": t.get("patient_id"),
            "therapist_id": k,
            "start_min": start,
            "end_min": start + face,
            "location": {"type": loc, "detail": {}},
            "resources": list(dict.fromkeys(t.get("required_resources", []))),
            "status": "PLANNED",
            "locked": True,
            "explainability": {"reasons": ["locked"]},
        })

    unscheduled: List[str] = []
    for t in tasks_sorted:
        tid = t["task_id"]
        if tid in locks and locks[tid].get("locked", True):
            continue
        readiness = t.get("readiness", "READY")
        must = bool(t.get("must_schedule", True)) and readiness == "READY"
        req = set(t.get("required_skills", []))
        allowed_locs = list(dict.fromkeys(t.get("allowed_locations", ["BEDSIDE"])))

        elig = []
        for k, th in therapists.items():
            if req.issubset(set(th.get("skills", []))):
                elig.append(k)
        if not elig:
            if must:
                unscheduled.append(tid)
            continue

        # enumerate options sorted by cost
        options = []
        for k in elig:
            for loc in allowed_locs:
                options.append((k, loc))
        options.sort(key=lambda kl: option_cost(t, kl[0], kl[1]))

        placed = False
        for k, loc in options:
            st = feasible_start(t, k, loc)
            if st is None:
                continue
            face = int(t["face_dur_min"])
            doc = int(t.get("doc_buffer_min", default_doc))
            _add_interval(th_cal[k], st, st + face + doc)
            if loc == "GYM":
                _add_interval(gym_cal, st, st + face + gym_turn)
            if bool(t.get("requires_assist", False)):
                _add_interval(assist_cal, st, st + face)
            start_by_task[tid] = st
            end_by_task[tid] = st + face

            resources_used = list(dict.fromkeys(t.get("required_resources", [])))
            if loc == "GYM" and gym_id not in resources_used:
                resources_used.append(gym_id)
            if bool(t.get("requires_assist", False)) and assist_id not in resources_used:
                resources_used.append(assist_id)

            assignments.append({
                "task_id": tid,
                "patient_id": t.get("patient_id"),
                "therapist_id": k,
                "start_min": st,
                "end_min": st + face,
                "location": {"type": loc, "detail": {}},
                "resources": resources_used,
                "status": "PLANNED",
                "locked": False,
                "explainability": {"reasons": ["greedy_earliest", f"cost={option_cost(t,k,loc)}"]},
            })
            placed = True
            break
        if not placed and must:
            unscheduled.append(tid)

    schedule = {
        "service_date": (tasks_sorted[0].get("service_date") if tasks_sorted else None),
        "mode": "GREEDY",
        "assignments": sorted(assignments, key=lambda a: (a["start_min"], a["therapist_id"], a["task_id"])),
        "unscheduled": unscheduled,
    }
    status = "FEASIBLE" if not unscheduled else "PARTIAL"
    return SolveResult(schedule=schedule, status=status)
