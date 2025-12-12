from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from ortools.sat.python import cp_model

from .utils import DAY_MIN

Location = str  # "BEDSIDE" | "GYM"

@dataclass
class SolveResult:
    schedule: Dict[str, Any]
    objective_value: Optional[int]
    status: str


def _build_unavailable_intervals_from_availability(
    model: cp_model.CpModel,
    avail: List[Dict[str, int]],
    horizon: int = DAY_MIN,
    name_prefix: str = "unavail",
) -> List[cp_model.IntervalVar]:
    """Given availability windows, create fixed *unavailable* intervals for the complement."""
    if not avail:
        # unavailable = whole day
        return [model.NewIntervalVar(0, horizon, horizon, f"{name_prefix}[all]")]

    av = sorted([(int(w["start"]), int(w["end"])) for w in avail], key=lambda x: x[0])
    # clamp / merge
    merged: List[Tuple[int,int]] = []
    for a,b in av:
        a = max(0, min(horizon, a)); b = max(0, min(horizon, b))
        if b <= a:
            continue
        if not merged or a > merged[-1][1]:
            merged.append((a,b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    unavail: List[cp_model.IntervalVar] = []
    cur = 0
    for a,b in merged:
        if a > cur:
            unavail.append(model.NewIntervalVar(cur, a-cur, a, f"{name_prefix}[{cur}-{a}]"))
        cur = max(cur, b)
    if cur < horizon:
        unavail.append(model.NewIntervalVar(cur, horizon-cur, horizon, f"{name_prefix}[{cur}-{horizon}]"))
    return unavail


def solve_cpsat(
    data: Dict[str, Any],
    *,
    separation_mode: str = "start_to_start",  # or "end_to_start"
    time_limit_s: int = 10,
    log: bool = False,
) -> SolveResult:
    """Solve one-day scheduling using OR-Tools CP-SAT.

    Expected input keys:
      - therapists, patients, resources, tasks, separations
      - config, weights
    """
    config = data.get("config", {})
    weights = {**data.get("weights", {})}

    default_doc_buffer = int(config.get("default_doc_buffer_min", 10))
    gym_turnover = int(config.get("gym_turnover_min", 5))
    max_ot = int(config.get("max_overtime_min", 120))
    assist_cap_default = int(config.get("assist_pool_capacity", 1))

    priority_weight_map = config.get("priority_weight", {1:1,2:2,3:4,4:7,5:10})
    # allow string keys from JSON
    priority_weight = {int(k): int(v) for k,v in priority_weight_map.items()}

    W_late = int(weights.get("late", 1))
    W_ot = int(weights.get("overtime", 20))
    W_travel = int(weights.get("travel", 3))
    W_cont = int(weights.get("continuity", 5))
    W_churn = int(weights.get("churn", 2))
    W_ther_churn = int(weights.get("ther_churn", 10))
    W_gym_pref = int(weights.get("gym_pref", 2))

    therapists = data.get("therapists", [])
    patients = data.get("patients", [])
    resources = data.get("resources", [])
    tasks = data.get("tasks", [])
    separations = data.get("separations", [])
    locks = {lk["task_id"]: lk for lk in data.get("locks", [])}

    patient_by_id = {p["patient_id"]: p for p in patients}
    therapist_by_id = {t["therapist_id"]: t for t in therapists}

    # resource lookups
    res_by_id = {r["resource_id"]: r for r in resources}
    gym_resources = [r for r in resources if r.get("resource_type") == "GYM_ROOM"]
    assist_resources = [r for r in resources if r.get("resource_type") == "ASSIST_POOL"]
    equip_resources = [r for r in resources if r.get("resource_type") == "EQUIPMENT_SET"]

    gym_id = gym_resources[0]["resource_id"] if gym_resources else "GYM_ROOM_1"
    assist_id = assist_resources[0]["resource_id"] if assist_resources else "ASSIST_POOL"

    # model
    model = cp_model.CpModel()

    I = [t["task_id"] for t in tasks]
    K = [th["therapist_id"] for th in therapists]

    task_by_id = {t["task_id"]: t for t in tasks}

    # global vars per task
    s: Dict[str, cp_model.IntVar] = {}
    e: Dict[str, cp_model.IntVar] = {}
    wsel: Dict[Tuple[str,int], cp_model.BoolVar] = {}

    for i in I:
        t = task_by_id[i]
        s[i] = model.NewIntVar(0, DAY_MIN-1, f"s[{i}]")
        e[i] = model.NewIntVar(0, DAY_MIN, f"e[{i}]")
        face = int(t["face_dur_min"])
        model.Add(e[i] == s[i] + face)

        windows = t.get("allowed_windows", [])
        if not windows:
            # if missing, allow any time (MVP) but flag through validator; keep wide
            continue
        if len(windows) == 1:
            a = int(windows[0]["start"]); b = int(windows[0]["end"])
            model.Add(s[i] >= a)
            model.Add(e[i] <= b)
        else:
            for widx, w in enumerate(windows):
                wsel[(i,widx)] = model.NewBoolVar(f"wsel[{i},{widx}]")
                model.Add(s[i] >= int(w["start"])).OnlyEnforceIf(wsel[(i,widx)])
                model.Add(e[i] <= int(w["end"])).OnlyEnforceIf(wsel[(i,widx)])
            model.Add(sum(wsel[(i,widx)] for widx in range(len(windows))) == 1)

    # option vars + optional intervals
    x: Dict[Tuple[str,str,Location], cp_model.BoolVar] = {}
    I_th: Dict[Tuple[str,str,Location], cp_model.IntervalVar] = {}
    I_pat: Dict[Tuple[str,str,Location], cp_model.IntervalVar] = {}

    # helper: compute travel cost if not provided
    def travel_cost(i: str, k: str) -> int:
        t = task_by_id[i]
        # allow override in input
        by = t.get("travel_cost_by_therapist", {})
        if k in by:
            return int(by[k])
        pid = t.get("patient_id")
        p = patient_by_id.get(pid, {})
        punit = p.get("unit")
        home = therapist_by_id[k].get("home_unit")
        if home in (None, "", "UNKNOWN"):
            return 2
        if home == "FLOAT":
            return 1
        return 0 if (punit is not None and punit == home) else 1

    # Build eligible options
    for i in I:
        t = task_by_id[i]
        req_skills = set(t.get("required_skills", []))
        allowed_locs = set(t.get("allowed_locations", ["BEDSIDE"]))
        face = int(t["face_dur_min"])
        doc_buf = int(t.get("doc_buffer_min", default_doc_buffer))
        # Locks: if locked, we will later fix x and s
        for k in K:
            th = therapist_by_id[k]
            if not req_skills.issubset(set(th.get("skills", []))):
                continue
            for loc in allowed_locs:
                x[(i,k,loc)] = model.NewBoolVar(f"x[{i},{k},{loc}]")
                # shift bounds (for therapist-occupied interval)
                model.Add(s[i] >= int(th["shift_start"])).OnlyEnforceIf(x[(i,k,loc)])
                model.Add(s[i] + face + doc_buf <= int(th["shift_end"]) + max_ot).OnlyEnforceIf(x[(i,k,loc)])

                pat_size = face + (gym_turnover if loc == "GYM" else 0)
                th_size = face + doc_buf
                I_pat[(i,k,loc)] = model.NewOptionalIntervalVar(
                    s[i], pat_size, s[i] + pat_size, x[(i,k,loc)], f"Ipat[{i},{k},{loc}]"
                )
                I_th[(i,k,loc)] = model.NewOptionalIntervalVar(
                    s[i], th_size, s[i] + th_size, x[(i,k,loc)], f"Ith[{i},{k},{loc}]"
                )

    # C1 assignment constraints
    for i in I:
        t = task_by_id[i]
        readiness = t.get("readiness", "READY")
        must = bool(t.get("must_schedule", True)) and readiness == "READY"
        opts = [xv for (ii,kk,ll), xv in x.items() if ii == i]
        if not opts:
            # If there are no eligible options, CP-SAT will be infeasible for must tasks.
            # Keep as-is; status will show infeasible.
            continue
        if must:
            model.Add(sum(opts) == 1)
        else:
            model.Add(sum(opts) <= 1)

    # breaks + therapist NoOverlap
    for k in K:
        intervals: List[cp_model.IntervalVar] = []
        for (i2,k2,loc2), iv in I_th.items():
            if k2 == k:
                intervals.append(iv)
        # add breaks
        for br in therapist_by_id[k].get("breaks", []):
            bs = int(br["start"]); be = int(br["end"])
            if be > bs:
                intervals.append(model.NewIntervalVar(bs, be-bs, be, f"break[{k},{bs}-{be}]"))
        model.AddNoOverlap(intervals)

    # gym NoOverlap + gym unavailability
    gym_intervals: List[cp_model.IntervalVar] = []
    for (i2,k2,loc2), iv in I_pat.items():
        if loc2 == "GYM":
            gym_intervals.append(iv)

    # Add fixed unavailable intervals for gym (if known)
    gym_avail = []
    if gym_id in res_by_id:
        gym_avail = res_by_id[gym_id].get("availability", [])
    gym_unavail = _build_unavailable_intervals_from_availability(model, gym_avail, name_prefix=f"gym_unavail[{gym_id}]")
    model.AddNoOverlap(gym_intervals + gym_unavail)

    # assist pool cumulative + unavailability
    assist_cap = assist_cap_default
    if assist_id in res_by_id:
        assist_cap = int(res_by_id[assist_id].get("capacity", assist_cap_default))
    assist_intervals: List[cp_model.IntervalVar] = []
    assist_demands: List[int] = []
    y_assist: Dict[str, cp_model.BoolVar] = {}
    for i in I:
        t = task_by_id[i]
        if bool(t.get("requires_assist", False)):
            # y[i] == scheduled(i)
            y_assist[i] = model.NewBoolVar(f"y_assist[{i}]")
            scheduled_i = [xv for (ii,kk,ll), xv in x.items() if ii == i]
            if scheduled_i:
                model.Add(y_assist[i] == sum(scheduled_i))
            face = int(t["face_dur_min"])
            assist_intervals.append(
                model.NewOptionalIntervalVar(s[i], face, s[i] + face, y_assist[i], f"Iassist[{i}]")
            )
            assist_demands.append(1)

    if assist_intervals:
        # availability constraints via fixed unavailability intervals:
        assist_unavail = []
        if assist_id in res_by_id:
            assist_unavail = _build_unavailable_intervals_from_availability(
                model, res_by_id[assist_id].get("availability", []), name_prefix=f"assist_unavail[{assist_id}]"
            )
        # Enforce unavailability via NoOverlap too (capacity=1) and Cumulative for capacity>1.
        if assist_cap == 1:
            model.AddNoOverlap(assist_intervals + assist_unavail)
        else:
            model.AddCumulative(assist_intervals, assist_demands, assist_cap)
            # also forbid use during unavailable by adding unavail as demands=assist_cap (blocks all)
            if assist_unavail:
                block_demands = [assist_cap for _ in assist_unavail]
                model.AddCumulative(assist_intervals + assist_unavail, assist_demands + block_demands, assist_cap)

    # equipment sets cumulative (optional)
    for res in equip_resources:
        rid = res["resource_id"]
        cap = int(res.get("capacity", 1))
        avail = res.get("availability", [])
        intervals: List[cp_model.IntervalVar] = []
        demands: List[int] = []
        for i in I:
            t = task_by_id[i]
            if rid in set(t.get("required_resources", [])):
                y = model.NewBoolVar(f"y_eq[{rid}][{i}]")
                scheduled_i = [xv for (ii,kk,ll), xv in x.items() if ii == i]
                if scheduled_i:
                    model.Add(y == sum(scheduled_i))
                face = int(t["face_dur_min"])
                intervals.append(model.NewOptionalIntervalVar(s[i], face, s[i] + face, y, f"Ieq[{rid}][{i}]")) 
                demands.append(1)
        if intervals:
            unavail = _build_unavailable_intervals_from_availability(model, avail, name_prefix=f"eq_unavail[{rid}]") if avail else []
            if cap == 1:
                model.AddNoOverlap(intervals + unavail)
            else:
                model.AddCumulative(intervals, demands, cap)
                if unavail:
                    block_demands = [cap for _ in unavail]
                    model.AddCumulative(intervals + unavail, demands + block_demands, cap)

    # Session separations
    for sep in separations:
        a = sep["task_a"]; b = sep["task_b"]
        gap = int(sep.get("min_gap_min", 0))
        if a not in s or b not in s:
            continue
        if separation_mode == "end_to_start":
            model.Add(s[b] >= e[a] + gap)
        else:
            model.Add(s[b] >= s[a] + gap)

    # Locks / freezes (if provided)
    for tid, lk in locks.items():
        if tid not in I:
            continue
        if not lk.get("locked", True):
            continue
        k_star = lk.get("therapist_id")
        start_star = lk.get("start_min")
        loc_star = lk.get("location", lk.get("location_type", None))
        if start_star is not None:
            model.Add(s[tid] == int(start_star))
        if k_star and loc_star:
            # force x[tid,k_star,loc_star] = 1 and others 0
            for (ii,kk,ll), xv in x.items():
                if ii != tid:
                    continue
                if kk == k_star and ll == loc_star:
                    model.Add(xv == 1)
                else:
                    model.Add(xv == 0)

    # Objective terms
    obj_terms: List[cp_model.LinearExpr] = []

    # lateness
    late: Dict[str, cp_model.IntVar] = {}
    for i in I:
        t = task_by_id[i]
        pref = t.get("preferred_time")
        if pref is None:
            windows = t.get("allowed_windows", [])
            pref = int(min(w["start"] for w in windows)) if windows else 0
        late[i] = model.NewIntVar(0, DAY_MIN, f"late[{i}]")
        model.Add(late[i] >= s[i] - int(pref))
        model.Add(late[i] >= 0)
        pr = int(t.get("priority", 3))
        pw = priority_weight.get(pr, 1)
        obj_terms.append(W_late * pw * late[i])

    # overtime
    ot: Dict[str, cp_model.IntVar] = {}
    end_k: Dict[str, cp_model.IntVar] = {}
    for k in K:
        th = therapist_by_id[k]
        end_k[k] = model.NewIntVar(0, DAY_MIN + max_ot, f"end_k[{k}]")
        model.Add(end_k[k] >= int(th["shift_start"]))
        for (i2,k2,loc2), xv in x.items():
            if k2 != k:
                continue
            t2 = task_by_id[i2]
            buf = int(t2.get("doc_buffer_min", default_doc_buffer))
            model.Add(end_k[k] >= s[i2] + int(t2["face_dur_min"]) + buf).OnlyEnforceIf(xv)
        ot[k] = model.NewIntVar(0, max_ot, f"ot[{k}]")
        model.Add(ot[k] >= end_k[k] - int(th["shift_end"]))
        model.Add(ot[k] >= 0)
        obj_terms.append(W_ot * ot[k])

    # travel approximation
    for (i2,k2,loc2), xv in x.items():
        cost = travel_cost(i2, k2)
        if cost:
            obj_terms.append(W_travel * cost * xv)

    # continuity (if prev_therapist_id present)
    for i in I:
        t = task_by_id[i]
        prev_k = t.get("prev_therapist_id") or t.get("prev_therapist")
        if not prev_k or prev_k not in K:
            continue
        is_prev = []
        for loc in set(t.get("allowed_locations", ["BEDSIDE"])):
            key = (i, prev_k, loc)
            if key in x:
                is_prev.append(x[key])
        if not is_prev:
            continue
        cont_break = model.NewBoolVar(f"cont_break[{i}]")
        model.Add(cont_break >= 1 - sum(is_prev))
        obj_terms.append(W_cont * cont_break)

    # churn (if planned_start_min) + therapist churn (planned_therapist_id)
    for i in I:
        t = task_by_id[i]
        planned = t.get("planned_start_min")
        if planned is not None:
            d = model.NewIntVar(0, DAY_MIN, f"churn_d[{i}]")
            model.Add(d >= s[i] - int(planned))
            model.Add(d >= int(planned) - s[i])
            obj_terms.append(W_churn * d)
        planned_k = t.get("planned_therapist_id")
        if planned_k and planned_k in K:
            is_planned = []
            for loc in set(t.get("allowed_locations", ["BEDSIDE"])):
                key = (i, planned_k, loc)
                if key in x:
                    is_planned.append(x[key])
            if is_planned:
                th_churn = model.NewBoolVar(f"ther_churn[{i}]")
                model.Add(th_churn >= 1 - sum(is_planned))
                obj_terms.append(W_ther_churn * th_churn)

    # gym preference
    for (i2,k2,loc2), xv in x.items():
        t = task_by_id[i2]
        prefers_gym = bool(t.get("prefers_gym", False))
        if prefers_gym and loc2 == "BEDSIDE" and "GYM" in set(t.get("allowed_locations", [])):
            obj_terms.append(W_gym_pref * xv)

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = bool(log)

    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)

    if status not in ("OPTIMAL", "FEASIBLE"):
        return SolveResult(schedule={"assignments": [], "unscheduled": I}, objective_value=None, status=status)

    # Extract solution
    assignments = []
    for i in I:
        chosen = None
        for (ii,kk,ll), xv in x.items():
            if ii != i:
                continue
            if solver.Value(xv) == 1:
                chosen = (kk,ll)
                break
        if not chosen:
            continue
        k, loc = chosen
        t = task_by_id[i]
        start = int(solver.Value(s[i]))
        end = start + int(t["face_dur_min"])
        pid = t.get("patient_id")
        p = patient_by_id.get(pid, {})
        resources_used = list(dict.fromkeys(t.get("required_resources", [])))  # de-dup, preserve order
        # ensure gym resource if scheduled in gym
        if loc == "GYM" and gym_id not in resources_used:
            resources_used.append(gym_id)
        if t.get("requires_assist", False) and assist_id not in resources_used:
            resources_used.append(assist_id)

        explain = {
            "reasons": [
                f"assigned:{k}",
                f"loc:{loc}",
                f"priority:{t.get('priority', 3)}",
            ],
            "scores": {
                "late": int(solver.Value(late[i])) if i in late else None,
                "travel": travel_cost(i, k),
            },
        }
        assignments.append({
            "task_id": i,
            "patient_id": pid,
            "therapist_id": k,
            "start_min": start,
            "end_min": end,
            "location": {"type": loc, "detail": {"unit": p.get("unit"), "room": p.get("room"), "resource_id": gym_id if loc=="GYM" else None}},
            "resources": resources_used,
            "status": "PLANNED",
            "locked": bool(locks.get(i, {}).get("locked", False)),
            "explainability": explain,
        })

    obj_val = int(solver.ObjectiveValue())
    schedule = {
        "service_date": (tasks[0].get("service_date") if tasks else None),
        "mode": "CPSAT",
        "objective_value": obj_val,
        "assignments": sorted(assignments, key=lambda a: (a["start_min"], a["therapist_id"], a["task_id"])),
        "unscheduled": [],
    }
    return SolveResult(schedule=schedule, objective_value=obj_val, status=status)
