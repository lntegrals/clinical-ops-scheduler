\
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Literal, Optional

from scheduler.cpsat import solve_cpsat
from scheduler.greedy import solve_greedy
from scheduler.validator import validate_schedule

app = FastAPI(title="PT Scheduler API", version="0.1.0")

class RunRequest(BaseModel):
    data: Dict[str, Any]
    mode: Literal["CPSAT","GREEDY"] = "CPSAT"
    separation_mode: Literal["start_to_start","end_to_start"] = "start_to_start"
    time_limit_s: int = 10

@app.post("/v1/schedule/run")
def run_schedule(req: RunRequest):
    if req.mode == "CPSAT":
        res = solve_cpsat(req.data, time_limit_s=req.time_limit_s, separation_mode=req.separation_mode)
        schedule = res.schedule
    else:
        res = solve_greedy(req.data, separation_mode=req.separation_mode)
        schedule = res.schedule

    violations = validate_schedule(req.data, schedule, separation_mode=req.separation_mode)
    schedule["validation"] = {
        "ok": len(violations) == 0,
        "violations": [v.__dict__ for v in violations],
    }
    return schedule
