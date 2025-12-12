\
import json
import traceback
from datetime import datetime, date

import pandas as pd
import streamlit as st

from scheduler.utils import min_to_hhmm
from scheduler.cpsat import solve_cpsat
from scheduler.greedy import solve_greedy
from scheduler.validator import validate_schedule

st.set_page_config(page_title="PT Scheduler MVP", layout="wide")

st.title("Inpatient PT Scheduler — MVP (Greedy + CP-SAT)")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload input JSON (minutes-based)", type=["json"])
    use_sample = st.checkbox("Use sample_input.json", value=uploaded is None)

    st.header("Solver")
    mode = st.selectbox("Mode", ["CPSAT", "GREEDY"])
    separation_mode = st.selectbox("Session separation mode", ["start_to_start", "end_to_start"])
    time_limit = st.slider("CP-SAT time limit (seconds)", 2, 60, 10)
    st.caption("If OR-Tools isn't installed, CPSAT will fail; choose GREEDY.")

    st.header("Output")
    show_debug = st.checkbox("Show debug / raw JSON", value=False)

def load_data() -> dict:
    if uploaded is not None:
        return json.loads(uploaded.getvalue().decode("utf-8"))
    if use_sample:
        with open("sample_input.json", "r") as f:
            return json.load(f)
    return {}

data = load_data()
if not data:
    st.info("Upload an input JSON or toggle the sample.")
    st.stop()

def pretty_schedule(assignments, data):
    rows = []

    for a in assignments:
        rows.append({
            "Start": min_to_hhmm(a["start_min"]),
            "End": min_to_hhmm(a["end_min"]),
            "Therapist": a["therapist_id"],
            "Task": a["task_id"],
            "Location": a["location"]["type"],
        })

    # EMPTY STATE FIX
    if not rows:
        return pd.DataFrame(
            columns=["Start", "End", "Therapist", "Task", "Location"]
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["Start", "Therapist", "Task"])

run = st.button("Run scheduler", type="primary")

if run:
    try:
        if mode == "CPSAT":
            res = solve_cpsat(data, time_limit_s=int(time_limit), separation_mode=separation_mode)
            schedule = res.schedule
        else:
            res = solve_greedy(data, separation_mode=separation_mode)
            schedule = res.schedule

        violations = validate_schedule(data, schedule, separation_mode=separation_mode)
        schedule["validation"] = {
            "ok": len(violations) == 0,
            "violations": [v.__dict__ for v in violations],
        }

        st.subheader("Status")
        cols = st.columns(4)
        cols[0].metric("Mode", schedule.get("mode", mode))
        cols[1].metric("Assignments", len(schedule.get("assignments", [])))
        cols[2].metric("Unscheduled", len(schedule.get("unscheduled", [])))
        cols[3].metric("Validation OK", "YES" if schedule["validation"]["ok"] else "NO")

        st.subheader("Schedule")
        df = pretty_schedule(schedule.get("assignments", []), data)
        st.dataframe(df, use_container_width=True, height=360)

        if schedule.get("unscheduled"):
            st.subheader("Unscheduled tasks")
            st.write(schedule["unscheduled"])

        if not schedule["validation"]["ok"]:
            st.subheader("Violations")
            vdf = pd.DataFrame(schedule["validation"]["violations"])
            st.dataframe(vdf, use_container_width=True, height=240)

        # Downloads
        st.subheader("Export")
        json_bytes = json.dumps(schedule, indent=2).encode("utf-8")
        st.download_button("Download schedule JSON", data=json_bytes, file_name="schedule.json", mime="application/json")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download schedule CSV", data=csv_bytes, file_name="schedule.csv", mime="text/csv")

        if show_debug:
            st.subheader("Raw output")
            st.code(json.dumps(schedule, indent=2), language="json")

    except Exception as e:
        st.error("Scheduler failed.")
        st.code(traceback.format_exc())
else:
    st.caption("Click **Run scheduler** to produce a schedule.")
    st.dataframe(pretty_schedule([], data), use_container_width=True, height=200)
