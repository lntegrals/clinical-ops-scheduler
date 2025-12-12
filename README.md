# PT Scheduler Shippable MVP (UI + Solver)

This repo gives you a **working UI** (Streamlit) that runs either:
- **Greedy** scheduler (fast baseline)
- **CP-SAT** scheduler (OR-Tools, higher-quality + harder constraints)

All times are **minutes from midnight**.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Sample data

The UI can run with `sample_input.json`.

## Optional: API server

```bash
uvicorn api.main:app --reload --port 8000
```

Then POST to:

`POST http://localhost:8000/v1/schedule/run`

Body:
```json
{"data": {...}, "mode":"CPSAT", "separation_mode":"start_to_start", "time_limit_s": 10}
```

## Notes / assumptions

- Session separation supports `start_to_start` (default) or `end_to_start`.
- Therapist occupancy includes `doc_buffer_min`.
- Gym occupancy includes `gym_turnover_min`.
- Gym/assist availability is respected if provided under `resources[].availability`.
