#!/usr/bin/env python3
import os
import sqlite3
import json
from datetime import datetime, date, timedelta
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import requests
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
from collections import defaultdict
import math
import statistics
import io
import csv

DB_FILE = "shift_backend.db"
SHIFT_HOURS = 8
MAX_HOURS_PER_WEEK = 40

# Business assumption: how many "covers" (customers) one staff member can serve in one shift.
# This is configurable here; you can tune it for your restaurants.
COVERS_PER_EMP = 10

PAYPAL_CLIENT = os.getenv("PAYPAL_CLIENT")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET")
PAYPAL_API = "https://api-m.sandbox.paypal.com"  # sandbox

app = FastAPI(title="Shift Scheduler Backend with Advanced Logic")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS restaurants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            planning_days INTEGER DEFAULT 7,
            paid_until TEXT
        );

        CREATE TABLE IF NOT EXISTS shifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id INTEGER,
            name TEXT,
            start_time TEXT,
            end_time TEXT
        );

        -- employees table now has 'skills' column (JSON array stored as TEXT)
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id INTEGER,
            name TEXT,
            skills TEXT
        );

        CREATE TABLE IF NOT EXISTS availability (
            employee_id INTEGER,
            day INTEGER,
            shift_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS load (
            restaurant_id INTEGER,
            day INTEGER,
            shift_id INTEGER,
            required INTEGER
        );

        CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id INTEGER,
            days INTEGER,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS schedule_items (
            schedule_id INTEGER,
            employee_id INTEGER,
            day INTEGER,
            shift_id INTEGER
        );

        -- New tables to store richer historical / POS / holiday data (additive, does not affect existing endpoints)
        CREATE TABLE IF NOT EXISTS sales_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id INTEGER,
            date TEXT,          -- ISO date (YYYY-MM-DD)
            shift_id INTEGER,
            covers INTEGER,     -- number of customers served during that shift (POS metric)
            revenue REAL        -- optional revenue metric
        );

        CREATE TABLE IF NOT EXISTS holidays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,          -- ISO date (YYYY-MM-DD)
            name TEXT
        );
        """)
    # For existing DBs created before this change, attempt to add skills column if missing
    try:
        with db() as c:
            # This will fail if column already exists; ignore exceptions
            c.execute("ALTER TABLE employees ADD COLUMN skills TEXT")
    except Exception:
        pass

    # For existing DBs created before this change, attempt to add start_time and end_time columns to shifts
    try:
        with db() as c:
            c.execute("ALTER TABLE shifts ADD COLUMN start_time TEXT")
    except Exception:
        pass
    try:
        with db() as c:
            c.execute("ALTER TABLE shifts ADD COLUMN end_time TEXT")
    except Exception:
        pass

    # For existing DBs created before this change, attempt to add required_skills column to shifts
    try:
        with db() as c:
            c.execute("ALTER TABLE shifts ADD COLUMN required_skills TEXT")
    except Exception:
        pass

init_db()

# ================= MODELS =================
class RestaurantCreate(BaseModel):
    name: str
    planning_days: Optional[int] = 7

class EmployeeCreate(BaseModel):
    name: str
    skills: Optional[List[str]] = None  # new optional field; list of skill tags/strings

class ShiftCreate(BaseModel):
    name: str
    start_time: Optional[str] = None  # "HH:MM" optional
    end_time: Optional[str] = None    # "HH:MM" optional
    # optional mapping skill -> minimum required count for the shift (e.g. {"cook":1,"senior":1})
    required_skills: Optional[Dict[str, int]] = None

class AvailabilityCreate(BaseModel):
    employee_id: int
    day: int
    shift_id: int

class LoadCreate(BaseModel):
    day: int
    shift_id: int
    required: int

# Models for POS / historical uploads
class POSRecord(BaseModel):
    date: str            # "YYYY-MM-DD"
    shift_id: int
    covers: int
    revenue: Optional[float] = None

class HolidayRecord(BaseModel):
    date: str            # "YYYY-MM-DD"
    name: Optional[str] = None

# New minimal model for payment legal acceptance (strict booleans)
class PaymentConsent(BaseModel):
    accept_terms: bool
    accept_privacy: bool
    amount: Optional[float] = None   # optional amount in USD
    months: Optional[int] = None    # optional months to subscribe for

# ================= PAYPAL =================
def get_access_token():
    if not PAYPAL_CLIENT or not PAYPAL_SECRET:
        raise HTTPException(status_code=500, detail="PAYPAL_CLIENT or PAYPAL_SECRET not configured")
    r = requests.post(
        f"{PAYPAL_API}/v1/oauth2/token",
        auth=(PAYPAL_CLIENT, PAYPAL_SECRET),
        data={"grant_type": "client_credentials"}
    )
    r.raise_for_status()
    return r.json()["access_token"]

def create_paypal_order(amount="10.00", currency="USD", restaurant_id=None, months: int = 1):
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # Ensure amount formatting is valid string
    try:
        amt = float(amount)
    except Exception:
        try:
            amt = float(str(amount))
        except Exception:
            amt = 10.00
    amt_str = f"{amt:.2f}"
    # store restaurant_id and months in custom_id so webhook can set correct duration
    custom = f"{restaurant_id}:{months}"
    data = {
        "intent": "CAPTURE",
        "purchase_units": [{"amount": {"currency_code": currency, "value": amt_str}, "custom_id": custom}]
    }
    r = requests.post(f"{PAYPAL_API}/v2/checkout/orders", json=data, headers=headers)
    r.raise_for_status()
    return r.json()

# Helper: ensure subscription active for restaurant (used to strictly block paid features)
def ensure_subscription_active(restaurant_id: int):
    c = db().cursor()
    r = c.execute("SELECT paid_until FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    paid_until = r["paid_until"]
    if not paid_until:
        raise HTTPException(status_code=403, detail="Subscription required")
    try:
        if datetime.utcnow() > datetime.fromisoformat(paid_until):
            raise HTTPException(status_code=403, detail="Subscription expired")
    except Exception:
        # if format issue, treat as expired
        raise HTTPException(status_code=403, detail="Subscription required")

# ================= REST API (unchanged endpoints preserved) =================
@app.post("/restaurants/")
def create_restaurant(r: RestaurantCreate):
    with db() as c:
        c.execute(
            "INSERT INTO restaurants (name, planning_days, paid_until) VALUES (?, ?, ?)",
            (r.name, r.planning_days, None)
        )
        rid = c.lastrowid

        # Auto-fill default shifts and load entries using the real inserted shift IDs
        default_shifts = ["Утро", "Вечер"]
        created_shift_ids = []
        for s in default_shifts:
            c.execute("INSERT INTO shifts (restaurant_id, name) VALUES (?, ?)", (rid, s))
            created_shift_ids.append(c.lastrowid)
        # If planning days > 0, populate the load table for the created shifts
        for d in range(r.planning_days):
            for s_id in created_shift_ids:
                c.execute("INSERT INTO load (restaurant_id, day, shift_id, required) VALUES (?, ?, ?, ?)",
                          (rid, d, s_id, 1))
    return {"id": rid, "name": r.name, "planning_days": r.planning_days, "default_shift_ids": created_shift_ids}

@app.get("/restaurants/")
def list_restaurants():
    c = db().cursor()
    rows = c.execute("SELECT id, name, planning_days, paid_until FROM restaurants").fetchall()
    return [{"id": r["id"], "name": r["name"], "planning_days": r["planning_days"], "paid_until": r["paid_until"]} for r in rows]

@app.get("/restaurants/{restaurant_id}/info/")
def restaurant_info(restaurant_id: int):
    c = db().cursor()
    r = c.execute("SELECT id, name, planning_days, paid_until FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    paid_until = r["paid_until"]
    has_access = False
    if paid_until:
        try:
            has_access = datetime.utcnow() <= datetime.fromisoformat(paid_until)
        except Exception:
            has_access = False
    return {"id": r["id"], "name": r["name"], "planning_days": r["planning_days"], "paid_until": paid_until, "has_access": has_access}

@app.get("/restaurants/{restaurant_id}/employees/")
def list_employees(restaurant_id: int):
    c = db().cursor()
    rows = c.execute("SELECT id, name, skills FROM employees WHERE restaurant_id=?", (restaurant_id,)).fetchall()
    out = []
    for r in rows:
        skills = []
        if r["skills"]:
            try:
                skills = json.loads(r["skills"])
            except Exception:
                # legacy CSV fallback
                skills = [s.strip() for s in (r["skills"] or "").split(",") if s.strip()]
        out.append({"id": r["id"], "name": r["name"], "skills": skills})
    return out

@app.get("/restaurants/{restaurant_id}/shifts/")
def list_shifts(restaurant_id: int):
    c = db().cursor()
    rows = c.execute("SELECT id, name, start_time, end_time, required_skills FROM shifts WHERE restaurant_id=?", (restaurant_id,)).fetchall()
    out = []
    for r in rows:
        req_sk = {}
        if r["required_skills"]:
            try:
                req_sk = json.loads(r["required_skills"])
            except Exception:
                # legacy: maybe semicolon separated tokens -> treat as keys with count=1
                tokens = [t.strip() for t in (r["required_skills"] or "").split(";") if t.strip()]
                for t in tokens:
                    req_sk[t] = req_sk.get(t, 0) + 1
        out.append({"id": r["id"], "name": r["name"], "start_time": r["start_time"], "end_time": r["end_time"], "required_skills": req_sk})
    return out

@app.post("/restaurants/{restaurant_id}/employees/")
def add_employee(restaurant_id: int, e: EmployeeCreate):
    with db() as c:
        # basic validation: restaurant exists
        r = c.execute("SELECT id FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        skills_json = None
        if e.skills:
            # store as JSON array
            try:
                skills_json = json.dumps(e.skills)
            except Exception:
                # fallback to CSV string if JSON fails
                skills_json = ",".join(e.skills)
        c.execute("INSERT INTO employees (restaurant_id, name, skills) VALUES (?, ?, ?)", (restaurant_id, e.name, skills_json))
        eid = c.lastrowid
    return {"id": eid, "name": e.name, "restaurant_id": restaurant_id, "skills": e.skills or []}

@app.post("/restaurants/{restaurant_id}/shifts/")
def add_shift(restaurant_id: int, s: ShiftCreate):
    with db() as c:
        r = c.execute("SELECT id FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        required_skills_json = None
        if s.required_skills:
            try:
                required_skills_json = json.dumps(s.required_skills)
            except Exception:
                required_skills_json = None
        c.execute("INSERT INTO shifts (restaurant_id, name, start_time, end_time, required_skills) VALUES (?, ?, ?, ?, ?)",
                  (restaurant_id, s.name, s.start_time, s.end_time, required_skills_json))
        sid = c.lastrowid
    return {"id": sid, "name": s.name, "restaurant_id": restaurant_id, "start_time": s.start_time, "end_time": s.end_time, "required_skills": s.required_skills or {}}

@app.post("/availability/")
def set_availability(a: AvailabilityCreate):
    with db() as c:
        # Validate employee exists
        emp = c.execute("SELECT restaurant_id FROM employees WHERE id=?", (a.employee_id,)).fetchone()
        if not emp:
            raise HTTPException(status_code=404, detail="Employee not found")
        # Validate shift exists
        sh = c.execute("SELECT restaurant_id FROM shifts WHERE id=?", (a.shift_id,)).fetchone()
        if not sh:
            raise HTTPException(status_code=404, detail="Shift not found")
        # Ensure employee and shift belong to same restaurant
        if emp["restaurant_id"] != sh["restaurant_id"]:
            raise HTTPException(status_code=400, detail="Employee and shift belong to different restaurants")
        # Prevent duplicate availability rows
        exists = c.execute("SELECT 1 FROM availability WHERE employee_id=? AND day=? AND shift_id=?",
                           (a.employee_id, a.day, a.shift_id)).fetchone()
        if exists:
            return {"status": "ok", "note": "already exists"}
        c.execute("INSERT INTO availability (employee_id, day, shift_id) VALUES (?, ?, ?)",
                  (a.employee_id, a.day, a.shift_id))
    return {"status": "ok"}

@app.get("/restaurants/{restaurant_id}/availability/")
def get_availability(restaurant_id: int):
    c = db().cursor()
    rows = c.execute("""
        SELECT a.employee_id, a.day, a.shift_id
        FROM availability a
        JOIN employees e ON e.id = a.employee_id
        WHERE e.restaurant_id=?
    """, (restaurant_id,)).fetchall()
    return [{"employee_id": r["employee_id"], "day": r["day"], "shift_id": r["shift_id"]} for r in rows]

@app.post("/load/{restaurant_id}/")
def set_load(restaurant_id: int, loads: List[LoadCreate]):
    with db() as c:
        # basic validation
        r = c.execute("SELECT id FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        for l in loads:
            # Upsert: remove existing entry for that restaurant/day/shift and insert new required
            c.execute("DELETE FROM load WHERE restaurant_id=? AND day=? AND shift_id=?",
                      (restaurant_id, l.day, l.shift_id))
            c.execute("INSERT INTO load (restaurant_id, day, shift_id, required) VALUES (?, ?, ?, ?)",
                      (restaurant_id, l.day, l.shift_id, l.required))
    return {"status": "ok"}

# ================= PAYPAL ENDPOINT =================
@app.post("/restaurants/{restaurant_id}/pay/")
def pay_subscription(restaurant_id: int, consent: PaymentConsent):
    # Validate legal acceptance strictly
    if not (consent.accept_terms and consent.accept_privacy):
        raise HTTPException(status_code=400, detail="Legal acceptance required: accept_terms and accept_privacy must be true")

    # Ensure restaurant exists
    with db() as c:
        r = c.execute("SELECT id FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")

    # Determine amount and months; fall back to 1 month / $30.00 to preserve previous behavior
    months = consent.months if consent.months and consent.months > 0 else 1
    if consent.amount is not None:
        try:
            amount = float(consent.amount)
        except Exception:
            amount = 30.00
    else:
        # default pricing: 1 month = 30, 2 months = 55, 3 months = 80 (matches frontend choices)
        if months == 1:
            amount = 30.00
        elif months == 2:
            amount = 55.00
        elif months == 3:
            amount = 80.00
        else:
            # generic: $30 per month fallback
            amount = 30.00 * months

    order = create_paypal_order(amount=amount, restaurant_id=restaurant_id, months=months)
    for link in order.get("links", []):
        if link.get("rel") == "approve":
            return {"approval_url": link["href"]}
    raise HTTPException(status_code=500, detail="No approval URL returned")

@app.post("/paypal/webhook/")
async def paypal_webhook(request: Request):
    event = await request.json()
    if event.get("event_type") == "CHECKOUT.ORDER.APPROVED":
        # custom_id format: "<restaurant_id>:<months>"
        custom = ""
        try:
            custom = event["resource"].get("custom_id", "")
        except Exception:
            custom = ""
        parts = str(custom).split(":") if custom else []
        restaurant_id = None
        if parts:
            try:
                restaurant_id = int(parts[0])
            except Exception:
                restaurant_id = None
        if restaurant_id is None:
            # fallback: attempt to parse resource.payee or other fields (best-effort)
            try:
                restaurant_id = int(event["resource"].get("custom_id", 0))
            except Exception:
                # cannot determine restaurant id; ignore
                return {"status": "ignored"}

        months = 1
        if len(parts) >= 2:
            try:
                months = max(1, int(parts[1]))
            except Exception:
                months = 1

        paid_until = (datetime.utcnow() + timedelta(days=30 * months)).isoformat()
        with db() as c:
            c.execute("UPDATE restaurants SET paid_until=? WHERE id=?", (paid_until, restaurant_id))
    return {"status": "ok"}

# ================= SCHEDULE GENERATION (skill-aware) =================
@app.post("/restaurants/{restaurant_id}/generate_schedule/")
def generate_schedule(restaurant_id: int):
    c = db().cursor()
    r = c.execute("SELECT planning_days, paid_until FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    days, paid_until = r["planning_days"], r["paid_until"]
    if paid_until and datetime.utcnow() > datetime.fromisoformat(paid_until):
        raise HTTPException(status_code=403, detail="Subscription expired")

    # Fetch shifts with potential required_skills
    shift_rows = c.execute("SELECT id, required_skills FROM shifts WHERE restaurant_id=?", (restaurant_id,)).fetchall()
    shifts = [s["id"] for s in shift_rows]
    shift_req_map = {}
    for s in shift_rows:
        rs = {}
        if s["required_skills"]:
            try:
                rs = json.loads(s["required_skills"])
            except Exception:
                rs = {}
        # normalize counts to ints
        for k in list(rs.keys()):
            try:
                rs[k] = int(rs[k])
            except Exception:
                rs[k] = 1
        shift_req_map[s["id"]] = rs

    employees_rows = c.execute("SELECT id, skills FROM employees WHERE restaurant_id=?", (restaurant_id,)).fetchall()
    employees = [e["id"] for e in employees_rows]
    # build employee -> skills set map
    emp_skills = {}
    for e in employees_rows:
        skills = []
        if e["skills"]:
            try:
                skills = json.loads(e["skills"])
            except Exception:
                skills = [s.strip() for s in (e["skills"] or "").split(",") if s.strip()]
        emp_skills[e["id"]] = set(s.lower() for s in skills if s)

    if not employees:
        raise HTTPException(status_code=400, detail="No employees to schedule")
    if not shifts:
        raise HTTPException(status_code=400, detail="No shifts defined for restaurant")

    availability_rows = c.execute("SELECT employee_id, day, shift_id FROM availability").fetchall()
    availability = set()
    for row in availability_rows:
        # only keep if employee is from this restaurant
        emp_rest = c.execute("SELECT restaurant_id FROM employees WHERE id=?", (row["employee_id"],)).fetchone()
        if emp_rest and emp_rest["restaurant_id"] == restaurant_id:
            availability.add((row["employee_id"], row["day"], row["shift_id"]))

    load_rows = c.execute("SELECT day, shift_id, required FROM load WHERE restaurant_id=?", (restaurant_id,)).fetchall()

    prob = LpProblem("Scheduling", LpMinimize)

    # ────────────────────────────────────────────────
    # Добавляем балансировку смен — мягкий штраф (как раньше)
    # ────────────────────────────────────────────────

    # 1. Сколько всего смен реально нужно покрыть (сумма required)
    total_required = sum(lr["required"] for lr in load_rows if lr["day"] < days)

    # 2. Примерное "справедливое" количество смен на человека
    employee_count = len(employees)
    if employee_count > 0:
        fair_share = total_required / employee_count
    else:
        fair_share = 0

    # 3. Decision variables: x for assignments (declare BEFORE referencing)
    x = {(e,d,s): LpVariable(f"x_{e}_{d}_{s}", 0,1,cat="Binary")
         for e in employees for d in range(days) for s in shifts}

    # 4. Переменные для количества смен каждого сотрудника (expressions using x)
    shifts_per_employee = {
        e: lpSum(x[(e, d, s)] for d in range(days) for s in shifts)
        for e in employees
    }

    # 5. Переменные отклонения (positive & negative) — стандартный приём
    deviation_pos = {e: LpVariable(f"dev_pos_{e}", 0) for e in employees}
    deviation_neg = {e: LpVariable(f"dev_neg_{e}", 0) for e in employees}

    # 6. Связываем отклонения с реальным количеством смен
    for e in employees:
        prob += shifts_per_employee[e] - fair_share == deviation_pos[e] - deviation_neg[e]

    # 7. Добавляем в целевую функцию штраф за любое отклонение и штраф за несоответствие скиллов
    BALANCE_PENALTY = 0.25
    SKILL_MISMATCH_PENALTY = 0.35  # adjust strength: higher -> prefer matching more

    objective_terms = []
    for (e,d,s), var in x.items():
        penalty = 0.0
        reqs = shift_req_map.get(s, {})
        if reqs:
            # if shift expects any skills, check if employee has ANY of those skills
            required_keys = [k.lower() for k in reqs.keys()]
            has_any = any((rk in emp_skills.get(e, set())) for rk in required_keys)
            if not has_any:
                penalty = SKILL_MISMATCH_PENALTY
        objective_terms.append(var * (1.0 + penalty))

    prob += lpSum(objective_terms) + BALANCE_PENALTY * lpSum(
        deviation_pos[e] + deviation_neg[e] for e in employees
    )

    # Load coverage constraints (required staff per day+shift)
    for lr in load_rows:
        d = lr["day"]
        s = lr["shift_id"]
        req = lr["required"]
        if d < days and s in shifts:
            prob += lpSum(x[(e,d,s)] for e in employees) >= req

    # skill-specific minimum constraints per shift (hard constraints when possible)
    for d in range(days):
        for s in shifts:
            reqs = shift_req_map.get(s, {})
            if not reqs:
                continue
            for skill, minc in reqs.items():
                skill_lower = skill.lower()
                # employees that have this skill
                eligible = [e for e in employees if skill_lower in emp_skills.get(e, set())]
                # If no eligible employees exist across whole staff, this constraint would make problem infeasible.
                # We add a relaxed approach: if eligible exist, enforce hard min; otherwise skip and rely on objective penalty.
                if not eligible:
                    continue
                prob += lpSum(x[(e,d,s)] for e in eligible) >= minc

    # availability & per-day and weekly hour constraints
    for (e,d,s),v in x.items():
        if (e,d,s) not in availability:
            prob += v == 0

    for e in employees:
        for d in range(days):
            prob += lpSum(x[(e,d,s)] for s in shifts) <= 1
        prob += lpSum(x[(e,d,s)]*SHIFT_HOURS for d in range(days) for s in shifts) <= MAX_HOURS_PER_WEEK

    prob.solve()
    if LpStatus[prob.status] != "Optimal":
        raise HTTPException(status_code=400, detail="No feasible schedule")

    with db() as c:
        c.execute("INSERT INTO schedules (restaurant_id, days, created_at) VALUES (?, ?, ?)",
                  (restaurant_id, days, datetime.utcnow().isoformat()))
        sid = c.lastrowid
        for (e,d,s),v in x.items():
            try:
                vv = value(v)
            except Exception:
                vv = 0
            if vv == 1:
                c.execute("INSERT INTO schedule_items (schedule_id, employee_id, day, shift_id) VALUES (?, ?, ?, ?)",
                          (sid,e,d,s))
    return {"schedule_id": sid, "status": "generated"}

@app.get("/schedules/{schedule_id}/")
def get_schedule(schedule_id: int):
    """
    Returns schedule items enriched with advisory fields expected by the frontend:
      - forecast (string)
      - reasons (list of strings)
      - confidence (float 0..1)
      - advisory (bool)
      - overlap_warnings (list)
      - skill_match (bool)
      - schedule_index (int) for frontend Apply action
    This endpoint preserves original stored schedule items and augments them with advisory metadata
    produced from load table, availability and AI forecast (if historical POS exists).
    """
    conn = db()
    c = conn.cursor()

    srow = c.execute("SELECT restaurant_id, days FROM schedules WHERE id=?", (schedule_id,)).fetchone()
    if not srow:
        raise HTTPException(status_code=404, detail="Schedule not found")
    restaurant_id = srow["restaurant_id"]
    days = srow["days"]

    # fetch raw items (including employee skills)
    rows = c.execute("""
        SELECT si.day, si.shift_id, sh.name AS shift_name, emp.name AS employee_name, emp.id AS employee_id, emp.skills AS skills
        FROM schedule_items si
        JOIN employees emp ON emp.id = si.employee_id
        JOIN shifts sh ON sh.id = si.shift_id
        WHERE si.schedule_id=?
        ORDER BY si.day, si.shift_id
    """, (schedule_id,)).fetchall()

    items = []
    # build overlap info (count assignments per employee per day)
    overlap_counts = defaultdict(int)
    for r in rows:
        overlap_counts[(r["employee_id"], r["day"])] += 1

    # get forecast predictions for restaurant (use ai_forecast_load logic)
    try:
        forecast_resp = ai_forecast_load(restaurant_id)  # function defined below
        forecast_list = forecast_resp.get("predictions", [])
    except HTTPException:
        forecast_list = []

    # build a map for quick lookup
    forecast_map = {(p["day"], p["shift_id"]): p for p in forecast_list}

    # gather load map
    load_rows = c.execute("SELECT day, shift_id, required FROM load WHERE restaurant_id=?", (restaurant_id,)).fetchall()
    load_map = {(lr["day"], lr["shift_id"]): lr["required"] for lr in load_rows}

    # availability map for checking if assigned employee was available
    avail_rows = c.execute("SELECT employee_id, day, shift_id FROM availability").fetchall()
    avail_set = set((ar["employee_id"], ar["day"], ar["shift_id"]) for ar in avail_rows)

    for idx, r in enumerate(rows):
        day = r["day"]
        sid = r["shift_id"]
        shift_name = r["shift_name"]
        employee_name = r["employee_name"]
        eid = r["employee_id"]

        # parse skills (JSON or CSV fallback)
        skills = []
        if r["skills"]:
            try:
                skills = json.loads(r["skills"])
            except Exception:
                skills = [s.strip() for s in (r["skills"] or "").split(",") if s.strip()]

        # forecast data
        f = forecast_map.get((day, sid))
        reasons = []
        forecast_str = ""
        predicted_covers = None
        recommended = None
        if f:
            predicted_covers = f.get("predicted_covers")
            recommended = f.get("recommended")
            forecast_str = f"Predicted covers ~{predicted_covers} -> recommend {recommended} staff"
            reasons.append(f"Forecast: predicted covers {predicted_covers} -> recommended {recommended}")
        else:
            # fallback to load table or default
            req = load_map.get((day, sid))
            if req is not None:
                forecast_str = f"Load table requires {req} staff"
                reasons.append(f"Load requires {req}")
            else:
                forecast_str = "No forecast available"
                reasons.append("No POS forecast or load")

        # availability reason
        if (eid, day, sid) in avail_set:
            reasons.append("Employee available for this shift.")
            has_availability = True
        else:
            reasons.append("Employee was not explicitly marked available for this shift.")
            has_availability = False

        # required value
        req = load_map.get((day, sid), None)

        # overlap warnings
        overlap_count = overlap_counts.get((eid, day), 0)
        overlap_warnings = []
        if overlap_count > 1:
            overlap_warnings.append(f"Employee assigned to {overlap_count} shifts on day {day+1}")

        # skill_match: True if any skill token matches shift name (case-insensitive substring)
        skill_match = False
        if skills and shift_name:
            low_shift = shift_name.lower()
            for sk in skills:
                if not sk:
                    continue
                if sk.lower() in low_shift:
                    skill_match = True
                    break

        if skill_match:
            reasons.append("Skill match for shift.")
        else:
            reasons.append("No explicit skill match found.")

        # compute confidence:
        # base confidence depends on availability and match between recommended and required
        conf = 0.5
        if has_availability:
            conf = 0.8
        else:
            conf = 0.55
        if recommended is not None and req is not None:
            # if recommended >= req, increase confidence
            if recommended >= req:
                conf += 0.1
            else:
                conf -= 0.05
            reasons.append(f"Recommended {recommended} vs required {req}")
        elif recommended is not None:
            # only recommended present
            conf += 0.05

        # reward skill match
        if skill_match:
            conf += 0.10
            reasons.append("Skill match increases confidence.")

        # penalize for overlap
        if overlap_count > 1:
            conf -= 0.25
            reasons.append("Potential overlap with other assigned shift on same day.")

        # clamp confidence between 0.05 and 0.99
        conf = max(0.05, min(0.99, conf))

        # advisory flag (frontend expects recommendations to be advisory)
        advisory = True

        # assemble enriched item
        item = {
            "day": day,
            "shift_id": sid,
            "shift_name": shift_name,
            "employee_name": employee_name,
            "forecast": forecast_str,
            "reasons": reasons,
            "confidence": round(conf, 3),
            "advisory": advisory,
            "overlap_warnings": overlap_warnings,
            "skill_match": bool(skill_match),
            # include original schedule index for frontend actions
            "schedule_index": idx,
            "employee_skills": skills
        }
        items.append(item)

    conn.close()
    return {"schedule_id": schedule_id, "items": items}

@app.post("/schedules/{schedule_id}/apply/{index}")
def apply_instruction(schedule_id: int, index: int):
    """
    Returns manual (English) instructions to apply a specific recommendation.
    Non-destructive: does not modify DB, only provides a clear step for managers to follow or export.
    This matches the frontend 'Apply' button which requests manual instructions.
    """
    conn = db()
    c = conn.cursor()
    # check schedule exists
    srow = c.execute("SELECT restaurant_id, days FROM schedules WHERE id=?", (schedule_id,)).fetchone()
    if not srow:
        raise HTTPException(status_code=404, detail="Schedule not found")
    # fetch items ordered same way as get_schedule
    rows = c.execute("""
        SELECT si.day, si.shift_id, sh.name AS shift_name, emp.name AS employee_name, emp.id AS employee_id, emp.skills AS skills
        FROM schedule_items si
        JOIN employees emp ON emp.id = si.employee_id
        JOIN shifts sh ON sh.id = si.shift_id
        WHERE si.schedule_id=?
        ORDER BY si.day, si.shift_id
    """, (schedule_id,)).fetchall()
    if index < 0 or index >= len(rows):
        raise HTTPException(status_code=400, detail="index out of range")
    r = rows[index]
    day = r["day"]
    shift_name = r["shift_name"]
    employee_name = r["employee_name"]
    # parse skills
    skills = []
    if r["skills"]:
        try:
            skills = json.loads(r["skills"])
        except Exception:
            skills = [s.strip() for s in (r["skills"] or "").split(",") if s.strip()]
    # attempt to compute confidence/reasons similarly to get_schedule (best-effort)
    # check availability
    avail = c.execute("SELECT 1 FROM availability WHERE employee_id=? AND day=? AND shift_id=?",
                      (r["employee_id"], day, r["shift_id"])).fetchone()
    has_availability = bool(avail)
    # load required
    lr = c.execute("SELECT required FROM load WHERE restaurant_id=? AND day=? AND shift_id=?",
                   (srow["restaurant_id"], day, r["shift_id"])).fetchone()
    req = lr["required"] if lr else None
    # forecast details
    try:
        forecast_resp = ai_forecast_load(srow["restaurant_id"])
        f = next((p for p in forecast_resp.get("predictions", []) if p["day"] == day and p["shift_id"] == r["shift_id"]), None)
    except Exception:
        f = None

    instructions_lines = [
        "Advisory Recommendation (non-binding):",
        f"- Day: {day+1}",
        f"- Shift: {shift_name}",
        f"- Suggested employee: {employee_name}",
    ]
    if skills:
        instructions_lines.append(f"- Employee skills: {', '.join(skills)}")
    if f:
        instructions_lines.append(f"- Forecasted covers: {f.get('predicted_covers')}, recommended staff: {f.get('recommended')}")
    if req is not None:
        instructions_lines.append(f"- Load required: {req}")
    instructions_lines.append(f"- Employee availability marked: {'yes' if has_availability else 'no'}")
    instructions_lines.append("- Confidence: advisory (see reasons)")
    instructions_lines.append("- Reasons: check availability, POS forecast, skills and load table.")
    instructions_lines.append("")
    instructions_lines.append("To apply manually: export this recommendation to CSV using the Export button in the UI,")
    instructions_lines.append("then import or copy the assignment into your payroll/rostering system.")
    instructions_lines.append("All recommendations are advisory and require manager confirmation.")
    conn.close()
    return {"instructions": "\n".join(instructions_lines)}

# ================= AI FEATURES (ENHANCED) =================
# Additive improvements: use sales_history, holidays, trend, weekday patterns.
# All endpoints remain advisory-only and do not modify existing data unless explicitly ingesting POS/holiday data.

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def parse_iso_date(s: str) -> date:
    return datetime.fromisoformat(s).date()

@app.post("/restaurants/{restaurant_id}/pos_data/")
def ingest_pos_data(restaurant_id: int, records: List[POSRecord]):
    """
    Ingest POS / sales historical records. This is additive storage used by AI.
    Each record: {date: "YYYY-MM-DD", shift_id, covers, revenue}
    """
    with db() as c:
        r = c.execute("SELECT id FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        for rec in records:
            # Basic validation of date format
            try:
                d = parse_iso_date(rec.date)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid date: {rec.date}")
            c.execute(
                "INSERT INTO sales_history (restaurant_id, date, shift_id, covers, revenue) VALUES (?, ?, ?, ?, ?)",
                (restaurant_id, rec.date, rec.shift_id, rec.covers, rec.revenue)
            )
    return {"status": "ok", "ingested": len(records)}

@app.post("/restaurants/{restaurant_id}/holidays/")
def ingest_holidays(restaurant_id: int, holidays: List[HolidayRecord]):
    """
    Add holiday definitions (date and name). Holidays are global; restaurant_id provided for API symmetry,
    but holidays table is shared across restaurants (you can change this behavior later).
    """
    with db() as c:
        for h in holidays:
            # validate date
            try:
                d = parse_iso_date(h.date)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid date: {h.date}")
            # avoid duplicate entries
            exists = c.execute("SELECT 1 FROM holidays WHERE date=?", (h.date,)).fetchone()
            if not exists:
                c.execute("INSERT INTO holidays (date, name) VALUES (?, ?)", (h.date, h.name))
    return {"status": "ok", "ingested": len(holidays)}

def is_holiday(curs, dt: date) -> bool:
    row = curs.execute("SELECT 1 FROM holidays WHERE date=?", (dt.isoformat(),)).fetchone()
    return bool(row)

def get_sales_history_for_restaurant(curs, restaurant_id: int):
    rows = curs.execute("SELECT date, shift_id, covers FROM sales_history WHERE restaurant_id=?", (restaurant_id,)).fetchall()
    # return list of tuples (date (date), shift_id, covers)
    out = []
    for r in rows:
        try:
            d = parse_iso_date(r["date"])
        except Exception:
            continue
        out.append((d, r["shift_id"], r["covers"]))
    return out

@app.get("/restaurants/{restaurant_id}/ai/forecast_load/")
def ai_forecast_load(restaurant_id: int):
    """
    Improved forecast:
    - Uses historical sales_history (covers) if present.
    - Considers weekday patterns, recency weighting and holidays.
    - Falls back to simple average / load table if no history.
    Returns recommended staff counts per day and shift (advisory only).
    STRICT: blocked for restaurants without active subscription.
    """
    # Strict check: AI forecast is a paid feature
    ensure_subscription_active(restaurant_id)

    c = db().cursor()
    r = c.execute("SELECT planning_days FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    planning_days = r["planning_days"] or 7

    shifts = [s["id"] for s in c.execute("SELECT id FROM shifts WHERE restaurant_id=?", (restaurant_id,)).fetchall()]
    if not shifts:
        return {"restaurant_id": restaurant_id, "predictions": []}

    sales = get_sales_history_for_restaurant(c, restaurant_id)
    today = datetime.utcnow().date()

    # Build stats keyed by (weekday, shift_id) and overall holiday adjustments
    weekday_shift_vals = defaultdict(list)
    holiday_vals = defaultdict(list)
    nonholiday_vals = defaultdict(list)

    for d, sid, covers in sales:
        weekday = d.weekday()
        weekday_shift_vals[(weekday, sid)].append((d, covers))
        if is_holiday(c, d):
            holiday_vals[(weekday, sid)].append((d, covers))
        else:
            nonholiday_vals[(weekday, sid)].append((d, covers))

    # compute holiday multiplier per (weekday, shift) as avg(holiday) / avg(nonholiday) if data exists
    holiday_multiplier = {}
    for key in set(list(weekday_shift_vals.keys())):
        hvals = [v for _, v in holiday_vals.get(key, [])]
        nhvals = [v for _, v in nonholiday_vals.get(key, [])]
        if hvals and nhvals:
            try:
                holiday_multiplier[key] = (statistics.mean(hvals) / (statistics.mean(nhvals) + 1e-6))
            except Exception:
                holiday_multiplier[key] = 1.0
        else:
            holiday_multiplier[key] = 1.0

    predictions = []
    # For trend: compute simple linear trend slope (covers per day) per (weekday, shift) using recent data if enough points
    for d_offset in range(planning_days):
        target_date = today + timedelta(days=d_offset)
        weekday = target_date.weekday()
        is_hol = is_holiday(c, target_date)
        for sid in shifts:
            key = (weekday, sid)
            hist = weekday_shift_vals.get(key, [])
            predicted_covers = None

            if hist:
                # recency-weighted average: weight = exp(-days_ago / 28)
                weighted_sum = 0.0
                weight_total = 0.0
                for hist_date, covers in hist:
                    days_ago = (today - hist_date).days
                    weight = math.exp(-max(0, days_ago) / 28.0)
                    weighted_sum += covers * weight
                    weight_total += weight
                base = (weighted_sum / weight_total) if weight_total > 0 else statistics.mean([c for _, c in hist])
                # apply holiday multiplier
                mult = holiday_multiplier.get(key, 1.0)
                if is_hol:
                    base *= mult
                # simple linear trend: compute slope across history if enough points
                if len(hist) >= 3:
                    # map dates to integer days and do simple linear regression (least squares)
                    xs = [(hist_date - hist[0][0]).days for hist_date, _ in hist]
                    ys = [covers for _, covers in hist]
                    x_mean = statistics.mean(xs)
                    y_mean = statistics.mean(ys)
                    denom = sum((x - x_mean) ** 2 for x in xs)
                    if denom != 0:
                        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
                        slope = num / denom
                        days_ahead = (target_date - hist[-1][0]).days
                        base = max(0, base + slope * days_ahead)
                predicted_covers = max(0.0, base)
            else:
                # fallback: if no history per (weekday,shift), try overall shift average
                all_shift_vals = [covers for _, s, covers in sales if s == sid]
                if all_shift_vals:
                    predicted_covers = statistics.mean(all_shift_vals)
                else:
                    # final fallback: check load table for required staff and map to covers via COVERS_PER_EMP
                    lr = c.execute("SELECT required FROM load WHERE restaurant_id=? AND day=? AND shift_id=?",
                                   (restaurant_id, d_offset, sid)).fetchone()
                    if lr:
                        predicted_covers = (lr["required"] * COVERS_PER_EMP)
                    else:
                        predicted_covers = COVERS_PER_EMP  # assume minimal covers

            # Map predicted covers to recommended staff
            recommended_staff = max(1, int(math.ceil(predicted_covers / COVERS_PER_EMP)))
            reason = f"Predicted covers ~{predicted_covers:.1f} based on historical POS and weekday patterns"
            if is_hol:
                reason += "; holiday multiplier applied"
            predictions.append({
                "day": d_offset,
                "date": target_date.isoformat(),
                "shift_id": sid,
                "predicted_covers": round(predicted_covers, 1),
                "recommended": recommended_staff,
                "reason": reason
            })

    return {"restaurant_id": restaurant_id, "predictions": predictions}

@app.get("/restaurants/{restaurant_id}/ai/hints/")
def ai_hints(restaurant_id: int):
    """
    Improved hints:
    - Compare forecasted recommended staff to current load.required to produce overstaffed/understaffed hints.
    - Uses POS-based forecast when available, otherwise falls back to load/availability heuristics.
    STRICT: blocked for restaurants without active subscription.
    """
    ensure_subscription_active(restaurant_id)

    c = db().cursor()
    r = c.execute("SELECT planning_days FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    planning_days = r["planning_days"] or 7

    # Get forecast (prediction per day+shift)
    forecast = ai_forecast_load(restaurant_id)["predictions"]

    # load per day+shift
    load_rows = c.execute("SELECT day, shift_id, required FROM load WHERE restaurant_id=?", (restaurant_id,)).fetchall()
    load_map = {(lr["day"], lr["shift_id"]): lr["required"] for lr in load_rows}

    # availability counts (only for this restaurant)
    avail_rows = c.execute("""
        SELECT a.employee_id, a.day, a.shift_id
        FROM availability a
        JOIN employees e ON e.id = a.employee_id
        WHERE e.restaurant_id=?
    """, (restaurant_id,)).fetchall()
    avail_map = defaultdict(set)
    for ar in avail_rows:
        avail_map[(ar["day"], ar["shift_id"])].add(ar["employee_id"])

    hints = []
    OVERSTAFF_DIFF = 2  # staff difference threshold
    for p in forecast:
        d = p["day"]
        sid = p["shift_id"]
        rec = p["recommended"]
        req = load_map.get((d, sid), None)
        available = len(avail_map.get((d, sid), set()))
        # Compare forecast recommended staff to required (load) if available
        if req is not None:
            if available >= rec + OVERSTAFF_DIFF:
                hints.append({"level": "info", "message": f"You are overstaffed on day {d+1} shift {sid}: recommended {rec}, available {available}, required {req}"})
            elif available < rec:
                hints.append({"level": "warning", "message": f"Risk of understaffing on day {d+1} shift {sid}: recommended {rec}, available {available}, required {req}"})
        else:
            # no explicit load -> still warn about availability vs recommended
            if available >= rec + OVERSTAFF_DIFF:
                hints.append({"level": "info", "message": f"Likely overstaffed on day {d+1} shift {sid}: recommended {rec}, available {available}"})
            elif available < rec:
                hints.append({"level": "warning", "message": f"Likely understaffed on day {d+1} shift {sid}: recommended {rec}, available {available}"})

    # aggregate hint
    if not hints:
        hints.append({"level":"info", "message":"No immediate staffing issues detected for the selected planning period (based on POS/forecast)."})
    return {"restaurant_id": restaurant_id, "hints": hints}

@app.post("/restaurants/{restaurant_id}/ai/suggest_schedule/")
def ai_suggest_schedule(restaurant_id: int, max_suggestions: Optional[int] = 100):
    """
    Suggest a schedule using forecasted recommended staff.
    Still advisory — does NOT save anything to DB.
    STRICT: blocked for restaurants without active subscription.
    """
    ensure_subscription_active(restaurant_id)

    c = db().cursor()
    r = c.execute("SELECT planning_days FROM restaurants WHERE id=?", (restaurant_id,)).fetchone()
    if not r:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    planning_days = r["planning_days"] or 7

    # get forecasted recommended staff
    forecast = ai_forecast_load(restaurant_id)["predictions"]
    # map forecast by (day, shift_id) -> recommended
    rec_map = {(p["day"], p["shift_id"]): p["recommended"] for p in forecast}

    # build availability per day+shift (only this restaurant)
    avail_rows = c.execute("""
        SELECT a.employee_id, a.day, a.shift_id
        FROM availability a
        JOIN employees e ON e.id = a.employee_id
        WHERE e.restaurant_id=?
    """, (restaurant_id,)).fetchall()
    avail_map = defaultdict(list)
    for ar in avail_rows:
        avail_map[(ar["day"], ar["shift_id"])].append(ar["employee_id")]

    # compute simple "workload" per employee (how many availability entries they have)
    emp_avail_count = defaultdict(int)
    for k, lst in avail_map.items():
        for emp in lst:
            emp_avail_count[emp] += 1

    suggestions = []
    suggestion_count = 0
    for d in range(planning_days):
        # collect shifts for this day from forecast (if none, fallback to load table)
        shifts_for_day = [p for p in forecast if p["day"] == d]
        if not shifts_for_day:
            # fallback to load table shifts
            shifts_for_day = []
            load_rows = c.execute("SELECT shift_id, required FROM load WHERE restaurant_id=? AND day=", (restaurant_id, d)).fetchall()
            for lr in load_rows:
                shifts_for_day.append({"day": d, "shift_id": lr["shift_id"], "recommended": lr["required"]})

        for p in shifts_for_day:
            sid = p["shift_id"]
            rec = rec_map.get((d, sid), p.get("recommended", 1))
            available = list(avail_map.get((d, sid), []))
            // ... (file continues; full original content will be saved)