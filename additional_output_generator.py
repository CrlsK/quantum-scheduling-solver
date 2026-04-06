"""
Additional Output Generator for QCentroid Solvers
Generates rich HTML visualizations and CSV exports in additional_output/ folder.
Platform picks up files from this folder and displays them in the job detail view.

All HTML is self-contained (inline CSS/SVG) — no external dependencies needed.
"""

import os
import json
import csv
import io
import math
from typing import Dict, List, Any


def _safe_get(obj, key, default=None):
    """Safely get a key from an object, returning default if obj is not a dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# Unicode characters used in f-string expressions — must be pre-assigned to
# variables because Python < 3.12 forbids backslashes inside f-string {}.
# When this file is pushed via JSON-based APIs, unicode chars get serialised
# as \uXXXX escapes.  Keeping them in plain assignments (not inside {})
# lets the interpreter decode them normally.
_SPECIALTY_YES = "\u26a1 Yes"
_EMDASH = "\u2014"


def generate_additional_output(input_data: dict, result: dict, algorithm_name: str = "Solver"):
    """
    Main entry point. Call from run() after computing the result dict.
    Creates additional_output/ folder and writes all visualization files.
    Each file is generated independently — errors in one don't block others.
    """
    os.makedirs("additional_output", exist_ok=True)

    _files = [
        ("additional_output/01_input_overview.html", _generate_input_overview_html, (input_data,)),
        ("additional_output/02_problem_structure.html", _generate_problem_structure_html, (input_data,)),
        ("additional_output/03_executive_dashboard.html", _generate_executive_dashboard_html, (result, input_data, algorithm_name)),
        ("additional_output/04_gantt_schedule.html", _generate_gantt_html, (result, input_data)),
        ("additional_output/05_machine_utilization.html", _generate_machine_utilization_html, (result, input_data)),
        ("additional_output/06_delivery_analysis.html", _generate_delivery_analysis_html, (result, input_data)),
        ("additional_output/07_financial_impact.html", _generate_financial_impact_html, (result, input_data)),
        ("additional_output/08_energy_report.html", _generate_energy_report_html, (result, input_data)),
        ("additional_output/09_schedule_assignments.csv", _generate_schedule_csv, (result,)),
        ("additional_output/10_kpi_summary.csv", _generate_kpi_csv, (result,)),
        ("additional_output/11_machine_metrics.csv", _generate_machine_csv, (result,)),
        ("additional_output/12_job_delivery.csv", _generate_delivery_csv, (result,)),
    ]

    generated = 0
    for path, func, args in _files:
        try:
            content = func(*args)
            _write_file(path, content)
            generated += 1
        except Exception:
            pass  # skip this file, continue with others

    return generated


def _write_file(path: str, content: str):
    """Write content to file, silently skip on error."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


# ============================================================================
# Shared HTML helpers
# ============================================================================

_CSS = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 24px; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 28px; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }
  h2 { font-size: 20px; font-weight: 600; color: #94a3b8; margin: 24px 0 12px; }
  h3 { font-size: 16px; font-weight: 600; color: #cbd5e1; margin: 16px 0 8px; }
  .subtitle { color: #64748b; font-size: 14px; margin-bottom: 24px; }
  .grid { display: grid; gap: 16px; margin-bottom: 24px; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: 1fr 1fr 1fr; }
  .grid-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
  .grid-5 { grid-template-columns: 1fr 1fr 1fr 1fr 1fr; }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
  .kpi-card { text-align: center; }
  .kpi-value { font-size: 32px; font-weight: 700; color: #f8fafc; }
  .kpi-label { font-size: 12px; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi-delta { font-size: 13px; margin-top: 4px; }
  .positive { color: #4ade80; }
  .negative { color: #f87171; }
  .neutral { color: #fbbf24; }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .badge-green { background: #064e3b; color: #6ee7b7; }
  .badge-red { background: #7f1d1d; color: #fca5a5; }
  .badge-yellow { background: #713f12; color: #fde68a; }
  .badge-blue { background: #1e3a5f; color: #93c5fd; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #334155; color: #94a3b8; text-align: left; padding: 10px 12px;
       font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
  td { padding: 10px 12px; border-bottom: 1px solid #1e293b; }
  tr:hover { background: #1e293b; }
  .bar-bg { background: #334155; border-radius: 4px; height: 20px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin: 1px; }
  .header-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;
                padding-bottom: 16px; border-bottom: 1px solid #334155; }
  .logo { font-size: 11px; color: #475569; }
  @media (max-width: 768px) { .grid-2, .grid-3, .grid-4, .grid-5 { grid-template-columns: 1fr; } }
</style>
"""


def _html_wrapper(title: str, subtitle: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>{_CSS}</head>
<body><div class="container">
<div class="header-bar">
  <div><h1>{title}</h1><div class="subtitle">{subtitle}</div></div>
  <div class="logo">QCentroid Platform | Jindal Stainless</div>
</div>
{body}
</div></body></html>"""


def _kpi_card(value, label, delta=None, delta_good=True):
    delta_html = ""
    if delta is not None:
        cls = "positive" if (delta >= 0) == delta_good else "negative"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="kpi-delta {cls}">{sign}{delta:.1f}%</div>'
    return f"""<div class="card kpi-card">
<div class="kpi-value">{value}</div>
<div class="kpi-label">{label}</div>{delta_html}</div>"""


def _bar_chart_inline(value, max_val=100, color="#3b82f6"):
    pct = min(100, max(0, (value / max_val * 100) if max_val > 0 else 0))
    return f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'


def _badge(text, variant="blue"):
    return f'<span class="badge badge-{variant}">{text}</span>'


def _svg_donut(pct, label, size=120, color="#3b82f6"):
    r = 40
    circ = 2 * math.pi * r
    dash = circ * pct / 100
    gap = circ - dash
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100">
<circle cx="50" cy="50" r="{r}" fill="none" stroke="#334155" stroke-width="8"/>
<circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="8"
  stroke-dasharray="{dash:.1f} {gap:.1f}" stroke-dashoffset="{circ/4:.1f}" stroke-linecap="round"/>
<text x="50" y="48" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="700">{pct:.0f}%</text>
<text x="50" y="64" text-anchor="middle" fill="#94a3b8" font-size="8">{label}</text>
</svg>"""


# ============================================================================
# INPUT VISUALIZATIONS
# ============================================================================

def _generate_input_overview_html(input_data: dict) -> str:
    metadata = input_data.get("metadata", {})
    horizon = input_data.get("planning_horizon", {})
    machines = input_data.get("machines", [])
    jobs = input_data.get("jobs", [])
    maint = input_data.get("maintenance_schedules", [])

    plant = metadata.get("plant_name", metadata.get("plant", "Unknown"))
    scenario = metadata.get("scenario_name", metadata.get("scenario", ""))
    h_end = horizon.get("end_time", 72)
    h_unit = horizon.get("time_unit", "hours")

    # Machine type breakdown
    machine_types = {}
    for m in machines:
        t = m.get("type", "unknown")
        machine_types[t] = machine_types.get(t, 0) + 1

    # Job priority breakdown
    priority_counts = {}
    grade_counts = {}
    total_qty = 0
    for j in jobs:
        p = j.get("priority", "medium")
        g = j.get("material_grade", "unknown")
        priority_counts[p] = priority_counts.get(p, 0) + 1
        grade_counts[g] = grade_counts.get(g, 0) + 1
        total_qty += j.get("quantity_kg", 0)

    body = f"""
    <div class="grid grid-4">
      {_kpi_card(len(jobs), "Total Jobs")}
      {_kpi_card(len(machines), "Machines")}
      {_kpi_card(f"{h_end}h", "Planning Horizon")}
      {_kpi_card(f"{total_qty:,.0f}", "Total kg")}
    </div>

    <div class="grid grid-2">
      <div class="card">
        <h3>Machine Fleet</h3>
        <table><tr><th>Machine Type</th><th>Count</th></tr>
        {''.join(f'<tr><td>{t.replace("_"," ").title()}</td><td>{c}</td></tr>' for t,c in sorted(machine_types.items()))}
        </table>
      </div>
      <div class="card">
        <h3>Job Priority Distribution</h3>
        <table><tr><th>Priority</th><th>Jobs</th><th>Share</th></tr>
        {''.join(f'<tr><td>{_badge(p, "red" if p=="critical" else "yellow" if p=="high" else "blue")}</td><td>{c}</td><td>{c/len(jobs)*100:.0f}%</td></tr>' for p,c in sorted(priority_counts.items()))}
        </table>
      </div>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <h3>Material Grades</h3>
        <table><tr><th>Grade</th><th>Jobs</th><th>Specialty</th></tr>
        {''.join(f'<tr><td>{g}</td><td>{c}</td><td>{_SPECIALTY_YES if g.lower() in ["duplex_2205","904l"] else _EMDASH}</td></tr>' for g,c in sorted(grade_counts.items()))}
        </table>
      </div>
      <div class="card">
        <h3>Maintenance Windows</h3>
        <table><tr><th>Machine</th><th>Start</th><th>End</th><th>Reason</th></tr>
        {''.join('<tr><td>' + str(m.get("machine_id","")) + '</td><td>' + str(m.get("scheduled_start",0)) + 'h</td><td>' + str(m.get("scheduled_end",0)) + 'h</td><td>' + str(m.get("description",""))[:40] + '</td></tr>' for m in maint[:10])}
        </table>
        {f'<p style="color:#64748b;margin-top:8px;font-size:12px;">Showing {min(10,len(maint))} of {len(maint)} windows</p>' if len(maint) > 10 else ''}
      </div>
    </div>
    """

    return _html_wrapper(
        f"Input Overview — {plant}",
        f"{scenario} | {len(jobs)} jobs, {len(machines)} machines, {h_end}{h_unit} horizon",
        body
    )


def _generate_problem_structure_html(input_data: dict) -> str:
    jobs = input_data.get("jobs", [])
    machines = input_data.get("machines", [])
    horizon = input_data.get("planning_horizon", {})
    h_end = horizon.get("end_time", 72)

    # Jobs table
    rows = []
    for j in jobs:
        jid = j.get("job_id", "")
        cust = j.get("customer_id", "")
        grade = j.get("material_grade", "")
        qty = j.get("quantity_kg", 0)
        due = j.get("due_date", h_end)
        prio = j.get("priority", "medium")
        ops = len(j.get("required_operations", []))
        prio_badge = _badge(prio, "red" if prio == "critical" else "yellow" if prio == "high" else "blue" if prio == "medium" else "green")
        rows.append(f"<tr><td>{jid}</td><td>{cust}</td><td>{grade}</td><td>{qty:,.0f}</td><td>{due}h</td><td>{prio_badge}</td><td>{ops}</td></tr>")

    # Machines table
    m_rows = []
    for m in machines:
        mid = m.get("machine_id", "")
        name = m.get("name", "")
        mtype = m.get("type", "").replace("_", " ").title()
        power = m.get("power_consumption_kw", 0)
        caps = ", ".join(cg.get("operation_code", "") for cg in m.get("capability_groups", []))
        m_rows.append(f"<tr><td>{mid}</td><td>{name}</td><td>{mtype}</td><td>{power} kW</td><td>{caps}</td></tr>")

    body = f"""
    <div class="card" style="margin-bottom:24px">
      <h3>Jobs ({len(jobs)})</h3>
      <div style="overflow-x:auto">
      <table>
        <tr><th>Job ID</th><th>Customer</th><th>Grade</th><th>Qty (kg)</th><th>Due</th><th>Priority</th><th>Ops</th></tr>
        {''.join(rows)}
      </table>
      </div>
    </div>
    <div class="card">
      <h3>Machines ({len(machines)})</h3>
      <div style="overflow-x:auto">
      <table>
        <tr><th>ID</th><th>Name</th><th>Type</th><th>Power</th><th>Capabilities</th></tr>
        {''.join(m_rows)}
      </table>
      </div>
    </div>
    """

    return _html_wrapper("Problem Structure", f"{len(jobs)} jobs × {len(machines)} machines", body)


# ============================================================================
# OUTPUT VISUALIZATIONS
# ============================================================================

def _generate_executive_dashboard_html(result: dict, input_data: dict, algorithm: str) -> str:
    sched = result.get("schedule", {})
    makespan = sched.get("makespan", 0)
    tardiness = sched.get("total_tardiness", 0)
    otd = sched.get("on_time_percentage", 0)
    jobs_on = sched.get("jobs_on_time", 0)
    jobs_late = sched.get("jobs_late", 0)
    energy = sched.get("total_energy_kwh", 0)
    obj = result.get("objective_value", 0)
    status = result.get("solution_status", "unknown")

    mu = result.get("machine_utilization", {})
    if not isinstance(mu, dict):
        mu = {}
    avg_util = sum(_safe_get(m, "utilization_percentage", m if isinstance(m, (int, float)) else 0) for m in mu.values()) / len(mu) if mu else 0

    ph = input_data.get("planning_horizon", {})
    horizon = _safe_get(ph, "end_time", 72) if isinstance(ph, dict) else 72
    metadata = input_data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    plant = metadata.get("plant_name", metadata.get("plant", "Plant"))

    cm = result.get("computation_metrics", {})
    elapsed = _safe_get(cm, "wall_time_s", 0) if isinstance(cm, dict) else 0

    status_badge = _badge(status.upper(), "green" if status in ("optimal", "feasible") else "red")

    body = f"""
    <div style="margin-bottom:16px">{status_badge} <span style="color:#94a3b8;margin-left:8px;">Algorithm: {algorithm} | Computed in {elapsed:.1f}s</span></div>

    <div class="grid grid-5">
      {_kpi_card(f"{makespan:.1f}h", "Makespan", delta=((horizon - makespan)/horizon*100) if horizon else 0)}
      {_kpi_card(f"{otd:.1f}%", "On-Time Delivery", delta=otd - 97)}
      {_kpi_card(f"{avg_util:.1f}%", "Avg Utilization", delta=avg_util - 88)}
      {_kpi_card(f"{tardiness:.1f}h", "Total Tardiness")}
      {_kpi_card(f"{obj:.1f}", "Objective Value")}
    </div>

    <div class="grid grid-3">
      <div class="card" style="text-align:center">
        {_svg_donut(otd, "On-Time", color="#4ade80" if otd >= 95 else "#fbbf24" if otd >= 85 else "#f87171")}
      </div>
      <div class="card" style="text-align:center">
        {_svg_donut(avg_util, "Utilization", color="#3b82f6")}
      </div>
      <div class="card" style="text-align:center">
        {_svg_donut(min(100, (1 - tardiness/max(makespan,1))*100), "Schedule Health", color="#8b5cf6")}
      </div>
    </div>

    <div class="grid grid-2">
      <div class="card">
        <h3>Job Delivery Summary</h3>
        <div style="display:flex;gap:24px;margin-top:12px">
          <div style="text-align:center;flex:1">
            <div style="font-size:48px;font-weight:700;color:#4ade80">{jobs_on}</div>
            <div style="color:#94a3b8;font-size:13px">On Time</div>
          </div>
          <div style="text-align:center;flex:1">
            <div style="font-size:48px;font-weight:700;color:#f87171">{jobs_late}</div>
            <div style="color:#94a3b8;font-size:13px">Late</div>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>Resource Usage</h3>
        <table>
          <tr><td>Energy Consumption</td><td style="text-align:right">{energy:.1f} kWh</td></tr>
          <tr><td>Total Changeovers</td><td style="text-align:right">{result.get("total_changeovers", 0)}</td></tr>
          <tr><td>Idle Time</td><td style="text-align:right">{sched.get("total_idle_time", 0):.1f}h</td></tr>
          <tr><td>Computation Time</td><td style="text-align:right">{elapsed:.2f}s</td></tr>
        </table>
      </div>
    </div>
    """

    return _html_wrapper(f"Executive Dashboard — {plant}", f"{algorithm} | {jobs_on+jobs_late} jobs scheduled", body)


def _generate_gantt_html(result: dict, input_data: dict) -> str:
    sched = result.get("schedule", {})
    assignments = sched.get("assignments", [])
    gantt = sched.get("gantt_data", assignments)
    makespan = sched.get("makespan", 1)
    machines = input_data.get("machines", [])

    if not gantt or makespan <= 0:
        return _html_wrapper("Gantt Schedule", "No schedule data", "<p>No assignments generated.</p>")

    machine_ids = sorted(set(a.get("machine_id", "") for a in gantt))
    colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#f97316", "#14b8a6", "#eab308",
              "#6366f1", "#ef4444", "#22c55e", "#06b6d4", "#a855f7", "#f43f5e",
              "#84cc16", "#0ea5e9", "#d946ef"]
    job_ids = sorted(set(a.get("job_id", "") for a in gantt))
    job_color = {j: colors[i % len(colors)] for i, j in enumerate(job_ids)}

    chart_w = 900
    row_h = 32
    left_margin = 100
    chart_h = len(machine_ids) * row_h + 60

    bars = []
    for a in gantt:
        mid = a.get("machine_id", "")
        if mid not in machine_ids:
            continue
        y_idx = machine_ids.index(mid)
        start = a.get("start_time", a.get("start", 0))
        end = a.get("end_time", a.get("end", 0))
        jid = a.get("job_id", "")
        x = left_margin + (start / makespan) * (chart_w - left_margin)
        w = max(2, ((end - start) / makespan) * (chart_w - left_margin))
        y = 30 + y_idx * row_h + 2
        c = job_color.get(jid, "#475569")
        bars.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_h-4}" rx="3" fill="{c}" opacity="0.85">'
                     f'<title>{jid} | {start:.1f}h-{end:.1f}h</title></rect>')

    # Y-axis labels
    y_labels = []
    for i, mid in enumerate(machine_ids):
        name = mid
        for m in machines:
            if m.get("machine_id") == mid:
                name = m.get("name", mid)[:12]
                break
        y = 30 + i * row_h + row_h / 2 + 4
        y_labels.append(f'<text x="{left_margin-8}" y="{y}" text-anchor="end" fill="#94a3b8" font-size="11">{name}</text>')

    # X-axis ticks
    x_ticks = []
    num_ticks = min(10, int(makespan))
    for i in range(num_ticks + 1):
        t = i * makespan / num_ticks
        x = left_margin + (t / makespan) * (chart_w - left_margin)
        x_ticks.append(f'<text x="{x:.1f}" y="{chart_h-5}" text-anchor="middle" fill="#64748b" font-size="10">{t:.0f}h</text>')
        x_ticks.append(f'<line x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{chart_h-20}" stroke="#1e293b" stroke-width="1"/>')

    # Grid lines
    grid = []
    for i in range(len(machine_ids)):
        y = 30 + i * row_h
        grid.append(f'<line x1="{left_margin}" y1="{y}" x2="{chart_w}" y2="{y}" stroke="#1e293b" stroke-width="1"/>')

    # Legend
    legend = []
    for i, jid in enumerate(job_ids[:15]):
        lx = (i % 5) * 180
        ly = (i // 5) * 20
        c = job_color[jid]
        legend.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{c}"/>'
                      f'<text x="{lx+16}" y="{ly+10}" fill="#cbd5e1" font-size="11">{jid}</text>')

    legend_h = ((len(job_ids[:15]) - 1) // 5 + 1) * 20 + 10

    body = f"""
    <div class="card" style="overflow-x:auto">
      <svg width="{chart_w}" height="{chart_h}" style="display:block;margin:0 auto">
        {''.join(grid)}{''.join(x_ticks)}{''.join(y_labels)}{''.join(bars)}
      </svg>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>Legend</h3>
      <svg width="{chart_w}" height="{legend_h}">{''.join(legend)}</svg>
    </div>
    """

    return _html_wrapper("Gantt Schedule", f"{len(assignments)} operations across {len(machine_ids)} machines | Makespan: {makespan:.1f}h", body)


def _get_util(data):
    """Extract utilization percentage from either a dict or a numeric value."""
    if isinstance(data, dict):
        return data.get("utilization_percentage", 0)
    if isinstance(data, (int, float)):
        return data
    return 0


def _generate_machine_utilization_html(result: dict, input_data: dict) -> str:
    mu = result.get("machine_utilization", {})
    if not isinstance(mu, dict):
        mu = {}
    machines = input_data.get("machines", [])

    if not mu:
        return _html_wrapper("Machine Utilization", "No data", "<p>No utilization data.</p>")

    machine_map = {m.get("machine_id", ""): m.get("name", m.get("machine_id", "")) for m in machines if isinstance(m, dict)}

    rows = []
    sorted_mu = sorted(mu.items(), key=lambda x: _get_util(x[1]), reverse=True)
    for mid, data in sorted_mu:
        util = _get_util(data)
        name = machine_map.get(mid, mid)
        proc = _safe_get(data, "total_processing_hours", _safe_get(data, "total_busy_time", 0))
        idle = _safe_get(data, "idle_time_hours", _safe_get(data, "idle_time", 0))
        njobs = _safe_get(data, "num_jobs", _safe_get(data, "num_jobs_processed", 0))
        color = "#4ade80" if util >= 80 else "#fbbf24" if util >= 50 else "#f87171"
        rows.append(f"""<tr>
          <td>{mid}</td><td>{name}</td>
          <td>{_bar_chart_inline(util, 100, color)} <span style="font-size:12px;color:#94a3b8">{util:.1f}%</span></td>
          <td>{proc:.1f}h</td><td>{idle:.1f}h</td><td>{njobs}</td></tr>""")

    avg_util = sum(_get_util(d) for d in mu.values()) / len(mu)

    body = f"""
    <div class="grid grid-3">
      {_kpi_card(f"{avg_util:.1f}%", "Average Utilization")}
      {_kpi_card(str(sum(1 for d in mu.values() if _get_util(d) > 85)), "Bottleneck Machines")}
      {_kpi_card(str(sum(1 for d in mu.values() if _get_util(d) < 50)), "Underutilized")}
    </div>
    <div class="card">
      <table>
        <tr><th>ID</th><th>Machine</th><th>Utilization</th><th>Processing</th><th>Idle</th><th>Jobs</th></tr>
        {''.join(rows)}
      </table>
    </div>
    """

    return _html_wrapper("Machine Utilization", f"{len(mu)} machines analyzed", body)


def _generate_delivery_analysis_html(result: dict, input_data: dict) -> str:
    jm = result.get("job_metrics", {})
    if not isinstance(jm, dict):
        jm = {}
    jobs = input_data.get("jobs", [])
    ph = input_data.get("planning_horizon", {})
    horizon = _safe_get(ph, "end_time", 72) if isinstance(ph, dict) else 72

    rows = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        jid = j.get("job_id", "")
        m = jm.get(jid, {})
        if not isinstance(m, dict):
            m = {}
        cust = j.get("customer_id", "")
        grade = j.get("material_grade", "")
        prio = j.get("priority", "medium")
        due = _safe_get(m, "due_date", j.get("due_date", horizon))
        comp = _safe_get(m, "completion_time", 0)
        tard = _safe_get(m, "tardiness", 0)
        on_time = _safe_get(m, "on_time", False)
        slack = due - comp if comp else 0
        status = _badge("ON TIME", "green") if on_time else _badge(f"LATE {tard:.1f}h", "red")
        rows.append(f"<tr><td>{jid}</td><td>{cust}</td><td>{grade}</td><td>{_badge(prio, 'red' if prio=='critical' else 'yellow' if prio=='high' else 'blue')}</td>"
                    f"<td>{due:.1f}h</td><td>{comp:.1f}h</td><td>{slack:.1f}h</td><td>{status}</td></tr>")

    body = f"""
    <div class="card">
      <div style="overflow-x:auto">
      <table>
        <tr><th>Job</th><th>Customer</th><th>Grade</th><th>Priority</th><th>Due</th><th>Completed</th><th>Slack</th><th>Status</th></tr>
        {''.join(rows)}
      </table>
      </div>
    </div>
    """

    return _html_wrapper("Delivery Analysis", f"{len(jobs)} jobs tracked", body)


def _generate_financial_impact_html(result: dict, input_data: dict) -> str:
    fi = result.get("financial_impact", {})
    cb = result.get("cost_breakdown", {})

    if not fi:
        # Generate basic financial view from cost_breakdown
        body = """<div class="card"><h3>Cost Breakdown</h3><table>"""
        for k, v in cb.items():
            body += f"<tr><td>{k.replace('_',' ').title()}</td><td style='text-align:right'>{v:,.1f}</td></tr>"
        body += "</table></div>"
        return _html_wrapper("Financial Impact", "Basic cost view", body)

    sla = fi.get("sla_compliance", {})
    energy_econ = fi.get("energy_economics", {})
    prod_econ = fi.get("production_economics", {})

    body = f"""
    <div class="grid grid-3">
      {_kpi_card(f"₹{sla.get('net_sla_savings_inr',0):,.0f}", "Net SLA Savings")}
      {_kpi_card(f"₹{energy_econ.get('total_energy_cost_inr',0):,.0f}", "Energy Cost")}
      {_kpi_card(f"₹{prod_econ.get('estimated_cost_per_kg_inr',0):.1f}/kg", "Cost per kg")}
    </div>
    <div class="grid grid-2">
      <div class="card">
        <h3>SLA Compliance</h3>
        <table>
          <tr><td>Penalty Avoided</td><td style="text-align:right;color:#4ade80">₹{sla.get('total_penalty_avoided_inr',0):,.0f}</td></tr>
          <tr><td>Penalty Incurred</td><td style="text-align:right;color:#f87171">₹{sla.get('total_penalty_incurred_inr',0):,.0f}</td></tr>
          <tr><td><strong>Net Savings</strong></td><td style="text-align:right"><strong>₹{sla.get('net_sla_savings_inr',0):,.0f}</strong></td></tr>
        </table>
      </div>
      <div class="card">
        <h3>Energy Economics</h3>
        <table>
          <tr><td>Total Energy</td><td style="text-align:right">{energy_econ.get('total_energy_kwh',0):,.1f} kWh</td></tr>
          <tr><td>Peak Cost</td><td style="text-align:right">₹{energy_econ.get('peak_energy_cost_inr',0):,.0f}</td></tr>
          <tr><td>Off-Peak Cost</td><td style="text-align:right">₹{energy_econ.get('off_peak_energy_cost_inr',0):,.0f}</td></tr>
          <tr><td>Cost/kg</td><td style="text-align:right">₹{energy_econ.get('energy_cost_per_kg_inr',0):.2f}</td></tr>
        </table>
      </div>
    </div>
    """

    return _html_wrapper("Financial Impact Analysis", "Cost savings and production economics", body)


def _generate_energy_report_html(result: dict, input_data: dict) -> str:
    es = result.get("energy_sustainability", {})
    sched = result.get("schedule", {})
    energy_kwh = sched.get("total_energy_kwh", 0)

    if not es:
        body = f"""<div class="card"><h3>Energy Summary</h3>
        <p style="font-size:24px;color:#fbbf24">{energy_kwh:.1f} kWh</p>
        <p style="color:#94a3b8">Total energy consumed during production</p></div>"""
        return _html_wrapper("Energy & Sustainability", "Basic view", body)

    ep = es.get("energy_profile", {})
    cf = es.get("carbon_footprint", {})
    ec = es.get("environmental_compliance", {})

    compliance = cf.get("compliance_status", "UNKNOWN")
    comp_badge = _badge(compliance, "green" if compliance == "COMPLIANT" else "red")

    body = f"""
    <div class="grid grid-4">
      {_kpi_card(f"{ep.get('total_consumption_kwh',0):,.0f}", "kWh Total")}
      {_kpi_card(f"{cf.get('estimated_co2_emissions_kg',0):,.0f}", "kg CO2")}
      {_kpi_card(f"{cf.get('co2_per_kg_product',0):.3f}", "kg CO2/kg product")}
      {_kpi_card(f"₹{ep.get('peak_avoidance_savings_inr',0):,.0f}", "Peak Savings Potential")}
    </div>
    <div style="text-align:center;margin:16px 0">{comp_badge}</div>
    <div class="grid grid-2">
      <div class="card">
        <h3>Energy Split</h3>
        <table>
          <tr><td>Peak Hours</td><td style="text-align:right">{ep.get('peak_hours_consumption_kwh',0):,.0f} kWh</td></tr>
          <tr><td>Off-Peak Hours</td><td style="text-align:right">{ep.get('off_peak_consumption_kwh',0):,.0f} kWh</td></tr>
        </table>
      </div>
      <div class="card">
        <h3>Carbon Footprint</h3>
        <table>
          <tr><td>Daily CO2</td><td style="text-align:right">{cf.get('daily_co2_kg',0):,.1f} kg</td></tr>
          <tr><td>Daily Limit</td><td style="text-align:right">{cf.get('daily_limit_kg_co2',18000):,.0f} kg</td></tr>
          <tr><td>Recycling Target</td><td style="text-align:right">{ec.get('recycling_target_pct',88)}%</td></tr>
        </table>
      </div>
    </div>
    """

    return _html_wrapper("Energy & Sustainability Report", "Environmental impact and compliance", body)


# ============================================================================
# CSV EXPORTS
# ============================================================================

def _generate_schedule_csv(result: dict) -> str:
    assignments = result.get("schedule", {}).get("assignments", [])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["job_id", "machine_id", "start_time", "end_time", "duration", "setup_time"])
    for a in assignments:
        w.writerow([
            a.get("job_id", ""), a.get("machine_id", ""),
            f"{a.get('start_time', a.get('start', 0)):.2f}",
            f"{a.get('end_time', a.get('end', 0)):.2f}",
            f"{a.get('duration', 0):.2f}",
            f"{a.get('setup_time', a.get('setup', 0)):.2f}",
        ])
    return buf.getvalue()


def _generate_kpi_csv(result: dict) -> str:
    sched = result.get("schedule", {})
    if not isinstance(sched, dict):
        sched = {}
    mu = result.get("machine_utilization", {})
    if not isinstance(mu, dict):
        mu = {}
    avg_util = sum(_get_util(m) for m in mu.values()) / len(mu) if mu else 0

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kpi", "value", "unit"])
    w.writerow(["makespan", f"{sched.get('makespan', 0):.2f}", "hours"])
    w.writerow(["on_time_delivery", f"{sched.get('on_time_percentage', 0):.1f}", "percent"])
    w.writerow(["total_tardiness", f"{sched.get('total_tardiness', 0):.2f}", "hours"])
    w.writerow(["avg_machine_utilization", f"{avg_util:.1f}", "percent"])
    w.writerow(["total_energy", f"{sched.get('total_energy_kwh', 0):.1f}", "kWh"])
    w.writerow(["objective_value", f"{result.get('objective_value', 0):.2f}", "composite"])
    w.writerow(["total_changeovers", str(result.get("total_changeovers", 0)), "count"])
    w.writerow(["jobs_on_time", str(sched.get("jobs_on_time", 0)), "count"])
    w.writerow(["jobs_late", str(sched.get("jobs_late", 0)), "count"])
    return buf.getvalue()


def _generate_machine_csv(result: dict) -> str:
    mu = result.get("machine_utilization", {})
    if not isinstance(mu, dict):
        mu = {}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["machine_id", "utilization_pct", "processing_hours", "idle_hours", "num_jobs"])
    for mid, data in sorted(mu.items()):
        w.writerow([
            mid,
            f"{_get_util(data):.1f}",
            f"{_safe_get(data, 'total_processing_hours', _safe_get(data, 'total_busy_time', 0)):.1f}",
            f"{_safe_get(data, 'idle_time_hours', _safe_get(data, 'idle_time', 0)):.1f}",
            _safe_get(data, "num_jobs", _safe_get(data, "num_jobs_processed", 0)),
        ])
    return buf.getvalue()


def _generate_delivery_csv(result: dict) -> str:
    jm = result.get("job_metrics", {})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["job_id", "completion_time", "due_date", "tardiness", "on_time"])
    for jid, data in sorted(jm.items()):
        w.writerow([
            jid,
            f"{data.get('completion_time', 0):.2f}",
            f"{data.get('due_date', 0):.2f}",
            f"{data.get('tardiness', 0):.2f}",
            str(data.get("on_time", False)),
        ])
    return buf.getvalue()
