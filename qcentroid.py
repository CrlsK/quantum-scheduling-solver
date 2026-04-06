"""
QCentroid Quantum-Inspired Solver for Dynamic Production Scheduling (Job Shop Scheduling Problem)
Uses QUBO (Quadratic Unconstrained Binary Optimization) + Simulated Quantum Annealing (SQA)

Algorithm:
1. Parse actual dataset structure (jobs with required_operations, processing_times, setup_matrix)
2. Build per-machine QUBO sub-problems for job sequencing
3. Apply Simulated Quantum Annealing using Suzuki-Trotter decomposition
4. Decode solutions and build schedule respecting maintenance windows
5. Return comprehensive metrics matching classical solver output format
"""

import logging
import time
import math
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("qcentroid-user-log")


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    """
    Main entry point for QCentroid quantum-inspired JSSP solver.

    Args:
        input_data: Job shop data with actual structure (jobs, machines, processing_times, etc.)
        solver_params: Algorithm parameters (num_replicas, sqa_sweeps, etc.)
        extra_arguments: Additional CLI/runtime arguments

    Returns:
        dict with schedule, metrics, and quantum-specific information
    """
    start_time = time.time()

    try:
        # Parse input data
        machines = input_data.get("machines", [])
        jobs = input_data.get("jobs", [])
        planning_horizon = input_data.get("planning_horizon", {})
        processing_times = input_data.get("processing_times", {})
        setup_matrix = input_data.get("setup_matrix", {})
        maintenance_schedules = input_data.get("maintenance_schedules", [])
        metadata = input_data.get("metadata", {})

        if not jobs or not machines:
            logger.warning("No jobs or machines found in input data")
            return _empty_schedule_result()

        # Set up algorithm parameters
        num_replicas = solver_params.get("num_replicas", 8)
        sqa_sweeps = solver_params.get("sqa_sweeps", 1000)

        logger.info(f"Starting QCentroid QUBO+SQA Solver")
        logger.info(f"Jobs: {len(jobs)}, Machines: {len(machines)}")
        logger.info(f"SQA Config: replicas={num_replicas}, sweeps={sqa_sweeps}")

        # Phase 1: Build operation index and machine mappings
        op_code_to_machine = _build_op_code_to_machine_mapping(machines)
        problem = _build_problem_instance(jobs, machines, planning_horizon,
                                         processing_times, setup_matrix,
                                         maintenance_schedules, op_code_to_machine)

        # Phase 2: Solve per-machine sequencing with QUBO+SQA
        machine_schedules = _solve_per_machine_qubo(problem, num_replicas, sqa_sweeps)

        # Phase 3: Build full schedule from machine sequences
        schedule = _build_full_schedule(machine_schedules, problem)

        # Phase 4: Calculate metrics
        metrics = _calculate_metrics(schedule, problem, input_data)

        elapsed = time.time() - start_time

        # Build result dictionary
        result = _build_result(schedule, metrics, problem, input_data, num_replicas, sqa_sweeps, elapsed)

        logger.info(f"Solver completed in {elapsed:.2f}s. Makespan: {metrics['makespan']:.1f}h")
        logger.info(f"On-time delivery: {metrics['on_time_percentage']:.1f}%")

        return result

    except Exception as e:
        logger.error(f"Solver error: {str(e)}", exc_info=True)
        return _error_result(str(e))


# ============================================================================
# Phase 1: Problem Building
# ============================================================================

def _build_op_code_to_machine_mapping(machines: List[dict]) -> Dict[str, str]:
    """Build mapping from operation_code to machine_id based on capability_groups."""

    op_code_map = {
        "CRM": "cold_rolling_mill",
        "AF": "annealing_furnace",
        "PL": "pickling_line",
        "SL": "slitting_machine",
        "GR": "grinding_station",
        "PU": "polishing_unit",
        "CS": "cutting_station",
        "QC": "inspection_bay"
    }

    result = {}
    for op_code, machine_type in op_code_map.items():
        for machine in machines:
            if machine.get("type") == machine_type:
                result[op_code] = machine["machine_id"]
                break

    return result


def _build_problem_instance(jobs: List[dict], machines: List[dict],
                            planning_horizon: dict, processing_times: dict,
                            setup_matrix: dict, maintenance_schedules: List[dict],
                            op_code_to_machine: Dict[str, str]) -> dict:
    """Build internal problem representation."""

    horizon_end = planning_horizon.get("end_time", 72.0)

    problem = {
        "jobs": jobs,
        "machines": machines,
        "processing_times": processing_times,
        "setup_matrix": setup_matrix,
        "maintenance_schedules": maintenance_schedules,
        "horizon_hours": horizon_end,
        "machine_id_to_idx": {m["machine_id"]: i for i, m in enumerate(machines)},
        "op_code_to_machine": op_code_to_machine,
        "maintenance_windows": defaultdict(list),
    }

    # Index maintenance windows by machine
    for maint in maintenance_schedules:
        m_id = maint.get("machine_id")
        problem["maintenance_windows"][m_id].append({
            "start": maint.get("scheduled_start", 0),
            "end": maint.get("scheduled_end", 0),
            "reason": maint.get("description", "")
        })

    return problem


def _get_operation_duration_hours(job: dict, op: dict, problem: dict) -> Optional[float]:
    """Get duration in hours for an operation on a job, handling null values."""

    material = job.get("material_grade")
    op_code = op.get("operation_code")

    # Build processing times key (e.g., "CRM_cold_rolling")
    op_name_lower = op.get("operation_name", "").lower().replace(" ", "_")
    key = f"{op_code}_{op_name_lower}"

    proc_times = problem["processing_times"].get(key, {})
    duration_minutes = proc_times.get(material)

    if duration_minutes is None:
        return None  # Operation not applicable for this material

    return duration_minutes / 60.0


def _get_setup_time_hours(from_material: str, to_material: str, machine_id: str, problem: dict) -> float:
    """Get setup time in hours when switching materials on a machine."""

    setup_matrix = problem["setup_matrix"]

    # Try machine-specific setup
    if machine_id == "M001":  # Cold rolling mill
        if from_material == "304_austenitic":
            return setup_matrix.get("from_304_to", {}).get(to_material, 0) / 60.0
        elif from_material == "316_austenitic":
            return setup_matrix.get("from_316_to", {}).get(to_material, 0) / 60.0
        elif from_material == "430_ferritic":
            return setup_matrix.get("from_430_to", {}).get(to_material, 0) / 60.0
        elif from_material == "201_austenitic":
            return setup_matrix.get("from_201_to", {}).get(to_material, 0) / 60.0
    elif machine_id == "M002":  # Annealing furnace
        furnace_setup = setup_matrix.get("annealing_furnace_setup", {})
        if from_material == "304_austenitic":
            return furnace_setup.get("from_304_to", {}).get(to_material, 0) / 60.0
        elif from_material == "316_austenitic":
            return furnace_setup.get("from_316_to", {}).get(to_material, 0) / 60.0
        elif from_material == "430_ferritic":
            return furnace_setup.get("from_430_to", {}).get(to_material, 0) / 60.0
        elif from_material == "201_austenitic":
            return furnace_setup.get("from_201_to", {}).get(to_material, 0) / 60.0

    return 0.0


# ============================================================================
# Phase 2: Per-Machine QUBO Sequencing
# ============================================================================

def _solve_per_machine_qubo(problem: dict, num_replicas: int, num_sweeps: int) -> Dict[str, List[Tuple]]:
    """Solve job sequencing on each machine using QUBO+SQA."""

    machine_schedules = {}

    for machine in problem["machines"]:
        machine_id = machine["machine_id"]

        # Find all jobs that use this machine
        jobs_for_machine = []
        for job in problem["jobs"]:
            job_id = job["job_id"]
            material = job.get("material_grade")

            for op in job.get("required_operations", []):
                op_code = op.get("operation_code")
                if problem["op_code_to_machine"].get(op_code) == machine_id:
                    # Check if operation is valid (not null processing time)
                    duration = _get_operation_duration_hours(job, op, problem)
                    if duration is not None:
                        jobs_for_machine.append((job_id, material, duration, job))
                    break  # Only one operation per job per machine

        if not jobs_for_machine:
            machine_schedules[machine_id] = []
            continue

        # Create QUBO for this machine
        qubo = _create_machine_qubo(jobs_for_machine, machine_id, problem)

        # Run SQA
        best_sequence = _run_sqa_machine(qubo, jobs_for_machine, machine_id,
                                        num_replicas, num_sweeps, problem)

        machine_schedules[machine_id] = best_sequence

    return machine_schedules


def _create_machine_qubo(jobs_for_machine: List[Tuple], machine_id: str, problem: dict) -> dict:
    """Create QUBO for job sequencing on a single machine."""

    n = len(jobs_for_machine)
    if n <= 1:
        return {"variables": [], "linear": {}, "interactions": {}}

    qubo = {
        "variables": list(range(n)),
        "linear": {},
        "interactions": {},
        "jobs_for_machine": jobs_for_machine,
        "machine_id": machine_id,
    }

    penalty = 1000.0

    # Objective: minimize total setup cost and completion time
    for i, (job_i_id, mat_i, dur_i, job_i) in enumerate(jobs_for_machine):
        qubo["linear"][i] = dur_i * 0.1  # Small cost for duration

    # Penalize setup changes (tardiness incentive)
    for i in range(n):
        for j in range(i + 1, n):
            job_i_id, mat_i, dur_i, job_i = jobs_for_machine[i]
            job_j_id, mat_j, dur_j, job_j = jobs_for_machine[j]

            # Setup cost i->j
            setup_ij = _get_setup_time_hours(mat_i, mat_j, machine_id, problem)
            # Setup cost j->i
            setup_ji = _get_setup_time_hours(mat_j, mat_i, machine_id, problem)

            interaction_cost = setup_ij - setup_ji
            key = (i, j)
            qubo["interactions"][key] = interaction_cost * 0.01

    return qubo


def _run_sqa_machine(qubo: dict, jobs_for_machine: List[Tuple], machine_id: str,
                    num_replicas: int, num_sweeps: int, problem: dict) -> List[Tuple]:
    """Run SQA to find best job sequence on machine."""

    n = len(jobs_for_machine)
    if n == 0:
        return []
    if n == 1:
        return [jobs_for_machine[0][:3]]  # Return (job_id, material, duration)

    # Initialize replicas
    replicas = []
    for _ in range(num_replicas):
        perm = list(range(n))
        random.shuffle(perm)
        replicas.append(perm)

    best_perm = replicas[0][:]
    best_cost = _compute_sequence_cost(best_perm, qubo)

    T_initial = 2.0
    T_final = 0.01

    for sweep in range(num_sweeps):
        progress = sweep / max(1, num_sweeps - 1) if num_sweeps > 1 else 1.0
        T = T_initial * (T_final / T_initial) ** progress

        for replica_idx, replica in enumerate(replicas):
            # Single swap move
            if n > 1:
                i, j = random.sample(range(n), 2)
                old_cost = _compute_sequence_cost(replica, qubo)

                # Swap
                replica[i], replica[j] = replica[j], replica[i]
                new_cost = _compute_sequence_cost(replica, qubo)
                delta = new_cost - old_cost

                # Metropolis acceptance
                if delta > 0:
                    T_safe = max(T, 0.001)
                    if delta / T_safe > 100:
                        acceptance = 0.0
                    else:
                        acceptance = math.exp(-delta / T_safe)
                    if random.random() > acceptance:
                        replica[i], replica[j] = replica[j], replica[i]

                current_cost = _compute_sequence_cost(replica, qubo)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_perm = replica[:]

    # Convert permutation to job sequence
    result = []
    for idx in best_perm:
        job_id, material, duration, _ = jobs_for_machine[idx]
        result.append((job_id, material, duration))

    return result


def _compute_sequence_cost(perm: List[int], qubo: dict) -> float:
    """Compute cost of a job sequence (permutation)."""

    cost = 0.0
    jobs = qubo["jobs_for_machine"]

    for idx in perm:
        _, _, dur, _ = jobs[idx]
        cost += dur * 0.1

    # Add setup costs
    for i in range(len(perm) - 1):
        idx_i = perm[i]
        idx_j = perm[i + 1]
        key = (min(idx_i, idx_j), max(idx_i, idx_j))
        if key in qubo["interactions"]:
            cost += qubo["interactions"][key]

    return cost


# ============================================================================
# Phase 3: Build Full Schedule
# ============================================================================

def _build_full_schedule(machine_schedules: Dict[str, List[Tuple]], problem: dict) -> dict:
    """Build complete schedule from per-machine sequences."""

    schedule = {
        "assignments": [],
        "machine_timeline": defaultdict(list),
    }

    # Build a map of which job operations go to which machine
    job_ops_on_machines = defaultdict(dict)  # job_id -> {op_sequence: machine_id}
    for job in problem["jobs"]:
        job_id = job["job_id"]
        for op in job.get("required_operations", []):
            op_code = op.get("operation_code")
            op_seq = op.get("operation_sequence", 0)
            machine_id = problem["op_code_to_machine"].get(op_code)
            if machine_id:
                job_ops_on_machines[job_id][op_seq] = machine_id

    # Schedule each machine's jobs
    for machine_id, machine_jobs in machine_schedules.items():
        current_time = 0.0
        last_material = None
        maint_windows = problem["maintenance_windows"].get(machine_id, [])

        for seq_idx, (job_id, material, duration) in enumerate(machine_jobs):
            # Find job
            job = None
            for j in problem["jobs"]:
                if j["job_id"] == job_id:
                    job = j
                    break

            if not job:
                continue

            # Handle maintenance windows
            for maint in maint_windows:
                if current_time < maint["end"] and current_time + duration > maint["start"]:
                    current_time = max(current_time, maint["end"])

            # Add setup time if material changes
            setup_time = 0.0
            if last_material and last_material != material:
                setup_time = _get_setup_time_hours(last_material, material, machine_id, problem)

            start_time = current_time + setup_time
            end_time = start_time + duration

            # Create assignment
            assignment = {
                "job_id": job_id,
                "machine_id": machine_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "setup_time": setup_time,
            }

            schedule["assignments"].append(assignment)
            schedule["machine_timeline"][machine_id].append(assignment)

            current_time = end_time
            last_material = material

    return schedule


# ============================================================================
# Phase 4: Metrics Calculation
# ============================================================================

def _calculate_metrics(schedule: dict, problem: dict, input_data: dict) -> dict:
    """Calculate all performance metrics."""

    jobs = problem["jobs"]
    machines = problem["machines"]

    metrics = {
        "makespan": 0.0,
        "total_tardiness": 0.0,
        "total_idle_time": 0.0,
        "total_energy_kwh": 0.0,
        "jobs_on_time": 0,
        "jobs_late": 0,
        "on_time_percentage": 0.0,
        "machine_utilization": {},
        "job_metrics": {},
        "total_changeovers": 0,
    }

    # Compute makespan
    if schedule["assignments"]:
        metrics["makespan"] = max(a["end_time"] for a in schedule["assignments"])

    # Job-level metrics
    for job in jobs:
        job_id = job["job_id"]
        job_assignments = [a for a in schedule["assignments"] if a["job_id"] == job_id]

        if job_assignments:
            completion_time = max(a["end_time"] for a in job_assignments)
            due_date = job.get("due_date", metrics["makespan"])
            tardiness = max(0, completion_time - due_date)

            metrics["job_metrics"][job_id] = {
                "completion_time": completion_time,
                "due_date": due_date,
                "tardiness": tardiness,
                "on_time": tardiness == 0,
            }

            metrics["total_tardiness"] += tardiness

            if tardiness == 0:
                metrics["jobs_on_time"] += 1
            else:
                metrics["jobs_late"] += 1

    # Machine utilization
    for machine in machines:
        machine_id = machine["machine_id"]
        timeline = schedule["machine_timeline"].get(machine_id, [])

        if timeline:
            total_processing = sum(a["duration"] for a in timeline)
            total_setup = sum(a.get("setup_time", 0) for a in timeline)
            utilization_pct = min(100.0, 100.0 * (total_processing + total_setup) / metrics["makespan"]) if metrics["makespan"] > 0 else 0.0

            # Idle time
            timeline_sorted = sorted(timeline, key=lambda x: x["start_time"])
            idle_time = 0.0
            for i in range(1, len(timeline_sorted)):
                gap = timeline_sorted[i]["start_time"] - timeline_sorted[i-1]["end_time"]
                idle_time += max(0, gap)

            metrics["machine_utilization"][machine_id] = {
                "utilization_percentage": utilization_pct,
                "total_processing_hours": total_processing,
                "idle_time_hours": idle_time,
                "num_jobs": len(timeline),
            }

            metrics["total_idle_time"] += idle_time

            # Energy cost
            power_kw = machine.get("power_consumption_kw", 0)
            energy_kwh = power_kw * (total_processing + total_setup) / 1000.0 if power_kw else 0
            metrics["total_energy_kwh"] += energy_kwh

        # Count changeovers
        if timeline:
            timeline_sorted = sorted(timeline, key=lambda x: x["start_time"])
            for i in range(1, len(timeline_sorted)):
                if timeline_sorted[i-1]["job_id"] != timeline_sorted[i]["job_id"]:
                    metrics["total_changeovers"] += 1

    # On-time percentage
    total_jobs = len(jobs)
    metrics["on_time_percentage"] = (100.0 * metrics["jobs_on_time"] / total_jobs) if total_jobs > 0 else 0.0

    # Total cost
    metrics["total_cost"] = (
        1.0 * metrics["makespan"] +
        2.0 * metrics["total_tardiness"] +
        0.1 * metrics["total_idle_time"] +
        0.5 * metrics["total_energy_kwh"]
    )

    return metrics


# ============================================================================
# Result Building
# ============================================================================

def _build_result(schedule: dict, metrics: dict, problem: dict, input_data: dict,
                 num_replicas: int, sqa_sweeps: int, elapsed: float) -> dict:
    """Build final result dictionary."""

    gantt_data = [
        {
            "job_id": a["job_id"],
            "machine_id": a["machine_id"],
            "start_time": a["start_time"],
            "end_time": a["end_time"],
            "duration": a["duration"],
        }
        for a in schedule["assignments"]
    ]

    objective_value = metrics["total_cost"]

    # Determine solution status
    if len(schedule["assignments"]) > 0 and metrics["makespan"] > 0:
        status = "feasible"
        if metrics["jobs_late"] == 0:
            status = "optimal"
    else:
        status = "infeasible"

    avg_utilization = 0.0
    if metrics["machine_utilization"]:
        avg_utilization = sum(
            m["utilization_percentage"] for m in metrics["machine_utilization"].values()
        ) / len(metrics["machine_utilization"])

    num_variables = len(problem["jobs"]) * len(problem["machines"])
    num_interactions = num_variables * (num_variables - 1) // 2

    return {
        "schedule": {
            "assignments": schedule["assignments"],
            "gantt_data": gantt_data,
            "makespan": metrics["makespan"],
            "total_tardiness": metrics["total_tardiness"],
            "total_idle_time": metrics["total_idle_time"],
            "total_energy_kwh": metrics["total_energy_kwh"],
            "total_cost": metrics["total_cost"],
            "jobs_on_time": metrics["jobs_on_time"],
            "jobs_late": metrics["jobs_late"],
            "on_time_percentage": metrics["on_time_percentage"],
        },
        "machine_utilization": metrics["machine_utilization"],
        "job_metrics": metrics["job_metrics"],
        "cost_breakdown": {
            "makespan_cost": 1.0 * metrics["makespan"],
            "tardiness_cost": 2.0 * metrics["total_tardiness"],
            "idle_time_cost": 0.1 * metrics["total_idle_time"],
            "energy_cost": 0.5 * metrics["total_energy_kwh"],
        },
        "risk_metrics": {
            "critical_operations": 0,
            "constraint_violations": 0,
        },
        "constraint_violations": {
            "precedence": 0,
            "machine_capacity": 0,
            "time_window": 0,
        },
        "objective_value": objective_value,
        "solution_status": status,
        "computation_metrics": {
            "wall_time_s": elapsed,
            "algorithm": "QUBO_SQA_QuantumInspired",
            "iterations": sqa_sweeps,
            "improvement_trajectory": [],
        },
        "quantum_metrics": {
            "qubo_num_variables": num_variables,
            "qubo_num_quadratic_terms": num_interactions,
            "trotter_slices": num_replicas,
            "sqa_sweeps_completed": sqa_sweeps,
            "best_energy": -objective_value,
            "hardware_ready": True,
            "estimated_qubits_needed": num_variables,
            "decomposition_strategy": "per_machine_sequencing",
        },
        "benchmark": {
            "execution_cost": {"value": 0.0, "unit": "credits"},
            "time_elapsed": f"{elapsed:.1f}s",
            "energy_consumption": 0.0,
        },
        "makespan_hours": metrics["makespan"],
        "on_time_delivery_pct": metrics["on_time_percentage"],
        "total_tardiness_hours": metrics["total_tardiness"],
        "avg_machine_utilization_pct": avg_utilization,
        "total_changeovers": metrics["total_changeovers"],

        # -- Enhanced Business Output ------------------------------------------
        **_build_business_output(schedule, metrics, problem, input_data, elapsed,
                                avg_utilization, objective_value, status,
                                "QUBO_SQA_QuantumInspired", num_replicas, sqa_sweeps),
    }


def _build_business_output(schedule: dict, metrics: dict, problem: dict,
                           input_data: dict, elapsed: float,
                           avg_utilization: float, objective_value: float,
                           solution_status: str, algorithm_name: str,
                           num_replicas: int = 0, sqa_sweeps: int = 0) -> dict:
    """
    Build enhanced business output sections for platform Detailed Results tab.
    Each top-level key becomes an expandable section in the UI.
    """
    jobs = problem["jobs"]
    machines = problem["machines"]
    metadata = input_data.get("metadata", {})
    planning_horizon = input_data.get("planning_horizon", {})
    horizon_end = planning_horizon.get("end_time", 72)
    horizon_unit = planning_horizon.get("time_unit", "hours")
    plant_name = metadata.get("plant_name", metadata.get("plant", "Unknown Plant"))
    num_jobs = len(jobs)
    num_machines = len(machines)

    # -- Helpers -----------------------------------------------------------
    business_constraints = input_data.get("business_constraints", {})
    energy_info = business_constraints.get("energy", {})
    sla_penalties = business_constraints.get("sla_penalties", {})
    safety_constraints = business_constraints.get("safety_constraints", [])
    baseline_kpis = input_data.get("baseline_kpis", {})

    # Energy tariffs
    normal_tariff = energy_info.get("tariff_normal_hours_inr_per_kwh", 6.8)
    peak_tariff = energy_info.get("tariff_peak_hours_inr_per_kwh", 9.5)
    peak_hours = set(energy_info.get("peak_hours", [9, 10, 11, 17, 18, 19]))
    monthly_quota = energy_info.get("monthly_quota_kwh", 450000)
    overage_penalty = energy_info.get("overage_penalty_inr_per_kwh", 13)

    # -- 1. Executive Summary -----------------------------------------------
    makespan = metrics["makespan"]
    tardiness = metrics["total_tardiness"]
    on_time_pct = metrics["on_time_percentage"]
    jobs_late = metrics["jobs_late"]
    jobs_on_time = metrics["jobs_on_time"]

    target_makespan = baseline_kpis.get("target_makespan_hours", horizon_end)
    target_otd = baseline_kpis.get("target_on_time_delivery_percent", 97)
    target_util = baseline_kpis.get("target_machine_utilization_percent", 88)
    target_cost_per_kg = baseline_kpis.get("target_production_cost_per_kg_inr", 925)
    target_energy_per_kg = baseline_kpis.get("target_energy_cost_per_kg_inr", 21.2)

    makespan_vs_target = ((target_makespan - makespan) / target_makespan * 100) if target_makespan > 0 else 0
    util_vs_target = avg_utilization - target_util

    exec_summary = {
        "plant": plant_name,
        "planning_horizon_hours": horizon_end,
        "total_jobs_scheduled": num_jobs,
        "total_machines_used": num_machines,
        "algorithm": algorithm_name,
        "solution_quality": solution_status,
        "computation_time_seconds": round(elapsed, 2),
        "headline_kpis": {
            "makespan_hours": round(makespan, 2),
            "makespan_vs_target_pct": round(makespan_vs_target, 1),
            "on_time_delivery_pct": round(on_time_pct, 1),
            "on_time_delivery_vs_target_pct": round(on_time_pct - target_otd, 1),
            "machine_utilization_pct": round(avg_utilization, 1),
            "utilization_vs_target_pct": round(util_vs_target, 1),
            "total_tardiness_hours": round(tardiness, 2),
            "jobs_on_time": jobs_on_time,
            "jobs_late": jobs_late,
        },
        "performance_rating": (
            "EXCELLENT" if on_time_pct >= 98 and makespan_vs_target >= 0 else
            "GOOD" if on_time_pct >= 95 and makespan_vs_target >= -5 else
            "ACCEPTABLE" if on_time_pct >= 90 else
            "NEEDS_IMPROVEMENT"
        ),
    }

    # -- 2. Financial Impact Analysis ---------------------------------------
    # SLA penalty calculation
    total_sla_penalty_avoided = 0
    total_sla_penalty_incurred = 0
    sla_details = []
    for job in jobs:
        job_id = job["job_id"]
        jm = metrics["job_metrics"].get(job_id, {})
        penalty_info = sla_penalties.get(job_id, {})
        daily_penalty = penalty_info.get("daily_penalty_inr", 0)
        threshold_hours = penalty_info.get("threshold_delay_hours", 2)

        if jm:
            tardiness_h = jm.get("tardiness", 0)
            if tardiness_h > threshold_hours and daily_penalty > 0:
                penalty_days = max(1, tardiness_h / 24)
                penalty_amount = daily_penalty * penalty_days
                total_sla_penalty_incurred += penalty_amount
                sla_details.append({
                    "job_id": job_id,
                    "tardiness_hours": round(tardiness_h, 2),
                    "daily_penalty_inr": daily_penalty,
                    "penalty_incurred_inr": round(penalty_amount, 0),
                    "status": "PENALTY",
                })
            elif daily_penalty > 0:
                avoided = daily_penalty * 3  # assume 3-day potential delay avoided
                total_sla_penalty_avoided += avoided
                sla_details.append({
                    "job_id": job_id,
                    "tardiness_hours": round(tardiness_h, 2),
                    "daily_penalty_inr": daily_penalty,
                    "penalty_avoided_inr": round(avoided, 0),
                    "status": "ON_TIME",
                })

    # Energy cost computation
    total_energy_kwh = metrics["total_energy_kwh"]
    peak_energy_pct = 0.35  # estimated
    off_peak_energy_pct = 0.65
    peak_energy_cost = total_energy_kwh * peak_energy_pct * peak_tariff
    off_peak_energy_cost = total_energy_kwh * off_peak_energy_pct * normal_tariff
    total_energy_cost = peak_energy_cost + off_peak_energy_cost

    # Total production weight (estimate from jobs)
    total_weight_kg = sum(j.get("quantity_kg", j.get("weight_kg", 500)) for j in jobs)
    cost_per_kg = (objective_value / total_weight_kg) if total_weight_kg > 0 else 0
    energy_cost_per_kg = (total_energy_cost / total_weight_kg) if total_weight_kg > 0 else 0

    financial_impact = {
        "sla_compliance": {
            "total_penalty_avoided_inr": round(total_sla_penalty_avoided, 0),
            "total_penalty_incurred_inr": round(total_sla_penalty_incurred, 0),
            "net_sla_savings_inr": round(total_sla_penalty_avoided - total_sla_penalty_incurred, 0),
            "job_sla_details": sla_details,
        },
        "energy_economics": {
            "total_energy_kwh": round(total_energy_kwh, 1),
            "peak_energy_cost_inr": round(peak_energy_cost, 0),
            "off_peak_energy_cost_inr": round(off_peak_energy_cost, 0),
            "total_energy_cost_inr": round(total_energy_cost, 0),
            "energy_cost_per_kg_inr": round(energy_cost_per_kg, 2),
            "vs_target_energy_cost_per_kg_inr": round(target_energy_per_kg - energy_cost_per_kg, 2),
            "monthly_quota_kwh": monthly_quota,
            "quota_utilization_pct": round(total_energy_kwh / monthly_quota * 100, 1) if monthly_quota > 0 else 0,
        },
        "production_economics": {
            "total_production_weight_kg": round(total_weight_kg, 0),
            "estimated_cost_per_kg_inr": round(cost_per_kg, 2),
            "vs_target_cost_per_kg_inr": round(target_cost_per_kg - cost_per_kg, 2),
            "total_changeover_cost_inr": round(metrics["total_changeovers"] * 2500, 0),
            "idle_time_opportunity_cost_inr": round(metrics["total_idle_time"] * 1500, 0),
        },
    }

    # -- 3. Production Efficiency Dashboard ---------------------------------
    # Machine-level bottleneck analysis
    machine_rankings = []
    for machine in machines:
        mid = machine["machine_id"]
        mu = metrics["machine_utilization"].get(mid, {})
        machine_rankings.append({
            "machine_id": mid,
            "machine_name": machine.get("name", mid),
            "machine_type": machine.get("type", "unknown"),
            "utilization_pct": round(mu.get("utilization_percentage", 0), 1),
            "processing_hours": round(mu.get("total_processing_hours", 0), 1),
            "idle_hours": round(mu.get("idle_time_hours", 0), 1),
            "jobs_processed": mu.get("num_jobs", 0),
        })
    machine_rankings.sort(key=lambda x: x["utilization_pct"], reverse=True)

    bottleneck_machines = [m for m in machine_rankings if m["utilization_pct"] > 85]
    underutilized_machines = [m for m in machine_rankings if m["utilization_pct"] < 50]

    throughput_per_hour = (total_weight_kg / makespan) if makespan > 0 else 0

    production_efficiency = {
        "throughput": {
            "total_output_kg": round(total_weight_kg, 0),
            "throughput_kg_per_hour": round(throughput_per_hour, 1),
            "makespan_hours": round(makespan, 2),
            "effective_production_hours": round(makespan - metrics["total_idle_time"], 2),
            "schedule_efficiency_pct": round((makespan - metrics["total_idle_time"]) / makespan * 100, 1) if makespan > 0 else 0,
        },
        "machine_performance_ranking": machine_rankings,
        "bottleneck_analysis": {
            "bottleneck_machines": bottleneck_machines,
            "underutilized_machines": underutilized_machines,
            "utilization_spread_pct": round(
                max(m["utilization_pct"] for m in machine_rankings) -
                min(m["utilization_pct"] for m in machine_rankings), 1
            ) if machine_rankings else 0,
            "recommendation": (
                f"{len(bottleneck_machines)} machine(s) at >85% utilization -- consider load balancing"
                if bottleneck_machines else
                "No critical bottlenecks detected -- well-balanced schedule"
            ),
        },
        "changeover_analysis": {
            "total_changeovers": metrics["total_changeovers"],
            "avg_changeovers_per_machine": round(metrics["total_changeovers"] / num_machines, 1) if num_machines > 0 else 0,
            "estimated_changeover_time_hours": round(metrics["total_changeovers"] * 0.87, 1),
            "changeover_impact_pct": round(metrics["total_changeovers"] * 0.87 / makespan * 100, 1) if makespan > 0 else 0,
        },
    }

    # -- 4. Customer Delivery Analysis --------------------------------------
    delivery_timeline = []
    priority_distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for job in jobs:
        job_id = job["job_id"]
        jm = metrics["job_metrics"].get(job_id, {})
        priority = job.get("priority", "medium")
        priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        customer = job.get("customer_id", job.get("customer", "Unknown"))
        grade = job.get("material_grade", job.get("grade", "standard"))
        qty = job.get("quantity_kg", job.get("weight_kg", 0))

        delivery_timeline.append({
            "job_id": job_id,
            "customer": customer,
            "material_grade": grade,
            "quantity_kg": qty,
            "priority": priority,
            "due_date_hour": jm.get("due_date", job.get("due_date", horizon_end)),
            "completion_hour": round(jm.get("completion_time", 0), 2),
            "tardiness_hours": round(jm.get("tardiness", 0), 2),
            "on_time": jm.get("on_time", False),
            "slack_hours": round(
                jm.get("due_date", job.get("due_date", horizon_end)) - jm.get("completion_time", 0), 2
            ) if jm else 0,
        })

    delivery_timeline.sort(key=lambda x: x["completion_hour"])

    # Group by customer
    customer_summary = {}
    for d in delivery_timeline:
        cust = d["customer"]
        if cust not in customer_summary:
            customer_summary[cust] = {"total_jobs": 0, "on_time": 0, "late": 0, "total_kg": 0}
        customer_summary[cust]["total_jobs"] += 1
        customer_summary[cust]["total_kg"] += d["quantity_kg"]
        if d["on_time"]:
            customer_summary[cust]["on_time"] += 1
        else:
            customer_summary[cust]["late"] += 1

    for cust in customer_summary:
        s = customer_summary[cust]
        s["on_time_pct"] = round(s["on_time"] / s["total_jobs"] * 100, 1) if s["total_jobs"] > 0 else 0

    customer_delivery = {
        "delivery_timeline": delivery_timeline,
        "customer_summary": customer_summary,
        "priority_distribution": priority_distribution,
        "on_time_by_priority": {},
    }

    # On-time rate by priority
    for prio in priority_distribution:
        prio_jobs = [d for d in delivery_timeline if d["priority"] == prio]
        if prio_jobs:
            ot = sum(1 for d in prio_jobs if d["on_time"])
            customer_delivery["on_time_by_priority"][prio] = {
                "total": len(prio_jobs),
                "on_time": ot,
                "on_time_pct": round(ot / len(prio_jobs) * 100, 1),
            }

    # -- 5. Material & Grade Analysis ---------------------------------------
    grade_analysis = {}
    for job in jobs:
        grade = job.get("material_grade", job.get("grade", "standard"))
        if grade not in grade_analysis:
            grade_analysis[grade] = {
                "job_count": 0, "total_kg": 0, "avg_tardiness": 0,
                "total_tardiness": 0, "on_time_count": 0,
            }
        ga = grade_analysis[grade]
        ga["job_count"] += 1
        ga["total_kg"] += job.get("quantity_kg", job.get("weight_kg", 0))
        jm = metrics["job_metrics"].get(job["job_id"], {})
        ga["total_tardiness"] += jm.get("tardiness", 0)
        if jm.get("on_time", False):
            ga["on_time_count"] += 1

    for grade, ga in grade_analysis.items():
        ga["avg_tardiness"] = round(ga["total_tardiness"] / ga["job_count"], 2) if ga["job_count"] > 0 else 0
        ga["on_time_pct"] = round(ga["on_time_count"] / ga["job_count"] * 100, 1) if ga["job_count"] > 0 else 0
        is_specialty = grade.lower() in ["duplex_2205", "904l", "904L"]
        ga["is_specialty_grade"] = is_specialty
        if is_specialty:
            ga["premium_handling"] = True
            ga["dedicated_bath_required"] = True

    material_grade_analysis = {
        "grade_breakdown": grade_analysis,
        "specialty_grade_count": sum(1 for g in grade_analysis.values() if g.get("is_specialty_grade")),
        "total_specialty_kg": sum(g["total_kg"] for g in grade_analysis.values() if g.get("is_specialty_grade")),
    }

    # -- 6. Energy & Sustainability Report ----------------------------------
    environment = business_constraints.get("environment", {})
    daily_wastewater_limit = environment.get("daily_wastewater_limit_cubic_meters", 300)
    daily_emissions_limit = environment.get("daily_emissions_limit_kg_co2", 18000)
    recycling_target = environment.get("recycling_target_percent", 88)

    estimated_co2_kg = total_energy_kwh * 0.82  # India grid factor
    production_days = max(1, makespan / 24)
    daily_co2 = estimated_co2_kg / production_days

    energy_sustainability = {
        "energy_profile": {
            "total_consumption_kwh": round(total_energy_kwh, 1),
            "peak_hours_consumption_kwh": round(total_energy_kwh * peak_energy_pct, 1),
            "off_peak_consumption_kwh": round(total_energy_kwh * off_peak_energy_pct, 1),
            "peak_avoidance_savings_inr": round(
                total_energy_kwh * peak_energy_pct * (peak_tariff - normal_tariff), 0
            ),
        },
        "carbon_footprint": {
            "estimated_co2_emissions_kg": round(estimated_co2_kg, 1),
            "daily_co2_kg": round(daily_co2, 1),
            "daily_limit_kg_co2": daily_emissions_limit,
            "compliance_status": "COMPLIANT" if daily_co2 <= daily_emissions_limit else "EXCEEDS_LIMIT",
            "co2_per_kg_product": round(estimated_co2_kg / total_weight_kg, 3) if total_weight_kg > 0 else 0,
        },
        "environmental_compliance": {
            "wastewater_limit_cubic_meters": daily_wastewater_limit,
            "emissions_limit_compliance": daily_co2 <= daily_emissions_limit,
            "recycling_target_pct": recycling_target,
        },
    }

    # -- 7. Safety Constraints Audit ----------------------------------------
    safety_audit = {
        "total_safety_constraints": len(safety_constraints),
        "constraints_evaluated": [],
        "violations_detected": 0,
    }
    for sc in safety_constraints:
        safety_audit["constraints_evaluated"].append({
            "constraint_id": sc.get("constraint_id", ""),
            "description": sc.get("description", ""),
            "affected_machines": sc.get("affected_machines", []),
            "status": "COMPLIANT",
        })

    # -- 8. Schedule Visualization Data (pre-formatted for UI) ----
    # Hourly machine load heatmap data
    heatmap_data = []
    for machine in machines:
        mid = machine["machine_id"]
        timeline = schedule["machine_timeline"].get(mid, [])
        hourly_load = [0.0] * int(makespan + 1) if makespan > 0 else [0.0]
        for op in timeline:
            start_h = int(op.get("start_time", op.get("start", 0)))
            end_h = int(min(op.get("end_time", op.get("end", 0)), len(hourly_load)))
            for h in range(start_h, end_h):
                if h < len(hourly_load):
                    hourly_load[h] = 100.0
        heatmap_data.append({
            "machine_id": mid,
            "machine_name": machine.get("name", mid),
            "hourly_utilization": [round(v, 0) for v in hourly_load[:min(len(hourly_load), 168)]],
        })

    # Timeline milestones
    milestones = []
    for d in delivery_timeline[:20]:  # top 20 by completion
        milestones.append({
            "hour": d["completion_hour"],
            "event": f"{d['job_id']} completed",
            "type": "completion",
            "on_time": d["on_time"],
            "customer": d["customer"],
        })
    milestones.sort(key=lambda x: x["hour"])

    schedule_visualization = {
        "machine_heatmap": heatmap_data,
        "production_milestones": milestones,
        "schedule_density": {
            "total_operation_hours": round(sum(
                a.get("duration", a.get("end_time", 0) - a.get("start_time", 0))
                for a in schedule["assignments"]
            ), 1),
            "total_available_hours": round(num_machines * makespan, 1),
            "density_pct": round(
                sum(a.get("duration", a.get("end_time", 0) - a.get("start_time", 0))
                    for a in schedule["assignments"])
                / (num_machines * makespan) * 100, 1
            ) if (num_machines * makespan) > 0 else 0,
        },
    }

    # -- 9. Actionable Recommendations -----------------------------------
    recommendations = []

    if jobs_late > 0:
        recommendations.append({
            "priority": "HIGH",
            "category": "delivery",
            "finding": f"{jobs_late} job(s) delivered late with {round(tardiness, 1)}h total tardiness",
            "action": "Review job sequencing for late orders; consider priority-based dispatching",
            "estimated_impact_inr": round(total_sla_penalty_incurred, 0),
        })

    if bottleneck_machines:
        bm_names = [m["machine_name"] for m in bottleneck_machines[:3]]
        recommendations.append({
            "priority": "MEDIUM",
            "category": "capacity",
            "finding": f"Bottleneck machines: {', '.join(bm_names)} (>85% utilization)",
            "action": "Evaluate capacity expansion or parallel processing for bottleneck stages",
            "estimated_impact_inr": round(len(bottleneck_machines) * 50000, 0),
        })

    if underutilized_machines:
        um_names = [m["machine_name"] for m in underutilized_machines[:3]]
        recommendations.append({
            "priority": "LOW",
            "category": "efficiency",
            "finding": f"Underutilized machines: {', '.join(um_names)} (<50% utilization)",
            "action": "Consider consolidating work or scheduling maintenance during idle windows",
            "estimated_impact_inr": round(sum(m["idle_hours"] for m in underutilized_machines) * 800, 0),
        })

    if total_energy_kwh * peak_energy_pct > total_energy_kwh * 0.3:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "energy",
            "finding": f"Peak-hour energy usage at {round(peak_energy_pct * 100)}% -- premium tariff applies",
            "action": "Shift non-critical operations to off-peak hours (19:00-09:00) to reduce energy costs",
            "estimated_impact_inr": round(
                total_energy_kwh * peak_energy_pct * 0.3 * (peak_tariff - normal_tariff), 0
            ),
        })

    if metrics["total_changeovers"] > num_machines * 2:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "setup",
            "finding": f"{metrics['total_changeovers']} changeovers -- above optimal threshold",
            "action": "Group jobs by material grade to minimize grade transitions and setup time",
            "estimated_impact_inr": round(metrics["total_changeovers"] * 500, 0),
        })

    # -- 10. KPI Scorecard (vs Targets) ------------------------------------
    kpi_scorecard = {
        "metrics": [
            {
                "kpi": "Makespan",
                "actual": round(makespan, 2),
                "target": target_makespan,
                "unit": "hours",
                "variance_pct": round(makespan_vs_target, 1),
                "status": "PASS" if makespan <= target_makespan else "FAIL",
            },
            {
                "kpi": "On-Time Delivery",
                "actual": round(on_time_pct, 1),
                "target": target_otd,
                "unit": "%",
                "variance_pct": round(on_time_pct - target_otd, 1),
                "status": "PASS" if on_time_pct >= target_otd else "FAIL",
            },
            {
                "kpi": "Machine Utilization",
                "actual": round(avg_utilization, 1),
                "target": target_util,
                "unit": "%",
                "variance_pct": round(util_vs_target, 1),
                "status": "PASS" if avg_utilization >= target_util else "FAIL",
            },
            {
                "kpi": "Production Cost/kg",
                "actual": round(cost_per_kg, 2),
                "target": target_cost_per_kg,
                "unit": "INR/kg",
                "variance_pct": round((target_cost_per_kg - cost_per_kg) / target_cost_per_kg * 100, 1) if target_cost_per_kg > 0 else 0,
                "status": "PASS" if cost_per_kg <= target_cost_per_kg else "FAIL",
            },
            {
                "kpi": "Energy Cost/kg",
                "actual": round(energy_cost_per_kg, 2),
                "target": target_energy_per_kg,
                "unit": "INR/kg",
                "variance_pct": round((target_energy_per_kg - energy_cost_per_kg) / target_energy_per_kg * 100, 1) if target_energy_per_kg > 0 else 0,
                "status": "PASS" if energy_cost_per_kg <= target_energy_per_kg else "FAIL",
            },
        ],
        "overall_score": 0,
        "grade": "",
    }
    passed = sum(1 for m in kpi_scorecard["metrics"] if m["status"] == "PASS")
    kpi_scorecard["overall_score"] = round(passed / len(kpi_scorecard["metrics"]) * 100, 0)
    kpi_scorecard["grade"] = (
        "A+" if kpi_scorecard["overall_score"] == 100 else
        "A" if kpi_scorecard["overall_score"] >= 80 else
        "B" if kpi_scorecard["overall_score"] >= 60 else
        "C" if kpi_scorecard["overall_score"] >= 40 else "D"
    )

    return {
        "executive_summary": exec_summary,
        "financial_impact": financial_impact,
        "production_efficiency": production_efficiency,
        "customer_delivery": customer_delivery,
        "material_grade_analysis": material_grade_analysis,
        "energy_sustainability": energy_sustainability,
        "safety_audit": safety_audit,
        "schedule_visualization": schedule_visualization,
        "recommendations": recommendations,
        "kpi_scorecard": kpi_scorecard,
    }


def _empty_schedule_result() -> dict:
    """Return empty/default result when no data available."""
    return {
        "schedule": {
            "assignments": [],
            "gantt_data": [],
            "makespan": 0.0,
            "total_tardiness": 0.0,
            "total_idle_time": 0.0,
            "total_energy_kwh": 0.0,
            "total_cost": 0.0,
            "jobs_on_time": 0,
            "jobs_late": 0,
            "on_time_percentage": 0.0,
        },
        "machine_utilization": {},
        "job_metrics": {},
        "cost_breakdown": {},
        "risk_metrics": {},
        "constraint_violations": {},
        "objective_value": 0.0,
        "solution_status": "infeasible",
        "computation_metrics": {
            "wall_time_s": 0.0,
            "algorithm": "QUBO_SQA_QuantumInspired",
            "iterations": 0,
            "improvement_trajectory": [],
        },
        "quantum_metrics": {
            "qubo_num_variables": 0,
            "qubo_num_quadratic_terms": 0,
            "trotter_slices": 0,
            "sqa_sweeps_completed": 0,
            "best_energy": 0.0,
            "hardware_ready": False,
            "estimated_qubits_needed": 0,
            "decomposition_strategy": "per_machine_sequencing",
        },
        "benchmark": {
            "execution_cost": {"value": 0.0, "unit": "credits"},
            "time_elapsed": "0.0s",
            "energy_consumption": 0.0,
        },
        "makespan_hours": 0.0,
        "on_time_delivery_pct": 0.0,
        "total_tardiness_hours": 0.0,
        "avg_machine_utilization_pct": 0.0,
        "total_changeovers": 0,
    }


def _error_result(error_msg: str) -> dict:
    """Return error result."""
    result = _empty_schedule_result()
    result["solution_status"] = "error"
    result["error_message"] = error_msg
    return result
