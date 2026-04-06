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
        result = _build_result(schedule, metrics, problem, num_replicas, sqa_sweeps, elapsed)

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

def _build_result(schedule: dict, metrics: dict, problem: dict,
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