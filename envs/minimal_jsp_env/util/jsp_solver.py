from util.solver import Solver
from util.object_factory import ObjectFactory

import collections
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from typing import List


class JSPSolverFactory(ObjectFactory):
    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


class JSPSolver(Solver):
    def solve_from_job_list(self, job_list: List):
        # Getting job format compatible with Google OR input
        google_or_instance_format = []
        for job in job_list:
            job_ops = [[i.machine_type, i.duration] for i in job]
            google_or_instance_format.append(job_ops)

        opt_time = self._solve_jsp(google_or_instance_format)
        return opt_time
    
    def solve(self, instance):
        solutions = dict()
        if instance.opt_time:
            solutions['opt'] = instance.opt_time
        else:
            solutions['opt'] = self.solve_from_job_list(instance.jobs)

        if instance.spt_time:
            solutions['spt'] = instance.spt_time

        return solutions

    @staticmethod
    def _solve_jsp(jobs_data, max_time_in_seconds=7200, verbose=False):
        """Minimal jobshop problem."""
        # Create the model.
        model = cp_model.CpModel()

        machines_count = 1 + max(task[0] for job in jobs_data for task in job)
        all_machines = range(machines_count)

        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in jobs_data for task in job)

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple('assigned_task_type',
                                                    'start job index duration')

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(
                    start=start_var, end=end_var, interval=interval_var)
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in all_machines:
            model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id +
                                    1].start >= all_tasks[job_id, task_id].end)

        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [
            all_tasks[job_id, len(job) - 1].end
            for job_id, job in enumerate(jobs_data)
        ])
        model.Minimize(obj_var)

        # Solve model.
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time_in_seconds
        status = solver.Solve(model)

        if (status == cp_model.OPTIMAL) or (status == cp_model.FEASIBLE):
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.Value(all_tasks[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[1]))

            # Create per machine output lines.
            output = ''
            gantt_data = []

            for machine in all_machines:
                # Sort by starting time.
                assigned_jobs[machine].sort()
                sol_line_tasks = 'Machine ' + str(machine) + ': '
                sol_line = '           '

                for assigned_task in assigned_jobs[machine]:
                    name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                    # Add spaces to output to align columns.
                    sol_line_tasks += '%-10s' % name

                    start = assigned_task.start
                    duration = assigned_task.duration
                    sol_tmp = '[%i,%i]' % (start, start + duration)
                    # Add spaces to output to align columns.
                    sol_line += '%-10s' % sol_tmp

                    # add Job_num,Mach_num,Job_time,Start_time
                    gantt_data.append([assigned_task.job, machine, duration, start])

                sol_line += '\n'
                sol_line_tasks += '\n'
                output += sol_line_tasks
                output += sol_line

            gantt_data = pd.DataFrame(data=gantt_data, columns=["Job_num", "Mach_num", "Job_time", "Start_time"])

            # Finally print the solution found.
            optimal_time = solver.ObjectiveValue()
            if verbose:
                print("################# OR Solver Solution #################")
                print('Optimal Schedule Length: %i' % optimal_time)
                print(output)
                print("#######################################################")
            status_dict = {4: 'OPTIMAL', 2: 'FEASIBLE'}
            return optimal_time#, output, gantt_data, status_dict[status]



class SPTSolver(Solver):
    def solve(self, instance):
        pass

jsp_solvers = JSPSolverFactory()
jsp_solvers.register_builder('opt', JSPSolver)

