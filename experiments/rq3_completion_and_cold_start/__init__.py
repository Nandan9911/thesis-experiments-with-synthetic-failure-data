"""RQ3: completing partially observed failure records, and the value of synthetic rows when failures are scarce.

completion_engines.py holds six ways to fill in the missing attributes of a failure record. They are compared in
run_completion_benchmark.py, checked for copying in run_completion_memorisation_audit.py and shown on a single
record by the prototype complete_failure_record.py. run_cold_start_experiment.py measures the cost as a function
of the number of real failures available for training, with and without synthetic rows.
"""
