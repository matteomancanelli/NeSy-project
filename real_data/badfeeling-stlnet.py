# same as before, but for the final result

from pathlib import Path

financial_traces = set()
with open(Path(__file__).parent / "stlnet" / "financial_log_for_experiments.dat") as f:
    financial_traces.update(l.strip() for l in f.readlines())

merge_traces = set()
with open(Path(__file__).parent / "stlnet" / "50-75-100-128-merge.dat") as f:
    merge_traces.update(l.strip() for l in f.readlines())

with open("a.log", "w") as f:
    pass

for trace in financial_traces:
    if trace.strip() not in merge_traces:
        print("ZAN ZAN ZAN")
        with open("a.log", "a") as f:
            f.write(trace+"\n")

for trace in merge_traces:
    if trace.strip() not in financial_traces:
        print("ZANMA ZANMA ZANMA")
        with open("a.log", "a") as f:
            f.write(trace+"\n")
