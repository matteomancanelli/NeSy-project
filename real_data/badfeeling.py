# i have the bad feeling that all traces in 50, 75, 100, 128 are also contained in financial_log
# so let's test that

from pathlib import Path
import xml.etree.ElementTree as ET
import re

def comparable_representation(trace):
    trace_str = ET.tostring(trace, encoding="unicode")
    # whitespace
    trace_str = re.sub(r'[ \n\t]+', r'', trace_str)
    # by manual inspection of the 18 traces that appear to differ,
    # most do only for an extra ".000" added as teh milliseconds in the timestamp,
    # which is obviously inconsequential, so let's equalize that
    trace_str = re.sub(
        # r'(key="(?:time:timestamp|REG_DATE)" value="[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}).000([+][0-9]{2}:[0-9]{2}")',
        r'''(key="(?:time:timestamp|REG_DATE)" value="[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})
            .000
            ([+][0-9]{2}:[0-9]{2}")''',
        r'\1\2',
        trace_str,
        flags=re.VERBOSE
    )
    return trace_str

# alright, first off, we hash all the traces in the big file
traces = set()
tree = ET.parse(Path(__file__).parent / "financial_log_for_experiments.xes")
root = tree.getroot()
found = root.findall("./trace")
for trace in found:
    traces.add(comparable_representation(trace))

# then let's check whether the traces from the other files are contained
misses = {"50.xes":0, "75.xes":0, "100.xes":0, "128.xes":0}
miss_list = []
for fname in misses.keys():
    tree = ET.parse(Path(__file__).parent / fname)
    root = tree.getroot()
    found = root.findall("./trace")
    for trace in found:
        if comparable_representation(trace) not in traces:
            misses[fname]+=1
            miss_list.append(trace)
print(misses)
miss_list.sort(key = lambda t: comparable_representation(t))
miss_root = ET.Element("misses")
for t in miss_list:
    miss_root.append(t)
miss_tree = ET.ElementTree(miss_root)
# miss_tree.write(Path(__file__).parent / "misses_a.xml")


# you know what, let's do the reverse check too

traces = set()
for fname in misses.keys():
    tree = ET.parse(Path(__file__).parent / fname)
    root = tree.getroot()
    found = root.findall("./trace")
    for trace in found:
        traces.add(comparable_representation(trace))

misses = 0
miss_list = []
tree = ET.parse(Path(__file__).parent / "financial_log_for_experiments.xes")
root = tree.getroot()
found = root.findall("./trace")
for trace in found:
    if comparable_representation(trace) not in traces:
        misses+=1
        miss_list.append(trace)
print(misses)
miss_list.sort(key = lambda t: comparable_representation(t))
miss_root = ET.Element("misses")
for t in miss_list:
    miss_root.append(t)
miss_tree = ET.ElementTree(miss_root)
# miss_tree.write(Path(__file__).parent / "misses_b.xml")
