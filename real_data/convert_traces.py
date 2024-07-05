# .trace is the scarlet format (more "readable")
# .dat is the stlnet format (actually used by the code)

# scarlet format is for readability only and NOT an accurate implementation of the format.
# for example, it's missing the positive-negative divide,
# as well as the conclusion with the operator and symbol lists.

from pathlib import Path
import xml.etree.ElementTree as ET
import pm4py
import json

from generate_alphabet import get_ct_symbol_name, END_OF_TRACE_NAME, PADDING_NAME

def trace_names_are_unique(fpath):
    tree = ET.parse(fpath)
    root = tree.getroot()
    names = root.findall("./trace/string[@key='concept:name']")
    names = [e.attrib["value"] for e in names]
    # don't think we can avoid this O(N^2)
    for i, name1 in enumerate(names):
        for name2 in names[:i]:
            if name1 == name2:
                return False
    return True

def symbol_to_scarlet(symbol, alphabet):
    onehot = ["1" if symbol==s else "0" for s in alphabet]
    return ",".join(onehot)

# def get_maxlen(log):
#     trace_names = log["case:concept:name"].unique()
#     maxlen = -1
#     for tn in trace_names:
#         trace_df = log[log["case:concept:name"] == tn]
#         maxlen = max(maxlen, trace_df.shape[0])
#     return maxlen

def get_alphabet(include_transitions):
    if include_transitions:
        alphabet_fname = "ALPHABET_ct.LIST"
    else:
        alphabet_fname = "ALPHABET_c.LIST"
    with open(Path(__file__).parent / alphabet_fname, "r") as f:
        alphabet = json.load(f)
    return alphabet

def to_symbol_list(trace_df, include_transitions):
    if include_transitions:
        symbolize = get_ct_symbol_name
        trace_df = trace_df[["concept:name", "lifecycle:transition"]]
    else:
        symbolize = lambda s:s
        trace_df = trace_df["concept:name"]
    trace_list = [symbolize(row) for row in trace_df.values]
    trace_list.append(END_OF_TRACE_NAME)
    return trace_list

# makes the final part of _convertxes a little tidier
def strdump(string, directory, filename, suffix):
    with open(Path(__file__).parent / directory / filename.with_suffix(suffix).name, "w") as f:
        f.write(string)

def xes_to_scarlet(include_transitions, padding):
    return _convertxes(include_transitions, do_scarlet=True, do_stlnet=False, do_padded=padding)

def xes_to_stlnet(include_transitions, padding):
    return _convertxes(include_transitions, do_scarlet=False, do_stlnet=True, do_padded=padding)

def _convertxes(include_transitions, *, do_scarlet, do_stlnet, do_padded):
    alphabet = get_alphabet(include_transitions)
    for filename in Path(__file__).parent.iterdir():
        if filename.suffix == ".xes":
            ### HANDLE FILE
            assert trace_names_are_unique(filename)
            log = pm4py.read_xes(str(filename))
            trace_names = log["case:concept:name"].unique()
            include_transitions = include_transitions and "lifecycle:transition" in log.columns
            maxlen = -1
            trace_lists = []
            traces_scarlet = []
            traces_scarlet_padded = []
            lengths = []
            for tn in trace_names:
                ### HANDLE TRACE (1)
                trace_df = log[log["case:concept:name"] == tn]
                trace_list = to_symbol_list(trace_df, include_transitions)
                lengths.append(str(len(trace_list))+"\n")
                maxlen = max(maxlen, len(trace_list))
                trace_lists.append(trace_list)
            for trace_list in trace_lists:
                ### HANDLE TRACE (2)
                trace_halfscarlet = [symbol_to_scarlet(s, alphabet) for s in trace_list]
                traces_scarlet.append(";".join(trace_halfscarlet))
                if do_padded:
                    padding_scarlet = symbol_to_scarlet(PADDING_NAME, alphabet)
                    while len(trace_halfscarlet) < maxlen:
                        trace_halfscarlet.append(padding_scarlet)
                    traces_scarlet_padded.append(";".join(trace_halfscarlet))
            # HANDLE FILE cont.
            superstring_scarlet = "\n".join(traces_scarlet)
            if do_scarlet:
                strdump(superstring_scarlet, "scarlet", filename, ".trace")
            if do_stlnet:
                superstring_stlnet = superstring_scarlet.replace(",", " ").replace(";", " ")
                strdump(superstring_stlnet, "stlnet", filename, ".dat")
            if do_padded:
                superstring_scarlet_padded = "\n".join(traces_scarlet_padded)
                if do_scarlet:
                    strdump(superstring_scarlet_padded, "scarlet_padded", filename, ".trace")
                if do_stlnet:
                    superstring_stlnet_padded = superstring_scarlet_padded.replace(",", " ").replace(";", " ")
                    strdump(superstring_stlnet_padded, "stlnet_padded", filename, ".dat")
            with open(Path(__file__).parent / "lengths" / filename.with_suffix(".txt").name, "w") as f:
                f.writelines(lengths)

def megamerge(directory, suffix):
    first_write = True
    with open(directory / ("50-75-100-128-merge"+suffix), "w") as fout:
        for filename in directory.iterdir():
            if filename.suffix == suffix and filename.stem in ("50","75","100","128"):
                with open(filename, "r") as fin:
                    if not first_write:
                        fout.write("\n")
                    fout.write(fin.read())
                    first_write = False

if __name__ == "__main__":
    _convertxes(True, do_scarlet=True, do_stlnet=True, do_padded=True)
    megamerge(Path(__file__).parent / "scarlet", ".trace")
    megamerge(Path(__file__).parent / "stlnet", ".dat")
