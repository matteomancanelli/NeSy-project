from pathlib import Path
import pm4py
import json

PADDING_NAME = "PADDING"
END_OF_TRACE_NAME = "END_OF_TRACE"

def generate_alphabet_concept():
    symbols = set()

    for filename in Path(__file__).parent.iterdir():
        if filename.suffix == ".xes":
            log = pm4py.read_xes(str(filename))
            symbols.update(log["concept:name"].unique())

    symbols_list = sorted(symbols)
    symbols_list.insert(0, "PADDING")
    symbols_list.append("END_OF_TRACE")

    with open(Path(__file__).parent / "ALPHABET_c.LIST", "w") as f:
        json.dump(symbols_list, f)
    
    return len(symbols_list)

def generate_alphabet_transition():
    symbols = set()

    for filename in Path(__file__).parent.iterdir():
        if filename.suffix == ".xes":
            log = pm4py.read_xes(str(filename))
            if "lifecycle:transition" in log.columns:
                symbols.update(log["lifecycle:transition"].unique())

    with open(Path(__file__).parent / "ALPHABET_t.LIST", "w") as f:
        json.dump(sorted(symbols), f)
    
    return len(symbols)

def generate_alphabet_concept_transition():
    big_symbols = set()
    small_symbols = set()

    for filename in Path(__file__).parent.iterdir():
        if filename.suffix == ".xes":
            log = pm4py.read_xes(str(filename))
            if "lifecycle:transition" in log.columns:
                log = log[["concept:name", "lifecycle:transition"]]
                big_symbols.update(tuple(row) for row in log.values.tolist())
            else:
                small_symbols.update(log["concept:name"].unique())

    symbols = set(map(lambda tup: "_".join(tup).replace(" ", "_").lower(), big_symbols))
    symbols.update(set(map(lambda s: s.lower(), small_symbols)))

    symbols_list = sorted(symbols)
    symbols_list.insert(0, PADDING_NAME)
    symbols_list.append(END_OF_TRACE_NAME)

    with open(Path(__file__).parent / "ALPHABET_ct.LIST", "w") as f:
        json.dump(symbols_list, f)

    return len(symbols_list)

# this is useful outside of this too
def get_ct_symbol_name(ct_pair):
    return "_".join(ct_pair).replace(" ", "_").lower()

if __name__ == "__main__":
    lc = generate_alphabet_concept()
    lt = generate_alphabet_transition()
    lct = generate_alphabet_concept_transition()

    print(lc, lt, lct)