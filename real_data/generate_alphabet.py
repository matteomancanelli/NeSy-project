from pathlib import Path
import pm4py
import json

def generate_alphabet():
    symbols = set()

    for filename in Path(__file__).parent.iterdir():
        if filename.suffix == ".xes":
            log = pm4py.read_xes(str(filename))
            symbols.update(set(log["concept:name"].unique()))

    with open(Path(__file__).parent / "ALPHABET.LIST", "w") as f:
        json.dump(sorted(list(symbols)), f)

if __name__ == "__main__":
    generate_alphabet()