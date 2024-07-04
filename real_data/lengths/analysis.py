d = {i:0 for i in range(1,51)}

with open("50.txt", "r") as f:
    lengths = [int(l) for l in f.readlines()]
    for l in lengths:
        d[l]+=1

with open("50-results.txt", "w") as f:
    for i in range(1,51):
        f.write(f"{i}  {d[i]}\n")



d = {i:0 for i in range(1,129)}

with open("financial_log_for_experiments.txt", "r") as f:
    lengths = [int(l) for l in f.readlines()]
    for l in lengths:
        d[l]+=1

with open("financial-results.txt", "w") as f:
    for i in range(1,129):
        f.write(f"{i}  {d[i]}\n")
