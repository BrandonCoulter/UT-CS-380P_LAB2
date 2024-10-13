#!/usr/bin/env python3
from subprocess import check_output as co

INPUTS = ['random-n2048-d16-c16.txt', "random-n16384-d24-c16.txt", "random-n65536-d32-c16.txt"]

print("Making executable: bin/kmeans")
out = co("make all", shell=True).decode("ascii")
print(out)

for input in INPUTS:
    print(f"Testing input: {input}")
    d = input.split("d")[-1].split("-")[0]
    k = input.split("c")[-1].split("-")[0]
    
    # bin/kmeans -k 16 -t 1e-5 -d 16 -i input/random-n2048-d16-c16.txt -m 200 -s 8675309 -
    cmd = f'bin/kmeans -k {k} -t 1e-5 -d {d} -i input/{input} -m 200 -s 8675309 -c > out.txt'

    out = co(cmd, shell=True).decode("ascii")

    if out is not None:
        answer = "-answer.txt".join(input.split(".txt"))
        cmd = f'./cluster_checker.py answers/{answer} out.txt 0.0001'
        out = co(cmd, shell=True).decode("ascii")

        print(out)


