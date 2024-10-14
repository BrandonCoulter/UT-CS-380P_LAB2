#!/usr/bin/env python3
import re
from subprocess import check_output as co

INPUTS = ['random-n2048-d16-c16.txt', "random-n16384-d24-c16.txt", "random-n65536-d32-c16.txt"]

print("Making executable: bin/kmeans")
clean = co("make clean", shell=True).decode("ascii")
make = co("make all", shell=True).decode("ascii")
if clean is None or make is None:
    print("Failed to make clean or make executable.")
    exit(1)
else:
    print(clean)
    print(make)

for input in INPUTS:
    print(f"Testing input: {input}")
    d = input.split("d")[-1].split("-")[0]
    k = input.split("c")[-1].split("-")[0]
    
    # bin/kmeans -k 16 -t 1e-5 -d 16 -i input/random-n2048-d16-c16.txt -m 200 -s 8675309 -
    cmd = f'bin/kmeans -k {k} -t 1e-5 -d {d} -i input/{input} -m 200 -s 8675309 -c -T --run_cuda> out.txt'

    out = co(cmd, shell=True).decode("ascii")
    with open("out.txt", "r") as out:
        time = out.readline()
        if "time" in time:
            print(time.strip())

        out.close()

    if out is not None:
        answer = "-answer.txt".join(input.split(".txt"))
        cmd = f'./cluster_checker.py answers/{answer} out.txt 0.0001'
        check = co(cmd, shell=True).decode("ascii")

        print(check)


