#!/usr/bin/env python3
import os
import re
import time as t
import argparse
from subprocess import check_output as co

os.system("clear")

parser = argparse.ArgumentParser()
parser.add_argument("-r", type=int, default=1, help="Number of times to run tests (int)")
parser.add_argument("-t", type=float, default=1e-5, help="Threshold for convergence check")
parser.add_argument("-i", nargs="+", type=str, default="all", help="Absolute path to input file")
parser.add_argument("-T", nargs="+", type=str, default="all", help="Type of run I.e., --run_cuda --run_shmem --run_thrust")
parser.add_argument("-o", type=str, default="out.txt", help="Absolute path to output file")
parser.add_argument("-s", type=int, default=8675309, help="Intial seed")
parser.add_argument("-c", action="store_true", help="Print Centroids")

args = parser.parse_args()

INPUTS = args.i if args.i != "all" else ['input/random-n2048-d16-c16.txt', "input/random-n16384-d24-c16.txt", "input/random-n65536-d32-c16.txt"]
RUN_TYPE = args.T if args.T != "all" else ["", "run_cuda", "run_shmem", "run_thrust"]
PRINT_CENT = "-c" if args.c else ""

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
    for run_type in RUN_TYPE:
        d = input.split("d")[-1].split("-")[0]
        k = input.split("c")[-1].split(".")[0]
        print(f"Testing input: {input} -d {d} -k {k} with {run_type if run_type != '' else 'sequential'}")

        # bin/kmeans -k 16 -t 1e-5 -d 16 -i input/random-n2048-d16-c16.txt -m 200 -s 8675309 -
        cmd = f'bin/kmeans -k {k} -t {args.t} -d {d} -i {input} -m 200 -s {args.s} {PRINT_CENT} -T {"--" if run_type != "" else ""}{run_type} > {args.o}'
        # print(f"\n\n{cmd}")

        out = co(cmd, shell=True).decode("ascii")

        with open("out.txt", "r") as out:
            time = out.readline()
            if "Time" in time:
                print(time.strip())
            out.close()

        if args.c:
            if out is not None:
                answer = "-answer.txt".join(input.split(".txt")).split("input/")[-1]
                cmd = f'./cluster_checker.py answers/{answer} out.txt 0.0001'
                check = co(cmd, shell=True).decode("ascii")

                print(check)


