import sys
a=sys.argv[:]
print(a)
import argparse
parser=argparse.ArgumentParser(description='dddd')

parser.add_argument("-v", "--verbose", action="store_true", default=False,help="Enable verbose debugging")
parser.add_argument("--name", action="store", dest="exp_name", type=str, default="_random",help="Name of experiment (for logging purposes)")
parser.add_argument("--gpuid", action="store", dest="gpuid", type=int, required=False, default=-1,help="GPU ID for cntk")
ps=parser.parse_args(sys.argv[1:])
print(ps.verbose)
print(ps.exp_name)
print(ps.gpuid)
