# execute as "python scinter.py ..."

import argparse

from scinter.scintillation import Scintillation

description = "Scinter v2.3"

parser = argparse.ArgumentParser(description=description)
parser.add_argument("data", help="specify the dynamic spectrum you want to analyze")
parser.add_argument("action", help="specify what you want to do (plot or compute)")
parser.add_argument("save", help="specify the requested result")
parser.add_argument("--archive", help="archive current results under this name before computing new ones")
parser.add_argument("--load", help="load specified archived results instead of computing new ones")

args = parser.parse_args()

scin = Scintillation(args.data)

if args.action == "compute":
    scin.compute(args.save,args.archive,args.load)
elif args.action == "plot":
    scin.plot(args.save,args.archive,args.load)
elif args.action == "animate":
    scin.animate(args.save,args.archive,args.load)
elif args.action == "remove":
    scin.remove(args.save)
