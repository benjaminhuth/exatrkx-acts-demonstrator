#!/usr/bin/env python3
import sys
import os
import yaml
import pprint
import time
import warnings
import json
import argparse
import math
from pathlib import Path

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector
from acts.examples.reconstruction import *
from acts.examples.simulation import *

u = acts.UnitConstants

logger = acts.logging.getLogger("main")


#########################
# Command line handling #
#########################

parser = argparse.ArgumentParser(description='Exa.TrkX data generation/reconstruction script')
parser.add_argument('events', help="how many events to run", type=int)
parser.add_argument('digi', help="digitization mode", type=str, choices=['truth', 'smear'])
parser.add_argument('--output', '-o', help="where to store output data", type=str, default="output")
args = vars(parser.parse_args())

assert args['events'] > 0

outputDir = Path(args['output'])
(outputDir / "csv").mkdir(exist_ok=True, parents=True)

###########################
# Load Open Data Detector #
###########################

baseDir = Path(os.path.dirname(__file__))

oddDir = Path("/home/iwsatlas1/bhuth/acts/thirdparty/OpenDataDetector")
oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert oddMaterialMap.exists()

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(oddDir, mdecorator=oddMaterialDeco)

if args['digi'] == 'smear':
    digiConfigFile = baseDir / "detector/odd-digi-smearing-config.json"
elif args['digi'] == 'truth':
    digiConfigFile = baseDir / "detector/odd-digi-true-config.json"
assert digiConfigFile.exists()


#####################
# Prepare sequencer #
#####################

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args['events'],
    numThreads=1,
    outputDir=str(outputDir),
)


#############################
# Simulation & Digitization #
#############################

s = addPythia8(
    s,
    rnd=rnd,
    hardProcess=["HardQCD:all = on"],
    #hardProcess=["Top:qqbar2ttbar=on"],
    outputDirCsv=str(outputDir/"csv"),
)

particleSelection = ParticleSelectorConfig(
    rho=(0.0, 2.0*u.mm),
    pt=(500*u.MeV, 20*u.GeV),
    absEta=(None, 3)
)

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    preselectParticles=particleSelection,
)

s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    outputDirCsv=str(outputDir/"csv"),
    rnd=rnd,
    logLevel=acts.logging.INFO,
)

s.run()
