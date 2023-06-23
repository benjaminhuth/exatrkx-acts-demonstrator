#!/usr/bin/env python3
import sys
import os
import yaml
import pprint
import time
import warnings
import json
import argparse
import subprocess
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

parser = argparse.ArgumentParser(
    description="Exa.TrkX data generation/reconstruction script"
)
parser.add_argument("events", help="how many events to run", type=int)
parser.add_argument("models", help="where the models are stored", type=str)
parser.add_argument(
    "digi", help="digitization mode", type=str, choices=["truth", "smear"]
)
parser.add_argument(
    "--output", "-o", help="where to store output data", type=str, default="output"
)
parser.add_argument(
    "--embdim", "-e", help="Hyperparameter embedding dim", type=int, default=8
)
parser.add_argument(
    "--verbose", "-v", help="Make ExaTrkX algorithm verbose", action="store_true"
)
args = vars(parser.parse_args())

assert args["events"] > 0

outputDir = Path(args["output"])
(outputDir / "train_all").mkdir(exist_ok=True, parents=True)

modelDir = Path(args["models"])

assert (modelDir / "embed.pt").exists()
assert (modelDir / "filter.pt").exists()
assert (modelDir / "gnn.pt").exists()


###########################
# Load Open Data Detector #
###########################

baseDir = Path(os.path.dirname(__file__))

oddDir = Path("/home/iwsatlas1/bhuth/acts/thirdparty/OpenDataDetector")
oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert oddMaterialMap.exists()

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(
    oddDir, mdecorator=oddMaterialDeco
)

geoSelectionExaTrkX = baseDir / "detector/odd-geo-selection-whole-detector.json"
assert geoSelectionExaTrkX.exists()

if args["digi"] == "smear":
    digiConfigFile = baseDir / "detector/odd-digi-smearing-config.json"
elif args["digi"] == "truth":
    digiConfigFile = baseDir / "detector/odd-digi-true-config.json"
assert digiConfigFile.exists()


#######################
# Start GPU profiling #
#######################

gpu_profiler_args = [
    "nvidia-smi",
    "--query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used",
    "--format=csv,nounits",
    "--loop-ms=10",
    "--filename={}".format(outputDir / "gpu_memory_profile.csv"),
]

gpu_profiler = subprocess.Popen(gpu_profiler_args)

#####################
# Prepare sequencer #
#####################

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args["events"],
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
    # hardProcess=["Top:qqbar2ttbar=on"],
    outputDirRoot=str(outputDir),
)

particleSelection = ParticleSelectorConfig(
    rho=(0.0 * u.mm, 2.0 * u.mm),
    pt=(500 * u.MeV, 20 * u.GeV),
    absEta=(0, 3),
    removeNeutral=True,
)

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    preSelectParticles=particleSelection,
    # postSelectParticles=particleSelection,
    outputDirRoot=str(outputDir),
)

s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    outputDirRoot=None,
    outputDirCsv=str(outputDir / "train_all"),
    rnd=rnd,
    logLevel=acts.logging.INFO,
)

s.addWriter(
    acts.examples.CsvSimHitWriter(
        level=acts.logging.INFO,
        inputSimHits="simhits",
        outputDir=str(outputDir / "train_all"),
        outputStem="truth",
    )
)

s.addWriter(
    acts.examples.CsvMeasurementWriter(
        level=acts.logging.INFO,
        inputMeasurements="measurements",
        inputClusters="clusters",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputDir=str(outputDir / "train_all"),
    )
)

s.addWriter(
    acts.examples.CsvTrackingGeometryWriter(
        level=acts.logging.INFO,
        trackingGeometry=trackingGeometry,
        outputDir=str(outputDir),
        writePerEvent=False,
    )
)

#########################
# ExaTrkX Track Finding #
#########################

s.addAlgorithm(
    acts.examples.SpacePointMaker(
        level=acts.logging.INFO,
        inputSourceLinks="sourcelinks",
        inputMeasurements="measurements",
        outputSpacePoints="exatrkx_spacepoints",
        trackingGeometry=trackingGeometry,
        geometrySelection=acts.examples.readJsonGeometryList(str(geoSelectionExaTrkX)),
    )
)

exatrkxLogLevel = acts.logging.VERBOSE if args["verbose"] else acts.logging.INFO

metricLearningConfig = {
    "level": exatrkxLogLevel,
    "modelPath": str(modelDir / "embed.pt"),
    "spacepointFeatures": 3,
    "embeddingDim": args["embdim"],
    "rVal": 0.2,
    "knnVal": 100,
}

filterConfig = {
    "level": exatrkxLogLevel,
    "cut": 0.01,
    "modelPath": str(modelDir / "filter.pt"),
    "nChunks": 5,
}

gnnConfig = {
    "level": exatrkxLogLevel,
    "cut": 0.5,
    "modelPath": str(modelDir / "gnn.pt"),
    "undirected": True,
}

for cfg in [metricLearningConfig, filterConfig, gnnConfig]:
    assert Path(cfg["modelPath"]).exists()

graphConstructor = acts.examples.TorchMetricLearning(**metricLearningConfig)
edgeClassifiers = [
    acts.examples.TorchEdgeClassifier(**filterConfig),
    acts.examples.TorchEdgeClassifier(**gnnConfig),
]
trackBuilder = acts.examples.BoostTrackBuilding(level=acts.logging.INFO)

s.addAlgorithm(
    acts.examples.TrackFindingAlgorithmExaTrkX(
        level=exatrkxLogLevel,
        inputSpacePoints="exatrkx_spacepoints",
        outputProtoTracks="exatrkx_prototracks",
        graphConstructor=graphConstructor,
        edgeClassifiers=edgeClassifiers,
        trackBuilder=trackBuilder,
        rScale=1000.0,
        phiScale=3.14,
        zScale=1000.0,
    )
)

s.addWriter(
    acts.examples.TrackFinderPerformanceWriter(
        level=acts.logging.INFO,
        inputProtoTracks="exatrkx_prototracks",
        inputParticles="particles_initial",
        inputMeasurementParticlesMap="measurement_particles_map",
        filePath=str(outputDir / "track_finding_performance_exatrkx.root"),
    )
)


#################
# Track fitting #
#################

# Need to wait for a prototracks-to-seeds algorithm to arrive in main
if False:
    s.addAlgorithm(
        acts.examples.TrackParamsEstimationAlgorithm(
            level=acts.logging.FATAL,
            inputSpacePoints=["exatrkx_spacepoints"],
            inputProtoTracks="exatrkx_prototracks",
            inputSourceLinks="sourcelinks",
            outputProtoTracks="exatrkx_estimated_prototracks",
            outputTrackParameters="exatrkx_estimated_parameters",
            trackingGeometry=trackingGeometry,
            magneticField=field,
        )
    )

    kalmanOptions = {
        "multipleScattering": True,
        "energyLoss": True,
        "reverseFilteringMomThreshold": 0.0,
        "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
    }

    s.addAlgorithm(
        acts.examples.TrackFittingAlgorithm(
            level=acts.logging.INFO,
            inputMeasurements="measurements",
            inputSourceLinks="sourcelinks",
            inputProtoTracks="exatrkx_estimated_prototracks",
            inputInitialTrackParameters="exatrkx_estimated_parameters",
            outputTrajectories="exatrkx_kalman_trajectories",
            directNavigation=False,
            pickTrack=-1,
            trackingGeometry=trackingGeometry,
            fit=acts.examples.makeKalmanFitterFunction(
                trackingGeometry, field, **kalmanOptions
            ),
        )
    )


s.run()

# stop GPU profiler
gpu_profiler.kill()
