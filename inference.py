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
parser.add_argument('models', help="where the models are stored", type=str)
parser.add_argument('--output', '-o', help="where to store output data (defaults to input if empty)", type=str, default="")
parser.add_argument('--input', '-i', help="where to get data from", type=str, default="output")
parser.add_argument('--embdim', '-e', help="Hyperparameter embedding dim", type=int, default=8)
parser.add_argument('--verbose', help="Make ExaTrkX algorithm verbose", action="store_true")
args = vars(parser.parse_args())

args['output'] = args['input'] if len(args["output"]) == 0 else args["output"]

inputDir = Path(args['input'])
assert (inputDir / "csv").exists()

outputDir = Path(args['output'])
outputDir.mkdir(exist_ok=True, parents=True)


modelDir = Path(args['models'])
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
detector, trackingGeometry, decorators = getOpenDataDetector(oddDir, mdecorator=oddMaterialDeco)

geoSelectionExaTrkX = baseDir / "detector/odd-geo-selection-whole-detector.json"
assert geoSelectionExaTrkX.exists()


###########################
# Prepare & run sequencer #
###########################

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

s = acts.examples.Sequencer(
    numThreads=1,
    outputDir=str(outputDir),
)

s.addReader(
    acts.examples.CsvMeasurementReader(
        level=acts.logging.INFO,
        outputSourceLinks="sourcelinks",
        outputMeasurements="measurements",
        outputMeasurementSimHitsMap="measurement_simhit_map",
        inputDir=str(inputDir/"csv")
    )
)


#################
# Track finding #
#################

s.addAlgorithm(
    acts.examples.SpacePointMaker(
        level=acts.logging.INFO,
        inputSourceLinks="sourcelinks",
        inputMeasurements="measurements",
        outputSpacePoints="exatrkx_spacepoints",
        trackingGeometry=trackingGeometry,
        geometrySelection=acts.examples.readJsonGeometryList(
            str(geoSelectionExaTrkX)
        ),
    )
)

exaTrkXConfig = {
    "modelDir" : str(modelDir),
    "spacepointFeatures" : 3,
    "embeddingDim" : args["embdim"],
    "rVal" : 0.2,
    "knnVal" : 500,
    "filterCut" : 0.01,
    "n_chunks" : 5,
    "edgeCut" : 0.5,
}

print("Exa.TrkX Configuration")
pprint.pprint(exaTrkXConfig, indent=4)

s.addAlgorithm(
    acts.examples.TrackFindingAlgorithmExaTrkX(
        level=acts.logging.VERBOSE if args['verbose'] else acts.logging.INFO,
        inputSpacePoints="exatrkx_spacepoints",
        outputProtoTracks="exatrkx_prototracks",
        trackFinderML=acts.examples.ExaTrkXTrackFindingTorch(**exaTrkXConfig),
        rScale = 1000.,
        phiScale = math.pi,
        zScale = 1000.,
    )
)

# FIXME currently not possible due to missing measurement_particles_map reader/writer
# s.addWriter(
#     acts.examples.TrackFinderPerformanceWriter(
#         level=acts.logging.INFO,
#         inputProtoTracks="exatrkx_prototracks",
#         inputParticles="particles_initial",
#         inputMeasurementParticlesMap="measurement_particles_map",
#         filePath=str(outputDir / "track_finding_performance_exatrkx.root"),
#     )
# )


#################
# Track fitting #
#################

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
