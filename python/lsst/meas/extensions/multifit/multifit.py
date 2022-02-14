import numpy
import logging

import lsst.pex.config as pexConfig
import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.pipe.tasks as pipeTasks
from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpConfig, MakeWarpConnections, MakeWarpTask
import lsst.utils as utils
import lsst.geom
from lsst.meas.algorithms import CoaddPsf, CoaddPsfConfig
from lsst.skymap import BaseSkyMap
# from lsst.utils.timer import timeMethod
# from .coaddBase import CoaddBaseTask
from lsst.pipe.tasks.coaddBase import CoaddBaseTask, makeSkyInfo
from collections.abc import Iterable

__all__ = ["MultifitConnections", "MultifitTask", "MultifitConfig"]

# The following block adds links to this task from the Task Documentation page.
# This works even for task(s) that are not in lsst.pipe.tasks.
## \addtogroup LSST_task_documentation
## \{
## \page pipeTasks_multifitTask
## \ref MultifitTask "MultifitTask"
##      Task doing Multifit.
## \}

log = logging.getLogger(__name__)

class MultifitConnections(MakeWarpConnections,
                          dimensions=("tract", "patch", "skymap", "instrument", "visit"),
                          defaultTemplates={"coaddName": "deep",
                                            "skyWcsName": "jointcal",
                                            "photoCalibName": "fgcm",
                                            "calexpType": ""}):

    measCat = pipeBase.connectionTypes.Input(
        doc="Coadd-measurement input catalog.",
        name="{coaddName}Coadd_meas",
        storageClass="SourceCatalog",
        dimensions=["band", "skymap", "tract", "patch"],
    )

    def __init__(self, *, config=None):  # **kwargs):
        # super().__init__(**kwargs)
        super().__init__(config=config)

class MultifitConfig(pipeBase.PipelineTaskConfig, MakeCoaddTempExpConfig,
                     pipelineConnections=MultifitConnections):

    def validate(self):
        super().validate()

class MultifitTask(MakeWarpTask):

    ConfigClass = MultifitConfig
    _DefaultName = "multifit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        print("Multifit runQuantum")
        log.info("Starting Multifit runQuantum")
        log.info(f"Here's my Butler: {butlerQC}")
        log.info(f"Here are my input refs: {inputRefs}")
        log.info(f"Here are my output refs: {outputRefs}")

        inputs = butlerQC.get(inputRefs)

        # Construct skyInfo expected by `run`.  We remove the SkyMap itself
        # from the dictionary so we can pass it as kwargs later.
        skyMap = inputs.pop("skyMap")
        quantumDataId = butlerQC.quantum.dataId
        skyInfo = makeSkyInfo(skyMap, tractId=quantumDataId['tract'], patchId=quantumDataId['patch'])

        # Construct list of input DataIds expected by `run`
        dataIdList = [ref.datasetRef.dataId for ref in inputRefs.calExpList]
        # Construct list of packed integer IDs expected by `run`
        ccdIdList = [dataId.pack("visit_detector") for dataId in dataIdList]

        # Run the selector and filter out calexps that were not selected
        # primarily because they do not overlap the patch
        cornerPosList = lsst.geom.Box2D(skyInfo.bbox).getCorners()
        coordList = [skyInfo.wcs.pixelToSky(pos) for pos in cornerPosList]
        goodIndices = self.select.run(**inputs, coordList=coordList, dataIds=dataIdList)
        inputs = self.filterInputs(indices=goodIndices, inputs=inputs)

        # Read from disk only the selected calexps
        inputs['calExpList'] = [ref.get() for ref in inputs['calExpList']]

        log.info(f"measCat input: {inputs['measCat']}")

        self.run(None, None, None, 0, None)

    # @timeMethod
    def run(self, calExpList, ccdIdList, skyInfo, visitId=0, dataIdList=None, **kwargs):
        log.info("Starting Multifit run")

    # def _getConfigName(self):
    #     return None
    #
    # def _getMetadataName(self):
    #     # Do not write metadata because I don't know how to create and reference multifit_metadata data type (for now)
    #     # I need to call:
    #     #   butler register-dataset-type REPO multifit_metadata storageClass Dim1 Dim2 DimN
    #     return None
