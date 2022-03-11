Version:1.0
StartHTML:0000000128
EndHTML:0000059602
StartFragment:0000000128
EndFragment:0000059602
SourceURL:about:blank
import numpy
import logging
import pickle
from os.path import exists

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

class MultifitConnections(pipeBase.PipelineTaskConnections,
                          dimensions=("tract", "patch", "skymap", "instrument", "band"),  #, "visit"),
                          defaultTemplates={"coaddName": "deep",
                                            "skyWcsName": "jointcal",
                                            "photoCalibName": "fgcm",
                                            "calexpType": ""}):
    calExpList = connectionTypes.Input(
        doc="Input exposures to be resampled and optionally PSF-matched onto a SkyMap projection/patch",
        name="{calexpType}calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    backgroundList = connectionTypes.Input(
        doc="Input backgrounds to be added back into the calexp if bgSubtracted=False",
        name="calexpBackground",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    skyCorrList = connectionTypes.Input(
        doc="Input Sky Correction to be subtracted from the calexp if doApplySkyCorr=True",
        name="skyCorr",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    skyMap = connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for warped exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    externalSkyWcsTractCatalog = connectionTypes.Input(
        doc=("Per-tract, per-visit wcs calibrations.  These catalogs use the detector "
             "id for the catalog id, sorted on id for fast lookup."),
        name="{skyWcsName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "tract"),
    )
    externalSkyWcsGlobalCatalog = connectionTypes.Input(
        doc=("Per-visit wcs calibrations computed globally (with no tract information). "
             "These catalogs use the detector id for the catalog id, sorted on id for "
             "fast lookup."),
        name="{skyWcsName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    externalPhotoCalibTractCatalog = connectionTypes.Input(
        doc=("Per-tract, per-visit photometric calibrations.  These catalogs use the "
             "detector id for the catalog id, sorted on id for fast lookup."),
        name="{photoCalibName}PhotoCalibCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "tract"),
    )
    externalPhotoCalibGlobalCatalog = connectionTypes.Input(
        doc=("Per-visit photometric calibrations computed globally (with no tract "
             "information).  These catalogs use the detector id for the catalog id, "
             "sorted on id for fast lookup."),
        name="{photoCalibName}PhotoCalibCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    # wcsList = connectionTypes.Input(
    #     doc="WCSs of calexps used by SelectImages subtask to determine if the calexp overlaps the patch",
    #     name="{calexpType}calexp.wcs",
    #     storageClass="Wcs",
    #     dimensions=("instrument", "visit", "detector"),
    #     multiple=True,
    # )
    # bboxList = connectionTypes.Input(
    #     doc="BBoxes of calexps used by SelectImages subtask to determine if the calexp overlaps the patch",
    #     name="{calexpType}calexp.bbox",
    #     storageClass="Box2I",
    #     dimensions=("instrument", "visit", "detector"),
    #     multiple=True,
    # )
    # visitSummary = connectionTypes.Input(
    #     doc="Consolidated exposure metadata from ConsolidateVisitSummaryTask",
    #     name="{calexpType}visitSummary",
    #     storageClass="ExposureCatalog",
    #     dimensions=("instrument", "visit",),
    #     multiple=True,
    # )
    srcList = connectionTypes.Input(
        doc="Source catalogs used by PsfWcsSelectImages subtask to further select on PSF stability",
        name="src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
    )
    measCat = pipeBase.connectionTypes.Input(
        doc="Coadd-measurement input catalog.",
        name="{coaddName}Coadd_meas",
        storageClass="SourceCatalog",
        dimensions=["band", "skymap", "tract", "patch"],
    )

    def __init__(self, *, config=None):  # **kwargs):
        # super().__init__(**kwargs)
        super().__init__(config=config)
        if config.bgSubtracted:
            self.inputs.remove("backgroundList")
        if not config.doApplySkyCorr:
            self.inputs.remove("skyCorrList")
        if config.doApplyExternalSkyWcs:
            if config.useGlobalExternalSkyWcs:
                self.inputs.remove("externalSkyWcsTractCatalog")
            else:
                self.inputs.remove("externalSkyWcsGlobalCatalog")
        else:
            self.inputs.remove("externalSkyWcsTractCatalog")
            self.inputs.remove("externalSkyWcsGlobalCatalog")
        if config.doApplyExternalPhotoCalib:
            if config.useGlobalExternalPhotoCalib:
                self.inputs.remove("externalPhotoCalibTractCatalog")
            else:
                self.inputs.remove("externalPhotoCalibGlobalCatalog")
        else:
            self.inputs.remove("externalPhotoCalibTractCatalog")
            self.inputs.remove("externalPhotoCalibGlobalCatalog")
        if not config.makeDirect:
            self.outputs.remove("direct")
        # if not config.makePsfMatched:
        #     self.outputs.remove("psfMatched")
        # TODO DM-28769: add connection per selectImages connections
        # if config.select.target != lsst.pipe.tasks.selectImages.PsfWcsSelectImagesTask:
        #     self.inputs.remove("visitSummary")
        #     self.inputs.remove("srcList")
        # elif not config.select.doLegacyStarSelectionComputation:
        #     # Remove backwards-compatibility connections.
        #     self.inputs.remove("srcList")
class MultifitConfig(pipeBase.PipelineTaskConfig, MakeCoaddTempExpConfig,
                     pipelineConnections=MultifitConnections):
    outFile = pexConfig.Field(
        doc="cutouts info output filename",
        dtype=str,
        default="/home/lsst/mnt/lsst-pkgs/rc2_subset/multifit_output.pickle",
    )
    def validate(self):
        super().validate()

class MultifitTask(MakeWarpTask):

    ConfigClass = MultifitConfig
    _DefaultName = "multifit"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        log.info("Starting Multifit runQuantum")
        log.info(f"Here are my input refs: {inputRefs}")
        log.info(f"Here are my output refs: {outputRefs}")

        inputs = butlerQC.get(inputRefs)
        # for key in inputs:
        #     if isinstance(inputs[key], list):
        #         log.info(f"input {key}: {len(inputs[key])}")
        # Construct skyInfo expected by `run`.  We remove the SkyMap itself
        # from the dictionary so we can pass it as kwargs later.
        # skyMap = inputs.pop("skyMap")
        # quantumDataId = butlerQC.quantum.dataId
        # skyInfo = makeSkyInfo(skyMap, tractId=quantumDataId['tract'], patchId=quantumDataId['patch'])
        # Construct list of input DataIds expected by `run`
        dataIdList = [ref.datasetRef.dataId for ref in inputRefs.calExpList]
        # Construct list of packed integer IDs expected by `run`
        # ccdIdList = [dataId.pack("visit_detector") for dataId in dataIdList]
        # Run the selector and filter out calexps that were not selected
        # primarily because they do not overlap the patch
        # cornerPosList = lsst.geom.Box2D(skyInfo.bbox).getCorners()
        # coordList = [skyInfo.wcs.pixelToSky(pos) for pos in cornerPosList]
        # TODO: enable select task to speed up loading:
        # goodIndices = self.select.run(**inputs, coordList=coordList, dataIds=dataIdList)
        # log.info(f"coordList: {len(coordList)}")
        # log.info(f"dataIdList: {len(dataIdList)}")
        # log.info(f"goodind: {len(goodIndices)}")
        # log.info(f"inputs: {len(inputs)}")
        # inputs = self.filterInputs(indices=goodIndices, inputs=inputs)
        # Read from disk only the selected calexps
        inputs['calExpList'] = [ref.get() for ref in inputs['calExpList']]

        # log.info(f"measCat input: {inputs['measCat']}")
        self.run(inputs['calExpList'], inputs['measCat'].asAstropy(), dataIdList)  # , None, 0, None)
    # @timeMethod
    def run(self, calExpList, measCat, dataIdList):  # ccdIdList, skyInfo, visitId=0, dataIdList=None, **kwargs):
        # log.info("Starting Multifit run")
        EXT_PIXELS = 100
        # deblend_modelType == MultiExtendedSource
        extended_sources = measCat[measCat['deblend_modelType'] == 'MultiExtendedSource']

        all_cutouts = []

        for source in extended_sources:
            ra = source['coord_ra']
            dec = source['coord_dec']

            coord = lsst.geom.SpherePoint(ra, dec, lsst.geom.radians)

            cutouts = []
            wcss = []
            psfs = []
            xs = []
            ys = []
            times = []
            offsets = []

            for i, calexp in enumerate(calExpList):
                dataId = dataIdList[i]
                wcs = calexp.getWcs()
                psf = calexp.getPsf().computeImage()
                sourcexy = wcs.skyToPixel(coord)

                # log.info(f"src in calexp: {sourcexy}")
                bb = calexp.getBBox()

                if bb.beginX + EXT_PIXELS/2 > sourcexy.x or bb.beginY + EXT_PIXELS/2 > sourcexy.y or \
                    bb.endX - EXT_PIXELS/2 < sourcexy.x or bb.endY - EXT_PIXELS/2 < sourcexy.y:
                    # log.info(f"Source not contained in calexp")
                    continue
                # img = calexp.image.getArray()
                extent = lsst.geom.ExtentI(EXT_PIXELS, EXT_PIXELS)
                cutout = calexp.getCutout(coord, extent)
                image_arr = cutout.getMaskedImage().getImage().array

                offset = (cutout.getBBox().getCorners()[0], cutout.getBBox().getCorners()[2], wcs.skyToPixel(coord))

                cutouts.append(image_arr)
                wcss.append(wcs)
                psfs.append(psf)
                xs.append(sourcexy.x)
                ys.append(sourcexy.y)
                times.append(calexp.visitInfo.date)
                offsets.append(offset)

            if len(cutouts) > 4:
                # log.info(f"Found {len(cutouts)} cutouts for source ({ra}, {dec})")
                cutouts_info = {}
                cutouts_info['dataId'] = dataId
                cutouts_info['id'] = source['id']
                cutouts_info['base_GaussianFlux_instFlux'] = source['base_GaussianFlux_instFlux']
                cutouts_info['base_PsfFlux_instFlux'] = source['base_PsfFlux_instFlux']
                cutouts_info['modelfit_CModel_initial_instFlux'] = source['modelfit_CModel_initial_instFlux']
                cutouts_info['ra'] = ra
                cutouts_info['dec'] = dec
                cutouts_info['xs'] = xs
                cutouts_info['ys'] = ys
                cutouts_info['times'] = times
                cutouts_info['cutouts'] = cutouts
                cutouts_info['wcss'] = wcss
                cutouts_info['psfs'] = psfs
                cutouts_info['offsets'] = offsets

                all_cutouts.append(cutouts_info)
            # break
        output = []
        if exists(self.config.outFile):
            with open(self.config.outFile, 'rb') as f:
                output = pickle.load(f)
        output.append(all_cutouts)

        with open(self.config.outFile, 'wb') as f:
            pickle.dump(output, f)
