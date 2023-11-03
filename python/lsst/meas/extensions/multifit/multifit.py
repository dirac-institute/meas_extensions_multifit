from .multifit_lib import sort_and_prepare_cutouts, generate_random_params, do_multifit
from .multifit_lib import create_output_schema, lnlike_fun, calculate_t0

import logging
import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
import lsst.afw.table as afwTable
from lsst.obs.base import ExposureIdInfo
from lsst.geom import SpherePoint, radians

from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpConfig, MakeWarpTask

import lsst.geom

from lsst.skymap import BaseSkyMap

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
                          dimensions=("tract", "patch", "skymap", "instrument", "band"),  # , "visit"),
                          defaultTemplates={"coaddName": "deep",
                                            "skyWcsName": "jointcal",
                                            "photoCalibName": "fgcm",
                                            "calexpType": ""}):
    calExpList = connectionTypes.Input(
        doc="Input exposures to be resampled and optionally PSF-matched onto a SkyMap projection/patch",
        name="{calexpType}calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector", "band"),
        multiple=True,
        deferLoad=True,
    )
    backgroundList = connectionTypes.Input(
        doc="Input backgrounds to be added back into the calexp if bgSubtracted=False",
        name="calexpBackground",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector", "band"),
        multiple=True,
    )
    skyCorrList = connectionTypes.Input(
        doc="Input Sky Correction to be subtracted from the calexp if doApplySkyCorr=True",
        name="skyCorr",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector", "band"),
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
    multifitCat = pipeBase.connectionTypes.Output(
        doc="Multifit measurement output catalog.",
        name="meas_multifit",
        storageClass="SourceCatalog",  # "DataFrame",
        dimensions=["band", "skymap", "tract", "patch"],
    )

    def __init__(self, *, config=None):
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


class MultifitConfig(pipeBase.PipelineTaskConfig, MakeCoaddTempExpConfig,
                     pipelineConnections=MultifitConnections):
    def validate(self):
        super().validate()


class MultifitTask(MakeWarpTask):
    ConfigClass = MultifitConfig
    _DefaultName = "multifit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.schema = create_output_schema()
        # self.outputSchema = afwTable.SourceCatalog(self.schema)
        # self.outputSchema.getTable().setMetadata(self.algMetadata)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        log.info("Starting Multifit runQuantum")
        # log.info(f"Here are my input refs: {inputRefs}")
        # log.info(f"Here are my output refs: {outputRefs}")

        inputs = butlerQC.get(inputRefs)

        # Construct list of input DataIds expected by `run`
        dataIdList = [ref.datasetRef.dataId for ref in inputRefs.calExpList]

        # Read from disk only the selected calexps
        inputs['calExpList'] = [ref.get() for ref in inputs['calExpList']]

        patch = inputRefs.measCat.dataId['patch']

        log.info(f"butler quantum dataId: {butlerQC.quantum.dataId}")

        expId, expBits = butlerQC.quantum.dataId.pack("tract_patch", returnMaxBits=True)
        idFactory = ExposureIdInfo(expId, expBits).makeSourceIdFactory()

        outputs = self.run(inputs['calExpList'], inputs['measCat'].asAstropy(), dataIdList, patch, idFactory)  # , None, 0, None)
        butlerQC.put(outputs, outputRefs)

    # @timeMethod
    def run(self, calExpList, measCat, dataIdList, patch, idFactory):  # ccdIdList, skyInfo, visitId=0, dataIdList=None, **kwargs):
        # log.info("Starting Multifit run")

        EXT_PIXELS = 21

        extended_sources = measCat  # [measCat['deblend_modelType'] == 'MultiExtendedSource']

        if len(extended_sources) > 0:
            log.info(f"Multifit measuring {len(extended_sources)} extended sources")

        table = afwTable.SourceTable.make(self.schema, idFactory)
        catalog = afwTable.SourceCatalog(table)
        catalog.reserve(len(extended_sources))

        for source in extended_sources:
            ra = source['coord_ra'] * 180 / np.pi
            dec = source['coord_dec'] * 180 / np.pi

            coord = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)

            clones = []
            cutouts = []
            dids = []
            wcss = []
            psfs = []
            xs = []
            ys = []
            times = []
            offsets = []
            variances = []
            masks = []
            calibMeans = []

            srcid = source['id']

            for i, calexp in enumerate(calExpList):
                dataId = dataIdList[i]
                wcs = calexp.getWcs()
                sourcexy = wcs.skyToPixel(coord)
                psf = calexp.getPsf().computeImage(sourcexy)
                # psf = calexp.getPsf(sourcexy).computeImage()

                bb = calexp.getBBox()

                if bb.beginX + EXT_PIXELS/2 > sourcexy.x or bb.beginY + EXT_PIXELS/2 > sourcexy.y or \
                   bb.endX - EXT_PIXELS/2 < sourcexy.x or bb.endY - EXT_PIXELS/2 < sourcexy.y:
                    continue

                extent = lsst.geom.ExtentI(EXT_PIXELS, EXT_PIXELS)
                cutout = calexp.getCutout(coord, extent)

                image_arr = cutout.getImage().array
                # image_arr = cutout.getMaskedImage().getImage().array

                offset = (cutout.getBBox().getCorners()[0], cutout.getBBox().getCorners()[2])
                bx = cutout.getBBox().getBeginX()
                by = cutout.getBBox().getBeginY()
                mask = calexp.getMask().getArray()[by:by + EXT_PIXELS, bx:bx + EXT_PIXELS]
                variance = calexp.getVariance().getArray()[by:by + EXT_PIXELS, bx:bx + EXT_PIXELS]
                calibMean = calexp.getPhotoCalib().getCalibrationMean()

                clones.append(calexp.clone())
                cutouts.append(image_arr)
                dids.append(dataId)
                wcss.append(wcs)
                psfs.append(psf)
                xs.append(sourcexy.x)
                ys.append(sourcexy.y)
                times.append(calexp.visitInfo.date)
                offsets.append(offset)
                variances.append(variance)
                masks.append(mask)
                calibMeans.append(calibMean)

            if len(cutouts) == 0:
                log.warn(f"No calexps of extended source {srcid}")
                continue

            cutouts_info = {}
            cutouts_info['dataIds'] = dids
            cutouts_info['id'] = srcid
            cutouts_info['ra'] = ra
            cutouts_info['dec'] = dec
            cutouts_info['xs'] = xs
            cutouts_info['ys'] = ys
            cutouts_info['times'] = times
            cutouts_info['cutouts'] = cutouts
            cutouts_info['wcss'] = wcss
            cutouts_info['psfs'] = psfs
            cutouts_info['offsets'] = offsets
            cutouts_info['masks'] = masks
            cutouts_info['variances'] = variances
            cutouts_info['calibMeans'] = calibMeans

            cutouts_info = sort_and_prepare_cutouts(cutouts_info, cutout_size=EXT_PIXELS)

            start_params = generate_random_params(1, mag_rrange=6, mag_roffs=19,
                                                  pos0_rrange=1/10, pos0_roffs=-1/20,
                                                  v_rrange=1/100, v_roffs=-1/200)[0]

            mr = do_multifit(cutouts_info, None, None, in_collection=None,
                             start_params=start_params, clones=clones, do_normalize=False,
                             method='powell', verbose=0)
            mag, x0, y0, vx, vy = mr.x
            mag_err, x0_err, y0_err, vx_err, vy_err = mr.errors

            lnlikeraw = lnlike_fun(mr.x, cutouts_info, clones)
            chisq = np.sum(lnlikeraw ** 2)
            chisqperdof = chisq / len(lnlikeraw)

            t0 = calculate_t0(cutouts_info)

            row = catalog.addNew()
            row.setParent(srcid)
            radec = SpherePoint(source['coord_ra'], source['coord_dec'], radians)
            row.setRa(radec.getRa())
            row.setDec(radec.getDec())
            row.set("ra0", ra+x0/3600)
            row.set("dec0", dec+y0/3600)
            row.set("mag", mag)
            row.set("t0", t0)
            row.set("x0", x0)
            row.set("y0", y0)
            row.set("vx", vx)
            row.set("vy", vy)
            row.set("mag_err", mag_err)
            row.set("x0_err", x0_err)
            row.set("y0_err", y0_err)
            row.set("vx_err", vx_err)
            row.set("vy_err", vy_err)
            row.set("chi_sq", chisq)
            row.set("chi_sq_per_dof", chisqperdof)

        return pipeBase.Struct(multifitCat=catalog)
