import cv2
import galsim
import numpy as np
from numpy.linalg import LinAlgError
import lsst
from lsst import geom
import lsst.afw.table as afwTable
from lsst.geom import SpherePoint, degrees, ExtentI
from lsst.pex.exceptions.wrappers import InvalidParameterError
from lmfit import Parameters, minimize
from scipy.optimize import least_squares, leastsq


# The method copied from lsst.pipe.tasks.insertFakes.py and modified so that:
#    - we can add fakes also to variance images and
#    - switch off noise.
def add_fake_sources(exposure, objects, add_to_variance=True, add_noise=True,
                     calibFluxRadius=12.0, logger=None):
    """Add fake sources to the given exposure

    Parameters
    ----------
    exposure : `lsst.afw.image.exposure.exposure.ExposureF`
        The exposure into which the fake sources should be added
    objects : `typing.Iterator` [`tuple` ['lsst.geom.SpherePoint`, `galsim.GSObject`]]
        An iterator of tuples that contains (or generates) locations and object
        surface brightness profiles to inject.
    add_to_variance : whether to also add noise to the variance image
    add_noise : if False will turn off Poisson noise
    calibFluxRadius : `float`, optional
        Aperture radius (in pixels) used to define the calibration for this
        exposure+catalog.  This is used to produce the correct instrumental fluxes
        within the radius.  The value should match that of the field defined in
        slot_CalibFlux_instFlux.
    logger : `lsst.log.log.log.Log` or `logging.Logger`, optional
        Logger.
    """
    exposure.mask.addMaskPlane("FAKE")
    bitmask = exposure.mask.getPlaneBitMask("FAKE")
    if logger:
        logger.info(f"Adding mask plane with bitmask {bitmask}")

    wcs = exposure.getWcs()
    psf = exposure.getPsf()

    bbox = exposure.getBBox()
    fullBounds = galsim.BoundsI(bbox.minX, bbox.maxX, bbox.minY, bbox.maxY)
    _add_fake_sources(exposure.image.array, fullBounds, wcs, psf, objects, exposure=exposure, bitmask=bitmask,
                      add_noise=add_noise, calibFluxRadius=calibFluxRadius, logger=logger)


OBJ = {}


def dataId_hash(dataId):
    h = ""
    for k in dataId.keys():
        h = h+"|"+str(dataId[k])
    return h


def get_calexp_clones(butler, dataIds, in_collection):
    clones = []
    for dref in dataIds:
        dstr = dataId_hash(dref)+'#clone'
        if dstr in OBJ:
            clones.append(OBJ[dstr])
        else:
            clone = butler.get('calexp', dataId=dref, collections=in_collection).clone()
            OBJ[dstr] = clone
            clones.append(clone)
    return clones


# caching this computation
APFLUXES = {}


def get_ap_flux(psf, cfr):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    key = f"{psf.computeImage().array.sum()}#{cfr}"
    if key in APFLUXES:
        return APFLUXES[key]
    apflux = psf.computeApertureFlux(cfr, psf.getAveragePosition())
    APFLUXES[key] = apflux
    return apflux


def _add_fake_sources(imgArr, fullBounds, wcs, psf, objects, exposure=None, bitmask=None, add_noise=True,
                      calibFluxRadius=12.0, logger=None):
    gsImg = galsim.Image(imgArr, bounds=fullBounds)

    pixScale = wcs.getPixelScale().asArcseconds()

    for spt, gsObj in objects:
        pt = wcs.skyToPixel(spt)
        posd = galsim.PositionD(pt.x, pt.y)
        posi = galsim.PositionI(pt.x // 1, pt.y // 1)
        if logger:
            print(f"Adding fake source at {pt}")
        # print(f"Adding fake source at {pt}")

        mat = wcs.linearizePixelToSky(spt, geom.arcseconds).getMatrix()
        gsWCS = galsim.JacobianWCS(mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])

        # This check is here because sometimes the WCS
        # is multivalued and objects that should not be
        # were being included.
        gsPixScale = np.sqrt(gsWCS.pixelArea())
        if gsPixScale < pixScale / 2 or gsPixScale > pixScale * 2:
            print("WCS check failed. Skipping the calexp...")
            continue

        try:
            psfArr = psf.computeKernelImage(pt).array
        except InvalidParameterError:
            # Try mapping to nearest point contained in bbox.
            bbox = exposure.getBBox()
            contained_pt = geom.Point2D(
                np.clip(pt.x, bbox.minX, bbox.maxX),
                np.clip(pt.y, bbox.minY, bbox.maxY)
            )
            if pt == contained_pt:  # no difference, so skip immediately
                print(f"Cannot compute Psf for object at {pt}; skipping")
                continue
            # otherwise, try again with new point
            try:
                psfArr = psf.computeKernelImage(contained_pt).array
            except InvalidParameterError:
                print(f"Cannot compute Psf for object at {pt}; skipping")
                continue

        # OPTIMIZED:
        # apCorr = psf.computeApertureFlux(calibFluxRadius, psf.getAveragePosition())
        apCorr = get_ap_flux(psf, calibFluxRadius)
        psfArr /= apCorr
        gsPSF = galsim.InterpolatedImage(galsim.Image(psfArr), wcs=gsWCS)

        conv = galsim.Convolve(gsObj, gsPSF)
        stampSize = conv.getGoodImageSize(gsWCS.minLinearScale())
        subBounds = galsim.BoundsI(posi).withBorder(stampSize // 2)
        subBounds &= fullBounds

        if subBounds.area() > 0:
            subImg = gsImg[subBounds]
            offset = posd - subBounds.true_center
            # Note, for calexp injection, pixel is already part of the PSF and
            # for coadd injection, it's incorrect to include the output pixel.
            # So for both cases, we draw using method='no_pixel'.

            if not add_noise:
                conv.drawImage(
                    subImg,
                    add_to_image=True,
                    offset=offset,
                    wcs=gsWCS,
                    method='fft',
                    # method='phot',
                    # method='no_pixel',
                    # method='sb',
                    # poisson_flux=False
                )
            else:
                conv.drawImage(
                    subImg,
                    add_to_image=True,
                    offset=offset,
                    wcs=gsWCS,
                    # method='phot'
                    # method='no_pixel'
                    method='fft'
                )

            subBox = geom.Box2I(
                geom.Point2I(subBounds.xmin, subBounds.ymin),
                geom.Point2I(subBounds.xmax, subBounds.ymax)
            )
            if exposure and add_noise:
                exposure[subBox].mask.array |= bitmask
        else:
            raise ValueError("Subbounds area 0")  # print("Subbounds area 0")


def calexp_contains_point(calexp, coord):
    p = calexp.getWcs().skyToPixel(SpherePoint(coord[0], coord[1], degrees))
    return p.x > 0 and p.y > 0 and p.x < calexp.getBBox().getWidth() and p.y < calexp.getBBox().getHeight()


def insert_fakes_in_calexp(calexp, objs, add_to_variance=True, add_noise=True):
    # object speed in arcseconds per day
    spoints = []
    stars1 = []
    mjd = calexp.getMetadata()['MJD']
    wcs = calexp.getWcs()
    for obj in objs:
        if calexp_contains_point(calexp, obj.position_at(mjd)):
            spoint = obj.sphere_point_at(mjd)
            spoints.append(spoint)
            f = calexp.getPhotoCalib().magnitudeToInstFlux(obj.mag, wcs.skyToPixel(spoint))
            stars1.append(galsim.DeltaFunction().withFlux(f))

    arr_copy = None
    if add_to_variance:
        arr_copy = calexp.image.array.copy()

    add_fake_sources(calexp, [(spoints[i], stars1[i]) for i in range(len(spoints))],
                     add_to_variance=add_to_variance, add_noise=add_noise)

    if add_to_variance:
        calexp.getVariance().array += (calexp.image.array - arr_copy)


def get_cutout(calexp, center_coord, SIZE):
    if not calexp_contains_point(calexp, center_coord):
        return None
    extent = ExtentI(SIZE, SIZE)
    return calexp.getCutout(SpherePoint(center_coord[0], center_coord[1], degrees), extent).\
        getImage().array.astype(np.float64)

REF_FLUX = 1e23 * pow(10, (48.6 / -2.5)) * 1e9
def magnitudeToFlux(magnitude, calibMean):
    return 10 ** (magnitude / -2.5) * REF_FLUX / calibMean

def generate_model_faster(obj, mjd, ra, dec, psf, wcs, calibMean, image_size=21):
    coord = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    sourcexy = wcs.skyToPixel(coord)
    # psfimg = psf.computeImage(sourcexy).array
    psfimg = psf.computeImage(sourcexy).array
    psfimg /= psfimg.sum()
    psfimg = psfimg * magnitudeToFlux(obj.mag, calibMean) # / get_ap_flux(psf, 12.0)
    raobj, decobj = obj.position_at(mjd)
    p0 = wcs.skyToPixel(SpherePoint(ra, dec, degrees))
    pobj = wcs.skyToPixel(SpherePoint(raobj, decobj, degrees))
    offsdec = pobj.x - p0.x
    offsra = pobj.y - p0.y
    
    psize = psfimg.shape[0]    
    if psize > image_size:
        sdiff = (psize - image_size) // 2
        psfimg = psfimg[sdiff:-sdiff,sdiff:-sdiff]
    
    sdiff = (image_size - psfimg.shape[0]) // 2
    
    # img = np.zeros((image_size, image_size))
    # transm = np.float32([[1, 0, sdiff+decm*offsdec/pixScale], [0, 1, sdiff+ram*offsra/pixScale]])
    transm = np.float32([[1, 0, sdiff+offsdec], [0, 1, sdiff+offsra]])
    simimg = cv2.warpAffine(psfimg, transm, (image_size, image_size), flags=cv2.INTER_LANCZOS4)
    return simimg


def generate_model(obj, ra, dec, calexp_clone, image_size=100):
    # ra, dec in degrees
    calexp_clone.getImage().set(0.)  # [:,:] = 0

    # mjd = calexp_clone.getMetadata()['MJD']
    # spoint = obj.position_at(mjd)
    # print(f"{spoint} is position at {mjd} center at {ra}, {dec}")

    insert_fakes_in_calexp(calexp_clone, [obj], add_to_variance=False, add_noise=False)
    return get_cutout(calexp_clone, (ra, dec), image_size)


class MovingSource():
    ''' ra0, dec0 - position at mjd0 in degrees
     sra, sdec - speeds in arcsec/day
     mag - magnitude '''
    def __init__(self, ra0, dec0, mjd0, sra, sdec, mag):
        self.ra0 = ra0
        self.dec0 = dec0
        self.mjd0 = mjd0
        self.sra = sra
        self.sdec = sdec
        self.mag = mag

        self.sra_degs = sra / 3600
        self.sdec_degs = sdec / 3600

    def position_at(self, mjd):
        ra = self.ra0 + (mjd - self.mjd0) * self.sra_degs
        dec = self.dec0 + (mjd - self.mjd0) * self.sdec_degs
        # print(f"{self.ra0} + ({mjd} - {self.mjd0}) * {self.sra_degs} = {ra}")
        # print(f"{self.dec0} + ({mjd} - {self.mjd0}) * {self.sdec_degs} = {dec}")
        return (ra, dec)

    def sphere_point_at(self, mjd):
        pos = self.position_at(mjd)
        return SpherePoint(pos[0], pos[1], degrees)

    def __repr__(self):
        return f"Moving source: \n\tMagnitude: {self.mag}, \n\tra0: {self.ra0} deg, " + \
            f"\n\tdec0: {self.dec0} deg, \n\tMJD0: {self.mjd0}, \n\tSpeed in ra: {self.sra} " + \
            f"arsec/day, \n\tSpeed in dec: {self.sdec} arcsec/day"


MAX_PIXEL_SCALE = 4.6831063024143785e-05 * 3600
PIX_SC = 2 * MAX_PIXEL_SCALE
MAX_SPEED = 5 * MAX_PIXEL_SCALE / 365  # max_speed = 5 pixels per year (in arcsec per day)


def normalize_params(params):
    mag, x0, y0, vx, vy = params
    return np.array([(mag-10)/20, (x0+PIX_SC)/2/PIX_SC, (y0+PIX_SC)/2/PIX_SC, (vx+MAX_SPEED)/2/MAX_SPEED,
                     (vy+MAX_SPEED)/2/MAX_SPEED])


def denormalize_params(params):
    mag, x0, y0, vx, vy = params
    return np.array([mag*20+10, x0*2*PIX_SC-PIX_SC, y0*2*PIX_SC-PIX_SC, vx*2*MAX_SPEED-MAX_SPEED,
                     vy*2*MAX_SPEED-MAX_SPEED])


def denormalize_range(params):
    mag, x0, y0, vx, vy = params
    return np.array([mag*20, x0*2*PIX_SC, y0*2*PIX_SC, vx*2*MAX_SPEED, vy*2*MAX_SPEED])


def sort_and_prepare_cutouts(imd, cutout_size=21):
    if len(imd['times']) == 0:
        return imd
    ljist = list(zip([t.get() if hasattr(t, 'get') else t for t in imd['times']], imd['dataIds'],
                     imd['xs'], imd['ys'], imd['cutouts'], imd['wcss'], imd['psfs'],
                     imd['masks'], imd['variances'], imd['calibMeans']))
    ljist.sort()
    t, dis, xs, ys, cut, wc, ps, msk, varnc, cmns = list(zip(*ljist))
    # maskPlaneDict={'BAD': 0,
    # 'SAT': 1,
    # 'INTRP': 2,
    # 'CR': 3,
    # 'EDGE': 4,
    # 'DETECTED': 5,
    # 'DETECTED_NEGATIVE': 6,
    # 'SUSPECT': 7,
    # 'NO_DATA': 8,
    # 'CROSSTALK': 9,
    # 'NOT_DEBLENDED': 10,
    # 'UNMASKEDNAN': 11,
    # 'BRIGHT_OBJECT': 12,
    # 'CLIPPED': 13,
    # 'FAKE': 14,
    # 'INEXACT_PSF': 15,
    # 'REJECTED': 16,
    # 'SENSOR_EDGE': 17,
    # 'STREAK': 18, }
    # ignore FAKE, DETECTED, DETECTED_NEGATIVE, NOT_DEBLENDED, BRIGHT_OBJECT
    MASK_BIT_MASK = 0b1111010101110011111
    # this will be true for all pixels whose mask values don't have
    # MASK_BIT_MASK bits set AND whose variance pixel i >= 0
    # this is precomputed so that we don't have to do it in each iteration
    # for each image
    prepmsk = [np.logical_and(np.bitwise_and(m, MASK_BIT_MASK) <= 0, varnc[i] > 0)
               for i, m in enumerate(msk)]

    def _resize_cutouts(cs, newsize):
        csize = cs[0].shape[0]
        assert newsize <= csize
        assert (csize % 2 == 0 and newsize % 2 == 0) or (csize % 2 == 1 and newsize % 2 == 1)
        margin = int((csize-newsize)/2)
        return [c[margin:csize-margin, margin:csize-margin] for c in cs]
    return {'dataIds': [d.full for d in dis],
            'id': imd['id'],
            'ra': imd['ra'],
            'ras': [imd['ra']] * len(t),
            'dec': imd['dec'],
            'decs': [imd['dec']] * len(t),
            'xs': list(xs),
            'ys': list(ys),
            'times': list(t),
            'cutouts': [c.astype(np.float64) for c in _resize_cutouts(cut, cutout_size)],
            'wcss': list(wc),
            'psfs': list(ps),
            'masks': _resize_cutouts(msk, cutout_size),  # [m.astype(np.float64) for m in msk],
            'variances': [v.astype(np.float64) for v in _resize_cutouts(varnc, cutout_size)],
            'prepmasks': _resize_cutouts(prepmsk, cutout_size),
            'calibMeans': list(cmns)}


def calculate_t0(imagedata):
    """Calculates the most likely t0 position"""
    times = imagedata['times']
    return np.mean(times)  # (times[-1] + times[0])/2


def calculate_pos0(imagedata, startpos, sra, sdec, simmjd0):
    # sra, sdec - arcsec/day
    t0 = calculate_t0(imagedata)
    tp = t0 - simmjd0  # imagedata['times'][0]
    return (startpos[0] + tp*sra/3600, startpos[1] + tp*sdec/3600)


def lnlike_fun(params, imagedata, calexp_clones, returnimgs=False, verbose=0, denormalize=False,
               dist_penalty_multi=None, fast_model=False):
    """Returns the sum of chi-square differences between simulated images
    (according to the provided parameters) and the input images (pstamps)."""

    if verbose > 1:
        print(f"iteration params: {params}")

    simimgs = []
    diffimgs = []
    maskedimgs = []

    cutouts = imagedata['cutouts']
    IMGSIZE = cutouts[0].shape[0]
    psfs = imagedata['psfs']
    wcss = imagedata['wcss']
    # offsets = imagedata['offsets']
    variances = imagedata['variances']
    ras = imagedata['ras']
    decs = imagedata['decs']
    # masks = imagedata['masks']
    prepmasks = imagedata['prepmasks']
    # calibMeans = imagedata['calibMeans']
    # if hasattr(imagedata['times'][0], 'get'):
    #     times = [t.get() for t in imagedata['times']]
    # else:
    #     times = imagedata['times']
    ra = imagedata['ra']
    dec = imagedata['dec']
    
    # coord = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)    

    # MODEL:
    # mag - magnitude
    # x0, y0 - object offset from the detected position at t=0 in arcsec
    # t0 - calculated from input time points; should be calculated so that it
    #      corresponds to the object's detected position (from the coadd)
    #      as much as possible
    # vx0, vy0 - object speed in x and y directions in arcsec per day
    if denormalize:
        params = denormalize_params(params)
    mag, x0, y0, vx0, vy0 = params
    t0 = calculate_t0(imagedata)
    # t0 = times[0]
    obj = MovingSource(ra+x0/3600, dec+y0/3600, t0, vx0, vy0, mag)

    if dist_penalty_multi is not None:
        MIN_DIST = 2 * 4.6831063024143785e-05 * 3600  # 2 pixels in arcsec
        dist_from_detection = np.sqrt(x0 ** 2 + y0 ** 2)
        distance_penalty = (dist_from_detection - MIN_DIST) * dist_penalty_multi
        if distance_penalty < 0:
            distance_penalty = 0
        distance_penalty = distance_penalty ** 2

    # chi2 degrees of freedom, i.e. number of pixels
    # doftotal = 0
    total = None  # np.zeros(cutouts[0].shape).flatten()

    for _i in range(len(cutouts)):
        if fast_model:
            # sourcexy = wcss[_i].skyToPixel(coord)
            # computeImage(sourcexy).
            simimg = generate_model_faster(obj, calexp_clones[_i].getMetadata()['MJD'], ras[_i], decs[_i], 
                                           psfs[_i], wcss[_i], 
                                           calexp_clones[_i].getPhotoCalib().getCalibrationMean(), 
                                           image_size=IMGSIZE)
        else:
            simimg = generate_model(obj, ras[_i], decs[_i], calexp_clones[_i], image_size=IMGSIZE)

        if returnimgs:
            simimgs.append(simimg.copy())

        diffimg = cutouts[_i] - simimg
        if returnimgs:
            diffimgs.append(diffimg.copy())
        msk = prepmasks[_i]
        varimg = variances[_i]
        imgsum = diffimg
        imgsum[msk] = (diffimg[msk] / np.sqrt(varimg[msk]))
        imgsum[~msk] = np.inf
        if returnimgs:
            maskedimgs.append(imgsum.copy())
        imgsum = imgsum[np.isfinite(imgsum)].flatten()

        if dist_penalty_multi is not None:
            imgsum = np.append(imgsum, [distance_penalty], axis=0)

        if total is not None:
            total = np.concatenate([total, imgsum])
        else:
            total = imgsum
        # total = np.nansum([total, imgsum], axis=0)

    if verbose == 2:
        print(f"{total.dtype} sumsq: {np.sum(total ** 2)}")

    if returnimgs:
        return (simimgs, diffimgs, maskedimgs, len(total))
    else:
        return total  # np.concatenate([total, [distance_penalty]])


def trajectory_diff(obj1, obj2, mjds):
    ''' Returns mean distance in arcsec of trajectories predicted by
        obj1 and obj2 at time points mjds '''
    ras1 = np.array([obj1.ra0 + (mjd - obj1.mjd0)*obj1.sra/3600 for mjd in mjds])
    ras2 = np.array([obj2.ra0 + (mjd - obj2.mjd0)*obj2.sra/3600 for mjd in mjds])
    decs1 = np.array([obj1.dec0 + (mjd - obj1.mjd0)*obj1.sdec/3600 for mjd in mjds])
    decs2 = np.array([obj2.dec0 + (mjd - obj2.mjd0)*obj2.sdec/3600 for mjd in mjds])

    return np.sum(np.sqrt((ras2-ras1) ** 2 + (decs2 - decs1) ** 2))/len(mjds) * 3600


class MultifitResult(object):
    def __init__(self, mov_src, res_norm, res_denorm, errors, errors_denorm, C, result,
                 real_solution, real_obj, object_mjds):
        self.moving_source = mov_src
        self.x_normalized = res_norm
        self.x = res_denorm
        self.errors_normalized = errors
        self.errors = errors_denorm
        self.C = C
        self.result_object = result
        self.real_solution = real_solution
        self.real_source = real_obj
        self.object_mjds = object_mjds

    def get_trajectory_diff(self):
        if self.real_source is None:
            return [np.inf] * len(self.object_mjds)
        return trajectory_diff(self.moving_source, self.real_source, self.object_mjds)

    def get_mag_diff(self):
        if self.real_source is None:
            return np.inf
        return self.moving_source.mag - self.real_source.mag

    def __getstate__(self):
        # omitting result_object because those can be HUGE
        return [self.moving_source, self.x_normalized, self.x, self.errors_normalized,
                self.errors, self.C, self.real_solution, self.real_source, self.object_mjds]

    def __setstate__(self, depickl):
        mov_src, res_norm, res_denorm, errors, errors_denorm, C, real_params, realsrc, mjds = depickl
        self.moving_source = mov_src
        self.x_normalized = res_norm
        self.x = res_denorm
        self.errors_normalized = errors
        self.errors = errors_denorm
        self.C = C
        self.real_solution = real_params
        self.real_source = realsrc
        self.object_mjds = mjds

    def __repr__(self):
        return f"ESTIMATED OBJECT: {self.moving_source}, " + \
            f"REAL OBJECT: {self.real_source}, " + \
            f"\nX: {self.x}, " + \
            f"\nX normalized: {self.x_normalized}, " + \
            f"\nStandard deviations: {self.errors}, " + \
            f"\nStandard deviations normalized: {self.errors_normalized}, " + \
            f"\nCovariance matrix: {self.C}, " + \
            f"\nReal parameters: {self.real_solution}, " + \
            f"\nAverage trajectory diff: {self.get_trajectory_diff()}, " + \
            f"\nMagnitude diff: {self.get_mag_diff()}, " + \
            "\nAlgorithm resulting object from the field 'result_object' ommited..."


def generate_random_params(nsamples, priors=None, mag_rrange=6, mag_roffs=19, pos0_rrange=1/10,
                           pos0_roffs=-1/20, v_rrange=1/100, v_roffs=-1/200):
    if priors is not None:
        means, stdevs = priors
        mags = means[0] + np.random.randn(nsamples) * stdevs[0]
        x0s = means[1] + np.random.randn(nsamples) * stdevs[1]
        y0s = means[2] + np.random.randn(nsamples) * stdevs[2]
        vxs = means[3] + np.random.randn(nsamples) * stdevs[3]
        vys = means[4] + np.random.randn(nsamples) * stdevs[4]
    else:
        mags = mag_rrange * np.random.rand(nsamples) + mag_roffs
        x0s = pos0_rrange * np.random.rand(nsamples) + pos0_roffs
        y0s = pos0_rrange * np.random.rand(nsamples) + pos0_roffs
        vxs = v_rrange * np.random.rand(nsamples) + v_roffs
        vys = v_rrange * np.random.rand(nsamples) + v_roffs
    return np.stack([mags, x0s, y0s, vxs, vys]).T


def get_object(params, imagedata):
    if params is None:
        return None
    mag, x0, y0, vx, vy = params
    t0 = calculate_t0(imagedata)
    return MovingSource(imagedata['ra']+x0/3600, imagedata['dec']+y0/3600, t0, vx, vy, mag)


def do_lmfit(imagedata, butler, real_solution, start_params=None, clones=None,
             in_collection='u//single_frame3', method='leastsq', bounds=(-np.inf, np.inf),
             do_normalize=False, verbose=0, fast_model=False):
    start_mag = 20

    x0 = [start_mag, 0, 0, 0, 0] if start_params is None else start_params
    if do_normalize:
        x0 = normalize_params(x0)

    maxs = [27.5, 2*MAX_PIXEL_SCALE, 2*MAX_PIXEL_SCALE, MAX_SPEED, MAX_SPEED]
    mins = [15, -2*MAX_PIXEL_SCALE, -2*MAX_PIXEL_SCALE, -MAX_SPEED, -MAX_SPEED]
    if do_normalize:
        maxs = normalize_params(maxs)
        mins = normalize_params(mins)
    fit_params = Parameters()
    fit_params.add('mag', value=x0[0], max=maxs[0], min=mins[0])
    fit_params.add('x0', value=x0[1], max=maxs[1], min=mins[1])
    fit_params.add('y0', value=x0[2], max=maxs[2], min=mins[2])
    fit_params.add('vx', value=x0[3], max=maxs[3], min=mins[3])
    fit_params.add('vy', value=x0[4], max=maxs[4], min=mins[4])

    # print("Getting clones")
    if clones is None:
        clones = get_calexp_clones(butler, imagedata['dataIds'], in_collection)
    # print("Getting clones done")

    def _lnlike_fun_wrap(params, imagedata, calexp_clones, returnimgs=False, verbose=0):
        ps = (params['mag'], params['x0'], params['y0'], params['vx'], params['vy'])
        return lnlike_fun(ps, imagedata, calexp_clones, returnimgs, verbose, do_normalize, fast_model=fast_model)

    result = minimize(_lnlike_fun_wrap, fit_params, args=(imagedata, clones, False, verbose),
                      method=method, calc_covar=True)

    if not hasattr(result, 'covar') or result.covar is None:
        raise LinAlgError("Covariance not found")

    C = result.covar
    diag = np.diag(result.covar)
    if (diag < 0).any():
        raise LinAlgError("Negative variances")
    errors = np.sqrt(diag)
    errors_denorm = errors
    if do_normalize:
        errors_denorm = denormalize_params(errors)

    resmag, resx0, resy0, resvx, resvy = (result.params['mag'].value, result.params['x0'].value,
                                          result.params['y0'].value, result.params['vx'].value,
                                          result.params['vy'].value)

    res_norm = np.array([resmag, resx0, resy0, resvx, resvy])
    if do_normalize:
        resmag, resx0, resy0, resvx, resvy = denormalize_params([resmag, resx0, resy0, resvx, resvy])

    res_denorm = np.array([resmag, resx0, resy0, resvx, resvy])
    ms = get_object(res_denorm, imagedata)

    if verbose > 0:
        print(f"result: {(resmag, resx0, resy0, resvx, resvy)}")
        print(f"estimated object: {ms}")

    return MultifitResult(ms, res_norm, res_denorm, errors, errors_denorm, C, result,
                          real_solution, get_object(real_solution, imagedata), imagedata['times'])

def do_multifit(imagedata, butler, real_solution, start_params=None, clones=None,
                in_collection='u//single_frame3', do_normalize=False, method='powell',
                bounds=(-np.inf, np.inf), verbose=0, fast_model=False):
    if method in ["cg", "lbfgsb", "bfgs", "tnc", "trust-constr", "slsqp", "shgo",
                  "nelder", "differential_evolution", "cobyla", "powell"]:
        return do_lmfit(imagedata, butler, real_solution, start_params, clones, in_collection,
                        method, bounds, do_normalize, verbose, fast_model=fast_model)

    start_mag = 20
    errors = None
    C = None

    x0 = [start_mag, 0, 0, 0, 0] if start_params is None else start_params
    if do_normalize:
        x0 = normalize_params(x0)

    if clones is None:
        clones = get_calexp_clones(butler, imagedata['dataIds'], in_collection)

    if method == 'leastsq':  # use old leastsq method
        result = leastsq(lnlike_fun, x0, args=(imagedata, clones, False, verbose, do_normalize, None, fast_model),
                         full_output=1)

        res_denorm = result[0]
        res_norm = res_denorm.copy()

        fjac = result[2]['fjac']
        if verbose > 1:
            print(f"fjac: {fjac}")

        # From: https://scipy-user.scipy.narkive.com/VSuLTUDT/optimize-leastsq-and-uncertianty-in-results
        # errors = np.sqrt(np.diagonal(np.linalg.inv(fjac @ fjac.T)))

        # from: https://mail.python.org/pipermail/scipy-user/2005-March/004209.html
        ipvt = result[2]['ipvt']
        if verbose > 1:
            print(f"ipvt: {ipvt}")

        n = len(x0)
        perm = np.mat(np.take(np.eye(n), ipvt-1, axis=1))
        if verbose > 1:
            print(f"perm: {perm}")

        r = np.mat(np.triu(fjac[:, :n]))
        if verbose > 1:
            print(f"r: {r}")

        jac = np.dot(r, perm)
        if verbose > 1:
            print(f"jac: {jac}")

        # C = (R.T @ R).I
        # errors = np.sqrt(np.diagonal(C))
    elif method in ['trf', 'dogbox', 'lm']:  # use least_squares method
        result = least_squares(lnlike_fun, x0, args=(imagedata, clones, False, verbose, do_normalize, None, fast_model),
                               method=method, verbose=verbose, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        res_denorm = result.x
        res_norm = res_denorm.copy()

        jac = result.jac
    else:
        raise NotImplementedError("Method not supported")

    # Jacobian J (result.jac): J^T J is a Gauss-Newton approximation of the
    # Hessian of the cost function.
    # Hessian is an inverse of covariance matrix C
    # so...
    if jac is None:
        raise LinAlgError("Jacobian not found")

    hess = jac.T @ jac
    C = np.linalg.inv(hess)  # this can raise an Exception

    errors = np.sqrt(np.diagonal(C))
    errors_denorm = errors.copy()
    if do_normalize:
        res_denorm = denormalize_params(res_denorm)
        errors_denorm = denormalize_range(errors)

    if verbose > 0:
        # print(f"result: {x}, msg: {msg}, flag: {flag}")
        print(result)
        print(f"Solution:     {np.array(res_denorm)}")
        print(f"Start params: {np.array(denormalize_params(x0))}")

    resmag, resx0, resy0, resvx, resvy = res_denorm

    ms = get_object(res_denorm, imagedata)

    return MultifitResult(ms, res_norm, res_denorm, errors, errors_denorm, C, result,
                          real_solution, get_object(real_solution, imagedata), imagedata['times'])


def create_output_schema():
    # dtypes = {
    #     "String": str,
    #     "B": np.uint8,
    #     "U": np.uint16,
    #     "I": np.int32,
    #     "L": np.int64,
    #     "F": np.float32,
    #     "D": np.float64,
    #     "Angle": lsst.geom.Angle,
    # }
    # def addField(self, field, type=None, doc="", units="", size=None,
    #         doReplace=False, parse_strict="raise"):
    """Add a field to the Schema.
    Parameters
    ----------
    field : `str` or `Field`
        The string name of the Field, or a fully-constructed Field object.
        If the latter, all other arguments besides doReplace are ignored.
    type : `str`, optional
        The type of field to create.  Valid types are the keys of the
        afw.table.Field dictionary.
    doc : `str`
        Documentation for the field.
    unit : `str`
        Units for the field, or an empty string if unitless.
    size : `int`
        Size of the field; valid for string and array fields only.
    doReplace : `bool`
        If a field with this name already exists, replace it instead of
        raising pex.exceptions.InvalidParameterError.
    parse_strict : `str`
        One of 'raise' (default), 'warn', or 'strict', indicating how to
        handle unrecognized unit strings.  See also astropy.units.Unit.
    Returns
    -------
    result :
        Result of the `Field` addition.
    """
    schema = afwTable.SourceTable.makeMinimalSchema()
    # schema.addField('id', "L", "Unique id.")
    # schema.addField('parent', "L", "Parent source id.")
    # CoordKey.addFields(schema, "coord", "position in ra/dec")
    schema.addField('ra0', "F", "Estimated RA position at t0.",
                    units="degree")
    schema.addField('dec0', "F", "Estimated Dec position at t0.",
                    units="degree")
    schema.addField('mag', "F", "Estimated magnitude.", units="mag")
    schema.addField('t0', "F", "MJD time point taken as central time (average time of all calexps).",
                    units="day")
    schema.addField('x0', "F", "Estimated offset at t0 from RA in the input catalog.",
                    units="arcsecond")
    schema.addField('y0', "F", "Estimated offset at t0 from Dec in the input catalog.",
                    units="arcsecond")
    schema.addField('vx', "D", "Estimated speed in RA direction in arcseconds per day.",
                    units="arcsecond")
    schema.addField('vy', "D", "Estimated speed in Dec direction in arcseconds per day.",
                    units="arcsecond")
    schema.addField('mag_err', "D", "Forced PSF flux measured on the direct image.",
                    units="arcsecond")
    schema.addField('x0_err', "D", "Forced PSF flux measured on the direct image.",
                    units="arcsecond")
    schema.addField('y0_err', "D", "Forced PSF flux measured on the direct image.",
                    units="arcsecond")
    schema.addField('vx_err', "D", "Forced PSF flux measured on the direct image.",
                    units="arcsecond")
    schema.addField('vy_err', "D", "Forced PSF flux measured on the direct image.",
                    units="arcsecond")
    schema.addField('chi_sq', "D", "Total chi2.")
    schema.addField('chi_sq_per_dof', "D", "Total chi2 per degrees of freedom.")
    # chi square
    return schema
