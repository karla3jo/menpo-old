import numpy as np
from pybug.activeappearancemodel.builder import align_with_noise, \
    gaussian_pyramid, rescale_to_reference_landmarks, \
    compute_mean_pointcloud, compute_features


def build_gaussian_pyramid(images, group='PTS', label='all',
                           interpolator='scipy', reference_landmarks=None,
                           scale=1, crop_boundary=0.2, n_levels=3):

    r"""

    Parameters
    ----------
    images:
    group: str, optional

        Default: 'PTS'
    label: tr, optional

        Default: 'all'
    interpolator: str, optional

        Default: 'scipy'
    reference_landmarks: optional

        Default: None
    scale: float, optional

        Default: 1
    crop_boundary: float, optional


    n_levels: int, optional

        Default: 3

    Returns
    -------
    image_pyramid :
    """

    print '- Cropping images'
    # crop images around their landmarks
    for i in images:
        i.crop_to_landmarks_proportion(crop_boundary, group=group, label=label)

    # TODO:
    if reference_landmarks is None:
        print '- Compute reference shape'
        # extract original shapes
        shapes = [i.landmarks[group][label].lms for i in images]
        # define reference shape
        reference_shape = compute_mean_pointcloud(shapes)

    # TODO:
    print '- Rescaling images to reference shape'
    # rescale images so that the scale of their corresponding shapes matches
    # the scale of the reference shape
    images = [rescale_to_reference_landmarks(i, reference_shape, scale=1,
                                             group=group, label=label,
                                             interpolator=interpolator)
              for i in images]

    # TODO:
    print '- Building gaussian pyramids'
    # build gaussian pyramids
    images_pyramid = [gaussian_pyramid(i, n_levels=n_levels) for i in images]

    return images_pyramid


def extract_local_patches(self, group=None, label=None,
                          patch_size=(16, 16)):
    r"""

    Parameters
    ----------
    self:
    group:
    label:
    patch_size

    Returns
    -------
    patches :
    """

    pc = self.landmarks[group][label].lms

    patch_size = np.array(patch_size)
    patch_half_size = patch_size / 2
    patch_list = []

    for point in pc.points:
        start = np.floor(point - patch_half_size).astype(int)
        finish = np.floor(point + patch_half_size).astype(int)
        x, y = np.mgrid[start[0]:finish[0], start[1]:finish[1]]

        # deal with boundaries
        x[x > self.shape[0]] = self.shape[0]
        y[y > self.shape[1]] = self.shape[1]
        x[x < 0] = 0
        y[y < 0] = 0

        # sample patch
        patch_data = self.pixels[x, y, :]
        patch_img = self.__class__(patch_data)
        patch_list.append(patch_img)

    return patch_list


def linear_regression1(Os, Ts):

    OOs = np.dot(Os.T, Os)
    OOs = (OOs + OOs.T) / 2
    OTs = np.dot(Os.T, Ts)
    w = np.linalg.solve(OOs, OTs)

    return w, OOs


def linear_regression2(Os, Ts):

    # number of independent variables
    M = Os.shape[1]
    # number of dependent variables
    K = Ts.shape[1]

    OOs = np.zeros(M, M)
    OTs = np.zeros(M, K)

    for x, y in enumerate(Os, Ts):

        OOs = OOs + np.dot(x, x.T)
        XYs = XYs + np.dot(x, y.T)

    OOs = (OOs + OOs.T) / 2
    w = np.linalg.solve(OOs, OTs)

    return w, OOs


def train_regressor_aam(aam, images, group='PTS', label='all',
                        interpolator='scipy', reference_landmarks=None,
                        scale=1, crop_boundary=0.2,
                        n_levels=3, levels=[0, 1 ,2],
                        n_shapes=[3, 6, 12], n_appearances=[250, 250, 250]):

    r"""

    Parameters
    ----------
    images:
    group: str, optional

        Default: 'PTS'
    label: tr, optional

        Default: 'all'
    interpolator: str, optional

        Default: 'scipy'
    reference_landmarks: optional

        Default: None
    scale: float, optional

        Default: 1
    crop_boundary: float, optional

        Default: 0.2
    reference_frame_boundary: int, optional

        Default: 3
    triangulation: dictionary, optional

        Default: None
    n_levels: int, optional

        Default: 3
    transform_cls: optional

        Default: PieceWiseAffine
    feature_space: dictionary, optional

        Default: None
    max_shape_components: float, optional

        Default: 0.95
    max_appearance_components: float, optional

        Deafult: 0.95


    Returns
    -------
    aam : :class:`pybug.activeappearancemodel.AAM`
    """

    feature_pyramid = [aam._feature_pyramid(f) for f in images]

    # initialize lists of regressors and covariances
    regressors = []
    covariances = []

    # for each level
    for l in levels:

        print ' - level {}'.format(l)

        # obtain level's features
        features = [f[l] for f in feature_pyramid]

        aam._md_transform

        for f in features:

            aam.md_transform.target = f.landmarks[group][label].lms
            p = aam.md_transform.as_vectors()

            for k in range(n_perturbations):

                global_transform = \
                    align_with_noise(aam.shape_model.mean,
                                     f.landmarks[group].lms, noise_std)

                perturbed_landmarks = \
                    global_transform.apply(aam.shape_model.mean)

                aam.md_transform.target = perturbed_landmarks
                p_k = aam.md_transform.as_vectors()

                for w in regressors:

                    regression_features = \
                        aam.regression_features(f, perturbed_landmarks)

                    delta_p = np.dot(w, regression_features)

                    perturbed_landmarks = \
                        aam.md_transform.from_vector(p_k + delta_p).target

                    p_k = aam.md_transform.as_vectors()


                delta_p = p - p_k


        w, cov = linear_regression(regression_features, delta_p)

        p_k = np.dot(w, regression_features)

        regressors.append(w)
        covariances.append(cov)

    return 1
