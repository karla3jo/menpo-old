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
    images = [rescale_to_reference_landmarks(i, reference_shape, scale=scale,
                                             group=group, label=label,
                                             interpolator=interpolator)
              for i in images]

    # TODO:
    print '- Building gaussian pyramids'
    # build gaussian pyramids
    images_pyramid = [gaussian_pyramid(i, n_levels=n_levels) for i in images]

    return images_pyramid


def regression_training_set(images, aam, n_perturbations, noise_std,
                            regressors, group, label):

    # initialize the list of features (independent variables) and the list
    # of delta_ps (dependent variables)
    features = []
    delta_ps = []

    # for each image
    for i in images:

        # obtain image's landmarks
        ground_truth_landmarks = i.landmarks[group][label].lms
        # TODO: This can be a method 'obtain_shape_params' in AAMs
        # compute the ground truth parameters of the model driven transform
        # defining the position of the previous landmarks
        aam.md_transform.target = ground_truth_landmarks
        p = aam.md_transform.as_vectors()

        # for n_perturbations
        for k in range(n_perturbations):

            # perturb ground truth landmarks
            global_transform = align_with_noise(aam.shape_model.mean,
                                                ground_truth_landmarks,
                                                noise_std)
            perturbed_landmarks = global_transform.apply(aam.shape_model.mean)

            # compute the ground truth parameters of the model driven transform
            # defining the position of the previously perturbed landmarks
            aam.md_transform.target = perturbed_landmarks
            p_k = aam.md_transform.as_vectors()

            # for each computed regressor already computed
            for r in regressors:
                # compute regression features at the perturbed landmarks
                # positions
                regression_features = aam.regression_features(
                    i, perturbed_landmarks)
                # use the previous features and the regressor to correct the
                # model driven transform parameters
                delta_p = np.dot(r, regression_features)
                p_k = p_k + delta_p
                # correct the perturbed landmarks using the previously
                # corrected parameters
                perturbed_landmarks = \
                    aam.md_transform.from_vector_inplace(p_k).target

            # compute regression features at the estimated landmarks positions
            features.append(aam.regression_features(i, perturbed_landmarks))
            # compute the difference between the true and the estimated
            # parameters
            delta_ps.append(p - p_k)

    return delta_p, features





def linear_regression(Os, Ts):

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
                        n_levels=3, levels=[0, 1, 2],
                        n_perturbations=10, noise_std=0.05):

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

    # build gaussian pyramid
    images_pyramid = build_gaussian_pyramid(
        images, group=group, label=label, interpolator=interpolator,
        reference_landmarks=reference_landmarks, scale=scale,
        crop_boundary=crop_boundary, n_levels=n_levels)

    # initialize lists of regressors and covariances
    regressors = []
    covariances = []

    # for each level
    for l in levels:

        print ' - level {}'.format(l)

        # obtain level images
        images = [i[l] for i in images_pyramid]

        features, delta_p = regression_training_set(
            images, aam, n_perturbations, noise_std, regressors, group, label)

        r, cov = linear_regression(features, delta_p)

        # compute current root mean squared error
        estimated_delta_p = np.dot(r, features)
        error = compute_rmse(delta_p, estimated_delta_p)
        print '  - error = {}'.format(error)

        # add the regressor and covariance to their respective lists
        regressors.append(r)
        covariances.append(cov)

    return 1
