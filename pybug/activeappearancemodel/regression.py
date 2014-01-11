from __future__ import division, print_function
import abc
import numpy as np
from pybug.transform import SimilarityTransform


def noisy_align(source, target, noise_std=0.05, rotation=False):
    r"""

    Parameters
    ----------
    source :
    target :
    noise_std:
    rotation:

    Returns
    -------
    noisy_transform :
    """
    transform = SimilarityTransform.align(source, target)
    parameters = transform.as_vector()
    if not rotation:
        parameters[1] = 0
    parameter_range = np.hstack((parameters[:2], target.range()))
    noise = (parameter_range * noise_std *
             np.random.randn(transform.n_parameters))
    parameters += noise
    return SimilarityTransform.from_vector(parameters)


def extract_local_patches(image, landmarks, patch_size=(24, 24)):
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
    patch_size = np.array(patch_size)
    patch_half_size = patch_size / 2
    patch_list = []

    for point in landmarks.points:
        start = np.floor(point - patch_half_size).astype(int)
        finish = np.floor(point + patch_half_size).astype(int)
        x, y = np.mgrid[start[0]:finish[0], start[1]:finish[1]]

        # deal with boundaries
        x[x > image.shape[0] - 1] = image.shape[0] - 1
        y[y > image.shape[1] - 1] = image.shape[1] - 1
        x[x < 0] = 0
        y[y < 0] = 0

        # sample patch
        patch_data = image.pixels[x, y, :]
        patch_img = image.__class__(patch_data)
        patch_list.append(patch_img)

    return patch_list


# def polynomial_features(features, order=2, bias=True):
#     r"""
#
#     Parameters
#     ----------
#     features:
#     order:
#     bias:
#
#     Returns
#     -------
#     polynomial_features:
#     """
#     poly_features = features
#     n_order = features
#     len_features = features.shape[0]
#
#     for _ in range(1, order):
#         n_order = np.dot(n_order[..., None], features[None, ...])
#         mask = np.zeros_like(n_order)
#         n_elements = mask.shape[0]
#         for j, r in enumerate(mask):
#             mask[-n_elements:, j] = np.ones(n_elements)
#             n_elements = np.floor((len_features-1)*n_elements/len_features)
#             if n_elements == 0 or j == len_features-1:
#                 break
#         n_order = n_order[mask == 1]
#
#         poly_features = np.hstack((n_order, poly_features))
#
#     if bias:
#         poly_features = np.hstack((poly_features, 1))
#
#     return poly_features


def polynomial_features(features, order=2, bias=False):
    r"""

    Parameters
    ----------
    features:
    order:
    bias:

    Returns
    -------
    polynomial_features:
    """
    poly_features = features
    n_order = features

    for _ in range(1, order):
        n_order = np.dot(n_order[..., None], features[None, ...])
        n_order = np.unique(n_order)
        poly_features = np.hstack((n_order, poly_features))

    if bias:
        poly_features = np.hstack((poly_features, 1))

    return poly_features


def linear_regression(features, targets):
    r"""

    Parameters
    ----------
    features:
    targets:

    Returns
    -------
    w:
    covariance:
    """
    covariance = np.dot(features.T, features)
    covariance = (covariance + covariance.T) / 2
    ft = np.dot(features.T, targets)
    r = np.linalg.solve(covariance, ft)

    return r, covariance


def linear_regression_memory_efficient(features, targets):
    r"""

    Parameters
    ----------
    features:
    targets:

    Returns
    -------
    w:
    covariance:
    """
    # number of independent variables
    m = features.shape[1]
    # number of dependent variables
    k = targets.shape[1]

    covariance = np.zeros((m, m))
    ft = np.zeros((m, k))

    for f, t in zip(features, targets):

        covariance = covariance + np.dot(f[..., None], t[None, ...])
        ft = ft + np.dot(f[..., None], t[None, ...])

    covariance = (covariance + covariance.T) / 2
    r = np.linalg.solve(covariance, ft)

    return r, covariance


class Regressor(object):

    def __init__(self, reference_landmarks, order=1, noise_std=0.05,
                 n_perturbations=10):
        self.reference_landmarks = reference_landmarks
        self.order = order
        self.noise_std = noise_std
        self.n_perturbations = n_perturbations

        self.r = None
        self.cov = None

    def _perturb_landmarks(self, original_landmarks):

        transform = noisy_align(self.reference_landmarks, original_landmarks,
                                self.noise_std)

        return transform.apply(self.reference_landmarks)

    @abc.abstractmethod
    def features(self, image, landmarks):
        pass

    def _polynomial_features(self, features):
        return polynomial_features(features, self.order)

    @abc.abstractmethod
    def _delta_ps(self, original_landmarks, perturbed_landmarks):
        pass

    def _regression_data(self, images, original_landmarks):

        n_images = len(images)
        features = []
        delta_ps = []
        for j, (i, l) in enumerate(zip(images, original_landmarks)):
            for _ in range(self.n_perturbations):
                perturbed_landmarks = self._perturb_landmarks(l)
                features.append(self.features(i, perturbed_landmarks))
                delta_ps.append(self._delta_ps(l, perturbed_landmarks))
            print(' - {} % '.format(round(100*(j+1)/n_images)),
                  end='\r')

        return np.asarray(features), np.asarray(delta_ps)

    def train(self, images, original_landmarks):

        print('- generating regression data')
        features, delta_ps = self._regression_data(images, original_landmarks)

        print('- performing regression')
        self.r, self.cov = linear_regression(features, delta_ps)

        # compute regression error
        print('- computing regression error')
        estimated_delta_ps = np.dot(features, self.r)
        error = np.sqrt(np.mean(np.sum((delta_ps - estimated_delta_ps) ** 2,
                                       axis=1)))
        print(' - error = {}'.format(error))

    @abc.abstractmethod
    def align(self, image, initial_landmarks, **kwargs):
        pass


class ParametricRegressor(Regressor):

    def __init__(self, appearance_model, transform, order=1,
                 features='parameters', update='additive', noise_std=0.05,
                 n_perturbations=10, interpolator='scipy'):

        super(ParametricRegressor, self).__init__(
            transform.source, order=order, noise_std=noise_std,
            n_perturbations=n_perturbations)

        self.appearance_model = appearance_model
        self.template = appearance_model.mean
        self.transform = transform

        self._features = self._select_features(features)
        self._update = self._select_update(update)
        self._interpolator = interpolator

    def _select_features(self, features):
        if features is 'parameters':
            return self._parameters
        elif features is 'appearance':
            return self._appearance
        elif features is 'difference':
            return self._difference
        elif features is 'project_out':
            return self._project_out
        elif features is 'probabilistic':
            return self._probabilistic
        else:
            raise ValueError('Unknown feature string selected. Valid'
                             'options are: parameters, appearance, '
                             'difference, project_out, probabilistic')

    def _parameters(self, warped_image):
        return self.appearance_model.project(warped_image)

    def _appearance(self, warped_image):
        return self.appearance_model.reconstruct(warped_image).as_vector()

    def _difference(self, warped_image):
        return warped_image.as_vector() - self._appearance(warped_image)

    def _project_out(self, warped_image):
        difference = (warped_image.as_vector() -
                      self.appearance_model.mean.as_vector())
        return self.appearance_model.distance_to_subspace_vector(
            difference).flatten()

    def _probabilistic(self, warped_image):
        difference = (warped_image.as_vector() -
                      self.appearance_model.mean.as_vector())
        project_out = self.appearance_model.distance_to_subspace_vector(
            difference).flatten()
        return (project_out +
                self.appearance_model.project_whitened_vector(
                    difference).flatten())

    def _select_update(self, update):
        if update is 'additive':
            return self._additive
        elif update is 'compositional':
            return self._compositional
        else:
            raise ValueError('Unknown update string selected. Valid'
                             'options are: additive, compositional')

    def _additive(self, delta_p):
        return self.transform.from_vector(self.transform.as_vector() +
                                          delta_p)

    # TODO: Make this more efficient
    def _compositional(self, delta_p):
        return self.transform.compose_after(
            self.transform.from_vector(delta_p))

    def features(self, image, landmarks):
        self.transform.target = landmarks
        warped_image = image.warp_to(self.template.mask, self.transform,
                                     interpolator=self._interpolator)
        return self._polynomial_features(self._features(warped_image))

    def _delta_ps(self, original_landmarks, perturbed_landmarks):
        self.transform.target = original_landmarks
        original_ps = self.transform.as_vector()
        self.transform.target = perturbed_landmarks
        perturbed_ps = self.transform.as_vector()
        return original_ps - perturbed_ps

    def align(self, image, initial_landmarks, **kwargs):
        features = self.features(image, initial_landmarks)
        delta_p = np.dot(features, self.r)
        return self._update(delta_p)


class NonParametricRegressor(Regressor):

    def __init__(self, reference_landmarks, patch_size=(24, 24), order=1,
                 noise_std=0.05, n_perturbations=10,
                 features_dic={'type': 'raw'}):

        super(NonParametricRegressor, self).__init__(
            reference_landmarks, order=order, noise_std=noise_std,
            n_perturbations=n_perturbations)

        self.features_dic = features_dic
        self._features = self._select_features(features_dic['type'])
        self._patch_size = patch_size

    def _select_features(self, features_type):
        if features_type is 'raw':
            return self._raw
        elif features_type is 'norm':
            return self._norm
        elif features_type is 'igo':
            return self._igo
        elif features_type is 'hog':
            return self._hog
        elif features_type is 'sift':
            return self._sift
        elif features_type is 'lbp':
            return self._lbp
        else:
            raise ValueError('Unknown feature string selected. Valid'
                             'options are: igo, hogs, sift, lbp')

    def _raw(self, patches):
        return [p.as_vector() for p in patches]

    def _norm(self, patches):
        normalized_patches = []
        for p in patches:
            p.normalize_inplace(**self.features_dic['options'])
            normalized_patches.append(p.as_vector())
        return normalized_patches

    def _igo(self, patches):
        return [p.igos(**self.features_dic['options']).as_vector()
                for p in patches]

    def _hog(self, patches):
        return [p.hogs(**self.features_dic['options']).as_vector()
                for p in patches]

    def _sift(self, patches):
        raise NotImplementedError('Sift features not supported yet')

    def _lbp(self, patches):
        raise NotImplementedError('LBP features not supported yet')

    def features(self, image, landmarks):
        patches = extract_local_patches(
            image, landmarks, patch_size=self._patch_size)
        return np.asarray(self._features(patches)).flatten()

    def _delta_ps(self, original_landmarks, perturbed_landmarks):
        return (original_landmarks.as_vector() -
                perturbed_landmarks.as_vector())

    def align(self, image, initial_landmarks, **kwargs):
        features = self.features(image, initial_landmarks)
        delta_landmarks = np.dot(features, self.r)
        return initial_landmarks.from_vector(initial_landmarks.as_vector() +
                                             delta_landmarks)


class AppearanceRegressor(Regressor):

    def __init__(self, appearance_model, transform, order=1,
                 features='parameters', update='additive', noise_std=0.05,
                 n_perturbations=10, interpolator='scipy'):

        super(AppearanceRegressor, self).__init__(
            transform.source, order=order, noise_std=noise_std,
            n_perturbations=n_perturbations)

        self.appearance_model = appearance_model
        self.template = appearance_model.mean
        self.transform = transform

        self._features = self._select_features(features)
        self._update = self._select_update(update)
        self._interpolator = interpolator

    def _select_features(self, features):
        if features is 'parameters':
            return self._parameters
        elif features is 'appearance':
            return self._appearance
        elif features is 'difference':
            return self._difference
        elif features is 'project_out':
            return self._project_out
        elif features is 'probabilistic':
            return self._probabilistic
        else:
            raise ValueError('Unknown feature string selected. Valid'
                             'options are: parameters, appearance, '
                             'difference, project_out, probabilistic')

    def _appearance(self, warped_image):
        return self.appearance_model.reconstruct(warped_image).as_vector()

    def _difference(self, warped_image):
        return warped_image.as_vector() - self._appearance(warped_image)

    def _project_out(self, warped_image):
        difference = (warped_image.as_vector() -
                      self.appearance_model.mean.as_vector())
        return self.appearance_model.distance_to_subspace_vector(
            difference).flatten()

    def _probabilistic(self, warped_image):
        difference = (warped_image.as_vector() -
                      self.appearance_model.mean.as_vector())
        project_out = self.appearance_model.distance_to_subspace_vector(
            difference).flatten()
        return (project_out +
                self.appearance_model.project_whitened_vector(
                    difference).flatten())

    def _select_update(self, update):
        if update is 'additive':
            return self._additive
        elif update is 'compositional':
            return self._compositional
        else:
            raise ValueError('Unknown update string selected. Valid'
                             'options are: additive, compositional')

    def _additive(self, delta_p):
        return self.transform.from_vector(self.transform.as_vector() +
                                          delta_p)

    # TODO: Make this more efficient
    def _compositional(self, delta_p):
        return self.transform.compose_after(
            self.transform.from_vector(delta_p))

    def features(self, image, landmarks):
        self.transform.target = landmarks
        warped_image = image.warp_to(self.template.mask, self.transform,
                                     interpolator=self._interpolator)
        return self._polynomial_features(self._features(warped_image))

    def _delta_ps(self, original_landmarks, perturbed_landmarks):
        self.transform.target = original_landmarks
        original_ps = self.transform.as_vector()
        self.transform.target = perturbed_landmarks
        perturbed_ps = self.transform.as_vector()
        return original_ps - perturbed_ps

    def align(self, image, initial_landmarks, **kwargs):
        features = self.features(image, initial_landmarks)
        delta_p = np.dot(self.r, features)
        return self._update(delta_p)

    def _regression_data(self, images, original_landmarks):

        n_images = len(images)
        features = []
        delta_ps = []
        for j, (i, l) in enumerate(zip(images, original_landmarks)):
            for _ in range(self.n_perturbations):
                perturbed_landmarks = self._perturb_landmarks(l)
                features.append(self.features(i, perturbed_landmarks))
                delta_ps.append(self._delta_ps(l, perturbed_landmarks))
            print(' - {} % '.format(round(100*(j+1)/n_images)),
                  end='\r')

        return np.asarray(features), np.asarray(delta_ps)

    def train(self, images, original_landmarks):

        print('- generating regression data')
        features, delta_ps = self._regression_data(images, original_landmarks)

        print('- performing regression')
        self.r, self.cov = linear_regression(delta_ps, features)

        #print(self.r.shape)
        #print(features.shape)

        self.r = np.linalg.pinv(self.r)

        # compute regression error
        print('- computing regression error')
        estimated_delta_ps = np.dot(self.r.T, features.T).T
        error = np.sqrt(np.mean(np.sum((delta_ps - estimated_delta_ps) ** 2,
                                       axis=1)))
        print(' - error = {}'.format(error))