from __future__ import division
import numpy as np
from copy import deepcopy
from pybug.transform import Scale, SimilarityTransform
from pybug.transform.modeldriven import OrthoMDTransform
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.transform.tps import TPS
from pybug.lucaskanade.residual import LSIntensity
from pybug.lucaskanade.appearance import AlternatingInverseCompositional
from pybug.activeappearancemodel.builder import (build_reference_frame,
                                                 build_patch_reference_frame)


class AAM(object):
    r"""
    An Active Appearance Model (AAM)


    Parameters
    -----------
    shape_model:
    reference_frame:
    appearance_model_pyramid:
    """

    def __init__(self, shape_model, reference_frame,
                 appearance_model_pyramid, features):
        self.shape_model = shape_model
        self.reference_frame = reference_frame
        self.appearance_model_pyramid = appearance_model_pyramid
        self.features = features

    @property
    def n_levels(self):
        return len(self.appearance_model_pyramid)

    def instance(self, shape_weights, appearance_weights, level=0,
                 transform_cls=PiecewiseAffineTransform, patches=False,
                 patch_size=[16, 16], interpolator='scipy', **warp_kwargs):
        r"""
        Create a novel AAM instance using the given shape and appearance
        weights.

        Parameters
        -----------
        shape_weights: (n_weights,) ndarray or list
            Weights of the shape model that should be used to create
            a novel shape instance.

        appearance_weights: (n_weights,) ndarray or list
            Weights of the appearance model that should be used to create
            a novel appearance instance.

        level: int, optional
            Index representing the appearance model to be used to create
            the instance.

            Default: 0
        transform_cls: :class:`pybug.transform.base.PureAlignmentTransform`
            Class of transform that should be used to warp the appearance
            instance onto the reference frame defined by the shape instance.

            Default: PieceWiseAffineTransform
        interpolator: 'scipy' or 'c', optional
            The interpolator that should be used to perform the warp.

            Default: 'scipy'
        kwargs: dict
            Passed through to the interpolator. See `pybug.interpolation`
            for details.

        Returns
        -------
        cropped_image: :class:`pybug.image.masked.MaskedNDImage`
            The novel AAM instance.
        """
        if patches:
            transform_cls = TPS

        # multiply weights by std
        n_shape_weights = len(shape_weights)
        shape_weights *= self.shape_model.eigenvalues[:n_shape_weights] ** 0.5
        # compute shape instance
        shape_instance = self.shape_model.instance(shape_weights)
        # build reference frame using to shape instance
        if patches:
            reference_frame = build_patch_reference_frame(
                shape_instance, patch_size=patch_size)

        else:
            reference_frame = build_reference_frame(shape_instance)[0]

        # select appearance model
        n_appearance_weights = len(appearance_weights)
        appearance_model = self.appearance_model_pyramid[level]
        # multiply weights by std
        appearance_weights *= appearance_model.eigenvalues[
            :n_appearance_weights] ** 0.5
        # compute appearance instance
        appearance_instance = appearance_model.instance(appearance_weights)

        # compute transform mapping appearance instance to previous
        # reference frame
        transform = transform_cls(
            reference_frame.landmarks['source'].lms,
            self.reference_frame.landmarks['source'].lms)
        # return warped appearance
        return appearance_instance.warp_to(reference_frame.mask, transform,
                                           interpolator, **warp_kwargs)

    def initialize_lk(self, md_transform_cls=OrthoMDTransform,
                      transform_cls=PiecewiseAffineTransform,
                      global_transform_cls=SimilarityTransform,
                      lk_algorithm=AlternatingInverseCompositional,
                      residual=LSIntensity, n_shape=None, n_appearance=None):
        r"""
        Initialize a particular Lucas-Kanade algorithm for fitting this AAM
        onto images.

        Parameters
        -----------
        md_transform_cls: :class:`pybug.transform.base.modeldriven`

            Default: OrthoMDTransform
        transform_cls: :class:`pybug.transform.base.PureAlignmentTransform`

            Default: PieceWiseAffineTransform
        global_transform_cls: :class:`pybug.transform.base.affine`

            Default: SimilarityTransform
        lk_algorithm: :class:`pybug.lucakanade.appearance`

            Default: PieceWiseAffineTransform
        residual: :class:`pybug.lucakanade.residual`

            Default: 'LSIntensity'
        """
        self._lk_pyramid = []

        for am, n_s, n_a in zip(self.appearance_model_pyramid, n_shape,
                              n_appearance):

            global_transform = global_transform_cls(np.eye(3, 3))
            source = self.reference_frame.landmarks['source'].lms

            sm = deepcopy(self.shape_model)
            if n_shape is not None:
                sm.n_active_components = n_s
            md_transform = md_transform_cls(sm,
                                            transform_cls,
                                            global_transform,
                                            source=source)

            am = deepcopy(am)
            if n_appearance is not None:
                am.n_active_components = n_a

            self._lk_pyramid.append(lk_algorithm(am, residual(),
                                                 md_transform))

    def lk_fit(self, image_pyramid, initial_landmarks, max_iters=20):
        r"""
        Fit the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image_pyramid:
        initial_landmarks: :class:`pybug.shape.PointCloud`
        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        Returns
        -------
        md_tranform_pyramid:
            A list containing the optimal transform per pyramidal level.
        """

        target = initial_landmarks
        md_transform_pyramid = []

        for i, lk in zip(image_pyramid, self._lk_pyramid):
            lk.transform.target = target
            md_transform = lk.align(i, lk.transform.as_vector(),
                                    max_iters=max_iters)
            md_transform_pyramid.append(md_transform)
            target = Scale(2, n_dims=md_transform.n_dims).apply(
                md_transform.target)

        return md_transform_pyramid

    # def feature_pyramid(self, image):
    #
    #     image = rescale_to_reference(image, self.shape_model)
    #
    #     image_pyramid = gaussian_pyramid(image).reverse()
    #
    #     return [compute_features(i, self.feature_space['type'],
    #                              self.feature_space['options'])
    #             for i in image_pyramid]
