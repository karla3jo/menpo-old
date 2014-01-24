from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from pybug.shape import TriMesh
from pybug.transform .affine import Scale, SimilarityTransform, UniformScale, \
    AffineTransform, Translation
from pybug.groupalign import GeneralizedProcrustesAnalysis
from pybug.transform.modeldriven import OrthoMDTransform, ModelDrivenTransform
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.transform.tps import TPS
from pybug.model import PCAModel
from pybug.lucaskanade.residual import LSIntensity
from pybug.lucaskanade.appearance import AlternatingInverseCompositional
from pybug.aam.functions import \
    (mean_pointcloud, build_reference_frame, build_patch_reference_frame,
     noisy_align, compute_features, compute_error_facesize,
     compute_error_me17, compute_error_p2p, compute_error_rms)


class AAM(object):
    r"""
    Active Appearance Model (AAM) class.

    Parameters
    -----------
    shape_model_pyramid: list of :class:`pybug.model.PCA`
        A list containing the shape models of the AAM.
    appearance_model_pyramid: list of :class:`pybug.model.PCA`
        A list containing the appearance models of the AAM.
    reference_shape: :class:`pybug.shape.PointCloud`
        The reference shape used to build the AAM.
    features: (str, dictionary)
        Tuple specifying the type of features used to build the AAM.
        The first element of the tuple is a string specifying the type of
        features. The second element is a dictionary specifying the
        possible feature options and which is passed through to the
        particular feature method being used. See `pybug.image
        .MaskedNDImage` for details on feature options.
    downscale: float
        The downscale factor used to create the different pyramidal levels
        of the AAM.
    transform_cls: :class:`pybug.transform.PureAlignmentTransform`
        The :class:`pybug.transform.PureAlignmentTransform` used to warp the
        images from which the AAM was constructed.
    interpolator:'scipy' or 'cinterp' or func
        The interpolator used in the previous warps.
    patch_size: tuple of ints or None
        If None, the AAM does not use a Patch-Based representation for its
        appearance models. If tuple, it specifies the size of the patches
        used to build the AAM.
    """

    def __init__(self, shape_model_pyramid, appearance_model_pyramid,
                 reference_shape, features, downscale,
                 scaled_reference_frames, transform_cls, interpolator,
                 patch_size):
        self.shape_model_pyramid = shape_model_pyramid
        self.appearance_model_pyramid = appearance_model_pyramid
        self.reference_shape = reference_shape
        self.features = features
        self.downscale = downscale
        self.scaled_reference_frame = scaled_reference_frames
        self.transform_cls = transform_cls
        self.interpolator = interpolator
        self.patch_size = patch_size

    @property
    def n_levels(self):
        return len(self.appearance_model_pyramid)

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
        r"""
        Creates a novel AAM instance.

        Parameters
        -----------
        shape_weights: (n_weights,) ndarray or list
            Weights of the shape model that should be used to create
            a novel shape instance. If None random weights will be used.

            Default: None
        appearance_weights: (n_weights,) ndarray or list
            Weights of the appearance model that should be used to create
            a novel appearance instance. If None random weights will be used.

            Default: None
        level: int, optional
            Index representing the pyramidal level to be used.

            Default: -1

        Returns
        -------
        image: :class:`pybug.image.masked.MaskedNDImage`
            The novel AAM instance.
        """
        sm = self.shape_model_pyramid[level]
        am = self.appearance_model_pyramid[level]

        template = am.mean
        landmarks = template.landmarks['source'].lms

        if shape_weights is None:
            shape_weights = np.random.randn(sm.n_active_components)
        if appearance_weights is None:
            appearance_weights = np.random.randn(am.n_active_components)

        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)

        n_appearance_weights = len(appearance_weights)
        appearance_weights *= am.eigenvalues[:n_appearance_weights] ** 0.5
        appearance_instance = am.instance(appearance_weights)

        # build reference frame
        if self.patch_size:
            reference_frame = build_patch_reference_frame(
                shape_instance, patch_size=self.patch_size)
        else:
            if type(landmarks) == TriMesh:
                trilist = template.landmarks['source'].lms.trilist
            else:
                trilist = None
            reference_frame = build_reference_frame(
                shape_instance, trilist=trilist)

        transform = self.transform_cls(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.warp_to(reference_frame.mask, transform,
                                           self.interpolator)

    def initialize_lk(self, lk_algorithm=AlternatingInverseCompositional,
                      residual=(LSIntensity,),
                      transform_cls=(OrthoMDTransform, SimilarityTransform),
                      n_shape=None, n_appearance=None):
        r"""
        Initializes Lucas-Kanade fitting procedure.

        Parameters
        -----------
        lk_algorithm: :class:`pybug.lucakanade.appearance`, optional
            The :class:`pybug.lucakanade.appearance` algorithm to be used.

            Default: AlternatingInverseCompositional

        residual: :class:`pybug.lucakanade.residual`, optional
            The :class:`pybug.lucakanade.residual` residual to be used

            Default: 'LSIntensity'

        md_transform_cls: tuple, optional
                          (:class:`pybug.transform.ModelDrivenTransform`,
                           :class:`pybug.transform.Affine`)
            The first element of the tuple is the
            :class:`pybug.transform.ModelDrivenTransform` to be used by the
            LK objects. In case this first element is one of the Global
            version of the ModelDrivenTransform the second element is the
            :class:`pybug.transform.Affine` to be used as a global transform.

            Default: (OrthoMDTransform, SimilarityTransform)

        n_shape: list, optional
            List containing the number of shape components to be used at each
            pyramidal level.

            Default: None

        n_appearance: list, optional
            List containing the number of appearance components to be used at
            each pyramidal level.

            Default: None
        """
        if n_shape is None:
            n_shape = [sm.n_active_components
                       for sm in self.shape_model_pyramid]
        if n_appearance is None:
            n_appearance = [am.n_active_components
                            for am in self.appearance_model_pyramid]

        if len(n_shape) is not self.n_levels:
            ValueError('n_shape must contain'
                       '{} elements'.format(self.n_levels))
        if len(n_appearance) is not self.n_levels:
                ValueError('n_appearance must contain'
                           '{} elements'.format(self.n_levels))

        if len(n_shape) is 1 and self.n_levels > 1:
            n_shape = [n_shape for _ in range(self.n_levels)]
        if len(n_appearance) is 1 and self.n_levels > 1:
            n_appearance = [n_appearance for _ in range(self.n_levels)]

        self._lk_pyramid = []
        for j, (am, sm) in enumerate(zip(self.appearance_model_pyramid,
                                         self.shape_model_pyramid)):

            if n_shape is not None:
                sm.n_active_components = n_shape[j]
            if n_appearance is not None:
                am.n_active_components = n_appearance[j]

            if transform_cls[0] is not ModelDrivenTransform:
                # ToDo: Do we need a blank (identity) method for Transforms?
                global_transform = transform_cls[1](np.eye(3, 3))
                md_transform = transform_cls[0](
                    sm, self.transform_cls, global_transform,
                    source=am.mean.landmarks['source'].lms)
            else:
                md_transform = transform_cls[0](
                    sm, self.transform_cls,
                    source=am.mean.landmarks['source'].lms)

            if len(residual) == 1:
                res = residual[0]()
            else:
                res = residual[0](residual[1])

            self._lk_pyramid.append(lk_algorithm(am, res, md_transform))

    def _lk_fit(self, image_pyramid, initial_landmarks, max_iters=20):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image_pyramid: :class:`pybug.image.MaskedNDImage` or list
            The image representation used by the fitting algorithm.

        initial_landmarks: :class:`pybug.shape.PointCloud`
            The initial position of the landmarks from which the fitting
            procedure is going to start from.

        max_iters: int, optional
            The maximum number of iterations per pyramidal level

            Default: 20

        Returns
        -------
        optimal_transforms: list of
                             :class:`pybug.transform.ModelDrivenTransform`
            A list containing the optimal transform per pyramidal level.
        """

        target = initial_landmarks
        optimal_transforms = []

        for i, lk in zip(image_pyramid, self._lk_pyramid):
            lk.transform.target = target
            md_transform = lk.align(i, lk.transform.as_vector(),
                                    max_iters=max_iters)
            optimal_transforms.append(md_transform)

            if not self.scaled_reference_frame:
                target = Scale(
                    self.downscale, n_dims=md_transform.n_dims).apply(
                        md_transform.target)

        return optimal_transforms

    def _prepare_image(self, image, group, label):
        r"""
        Prepares an image to be be fitted by the AAM. The image is first
        rescaled wrt the reference landmarks, then a gaussian
        pyramid is computed and, finally, features are extracted
        from each pyramidal element.

        Parameters
        -----------
        image: :class:`pybug.image.IntensityImage`
            The image to be fitted.

        Returns
        -------
        image_pyramid: list of :class:`pybug.image.MaskedNDImage`
            The image representation used by the fitting algorithm.
        """
        image = image.rescale_to_reference_landmarks(self.reference_shape,
                                                     group=group, label=label)

        if self.scaled_reference_frame:
            pyramid = image.smoothing_pyramid(n_levels=self.n_levels,
                                              downscale=self.downscale)
        else:
            pyramid = image.gaussian_pyramid(n_levels=self.n_levels,
                                             downscale=self.downscale)

        image_pyramid = [compute_features(p, self.features)
                         for p in pyramid]
        image_pyramid.reverse()
        return image_pyramid

    def lk_fit_landmarked_image(self, image, group=None, label='all',
                                runs=10, noise_std=0.05, rotation=False,
                                max_iters=20, verbose=True, view=False):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image: :class:`pybug.image.IntensityImage`
            The landmarked image to be fitted.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        noise_std: float
            The standard deviation of the white noise used to perturb the
            landmarks.

            Default: 0.05

        rotation: boolean
            If False the second parameter of the SimilarityTransform,
            which captures captures inplane rotations, is set to 0.

            Default:False

        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        verbose: boolean
            If True the error between the ground truth landmarks and the
            result of the fitting is displayed.

            Default: True

        view: boolean
            If True the final result of the fitting procedure is shown.

            Default: False

        Returns
        -------
        optimal_transforms: list of
                             :class:`pybug.transform.ModelDrivenTransform`
            A list containing the optimal transform per pyramidal level.
        """

        image_pyramid = self._prepare_image(image, group=group, label=label)

        original_landmarks = image.landmarks[group][label].lms

        affine_correction = AffineTransform.align(
            image_pyramid[-1].landmarks[group][label].lms, original_landmarks)

        optimal_transforms = []
        for j in range(runs):
            reference_landmarks = self.shape_model_pyramid[0].mean
            scaled_landmarks = image_pyramid[0].landmarks[group][label].lms

            transform = noisy_align(reference_landmarks, scaled_landmarks,
                                    noise_std=noise_std, rotation=rotation)
            initial_landmarks = transform.apply(reference_landmarks)

            optimal_transforms.append(self._lk_fit(
                image_pyramid, initial_landmarks, max_iters=max_iters))

            # ToDo: All this will need to change with the new LK structure
            fitted_landmarks = optimal_transforms[j][-1].target
            fitted_landmarks = affine_correction.apply(fitted_landmarks)
            if verbose:
                error = compute_error_facesize(fitted_landmarks.points,
                                               original_landmarks.points)
                print ' - run {} of {} with error: {}'.format(j+1, runs,
                                                              error)
            if self.scaled_reference_frame:
                scale = 1
            else:
                scale = self.downscale ** (self.n_levels-1)
            image.landmarks['initial_{}'.format(j)] = \
                affine_correction.apply(UniformScale(scale, 2).apply(
                    initial_landmarks))
            image.landmarks['fitted_{}'.format(j)] = fitted_landmarks
            if view:
                image.landmarks['initial_{}'.format(j)].view(
                    include_labels=False)
                image.landmarks['fitted_{}'.format(j)].view(
                    include_labels=False)
                plt.show()

        return optimal_transforms

    def lk_fit_landmarked_database(self, images, group=None, label='all',
                                   runs=10, noise_std=0.5, rotation=False,
                                   max_iters=20, verbose=True, view=False):
        r"""
        Fits the AAM to a list of landmark images using Lukas-Kanade.

        Parameters
        -----------
        images: list of :class:`pybug.image.IntensityImage`
            The list of landmarked images to be fitted.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        noise_std: float
            The standard deviation of the white noise used to perturb the
            landmarks.

            Default: 0.05

        rotation: boolean
            If False the second parameter of the SimilarityTransform,
            which captures captures inplane rotations, is set to 0.

            Default:False

        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        verbose: boolean
            If True the error between the ground truth landmarks and the
            result of the fitting is displayed.

            Default: True

        view: boolean
            If True the final result of the fitting procedure is shown.

            Default: False

        Returns
        -------
        optimal_transforms: list of list
                             :class:`pybug.transform.ModelDrivenTransform`
            A list containing the optimal transform per pyramidal level for
            all images.
        """
        n_images = len(images)
        optimal_transforms = []
        for j, i in enumerate(images):
            if verbose:
                print '- fitting image {} of {}'.format(j+1, n_images)
            optimal_transforms.append(self.lk_fit_landmarked_image(
                i, runs=runs, label=label, noise_std=noise_std,
                rotation=rotation, group=group, max_iters=max_iters,
                verbose=verbose, view=view))

        return optimal_transforms


def aam_builder(images, group=None, label='all', interpolator='scipy',
                reference_shape=None, scale=1, boundary=3,
                transform_cls=PiecewiseAffineTransform, trilist=None,
                patch_size=None, n_levels=3, downscale=2,
                scaled_reference_frames=False, features=None,
                max_shape_components=None, max_appearance_components=None):

    r"""
    Builds an AAM object from a set of landmark images.

    Parameters
    ----------
    images: list of :class:`pybug.image.IntensityImage`
        The set of landmarked images from which to build the AAM

    group : string, Optional
        The key of the landmark set that should be used. If None,
        and if there is only one set of landmarks, this set will be used.

        Default: None

    label: string, Optional
        The label of of the landmark manager that you wish to use. If no
        label is passed, the convex hull of all landmarks is used.

        Default: 'all'

    interpolator:'scipy' or 'cinterp' or func, Optional
        The interpolator that should be used to perform the warps.

        Default: 'scipy'

    reference_landmarks: :class:`pybug.shape.PointCloud`, Optional
        Particular set of reference landmarks that will be used to rescale
        all images and build the reference frame. If None the mean of the
        all landmarks will be used instead.

        Default: None

    scale: float, Optional
        Scale factor to be applied to the previous reference landmarks.
        Allows us to control the size of the reference frame (ie the number
        of pixels constituting the appearance models).

        Default: 1

    boundary: int, Optional
        The number of pixels to be left as a "save" margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3

    transform_cls: :class:`pybug.transform.PureAlignmentTransform`, Optional
        The :class:`pybug.transform.PureAlignmentTransform` that will be
        used to warp the images.

        Default: PieceWiseAffine

    trilist: (t, 3) ndarray, Optional
        Triangle list that will be used to build the reference frame. If None
        defaults to performing Delaunay triangulation on the points.

        Default: None

        .. note::

            This kwarg will be completely ignored if the kwarg transform_cls
            is not set :class:`pybug.transform.PiecewiseAffineTransform` or
            if the kwarg patch_size is not set to None.

    patch_size: tuple of ints or None, Optional
        If tuple, the appearance model of the AAM will be obtained from by
        sampling the appearance patches around the landmarks. If None, the
        standard representation for the AAMs' appearance model will be used
        instead.

        Default: None

        .. note::

            If tuple, the kwarg transform_cls will be automatically set to
            :class:`pybug.transform.TPS`.

    n_levels: int, Optional
        The number of multi-resolution pyramidal levels to be used.

        Default: 3

    downscale: float > 1, Optional
        The downscale factor that will be used to create the different AAM
        pyramidal levels.

        Default: 2

    scaled_reference_frames: boolean, Optional
        If False, the resolution of all reference frame used to build the
        appearance model will be fixed (the original images will be
        both smooth and scaled using a Gaussian pyramid). consequently, all
        appearance models will have the same dimensionality.
        If True, the reference frames used to create the appearance model
        will be themselves scaled (the original images will only be smooth).
        Consequently the dimensionality of all appearance models will be
        different.

        Default: False

    features: tuple or None, Optional
        If tuple, the first element is a string that specifies the type of
        features that will be used to build the AAM. The second element is a
        dictionary that specifies possible feature options and which is passed
        through to the particular feature method being used. See
        `pybug.image.MaskedNDImage` for details on feature options.

        Default: None

    max_shape_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original shape model to be retained.

        Default: None

    max_appearance_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original appearance model to be retained.

        Default: None

    Returns
    -------
    aam : :class:`pybug.aam.AAM`
        The AAM object
    """

    if patch_size is not None:
        transform_cls = TPS

    if reference_shape is None:
        print '- Computing reference shape'
        shapes = [i.landmarks[group][label].lms for i in images]
        reference_shape = mean_pointcloud(shapes)

    print '- Rescaling images to reference shape'
    images = [i.rescale_to_reference_landmarks(reference_shape,
                                               group=group, label=label,
                                               interpolator=interpolator)
              for i in images]

    if scaled_reference_frames:
        print '- Setting gaussian smoothing generators'
        generator = [i.smoothing_pyramid(n_levels=n_levels,
                                         downscale=downscale)
                     for i in images]
    else:
        print '- Setting gaussian pyramid generators'
        generator = [i.gaussian_pyramid(n_levels=n_levels,
                                        downscale=downscale)
                     for i in images]

    print '- Computing features'
    images = [compute_features(g.next(), features) for g in generator]
    # extract potentially rescaled shapes
    shapes = [i.landmarks[group][label].lms for i in images]

    print '- Building shape model'
    # centralize shapes
    centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source for s in gpa.transforms]
    # scale shapes if necessary
    if scale != 1:
        aligned_shapes = [Scale(scale, n_dims=reference_shape.n_dims).apply(s)
                          for s in aligned_shapes]

    # build shape model
    shape_model = PCAModel(aligned_shapes)
    if max_shape_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_shape_components)

    print '- Building reference frame'
    reference_shape = mean_pointcloud(aligned_shapes)
    if patch_size is not None:
        # build patch based reference frame
        reference_frame = build_patch_reference_frame(
            reference_shape, boundary=boundary, patch_size=patch_size)
    else:
        # build reference frame
        reference_frame = build_reference_frame(
            reference_shape, boundary=boundary, trilist=trilist)

    print '- Building model pyramids'
    shape_model_pyramid = []
    appearance_model_pyramid = []
    # for each level
    for j in range(n_levels):
        print ' - Level {}'.format(j)

        if j != 0:
            print ' - Computing features'
            images = [compute_features(g.next(), features) for g in generator]

            print ' - Building shape model'
            if scaled_reference_frames:
                shapes = [Scale(1/downscale,
                                n_dims=reference_shape.n_dims).apply(s)
                          for s in shapes]
                centered_shapes = [Translation(-s.centre).apply(s)
                                   for s in shapes]
                gpa = GeneralizedProcrustesAnalysis(centered_shapes)
                aligned_shapes = [s.aligned_source for s in gpa.transforms]
                if scale != 1:
                    aligned_shapes = \
                        [Scale(scale, n_dims=reference_shape.n_dims).apply(s)
                         for s in aligned_shapes]
                # build shape model
                shape_model = PCAModel(aligned_shapes)
                if max_shape_components is not None:
                    # trim shape model if required
                    shape_model.trim_components(max_shape_components)

                print ' - Building reference frame'
                reference_shape = mean_pointcloud(aligned_shapes)
                if patch_size is not None:
                    # build patch based reference frame
                    reference_frame = build_patch_reference_frame(
                        reference_shape, boundary=boundary,
                        patch_size=patch_size)
                else:
                    # build reference frame
                    reference_frame = build_reference_frame(
                        reference_shape, boundary=boundary, trilist=trilist)

        # add shape model to the list
        shape_model_pyramid.append(shape_model)

        print ' - Computing transforms'
        transforms = [transform_cls(reference_frame.landmarks['source'].lms,
                                    i.landmarks[group][label].lms)
                      for i in images]

        print ' - Warping images'
        images = [i.warp_to(reference_frame.mask, t,
                            interpolator=interpolator)
                  for i, t in zip(images, transforms)]

        for i in images:
            i.landmarks['source'] = reference_frame.landmarks['source']
        if patch_size:
            for i in images:
                i.build_mask_around_landmarks(patch_size, group='source')
        else:
            for i in images:
                i.constrain_mask_to_landmarks(group='source', trilist=trilist)

        print ' - Building appearance model'
        appearance_model = PCAModel(images)
        # trim appearance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)

        # add appearance model to the list
        appearance_model_pyramid.append(appearance_model)

    # reverse the list of shape and appearance models so that they are
    # ordered from lower to higher resolution
    shape_model_pyramid.reverse()
    appearance_model_pyramid.reverse()

    return AAM(shape_model_pyramid, appearance_model_pyramid,
               reference_shape, features, downscale,
               scaled_reference_frames, transform_cls,
               interpolator, patch_size)
