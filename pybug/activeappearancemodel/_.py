from __future__ import division
import numpy as np
from skimage.transform import pyramid_gaussian
from pybug.io import auto_import
from pybug.shape import PointCloud
from pybug.landmark.labels import labeller, ibug_68_points, ibug_68_trimesh
from pybug.transform import Scale, Translation
from pybug.transform.affine import UniformScale
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.groupalign import GeneralizedProcrustesAnalysis
from pybug.image import MaskedNDImage
from pybug.model import PCAModel


def aam_builder(path, max_images=None, group='PTS', label='all',
                crop_boundary=0.2, interpolator='scipy', warp_kwargs=None,
                scale=1, max_shape_components=25, reference_frame_boundary=3,
                labels=None, triangulation_label='ibug_68_trimesh',
                n_multiresolution_levels=3,
                transform_cls=PiecewiseAffineTransform,
                max_appearance_components=250):

    # TODO:
    # load images
    images = auto_import(path, max_images=max_images)
    # crop images around their landmarks
    for i in images:
        i.crop_to_landmarks_proportion(group=group, label=label,
                                       boundary=crop_boundary)

    # TODO:
    # extract shapes
    shapes = [i.landmarks[group][label].lms.points for i in images]
    # define reference shape
    reference_shape = PointCloud(np.mean(shapes, axis=0))
    # compute scale difference between all shapes and reference shape
    scales = [UniformScale.align(s, reference_shape).as_vector()
              for s in shapes]
    # rescale all images using previous scales
    images = [i.rescale(s, interpolator=interpolator, **warp_kwargs)
              for i, s in zip(images, scales)]
    # extract rescaled shapes
    shapes = [i.landmarks[group][label].lms.points for i in images]

    # TODO:
    pyramid_iterator = [pyramid_gaussian(i.pixels) for i in images]

    # TODO:
    # centralize shapes
    centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source for s in gpa.transforms]

    # TODO:
    # scale shape if necessary
    if scale is not 1:
        aligned_shapes = [Scale(scale, n_dims=reference_shape.n_dims).apply(s)
                          for s in aligned_shapes]
    # build shape model
    shape_model = PCAModel(aligned_shapes)
    # trim shape model if required
    if max_shape_components is not None:
        shape_model.trim_components(max_shape_components)

    # TODO:
    # scale reference shape if necessary
    if scale is not 1:
        scaled_reference_shape = Scale(
            scale, n_dims=reference_shape.n_dims).apply(reference_shape)
    # compute lower bound
    lower_bound = scaled_reference_shape.bounds(
        boundary=reference_frame_boundary)[0]
    # translate reference shape using lower bound
    reference_landmarks = Translation(-lower_bound).apply(
        scaled_reference_shape)
    # compute reference frame resolution
    reference_resolution = reference_landmarks.range(
        boundary=reference_frame_boundary)
    # build reference frame
    reference_frame = MaskedNDImage.blank(reference_resolution)
    # assign landmarks using the default group
    reference_frame.landmarks[group] = reference_landmarks
    # label reference frame
    for l in labels:
        labeller([reference_frame], group, l)
    # check for precomputed triangulation
    if triangulation_label is not None:
        labeller([reference_frame], group, triangulation_label)
        trilist = reference_frame.landmarks[triangulation_label].lms.trilist
    else:
        trilist = None
    # mask reference frame
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    # TODO:
    # extract landmarks
    landmarks = [i.landmarks[group][label].lms.points for i in images]
    # initialize list of appearance models
    appearance_model_list = []

    # for each level
    for j in range(0, n_multiresolution_levels):
        # obtain images
        level_images = [MaskedNDImage(p.next().squeeze()) for p in
                        pyramid_iterator]
        # rescale and reassign landmarks if necessary
        if j is not 0:
            for i, l in zip(level_images, landmarks):
                i.landmarks[group].lms.points = l.points / (2 ** j)
        # mask level_images
        for i in level_images:
            i.constrain_mask_to_landmarks(group=group, trilist=trilist)

        # compute features
        feature_images = [i.normalize_inplace() for i in level_images]

        # compute transforms
        transforms = [transform_cls(reference_frame.landmarks[group].lms,
                                    i.landmarks[group].lms)
                      for i in feature_images]
        # warp images
        warped_images = [i.warp_to(reference_frame.mask, t,
                                   warp_landmarks=True,
                                   interpolator=interpolator, **warp_kwargs)
                         for i, t in zip(level_images, transforms)]
        # label reference frame
        for l in labels:
            labeller([reference_frame], group, l)
        # check for precomputed triangulation
        if triangulation_label is not None:
            labeller([reference_frame], group, triangulation_label)
            trilist = reference_frame.landmarks[triangulation_label].lms.trilist
        else:
            trilist = None
        # mask warped images
        for i in warped_images:
            i.constrain_mask_to_landmarks(group=group, trilist=trilist)

        # build appearance model
        appearance_model = PCAModel(warped_images)
        # trim apperance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)
        # add appearance model to the list
        appearance_model_list.append()







class ActiveAppearanceModel(object):

    def __init__(self):
        pass


def ActiveAppearanceModelBuilder(object):

    def __init__(self):
        self._shapes = None
        self._mean_shape = None
        self._centered_shapes = None
        self._centered_mean_shape = None
        self._aligned_centered_shapes = None
        self._aligned_centered_mean_shape = None

    def load_dataset(self, path, max_images=None):
        self._flush()
        # load the images using the auto-importer
        self.images = auto_import(path, max_images=max_images)

    def crop_images(self, margin=0.2):
        for img in self.images:
            boundary = margin * np.max(img.landmarks['PTS'].lms.range())
            img.crop_to_landmarks(group='PTS', boundary=boundary)

    @staticmethod
    def _compute_mean_shape(shape_list):
        return PointCloud(np.mean([s.points for s in shape_list], axis=0))

    def _flush(self):
        self._shapes = None
        self._mean_shape = None
        self._centered_shapes = None
        self._centered_mean_shape = None
        self._aligned_centered_shapes = None
        self._aligned_centered_mean_shape = None

    @property
    def shapes(self):
        if self._shapes is None:
            self._shapes = [img.landmarks['PTS'].lms for img in self.images]
        return self._shapes

    @property
    def mean_shape(self):
        if self._mean_shape is None:
            self._mean_shape = self._compute_mean_shape(self.shapes)
        return self._mean_shape

    @property
    def centered_shapes(self):
        if self._centered_shapes is None:
            self._centered_shapes = [Translation(-s.centre).apply(s) for s in
                                     self.shapes]
        return self._centered_shapes

    @property
    def centered_mean_shape(self):
        if self._centered_mean_shape is None:
            self._centered_mean_shape = self._compute_mean_shape(
                self.centered_shapes)
        return self._centered_mean_shape

    @property
    def aligned_centered_shapes(self):
        if self._aligned_centered_shapes is None:
            gpa = GeneralizedProcrustesAnalysis(self.centered_shapes)
            self._aligned_centered_shapes = [s.aligned_source for s in
                                             gpa.transforms]
        return self._aligned_centered_shapes

    @property
    def aligned_centered_mean_shape(self):
        if self._aligned_centered_mean_shape is None:
            self._aligned_centered_mean_shape = self._compute_mean_shape(
                self.aligned_centered_shapes)
        return self._aligned_centered_mean_shape

    def normalize_scale_space(self, reference_shape=None,
                              warp_landmarks=True, interpolator='scipy',
                              **kwargs):
        if reference_shape is None:
            reference_shape = self.aligned_centered_mean_shape

        scales = [UniformScale.align(s, reference_shape).as_vector()
                  for s in self.shapes]

        self._flush()
        self.images = [img.rescale(s, warp_landmarks=warp_landmarks,
                                   interpolator=interpolator, **kwargs)
                       for img, s in zip(self.images, scales)]

    def gaussian_pyramid(self, n_levels=3):
        pyramid_iterator = [pyramid_gaussian(img.pixels, max_layer=n_levels)
                            for img in self.images]

        return [[img.__class__(it.next().squeeze()) for it, img in
                 zip(pyramid_iterator, self.images)]
                for _ in range(n_levels)]

    def reference_frame(self, reference_shape=None, boundary=3, scale=1.0):
        if reference_shape is None:
            reference_shape = self.aligned_centered_mean_shape
        scaled_mean_shape = Scale(scale, n_dims=reference_shape.n_dims).apply(
            reference_shape)
        lower_bound = scaled_mean_shape.bounds(boundary=boundary)[0]
        landmarks = Translation(-lower_bound).apply(scaled_mean_shape)

        resolution = landmarks.range(boundary=boundary)
        reference_frame = MaskedNDImage.blank(resolution)

        reference_frame.landmarks['PTS'] = landmarks
        labeller([reference_frame], 'PTS', ibug_68_points)
        labeller([reference_frame], 'PTS', ibug_68_trimesh)

        trilist = reference_frame.landmarks['ibug_68_trimesh'].lms.trilist
        reference_frame.constrain_mask_to_landmarks(group='PTS',
                                                    trilist=trilist)

        return reference_frame

    def warp_to_reference_frame(self, reference_frame,
                                transform=PiecewiseAffineTransform,
                                warp_landmarks=True, interpolator='scipy',
                                **kwargs):
        transforms = [transform(
            reference_frame.landmarks['ibug_68_trimesh'].lms,
            img.landmarks['PTS'].lms) for img in self.images]
        return [img.warp_to(reference_frame.mask, t,
                            warp_landmarks=warp_landmarks,
                            interpolator=interpolator, **kwargs)
                for img, t in zip(self.images, transforms)]


class AAMBuilder(DatasetManager)

    def __init__()

