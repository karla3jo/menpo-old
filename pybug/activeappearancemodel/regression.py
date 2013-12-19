
def extract_local_patches(self, group=None, label=None,
                              patch_size=(16, 16)):
    r"""
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


def train_regressor(images, group='PTS', label='all', interpolator='scipy',
                    reference_landmarks=None, scale=1, crop_boundary=0.2,
                    reference_frame_boundary=3, triangulation=None, n_levels=3,
                    transform_cls=PiecewiseAffineTransform, feature_space=None,
                    max_shape_components=0.95, max_appearance_components=0.95):

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

    # TODO:
    print '- Converting images to greyscale'
    # convert images to greyscale
    images = [i.as_greyscale() if type(i) is RGBImage else i for i in images]

    # TODO:
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
    images = [rescale_to_reference_landmarks(i, reference_shape,
                                                   group=group,
                                                   label=label,
                                                   interpolator=interpolator)
                    for i in images]
    # extract rescaled shapes
    shapes = [i.landmarks[group][label].lms for i in images]

    # TODO:
    print '- Building gaussian pyramids'
    # build gaussian pyramids
    images_pyramid = [gaussian_pyramid(i, max_layer=n_levels) for i in images]

    # free memory
    del images, aligned_shapes, gpa, centered_shapes, shapes


    return AAM(shape_model, reference_frame, appearance_model_pyramid,
               feature_space)
