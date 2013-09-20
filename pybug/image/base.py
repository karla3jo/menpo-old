import numpy as np
from copy import deepcopy
from pybug.base import Vectorizable
from pybug.landmark import Landmarkable
from pybug.transform.affine import Translation
from pybug.visualize.base import Viewable, ImageViewer


class AbstractNDImage(Vectorizable, Landmarkable, Viewable):
    r"""
    An abstract representation of an image. All images can be
    vectorized/built from vector, viewed, all have a ``shape``,
    all are ``n_dimensional``, and all have ``n_channels``.

    Images are also :class:`pybug.landmark.Landmarkable`.

    Parameters
    -----------
    image_data: (M, N, ..., C) ndarray
        Array representing the image pixels, with the last axis being
        channels.
    """
    def __init__(self, image_data):
        Landmarkable.__init__(self)
        # asarray will pass through ndarrays unchanged
        image_data = np.asarray(image_data)
        if image_data.ndim < 3:
            raise ValueError("Abstract Images have to build from at least 3D"
                             " image data arrays (2D + n_channels) - a {} "
                             "dim array was provided".format(image_data.ndim))
        self.pixels = image_data

    @classmethod
    def _init_with_channel(cls, image_data_with_channel):
        r"""
        Constructor that always requires the image has a
        channel on the last axis. Only used by from_vector. By default,
        just calls the constructor. Subclasses with constructors that don't
        require channel axes need to overwrite this.
        """
        return cls(image_data_with_channel)

    @property
    def n_dims(self):
        r"""
        The number of dimensions in the image. The minimum possible n_dims is
        2.

        :type: int
        """
        return len(self.shape)

    @property
    def n_pixels(self):
        r"""
        Total number of pixels in the image (``prod(shape)``)

        :type: int
        """
        return self.pixels[..., 0].size

    @property
    def n_elements(self):
        r"""
        Total number of data points in the image (``prod(shape) x
        n_channels``)

        :type: int
        """
        return self.pixels.size

    @property
    def n_channels(self):
        """
        The number of channels on each pixel in the image.

        :type: int
        """
        return self.pixels.shape[-1]

    @property
    def width(self):
        r"""
        The width of the image.

        This is the width according to image semantics, and is thus the size
        of the **second** dimension.

        :type: int
        """
        return self.pixels.shape[1]

    @property
    def height(self):
        r"""
        The height of the image.

        This is the height according to image semantics, and is thus the size
        of the **first** dimension.

        :type: int
        """
        return self.pixels.shape[0]

    @property
    def depth(self):
        r"""
        The depth of the image.

        This is the depth according to image semantics, and is thus the size
        of the **third** dimension. If the n_dim of the image is 2, this is 0.

        :type: int
        """
        if self.n_dims == 2:
            return 0
        else:
            return self.pixels.shape[0]

    @property
    def shape(self):
        r"""
        The shape of the image
        (with ``n_channel`` values at each point).

        :type: tuple
        """
        return self.pixels.shape[:-1]

    @property
    def centre(self):
        r"""
        The geometric centre of the Image - the subpixel that is in the
        middle.

        Useful for aligning shapes and images.

        :type: (D,) ndarray
        """
        # noinspection PyUnresolvedReferences
        return np.array(self.shape, dtype=np.double) / 2

    @property
    def _str_shape(self):
        if self.n_dims > 3:
            return reduce(lambda x, y: str(x) + ' x ' + str(y),
                          self.shape) + ' (in memory)'
        elif self.n_dims == 3:
            return (str(self.width) + 'W x ' + str(self.height) + 'H x ' +
                    str(self.depth) + 'D')
        elif self.n_dims == 2:
            return str(self.width) + 'W x ' + str(self.height) + 'H'

    def as_vector(self, keep_channels=False):
        r"""
        Convert the Image to a vectorized form.

        Parameters
        ----------
        keep_channels : bool, optional

            ========== =================
            Value      Return shape
            ========== =================
            ``True``   (``n_pixels``,``n_channels``)
            ``False``  (``n_pixels`` x ``n_channels``,)
            ========== =================

            Default: ``False``

        Returns
        -------
        vectorized_image : (shape given by ``keep_channels``) ndarray
            Vectorized image
        """
        if keep_channels:
            return self.pixels.reshape([-1, self.n_channels])
        else:
            return self.pixels.flatten()

    def _view(self, figure_id=None, new_figure=False, channel=None, **kwargs):
        r"""
        View the image using the default image viewer. Currently only
        supports the rendering of 2D images.

        Returns
        -------
        image_viewer : :class:`pybug.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within

        Raises
        ------
        DimensionalityError
            If Image is not 2D
        """
        pixels_to_view = self.pixels
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channel=channel).render(**kwargs)

    def crop_self(self, *slice_args):
        r"""
        Crops this image using the given slice objects. Expects
        ``len(args) == self.n_dims``. Landmarks are correctly adjusted so they
        maintain their position relative to the newly cropped image.

        Parameters
        -----------
        slice_args: The slices to take over each axis
        slice_args: List of slice objects

        Returns
        -------
        cropped_image : :class:`self`
            This image, but cropped.
        """
        assert(self.n_dims == len(slice_args))
        self.pixels = self.pixels[slice_args]
        lm_translation = Translation(-np.array([x.start for x in slice_args]))
        # update all our landmarks
        for manager in self.landmarks.values():
            for label, landmarks in manager:
                lm_translation.apply(landmarks)
        return self

    def crop(self, *slice_args):
        r"""
        Returns a cropped version of this image using the given slice
        objects. Expects
        ``len(args) == self.n_dims``. Landmarks are correctly adjusted so they
        maintain their position relative to the newly cropped image.

        Parameters
        -----------
        slice_args: The slices to take over each axis
        slice_args: List of slice objects

        Returns
        -------
        cropped_image : :class:`Image`
            A new instance of self, cropped.
        """
        cropped_image = deepcopy(self)
        return cropped_image.crop_self(*slice_args)
