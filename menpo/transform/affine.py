import abc
import copy
import numpy as np

from menpo.base import Vectorizable
from menpo.transform.base import Alignable, VComposableTransform, VInvertible
from menpo.exception import DimensionalityError


class AffineTransform(Vectorizable, VComposableTransform, VInvertible,
                      Alignable):
    r"""
    The base class for all n-dimensional affine transformations. Provides
    methods to break the transform down into it's constituent
    scale/rotation/translation, to view the homogeneous matrix equivalent,
    and to chain this transform with other affine transformations.

    Parameters
    ----------
    h_matrix : (n_dims + 1, n_dims + 1) ndarray
        The homogeneous matrix of the affine transformation.
    """

    @property
    def composes_inplace_with(self):
        return AffineTransform

    def __init__(self, h_matrix):
        super(AffineTransform, self).__init__()
        self._h_matrix = None
        # let the setter handle initialization
        self.h_matrix = h_matrix

    @classmethod
    def _align(cls, source, target, **kwargs):
        r"""
        Alternative Transform constructor. Constructs an AffineTransform by
        finding the optimal transform to align source to target.

        Parameters
        ----------

        source: :class:`menpo.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`menpo.shape.PointCloud`
            The target pointcloud instance used in the alignment

        This is called automatically by align once verification of source and
        target is performed.

        Returns
        -------

        alignment_transform: :class:`menpo.transform.AffineTransform`
            An AffineTransform object that is_alignment.


        Notes
        -----

        We want to find the optimal transform M which satisfies

            M a = b

        where `a` and `b` are the source and target homogeneous vectors
        respectively.

           (M a)' = b'
           a' M' = b'
           a a' M' = a b'

           `a a'` is of shape `(n_dim + 1, n_dim + 1)` and so can be inverted
           to solve for M.

           This approach is the analytical linear least squares solution to
           the problem at hand. It will have a solution as long as `(a a')`
           is non-singular, which generally means at least 2 corresponding
           points are required.
        """
        optimal_h = AffineTransform._build_alignment_h_matrix(source, target)
        affine_transform = AffineTransform(optimal_h)
        affine_transform._source = source
        affine_transform._target = target
        return affine_transform

    def _target_setter(self, new_target):
        self.h_matrix = self._build_alignment_h_matrix(self.source, new_target)
        self._target = new_target

    @property
    def n_dims(self):
        return self.h_matrix.shape[0] - 1

    @property
    def n_parameters(self):
        r"""
        ``n_dims * (n_dims + 1)`` parameters - every element of the matrix bar
        the homogeneous part.

        :type: int

        Examples
        --------
        2D Affine: 6 parameters::

            [p1, p3, p5]
            [p2, p4, p6]

        3D Affine: 12 parameters::

            [p1, p4, p7, p10]
            [p2, p5, p8, p11]
            [p3, p6, p9, p12]
        """
        return self.n_dims * (self.n_dims + 1)

    @property
    def h_matrix(self):
        return self._h_matrix

    @h_matrix.setter
    def h_matrix(self, value):
        shape = value.shape
        if len(shape) != 2 and shape[0] != shape[1]:
            raise ValueError("You need to provide a square homogeneous matrix")
        if self.h_matrix is not None:
            # already have a matrix set! The update better be the same size
            if self.n_dims != shape[0] - 1:
                raise DimensionalityError("Trying to update the homogeneous "
                                          "matrix to a different dimension")
        elif shape[0] - 1 not in [2, 3]:
            raise DimensionalityError("Affine Transforms can only be 2D or 3D")
            # TODO add a check here that the matrix is actually valid
        self._h_matrix = value.copy()

    @property
    def linear_component(self):
        r"""
        Returns just the linear transform component of this affine
        transform.

        :type: (D, D) ndarray
        """
        return self.h_matrix[:-1, :-1]

    @property
    def translation_component(self):
        r"""
        Returns just the translation component.

        :type: (D,) ndarray
        """
        return self.h_matrix[:-1, -1]

    @property
    def has_true_inverse(self):
        return True

    def _build_pseudoinverse(self):
        return AffineTransform(np.linalg.inv(self.h_matrix))

    def __eq__(self, other):
        return np.allclose(self.h_matrix, other.h_matrix)

    def __str__(self):
        rep = repr(self) + '\n'
        rep += str(self.h_matrix) + '\n'
        rep += self._transform_str()
        return rep

    def _transform_str(self):
        r"""
        A string representation explaining what this affine transform does.
        Has to be implemented by base classes.

        Returns
        -------
        str : string
            String representation of transform.
        """
        list_str = [t._transform_str() for t in self.decompose()]
        return reduce(lambda x, y: x + '\n' + y, list_str)

    def _apply(self, x, **kwargs):
        r"""
        Applies this transform to a new set of vectors.

        Parameters
        ----------
        x : (N, D) ndarray
            Array to apply this transform to.


        Returns
        -------
        transformed_x : (N, D) ndarray
            The transformed array.
        """
        return np.dot(x, self.linear_component.T) + self.translation_component

    def _compose_after(self, transform):
        return transform._compose_before(self)

    def _compose_before(self, transform):
        r"""
        Chains an affine family transform with another transform of the
        same family, producing a new transform that is the composition of
        the two.

        .. note::

            This will succeed if and only if transform is a transform of
            that belongs to the affine family of transforms. The type of the
            returned transform is always the first common ancestor between
            self and transform.

        Parameters
        ----------
        transform : :class:`AffineTransform`
            Transform to be applied **after** self

        Returns
        --------
        transform : :class:`AffineTransform`
            The resulting affine transform.
        """
        # note that this overload of the basic compose_before is just to
        # deal with the complexities of maintaining the correct class of
        # transform upon composition
        if isinstance(transform, type(self)):
            new_self = copy.deepcopy(self)
            new_self._compose_before_inplace(transform)
        elif isinstance(self, type(transform)):
            new_self = transform._compose_before(self)
        elif (isinstance(self, SimilarityTransform) and
              isinstance(transform, SimilarityTransform)):
            new_self = SimilarityTransform(self.h_matrix)
            new_self._compose_before_inplace(transform)
        elif isinstance(transform, AffineTransform):
            new_self = AffineTransform(self.h_matrix)
            new_self._compose_before_inplace(transform)
        else:
            # TODO this shouldn't be possible with composes_with
            raise ValueError("Trying to compose_before a {} with "
                             " a {}".format(type(self), type(transform)))
        return new_self

    def _compose_before_inplace(self, transform):
        r"""
        Chains an affine family transform with another transform of the
        exact same type, updating the first to be the compose_before of the
        two.

        Parameters
        ----------
        affine_transform : :class:`AffineTransform`
            Transform to be applied **after** self
        """
        # note we dot this way as we have our data in the transposed
        # representation to normal
        self.h_matrix = np.dot(transform.h_matrix, self.h_matrix)

    def _compose_after_inplace(self, transform):
        r"""
        Chains an affine family transform with another transform of the
        exact same type, updating the first to be the compose_after of the
        two.

        Parameters
        ----------
        affine_transform : :class:`AffineTransform`
            Transform to be applied **before** self
        """
        # note we dot this way as we have our data in the transposed
        # representation to normal
        self.h_matrix = np.dot(self.h_matrix, transform.h_matrix)

    def jacobian(self, points):
        r"""
        Computes the Jacobian of the transform w.r.t the parameters. This is
        constant for affine transforms.

        The Jacobian generated (for 2D) is of the form::

            x 0 y 0 1 0
            0 x 0 y 0 1

        This maintains a parameter order of::

          W(x;p) = [1 + p1  p3      p5] [x]
                   [p2      1 + p4  p6] [y]
                                        [1]

        Parameters
        ----------
        points : (N, D) ndarray
            The set of points to calculate the jacobian for.

        Returns
        -------
        dW_dp : (N, P, D) ndarray
            A (``n_points``, ``n_params``, ``n_dims``) array representing
            the Jacobian of the transform.
        """
        n_points, points_n_dim = points.shape
        if points_n_dim != self.n_dims:
            raise DimensionalityError(
                "Trying to sample jacobian in incorrect dimensions "
                "(transform is {0}D, sampling at {1}D)".format(
                    self.n_dims, points_n_dim))
            # prealloc the jacobian
        jac = np.zeros((n_points, self.n_parameters, self.n_dims))
        # a mask that we can apply at each iteration
        dim_mask = np.eye(self.n_dims, dtype=np.bool)

        for i, s in enumerate(
                range(0, self.n_dims * self.n_dims, self.n_dims)):
            # i is current axis
            # s is slicing offset
            # make a mask for a single points jacobian
            full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=bool)
            # fill the mask in for the ith axis
            full_mask[slice(s, s + self.n_dims)] = dim_mask
            # assign the ith axis points to this mask, broadcasting over all
            # points
            jac[:, full_mask] = points[:, i][..., None]
            # finally, just repeat the same but for the ones at the end
        full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=bool)
        full_mask[slice(s + self.n_dims, s + 2 * self.n_dims)] = dim_mask
        jac[:, full_mask] = 1
        return jac

    def jacobian_points(self, points):
        r"""
        Computes the Jacobian of the transform wrt the points to which
        the transform is applied to. This is constant for affine transforms.

        The Jacobian for a given point (for 2D) is of the form::

            Jx = [(1 + a),     -b  ]
            Jy = [   b,     (1 + a)]
            J =  [Jx, Jy] = [[(1 + a), -b], [b, (1 + a)]]

        where a and b come from:

            W(x;p) = [1 + a   -b      tx] [x]
                     [b       1 + a   ty] [y]
                                          [1]

        Returns
        -------
        dW/dx: dW/dx: (N, D, D) ndarray
            The Jacobian of the transform wrt the points to which the
            transform is applied to.
        """
        return self.linear_component[None, ...]

    @classmethod
    def identity(cls, n_dims):
        return AffineTransform(np.eye(n_dims + 1))

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. This does not
        include the homogeneous part of the warp. Note that it flattens using
        Fortran ordering, to stay consistent with Matlab.

        **2D**

        ========= ===========================================
        parameter definition
        ========= ===========================================
        p1        Affine parameter
        p2        Affine parameter
        p3        Affine parameter
        p4        Affine parameter
        p5        Translation in ``x``
        p6        Translation in ``y``
        ========= ===========================================

        3D and higher transformations follow a similar format to the 2D case.

        Returns
        -------
        params : (P,) ndarray
            The values that paramaterise the transform.
        """
        params = self.h_matrix - np.eye(self.n_dims + 1)
        return params[:self.n_dims, :].flatten(order='F')

    def from_vector_inplace(self, p):
        r"""
        Updates this AffineTransform in-place from the new parameters. See
        from_vector for details of the parameter format
        """
        self.h_matrix = self._h_matrix_from_parameters(p)
        self._sync_target()  # update the target (if we are an alignment)

    @staticmethod
    def _h_matrix_from_parameters(p):
        r"""
        See from_vector for details of the parameter format expected.
        """
        h_matrix = None
        if p.shape[0] is 6:  # 2D affine
            h_matrix = np.eye(3)
            h_matrix[:2, :] += p.reshape((2, 3), order='F')
        elif p.shape[0] is 12:  # 3D affine
            h_matrix = np.eye(4)
            h_matrix[:3, :] += p.reshape((3, 4), order='F')
        else:
            ValueError("Only 2D (6 parameters) or 3D (12 parameters) "
                       "homogeneous matrices are supported.")
        return h_matrix

    @staticmethod
    def _build_alignment_h_matrix(source, target):
        r"""
        See _align() for details. This is a separate method just so it can
        be shared by _target_setter().
        """

        def _homogeneous_points(pc):
            r"""
            Pulls out the points from a pointcloud as homogeneous points of
            shape (n_dims + 1, n_points)
            """
            return np.concatenate((pc.points.T, np.ones(pc.n_points)[None, :]))

        a = _homogeneous_points(source)
        b = _homogeneous_points(target)
        return np.linalg.solve(np.dot(a, a.T), np.dot(a, b.T)).T

    def decompose(self):
        r"""
        Uses an SVD to decompose this transform into discrete Affine
        Transforms.

        Returns
        -------
        transforms: list of :class`DiscreteAffineTransform` that
            Equivalent to this affine transform, such that:

            ``reduce(lambda x,y: x.chain(y), self.decompose()) == self``
        """
        U, S, V = np.linalg.svd(self.linear_component)
        rotation_2 = Rotation(U)
        rotation_1 = Rotation(V)
        scale = Scale(S)
        translation = Translation(self.translation_component)
        return [rotation_1, scale, rotation_2, translation]


class SimilarityTransform(AffineTransform):
    r"""
    Specialist version of an :class:`AffineTransform` that is guaranteed to be
    a Similarity transform.

    Parameters
    ----------
    h_matrix : (D + 1, D + 1) ndarray
        The homogeneous matrix of the similarity transform.
    """

    def __init__(self, h_matrix):
        #TODO check that I am a similarity transform
        super(SimilarityTransform, self).__init__(h_matrix)

    @classmethod
    def _align(cls, source, target, rotation=True, **kwargs):
        """
        Infers the similarity transform relating two vectors with the same
        dimensionality. This is simply the procrustes alignment of the
        source to the target.


        source: :class:`menpo.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`menpo.shape.PointCloud`
            The target pointcloud instance used in the alignment

        rotation: boolean, optional
            If False, the rotation component of the similarity transform is not
            inferred.

            Default: True

        This is called automatically by align once verification of source and
        target is performed.

        Returns
        -------

        alignment_transform: :class:`menpo.transform.SimilarityTransform`
            A SimilarityTransform object that is_alignment.
        """
        similarity = cls._procrustes_alignment(source, target, rotation)
        similarity._source = source
        similarity._target = target
        return similarity

    @staticmethod
    def _procrustes_alignment(source, target, rotation=True):
        r"""
        Returns the similarity transform that aligns the source to the target.
        """
        target_translation = Translation(-target.centre)
        centred_target = target_translation.apply(target)
        # now translate the source to the origin
        translation = Translation(-source.centre)
        # apply the translation to the source
        aligned_source = translation.apply(source)
        scale = UniformScale(target.norm() / source.norm(), source.n_dims)
        scaled_aligned_source = scale.apply(aligned_source)
        # compute the target's inverse translation
        inv_target_translation = target_translation.pseudoinverse
        if rotation:
            # calculate the correlation along each dimension + find the optimal
            # rotation to maximise it
            correlation = np.dot(centred_target.points.T,
                                 scaled_aligned_source.points)
            U, D, Vt = np.linalg.svd(correlation)

            rotation = Rotation(np.dot(U, Vt))
            return translation.compose_before(scale).compose_before(
                rotation).compose_before(inv_target_translation)
        else:
            return translation.compose_before(scale).compose_before(
                inv_target_translation)

    def _target_setter(self, new_target):
        similarity = self._procrustes_alignment(self.source, new_target)
        self.h_matrix = similarity.h_matrix
        self._target = new_target

    @property
    def n_parameters(self):
        r"""
        2D Similarity: 4 parameters::

            [(1 + a), -b,      tx]
            [b,       (1 + a), ty]

        3D Similarity: Currently not supported

        Returns
        -------
        4

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.
        """
        if self.n_dims == 2:
            return 4
        elif self.n_dims == 3:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise ValueError("Only 2D and 3D Similarity transforms "
                             "are currently supported.")

    def jacobian(self, points):
        r"""
        Computes the Jacobian of the transform w.r.t the parameters.

        The Jacobian generated (for 2D) is of the form::

            x -y 1 0
            y  x 0 1

        This maintains a parameter order of::

          W(x;p) = [1 + a  -b   ] [x] + tx
                   [b      1 + a] [y] + ty

        Parameters
        ----------
        points : (N, D) ndarray
            The points to calculate the jacobian over

        Returns
        -------
        dW_dp : (N, P, D) ndarray
            A (``n_points``, ``n_params``, ``n_dims``) array representing
            the Jacobian of the transform.

        Raises
        ------
        DimensionalityError
            ``points.n_dims != self.n_dims`` or transform is not 2D
        """
        n_points, points_n_dim = points.shape
        if points_n_dim != self.n_dims:
            raise DimensionalityError('Trying to sample jacobian in incorrect '
                                      'dimensions (transform is {0}D, '
                                      'sampling at {1}D)'.format(self.n_dims,
                                                                 points_n_dim))
        elif self.n_dims != 2:
            # TODO: implement 3D Jacobian
            raise DimensionalityError("Only the Jacobian of a 2D similarity "
                                      "transform is currently supported.")

        # prealloc the jacobian
        jac = np.zeros((n_points, self.n_parameters, self.n_dims))
        ones = np.ones_like(points)

        # Build a mask and apply it to the points to build the jacobian
        # Do this for each parameter - [a, b, tx, ty] respectively
        self._apply_jacobian_mask(jac, np.array([1, 1]), 0, points)
        self._apply_jacobian_mask(jac, np.array([-1, 1]), 1, points[:, ::-1])
        self._apply_jacobian_mask(jac, np.array([1, 0]), 2, ones)
        self._apply_jacobian_mask(jac, np.array([0, 1]), 3, ones)

        return jac

    def _apply_jacobian_mask(self, jac, param_mask, row_index, points):
        # make a mask for a single points jacobian
        full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=np.bool)
        # fill the mask in for the ith axis
        full_mask[row_index] = [True, True]
        # assign the ith axis points to this mask, broadcasting over all
        # points
        jac[:, full_mask] = points * param_mask

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order ``[a, b, tx, ty]``, given that
        ``a = k cos(theta) - 1`` and ``b = k sin(theta)`` where ``k`` is a
        uniform scale and ``theta`` is a clockwise rotation in radians.

        **2D**

        ========= ===========================================
        parameter definition
        ========= ===========================================
        a         ``a = k cos(theta) - 1``
        b         ``b = k sin(theta)``
        tx        Translation in ``x``
        ty        Translation in ``y``
        ========= ===========================================

        .. note::

            Only 2D transforms are currently supported.

        Returns
        -------
        params : (P,) ndarray
            The values that parameterise the transform.

        Raises
        ------
        DimensionalityError, NotImplementedError
            If the transform is not 2D
        """
        n_dims = self.n_dims
        if n_dims == 2:
            params = self.h_matrix - np.eye(n_dims + 1)
            # Pick off a, b, tx, ty
            params = params[:n_dims, :].flatten(order='F')
            # Pick out a, b, tx, ty
            return params[[0, 1, 4, 5]]
        elif n_dims == 3:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise DimensionalityError("Only 2D and 3D Similarity transforms "
                                      "are currently supported.")

    def from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Similarity: 4 parameters::

            [a, b, tx, ty]

        Parameters
        ----------
        p : (P,) ndarray
            The array of parameters.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Only 2D transforms are supported.
        """
        # See affine from_vector with regards to classmethod decorator
        if p.shape[0] == 4:
            homo = np.eye(3)
            homo[0, 0] += p[0]
            homo[1, 1] += p[0]
            homo[0, 1] = -p[1]
            homo[1, 0] = p[1]
            homo[:2, 2] = p[2:]
            self.h_matrix = homo
            self._sync_target()  # update the target (if we are an alignment)
        elif p.shape[0] == 7:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise DimensionalityError("Only 2D and 3D Similarity transforms "
                                      "are currently supported.")

    def _build_pseudoinverse(self):
        return SimilarityTransform(np.linalg.inv(self.h_matrix))

    @classmethod
    def identity(cls, n_dims):
        return SimilarityTransform(np.eye(n_dims + 1))


class DiscreteAffineTransform(object):
    r"""
    A discrete Affine transform operation (such as a :meth:`Scale`,
    :class:`Translation` or :meth:`Rotation`). Has to be able to invertable.
    Make sure you inherit from :class:`DiscreteAffineTransform` first,
    for optimal ``decompose()`` behavior.
    """

    __metaclass__ = abc.ABCMeta

    def decompose(self):
        r"""
        A :class:`DiscreteAffineTransform` is already maximally decomposed -
        return a copy of self in a list.

        Returns
        -------
        transform : :class:`DiscreteAffineTransform`
            Deep copy of ``self``.
        """
        return [copy.deepcopy(self)]


def Rotation(rotation_matrix):
    r"""
    Factory function for producing :class:`AbstractRotation` transforms.

    Parameters
    ----------
    rotation_matrix : (D, D) ndarray
        A square legal 2D or 3D rotation matrix

    Returns
    -------
    rotation : :class:`Rotation2D` or :class:`Rotation3D`
        A 2D or 3D rotation transform

    Raises
    ------
    DimensionalityError
        Only 2D and 3D transforms are supported.
    """
    if rotation_matrix.shape[0] == 2:
        return Rotation2D(rotation_matrix)
    elif rotation_matrix.shape[0] == 3:
        return Rotation3D(rotation_matrix)
    else:
        raise DimensionalityError("Can only construct 2D or 3D Rotations")
    # TODO build rotations about axis, euler angles etc
    # see http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    # for details.


class AbstractRotation(DiscreteAffineTransform, SimilarityTransform):
    r"""
    Abstract ``n_dims`` rotation transform.

    Parameters
    ----------
    rotation_matrix : (D, D) ndarray
        A valid, square rotation matrix
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, rotation_matrix):
        #TODO check that I am a valid rotation
        h_matrix = np.eye(rotation_matrix.shape[0] + 1)
        h_matrix[:-1, :-1] = rotation_matrix
        super(AbstractRotation, self).__init__(h_matrix)

    @property
    def rotation_matrix(self):
        r"""
        The rotation matrix.

        :type: (D, D) ndarray
        """
        return self.linear_component

    @property
    def inverse(self):
        r"""
        The inverse rotation matrix.

        :type: (D, D) ndarray
        """
        return Rotation(np.linalg.inv(self.rotation_matrix))

    def _transform_str(self):
        axis, rad_angle_of_rotation = self.axis_and_angle_of_rotation()
        if axis is None:
            return "NO OP"
        angle_of_rot = (rad_angle_of_rotation * 180.0) / np.pi
        message = 'CCW Rotation of %d degrees about %s' % (angle_of_rot, axis)
        return message

    @abc.abstractmethod
    def axis_and_angle_of_rotation(self):
        r"""
        Abstract method for computing the axis and angle of rotation.

        Returns
        -------
        axis : (D,) ndarray
            The unit vector representing the axis of rotation
        angle_of_rotation : double
            The angle in radians of the rotation about the axis. The angle is
            signed in a right handed sense.
        """
        pass


class Rotation2D(AbstractRotation):
    r"""
    A 2-dimensional rotation. Parametrised by a single parameter, ``theta``,
    which represents the right-handed rotation around ``[0, 0, 1]``.

    Parameters
    ----------
    rotation_matrix : (2, 2) ndarray
        The 2D rotation matrix.

    Raises
    ------
    DimensionalityError
        Only 2D rotation matrices are supported.
    """

    def __init__(self, rotation_matrix):
        super(Rotation2D, self).__init__(rotation_matrix)
        if self.n_dims != 2:
            raise DimensionalityError("Rotation2D has to be built from a 2D"
                                      " rotation matrix")

    def axis_and_angle_of_rotation(self):
        r"""
        Decomposes this 2D rotation's rotation matrix into a angular rotation
        The rotation is considered in a right handed sense. The axis is, by
        definition, [0, 0, 1].

        Returns
        -------
        axis : (2,) ndarray
            The vector representing the axis of rotation
        angle_of_rotation : double
            The angle in radians of the rotation about the axis. The angle is
            signed in a right handed sense.
        """
        axis = np.array([0, 0, 1])
        test_vector = np.array([1, 0])
        transformed_vector = np.dot(self.rotation_matrix,
                                    test_vector)
        angle_of_rotation = np.arccos(
            np.dot(transformed_vector, test_vector))
        return axis, angle_of_rotation

    @property
    def n_parameters(self):
        r"""
        The number of parameters: 1

        :type: int
        """
        return 1

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [theta].

        +----------+--------------------------------------------+
        |parameter | definition                                 |
        +==========+============================================+
        |theta     | The angle of rotation around ``[0, 0, 1]`` |
        +----------+--------------------------------------------+

        Returns
        -------
        theta : double
            Angle of rotation around axis. Right-handed.
        """
        return self.axis_and_angle_of_rotation()[1]

    def from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Rotation: 1 parameter::

            [theta]

        Parameters
        ----------
        p : (1,) ndarray
            The array of parameters.

        Returns
        -------
        transform : :class:`Rotation2D`
            The transform initialised to the given parameters.
        """
        self.h_matrix[:2, :2] = np.array([[np.cos(p), -np.sin(p)],
                                                    [np.sin(p), np.cos(p)]])
        self._sync_target()  # update the target (if we are an alignment)

    @classmethod
    def identity(cls):
        return Rotation2D(np.eye(2))


class Rotation3D(AbstractRotation):
    r"""
    A 3-dimensional rotation. **Currently no parametrisation is implemented**.

    Parameters
    ----------
    rotation_matrix : (D, D) ndarray
        The 3D rotation matrix.

    Raises
    ------
    DimensionalityError
        Only 3D rotation matrices are supported.
    """

    def __init__(self, rotation_matrix):
        super(Rotation3D, self).__init__(rotation_matrix)
        if self.n_dims != 3:
            raise DimensionalityError("Rotation3D has to be built from a 3D"
                                      " rotation matrix")

    def axis_and_angle_of_rotation(self):
        r"""
        Decomposes this 3D rotation's rotation matrix into a angular rotation
        about an axis. The rotation is considered in a right handed sense.

        Returns
        -------
        axis : (3,) ndarray
            A unit vector, the axis about which the rotation takes place
        angle_of_rotation : double
            The angle in radians of the rotation about the ``axis``.
            The angle is signed in a right handed sense.

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis
        """
        eval_, evec = np.linalg.eig(self.rotation_matrix)
        real_eval_mask = np.isreal(eval_)
        real_eval = np.real(eval_[real_eval_mask])
        evec_with_real_eval = np.real_if_close(evec[:, real_eval_mask])
        error = 1e-7
        below_margin = np.abs(real_eval) < (1 + error)
        above_margin = (1 - error) < np.abs(real_eval)
        re_unit_eval_mask = np.logical_and(below_margin, above_margin)
        evec_with_real_unitary_eval = evec_with_real_eval[:, re_unit_eval_mask]
        # all the eigenvectors with real unitary eigenvalues are now all
        # equally 'valid' if multiple remain that probably means that this
        # rotation is actually a no op (i.e. rotate by 360 degrees about any
        #  axis is an invariant transform) but need to check this. For now,
        # just take the first
        if evec_with_real_unitary_eval.shape[1] != 1:
            # TODO confirm that multiple eigenvalues of 1 means the rotation
            #  does nothing
            return None, None
        axis = evec_with_real_unitary_eval[:, 0]
        axis /= np.sqrt((axis ** 2).sum())  # normalize to unit vector
        # to find the angle of rotation, build a new unit vector perpendicular
        # to the axis, and see how it rotates
        axis_temp_vector = axis - np.random.rand(axis.size)
        perpendicular_vector = np.cross(axis, axis_temp_vector)
        perpendicular_vector /= np.sqrt((perpendicular_vector ** 2).sum())
        transformed_vector = np.dot(self.rotation_matrix,
                                    perpendicular_vector)
        angle_of_rotation = np.arccos(
            np.dot(transformed_vector, perpendicular_vector))
        chirality_of_rotation = np.dot(axis, np.cross(perpendicular_vector,
                                                      transformed_vector))
        if chirality_of_rotation < 0:
            angle_of_rotation *= -1.0
        return axis, angle_of_rotation

    @property
    def n_parameters(self):
        r"""
        Not yet implemented.

        Raises
        -------
        NotImplementedError
            Not yet implemented.
        """
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    def as_vector(self):
        r"""
        Not yet implemented.

        Raises
        -------
        NotImplementedError
            Not yet implemented.
        """
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    def from_vector_inplace(self, p):
        r"""
        Not yet implemented.

        Raises
        -------
        NotImplementedError
            Not yet implemented.
        """
        # See affine from_vector with regards to classmethod decorator
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    @classmethod
    def identity(cls):
        return Rotation3D(np.eye(3))


def Scale(scale_factor, n_dims=None):
    r"""
    Factory function for producing Scale transforms. Zero scale factors are not
    permitted.

    A :class:`UniformScale` will be produced if:

        - A float ``scale_factor`` and a ``n_dims`` kwarg are provided
        - A ndarray scale_factor with shape (``n_dims``, ) is provided with all
          elements being the same

    A :class:`NonUniformScale` will be provided if:

        - A ndarray ``scale_factor`` with shape (``n_dims``, ) is provided with
          at least two differing scale factors.

    Parameters
    ----------
    scale_factor: double or (D,) ndarray
        Scale for each axis.
    n_dims: int
        The dimensionality of the output transform.

    Returns
    -------
    scale : :class:`UniformScale` or :class:`NonUniformScale`
        The correct type of scale

    Raises
    -------
    ValueError
        If any of the scale factors is zero
    """
    from numbers import Number
    if not isinstance(scale_factor, Number):
        # some array like thing - make it a numpy array for sure
        scale_factor = np.asarray(scale_factor)
    if not np.all(scale_factor):
        raise ValueError('Having a zero in one of the scales is invalid')

    if n_dims is None:
        # scale_factor better be a numpy array then
        if np.allclose(scale_factor, scale_factor[0]):
            return UniformScale(scale_factor[0], scale_factor.shape[0])
        else:
            return NonUniformScale(scale_factor)
    else:
        # interpret as a scalar then
        return UniformScale(scale_factor, n_dims)


class NonUniformScale(DiscreteAffineTransform, AffineTransform):
    r"""
    An ``n_dims`` scale transform, with a scale component for each dimension.

    Parameters
    ----------
    scale : (D,) ndarray
        A scale for each axis.
    """

    def __init__(self, scale):
        scale = np.asarray(scale)
        h_matrix = np.eye(scale.size + 1)
        np.fill_diagonal(h_matrix, scale)
        h_matrix[-1, -1] = 1
        AffineTransform.__init__(self, h_matrix)

    @classmethod
    def _align(cls, source, target, **kwargs):
        #TODO scale per dim should be used.
        pass

    @property
    def n_parameters(self):
        """
        The number of parameters: ``n_dims``.

        :type: int

        ``n_dims`` parameters - ``[scale_x, scale_y, ....]`` - The scalar values
        representing the scale across each axis.
        """
        return self.scale.size

    @property
    def scale(self):
        r"""
        The scale vector.

        :type: (D,) ndarray
        """
        return self.h_matrix.diagonal()[:-1]

    def _build_pseudoinverse(self):
        """
        The inverse scale.

        :type: :class:`NonUniformScale`
        """
        return NonUniformScale(1.0 / self.scale)

    def _transform_str(self):
        message = 'NonUniformScale by %s ' % self.scale
        return message

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [s0, s1, ...].

        +----------+--------------------------------------------+
        |parameter | definition                                 |
        +==========+============================================+
        |s0        | The scale across the first axis            |
        +----------+--------------------------------------------+
        |s1        | The scale across the second axis           |
        +----------+--------------------------------------------+
        |...       | ...                                        |
        +----------+--------------------------------------------+
        |sn        | The scale across the nth axis              |
        +----------+--------------------------------------------+

        Returns
        -------
        s : (D,) ndarray
            The scale across each axis.
        """
        return self.scale

    def from_vector_inplace(self, vector):
        r"""
        Updates the NonUniformScale inplace.

        Parameters
        ----------
        vector : (D,) ndarray
            The array of parameters.

        """
        np.fill_diagonal(self.h_matrix, vector)
        self.h_matrix[-1, -1] = 1
        self._sync_target()  # update the target (if we are an alignment)

    @classmethod
    def identity(cls, n_dims):
        return NonUniformScale(np.ones(n_dims))


class UniformScale(DiscreteAffineTransform, SimilarityTransform):
    r"""
    An abstract similarity scale transform, with a single scale component
    applied to all dimensions. This is abstracted out to remove unnecessary
    code duplication.
    """

    def __init__(self, scale, n_dims):
        h_matrix = np.eye(n_dims + 1)
        np.fill_diagonal(h_matrix, scale)
        h_matrix[-1, -1] = 1
        SimilarityTransform.__init__(self, h_matrix)

    @classmethod
    def _align(cls, source, target, **kwargs):
        uniform_scale = cls(target.norm() / source.norm(), source.n_dims)
        uniform_scale._source = source
        uniform_scale._target = target
        return uniform_scale

    def _target_setter(self, new_target):
        new_scale = new_target.norm() / self.source.norm()
        np.fill_diagonal(self.h_matrix, new_scale)
        self.h_matrix[-1, -1] = 1
        self._target = new_target

    @property
    def n_parameters(self):
        r"""
        The number of parameters: 1

        :type: int
        """
        return 1

    @property
    def scale(self):
        r"""
        The single scale value.

        :type: double
        """
        return self.h_matrix[0, 0]

    def _build_pseudoinverse(self):
        r"""
        The inverse scale.

        :type: type(self)
        """
        return type(self)(1.0 / self.scale, self.n_dims)

    def _transform_str(self):
        message = 'UniformScale by %f ' % self.scale
        return message

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [s].

        +----------+--------------------------------+
        |parameter | definition                     |
        +==========+================================+
        |s         | The scale across each axis     |
        +----------+--------------------------------+

        Returns
        -------
        s : double
            The scale across each axis.
        """
        return self.scale

    def from_vector_inplace(self, p):
        np.fill_diagonal(self.h_matrix, p)
        self.h_matrix[-1, -1] = 1
        self._sync_target()  # update the target (if we are an alignment)

    @classmethod
    def identity(cls, n_dims):
        return UniformScale(1, n_dims)


class Translation(DiscreteAffineTransform, SimilarityTransform):
    r"""
    An N-dimensional translation transform.

    Parameters
    ----------
    translation : (D,) ndarray
        The translation in each axis.
    """

    def __init__(self, translation):
        translation = np.asarray(translation)
        h_matrix = np.eye(translation.shape[0] + 1)
        h_matrix[:-1, -1] = translation
        SimilarityTransform.__init__(self, h_matrix)

    @classmethod
    def _align(cls, source, target, **kwargs):
        translation = cls(target.centre - source.centre)
        translation._source = source
        translation._target = target
        return translation

    def _target_setter(self, new_target):
        translation = new_target.centre - self.source.centre
        self.h_matrix[:-1, -1] = translation
        self._target = new_target

    @property
    def n_parameters(self):
        r"""
        The number of parameters: ``n_dims``

        :type: int
        """
        return self.n_dims

    def _build_pseudoinverse(self):
        r"""
        The inverse translation (negated).

        :return: :class:`Translation`
        """
        return Translation(-self.translation_component)

    def _transform_str(self):
        message = 'Translate by %s ' % self.translation_component
        return message

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [t0, t1, ...].

        +-----------+--------------------------------------------+
        |parameter | definition                                  |
        +==========+=============================================+
        |t0        | The translation in the first axis           |
        |t1        | The translation in the second axis          |
        |...       | ...                                         |
        |tn        | The translation in the nth axis             |
        +----------+---------------------------------------------+

        Returns
        -------
        ts : (D,) ndarray
            The translation in each axis.
        """
        return self.h_matrix[:-1, -1]

    def from_vector_inplace(self, p):
        self.h_matrix[:-1, -1] = p
        self._sync_target()  # update the target (if we are an alignment)

    @classmethod
    def identity(cls, n_dims):
        return Translation(np.zeros(n_dims))
