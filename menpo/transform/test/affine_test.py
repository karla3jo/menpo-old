import numpy as np
from numpy.testing import assert_allclose, assert_equal
from menpo.shape import PointCloud
from menpo.transform.affine import Rotation, Translation, \
    AffineTransform, SimilarityTransform, NonUniformScale, \
    Rotation2D, Rotation3D, UniformScale, Scale
from menpo.exception import DimensionalityError
from nose.tools import raises


@raises(DimensionalityError)
def test_1d():
    t_vec = np.array([1])
    Translation(t_vec)


@raises(DimensionalityError)
def test_5d():
    t_vec = np.ones(5)
    Translation(t_vec)


def test_translation():
    t_vec = np.array([1, 2, 3])
    starting_vector = np.random.rand(10, 3)
    transform = Translation(t_vec)
    transformed = transform.apply(starting_vector)
    assert_allclose(starting_vector + t_vec, transformed)


def test_align_2d_translation():
    t_vec = np.array([1, 2])
    translation = Translation(t_vec)
    source = PointCloud(np.array([[0, 1],
                                  [1, 1],
                                  [-1, -5],
                                  [3, -5]]))
    target = translation.apply(source)
    # estimate the transform from source and target
    estimate = Translation.align(source, target)
    # check the estimates is correct
    assert_allclose(translation.h_matrix,
                    estimate.h_matrix)


def test_basic_2d_rotation():
    rotation_matrix = np.array([[0, 1],
                                [-1, 0]])
    rotation = Rotation(rotation_matrix)
    assert_allclose(np.array([0, -1]), rotation.apply(np.array([1, 0])))


def test_basic_2d_rotation_axis_angle():
    rotation_matrix = np.array([[0, 1],
                                [-1, 0]])
    rotation = Rotation(rotation_matrix)
    axis, angle = rotation.axis_and_angle_of_rotation()
    assert_allclose(axis, np.array([0, 0, 1]))
    assert_allclose((90 * np.pi)/180, angle)


def test_align_2d_rotation():
    rotation_matrix = np.array([[0, 1],
                                [-1, 0]])
    rotation = Rotation(rotation_matrix)
    source = PointCloud(np.array([[0, 1],
                                  [1, 1],
                                  [-1, -5],
                                  [3, -5]]))
    target = rotation.apply(source)
    # estimate the transform from source and target
    estimate = Rotation2D.align(source, target)
    # check the estimates is correct
    assert_allclose(rotation.h_matrix,
                    estimate.h_matrix, atol=1e-14)


def test_basic_3d_rotation():
    a = np.sqrt(3.0)/2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    starting_vector = np.array([0, 1, 0])
    transformed = rotation.apply(starting_vector)
    assert_allclose(np.array([0, a, -b]), transformed)


def test_basic_3d_rotation_axis_angle():
    a = np.sqrt(3.0)/2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    axis, angle = rotation.axis_and_angle_of_rotation()
    assert_allclose(axis, np.array([1, 0, 0]))
    assert_allclose((-30 * np.pi)/180, angle)


def test_3d_rotation_inverse_eye():
    a = np.sqrt(3.0)/2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    transformed = rotation.compose_before(rotation.inverse)
    print transformed.h_matrix
    assert_allclose(np.eye(4), transformed.h_matrix, atol=1e-15)


def test_basic_2d_affine():
    linear_component = np.array([[1, -6],
                                 [-3, 2]])
    translation_component = np.array([7, -8])
    h_matrix = np.eye(3, 3)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    affine = AffineTransform(h_matrix)
    x = np.array([[0, 1],
                  [1, 1],
                  [-1, -5],
                  [3, -5]])
    # transform x explicitly
    solution = np.dot(x, linear_component.T) + translation_component
    # transform x using the affine transform
    result = affine.apply(x)
    # check that both answers are equivalent
    assert_allclose(solution, result)
    # create several copies of x
    x_copies = np.array([x, x, x, x, x, x, x, x])
    # transform all of copies at once using the affine transform
    results = affine.apply(x_copies)
    # check that all copies have been transformed correctly
    for r in results:
        assert_allclose(solution, r)


def test_align_2d_affine():
    linear_component = np.array([[1, -6],
                                 [-3, 2]])
    translation_component = np.array([7, -8])
    h_matrix = np.eye(3, 3)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    affine = AffineTransform(h_matrix)
    source = PointCloud(np.array([[0, 1],
                                  [1, 1],
                                  [-1, -5],
                                  [3, -5]]))
    target = affine.apply(source)
    # estimate the transform from source and target
    estimate = AffineTransform.align(source, target)
    # check the estimates is correct
    assert_allclose(affine.h_matrix, estimate.h_matrix)


def test_basic_3d_affine():
    linear_component = np.array([[1, 6, -4],
                                 [-3, -2, 5],
                                 [5, -1, 3]])
    translation_component = np.array([7, -8, 9])
    h_matrix = np.eye(4, 4)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    affine = AffineTransform(h_matrix)
    x = np.array([[0, 1,  2],
                  [1, 1, 1],
                  [-1, 2, -5],
                  [1, -5, -1]])
    # transform x explicitly
    solution = np.dot(x, linear_component.T) + translation_component
    # transform x using the affine transform
    result = affine.apply(x)
    # check that both answers are equivalent
    assert_allclose(solution, result)
    # create several copies of x
    x_copies = np.array([x, x, x, x, x, x, x, x])
    # transform all of copies at once using the affine transform
    results = affine.apply(x_copies)
    # check that all copies have been transformed correctly
    for r in results:
        assert_allclose(solution, r)


def test_basic_2d_similarity():
    linear_component = np.array([[2, -6],
                                 [6, 2]])
    translation_component = np.array([7, -8])
    h_matrix = np.eye(3, 3)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    similarity = SimilarityTransform(h_matrix)
    x = np.array([[0, 1],
                  [1, 1],
                  [-1, -5],
                  [3, -5]])
    # transform x explicitly
    solution = np.dot(x, linear_component.T) + translation_component
    # transform x using the affine transform
    result = similarity.apply(x)
    # check that both answers are equivalent
    assert_allclose(solution, result)
    # create several copies of x
    x_copies = np.array([x, x, x, x, x, x, x, x])
    # transform all of copies at once using the affine transform
    results = similarity.apply(x_copies)
    # check that all copies have been transformed correctly
    for r in results:
        assert_allclose(solution, r)


def test_align_2d_similarity():
    linear_component = np.array([[2, -6],
                                 [6, 2]])
    translation_component = np.array([7, -8])
    h_matrix = np.eye(3, 3)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    similarity = SimilarityTransform(h_matrix)
    source = PointCloud(np.array([[0, 1],
                                  [1, 1],
                                  [-1, -5],
                                  [3, -5]]))
    target = similarity.apply(source)
    # estimate the transform from source and target
    estimate = SimilarityTransform.align(source, target)
    # check the estimates is correct
    assert_allclose(similarity.h_matrix,
                    estimate.h_matrix)


jac_solution2d = np.array(
    [[[0.,  0.],
    [0.,  0.],
    [0.,  0.],
    [0.,  0.],
    [1.,  0.],
    [0.,  1.]],
    [[0.,  0.],
    [0.,  0.],
    [1.,  0.],
    [0.,  1.],
    [1.,  0.],
    [0.,  1.]],
    [[0.,  0.],
    [0.,  0.],
    [2.,  0.],
    [0.,  2.],
    [1.,  0.],
    [0.,  1.]],
    [[1.,  0.],
    [0.,  1.],
    [0.,  0.],
    [0.,  0.],
    [1.,  0.],
    [0.,  1.]],
    [[1.,  0.],
    [0.,  1.],
    [1.,  0.],
    [0.,  1.],
    [1.,  0.],
    [0.,  1.]],
    [[1.,  0.],
    [0.,  1.],
    [2.,  0.],
    [0.,  2.],
    [1.,  0.],
    [0.,  1.]]])

jac_solution3d = np.array(
    [[[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]]])

sim_jac_solution2d = np.array([[[0.,  0.],
                              [0.,  0.],
                              [1.,  0.],
                              [0.,  1.]],
                              [[0.,  1.],
                              [-1.,  0.],
                              [1.,  0.],
                              [0.,  1.]],
                              [[0.,  2.],
                              [-2.,  0.],
                              [1.,  0.],
                              [0.,  1.]],
                              [[1.,  0.],
                              [0.,  1.],
                              [1.,  0.],
                              [0.,  1.]],
                              [[1.,  1.],
                              [-1.,  1.],
                              [1.,  0.],
                              [0.,  1.]],
                              [[1.,  2.],
                              [-2.,  1.],
                              [1.,  0.],
                              [0.,  1.]]])


def test_affine_jacobian_2d_with_positions():
    params = np.array([0, 0.1, 0.2, 0, 30, 70])
    t = AffineTransform.identity(2).from_vector(params)
    explicit_pixel_locations = np.array(
        [[0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2]])
    dW_dp = t.jacobian(explicit_pixel_locations)
    assert_equal(dW_dp, jac_solution2d)


def test_affine_jacobian_3d_with_positions():
    params = np.ones(12)
    t = AffineTransform.identity(3).from_vector(params)
    explicit_pixel_locations = np.array(
        [[0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2, 0],
        [0, 2, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 2, 0],
        [1, 2, 1]])
    dW_dp = t.jacobian(explicit_pixel_locations)
    assert_equal(dW_dp, jac_solution3d)


def test_similarity_jacobian_2d():
    params = np.ones(4)
    t = SimilarityTransform.identity(2).from_vector(params)
    explicit_pixel_locations = np.array(
        [[0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2]])
    dW_dp = t.jacobian(explicit_pixel_locations)
    assert_equal(dW_dp, sim_jac_solution2d)


@raises(DimensionalityError)
def test_similarity_jacobian_3d_raises_dimensionalityerror():
    t = SimilarityTransform(np.eye(4))
    t.jacobian(np.ones([2, 3]))


@raises(DimensionalityError)
def test_similarity_2d_points_raises_dimensionalityerror():
    params = np.ones(4)
    t = SimilarityTransform.identity(2).from_vector(params)
    t.jacobian(np.ones([2, 3]))


def test_similarity_2d_from_vector():
    params = np.array([0.2, 0.1, 1, 2])
    homo = np.array([[params[0] + 1, -params[1], params[2]],
                     [params[1], params[0] + 1, params[3]],
                     [0, 0, 1]])

    sim = SimilarityTransform.identity(2).from_vector(params)

    assert_equal(sim.h_matrix, homo)


def test_similarity_2d_as_vector():
    params = np.array([0.2, 0.1, 1.0, 2.0])
    homo = np.array([[params[0] + 1.0, -params[1], params[2]],
                     [params[1], params[0] + 1.0, params[3]],
                     [0.0, 0.0, 1.0]])

    vec = SimilarityTransform(homo).as_vector()

    assert_allclose(vec, params)


def test_translation_2d_from_vector():
    params = np.array([1, 2])
    homo = np.array([[1, 0, params[0]],
                     [0, 1, params[1]],
                     [0, 0, 1]])

    tr = Translation.identity(2).from_vector(params)

    assert_equal(tr.h_matrix, homo)


def test_translation_2d_as_vector():
    params = np.array([1, 2])
    vec = Translation(params).as_vector()
    assert_allclose(vec, params)


def test_translation_3d_from_vector():
    params = np.array([1, 2, 3])
    homo = np.array([[1, 0, 0, params[0]],
                     [0, 1, 0, params[1]],
                     [0, 0, 1, params[2]],
                     [0, 0, 0, 1]])

    tr = Translation.identity(3).from_vector(params)

    assert_equal(tr.h_matrix, homo)


def test_translation_3d_as_vector():
    params = np.array([1, 2, 3])
    vec = Translation(params).as_vector()
    assert_allclose(vec, params)


def test_uniformscale2d_update_from_vector():
    # make a uniform scale of 1, 2 dimensional
    uniform_scale = UniformScale(1, 2)
    new_scale = 2
    homo = np.array([[new_scale, 0, 0],
                     [0, new_scale, 0],
                     [0, 0, 1]])

    uniform_scale.from_vector_inplace(new_scale)
    assert_equal(uniform_scale.h_matrix, homo)


def test_uniformscale2d_as_vector():
    scale = 2
    vec = UniformScale(scale, 2).as_vector()
    assert_allclose(vec, scale)


def test_nonuniformscale2d_from_vector():
    scale = np.array([1, 2])
    homo = np.array([[scale[0], 0, 0],
                     [0, scale[1], 0],
                     [0, 0, 1]])

    tr = NonUniformScale.identity(2).from_vector(scale)

    assert_equal(tr.h_matrix, homo)


def test_nonuniformscale2d_update_from_vector():
    scale = np.array([3, 4])
    homo = np.array([[scale[0], 0, 0],
                     [0, scale[1], 0],
                     [0, 0, 1]])
    tr = NonUniformScale(np.array([1, 2]))
    tr.from_vector_inplace(scale)
    assert_equal(tr.h_matrix, homo)


def test_nonuniformscale2d_as_vector():
    scale = np.array([1, 2])
    vec = NonUniformScale(scale).as_vector()
    assert_allclose(vec, scale)


def test_uniformscale3d_from_vector():
    scale = 2
    homo = np.array([[scale, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, scale, 0],
                     [0, 0, 0, 1]])

    uniform_scale = UniformScale(1, 3)
    tr = uniform_scale.from_vector(scale)
    assert_equal(tr.h_matrix, homo)


def test_uniformscale3d_as_vector():
    scale = 2
    vec = UniformScale(scale, 3).as_vector()
    assert_allclose(vec, scale)


def test_uniformscale_build_2d():
    scale = 2
    homo = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]])

    tr = UniformScale(scale, 2)
    assert_equal(tr.h_matrix, homo)


def test_uniformscale_build_3d():
    scale = 2
    homo = np.array([[scale, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, scale, 0],
                     [0, 0, 0, 1]])

    tr = UniformScale(scale, 3)

    assert(isinstance(tr, UniformScale))
    assert_equal(tr.h_matrix, homo)


@raises(DimensionalityError)
def test_uniformscale_build_4d_raise_dimensionalityerror():
    UniformScale(1, 4)


def test_scale_build_2d_uniform_pass_dim():
    scale = 2
    ndim = 2
    tr = Scale(scale, ndim)

    assert(isinstance(tr, UniformScale))


def test_scale_build_3d_uniform_pass_dim():
    scale = 2
    ndim = 3
    tr = Scale(scale, ndim)

    assert(isinstance(tr, UniformScale))


def test_scale_build_2d_nonuniform():
    scale = np.array([1, 2])
    tr = Scale(scale)

    assert(isinstance(tr, NonUniformScale))


def test_scale_build_2d_uniform_from_vec():
    scale = np.array([2, 2])
    tr = Scale(scale)

    assert(isinstance(tr, UniformScale))


@raises(ValueError)
def test_scale_zero_scale_raise_valuerror():
    Scale(np.array([1, 0]))


def test_rotation2d_from_vector():
    theta = np.pi / 2
    homo = np.array([[0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

    tr = Rotation2D.identity().from_vector(theta)

    assert_allclose(tr.h_matrix, homo, atol=1**-15)


def test_rotation2d_as_vector():
    theta = np.pi / 2
    rot_matrix = np.array([[0.0, -1.0],
                          [1.0, 0.0]])

    vec = Rotation2D(rot_matrix).as_vector()

    assert_allclose(vec, theta)


@raises(NotImplementedError)
def test_rotation3d_from_vector_raises_notimplementederror():
    Rotation3D.identity().from_vector(0)


@raises(NotImplementedError)
def test_rotation3d_as_vector_raises_notimplementederror():
    homo = np.eye(3)
    Rotation3D(homo).as_vector()


def test_affine_2d_n_parameters():
    homo = np.eye(3)
    t = AffineTransform(homo)
    assert(t.n_parameters == 6)


def test_affine_3d_n_parameters():
    homo = np.eye(4)
    t = AffineTransform(homo)
    assert(t.n_parameters == 12)


def test_similarity_2d_n_parameters():
    homo = np.eye(3)
    t = SimilarityTransform(homo)
    assert(t.n_parameters == 4)


@raises(NotImplementedError)
def test_similarity_3d_n_parameters_raises_notimplementederror():
    homo = np.eye(4)
    t = SimilarityTransform(homo)
    # Raises exception
    t.n_parameters


def test_uniformscale2d_n_parameters():
    scale = 2
    t = UniformScale(scale, 2)
    assert(t.n_parameters == 1)


def test_uniformscale3d_n_parameters():
    scale = 2
    t = UniformScale(scale, 3)
    assert(t.n_parameters == 1)


def test_nonuniformscale_2d_n_parameters():
    scale = np.array([1, 2])
    t = NonUniformScale(scale)
    assert(t.n_parameters == 2)


def test_translation_2d_n_parameters():
    trans = np.array([1, 2])
    t = Translation(trans)
    assert(t.n_parameters == 2)


def test_translation_3d_n_parameters():
    trans = np.array([1, 2, 3])
    t = Translation(trans)
    assert(t.n_parameters == 3)


def test_rotation2d_n_parameters():
    rot_matrix = np.eye(2)
    t = Rotation2D(rot_matrix)
    assert(t.n_parameters == 1)

@raises(NotImplementedError)
def test_rotation3d_n_parameters_raises_notimplementederror():
    rot_matrix = np.eye(3)
    t = Rotation3D(rot_matrix)
    # Throws exception
    t.n_parameters


def test_rotation2d_identity():
    assert_allclose(Rotation2D.identity().h_matrix, np.eye(3))


def test_rotation3d_identity():
    assert_allclose(Rotation3D.identity().h_matrix, np.eye(4))


def test_affine_identity_2d():
    assert_allclose(AffineTransform.identity(2).h_matrix, np.eye(3))


def test_affine_identity_3d():
    assert_allclose(AffineTransform.identity(3).h_matrix, np.eye(4))


def test_similarity_identity_2d():
    assert_allclose(SimilarityTransform.identity(2).h_matrix,
                    np.eye(3))


def test_similarity_identity_3d():
    assert_allclose(SimilarityTransform.identity(3).h_matrix,
                    np.eye(4))


def test_uniformscale_identity_2d():
    assert_allclose(UniformScale.identity(2).h_matrix, np.eye(3))


def test_uniformscale_identity_3d():
    assert_allclose(UniformScale.identity(3).h_matrix, np.eye(4))


def test_nonuniformscale_identity_2d():
    assert_allclose(NonUniformScale.identity(2).h_matrix, np.eye(3))


def test_nonuniformscale_identity_3d():
    assert_allclose(NonUniformScale.identity(3).h_matrix, np.eye(4))


def test_translation_identity_2d():
    assert_allclose(Translation.identity(2).h_matrix, np.eye(3))


def test_translation_identity_3d():
    assert_allclose(Translation.identity(3).h_matrix, np.eye(4))