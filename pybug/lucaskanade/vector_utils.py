import numpy as np


def normalise_vector(v):
    n = v / np.sqrt(np.sum(v ** 2, axis=-1))[..., None]
    n = np.nan_to_num(n)
    return n


def row_norm(v):
    return np.sqrt(np.sum(v ** 2, axis=-1))


def cart2sph(x, y, z, theta_origin='xy'):
    """
    theta_origin : {'xy', 'z'}
        Defines where to take the 0 value for the elevation angle, theta. xy
        implies the origin is at the xy-plane and and 90 is at the z-axis.
        z implies that the origin is at the z-axis and 90 is at the xy-plane.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)

    if theta_origin == 'xy':
        xy = np.sqrt(x**2 + y**2)
        angles = np.concatenate([phi[..., None],
                                 np.arctan2(z, xy)[..., None],
                                 r[..., None]], axis=-1)
    elif theta_origin == 'z':
        angles = np.concatenate([phi[..., None],
                                 np.arccos(z / r)[..., None],
                                 r[..., None]], axis=-1)
    else:
        raise ValueError('Unknown value for the theta origin, valid values '
                         'are: xy, z')
    return angles


def sph2cart(azimuth, elevation, r, theta_origin='xy'):
    """
    theta_origin : {'xy', 'z'}
        Defines where to take the 0 value for the elevation angle, theta. xy
        implies the origin is at the xy-plane and and 90 is at the z-axis.
        z implies that the origin is at the z-axis and 90 is at the xy-plane.
    """
    azi_cos = np.cos(azimuth)
    ele_cos = np.cos(elevation)
    azi_sin = np.sin(azimuth)
    ele_sin = np.sin(elevation)

    if theta_origin == 'xy':
        cart = np.concatenate([(r * ele_cos * azi_cos)[..., None],
                               (r * ele_cos * azi_sin)[..., None],
                               (r * ele_sin)[..., None]], axis=-1)
    elif theta_origin == 'z':
        cart = np.concatenate([(r * ele_sin * azi_cos)[..., None],
                               (r * ele_sin * azi_sin)[..., None],
                               (r * ele_cos)[..., None]], axis=-1)
    else:
        raise ValueError('Unknown value for the theta origin, valid values '
                         'are: xy, z')
    return cart


def normalise_image(image):
    """
    For normalising an image that represents a set of vectors.
    """
    vectors = image.as_vector(keep_channels=True)
    return image.from_vector(normalise_vector(vectors))


class Spherical(object):

    def __init__(self):
        super(Spherical, self).__init__()

    def logmap(self, tangent_vectors):
        if len(tangent_vectors.shape) < 3:
            tangent_vectors = tangent_vectors[None, ...]

        x = tangent_vectors[..., 0]
        y = tangent_vectors[..., 1]
        z = tangent_vectors[..., 2]

        xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        xy = np.sqrt(x ** 2 + y ** 2)

        gx = x / xy
        gy = y / xy
        gz = z / xyz
        sgz = np.sqrt(1 - gz ** 2)

        spher = np.concatenate([gx[..., None], gy[..., None],
                                gz[..., None], sgz[..., None]], axis=-1)
        spher[np.isnan(spher)] = 0.0

        return spher

    def expmap(self, sd_vectors):
        if len(sd_vectors.shape) < 3:
            sd_vectors = sd_vectors[None, ...]

        gx = sd_vectors[..., 0]
        gy = sd_vectors[..., 1]
        gz = sd_vectors[..., 2]
        sgz = sd_vectors[..., 3]

        gzsgz = np.sqrt(gz ** 2 + sgz ** 2)

        gxgy = np.sqrt(gx ** 2 + gy ** 2)

        gx = gx / gxgy
        gy = gy / gxgy
        gz = gz / gzsgz
        sgz = sgz / gzsgz

        phi = np.arctan2(gy, gx)
        theta = np.arctan2(sgz, gz)

        cart = sph2cart(phi, theta, np.ones_like(phi), theta_origin='z')
        cart[np.isnan(cart)] = 0.0

        return cart