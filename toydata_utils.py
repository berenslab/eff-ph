import numpy as np

#######################################################
# toy datasets
#######################################################


def get_circle(n=1000, r=1.0):
    """
    Creates equidistant points on a circle around (0,0).
    :param n: number of points
    :param r: radius of the circle
    :return: points on the circle (np.ndarray, (n, 2))
    """
    theta = np.linspace(0, 2*np.pi, n)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.stack([x,y], axis=1)


def get_two_circles(n=1000, r=1.0, sep=2.5):
    """
    Creates equidistant points on two circles around (0,0) and (sep, 0) with a given separation sep.
    :param n: number of poitns
    :param r: radius of the circles
    :param sep: separation of the circle centers in the x-direction
    :return: points on two circles (np.ndarray, (n, 2))
    """
    n1 = n//2
    n2 = n - n1
    r1 = get_circle(n1, r)
    r2 = get_circle(n2, r)

    r2[:, 0] += sep
    return np.concatenate([r1, r2], axis=0)


def get_blob(n=1000):
    """
    Creates n points at (0, 0). The blob is purely formed by noise around these identical points.
    :param n: number of points
    :return: points at (0, 0) (np.ndarray, (n, 2))
    """
    return np.zeros((n, 2))


def get_torus(n, r=1.0, R=2.0, seed=0, uniform=True):
    """
    Creates n points on a torus with major radius R and minor radius r.
    :param n: number of points
    :param r: minor radius
    :param R: major radius
    :param seed: random seed
    :param uniform: whether the points should be uniformly distributed on the torus surface or uniform in the typical
    parametrization
    :return: points on a torus (np.ndarray, (n, 3))
    """
    # note tadaset's torus is non-uniform
    # solution based on https://math.stackexchange.com/questions/2017079/uniform-random-points-on-a-torus
    # or Sampling from a Manifold 2013
    np.random.seed(seed)

    # use rejection sampling to ensure true uniformity
    # phi is the angle for the large ring, theta the angle for the tube

    phi = 2 * np.pi * np.random.rand(n)

    if uniform:
        # we start with 2+ r/R times the number of points because on average only every 1+r/R -th will be accepted
        m = np.ceil(2 + r/R).astype(int)
        theta = 2 * np.pi * np.random.rand(m*n)
        w = np.random.rand(m*n)
        reject = w > (R + r*np.cos(theta)) / (R + r)

        # resample a new batch of n if too many were rejected
        while (1-reject).sum() < n:
            theta_new = 2*np.pi * np.random.rand(2*n)
            w_new = np.random.rand(2*n)

            reject_new = w_new > (R + r*np.cos(theta_new)) / (R + r)

            theta = np.concatenate([theta, theta_new])
            reject = np.concatenate([reject, reject_new])

        theta = theta[~reject][:n]
    else:
        theta = 2 * np.pi * np.random.rand(n)

    x = (R + r*np.cos(theta))*np.cos(phi)
    y = (R + r*np.cos(theta))*np.sin(phi)
    z = r*np.sin(theta)

    return np.stack([x, y, z], axis=1)



def invert_torus(p, r=1.0, R=2.0):
    # Returns parametrization values of a point on a torus of major radius R and minor radius r centered at the origin.
    # Assumes parametrization theta, phi in [0, 2pi)
    # x = (R + r*np.cos(theta))*np.cos(phi)
    # y = (R + r*np.cos(theta))*np.sin(phi)
    # z = r*np.sin(theta)
    # phi is angle of large circle, theta is angle of small circle
    # assumes r < R

    # we need to make case distinctions because the output ranges of arccos and arcsin are not [0, 2pi)

    x, y, z = p.T

    norm_xy = np.sqrt(x ** 2 + y ** 2)

    theta = np.zeros(len(x))

    # get correct theta
    mask1 = norm_xy < R
    theta[mask1] = np.pi - np.arcsin(z[mask1] / r)

    mask2 = ~mask1 * (z >= 0)
    theta[mask2] = np.arcsin(z[mask2] / r)

    mask3 = ~mask1 * (z < 0)
    theta[mask3] = 2 * np.pi + np.arcsin(z[mask3] / r)

    # sanity check
    assert np.all(mask1 + mask2 + mask3)

    # get correct phi
    phi = np.zeros(len(x))
    mask4 = y >= 0
    phi[mask4] = np.arccos(x[mask4] / (R + r * np.cos(theta[mask4])))
    mask5 = ~mask4
    phi[mask5] = 2 * np.pi - np.arccos(x[mask5] / (R + r * np.cos(theta[mask5])))

    return phi, theta


def get_sphere(n=1000, r=1.0, d=3, seed=0):
    """
    Creates n points on a (d-1)-sphere with radius r centered at (0, 0, 0).
    :param n: number of points
    :param r: radius of the sphere
    :param d: ambient dimension of the sphere, i.e., the sphere will be of dimension d-1.
    :param seed: random seed
    :return: points on a (d-1)-sphere (np.ndarray, (n, d))
    """
    np.random.seed(seed)
    x = np.random.randn(n, d)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return r*x




def get_eyeglasses(n_obs, r=1.0):
    '''
    Sample on eyeglasses curve with noise and outliers. Adapted from intrinsicPH GitHub repo
    https://github.com/ximenafernandez/intrinsicPH/blob/main/src/datasets.py. Stripped from random seeds and outlier
    related code.
    Input:
    :param n_obs: an integer, number of points on the Eyeglasses
    :param r: a float, radius of the Eyeglasses
    Output:
    data: a nx2 array, representing points in R^2
    '''

    # number of points on the circle segments
    n = int(0.85 * (n_obs / 2))
    # number of points on the line segments
    m = int(0.15 * (n_obs / 2))
    phi1 = np.linspace(np.pi - 1.2, 2 * np.pi + 1.2, n)
    phi2 = np.linspace(np.pi + 1.92, 2 * np.pi + 4.35, n)

    # x-values of the segments
    seg = np.linspace(-0.53, 0.53, m)

    # circle segments 1 and 2
    x1 = np.sin(phi1) - 1.5
    y1 = np.cos(phi1)
    x2 = np.sin(phi2) + 1.5
    y2 = np.cos(phi2)
    # line segments 1 and 2
    x3 = seg
    y3 = 0.35 * np.ones(m)
    x4 = seg
    y4 = -0.35 * np.ones(m)

    X = np.concatenate([x1, x2, x3, x4])
    Y = np.concatenate([y1, y2, y3, y4])

    data = np.column_stack((X, Y))

    return data * r


def get_eyeglasses_order(n_obs):
    # returns indices that order the eyeglasses dataset along the curve
    n = int(0.85 * (n_obs / 2))
    m = int(0.15 * (n_obs / 2))
    id_normal = np.arange(n_obs, dtype=int)
    return np.concatenate([id_normal[:n], id_normal[2*n:2*n+m], id_normal[n:2*n], id_normal[2*n+m:][::-1]])


def get_interlinked_circles(n, r=1.0):
    """
    Creates n points on two interlinked circles with radius r.
    :param n: number of points on both rings together
    :param r: radius of the rings
    :return:
    """
    # get two rings of half the total number of points
    n1 = n//2
    n2 = n - n1
    c1 = get_circle(n1, r)
    c2 = get_circle(n2, r)

    # add third dimensions
    c1 = np.concatenate([c1, np.zeros(n1)[:, None]], axis=1)
    c2 = np.concatenate([c2, np.zeros(n2)[:, None]], axis=1)
    c2 = np.roll(c2, axis=1, shift=-1)

    # shift second circle by the radius in the first dimension
    c2 += np.array([r, 0, 0])[None]

    return np.concatenate([c1, c2], axis=0)


#######################################################
# noisy ambient embedding
#######################################################

def add_gaussian(x, sigma=0.1, seed=0):
    """
    Adds Gaussian noise to the data.
    :param x: data
    :param sigma: standard deviation of the Gaussian noise
    :param seed: random seed
    :return: noised data
    """
    np.random.seed(seed)
    return x + np.random.normal(0, sigma, size=x.shape)


def add_uniform(x, n_noise=100, seed=0, scale=1.0):
    """
    Adds uniform noise to the data from an axis aligned box around the data.
    :param x: data
    :param n_noise: number of noise points
    :param seed: random seed
    :param scale: multiplicative factor for the size of the box relative to the spread of the data.
    :return:
    """
    # get mins and maxs of the box
    mins = x.min(axis=0) * scale
    maxs = x.max(axis=0) * scale
    # add noise points
    np.random.seed(seed)
    noise = np.random.uniform(mins, maxs, size=(n_noise, x.shape[1]))
    return np.concatenate([x, noise])


def get_orthonormal_basis(out_d=50, in_d=2, seed=0):
    """
    Creates an orthonormal basis for a subspace of dimension in_d in a space of dimension out_d.
    :param out_d: dimension of ambient space
    :param in_d: dimensions in which the data is given
    :param seed: random seed
    :return: basis matrix (in_d, out_d)
    """
    assert out_d >= in_d

    # create orthogonal normal basis by Gauss elemination procedure from random vectors
    np.random.seed(seed)
    basis = np.random.randn(in_d, out_d)
    for i, _ in enumerate(basis):
        basis[i] /= np.linalg.norm(basis[i])
        for j, _ in enumerate(basis):
            if j <= i:
                continue
            basis[j] = basis[j] - np.dot(basis[i], basis[j]) * basis[i]
            assert np.allclose(np.dot(basis[i], basis[j]), 0)  # check that the vectors are orthogonal
    return basis



#######################################################
# wrapper
#######################################################
def get_toy_data(n, dataset, seed=0, r=1.0, d=50, **noise_kwargs):
    """
    Wrapper function to create toy data. The data in embedded into a d-dimensional space and then noise is added.
    :param n: number of points
    :param dataset: name of the dataset. Must be one of 'toy_circle', 'two_circles', 'toy_blob', 'toy_sphere', 'torus',
    'eyeglasses', 'inter_circles'
    :param seed: randome seed
    :param r: radius of the data, used for dataset with circles
    :param d: dimension of the ambient space
    :param noise_kwargs: key word arguments for the noise function
    :return: noised dataset in ambient space (np.ndarray (n, d))
    """
    if dataset == 'toy_circle':
        data = get_circle(n, r=r)
        in_d = 2
    elif dataset == 'two_circles':
        data = get_two_circles(n, r=r, sep=2.5*r)
        in_d = 2
    elif dataset == 'toy_blob':
        data = get_blob(n)
        in_d = 2
    elif dataset == 'toy_sphere':
        data = get_sphere(n, r=r, d=3, seed=seed+1)  # need to use other seed otherwise there is correlation between the data and the noise
        in_d = 3
    elif dataset == "torus":
        data = get_torus(n, r=r, R=2*r, seed=seed+1) # need to use other seed otherwise there is correlation between the data and the noise
        in_d = 3
    elif dataset == "eyeglasses":
        data = get_eyeglasses(n, r=r)
        in_d = 2
    elif dataset == "inter_circles":
        data = get_interlinked_circles(n, r=r)
        in_d = 3
    else:
        raise NotImplementedError

    assert d >= in_d

    # embed in higher dimensions
    basis = get_orthonormal_basis(out_d=d, in_d=in_d, seed=seed)
    data = np.dot(data, basis)

    # add noise
    if "gaussian" in noise_kwargs:
        data = add_gaussian(data, seed=seed, **noise_kwargs["gaussian"])
    if "uniform" in noise_kwargs:
        data = add_uniform(data, seed=seed, **noise_kwargs["uniform"])

    return data