import numpy as np
import math
_EPS = np.finfo(float).eps * 4.0

'''
def find_transform_matrix(l1, l2):
    ret = np.dot(np.linalg.inv(l1), l2)
    return ret
'''


'''
def find_transform_matrix(l1, l2):
    # P = np.empty([0, 16])
    P = np.empty([0, 13])
    for pt, ptp in zip(l1, l2):
        x, y, z = pt
        u, v, w = ptp
        # arr1 = np.array([x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])
        # arr2 = np.array([0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0, -v*x, -v*y, -v*z, -v])
        # arr3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1, -w*x, -w*y, -w*z, -w])
        arr1 = np.array([x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0, -u])
        arr2 = np.array([0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0, -v])
        arr3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1, -w])
        P = np.concatenate((P, arr1[np.newaxis, :], arr2[np.newaxis, :], arr3[np.newaxis, :]))
    U, S, Vt = np.linalg.svd(P)
    H = Vt[-1]
    H = np.concatenate((H[:-1], np.array([0,0,0]), H[-1:]))
    # H /= H[-1]
    H = np.reshape(H, (4,4))
    return H
'''

def quaternion_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """
    https://github.com/cgohlke/transformations/blob/deb1a195dab70f0f36365a104f9b70505e37b473/transformations/transformations.py#L920
    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError('input arrays are of wrong shape or type')

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims : 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def vector_norm(data, axis=None, out=None):
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)
    return None


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
    v1 = np.array(v1, dtype=np.float64, copy=False)[:3]
    return affine_matrix_from_points(
        v0, v1, shear=False, scale=scale, usesvd=usesvd
    )


def L2_dist(pnt1, pnt2):
    return np.sqrt(sum((pnt1-pnt2)**2))


def main():
    # blk1 points
    l1 = np.array([[0.922949380986978, 0.03988826901495437, - 0.25543931832312905, 1],
                   [0.9020137551534037, -0.8686446763605445, 6.727590088701047, 1],
                   [-0.9719236463924485, 0.09775513785617239, -0.06244366107081944, 1],
                   [-0.10462858959566317, -0.709808680366534, 6.061564583902041, 1]
                  ])

    # blk2 point
    l2 = np.array([[1.6026888143101714, 1.0198958823694528, -6.004233240203687, 1],
                   [0.3790912935330495, -0.0780825253723753, 0.3879320949984825, 1],
                   [-0.17131868229679376, 1.1000097357483747, -6.152928034818453, 1],
                   [-0.42999441908280667, 0.08113574861038235, -0.3659528471479385, 1]
                  ])
    mtx = superimposition_matrix(l1.T, l2.T, scale=True)
    print(mtx)

    for pnt1, pnt2 in zip(l1, l2):
        warp_pnt = np.dot(mtx, pnt1)
        warp_pnt = warp_pnt[:3]
        dist = L2_dist(warp_pnt, pnt2[:3])
        print(dist)


if __name__ == "__main__":
    main()
