"""
reference:
[1] J. Shi and J. Malik, “Normalized cuts and image segmentation”,
    IEEE Trans. on Pattern Analysisand Machine Intelligence, Vol 22
[2] https://github.com/SatyabratSrikumar/Normalized-Cuts-and-Image-Segmentation-Matlab-Implementation
edited by Henry Yu @ 2021-07-21
"""

import cv2
import numpy as np
from scipy.linalg.decomp import eig
from scipy import sparse
from scipy.sparse.linalg import eigs
import math


class Ncut_origin(object):
    '''
    This class is write for RGB image, so if you want to processing grayscale,
    some adjustment should worked on F_maker, W_maker function :)
    '''

    def __init__(self, img):
        '''
        :param img: better no larger than 300px,300px
        '''
        self.no_rows, self.no_cols, self.channel = img.shape
        self.N = self.no_rows * self.no_cols
        self.V_nodes = self.V_node_maker(img)
        self.X = self.X_maker()
        self.F = self.F_maker(img)
        # parameter for W clculate
        self.r = 2
        self.sigma_I = 4
        self.sigma_X = 6
        # Dense W,D
        self.W = self.W_maker()
        self.D = self.D_maker()

    # V_nodes shape : [self.N,1,3]
    def V_node_maker(self, img):
        b, g, r = cv2.split(img)
        b = b.flatten()
        g = g.flatten()
        r = r.flatten()
        V_nodes = np.vstack((b, g))
        V_nodes = np.vstack((V_nodes, r))
        return V_nodes

    def X_maker(self):
        X_temp = np.arange(self.N)
        X_temp = X_temp.reshape((self.no_rows, self.no_cols))
        X_temp_rows = X_temp // self.no_rows
        X_temp_cols = (X_temp // self.no_cols).T
        X = np.zeros((self.N, 1, 2))
        X[:, :, 0] = X_temp_rows.reshape(self.N, 1)
        X[:, :, 1] = X_temp_cols.reshape(self.N, 1)
        return X

    def F_maker(self, img):
        if self.channel < 2:
            return self.gray_feature_maker(img)
        else:
            return self.color_img_feature_maker(img)

    def gray_feature_maker(self, img):
        print('need to ')

    def color_img_feature_maker(self, img):
        F = img.flatten().reshape((self.N, 1, self.channel))
        F = F.astype('uint8')
        return F

    def W_maker(self):
        X = self.X.repeat(self.N, axis=1)
        X_T = self.X.reshape((1, self.N, 2)).repeat(self.N, axis=0)
        diff_X = X - X_T
        diff_X = diff_X[:, :, 0]**2 + diff_X[:, :, 1]**2

        F = self.F.repeat(self.N, axis=1)
        F_T = self.F.reshape((1, self.N, 3)).repeat(self.N, axis=0)
        diff_F = F - F_T
        diff_F = diff_F[:, :, 0]**2 + diff_F[:, :, 1]**2 + diff_F[:, :, 2]**2

        W_map = diff_X < self.r**2  # valid map for W

        W = np.exp(-((diff_F / (self.sigma_I**2)) +
                   (diff_X / (self.sigma_X**2))))
        WW = W * W_map
        print(f"W max: {WW.max()}")
        print(f"W min: {WW.min()}")
        return WW

    def D_maker(self):
        # D is a diagonal matrix using di as diagonal, di is the sum of weight
        # of node i with all other nodes
        d_i = np.sum(self.W, axis=1)
        D = np.diag(d_i)
        return D

    def EigenSolver(self):
        L = self.D - self.W
        R = self.D
        lam, y = eig(L, R)
        index = np.argsort(lam)

        top2 = lam[index[:2]].real
        smallest_2 = y[:, index[1]]
        print('dense eigenvector')
        return smallest_2.real

    def EigenSolver_sparse(self):
        s_D = sparse.csr_matrix(self.D)
        s_W = sparse.csr_matrix(self.W)
        s_D_nhalf = np.sqrt(s_D).power(-1)
        L = s_D_nhalf @ (s_D - s_W) @ s_D_nhalf
        lam, y = eigs(L)
        index = np.argsort(lam)

        top2 = lam[index[:2]].real
        smallest_2 = y[:, index[1]]
        print('sparse eigenvector')
        return smallest_2.real


class Ncut(object):
    '''
    This class is write for RGB image, so if you want to processing grayscale,
    some adjustment should worked on F_maker, W_maker function :)
    '''

    def __init__(self, match_path, init_num):

        self.path = match_path
        # self.N = self._N_maker() if init_num == -1 else init_num
        self.N = init_num
        self.W = self._W_maker()
        self.d = self._d_maker()
        self.D = self._D_maker()

    # V_nodes shape : [self.N,1,3]
    # def _N_maker(self):
    #     with open(self.path, "r") as f:
    #         lines = f.readlines()
    #         N = int(math.sqrt(len(lines)))
    #     return N

    def _W_maker(self):
        """
            Make weight matrix.
            W[i, j] means the weight(similarity score) between image i and j.
        """
        with open(self.path, "r") as f:
            lines = f.readlines()
            W = np.zeros((self.N, self.N))

            cur_i = -1
            count = 0
            top_p = 50

            for line in lines:
                i, j, score = line.split()
                i, j = int(i), int(j)
                score = float(score)
                if i != cur_i:
                    count = 0
                    cur_i = i
                    continue
                else:
                    if count <= top_p and i < self.N and j < self.N:
                        count += 1
                        W[i, j] = score
                    else:
                        continue
            # for i in range(self.N):
            #     W[i, i] = 0
        print(f"W max: {W.max()}")
        print(f"W min: {W.min()}")
        return W

    def _d_maker(self):
        # di is the sum of weight of node i with all other nodes
        d_i = np.sum(self.W, axis=1)
        d_i = np.array(d_i)
        return d_i

    def _D_maker(self):
        # D is a diagonal matrix using di as diagonal,
        D = np.diag(self.d)
        return D

    def _find_thresh(self, vect):
        min_ncut = 1e6
        ret = -1
        one_matrix = np.array([1] * self.N)
        for thresh in vect:
            bin_vect = vect.copy()
            bin_vect[bin_vect >= thresh] = 1
            bin_vect[bin_vect < thresh] = 0

            num, den = 0, 0
            for idx, val in enumerate(bin_vect):
                if val == 1:
                    num += self.d[idx]
                else:
                    den += self.d[idx]
            b = num / den
            bin_vect[bin_vect == 0] = -b

            ncut = (bin_vect.T@(self.D-self.W)@bin_vect) / \
                (bin_vect.T@self.D@bin_vect)
            print(ncut, thresh)
            if ncut < min_ncut and bin_vect.T@self.D@one_matrix == 0:
                min_ncut = ncut
                ret = thresh

        return ret

    def EigenSolver(self):
        print(f"Calculating eigenvector.")
        L = self.D - self.W
        R = self.D
        lam, y = eig(L, R)
        index = np.argsort(lam)

        top2 = lam[index[:2]].real
        smallest_2 = y[:, index[1]]

        print(f"Finding best threshold.")
        vect = smallest_2.real
        # print(f"Vect max: {vect.max()}")
        # print(f"Vect min: {vect.min()}")
        # thresh = self._find_thresh(vect)
        # print(f"Thresh is {thresh}")
        # vect[vect >= thresh] = 1
        # vect[vect < thresh] = -1
        # print(f"Eigenvector is {vect}")

        return vect
        # return smallest_2.real

    def EigenSolver_sparse(self):
        s_D = sparse.csr_matrix(self.D)
        s_W = sparse.csr_matrix(self.W)
        s_D_nhalf = np.sqrt(s_D).power(-1)
        L = s_D_nhalf @ (s_D - s_W) @ s_D_nhalf
        lam, y = eigs(L)
        index = np.argsort(lam)

        top2 = lam[index[:2]].real
        smallest_2 = y[:, index[1]]
        print('sparse eigenvector')
        return smallest_2.real


if __name__ == '__main__':
    # This is dense eigenvector method
    # img = cv2.imread('rgb', cv2.IMREAD_COLOR)
    # cutter = Ncut_origin(img)
    # eigenvector = cutter.EigenSolver()

    # the process is cost too much time, so I saved the results in a txt file,
    # just ignore this part if you need't

    # file = open('result.txt','w')
    # for i in eigenvector:
    #     file.write(str(i))
    #     file.write(',')
    file = open('result.txt', 'r')
    a = file.read()
    b = np.array(a.split(','))
    b = b[:-1]
    # This is sparse eigenvector method
    # img = cv2.imread('picture/Ncut_test.png', cv2.IMREAD_COLOR)
    # cutter = Ncut(img)
    # eigenvector = cutter.EigenSolver_sparse()
    # b = eigenvector
    print(b.shape)
    print(type(b))
    b = b.astype('float64')
    print(f"Result max: {b.max()}")
    print(f"Result min: {b.min()}")
    b = b.reshape((50, 50))
    b = (b/b.max())*255

    cv2.imshow('eigvec', b.astype('uint8'))
    cv2.waitKey()
