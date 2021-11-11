import numpy as np


class SimilarityMeasure:
    """
    Base class for similarity measures
    """

    def measure(self, object1, object2):
        """
        Returns the measure value between two objects
        """
        return 0

    def matrix(self, container1, container2):
        """
        Returns the matrix of measure values between two sets of objects
        Sometimes can be implemented in a faster way than making couplewise measurements
        """
        matrix = np.zeros((len(container1), len(container2)))
        for i, object1 in enumerate(container1):
            for j, object2 in enumerate(container2):
                matrix[i, j] = self.measure(object1, object2)
        return matrix


class Minkovsky2DSimilarity(SimilarityMeasure):
    def __init__(self, p=2, scale=1.0):
        self.p = p
        self.scale = scale
        

    def measure(self, point1, point2):
        x_diff = point1.x_coords()[0] - point2.x_coords()[0]
        y_diff = point1.y_coords()[0] - point2.y_coords()[0]
        diffs = np.hstack([np.abs(x_diff), np.abs(y_diff)])
        powers = np.power(diffs, self.p)
        distance = np.power(powers.sum(), 1 / self.p) / self.scale

        return distance

    def matrix(self, points1, points2):

        coords1 = np.vstack([points1.x_coords(), points1.y_coords()])
        coords2 = np.vstack([points2.x_coords(), points2.y_coords()])

        diffs = np.abs(coords1[:, :, np.newaxis] - coords2[:, np.newaxis, :])

        powers = np.power(diffs, self.p)
        matrix = np.power(powers.sum(axis=0), 1 / self.p) / self.scale

        return matrix


class KPSimilarity(Minkovsky2DSimilarity):
    """
    Keypoints similarity
    """

    def __init__(self, p=2, scale=1.0, class_agnostic=True):
        super().__init__(p, scale)
        self.class_agnostic = class_agnostic

    def _exp_square(self, arr):
        return np.exp(-np.power(arr, 2) / 2.0)

    def measure(self, point1, point2):
        distance = super().measure(point1, point2)
        return self._exp_square(distance)

    def matrix(self, points1, points2):
        distance = super().matrix(points1, points2)
        matrix = self._exp_square(distance)
        if not self.class_agnostic:
            class_matrix = points1.classes().reshape(-1, 1) == points2.classes().reshape(1,-1)
            matrix = matrix * class_matrix
        return matrix
