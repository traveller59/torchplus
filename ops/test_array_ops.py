import unittest
import numpy as np
import os
# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
import torch
from torchplus.tools import to_tensor
from torchplus.ops.array_ops import _scatter_nd, _gather_nd
from torchplus.framework.test import TestCase

def scatter_nd_gradient(indices, grad):
    # return grads for three input
    return None, _gather_nd(grad, indices), None


class ScatterNdTest(TestCase):
    """scatter_nd test from tensorflow.
    """

    def scatter_nd(self, indices, updates, shape, input_=None):
        del input_  # input_ is not used in scatter_nd
        indices = to_tensor(indices).long().cuda()
        updates = to_tensor(updates).cuda()
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()
        ret = _scatter_nd(indices, updates, shape)
        return ret.cpu().numpy()

    def scatter_nd_gradient(self, indices, grad):
        indices = to_tensor(indices).long()
        grad = to_tensor(grad)
        _, ret, _ = scatter_nd_gradient(indices, grad)
        return None, ret.numpy(), None
    """
    """
    def testRank3ValidShape(self):
        indices = np.zeros([2, 2, 2], np.int32)
        updates = np.zeros([2, 2, 2], np.int32)
        shape = np.array([2, 2, 2])
        self.assertAllEqual(
            self.scatter_nd(indices, updates, shape).shape, shape)

    def testExtraIndicesDimensions(self):
        indices = np.zeros([1, 1, 2], np.int32)
        updates = np.zeros([1, 1], np.int32)
        shape = np.array([2, 2])
        scatter = self.scatter_nd(indices, updates, shape)
        self.assertAllEqual(scatter.shape, shape)
        expected_result = np.zeros([2, 2], dtype=np.int32)
        self.assertAllEqual(expected_result, scatter)

    def testGradientsRank2ElementUpdate(self):
        indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
        updates = np.array([1, 4], dtype=np.float64)
        shape = np.array([2, 2], dtype=np.int32)
        input_ = np.zeros(shape, dtype=np.float64)
        outputs = self.scatter_nd(indices, updates, shape, input_)
        expected_outputs = np.array([[1, 0], [0, 4]], dtype=np.float64)
        self.assertAllEqual(expected_outputs, outputs)
        grad_vals = np.array([[1, 2], [3, 4]], dtype=np.float64)
        _, updates_grad, _ = self.scatter_nd_gradient(indices, grad_vals)
        expected_updates_grad = np.array([1, 4], dtype=np.float64)
        self.assertAllEqual(expected_updates_grad, updates_grad)
    
    def testGradientsRank2SliceUpdate(self):
        indices = np.array([[1], [0]], dtype=np.int32)
        updates = np.array([[3, 4], [1, 2]], dtype=np.float64)
        shape = np.array([2, 2], dtype=np.int32)
        input_ = np.zeros(shape, dtype=np.float64)
        outputs = self.scatter_nd(indices, updates, shape, input_)
        expected_outputs = np.array([[1, 2], [3, 4]], dtype=np.float64)
        self.assertAllEqual(expected_outputs, outputs)

        grad_vals = np.array([[3, 4], [1, 2]], dtype=np.float64)
        _, updates_grad, _ = self.scatter_nd_gradient(indices, grad_vals)
        expected_updates_grad = np.array([[1, 2], [3, 4]], dtype=np.float64)
        self.assertAllEqual(expected_updates_grad, updates_grad)
    
    def testGradientsRank3SliceUpdate(self):
        indices = np.array(
            [[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.int32)
        updates = np.array(
            [[[5, 7], [2, 4]], [[1, 3], [6, 8]]], dtype=np.float64)
        shape = np.array([2, 2, 2], dtype=np.int32)
        input_ = np.zeros(shape, dtype=np.float64)
        outputs = self.scatter_nd(indices, updates, shape, input_)
        expected_outputs = np.array(
            [[[1, 3], [5, 7]], [[2, 4], [6, 8]]], dtype=np.float64)
        self.assertAllEqual(expected_outputs, outputs)
        grad_vals = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
        _, updates_grad, _ = self.scatter_nd_gradient(indices, grad_vals)
        expected_updates_grad = np.array(
            [[[3, 4], [5, 6]], [[1, 2], [7, 8]]], dtype=np.float64)
        self.assertAllEqual(expected_updates_grad, updates_grad)

    def testGradientsRank7SliceUpdate(self):
        indices = np.array(
            [[[[[[[0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]]]],
               [[[[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]]]]]],
            dtype=np.int32)
        updates = np.array(
            [[[[[[[5, 6], [2, 4]]]], [[[[1, 3], [6, 8]]]]]]],
            dtype=np.float64)
        shape = np.array([1, 1, 2, 1, 1, 2, 2], dtype=np.int32)
        input_ = np.zeros(shape, dtype=np.float64)
        outputs = self.scatter_nd(indices, updates, shape, input_)
        grad_vals = np.array(
            [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]],
            dtype=np.float64)
        _, updates_grad, _ = self.scatter_nd_gradient(indices, grad_vals)
        expected_updates_grad = np.array(
            [[[[[[[3, 4], [5, 6]]]], [[[[1, 2], [7, 8]]]]]]], dtype=np.float64)
        expected_input_grad = np.array(
            [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]], dtype=np.float64)
        self.assertAllEqual(expected_updates_grad, updates_grad)
    
    def testScatterNdRepatedIndicesAdd(self):
        # failed. but it seems that we don't need this feature
        indices = np.zeros([10, 1], np.int32)
        values = np.random.randn(10)
        shape = [1]
        val = self.scatter_nd(indices, values, shape)
        self.assertAllClose([np.sum(values)], val)
    
    def testSmokeScatterNdBatch2DSliceDim2(self):
        indices = np.zeros([3, 5, 2], dtype=np.int32)
        values = np.zeros([3, 5, 7])
        shape = [4, 6, 7]
        self.scatter_nd(indices, values, shape)

    def testSmokeScatterNdBatch1DSliceDim3ShapeRank7(self):
        indices = np.zeros([1, 3], dtype=np.int32)
        values = np.zeros([1, 6, 7, 8, 9])
        shape = [3, 4, 5, 6, 7, 8, 9]
        self.scatter_nd(indices, values, shape)

    def testSmokeScatterNdBatch2DSliceDim3ShapeRank7(self):
        indices = np.zeros([1, 2, 3], dtype=np.int32)
        values = np.zeros([1, 2, 6, 7, 8, 9])
        shape = [3, 4, 5, 6, 7, 8, 9]
        self.scatter_nd(indices, values, shape)
    """
    """
if __name__ == '__main__':
    unittest.main()