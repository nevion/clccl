import unittest
from scipy.misc import imread
import os
from os.path import join as pjoin
import numpy as np
from kernel_common import *
from kernels import *
import cv2

filename = 'spiralc-2044.png'
if __name__ != '__main__':
    frame = imread(filename)
debug = False

def load_tests(loader, tests, pattern):
    def make_class(cls_img_dtype, cls_label_dtype, cls_connectivity_dtype, cls_img):
        class CCLTests(unittest.TestCase):
            img_dtype=cls_img_dtype
            label_dtype=cls_label_dtype
            connectivity_dtype = cls_connectivity_dtype
            img = cls_img
            ccl = None
            def __init__(self, method):
                unittest.TestCase.__init__(self, method)
                self.method = method

            @staticmethod
            def setUpClass():
                cls = CCLTests
                img_dtype = cls.img_dtype
                CCLTests.ccl = CCL(cls.img.shape, cls.img_dtype, cls.label_dtype, cls.connectivity_dtype, debug=debug)
                CCLTests.ccl.compile()

            def test_connectivity_image(self):
                ccl = self.ccl
                cl_img = ccl.make_input_buffer(queue)
                event = cl.enqueue_copy(queue, cl_img.data, self.img)
                event, connectivityim = ccl.make_connectivity_image(queue, cl_img, wait_for=[event])
                event.wait()

            def test_labeled_tiles(self):
                ccl = self.ccl
                cl_img = ccl.make_input_buffer(queue)
                event = cl.enqueue_copy(queue, cl_img.data, self.img)
                event, connectivityim = ccl.make_connectivity_image(queue, cl_img, wait_for=[event])
                event, labelim = ccl.label_tiles(queue, connectivityim, wait_for = [event])
                event.wait()
                labelim_h = labelim.get()
                connectivityim_h = connectivityim.get()
                ccl = self.ccl
                img_size = ccl.img_size
                im_cols = img_size[1]
                TILE_ROWS, TILE_COLS = ccl.TILE_ROWS, ccl.TILE_COLS
                blocks_rc = divUp(img_size[0], TILE_ROWS), divUp(img_size[1], TILE_COLS)
                for r_block in range(blocks_rc[0]):
                    tile_row_start = r_block * TILE_ROWS
                    tile_row_end = min(tile_row_start + TILE_ROWS, img_size[0])
                    tile_rows = tile_row_end - tile_row_start
                    for c_block in range(blocks_rc[1]):
                        tile_col_start = c_block * ccl.TILE_COLS
                        tile_col_end = min(tile_col_start + TILE_COLS, img_size[1])
                        tile_cols = tile_col_end - tile_col_start
                        im_tile = self.img[tile_row_start:tile_row_end, tile_col_start:tile_col_end].astype(np.uint8)
                        labelim_tile = labelim_h[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
                        conn_tile = connectivityim_h[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
                        labelim_tilep = labelim_tile.copy()
                        img_fg = im_tile.ravel() != 0

                        #get the labels relative to the tile again
                        l_rs = labelim_tile.ravel() / im_cols
                        l_cs = labelim_tile.ravel() % im_cols
                        l_rs -= tile_row_start
                        l_cs -= tile_col_start
                        linear_labels_local = l_rs * tile_cols + l_cs
                        labelim_is_root = ((linear_labels_local == np.arange(tile_rows * tile_cols, dtype=np.uint32)) & (img_fg)).astype(np.uint32)
                        linear_labels = inclusive_prefix_sum(labelim_is_root, dtype=np.uint32)
                        relabeled_labelim_tile = np.zeros_like(labelim_tile)
                        relabeled_labelim_tile.ravel()[img_fg] = linear_labels[linear_labels_local[img_fg]]

                        N_ref, cc_tile_ref = cv2.connectedComponents(im_tile)
                        np.testing.assert_array_equal(relabeled_labelim_tile, cc_tile_ref)

            def test_ccl_agreement(self):
                ccl = self.ccl
                cl_img = ccl.make_input_buffer(queue)
                img = self.img
                event = cl.enqueue_copy(queue, cl_img.data, img)
                event, N_d, labelim_d, labelim_i_d, prefix_sums_d, connectivityim_d = ccl(queue, cl_img, wait_for=[event], all_outputs = True)
                event, dcountim_i_d = ccl.count_invalid_labels(queue, labelim_i_d, connectivityim_d, wait_for=[event])
                event, dcountim_d = ccl.count_invalid_labels(queue, labelim_d, connectivityim_d, wait_for=[event])
                event.wait()
                N = N_d.get()
                labelim = labelim_d.get()
                labelim_i = labelim_i_d.get()
                prefix_sums = prefix_sums_d.get()
                N_ref, cc_ref = cv2.connectedComponents(img.astype(np.uint8))
                dcountim_i = dcountim_i_d.get()
                dcountim = dcountim_d.get()
                connectivityim = connectivityim_d.get()

                xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
                linear_index = xx + yy * img.shape[1]
                prefix_sum_ref = inclusive_prefix_sum(((labelim_i == linear_index) & (img != 0)).ravel())
                np.testing.assert_equal(prefix_sum_ref, prefix_sums.ravel())

                np.testing.assert_array_equal(labelim, cc_ref)
                self.assertEqual(N, N_ref)


        return CCLTests

    suite = unittest.TestSuite()
    #pixel, label, connectivity
    dtype_configs = [
        (np.uint32, np.uint32, np.uint32),
        (np.uint16, np.uint32, np.uint32),
    ]
    dtype_configs = [[np.dtype(x) for x in dtypes] for dtypes in dtype_configs]
    methods = 'labeled_tiles', 'ccl_agreement'

    for (pixel_dtype, label_dtype, connectivity_dtype) in dtype_configs:
        img = frame.astype(pixel_dtype)
        CCLTestCaseClass = make_class(pixel_dtype, label_dtype, connectivity_dtype, img)
        for method in methods:
            suite.addTest(CCLTestCaseClass('test_'+method))
    return suite

if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default=filename)
    parser.add_argument('--debug', action='store_true', default=debug)
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    filename = args.filename
    frame = imread(filename)
    debug = args.debug

    # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
    sys.argv[1:] = args.unittest_args
    unittest.main()


