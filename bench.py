#!/usr/bin/env python

from kernels import *
from scipy.misc import imread
import time
import sys

debug = False
max_cus = 4 * compute_units
filename = 'image.png'
iters = 500

if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', default=filename)
    parser.add_argument('--debug', action='store_true', default=debug)
    parser.add_argument('--max-cus', type=int, default=max_cus)
    parser.add_argument('-i','--iterations', type=int, default=iters)
    args = parser.parse_args()
    filename = args.filename
    debug = args.debug
    max_cus = args.max_cus
    iters = args.iterations

frame = imread(filename)

pixel_dtype = np.dtype(np.uint32)
label_dtype = np.dtype(np.uint32)
connectivity_dtype = np.dtype(np.uint32)
wg_size = default_wg_size
img = frame.astype(pixel_dtype)

h,w = img.shape
ccl = CCL(img.shape, pixel_dtype, label_dtype, connectivity_dtype, debug=debug, best_wg_size = wg_size, max_cus = max_cus, use_fused_mark=False)
ccl.compile()

cl_src_img = ccl.make_input_buffer(queue)
cl_dst_img = None
dst_img = ccl.make_host_output_buffer()

def upload(wait_for = None):
    event = cl.enqueue_copy(queue, cl_src_img.data, img, wait_for=wait_for)
    return event
def download(wait_for = None):
    event = cl.enqueue_copy(queue, dst_img, cl_dst_img.data, wait_for = wait_for)
    return event

def core_loop(wait_for = None):
    global cl_dst_img
    event, count, cl_dst_img = ccl(queue, cl_src_img, wait_for=wait_for)
    return event

def full_loop():
    event = upload()
    event = core_loop(wait_for = [event])
    download(wait_for = [event])

print 'compiled'
upload().wait()
print 'uploaded'

times = np.zeros((iters, 2), np.double)
loop_start = time.time()
for x in range(iters):
    start = time.time()
    core_loop().wait()
    end = time.time()
    times[x, :] = start, end
loop_end = time.time()
loop_total = loop_end - loop_start
loop_avg = (loop_total / iters)*1e3
#import timeit
#iters = 100*15
#print timeit.timeit(lambda: core_loop().wait(), number=iters)*1e3/iters
timings = np.squeeze(np.diff(times, axis=1))*1e3
print timings
print 'total: %r loop avg: %r best: %r iterations: %d std: %r'%(loop_total, loop_avg, timings.min(), iters, np.std(timings))

print 'download'
download().wait()
