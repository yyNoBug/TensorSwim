import ctypes
import time
import numpy as np

ll = ctypes.cdll.LoadLibrary  # call c/c++ function
lib = ll("./libpycall.so")


def zero_extend(input, u, d, l, r):
    output = np.zeros((input.shape[0], input.shape[1] + u + d,
                       input.shape[2] + l + r, input.shape[3]))
    output[:, u : u + input.shape[1], l : l + input.shape[2], :] = input[:, :, :, :]
    return output


def get_pointer(input):
    # This is amazing!
    return input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def conv2d(input, filter, strides, padding):
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    in_channels = input.shape[3]

    filter_height = filter.shape[0]
    filter_width = filter.shape[1]
    assert in_channels == filter.shape[2]
    out_channels = filter.shape[3]

    if padding == "SAME":
        out_height = (in_height - 1) // strides[1] + 1
        out_width = (in_width - 1) // strides[2] + 1
        output = np.zeros((batch, out_height, out_width, out_channels), dtype=np.float32)
        ri_height = (out_height - 1) * strides[1] + filter_height
        ri_width = (out_width - 1) * strides[2] + filter_width
        ri_channel = in_channels
        ri = zero_extend(input, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)

    elif padding == "VALID":
        out_height = (in_height - filter_height) // strides[1] + 1
        out_width = (in_width - filter_width) // strides[2] + 1
        output = np.zeros((batch, out_height, out_width, out_channels), dtype=np.float32)
        ri_height = in_height
        ri_width = in_width
        ri_channel = in_channels
        ri = input

    else:
        assert False

    ri = ri.astype(np.float32)
    filter = filter.astype(np.float32)
    lib.cov2d(
        batch,
        get_pointer(ri),
        ri_height,
        ri_width,
        ri_channel,
        strides[1],
        strides[2],
        get_pointer(filter),
        filter_height,
        filter_width,
        out_channels,
        get_pointer(output),
        out_height,
        out_width
    )
    return output


def conv2dGrad1(input, filter, strides, padding):
    pass


def conv2dGrad2(input, filter, strides, padding):
    pass






def fact(n):  # function write in python to compare
    if n <= 1: return 1
    else: return n * fact(n - 1)


now = time.time()
for i in range(10):
    n = fact(100)
end = time.time()
print('the python fact takes:', end - now)


for i in range(10):
    n = lib.fact(100)
print('the c fact takes:', time.time() - end)
