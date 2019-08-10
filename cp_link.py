import ctypes
import time
import numpy as np

ll = ctypes.cdll.LoadLibrary  # call c/c++ function
lib = ll("./libpycall.so")


def get_pointer(input):
    # This is amazing!
    return input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def zero_extend(input, u, d, l, r):
    output = np.zeros((input.shape[0], input.shape[1] + u + d,
                       input.shape[2] + l + r, input.shape[3]))
    output[:, u : u + input.shape[1], l : l + input.shape[2], :] = input[:, :, :, :]
    return output


def zero_extend_c(input, u, d, l, r):
    output = np.zeros((input.shape[0], input.shape[1] + u + d,
                       input.shape[2] + l + r, input.shape[3]), dtype=np.float32)
    input = input.astype(np.float32)
    input_pointer = get_pointer(input)
    output_pointer = get_pointer(output)
    # time1 = time.time()
    lib.zero_extend(input_pointer, output_pointer, u, d, l, r,
                    input.shape[0], input.shape[1], input.shape[2], input.shape[3])
    """
    time2 = time.time()
    output2 = np.zeros((input.shape[0], input.shape[1] + u + d,
                       input.shape[2] + l + r, input.shape[3]))
    output2[:, u: u + input.shape[1], l: l + input.shape[2], :] = input[:, :, :, :]
    time3 = time.time()
    print("compare")
    print(time2 - time1)
    print(time3 - time2)
    """
    return output


def input_extend(input, filter, strides, padding):
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
        ri_height = (out_height - 1) * strides[1] + filter_height
        ri_width = (out_width - 1) * strides[2] + filter_width
        ri = zero_extend_c(input, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)

    else:
        assert False

    return ri


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
        time1 = time.time()
        ri = zero_extend_c(input, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)
        time2 = time.time()
        ri = zero_extend(input, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                           (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)
        time3 = time.time()
        print("asdfg", time2 - time1, time3 - time2)

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
    lib.conv2d(
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

def conv2dGrad1(input, filter, output, strides, padding):
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    in_channels = input.shape[3]

    filter_height = filter.shape[0]
    filter_width = filter.shape[1]
    assert in_channels == filter.shape[2]
    out_channels = filter.shape[3]

    assert batch == output.shape[0]
    out_height = output.shape[1]
    out_width = output.shape[2]
    assert out_channels == output.shape[3]

    if padding == "SAME":
        ri_height = (out_height - 1) * strides[1] + filter_height
        ri_width = (out_width - 1) * strides[2] + filter_width
        ri_channel = in_channels
        ri = np.zeros(input.shape, dtype=np.float32)
        time1 = time.time()
        ri = zero_extend(ri, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)
        time2 = time.time()
        ri = zero_extend(ri, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)
        time3 = time.time()
        print("erewrq", time2 - time1, time3 - time2)

    else:
        assert False

    ri = ri.astype(np.float32)
    filter = filter.astype(np.float32)
    output = output.astype(np.float32)

    lib.cov2d_grad1(
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

    return ri[:, (ri_height - in_height) // 2 : (ri_height - in_height) // 2 + input.shape[1],
           (ri_width - in_width) // 2 : (ri_width - in_width) // 2 + input.shape[2], :]


def conv2dGrad2(input, filter, output, strides, padding):
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    in_channels = input.shape[3]

    filter_height = filter.shape[0]
    filter_width = filter.shape[1]
    assert in_channels == filter.shape[2]
    out_channels = filter.shape[3]

    assert batch == output.shape[0]
    out_height = output.shape[1]
    out_width = output.shape[2]
    assert out_channels == output.shape[3]

    if padding == "SAME":
        ri_height = (out_height - 1) * strides[1] + filter_height
        ri_width = (out_width - 1) * strides[2] + filter_width
        ri_channel = in_channels
        ri = zero_extend(input, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)

    else:
        assert False

    rf = np.zeros(filter.shape, dtype=np.float32)
    ri = ri.astype(np.float32)
    output = output.astype(np.float32)

    lib.conv2d_grad2(
        batch,
        get_pointer(ri),
        ri_height,
        ri_width,
        ri_channel,
        strides[1],
        strides[2],
        get_pointer(rf),
        filter_height,
        filter_width,
        out_channels,
        get_pointer(output),
        out_height,
        out_width
    )

    return rf


def max_pool(value, ksize, strides, padding):
    batch = value.shape[0]
    in_height = value.shape[1]
    in_width = value.shape[2]
    in_channels = value.shape[3]

    filter_height = ksize[1]
    filter_width = ksize[2]
    out_channels = in_channels

    if padding == "SAME":
        out_height = (in_height - 1) // strides[1] + 1
        out_width = (in_width - 1) // strides[2] + 1
        output = np.zeros((batch, out_height, out_width, out_channels), dtype=np.float32)
        ri_height = (out_height - 1) * strides[1] + filter_height
        ri_width = (out_width - 1) * strides[2] + filter_width
        ri_channel = in_channels
        ri = zero_extend(value, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)

    else:
        assert False

    ri = ri.astype(np.float32)

    lib.max_pool(
        batch,
        get_pointer(ri),
        ri_height,
        ri_width,
        ri_channel,
        strides[1],
        strides[2],
        filter_height,
        filter_width,
        out_channels,
        get_pointer(output),
        out_height,
        out_width
    )
    return output


def max_pool_grad(value, output, ksize, strides, padding):
    batch = value.shape[0]
    in_height = value.shape[1]
    in_width = value.shape[2]
    in_channels = value.shape[3]

    filter_height = ksize[1]
    filter_width = ksize[2]

    assert output.shape[0] == batch
    out_height = output.shape[1]
    out_width = output.shape[2]
    out_channels = output.shape[3]
    assert output.shape[3] == in_channels

    if padding == "SAME":
        ri_height = (out_height - 1) * strides[1] + filter_height
        ri_width = (out_width - 1) * strides[2] + filter_width
        ri_channel = in_channels
        ri = zero_extend(value, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)
        gradi = np.zeros(value.shape)
        gradi = zero_extend(gradi, (ri_height - in_height) // 2, (ri_height - in_height + 1) // 2,
                         (ri_width - in_width) // 2, (ri_width - in_width + 1) // 2)

    else:
        assert False

    output = output.astype(np.float32)
    ri = ri.astype(np.float32)
    gradi = gradi.astype(np.float32)

    lib.max_pool_grad(
        batch,
        get_pointer(ri),
        get_pointer(gradi),
        ri_height,
        ri_width,
        ri_channel,
        strides[1],
        strides[2],
        filter_height,
        filter_width,
        out_channels,
        get_pointer(output),
        out_height,
        out_width
    )
    return gradi





def conv2dGrad22(input, filter, strides, padding):
    # print("is:", input.shape)
    # print(input)
    # print("fs:", filter.shape)
    # print(filter)
    batch = input.shape[0]
    in_height = input.shape[1]
    in_width = input.shape[2]
    in_channels = input.shape[3]

    filter_height = filter.shape[0]
    filter_width = filter.shape[1]
    assert in_channels == filter.shape[2]
    out_channels = filter.shape[3]

    assert padding == "SAME"
    assert strides == [1, 1, 1, 1]

    out_height = (in_height - filter_height) + 1
    out_width = (in_width - filter_width) + 1
    output = np.zeros((batch, out_height, out_width, out_channels), dtype=np.float32)

    input = input.astype(np.float32)
    input = np.ascontiguousarray(input)
    filter = filter.astype(np.float32)
    filter = np.ascontiguousarray(filter)
    lib.conv2d(
        batch,
        get_pointer(input),
        in_height,
        in_width,
        in_channels,
        1,
        1,
        get_pointer(filter),
        filter_height,
        filter_width,
        out_channels,
        get_pointer(output),
        out_height,
        out_width
    )
    return output