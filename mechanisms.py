# Bounded Laplace Mechanism (https://arxiv.org/pdf/1808.10410.pdf)
from diffprivlib.mechanisms import LaplaceBoundedDomain
import numpy as np


def BoundedLaplace(x, epsilon, lower, upper):
    LBD = LaplaceBoundedDomain(
        epsilon=epsilon,
        delta=0,
        sensitivity=(upper-lower),
        lower=lower,
        upper=upper
    )
    return LBD.randomise(x)


def Body_LDP(replay):
    height = np.median(replay[:,1]).clip(1.496, 1.826)
    wingspan = np.max(np.abs(replay[:,14] - replay[:,7])).clip(1.556, 1.899)
    room_width = np.max(np.abs(replay[:,0])).clip(0.1, 1)
    room_height = np.max(np.abs(replay[:,2])).clip(0.1, 1)

    height_f = BoundedLaplace(height, 3, 1.496, 1.826)
    wingspan_f = BoundedLaplace(wingspan, 1, 1.556, 1.899)
    room_width_f = BoundedLaplace(room_width, 1, 0.1, 1)
    room_height_f = BoundedLaplace(room_height, 1, 0.1, 1)

    replay[:,1] = replay[:,1] * (height_f / height)
    replay[:,7] = replay[:,7] * (wingspan_f / wingspan)
    replay[:,14] = replay[:,14] * (wingspan_f / wingspan)
    replay[:,0] = replay[:,0] * (room_width_f / room_width)
    replay[:,2] = replay[:,2] * (room_height_f / room_height)

    return replay

def temporal_downsampling(gaze_values, step=2):
    length = gaze_values.shape[1]
    indicies = np.arange(0, length, step)
    indicies = np.repeat(indicies, step)

    return gaze_values[:, indicies][:, :length]


def spatial_downsampling(gaze_values, denom=2):
    ratio = 180 / (2160 / denom)
    val = np.floor(gaze_values / ratio) * ratio
    print(val.shape)

    return val

def gaussian(gaze_values, epsilon=2):
    noise = np.random.normal(loc=0., scale=epsilon, size=gaze_values.shape)
    gaze_values += noise

    return gaze_values

def smoothing(gaze_values, initial_mechanism='smoothing', custom_buffer_size=None, epsilon=2):
    global target_values

    # note this was 10 and 50 before
    first_buffer_size = 10
    second_buffer_size = 50

    if custom_buffer_size is not None:
        first_buffer_size = 1
        second_buffer_size = custom_buffer_size
        epsilon = custom_buffer_size

    # first add gaussian noise as usual
    original_values = gaze_values.copy()
    if target_values is None:
        target_values = original_values

    # now apply a weighted smoothing function using the noisy values
    smooth_xs = []
    smooth_ys = []

    first_buffer = []
    second_buffer = []

    # initialize the buffer with forward vectors
    for _ in range(first_buffer_size):
        first_buffer.append([0, 0])
    for _ in range(second_buffer_size):
        second_buffer.append([0, 0])

    def get_weighted_value(buffer):
        denom = 0

        weighted_value = [0, 0]

        for i in range(1, len(buffer) + 1):
            denom += i
            weighted_value[0] += buffer[i - 1][0] * i
            weighted_value[1] += buffer[i - 1][1] * i

        weighted_value[0] = weighted_value[0] / denom
        weighted_value[1] = weighted_value[1] / denom

        return weighted_value

    i = 0
    while i < gaze_values.shape[1]:
        first_buffer.remove(first_buffer[0])
        first_buffer.append([gaze_values[0, i], gaze_values[1, i]])
        buffered_value = get_weighted_value(first_buffer)

        second_buffer.remove(second_buffer[0])
        second_buffer.append(buffered_value)
        weighted_value = get_weighted_value(second_buffer)

        smooth_xs.append(weighted_value[0])
        smooth_ys.append(weighted_value[1])

        i += 1

    gaze_values = np.array([smooth_xs, smooth_ys])

    return gaze_values


def composite(x, epsilon, lower, upper, initial_mechanism='smoothing', custom_buffer_size=None):
    x = Body_LDP(x, epsilon, lower, upper)
    x = smoothing(x, initial_mechanism, custom_buffer_size, epsilon)
    return x