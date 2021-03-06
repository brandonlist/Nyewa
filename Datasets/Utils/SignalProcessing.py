import scipy
import scipy.signal
import pandas as pd
from scipy import fftpack
from matplotlib import pyplot as plt
import numpy as np

"""Mainly operate on numpy array data"""

def HighpassCnt(data, low_cut_hz, fs, filt_order=8, axis=0):
    """
     Highpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    fs: float
    filt_order: int

    Returns
    -------
    highpassed_data: 2d-array
        Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        print('Not doing any highpass, since low 0 or None')
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, low_cut_hz / (fs / 2.0), btype="highpass"
    )
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed

def LowpassCnt(data, high_cut_hz, fs, filt_order=8, axis=0):
    """
     Lowpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    high_cut_hz: float
    fs: float
    filt_order: int

    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    if (high_cut_hz is None) or (high_cut_hz == fs / 2.0):
        'Not doing any lowpass, since high cut hz is None or nyquist freq.'
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, high_cut_hz / (fs / 2.0), btype="lowpass"
    )
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed

def BandpassCnt(
        data, low_cut_hz, high_cut_hz, fs, filt_order=8, axis=0, filtfilt=False
):
    """
     Bandpass signal applying **causal** butterworth filter of given order.
    Can be used as low-pass / high-pass filters by setting low_cut_hz=0 / high_cut_hz=None or Nyquist
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter

    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    if (low_cut_hz == 0 or low_cut_hz is None) and (
            high_cut_hz == None or high_cut_hz == fs / 2.0
    ):
        print('low_cut cant be 0 or None, and high_cut_hz cant be None or higher than Nyquist')
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        print('Using lowpass filter since low cut hz is 0 or None')
        return LowpassCnt(
            data, high_cut_hz, fs, filt_order=filt_order, axis=axis
        )
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        print('Using highpass filter since high cut hz is None or nyquist freq')
        return HighpassCnt(
            data, low_cut_hz, fs, filt_order=filt_order, axis=axis
        )

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass")
    assert filter_is_stable(a), 'Filter should be stable...'
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed

def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.

    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.

    Returns
    -------
    is_stable: bool
        Filter is stable or not.
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.

    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a))
    )
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)

def ExponentialRunningStandardize(
        data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    """
    Perform exponential running standardization.

    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential running variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.


    Parameters
    ----------
    data: 2darray (channels, time)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    data = data.T
    df = pd.DataFrame(data)
    #ewm: exponential weighted; equation at https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#stats-moments-exponentially-weighted
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized).T
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
                                          data[0:init_block_size] - init_mean
                                  ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized

def Envelop(data,display=0):
    """

    :param data:1-d array
    :param display:
    :return: data's up envelope
    """
    data_a = data - data.mean()
    hx_a = fftpack.hilbert(data_a)
    data_up = np.sqrt(data_a**2 + hx_a**2)+ data.mean()
    # data_a_dw = -data_a
    # hx_a_dw = fftpack.hilbert(data_a_dw)
    # data_dw = -np.sqrt(data_a_dw**2 + hx_a_dw**2)+ data.mean()
    # data_mean = (data_dw+data_up)/2
    if display:
        plt.plot(data,"b",linewidth=2, label='signal')
        plt.plot(data_up,"r",linewidth=2, label='envelop')
        plt.legend()
        plt.show()
    return data_up
