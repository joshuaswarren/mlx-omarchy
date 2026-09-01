"""mlx.core.fft: Fast Fourier Transforms."""

from collections.abc import Sequence

import mlx.core


def fft(a: mlx.core.array, n: int | None = None, axis: int = -1, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    One dimensional discrete Fourier Transform.

    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n``. The default value is ``a.shape[axis]``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The DFT of the input along the given axis.
    """

def ifft(a: mlx.core.array, n: int | None = None, axis: int = -1, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    One dimensional inverse discrete Fourier Transform.

    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n``. The default value is ``a.shape[axis]``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The inverse DFT of the input along the given axis.
    """

def fft2(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = [-2, -1], norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    Two dimensional discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The DFT of the input along the given axes.
    """

def ifft2(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = [-2, -1], norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    Two dimensional inverse discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The inverse DFT of the input along the given axes.
    """

def fftn(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = None, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    n-dimensional discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes are or all axes if ``s`` is also ``None``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The DFT of the input along the given axes.
    """

def ifftn(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = None, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    n-dimensional inverse discrete Fourier Transform.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes or all axes if ``s`` is also ``None``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The inverse DFT of the input along the given axes.
    """

def rfft(a: mlx.core.array, n: int | None = None, axis: int = -1, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    One dimensional discrete Fourier Transform on a real input.

    The output has the same shape as the input except along ``axis`` in
    which case it has size ``n // 2 + 1``.

    Args:
        a (array): The input array. If the array is complex it will be silently
           cast to a real type.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n``. The default value is ``a.shape[axis]``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The DFT of the input along the given axis. The output
        data type will be complex.
    """

def irfft(a: mlx.core.array, n: int | None = None, axis: int = -1, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    The inverse of :func:`rfft`.

    The output has the same shape as the input except along ``axis`` in
    which case it has size ``n``.

    Args:
        a (array): The input array.
        n (int, optional): Size of the transformed axis. The
           corresponding axis in the input is truncated or padded with
           zeros to match ``n // 2 + 1``. The default value is
           ``a.shape[axis] // 2 + 1``.
        axis (int, optional): Axis along which to perform the FFT. The
           default is ``-1``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The real array containing the inverse of :func:`rfft`.
    """

def rfft2(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = [-2, -1], norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    Two dimensional real discrete Fourier Transform.

    The output has the same shape as the input except along the dimensions in
    ``axes`` in which case it has sizes from ``s``. The last axis in ``axes`` is
    treated as the real axis and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array. If the array is complex it will be silently
           cast to a real type.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The real DFT of the input along the given axes. The output
        data type will be complex.
    """

def irfft2(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = [-2, -1], norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    The inverse of :func:`rfft2`.

    Note the input is generally complex. The dimensions of the input
    specified in ``axes`` are padded or truncated to match the sizes
    from ``s``. The last axis in ``axes`` is treated as the real axis
    and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s`` except for the last axis
           which has size ``s[-1] // 2 + 1``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``[-2, -1]``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The real array containing the inverse of :func:`rfft2`.
    """

def rfftn(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = None, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    n-dimensional real discrete Fourier Transform.

    The output has the same shape as the input except along the dimensions in
    ``axes`` in which case it has sizes from ``s``. The last axis in ``axes`` is
    treated as the real axis and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array. If the array is complex it will be silently
           cast to a real type.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes or all axes if ``s`` is also ``None``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The real DFT of the input along the given axes. The output
        data type will be complex.
    """

def irfftn(a: mlx.core.array, s: tuple[int, ...] | None = None, axes: Sequence[int] | None = None, norm: str = 'backward', stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    The inverse of :func:`rfftn`.

    Note the input is generally complex. The dimensions of the input
    specified in ``axes`` are padded or truncated to match the sizes
    from ``s``. The last axis in ``axes`` is treated as the real axis
    and will have size ``s[-1] // 2 + 1``.

    Args:
        a (array): The input array.
        s (list(int), optional): Sizes of the transformed axes. The
           corresponding axes in the input are truncated or padded with
           zeros to match the sizes in ``s``. The default value is the
           sizes of ``a`` along ``axes``.
        axes (list(int), optional): Axes along which to perform the FFT.
           The default is ``None`` in which case the FFT is over the last
           ``len(s)`` axes or all axes if ``s`` is also ``None``.
        norm (str, optional): One of ``"backward"``, ``"ortho"``, or
           ``"forward"``. Default is ``"backward"``.

    Returns:
        array: The real array containing the inverse of :func:`rfftn`.
    """

def fftfreq(n: int, d: float = 1.0, stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    Return the discrete Fourier Transform sample frequencies.

    Args:
        n (int): Window length.
        d (float, optional): Sample spacing. The default is ``1.0``.

    Returns:
        array: The sample frequencies as a one-dimensional array of type ``float32``.
    """

def rfftfreq(n: int, d: float = 1.0, stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    Return the discrete Fourier Transform sample frequencies
    for use with :func:`rfft` and :func:`irfft`.

    The returned array contains the non-negative frequency terms
    in the range ``[0, floor(n/2)]``.

    Args:
        n (int): Window length.
        d (float, optional): Sample spacing. The default is ``1.0``.

    Returns:
        array: The sample frequencies as a one-dimensional array of type ``float32``.
    """

def fftshift(a: mlx.core.array, axes: int | Sequence[int] | None = None, stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    Shift the zero-frequency component to the center of the spectrum.

    Args:
        a (array): The input array.
        axes (int or list(int), optional): Axis or axes over which to perform the shift.
           If ``None``, shift all axes. 

    Returns:
        array: The shifted array with the same shape as the input.
    """

def ifftshift(a: mlx.core.array, axes: int | Sequence[int] | None = None, stream: mlx.core.Stream | mlx.core.ThreadLocalStream | mlx.core.Device | mlx.core.DeviceType | None = None) -> mlx.core.array:
    """
    The inverse of :func:`fftshift`. While identical to :func:`fftshift` for even-length axes,
    the behavior differs for odd-length axes.

    Args:
        a (array): The input array.
        axes (int or list(int), optional): Axis or axes over which to perform the inverse shift.
           If ``None``, shift all axes. 

    Returns:
        array: The inverse-shifted array with the same shape as the input.
    """
