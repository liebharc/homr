import numpy as np

from homr.type_definitions import NDArray


def find_peaks(
    x: NDArray,
    height: float | None = None,
    distance: float | None = None,
    prominence: float | None = None,
) -> tuple[NDArray, dict]:
    """
    Find peaks in a 1D array without using scipy.

    Parameters:
    -----------
    x : array-like
        1D signal data
    height : float, optional
        Minimum height of peaks
    distance : float, optional
        Minimum distance between peaks (in samples)
    prominence : float, optional
        Minimum prominence of peaks

    Returns:
    --------
    peaks : ndarray
        Indices of peaks
    properties : dict
        Dictionary with empty values (for compatibility)
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if len(x) < 3:
        return np.array([], dtype=int), {}

    # Find local maxima, handling flat regions
    peaks_list: list[int] = []
    i = 1
    while i < len(x) - 1:
        # Check if we're at a local maximum or start of plateau
        if x[i] > x[i - 1]:
            # Find end of plateau
            j = i
            while j < len(x) - 1 and x[j] == x[j + 1]:
                j += 1

            # Check if plateau ends with a descent (making it a peak)
            if j < len(x) - 1 and x[j] > x[j + 1]:
                # Peak is at the midpoint of the plateau
                peak_idx = (i + j) // 2
                peaks_list.append(peak_idx)
                i = j + 1
            else:
                i = j + 1
        elif x[i] == x[i - 1]:
            # Start of a flat region from the left
            j = i
            while j < len(x) - 1 and x[j] == x[j + 1]:
                j += 1

            # Check if it's a peak (higher than left, and either at end or higher than right)
            if j < len(x) - 1 and x[j] > x[j + 1]:
                peak_idx = (i + j) // 2
                peaks_list.append(peak_idx)
            i = j + 1
        else:
            i += 1

    peaks = np.array(peaks_list, dtype=int)

    if len(peaks) == 0:
        return np.array([], dtype=int), {}

    # Apply height threshold
    if height is not None:
        mask = x[peaks] >= height
        peaks = peaks[mask]

    if len(peaks) == 0:
        return np.array([], dtype=int), {}

    # Apply prominence filter
    if prominence is not None:
        valid_peaks: list[int] = []
        for peak_val in peaks:
            peak = int(peak_val)
            # Calculate prominence properly
            # Find the lowest contour line around the peak
            # Look left for the minimum until we hit a higher peak or boundary
            left_min = x[peak]
            for k in range(peak - 1, -1, -1):
                if x[k] > x[peak]:
                    break
                left_min = min(left_min, x[k])

            # Look right for the minimum until we hit a higher peak or boundary
            right_min = x[peak]
            for k in range(peak + 1, len(x)):
                if x[k] > x[peak]:
                    break
                right_min = min(right_min, x[k])

            # Prominence is height above the higher of the two minima
            peak_prominence = x[peak] - max(left_min, right_min)

            if peak_prominence >= prominence:
                valid_peaks.append(peak)
        peaks = np.array(valid_peaks, dtype=int)

    if len(peaks) == 0:
        return np.array([], dtype=int), {}

    # Apply distance constraint (keep highest peaks)
    if distance is not None and len(peaks) > 1:
        # Sort peaks by height (descending)
        sorted_indices = np.argsort(x[peaks])[::-1]
        sorted_peaks = peaks[sorted_indices]

        keep: list[int] = []
        for peak_val in sorted_peaks:
            peak = int(peak_val)
            # Check if this peak is far enough from all kept peaks
            if len(keep) == 0 or all(abs(k - peak) >= distance for k in keep):
                keep.append(peak)

        # Sort back to original order
        peaks = np.array(sorted(keep), dtype=int)

    return peaks, {}
