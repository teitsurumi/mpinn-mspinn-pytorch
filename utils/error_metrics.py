import numpy as np

def _validate_shape(actual: np.ndarray, predicted: np.ndarray) -> None:
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} != predicted {predicted.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Input arrays must contain only finite values.")


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    _validate_shape(actual, predicted)
    return np.float64(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))

def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > eps
    if not np.any(mask): return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0

def smape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    denom = np.abs(actual) + np.abs(predicted)
    valid = denom > eps
    if not np.any(valid): return np.nan
    return np.mean(2.0 * np.abs(predicted[valid] - actual[valid]) / denom[valid]) * 100.0

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    _validate_shape(actual, predicted)
    return np.float64(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual))**2)))

def l_inf(actual: np.ndarray, predicted: np.ndarray) -> float:
    _validate_shape(actual, predicted)
    return np.float64(np.max(np.abs(np.asarray(predicted) - np.asarray(actual))))

def relative_l1(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return mae(actual, predicted) / (np.mean(np.abs(actual)) + eps)

def relative_l2(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return rmse(actual, predicted) / (np.sqrt(np.mean(actual**2)) + eps)

def relative_l_inf(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return np.max(np.abs(predicted - actual)) / (np.max(np.abs(actual)) + eps)

def gradient_l2_error(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    grad_a, grad_p = np.gradient(actual), np.gradient(predicted)
    diff_norm = np.sqrt(sum(np.mean((g_p - g_a)**2) for g_a, g_p in zip(grad_a, grad_p)))
    actual_norm = np.sqrt(sum(np.mean(g_a**2) for g_a in grad_a)) + eps
    return diff_norm / actual_norm

def spectral_relative_error(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    fft_a, fft_p = np.fft.fftn(actual), np.fft.fftn(predicted)
    diff_norm = np.sqrt(np.mean(np.abs(fft_p - fft_a)**2))
    actual_norm = np.sqrt(np.mean(np.abs(fft_a)**2)) + eps
    return diff_norm / actual_norm

def relative_energy_error(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-16) -> float:
    _validate_shape(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    energy_a = np.sum(actual**2)
    energy_p = np.sum(predicted**2)
    return np.abs(energy_p - energy_a) / (energy_a + eps)

def pearson_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    _validate_shape(actual, predicted)
    a, p = np.asarray(actual).ravel(), np.asarray(predicted).ravel()
    if len(a) < 2: return np.nan
    a_c, p_c = a - np.mean(a), p - np.mean(p)
    std_a, std_p = np.sqrt(np.mean(a_c**2)), np.sqrt(np.mean(p_c**2))
    if std_a == 0 or std_p == 0: return np.nan if std_a != std_p else 1.0
    return np.mean(a_c * p_c) / (std_a * std_p)

def robust_median_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    _validate_shape(actual, predicted)
    return np.float64(np.median(np.abs(np.asarray(predicted) - np.asarray(actual))))