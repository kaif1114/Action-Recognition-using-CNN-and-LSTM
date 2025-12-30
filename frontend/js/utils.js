/**
 * Utility Functions for Action Recognition System
 * Contains helper functions used throughout the application
 */

/**
 * Format action name from snake_case to Title Case
 * @param {string} snakeCase - Action name in snake_case (e.g., "riding_a_bike")
 * @returns {string} Formatted action name (e.g., "Riding A Bike")
 */
export function formatActionName(snakeCase) {
    if (!snakeCase) return '';
    return snakeCase
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

/**
 * Format confidence value as percentage
 * @param {number} value - Confidence value (0-1)
 * @param {number} decimals - Number of decimal places (default: 1)
 * @returns {string} Formatted percentage (e.g., "87.3%")
 */
export function formatConfidence(value, decimals = 1) {
    if (typeof value !== 'number' || isNaN(value)) return '0.0%';
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Get confidence level category
 * @param {number} confidence - Confidence value (0-1)
 * @returns {string} 'high', 'medium', or 'low'
 */
export function getConfidenceLevel(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
}

/**
 * Validate image file
 * @param {File} file - File object to validate
 * @returns {object} {valid: boolean, error: string}
 */
export function validateImageFile(file) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const validTypes = [
        'image/jpeg',
        'image/png',
        'image/bmp',
        'image/gif',
        'image/tiff',
        'image/webp'
    ];

    if (!file) {
        return { valid: false, error: 'No file selected' };
    }

    if (!validTypes.includes(file.type)) {
        return {
            valid: false,
            error: `Invalid file type: ${file.type}. Supported: JPEG, PNG, BMP, GIF, TIFF, WebP`
        };
    }

    if (file.size > maxSize) {
        return {
            valid: false,
            error: `File too large: ${formatBytes(file.size)}. Maximum: 10MB`
        };
    }

    return { valid: true, error: null };
}

/**
 * Format bytes to human-readable size
 * @param {number} bytes - File size in bytes
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted size (e.g., "2.4 MB")
 */
export function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format timestamp to readable string
 * @param {Date} date - Date object (default: now)
 * @returns {string} Formatted timestamp (e.g., "Dec 30, 2025 2:30 PM")
 */
export function formatTimestamp(date = new Date()) {
    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    };
    return date.toLocaleString('en-US', options);
}

/**
 * Format processing time
 * @param {number} seconds - Processing time in seconds
 * @returns {string} Formatted time (e.g., "89ms" or "1.2s")
 */
export function formatProcessingTime(seconds) {
    if (seconds < 1) {
        return Math.round(seconds * 1000) + 'ms';
    }
    return seconds.toFixed(2) + 's';
}

/**
 * Get image dimensions from file
 * @param {File} file - Image file
 * @returns {Promise<{width: number, height: number}>}
 */
export function getImageDimensions(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);

        img.onload = () => {
            URL.revokeObjectURL(url);
            resolve({
                width: img.naturalWidth,
                height: img.naturalHeight
            });
        };

        img.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error('Failed to load image'));
        };

        img.src = url;
    });
}

/**
 * Debounce function - delays execution until after wait time has elapsed
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Save data to localStorage with error handling
 * @param {string} key - Storage key
 * @param {any} value - Value to store (will be JSON stringified)
 * @returns {boolean} Success status
 */
export function saveToLocalStorage(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
        return true;
    } catch (error) {
        console.error('Failed to save to localStorage:', error);
        return false;
    }
}

/**
 * Load data from localStorage with error handling
 * @param {string} key - Storage key
 * @param {any} defaultValue - Default value if key doesn't exist
 * @returns {any} Stored value or default
 */
export function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.error('Failed to load from localStorage:', error);
        return defaultValue;
    }
}

/**
 * Create a download link for data
 * @param {Blob|string} data - Data to download
 * @param {string} filename - Download filename
 * @param {string} mimeType - MIME type (default: 'text/plain')
 */
export function downloadFile(data, filename, mimeType = 'text/plain') {
    const blob = data instanceof Blob ? data : new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Wait for specified milliseconds
 * @param {number} ms - Milliseconds to wait
 * @returns {Promise}
 */
export function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Clamp a number between min and max
 * @param {number} value - Value to clamp
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} Clamped value
 */
export function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

/**
 * Generate a unique ID
 * @returns {string} Unique ID
 */
export function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

export default {
    formatActionName,
    formatConfidence,
    getConfidenceLevel,
    validateImageFile,
    formatBytes,
    formatTimestamp,
    formatProcessingTime,
    getImageDimensions,
    debounce,
    saveToLocalStorage,
    loadFromLocalStorage,
    downloadFile,
    sleep,
    clamp,
    generateId
};
