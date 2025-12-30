/**
 * API Communication Layer
 * Handles all requests to the FastAPI backend
 */

// API configuration
let API_BASE_URL = 'http://localhost:8000';

/**
 * Set the API base URL
 * @param {string} url - New base URL
 */
export function setApiBaseUrl(url) {
    API_BASE_URL = url.replace(/\/$/, ''); // Remove trailing slash
}

/**
 * Get current API base URL
 * @returns {string} Current base URL
 */
export function getApiBaseUrl() {
    return API_BASE_URL;
}

/**
 * Generic API request wrapper with error handling
 * @param {string} endpoint - API endpoint (e.g., '/api/predict')
 * @param {object} options - Fetch options
 * @returns {Promise<any>} Response data
 * @throws {Error} API error with details
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                ...options.headers,
            },
        });

        // Handle non-2xx responses
        if (!response.ok) {
            let errorDetail = `HTTP ${response.status}: ${response.statusText}`;

            try {
                const errorData = await response.json();
                if (errorData.detail) {
                    errorDetail = errorData.detail;
                }
            } catch (e) {
                // If response is not JSON, use status text
            }

            const error = new Error(errorDetail);
            error.status = response.status;
            throw error;
        }

        // Parse JSON response
        const data = await response.json();
        return data;

    } catch (error) {
        // Network or parsing errors
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            const connectionError = new Error(
                `Cannot connect to API at ${API_BASE_URL}. Is the server running?`
            );
            connectionError.status = 0;
            throw connectionError;
        }

        // Re-throw the error with additional context
        throw error;
    }
}

/**
 * Check API health status
 * @returns {Promise<object>} Health status data
 * @example
 * {
 *   "status": "healthy",
 *   "model_loaded": true,
 *   "device": "cuda:0",
 *   "num_actions": 40
 * }
 */
export async function getHealthStatus() {
    try {
        const data = await apiRequest('/health');
        return {
            success: true,
            healthy: data.model_loaded === true,
            data
        };
    } catch (error) {
        return {
            success: false,
            healthy: false,
            error: error.message
        };
    }
}

/**
 * Get list of available action classes
 * @returns {Promise<string[]>} Array of action names
 */
export async function getAvailableActions() {
    try {
        const data = await apiRequest('/api/actions');
        return {
            success: true,
            actions: data.actions || []
        };
    } catch (error) {
        console.error('Failed to fetch available actions:', error);
        return {
            success: false,
            actions: [],
            error: error.message
        };
    }
}

/**
 * Predict action from image
 * @param {File} imageFile - Image file to analyze
 * @param {number} topK - Number of top predictions to return (1-10)
 * @param {boolean} includeAttention - Include attention heatmap in response
 * @returns {Promise<object>} Prediction results
 * @example
 * {
 *   "action": "riding_a_bike",
 *   "confidence": 0.8734,
 *   "top_k": [
 *     {"action": "riding_a_bike", "confidence": 0.8734},
 *     {"action": "walking_dog", "confidence": 0.0654}
 *   ],
 *   "attention_heatmap": [[0.1, 0.2, ...], ...], // 7x7 grid if includeAttention=true
 *   "processing_time": 0.089,
 *   "device": "cuda:0"
 * }
 */
export async function predictAction(imageFile, topK = 5, includeAttention = false) {
    // Validate inputs
    if (!imageFile || !(imageFile instanceof File)) {
        throw new Error('Invalid image file');
    }

    if (topK < 1 || topK > 10) {
        throw new Error('top_k must be between 1 and 10');
    }

    // Create FormData for multipart/form-data request
    const formData = new FormData();
    formData.append('file', imageFile);

    // Build query parameters
    const params = new URLSearchParams({
        top_k: topK.toString(),
        include_attention: includeAttention.toString()
    });

    try {
        const data = await apiRequest(`/api/predict?${params}`, {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header - browser will set it with boundary for FormData
        });

        return {
            success: true,
            data
        };
    } catch (error) {
        console.error('Prediction failed:', error);
        return {
            success: false,
            error: error.message,
            status: error.status
        };
    }
}

/**
 * Test API connection
 * @returns {Promise<boolean>} True if API is reachable
 */
export async function testConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        return response.ok;
    } catch (error) {
        return false;
    }
}

/**
 * Fetch with timeout
 * @param {string} url - URL to fetch
 * @param {object} options - Fetch options
 * @param {number} timeout - Timeout in milliseconds (default: 30000)
 * @returns {Promise<Response>}
 */
export async function fetchWithTimeout(url, options = {}, timeout = 30000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error('Request timeout - server took too long to respond');
        }
        throw error;
    }
}

export default {
    setApiBaseUrl,
    getApiBaseUrl,
    getHealthStatus,
    getAvailableActions,
    predictAction,
    testConnection,
    fetchWithTimeout
};
