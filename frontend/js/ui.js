/**
 * UI Manipulation Module
 * Handles all UI updates and interactions
 */

import {
    formatActionName,
    formatConfidence,
    getConfidenceLevel,
    formatBytes,
    formatTimestamp,
    formatProcessingTime,
    generateId
} from './utils.js';

// Toast notification queue
const toastQueue = [];

/**
 * Show image preview
 * @param {File} file - Image file
 * @param {object} dimensions - Image dimensions {width, height}
 */
export function showImagePreview(file, dimensions) {
    const previewSection = document.getElementById('previewSection');
    const previewImage = document.getElementById('previewImage');
    const previewFilename = document.getElementById('previewFilename');
    const previewMeta = document.getElementById('previewMeta');

    // Create object URL for preview
    const imageUrl = URL.createObjectURL(file);
    previewImage.src = imageUrl;

    // Set filename
    previewFilename.textContent = file.name;

    // Set metadata
    const meta = `${dimensions.width} × ${dimensions.height} • ${formatBytes(file.size)}`;
    previewMeta.textContent = meta;

    // Show preview section
    previewSection.classList.remove('hidden');
    previewSection.classList.add('fade-in');

    // Hide results and heatmap sections
    hideResults();
    hideHeatmap();
}

/**
 * Clear image preview
 */
export function clearPreview() {
    const previewSection = document.getElementById('previewSection');
    const previewImage = document.getElementById('previewImage');

    // Revoke object URL to free memory
    if (previewImage.src) {
        URL.revokeObjectURL(previewImage.src);
    }

    previewSection.classList.add('hidden');
    hideResults();
    hideHeatmap();
}

/**
 * Show loading state
 * @param {string} message - Loading message (default: "Analyzing image...")
 * @param {string} subtext - Subtext (optional)
 */
export function showLoading(message = 'Analyzing image...', subtext = '') {
    const loadingSection = document.getElementById('loadingSection');
    const loadingText = loadingSection.querySelector('.loading-text');
    const loadingSubtext = document.getElementById('loadingSubtext');
    const predictBtn = document.getElementById('predictBtn');

    loadingText.textContent = message;
    loadingSubtext.textContent = subtext;

    loadingSection.classList.remove('hidden');
    loadingSection.classList.add('fade-in');

    // Disable predict button
    if (predictBtn) {
        predictBtn.disabled = true;
    }

    // Hide results
    hideResults();
    hideHeatmap();
}

/**
 * Hide loading state
 */
export function hideLoading() {
    const loadingSection = document.getElementById('loadingSection');
    const predictBtn = document.getElementById('predictBtn');

    loadingSection.classList.add('hidden');

    // Re-enable predict button
    if (predictBtn) {
        predictBtn.disabled = false;
    }
}

/**
 * Display prediction results
 * @param {object} data - Prediction data from API
 * @param {string} imageUrl - Image URL for display
 */
export function displayPredictionResults(data, imageUrl) {
    const resultsSection = document.getElementById('resultsSection');
    const resultImage = document.getElementById('resultImage');
    const predictedAction = document.getElementById('predictedAction');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const predictionsList = document.getElementById('predictionsList');
    const resultMetadata = document.getElementById('resultMetadata');

    // Set result image
    resultImage.src = imageUrl;

    // Set top prediction
    const actionName = formatActionName(data.action);
    predictedAction.textContent = actionName;

    // Set confidence
    const confidence = data.confidence;
    const confidenceLevel = getConfidenceLevel(confidence);
    const confidencePercent = formatConfidence(confidence);

    confidenceValue.textContent = confidencePercent;
    confidenceValue.className = `confidence-value ${confidenceLevel}`;

    // Animate confidence bar
    confidenceFill.style.width = '0%';
    confidenceFill.className = `confidence-fill ${confidenceLevel}`;
    setTimeout(() => {
        confidenceFill.style.width = `${confidence * 100}%`;
    }, 100);

    // Render top-k predictions
    renderTopKPredictions(data.top_k, predictionsList);

    // Render metadata
    renderMetadata(data, resultMetadata);

    // Show results section
    hideLoading();
    resultsSection.classList.remove('hidden');
    resultsSection.classList.add('fade-in');
}

/**
 * Render top-k predictions list
 * @param {Array} predictions - Array of {action, confidence} objects
 * @param {HTMLElement} container - Container element
 */
export function renderTopKPredictions(predictions, container) {
    container.innerHTML = '';

    predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animationDelay = `${index * 50}ms`;

        const rank = document.createElement('div');
        rank.className = 'prediction-rank';
        rank.textContent = `${index + 1}.`;

        const name = document.createElement('div');
        name.className = 'prediction-name';
        name.textContent = formatActionName(pred.action);

        const confidence = document.createElement('div');
        confidence.className = 'prediction-confidence';

        const percentage = document.createElement('span');
        percentage.className = 'prediction-percentage';
        percentage.textContent = formatConfidence(pred.confidence);

        const bar = document.createElement('div');
        bar.className = 'prediction-bar';

        const barFill = document.createElement('div');
        barFill.className = 'prediction-bar-fill';
        barFill.style.width = '0%';

        bar.appendChild(barFill);
        confidence.appendChild(percentage);

        item.appendChild(rank);
        item.appendChild(name);
        item.appendChild(confidence);

        container.appendChild(item);

        // Animate bar fill
        setTimeout(() => {
            barFill.style.width = `${pred.confidence * 100}%`;
        }, 100 + index * 50);
    });
}

/**
 * Render result metadata
 * @param {object} data - Prediction data
 * @param {HTMLElement} container - Container element
 */
function renderMetadata(data, container) {
    const metadata = [
        {
            label: 'Processing Time',
            value: formatProcessingTime(data.processing_time)
        },
        {
            label: 'Device',
            value: data.device
        },
        {
            label: 'Timestamp',
            value: formatTimestamp()
        }
    ];

    container.innerHTML = metadata.map(item => `
        <div class="metadata-item">
            <div class="metadata-label">${item.label}</div>
            <div class="metadata-value">${item.value}</div>
        </div>
    `).join('');
}

/**
 * Hide results section
 */
export function hideResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.add('hidden');
}

/**
 * Show heatmap section
 */
export function showHeatmap() {
    const heatmapSection = document.getElementById('heatmapSection');
    heatmapSection.classList.remove('hidden');
    heatmapSection.classList.add('fade-in');
}

/**
 * Hide heatmap section
 */
export function hideHeatmap() {
    const heatmapSection = document.getElementById('heatmapSection');
    heatmapSection.classList.add('hidden');
}

/**
 * Display available actions
 * @param {string[]} actions - Array of action names
 */
export function displayAvailableActions(actions) {
    const actionsGrid = document.getElementById('actionsGrid');

    if (!actions || actions.length === 0) {
        actionsGrid.innerHTML = '<p class="text-center">No actions available</p>';
        return;
    }

    actionsGrid.innerHTML = actions.map(action => {
        const formattedName = formatActionName(action);
        return `<div class="action-tag" title="${formattedName}">${formattedName}</div>`;
    }).join('');
}

/**
 * Update health badge
 * @param {boolean} healthy - Whether API is healthy
 * @param {string} statusText - Status text
 */
export function updateHealthBadge(healthy, statusText = '') {
    const healthBadge = document.getElementById('healthBadge');
    const statusDot = healthBadge.querySelector('.status-dot');
    const statusTextEl = healthBadge.querySelector('.status-text');

    if (healthy) {
        statusDot.classList.add('healthy');
        statusDot.classList.remove('unhealthy');
        statusTextEl.textContent = statusText || 'Online';
    } else {
        statusDot.classList.add('unhealthy');
        statusDot.classList.remove('healthy');
        statusTextEl.textContent = statusText || 'Offline';
    }
}

/**
 * Show toast notification
 * @param {string} message - Notification message
 * @param {string} type - Type: 'success', 'error', 'warning', 'info'
 * @param {number} duration - Duration in ms (default: 5000, 0 = no auto-dismiss)
 */
export function showToast(message, type = 'info', duration = 5000) {
    const toastContainer = document.getElementById('toastContainer');
    const toastId = generateId();

    const icons = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.id = toastId;
    toast.innerHTML = `
        <div class="toast-icon">${icons[type] || icons.info}</div>
        <div class="toast-content">
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" aria-label="Close">×</button>
    `;

    // Close button handler
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => {
        removeToast(toastId);
    });

    // Add to container
    toastContainer.appendChild(toast);

    // Auto-dismiss
    if (duration > 0) {
        setTimeout(() => {
            removeToast(toastId);
        }, duration);
    }
}

/**
 * Remove toast notification
 * @param {string} toastId - Toast ID to remove
 */
function removeToast(toastId) {
    const toast = document.getElementById(toastId);
    if (toast) {
        toast.style.animation = 'slideInRight 0.3s ease-out reverse';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }
}

/**
 * Show error message
 * @param {string} message - Error message
 * @param {string} technical - Technical details (optional)
 */
export function showError(message, technical = '') {
    let fullMessage = message;
    if (technical) {
        fullMessage += `<br><small style="opacity: 0.7">${technical}</small>`;
    }
    showToast(fullMessage, 'error', 7000);
}

/**
 * Show success message
 * @param {string} message - Success message
 */
export function showSuccess(message) {
    showToast(message, 'success', 3000);
}

/**
 * Show warning message
 * @param {string} message - Warning message
 */
export function showWarning(message) {
    showToast(message, 'warning', 5000);
}

/**
 * Show info message
 * @param {string} message - Info message
 */
export function showInfo(message) {
    showToast(message, 'info', 4000);
}

export default {
    showImagePreview,
    clearPreview,
    showLoading,
    hideLoading,
    displayPredictionResults,
    renderTopKPredictions,
    hideResults,
    showHeatmap,
    hideHeatmap,
    displayAvailableActions,
    updateHealthBadge,
    showToast,
    showError,
    showSuccess,
    showWarning,
    showInfo
};
