/**
 * Main Application Controller
 * Coordinates all modules and handles user interactions
 */

import {
    validateImageFile,
    getImageDimensions,
    saveToLocalStorage,
    loadFromLocalStorage
} from './utils.js';

import {
    setApiBaseUrl,
    getApiBaseUrl,
    getHealthStatus,
    getAvailableActions,
    predictAction
} from './api.js';

import {
    showImagePreview,
    clearPreview,
    showLoading,
    hideLoading,
    displayPredictionResults,
    displayAvailableActions,
    updateHealthBadge,
    showError,
    showSuccess,
    showWarning,
    showHeatmap,
    hideHeatmap
} from './ui.js';

import {
    renderAttentionHeatmap,
    updateOpacity,
    changeMode,
    downloadHeatmap,
    clearHeatmap
} from './visualization.js';

// Application state
const state = {
    currentFile: null,
    currentImageUrl: null,
    settings: {
        apiEndpoint: 'http://localhost:8000',
        topK: 5,
        includeAttention: false
    }
};

/**
 * Initialize application
 */
async function init() {
    console.log('Initializing Action Recognition System...');

    // Load settings from localStorage
    loadSettings();

    // Setup event listeners
    setupEventListeners();

    // Check API health
    await checkApiHealth();

    // Load available actions
    await loadAvailableActions();

    console.log('Application initialized successfully');
}

/**
 * Load settings from localStorage
 */
function loadSettings() {
    const savedSettings = loadFromLocalStorage('actionRecognitionSettings');

    if (savedSettings) {
        state.settings = { ...state.settings, ...savedSettings };

        // Apply settings to UI
        const apiEndpointInput = document.getElementById('apiEndpoint');
        const topKSlider = document.getElementById('topKSlider');
        const topKValue = document.getElementById('topKValue');
        const includeAttentionCheckbox = document.getElementById('includeAttention');

        if (apiEndpointInput) {
            apiEndpointInput.value = state.settings.apiEndpoint;
            setApiBaseUrl(state.settings.apiEndpoint);
        }

        if (topKSlider) {
            topKSlider.value = state.settings.topK;
            topKValue.textContent = state.settings.topK;
        }

        if (includeAttentionCheckbox) {
            includeAttentionCheckbox.checked = state.settings.includeAttention;
        }
    }
}

/**
 * Save settings to localStorage
 */
function saveSettings() {
    saveToLocalStorage('actionRecognitionSettings', state.settings);
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Upload area events
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const filePickerBtn = document.getElementById('filePickerBtn');

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // File picker
    filePickerBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Advanced options toggle
    const optionsToggle = document.getElementById('optionsToggle');
    const optionsContent = document.getElementById('optionsContent');

    optionsToggle.addEventListener('click', () => {
        optionsToggle.classList.toggle('active');
        optionsContent.classList.toggle('open');
    });

    // Top-K slider
    const topKSlider = document.getElementById('topKSlider');
    const topKValue = document.getElementById('topKValue');

    topKSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        topKValue.textContent = value;
        state.settings.topK = value;
        saveSettings();
    });

    // Top-K presets
    const presetBtns = document.querySelectorAll('.preset-btn');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const value = parseInt(btn.dataset.value);
            topKSlider.value = value;
            topKValue.textContent = value;
            state.settings.topK = value;
            saveSettings();

            // Update active state
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Include attention checkbox
    const includeAttentionCheckbox = document.getElementById('includeAttention');
    includeAttentionCheckbox.addEventListener('change', (e) => {
        state.settings.includeAttention = e.target.checked;
        saveSettings();
    });

    // API endpoint input
    const apiEndpointInput = document.getElementById('apiEndpoint');
    apiEndpointInput.addEventListener('change', (e) => {
        const newEndpoint = e.target.value.trim();
        state.settings.apiEndpoint = newEndpoint;
        setApiBaseUrl(newEndpoint);
        saveSettings();
        checkApiHealth();
    });

    // Predict button
    const predictBtn = document.getElementById('predictBtn');
    predictBtn.addEventListener('click', handlePredict);

    // Clear button
    const clearBtn = document.getElementById('clearBtn');
    clearBtn.addEventListener('click', handleClear);

    // Heatmap controls
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityValue = document.getElementById('opacityValue');

    opacitySlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        opacityValue.textContent = value + '%';
        updateOpacity(value / 100);
    });

    // View mode buttons
    const modeBtns = document.querySelectorAll('.mode-btn');
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            changeMode(mode);

            // Update active state
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Download heatmap button
    const downloadHeatmapBtn = document.getElementById('downloadHeatmapBtn');
    downloadHeatmapBtn.addEventListener('click', () => {
        downloadHeatmap('attention_heatmap.png');
        showSuccess('Heatmap downloaded successfully');
    });
}

/**
 * Handle drag over event
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-over');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');
}

/**
 * Handle drop event
 */
async function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        await processFile(files[0]);
    }
}

/**
 * Handle file select from input
 */
async function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        await processFile(files[0]);
    }
}

/**
 * Process uploaded file
 */
async function processFile(file) {
    // Validate file
    const validation = validateImageFile(file);

    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    try {
        // Get image dimensions
        const dimensions = await getImageDimensions(file);

        // Store file and create URL
        state.currentFile = file;

        // Revoke previous URL if exists
        if (state.currentImageUrl) {
            URL.revokeObjectURL(state.currentImageUrl);
        }

        state.currentImageUrl = URL.createObjectURL(file);

        // Show preview
        showImagePreview(file, dimensions);

        showSuccess('Image loaded successfully');

    } catch (error) {
        showError('Failed to process image', error.message);
        console.error(error);
    }
}

/**
 * Handle predict action
 */
async function handlePredict() {
    if (!state.currentFile) {
        showWarning('Please upload an image first');
        return;
    }

    const { topK, includeAttention } = state.settings;

    try {
        // Show loading state
        const estimatedTime = includeAttention ? '~100-150ms' : '~50-100ms';
        showLoading('Analyzing image...', `Estimated time: ${estimatedTime}`);

        // Make prediction request
        const result = await predictAction(state.currentFile, topK, includeAttention);

        if (!result.success) {
            throw new Error(result.error || 'Prediction failed');
        }

        // Hide loading
        hideLoading();

        // Display results
        displayPredictionResults(result.data, state.currentImageUrl);

        // Handle attention heatmap if included
        if (includeAttention && result.data.attention_heatmap) {
            renderAttentionHeatmap(
                result.data.attention_heatmap,
                state.currentImageUrl,
                0.6,
                'overlay'
            );
            showHeatmap();
        } else {
            hideHeatmap();
        }

        showSuccess('Prediction completed successfully');

    } catch (error) {
        hideLoading();

        let errorMessage = 'Failed to predict action';
        let technical = error.message;

        // Handle specific error cases
        if (error.message.includes('Cannot connect')) {
            errorMessage = 'Cannot connect to API server';
            technical = `Please ensure the server is running at ${getApiBaseUrl()}`;
        } else if (error.message.includes('Model or action labels not loaded')) {
            errorMessage = 'Model not loaded on server';
            technical = 'The server is running but the model is not initialized';
        }

        showError(errorMessage, technical);
        console.error('Prediction error:', error);
    }
}

/**
 * Handle clear action
 */
function handleClear() {
    // Clear state
    state.currentFile = null;

    if (state.currentImageUrl) {
        URL.revokeObjectURL(state.currentImageUrl);
        state.currentImageUrl = null;
    }

    // Clear UI
    clearPreview();
    clearHeatmap();

    // Reset file input
    const fileInput = document.getElementById('fileInput');
    fileInput.value = '';
}

/**
 * Check API health status
 */
async function checkApiHealth() {
    try {
        const result = await getHealthStatus();

        if (result.success && result.healthy) {
            updateHealthBadge(true, 'Online');

            // Show model info if available
            if (result.data && result.data.device) {
                console.log(`API healthy - Device: ${result.data.device}`);
            }
        } else {
            updateHealthBadge(false, 'Offline');

            if (result.error) {
                console.warn('API health check failed:', result.error);
            }
        }
    } catch (error) {
        updateHealthBadge(false, 'Error');
        console.error('Health check error:', error);
    }
}

/**
 * Load available actions from API
 */
async function loadAvailableActions() {
    try {
        const result = await getAvailableActions();

        if (result.success) {
            displayAvailableActions(result.actions);
            console.log(`Loaded ${result.actions.length} available actions`);
        } else {
            console.warn('Failed to load available actions:', result.error);
            displayAvailableActions([]);
        }
    } catch (error) {
        console.error('Error loading available actions:', error);
        displayAvailableActions([]);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for debugging
window.appState = state;
window.checkApiHealth = checkApiHealth;
