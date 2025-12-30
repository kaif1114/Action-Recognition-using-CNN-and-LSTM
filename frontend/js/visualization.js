/**
 * Visualization Module
 * Handles attention heatmap rendering and manipulation
 */

import { downloadFile, clamp } from './utils.js';

// Current state
let currentImage = null;
let currentHeatmap = null;
let currentMode = 'overlay';
let currentOpacity = 0.6;

/**
 * Render attention heatmap on canvas
 * @param {Array<Array<number>>} attentionGrid - 7x7 grid of attention values
 * @param {string} imageUrl - URL of the original image
 * @param {number} opacity - Opacity for overlay mode (0-1)
 * @param {string} mode - Display mode: 'original', 'heatmap', 'overlay'
 */
export async function renderAttentionHeatmap(attentionGrid, imageUrl, opacity = 0.6, mode = 'overlay') {
    const canvas = document.getElementById('heatmapCanvas');
    const ctx = canvas.getContext('2d');

    // Store current state
    currentHeatmap = attentionGrid;
    currentOpacity = opacity;
    currentMode = mode;

    // Load image
    const img = await loadImage(imageUrl);
    currentImage = img;

    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Render based on mode
    switch (mode) {
        case 'original':
            renderOriginal(ctx, img);
            break;
        case 'heatmap':
            renderHeatmapOnly(ctx, attentionGrid, img.width, img.height);
            break;
        case 'overlay':
            renderOverlay(ctx, img, attentionGrid, opacity);
            break;
        default:
            console.error('Invalid render mode:', mode);
    }
}

/**
 * Render original image only
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {HTMLImageElement} img - Image element
 */
function renderOriginal(ctx, img) {
    ctx.drawImage(img, 0, 0);
}

/**
 * Render heatmap only (no original image)
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Array<Array<number>>} grid - 7x7 attention grid
 * @param {number} width - Canvas width
 * @param {number} height - Canvas height
 */
function renderHeatmapOnly(ctx, grid, width, height) {
    // Interpolate grid to canvas dimensions
    const interpolated = interpolateGrid(grid, width, height);

    // Draw heatmap
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = interpolated[y][x];
            const color = valueToColor(value);
            const idx = (y * width + x) * 4;

            data[idx] = color.r;
            data[idx + 1] = color.g;
            data[idx + 2] = color.b;
            data[idx + 3] = 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

/**
 * Render overlay (original image + heatmap)
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {HTMLImageElement} img - Image element
 * @param {Array<Array<number>>} grid - 7x7 attention grid
 * @param {number} opacity - Heatmap opacity (0-1)
 */
function renderOverlay(ctx, img, grid, opacity) {
    // Draw original image
    ctx.drawImage(img, 0, 0);

    // Create heatmap overlay
    const width = img.width;
    const height = img.height;
    const interpolated = interpolateGrid(grid, width, height);

    // Draw heatmap with opacity
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = interpolated[y][x];
            const color = valueToColor(value);
            const idx = (y * width + x) * 4;

            data[idx] = color.r;
            data[idx + 1] = color.g;
            data[idx + 2] = color.b;
            data[idx + 3] = opacity * 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);
}

/**
 * Interpolate 7x7 grid to target dimensions using bilinear interpolation
 * @param {Array<Array<number>>} grid - 7x7 attention grid
 * @param {number} targetWidth - Target width
 * @param {number} targetHeight - Target height
 * @returns {Array<Array<number>>} Interpolated grid
 */
function interpolateGrid(grid, targetWidth, targetHeight) {
    const gridSize = grid.length; // Should be 7
    const result = Array(targetHeight).fill(null).map(() => Array(targetWidth).fill(0));

    for (let y = 0; y < targetHeight; y++) {
        for (let x = 0; x < targetWidth; x++) {
            // Map pixel coordinates to grid coordinates
            const gx = (x / targetWidth) * (gridSize - 1);
            const gy = (y / targetHeight) * (gridSize - 1);

            // Get grid cell indices
            const x0 = Math.floor(gx);
            const x1 = Math.min(x0 + 1, gridSize - 1);
            const y0 = Math.floor(gy);
            const y1 = Math.min(y0 + 1, gridSize - 1);

            // Get fractional parts
            const fx = gx - x0;
            const fy = gy - y0;

            // Bilinear interpolation
            const v00 = grid[y0][x0];
            const v10 = grid[y0][x1];
            const v01 = grid[y1][x0];
            const v11 = grid[y1][x1];

            const v0 = v00 * (1 - fx) + v10 * fx;
            const v1 = v01 * (1 - fx) + v11 * fx;
            const value = v0 * (1 - fy) + v1 * fy;

            result[y][x] = value;
        }
    }

    return result;
}

/**
 * Map attention value (0-1) to color
 * Color scheme: Blue (low) → Green → Yellow → Red (high)
 * @param {number} value - Attention value (0-1)
 * @returns {object} RGB color {r, g, b}
 */
function valueToColor(value) {
    // Clamp value to [0, 1]
    value = clamp(value, 0, 1);

    // Define color stops (from blue to red)
    const colors = [
        { pos: 0.0, r: 0, g: 0, b: 255 },      // Blue
        { pos: 0.25, r: 0, g: 255, b: 255 },   // Cyan
        { pos: 0.5, r: 0, g: 255, b: 0 },      // Green
        { pos: 0.75, r: 255, g: 255, b: 0 },   // Yellow
        { pos: 1.0, r: 255, g: 0, b: 0 }       // Red
    ];

    // Find color segment
    for (let i = 0; i < colors.length - 1; i++) {
        const c0 = colors[i];
        const c1 = colors[i + 1];

        if (value >= c0.pos && value <= c1.pos) {
            // Linear interpolation between c0 and c1
            const t = (value - c0.pos) / (c1.pos - c0.pos);
            return {
                r: Math.round(c0.r + t * (c1.r - c0.r)),
                g: Math.round(c0.g + t * (c1.g - c0.g)),
                b: Math.round(c0.b + t * (c1.b - c0.b))
            };
        }
    }

    // Fallback (should not reach here)
    return { r: 255, g: 0, b: 0 };
}

/**
 * Load image from URL
 * @param {string} url - Image URL
 * @returns {Promise<HTMLImageElement>}
 */
function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = url;
    });
}

/**
 * Update heatmap opacity
 * @param {number} opacity - New opacity (0-1)
 */
export function updateOpacity(opacity) {
    if (!currentImage || !currentHeatmap) {
        console.warn('No heatmap to update');
        return;
    }

    currentOpacity = clamp(opacity, 0, 1);
    renderAttentionHeatmap(
        currentHeatmap,
        currentImage.src,
        currentOpacity,
        currentMode
    );
}

/**
 * Change display mode
 * @param {string} mode - Display mode: 'original', 'heatmap', 'overlay'
 */
export function changeMode(mode) {
    if (!currentImage || !currentHeatmap) {
        console.warn('No heatmap to update');
        return;
    }

    currentMode = mode;
    renderAttentionHeatmap(
        currentHeatmap,
        currentImage.src,
        currentOpacity,
        currentMode
    );
}

/**
 * Download heatmap as PNG
 * @param {string} filename - Output filename
 */
export function downloadHeatmap(filename = 'attention_heatmap.png') {
    const canvas = document.getElementById('heatmapCanvas');

    if (!canvas) {
        console.error('Canvas not found');
        return;
    }

    // Convert canvas to blob
    canvas.toBlob((blob) => {
        if (blob) {
            downloadFile(blob, filename, 'image/png');
        } else {
            console.error('Failed to create blob from canvas');
        }
    }, 'image/png');
}

/**
 * Get current heatmap state
 * @returns {object} Current state
 */
export function getCurrentState() {
    return {
        image: currentImage,
        heatmap: currentHeatmap,
        mode: currentMode,
        opacity: currentOpacity
    };
}

/**
 * Clear heatmap state
 */
export function clearHeatmap() {
    currentImage = null;
    currentHeatmap = null;
    currentMode = 'overlay';
    currentOpacity = 0.6;

    const canvas = document.getElementById('heatmapCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

export default {
    renderAttentionHeatmap,
    updateOpacity,
    changeMode,
    downloadHeatmap,
    getCurrentState,
    clearHeatmap
};
