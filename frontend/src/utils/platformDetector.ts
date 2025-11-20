/**
 * Platform detection utility to identify PyQt environment
 * and disable animations for better performance
 */

/**
 * Check if the app is running inside PyQt WebEngine
 * by detecting the 'platform=pyqt' URL parameter
 */
export const isPyQtEnvironment = (): boolean => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('platform') === 'pyqt';
};

/**
 * Disable CSS animations when running in PyQt environment
 * to prevent flickering issues with WebEngine rendering
 */
export const disableAnimationsForPyQt = (): void => {
    if (isPyQtEnvironment()) {
        document.body.classList.add('disable-animations');
        console.log('[PyQt] Animations disabled for PyQt environment');
    } else {
        console.log('[Web] Running in browser, animations enabled');
    }
};
