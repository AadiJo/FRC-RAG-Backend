/**
 * Global state management for the application.
 * I've centralized the state here to make it easier to track and modify.
 */

// Load settings from localStorage
function loadSettings() {
    try {
        const saved = localStorage.getItem('frc_rag_settings');
        if (saved) {
            return JSON.parse(saved);
        }
    } catch (e) {
        console.error('Error loading settings:', e);
    }
    return {
        customApiKey: null,
        customModel: null,
        apiKeyValidated: false
    };
}

const savedSettings = loadSettings();

export const state = {
    currentQuery: '',
    conversationHistory: [],
    isFollowingStream: false, // When true, auto-scroll to bottom during streaming
    isStreaming: false, // Whether we're currently receiving streamed content
    isProgrammaticScroll: false, // Flag to track our own scrolls
    forceScrollButtonVisible: false,
    lastScrollTop: 0,
    userIntentToScroll: false,
    scrollToResponseTop: null, // Element to scroll to top of viewport when streaming starts
    isTableInteracting: false,
    shouldRunAfterInteraction: false,
    // Settings state - API key is never persisted for security
    customApiKey: null,
    customModel: savedSettings.customModel,
    apiKeyValidated: false
};

export function resetConversation() {
    state.conversationHistory = [];
    state.currentQuery = '';
}

export function saveSettings() {
    try {
        // Note: We intentionally don't save the API key for security
        // User will need to re-enter it on page reload
        localStorage.setItem('frc_rag_settings', JSON.stringify({
            customModel: state.customModel
            // customApiKey and apiKeyValidated are not persisted
        }));
    } catch (e) {
        console.error('Error saving settings:', e);
    }
}

export function clearSettings() {
    state.customApiKey = null;
    state.customModel = null;
    state.apiKeyValidated = false;
    try {
        localStorage.removeItem('frc_rag_settings');
    } catch (e) {
        console.error('Error clearing settings:', e);
    }
}
