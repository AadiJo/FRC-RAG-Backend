/**
 * Global state management for the application.
 * I've centralized the state here to make it easier to track and modify.
 */
export const state = {
    currentQuery: '',
    conversationHistory: [],
    isAutoScrollEnabled: false, // Start disabled by default
    isUserScrolling: false, // Track if user is manually scrolling
    lastScrollTop: 0,
    isProgrammaticScroll: false, // Flag to track our own scrolls
    pendingTopMessage: null,
    isTableInteracting: false,
    shouldRunAfterInteraction: false
};

export function resetConversation() {
    state.conversationHistory = [];
    state.currentQuery = '';
}
