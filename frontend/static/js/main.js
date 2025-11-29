/**
 * Main application entry point.
 * I've orchestrated the application logic here, connecting the UI, API, and State.
 */
import { state, resetConversation } from './state.js';
import { streamQuery } from './api.js';
import { 
    elements, 
    addMessage, 
    showLoading, 
    hideLoading, 
    scrollToBottom, 
    autoScrollToBottom, 
    checkPendingTopMessage, 
    isAtBottom, 
    createStreamingMessageContainer, 
    finalizeStreamingMessage, 
    closeModal,
    openImageModal
} from './ui.js';
import { formatText } from './utils.js';

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM loaded, focusing input');
    elements.messageInput.focus();
    console.log('Page fully loaded and ready!');

    // New Chat button handler
    const newChatBtn = document.querySelector('.new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', function () {
            resetConversation();

            // Clear chat messages except welcome message
            const welcomeMessage = elements.chatMessages.querySelector('.welcome-message');
            elements.chatMessages.innerHTML = '';
            if (welcomeMessage) {
                elements.chatMessages.appendChild(welcomeMessage);
            }

            // Clear input
            elements.messageInput.value = '';
            elements.messageInput.focus();

            console.log('New chat started - conversation history cleared');
        });
    }
});

// Event Listeners
console.log('Setting up event listeners...');

elements.sendButton.addEventListener('click', function () {
    console.log('Send button clicked!');
    sendMessage();
});

elements.messageInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

elements.modalClose.addEventListener('click', closeModal);
elements.imageModal.addEventListener('click', function (e) {
    if (e.target === elements.imageModal) {
        closeModal();
    }
});

elements.scrollToBottomBtn.addEventListener('click', scrollToBottom);

elements.chatMessages.addEventListener('scroll', function (e) {
    if (state.isProgrammaticScroll) {
        return;
    }

    const currentScrollTop = elements.chatMessages.scrollTop;
    if (state.isAutoScrollEnabled) {
        console.log('User scrolled - disabling follow mode');
    }
    state.isUserScrolling = true;
    state.isAutoScrollEnabled = false;
    state.pendingTopMessage = null;
    if (!isAtBottom()) {
        elements.scrollToBottomBtn.style.display = 'block';
    } else {
        elements.scrollToBottomBtn.style.display = 'none';
    }

    state.lastScrollTop = currentScrollTop;
});


function sendMessage(queryOverride) {
    console.log('sendMessage function called');
    const message = queryOverride || elements.messageInput.value.trim();
    console.log('Message:', message);
    if (!message) {
        console.log('No message, returning');
        return;
    }
    state.currentQuery = message;

    // Add user message to conversation history
    state.conversationHistory.push({
        role: 'user',
        content: message
    });
    if (state.conversationHistory.length > 20) {
        state.conversationHistory = state.conversationHistory.slice(-20);
    }

    console.log('Adding user message to chat');
    // Add user message to chat
    addMessage(message, 'user');

    // Clear input
    elements.messageInput.value = '';

    // Show loading
    showLoading();
    const userMessages = elements.chatMessages.querySelectorAll('.user-message');
    const lastUserMessage = userMessages[userMessages.length - 1];
    if (lastUserMessage) {
        state.pendingTopMessage = lastUserMessage;
        state.isAutoScrollEnabled = true;
        state.isUserScrolling = false;
        elements.scrollToBottomBtn.style.display = 'none';
        state.isProgrammaticScroll = true;
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
        if (window.scrollTimeout) clearTimeout(window.scrollTimeout);
        window.scrollTimeout = setTimeout(() => { state.isProgrammaticScroll = false; }, 100);
    }
    sendMessageWithStreaming(message);
}

function sendMessageWithStreaming(message) {
    console.log('Sending streaming request to backend...');
    let assistantMessageDiv = null;
    let textDiv = null;
    let currentText = '';
    let metadata = null;
    let hasRenderedVisibleContent = false;
    let pendingUpdate = false;
    let lastUpdateTime = 0;
    let updateTimeout = null;
    const UPDATE_INTERVAL = 200;
    let tableScrollPositions = [];
    
    // Reset interaction state for new message
    state.isTableInteracting = false;
    state.shouldRunAfterInteraction = false;

    function attachTableInteractionHandlers() {
        if (!textDiv) return;
        textDiv.querySelectorAll('.table-wrapper').forEach(wrapper => {
            if (wrapper.dataset.interactionBound === 'true') return;
            wrapper.dataset.interactionBound = 'true';

            const startInteraction = () => {
                state.isTableInteracting = true;
            };
            const endInteraction = () => {
                if (!state.isTableInteracting) return;
                state.isTableInteracting = false;
                if (state.shouldRunAfterInteraction) {
                    state.shouldRunAfterInteraction = false;
                    updateContent(true);
                }
            };
            
            wrapper.addEventListener('pointerdown', startInteraction, { passive: true });
            wrapper.addEventListener('pointerup', endInteraction, { passive: true });
            wrapper.addEventListener('pointerleave', endInteraction, { passive: true });
            wrapper.addEventListener('pointercancel', endInteraction, { passive: true });
        });
    }

    function updateContent(force = false) {
        if (!textDiv) return;
        if (!force && state.isTableInteracting) {
            state.shouldRunAfterInteraction = true;
            pendingUpdate = true;
            return;
        }
        const currentWrappers = Array.from(textDiv.querySelectorAll('.table-wrapper'));
        tableScrollPositions = currentWrappers.map(wrapper => wrapper.scrollLeft);
        textDiv.innerHTML = formatText(currentText) + '<span class="streaming-cursor">|</span>';
        requestAnimationFrame(() => {
            const refreshedWrappers = textDiv.querySelectorAll('.table-wrapper');
            refreshedWrappers.forEach((wrapper, index) => {
                if (typeof tableScrollPositions[index] === 'number') {
                    wrapper.scrollLeft = tableScrollPositions[index];
                }
            });
        });

        pendingUpdate = false;
        attachTableInteractionHandlers();
    }

    streamQuery(message, state.conversationHistory, {
        onMetadata: (data) => {
            metadata = data;
            console.log('Metadata received:', metadata);
            assistantMessageDiv = createStreamingMessageContainer(metadata);
            textDiv = assistantMessageDiv.querySelector('.message-text');
        },
        onContent: (data) => {
            if (textDiv) {
                currentText += data;
                const now = Date.now();

                if (updateTimeout) {
                    clearTimeout(updateTimeout);
                    updateTimeout = null;
                }

                if (!pendingUpdate && (now - lastUpdateTime) >= UPDATE_INTERVAL) {
                    lastUpdateTime = now;
                    updateContent();
                } else {
                    pendingUpdate = true;
                    const delay = Math.max(0, UPDATE_INTERVAL - (now - lastUpdateTime));
                    updateTimeout = setTimeout(() => {
                        lastUpdateTime = Date.now();
                        updateContent();
                        updateTimeout = null;
                    }, delay);
                }
                if (!hasRenderedVisibleContent) {
                    requestAnimationFrame(() => {
                        const renderedText = textDiv.innerText.replace('|', '').trim();
                        if (!hasRenderedVisibleContent && renderedText.length > 0) {
                            hasRenderedVisibleContent = true;
                            hideLoading();
                            console.log('First content rendered on screen - hiding loading');
                            assistantMessageDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    });
                }
                try {
                    textDiv.querySelectorAll('pre code').forEach((block) => {
                        if (window.hljs) window.hljs.highlightElement(block);
                    });
                } catch (e) { /* no-op */ }
                autoScrollToBottom();
                checkPendingTopMessage();
            }
        },
        onDone: () => {
            console.log('Response streaming complete');
            if (updateTimeout) {
                clearTimeout(updateTimeout);
                updateTimeout = null;
            }
            if (pendingUpdate) {
                updateContent(true);
            }

            if (!hasRenderedVisibleContent) {
                hideLoading();
            }
            checkPendingTopMessage(true);
            if (assistantMessageDiv && metadata) {
                // Update history
                state.conversationHistory.push({
                    role: 'assistant',
                    content: currentText
                });
                if (state.conversationHistory.length > 20) {
                    state.conversationHistory = state.conversationHistory.slice(-20);
                }

                finalizeStreamingMessage(assistantMessageDiv, currentText, metadata, (query) => {
                    elements.messageInput.value = query;
                    sendMessage();
                });
            }
        },
        onError: (error) => {
            console.error('Stream error:', error);
            hideLoading();
            addMessage(`Error: ${error}`, 'assistant', [], true);
        }
    });
}
