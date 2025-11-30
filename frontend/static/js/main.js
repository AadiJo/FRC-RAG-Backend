/**
 * Main application entry point.
 * I've compiled the application logic here, connecting the UI, API, and State.
 */
import { state, resetConversation, saveSettings, clearSettings } from './state.js';
import { streamQuery, validateApiKey, getChutesModels } from './api.js';
import { 
    elements, 
    addMessage, 
    showLoading, 
    hideLoading, 
    scrollToBottom, 
    autoScrollIfFollowing, 
    updateScrollButton,
    updateSendButtonIcon,
    scrollToUserMessage,
    handleUserScroll,
    isAtBottom, 
    createStreamingMessageContainer, 
    finalizeStreamingMessage, 
    closeModal,
    openImageModal
} from './ui.js';
import { formatText } from './utils.js';

function updateSendButtonState() {
    if (elements.messageInput && elements.sendButton) {
        if (state.isStreaming) {
            elements.sendButton.disabled = false;
        } else {
            elements.sendButton.disabled = elements.messageInput.value.trim().length === 0;
        }
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM loaded, focusing input');
    elements.messageInput.focus();
    updateSendButtonState();
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
            updateSendButtonState();
            elements.messageInput.focus();

            console.log('New chat started - conversation history cleared');
        });
    }
    
    // Initialize settings modal
    initializeSettingsModal();
});

// Settings Modal Functions
async function initializeSettingsModal() {
    // Load saved API key into input if exists
    if (state.customApiKey) {
        elements.apiKeyInput.value = state.customApiKey;
    }
    
    // Load system prompt if exists
    if (state.systemPrompt) {
        elements.systemPromptInput.value = state.systemPrompt;
    } else {
        elements.systemPromptInput.value = '';
    }
    
    // Setup custom dropdown listeners
    setupCustomDropdown();
    
    // Initialize model selector state
    updateModelSelectorState();
    
    // If we have a validated API key, load models
    if (state.apiKeyValidated && state.customApiKey) {
        await loadModels();
        if (state.customModel) {
            selectModel(state.customModel);
        }
    }
}

function setupCustomDropdown() {
    // Toggle dropdown
    elements.modelSelectorTrigger.addEventListener('click', (e) => {
        if (elements.modelSelectorTrigger.classList.contains('disabled')) return;
        e.stopPropagation();
        elements.modelOptionsList.classList.toggle('show');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!elements.modelSelectorContainer.contains(e.target)) {
            elements.modelOptionsList.classList.remove('show');
        }
    });
}

function selectModel(modelId, modelName = null) {
    elements.modelSelector.value = modelId;
    state.customModel = modelId;
    
    // Update display
    if (!modelName) {
        // Try to find name from options if not provided
        const option = Array.from(elements.modelOptionsList.children).find(opt => opt.dataset.value === modelId);
        if (option) {
            modelName = option.querySelector('.model-name-text').textContent;
        } else {
            modelName = modelId; // Fallback
        }
    }
    
    const triggerText = elements.modelSelectorTrigger.querySelector('.current-model-name');
    if (triggerText) triggerText.textContent = modelName;

    // Update selected state in list
    Array.from(elements.modelOptionsList.children).forEach(opt => {
        if (opt.dataset.value === modelId) opt.classList.add('selected');
        else opt.classList.remove('selected');
    });
    
    elements.modelOptionsList.classList.remove('show');
}

const DEFAULT_SERVER_MODEL = 'openai/gpt-oss-20b';
const DEFAULT_SERVER_MODEL_NAME = 'GPT-OSS 20B (Server Default)';

function updateModelSelectorState() {
    if (state.apiKeyValidated) {
        elements.modelSelectorTrigger.classList.remove('disabled');
        elements.modelStatus.innerHTML = '<i class="fas fa-check-circle"></i> Model selection enabled';
        elements.modelStatus.style.color = '#22c55e';
        elements.apiKeyInput.classList.remove('invalid');
    } else {
        elements.modelSelectorTrigger.classList.add('disabled');
        elements.modelStatus.innerHTML = '<i class="fas fa-lock"></i> Provide a valid API key to select models';
        elements.modelStatus.style.color = '#8e8ea0';
        
        // Reset to server default model
        elements.modelOptionsList.innerHTML = '';
        selectModel(DEFAULT_SERVER_MODEL, DEFAULT_SERVER_MODEL_NAME);
        state.customModel = null;
    }
}

async function loadModels() {
    console.log('Loading models...');
    const data = await getChutesModels();
    console.log('Models response:', data);
    
    // Clear the model selector options
    elements.modelOptionsList.innerHTML = '';
    
    if (data.models && data.models.length > 0) {
        console.log('Found', data.models.length, 'models');
        
        // Sort: free models first, then paid
        const sortedModels = [...data.models].sort((a, b) => {
            if (a.free && !b.free) return -1;
            if (!a.free && b.free) return 1;
            return 0;
        });
        
        sortedModels.forEach(model => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'model-option';
            optionDiv.dataset.value = model.id;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'model-item-content';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'model-name-text';
            nameSpan.textContent = model.name;
            
            contentDiv.appendChild(nameSpan);

            if (!model.free) {
                const badgeSpan = document.createElement('span');
                badgeSpan.className = 'model-badge badge-paid';
                badgeSpan.textContent = 'PAID';
                contentDiv.appendChild(badgeSpan);
            }
            
            optionDiv.appendChild(contentDiv);
            
            optionDiv.addEventListener('click', () => {
                selectModel(model.id, model.name);
            });
            
            elements.modelOptionsList.appendChild(optionDiv);
        });
        
        // Auto-select first model (which should be free since we sorted)
        const firstFreeModel = sortedModels.find(m => m.free);
        if (firstFreeModel) {
            selectModel(firstFreeModel.id, firstFreeModel.name);
        } else if (sortedModels.length > 0) {
            selectModel(sortedModels[0].id, sortedModels[0].name);
        }
    } else {
        console.log('No models found in response');
        // Fallback option if no models loaded
        const fallbackDiv = document.createElement('div');
        fallbackDiv.className = 'model-option';
        fallbackDiv.dataset.value = 'openai/gpt-oss-20b';
        fallbackDiv.textContent = 'GPT-OSS 20B';
        fallbackDiv.addEventListener('click', () => selectModel('openai/gpt-oss-20b', 'GPT-OSS 20B'));
        elements.modelOptionsList.appendChild(fallbackDiv);
        
        selectModel('openai/gpt-oss-20b', 'GPT-OSS 20B');
    }
}

function openSettingsModal() {
    elements.settingsModal.style.display = 'block';
    // Restore current state
    if (state.customApiKey) {
        elements.apiKeyInput.value = state.customApiKey;
    }
    // Restore system prompt state
    elements.systemPromptInput.value = state.systemPrompt || '';

    if (state.apiKeyValidated) {
        elements.apiKeyStatus.innerHTML = '<i class="fas fa-check-circle"></i> API key validated';
        elements.apiKeyStatus.className = 'api-key-status success';
    }
}

function closeSettingsModal() {
    elements.settingsModal.style.display = 'none';
}

async function handleValidateApiKey() {
    const apiKey = elements.apiKeyInput.value.trim();
    
    if (!apiKey) {
        elements.apiKeyStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Please enter an API key';
        elements.apiKeyStatus.className = 'api-key-status error';
        return;
    }
    
    // Show loading state
    elements.validateApiKey.classList.add('loading');
    elements.validateApiKey.querySelector('i').className = 'fas fa-spinner';
    elements.apiKeyStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
    elements.apiKeyStatus.className = 'api-key-status loading';
    
    try {
        const result = await validateApiKey(apiKey);
        
        if (result.valid) {
            elements.apiKeyStatus.innerHTML = '<i class="fas fa-check-circle"></i> ' + result.message;
            elements.apiKeyStatus.className = 'api-key-status success';
            state.customApiKey = apiKey;
            state.apiKeyValidated = true;
            updateModelSelectorState();
            await loadModels();
        } else {
            elements.apiKeyStatus.innerHTML = '<i class="fas fa-times-circle"></i> ' + result.message;
            elements.apiKeyStatus.className = 'api-key-status error';
            elements.apiKeyInput.classList.add('invalid');
            state.apiKeyValidated = false;
            updateModelSelectorState();
        }
    } catch (error) {
        elements.apiKeyStatus.innerHTML = '<i class="fas fa-times-circle"></i> Validation failed';
        elements.apiKeyStatus.className = 'api-key-status error';
        elements.apiKeyInput.classList.add('invalid');
        state.apiKeyValidated = false;
        updateModelSelectorState();
    } finally {
        elements.validateApiKey.classList.remove('loading');
        elements.validateApiKey.querySelector('i').className = 'fas fa-check';
    }
}

function handleToggleApiKeyVisibility() {
    const input = elements.apiKeyInput;
    const icon = elements.toggleApiKeyVisibility.querySelector('i');
    
    if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'fas fa-eye-slash';
    } else {
        input.type = 'password';
        icon.className = 'fas fa-eye';
    }
}

function handleSaveSettings() {
    // Update state with current values
    state.customApiKey = elements.apiKeyInput.value.trim() || null;
    state.customModel = elements.modelSelector.value || null;
    state.systemPrompt = elements.systemPromptInput.value.trim() || null;
    
    // If API key was cleared, reset validation state
    if (!state.customApiKey) {
        state.apiKeyValidated = false;
        state.customModel = null;
    }
    
    // Save to localStorage
    saveSettings();
    
    // Close modal
    closeSettingsModal();
    
    console.log('Settings saved:', { 
        hasApiKey: !!state.customApiKey, 
        model: state.customModel,
        validated: state.apiKeyValidated 
    });
}

// Settings Modal Event Listeners
if (elements.settingsBtn) {
    elements.settingsBtn.addEventListener('click', openSettingsModal);
}

if (elements.settingsClose) {
    elements.settingsClose.addEventListener('click', closeSettingsModal);
}

if (elements.settingsModal) {
    elements.settingsModal.addEventListener('click', function(e) {
        if (e.target === elements.settingsModal) {
            closeSettingsModal();
        }
    });
}

if (elements.toggleApiKeyVisibility) {
    elements.toggleApiKeyVisibility.addEventListener('click', handleToggleApiKeyVisibility);
}

if (elements.validateApiKey) {
    elements.validateApiKey.addEventListener('click', handleValidateApiKey);
}

if (elements.apiKeyInput) {
    elements.apiKeyInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleValidateApiKey();
        }
    });
    
    // Clear validation status and invalid styling when key changes
    elements.apiKeyInput.addEventListener('input', function() {
        // Remove invalid styling when user starts typing
        elements.apiKeyInput.classList.remove('invalid');
        
        if (state.customApiKey !== this.value.trim()) {
            state.apiKeyValidated = false;
            elements.apiKeyStatus.innerHTML = '';
            elements.apiKeyStatus.className = 'api-key-status';
            updateModelSelectorState();
        }
    });
}

if (elements.settingsSave) {
    elements.settingsSave.addEventListener('click', handleSaveSettings);
}

// Event Listeners
console.log('Setting up event listeners...');

elements.messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value === '') {
        this.style.height = '40px';
    }
    // Toggle overflow based on content height vs max height
    if (this.scrollHeight > 200) {
        this.classList.remove('overflow-y-hidden');
        this.classList.add('overflow-y-auto');
    } else {
        this.classList.add('overflow-y-hidden');
        this.classList.remove('overflow-y-auto');
    }
    updateSendButtonState();
});

if (elements.toolsToggle && elements.toolsMenu) {
    elements.toolsToggle.addEventListener('click', function(e) {
        e.stopPropagation();
        const isHidden = elements.toolsMenu.classList.contains('opacity-0');
        
        if (isHidden) {
            // Show
            elements.toolsMenu.classList.remove('opacity-0', 'scale-95', 'pointer-events-none');
            elements.toolsMenu.classList.add('opacity-100', 'scale-100', 'pointer-events-auto');
            this.classList.add('text-primary');
            this.classList.remove('text-[#8e8ea0]');
        } else {
            // Hide
            elements.toolsMenu.classList.add('opacity-0', 'scale-95', 'pointer-events-none');
            elements.toolsMenu.classList.remove('opacity-100', 'scale-100', 'pointer-events-auto');
            this.classList.remove('text-primary');
            this.classList.add('text-[#8e8ea0]');
        }
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!elements.toolsMenu.contains(e.target) && e.target !== elements.toolsToggle && !elements.toolsToggle.contains(e.target)) {
            elements.toolsMenu.classList.add('opacity-0', 'scale-95', 'pointer-events-none');
            elements.toolsMenu.classList.remove('opacity-100', 'scale-100', 'pointer-events-auto');
            elements.toolsToggle.classList.remove('text-primary');
            elements.toolsToggle.classList.add('text-[#8e8ea0]');
        }
    });
}

// Handle Reasoning Tool Chip
if (elements.showReasoning) {
    elements.showReasoning.addEventListener('change', function() {
        updateToolChips();
    });
}

// Initialize chips on load
updateToolChips();

function updateToolChips() {
    const activeTools = elements.activeTools;
    if (!activeTools) return;
    
    activeTools.innerHTML = '';
    
    if (elements.showReasoning && elements.showReasoning.checked) {
        const chip = document.createElement('div');
        chip.className = 'flex items-center gap-1 bg-[#1e3a8a] text-[#60a5fa] px-3 py-1.5 rounded-full text-sm font-medium cursor-pointer hover:bg-[#1e40af] transition-colors group select-none';
        chip.innerHTML = `
            <div class="w-4 h-4 flex items-center justify-center relative">
                <i class="fas fa-brain text-xs absolute transition-opacity duration-200 opacity-100 group-hover:opacity-0"></i>
                <i class="fas fa-times text-xs absolute transition-opacity duration-200 opacity-0 group-hover:opacity-100"></i>
            </div>
            <span class="whitespace-nowrap relative -top-[1px]">Reasoning</span>
        `;
        
        chip.addEventListener('click', function() {
            elements.showReasoning.checked = false;
            updateToolChips();
        });
        
        activeTools.appendChild(chip);
    }
}

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
    if (e.target === elements.imageModal || e.target.classList.contains('image-modal-content')) {
        closeModal();
    }
});

elements.scrollToBottomBtn.addEventListener('click', scrollToBottom);

elements.chatMessages.addEventListener('scroll', function (e) {
    handleUserScroll();
});


function sendMessage(queryOverride) {
    console.log('sendMessage function called');
    
    // Handle stop button click
    if (state.isStreaming) {
        if (state.abortController) {
            state.abortController.abort();
            state.abortController = null;
            // The streamQuery error handler will catch the abort error
        }
        return;
    }

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
    elements.messageInput.style.height = '40px';
    updateSendButtonState();

    // Show loading
    showLoading();
    
    // Reset scroll state for new message - don't follow stream initially
    state.isFollowingStream = false;
    state.isStreaming = true;
    updateSendButtonIcon();
    updateSendButtonState();
    
    state.forceScrollButtonVisible = false;
    elements.scrollToBottomBtn.setAttribute('data-visible', 'false'); // Hide until we can show it appropriately
    
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
    
    // Create abort controller
    state.abortController = new AbortController();

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
        // Don't update if streaming has ended (finalizeStreamingMessage will handle it)
        if (!state.isStreaming) return;
        
        if (!force && state.isTableInteracting) {
            state.shouldRunAfterInteraction = true;
            pendingUpdate = true;
            return;
        }
        
        // Save scroll positions of all table wrappers before update
        const currentWrappers = Array.from(textDiv.querySelectorAll('.table-wrapper'));
        tableScrollPositions = currentWrappers.map(wrapper => ({
            scrollLeft: wrapper.scrollLeft,
            scrollTop: wrapper.scrollTop
        }));
        
        // Use requestAnimationFrame for smoother updates
        requestAnimationFrame(() => {
            // Double-check streaming is still active (could have ended during RAF delay)
            if (!state.isStreaming) return;
            
            textDiv.innerHTML = formatText(currentText) + '<span class="streaming-cursor">|</span>';
            
            // Restore scroll positions after DOM update
            const refreshedWrappers = textDiv.querySelectorAll('.table-wrapper');
            refreshedWrappers.forEach((wrapper, index) => {
                if (tableScrollPositions[index]) {
                    wrapper.scrollLeft = tableScrollPositions[index].scrollLeft;
                    wrapper.scrollTop = tableScrollPositions[index].scrollTop;
                }
            });
            
            attachTableInteractionHandlers();
        });

        pendingUpdate = false;
    }

    const options = {
        show_reasoning: elements.showReasoning ? elements.showReasoning.checked : false,
        signal: state.abortController.signal
    };

    streamQuery(message, state.conversationHistory, options, {
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
                            // Scroll to show the user message at the top
                            const userMessages = elements.chatMessages.querySelectorAll('.user-message');
                            const lastUserMessage = userMessages[userMessages.length - 1];
                            scrollToUserMessage(lastUserMessage);
                            // Force-show the follow arrow so user can opt in even if at bottom
                            state.forceScrollButtonVisible = true;
                            updateScrollButton();
                        }
                    });
                }
                try {
                    textDiv.querySelectorAll('pre code').forEach((block) => {
                        if (window.hljs) window.hljs.highlightElement(block);
                    });
                } catch (e) { /* no-op */ }
                
                // Auto-scroll if user clicked follow button
                autoScrollIfFollowing();
            }
        },
        onDone: () => {
            console.log('Response streaming complete');
            if (updateTimeout) {
                clearTimeout(updateTimeout);
                updateTimeout = null;
            }
            
            // Mark streaming as done FIRST to prevent updateContent from running
            state.isStreaming = false;
            state.abortController = null;
            updateSendButtonIcon();
            updateSendButtonState();

            state.forceScrollButtonVisible = false;

            if (!hasRenderedVisibleContent) {
                hideLoading();
            }
            
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
            if (updateTimeout) {
                clearTimeout(updateTimeout);
                updateTimeout = null;
            }
            
            state.isStreaming = false;
            state.abortController = null;
            updateSendButtonIcon();
            updateSendButtonState();
            
            state.forceScrollButtonVisible = false;
            hideLoading();
            
            // If aborted, we might want to keep what we have so far
            // Check for various abort error signatures
            const errorMessage = typeof error === 'string' ? error : (error.message || '');
            const isAbort = error.name === 'AbortError' || 
                           errorMessage.toLowerCase().includes('aborted') ||
                           error === 'AbortError';

            if (isAbort) {
                console.log('Request aborted by user');
                if (assistantMessageDiv && metadata) {
                    // Update history with partial response
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
            } else {
                addMessage(`Error: ${error.message || error}`, 'assistant', [], true);
            }
        }
    });
}
