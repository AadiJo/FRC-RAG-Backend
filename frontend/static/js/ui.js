/**
 * UI components and DOM manipulation.
 * I've grouped all visual logic here to keep the presentation layer separate.
 */
import { getApiUrl, formatText, getTeamDisplayName } from './utils.js';
import { loadImage, submitFeedback } from './api.js';
import { state } from './state.js';

// DOM Elements
export const elements = {
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingStage: document.getElementById('loadingStage'),
    imageModal: document.getElementById('imageModal'),
    modalImage: document.getElementById('modalImage'),
    modalTitle: document.getElementById('modalTitle'),
    modalDescription: document.getElementById('modalDescription'),
    modalClose: document.getElementById('modalClose'),
    scrollToBottomBtn: document.getElementById('scrollToBottomBtn')
};

export function showLoading() {
    elements.loadingOverlay.style.display = 'flex';
    elements.loadingStage.textContent = 'Querying';
    setTimeout(() => {
        elements.loadingStage.textContent = 'Processing';
    }, 800);

    setTimeout(() => {
        elements.loadingStage.textContent = 'Finishing up';
    }, 2000);
}

export function hideLoading() {
    elements.loadingOverlay.style.display = 'none';
}

export function isAtBottom() {
    const threshold = 50; // pixels from bottom
    return elements.chatMessages.scrollHeight - elements.chatMessages.scrollTop - elements.chatMessages.clientHeight < threshold;
}

export function scrollToBottom() {
    console.log('Scroll button clicked - enabling follow mode');
    state.isAutoScrollEnabled = true;
    state.isUserScrolling = false;
    state.isProgrammaticScroll = true;
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    state.lastScrollTop = elements.chatMessages.scrollTop;
    elements.scrollToBottomBtn.style.display = 'none';
    if (window.scrollTimeout) clearTimeout(window.scrollTimeout);
    window.scrollTimeout = setTimeout(() => { state.isProgrammaticScroll = false; }, 100);
}

export function autoScrollToBottom() {
    if (state.isAutoScrollEnabled && !state.isUserScrolling) {
        state.isProgrammaticScroll = true;
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
        state.lastScrollTop = elements.chatMessages.scrollTop;
        elements.scrollToBottomBtn.style.display = 'none';
        if (window.scrollTimeout) clearTimeout(window.scrollTimeout);
        window.scrollTimeout = setTimeout(() => { state.isProgrammaticScroll = false; }, 100);
    } else if (!state.isAutoScrollEnabled && !isAtBottom()) {
        elements.scrollToBottomBtn.style.display = 'block';
    }
}

export function hasMessageReachedTop(messageElement) {
    if (!messageElement || !elements.chatMessages) return false;
    const chatRect = elements.chatMessages.getBoundingClientRect();
    const messageRect = messageElement.getBoundingClientRect();
    return messageRect.top <= chatRect.top;
}

export function checkPendingTopMessage(force = false) {
    if (!state.pendingTopMessage) return;

    if (force || hasMessageReachedTop(state.pendingTopMessage)) {
        console.log('Pending message reached top, disabling auto-scroll.');
        state.pendingTopMessage = null;
        state.isAutoScrollEnabled = false;
        if (!isAtBottom()) {
            elements.scrollToBottomBtn.style.display = 'block';
        }
    }
}

export function openImageModal(image) {
    // Clear previous image to prevent flashing
    elements.modalImage.src = '';
    elements.modalImage.alt = 'Loading...';

    console.log('Opening modal for image:', image.filename);
    // Use loadImage helper to handle ngrok headers
    loadImage(getApiUrl(`/images/${image.web_path}`), elements.modalImage);
    elements.modalTitle.textContent = getTeamDisplayName(image.web_path || image.file_path);

    let modalContent = `
        <strong>Page:</strong> ${image.page || 'N/A'}<br>
        <strong>Path:</strong> ${image.file_path}<br>
    `;

    // Show formatted context if available, otherwise fall back to OCR text
    if (image.formatted_context && image.formatted_context.trim()) {
        modalContent += `<div class="formatted-context"><strong>Context:</strong><br><pre class="context-content">${image.formatted_context}</pre></div>`;
    } else if (image.ocr_text && image.ocr_text.trim()) {
        modalContent += `<div class="formatted-context"><strong>OCR Text:</strong><br><pre class="context-content">${image.ocr_text}</pre></div>`;
    }

    elements.modalDescription.innerHTML = modalContent;
    elements.imageModal.style.display = 'block';
    console.log('Modal should now be visible');
}

export function closeModal() {
    elements.imageModal.style.display = 'none';
}

function createImageItem(image, index) {
    const imageDiv = document.createElement('div');
    imageDiv.className = 'image-item';

    if (image.exists) {
        const imgContainer = document.createElement('div');
        imgContainer.className = 'image-thumbnail-container';

        const img = document.createElement('img');
        // Use loadImage helper to handle ngrok headers
        loadImage(getApiUrl(`/images/${image.web_path}`), img);
        img.alt = image.filename;
        img.className = 'image-thumbnail';

        const overlay = document.createElement('div');
        overlay.className = 'image-overlay';
        overlay.innerHTML = `
            <i class="fas fa-search-plus"></i>
            <span>Click to view</span>
        `;
        const clickHandler = () => openImageModal(image);
        imgContainer.addEventListener('click', clickHandler);

        imgContainer.appendChild(img);
        imgContainer.appendChild(overlay);
        imageDiv.appendChild(imgContainer);
    } else {
        imageDiv.className += ' image-not-found';
        imageDiv.innerHTML = `
            <div class="image-placeholder">
                <i class="fas fa-image"></i>
                <span>Image not found</span>
            </div>
        `;
    }

    const infoDiv = document.createElement('div');
    infoDiv.className = 'image-info';
    const basicInfo = document.createElement('div');
    basicInfo.className = 'image-basic-info';
    basicInfo.innerHTML = `
        <div class="image-filename">${getTeamDisplayName(image.web_path || image.file_path)}</div>
        <div class="image-page">Page ${image.page || 'N/A'}</div>
    `;
    infoDiv.appendChild(basicInfo);

    // Add context summary if available with improved formatting
    if (image.context_summary && image.context_summary.trim()) {
        const contextDiv = document.createElement('div');
        contextDiv.className = 'image-context-summary';
        let contextText = image.context_summary.trim();
        if (contextText.length > 120) {
            contextText = contextText.substring(0, 120) + '...';
        }

        contextDiv.innerHTML = `
            <div class="context-label"><i class="fas fa-info-circle"></i> Context</div>
            <div class="context-text">${contextText}</div>
        `;
        infoDiv.appendChild(contextDiv);
    }
    infoDiv.addEventListener('click', (e) => {
        e.stopPropagation();
        if (image.exists) {
            openImageModal(image);
        }
    });

    imageDiv.appendChild(infoDiv);
    return imageDiv;
}

export function addMessage(text, sender, images = [], isError = false, enhancedData = null, onRedo = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message${isError ? ' error-message' : ''}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    if (sender === 'user') {
        avatarDiv.innerHTML = '<div class="user-avatar"><i class="fas fa-user"></i></div>';
    } else {
        avatarDiv.innerHTML = '<div class="assistant-avatar"><i class="fas fa-robot"></i></div>';
    }

    const textContentDiv = document.createElement('div');
    textContentDiv.className = 'message-text-container';

    // Add game piece mapping info if available
    if (enhancedData && enhancedData.matched_pieces) {
        let matchedPiecesText = '';
        if (Array.isArray(enhancedData.matched_pieces)) {
            matchedPiecesText = enhancedData.matched_pieces.join(', ');
        } else {
            matchedPiecesText = enhancedData.matched_pieces.toString();
        }

        if (matchedPiecesText.trim()) {
            const mappingDiv = document.createElement('div');
            mappingDiv.className = 'game-piece-mapping';
            mappingDiv.innerHTML = `
                <div class="mapping-header">
                    <i class="fas fa-link"></i> Game Piece Mapping
                </div>
                <div class="mapping-content">
                    Detected: <strong>${matchedPiecesText}</strong>
                    ${enhancedData.enhanced_query !== enhancedData.query ?
                    `<br><small>Enhanced search: "${enhancedData.enhanced_query}"</small>` : ''}
                </div>
            `;
            textContentDiv.appendChild(mappingDiv);
        }
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = formatText(text);

    textContentDiv.appendChild(textDiv);
    try {
        textDiv.querySelectorAll('pre code').forEach((block) => {
            if (window.hljs) window.hljs.highlightElement(block);
        });
    } catch (e) { /* no-op */ }

    // Add images if present
    if (images && images.length > 0) {
        const imagesDiv = document.createElement('div');
        imagesDiv.className = 'message-images';

        const imageHeader = document.createElement('div');
        imageHeader.className = 'image-header';
        imageHeader.innerHTML = `<i class="fas fa-images"></i> Related Images (${images.length})`;
        imagesDiv.appendChild(imageHeader);

        const imageGrid = document.createElement('div');
        imageGrid.className = 'image-grid';

        images.forEach((image, index) => {
            const imageItem = createImageItem(image, index);
            imageGrid.appendChild(imageItem);
        });

        imagesDiv.appendChild(imageGrid);
        textContentDiv.appendChild(imagesDiv);
    }

    // Add feedback buttons for assistant messages (non-error)
    if (sender === 'assistant' && !isError) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-buttons';

        const feedbackLabel = document.createElement('div');
        feedbackLabel.className = 'feedback-label';
        feedbackLabel.textContent = 'Was this response helpful?';

        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'feedback-button-container';
        const goodButton = document.createElement('button');
        goodButton.className = 'feedback-btn feedback-good';
        goodButton.innerHTML = '<i class="fas fa-thumbs-up"></i> Good';
        goodButton.onclick = () => handleFeedback(text, 'good', feedbackDiv, onRedo);
        const badButton = document.createElement('button');
        badButton.className = 'feedback-btn feedback-bad';
        badButton.innerHTML = '<i class="fas fa-thumbs-down"></i> Bad';
        badButton.onclick = () => handleFeedback(text, 'bad', feedbackDiv, onRedo);

        buttonsContainer.appendChild(goodButton);
        buttonsContainer.appendChild(badButton);
        messageDiv.classList.add('latest-assistant-message');

        feedbackDiv.appendChild(feedbackLabel);
        feedbackDiv.appendChild(buttonsContainer);

        textContentDiv.appendChild(feedbackDiv);
    }

    contentDiv.appendChild(avatarDiv);
    contentDiv.appendChild(textContentDiv);

    messageDiv.appendChild(contentDiv);

    elements.chatMessages.appendChild(messageDiv);

    // If this is an assistant message, manage redo button visibility
    if (sender === 'assistant' && !isError) {
        // Remove redo button from all previous assistant messages
        const previousLatestMessages = elements.chatMessages.querySelectorAll('.latest-assistant-message');
        previousLatestMessages.forEach(msg => {
            if (msg !== messageDiv) {
                msg.classList.remove('latest-assistant-message');
                const redoBtn = msg.querySelector('.feedback-redo');
                if (redoBtn) {
                    redoBtn.remove();
                }
            }
        });

        // Add redo button to the current (latest) message
        const buttonsContainer = messageDiv.querySelector('.feedback-button-container');
        if (buttonsContainer && onRedo) {
            const redoButton = document.createElement('button');
            redoButton.className = 'feedback-btn feedback-redo';
            redoButton.innerHTML = '<i class="fas fa-redo"></i> Redo';
            redoButton.onclick = () => handleFeedback(text, 'redo', messageDiv.querySelector('.feedback-buttons'), onRedo);
            buttonsContainer.appendChild(redoButton);
        }
    }

    autoScrollToBottom();
}

function handleFeedback(responseText, feedbackType, feedbackDiv, onRedo) {
    const buttons = feedbackDiv.querySelectorAll('.feedback-btn');
    buttons.forEach(btn => btn.disabled = true);
    const buttonsContainer = feedbackDiv.querySelector('.feedback-button-container');
    buttonsContainer.innerHTML = '<span class="feedback-loading"><i class="fas fa-spinner fa-spin"></i> Submitting feedback...</span>';

    submitFeedback(state.currentQuery, responseText, feedbackType, 
        (data) => {
            buttonsContainer.innerHTML = `<span class="feedback-success"><i class="fas fa-check"></i> Thank you for your feedback!</span>`;
            if (feedbackType === 'redo' && onRedo) {
                setTimeout(() => {
                    onRedo(state.currentQuery);
                }, 1000);
            }
        },
        (error) => {
            buttonsContainer.innerHTML = '<span class="feedback-error"><i class="fas fa-exclamation-triangle"></i> Failed to submit feedback</span>';
            setTimeout(() => {
                location.reload();
            }, 2000);
        }
    );
}

export function createStreamingMessageContainer(metadata) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = '<div class="assistant-avatar"><i class="fas fa-robot"></i></div>';

    const textContentDiv = document.createElement('div');
    textContentDiv.className = 'message-text-container';

    // Add game piece mapping info if available
    if (metadata && metadata.matched_pieces) {
        let matchedPiecesText = '';
        if (Array.isArray(metadata.matched_pieces)) {
            matchedPiecesText = metadata.matched_pieces.join(', ');
        } else {
            matchedPiecesText = metadata.matched_pieces.toString();
        }

        if (matchedPiecesText.trim()) {
            const mappingDiv = document.createElement('div');
            mappingDiv.className = 'game-piece-mapping';
            mappingDiv.innerHTML = `
                <div class="mapping-header">
                    <i class="fas fa-link"></i> Game Piece Mapping
                </div>
                <div class="mapping-content">
                    Detected: <strong>${matchedPiecesText}</strong>
                    ${metadata.enhanced_query !== metadata.original_query ?
                    `<br><small>Enhanced search: "${metadata.enhanced_query}"</small>` : ''}
                </div>
            `;
            textContentDiv.appendChild(mappingDiv);
        }
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = '<span class="streaming-cursor">|</span>';

    textContentDiv.appendChild(textDiv);
    contentDiv.appendChild(avatarDiv);
    contentDiv.appendChild(textContentDiv);
    messageDiv.appendChild(contentDiv);

    elements.chatMessages.appendChild(messageDiv);
    autoScrollToBottom();

    return messageDiv;
}

export function finalizeStreamingMessage(messageDiv, finalText, metadata, onRedo) {
    const textContentDiv = messageDiv.querySelector('.message-text-container');
    const textDiv = messageDiv.querySelector('.message-text');

    textDiv.innerHTML = formatText(finalText);

    // Add images if present
    if (metadata.images && metadata.images.length > 0) {
        const imagesDiv = document.createElement('div');
        imagesDiv.className = 'message-images';

        const imageHeader = document.createElement('div');
        imageHeader.className = 'image-header';
        imageHeader.innerHTML = `<i class="fas fa-images"></i> Related Images (${metadata.images.length})`;
        imagesDiv.appendChild(imageHeader);

        const imageGrid = document.createElement('div');
        imageGrid.className = 'image-grid';

        metadata.images.forEach((image, index) => {
            const imageItem = createImageItem(image, index);
            imageGrid.appendChild(imageItem);
        });

        imagesDiv.appendChild(imageGrid);
        textContentDiv.appendChild(imagesDiv);
    }

    // Add feedback buttons
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = 'feedback-buttons';

    const feedbackLabel = document.createElement('div');
    feedbackLabel.className = 'feedback-label';
    feedbackLabel.textContent = 'Was this response helpful?';

    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'feedback-button-container';

    const goodButton = document.createElement('button');
    goodButton.className = 'feedback-btn feedback-good';
    goodButton.innerHTML = '<i class="fas fa-thumbs-up"></i> Good';
    goodButton.onclick = () => handleFeedback(finalText, 'good', feedbackDiv, onRedo);

    const badButton = document.createElement('button');
    badButton.className = 'feedback-btn feedback-bad';
    badButton.innerHTML = '<i class="fas fa-thumbs-down"></i> Bad';
    badButton.onclick = () => handleFeedback(finalText, 'bad', feedbackDiv, onRedo);

    const redoButton = document.createElement('button');
    redoButton.className = 'feedback-btn feedback-redo';
    redoButton.innerHTML = '<i class="fas fa-redo"></i> Redo';
    redoButton.onclick = () => handleFeedback(finalText, 'redo', feedbackDiv, onRedo);

    buttonsContainer.appendChild(goodButton);
    buttonsContainer.appendChild(badButton);
    buttonsContainer.appendChild(redoButton);

    feedbackDiv.appendChild(feedbackLabel);
    feedbackDiv.appendChild(buttonsContainer);
    textContentDiv.appendChild(feedbackDiv);
    messageDiv.classList.add('latest-assistant-message');
    
    const previousLatestMessages = elements.chatMessages.querySelectorAll('.latest-assistant-message');
    previousLatestMessages.forEach(msg => {
        if (msg !== messageDiv) {
            msg.classList.remove('latest-assistant-message');
            const redoBtn = msg.querySelector('.feedback-redo');
            if (redoBtn) redoBtn.remove();
        }
    });

    autoScrollToBottom();
}
