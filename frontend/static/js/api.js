/**
 * API interaction layer.
 * I've separated all network requests here to isolate side effects.
 */
import { getApiUrl } from './utils.js';

// Helper to load images with headers (for ngrok support)
export async function loadImage(url, imgElement) {
    try {
        // Check if need to use fetch (e.g. for ngrok)
        const isNgrok = url.includes('ngrok');
        
        if (isNgrok) {
            const response = await fetch(url, {
                headers: {
                    'ngrok-skip-browser-warning': 'true'
                }
            });
            if (!response.ok) throw new Error('Network response was not ok');
            const blob = await response.blob();
            const objectUrl = URL.createObjectURL(blob);
            imgElement.src = objectUrl;
        } else {
            // Normal loading
            imgElement.src = url;
        }
    } catch (error) {
        console.error('Error loading image:', error);
        imgElement.alt = 'Failed to load image';
        // Fallback to direct src just in case it works or to show broken image
        imgElement.src = url;
    }
}

export function submitFeedback(query, responseText, feedbackType, onSuccess, onError) {
    fetch(getApiUrl('/api/feedback'), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: query,
            response: responseText,
            feedback_type: feedbackType
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (onSuccess) onSuccess(data);
            } else {
                if (onError) onError('Failed to submit feedback');
            }
        })
        .catch(error => {
            console.error('Feedback error:', error);
            if (onError) onError(error);
        });
}

export async function streamQuery(query, conversationHistory, options, callbacks) {
    const { onMetadata, onContent, onDone, onError } = callbacks;

    try {
        const response = await fetch(getApiUrl('/api/query/stream'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                conversation_history: conversationHistory,
                ...options
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                console.log('Stream complete');
                break;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));

                        if (data.type === 'metadata') {
                            if (onMetadata) onMetadata(data.data);
                        } else if (data.type === 'content') {
                            if (onContent) onContent(data.data);
                        } else if (data.type === 'done') {
                            if (onDone) onDone();
                        } else if (data.type === 'error') {
                            if (onError) onError(data.error);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e, line);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Detailed error:', error);
        if (onError) onError(error.message || error);
    }
}
