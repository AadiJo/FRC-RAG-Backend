/**
 * Utility functions for the application.
 * I've gathered these helpers here to keep the main logic clean.
 */

// Helper to get full API URL
export function getApiUrl(path) {
    const baseUrl = (window.FRC_RAG_CONFIG && window.FRC_RAG_CONFIG.API_BASE_URL) || '';
    return `${baseUrl}${path}`;
}

export function getTeamDisplayName(imagePath) {
    if (!imagePath) return 'Unknown';
    const pathParts = imagePath.split('/');
    for (let part of pathParts) {
        const match = part.match(/^(\d+)-(\d{4})$/);
        if (match) {
            const teamNumber = match[1];
            return `FRC ${teamNumber}`;
        }
    }
    return imagePath.split('/').pop();
}

export function formatText(text) {
    try {
        // Assuming marked and DOMPurify are available globally via CDN
        if (window.marked) {
            window.marked.setOptions({ gfm: true, breaks: true, headerIds: false, mangle: false });
            const html = window.marked.parse(text || '');
            const sanitized = window.DOMPurify.sanitize(html);
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = sanitized;
            const tables = tempDiv.querySelectorAll('table');
            tables.forEach(table => {
                const wrapper = document.createElement('div');
                wrapper.className = 'table-wrapper';
                table.parentNode.insertBefore(wrapper, table);
                wrapper.appendChild(table);
            });

            return tempDiv.innerHTML;
        }
        return text;
    } catch (err) {
        console.error('Error formatting text:', err);
        const escaped = (text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
        return escaped;
    }
}
