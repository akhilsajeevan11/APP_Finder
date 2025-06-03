document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Linkify function to find URLs and wrap them in <a> tags
    function linkify(inputText) {
        // Regular expression to find URLs (http, https, ftp)
        const urlRegex = /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig;
        return inputText.replace(urlRegex, function(url) {
            // Check if the URL is already part of an anchor tag to avoid nested anchors
            // This is a simple check; more robust parsing might be needed for complex HTML.
            // However, given we control bot output, it should generally be plain text with URLs.
            // A more advanced check could involve DOM parsing if input text could be HTML.
            // For now, assuming bot output is primarily text with potential URLs.
            return '<a href="' + url + '" target="_blank">' + url + '</a>';
        });
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender + '-message');

        const p = document.createElement('p');
        if (sender === 'bot') {
            p.innerHTML = linkify(text); // Use linkify for bot messages
        } else {
            p.textContent = text; // Keep user messages as plain text
        }
        messageDiv.appendChild(p);

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom
    }

    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === '') {
            return;
        }

        addMessage(messageText, 'user');
        userInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                addMessage(errorData.response || `Error: ${response.status}`, 'bot');
                return;
            }

            const data = await response.json();
            addMessage(data.response, 'bot');

        } catch (error) {
            console.error('Error sending message:', error);
            addMessage('Sorry, something went wrong while connecting to the server.', 'bot');
        }
    }

    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
});
