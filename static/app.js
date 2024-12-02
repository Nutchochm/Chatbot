class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        };

        this.state = false;
        this.message = [];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const inputField = chatBox.querySelector('input');
        inputField.addEventListener("keyup", (event) => {
            if (event.key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatBox) {
        this.state = !this.state;
        if (this.state) {
            chatBox.classList.add('chatbox--active');
        } else {
            chatBox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatBox) {
        const textField = chatBox.querySelector('input');
        const text1 = textField.value.trim();
        if (!text1) {
            return;
        }

        const msg1 = { name: "User", message: text1 };
        this.message.push(msg1);

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then((response) => response.json())
            .then((data) => {
                const msg2 = { name: 'Sam', message: data.answer };
                this.message.push(msg2);
                this.updateChatText(chatBox);
                textField.value = ""; // Clear input on success
            })
            .catch((error) => {
                console.error('Error:', error);
                const msg2 = { name: 'Sam', message: "Something went wrong. Please try again!" };
                this.message.push(msg2);
                this.updateChatText(chatBox);
            });
    }

    updateChatText(chatBox) {
        const chatMessages = chatBox.querySelector('.chatbox__messages');
        chatMessages.innerHTML = this.message
            .slice()
            .reverse()
            .map((item) =>
                item.name === "Sam"
                    ? `<div class="messages__item messages__item--visitor">${item.message}</div>`
                    : `<div class="messages__item messages__item--operator">${item.message}</div>`
            )
            .join('');
    }
}

const chatbox = new Chatbox();
chatbox.display();
