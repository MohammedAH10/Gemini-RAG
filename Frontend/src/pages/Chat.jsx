import { useState } from 'react';
import { FiSend, FiLoader, FiPlus, FiMessageCircle } from 'react-icons/fi';
import styles from '../styles/Chat.module.css';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    // TODO: Wire up API endpoint
    console.log('Query:', input);
  };

  return (
    <div className={styles.chatPage}>
      <div className={styles.chatContainer}>
        {messages.length === 0 ? (
          <div className={styles.emptyState}>
            <FiMessageCircle className={styles.emptyIcon} />
            <h2>Start a Conversation</h2>
            <p>Ask anything about your uploaded documents</p>
          </div>
        ) : (
          <div className={styles.messages}>
            {messages.map((msg, idx) => (
              <div key={idx} className={`${styles.message} ${msg.role === 'user' ? styles.userMsg : styles.aiMsg}`}>
                <div className={styles.messageContent}>{msg.content}</div>
              </div>
            ))}
          </div>
        )}

        <div className={styles.inputArea}>
          <div className={styles.inputWrapper}>
            <input
              type="text"
              placeholder="Ask a question about your documents..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              disabled={isLoading}
            />
            <button
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
            >
              {isLoading ? <FiLoader className={styles.spin} /> : <FiSend />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
