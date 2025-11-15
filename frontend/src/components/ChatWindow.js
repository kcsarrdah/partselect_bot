import React, { useState, useEffect, useRef } from "react";
import { getAIMessage } from "../api/api";
import { marked } from "marked";
import { themeColors, colors } from "../constants/colors";

function ChatWindow() {
  const defaultMessage = [{
    role: "assistant",
    content: "Hi, how can I help you today?"
  }];

  const [messages, setMessages] = useState(defaultMessage);
  const [input, setInput] = useState("");

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (input) => {
    if (input.trim() !== "") {
      setMessages(prevMessages => [...prevMessages, { role: "user", content: input }]);
      setInput("");

      const newMessage = await getAIMessage(input);
      setMessages(prevMessages => [...prevMessages, newMessage]);
    }
  };

  // Styles
  const styles = {
    messagesContainer: {
      flex: 1,
      overflowY: "auto",
      padding: "20px",
      boxSizing: "border-box",
      display: "flex",
      flexDirection: "column",
      paddingBottom: "2px",
      fontSize: "16px",
      marginTop: "60px",
      marginBottom: "70px",
      backgroundColor: themeColors.bgPage,
    },
    messageContainer: {
      display: "flex",
      flexDirection: "column",
      maxWidth: "100%",
      margin: "4px 0",
    },
    userMessageContainer: {
      alignSelf: "flex-end",
      alignItems: "flex-end",
    },
    assistantMessageContainer: {
      alignItems: "flex-start",
    },
    message: {
      whiteSpace: "pre-line",
      padding: "14px",
      margin: "2px 0",
      borderRadius: "10px",
      fontSize: "13px",
      fontWeight: 400,
      lineHeight: 1.4,
      textAlign: "left",
    },
    userMessage: {
      alignSelf: "flex-end",
      backgroundColor: themeColors.userMessage,
      color: colors.secondary.white,
      borderTopRightRadius: 0,
    },
    assistantMessage: {
      alignSelf: "flex-start",
      backgroundColor: themeColors.assistantMessage,
      borderTopLeftRadius: 0,
      color: themeColors.textPrimary,
      width: "100%",
      boxSizing: "border-box",
    },
    inputArea: {
      fontSize: "15px",
      padding: "10px",
      bottom: 0,
      width: "calc(100% - 40px)",
      display: "flex",
      borderTop: `1px solid ${themeColors.border}`,
      background: colors.secondary.white,
      position: "fixed",
    },
    input: {
      flex: 1,
      padding: "10px",
      marginRight: "10px",
      borderRadius: "5px",
      border: `1px solid ${themeColors.border}`,
      fontSize: "13px",
      outline: "none",
    },
    button: {
      padding: "10px 20px",
      border: "none",
      borderRadius: "5px",
      backgroundColor: themeColors.buttonPrimary,
      color: colors.secondary.white,
      cursor: "pointer",
      fontSize: "13px",
      fontWeight: 600,
    },
  };

  return (
    <div style={styles.messagesContainer}>
      {messages.map((message, index) => (
        <div
          key={index}
          style={{
            ...styles.messageContainer,
            ...(message.role === "user" ? styles.userMessageContainer : styles.assistantMessageContainer),
          }}
        >
          {message.content && (
            <div
              style={{
                ...styles.message,
                ...(message.role === "user" ? styles.userMessage : styles.assistantMessage),
              }}
            >
              <div dangerouslySetInnerHTML={{ __html: marked(message.content).replace(/<p>|<\/p>/g, "") }} />
            </div>
          )}
        </div>
      ))}
      <div ref={messagesEndRef} />
      <div style={styles.inputArea}>
        <input
          style={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              handleSend(input);
              e.preventDefault();
            }
          }}
        />
        <button style={styles.button} onClick={() => handleSend(input)}>
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatWindow;