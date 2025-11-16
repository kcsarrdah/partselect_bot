import React, { useState, useEffect, useRef } from "react";
import { getAIMessage } from "../api/api";
import { marked } from "marked";
import { themeColors, colors } from "../constants/colors";

// Configure marked to open links in new tabs
marked.use({
  renderer: {
    link(href, title, text) {
      // Handle case where href is a token object (marked.js v4+)
      let actualHref = href;
      let actualText = text;
      let actualTitle = title;
      
      if (typeof href === 'object' && href !== null) {
        // It's a token object, extract the values
        actualHref = href.href || href.url || href;
        actualText = href.text || text || '';
        actualTitle = href.title || title || null;
      }
      
      // Validate href - if it's undefined, null, or not a string, don't render the link
      if (!actualHref || typeof actualHref !== 'string' || actualHref === 'undefined' || actualHref === 'null') {
        // Just return the text without the link
        return actualText || '';
      }
      
      // Ensure href is properly formatted
      const safeHref = actualHref.trim();
      if (!safeHref.startsWith('http://') && !safeHref.startsWith('https://')) {
        // If it's not a valid URL, don't render as a link
        return actualText || '';
      }
      
      // Render link with target="_blank" to open in new tab
      return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer" title="${actualTitle || ''}">${actualText || ''}</a>`;
    }
  }
});

// Loading status messages - defined outside component since they never change
const LOADING_MESSAGES = [
  "Running around the warehouse...",
  "Finding the right aisle...",
  "Consulting our expert...",
  "Compiling all the info...",
  "Almost there..."
];

function ChatWindow({ messages, setMessages }) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("");

  const messagesEndRef = useRef(null);
  const loadingTimerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Cycle through loading messages
  useEffect(() => {
    if (isLoading) {
      let currentIndex = 0;
      setLoadingStatus(LOADING_MESSAGES[0]);

      // Change message every 1 second
      loadingTimerRef.current = setInterval(() => {
        currentIndex++;
        if (currentIndex < LOADING_MESSAGES.length) {
          setLoadingStatus(LOADING_MESSAGES[currentIndex]);
        } else {
          // Loop back to last message if taking too long
          setLoadingStatus(LOADING_MESSAGES[LOADING_MESSAGES.length - 1]);
        }
      }, 1000);

      // Cleanup on unmount or when loading stops
      return () => {
        if (loadingTimerRef.current) {
          clearInterval(loadingTimerRef.current);
        }
      };
    } else {
      setLoadingStatus("");
      if (loadingTimerRef.current) {
        clearInterval(loadingTimerRef.current);
      }
    }
  }, [isLoading]);

  const handleSend = async (input) => {
    if (input.trim() !== "" && !isLoading) {
      // Add user message
      setMessages(prevMessages => [...prevMessages, { role: "user", content: input }]);
      setInput("");
      setIsLoading(true);

      // Add loading indicator message
      setMessages(prevMessages => [
        ...prevMessages, 
        { role: "assistant", content: "", isLoading: true }
      ]);

      try {
        const newMessage = await getAIMessage(input);
        
        // Remove loading message and add actual response
        setMessages(prevMessages => {
          const withoutLoading = prevMessages.filter(msg => !msg.isLoading);
          return [...withoutLoading, newMessage];
        });
      } catch (error) {
        // Remove loading and show error
        setMessages(prevMessages => {
          const withoutLoading = prevMessages.filter(msg => !msg.isLoading);
          return [...withoutLoading, {
            role: "assistant",
            content: "Sorry, something went wrong. Please try again."
          }];
        });
      } finally {
        setIsLoading(false);
      }
    }
  };

  // Loading indicator component with animated dots
  const LoadingDots = () => (
    <div style={styles.loadingContainer}>
      <span style={styles.loadingDot}>●</span>
      <span style={{...styles.loadingDot, animationDelay: "0.2s"}}>●</span>
      <span style={{...styles.loadingDot, animationDelay: "0.4s"}}>●</span>
    </div>
  );

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
      opacity: isLoading ? 0.6 : 1,
      cursor: isLoading ? "not-allowed" : "text",
    },
    button: {
      padding: "10px 20px",
      border: "none",
      borderRadius: "5px",
      backgroundColor: isLoading ? themeColors.border : themeColors.buttonPrimary,
      color: colors.secondary.white,
      cursor: isLoading ? "not-allowed" : "pointer",
      fontSize: "13px",
      fontWeight: 600,
      opacity: isLoading ? 0.6 : 1,
    },
    loadingContainer: {
      display: "flex",
      gap: "4px",
      alignItems: "center",
      padding: "8px 0",
    },
    loadingDot: {
      fontSize: "20px",
      color: themeColors.buttonPrimary,
      animation: "bounce 1.4s infinite ease-in-out",
    },
    loadingStatusText: {
      fontSize: "12px",
      color: themeColors.textSecondary,
      marginLeft: "8px",
      fontWeight: 500,
      animation: "fadeIn 0.3s ease-in",
    },
    loadingProgressBar: {
      position: "fixed",
      top: "60px",
      left: 0,
      width: "100%",
      height: "3px",
      backgroundColor: colors.secondary.lightGray,
      zIndex: 1000,
    },
    loadingProgressFill: {
      height: "100%",
      backgroundColor: themeColors.buttonPrimary,
      animation: "progressBar 5s ease-out forwards",
      width: "0%",
    },
  };

  return (
    <>
      {/* CSS animations */}
      <style>
        {`
          @keyframes bounce {
            0%, 80%, 100% { 
              transform: scale(0.8);
              opacity: 0.5;
            }
            40% { 
              transform: scale(1);
              opacity: 1;
            }
          }
          
          @keyframes fadeIn {
            from {
              opacity: 0;
              transform: translateY(-5px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
          
          @keyframes progressBar {
            0% { width: 0%; }
            20% { width: 30%; }
            40% { width: 50%; }
            60% { width: 70%; }
            80% { width: 85%; }
            100% { width: 95%; }
          }
        `}
      </style>
      
      {/* Progress bar at top */}
      {isLoading && (
        <div style={styles.loadingProgressBar}>
          <div style={styles.loadingProgressFill} />
        </div>
      )}
      
      <div style={styles.messagesContainer}>
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              ...styles.messageContainer,
              ...(message.role === "user" ? styles.userMessageContainer : styles.assistantMessageContainer),
            }}
          >
            {message.isLoading ? (
              // Show loading indicator with dynamic status
              <div style={{...styles.message, ...styles.assistantMessage}}>
                <div style={{display: "flex", alignItems: "center"}}>
                  <LoadingDots />
                  <span style={styles.loadingStatusText} key={loadingStatus}>
                    {loadingStatus}
                  </span>
                </div>
              </div>
            ) : message.content && (
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
            placeholder={isLoading ? "Waiting for response..." : "Type a message..."}
            disabled={isLoading}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !isLoading) {
                handleSend(input);
                e.preventDefault();
              }
            }}
          />
          <button 
            style={styles.button} 
            onClick={() => handleSend(input)}
            disabled={isLoading}
          >
            {isLoading ? "Thinking..." : "Send"}
          </button>
        </div>
      </div>
    </>
  );
}

export default ChatWindow;