import "./App.css";
import ChatWindow from "./components/ChatWindow";
import { useState, useEffect } from "react";

function App() {
  const defaultMessage = [{
    role: "assistant",
    content: "Hi, how can I help you today?"
  }];

  // Lift messages state to App level
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem("partselect_chat_history");
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return defaultMessage;
      }
    }
    return defaultMessage;
  });

  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("partselect_chat_history", JSON.stringify(messages));
  }, [messages]);

  // Clear chat handler
  const handleClearChat = () => {
    setMessages(defaultMessage);
    localStorage.setItem("partselect_chat_history", JSON.stringify(defaultMessage));
  };

  return (
    <div className="App">
      <div className="heading">
        <img 
          src="/PartSelect Header Logo.svg" 
          alt="PartSelect Logo" 
          className="logo"
        />
        <span className="heading-text">Assistant</span>
        
        {/* Clear button in header */}
        {messages.length > 1 && (
          <button className="clear-button-header" onClick={handleClearChat}>
            🗑️ Clear Chat
          </button>
        )}
      </div>
      <ChatWindow messages={messages} setMessages={setMessages} />
    </div>
  );
}

export default App;