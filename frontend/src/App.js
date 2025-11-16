import "./App.css";
import ChatWindow from "./components/ChatWindow";
import { useState, useEffect } from "react";

function App() {
  const defaultMessage = [{
    role: "assistant",
    content: `👋 **Welcome to PartSelect Assistant!**

I'm here to help you find the right parts for your **refrigerator** or **dishwasher**. I can help you with:

🔧 **Finding parts** - Just ask about any part or provide a part number (like PS123456)
💰 **Checking prices & availability** - Want to know if something's in stock?
🛠️ **Troubleshooting** - Having issues? Describe the problem and I'll help diagnose it
📹 **Installation help** - Need guidance on how to install a part? I've got you covered!

What can I help you with today?`
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