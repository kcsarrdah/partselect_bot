const API_BASE_URL = "http://localhost:8000";

export const getAIMessage = async (userQuery) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: userQuery,
        k: 5,
      }),
    });

    if (!response.ok) {
      // Handle different HTTP error codes
      if (response.status === 404) {
        throw new Error("NOT_FOUND");
      } else if (response.status === 500) {
        throw new Error("SERVER_ERROR");
      } else if (response.status === 503) {
        throw new Error("SERVICE_UNAVAILABLE");
      } else {
        throw new Error(`API_ERROR_${response.status}`);
      }
    }

    const data = await response.json();

    // Check if response has an error status
    if (data.status_code && data.status_code !== 200) {
      throw new Error(data.message || "RESPONSE_ERROR");
    }

    // Use the answer directly - links are already embedded by backend
    let content = data.answer || data.message || "I couldn't generate a response. Please try again.";

    return {
      role: "assistant",
      content: content,
    };

  } catch (error) {
    console.error("Error calling backend:", error);
    
    // Handle different error types gracefully
    let errorMessage;
    
    if (error.message === "Failed to fetch" || error.name === "TypeError") {
      // Network error - backend is likely down
      errorMessage = `🔌Hmm, that didn't work... I must be getting old. Hopefully I'm wiser now though! Please try again in a few moments!`;
      
    } else if (error.message === "NOT_FOUND") {
      errorMessage = `🔍 **Not Found**

I couldn't find the information you're looking for. Try:
• Rephrasing your question
• Being more specific about the part or issue
• Including your appliance model number if you have it`;
      
    } else if (error.message === "SERVER_ERROR") {
      errorMessage = `I'm feeling a little under the weather. I might need to visit the shop and check whats under the hood! Please try again in a few moments!`;
      
    } else if (error.message === "SERVICE_UNAVAILABLE") {
      errorMessage = `🚧 **Service Temporarily Unavailable**

The service is currently being updated or is temporarily down. 

Please try again in a few moments!`;
      
    } else if (error.message === "RESPONSE_ERROR") {
      errorMessage = `🤔 **Response Error**

I received your question but couldn't process it properly. 

Try rephrasing your question or asking about something else!`;
      
    } else {
      // Generic fallback
      errorMessage = `😕 **Oops! Something went wrong**

I encountered an error: \`${error.message}\`

Please try:
• Checking that the backend is running
• Rephrasing your question
• Trying again in a moment`;
    }
    
    return {
      role: "assistant",
      content: errorMessage,
    };
  }
};