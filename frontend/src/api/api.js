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
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    // ✅ Use the answer directly - links are already embedded by backend
    let content = data.answer;
    
    // ✅ REMOVED: No longer appending Sources section
    // The backend now embeds links directly in the response text

    return {
      role: "assistant",
      content: content,
    };

  } catch (error) {
    console.error("Error calling backend:", error);
    return {
      role: "assistant",
      content: `Sorry, I encountered an error: ${error.message}. Please make sure the backend is running on http://localhost:8000`,
    };
  }
};