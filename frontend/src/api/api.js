const API_BASE_URL = "http://localhost:8000";

export const getAIMessage = async (userQuery) => {
  try {
    // Call your FastAPI backend
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: userQuery,
        k: 5, // Retrieve 5 documents
      }),
    });

    // Check if request was successful
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    // Format response for ChatWindow
    // Convert backend response to the format ChatWindow expects
    let content = data.answer;

    // Optionally append sources
    if (data.sources && data.sources.length > 0) {
      content += "\n\n**Sources:**\n";
      data.sources.forEach((source, index) => {
        if (source.type === "part") {
          content += `\n${index + 1}. **${source.part_name}** (${source.part_id}) - ${source.price || "Price N/A"}`;
        } else if (source.type === "blog") {
          content += `\n${index + 1}. [${source.title}](${source.url})`;
        } else if (source.type === "repair") {
          content += `\n${index + 1}. **${source.symptom}** (${source.difficulty})`;
        }
      });
    }

    return {
      role: "assistant",
      content: content,
    };

  } catch (error) {
    console.error("Error calling backend:", error);
    
    // Return error message to user
    return {
      role: "assistant",
      content: `Sorry, I encountered an error: ${error.message}. Please make sure the backend is running on http://localhost:8000`,
    };
  }
};