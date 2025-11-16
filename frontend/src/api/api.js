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

    // Start with the LLM's answer
    let content = data.answer;
    
    // Remove any "(undefined)" text the LLM might have added
    content = content.replace(/\(undefined\)/g, '');
    
    // Always append properly formatted sources at the end
    if (data.sources && data.sources.length > 0) {
      content += "\n\n**Sources:**\n";
      data.sources.forEach((source, index) => {
        if (source.type === "part") {
          const partName = source.part_name || "Unknown Part";
          const partId = source.part_id || "N/A";
          const price = source.price || "Price N/A";
          content += `\n${index + 1}. **${partName}** (${partId}) - ${price}`;
        } else if (source.type === "blog" && source.url && source.title) {
          // Only add blog sources if both URL and title exist
          content += `\n${index + 1}. [${source.title}](${source.url})`;
        } else if (source.type === "repair") {
          const symptom = source.symptom || "Repair Guide";
          const difficulty = source.difficulty || "N/A";
          content += `\n${index + 1}. **${symptom}** (${difficulty})`;
        }
      });
    }

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