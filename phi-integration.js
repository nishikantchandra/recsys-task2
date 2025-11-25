/**
 * Phi-3.5-mini-instruct Integration Module
 * Uses HuggingFace Inference API for query processing
 */

/**
 * Call HuggingFace Inference API with Phi-3.5-mini-instruct
 * @param {string} apiToken - HuggingFace API token
 * @param {string} query - User query text
 * @returns {Promise<Object>} Query analysis result
/**
 * Phi-3.5-mini-instruct Integration Module
 * Uses HuggingFace Inference API for query processing
 */

// Hard-coded HuggingFace API token
const HUGGINGFACE_API_TOKEN = "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";

/**
 * Call HuggingFace Inference API with Phi-3.5-mini-instruct
 * @param {string} query - User query text
 * @returns {Promise<Object>} Query analysis result
 */
async function analyzeQueryWithPhi(query) {
    const apiToken = HUGGINGFACE_API_TOKEN;
    if (!apiToken || apiToken.trim() === '') {
        // Fallback to basic analysis if no API token
        return analyzeQueryBasic(query);
    }

    const model = "microsoft/Phi-3.5-mini-instruct";
    const apiUrl = `https://api-inference.huggingface.co/models/${model}`;

    // Create prompt for query analysis
    const systemPrompt = `You are a movie recommendation assistant. Analyze the user's query and extract relevant information.
Return ONLY a JSON object with this structure:
{
  "intent": "brief description of what user wants",
  "keywords": ["key", "words", "from", "query"],
  "genres": ["detected", "genres"],
  "themes": ["detected", "themes"]
}`;

    const userPrompt = `Query: "${query}"

Extract the intent, keywords, genres, and themes. Return ONLY the JSON object.`;

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                inputs: `${systemPrompt}\n\n${userPrompt}`,
                parameters: {
                    max_new_tokens: 256,
                    temperature: 0.3,
                    top_p: 0.9,
                    return_full_text: false
                }
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.warn('HuggingFace API error:', errorText);
            // Fallback to basic analysis
            return analyzeQueryBasic(query);
        }

        const result = await response.json();

        // Handle different response formats
        let generatedText = '';
        if (Array.isArray(result) && result.length > 0) {
            generatedText = result[0].generated_text || '';
        } else if (result.generated_text) {
            generatedText = result.generated_text;
        }

        // Try to extract JSON from response
        const jsonMatch = generatedText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const analysis = JSON.parse(jsonMatch[0]);
            return {
                intent: analysis.intent || query,
                keywords: analysis.keywords || [],
                genres: analysis.genres || [],
                themes: analysis.themes || [],
                source: 'phi-3.5'
            };
        }

        // Fallback if JSON parsing fails
        return analyzeQueryBasic(query);

    } catch (error) {
        console.warn('Error calling Phi-3.5:', error);
        // Fallback to basic analysis
        return analyzeQueryBasic(query);
    }
}

/**
 * Basic query analysis without LLM (fallback)
 * @param {string} query - User query text
 * @returns {Object} Query analysis result
 */
function analyzeQueryBasic(query) {
    const features = extractQueryFeatures(query);

    // Build intent description
    let intent = 'Find movies';
    if (features.genres.length > 0) {
        intent += ` in ${features.genres.join(', ')} genre(s)`;
    }
    if (features.keywords.length > 0) {
        intent += ` related to: ${features.keywords.slice(0, 3).join(', ')}`;
    }

    return {
        intent: intent,
        keywords: features.keywords,
        genres: features.genres,
        themes: [], // Basic analysis doesn't detect themes
        source: 'basic'
    };
}

/**
 * Format query analysis for display
 * @param {Object} analysis - Query analysis result
 * @returns {string} HTML formatted analysis
 */
function formatQueryAnalysis(analysis) {
    let html = '<div class="analysis-item"><strong>Intent:</strong> ' + analysis.intent + '</div>';

    if (analysis.keywords && analysis.keywords.length > 0) {
        html += '<div class="analysis-item"><strong>Keywords:</strong> ' + analysis.keywords.join(', ') + '</div>';
    }

    if (analysis.genres && analysis.genres.length > 0) {
        html += '<div class="analysis-item"><strong>Detected Genres:</strong> ' + analysis.genres.join(', ') + '</div>';
    }

    if (analysis.themes && analysis.themes.length > 0) {
        html += '<div class="analysis-item"><strong>Themes:</strong> ' + analysis.themes.join(', ') + '</div>';
    }

    html += '<div class="analysis-item"><small style="color:#666;">Analysis by: ' +
        (analysis.source === 'phi-3.5' ? 'Phi-3.5-mini-instruct' : 'Basic keyword extraction') +
        '</small></div>';

    return html;
}
