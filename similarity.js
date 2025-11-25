/**
 * Similarity Calculation Module
 * 
 * Implements dual similarity approach:
 * 1. Content-based similarity: query text → item metadata comparison
 * 2. Collaborative filtering similarity: user tower → item tower comparison
 */

/**
 * Compute cosine similarity between two vectors
 * @param {Array} vec1 - First vector
 * @param {Array} vec2 - Second vector
 * @returns {number} Cosine similarity [-1, 1]
 */
function cosineSimilarity(vec1, vec2) {
    if (vec1.length !== vec2.length) {
        throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) {
        return 0;
    }

    return dotProduct / (norm1 * norm2);
}

/**
 * Create a simple text embedding using TF-IDF style approach
 * @param {string} text - Input text
 * @param {Object} vocabulary - Global vocabulary with word frequencies
 * @param {number} dim - Embedding dimension
 * @returns {Array} Text embedding vector
 */
function simpleTextEmbedding(text, vocabulary, dim = 32) {
    const words = text.toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(w => w.length > 2);

    // Initialize embedding vector
    const embedding = new Array(dim).fill(0);

    // Simple hash-based embedding
    words.forEach(word => {
        const hash = hashString(word);
        const idx = Math.abs(hash) % dim;

        // Increment the corresponding dimension
        embedding[idx] += 1;

        // Also increment neighboring dimensions for better representation
        embedding[(idx + 1) % dim] += 0.5;
        embedding[(idx - 1 + dim) % dim] += 0.5;
    });

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
        return embedding.map(val => val / norm);
    }

    return embedding;
}

/**
 * Simple string hash function
 * @param {string} str - String to hash
 * @returns {number} Hash value
 */
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
}

/**
 * Extract keywords from query text
 * @param {string} query - User query
 * @returns {Object} Extracted features {keywords, genres}
 */
function extractQueryFeatures(query) {
    const lowerQuery = query.toLowerCase();

    // Common movie genres
    const genreMap = {
        'action': ['action', 'fight', 'battle', 'combat'],
        'adventure': ['adventure', 'quest', 'journey'],
        'animation': ['animation', 'animated', 'cartoon'],
        'comedy': ['comedy', 'funny', 'humor', 'laugh'],
        'crime': ['crime', 'detective', 'murder', 'heist'],
        'documentary': ['documentary', 'real', 'true story'],
        'drama': ['drama', 'emotional', 'serious'],
        'fantasy': ['fantasy', 'magic', 'wizard', 'mythical'],
        'horror': ['horror', 'scary', 'terror', 'fear'],
        'mystery': ['mystery', 'puzzle', 'secret'],
        'romance': ['romance', 'romantic', 'love', 'relationship'],
        'sci-fi': ['sci-fi', 'science fiction', 'space', 'future', 'robot', 'alien'],
        'thriller': ['thriller', 'suspense', 'tension'],
        'war': ['war', 'military', 'soldier', 'battle'],
        'western': ['western', 'cowboy', 'frontier']
    };

    // Detect genres
    const detectedGenres = [];
    for (const [genre, keywords] of Object.entries(genreMap)) {
        if (keywords.some(keyword => lowerQuery.includes(keyword))) {
            detectedGenres.push(genre);
        }
    }

    // Extract keywords (simple tokenization)
    const stopwords = new Set(['a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'about', 'like', 'movies', 'movie', 'film', 'films']);
    const keywords = lowerQuery
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(word => word.length > 2 && !stopwords.has(word));

    return {
        keywords: [...new Set(keywords)],
        genres: detectedGenres
    };
}

/**
 * Compute content-based similarity between query and items
 * @param {string} query - User query text
 * @param {Array} items - Array of items with metadata
 * @param {number} dim - Embedding dimension
 * @returns {Array} Array of {itemIdx, score} sorted by score
 */
function computeContentSimilarity(query, items, dim = 32) {
    // Create query embedding
    const queryEmb = simpleTextEmbedding(query, null, dim);

    // Compute similarity for each item
    const results = [];

    items.forEach((item, idx) => {
        // Create item text representation (title + genres)
        const itemText = `${item.title} ${item.genres ? item.genres.join(' ') : ''}`;
        const itemEmb = simpleTextEmbedding(itemText, null, dim);

        // Compute cosine similarity
        const similarity = cosineSimilarity(queryEmb, itemEmb);

        results.push({
            itemIdx: idx,
            score: similarity
        });
    });

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    return results;
}

/**
 * Create a synthetic user profile from query
 * Useful for collaborative filtering when we don't have a real user
 * 
 * @param {string} query - User query
 * @param {Array} items - Array of items
 * @param {number} embDim - Embedding dimension
 * @returns {Array} Synthetic user embedding
 */
function createSyntheticUserProfile(query, items, embDim = 32) {
    const features = extractQueryFeatures(query);

    // Find items that match query features
    const matchingItems = [];

    items.forEach((item, idx) => {
        let matchScore = 0;

        // Check genre match
        if (item.genres && features.genres.length > 0) {
            const itemGenres = item.genres.map(g => g.toLowerCase());
            const genreMatches = features.genres.filter(g =>
                itemGenres.some(ig => ig.includes(g) || g.includes(ig))
            );
            matchScore += genreMatches.length * 2;
        }

        // Check keyword match in title
        const titleLower = item.title.toLowerCase();
        const keywordMatches = features.keywords.filter(kw =>
            titleLower.includes(kw)
        );
        matchScore += keywordMatches.length;

        if (matchScore > 0) {
            matchingItems.push({ idx, score: matchScore });
        }
    });

    // If we found matching items, create a profile based on their embeddings
    // Otherwise, return a random profile
    if (matchingItems.length === 0) {
        // Random profile
        return Array(embDim).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    }

    // For now, return a simple weighted representation
    // In practice, this would use actual learned embeddings
    const profile = new Array(embDim).fill(0);
    matchingItems.forEach(item => {
        const hash = hashString(items[item.idx].title);
        const idx = Math.abs(hash) % embDim;
        profile[idx] += item.score;
    });

    // Normalize
    const norm = Math.sqrt(profile.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
        return profile.map(val => val / norm);
    }

    return profile;
}

/**
 * Combine content-based and collaborative filtering scores
 * @param {Array} contentScores - Content-based scores [{itemIdx, score}]
 * @param {Array} collabScores - Collaborative filtering scores [{itemIdx, score}]
 * @param {number} alpha - Weight for content scores (0-1)
 * @returns {Array} Combined scores [{itemIdx, score, contentScore, collabScore}]
 */
function combineScores(contentScores, collabScores, alpha = 0.4) {
    const beta = 1 - alpha;

    // Create a map for quick lookup
    const collabMap = new Map();
    collabScores.forEach(item => {
        collabMap.set(item.itemIdx, item.score);
    });

    const combined = [];

    contentScores.forEach(item => {
        const collabScore = collabMap.get(item.itemIdx) || 0;
        const combinedScore = alpha * item.score + beta * collabScore;

        combined.push({
            itemIdx: item.itemIdx,
            score: combinedScore,
            contentScore: item.score,
            collabScore: collabScore
        });
    });

    // Sort by combined score
    combined.sort((a, b) => b.score - a.score);

    return combined;
}

/**
 * Normalize scores to [0, 1] range
 * @param {Array} scores - Array of score objects
 * @param {string} scoreKey - Key of the score field
 * @returns {Array} Scores with normalized values
 */
function normalizeScores(scores, scoreKey = 'score') {
    if (scores.length === 0) return scores;

    const values = scores.map(s => s[scoreKey]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    if (range === 0) {
        return scores.map(s => ({ ...s, [`${scoreKey}Normalized`]: 0.5 }));
    }

    return scores.map(s => ({
        ...s,
        [`${scoreKey}Normalized`]: (s[scoreKey] - min) / range
    }));
}
