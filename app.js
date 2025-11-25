/**
 * Main Application Logic
 * Handles data loading, training, and recommendation generation
 */

// Global state
let model = null;
let interactions = [];
let items = [];
let users = new Set();
let userIdxMap = new Map();  // userId -> index
let itemIdxMap = new Map();  // itemId -> index
let idxToUserId = new Map();  // index -> userId
let idxToItemId = new Map();  // index -> itemId
let lossHistory = [];
let isDataLoaded = false;
let isModelTrained = false;

/**
 * Load MovieLens data files
 */
async function loadData() {
    const statusEl = document.getElementById('status');
    const loadBtn = document.getElementById('load-btn');

    try {
        loadBtn.disabled = true;
        statusEl.textContent = 'Loading data files...';
        statusEl.className = 'status-message loading';

        // Load u.data (ratings)
        const ratingsResponse = await fetch('data/u.data');
        if (!ratingsResponse.ok) throw new Error('Failed to load u.data');
        const ratingsText = await ratingsResponse.text();

        // Load u.item (movies)
        const itemsResponse = await fetch('data/u.item');
        if (!itemsResponse.ok) throw new Error('Failed to load u.item');
        const itemsText = await itemsResponse.text();

        // Parse ratings (tab separated: userId, movieId, rating, timestamp)
        const ratingsLines = ratingsText.trim().split('\n');
        const rawInteractions = ratingsLines.map(line => {
            const [userId, itemId, rating, timestamp] = line.split('\t').map(Number);
            return { userId, itemId, rating, timestamp };
        });

        // Parse items (pipe separated)
        const itemsLines = itemsText.trim().split('\n');
        const genres = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime',
            'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery',
            'romance', 'sci-fi', 'thriller', 'war', 'western'];

        items = itemsLines.map(line => {
            const parts = line.split('|');
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            const releaseDate = parts[2];

            // Extract genres (binary flags start at index 5)
            const itemGenres = [];
            for (let i = 0; i < genres.length; i++) {
                if (parts[5 + i] === '1') {
                    itemGenres.push(genres[i]);
                }
            }

            return {
                itemId,
                title,
                releaseDate,
                genres: itemGenres
            };
        });

        // Create index mappings
        const uniqueUserIds = [...new Set(rawInteractions.map(int => int.userId))].sort((a, b) => a - b);
        const uniqueItemIds = [...new Set(items.map(item => item.itemId))].sort((a, b) => a - b);

        uniqueUserIds.forEach((userId, idx) => {
            userIdxMap.set(userId, idx);
            idxToUserId.set(idx, userId);
        });

        uniqueItemIds.forEach((itemId, idx) => {
            itemIdxMap.set(itemId, idx);
            idxToItemId.set(idx, itemId);
        });

        // Convert to indexed interactions
        interactions = rawInteractions
            .filter(int => int.rating >= 4)  // Only use positive ratings (4 and 5)
            .map(int => ({
                userIdx: userIdxMap.get(int.userId),
                itemIdx: itemIdxMap.get(int.itemId),
                rating: int.rating,
                timestamp: int.timestamp
            }));

        // Resort items array by index for easier lookup
        const sortedItems = new Array(uniqueItemIds.length);
        items.forEach(item => {
            const idx = itemIdxMap.get(item.itemId);
            sortedItems[idx] = item;
        });
        items = sortedItems;

        users = new Set(uniqueUserIds);

        statusEl.innerHTML = `✅ Data loaded successfully!<br>
            <small>${users.size} users, ${items.length} items, ${interactions.length} interactions (rating ≥ 4)</small>`;
        statusEl.className = 'status-message success';

        document.getElementById('train-btn').disabled = false;
        isDataLoaded = true;

    } catch (error) {
        console.error('Error loading data:', error);
        statusEl.textContent = `❌ Error loading data: ${error.message}`;
        statusEl.className = 'status-message error';
        loadBtn.disabled = false;
    }
}

/**
 * Train the Two-Tower model
 */
async function trainModel() {
    if (!isDataLoaded) {
        alert('Please load data first!');
        return;
    }

    const statusEl = document.getElementById('status');
    const trainBtn = document.getElementById('train-btn');

    try {
        trainBtn.disabled = true;
        statusEl.textContent = 'Initializing model...';
        statusEl.className = 'status-message loading';

        // Initialize model
        const embeddingDim = 32;
        model = new TwoTowerModel(users.size, items.length, embeddingDim);

        statusEl.textContent = 'Training model... (this may take a minute)';

        // Training configuration
        const config = {
            epochs: 15,
            batchSize: 256,
            learningRate: 0.01,
            onEpochEnd: (epoch, loss) => {
                statusEl.innerHTML = `Training... Epoch ${epoch + 1}/15<br>
                    <small>Loss: ${loss.toFixed(4)}</small>`;

                // Update loss chart
                drawLossChart(lossHistory);
            },
            onBatchEnd: null
        };

        // Train model
        lossHistory = await trainTwoTowerModel(model, interactions, config);

        // Draw final visualizations
        drawLossChart(lossHistory);
        await drawEmbeddingProjection();

        statusEl.innerHTML = `✅ Training complete!<br>
            <small>Final loss: ${lossHistory[lossHistory.length - 1].toFixed(4)}</small>`;
        statusEl.className = 'status-message success';

        // Enable query section
        document.getElementById('query-section').style.display = 'block';
        isModelTrained = true;

    } catch (error) {
        console.error('Error training model:', error);
        statusEl.textContent = `❌ Error training model: ${error.message}`;
        statusEl.className = 'status-message error';
        trainBtn.disabled = false;
    }
}

/**
 * Get recommendations based on user query
 */
async function getRecommendations() {
    if (!isModelTrained) {
        alert('Please train the model first!');
        return;
    }

    const query = document.getElementById('query-input').value.trim();
    const numRecs = parseInt(document.getElementById('num-recommendations').value);
    const contentWeight = parseFloat(document.getElementById('content-weight').value);

    if (!query) {
        alert('Please enter a search query!');
        return;
    }

    const statusEl = document.getElementById('status');
    const resultsSection = document.getElementById('results-section');
    const queryAnalysisEl = document.getElementById('query-analysis');
    const recommendationsEl = document.getElementById('recommendations');

    try {
        statusEl.textContent = 'Analyzing query...';
        statusEl.className = 'status-message loading';

        // Step 1: Analyze query (with Phi-3.5 or fallback)
        const analysis = await analyzeQueryWithPhi(query);

        // Display query analysis
        queryAnalysisEl.innerHTML = '<h4>📝 Query Analysis</h4>' + formatQueryAnalysis(analysis);
        resultsSection.style.display = 'block';

        statusEl.textContent = 'Computing recommendations...';

        // Step 2: Compute content-based similarity
        const contentScores = computeContentSimilarity(query, items, model.embDim);

        // Step 3: Create synthetic user profile and compute collaborative scores
        const syntheticUserEmb = createSyntheticUserProfile(query, items, model.embDim);
        const collabScores = await model.scoreAllItems(syntheticUserEmb, []);

        // Step 4: Combine scores
        const combinedScores = combineScores(contentScores, collabScores, contentWeight);

        // Step 5: Get top N recommendations
        const topN = combinedScores.slice(0, numRecs);

        // Display recommendations
        displayRecommendations(topN, contentWeight);

        statusEl.innerHTML = `✅ Found ${numRecs} recommendations!<br>
            <small>Content weight: ${contentWeight}, Collaborative weight: ${(1 - contentWeight).toFixed(1)}</small>`;
        statusEl.className = 'status-message success';

    } catch (error) {
        console.error('Error getting recommendations:', error);
        statusEl.textContent = `❌ Error: ${error.message}`;
        statusEl.className = 'status-message error';
    }
}

/**
 * Display recommendations in a table
 */
function displayRecommendations(recommendations, alpha) {
    const recommendationsEl = document.getElementById('recommendations');

    let html = '<h4>🎯 Top Recommendations</h4>';
    html += '<table class="recommendations-table">';
    html += '<thead><tr>';
    html += '<th>#</th>';
    html += '<th>Title</th>';
    html += '<th>Genres</th>';
    html += '<th>Combined Score</th>';
    html += `<th>Content (α=${alpha})</th>`;
    html += `<th>Collaborative (β=${(1 - alpha).toFixed(1)})</th>`;
    html += '</tr></thead><tbody>';

    recommendations.forEach((rec, idx) => {
        const item = items[rec.itemIdx];
        const scoreClass = rec.score > 0.7 ? 'score-high' : rec.score > 0.4 ? 'score-medium' : 'score-low';

        html += '<tr>';
        html += `<td>${idx + 1}</td>`;
        html += `<td><strong>${item.title}</strong></td>`;
        html += `<td>${item.genres.slice(0, 3).join(', ')}</td>`;
        html += `<td><span class="score-badge ${scoreClass}">${rec.score.toFixed(3)}</span></td>`;
        html += `<td>${rec.contentScore.toFixed(3)}</td>`;
        html += `<td>${rec.collabScore.toFixed(3)}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    recommendationsEl.innerHTML = html;
}

/**
 * Draw loss chart
 */
function drawLossChart(losses) {
    const canvas = document.getElementById('loss-chart');
    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (losses.length === 0) return;

    const padding = 40;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;

    // Find min/max
    const maxLoss = Math.max(...losses);
    const minLoss = Math.min(...losses);
    const range = maxLoss - minLoss;

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    // Draw labels
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    // Draw loss line
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    ctx.beginPath();

    losses.forEach((loss, i) => {
        const x = padding + (i / (losses.length - 1)) * width;
        const y = canvas.height - padding - ((loss - minLoss) / range) * height;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Draw points
    ctx.fillStyle = '#667eea';
    losses.forEach((loss, i) => {
        const x = padding + (i / (losses.length - 1)) * width;
        const y = canvas.height - padding - ((loss - minLoss) / range) * height;

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    });
}

/**
 * Draw item embeddings 2D projection using PCA
 */
async function drawEmbeddingProjection() {
    const canvas = document.getElementById('embedding-chart');
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!model) return;

    try {
        // Get all item embeddings
        const embeddings = await model.getAllItemEmbeddings();

        // Simple PCA projection to 2D
        const projected = simplePCA(embeddings, 2);

        // Find min/max for scaling
        const xValues = projected.map(p => p[0]);
        const yValues = projected.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);

        const padding = 40;
        const width = canvas.width - 2 * padding;
        const height = canvas.height - 2 * padding;

        // Draw axes
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, canvas.height / 2);
        ctx.lineTo(canvas.width - padding, canvas.height / 2);
        ctx.moveTo(canvas.width / 2, padding);
        ctx.lineTo(canvas.width / 2, canvas.height - padding);
        ctx.stroke();

        // Sample points for visualization (too many will be cluttered)
        const sampleSize = Math.min(200, projected.length);
        const step = Math.floor(projected.length / sampleSize);

        // Draw points
        ctx.fillStyle = 'rgba(102, 126, 234, 0.6)';
        for (let i = 0; i < projected.length; i += step) {
            const x = padding + ((projected[i][0] - xMin) / (xMax - xMin)) * width;
            const y = padding + ((projected[i][1] - yMin) / (yMax - yMin)) * height;

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        }

        // Add title
        ctx.fillStyle = '#333';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${sampleSize} items visualized (PCA)`, canvas.width / 2, 20);

    } catch (error) {
        console.error('Error drawing embeddings:', error);
    }
}

/**
 * Simple PCA implementation
 */
function simplePCA(data, numComponents = 2) {
    // Center the data
    const numSamples = data.length;
    const numFeatures = data[0].length;

    const means = new Array(numFeatures).fill(0);
    for (let i = 0; i < numSamples; i++) {
        for (let j = 0; j < numFeatures; j++) {
            means[j] += data[i][j];
        }
    }
    for (let j = 0; j < numFeatures; j++) {
        means[j] /= numSamples;
    }

    const centered = data.map(row =>
        row.map((val, j) => val - means[j])
    );

    // For simplicity, just use first 2 dimensions
    // A proper PCA would compute covariance matrix and eigenvectors
    return centered.map(row => [row[0], row[1]]);
}
