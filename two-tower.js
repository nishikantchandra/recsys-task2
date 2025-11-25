/**
 * Two-Tower Recommendation Model
 * 
 * Architecture:
 * - User Tower: user_id → embedding
 * - Item Tower: item_id → embedding
 * - Scoring: dot product (or cosine similarity)
 * - Loss: in-batch sampled softmax (contrastive learning)
 */

class TwoTowerModel {
    /**
     * Initialize the Two-Tower model
     * @param {number} numUsers - Total number of users
     * @param {number} numItems - Total number of items
     * @param {number} embDim - Embedding dimension (default: 32)
     */
    constructor(numUsers, numItems, embDim = 32) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;

        // Initialize embedding tables with small random values
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.05),
            true,
            'userEmbedding'
        );

        this.itemEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.05),
            true,
            'itemEmbedding'
        );

        // L2 normalization for embeddings (important for cosine similarity)
        this.normalizeEmbeddings = true;
    }

    /**
     * Get user embedding for given user indices
     * @param {tf.Tensor} userIdxTensor - Tensor of user indices
     * @returns {tf.Tensor} User embeddings
     */
    userForward(userIdxTensor) {
        return tf.tidy(() => {
            let embeddings = tf.gather(this.userEmbedding, userIdxTensor);
            if (this.normalizeEmbeddings) {
                const norm = tf.norm(embeddings, 'euclidean', -1, true);
                embeddings = tf.div(embeddings, norm);
            }
            return embeddings;
        });
    }

    /**
     * Get item embedding for given item indices
     * @param {tf.Tensor} itemIdxTensor - Tensor of item indices
     * @returns {tf.Tensor} Item embeddings
     */
    itemForward(itemIdxTensor) {
        return tf.tidy(() => {
            let embeddings = tf.gather(this.itemEmbedding, itemIdxTensor);
            if (this.normalizeEmbeddings) {
                const norm = tf.norm(embeddings, 'euclidean', -1, true);
                embeddings = tf.div(embeddings, norm);
            }
            return embeddings;
        });
    }

    /**
     * Compute dot product score between user and item embeddings
     * @param {tf.Tensor} uEmb - User embeddings [batch, dim]
     * @param {tf.Tensor} iEmb - Item embeddings [batch, dim]
     * @returns {tf.Tensor} Scores [batch]
     */
    score(uEmb, iEmb) {
        return tf.tidy(() => {
            // Sum of element-wise product along last dimension
            return tf.sum(tf.mul(uEmb, iEmb), -1);
        });
    }

    /**
     * Compute in-batch sampled softmax loss
     * Uses all items in the batch as negatives for each user
     * 
     * @param {tf.Tensor} userIdx - User indices [batch]
     * @param {tf.Tensor} posItemIdx - Positive item indices [batch]
     * @returns {tf.Tensor} Scalar loss value
     */
    computeLoss(userIdx, posItemIdx) {
        return tf.tidy(() => {
            // Get embeddings
            const userEmb = this.userForward(userIdx);  // [batch, dim]
            const itemEmb = this.itemForward(posItemIdx);  // [batch, dim]

            // Compute logits matrix: userEmb @ itemEmb^T
            // This gives scores for all user-item pairs in the batch
            const logits = tf.matMul(userEmb, itemEmb, false, true);  // [batch, batch]

            // Labels: diagonal elements are positives (user i paired with item i)
            const batchSize = userIdx.shape[0];
            const labels = tf.range(0, batchSize, 1, 'int32');

            // Softmax cross-entropy loss
            const loss = tf.losses.softmaxCrossEntropy(
                tf.oneHot(labels, batchSize),
                logits
            );

            return loss;
        });
    }

    /**
     * Get a single user's embedding
     * @param {number} userIdx - User index
     * @returns {Promise<Array>} User embedding as array
     */
    async getUserEmbedding(userIdx) {
        return tf.tidy(() => {
            const embedding = this.userForward(tf.tensor1d([userIdx], 'int32'));
            return embedding.squeeze().arraySync();
        });
    }

    /**
     * Get a single item's embedding
     * @param {number} itemIdx - Item index
     * @returns {Promise<Array>} Item embedding as array
     */
    async getItemEmbedding(itemIdx) {
        return tf.tidy(() => {
            const embedding = this.itemForward(tf.tensor1d([itemIdx], 'int32'));
            return embedding.squeeze().arraySync();
        });
    }

    /**
     * Get all item embeddings as a matrix
     * @returns {Promise<Array>} All item embeddings [numItems, embDim]
     */
    async getAllItemEmbeddings() {
        return tf.tidy(() => {
            const allIndices = tf.range(0, this.numItems, 1, 'int32');
            const embeddings = this.itemForward(allIndices);
            return embeddings.arraySync();
        });
    }

    /**
     * Compute scores for a user against all items
     * @param {Array} userEmbArray - User embedding as array
     * @param {Array} excludeItems - Item indices to exclude (optional)
     * @returns {Promise<Array>} Array of {itemIdx, score} objects, sorted by score
     */
    async scoreAllItems(userEmbArray, excludeItems = []) {
        return tf.tidy(() => {
            const userEmb = tf.tensor2d([userEmbArray]);  // [1, dim]

            // Get all item embeddings
            const allItemIdx = tf.range(0, this.numItems, 1, 'int32');
            const allItemEmb = this.itemForward(allItemIdx);  // [numItems, dim]

            // Compute scores: userEmb @ allItemEmb^T
            const scores = tf.matMul(userEmb, allItemEmb, false, true).squeeze();  // [numItems]

            const scoresArray = scores.arraySync();

            // Create array of {itemIdx, score}
            let results = [];
            for (let i = 0; i < this.numItems; i++) {
                if (!excludeItems.includes(i)) {
                    results.push({ itemIdx: i, score: scoresArray[i] });
                }
            }

            // Sort by score descending
            results.sort((a, b) => b.score - a.score);

            return results;
        });
    }

    /**
     * Compute cosine similarity between two vectors
     * @param {Array} vec1 - First vector
     * @param {Array} vec2 - Second vector
     * @returns {number} Cosine similarity score
     */
    static cosineSimilarity(vec1, vec2) {
        return tf.tidy(() => {
            const t1 = tf.tensor1d(vec1);
            const t2 = tf.tensor1d(vec2);

            // Normalize vectors
            const norm1 = tf.norm(t1);
            const norm2 = tf.norm(t2);
            const n1 = tf.div(t1, norm1);
            const n2 = tf.div(t2, norm2);

            // Dot product
            const similarity = tf.sum(tf.mul(n1, n2)).arraySync();
            return similarity;
        });
    }

    /**
     * Dispose model resources
     */
    dispose() {
        this.userEmbedding.dispose();
        this.itemEmbedding.dispose();
    }
}

/**
 * Train the Two-Tower model
 * @param {TwoTowerModel} model - The model to train
 * @param {Array} interactions - Training data [{userIdx, itemIdx}]
 * @param {Object} config - Training configuration
 * @returns {Promise<Array>} Training loss history
 */
async function trainTwoTowerModel(model, interactions, config = {}) {
    const {
        epochs = 10,
        batchSize = 256,
        learningRate = 0.01,
        onEpochEnd = null,
        onBatchEnd = null
    } = config;

    // Create optimizer
    const optimizer = tf.train.adam(learningRate);

    // Shuffle interactions
    const shuffled = [...interactions];

    const lossHistory = [];
    const numBatches = Math.ceil(shuffled.length / batchSize);

    for (let epoch = 0; epoch < epochs; epoch++) {
        // Shuffle data each epoch
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }

        let epochLoss = 0;

        for (let b = 0; b < numBatches; b++) {
            const start = b * batchSize;
            const end = Math.min(start + batchSize, shuffled.length);
            const batch = shuffled.slice(start, end);

            // Prepare batch tensors
            const userIndices = batch.map(item => item.userIdx);
            const itemIndices = batch.map(item => item.itemIdx);

            // Compute loss and gradients
            const loss = optimizer.minimize(() => {
                const userIdx = tf.tensor1d(userIndices, 'int32');
                const itemIdx = tf.tensor1d(itemIndices, 'int32');
                const batchLoss = model.computeLoss(userIdx, itemIdx);

                // Clean up tensors
                userIdx.dispose();
                itemIdx.dispose();

                return batchLoss;
            }, true);

            const lossValue = await loss.data();
            epochLoss += lossValue[0];
            loss.dispose();

            if (onBatchEnd) {
                onBatchEnd(epoch, b, lossValue[0]);
            }
        }

        const avgLoss = epochLoss / numBatches;
        lossHistory.push(avgLoss);

        if (onEpochEnd) {
            onEpochEnd(epoch, avgLoss);
        }
    }

    return lossHistory;
}
