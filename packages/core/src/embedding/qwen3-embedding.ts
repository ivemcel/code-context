import axios from 'axios';
import { Embedding, EmbeddingVector } from './base-embedding';

export interface Qwen3EmbeddingConfig {
    model: string;
    apiKey: string;
    baseURL?: string;
}

export class Qwen3Embedding extends Embedding {
    private config: Qwen3EmbeddingConfig;
    private dimension: number = 1536; // Default dimension for qwen embedding models
    protected maxTokens: number = 8192; // Maximum tokens limit
    private MAX_BATCH_SIZE: number = 10; // Maximum batch size for Qwen3 API - reduced from 25 to 10 as per API limit

    constructor(config: Qwen3EmbeddingConfig) {
        super();
        this.config = config;
        
        // Set dimension based on model
        this.updateDimensionForModel(config.model || 'text-embedding-v4');
    }

    private updateDimensionForModel(model: string): void {
        // Update dimensions based on model version
        if (model.includes('text-embedding-v1-mini')) {
            this.dimension = 1024;
        } else if (model.includes('text-embedding-v1-huge')) {
            this.dimension = 4096;
        } else if (model.includes('text-embedding-v4-huge')) {
            this.dimension = 4096; // 为v4-huge版本设置4096维度
        } else if (model.includes('text-embedding-v4')) {
            this.dimension = 1024; // 修正为1024维度，之前错误地设置为1536
        } else {
            // Default model (text-embedding-v1)
            this.dimension = 1536;
        }
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processedText = this.preprocessText(text);
        const model = this.config.model || 'text-embedding-v4';
        const baseURL = this.config.baseURL || 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding';

        try {
            const response = await axios.post(
                `${baseURL}/text-embedding`,
                {
                    model: model,
                    input: {
                        texts: [processedText]
                    }
                },
                {
                    headers: {
                        'Authorization': `Bearer ${this.config.apiKey}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
    
            if (!response.data || !response.data.output || !response.data.output.embeddings || !response.data.output.embeddings[0]) {
                console.error('Invalid API response:', JSON.stringify(response.data, null, 2));
                throw new Error('Qwen3 API returned invalid response');
            }
    
            // 确保向量是数组格式，并转换为数字数组
            let embedding = response.data.output.embeddings[0];
            
            // 如果不是数组，尝试转换
            if (!Array.isArray(embedding)) {
                if (typeof embedding === 'string') {
                    // 如果是字符串，尝试将其解析为数组
                    try {
                        embedding = JSON.parse(embedding);
                    } catch (e) {
                        // 如果解析失败，尝试将字符串分割为数组
                        embedding = embedding.split(',').map(Number);
                    }
                } else if (typeof embedding === 'object') {
                    // 如果是对象，尝试提取值
                    embedding = Object.values(embedding);
                }
            }
            
            // 确保所有元素都是数字
            const vector = embedding.map((val: any) => {
                if (typeof val === 'string') {
                    return parseFloat(val);
                }
                return val;
            });
    
            return {
                vector: vector,
                dimension: vector.length || this.dimension
            };
        } catch (error) {
            console.error('Error calling Qwen3 API:', error);
            throw error;
        }
    }

    /**
     * Embed multiple texts in batch
     * @param texts Array of texts to embed
     * @returns Array of embedding vectors
     */
    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        if (texts.length === 0) {
            return [];
        }

        // 预处理所有文本
        const processedTexts = texts.map(text => this.preprocessText(text));
        
        console.log(`🔢 Qwen3Embedding: 处理 ${texts.length} 个文本的批量嵌入请求`);
        
        // 强制限制批量大小，使用类中定义的 MAX_BATCH_SIZE 常量
        const results: EmbeddingVector[] = [];
        
        // 分批处理
        for (let i = 0; i < processedTexts.length; i += this.MAX_BATCH_SIZE) {
            const batchTexts = processedTexts.slice(i, i + this.MAX_BATCH_SIZE);
            console.log(`  - 处理子批次 ${Math.floor(i / this.MAX_BATCH_SIZE) + 1}/${Math.ceil(processedTexts.length / this.MAX_BATCH_SIZE)}, 大小: ${batchTexts.length}`);
            
            try {
                const batchResults = await this._embedBatchInternal(batchTexts);
                results.push(...batchResults);
                console.log(`  ✓ 成功处理子批次 ${Math.floor(i / this.MAX_BATCH_SIZE) + 1}`);
            } catch (error) {
                console.error(`  ❌ 处理子批次 ${Math.floor(i / this.MAX_BATCH_SIZE) + 1} 失败:`, error);
                throw error; // 重新抛出错误，让调用者决定如何处理
            }
        }
        
        console.log(`✅ Qwen3Embedding: 批量嵌入请求完成，共处理 ${results.length}/${texts.length} 个文本`);
        return results;
    }
    
    /**
     * Internal method to embed a batch of texts (max 25 items)
     * @param texts Array of texts to embed (must be <= 25)
     * @returns Array of embedding vectors
     * @private
     */
    private async _embedBatchInternal(texts: string[]): Promise<EmbeddingVector[]> {
        if (texts.length === 0) {
            return [];
        }
        
        if (texts.length > this.MAX_BATCH_SIZE) {
            throw new Error(`批量大小不能超过 ${this.MAX_BATCH_SIZE} 个项目`);
        }

        try {
            const baseURL = this.config.baseURL || 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding';
            const response = await axios.post(
                `${baseURL}/text-embedding`,
                {
                    model: this.config.model,
                    input: {
                        texts: texts
                    }
                },
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.config.apiKey}`
                    }
                }
            );

            if (response.data && response.data.output && response.data.output.embeddings) {
                return response.data.output.embeddings.map((item: any) => {
                    // 处理嵌入向量，确保格式一致
                    let embedding = item;
                    
                    // 如果是对象并且有embedding字段，说明是嵌套的结构
                    if (typeof item === 'object' && item !== null && 'embedding' in item) {
                        embedding = item.embedding;
                    }
                    
                    // 确保向量是数组格式
                    if (!Array.isArray(embedding)) {
                        if (typeof embedding === 'string') {
                            try {
                                embedding = JSON.parse(embedding);
                            } catch (e) {
                                embedding = embedding.split(',').map(Number);
                            }
                        } else if (typeof embedding === 'object') {
                            embedding = Object.values(embedding);
                        }
                    }
                    
                    // 确保所有元素都是数字
                    const vector = embedding.map((val: any) => {
                        if (typeof val === 'string') {
                            return parseFloat(val);
                        }
                        return val;
                    });
                    
                    return {
                        vector: vector,
                        dimension: vector.length || this.dimension,
                        tokenCount: item.token_count || 0
                    };
                });
            } else {
                console.error('Invalid API response:', JSON.stringify(response.data, null, 2));
                throw new Error('Invalid response format from Qwen3 API');
            }
        } catch (error) {
            console.error('Error calling Qwen3 API:', error);
            throw error;
        }
    }

    getDimension(): number {
        return this.dimension;
    }

    getProvider(): string {
        return 'Qwen3';
    }

    /**
     * Set model type
     * @param model Model name
     */
    setModel(model: string): void {
        this.config.model = model;
        this.updateDimensionForModel(model);
    }

    /**
     * Get list of supported models
     */
    static getSupportedModels(): Record<string, { dimension: number; description: string }> {
        return {
            'text-embedding-v1': {
                dimension: 1536,
                description: 'Default Qwen embedding model with 1536 dimensions'
            },
            'text-embedding-v1-mini': {
                dimension: 1024,
                description: 'Qwen embedding model with 1024 dimensions'
            },
            'text-embedding-v1-huge': {
                dimension: 4096,
                description: 'Qwen embedding model with 4096 dimensions for high-precision tasks'
            },
            'text-embedding-v4': {
                dimension: 1024,
                description: 'Latest Qwen embedding model with 1024 dimensions'
            },
            'text-embedding-v4-huge': {
                dimension: 4096,
                description: 'High-dimensional Qwen v4 embedding model with 4096 dimensions for high-precision tasks'
            }
        };
    }
} 