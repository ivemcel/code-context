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

    constructor(config: Qwen3EmbeddingConfig) {
        super();
        this.config = config;
        
        // Set dimension based on model
        this.updateDimensionForModel(config.model || 'text-embedding-v1');
    }

    private updateDimensionForModel(model: string): void {
        // Update dimensions based on model version
        if (model.includes('text-embedding-v1-mini')) {
            this.dimension = 1024;
        } else if (model.includes('text-embedding-v1-huge')) {
            this.dimension = 4096;
        } else {
            // Default model (text-embedding-v1)
            this.dimension = 1536;
        }
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processedText = this.preprocessText(text);
        const model = this.config.model || 'text-embedding-v1';
        const baseURL = this.config.baseURL || 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding';

        try {
            const response = await axios.post(
                baseURL,
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

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        const processedTexts = this.preprocessTexts(texts);
        const model = this.config.model || 'text-embedding-v1';
        const baseURL = this.config.baseURL || 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding';

        try {
            const response = await axios.post(
                baseURL,
                {
                    model: model,
                    input: {
                        texts: processedTexts
                    }
                },
                {
                    headers: {
                        'Authorization': `Bearer ${this.config.apiKey}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
    
            if (!response.data || !response.data.output || !response.data.output.embeddings) {
                console.error('Invalid API response:', JSON.stringify(response.data, null, 2));
                throw new Error('Qwen3 API returned invalid response');
            }
    
            return response.data.output.embeddings.map((embedding: any) => {
                // 确保向量是数组格式，并转换为数字数组
                let vector = embedding;
                
                // 如果不是数组，尝试转换
                if (!Array.isArray(embedding)) {
                    if (typeof embedding === 'string') {
                        // 如果是字符串，尝试将其解析为数组
                        try {
                            vector = JSON.parse(embedding);
                        } catch (e) {
                            // 如果解析失败，尝试将字符串分割为数组
                            vector = embedding.split(',').map(Number);
                        }
                    } else if (typeof embedding === 'object') {
                        // 如果是对象，尝试提取值
                        vector = Object.values(embedding);
                    }
                }
                
                // 确保所有元素都是数字
                const numericVector = vector.map((val: any) => {
                    if (typeof val === 'string') {
                        return parseFloat(val);
                    }
                    return val;
                });
    
                return {
                    vector: numericVector,
                    dimension: numericVector.length || this.dimension
                };
            });
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
            }
        };
    }
} 