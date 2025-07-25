import axios, { AxiosError } from 'axios';
import { Embedding, EmbeddingVector } from './base-embedding';

export interface StarFactoryEmbeddingConfig {
    apiKey: string;
    baseURL: string;
    model?: string;
}

export class StarFactoryEmbedding extends Embedding {
    private config: StarFactoryEmbeddingConfig;
    private dimension: number = 4096; // Default dimension for NV-Embed-v2 model
    protected maxTokens: number = 32768; // Maximum tokens supported according to docs

    constructor(config: StarFactoryEmbeddingConfig) {
        super();
        this.config = {
            ...config,
            model: config.model || 'NV-Embed-v2', // Default model
        };
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processedText = this.preprocessText(text);
        
        // The API requires input as an array of strings
        const embeddings = await this.embedBatch([processedText]);
        return embeddings[0];
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        const processedTexts = this.preprocessTexts(texts);
        
        try {
            // 注意：此API需要将输入作为字符串列表，而不是单个字符串
            const response = await axios.post(
                `${this.config.baseURL}/codegen/codebaseEmbedding`,
                {
                    input: processedTexts
                },
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'api-key': this.config.apiKey
                    }
                }
            );

            if (!response.data) {
                throw new Error('StarFactory Embedding API: Empty response');
            }

            // 对返回数据的代码进行处理，确保比较的是字符串
            const code = String(response.data.code);
            
            // 代码 200 和 10000 都视为成功
            if (code === '200' || code === '10000') {
                if (!response.data.data || !Array.isArray(response.data.data)) {
                    console.warn('StarFactory Embedding API: Response data is not an array');
                    return [];
                }
                
                // 确保每个项都有嵌入向量
                return response.data.data.map((item: any, index: number) => {
                    if (!item.embedding || !Array.isArray(item.embedding)) {
                        throw new Error(`StarFactory Embedding API: Item ${index} missing embedding vector`);
                    }
                    
                    return {
                        vector: item.embedding,
                        dimension: this.dimension
                    };
                });
            } else if (code === '507') {
                throw new Error('StarFactory Embedding API: Insufficient memory (code 507)');
            } else if (code === '508') {
                throw new Error('StarFactory Embedding API: Server busy, timeout exceeded (code 508)');
            } else {
                throw new Error(`StarFactory Embedding API: Error with code ${code}`);
            }
        } catch (error: unknown) {
            if (axios.isAxiosError(error)) {
                throw new Error(`StarFactory Embedding API request failed: ${error.message}`);
            }
            throw error;
        }
    }

    getDimension(): number {
        return this.dimension;
    }

    getProvider(): string {
        return 'StarFactory';
    }

    /**
     * Get token count for input texts
     * @param texts Array of input texts
     * @returns Promise with array of token counts
     */
    async getTokenCounts(texts: string[]): Promise<number[]> {
        try {
            const response = await axios.post(
                `${this.config.baseURL}/v1/token_count`,
                {
                    input: texts
                },
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'api-key': this.config.apiKey
                    }
                }
            );

            if (response.data && response.data.token_counts) {
                return response.data.token_counts;
            } else {
                throw new Error('StarFactory token count API: No token counts returned');
            }
        } catch (error: unknown) {
            if (axios.isAxiosError(error)) {
                throw new Error(`StarFactory token count API request failed: ${error.message}`);
            }
            throw error;
        }
    }

    /**
     * Set base URL for API
     * @param baseURL Base URL for API
     */
    setBaseURL(baseURL: string): void {
        this.config.baseURL = baseURL;
    }

    /**
     * Get information about batch processing capabilities based on token count
     * @returns Record of token counts to batch sizes and processing times
     */
    static getBatchProcessingInfo(): Record<number, { batchSize: number, processingTimeSec: number }> {
        return {
            32768: { batchSize: 2, processingTimeSec: 20.3 },
            30000: { batchSize: 2, processingTimeSec: 17.3 },
            25000: { batchSize: 3, processingTimeSec: 19.0 },
            20000: { batchSize: 4, processingTimeSec: 17.6 },
            15000: { batchSize: 4, processingTimeSec: 11.2 },
            10000: { batchSize: 7, processingTimeSec: 10.6 },
            7500: { batchSize: 11, processingTimeSec: 11.1 },
            5000: { batchSize: 15, processingTimeSec: 8.8 },
            4500: { batchSize: 17, processingTimeSec: 8.7 },
            4000: { batchSize: 20, processingTimeSec: 8.9 },
            3500: { batchSize: 23, processingTimeSec: 8.6 },
            3000: { batchSize: 27, processingTimeSec: 8.4 },
            2500: { batchSize: 32, processingTimeSec: 8.1 },
            2000: { batchSize: 39, processingTimeSec: 7.6 },
            1500: { batchSize: 48, processingTimeSec: 6.8 },
            1000: { batchSize: 77, processingTimeSec: 7.2 },
            750: { batchSize: 95, processingTimeSec: 6.6 },
            500: { batchSize: 117, processingTimeSec: 5.5 },
            250: { batchSize: 154, processingTimeSec: 3.8 },
            100: { batchSize: 194, processingTimeSec: 3.1 },
            75: { batchSize: 196, processingTimeSec: 2.8 },
            50: { batchSize: 197, processingTimeSec: 2.3 },
            25: { batchSize: 245, processingTimeSec: 2.4 },
            10: { batchSize: 251, processingTimeSec: 2.2 }
        };
    }

    /**
     * Get default API key (MD5 hash of "text2vector")
     * @returns Default API key for StarFactory embedding service
     */
    static getDefaultApiKey(): string {
        return 'f4e60824193fc9cbde1110e30c947a75'; // MD5 hash of "text2vector"
    }
} 