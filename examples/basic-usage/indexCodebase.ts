import {
    CodeIndexer,
    MilvusVectorDatabase,
    OllamaEmbedding,
    // Uncomment to use OpenAI or VoyageAI
    OpenAIEmbedding,
    VoyageAIEmbedding,
    AstCodeSplitter,
    StarFactoryEmbedding,
    Qwen3Embedding,
    MilvusRestfulVectorDatabase,
    EmbeddingVector
} from '@code-indexer/core';
import { EnhancedAstSplitter } from '@code-indexer/core/src/splitter/enhanced-ast-splitter';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

// Try to load .env file
try {
    require('dotenv').config();
} catch (error) {
    // dotenv is not required, skip if not installed
}

// Custom rate-limited embedding wrapper for VoyageAI (free tier: 3 RPM, 10K TPM)
class RateLimitedVoyageEmbedding extends VoyageAIEmbedding {
    private lastEmbedTime: number = 0;
    private embeddingQueue: Array<{
        texts: string[],
        resolve: (value: EmbeddingVector[]) => void,
        reject: (reason: any) => void
    }> = [];
    private processingQueue: boolean = false;
    private tokenCount: number = 0;
    private tokenResetTimeout: NodeJS.Timeout | null = null;
    private requestCount: number = 0;
    private requestResetTimeout: NodeJS.Timeout | null = null;
    
    // Conservative rate limits for free tier
    private readonly MIN_REQUEST_DELAY_MS: number = 25000; // 25 seconds between requests (conservative)
    private readonly MAX_REQUESTS_PER_MINUTE: number = 2; // Limit to 2 per minute to be safe
    private readonly MAX_TOKENS_PER_MINUTE: number = 9000; // Keep under 10K TPM limit
    private readonly MAX_BATCH_SIZE: number = 2; // Small batch size to reduce chances of hitting limits

    constructor(config: any) {
        super(config);
        console.log('🕒 Using strict rate-limited VoyageAI embedding (enforce 25s between requests)');
        
        // Reset token count every minute
        this.resetTokenCounter();
        this.resetRequestCounter();
    }

    private resetTokenCounter() {
        if (this.tokenResetTimeout) {
            clearTimeout(this.tokenResetTimeout);
        }
        
        this.tokenResetTimeout = setTimeout(() => {
            console.log(`🔄 Resetting token count (was ${this.tokenCount})`);
            this.tokenCount = 0;
            this.resetTokenCounter();
            // Process any pending requests
            this.processQueue();
        }, 60 * 1000); // 1 minute
    }
    
    private resetRequestCounter() {
        if (this.requestResetTimeout) {
            clearTimeout(this.requestResetTimeout);
        }
        
        this.requestResetTimeout = setTimeout(() => {
            console.log(`🔄 Resetting request count (was ${this.requestCount})`);
            this.requestCount = 0;
            this.resetRequestCounter();
            // Process any pending requests
            this.processQueue();
        }, 60 * 1000); // 1 minute
    }

    // Add cleanup method to clear timers
    public cleanup() {
        if (this.tokenResetTimeout) {
            clearTimeout(this.tokenResetTimeout);
            this.tokenResetTimeout = null;
        }
        if (this.requestResetTimeout) {
            clearTimeout(this.requestResetTimeout);
            this.requestResetTimeout = null;
        }
        console.log('🧹 Cleaned up rate limiter timers');
    }

    private estimateTokens(texts: string[]): number {
        // Rough estimation: ~1 token per 4 characters
        return texts.reduce((sum, text) => sum + Math.ceil(text.length / 4), 0);
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        const estimatedTokens = this.estimateTokens(texts);
        
        // If this would exceed our rate limit, split into smaller batches
        if (texts.length > this.MAX_BATCH_SIZE) {
            console.log(`⚠️ Large batch detected (${texts.length} chunks, ~${estimatedTokens} tokens), splitting into smaller batches`);
            
            // Process in smaller batches to avoid hitting rate limits
            const batchSize = this.MAX_BATCH_SIZE; // Small batch size to stay under limits
            const results: EmbeddingVector[] = [];
            
            for (let i = 0; i < texts.length; i += batchSize) {
                const batch = texts.slice(i, i + batchSize);
                console.log(`📦 Processing mini-batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(texts.length/batchSize)} (${batch.length} chunks)`);
                const batchResults = await this.embedBatchWithRateLimit(batch);
                results.push(...batchResults);
                
                // Always add significant delay between batches
                if (i + batchSize < texts.length) {
                    const delayMs = this.MIN_REQUEST_DELAY_MS + 5000; // Add extra 5 seconds buffer
                    console.log(`⏱️ Adding ${delayMs/1000}s delay between batches to respect rate limits...`);
                    await new Promise(resolve => setTimeout(resolve, delayMs));
                }
            }
            
            return results;
        }
        
        return this.embedBatchWithRateLimit(texts);
    }

    private async embedBatchWithRateLimit(texts: string[]): Promise<EmbeddingVector[]> {
        return new Promise((resolve, reject) => {
            this.embeddingQueue.push({ texts, resolve, reject });
            this.processQueue();
        });
    }

    private async processQueue() {
        if (this.processingQueue || this.embeddingQueue.length === 0) {
            return;
        }

        this.processingQueue = true;
        const { texts, resolve, reject } = this.embeddingQueue.shift()!;
        
        // Estimate tokens for this request
        const estimatedTokens = this.estimateTokens(texts);
        
        // Check if this would exceed our token limit
        if (this.tokenCount + estimatedTokens > this.MAX_TOKENS_PER_MINUTE) {
            console.log(`⏳ Token limit reached (${this.tokenCount}+${estimatedTokens} > ${this.MAX_TOKENS_PER_MINUTE}), waiting for token counter reset...`);
            
            // Put the request back at the front of the queue
            this.embeddingQueue.unshift({ texts, resolve, reject });
            
            // Retry after the token counter resets
            this.processingQueue = false;
            return;
        }
        
        // Check if we've hit the request count limit
        if (this.requestCount >= this.MAX_REQUESTS_PER_MINUTE) {
            console.log(`⏳ Request limit reached (${this.requestCount} >= ${this.MAX_REQUESTS_PER_MINUTE}), waiting for request counter reset...`);
            
            // Put the request back at the front of the queue
            this.embeddingQueue.unshift({ texts, resolve, reject });
            
            // Retry after the request counter resets
            this.processingQueue = false;
            return;
        }
        
        // Check rate limit (enforce minimum delay between requests)
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastEmbedTime;
        
        if (timeSinceLastRequest < this.MIN_REQUEST_DELAY_MS) {
            const waitTime = this.MIN_REQUEST_DELAY_MS - timeSinceLastRequest;
            console.log(`⏳ Rate limit: waiting ${waitTime/1000}s before next API call`);
            
            setTimeout(() => {
                this.processingQueue = false;
                this.processQueue();
            }, waitTime);
            
            // Put the request back at the front of the queue
            this.embeddingQueue.unshift({ texts, resolve, reject });
            return;
        }

        // Process the request
        try {
            console.log(`🔤 Processing batch of ${texts.length} texts (~${estimatedTokens} tokens)`);
            this.lastEmbedTime = now;
            this.tokenCount += estimatedTokens;
            this.requestCount++;
            
            // Call parent class implementation
            super.embedBatch(texts)
                .then(result => {
                    resolve(result);
                    this.processingQueue = false;
                    // Add fixed delay before processing next queue item
                    setTimeout(() => {
                        this.processQueue(); // Process next in queue
                    }, 2000); // Small buffer to ensure system has time to breathe
                })
                .catch(error => {
                    // If we hit a rate limit, add a long delay before retrying
                    if (error.message && error.message.includes('429')) {
                        console.error(`❌ Hit rate limit (429): ${error.message}`);
                        console.log('⏱️ Adding extended delay of 60s before retrying...');
                        
                        // Put the request back in the queue to retry
                        this.embeddingQueue.unshift({ texts, resolve, reject });
                        
                        // Pause for a full minute
                        setTimeout(() => {
                            this.processingQueue = false;
                            this.processQueue();
                        }, 60000);
                    } else {
                        console.error(`❌ Embedding error: ${error.message}`);
                        reject(error);
                        this.processingQueue = false;
                        this.processQueue(); // Process next in queue
                    }
                });
        } catch (error) {
            console.error(`❌ Unexpected error in rate limiter: ${error}`);
            reject(error);
            this.processingQueue = false;
            this.processQueue(); // Process next in queue
        }
    }

    // Override embed method to use our rate-limited batch process
    async embed(text: string): Promise<EmbeddingVector> {
        const results = await this.embedBatch([text]);
        return results[0];
    }
}

async function main() {
    // 替换为中国时间格式 (UTC+8)
    const timestamp = new Date().toLocaleString('zh-CN', {
        timeZone: 'Asia/Shanghai', 
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    }).replace(/[\/-]/g, '-').replace(/[,]/g, ' ');
    
    try {
        console.log(`[${timestamp}] ==== index codebase 测试 ====`);
        // 使用StarFactory作为默认嵌入模型
        const starFactoryApiKey = process.env.STARFACTORY_API_KEY || StarFactoryEmbedding.getDefaultApiKey();
        const starFactoryBaseURL = process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085';
        //const milvusAddress = process.env.MILVUS_ADDRESS || 'localhost:19530';
        const milvusToken = process.env.MILVUS_TOKEN;
        //const codebasePath = process.env.TEST_CODEBASE_PATH || '/Users/ivem/Desktop/test';
        
        //const codebasePath = "/Users/ivem/IdeaProjects/star-factory";
        //const codebasePath ="/Users/ivem/IdeaProjects/star-factory-codebase";
        //const codebasePath = "/Users/ivem/Desktop/test-qwen";
        //const codebasePath = "/Users/ivem/Desktop/test-voyage";
        //const codebasePath = "/Users/ivem/Desktop/test-starfactory";
        
        //const codebasePath = "/Users/ivem/Desktop/user-data-qwen";
        //const codebasePath = "/Users/ivem/Desktop/user-data-voyage";
        //const codebasePath = "/Users/ivem/Desktop/user-data-starfactory";

        const codebasePath = "/Users/ivem/Desktop/test";

        //const vectorDatabase = new MilvusVectorDatabase({ address: milvusAddress, ...(milvusToken && { token: milvusToken }) });
        const milvusAddress = "http://10.142.99.29:8085";
        const vectorDatabase = new MilvusRestfulVectorDatabase({ address: milvusAddress, ...(milvusToken && { token: milvusToken }) });
        // Get API key from environment variables
        const voyageApiKey = process.env.VOYAGE_API_KEY || 'pa-Weutp7FYlGyUXb8mU46hQdDcvJZhs53WJ3IWQGzszQl';

        //Use rate-limited embedding for VoyageAI to respect free tier limits
        // let embedding = new RateLimitedVoyageEmbedding({
        //     apiKey: voyageApiKey,
        //     model: 'voyage-code-3' // 使用voyage-code-3模型，针对代码优化
        // });
        

        // const qwen3ApiKey = "sk-3b6eca9223744941b801b4332a70a694"
        // const embedding = new Qwen3Embedding({
        //     apiKey: qwen3ApiKey,
        //     model: 'text-embedding-v4' // 或者使用mini/huge版本
        // });

        //// 使用star-factory-embedding
        const embedding = new StarFactoryEmbedding({
            baseURL: 'http://10.142.99.29:8085',
            apiKey: starFactoryApiKey
        });
        
        
        //const embedding = new OllamaEmbedding();
        // 使用增强型AST分割器，支持保留注释
        const codeSplitter = new EnhancedAstSplitter(0, 0);

        // Configure very small batch size for IndexCodebase
        process.env.EMBEDDING_BATCH_SIZE = '20'; // Use smallest practical batch size
        
        const indexer = new CodeIndexer({ vectorDatabase, embedding, codeSplitter, supportedExtensions: ['.ts', '.js', '.py', '.java', '.cpp', '.go', '.rs'] });

        ////1. 全量索引
        console.log('1. 执行全量索引...');
        try {
            // 强制清除现有索引，确保使用新代码创建
            console.log('强制清除现有本地索引文件，确保使用修复后的代码...');
            // 注意clearIndex会删除集合重新创建！！！！丢失数据！！！！
            await indexer.clearIndex(codebasePath);
            
            // 添加延迟确保旧索引被完全删除
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            console.log('开始创建新索引...');
            await indexer.indexCodebase(codebasePath);
            console.log('全量索引完成');
        } catch (error) {
            console.error('全量索引失败:', error);
            return;
        }
        
        // 2. 增量索引
        console.log('2. 执行 reindexByChange...');
        try {
            const stats = await indexer.reindexByChange(codebasePath);
            console.log('reindexByChange 结果:', stats);
        } catch (error) {
            console.error('增量索引失败:', error);
            return;
        }

        // // 3. 检查快照文件
        // const normalizedPath = path.resolve(codebasePath);
        // const hash = require('crypto').createHash('md5').update(normalizedPath).digest('hex');
        // const snapshotDir = path.join(os.homedir(), '.codeindexer', 'merkle');
        // const snapshotFile = path.join(snapshotDir, `code_chunks_${hash.substring(0, 8)}.json`);
        // const exists = fs.existsSync(snapshotFile);
        // console.log('快照文件路径:', snapshotFile);
        // console.log('快照文件是否存在:', exists);

    } catch (error) {
        console.error('❌ Error occurred:', error);
        // Add specific error handling for different services if needed
        process.exit(1);
    } finally {
        // 清理定时器资源
        // if (embedding) {
        //     embedding.cleanup();
        // }
        // 确保程序正常退出，不留下悬挂的定时器
        console.log('程序执行完成，即将退出');
        
        // 更新结束时间戳为中国时间
        const endTimestamp = new Date().toLocaleString('zh-CN', {
            timeZone: 'Asia/Shanghai', 
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        }).replace(/[\/-]/g, '-').replace(/[,]/g, ' ');
        
        console.log(`[${endTimestamp}] ==== index codebase 测试完成 ====`);
        // 给一点时间让最后的日志输出
        setTimeout(() => {
            process.exit(0);
        }, 500);
    }
}

// Run main program
if (require.main === module) {
    main().catch(console.error);
}

export { main };