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
    console.log(`[${timestamp}] ==== index codebase 测试 ====`);

    // Set environment variables for comment generation performance
    process.env.ENABLE_COMMENTS = process.env.ENABLE_COMMENTS || 'true';
    process.env.COMMENT_BATCH_SIZE = process.env.COMMENT_BATCH_SIZE || '20'; // Process 20 chunks at once
    process.env.MAX_PARALLEL_BATCHES = process.env.MAX_PARALLEL_BATCHES || '10'; // 10 concurrent batches
    process.env.EMBEDDING_BATCH_SIZE = process.env.EMBEDDING_BATCH_SIZE || '100'; // Bigger batch size for embeddings

    // 1. Configure vector database client
    console.log('🔌 Connecting to vector database at: ', process.env.MILVUS_HOST || 'localhost:19530');
    //const milvusAddress = "http://10.142.99.29:8085/codegen/milvus";
    const vectorDb = new MilvusRestfulVectorDatabase({
        address: process.env.MILVUS_HOST || 'localhost:19530',
        username: "root",
        password: "Y2GuWnu#ksvbQ*TRd", 
        database: "tongwen"
    });
    
    // 2. Configure text embedding client
    let embedding: any;

    // Use environment variable to select embedding provider
    const embeddingProvider = process.env.EMBEDDING_PROVIDER || 'starfactory';
    console.log(`🧠 Using embedding provider: ${embeddingProvider}`);
    
    switch (embeddingProvider.toLowerCase()) {
        case 'openai':
            if (!process.env.OPENAI_API_KEY) {
                throw new Error('OPENAI_API_KEY environment variable is required for OpenAI embeddings');
            }
            embedding = new OpenAIEmbedding({
                apiKey: process.env.OPENAI_API_KEY,
                model: process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-ada-002',
                ...(process.env.OPENAI_BASE_URL && { baseURL: process.env.OPENAI_BASE_URL })
            });
            break;
        
        case 'starfactory':
            embedding = new StarFactoryEmbedding({
                baseURL: 'http://10.142.99.29:8085',
                apiKey: StarFactoryEmbedding.getDefaultApiKey()
            });
            break;
            
        case 'qwen':
            if (!process.env.QWEN_API_KEY) {
                throw new Error('QWEN_API_KEY environment variable is required for Qwen embeddings');
            }
            embedding = new Qwen3Embedding({
                apiKey: process.env.QWEN_API_KEY,
                model: process.env.QWEN_EMBEDDING_MODEL || 'text-embedding-v1',
            });
            break;
            
        case 'voyage':
            if (!process.env.VOYAGE_API_KEY) {
                throw new Error('VOYAGE_API_KEY environment variable is required for Voyage AI embeddings');
            }
            // Use rate-limited client for free tier to avoid rate limiting errors
            if (process.env.VOYAGE_USE_RATE_LIMIT === 'true') {
                embedding = new RateLimitedVoyageEmbedding({
                    apiKey: process.env.VOYAGE_API_KEY,
                    model: process.env.VOYAGE_EMBEDDING_MODEL || 'voyage-2',
                });
            } else {
                embedding = new VoyageAIEmbedding({
                    apiKey: process.env.VOYAGE_API_KEY,
                    model: process.env.VOYAGE_EMBEDDING_MODEL || 'voyage-2',
                });
            }
            break;
            
        case 'ollama':
            embedding = new OllamaEmbedding({
                host: process.env.OLLAMA_HOST || 'http://localhost:11434',
                model: process.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text',
            });
            break;
            
        default:
            throw new Error(`Unknown embedding provider: ${embeddingProvider}`);
    }
    
    // 3. Configure code splitter
    console.log('🧩 Initializing code splitter...');
    
    // Use enhanced AST splitter by default, fall back to regular if needed
    let splitter;
    try {
        splitter = new EnhancedAstSplitter(0, 0);
        console.log('✓ Using EnhancedAstSplitter');
    } catch (error) {
        console.warn('⚠️ Failed to initialize EnhancedAstSplitter, falling back to AstCodeSplitter', error);
        splitter = new AstCodeSplitter(2500, 300);
    }

    // 4. Initialize code indexer
    const indexer = new CodeIndexer({
        embedding,
        vectorDatabase: vectorDb,
        codeSplitter: splitter,
        supportedExtensions: ['.java'],
        enableSparseVectors: false
    });

    // Customize ignore patterns
    // try {
    //     // This tool will automatically merge with default patterns
    //     const gitIgnorePatterns = await CodeIndexer.getIgnorePatternsFromFile(path.join(process.cwd(), '.gitignore'));
    //     indexer.updateIgnorePatterns(gitIgnorePatterns);
    // } catch (error) {
    //     console.warn('⚠️ Could not read .gitignore file, using default patterns only');
    // }

    // 5. Process command line arguments
    const args = process.argv.slice(2);
    const command = args[0] || 'index';
    // Default to current directory if not specified
    //const targetDir = args[1] || process.env.TARGET_CODEBASE_PATH || process.cwd();
    const targetDir = "/Users/ivem/Desktop/test";
    // 检查文件夹是否存在
    if (!fs.existsSync(targetDir)) {
        console.error(`❌ 文件夹不存在: ${targetDir}`);
        process.exit(1);
    }
    try {
        console.log(`1. 执行全量索引...`);
        
        // Force clear index first if requested (for clean state testing)
        console.log(`process.env.FORCE_CLEAR_INDEX: ${process.env.FORCE_CLEAR_INDEX}`);
        if (process.env.FORCE_CLEAR_INDEX === 'true') {
            console.log('强制清除现有本地索引文件，确保使用修复后的代码...');
            await indexer.clearIndex(targetDir);
        }
        
        // Check for existing index
        const hasIndex = await indexer.hasIndex(targetDir);
        
        if (hasIndex) {
            console.log('本地索引已存在，使用增量索引...');
            await indexer.reindexByChange(targetDir, progress => {
                // Update progress in logs
                if (progress.percentage % 10 === 0) {
                    console.log(`📊 Progress: ${progress.phase} (${progress.percentage}%)`);
                }
            });
        } else {
            console.log('开始创建新索引...');
            await indexer.indexCodebase(targetDir, progress => {
                // Update progress in logs
                if (progress.percentage % 10 === 0) {
                    console.log(`📊 Progress: ${progress.phase} (${progress.percentage}%)`);
                }
            });
        }
        
        // Clean up any rate limiters if used
        if (embedding.cleanup && typeof embedding.cleanup === 'function') {
            embedding.cleanup();
        }
        
        console.log('索引完成！✅');
    } catch (error) {
        console.error('❌ Error during indexing:', error);
        process.exit(1);
    }
}

// Run main program
if (require.main === module) {
    main().catch(console.error);
}

export { main };