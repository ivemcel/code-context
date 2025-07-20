import {
    CodeIndexer,
    MilvusVectorDatabase,
    OllamaEmbedding,
    // Uncomment to use OpenAI or VoyageAI
    OpenAIEmbedding,
    // VoyageAIEmbedding,
    AstCodeSplitter
} from '@code-indexer/core';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

// Try to load .env file
try {
    require('dotenv').config();
} catch (error) {
    // dotenv is not required, skip if not installed
}

async function main() {
    console.log('🚀 CodeIndexer Real Usage Example');
    console.log('===============================');
    
    // Only run mainReindexByChange if explicitly requested
    if (process.env.REINDEX_TEST === '1') {
        await mainReindexByChange().catch(error => {
            console.error('❌ reindexByChange 测试失败:', error);
        });
        return;
    }

    try {
        // 1. Configure Embedding Provider
        // ------------------------------------
        // Option A: Ollama (local model, recommended for getting started)
        const embedding = new OllamaEmbedding({
            model: "mxbai-embed-large" // Make sure you have pulled this model with `ollama pull mxbai-embed-large`
            // ollamaUrl: "http://localhost:11434", // 已移除，避免类型错误
        });
        console.log('🔧 Using Ollama embedding model');

        
        // // Option B: OpenAI
        // if (!process.env.OPENAI_API_KEY) {
        //     throw new Error('OPENAI_API_KEY environment variable is not set.');
        // }
        // const embedding = new OpenAIEmbedding({
        //     apiKey: process.env.OPENAI_API_KEY,
        //     model: 'text-embedding-3-small'
        // });
        // console.log('🔧 Using OpenAI embedding model');
        

        /*
        // Option C: VoyageAI
        if (!process.env.VOYAGE_API_KEY) {
            throw new Error('VOYAGE_API_KEY environment variable is not set.');
        }
        const embedding = new VoyageAIEmbedding({
            apiKey: process.env.VOYAGE_API_KEY,
            model: 'voyage-2'
        });
        console.log('🔧 Using VoyageAI embedding model');
        */


        // 2. Configure Vector Database
        // --------------------------------
        const milvusAddress = process.env.MILVUS_ADDRESS || 'localhost:19530';
        const milvusToken = process.env.MILVUS_TOKEN; // Optional
        console.log(`🔌 Connecting to Milvus at: ${milvusAddress}`);

        const vectorDatabase = new MilvusVectorDatabase({
            address: milvusAddress,
            ...(milvusToken && { token: milvusToken })
        });


        // 3. Create CodeIndexer instance
        // ----------------------------------
        const codeSplitter = new AstCodeSplitter(2500, 300);
        const indexer = new CodeIndexer({
            embedding, // Pass the configured embedding provider
            vectorDatabase,
            codeSplitter,
            supportedExtensions: ['.ts', '.js', '.py', '.java', '.cpp', '.go', '.rs']
        });


        // 4. Index the codebase
        // -------------------------
        console.log('\n📖 Starting to index codebase...');
        const codebasePath = path.join(__dirname, '../..'); // Index the entire monorepo
        
        // The collection name is now derived internally from the codebasePath
        // const collectionName = indexer.getCollectionName(codebasePath); // 私有方法，不能外部调用
        // console.log(`ℹ️  Using collection: ${collectionName}`);

        // Check if index already exists and clear if needed
        const hasExistingIndex = await indexer.hasIndex(codebasePath);
        if (hasExistingIndex) {
            console.log('🗑️  Existing index found, clearing it first...');
            await indexer.clearIndex(codebasePath);
        }

        // Index with progress tracking - API has changed
        const indexStats = await indexer.indexCodebase(codebasePath, (progress) => {
            console.log(`   [${progress.phase}] ${progress.percentage.toFixed(2)}%`);
        });
        console.log(`\n📊 Indexing stats: ${indexStats.indexedFiles} files, ${indexStats.totalChunks} code chunks`);


        // 5. Perform semantic search
        // ----------------------------
        /*
        console.log('\n🔍 Performing semantic search...');
        const queries = [
            'vector database operations',
            'code splitting functions',
            'embedding generation',
        ];

        for (const query of queries) {
            console.log(`\n🔎 Search: "${query}"`);
            const results = await indexer.semanticSearch(collectionName, query, 3, 0.3);

            if (results.length > 0) {
                results.forEach((result, index) => {
                    console.log(`   ${index + 1}. Similarity: ${(result.score * 100).toFixed(2)}%`);
                    console.log(`      File: ${result.relativePath}`);
                    console.log(`      Language: ${result.language}`);
                    console.log(`      Lines: ${result.startLine}-${result.endLine}`);
                    console.log(`      Preview: ${result.content.substring(0, 100).replace(/\n/g, ' ')}...`);
                });
            } else {
                console.log('   No relevant results found');
            }
        }
        */
        console.log('\n🎉 Example completed successfully!');

    } catch (error) {
        console.error('❌ Error occurred:', error);
        // Add specific error handling for different services if needed
        process.exit(1);
    }
}

/**
 * 独立的 reindexByChange 测试入口
 * 1. 先全量索引一次
 * 2. 再执行增量索引
 * 3. 检查快照文件是否存在
 */
async function mainReindexByChange() {
    console.log('==== reindexByChange 测试 ====');
    const ollamaModel = process.env.OLLAMA_MODEL || 'mxbai-embed-large';
    const milvusAddress = process.env.MILVUS_ADDRESS || 'localhost:19530';
    const milvusToken = process.env.MILVUS_TOKEN;
    const codebasePath = process.env.TEST_CODEBASE_PATH || '/Users/ivem/Desktop/test';

    const vectorDatabase = new MilvusVectorDatabase({ address: milvusAddress, ...(milvusToken && { token: milvusToken }) });
    const embedding = new OllamaEmbedding({ model: ollamaModel });
    const codeSplitter = new AstCodeSplitter(2500, 300);
    const indexer = new CodeIndexer({ vectorDatabase, embedding, codeSplitter, supportedExtensions: ['.ts', '.js', '.py', '.java', '.cpp', '.go', '.rs'] });

    // 1. 全量索引
    console.log('1. 执行全量索引...');
    try {
        // 检查索引是否已存在
        const hasExistingIndex = await indexer.hasIndex(codebasePath);
        if (hasExistingIndex) {
            console.log('索引已存在，清除旧索引...');
            await indexer.clearIndex(codebasePath);
        }
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

    // 3. 检查快照文件
    const normalizedPath = path.resolve(codebasePath);
    const hash = require('crypto').createHash('md5').update(normalizedPath).digest('hex');
    const snapshotDir = path.join(os.homedir(), '.codeindexer', 'merkle');
    const snapshotFile = path.join(snapshotDir, `code_chunks_${hash.substring(0, 8)}.json`);
    const exists = fs.existsSync(snapshotFile);
    console.log('快照文件路径:', snapshotFile);
    console.log('快照文件是否存在:', exists);
}

if (require.main === module && process.env.REINDEX_TEST === '1') {
    mainReindexByChange().catch(console.error);
}

// Run main program
if (require.main === module) {
    main().catch(console.error);
}

export { main };