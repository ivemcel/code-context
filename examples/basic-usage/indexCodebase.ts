import {
    CodeIndexer,
    MilvusVectorDatabase,
    OllamaEmbedding,
    // Uncomment to use OpenAI or VoyageAI
    OpenAIEmbedding,
    // VoyageAIEmbedding,
    AstCodeSplitter,
    StarFactoryEmbedding,
    MilvusRestfulVectorDatabase
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

async function main() {
    try {
        console.log('==== reindexByChange 测试 ====');
        // 使用StarFactory作为默认嵌入模型
        const starFactoryApiKey = process.env.STARFACTORY_API_KEY || StarFactoryEmbedding.getDefaultApiKey();
        const starFactoryBaseURL = process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085';
        const milvusAddress = process.env.MILVUS_ADDRESS || 'localhost:19530';
        const milvusToken = process.env.MILVUS_TOKEN;
        //const codebasePath = process.env.TEST_CODEBASE_PATH || '/Users/ivem/Desktop/test';
        //const codebasePath = "/Users/ivem/IdeaProjects/star-factory/star-factory-user";
        //const codebasePath = "/Users/ivem/IdeaProjects/star-factory-test";
        const codebasePath = "/Users/ivem/Desktop/rag-codebase";

        const vectorDatabase = new MilvusVectorDatabase({ address: milvusAddress, ...(milvusToken && { token: milvusToken }) });
        const embedding = new StarFactoryEmbedding({ 
            apiKey: starFactoryApiKey, 
            baseURL: starFactoryBaseURL,
            model: 'NV-Embed-v2'
        });
        //const embedding = new OllamaEmbedding();
        // 使用增强型AST分割器，支持保留注释
        const codeSplitter = new EnhancedAstSplitter(0, 0);

        const indexer = new CodeIndexer({ vectorDatabase, embedding, codeSplitter, supportedExtensions: ['.ts', '.js', '.py', '.java', '.cpp', '.go', '.rs'] });

        // 1. 全量索引
        console.log('1. 执行全量索引...');
        try {
            // 强制清除现有索引，确保使用新代码创建
            console.log('强制清除现有索引，确保使用修复后的代码...');
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

        // 3. 检查快照文件
        const normalizedPath = path.resolve(codebasePath);
        const hash = require('crypto').createHash('md5').update(normalizedPath).digest('hex');
        const snapshotDir = path.join(os.homedir(), '.codeindexer', 'merkle');
        const snapshotFile = path.join(snapshotDir, `code_chunks_${hash.substring(0, 8)}.json`);
        const exists = fs.existsSync(snapshotFile);
        console.log('快照文件路径:', snapshotFile);
        // console.log('快照文件是否存在:', exists);

    } catch (error) {
        console.error('❌ Error occurred:', error);
        // Add specific error handling for different services if needed
        process.exit(1);
    }
}

// Run main program
if (require.main === module) {
    main().catch(console.error);
}

export { main };