/**
 * 这是一个演示Milvus BM25混合搜索功能的独立测试脚本。
 * 配置好MilvusVectorDatabase并使用其search方法，结合查询文本和向量进行混合搜索。
 */

import {
    MilvusVectorDatabase,
    StarFactoryEmbedding
} from '@code-indexer/core';

// Try to load .env file
try {
    require('dotenv').config();
} catch (error) {
    // dotenv is not required, skip if not installed
}

/**
 * 演示如何使用Milvus BM25混合搜索功能
 */
async function testHybridSearch(collectionName: string, queryText: string) {
    console.log('🔍 Milvus BM25 混合搜索示例');
    console.log('====================================');
    
    try {
        // 1. 配置向量数据库，启用BM25
        const milvusAddress = process.env.MILVUS_ADDRESS || 'localhost:19530';
        const milvusToken = process.env.MILVUS_TOKEN;
        
        console.log(`🔌 连接到Milvus: ${milvusAddress}`);
        console.log('🔧 启用BM25混合搜索');
        
        const vectorDatabase = new MilvusVectorDatabase({
            address: milvusAddress,
            ...(milvusToken && { token: milvusToken }),
            enableBM25: true,
            consistencyLevel: 'Bounded' as 'Bounded',
            rankerType: 'weight' as 'weight',
            rankerParams: { k: 100 }
        });
        
        // 2. 配置嵌入模型
        const starFactoryApiKey = process.env.STARFACTORY_API_KEY || StarFactoryEmbedding.getDefaultApiKey();
        const starFactoryBaseURL = process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085';
        
        const embedding = new StarFactoryEmbedding({
            apiKey: starFactoryApiKey,
            baseURL: starFactoryBaseURL,
            model: 'NV-Embed-v2'
        });
        
        console.log(`📚 使用集合: ${collectionName}`);
        
        // 3. 生成查询向量
        console.log(`🔎 为查询文本生成向量嵌入: "${queryText}"`);
        const queryEmbedding = await embedding.embed(queryText);
        
        // 4. 执行混合搜索
        console.log('🔍 执行混合搜索...');
        const searchResults = await vectorDatabase.search(
            collectionName,
            queryEmbedding.vector,
            {
                topK: 5,
                threshold: 0.5,
                query: queryText // 添加文本查询用于BM25部分
            }
        );
        
        // 5. 显示结果
        console.log(`\n✅ 找到 ${searchResults.length} 个结果:`);
        
        if (searchResults.length > 0) {
            searchResults.forEach((result, index) => {
                console.log(`\n结果 ${index + 1} - 相似度: ${(result.score * 100).toFixed(2)}%`);
                console.log(`文件: ${result.document.relativePath}`);
                console.log(`行号: ${result.document.startLine}-${result.document.endLine}`);
                // 只显示内容的前100个字符
                const preview = result.document.content.substring(0, 100).replace(/\n/g, ' ') + '...';
                console.log(`预览: ${preview}`);
            });
        } else {
            console.log('未找到相关结果。');
        }
        
    } catch (error) {
        console.error('❌ 发生错误:', error);
    }
}

// 执行测试
if (require.main === module) {
    // 要搜索的集合名称
    const collectionName = process.argv[2] || 'code_chunks_05a7aa8c';
    
    // 查询文本
    const queryText = process.argv[3] || '用户登录和注册功能的核心逻辑';

    testHybridSearch(collectionName, queryText).catch(error => {
        console.error('❌ 致命错误:', error);
        process.exit(1);
    });
}

export { testHybridSearch }; 