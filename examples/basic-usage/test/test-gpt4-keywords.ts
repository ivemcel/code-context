/**
 * 测试GPT-4关键词提取功能的脚本
 * 
 * 该脚本测试使用GPT-4提取中文关键词的功能，用于改进混合搜索
 */

import { MilvusConfig, MilvusVectorDatabase } from '@code-indexer/core/src/vectordb/milvus-vectordb';
import { StarFactoryEmbedding } from '@code-indexer/core/src/embedding/starfactory-embedding';
import { HybridSearchOptions } from '@code-indexer/core/src/vectordb';

async function main() {
    if (process.argv.length < 3) {
        console.log('用法: npx ts-node test-gpt4-keywords.ts "查询文本"');
        process.exit(1);
    }

    const query = process.argv[2];
    console.log('====================================');
    console.log('🔍 GPT-4关键词提取测试');
    console.log('====================================');
    console.log(`📝 测试查询: "${query}"`);
    
    // 连接到Milvus数据库
    console.log('🔌 连接到Milvus数据库...');
    const config: MilvusConfig = {
        address: 'localhost:19530',
        enableBM25: true,
        consistencyLevel: 'Bounded' as 'Bounded',
        rankerType: 'weight' as 'weight'
    };

    // 创建MilvusVectorDatabase实例
    const vectorDb = new MilvusVectorDatabase(config);
    
    // 测试在代码库中搜索
    try {
        console.log('🔎 为查询文本生成向量嵌入...');
        const embedding = new StarFactoryEmbedding({
            apiKey: 'knJ6mlyfvRP6OHwl79d2AF2mgCEhwO4d',
            baseURL: 'http://10.142.99.29:8085'
        });
        const vector = (await embedding.embed(query)).vector;
        
        // 定义搜索选项
        const searchOptions: HybridSearchOptions = {
            query: query,
            topK: 5
        };
        
        console.log('🔍 执行混合搜索...');
        // 使用code_chunks_05a7aa8c集合进行测试
        const collectionName = 'code_chunks_05a7aa8c';
        const results = await vectorDb.search(collectionName, vector, searchOptions);
        
        console.log(`\n✅ 找到 ${results.length} 个结果:`);
        if (results.length === 0) {
            console.log('未找到相关结果。');
        } else {
            results.forEach((result, index) => {
                const doc = result.document;
                console.log(`\n结果 ${index + 1} - 相似度: ${(result.score * 100).toFixed(2)}%`);
                console.log(`文件: ${doc.relativePath}`);
                console.log(`行号: ${doc.startLine}-${doc.endLine}`);
                const preview = doc.content.length > 100 
                    ? doc.content.substring(0, 100) + '...'
                    : doc.content;
                console.log(`预览: ${preview}`);
            });
        }
    } catch (error) {
        console.error('❌ 搜索过程中发生错误:', error);
    }
}

// 执行主函数
main().catch(error => {
    console.error('执行过程中发生错误:', error);
    process.exit(1);
}); 