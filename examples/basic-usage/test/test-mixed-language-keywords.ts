/**
 * 测试中英文混合查询关键词提取功能
 * 
 * 该脚本测试GPT-4在提取包含中英文混合内容（特别是代码元素）的查询关键词方面的能力
 */

import { MilvusConfig, MilvusVectorDatabase } from '@code-indexer/core/src/vectordb/milvus-vectordb';
import { StarFactoryEmbedding } from '@code-indexer/core/src/embedding/starfactory-embedding';
import { HybridSearchOptions } from '@code-indexer/core/src/vectordb';

// 测试查询示例
const TEST_QUERIES = [
  "实现LoginService接口中的用户认证方法",
  "如何优化数据库connection pool性能",
  "使用JWT token实现用户authentication",
  "查找getUserInfo方法的所有调用",
  "LoginController中的登录逻辑实现"
];

async function main() {
    let query = "";
    if (process.argv.length >= 3) {
        // 使用命令行提供的查询
        query = process.argv[2];
    } else {
        // 使用预设测试查询
        query = TEST_QUERIES[Math.floor(Math.random() * TEST_QUERIES.length)];
    }
    
    console.log('====================================');
    console.log('🔍 中英文混合查询关键词提取测试');
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