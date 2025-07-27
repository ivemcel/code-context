// 简单测试 Qwen3Embedding 的批处理功能

import { Qwen3Embedding } from '@code-indexer/core/src/embedding/qwen3-embedding';
import { EmbeddingVector } from '@code-indexer/core/src/embedding/base-embedding';

// Try to load .env file
try {
    require('dotenv').config();
} catch (error) {
    // dotenv is not required, skip if not installed
}

async function main() {
    console.log('=== Qwen3Embedding 批处理测试 ===');
    
    // 获取API密钥
    const qwen3ApiKey = process.env.QWEN_API_KEY || 'sk-3b6eca9223744941b801b4332a70a694';
    
    // 初始化 Qwen3Embedding
    const embedding = new Qwen3Embedding({
        apiKey: qwen3ApiKey,
        model: 'text-embedding-v1'
    });
    
    // 创建一个较大的测试数据集（超过最大批处理限制，确保能测试批处理逻辑）
    const testData = [];
    for (let i = 0; i < 35; i++) {
        testData.push(`这是测试文本 ${i}，用于验证批处理功能。这个批次将会被分成多个子批次处理，因为API限制每批最多10个请求。`);
    }
    
    console.log(`创建了 ${testData.length} 个测试文本`);
    
    try {
        // 测试批处理嵌入
        console.log('开始批处理嵌入...');
        const results = await embedding.embedBatch(testData);
        
        console.log(`成功获取 ${results.length} 个嵌入向量`);
        if (results.length > 0 && results[0].vector) {
            console.log(`第一个向量的维度: ${results[0].vector.length}`);
        }
        console.log('测试成功!');
    } catch (error) {
        console.error('测试失败:', error);
    }
}

main().catch(console.error); 