/**
 * Qwen3 Embedding Test
 * 
 * This example demonstrates how to use the Qwen3 embedding model
 * to generate embeddings for code snippets and perform semantic search.
 * 
 * To run this example:
 * ts-node --transpile-only qwen3-embedding-test.ts
 */

import { Qwen3Embedding } from '../../packages/core/src/embedding';

// 配置
const API_KEY = 'sk-3b6eca9223744941b801b4332a70a694'; // Qwen3 API 密钥
const MODEL = 'text-embedding-v1'; // 默认模型
const DEBUG = process.env.DEBUG === 'true';

// 格式化向量元素的函数，兼容不同类型的数据
function formatVectorElement(val: any): string {
    if (typeof val === 'number') {
        return val.toFixed(4);
    } else if (val === null || val === undefined) {
        return 'null';
    } else {
        return String(val);
    }
}

// 辅助函数：余弦相似度计算
function cosineSimilarity(a: number[], b: number[]): number {
    try {
        if (!Array.isArray(a) || !Array.isArray(b)) {
            console.error('Invalid vector types:', typeof a, typeof b);
            return NaN;
        }
        
        if (a.length !== b.length) {
            console.error(`Vector length mismatch: ${a.length} vs ${b.length}`);
            return NaN;
        }
        
        let dotProduct = 0;
        let magA = 0;
        let magB = 0;
        
        for (let i = 0; i < a.length; i++) {
            // 确保元素是数字
            const numA = typeof a[i] === 'number' ? a[i] : Number(a[i]);
            const numB = typeof b[i] === 'number' ? b[i] : Number(b[i]);
            
            if (isNaN(numA) || isNaN(numB)) {
                console.error('非数值元素:', a[i], b[i]);
                continue;
            }
            
            dotProduct += numA * numB;
            magA += numA * numA;
            magB += numB * numB;
        }
        
        magA = Math.sqrt(magA);
        magB = Math.sqrt(magB);
        
        if (magA === 0 || magB === 0) {
            console.error('Zero magnitude vector detected');
            return 0;
        }
        
        return dotProduct / (magA * magB);
    } catch (e) {
        console.error('Error calculating cosine similarity:', e);
        return NaN;
    }
}

interface SearchResult {
    sample: string;
    similarity: number;
}

async function main() {
    try {
        console.log('📊 Qwen3 Embedding API 测试');
        console.log(`🔧 使用模型: ${MODEL}`);
        if (DEBUG) console.log('调试模式: 开启');
        
        // 初始化 Qwen3Embedding 实例
        const qwen3Embedding = new Qwen3Embedding({
            apiKey: API_KEY,
            model: MODEL
        });
        
        console.log(`📏 向量维度: ${qwen3Embedding.getDimension()}`);
        console.log(`🏢 提供商: ${qwen3Embedding.getProvider()}`);
        
        // 准备测试样本
        const samples = [
            // 代码样本
            `function calculateArea(radius) {
                return Math.PI * radius * radius;
            }`,
            
            // 自然语言样本
            "人工智能在改变我们的工作方式，特别是在软件开发领域。",
            
            // 混合样本
            `class UserService {
                /**
                 * 查询用户信息
                 * @param userId 用户ID
                 * @return 用户详细信息
                 */
                public UserInfo getUserById(String userId) {
                    return userRepository.findById(userId);
                }
            }`
        ];
        
        console.log('\n🔍 单个文本嵌入测试');
        console.time('单个嵌入耗时');
        const singleEmbedding = await qwen3Embedding.embed(samples[0]);
        console.timeEnd('单个嵌入耗时');
        
        if (singleEmbedding && Array.isArray(singleEmbedding.vector)) {
            console.log(`✅ 获取到 ${singleEmbedding.dimension} 维向量`);
            console.log(`✨ 向量前5个值: [${singleEmbedding.vector.slice(0, 5).map(formatVectorElement).join(', ')}...]`);
        } else {
            console.error('❌ 向量格式错误:', typeof singleEmbedding.vector);
        }
        
        console.log('\n🔍 批量文本嵌入测试');
        console.time('批量嵌入耗时');
        const batchEmbeddings = await qwen3Embedding.embedBatch(samples);
        console.timeEnd('批量嵌入耗时');
        
        console.log(`✅ 成功获取 ${batchEmbeddings.length} 个嵌入向量`);
        for (let i = 0; i < batchEmbeddings.length; i++) {
            if (batchEmbeddings[i] && Array.isArray(batchEmbeddings[i].vector)) {
                console.log(`样本 ${i+1} - 前5个向量元素: [${batchEmbeddings[i].vector.slice(0, 5).map(formatVectorElement).join(', ')}...]`);
            } else {
                console.error(`❌ 样本 ${i+1} 向量格式错误:`, typeof batchEmbeddings[i].vector);
            }
        }
        
        // 计算相似度矩阵
        console.log('\n📊 样本间相似度矩阵:');
        const similarityMatrix: Array<Array<string>> = [];
        let validVectors = true;
        
        for (let i = 0; i < samples.length; i++) {
            const row: Array<string> = [];
            for (let j = 0; j < samples.length; j++) {
                if (!Array.isArray(batchEmbeddings[i].vector) || !Array.isArray(batchEmbeddings[j].vector)) {
                    validVectors = false;
                    row.push('N/A');
                    continue;
                }
                
                const similarity = cosineSimilarity(
                    batchEmbeddings[i].vector, 
                    batchEmbeddings[j].vector
                );
                
                row.push(isNaN(similarity) ? 'N/A' : similarity.toFixed(4));
            }
            similarityMatrix.push(row);
        }
        
        if (validVectors) {
            console.table(similarityMatrix);
        } else {
            console.error('❌ 无法计算相似度矩阵: 向量格式无效');
        }
        
        // 查询示例
        console.log('\n🔍 语义搜索示例');
        const query = "如何获取用户数据?";
        console.log(`查询: "${query}"`);
        
        console.time('查询嵌入生成');
        const queryEmbedding = await qwen3Embedding.embed(query);
        console.timeEnd('查询嵌入生成');
        
        if (!Array.isArray(queryEmbedding.vector)) {
            console.error('❌ 查询向量格式错误:', typeof queryEmbedding.vector);
            return;
        }
        
        // 计算查询与样本的相似度
        const searchResults: SearchResult[] = [];
        for (let i = 0; i < samples.length; i++) {
            if (!Array.isArray(batchEmbeddings[i].vector)) {
                continue;
            }
            
            const similarity = cosineSimilarity(queryEmbedding.vector, batchEmbeddings[i].vector);
            if (!isNaN(similarity)) {
                searchResults.push({
                    sample: samples[i].length > 50 ? samples[i].substring(0, 50) + '...' : samples[i],
                    similarity: similarity
                });
            }
        }
        
        // 按相似度排序
        searchResults.sort((a, b) => b.similarity - a.similarity);
        
        // 显示结果
        console.log('\n📋 搜索结果 (按相似度排序):');
        for (const result of searchResults) {
            console.log(`${result.similarity.toFixed(4)} - ${result.sample}`);
        }
        
        // 显示支持的模型信息
        console.log('\n📚 Qwen 支持的模型:');
        const supportedModels = Qwen3Embedding.getSupportedModels();
        for (const [model, info] of Object.entries(supportedModels)) {
            console.log(`- ${model}: ${info.dimension}维 - ${info.description}`);
        }
        
    } catch (error) {
        console.error('❌ 错误:', error);
    }
}

// 运行测试
main(); 