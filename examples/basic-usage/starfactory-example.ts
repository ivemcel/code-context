/**
 * StarFactory Embedding Example
 * 
 * This example demonstrates how to use the StarFactory embedding service
 * to generate embeddings for code snippets and perform semantic search.
 * 
 * To run this example:
 * ts-node --transpile-only starfactory-example.ts
 */

const axios = require('axios');

// 配置
const BASE_URL = process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085';
const API_KEY = process.env.STARFACTORY_API_KEY || 'f4e60824193fc9cbde1110e30c947a75'; // MD5 of "text2vector"
const DIMENSION = 4096;
const DEBUG = process.env.DEBUG === 'true';

// 辅助函数：余弦相似度计算
function cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magA * magB);
}

// 嵌入向量生成
async function generateEmbeddings(texts) {
    try {
        console.log(`正在为 ${texts.length} 个文本生成嵌入向量...`);
        
        const response = await axios.post(
            `${BASE_URL}/codegen/codebaseEmbedding`,
            { input: texts },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'api-key': API_KEY
                }
            }
        );

        if (DEBUG) {
            console.log('API 响应状态:', JSON.stringify({
                code: response.data?.code,
                message: response.data?.message
            }, null, 2));
        }

        // 判断响应是否包含code和data字段
        if (!response.data) {
            throw new Error('API 响应为空');
        }
        
        if (!response.data.data || !Array.isArray(response.data.data) || response.data.data.length === 0) {
            throw new Error('API 响应中没有有效的数据');
        }

        // 检查每个数据项是否有embedding字段
        console.log(`成功获取 ${response.data.data.length} 个嵌入向量`);
        return response.data.data.map((item, index) => {
            if (!item.embedding || !Array.isArray(item.embedding)) {
                throw new Error(`数据项 ${index} 中没有有效的嵌入向量`);
            }
            
            return {
                vector: item.embedding,
                dimension: item.embedding.length
            };
        });
    } catch (error) {
        console.error('API 调用失败:', error.message);
        throw error;
    }
}

// 获取token计数
async function getTokenCounts(texts) {
    try {
        const response = await axios.post(
            `${BASE_URL}/v1/token_count`,
            { input: texts },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'api-key': API_KEY
                }
            }
        );

        if (!response.data || !response.data.token_counts) {
            throw new Error('API 没有返回 token 计数');
        }

        return response.data.token_counts;
    } catch (error) {
        console.error('获取 token 计数失败:', error.message);
        throw error;
    }
}

async function main() {
    try {
        console.log('📊 StarFactory Embedding API 示例');
        console.log('🔗 API 基础 URL:', BASE_URL);
        if (DEBUG) console.log('调试模式: 开启');
        
        // 示例代码片段
        const codeSnippets = [
            "package com.starfactory.data.shared.enums;\n" +
                "\n" +
                "import lombok.AllArgsConstructor;\n" +
                "import lombok.Getter;\n" +
                "\n" +
                "@AllArgsConstructor\n" +
                "@Getter\n" +
                "public enum QueryTypeEnum {\n" +
                "\n" +
                "    PERSON(1, \"个人\"),\n" +
                "    DEPARTMENT(2, \"部门\");\n" +
                "\n" +
                "    private Integer type;\n" +
                "\n" +
                "    private String desc;\n" +
                "}\n"
        ];
        
        console.log('\n🔍 生成代码片段的嵌入向量');
        console.time('生成嵌入向量');
        const embeddings = await generateEmbeddings(codeSnippets);
        console.timeEnd('生成嵌入向量');
        
        console.log(`✅ 成功生成 ${embeddings.length} 个嵌入向量`);
        for (let i = 0; i < embeddings.length; i++) {
            console.log(`代码片段 ${i+1} - 前 3 个向量元素: [${embeddings[i].vector.slice(0, 3).join(', ')}...]`);
        }
        
        // 计算相似度
        console.log('\n🔍 计算代码片段之间的相似度');
        console.log('Java vs JavaScript 相似度:', cosineSimilarity(embeddings[0].vector, embeddings[1].vector).toFixed(4));
        console.log('Java vs Python 相似度:', cosineSimilarity(embeddings[0].vector, embeddings[2].vector).toFixed(4));
        console.log('JavaScript vs Python 相似度:', cosineSimilarity(embeddings[1].vector, embeddings[2].vector).toFixed(4));
        
        // 尝试获取 token 计数
        try {
            console.log('\n🔍 获取代码片段的 token 计数');
            console.time('获取 token 计数');
            const tokenCounts = await getTokenCounts(codeSnippets);
            console.timeEnd('获取 token 计数');
            
            console.log('✅ Token 计数:');
            for (let i = 0; i < tokenCounts.length; i++) {
                console.log(`代码片段 ${i+1}: ${tokenCounts[i]} tokens`);
            }
        } catch (error) {
            console.log('❌ 无法获取 token 计数，这可能是因为此功能在当前 API 中不可用');
        }
        
        // 示范搜索查询
        console.log('\n🔍 示范搜索查询');
        const query = "How to print hello message";
        console.log(`查询: "${query}"`);
        
        try {
            console.time('查询嵌入向量');
            const queryEmbedding = await generateEmbeddings([query]);
            console.timeEnd('查询嵌入向量');
            
            console.log('计算与代码片段的相似度:');
            const similarities = codeSnippets.map((snippet, i) => ({
                snippet: snippet.substring(0, 40) + '...',
                similarity: cosineSimilarity(queryEmbedding[0].vector, embeddings[i].vector)
            }));
            
            // 按相似度排序
            similarities.sort((a, b) => b.similarity - a.similarity);
            
            // 显示结果
            for (const result of similarities) {
                console.log(`${result.similarity.toFixed(4)} - ${result.snippet}`);
            }
        } catch (error) {
            console.log('❌ 执行搜索查询时出错:', error.message);
        }
        
    } catch (error) {
        console.error('❌ 错误:', error);
    }
}

// 运行示例
main(); 