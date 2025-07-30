import { CodeIndexer } from '../../packages/core/src';
import { MilvusVectorDatabase } from '../../packages/core/src/vectordb';
import { StarFactoryEmbedding } from '../../packages/core/src/embedding';
import path from 'path';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import { generateCode } from '../../packages/core/src/utils/gpt4-client';
import { EnhancedAstSplitter } from '@code-indexer/core/src/splitter/enhanced-ast-splitter';

// Load environment variables from .env file
dotenv.config();

// 支持的文件扩展名
const SUPPORTED_EXTENSIONS = [
    // Programming languages
    //'.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h', '.hpp',
    //'.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.m', '.mm',
    '.java',
    // Text and markup files
    '.md', '.markdown', '.ipynb',
    //'.txt',  '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
    //'.css', '.scss', '.less', '.sql', '.sh', '.bash', '.env'
];

async function rewriteQuery(text: string): Promise<string[]> {
    // 1. 尝试使用GPT-4提取关键词
    try {
        console.log('尝试使用GPT-4提取中文关键词...');
        const prompt = `
请从以下查询中提取重要的关键词、短语和代码元素，以便用于代码库向量搜索。
只返回JSON数组格式的关键词列表，不要有任何其他文字。

关键词应包括：
1. 中文技术术语及其可能的英文对应（如"用户认证"和"user authentication"）
2. 代码中的关键元素（类名、方法名、变量名、API名称等）
3. 编程概念（如"接口"/"interface"、"继承"/"inheritance"等）
4. 中英文混合的技术术语
5. 查询中的单词和短语，无论是中文还是英文
6. 潜在的相关技术术语（即使查询中未直接提及）

特别注意：
- 保留所有可能的驼峰命名法标识符（如LoginService, getUserAuth）
- 提取查询中明确的代码标识符以及隐含的可能代码实现
- 对中文描述的功能，尝试推断可能的英文代码表示方式

查询文本: "${text}"

返回格式示例: ["关键词1", "getUserInfo", "用户验证", "authentication", "LoginService", ...]
`;
        const response = await generateCode(prompt, 'gpt-4', 1000, 0);
        
        try {
            // 处理可能带有Markdown代码块的响应
            let jsonText = response.trim();
            
            // 去除Markdown代码块标记
            if (jsonText.startsWith('```')) {
                const endMarkdownIndex = jsonText.indexOf('```', 3);
                if (endMarkdownIndex !== -1) {
                    // 完整的代码块，去除开始和结束标记
                    jsonText = jsonText.substring(jsonText.indexOf('\n') + 1, endMarkdownIndex).trim();
                } else {
                    // 只有开始标记，去除它
                    jsonText = jsonText.substring(jsonText.indexOf('\n') + 1).trim();
                }
            }
            
            // 尝试解析处理后的JSON
            const keywords = JSON.parse(jsonText);
            if (Array.isArray(keywords) && keywords.length > 0) {
                console.log('GPT-4提取的关键词:', keywords);
                return keywords;
            }
        } catch (parseError) {
            console.warn('无法解析GPT-4返回的关键词:', parseError);
            console.log('原始响应:', response);
        }
    } catch (error) {
        console.warn('使用GPT-4提取关键词失败，降级到其他方法:', error);
    }
    return [text]; // 如果不能重写，就返回原始查询
}

async function main() {
    // Configuration
    const codebasePath = "/Users/ivem/IdeaProjects/star-factory-hybrid-search";
    //const codebasePath = "/Users/ivem/Desktop/test";

    const forceReindex = false;
    
    console.log(`🚀 Starting hybrid search demo on codebase: ${codebasePath}`);

    try {
        // Initialize vector database with sparse vector support
        const vectorDb = new MilvusVectorDatabase({
            address: process.env.MILVUS_ADDRESS || 'http://localhost:19530',
            username: process.env.MILVUS_USERNAME,
            password: process.env.MILVUS_PASSWORD,
            token: process.env.MILVUS_TOKEN || 'root:Milvus',
            enableSparseVectors: true, // Enable sparse vector support
        });
        
        // 使用StarFactoryEmbedding替代OpenAIEmbedding，根据实际环境配置URL
        const embedding = new StarFactoryEmbedding({
            apiKey: process.env.STARFACTORY_API_KEY || StarFactoryEmbedding.getDefaultApiKey(),
            // 使用本地地址或环境变量中的地址
            baseURL: process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085',
            model: 'NV-Embed-v2' // 使用默认模型
        });
        
        console.log(`🔌 Using embedding provider: ${embedding.getProvider()}`);
        console.log(`📏 Vector dimension: ${embedding.getDimension()}`);
        console.log(`🌐 API endpoint: ${process.env.STARFACTORY_BASE_URL || 'http://localhost:8000'}`);
        
        // 使用增强型AST分割器，支持保留注释
        const codeSplitter = new EnhancedAstSplitter(0, 0);

        // Initialize code indexer with sparse vector support
        const indexer = new CodeIndexer({
            vectorDatabase: vectorDb,
            embedding,
            codeSplitter,
            enableSparseVectors: true, // Enable sparse vector support
            supportedExtensions: SUPPORTED_EXTENSIONS // 使用更多文件扩展名
        });
        
        // 检查是否需要重建索引
        const hasIndex = await indexer.hasIndex(codebasePath);
        
        if (hasIndex && !forceReindex) {
            console.log('✅ Index already exists, proceeding to search');
        } else {
            if (hasIndex) {
                console.log('🗑️  Clearing existing index...');
                await indexer.clearIndex(codebasePath);
            }
            const startTime = new Date();
            console.log(`🕒 indexCodebase开始时间: ${startTime.toLocaleString()}`);
            // Index the codebase with progress tracking
            await indexer.indexCodebase(codebasePath, (progress) => {
                console.log(`${progress.phase} - ${progress.percentage}%`);
            });

            console.log('✅ Indexing complete!');
            const endTime = new Date();
            console.log(`🕒 indexCodebase结束时间: ${endTime.toLocaleString()}`);
            console.log(`🕒 indexCodebase耗时: ${endTime.getTime() - startTime.getTime()}ms`);
        }
        
        // Execute search queries
        const queries = [
            '分析用户注册功能相关代码，梳理核心链路和主要逻辑',
            '分析用户登录功能相关代码，梳理核心链路和主要逻辑',
            '分析aiMetricsDataReporting接口核心链路和主要逻辑',
        ];
        
        for (const query of queries) {
            console.log(`\n🔍 Searching for: "${query}"`);
            const results = await indexer.semanticSearch(
                codebasePath, 
                query, 
                30, 
                0.5
            );
            
            console.log(`Found ${results.length} results:`);
            results.forEach((result, i) => {
                console.log(`\n${i + 1}. ${result.relativePath}:${result.startLine}-${result.endLine} (Score: ${result.score.toFixed(4)})`);
                console.log(`${result.content.substring(0, 150)}...`);
            });
            
            // 保存结果到markdown文件
            await saveResultsToMarkdown(query, results);
        }
        
    } catch (error) {
        console.error('❌ Error:', error);
    }
}

/**
 * 将搜索结果保存到Markdown文件
 * @param query 搜索查询
 * @param results 搜索结果
 */
async function saveResultsToMarkdown(query: string, results: any[]) {
    // 创建一个安全的文件名
    const safeFilename = query.replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '_');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19) + 'Z';
    const filename = `search_results_Original:_"${safeFilename}_${timestamp}.md`;
    const outputDir = path.join(process.cwd(), 'docs');
    
    // 确保输出目录存在
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    const outputPath = path.join(outputDir, filename);
    
    let markdown = `# 搜索结果: "${query}"\n\n`;
    markdown += `搜索时间: ${new Date().toLocaleString()}\n\n`;
    markdown += `共找到 ${results.length} 个结果\n\n`;
    
    // 添加每个搜索结果
    results.forEach((result, i) => {
        markdown += `## 结果 ${i + 1}\n\n`;
        markdown += `- **文件路径**: ${result.relativePath}\n`;
        markdown += `- **位置**: 第 ${result.startLine}-${result.endLine} 行\n`;
        markdown += `- **相关度得分**: ${result.score.toFixed(4)}\n\n`;
        markdown += `### 代码内容:\n\n`;
        markdown += '```\n';
        markdown += result.content;
        markdown += '\n```\n\n';
    });
    
    // 写入文件
    fs.writeFileSync(outputPath, markdown);
    console.log(`✅ 搜索结果已保存到: ${outputPath}`);
}

main(); 