import { CodeIndexer } from '../../packages/core/src';
import { MilvusVectorDatabase } from '../../packages/core/src/vectordb';
import { StarFactoryEmbedding } from '../../packages/core/src/embedding';
import path from 'path';
import * as dotenv from 'dotenv';
import * as fs from 'fs';

// Load environment variables from .env file
dotenv.config();

// 支持的文件扩展名
const SUPPORTED_EXTENSIONS = [
    // Programming languages
    '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.m', '.mm',
    // Text and markup files
    '.md', '.markdown', '.ipynb',
    '.txt',  '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
    '.css', '.scss', '.less', '.sql', '.sh', '.bash', '.env'
];

async function main() {
    // Configuration
    const codebasePath = "/Users/ivem/Desktop/test";
    const forceReindex = true;
    
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
        
        // Initialize code indexer with sparse vector support
        const indexer = new CodeIndexer({
            vectorDatabase: vectorDb,
            embedding,
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

            // Index the codebase with progress tracking
            await indexer.indexCodebase(codebasePath, (progress) => {
                console.log(`${progress.phase} - ${progress.percentage}%`);
            });
            
            console.log('✅ Indexing complete!');
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
                5, 
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