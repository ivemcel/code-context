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
import { EnhancedAstSplitter } from '../../packages/core/src/splitter/enhanced-ast-splitter';
import { generateCode, GPT4Client } from '../../packages/core/src/utils/gpt4-client';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

// Try to load .env file
try {
    require('dotenv').config();
} catch (error) {
    // dotenv is not required, skip if not installed
}

/**
 * Ensures that the directory exists, creating it if necessary
 */
function ensureDirectoryExists(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        console.log(`📁 Created directory: ${dirPath}`);
    }
}

/**
 * Writes search results to a file in the docs folder
 */
function writeResultsToFile(query: string, results: any[], docsPath: string): void {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const fileName = `search_results_${query.replace(/\s+/g, '_').substring(0, 30)}_${timestamp}.md`;
    const filePath = path.join(docsPath, fileName);

    let content = `# Search Results for: "${query}"\n`;
    content += `Generated on: ${new Date().toLocaleString()}\n\n`;

    if (results.length > 0) {
        content += `## Found ${results.length} results\n\n`;
        results.forEach((result, index) => {
            content += `### Result ${index + 1} - Similarity: ${(result.score * 100).toFixed(2)}%\n`;
            content += `- **File**: ${result.relativePath}\n`;
            content += `- **Language**: ${result.language}\n`;
            content += `- **Lines**: ${result.startLine}-${result.endLine}\n`;
            content += `- **Preview**:\n\`\`\`\n${result.content.substring(0, 300)}...\n\`\`\`\n\n`;
        });
    } else {
        content += "No relevant results found.\n";
    }

    fs.writeFileSync(filePath, content);
    console.log(`📄 Results saved to: ${filePath}`);
}

/**
 * Rewrites a user query using GPT-4 to make it more effective for semantic search
 * @param originalQuery The original user query
 * @returns The rewritten query optimized for semantic search
 */
async function rewriteQueryWithGPT4(originalQuery: string): Promise<string> {
    console.log(`🤖 Rewriting query with GPT-4: "${originalQuery}"`);
    
    try {
        const prompt = `
I need to improve this query for semantic code search in a codebase. 
Please rewrite it to be more effective at finding relevant code by:
1. Extracting key technical terms and concepts as keywords
2. Preserving the original user intent completely
3. Including likely function/class names and code patterns
4. Adding relevant technical synonyms where appropriate
5. 生成的问题包括中英文并以 JSON 格式返回，示例如下：{
"originalQuery": "分析用户注册功能相关代码，梳理核心链路和主要逻辑",
"rewrittenQuery": "用户注册功能 核心链路 主要逻辑"}
Original query: "${originalQuery}"`;

        const rewrittenQuery = await generateCode(prompt, 'gpt-4', 300, 0);
        
        // Try to parse the response as JSON
        try {
            const jsonResponse = JSON.parse(rewrittenQuery.trim());
            if (jsonResponse && jsonResponse.rewrittenQuery) {
                console.log(`🔄 Rewritten query: "${jsonResponse.rewrittenQuery}"`);
                return jsonResponse.rewrittenQuery;
            }
        } catch (parseError) {
            console.warn('⚠️ Failed to parse response as JSON, falling back to text extraction');
        }
        
        // Fallback to the previous text extraction method
        let cleanQuery = rewrittenQuery.trim();
        if (cleanQuery.startsWith('"') && cleanQuery.endsWith('"')) {
            cleanQuery = cleanQuery.substring(1, cleanQuery.length - 1);
        }
        
        // Extract just the first paragraph if there are multiple
        const paragraphs = cleanQuery.split('\n\n');
        cleanQuery = paragraphs[0].trim();
        
        console.log(`🔄 Rewritten query: "${cleanQuery}"`);
        return cleanQuery;
    } catch (error) {
        console.error('❌ Error rewriting query with GPT-4:', error);
        console.log('⚠️ Using original query instead.');
        return originalQuery;
    }
}

async function main() {
    console.log('🚀 CodeIndexer Real Usage Example');
    console.log('===============================');
    
    // Only run mainReindexByChange if explicitly requested
    if (process.env.REINDEX_TEST === '1') {
        console.log('⚠️ REINDEX_TEST is set, but mainReindexByChange is not implemented in this file.');
        console.log('⚠️ Please run reindexByChange.ts separately if needed.');
        return;
    }

    try {
        // 1. Configure Embedding Provider
        // ------------------------------------
        
        // Option A: StarFactory (适合中英文多语言文本)
        // const embedding = new StarFactoryEmbedding({
        //     apiKey: process.env.STARFACTORY_API_KEY || StarFactoryEmbedding.getDefaultApiKey(), // 默认API密钥
        //     baseURL: process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085',
        //     model: 'NV-Embed-v2' // 默认模型
        // });
        // console.log('🔧 Using StarFactory embedding model');
        // console.log('🔗 API Base URL:', process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085');
        
        
        // Option B: Ollama (local model)
        const embedding = new OllamaEmbedding({
            model: "mxbai-embed-large" // Make sure you have pulled this model with `ollama pull mxbai-embed-large`
        });
        console.log('🔧 Using Ollama embedding model');
        

        /*
        // Option C: OpenAI
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('OPENAI_API_KEY environment variable is not set.');
        }
        const embedding = new OpenAIEmbedding({
            apiKey: process.env.OPENAI_API_KEY,
            model: 'text-embedding-3-small'
        });
        console.log('🔧 Using OpenAI embedding model');
        */

        /*
        // Option D: VoyageAI
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

        // const vectorDatabase = new MilvusRestfulVectorDatabase({
        //     address: milvusAddress,
        //     ...(milvusToken && { token: milvusToken })
        // });

        // 3. Create CodeIndexer instance
        // ----------------------------------
        //const codeSplitter = new AstCodeSplitter(0, 0);
        const codeSplitter = new EnhancedAstSplitter(0, 0);

        // Using AstCodeSplitter instead of EnhancedAstSplitter
        const indexer = new CodeIndexer({
            embedding, // Pass the configured embedding provider
            vectorDatabase,
            codeSplitter,
            supportedExtensions: ['.java']
            //supportedExtensions: ['.ts', '.js', '.py', '.java', '.cpp', '.go', '.rs']
        });


        // 4. Index the codebase
        // -------------------------
        // console.log('\n📖 Starting to index codebase...');
        // //const codebasePath = path.join(__dirname, '../..'); // Index the entire monorepo
        //const codebasePath = "/Users/ivem/WebstormProjects/code-context/packages/core";
        
        // The collection name is now derived internally from the codebasePath
        // const collectionName = indexer.getCollectionName(codebasePath); // 私有方法，不能外部调用
        // console.log(`ℹ️  Using collection: ${collectionName}`);

        // // Check if index already exists and clear if needed
        // const hasExistingIndex = await indexer.hasIndex(codebasePath);
        // if (hasExistingIndex) {
        //     console.log('🗑️  Existing index found, clearing it first...');
        //     await indexer.clearIndex(codebasePath);
        // }

        // // Index with progress tracking - API has changed
        // const indexStats = await indexer.indexCodebase(codebasePath, (progress) => {
        //     console.log(`   [${progress.phase}] ${progress.percentage.toFixed(2)}%`);
        // });
        // console.log(`\n📊 Indexing stats: ${indexStats.indexedFiles} files, ${indexStats.totalChunks} code chunks`);


        // 5. Perform semantic search
        // ----------------------------
        console.log('\n🔍 Performing semantic search...');
        const queries = [
            // '什么接口是根据按日期范围查询用户指标数据的？', 
            // '获取详情数据统计接口用到了什么方法',
            //'给埋点日志上报接口及其方法添加日志',
            '分析用户注册功能相关代码，梳理核心链路和主要逻辑',
            '分析用户登录功能相关代码，梳理核心链路和主要逻辑',
            //'分析aiMetricsDataReporting接口核心链路和主要逻辑',
            //'中文：分析aiMetricsDataReporting接口核心链路和主要逻辑；英文：Analyze the core workflow and primary logic of the aiMetricsDataReporting API.',
            //'用户注册,register,signup,注册功能,用户创建,账户注册,注册接口,用户管理,创建用户',
            //'login,logout,authentication,authorization,username,password,token,security,auth,captcha,session,jwt,verification,signin,register,account'
            //'Analyze user registration and login functionality, organize core pathways and main logic',
            //'总结LoginController中的register方法逻辑'
        ];

        // Ensure docs directory exists
        const docsPath = path.join(__dirname, '../../docs');
        ensureDirectoryExists(docsPath);
        console.log(`\n📁 Results will be saved to: ${docsPath}`);

        //const codebasePath = "/Users/ivem/IdeaProjects/star-factory";
        const codebasePath = "/Users/ivem/Desktop/rag-codebase";
        //const codebasePath = "/Users/ivem/IdeaProjects/star-factory/star-factory-user";

        for (const originalQuery of queries) {
            console.log(`\n🔎 Original Search Query: "${originalQuery}"`);
            
            // Rewrite the query using GPT-4
            //const enhancedQuery = await rewriteQueryWithGPT4(originalQuery);
            
            // Perform semantic search with the enhanced query
            //console.log(`\n🔍 Searching with enhanced query: "${enhancedQuery}"`);
            //const results = await indexer.semanticSearch(codebasePath, enhancedQuery, 20, 0.3);
            const results = await indexer.semanticSearch(codebasePath, originalQuery, 20, 0.3);

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

            // Write results to file with both original and enhanced queries
            const queryInfo = `Original: "${originalQuery}"`;
            writeResultsToFile(queryInfo, results, docsPath);
        }
        console.log('\n🎉 Example completed successfully!');

    } catch (error) {
        console.error('❌ Error occurred:', error);
        // Add specific error handling for different services if needed
        process.exit(1);
    }
}


export { main };

// Run the main function if this file is executed directly
if (require.main === module) {
    main().catch(error => {
        console.error('❌ Fatal error:', error);
        process.exit(1);
    });
}