import {
    MilvusVectorDatabase,
    StarFactoryEmbedding
} from '@code-indexer/core';
import * as path from 'path';
import * as crypto from 'crypto';
import * as fs from 'fs';

// Try to load .env file
try {
    require('dotenv').config();
} catch (error) {
    // dotenv is not required, skip if not installed
}


function generateCollectionName(codebasePath: string): string {
    const normalizedPath = path.resolve(codebasePath);
    const hash = crypto.createHash('md5').update(normalizedPath).digest('hex');
    return `code_chunks_${hash.substring(0, 8)}`;
}

/**
 * Save search results to a file in the docs directory
 */
function saveResultsToFile(searchType: string, query: string, results: any[]): string {
    const timestamp = new Date().toISOString().replace(/:/g, '-').replace(/\./g, '-');
    const sanitizedQuery = query.replace(/\s+/g, '_').substring(0, 30);
    const filename = `search_results_${searchType}_${sanitizedQuery}_${timestamp}.md`;
    const docsDir = path.join(__dirname, '../../docs/search-comparison');
    
    // Create docs directory if it doesn't exist
    if (!fs.existsSync(docsDir)) {
        fs.mkdirSync(docsDir, { recursive: true });
    }
    
    const filePath = path.join(docsDir, filename);
    
    let content = `# ${searchType} Search Results for: "${query}"\n\n`;
    content += `Search Time: ${new Date().toISOString()}\n\n`;
    content += `Total Results: ${results.length}\n\n`;
    
    if (results.length > 0) {
        results.forEach((result, index) => {
            content += `## Result ${index + 1} - Score: ${(result.score * 100).toFixed(2)}%\n\n`;
            content += `- File: ${result.document.relativePath}\n`;
            content += `- Lines: ${result.document.startLine}-${result.document.endLine}\n\n`;
            content += "```\n";
            content += result.document.content + "\n";
            content += "```\n\n";
        });
    } else {
        content += "No results found.\n";
    }
    
    fs.writeFileSync(filePath, content);
    console.log(`📝 Results saved to: ${filePath}`);
    return filePath;
}

/**
 * Example function demonstrating how to use Milvus BM25 hybrid search
 */
export async function hybridSearch(codebasePath: string, query: string) {
    console.log('🔍 Milvus BM25 Hybrid Search Example');
    console.log('====================================');
    const starFactoryApiKey = process.env.STARFACTORY_API_KEY || StarFactoryEmbedding.getDefaultApiKey();
    const starFactoryBaseURL = process.env.STARFACTORY_BASE_URL || 'http://10.142.99.29:8085';
    try {
        // 1. Configure embedding provider
        const embedding = new StarFactoryEmbedding({
            apiKey: process.env.STARFACTORY_API_KEY || starFactoryApiKey,
            baseURL: process.env.STARFACTORY_BASE_URL || starFactoryBaseURL,
            model: 'NV-Embed-v2'
        });
        
        // 2. Configure vector database with BM25 enabled
        const milvusAddress = process.env.MILVUS_ADDRESS || 'localhost:19530';
        const milvusToken = process.env.MILVUS_TOKEN;
        
        console.log(`🔌 Connecting to Milvus at: ${milvusAddress}`);
        console.log('🔧 BM25 hybrid search is enabled');
        
        const vectorDatabase = new MilvusVectorDatabase({
            address: milvusAddress,
            ...(milvusToken && { token: milvusToken }),
            // @ts-ignore: The MilvusConfig has been extended in the implementation
            enableBM25: true,
            consistencyLevel: 'Bounded',
            rankerType: 'weight',
            rankerParams: { k: 100 }
        });
        
        // 3. Generate collection name
        const collectionName = generateCollectionName(codebasePath); 
        console.log(`📚 Using collection: ${collectionName}`);
        
        // 4. Generate query embedding
        console.log(`🔎 Generating embedding for query: "${query}"`);
        const queryEmbedding = await embedding.embed(query);
        
        // 5. Execute standard vector search
        console.log('\n🔍 Executing standard vector search...');
        const standardSearchResults = await vectorDatabase.search(
            collectionName,
            queryEmbedding.vector,
            {
                topK: 10,
                threshold: 0.3
            }
        );
        
        // 6. Display standard search results
        console.log(`\n✅ Found ${standardSearchResults.length} results with standard search:`);
        
        if (standardSearchResults.length > 0) {
            standardSearchResults.forEach((result, index) => {
                console.log(`\nResult ${index + 1} - Score: ${(result.score * 100).toFixed(2)}%`);
                console.log(`ID: ${result.document.id}`);
                console.log(`File: ${result.document.relativePath}`);
                console.log(`Lines: ${result.document.startLine}-${result.document.endLine}`);
                console.log(`Preview: ${result.document.content.substring(0, 100).replace(/\n/g, ' ')}...`);
            });
        } else {
            console.log('No relevant results found.');
        }
        
        // 7. Execute hybrid search
        console.log('\n🔍 Executing hybrid search...');
        const hybridSearchResults = await vectorDatabase.hybridSearch(
            collectionName,
            queryEmbedding.vector,
            {
                topK: 10,
                threshold: 0.3,
                query: query
            }
        );
        
        // 8. Display hybrid search results
        console.log(`\n✅ Found ${hybridSearchResults.length} results with hybrid search:`);
        
        if (hybridSearchResults.length > 0) {
            hybridSearchResults.forEach((result, index) => {
                console.log(`\nResult ${index + 1} - Score: ${(result.score * 100).toFixed(2)}%`);
                console.log(`ID: ${result.document.id}`);
                console.log(`File: ${result.document.relativePath}`);
                console.log(`Lines: ${result.document.startLine}-${result.document.endLine}`);
                console.log(`Preview: ${result.document.content.substring(0, 100).replace(/\n/g, ' ')}...`);
            });
        } else {
            console.log('No relevant results found.');
        }
        
        // 9. Save results to files
        console.log('\n💾 Saving search results to files...');
        const standardResultsFile = saveResultsToFile('Standard', query, standardSearchResults);
        const hybridResultsFile = saveResultsToFile('Hybrid', query, hybridSearchResults);
        
        // 10. Compare results
        console.log('\n🔍 Comparing search results:');
        const sameResults = standardSearchResults.length === hybridSearchResults.length && 
            standardSearchResults.every((result, i) => result.document.id === hybridSearchResults[i].document.id);
        
        if (sameResults) {
            console.log('❗ NOTICE: Standard and hybrid search returned identical results!');
            console.log('This might indicate that:');
            console.log('1. The collection does not have sparse vectors properly configured');
            console.log('2. The hybridSearch method is not using text filters correctly');
            console.log('3. For this particular query, both methods find the same optimal results');
        } else {
            console.log('✅ Standard and hybrid search returned different results!');
            
            // Count how many results are different
            const standardIds = standardSearchResults.map(r => r.document.id);
            const hybridIds = hybridSearchResults.map(r => r.document.id);
            const uniqueToStandard = standardIds.filter(id => !hybridIds.includes(id));
            const uniqueToHybrid = hybridIds.filter(id => !standardIds.includes(id));
            
            console.log(`Different results: ${uniqueToStandard.length + uniqueToHybrid.length} out of ${Math.max(standardIds.length, hybridIds.length)}`);
            console.log(`Results unique to standard search: ${uniqueToStandard.length}`);
            console.log(`Results unique to hybrid search: ${uniqueToHybrid.length}`);
        }
        
    } catch (error) {
        console.error('❌ Error occurred:', error);
    }
}

// Execute the example if this file is run directly
if (require.main === module) {
    // Change these values to match your environment
    //const codebasePath = process.env.TEST_CODEBASE_PATH || path.join(__dirname, '../..');
    //const query = process.argv[2] || '用户登录和注册功能的核心逻辑';
    
    const codebasePath = '/Users/ivem/Desktop/test';
    const query = '分析用户注册功能相关代码，梳理核心链路和主要逻辑';

    hybridSearch(codebasePath, query).catch(error => {
        console.error('❌ Fatal error:', error);
        process.exit(1);
    });
} 