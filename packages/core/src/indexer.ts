import {
    Splitter,
    CodeChunk,
    AstCodeSplitter
} from './splitter';
import {
    Embedding,
    EmbeddingVector,
    OpenAIEmbedding
} from './embedding';
import {
    VectorDatabase,
    VectorDocument,
    VectorSearchResult
} from './vectordb';
import { SemanticSearchResult } from './types';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { FileSynchronizer } from './sync/synchronizer';
import { generateCode } from './utils/gpt4-client';

const DEFAULT_SUPPORTED_EXTENSIONS = [
    // Programming languages
    '.java',
//    '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h', '.hpp',
//    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.m', '.mm',
    // Text and markup files
    // '.md', '.markdown', '.ipynb',
    // '.txt',  '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
    // '.css', '.scss', '.less', '.sql', '.sh', '.bash', '.env'
];

const DEFAULT_IGNORE_PATTERNS = [
    // Common build output and dependency directories
    'node_modules/**',
    'dist/**',
    'build/**',
    'out/**',
    'target/**',
    'coverage/**',
    '.nyc_output/**',

    // IDE and editor files
    '.vscode/**',
    '.idea/**',
    '*.swp',
    '*.swo',

    // Version control
    '.git/**',
    '.svn/**',
    '.hg/**',

    // Cache directories
    '.cache/**',
    '__pycache__/**',
    '.pytest_cache/**',

    // Logs and temporary files
    'logs/**',
    'tmp/**',
    'temp/**',
    '*.log',

    // Environment and config files
    '.env',
    '.env.*',
    '*.local',

    // Minified and bundled files
    '*.min.js',
    '*.min.css',
    '*.min.map',
    '*.bundle.js',
    '*.bundle.css',
    '*.chunk.js',
    '*.vendor.js',
    '*.polyfills.js',
    '*.runtime.js',
    '*.map', // source map files

    '.specstory/**',
    '.star-factory/**',
    '.cursor/**',
    '/docs/**',
    '/test/**',
    // 所有.开头的
    '.*/**'

];

export interface CodeIndexerConfig {
    embedding?: Embedding;
    vectorDatabase?: VectorDatabase;
    codeSplitter?: Splitter;
    supportedExtensions?: string[];
    ignorePatterns?: string[];
    enableSparseVectors?: boolean; // Add option to enable sparse vector indexing
}

export class CodeIndexer {
    private embedding: Embedding;
    private vectorDatabase: VectorDatabase;
    private codeSplitter: Splitter;
    private supportedExtensions: string[];
    private ignorePatterns: string[];
    private synchronizers = new Map<string, FileSynchronizer>();
    private enableSparseVectors: boolean; // Flag for sparse vector support

    constructor(config: CodeIndexerConfig = {}) {
        // Initialize services
        this.embedding = config.embedding || new OpenAIEmbedding({
            apiKey: process.env.OPENAI_API_KEY || 'your-openai-api-key',
            model: 'text-embedding-3-small',
            ...(process.env.OPENAI_BASE_URL && { baseURL: process.env.OPENAI_BASE_URL })
        });

        if (!config.vectorDatabase) {
            throw new Error('VectorDatabase is required. Please provide a vectorDatabase instance in the config.');
        }
        this.vectorDatabase = config.vectorDatabase;

        this.codeSplitter = config.codeSplitter || new AstCodeSplitter(2500, 300);

        this.supportedExtensions = config.supportedExtensions || DEFAULT_SUPPORTED_EXTENSIONS;
        this.ignorePatterns = config.ignorePatterns || DEFAULT_IGNORE_PATTERNS;
        
        // Initialize sparse vector support
        this.enableSparseVectors = config.enableSparseVectors || false;
        
        // If using MilvusVectorDatabase, pass the enableSparseVectors option
        if (this.enableSparseVectors && this.vectorDatabase.constructor.name.includes('Milvus')) {
            // Update the vector database configuration if it's a Milvus database
            const milvusDb = this.vectorDatabase as any;
            if (milvusDb && milvusDb.config) {
                milvusDb.config.enableSparseVectors = true;
            }
        }
    }

    /**
     * Generate collection name based on codebase path
     */
    private getCollectionName(codebasePath: string): string {
        const normalizedPath = path.resolve(codebasePath);
        const hash = crypto.createHash('md5').update(normalizedPath).digest('hex');
        return `code_chunks_${hash.substring(0, 8)}`;
    }

    /**
     * Index entire codebase
     * @param codebasePath Codebase path
     * @param progressCallback Optional progress callback function
     * @returns Indexing statistics
     */
    async indexCodebase(
        codebasePath: string,
        progressCallback?: (progress: { phase: string; current: number; total: number; percentage: number }) => void
    ): Promise<{ indexedFiles: number; totalChunks: number }> {
        console.log(`🚀 Starting to index codebase: ${codebasePath}`);

        // 1. Check and prepare vector collection
        progressCallback?.({ phase: 'Preparing collection...', current: 0, total: 100, percentage: 0 });
        await this.prepareCollection(codebasePath);

        // 2. Recursively traverse codebase to get all supported files
        progressCallback?.({ phase: 'Scanning files...', current: 5, total: 100, percentage: 5 });
        const codeFiles = await this.getCodeFiles(codebasePath);
        console.log(`📁 Found ${codeFiles.length} code files`);

        if (codeFiles.length === 0) {
            progressCallback?.({ phase: 'No files to index', current: 100, total: 100, percentage: 100 });
            return { indexedFiles: 0, totalChunks: 0 };
        }

        // 3. Process each file with streaming chunk processing
        // Reserve 10% for preparation, 90% for actual indexing
        const indexingStartPercentage = 10;
        const indexingEndPercentage = 100;
        const indexingRange = indexingEndPercentage - indexingStartPercentage;

        const result = await this.processFileList(
            codeFiles,
            codebasePath,
            (filePath, fileIndex, totalFiles) => {
                // Calculate progress percentage
                const progressPercentage = indexingStartPercentage + (fileIndex / totalFiles) * indexingRange;

                console.log(`📊 Processed ${fileIndex}/${totalFiles} files`);
                progressCallback?.({
                    phase: `Processing files (${fileIndex}/${totalFiles})...`,
                    current: fileIndex,
                    total: totalFiles,
                    percentage: Math.round(progressPercentage)
                });
            }
        );

        console.log(`✅ Codebase indexing completed! Processed ${result.processedFiles} files in total, generated ${result.totalChunks} code chunks`);

        progressCallback?.({
            phase: 'Indexing complete!',
            current: result.processedFiles,
            total: codeFiles.length,
            percentage: 100
        });

        return {
            indexedFiles: result.processedFiles,
            totalChunks: result.totalChunks
        };
    }

    async reindexByChange(
        codebasePath: string,
        progressCallback?: (progress: { phase: string; current: number; total: number; percentage: number }) => void
    ): Promise<{ added: number, removed: number, modified: number }> {
        const collectionName = this.getCollectionName(codebasePath);
        const synchronizer = this.synchronizers.get(collectionName);

        if (!synchronizer) {
            // To be safe, let's initialize if it's not there.
            const newSynchronizer = new FileSynchronizer(codebasePath, this.ignorePatterns);
            await newSynchronizer.initialize();
            this.synchronizers.set(collectionName, newSynchronizer);
        }

        const currentSynchronizer = this.synchronizers.get(collectionName)!;

        progressCallback?.({ phase: 'Checking for file changes...', current: 0, total: 100, percentage: 0 });
        const { added, removed, modified } = await currentSynchronizer.checkForChanges();
        const totalChanges = added.length + removed.length + modified.length;

        if (totalChanges === 0) {
            progressCallback?.({ phase: 'No changes detected', current: 100, total: 100, percentage: 100 });
            console.log('✅ No file changes detected.');
            return { added: 0, removed: 0, modified: 0 };
        }

        console.log(`🔄 Found changes: ${added.length} added, ${removed.length} removed, ${modified.length} modified.`);

        let processedChanges = 0;
        const updateProgress = (phase: string) => {
            processedChanges++;
            const percentage = Math.round((processedChanges / (removed.length + modified.length + added.length)) * 100);
            progressCallback?.({ phase, current: processedChanges, total: totalChanges, percentage });
        };

        // Handle removed files
        for (const file of removed) {
            await this.deleteFileChunks(collectionName, file);
            updateProgress(`Removed ${file}`);
        }

        // Handle modified files
        for (const file of modified) {
            await this.deleteFileChunks(collectionName, file);
            updateProgress(`Deleted old chunks for ${file}`);
        }

        // Handle added and modified files
        const filesToIndex = [...added, ...modified].map(f => path.join(codebasePath, f));

        if (filesToIndex.length > 0) {
            await this.processFileList(
                filesToIndex,
                codebasePath,
                (filePath, fileIndex, totalFiles) => {
                    updateProgress(`Indexed ${filePath} (${fileIndex}/${totalFiles})`);
                }
            );
        }

        console.log(`✅ Re-indexing complete. Added: ${added.length}, Removed: ${removed.length}, Modified: ${modified.length}`);
        progressCallback?.({ phase: 'Re-indexing complete!', current: totalChanges, total: totalChanges, percentage: 100 });

        return { added: added.length, removed: removed.length, modified: modified.length };
    }

    private async deleteFileChunks(collectionName: string, relativePath: string): Promise<void> {
        const results = await this.vectorDatabase.query(
            collectionName,
            `relativePath == "${relativePath}"`,
            ['id']
        );

        if (results.length > 0) {
            const ids = results.map(r => r.id as string).filter(id => id);
            if (ids.length > 0) {
                await this.vectorDatabase.delete(collectionName, ids);
                console.log(`Deleted ${ids.length} chunks for file ${relativePath}`);
            }
        }
    }

    /**
     * Semantic search
     * @param codebasePath Codebase path to search in
     * @param query Search query
     * @param topK Number of results to return
     * @param threshold Similarity threshold
     */
    async semanticSearch(codebasePath: string, query: string, topK: number = 5, threshold: number = 0.5): Promise<SemanticSearchResult[]> {
        console.log(`🔍 Executing semantic search: "${query}" in ${codebasePath}`);

        // 1. Generate query vector
        const queryEmbedding: EmbeddingVector = await this.embedding.embed(query);

        // 2. Search in vector database
        let searchResults: VectorSearchResult[];
        
        // Check if hybrid search is available and enabled
        if (this.enableSparseVectors && typeof this.vectorDatabase.hybridSearch === 'function') {
            console.log(`🔍 Using hybrid search with sparse vectors for: "${query}"`);
            searchResults = await this.vectorDatabase.hybridSearch(
                this.getCollectionName(codebasePath),
                query, // Text query for sparse vector search
                queryEmbedding.vector, // Dense vector for semantic search
                { topK, threshold }
            );
        } else {
            // Fall back to standard vector search
            searchResults = await this.vectorDatabase.search(
                this.getCollectionName(codebasePath),
                queryEmbedding.vector,
                { topK, threshold }
            );
        }

        // 3. Convert to semantic search result format
        const results: SemanticSearchResult[] = searchResults.map(result => ({
            content: result.document.content,
            relativePath: result.document.relativePath,
            startLine: result.document.startLine,
            endLine: result.document.endLine,
            language: result.document.metadata.language || 'unknown',
            score: result.score
        }));

        console.log(`✅ Found ${results.length} relevant results with ${this.enableSparseVectors ? 'hybrid' : 'semantic'} search`);
        return results;
    }

    /**
     * Check if index exists for codebase
     * @param codebasePath Codebase path to check
     * @returns Whether index exists
     */
    async hasIndex(codebasePath: string): Promise<boolean> {
        const collectionName = this.getCollectionName(codebasePath);
        return await this.vectorDatabase.hasCollection(collectionName);
    }

    /**
     * Clear index
     * @param codebasePath Codebase path to clear index for
     * @param progressCallback Optional progress callback function
     */
    async clearIndex(
        codebasePath: string,
        progressCallback?: (progress: { phase: string; current: number; total: number; percentage: number }) => void
    ): Promise<void> {
        console.log(`🧹 Cleaning index data for ${codebasePath}...`);

        progressCallback?.({ phase: 'Checking existing index...', current: 0, total: 100, percentage: 0 });

        const collectionName = this.getCollectionName(codebasePath);
        console.log(`🔍 collectionName: ${collectionName}`);
        
        const collectionExists = await this.vectorDatabase.hasCollection(collectionName);
        console.log(`🔍 collectionExists: ${collectionExists}`);
        progressCallback?.({ phase: 'Removing index data...', current: 50, total: 100, percentage: 50 });

        if (collectionExists) {
            await this.vectorDatabase.dropCollection(collectionName);
        }

        // Delete snapshot file
        await FileSynchronizer.deleteSnapshot(codebasePath);

        progressCallback?.({ phase: 'Index cleared', current: 100, total: 100, percentage: 100 });
        console.log('✅ Index data cleaned');
    }

    /**
     * Update ignore patterns (merges with default patterns)
     * @param ignorePatterns Array of ignore patterns to add to defaults
     */
    updateIgnorePatterns(ignorePatterns: string[]): void {
        // Merge with default patterns, avoiding duplicates
        const mergedPatterns = [...DEFAULT_IGNORE_PATTERNS, ...ignorePatterns];
        this.ignorePatterns = [...new Set(mergedPatterns)]; // Remove duplicates
        console.log(`🚫 Updated ignore patterns: ${ignorePatterns.length} from .gitignore + ${DEFAULT_IGNORE_PATTERNS.length} default = ${this.ignorePatterns.length} total patterns`);
    }

    /**
     * Reset ignore patterns to defaults only
     */
    resetIgnorePatternsToDefaults(): void {
        this.ignorePatterns = [...DEFAULT_IGNORE_PATTERNS];
        console.log(`🔄 Reset ignore patterns to defaults: ${this.ignorePatterns.length} patterns`);
    }

    /**
     * Update embedding instance
     * @param embedding New embedding instance
     */
    updateEmbedding(embedding: Embedding): void {
        this.embedding = embedding;
        console.log(`🔄 Updated embedding provider: ${embedding.getProvider()}`);
    }

    /**
     * Update vector database instance
     * @param vectorDatabase New vector database instance
     */
    updateVectorDatabase(vectorDatabase: VectorDatabase): void {
        this.vectorDatabase = vectorDatabase;
        console.log(`🔄 Updated vector database`);
    }

    /**
     * Update splitter instance
     * @param splitter New splitter instance
     */
    updateSplitter(splitter: Splitter): void {
        this.codeSplitter = splitter;
        console.log(`🔄 Updated splitter instance`);
    }

    /**
     * Prepare vector collection
     */
    private async prepareCollection(codebasePath: string): Promise<void> {
        // Create new collection
        const collectionName = this.getCollectionName(codebasePath);

        // For Ollama embeddings, ensure dimension is detected before creating collection
        if (this.embedding.getProvider() === 'Ollama' && typeof (this.embedding as any).initializeDimension === 'function') {
            await (this.embedding as any).initializeDimension();
        }

        const dimension = this.embedding.getDimension();
        await this.vectorDatabase.createCollection(collectionName, dimension, `Code chunk vector storage collection for codebase: ${codebasePath}`);
        console.log(`✅ Collection ${collectionName} created successfully (dimension: ${dimension})`);
    }

    /**
     * Recursively get all code files in the codebase
     */
    private async getCodeFiles(codebasePath: string): Promise<string[]> {
        const files: string[] = [];

        const traverseDirectory = async (currentPath: string) => {
            const entries = await fs.promises.readdir(currentPath, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(currentPath, entry.name);

                // Check if path matches ignore patterns
                if (this.matchesIgnorePattern(fullPath, codebasePath)) {
                    continue;
                }

                if (entry.isDirectory()) {
                    // Skip common ignored directories
                    if (!this.shouldIgnoreDirectory(entry.name)) {
                        await traverseDirectory(fullPath);
                    }
                } else if (entry.isFile()) {
                    const ext = path.extname(entry.name);
                    if (this.supportedExtensions.includes(ext)) {
                        files.push(fullPath);
                    }
                }
            }
        };

        await traverseDirectory(codebasePath);
        return files;
    }

    /**
     * Determine whether directory should be ignored
     */
    private shouldIgnoreDirectory(dirName: string): boolean {
        const ignoredDirs = [
            'node_modules', '.git', '.svn', '.hg', 'build', 'dist', 'out',
            'target', '.vscode', '.idea', '__pycache__', '.pytest_cache',
            'coverage', '.nyc_output', 'logs', 'tmp', 'temp'
        ];
        return ignoredDirs.includes(dirName) || dirName.startsWith('.');
    }

    /**
 * Process a list of files with streaming chunk processing
 * Each chunk will be combined with its source document for embedding
 * @param filePaths Array of file paths to process
 * @param codebasePath Base path for the codebase
 * @param onFileProcessed Callback called when each file is processed
 * @returns Object with processed file count and total chunk count
 */
    private async processFileList(
        filePaths: string[],
        codebasePath: string,
        onFileProcessed?: (filePath: string, fileIndex: number, totalFiles: number) => void
    ): Promise<{ processedFiles: number; totalChunks: number }> {
        const EMBEDDING_BATCH_SIZE = Math.max(1, parseInt(process.env.EMBEDDING_BATCH_SIZE || '100', 10));
        console.log(`🔧 Using EMBEDDING_BATCH_SIZE: ${EMBEDDING_BATCH_SIZE}`);

        let chunkBuffer: Array<{ chunk: CodeChunk; codebasePath: string }> = [];
        let processedFiles = 0;
        let totalChunks = 0;

        for (let i = 0; i < filePaths.length; i++) {
            const filePath = filePaths[i];

            try {
                const content = await fs.promises.readFile(filePath, 'utf-8');
                const language = this.getLanguageFromExtension(path.extname(filePath));
                const chunks = await this.codeSplitter.split(content, language, filePath);

                // Log files with many chunks or large content
                if (chunks.length > 50) {
                    console.warn(`⚠️  File ${filePath} generated ${chunks.length} chunks (${Math.round(content.length / 1024)}KB)`);
                } else if (content.length > 100000) {
                    console.log(`📄 Large file ${filePath}: ${Math.round(content.length / 1024)}KB -> ${chunks.length} chunks`);
                }

                // Add chunks to buffer
                for (const chunk of chunks) {
                    chunkBuffer.push({ chunk, codebasePath });
                    totalChunks++;

                    // Process batch when buffer reaches EMBEDDING_BATCH_SIZE
                    if (chunkBuffer.length >= EMBEDDING_BATCH_SIZE) {
                        try {
                            await this.processChunkBuffer(chunkBuffer);
                        } catch (error) {
                            console.error(`❌ Failed to process chunk batch: ${error}`);
                        } finally {
                            chunkBuffer = []; // Always clear buffer, even on failure
                        }
                    }
                }

                processedFiles++;
                onFileProcessed?.(filePath, i + 1, filePaths.length);

            } catch (error) {
                console.warn(`⚠️  Skipping file ${filePath}: ${error}`);
            }
        }

        // Process any remaining chunks in the buffer
        if (chunkBuffer.length > 0) {
            console.log(`📝 Processing final batch of ${chunkBuffer.length} chunks`);
            try {
                await this.processChunkBuffer(chunkBuffer);
            } catch (error) {
                console.error(`❌ Failed to process final chunk batch: ${error}`);
            }
        }

        return { processedFiles, totalChunks };
    }

    /**
     * Generate semantic-rich Chinese comments for code chunks using GPT-4
     * @param chunks Array of code chunks
     * @param codebasePath Base path of the codebase
     * @returns Enhanced chunks with comments in their content
     */
    private async generateChunkComments(chunks: CodeChunk[], codebasePath: string): Promise<CodeChunk[]> {
        console.log(`🤖 Generating semantic-rich Chinese comments for ${chunks.length} chunks...`);
        const startTime = Date.now();
        
        try {
            // Configuration for batch processing
            const BATCH_SIZE = 10; // Process N chunks in a single API call
            const MAX_PARALLEL_BATCHES = 20; // Max parallel API calls
            console.log(`📊 Using batch size: ${BATCH_SIZE}, max parallel batches: ${MAX_PARALLEL_BATCHES}`);
            
            // Create batches of chunks
            const batches: CodeChunk[][] = [];
            for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
                batches.push(chunks.slice(i, Math.min(i + BATCH_SIZE, chunks.length)));
            }
            
            // Create a result array pre-filled with original chunks as fallback
            const resultChunks = [...chunks];
            
            // Process batches with limited parallelism
            for (let i = 0; i < batches.length; i += MAX_PARALLEL_BATCHES) {
                const currentBatches = batches.slice(i, i + MAX_PARALLEL_BATCHES);
                const batchStartTime = Date.now();
                const batchPromises = currentBatches.map((batch, batchIndex) => 
                    this.processCommentBatch(batch, chunks, resultChunks, codebasePath, i + batchIndex, BATCH_SIZE)
                );
                
                // Wait for the current set of batches to complete
                await Promise.all(batchPromises);
                const batchDuration = Date.now() - batchStartTime;
                console.log(`✅ Completed processing ${Math.min((i + MAX_PARALLEL_BATCHES), batches.length)}/${batches.length} batches in ${(batchDuration/1000).toFixed(2)}s`);
            }
            
            // 验证结果数组长度是否与输入数组一致
            if (resultChunks.length !== chunks.length) {
                console.warn(`⚠️ 警告: 结果块数量(${resultChunks.length})与输入块数量(${chunks.length})不一致!`);
            }
            
            // 验证每个代码块是否都有对应的增强注释
            let missingComments = 0;
            for (let i = 0; i < resultChunks.length; i++) {
                if (resultChunks[i].content === chunks[i].content) {
                    console.warn(`⚠️ 警告: 代码块 ${i} 没有生成增强注释!`);
                    missingComments++;
                }
            }
            if (missingComments > 0) {
                console.warn(`⚠️ 总计 ${missingComments} 个代码块没有生成增强注释!`);
            }
            
            const totalDuration = Date.now() - startTime;
            console.log(`✅ Successfully generated comments for all chunks in ${(totalDuration/1000).toFixed(2)}s (${(totalDuration/chunks.length).toFixed(2)}ms per chunk)`);
            return resultChunks;
        } catch (error) {
            const totalDuration = Date.now() - startTime;
            console.error(`❌ Failed to generate comments batch after ${(totalDuration/1000).toFixed(2)}s: ${error}`);
            return chunks; // Return original chunks on error
        }
    }

    /**
     * Process a single batch of chunks for comment generation
     * @param batch The batch of chunks to process
     * @param allChunks All chunks (for reference)
     * @param resultChunks The result array to update
     * @param codebasePath The codebase path
     * @param batchNumber The batch number for logging
     * @param batchSize The size of each batch
     */
    private async processCommentBatch(
        batch: CodeChunk[], 
        allChunks: CodeChunk[], 
        resultChunks: CodeChunk[], 
        codebasePath: string,
        batchNumber: number,
        batchSize: number
    ): Promise<void> {
        const batchStartTime = Date.now();
        let promptTime = 0;
        let apiCallTime = 0;
        let parseTime = 0;
        
        try {
            // Prepare batch for comment generation
            const prepStartTime = Date.now();
            // 计算当前批次在所有代码块中的实际起始索引位置
            const batchStartIndex = batchNumber * batchSize;
            
            const batchPrompts = batch.map((chunk, localIndex) => {
                // 使用全局索引而非局部索引
                const globalChunkIndex = batchStartIndex + localIndex;
                
                const language = chunk.metadata.language || 'unknown';
                const relativePath = path.relative(codebasePath, chunk.metadata.filePath || '');
                let nodeType = chunk.metadata.nodeType || 'unknown';
                
                // 如果metadata中没有nodeType，则通过内容推断
                if (nodeType === 'unknown' && language === 'java') {
                    if (chunk.content.includes('class ') && !chunk.content.includes('interface ')) {
                        nodeType = 'class_declaration';
                    } else if (chunk.content.includes('interface ')) {
                        nodeType = 'interface_declaration';
                    } else if (chunk.content.match(/\b[A-Z][A-Za-z0-9_]*\s*\([^)]*\)\s*(\{|throws)/)) {
                        nodeType = 'constructor_declaration';
                    } else if (chunk.content.match(/\b(public|private|protected)?\s+(static\s+)?(final\s+)?\w+(<[^>]+>)?\s+\w+\s*\([^)]*\)/)) {
                        nodeType = 'method_declaration';
                    }
                }
                
                return {
                    // 关键修改：直接使用全局索引，而不是allChunks.indexOf(chunk)或局部索引
                    chunkIndex: globalChunkIndex,
                    language,
                    relativePath,
                    nodeType,
                    content: chunk.content
                };
            });
            
            // Build a single batch prompt for GPT-4
            const batchPrompt = `你是一位专业的代码文档专家。请为以下${batchPrompts.length}个代码片段生成详尽、语义丰富的中文注释。

对每个代码片段，注释需要包括：
1. 功能描述：代码的主要功能和目的
2. 输入参数：每个参数的作用和类型
3. 返回结果：返回值的含义和格式
4. 依赖关系：与其他函数或模块的依赖关系，包括调用关系和数据流

请按照下面的格式回复，确保为每个代码片段生成独立的注释，并保留正确的chunkIndex。格式为JSON数组：

\`\`\`json
[
  {
    "chunkIndex": ${batchStartIndex}, // 注意: 这是全局索引，而非局部索引
    "comments": "这里是对第一个代码片段的详细注释..."
  },
  {
    "chunkIndex": ${batchStartIndex + 1}, // 请根据全局索引递增
    "comments": "这里是对第二个代码片段的详细注释..."
  },
  // 以此类推...
]
\`\`\`

下面是代码片段：

${batchPrompts.map((prompt, index) => `
--- 代码片段 ${prompt.chunkIndex} (来自 ${prompt.relativePath}) ---
语言: ${prompt.language}
节点类型: ${prompt.nodeType}
依赖提示: ${prompt.language === 'java' ? this.getJavaDependencyPrompt(prompt.nodeType) : '与其他函数或模块的依赖关系'}

\`\`\`
${prompt.content}
\`\`\`
`).join('\n\n')}`;

            promptTime = Date.now() - prepStartTime;
            console.log(`🔄 Processing batch #${batchNumber} with ${batch.length} chunks (prompt prep: ${promptTime}ms)...`);
            
            // Call GPT-4 API once for the batch
            const apiStartTime = Date.now();
            const batchCommentsResponse = await generateCode(batchPrompt, 'gpt-4', 32000, 0);
            //console.log(`🔄 batchCommentsResponse: ${batchCommentsResponse}`);
            apiCallTime = Date.now() - apiStartTime;
            const duration = Date.now() - batchStartTime;
            console.log(`⏱️ Batch #${batchNumber} API call completed in ${(apiCallTime/1000).toFixed(2)}s (${(apiCallTime/batch.length).toFixed(0)}ms/chunk)`);
            
            // Parse the JSON response
            const parseStartTime = Date.now();
            let commentsData: { chunkIndex: number; comments: string }[] = [];
            try {
                // Extract JSON array from the response
                const jsonMatch = batchCommentsResponse.match(/```json\n([\s\S]*?)\n```/) || 
                                 batchCommentsResponse.match(/```\n([\s\S]*?)\n```/) ||
                                 [null, batchCommentsResponse];
                
                const jsonContent = jsonMatch ? jsonMatch[1] : batchCommentsResponse;
                commentsData = JSON.parse(jsonContent);

                // 验证返回数据，确保与批次对应
                console.log(`🔍 验证批次#${batchNumber} API返回: 收到${commentsData.length}个结果, 应该处理索引范围[${batchStartIndex}-${batchStartIndex + batch.length - 1}]`);
                
                // 检查并修复索引 - 尽管我们请求使用全局索引，但仍需防止API不遵循指令
                for (let i = 0; i < commentsData.length; i++) {
                    const expectedIndex = batchStartIndex + i;
                    if (commentsData[i].chunkIndex !== expectedIndex) {
                        console.warn(`⚠️ 索引不匹配: API返回chunkIndex=${commentsData[i].chunkIndex}, 预期=${expectedIndex}, 已修正`);
                        commentsData[i].chunkIndex = expectedIndex;
                    }
                }
                
                // 确保所有批次中的代码块都有对应的评论
                if (commentsData.length < batch.length) {
                    console.warn(`⚠️ 警告: API返回结果数量(${commentsData.length})小于批次大小(${batch.length})!`);
                    // 为缺失的代码块创建占位评论
                    for (let i = 0; i < batch.length; i++) {
                        const expectedIndex = batchStartIndex + i;
                        if (!commentsData.some(data => data.chunkIndex === expectedIndex)) {
                            console.warn(`⚠️ 为缺失的代码块索引 ${expectedIndex} 创建占位评论`);
                            commentsData.push({
                                chunkIndex: expectedIndex,
                                comments: `⚠️ 由于API响应不完整，未能获取此代码块的详细注释。`
                            });
                        }
                    }
                }
            } catch (error) {
                console.error(`Failed to parse comments JSON for batch #${batchNumber}: ${error}`);
                // Fallback: treat the entire response as a single comment for all chunks
                commentsData = batchPrompts.map(p => ({ 
                    chunkIndex: p.chunkIndex, 
                    comments: `解析失败，原始回复:\n${batchCommentsResponse}` 
                }));
            }
            
            // Apply the comments to the result chunks
            for (const commentData of commentsData) {
                const chunkIndex = commentData.chunkIndex;
                if (chunkIndex >= 0 && chunkIndex < allChunks.length) {
                    const originalChunk = allChunks[chunkIndex];
                    
                    // 记录对应关系，确保每个代码块都有正确的增强注释
                    const relativePath = path.relative(codebasePath, originalChunk.metadata.filePath || '');
                    console.log(`🔄 处理代码块 ${chunkIndex}: ${relativePath}, 节点类型: ${originalChunk.metadata.nodeType || 'unknown'}${originalChunk.metadata.nodeName ? `, 名称: ${originalChunk.metadata.nodeName}` : ''}`);
                    
                    // 检查是否已经处理过这个代码块，避免重复处理
                    if (resultChunks[chunkIndex].content !== allChunks[chunkIndex].content) {
                        console.log(`⚠️ 代码块 ${chunkIndex} 已被处理过，跳过重复处理`);
                        continue;
                    }
                    
                    resultChunks[chunkIndex] = {
                        content: `# 代码块: ${chunkIndex}
## 元数据
- 文件: ${path.basename(originalChunk.metadata.filePath || '')}
- 语言: ${originalChunk.metadata.language || 'unknown'}
- 类型: ${originalChunk.metadata.nodeType || 'unknown'}${originalChunk.metadata.nodeName ? `\n- 名称: ${originalChunk.metadata.nodeName}` : ''}

## 功能摘要
${commentData.comments.split('\n')[0] || ''}

## 原始代码
\`\`\`${originalChunk.metadata.language || ''}
${originalChunk.content}
\`\`\`

## 详细注释
${commentData.comments}`,
                        metadata: { ...originalChunk.metadata }
                    };
                } else {
                    console.warn(`⚠️ 警告: 收到无效的chunkIndex ${chunkIndex}, 超出范围 [0, ${allChunks.length - 1}]`);
                }
            }
            parseTime = Date.now() - parseStartTime;
            
            const totalDuration = Date.now() - batchStartTime;
            console.log(`✅ Processed batch #${batchNumber} with ${batch.length} chunks in ${(totalDuration/1000).toFixed(2)}s
            - Prompt preparation: ${(promptTime/1000).toFixed(2)}s (${Math.round(promptTime/totalDuration*100)}%)
            - API call: ${(apiCallTime/1000).toFixed(2)}s (${Math.round(apiCallTime/totalDuration*100)}%)
            - Parse & process: ${(parseTime/1000).toFixed(2)}s (${Math.round(parseTime/totalDuration*100)}%)
            - Per chunk: ${(totalDuration/batch.length).toFixed(0)}ms`);
        } catch (error) {
            const duration = Date.now() - batchStartTime;
            console.error(`❌ Failed to generate comments for batch #${batchNumber} after ${(duration/1000).toFixed(2)}s: ${error}`);
            // Original chunks are already in the resultChunks array as fallback
        }
    }

    /**
     * Get Java dependency prompt based on node type
     * @param nodeType Type of Java node (class, method, interface, constructor)
     * @returns Specific dependency prompt
     */
    private getJavaDependencyPrompt(nodeType: string): string {
        switch (nodeType) {
            case 'class_declaration':
                return '详细分析类的继承关系、实现的接口、类的依赖关系（例如：类成员、内部类、使用的其他类）';
            case 'method_declaration':
                return '详细分析方法调用的其他方法或服务、使用的类或对象、调用链路、操作的数据结构';
            case 'interface_declaration':
                return '详细分析接口的继承关系、定义的方法声明、哪些类实现了该接口、接口与其他接口的关系';
            case 'constructor_declaration':
                return '详细分析构造函数初始化的对象依赖、注入的服务、调用的其他构造函数、创建的对象生命周期';
            default:
                return '与其他模块或函数的依赖关系、调用链路和数据流';
        }
    }

    /**
     * Process accumulated chunk buffer
     */
    private async processChunkBuffer(chunkBuffer: Array<{ chunk: CodeChunk; codebasePath: string }>): Promise<void> {
        if (chunkBuffer.length === 0) return;

        // Extract chunks and ensure they all have the same codebasePath
        const chunks = chunkBuffer.map(item => item.chunk);
        const codebasePath = chunkBuffer[0].codebasePath;

        // Estimate tokens (rough estimation: 1 token ≈ 4 characters)
        const estimatedTokens = chunks.reduce((sum, chunk) => sum + Math.ceil(chunk.content.length / 4), 0);
        
        console.log(`🔄 Processing batch of ${chunks.length} chunks (~${estimatedTokens} tokens)`);
        const startTime = Date.now();
        
        try {
            // Generate semantic-rich Chinese comments for chunks - only if enabled
            const ENABLE_COMMENTS = process.env.ENABLE_COMMENTS === 'true';
            let enhancedChunks: CodeChunk[] = chunks;
            
            if (ENABLE_COMMENTS) {
                console.log(`🤖 Generating comments enabled - processing chunks with comments`);
                const commentStartTime = Date.now();
                enhancedChunks = await this.generateChunkComments(chunks, codebasePath);
                const commentDuration = Date.now() - commentStartTime;
                console.log(`📊 Comment generation completed in ${(commentDuration/1000).toFixed(2)}s`);
            }
            
            // Process chunks with comments
            const embeddingStartTime = Date.now();
            await this.processChunkBatch(codebasePath, chunks, enhancedChunks);
            const embeddingDuration = Date.now() - embeddingStartTime;
            const totalDuration = Date.now() - startTime;
            
            console.log(`📈 Performance stats: 
            - Total processing time: ${(totalDuration/1000).toFixed(2)}s
            - Embedding time: ${(embeddingDuration/1000).toFixed(2)}s (${Math.round(embeddingDuration/totalDuration*100)}%)
            - Avg. time per chunk: ${(totalDuration/chunks.length).toFixed(2)}ms`);
            
        } catch (error) {
            const duration = Date.now() - startTime;
            console.error(`❌ Failed during chunk processing after ${(duration/1000).toFixed(2)}s: ${error}`);
            // Fallback to processing without comments
            await this.processChunkBatch(codebasePath, chunks);
        }
    }

    /**
     * Process a batch of chunks
     */
    private async processChunkBatch(codebasePath: string, chunks: CodeChunk[], enhancedChunks?: CodeChunk[]): Promise<void> {
        // Generate embedding vectors
        const chunkContents = chunks.map(chunk => chunk.content);
        //console.log(`🔄 chunkContents: ${chunkContents}`);

        // 如果enhancedChunks不为空，则使用enhancedChunks的content
        const enhancedChunkContents = enhancedChunks ? enhancedChunks.map(chunk => chunk.content) : chunkContents;
        console.log(`🔄 处理增强后的代码块: ${enhancedChunks ? enhancedChunks.length : 0} 个`);
        
        // 验证增强块与原始块的数量是否一致
        if (enhancedChunks && enhancedChunks.length !== chunks.length) {
            console.warn(`⚠️ 警告: 增强块数量(${enhancedChunks.length})与原始块数量(${chunks.length})不一致!`);
        }
        
        // 不要直接打印整个数组内容，避免内容混合在一起
        // enhancedChunkContents.forEach(content => {
        //     console.log(`🔄 处理增强后的代码块: ${content}`);
        // });

        // 对enhancedChunkContents进行embedding；colelction中存的还是原始chunk(包括原有注释)
        const embeddings: EmbeddingVector[] = await this.embedding.embedBatch(enhancedChunkContents);

        // Prepare vector documents
        const documents: VectorDocument[] = chunks.map((chunk, index) => {
            if (!chunk.metadata.filePath) {
                throw new Error(`Missing filePath in chunk metadata at index ${index}`);
            }

            const relativePath = path.relative(codebasePath, chunk.metadata.filePath);
            const fileExtension = path.extname(chunk.metadata.filePath);

            // Extract metadata that should be stored separately
            const { filePath, startLine, endLine, ...restMetadata } = chunk.metadata;

            return {
                id: this.generateId(relativePath, chunk.metadata.startLine || 0, chunk.metadata.endLine || 0, chunk.content),
                vector: embeddings[index].vector,
                content: chunk.content,
                relativePath,
                startLine: chunk.metadata.startLine || 0,
                endLine: chunk.metadata.endLine || 0,
                fileExtension,
                metadata: {
                    ...restMetadata,
                    codebasePath,
                    language: chunk.metadata.language || 'unknown',
                    chunkIndex: index
                }
            };
        });

        // Store to vector database
        await this.vectorDatabase.insert(this.getCollectionName(codebasePath), documents);
    }



    /**
     * Get programming language based on file extension
     */
    private getLanguageFromExtension(ext: string): string {
        const languageMap: Record<string, string> = {
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.py': 'python',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.m': 'objective-c',
            '.mm': 'objective-c',
            '.ipynb': 'jupyter'
        };
        return languageMap[ext] || 'text';
    }

    /**
     * Generate unique ID based on chunk content and location
     * @param relativePath Relative path to the file
     * @param startLine Start line number
     * @param endLine End line number
     * @param content Chunk content
     * @returns Hash-based unique ID
     */
    private generateId(relativePath: string, startLine: number, endLine: number, content: string): string {
        const combinedString = `${relativePath}:${startLine}:${endLine}:${content}`;
        const hash = crypto.createHash('sha256').update(combinedString, 'utf-8').digest('hex');
        return `chunk_${hash.substring(0, 16)}`;
    }

    /**
     * Read ignore patterns from file (e.g., .gitignore)
     * @param filePath Path to the ignore file
     * @returns Array of ignore patterns
     */
    static async getIgnorePatternsFromFile(filePath: string): Promise<string[]> {
        try {
            const content = await fs.promises.readFile(filePath, 'utf-8');
            return content
                .split('\n')
                .map(line => line.trim())
                .filter(line => line && !line.startsWith('#')); // Filter out empty lines and comments
        } catch (error) {
            console.warn(`⚠️  Could not read ignore file ${filePath}: ${error}`);
            return [];
        }
    }

    /**
     * Check if a path matches any ignore pattern
     * @param filePath Path to check
     * @param basePath Base path for relative pattern matching
     * @returns True if path should be ignored
     */
    private matchesIgnorePattern(filePath: string, basePath: string): boolean {
        if (this.ignorePatterns.length === 0) {
            return false;
        }

        const relativePath = path.relative(basePath, filePath);
        const normalizedPath = relativePath.replace(/\\/g, '/'); // Normalize path separators

        for (const pattern of this.ignorePatterns) {
            if (this.isPatternMatch(normalizedPath, pattern)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Simple glob pattern matching
     * @param filePath File path to test
     * @param pattern Glob pattern
     * @returns True if pattern matches
     */
    private isPatternMatch(filePath: string, pattern: string): boolean {
        // Handle directory patterns (ending with /)
        if (pattern.endsWith('/')) {
            const dirPattern = pattern.slice(0, -1);
            const pathParts = filePath.split('/');
            return pathParts.some(part => this.simpleGlobMatch(part, dirPattern));
        }

        // Handle file patterns
        if (pattern.includes('/')) {
            // Pattern with path separator - match exact path
            return this.simpleGlobMatch(filePath, pattern);
        } else {
            // Pattern without path separator - match filename in any directory
            const fileName = path.basename(filePath);
            return this.simpleGlobMatch(fileName, pattern);
        }
    }

    /**
     * Simple glob matching supporting * wildcard
     * @param text Text to test
     * @param pattern Pattern with * wildcards
     * @returns True if pattern matches
     */
    private simpleGlobMatch(text: string, pattern: string): boolean {
        // Convert glob pattern to regex
        const regexPattern = pattern
            .replace(/[.+^${}()|[\]\\]/g, '\\$&') // Escape regex special chars except *
            .replace(/\*/g, '.*'); // Convert * to .*

        const regex = new RegExp(`^${regexPattern}$`);
        return regex.test(text);
    }

    /**
     * Get current splitter information
     */
    getSplitterInfo(): { type: string; hasBuiltinFallback: boolean; supportedLanguages?: string[] } {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
            return {
                type: 'ast',
                hasBuiltinFallback: true,
                supportedLanguages: AstCodeSplitter.getSupportedLanguages()
            };
        } else {
            return {
                type: 'langchain',
                hasBuiltinFallback: false
            };
        }
    }

    /**
     * Check if current splitter supports a specific language
     * @param language Programming language
     */
    isLanguageSupported(language: string): boolean {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
            return AstCodeSplitter.isLanguageSupported(language);
        }

        // LangChain splitter supports most languages
        return true;
    }

    /**
     * Get which strategy would be used for a specific language
     * @param language Programming language
     */
    getSplitterStrategyForLanguage(language: string): { strategy: 'ast' | 'langchain'; reason: string } {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
            const isSupported = AstCodeSplitter.isLanguageSupported(language);

            return {
                strategy: isSupported ? 'ast' : 'langchain',
                reason: isSupported
                    ? 'Language supported by AST parser'
                    : 'Language not supported by AST, will fallback to LangChain'
            };
        } else {
            return {
                strategy: 'langchain',
                reason: 'Using LangChain splitter directly'
            };
        }
    }
}
