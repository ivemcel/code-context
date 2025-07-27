import { MilvusClient, DataType, MetricType } from '@zilliz/milvus2-sdk-node';
import {
    VectorDatabase,
    VectorDocument,
    SearchOptions,
    VectorSearchResult
} from './index';
import { generateCode } from '../utils/gpt4-client';

export interface MilvusConfig {
    address: string;
    username?: string;
    password?: string;
    token?: string;
    ssl?: boolean;
    /** Enable hybrid search with BM25 sparse embedding */
    enableBM25?: boolean;
    /** Consistency level for Milvus operations */
    consistencyLevel?: 'Strong' | 'Session' | 'Bounded' | 'Eventually';
    /** Ranker type for hybrid search (defaults to 'weight' if not specified) */
    rankerType?: 'rrf' | 'weight';
    /** Ranker parameters for hybrid search */
    rankerParams?: Record<string, any>;
}

// Extend the SearchOptions interface to include query for BM25
export interface HybridSearchOptions extends SearchOptions {
    /** Text query for BM25 search */
    query?: string;
}

export class MilvusVectorDatabase implements VectorDatabase {
    private client: MilvusClient;
    private config: MilvusConfig;

    constructor(config: MilvusConfig) {
        this.config = config;
        console.log('🔌 Connecting to vector database at: ', config.address);
        // 在构造函数中打印enableBM25配置
        console.log('🔍 MilvusConfig:', JSON.stringify({
            enableBM25: config.enableBM25,
            consistencyLevel: config.consistencyLevel,
            rankerType: config.rankerType
        }, null, 2));
        
        this.client = new MilvusClient({
            address: config.address,
            username: config.username,
            password: config.password,
            token: config.token,
            ssl: config.ssl || false,
        });
    }

    /**
     * 使用更智能的中文分词方法
     * 尝试使用Milvus的中文分析器进行分词，或者降级使用本地简单分词
     * @param text 要分词的中文文本
     * @returns 分词结果数组
     */
    private async tokenizeChinese(text: string): Promise<string[]> {
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

        // 2. 尝试使用Milvus的run_analyzer API（如果可用）
        try {
            // 检查是否可以使用runAnalyzer
            if ('runAnalyzer' in this.client) {
                console.log('尝试使用Milvus中文分析器进行分词...');
                const analyzerParams = {
                    type: "chinese"
                };
                
                const result = await (this.client as any).runAnalyzer({
                    texts: [text],
                    analyzer_params: analyzerParams
                });
                
                // 如果成功获取分词结果
                if (result && result.results && result.results.length > 0) {
                    console.log('Milvus中文分析器结果:', JSON.stringify(result.results[0]).substring(0, 100) + '...');
                    
                    // 处理Milvus返回的分词格式，提取token字段
                    const tokens = result.results[0].tokens
                        .filter((t: any) => t && t.token && t.token !== 'undefined')
                        .map((t: any) => t.token);
                    
                    if (tokens.length > 0) {
                        console.log('提取的分词结果:', tokens);
                        return tokens;
                    }
                }
            }
        } catch (error) {
            console.warn('无法使用Milvus中文分析器进行分词:', error);
        }
        
        // 3. 如果两种方法都失败，返回查询文本本身作为单个关键词
        console.warn('所有分词方法都失败，使用原始查询文本作为关键词');
        return [text];
    }

    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        console.log(`🔧 创建集合 ${collectionName}，维度 ${dimension}，是否启用BM25: ${this.config.enableBM25 ? '是' : '否'}`);
        
        const schema = [
            {
                name: 'id',
                description: 'Document ID',
                data_type: DataType.VarChar,
                max_length: 512,
                is_primary_key: true,
            },
            {
                name: 'vector',
                description: 'Dense embedding vector',
                data_type: DataType.FloatVector,
                dim: dimension,
            },
            {
                name: 'content',
                description: 'Document content',
                data_type: DataType.VarChar,
                max_length: 65535,
            },
            {
                name: 'relativePath',
                description: 'Relative path to the codebase',
                data_type: DataType.VarChar,
                max_length: 1024,
            },
            {
                name: 'startLine',
                description: 'Start line number of the chunk',
                data_type: DataType.Int64,
            },
            {
                name: 'endLine',
                description: 'End line number of the chunk',
                data_type: DataType.Int64,
            },
            {
                name: 'fileExtension',
                description: 'File extension',
                data_type: DataType.VarChar,
                max_length: 32,
            },
            {
                name: 'metadata',
                description: 'Additional document metadata as JSON string',
                data_type: DataType.VarChar,
                max_length: 65535,
            },
        ];

        // 添加稀疏向量字段（如果启用BM25）
        if (this.config.enableBM25) {
            console.log('添加sparse字段（SparseFloatVector）...');
            try {
                schema.push({
                    name: 'sparse',
                    description: 'Sparse vector field for hybrid search',
                    data_type: DataType.SparseFloatVector
                    // SparseFloatVector不需要指定维度
                } as any); // 使用类型断言
                console.log('✅ 已添加sparse字段');
            } catch (e) {
                console.error('❌ 无法添加sparse字段:', e);
            }
        }
        
        console.log('📄 集合字段:', schema.map(field => field.name).join(', '));
        
        const createCollectionParams = {
            collection_name: collectionName,
            description: description || `Code indexer collection: ${collectionName}`,
            fields: schema,
            consistency_level: this.config.consistencyLevel || 'Bounded',
        };
        
        await this.client.createCollection(createCollectionParams);

        // Create index for dense vector
        const indexParams = {
            collection_name: collectionName,
            field_name: 'vector',
            index_type: 'AUTOINDEX',
            metric_type: MetricType.COSINE,
        };

        await this.client.createIndex(indexParams);
        
        // 尝试为sparse字段创建索引（如果存在）
        if (this.config.enableBM25) {
            try {
                console.log('尝试为sparse字段创建索引...');
                const sparseIndexParams = {
                    collection_name: collectionName,
                    field_name: 'sparse',
                    index_type: 'SPARSE_INVERTED_INDEX', // 稀疏向量的索引类型
                    metric_type: MetricType.IP, // 内积作为度量类型
                };
                
                await this.client.createIndex(sparseIndexParams);
                console.log('✅ sparse字段索引创建成功');
            } catch (error) {
                console.error('❌ 为sparse字段创建索引失败:', error);
            }
        }

        // Load collection to memory
        await this.client.loadCollection({
            collection_name: collectionName,
        });

        // Verify collection is created correctly
        const collectionInfo = await this.client.describeCollection({
            collection_name: collectionName,
        });
        
        console.log('集合创建完成，字段列表:');
        console.log(collectionInfo.schema.fields.map((f: any) => f.name).join(', '));
        
        // 检查是否存在sparse字段
        const hasSparseField = collectionInfo.schema.fields.some((f: any) => f.name === 'sparse');
        console.log(`集合是否包含sparse字段: ${hasSparseField ? '是' : '否'}`);
    }

    async dropCollection(collectionName: string): Promise<void> {
        await this.client.dropCollection({
            collection_name: collectionName,
        });
    }

    async hasCollection(collectionName: string): Promise<boolean> {
        const result = await this.client.hasCollection({
            collection_name: collectionName,
        });

        return Boolean(result.value);
    }

    async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
        const data = documents.map(doc => {
            // 基础字段
            const baseDoc = {
                id: doc.id,
                vector: doc.vector,
                content: doc.content,
                relativePath: doc.relativePath,
                startLine: doc.startLine,
                endLine: doc.endLine,
                fileExtension: doc.fileExtension,
                metadata: JSON.stringify(doc.metadata),
            };
            
            // 如果有sparse字段且启用了BM25，则添加该字段
            if (doc.sparse && this.config.enableBM25) {
                return {
                    ...baseDoc,
                    sparse: doc.sparse
                };
            }
            
            return baseDoc;
        });

        await this.client.insert({
            collection_name: collectionName,
            data: data,
        });
    }
    // 向量搜索 默认使用向量搜索    
    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        console.log('🔍 执行向量搜索...');
        const searchParams = {
            collection_name: collectionName,
            data: [queryVector],
            limit: options?.topK || 10,
            output_fields: ['id', 'content', 'relativePath', 'startLine', 'endLine', 'fileExtension', 'metadata'],
        };

        const searchResult = await this.client.search(searchParams);

        if (!searchResult.results || searchResult.results.length === 0) {
            return [];
        }

        return searchResult.results.map((result: any) => ({
            document: {
                id: result.id,
                vector: queryVector,
                content: result.content,
                relativePath: result.relativePath,
                startLine: result.startLine,
                endLine: result.endLine,
                fileExtension: result.fileExtension,
                metadata: JSON.parse(result.metadata || '{}'),
            },
            score: result.score,
        }));
    }

    async hybridSearch(collectionName: string, queryVector: number[], options?: HybridSearchOptions): Promise<VectorSearchResult[]> {
        // 构建正确格式的搜索参数 - 注意用vectors而不是data
        const searchParams: any = {
            collection_name: collectionName,
            vectors: [queryVector], // 使用vectors而不是data
            anns_field: 'vector',   // 必须指定向量字段名称
            limit: options?.topK || 10,
            output_fields: ['id', 'content', 'relativePath', 'startLine', 'endLine', 'fileExtension', 'metadata']
        };

        let usedTextFilter = false;  // 标记是否使用了文本过滤

        // 检查是否支持混合搜索
        if (this.config.enableBM25 && options?.query) {
            console.log(`执行混合搜索，查询文本: "${options.query}"`);
            
            try {
                // 获取集合信息，检查是否有sparse字段
                const collInfo = await this.client.describeCollection({
                    collection_name: collectionName
                });
                
                const hasSparseField = collInfo.schema.fields.some((f: any) => f.name === 'sparse');
                if (hasSparseField) {
                    console.log('检测到sparse字段，尝试混合搜索');
                    
                    // 构造文本过滤作为混合搜索的退化方案
                    if (options.query) {
                        // 使用智能中文分词替代简单的字符分割
                        const chineseTokens = await this.tokenizeChinese(options.query);
                        
                        // 合并所有可能的关键词
                        const allKeywords = [ ...chineseTokens];
                        
                        if (allKeywords.length > 0) {
                            const uniqueKeywords = [...new Set(allKeywords)]; // 去重
                            console.log(`提取的关键词 (${uniqueKeywords.length}): ${JSON.stringify(uniqueKeywords.slice(0, 10))}${uniqueKeywords.length > 10 ? '...' : ''}`);
                            
                            // 策略1：只选择较长的关键词，减少过滤范围，提高精确度
                            const keywordsForFiltering = uniqueKeywords
                                .filter(word => word && word.length > 2) // 只用长度>2的关键词
                                .slice(0, 5);  // 限制最多使用5个关键词避免过滤太严格
                            
                            let filterCondition = "";
                            
                            // 如果找不到足够长的关键词，使用所有关键词中最长的几个
                            if (keywordsForFiltering.length === 0) {
                                const sortedByLength = [...uniqueKeywords]
                                    .filter(word => word && word.length > 0)
                                    .sort((a, b) => b.length - a.length)
                                    .slice(0, 3); // 取最长的3个
                                
                                if (sortedByLength.length > 0) {
                                    const conditions = sortedByLength
                                        .map(word => `content like '%${word}%'`)
                                        .join(' or ');
                                    
                                    filterCondition = `(${conditions})`;
                                }
                            } else {
                                // 使用筛选出的关键词
                                const conditions = keywordsForFiltering
                                    .map(word => `content like '%${word}%'`)
                                    .join(' or ');
                                
                                filterCondition = `(${conditions})`;
                            }
                            
                            // 只有当成功构建过滤条件时才应用它
                            if (filterCondition) {
                                searchParams.filter = filterCondition;
                                console.log(`添加文本过滤条件: ${searchParams.filter.substring(0, 100)}${searchParams.filter.length > 100 ? '...' : ''}`);
                                usedTextFilter = true;
                                
                                // 注意：不再设置search_params，因为它可能与Milvus API不兼容
                            }
                        }
                    }
                } else {
                    console.log('未找到sparse字段，使用普通向量搜索');
                }
            } catch (error) {
                console.error('❌ 混合搜索失败，降级为向量搜索:', error);
            }
        }

        // 执行搜索
        try {
            console.log(`执行搜索请求，参数: ${JSON.stringify({
                collection_name: searchParams.collection_name,
                anns_field: searchParams.anns_field,
                limit: searchParams.limit,
                has_filter: !!searchParams.filter,
                used_text_filter: usedTextFilter,
                filter_condition: searchParams.filter ? searchParams.filter.substring(0, 50) + '...' : 'none'
            })}`);
            
            const searchResult = await this.client.search(searchParams);
            
            console.log(`搜索完成，结果数量: ${searchResult.results?.length || 0}`);

            if (!searchResult.results || searchResult.results.length === 0) {
                // 如果结果为空且使用了文本过滤，尝试不使用过滤器进行普通搜索
                if (usedTextFilter) {
                    console.log('混合搜索未找到结果，降级为普通向量搜索...');
                    
                    // 去除过滤条件
                    const fallbackParams = { ...searchParams };
                    delete fallbackParams.filter;
                    
                    const fallbackResult = await this.client.search(fallbackParams);
                    console.log(`降级搜索完成，结果数量: ${fallbackResult.results?.length || 0}`);
                    
                    if (!fallbackResult.results || fallbackResult.results.length === 0) {
                        return [];
                    }
                    
                    return fallbackResult.results.map((result: any) => ({
                        document: {
                            id: result.id,
                            vector: queryVector,
                            content: result.content,
                            relativePath: result.relativePath,
                            startLine: result.startLine,
                            endLine: result.endLine,
                            fileExtension: result.fileExtension,
                            metadata: JSON.parse(result.metadata || '{}'),
                        },
                        score: result.score * 0.9, // 降低降级搜索的分数，表明这不是最理想的结果
                    }));
                }
                return [];
            }

            return searchResult.results.map((result: any) => ({
                document: {
                    id: result.id,
                    vector: queryVector,
                    content: result.content,
                    relativePath: result.relativePath,
                    startLine: result.startLine,
                    endLine: result.endLine,
                    fileExtension: result.fileExtension,
                    metadata: JSON.parse(result.metadata || '{}'),
                },
                score: result.score,
            }));
        } catch (error) {
            console.error('❌ 搜索执行失败:', error);
            
            // 如果搜索失败且使用了文本过滤，尝试不使用过滤器重试搜索
            if (usedTextFilter) {
                console.log('搜索失败，尝试不使用文本过滤重新搜索...');
                // 移除过滤条件
                delete searchParams.filter;
                
                try {
                    const fallbackResult = await this.client.search(searchParams);
                    
                    if (!fallbackResult.results || fallbackResult.results.length === 0) {
                        return [];
                    }
                    
                    console.log(`降级搜索成功，结果数量: ${fallbackResult.results.length}`);
                    
                    return fallbackResult.results.map((result: any) => ({
                        document: {
                            id: result.id,
                            vector: queryVector,
                            content: result.content,
                            relativePath: result.relativePath,
                            startLine: result.startLine,
                            endLine: result.endLine,
                            fileExtension: result.fileExtension,
                            metadata: JSON.parse(result.metadata || '{}'),
                        },
                        score: result.score * 0.8, // 降低降级搜索的分数
                    }));
                } catch (fallbackError) {
                    console.error('❌ 降级搜索也失败:', fallbackError);
                }
            }
            
            throw error;
        }
    }

    async delete(collectionName: string, ids: string[]): Promise<void> {
        await this.client.delete({
            collection_name: collectionName,
            filter: `id in [${ids.map(id => `"${id}"`).join(', ')}]`,
        });
    }

    async query(collectionName: string, filter: string, outputFields: string[]): Promise<Record<string, any>[]> {
        try {
            const result = await this.client.query({
                collection_name: collectionName,
                filter: filter,
                output_fields: outputFields,
            });

            if (result.status.error_code !== 'Success') {
                throw new Error(`Failed to query Milvus: ${result.status.reason}`);
            }

            return result.data || [];
        } catch (error) {
            console.error(`❌ Failed to query collection '${collectionName}':`, error);
            throw error;
        }
    }
} 