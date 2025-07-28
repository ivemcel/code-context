import { MilvusClient, DataType, MetricType, FunctionType } from '@zilliz/milvus2-sdk-node';
import {
    VectorDatabase,
    VectorDocument,
    SearchOptions,
    VectorSearchResult
} from './index';

export interface MilvusConfig {
    address: string;
    username?: string;
    password?: string;
    token?: string;
    ssl?: boolean;
    enableSparseVectors?: boolean; // New option for enabling sparse vectors
}

export class MilvusVectorDatabase implements VectorDatabase {
    private client: MilvusClient;
    private config: MilvusConfig;

    constructor(config: MilvusConfig) {
        this.config = config;
        console.log('🔌 Connecting to vector database at: ', config.address);
        this.client = new MilvusClient({
            address: config.address,
            username: config.username,
            password: config.password,
            token: config.token,
            ssl: config.ssl || false,
        });
    }



    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        // Base fields for schema
        const baseFields = [
            {
                name: 'id',
                description: 'Document ID',
                data_type: DataType.VarChar,
                max_length: 512,
                is_primary_key: true,
            },
            {
                name: 'vector',
                description: 'Embedding vector',
                data_type: DataType.FloatVector,
                dim: dimension,
            },
            {
                name: 'content',
                description: 'Document content',
                data_type: DataType.VarChar,
                max_length: 65535,
                // Enable full-text search capabilities for sparse vectors
                enable_match: this.config.enableSparseVectors === true,
                enable_analyzer: this.config.enableSparseVectors === true,
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

        // Add sparse vector field if enabled
        const fields = [...baseFields];
        if (this.config.enableSparseVectors) {
            fields.push({
                name: 'sparseVector',
                description: 'Sparse vector for BM25 indexing',
                data_type: DataType.SparseFloatVector,
            });
        }

        const createCollectionParams = {
            collection_name: collectionName,
            description: description || `Code indexer collection: ${collectionName}`,
            fields: fields,
        };

        // Define BM25 function if sparse vectors are enabled
        let functions: any[] = [];
        if (this.config.enableSparseVectors) {
            functions = [
                {
                    name: "content_bm25_emb",
                    description: "content bm25 function",
                    type: FunctionType.BM25,
                    input_field_names: ["content"],
                    output_field_names: ["sparseVector"],
                    params: {},
                },
            ];
        }

        // Create collection with fields and functions if specified
        if (this.config.enableSparseVectors && functions.length > 0) {
            await this.client.createCollection({
                ...createCollectionParams,
                functions: functions,
            });
        } else {
            await this.client.createCollection(createCollectionParams);
        }

        // Create indices
        // Dense vector index
        const denseIndexParams = {
            collection_name: collectionName,
            field_name: 'vector',
            index_name: 'vector_index',
            index_type: 'AUTOINDEX',
            metric_type: MetricType.COSINE,
        };
        await this.client.createIndex(denseIndexParams);

        // Create sparse vector index if enabled
        if (this.config.enableSparseVectors) {
            const sparseIndexParams = {
                collection_name: collectionName,
                field_name: 'sparseVector',
                index_name: 'sparseVector_index',
                index_type: 'SPARSE_INVERTED_INDEX',
                metric_type: 'BM25',
                params: {
                    inverted_index_algo: "DAAT_MAXSCORE",
                }
            };
            await this.client.createIndex(sparseIndexParams);
        }

        // Load collection to memory
        await this.client.loadCollection({
            collection_name: collectionName,
        });

        // Verify collection is created correctly
        await this.client.describeCollection({
            collection_name: collectionName,
        });
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
        const data = documents.map(doc => ({
            id: doc.id,
            vector: doc.vector,
            // Sparse vector will be automatically generated by the BM25 function
            content: doc.content,
            relativePath: doc.relativePath,
            startLine: doc.startLine,
            endLine: doc.endLine,
            fileExtension: doc.fileExtension,
            metadata: JSON.stringify(doc.metadata),
        }));

        await this.client.insert({
            collection_name: collectionName,
            data: data,
        });
    }

    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        // Default is dense vector search only
        let searchParams = {
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
                vector: queryVector, // We don't get vectors back from search
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

    // Add a new hybrid search method
    async hybridSearch(collectionName: string, query: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        if (!this.config.enableSparseVectors) {
            // Fall back to regular search if sparse vectors not enabled
            return this.search(collectionName, queryVector, options);
        }

        // Create search parameters for dense and sparse search
        const denseSearchParam = {
            "data": queryVector,
            "anns_field": "vector",
            "param": {"nprobe": 10},
            "limit": options?.topK || 10
        };

        const sparseSearchParam = {
            "data": query,
            "anns_field": "sparseVector",
            "param": {"drop_ratio_search": 0.2},
            "limit": options?.topK || 10
        };

        // Use RRF ranker for hybrid search
        // Need to import RRFRanker from SDK
        const { RRFRanker } = require('@zilliz/milvus2-sdk-node');
        const rerank = RRFRanker(100);

        // Execute hybrid search
        const searchResult = await this.client.search({
            collection_name: collectionName,
            data: [denseSearchParam, sparseSearchParam],
            output_fields: ['id', 'content', 'relativePath', 'startLine', 'endLine', 'fileExtension', 'metadata'],
            limit: options?.topK || 10,
            rerank: rerank
        });

        if (!searchResult.results || searchResult.results.length === 0) {
            return [];
        }

        return searchResult.results.map((result: any) => ({
            document: {
                id: result.id,
                vector: queryVector, // We don't get vectors back from search
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