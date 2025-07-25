import { MilvusVectorDatabase } from '../../packages/core/src/vectordb/milvus-vectordb';

/**
 * Test script to verify the createCollection functionality of MilvusVectorDatabase
 * 
 * This script tests:
 * 1. Connection to Milvus
 * 2. Creating a collection with default parameters
 * 3. Verifying the collection exists
 */
async function main() {
  // Configure Milvus connection
  const milvusConfig = {
    address: 'localhost:19530', // Default Milvus address
    // For authentication, uncomment and fill these:
    // username: 'username',
    // password: 'password',
    // token: 'your-token',
    // ssl: false
  };

  console.log('Creating MilvusVectorDatabase instance...');
  const milvusDB = new MilvusVectorDatabase(milvusConfig);

  // Test parameters
  const collectionName = 'test_collection_' + Date.now(); // Use timestamp to avoid conflicts
  const dimension = 1536; // Common dimension for embeddings (e.g. OpenAI)
  const description = 'Test collection created by test script';

  try {
    console.log(`Creating collection: ${collectionName} with dimension: ${dimension}`);
    await milvusDB.createCollection(collectionName, dimension, description);
    
    // Verify the collection was created
    const exists = await milvusDB.hasCollection(collectionName);
    console.log(`Collection exists: ${exists}`);
    
    if (exists) {
      console.log('✅ Test passed: Collection was created successfully!');
      
      // Optional: Clean up by dropping the test collection
      console.log(`Cleaning up: Dropping collection ${collectionName}`);
      await milvusDB.dropCollection(collectionName);
      
      const existsAfterDrop = await milvusDB.hasCollection(collectionName);
      console.log(`Collection exists after drop: ${existsAfterDrop}`);
    } else {
      console.error('❌ Test failed: Collection was not created');
    }
  } catch (error) {
    console.error('❌ Error during test:', error);
  }
}

main().catch(console.error); 