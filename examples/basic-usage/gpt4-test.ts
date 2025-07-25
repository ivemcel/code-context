/**
 * Example of using the GPT-4 client
 */

import { GPT4Client, generateCode } from '../../packages/core/src/utils/gpt4-client';

async function simpleExample() {
  console.log('Running simple example...');
  try {
    // Simple usage with the utility function
    const result = await generateCode(
      'Generate a JavaScript function to calculate factorial',
      'gpt-4',
      1000,
      0  // Use temperature=0 for deterministic results
    );
    
    console.log('Generated code:');
    console.log(result);
    console.log('\n---\n');
    return true;
  } catch (error) {
    console.error('Error in simple example:', error);
    return false;
  }
}

async function advancedExample() {
  console.log('Running advanced example...');
  try {
    // Advanced usage with the class
    const client = new GPT4Client();
    
    const response = await client.generateCode({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful programming assistant. Provide concise, well-documented code examples.'
        },
        {
          role: 'user',
          content: 'Generate a TypeScript function to calculate the Fibonacci sequence'
        }
      ],
      max_tokens: 1000,
      temperature: 0,  // Use temperature=0 for deterministic results
      stream: false
    });
    
    console.log('Generated code with advanced options:');
    if (response && response.data) {
      console.log(response.data);
    } else {
      console.log('No response data available');
    }
    return true;
  } catch (error) {
    console.error('Error in advanced example:', error);
    return false;
  }
}


async function main() {
  console.log('Starting GPT-4 code generation examples...\n');
  
  console.log('=== Example 1: Simple API Call ===');
  const simpleSuccess = await simpleExample();
  
  console.log('\n=== Example 2: Advanced API Call ===');
  const advancedSuccess = await advancedExample();
  
  console.log('\nExamples completed.');
  console.log(`Results: 
  - Simple example: ${simpleSuccess ? 'succeeded' : 'failed'}
  - Advanced example: ${advancedSuccess ? 'succeeded' : 'failed'}`);
  
  process.exit(simpleSuccess && advancedSuccess ? 0 : 1);
}

// Add proper error handling for the main function
main().catch(error => {
  console.error('Unhandled error in main:', error);
  process.exit(1);
}); 