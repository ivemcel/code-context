/**
 * GPT-4 API client for code generation
 */

import axios from 'axios';

/**
 * Message interface for GPT-4 API
 */
export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * Parameters for GPT-4 API request
 */
export interface GPT4RequestParams {
  model: string;
  messages: Message[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
}

/**
 * Response from the actual API
 */
export interface ActualAPIResponse {
  code: string;
  data: string;
  log: any;
  message: string;
  timestamp: string;
}

/**
 * GPT-4 API client
 */
export class GPT4Client {
  private apiUrl = 'http://10.142.99.29:8085/codegen/genCode';
  private token = 'knJ6mlyfvRP6OHwl79d2AF2mgCEhwO4d';

  /**
   * Generate code using GPT-4
   * 
   * @param params - Request parameters
   * @returns API response
   */
  async generateCode(params: GPT4RequestParams): Promise<ActualAPIResponse> {
    try {
      console.log(`Sending request to ${this.apiUrl}...`);
      console.log('Request params:', JSON.stringify(params, null, 2));
      
      const response = await axios.post(this.apiUrl, params, {
        headers: {
          'token': this.token,
          'Content-Type': 'application/json'
        }
      });
      
      console.log('Response status:', response.status);
      console.log('Response data:', JSON.stringify(response.data, null, 2));
      
      return response.data;
    } catch (error) {
      console.error('Error generating code with GPT-4:', error);
      if (axios.isAxiosError(error) && error.response) {
        console.error('Response status:', error.response.status);
        console.error('Response data:', error.response.data);
      }
      throw error;
    }
  }

  /**
   * Generate code using GPT-4 with default parameters
   * 
   * @param prompt - User prompt
   * @param model - Model to use (default: 'gpt-4')
   * @returns Generated code as string
   */
  async generateCodeWithPrompt(
    prompt: string, 
    model = 'gpt-4',
    maxTokens = 1000,
    temperature = 0
  ): Promise<string> {
    const params: GPT4RequestParams = {
      model,
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ],
      max_tokens: maxTokens,
      temperature,
      stream: false
    };

    const response = await this.generateCode(params);
    if (!response || !response.data) {
      throw new Error('Invalid response from API: no data returned');
    }
    return response.data;
  }
}

// Export a singleton instance
export const gpt4Client = new GPT4Client();

// Export a convenient function for quick access
export async function generateCode(
  prompt: string, 
  model = 'gpt-4',
  maxTokens = 10000,
  temperature = 0
): Promise<string> {
  return gpt4Client.generateCodeWithPrompt(prompt, model, maxTokens, temperature);
} 