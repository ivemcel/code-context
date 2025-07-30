/**
 * File Splitter Example
 * 
 * This script demonstrates how to read a file, split it using the AstCodeSplitter,
 * and save the resulting chunks to the docs directory.
 * 
 * Usage:
 * tsx file-splitter.ts <file_path> <language>
 * 
 * Example:
 * tsx file-splitter.ts ../packages/core/src/indexer.ts typescript
 */

import { AstCodeSplitter, CodeChunk } from '@code-indexer/core';
import * as path from 'path';
import * as fs from 'fs';
import { EnhancedAstSplitter } from '@code-indexer/core/src/splitter/enhanced-ast-splitter';
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
 * Writes the code chunks to a markdown file in the docs directory
 */
function writeChunksToFile(filePath: string, language: string, chunks: CodeChunk[], docsPath: string): void {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const baseFileName = path.basename(filePath);
    const fileName = `code_chunks_${baseFileName}_${timestamp}.md`;
    const outputPath = path.join(docsPath, fileName);

    let content = `# Code Chunks for: "${filePath}"\n`;
    content += `Generated on: ${new Date().toLocaleString()}\n`;
    content += `Language: ${language}\n\n`;

    if (chunks.length > 0) {
        content += `## Found ${chunks.length} chunks\n\n`;
        chunks.forEach((chunk, index) => {
            content += `### Chunk ${index + 1}\n`;
            content += `- **Lines**: ${chunk.metadata.startLine}-${chunk.metadata.endLine}\n`;
            content += `- **Content**:\n\`\`\`${language}\n${chunk.content}\n\`\`\`\n\n`;
        });
    } else {
        content += "No chunks were generated.\n";
    }

    fs.writeFileSync(outputPath, content);
    console.log(`📄 Chunks saved to: ${outputPath}`);
}

async function main() {
    // Process command line arguments
    const args = process.argv.slice(2);
    
    //let filePath = "/Users/ivem/WebstormProjects/code-context/docs/LoginController.java";
    let filePath = "/Users/ivem/WebstormProjects/code-context/docs/codefile/UserServiceImpl.java";
    let language = "java";
    
    // 使用命令行参数（如果提供）
    if (args.length >= 2) {
        filePath = args[0];
        language = args[1];
    } else {
        console.log('ℹ️ 使用默认文件路径和语言。要指定其他文件，请使用: tsx file-splitter.ts <file_path> <language>');
    }
    
    try {
        console.log('🚀 代码分割器示例');
        console.log('======================');
        
        // 检查文件是否存在
        if (!fs.existsSync(filePath)) {
            console.error(`❌ 错误: 文件 "${filePath}" 不存在.`);
            process.exit(1);
        }

        // 读取文件内容
        console.log(`📖 读取文件: ${filePath}`);
        const content = fs.readFileSync(filePath, 'utf8');
        console.log(`✅ 文件读取成功. 大小: ${content.length} 字符`);

        // 根据文件类型选择分割器
        let codeSplitter;
        const lowerLang = language.toLowerCase();
        
        // 对Java文件使用增强型分割器（保留注释）
        // 对其他语言默认也使用增强型，但后续可根据需求调整
        codeSplitter = new EnhancedAstSplitter(0, 0);

        // if (lowerLang === 'java') {
        //     console.log(`🔍 检测到Java文件，使用增强型AST分割器（支持保留注释）`);
        //     codeSplitter = new EnhancedAstSplitter(5000, 0);
        // } else {
        //     console.log(`📃 使用标准AST分割器处理${language}文件`);
        //     codeSplitter = new AstCodeSplitter(5000, 0);
        // }
        
        // 分割代码
        console.log(`🔪 使用语言 ${language} 分割代码`);
        const chunks = await codeSplitter.split(content, language, filePath);
        console.log(`✅ 代码分割成功. 生成了 ${chunks.length} 个代码块.`);
        
        // 确保docs目录存在
        const docsPath = path.join(__dirname, '../../docs');
        ensureDirectoryExists(docsPath);
        
        // 将代码块写入文件
        writeChunksToFile(filePath, language, chunks, docsPath);
        console.log('\n🎉 文件分割成功完成!');

    } catch (error) {
        console.error('❌ 发生错误:', error);
        process.exit(1);
    }
}

// 运行主程序
if (require.main === module) {
    main().catch(console.error);
}

export { main }; 