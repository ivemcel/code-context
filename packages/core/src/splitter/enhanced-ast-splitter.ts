import { AstCodeSplitter } from '../splitter/ast-splitter';
import { CodeChunk } from '../splitter';

/**
 * 增强型AST代码分割器
 * 
 * 该分割器扩展了标准AST分割器功能，主要增强了以下能力：
 * 1. 保留方法和类上的注释，使它们与对应的代码块一起分割
 * 2. 智能识别JavaDoc等文档注释与代码节点的关联关系
 * 3. 支持多语言的注释处理
 */
export class EnhancedAstSplitter extends AstCodeSplitter {
    constructor(chunkSize?: number, chunkOverlap?: number) {
        super(chunkSize, chunkOverlap);
    }

    /**
     * 重写分割方法，增加注释处理逻辑
     */
    async split(code: string, language: string, filePath?: string): Promise<CodeChunk[]> {
        try {
            console.log(`🌳 使用增强型AST分割器处理 ${language} 文件: ${filePath || '未知'}`);
            
            // 由于无法直接访问父类的私有方法，我们将使用父类的split方法
            // 但通过自定义后处理来关联注释与代码块
            const originalChunks = await super.split(code, language, filePath);
            
            // 为每个代码块关联注释
            const enhancedChunks = this.enhanceChunksWithComments(originalChunks, code, language);
            
            return enhancedChunks;
        } catch (error) {
            console.warn(`⚠️ 增强型AST分割器出错，回退到标准方式: ${error}`);
            return super.split(code, language, filePath);
        }
    }

    /**
     * 增强代码块，为其添加前置注释
     */
    private enhanceChunksWithComments(
        chunks: CodeChunk[], 
        fullCode: string, 
        language: string
    ): CodeChunk[] {
        // 获取代码的所有行
        const codeLines = fullCode.split('\n');
        
        // 增强后的代码块
        const enhancedChunks: CodeChunk[] = [];
        
        for (const chunk of chunks) {
            // 获取当前代码块的起始行和结束行，提供默认值避免 undefined
            const startLine = chunk.metadata.startLine || 1;
            const endLine = chunk.metadata.endLine || codeLines.length;
            
            // 初始化注释起始行
            let commentStartLine = startLine;
            
            // 向上查找可能的注释块，不限制查找行数
            let lineIndex = startLine - 2; // 转换为0-based并从代码块上方第一行开始
            let inComment = false;
            let foundCommentStart = false;
            
            // 状态变量，用于追踪不同类型的注释
            let inBlockComment = false;
            let inJavadocComment = false;
            let consecutiveSingleLineComments = false;
            
            // 从代码块起始行向上查找
            while (lineIndex >= 0) {
                const line = codeLines[lineIndex].trim();
                
                // 空行处理：如果在注释中，空行可能是注释的一部分；否则，如果发现了注释，空行是分隔符
                if (line === '') {
                    if (!inComment && foundCommentStart) {
                        // 空行分隔了注释和代码块，结束查找
                        break;
                    }
                    // 空行，继续向上查找
                    lineIndex--;
                    continue;
                }
                
                // 检查是否遇到了非注释的代码行
                if (!inComment && 
                    !line.startsWith('//') && 
                    !line.startsWith('/*') && 
                    !line.startsWith('*') &&
                    !line.endsWith('*/')) {
                    // 如果已经找到了注释并且现在遇到非注释行，说明注释部分结束了
                    if (foundCommentStart) {
                        break;
                    }
                    // 没有找到注释就遇到了代码行，终止查找
                    break;
                }
                
                // 检测注释的开始和结束
                
                // JavaDoc注释开始
                if (line.startsWith('/**')) {
                    inComment = true;
                    inJavadocComment = true;
                    foundCommentStart = true;
                    commentStartLine = lineIndex + 1; // 转为1-based
                    break; // 找到JavaDoc开始，结束查找
                } 
                // 块注释开始
                else if (line.startsWith('/*')) {
                    inComment = true;
                    inBlockComment = true;
                    foundCommentStart = true;
                    commentStartLine = lineIndex + 1; // 转为1-based
                    break; // 找到块注释开始，结束查找
                }
                // 块注释结束
                else if (line.endsWith('*/')) {
                    // 当前行是块注释的结尾，需要继续向上查找注释开始
                    inComment = true;
                    lineIndex--;
                    continue;
                }
                // JavaDoc或块注释的中间行
                else if (line.startsWith('*')) {
                    // 处于注释中间
                    inComment = true;
                    lineIndex--;
                    continue;
                }
                // 单行注释
                else if (line.startsWith('//')) {
                    // 标记找到了注释
                    foundCommentStart = true;
                    // 记录第一个找到的单行注释的位置
                    if (!inComment) {
                        commentStartLine = lineIndex + 1; // 转为1-based
                    }
                    inComment = true;
                    consecutiveSingleLineComments = true;
                    lineIndex--;
                    continue;
                }
                // 其他非注释内容
                else {
                    // 如果已找到注释但遇到非注释行，结束查找
                    if (foundCommentStart) {
                        break;
                    }
                    // 非注释非空行，结束查找
                    break;
                }
            }
            
            // 如果找到了注释，创建包含注释的增强代码块
            if (foundCommentStart) {
                // 提取从注释开始到代码结束的文本
                const enhancedContent = codeLines.slice(commentStartLine - 1, endLine).join('\n');
                
                enhancedChunks.push({
                    content: enhancedContent,
                    metadata: {
                        ...chunk.metadata,
                        startLine: commentStartLine
                    }
                });
            } else {
                // 没有找到注释，保持原样
                enhancedChunks.push(chunk);
            }
        }
        
        return enhancedChunks;
    }
} 