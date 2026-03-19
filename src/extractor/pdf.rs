use super::ExtractedText;

/// 从 PDF 文件中提取文本（按页）
pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let bytes = std::fs::read(file_path)
        .map_err(|e| format!("读取 PDF 文件失败: {}", e))?;

    let text = pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| format!("解析 PDF 失败: {}", e))?;

    // pdf-extract 将所有页面合并输出，我们用分页符 '\x0C' (form feed) 分割
    let pages: Vec<&str> = text.split('\x0C').collect();

    let mut results = Vec::new();
    for (i, page_text) in pages.iter().enumerate() {
        let trimmed = page_text.trim();
        if !trimmed.is_empty() {
            results.push(ExtractedText {
                page_number: (i + 1) as i32,
                text: trimmed.to_string(),
            });
        }
    }

    // 如果分页失败，整体作为第 1 页
    if results.is_empty() && !text.trim().is_empty() {
        results.push(ExtractedText {
            page_number: 1,
            text: text.trim().to_string(),
        });
    }

    Ok(results)
}
