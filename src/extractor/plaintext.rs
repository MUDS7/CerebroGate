use super::ExtractedText;

/// 从纯文本文件（TXT/MD）中提取文本
pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let text = std::fs::read_to_string(file_path)
        .map_err(|e| format!("读取文本文件失败: {}", e))?;

    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![ExtractedText {
        page_number: 1,
        text: text.trim().to_string(),
    }])
}
