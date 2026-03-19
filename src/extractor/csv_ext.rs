use super::ExtractedText;

/// 从 CSV 文件中提取文本
pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(false)
        .from_path(file_path)
        .map_err(|e| format!("打开 CSV 文件失败: {}", e))?;

    let mut text = String::new();
    for record in reader.records() {
        let record = record.map_err(|e| format!("读取 CSV 行失败: {}", e))?;
        let row: Vec<&str> = record.iter().collect();
        text.push_str(&row.join("\t"));
        text.push('\n');
    }

    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![ExtractedText {
        page_number: 1,
        text: text.trim().to_string(),
    }])
}
