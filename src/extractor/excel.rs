use super::ExtractedText;
use calamine::{Reader, open_workbook_auto};

/// 从 XLS/XLSX 文件中提取文本
///
/// 按 Sheet 读取，每个 Sheet 作为一个页面
pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let mut workbook = open_workbook_auto(file_path)
        .map_err(|e| format!("打开 Excel 文件失败: {}", e))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    let mut results = Vec::new();

    for (i, name) in sheet_names.iter().enumerate() {
        if let Ok(range) = workbook.worksheet_range(name) {
            let mut sheet_text = String::new();
            sheet_text.push_str(&format!("[Sheet: {}]\n", name));

            for row in range.rows() {
                let row_text: Vec<String> = row
                    .iter()
                    .map(|cell| format!("{}", cell))
                    .collect();
                sheet_text.push_str(&row_text.join("\t"));
                sheet_text.push('\n');
            }

            let trimmed = sheet_text.trim();
            if !trimmed.is_empty() {
                results.push(ExtractedText {
                    page_number: (i + 1) as i32,
                    text: trimmed.to_string(),
                });
            }
        }
    }

    Ok(results)
}
