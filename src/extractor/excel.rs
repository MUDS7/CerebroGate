use super::ExtractedText;
use calamine::{Reader, open_workbook_auto};

/// 从 XLS/XLSX 文件中提取文本
///
/// 按 Sheet 读取，每个 Sheet 作为一个页面，并转为 Markdown 表格格式进行提取，加强可读性
pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let mut workbook = open_workbook_auto(file_path)
        .map_err(|e| format!("打开 Excel 文件失败: {}", e))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    let mut results = Vec::new();

    for (i, name) in sheet_names.iter().enumerate() {
        if let Ok(range) = workbook.worksheet_range(name) {
            let mut sheet_text = String::new();
            sheet_text.push_str(&format!("### 工作表 [Sheet: {}]\n\n", name));

            // 获取整个表数据的最大有效列宽
            let mut max_col = 0;
            for row in range.rows() {
                for (idx, cell) in row.iter().enumerate().rev() {
                    if !format!("{}", cell).trim().is_empty() {
                        if idx > max_col {
                            max_col = idx;
                        }
                        break;
                    }
                }
            }

            let mut is_first_row = true;
            for row in range.rows() {
                // 检查是否为全空行
                let row_len = std::cmp::min(max_col + 1, row.len());
                let is_empty_row = row[0..row_len]
                    .iter()
                    .all(|c| format!("{}", c).trim().is_empty());

                if is_empty_row {
                    continue;
                }

                let mut row_cells = Vec::new();
                for col_idx in 0..=max_col {
                    let cell_text = if col_idx < row.len() {
                        format!("{}", &row[col_idx])
                            .replace('\n', " ")
                            .replace('\r', "")
                            .replace('|', "\\|")
                            .trim()
                            .to_string()
                    } else {
                        String::new()
                    };
                    row_cells.push(cell_text);
                }

                sheet_text.push_str("| ");
                sheet_text.push_str(&row_cells.join(" | "));
                sheet_text.push_str(" |\n");

                // 在第一行输出后补上 Markdown 的表头分界线
                if is_first_row {
                    sheet_text.push_str("|");
                    for _ in 0..=max_col {
                        sheet_text.push_str("---|");
                    }
                    sheet_text.push('\n');
                    is_first_row = false;
                }
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
