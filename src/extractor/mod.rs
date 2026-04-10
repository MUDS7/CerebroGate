pub mod csv_ext;
pub mod docx;
pub mod excel;
pub mod pdf;
pub mod plaintext;

#[derive(Debug, Clone)]
pub struct ExtractedText {
    pub page_number: i32,
    pub text: String,
}

pub fn extract_text(file_path: &str, extension: &str) -> Result<Vec<ExtractedText>, String> {
    match extension {
        "pdf" => pdf::extract(file_path),
        "docx" => docx::extract(file_path),
        "doc" => doc_extract(file_path),
        "xlsx" | "xls" => excel::extract(file_path),
        "csv" => csv_ext::extract(file_path),
        "md" | "txt" => plaintext::extract(file_path),
        "png" | "jpg" | "jpeg" => Ok(vec![]),
        _ => Err(format!("不支持的文件格式: {}", extension)),
    }
}

fn doc_extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    if let Ok(text) = extract_doc_with_litchi(file_path) {
        if !text.trim().is_empty() {
            return Ok(vec![ExtractedText {
                page_number: 1,
                text,
            }]);
        }
    }

    if let Ok(text) = extract_doc_from_raw_utf16(file_path) {
        if !text.trim().is_empty() {
            return Ok(vec![ExtractedText {
                page_number: 1,
                text,
            }]);
        }
    }

    doc_extract_with_libreoffice(file_path)
}

fn extract_doc_with_litchi(file_path: &str) -> Result<String, String> {
    let document = litchi::Document::open(file_path)
        .map_err(|e| format!("纯 Rust DOC 解析失败: {}", e))?;
    let text = document
        .text()
        .map_err(|e| format!("提取 DOC 文本失败: {}", e))?;
    Ok(text.trim().to_string())
}

fn doc_extract_with_libreoffice(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    use std::path::Path;
    use std::process::Command;

    let path = Path::new(file_path);
    let parent_dir = path.parent().unwrap_or(Path::new("."));

    let output = Command::new("soffice")
        .args([
            "--headless",
            "--convert-to",
            "txt:Text",
            "--outdir",
            parent_dir.to_str().unwrap_or("."),
            file_path,
        ])
        .output()
        .map_err(|e| {
            format!(
                "DOC 提取失败。纯 Rust 解析不可用，且调用 LibreOffice 失败（请确认已安装 LibreOffice）: {}",
                e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("LibreOffice 转换失败: {}", stderr));
    }

    let txt_path = path.with_extension("txt");
    let text = std::fs::read_to_string(&txt_path)
        .map_err(|e| format!("读取 LibreOffice 转换后的文本失败: {}", e))?;
    let _ = std::fs::remove_file(&txt_path);

    Ok(vec![ExtractedText {
        page_number: 1,
        text,
    }])
}

fn extract_doc_from_raw_utf16(file_path: &str) -> Result<String, String> {
    let bytes = std::fs::read(file_path).map_err(|e| format!("读取 DOC 文件失败: {}", e))?;
    if bytes.len() < 2 {
        return Err("DOC 文件内容过短".to_string());
    }

    let even_len = bytes.len() - (bytes.len() % 2);
    let utf16: Vec<u16> = bytes[..even_len]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let decoded = String::from_utf16_lossy(&utf16);
    let normalized = decoded
        .chars()
        .map(|ch| if is_doc_text_char(ch) { ch } else { '\n' })
        .collect::<String>();

    let lines = normalized
        .lines()
        .map(str::trim)
        .filter(|line| line.chars().count() >= 8)
        .filter(|line| {
            let meaningful = line.chars().filter(|c| is_meaningful_text_char(*c)).count();
            meaningful * 2 >= line.chars().count()
        })
        .collect::<Vec<_>>();

    let text = lines.join("\n");
    if text.trim().is_empty() {
        Err("原始 DOC 字节流未提取到有效文本".to_string())
    } else {
        Ok(text)
    }
}

fn is_doc_text_char(ch: char) -> bool {
    is_meaningful_text_char(ch) || ch.is_whitespace()
}

fn is_meaningful_text_char(ch: char) -> bool {
    ch.is_alphanumeric()
        || matches!(
            ch,
            '\u{4E00}'..='\u{9FFF}'
                | '\u{3400}'..='\u{4DBF}'
                | '，'
                | '。'
                | '；'
                | '：'
                | '、'
                | '（'
                | '）'
                | '《'
                | '》'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '-'
                | '_'
                | '+'
                | '/'
                | '\\'
                | '.'
                | ','
                | ':'
                | ';'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '%'
                | '#'
                | '@'
                | '&'
                | '*'
                | '='
                | '<'
                | '>'
                | '"'
                | '\''
        )
}
