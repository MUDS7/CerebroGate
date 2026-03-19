pub mod pdf;
pub mod docx;
pub mod excel;
pub mod csv_ext;
pub mod plaintext;

/// 从文件中提取出的文本片段，附带页码信息
#[derive(Debug, Clone)]
pub struct ExtractedText {
    /// 页码（PDF 有效，其他格式从 1 开始递增）
    pub page_number: i32,
    /// 提取出的文本内容
    pub text: String,
}

/// 根据文件扩展名提取文本
///
/// 对于 PNG/JPG 图片文件，返回空的 Vec（不做文本提取）
pub fn extract_text(file_path: &str, extension: &str) -> Result<Vec<ExtractedText>, String> {
    match extension {
        "pdf" => pdf::extract(file_path),
        "docx" => docx::extract(file_path),
        "doc" => doc_extract(file_path),
        "xlsx" | "xls" => excel::extract(file_path),
        "csv" => csv_ext::extract(file_path),
        "md" | "txt" => plaintext::extract(file_path),
        "png" | "jpg" | "jpeg" => Ok(vec![]), // 图片不做文本提取
        _ => Err(format!("不支持的文件格式: {}", extension)),
    }
}

/// DOC 格式提取：调用 LibreOffice 命令行转换为文本
fn doc_extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    use std::process::Command;
    use std::path::Path;

    let path = Path::new(file_path);
    let parent_dir = path.parent().unwrap_or(Path::new("."));

    // 使用 LibreOffice 将 .doc 转换为 .txt
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
        .map_err(|e| format!("调用 LibreOffice 失败（请确保已安装 LibreOffice）: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("LibreOffice 转换失败: {}", stderr));
    }

    // 读取生成的 .txt 文件
    let txt_path = path.with_extension("txt");
    let text = std::fs::read_to_string(&txt_path)
        .map_err(|e| format!("读取转换后的文本文件失败: {}", e))?;

    // 清理临时文件
    let _ = std::fs::remove_file(&txt_path);

    Ok(vec![ExtractedText {
        page_number: 1,
        text,
    }])
}
