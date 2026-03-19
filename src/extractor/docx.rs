use super::ExtractedText;
use std::io::Read;

/// 从 DOCX 文件中提取文本
///
/// DOCX 本质是 ZIP 压缩包，核心内容在 word/document.xml 中
pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let file = std::fs::File::open(file_path)
        .map_err(|e| format!("打开 DOCX 文件失败: {}", e))?;

    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("解析 DOCX ZIP 结构失败: {}", e))?;

    // 读取 word/document.xml
    let mut xml_content = String::new();
    {
        let mut doc_file = archive
            .by_name("word/document.xml")
            .map_err(|e| format!("DOCX 中未找到 document.xml: {}", e))?;
        doc_file
            .read_to_string(&mut xml_content)
            .map_err(|e| format!("读取 document.xml 失败: {}", e))?;
    }

    // 从 XML 中提取纯文本
    let text = extract_text_from_xml(&xml_content);

    if text.trim().is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![ExtractedText {
        page_number: 1,
        text: text.trim().to_string(),
    }])
}

/// 从 DOCX 的 XML 内容中提取纯文本
///
/// 主要提取 <w:t> 标签中的内容，遇到 <w:p> 结束时添加换行
fn extract_text_from_xml(xml: &str) -> String {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml);
    let mut text = String::new();
    let mut in_w_t = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local_name = e.local_name();
                if local_name.as_ref() == b"t" {
                    in_w_t = true;
                }
            }
            Ok(Event::End(ref e)) => {
                let local_name = e.local_name();
                if local_name.as_ref() == b"t" {
                    in_w_t = false;
                } else if local_name.as_ref() == b"p" {
                    // 段落结束，添加换行
                    text.push('\n');
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_w_t {
                    if let Ok(t) = e.unescape() {
                        text.push_str(&t);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    text
}
