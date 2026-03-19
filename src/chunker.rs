use crate::extractor::ExtractedText;

/// 切片后的文本块
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// 文本内容
    pub text: String,
    /// 来源页码
    pub page_number: i32,
    /// 切片在文件中的全局序号（从 0 开始）
    pub chunk_index: i32,
}

/// 将提取出的文本按指定大小切片
///
/// # 参数
/// - `extracted`: 提取出的文本列表（每个元素对应一页/一个 Sheet）
/// - `chunk_size`: 每个切片的最大字符数
/// - `chunk_overlap`: 相邻切片之间的重叠字符数
pub fn chunk_text(
    extracted: &[ExtractedText],
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<TextChunk> {
    let mut chunks = Vec::new();
    let mut global_index: i32 = 0;

    for page in extracted {
        let text = &page.text;
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();

        if total == 0 {
            continue;
        }

        // 如果文本长度小于等于 chunk_size，直接作为一个切片
        if total <= chunk_size {
            chunks.push(TextChunk {
                text: text.clone(),
                page_number: page.page_number,
                chunk_index: global_index,
            });
            global_index += 1;
            continue;
        }

        // 滑动窗口切片
        let step = chunk_size.saturating_sub(chunk_overlap).max(1);
        let mut start = 0;

        while start < total {
            let end = (start + chunk_size).min(total);
            let chunk_text: String = chars[start..end].iter().collect();

            chunks.push(TextChunk {
                text: chunk_text,
                page_number: page.page_number,
                chunk_index: global_index,
            });
            global_index += 1;

            // 如果已到末尾，退出
            if end >= total {
                break;
            }

            start += step;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::ExtractedText;

    #[test]
    fn test_chunk_short_text() {
        let extracted = vec![ExtractedText {
            page_number: 1,
            text: "短文本".to_string(),
        }];
        let chunks = chunk_text(&extracted, 500, 50);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "短文本");
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn test_chunk_with_overlap() {
        // 创建一个 10 字符的文本，chunk_size=4, overlap=2
        let extracted = vec![ExtractedText {
            page_number: 1,
            text: "0123456789".to_string(),
        }];
        let chunks = chunk_text(&extracted, 4, 2);
        // step = 4 - 2 = 2
        // chunk 0: [0,4) = "0123"
        // chunk 1: [2,6) = "2345"
        // chunk 2: [4,8) = "4567"
        // chunk 3: [6,10) = "6789"
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].text, "0123");
        assert_eq!(chunks[1].text, "2345");
        assert_eq!(chunks[2].text, "4567");
        assert_eq!(chunks[3].text, "6789");
    }
}
