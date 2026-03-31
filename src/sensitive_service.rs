use crate::handler::AppState;
use crate::sensitive::{SENSITIVE_WORDS_TABLE, SensitiveWord};
use arrow_array::{RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use tracing::warn;

use regex::Regex;

/// 获取所有敏感词列表
pub async fn get_all_sensitive_words(state: &AppState) -> Vec<SensitiveWord> {
    let table = match state.db.open_table(SENSITIVE_WORDS_TABLE).execute().await {
        Ok(t) => t,
        Err(e) => {
            warn!("打开敏感词库失败: {}", e);
            return Vec::new();
        }
    };

    let batches: Vec<RecordBatch> = match table.query().execute().await {
        Ok(stream) => match stream.try_collect().await {
            Ok(b) => b,
            Err(e) => {
                warn!("收集敏感词结果失败: {}", e);
                return Vec::new();
            }
        },
        Err(e) => {
            warn!("查询敏感词失败: {}", e);
            return Vec::new();
        }
    };

    let mut words = Vec::new();
    for batch in &batches {
        let word_col = batch.column_by_name("word").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let created_at_col = batch.column_by_name("created_at").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let match_strategy_col = batch.column_by_name("match_strategy").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let replace_strategy_col = batch.column_by_name("replace_strategy").and_then(|c| c.as_any().downcast_ref::<StringArray>());

        if let (Some(w), Some(c), Some(m), Some(r)) = (word_col, created_at_col, match_strategy_col, replace_strategy_col) {
            for i in 0..batch.num_rows() {
                words.push(SensitiveWord {
                    word: w.value(i).to_string(),
                    created_at: c.value(i).to_string(),
                    match_strategy: m.value(i).to_string(),
                    replace_strategy: r.value(i).to_string(),
                });
            }
        }
    }
    words
}

/// 处理敏感词替换
pub fn process_sensitive_words(content: &str, sensitive_words: &[SensitiveWord]) -> String {
    let mut result = content.to_string();
    for sw in sensitive_words {
        let strategy_raw = sw.match_strategy.trim();
        let strategy = strategy_raw.to_lowercase();
        let replacement_for = |matched: &str| match sw.replace_strategy.as_str() {
            "remove" => "".to_string(),
            "mask" => "*".repeat(matched.chars().count()),
            custom => custom.to_string(),
        };

        if strategy == "exact" {
            if result.contains(&sw.word) {
                let replacement = replacement_for(&sw.word);
                result = result.replace(&sw.word, &replacement);
            }
            continue;
        }

        if matches!(strategy.as_str(), "fuzzy" | "regex" | "re") {
            warn!(
                "敏感词配置不符合约定: match_strategy='{}'，请直接填写正则到 match_strategy 字段",
                sw.match_strategy
            );
            continue;
        }

        // match_strategy 直接作为正则匹配
        let mut patterns = vec![strategy_raw.to_string()];
        if strategy_raw.contains("\\\\") {
            let normalized = strategy_raw.replace("\\\\", "\\");
            if normalized != strategy_raw {
                patterns.push(normalized);
            }
        }

        for pattern in patterns {
            match Regex::new(&pattern) {
                Ok(re) => {
                    if re.is_match(&result) {
                        result = re
                            .replace_all(&result, |caps: &regex::Captures| {
                                let m = caps.get(0).map(|m| m.as_str()).unwrap_or("");
                                replacement_for(m)
                            })
                            .into_owned();
                        break;
                    }
                }
                Err(e) => {
                    warn!("敏感词正则解析失败: '{}', 错误: {}", pattern, e);
                }
            }
        }
    }
    result
}
