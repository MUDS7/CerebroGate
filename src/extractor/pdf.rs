use super::ExtractedText;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

static OCR_INITIALIZED: OnceLock<Result<(), String>> = OnceLock::new();

fn project_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn pdfium_dir() -> PathBuf {
    project_root().join("data").join("pdfium")
}

fn ocr_model_dir() -> PathBuf {
    project_root().join("data").join("ocrs")
}

fn download_with_curl(url: &str, output: &Path) -> Result<(), String> {
    let output_str = output
        .to_str()
        .ok_or_else(|| format!("无法处理下载目标路径: {}", output.display()))?;

    let status = Command::new("curl.exe")
        .args(["-L", "--http1.1", "-o", output_str, url])
        .status()
        .map_err(|e| format!("调用 curl 下载失败: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("下载失败: {}", url))
    }
}

fn extract_tgz(archive_path: &Path, destination: &Path) -> Result<(), String> {
    let archive_str = archive_path
        .to_str()
        .ok_or_else(|| format!("无法处理压缩包路径: {}", archive_path.display()))?;
    let dest_str = destination
        .to_str()
        .ok_or_else(|| format!("无法处理解压目录路径: {}", destination.display()))?;

    let status = Command::new("tar")
        .args(["-xzf", archive_str, "-C", dest_str])
        .status()
        .map_err(|e| format!("调用 tar 解压 pdfium 失败: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("解压 pdfium 压缩包失败: {}", archive_path.display()))
    }
}

fn ensure_pdfium() -> Result<PathBuf, String> {
    let dll_path = pdfium_dir().join("bin").join("pdfium.dll");
    if dll_path.exists() {
        return Ok(dll_path);
    }

    let dir = pdfium_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("创建 pdfium 目录失败: {}", e))?;

    let archive_path = dir.join("pdfium-win-x64.tgz");
    let url = "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz";
    tracing::info!("正在下载 pdfium 到 {}", archive_path.display());
    download_with_curl(url, &archive_path)?;
    extract_tgz(&archive_path, &dir)?;
    let _ = std::fs::remove_file(&archive_path);

    if dll_path.exists() {
        Ok(dll_path)
    } else {
        Err(format!(
            "pdfium 解压后未找到 DLL: {}",
            dll_path.display()
        ))
    }
}

fn ensure_ocr_model(file_name: &str) -> Result<PathBuf, String> {
    let path = ocr_model_dir().join(file_name);
    if path.exists() {
        return Ok(path);
    }

    std::fs::create_dir_all(ocr_model_dir()).map_err(|e| format!("创建 OCR 模型目录失败: {}", e))?;

    let url = format!(
        "https://ocrs-models.s3-accelerate.amazonaws.com/{}",
        file_name
    );
    tracing::info!("正在下载 OCR 模型 {} 到 {}", file_name, path.display());
    download_with_curl(&url, &path)?;

    if path.exists() {
        Ok(path)
    } else {
        Err(format!("OCR 模型下载后缺失: {}", path.display()))
    }
}

fn init_ocr_dependencies() -> Result<(), String> {
    OCR_INITIALIZED
        .get_or_init(|| {
            ensure_pdfium()?;
            ensure_ocr_model("text-detection.rten")?;
            ensure_ocr_model("text-recognition.rten")?;
            Ok(())
        })
        .clone()
}

fn extract_with_ocr(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    init_ocr_dependencies()?;

    let det_data = std::fs::read(ocr_model_dir().join("text-detection.rten"))
        .map_err(|e| format!("读取 OCR 检测模型失败: {}", e))?;
    let rec_data = std::fs::read(ocr_model_dir().join("text-recognition.rten"))
        .map_err(|e| format!("读取 OCR 识别模型失败: {}", e))?;
    let det_model = rten::Model::load(det_data).map_err(|e| e.to_string())?;
    let rec_model = rten::Model::load(rec_data).map_err(|e| e.to_string())?;

    let engine = ocrs::OcrEngine::new(ocrs::OcrEngineParams {
        detection_model: Some(det_model),
        recognition_model: Some(rec_model),
        ..Default::default()
    })
    .map_err(|e| e.to_string())?;

    let bindings = pdfium_render::prelude::Pdfium::bind_to_library(
        pdfium_render::prelude::Pdfium::pdfium_platform_library_name_at_path(
            &pdfium_dir().join("bin"),
        ),
    )
    .or_else(|_| pdfium_render::prelude::Pdfium::bind_to_system_library())
    .map_err(|e| format!("Pdfium 绑定失败: {:?}", e))?;

    let pdfium = pdfium_render::prelude::Pdfium::new(bindings);
    let doc = pdfium
        .load_pdf_from_file(file_path, None)
        .map_err(|e| format!("加载 PDF 失败: {:?}", e))?;

    let mut results = Vec::new();

    for (i, page) in doc.pages().iter().enumerate() {
        let render_config = pdfium_render::prelude::PdfRenderConfig::new().set_target_width(2000);
        let bitmap = page
            .render_with_config(&render_config)
            .map_err(|e| format!("渲染 PDF 页面失败: {:?}", e))?;

        let dynamic_img = bitmap.as_image();
        let rgb_img = dynamic_img.into_rgb8();
        let (width, height) = rgb_img.dimensions();

        let img_source = ocrs::ImageSource::from_bytes(rgb_img.as_raw(), (width, height))
            .map_err(|e| format!("构建 OCR 输入失败: {:?}", e))?;

        let ocr_input = engine
            .prepare_input(img_source)
            .map_err(|e| format!("OCR 预处理失败: {:?}", e))?;

        let word_rects = engine
            .detect_words(&ocr_input)
            .map_err(|e| format!("OCR 词检测失败: {:?}", e))?;
        let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
        let text_lines = engine
            .recognize_text(&ocr_input, &line_rects)
            .map_err(|e| format!("OCR 识别失败: {:?}", e))?;

        let text = text_lines
            .into_iter()
            .flatten()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let trimmed = text.trim();
        if !trimmed.is_empty() {
            results.push(ExtractedText {
                page_number: (i + 1) as i32,
                text: trimmed.to_string(),
            });
        }
    }

    Ok(results)
}

pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let bytes = std::fs::read(file_path).map_err(|e| format!("读取 PDF 文件失败: {}", e))?;

    let text = pdf_extract::extract_text_from_mem(&bytes).unwrap_or_default();
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

    if results.is_empty() {
        tracing::info!("PDF 未提取到标准文本，切换到 OCR 流程");
        return extract_with_ocr(file_path);
    }

    Ok(results)
}
