use super::ExtractedText;
use std::sync::OnceLock;
use std::process::Command;
use std::path::Path;

static OCR_INITIALIZED: OnceLock<bool> = OnceLock::new();

/// 初始化本地 OCR 和 PDF 渲染环境
fn init_ocr_dependencies() -> Result<(), String> {
    if OCR_INITIALIZED.get().is_some() {
        return Ok(());
    }

    let dll_path = Path::new("pdfium.dll");
    if !dll_path.exists() {
        tracing::info!("Downloading pdfium.dll...");
        let url = "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.zip";
        let status = Command::new("powershell")
            .args([
                 "-Command",
                 &format!("Invoke-WebRequest -Uri '{}' -OutFile 'pdfium.zip'; Expand-Archive -Path 'pdfium.zip' -DestinationPath '.' -Force; Remove-Item 'pdfium.zip'", url)
            ])
            .status()
            .map_err(|e| format!("PowerShell error: {}", e))?;
        if !status.success() {
            return Err("Failed to download pdfium".to_string());
        }
    }

    let det_model = "text-detection.rten";
    let rec_model = "text-recognition.rten";

    for model in &[det_model, rec_model] {
        if !Path::new(model).exists() {
            tracing::info!("Downloading {}...", model);
            let url = format!("https://ocrs-models.s3-accelerate.amazonaws.com/{}", model);
            let status = Command::new("powershell")
                .args([
                    "-Command",
                    &format!("Invoke-WebRequest -Uri '{}' -OutFile '{}'", url, model)
                ])
                .status()
                .map_err(|e| format!("PowerShell download err: {}", e))?;
            if !status.success() {
                return Err(format!("Failed to download model {}", model));
            }
        }
    }

    let _ = OCR_INITIALIZED.set(true);
    Ok(())
}

/// 使用 OCR 提取扫描版 PDF 的内容
fn extract_with_ocr(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    init_ocr_dependencies()?;

    // Load models
    let det_data = std::fs::read("text-detection.rten").map_err(|e| e.to_string())?;
    let rec_data = std::fs::read("text-recognition.rten").map_err(|e| e.to_string())?;
    let det_model = rten::Model::load(det_data).map_err(|e| e.to_string())?;
    let rec_model = rten::Model::load(rec_data).map_err(|e| e.to_string())?;

    // Create OCR Engine
    let engine = ocrs::OcrEngine::new(ocrs::OcrEngineParams {
        detection_model: Some(det_model),
        recognition_model: Some(rec_model),
        ..Default::default()
    }).map_err(|e| e.to_string())?;

    // Bind and initialize Pdfium
    let bindings = pdfium_render::prelude::Pdfium::bind_to_library(
        pdfium_render::prelude::Pdfium::pdfium_platform_library_name_at_path("./")
    ).or_else(|_| pdfium_render::prelude::Pdfium::bind_to_system_library())
     .map_err(|e| format!("Pdfium binding fail: {:?}", e))?;
     
    let pdfium = pdfium_render::prelude::Pdfium::new(bindings);
    let doc = pdfium.load_pdf_from_file(file_path, None).map_err(|e| format!("Load PDF fail: {:?}", e))?;
    
    let mut results = Vec::new();

    for (i, page) in doc.pages().iter().enumerate() {
        // Render at a high resolution (2000px width)
        let render_config = pdfium_render::prelude::PdfRenderConfig::new().set_target_width(2000);
        let mut bitmap = page.render_with_config(&render_config).map_err(|e| format!("Render page fail: {:?}", e))?;
        
        let dynamic_img = bitmap.as_image(); 
        let rgb_img = dynamic_img.into_rgb8();
        let (width, height) = rgb_img.dimensions();
        
        // ocrs 0.12+ uses NdTensor layout `[1, 1, height, width]` for grayscale (via rten_imageproc)
        // Or we can use ImageSource struct
        let img_source = ocrs::ImageSource::from_bytes(rgb_img.as_raw(), (width, height))
               .map_err(|e| format!("ImageSource err: {:?}", e))?;

        let ocr_input = engine.prepare_input(img_source)
            .map_err(|e| format!("OCR prep fail: {:?}", e))?;
            
        let word_rects = engine.detect_words(&ocr_input)
            .map_err(|e| format!("Detect words fail: {:?}", e))?;
            
        let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
        
        let text_lines = engine.recognize_text(&ocr_input, &line_rects)
            .map_err(|e| format!("Recognize text fail: {:?}", e))?;

        let text: String = text_lines
            .into_iter()
            .filter_map(|l| l)
            .map(|l| l.to_string())
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

/// 从 PDF 文件中提取文本（按页）
pub fn extract(file_path: &str) -> Result<Vec<ExtractedText>, String> {
    let bytes = std::fs::read(file_path)
        .map_err(|e| format!("读取 PDF 文件失败: {}", e))?;

    let text_result = pdf_extract::extract_text_from_mem(&bytes);
    let mut text = String::new();
    if let Ok(t) = text_result {
        text = t;
    }

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

    // 如果未提取出有意义的文本内容，则可能是扫描版或者图片构成的 PDF
    // 降级使用 OCR 执行提取
    if results.is_empty() {
        tracing::info!("PDF doesn't contain standard extractable text. Falling back to OCR...");
        return extract_with_ocr(file_path);
    }

    Ok(results)
}
