""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os
import copy

import cv2
from omegaconf import OmegaConf
from PIL import Image
from bs4 import BeautifulSoup

from chat import DOLPHIN
from utils.utils import *


def parse_html_to_json(html_text):
    """Parse HTML content into structured JSON format using BeautifulSoup
    
    Args:
        html_text: String containing HTML content
        
    Returns:
        dict: Structured representation of the HTML content
    """
    if not html_text or not isinstance(html_text, str):
        return {"type": "text", "content": html_text}
    
    # Check if the text contains HTML tags
    if not ('<' in html_text and '>' in html_text):
        return {"type": "text", "content": html_text}
    
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # Check if it's primarily a table
        tables = soup.find_all('table')
        if tables:
            parsed_tables = []
            for table in tables:
                table_data = {
                    "type": "table",
                    "rows": []
                }
                
                # Extract table rows
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = []
                    for cell in cells:
                        cell_text = cell.get_text(strip=True)
                        cell_info = {
                            "content": cell_text,
                            "tag": cell.name,
                            "colspan": int(cell.get('colspan', 1)) if cell.get('colspan', '1').isdigit() else 1,
                            "rowspan": int(cell.get('rowspan', 1)) if cell.get('rowspan', '1').isdigit() else 1
                        }
                        row_data.append(cell_info)
                    table_data["rows"].append(row_data)
                
                parsed_tables.append(table_data)
            
            if len(parsed_tables) == 1:
                return parsed_tables[0]
            else:
                return {"type": "multiple_tables", "tables": parsed_tables}
        
        # Check for other structured elements
        structured_content = []
        
        # Parse lists
        lists = soup.find_all(['ul', 'ol'])
        for list_elem in lists:
            list_items = list_elem.find_all('li')
            list_data = {
                "type": "list",
                "list_type": list_elem.name,
                "items": [item.get_text(strip=True) for item in list_items]
            }
            structured_content.append(list_data)
        
        # Parse headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            heading_data = {
                "type": "heading",
                "level": int(heading.name[1]),
                "content": heading.get_text(strip=True)
            }
            structured_content.append(heading_data)
        
        # Parse paragraphs
        paragraphs = soup.find_all('p')
        for para in paragraphs:
            para_data = {
                "type": "paragraph",
                "content": para.get_text(strip=True)
            }
            structured_content.append(para_data)
        
        # If we found structured content, return it
        if structured_content:
            if len(structured_content) == 1:
                return structured_content[0]
            else:
                return {"type": "mixed_content", "elements": structured_content}
        
        # If no specific structure found, return the text content
        text_content = soup.get_text(strip=True)
        return {"type": "text", "content": text_content}
        
    except Exception as e:
        print(f"Error parsing HTML with BeautifulSoup: {str(e)}")
        # Return original text if parsing fails
        return {"type": "text", "content": html_text, "parse_error": str(e)}


def process_document(document_path, model, save_dir, max_batch_size):
    """Parse documents - Handles both images and PDFs"""
    file_ext = os.path.splitext(document_path)[1].lower()
    
    if file_ext == '.pdf':
        # Process PDF file
        # Convert PDF to images
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"Failed to convert PDF {document_path} to images")
        
        all_results = []
        
        # Process each page
        for page_idx, pil_image in enumerate(images):
            print(f"Processing page {page_idx + 1}/{len(images)}")
            
            # Generate output name for this page
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"
            
            # Process this page (don't save individual page results)
            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, save_individual=False
            )
            
            # Add page information to results
            page_results = {
                "page_number": page_idx + 1,
                "elements": recognition_results
            }
            all_results.append(page_results)
        
        # Save combined results for multi-page PDF with HTML parsing
        combined_json_path = save_combined_pdf_results_with_html_parsing(all_results, document_path, save_dir)
        
        return combined_json_path, all_results

    else:
        # Process regular image file
        pil_image = Image.open(document_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(pil_image, model, save_dir, base_name, max_batch_size)


def process_single_image(image, model, save_dir, image_name, max_batch_size, save_individual=True):
    """Process a single image (either from file or converted from PDF page)
    
    Args:
        image: PIL Image object
        model: DOLPHIN model instance
        save_dir: Directory to save results
        image_name: Name for the output file
        max_batch_size: Maximum batch size for processing
        save_individual: Whether to save individual results (False for PDF pages)
        
    Returns:
        Tuple of (json_path, recognition_results)
    """
    # Stage 1: Page-level layout and reading order parsing
    layout_output = model.chat("Parse the reading order of this document.", image)

    # Stage 2: Element-level content parsing
    padded_image, dims = prepare_image(image)
    recognition_results = process_elements(layout_output, padded_image, dims, model, max_batch_size, save_dir, image_name)

    # Save outputs only if requested (skip for PDF pages)
    json_path = None
    if save_individual:
        # Create a dummy image path for save_outputs function
        dummy_image_path = f"{image_name}.jpg"  # Extension doesn't matter, only basename is used
        json_path = save_outputs_with_html_parsing(recognition_results, dummy_image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results, padded_image, dims, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    layout_results = parse_layout_string(layout_results)

    text_table_elements = []  # Elements that need processing
    figure_results = []  # Figure elements (no processing needed)
    previous_box = None
    reading_order = 0

    # Collect elements for processing
    for bbox, label in layout_results:
        try:
            # Adjust coordinates
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # Crop and parse element
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0:
                if label == "fig":
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    
                    # 修改：保存figure到本地文件而不是base64
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    
                    # For figure regions, store relative path instead of base64
                    figure_results.append(
                        {
                            "label": label,
                            "text": f"![Figure](figures/{figure_filename})",  # 相对路径
                            "figure_path": f"figures/{figure_filename}",  # 添加专门的路径字段
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }
                    )
                else:
                    # For text or table regions, prepare for parsing
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    prompt = "Parse the table in the image." if label == "tab" else "Read text in the image."
                    text_table_elements.append(
                        {
                            "crop": pil_crop,
                            "prompt": prompt,
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }
                    )

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    # Parse text/table elements in parallel
    recognition_results = figure_results
    if text_table_elements:
        crops_list = [elem["crop"] for elem in text_table_elements]
        prompts_list = [elem["prompt"] for elem in text_table_elements]

        # Inference in batch
        batch_results = model.chat(prompts_list, crops_list, max_batch_size=max_batch_size)

        # Add batch results to recognition_results
        for i, result in enumerate(batch_results):
            elem = text_table_elements[i]
            recognition_results.append(
                {
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                }
            )

    # Sort elements by reading order
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def save_combined_pdf_results_with_html_parsing(all_page_results, pdf_path, save_dir):
    """Save combined results for multi-page PDF with both JSON and Markdown, including HTML parsing for JSON
    
    Args:
        all_page_results: List of results for all pages
        pdf_path: Path to original PDF file
        save_dir: Directory to save results
        
    Returns:
        Path to saved combined JSON file
    """
    # Create output filename based on PDF name
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Process all page results to add HTML parsing to JSON
    processed_page_results = []
    for page_data in all_page_results:
        processed_page = copy.deepcopy(page_data)
        processed_elements = []
        
        for element in page_data.get("elements", []):
            processed_element = copy.deepcopy(element)
            
            # Parse HTML content in the text field for JSON output
            if 'text' in processed_element and processed_element['text']:
                parsed_content = parse_html_to_json(processed_element['text'])
                processed_element['parsed_content'] = parsed_content
            
            processed_elements.append(processed_element)
        
        processed_page["elements"] = processed_elements
        processed_page_results.append(processed_page)
    
    # Prepare combined results with HTML parsing
    combined_results = {
        "source_file": pdf_path,
        "total_pages": len(processed_page_results),
        "pages": processed_page_results
    }
    
    # Save combined JSON results with HTML parsing
    json_filename = f"{base_name}.json"
    json_path = os.path.join(save_dir, "recognition_json", json_filename)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    # Generate and save combined markdown (using original data without HTML parsing)
    try:
        markdown_converter = MarkdownConverter()
        
        # Combine all page results into a single list for markdown conversion
        all_elements = []
        for page_data in all_page_results:  # Use original data for markdown
            page_elements = page_data.get("elements", [])
            if page_elements:
                # Add page separator if not the first page
                if all_elements:
                    all_elements.append({
                        "label": "page_separator",
                        "text": f"\n\n---\n\n",
                        "reading_order": len(all_elements)
                    })
                all_elements.extend(page_elements)
        
        # Generate markdown content
        markdown_content = markdown_converter.convert(all_elements)
        
        # Save markdown file
        markdown_filename = f"{base_name}.md"
        markdown_path = os.path.join(save_dir, "markdown", markdown_filename)
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
    except ImportError:
        print("MarkdownConverter not available, skipping markdown generation")
    except Exception as e:
        print(f"Error generating markdown: {e}")
    
    return json_path


def save_outputs_with_html_parsing(recognition_results, image_path, save_dir):
    """Save JSON and markdown outputs with HTML parsing for JSON"""
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Create a copy of recognition_results for JSON with parsed HTML
    json_results = []
    for result in recognition_results:
        json_result = copy.deepcopy(result)
        
        # Parse HTML content in the text field for JSON output
        if 'text' in json_result and json_result['text']:
            parsed_content = parse_html_to_json(json_result['text'])
            json_result['parsed_content'] = parsed_content
            # Keep original text for backward compatibility but add parsed version
        
        json_results.append(json_result)

    # Save JSON file with parsed HTML content
    json_path = os.path.join(save_dir, "recognition_json", f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    # Generate and save markdown file (using original recognition_results without HTML parsing)
    markdown_converter = MarkdownConverter()
    markdown_content = markdown_converter.convert(recognition_results)
    markdown_path = os.path.join(save_dir, "markdown", f"{basename}.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return json_path


def main():
    parser = argparse.ArgumentParser(description="Document parsing based on DOLPHIN")
    parser.add_argument("--config", default="./config/Dolphin.yaml", help="Path to configuration file")
    parser.add_argument("--input_path", type=str, default="./demo", help="Path to input image/PDF or directory of files")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input directory)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=4,
        help="Maximum number of document elements to parse in a single batch (default: 4)",
    )
    args = parser.parse_args()

    # Load Model
    config = OmegaConf.load(args.config)
    model = DOLPHIN(config)

    # Collect Document Files (images and PDFs)
    if os.path.isdir(args.input_path):
        # Support both image and PDF files
        file_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
        
        document_files = []
        for ext in file_extensions:
            document_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        document_files = sorted(document_files)
    else:
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input path {args.input_path} does not exist")
        
        # Check if it's a supported file type
        file_ext = os.path.splitext(args.input_path)[1].lower()
        supported_exts = ['.jpg', '.jpeg', '.png', '.pdf']
        
        if file_ext not in supported_exts:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported types: {supported_exts}")
        
        document_files = [args.input_path]

    save_dir = args.save_dir or (
        args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    )
    setup_output_dirs(save_dir)

    total_samples = len(document_files)
    print(f"\nTotal files to process: {total_samples}")

    # Process All Document Files
    for file_path in document_files:
        print(f"\nProcessing {file_path}")
        try:
            json_path, recognition_results = process_document(
                document_path=file_path,
                model=model,
                save_dir=save_dir,
                max_batch_size=args.max_batch_size,
            )

            print(f"Processing completed. Results saved to {save_dir}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()