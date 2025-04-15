import gradio as gr
import google.generativeai as genai
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
import time
import traceback
import tempfile
from dotenv import load_dotenv 

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Error configuring Gemini SDK: {e}")
        GOOGLE_API_KEY = None 

MODEL_NAME = "gemini-2.0-flash" 

async def extract_text_from_url(session, url):
    """Asynchronously fetches and extracts text content from a URL."""
    print(f"Fetching: {url}...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        async with session.get(url, timeout=60, headers=headers, allow_redirects=True) as response:
            response.raise_for_status() 
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                print(f"Warning: Content-Type for {url} is '{content_type}', not text/html. Attempting basic text extraction.")
                try:
                    text = await response.text(errors='ignore') 
                    if '\0' in text[:1000]: 
                         print(f"Warning: Content for {url} appears to be binary. Skipping.")
                         return None
                    return ' '.join(text.split()) 
                except Exception as text_err:
                    print(f"Error reading non-HTML content as text from {url}: {text_err}")
                    return None

            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button", "img", "svg", "iframe", "link", "meta"]):
                element.decompose()
            main_content = soup.find('main') or \
                           soup.find('article') or \
                           soup.find(id='content') or \
                           soup.find(id='main') or \
                           soup.find(class_='content') or \
                           soup.find(class_='main') or \
                           soup.find(role='main') or \
                           soup.body 

            if main_content:
                text = ' '.join(line.strip() for line in main_content.stripped_strings)
            else: 
                text = ' '.join(line.strip() for line in soup.stripped_strings)

            return text if text else None 
    except asyncio.TimeoutError:
        print(f"Error: Timeout fetching URL {url}")
        return None
    except aiohttp.ClientResponseError as e:
        print(f"Error fetching URL {url}: HTTP {e.status} {e.message}")
        return None
    except aiohttp.ClientError as e: 
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None


async def upload_pdf_to_gemini(file_path):
    """Asynchronously uploads a PDF file to the Gemini File API."""
    print(f"[PDF Worker] Checking PDF: {file_path}...")
    if not GOOGLE_API_KEY:
        print("[PDF Worker] Error: GOOGLE_API_KEY is not configured.")
        return None
    try:
        if not os.path.exists(file_path):
             print(f"[PDF Worker] Error: File not found at {file_path}")
             return None
        if os.path.getsize(file_path) == 0:
             print(f"[PDF Worker] Error: File is empty at {file_path}")
             return None

        print(f"[PDF Worker] Uploading PDF: {file_path}...")
        loop = asyncio.get_running_loop()
        display_name = f"upload_{os.path.basename(file_path)}_{int(time.time())}"

        pdf_file_object = await loop.run_in_executor(
            None, 
            lambda: genai.upload_file(path=file_path, display_name=display_name)
        )
        print(f"[PDF Worker] File uploaded: {pdf_file_object.name} ({pdf_file_object.uri})")

        print("[PDF Worker] Waiting for file processing...")
        start_wait_time = time.time()
        max_wait_time = 180 
        check_interval = 10 

        while True:
            current_time = time.time()
            if current_time - start_wait_time > max_wait_time:
                print(f"[PDF Worker] Error: Timeout waiting for file {pdf_file_object.name} to become active.")
                try:
                    await loop.run_in_executor(None, lambda: genai.delete_file(pdf_file_object.name))
                    print(f"[PDF Worker] Cleaned up timed-out file: {pdf_file_object.name}")
                except Exception as del_e:
                    print(f"[PDF Worker] Warning: Failed to delete timed-out file {pdf_file_object.name}: {del_e}")
                return None

            try:
                file_check = await loop.run_in_executor(
                    None,
                    lambda: genai.get_file(name=pdf_file_object.name)
                )
                state = file_check.state.name
                print(f"[PDF Worker] File '{pdf_file_object.name}' state: {state} (Elapsed: {current_time - start_wait_time:.0f}s)")

                if state == "ACTIVE":
                    print(f"[PDF Worker] File '{pdf_file_object.name}' is ACTIVE.")
                    return pdf_file_object 
                elif state == "FAILED":
                     print(f"[PDF Worker] Error: File processing FAILED for {pdf_file_object.name}.")
                     try:
                         await loop.run_in_executor(None, lambda: genai.delete_file(pdf_file_object.name))
                         print(f"[PDF Worker] Cleaned up failed file: {pdf_file_object.name}")
                     except Exception as del_e:
                          print(f"[PDF Worker] Warning: Failed to delete failed file {pdf_file_object.name}: {del_e}")
                     return None
                await asyncio.sleep(check_interval)

            except Exception as e_check:
                print(f"[PDF Worker] Error checking file status for {pdf_file_object.name}: {e_check}")
                await asyncio.sleep(check_interval * 2) 

    except Exception as e:
        print(f"[PDF Worker] Error uploading or processing PDF {file_path}: {e}\n{traceback.format_exc()}")
        if 'pdf_file_object' in locals() and pdf_file_object and hasattr(pdf_file_object, 'name'):
             try:
                 print(f"[PDF Worker] Attempting cleanup of file {pdf_file_object.name} after error...")
                 await loop.run_in_executor(None, lambda: genai.delete_file(pdf_file_object.name))
                 print(f"[PDF Worker] Cleaned up file {pdf_file_object.name} after error.")
             except Exception as del_e:
                 print(f"[PDF Worker] Warning: Failed to delete file {pdf_file_object.name} during error cleanup: {del_e}")
        return None


async def summarize_content(model, content_source, source_name):
    """
    Asynchronously generates a summary using the Gemini API.
    Correctly handles both text strings and File objects as input.
    """
    print(f"[Summarizer] Generating summary for: {source_name}...")
    if not GOOGLE_API_KEY:
        return f"Error: GOOGLE_API_KEY is not configured. Cannot summarize {source_name}."

    prompt_text = f"Please provide a concise summary of the document sourced from '{source_name}'. Focus on the main topic, key arguments, findings, and important conclusions. Do not include introductory phrases like 'This document discusses...' or 'The summary is...'. Present the summary directly."
    api_payload = []
    content_description = ""

    if isinstance(content_source, str):
        content_description = f"Text (length: {len(content_source)})"
        if not content_source.strip():
            print(f"[Summarizer] Warning: Input text for {source_name} is empty.")
            return "Error: Input text content is empty."

        estimated_tokens = len(content_source) / 3
        max_direct_text_chars = 100_000

        if len(content_source) > max_direct_text_chars:
            print(f"[Summarizer] Warning: Input text for {source_name} is very long ({len(content_source)} chars). Truncating for summary.")
            content_source = content_source[:max_direct_text_chars] + "... [Content Truncated]"

        api_payload = [f"{prompt_text}\n\n--- Document Content ---\n{content_source}\n\n--- Concise Summary ---"]

    elif hasattr(content_source, 'uri') and content_source.uri:
        content_description = f"File Object ({content_source.name}, {content_source.uri})"
        print(f"[Summarizer] Using uploaded file object: {content_source.name} ({content_source.uri})")
        api_payload = [prompt_text, content_source, "\n\n--- Concise Summary ---"]

    else:
        print(f"[Summarizer] Error: Invalid content_source type ({type(content_source)}) for {source_name}.")
        return f"Error: Invalid input type for summarization ({type(content_source)})."

    print(f"[Summarizer] Preparing API call for {source_name} ({content_description})")
    response = None
    try:
        safety_settings = {
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
           }
        generation_config = genai.types.GenerationConfig(
             temperature=0.5,
        )
        request_options = genai.types.RequestOptions(timeout=120) 

        response = await model.generate_content_async(
            api_payload,
            generation_config=generation_config,
            safety_settings=safety_settings,
            request_options=request_options
        )

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason
            print(f"[Summarizer] Warning: Prompt blocked for {source_name}. Reason: {reason}")
            details = ""
            if response.prompt_feedback.safety_ratings:
                 details = ", ".join([f"{sr.category}: {sr.probability}" for sr in response.prompt_feedback.safety_ratings])
            return f"Error: Could not generate summary (prompt blocked - {reason}). Details: {details}"

        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
             finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
             print(f"[Summarizer] Warning: No valid content parts found in response for {source_name}. Finish Reason: {finish_reason}")
             safety_info = ""
             if response.candidates and response.candidates[0].safety_ratings:
                  safety_info = ", ".join([f"{sr.category}: {sr.probability}" for sr in response.candidates[0].safety_ratings])
             return f"Error: Could not generate summary (API response missing valid content). Finish Reason: {finish_reason}. Safety: {safety_info}"

        summary = getattr(response, 'text', None)
        if summary is None:
             try:
                 summary = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
             except Exception:
                  pass 

        if summary is None or not summary.strip():
             print(f"[Summarizer] Warning: Extracted summary text is empty for {source_name}.")
             try:
                 print(f"[Summarizer] Raw response parts: {response.candidates[0].content.parts}")
             except: pass
             return "Error: Could not generate summary (extracted text is empty)."

        print(f"[Summarizer] Finished summary for: {source_name} (Length: {len(summary.strip())})")
        return summary.strip()

    except Exception as e:
        print(f"[Summarizer] Error generating summary for {source_name}: {type(e).__name__} - {e}")
        feedback = getattr(response, 'prompt_feedback', None) or getattr(response, 'candidates[0].finish_reason', None)
        if feedback:
            print(f"[Summarizer] Feedback/Reason: {feedback}")
        print(f"[Summarizer] Full error traceback:\n{traceback.format_exc()}")
        return f"Error generating summary: {type(e).__name__}"

async def process_source(session, model, source):
    """
    Processes a single source (URL or local PDF path).
    For URLs, checks Content-Type to determine if it's PDF or HTML.
    Returns a tuple: (source_display_name, summary_or_error_message)
    """
    source_lower = source.lower().strip()
    final_summary = f"Error: Processing failed for {source}" 
    temp_pdf_path = None 
    pdf_file_object_api = None 
    source_display_name = os.path.basename(source) if source_lower.endswith(".pdf") and not source_lower.startswith("http") else source
    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        print(f"[Router] Processing URL: {source}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/pdf' 
            }
            async with session.get(source, timeout=60, headers=headers, allow_redirects=True) as response:
                response.raise_for_status() 
                content_type = response.headers.get('Content-Type', '').lower().split(';')[0].strip()
                final_url = str(response.url) 
                source_display_name = final_url

                print(f"[Router] URL: {final_url} | Content-Type: {content_type}")
                if content_type == 'application/pdf':
                    print(f"[Router] Content identified as PDF. Downloading...")
                    with tempfile.NamedTemporaryFile(mode='wb', suffix=".pdf", delete=False) as temp_file:
                        temp_pdf_path = temp_file.name
                        downloaded_bytes = 0
                        try:
                            async for chunk in response.content.iter_chunked(8192):
                                if chunk:
                                    temp_file.write(chunk)
                                    downloaded_bytes += len(chunk)
                        except Exception as download_err:
                             print(f"[Router] Error during PDF download from {final_url}: {download_err}")
                             final_summary = f"Error: Failed during PDF download ({download_err})."
                             if temp_pdf_path and os.path.exists(temp_pdf_path):
                                 try: os.remove(temp_pdf_path)
                                 except: pass
                             temp_pdf_path = None 
                             return (source_display_name, final_summary)


                    print(f"[Router] Saved PDF ({downloaded_bytes} bytes) to temp file: {temp_pdf_path}")

                    if downloaded_bytes == 0:
                        print(f"[Router] Error: Downloaded PDF from {final_url} was empty.")
                        final_summary = "Error: Downloaded PDF content was empty."
                    elif temp_pdf_path and os.path.exists(temp_pdf_path):
                        pdf_file_object_api = await upload_pdf_to_gemini(temp_pdf_path)
                        if pdf_file_object_api:
                            final_summary = await summarize_content(model, pdf_file_object_api, final_url)
                            try:
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(None, lambda: genai.delete_file(pdf_file_object_api.name))
                                print(f"[Router] Cleaned up Gemini file: {pdf_file_object_api.name}")
                            except Exception as del_e:
                                print(f"[Router] Warning: Failed to delete Gemini file {pdf_file_object_api.name}: {del_e}")
                        else:
                            final_summary = "Error: Failed to upload or process downloaded PDF via API."
                    else:
                        final_summary = "Error: Downloaded PDF temp file is missing or invalid."

                elif 'text/html' in content_type:
                    print(f"[Router] Content identified as HTML. Reading and extracting text...")
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    for element in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button", "img", "svg", "iframe", "link", "meta"]):
                        element.decompose()
                    main_content = soup.find('main') or soup.find('article') or soup.find(id='content') or soup.find(id='main') or soup.find(class_='content') or soup.find(class_='main') or soup.find(role='main') or soup.body
                    if main_content:
                        text = ' '.join(line.strip() for line in main_content.stripped_strings)
                    else:
                        text = ' '.join(line.strip() for line in soup.stripped_strings)

                    if text and text.strip():
                        print(f"[Router] Extracted HTML text (Length: {len(text)}). Summarizing...")
                        final_summary = await summarize_content(model, text, final_url)
                    else:
                        final_summary = "Error: Could not extract meaningful text from HTML."

                else:
                    print(f"[Router] Skipping URL {final_url} due to unsupported Content-Type: {content_type}")
                    final_summary = f"Error: Unsupported Content-Type '{content_type}' for URL."

        except asyncio.TimeoutError:
            print(f"[Router] Error: Timeout accessing URL {source}")
            final_summary = f"Error: Timeout accessing URL."
        except aiohttp.ClientResponseError as e: 
             print(f"[Router] Error: HTTP error {e.status} accessing URL {source}: {e.message}")
             final_summary = f"Error: HTTP error {e.status} accessing URL."
        except aiohttp.ClientError as e: 
            print(f"[Router] Error: Client error accessing URL {source}: {e}")
            final_summary = f"Error: Client error accessing URL ({type(e).__name__})."
        except Exception as e:
            print(f"[Router] Error: Unexpected error processing URL {source}: {type(e).__name__} - {e}")
            print(f"Full traceback for URL {source}:\n{traceback.format_exc()}")
            final_summary = f"Error: Unexpected error processing URL: {type(e).__name__}"

        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    print(f"[Router] Cleaned up temporary PDF file: {temp_pdf_path}")
                except Exception as e_clean:
                    print(f"[Router] Warning: Failed to delete temporary file {temp_pdf_path}: {e_clean}")

    elif source_lower.endswith(".pdf"):
        print(f"[Router] Processing local PDF path: {source}")
        if not GOOGLE_API_KEY:
            return (source_display_name, "Error: GOOGLE_API_KEY is not configured. Cannot process local PDF.")

        if not os.path.exists(source):
             print(f"[Router] Error: Local PDF file not found: {source}")
             return (source_display_name, f"Error: Local PDF file not found.")
        if os.path.getsize(source) == 0:
            print(f"[Router] Error: Local PDF file is empty: {source}")
            return (source_display_name, f"Error: Local PDF file is empty.")

        pdf_file_object_api = await upload_pdf_to_gemini(source) 
        if pdf_file_object_api:
            final_summary = await summarize_content(model, pdf_file_object_api, source) 
            try:
                 loop = asyncio.get_running_loop()
                 await loop.run_in_executor(None, lambda: genai.delete_file(pdf_file_object_api.name))
                 print(f"[Router] Cleaned up Gemini file: {pdf_file_object_api.name}")
            except Exception as del_e:
                 print(f"[Router] Warning: Failed to delete Gemini file {pdf_file_object_api.name}: {del_e}")
        else:
            final_summary = "Error: Failed to upload or process local PDF via API."

    elif not source.strip():
         print("[Router] Skipping empty source line.")
         return None 
    else:
        print(f"[Router] Skipping unsupported source type: {source}")
        final_summary = "Error: Unsupported source type (must be URL or local .pdf path)."

    return (source_display_name, final_summary)

async def generate_review_paper(model, results):
    """
    Takes individual summaries and prompts the model to synthesize them
    into a review paper format.
    """
    print("\n" + "-"*30 + " Synthesizing Review Paper " + "-"*30)
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY is not configured. Cannot generate review paper."

    successful_summaries = []
    failed_sources = []
    source_map = {}

    for result_item in results:
         if result_item is None:
              continue

         if not isinstance(result_item, (tuple, list)) or len(result_item) != 2:
              print(f"[Synthesizer] Warning: Skipping malformed result item: {result_item}")
              failed_sources.append(f"Malformed Result ({result_item})")
              continue

         source, summary_or_error = result_item

         if summary_or_error and isinstance(summary_or_error, str) and not summary_or_error.strip().startswith("Error:"):
            max_summary_len = 15000 
            summary_text = summary_or_error.strip()
            if len(summary_text) > max_summary_len:
                 print(f"[Synthesizer] Warning: Truncating long summary for '{source}' ({len(summary_text)} chars)")
                 summary_text = summary_text[:max_summary_len] + "... [Summary Truncated]"

            successful_summaries.append({
                "source": source,
                "summary": summary_text
            })
            source_map[source] = summary_text 
         else:
            error_msg = summary_or_error if isinstance(summary_or_error, str) else "Unknown Processing Error"
            print(f"[Synthesizer] Excluding source '{source}' due to error: {error_msg}")
            failed_sources.append(f"{source} ({error_msg})")


    if not successful_summaries:
        report = f"--- Review Paper Generation Failed ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n"
        report += "No successful summaries were generated from the provided sources.\n"
        if failed_sources:
            report += f"\nFailed or skipped sources ({len(failed_sources)}):\n - " + '\n - '.join(failed_sources) + "\n"
        return report

    prompt_parts = [
        "You are an expert academic researcher tasked with writing a concise review paper based *only* on the following summaries provided.",
        "Your goal is to synthesize the information, identify key themes, common findings, and any contrasting points, presenting it in a standard academic review format.",
        "Structure the review paper clearly, potentially using sections like: Title, Abstract, Introduction, Thematic Review (use themes you identify), Discussion/Synthesis, Conclusion, and References.",
        "In the main review section(s), **synthesize** the findings. Group related information thematically. DO NOT just list the summaries one by one.",
        "When incorporating information from a specific summary, you **must** cite the source using its provided name/URL in parentheses, exactly as given, like this: (Source Name or URL). Ensure citations are accurate.",
        "The 'References' section should list all the sources (using the exact names/URLs provided) from which summaries were successfully generated and used in the review.",
        "Maintain a formal, objective, and analytical academic tone throughout. Avoid speculation beyond the provided summaries.",
        "\n--- Provided Summaries ---"
    ]

    for i, item in enumerate(successful_summaries):
        source_name_cleaned = str(item['source']).replace('\n', ' ').strip()
        prompt_parts.append(f"\n[{i+1}] Source: {source_name_cleaned}\nSummary:\n{item['summary']}\n---")

    prompt_parts.append("\n--- Review Paper ---")
    synthesis_prompt = "\n".join(prompt_parts)

    print(f"[Synthesizer] Sending {len(successful_summaries)} summaries to the model for review paper generation...")

    response = None
    review_paper_text = "Error: Failed to generate review paper." # Default error
    try:
        safety_settings = {
             'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
             'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
            }
        generation_config = genai.types.GenerationConfig(
            temperature=0.6, 
            max_output_tokens=8192 
        )
        request_options = genai.types.RequestOptions(timeout=300) 

        response = await model.generate_content_async(
            synthesis_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            request_options=request_options
            )

        if response.prompt_feedback and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason
             details = ""
             if response.prompt_feedback.safety_ratings:
                  details = ", ".join([f"{sr.category}: {sr.probability}" for sr in response.prompt_feedback.safety_ratings])
             print(f"[Synthesizer] Warning: Synthesis prompt blocked. Reason: {reason}. Details: {details}")
             review_paper_text = f"Error: Could not generate review paper (synthesis prompt blocked - {reason}). Details: {details}"
        elif not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
            safety_info = ""
            if response.candidates and response.candidates[0].safety_ratings:
                 safety_info = ", ".join([f"{sr.category}: {sr.probability}" for sr in response.candidates[0].safety_ratings])
            print(f"[Synthesizer] Warning: No valid content parts found in synthesis response. Finish Reason: {finish_reason}. Safety: {safety_info}")
            review_paper_text = f"Error: Could not generate review paper (API response missing valid content). Finish Reason: {finish_reason}. Safety: {safety_info}"
        else:
             generated_text = getattr(response, 'text', None)
             if generated_text is None:
                 try: 
                      generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                 except Exception: pass

             if generated_text and generated_text.strip():
                 review_paper_text = generated_text.strip()
                 print("[Synthesizer] Successfully generated review paper.")
             else:
                 print(f"[Synthesizer] Warning: No text found in synthesis response parts.")
                 try: print(f"[Synthesizer] Raw synthesis response parts: {response.candidates[0].content.parts}")
                 except: pass
                 review_paper_text = "Error: Could not generate review paper (no text in response)."

    except Exception as e:
        print(f"[Synthesizer] Error during review paper synthesis: {type(e).__name__} - {e}")
        feedback = getattr(response, 'prompt_feedback', None) or getattr(response, 'candidates[0].finish_reason', None)
        if feedback: print(f"[Synthesizer] Feedback/Reason: {feedback}")
        print(f"[Synthesizer] Full synthesis error traceback:\n{traceback.format_exc()}")
        review_paper_text = f"Error generating review paper during synthesis: {type(e).__name__}"

    final_output = f"--- Generated Review ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n"
    final_output += review_paper_text
    final_output += f"\n\n" + "="*40 + "\n" 
    print("[Synthesizer] Note: Uploaded temporary PDF files on the Gemini API are automatically deleted after summarization attempts.")

    return final_output

async def main_process(sources):
    """Main async function to orchestrate the concurrent processing and synthesis."""
    start_time = time.time()
    print(f"\n--- Starting Document Processing and Review Generation ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
    print(f"Processing {len(sources)} sources...")

    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY is not configured. Cannot proceed."

    try:
        system_instruction = "You are a helpful assistant skilled in summarizing academic and web documents accurately and synthesizing information into structured reviews."
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=system_instruction)
    except Exception as model_init_err:
        print(f"Error initializing Generative Model: {model_init_err}")
        return f"Error: Failed to initialize the AI model. Please check API key and configuration. Details: {model_init_err}"

    tasks = []
    connector = aiohttp.TCPConnector(limit=10) 
    async with aiohttp.ClientSession(connector=connector) as session:
        for source in sources:
            if source.strip(): 
                task = asyncio.create_task(process_source(session, model, source.strip()))
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    original_sources_map = {i: sources[i].strip() for i in range(len(sources)) if sources[i].strip()} 
    processed_indices = set()

    for i, result in enumerate(results):
        original_source = "Unknown Source (Order Mismatch?)"
        current_task_index = -1
        original_list_idx = -1
        for idx, src in enumerate(sources):
             if src.strip():
                  current_task_index += 1
                  if current_task_index == i:
                       original_source = src.strip()
                       original_list_idx = idx
                       processed_indices.add(idx)
                       break

        if isinstance(result, Exception):
            print(f"[Main] Critical Error processing source '{original_source}': {result}")
            processed_results.append((original_source, f"Error: Task failed unexpectedly: {result}"))
            print(f"[Main] Source '{original_source}' was skipped (empty line or explicit skip).")
        elif isinstance(result, (tuple, list)) and len(result) == 2:
             processed_results.append(result)
        else:
            print(f"[Main] Warning: Unexpected result format for source '{original_source}': {result}")
            processed_results.append((original_source, "Error: Internal processing error (unexpected result format)."))

    final_report = await generate_review_paper(model, processed_results)

    print("\n" + "="*60 + "\n")
    print("="*60)
    end_time = time.time()
    print(f"--- Document Processing and Review Generation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

    return final_report 

async def run_analysis(sources_text, uploaded_files): 
    """
    Wrapper function called by the Gradio button click.
    It parses URLs from the text input, gets paths from uploaded files,
    calls the main async processing logic, and returns the final report string.
    """
    start_run_time = time.time()
    print("\n[Gradio] Received request...")

    if not GOOGLE_API_KEY:
         print("CRITICAL ERROR: Backend GOOGLE_API_KEY is not configured.")
         return "Configuration Error: The application's API key is missing. Please contact the administrator."

    sources_list = []

    if sources_text and sources_text.strip():
        print("[Gradio] Processing sources from text input...")
        text_sources = [line.strip() for line in sources_text.splitlines() if line.strip()]
        sources_list.extend(text_sources)
        print(f"[Gradio] Added {len(text_sources)} sources from text.")

    if uploaded_files:
        print(f"[Gradio] Processing {len(uploaded_files)} uploaded files...")
        for uploaded_file in uploaded_files:
            temp_pdf_path = uploaded_file.name
            print(f"[Gradio] Added uploaded file: {os.path.basename(temp_pdf_path)} (Temp Path: {temp_pdf_path})")
            sources_list.append(temp_pdf_path)
    if not sources_list:
        return "Error: Please enter at least one URL in the textbox OR upload at least one PDF file."

    print(f"[Gradio] Combined sources to process ({len(sources_list)} total):")
    for i, src in enumerate(sources_list):
        display_src = os.path.basename(src) if src.lower().endswith(".pdf") and os.path.exists(src) else src
        print(f"  {i+1}: {display_src}") 
    try:
        result_report = await main_process(sources_list)
        total_time = time.time() - start_run_time
        print(f"[Gradio] Request completed in {total_time:.2f} seconds.")
        return result_report
    except Exception as e:
        print(f"[Gradio] An unexpected error occurred during the main process: {type(e).__name__} - {e}")
        traceback.print_exc() 
        return f"An unexpected error occurred: {type(e).__name__} - {e}. Please check the logs or try again."

input_textbox = gr.Textbox(
    lines=5, 
    label="Sources (Enter URLs here, one per line)",
    placeholder="Example URLs:\nhttps://www.nature.com/articles/s41586-023-06084-3\nhttps://arxiv.org/abs/2303.08774"
)

input_files_component = gr.Files(
    label="Upload PDF Files (Optional)",
    file_types=[".pdf"],
    type="filepath" 
)

output_textbox = gr.Textbox(
    label="Generated Output",
    lines=25,
    interactive=False 
)

iface = gr.Interface(
    fn=run_analysis,
    inputs=[input_textbox, input_files_component],
    outputs=output_textbox,
    title="📄 Document Summarizer 📄",
    description="""Enter URLs in the text box (one per line) **AND/OR** upload PDF files below.
The tool will fetch content from URLs, use uploaded PDFs, summarize each source using Google Gemini,
and then synthesize the summaries into a structured review paper.
**Uploading is the recommended way to process local PDF files.**
Processing can take several minutes depending on the number and size of sources. Check terminal for progress logs.
""",
    allow_flagging='never',
    theme=gr.themes.Soft() 
)
if __name__ == "__main__":
    print("Launching Gradio Interface...")
    iface.launch(show_error=True)