# Agentic-Rag-Summarizer
## A Document Summarizer and Review Paper Generator

A powerful tool that uses Google's Gemini AI to summarize documents from URLs and PDFs, then synthesizes them into a structured academic review paper.

## Features

- **Multi-source Processing**: Process multiple sources simultaneously (URLs and PDFs)
- **Intelligent Content Extraction**: Automatically extracts meaningful content from web pages and PDFs
- **AI-Powered Summarization**: Uses Google's Gemini AI model to generate concise summaries
- **Review Paper Synthesis**: Combines multiple summaries into a structured academic review paper
- **User-Friendly Interface**: Simple Gradio web interface for easy interaction

## Requirements

- Python 3.7+
- Google API Key for Gemini AI
- Internet connection for processing web content

## Installation

1. Clone this repository:
   ```
   git clone [https://github.com/yourusername/document-summarizer.git](https://github.com/PeakScripter/Agentic-Rag-Summarizer)
   cd document-summarizer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   - Create a `.env` file in the project root
   - Add your API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

1. Run the application:
   ```
   python mark2.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://127.0.0.1:7860)

3. In the web interface:
   - Enter URLs in the text box (one per line)
   - Upload PDF files (optional)
   - Click the submit button to start processing

4. Wait for the processing to complete (this may take several minutes depending on the number and size of sources)

5. The generated review paper will appear in the output text box

## How It Works

1. **Source Processing**: The tool concurrently processes each source (URL or PDF)
2. **Content Extraction**: For URLs, it extracts text content; for PDFs, it uploads them to Gemini
3. **Summarization**: Each source is summarized using the Gemini AI model
4. **Synthesis**: The summaries are combined into a structured academic review paper
5. **Output**: The final review paper is displayed in the interface

## Limitations

- Processing time depends on the number and size of sources
- Some websites may block automated content extraction
- PDF processing requires a valid Google API key

## License

[MIT License](LICENSE)

## Acknowledgements

- [Google Gemini AI](https://ai.google.dev/) for providing the AI model
- [Gradio](https://www.gradio.app/) for the web interface
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) for HTML parsing 
