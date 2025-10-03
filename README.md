# Acolad Style Guide Cleaner 🚀

A powerful Streamlit application that cleans and optimizes DOCX style guide files for LLM translation. Features AI-powered content optimization using OpenAI, advanced cleaning algorithms, and comprehensive document analysis.

## 🆕 Key Features

- **🤖 AI-Powered Optimization**: Uses OpenAI to rewrite content for better conciseness and clarity
- **🧠 Smart Analysis**: Intelligent complexity scoring and cleaning opportunity detection
- **📊 Enhanced Analytics**: Detailed document analysis with exportable reports
- **⚡ Advanced Cleaning**: More sophisticated algorithms based on real-world style guide analysis
- **🎯 Context-Aware**: Considers document structure and style when optimizing content

## Features

- **DOCX Processing**: Upload and parse Microsoft Word documents
- **Content Cleaning**: Remove formatting artifacts, page numbers, and metadata
- **LLM Optimization**: Clean text specifically for language model consumption
- **Multiple Output Formats**: Export as Markdown or plain text
- **Document Analysis**: Analyze document complexity and style distribution
- **Interactive Interface**: Easy-to-use web interface with preview functionality

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install pipenv
pipenv install
```

3. (Optional) Set up OpenAI integration:
   - Copy `env_example.txt` to `.env`
   - Add your OpenAI API key to the `.env` file
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## Usage

### Local Development

```bash
streamlit run app.py
# or
python run.py
```

2. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)

## 🚀 Deployment

### Streamlit Community Cloud

1. **Fork this repository** on GitHub
2. **Go to [Streamlit Community Cloud](https://share.streamlit.io/)**
3. **Click "New app"** and connect your GitHub repository
4. **Set up secrets** in the Streamlit dashboard:
   - Go to your app settings
   - Navigate to "Secrets"
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "your_api_key_here"
     ```
5. **Deploy!** Your app will be available at `https://your-app-name.streamlit.app/`

### Local Deployment

```bash
# Install dependencies
pip install pipenv
pipenv install

# Set up environment (optional)
cp env_example.txt .env
# Edit .env with your OpenAI API key

# Run the application
streamlit run app.py
```

3. Upload a DOCX file using the file uploader

4. Configure cleaning options in the sidebar:

   - **OpenAI API Key**: Enter your API key for AI optimization (Pro version)
   - **Output Format**: Choose between Markdown or Plain Text
   - **Remove Tables**: Option to exclude tables from output
   - **Aggressive Cleaning**: Apply more thorough cleaning for heavily formatted documents
   - **AI Content Optimization**: Use OpenAI to rewrite content for better conciseness (Pro version)

5. Review the cleaned content in the preview section

6. Download the cleaned file

## What Gets Cleaned

The application removes or cleans the following elements to optimize content for LLMs:

### Text Cleaning

- Excessive whitespace and formatting artifacts
- Page numbers and cross-references (e.g., "see page 5")
- Version numbers and dates
- Document metadata references
- Inconsistent bullet points and numbering
- Excessive punctuation

### Structure Optimization

- Converts headings to proper markdown hierarchy
- Standardizes bullet points to markdown format
- Preserves table structure when tables are included
- Maintains logical content flow

## Output Formats

### Markdown

- Structured with proper heading hierarchy
- Tables converted to markdown format
- Clean, readable formatting
- Optimal for LLM processing

### Plain Text

- Simple text format
- Preserves paragraph structure
- Removes all formatting artifacts
- Lightweight output

## Document Analysis

The application provides insights about your document:

- **Paragraph Count**: Total number of paragraphs
- **Table Count**: Number of tables in the document
- **Complexity Score**: Low, Medium, or High based on content volume
- **Style Distribution**: Chart showing paragraph style usage
- **Text Length**: Average paragraph length and total content

## Best Practices

### Input Documents

- Use well-structured DOCX files with clear headings
- Avoid overly complex formatting that isn't essential
- Ensure consistent paragraph styles throughout the document

### Cleaning Options

- Use **Aggressive Cleaning** for heavily formatted documents
- **Remove Tables** if tables contain formatting artifacts rather than useful data
- Choose **Markdown** output for better LLM readability

### Review Process

- Always review the preview before downloading
- Check that important content hasn't been removed
- Adjust cleaning options if needed

## Technical Details

### Dependencies

- `streamlit`: Web application framework
- `python-docx`: DOCX file processing
- `markdown`: Markdown conversion
- `Pillow`: Image processing support
- `pandas`: Data analysis and visualization

### File Processing

- Extracts paragraphs, tables, headers, and footers
- Preserves document structure and hierarchy
- Handles various paragraph styles and formatting
- Processes tables with proper markdown conversion

## Troubleshooting

### Common Issues

**File Upload Errors**

- Ensure the file is a valid DOCX format
- Check file size (large files may take longer to process)
- Verify the document isn't corrupted

**Content Issues**

- If too much content is removed, try disabling "Aggressive Cleaning"
- For tables with important data, ensure "Remove Tables" is unchecked
- Review the preview to verify cleaning results

**Performance**

- Large documents may take several seconds to process
- Complex tables with many cells may increase processing time

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.

## License

This project is open source and available under the MIT License.
