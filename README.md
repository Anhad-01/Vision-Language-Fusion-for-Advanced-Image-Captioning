# Advanced Image Captioning Application

This application generates high-quality, descriptive captions for images using state-of-the-art vision-language models with advanced NLP post-processing.

![image](https://github.com/user-attachments/assets/7009722c-67e9-4890-9a49-34bfd0169e4f)
![image](https://github.com/user-attachments/assets/9d38093c-917e-486e-89d5-e5daea53c706)
![image](https://github.com/user-attachments/assets/96c939a9-5324-458a-9458-0576ccca5566)

## Features

- **Multiple Model Options**:
  - **Standard (ViT-GPT2)**: Fast and efficient image captioning
  - **Advanced (BLIP)**: More detailed and accurate captions
  - **State-of-the-Art (BLIP-2)**: Highest quality captions using the latest technology
  - **Enhanced BLIP**: Optimized BLIP model with improved post-processing

- **Customization Options**:
  - Generate single or multiple captions for each image
  - Use beam search or sampling for more creative captions
  - Apply custom text prompts to guide caption generation

- **NLP Analysis**:
  - Part-of-speech tagging
  - Noun phrase extraction
  - Content word analysis
  - Adjective identification

- **User-Friendly Interface**:
  - Upload images or use image URLs
  - Side-by-side display of image and captions
  - Expandable sections for advanced analysis

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended for BLIP-2 model
- Packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

Or use the provided batch file on Windows:

```bash
run_app.bat
```

## Model Information

### Standard Model (ViT-GPT2)
The standard model uses a Vision Transformer (ViT) to process the image and extract visual features, and a GPT-2 language model to generate captions based on these features. It's faster and requires less memory but may produce simpler captions.

### Advanced Model (BLIP)
BLIP (Bootstrapping Language-Image Pre-training) is a more powerful model that better aligns vision and language through a contrastive approach. It provides more detailed and accurate captions and supports advanced NLP analysis.

### State-of-the-Art Model (BLIP-2)
BLIP-2 is the latest generation model that significantly improves on BLIP's performance. It uses a frozen image encoder and language model with a lightweight Querying Transformer in between. This model produces the highest quality captions and supports custom text prompts for more targeted descriptions.

### Enhanced BLIP
Enhanced BLIP uses the BLIP architecture with additional specialized linguistic analysis and post-processing. This model is optimized for grammatical correctness and focuses on producing clear, accurate descriptions of images. It's particularly good at identifying important details and producing well-structured captions.

## Advanced Usage

### Custom Prompts
When using BLIP or BLIP-2 models, you can provide custom text prompts to guide the caption generation. For example:
- "Describe this image in detail"
- "What objects are in this image?"
- "Explain what's happening in this scene"

### Multiple Captions
Generate multiple diverse captions for the same image to get different perspectives or descriptions.

### NLP Analysis
View detailed NLP analysis of generated captions, including:
- Part-of-speech breakdown
- Identification of key noun phrases
- Count of content words vs. function words
- List of descriptive adjectives used

## Troubleshooting

### Model Download Issues
If you encounter issues downloading models from HuggingFace:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Enable "Use HuggingFace Authentication" in the sidebar and provide your token

### Performance Issues
- The BLIP-2 model requires significant memory and may be slow on CPU-only systems
- For faster performance, use a CUDA-capable GPU
- If you're experiencing out-of-memory errors, try the Standard or Advanced models instead

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This application uses models from the Hugging Face Transformers library and relies on research from:
- ViT-GPT2: [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
- BLIP: [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)
- BLIP-2: [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) 
