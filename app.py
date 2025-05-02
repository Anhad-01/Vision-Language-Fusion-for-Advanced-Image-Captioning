import streamlit as st
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
from image_captioner import ImageCaptioner
from advanced_captioner import AdvancedCaptioner
from coca_captioner import CocaCaptioner
import pandas as pd
import nltk

# Initialize NLTK resources at startup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

def main():
    st.set_page_config(
        page_title="Advanced Image Captioning",
        page_icon="🖼️",
        layout="wide"
    )
    
    # App title and description
    st.title("🖼️ Advanced Image Captioning")
    st.markdown("""
    This application generates descriptive captions for your images using state-of-the-art
    vision-language models with advanced NLP post-processing.
    """)
    
    # HuggingFace authentication setup
    with st.sidebar.expander("HuggingFace Authentication", expanded=False):
        st.info("Models may require authentication for downloading from HuggingFace.")
        use_auth = st.checkbox("Use HuggingFace Authentication", value=False)
        hf_token = st.text_input("HuggingFace Token (optional)", type="password", 
                                help="Get your token from huggingface.co/settings/tokens")
    
    # Sidebar options
    st.sidebar.title("Options")
    
    # Model selection with improved options
    model_type = st.sidebar.radio(
        "Captioning Model",
        ["Standard (ViT-GPT2)", "Advanced (BLIP)", "State-of-the-Art (BLIP-2)", "Enhanced BLIP"],
        help="Standard model is faster, Advanced BLIP produces better captions, BLIP-2 offers high quality but requires more resources, Enhanced BLIP uses additional post-processing techniques"
    )
    
    # Caption method selection
    caption_method = st.sidebar.radio(
        "Caption Generation Method",
        ["Single Caption", "Multiple Captions"]
    )
    
    if caption_method == "Multiple Captions":
        num_captions = st.sidebar.slider("Number of Captions", 2, 5, 3)
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        if model_type in ["Advanced (BLIP)", "State-of-the-Art (BLIP-2)"]:
            beam_search = st.checkbox("Use Beam Search", value=True, 
                        help="Beam search is more deterministic, unchecking enables sampling for more creative captions")
            
            # Custom text prompt for BLIP models
            use_custom_prompt = st.checkbox("Use Custom Prompt", value=False,
                        help="Customize the text prompt used to generate captions")
            
            if use_custom_prompt:
                custom_prompt = st.text_input("Custom Prompt", value="Describe this image in detail")
            else:
                custom_prompt = "Describe this image in detail"
                
        show_analysis = st.checkbox("Show NLP Analysis", value=False)
    
    # Initialize the selected image captioner
    @st.cache_resource
    def load_standard_captioner(use_auth=False, token=None):
        with st.spinner("Loading standard image captioning model (ViT-GPT2)..."):
            return ImageCaptioner(use_auth=use_auth, token=token)
    
    @st.cache_resource
    def load_advanced_captioner(use_auth=False, token=None):
        with st.spinner("Loading advanced image captioning model (BLIP)..."):
            return AdvancedCaptioner(use_auth=use_auth, token=token)
    
    @st.cache_resource
    def load_blip2_captioner(use_auth=False, token=None):
        with st.spinner("Loading BLIP-2 image captioning model (state-of-the-art)..."):
            return AdvancedCaptioner(use_auth=use_auth, token=token, use_blip2=True)
    
    @st.cache_resource
    def load_coca_captioner(use_auth=False, token=None):
        with st.spinner("Loading CoCa image captioning model (contrastive captioner)..."):
            return CocaCaptioner(use_auth=use_auth, token=token)
    
    # Load appropriate captioner based on selection with authentication options
    try:
        if model_type == "Standard (ViT-GPT2)":
            captioner = load_standard_captioner(use_auth=use_auth, token=hf_token if use_auth else None)
        elif model_type == "Advanced (BLIP)":
            captioner = load_advanced_captioner(use_auth=use_auth, token=hf_token if use_auth else None)
        elif model_type == "State-of-the-Art (BLIP-2)":
            captioner = load_blip2_captioner(use_auth=use_auth, token=hf_token if use_auth else None)
        else:  # Enhanced BLIP
            captioner = load_coca_captioner(use_auth=use_auth, token=hf_token if use_auth else None)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Try enabling HuggingFace authentication in the sidebar and providing a valid token.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Alternative: URL input
    image_url = st.text_input("Or enter an image URL:", "")
    
    image_path = None
    
    # Process uploaded file or URL
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, width=400)
        
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)
            image_path = temp_path
    
    elif image_url:
        try:
            import requests
            from io import BytesIO
            
            # Download image from URL
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Image from URL")
                st.image(image, width=400)
            
            # Save the image to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_path = temp_file.name
                image.save(temp_path)
                image_path = temp_path
        except Exception as e:
            st.error(f"Error loading image from URL: {str(e)}")
    
    # Generate caption if an image is provided
    if image_path:
        # Generate caption(s)
        with st.spinner(f"Generating caption with {model_type}..."):
            start_time = time.time()
            
            if caption_method == "Single Caption":
                if model_type == "Standard (ViT-GPT2)":
                    result = captioner.generate_caption(image_path)
                elif model_type in ["Advanced (BLIP)", "State-of-the-Art (BLIP-2)"]:
                    # For BLIP and BLIP-2 models
                    result = captioner.generate_caption(
                        image_path, 
                        use_beam_search=beam_search,
                        text_prompt=custom_prompt if 'use_custom_prompt' in locals() and use_custom_prompt else "Describe this image in detail"
                    )
                else:  # Enhanced BLIP
                    result = captioner.generate_caption(image_path)
                
                with col2:
                    st.subheader("Generated Caption")
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.markdown("**Original:** " + result["original_caption"])
                        st.markdown("**Enhanced:** " + result["enhanced_caption"])
                        
                        # Show NLP analysis for supported models if requested
                        if model_type in ["Advanced (BLIP)", "State-of-the-Art (BLIP-2)", "Enhanced BLIP"] and show_analysis and "nlp_analysis" in result:
                            with st.expander("NLP Analysis"):
                                analysis = result["nlp_analysis"]
                                st.markdown(f"**Total Words:** {analysis['total_word_count']}")
                                if 'content_word_count' in analysis:
                                    st.markdown(f"**Content Words:** {analysis['content_word_count']}")
                                
                                # Only show these if they exist in the analysis
                                if 'adjectives' in analysis:
                                    st.markdown("**Adjectives:**")
                                    st.write(", ".join(analysis["adjectives"]) if analysis["adjectives"] else "None")
                                
                                if 'noun_phrases' in analysis:
                                    st.markdown("**Noun Phrases:**")
                                    for phrase in analysis["noun_phrases"]:
                                        st.markdown(f"- {phrase}")
                                
                                if 'pos_tags' in analysis:
                                    st.markdown("**Part-of-Speech Tags:**")
                                    pos_df = pd.DataFrame(analysis["pos_tags"], columns=["Word", "POS Tag"])
                                    st.dataframe(pos_df)
            else:
                # Handle multiple captions
                if model_type == "Standard (ViT-GPT2)":
                    result = captioner.generate_multiple_captions(image_path, num_captions)
                elif model_type in ["Advanced (BLIP)", "State-of-the-Art (BLIP-2)"]:
                    # For BLIP and BLIP-2 models with custom prompts
                    if 'use_custom_prompt' in locals() and use_custom_prompt:
                        # Create variations of the custom prompt for diversity
                        text_prompts = [
                            custom_prompt,
                            f"Detailed description: {custom_prompt}",
                            f"{custom_prompt} comprehensively"
                        ]
                        # Add more variations if needed
                        if num_captions > 3:
                            additional = [custom_prompt] * (num_captions - 3)
                            text_prompts.extend(additional)
                            
                        result = captioner.generate_multiple_captions(image_path, num_captions, text_prompts)
                    else:
                        result = captioner.generate_multiple_captions(image_path, num_captions)
                else:  # Enhanced BLIP
                    result = captioner.generate_multiple_captions(image_path, num_captions)
                
                with col2:
                    st.subheader("Generated Captions")
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        for i, (orig, enhanced) in enumerate(zip(result["original_captions"], 
                                                             result["enhanced_captions"])):
                            with st.expander(f"Caption {i+1}", expanded=True):
                                st.markdown("**Original:** " + orig)
                                st.markdown("**Enhanced:** " + enhanced)
            
            elapsed_time = time.time() - start_time
            st.caption(f"Caption generated in {elapsed_time:.2f} seconds")
        
        # Clean up the temporary file
        os.unlink(image_path)
    
    # Add explanation of the technology
    with st.expander("How does it work?"):
        st.markdown("""
        ### Technical Details
        
        This application offers four different models:
        
        #### Standard Model (ViT-GPT2)
        - **Vision Transformer (ViT)**: Processes the image and extracts visual features
        - **GPT-2**: Generates natural language captions based on image features
        - Faster but may produce simpler captions
        
        #### Advanced Model (BLIP)
        - **BLIP** (Bootstrapping Language-Image Pre-training): A powerful model that better aligns vision and language
        - Provides more detailed and accurate captions
        - Includes advanced NLP analysis capabilities
        
        #### State-of-the-Art Model (BLIP-2)
        - **BLIP-2**: A next-generation model with significantly improved performance
        - Uses a frozen image encoder and LLM with a lightweight Querying Transformer
        - Produces highly detailed and accurate captions
        - Supports custom text prompts for more targeted descriptions
        
        #### Enhanced BLIP
        - An optimized version of the BLIP model with additional post-processing
        - Uses specialized linguistic analysis to improve caption quality
        - Focuses on accurate and grammatically correct descriptions
        - Particularly good at identifying important details in images
        
        All models use NLP post-processing to enhance the captions with:
        - Tokenization
        - Stopword handling
        - Grammar correction
        - Sentence structure improvements
        """)
    
    # Adding troubleshooting section
    with st.expander("Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        #### Model Download Errors
        If you see errors related to downloading models from HuggingFace:
        1. Try enabling HuggingFace Authentication in the sidebar
        2. Create a free account at [huggingface.co](https://huggingface.co)
        3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
        4. Enter your token in the sidebar field
        
        #### Performance Issues
        - The advanced BLIP model is larger and may take longer to load
        - The BLIP-2 and Enhanced BLIP models are the largest and may require a GPU for optimal performance
        - If you're on a machine with limited memory, try using the Standard model
        
        #### CUDA Out of Memory Error
        If you get a CUDA out of memory error with BLIP-2 or Enhanced BLIP:
        1. Try closing other applications that use GPU memory
        2. Switch to the BLIP or ViT-GPT2 model instead
        """)

if __name__ == "__main__":
    main() 