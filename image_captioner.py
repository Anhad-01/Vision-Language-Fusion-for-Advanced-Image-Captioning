import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import os
from huggingface_hub import login, HfFolder

class ImageCaptioner:
    def __init__(self, use_auth=False, token=None, model_path="nlpconnect/vit-gpt2-image-captioning"):
        # Download basic NLTK resources only
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        
        # Handle HuggingFace authentication
        if use_auth and token:
            # Set up HuggingFace credentials
            login(token=token)
        
        # Load pre-trained models with offline mode option if no auth provided
        try:
            # Use base model - the large version doesn't exist on HuggingFace
            model_path = "nlpconnect/vit-gpt2-image-captioning"
            
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
            self.feature_extractor = ViTImageProcessor.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except OSError as e:
            print(f"Error loading model: {e}")
            print("Attempting to use local models or download without auth...")
            # Try with local files only option
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=False)
            self.feature_extractor = ViTImageProcessor.from_pretrained(model_path, local_files_only=False)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Set improved generation parameters for better captions
        self.max_length = 32  # Increased from 16 for more detailed captions
        self.num_beams = 5    # Increased from 4 for better beam search
        self.gen_kwargs = {
            "max_length": self.max_length, 
            "num_beams": self.num_beams,
            "no_repeat_ngram_size": 2,  # Avoid repetition of phrases
            "length_penalty": 1.0       # Favor slightly longer captions that are more descriptive
        }

    def preprocess_image(self, image_path):
        """Preprocess the image for the model"""
        image = Image.open(image_path).convert('RGB')
        
        # Apply image normalization and preprocessing
        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        return pixel_values, image
    
    def generate_caption(self, image_path):
        """Generate a caption for the image"""
        try:
            pixel_values, original_image = self.preprocess_image(image_path)
            
            # Generate caption
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
            
            # Decode the predicted tokens
            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            caption = preds[0].strip()
            
            # Apply basic enhancements
            enhanced_caption = self.enhance_caption(caption)
            
            return {
                "original_caption": caption,
                "enhanced_caption": enhanced_caption
            }
        except Exception as e:
            return {"error": str(e)}
    
    def enhance_caption(self, caption):
        """Apply basic enhancements to the caption"""
        # Tokenize the caption
        tokens = word_tokenize(caption.lower())
        
        # Improved processing logic - keep structure but remove obvious filler words
        filtered_tokens = []
        for i, word in enumerate(tokens):
            # Keep articles and prepositions for natural language flow
            if word.isalnum() or word in [".", ",", "!"]:
                filtered_tokens.append(word)
        
        # Capitalize first letter and ensure proper punctuation
        if filtered_tokens:
            if filtered_tokens[0].isalnum():
                filtered_tokens[0] = filtered_tokens[0].capitalize()
        
        enhanced_caption = " ".join(filtered_tokens)
        
        # Ensure proper punctuation at the end
        if enhanced_caption and not enhanced_caption.endswith(('.', '!', '?')):
            enhanced_caption += '.'
            
        return enhanced_caption
    
    def generate_multiple_captions(self, image_path, num_captions=3):
        """Generate multiple diverse captions for the image"""
        try:
            pixel_values, original_image = self.preprocess_image(image_path)
            
            # Generate multiple captions with improved diverse beam search
            output_ids = self.model.generate(
                pixel_values, 
                num_beams=self.num_beams * 2,
                num_return_sequences=num_captions,
                num_beam_groups=num_captions,
                diversity_penalty=0.7,  # Increased from 0.5 for more diverse outputs
                **self.gen_kwargs
            )
            
            # Decode the predicted tokens
            captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Apply basic enhancements to each caption
            enhanced_captions = [self.enhance_caption(caption.strip()) for caption in captions]
            
            return {
                "original_captions": captions,
                "enhanced_captions": enhanced_captions
            }
        except Exception as e:
            return {"error": str(e)} 