import torch
from transformers import CLIPVisionModel, AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from huggingface_hub import login, HfFolder

class CocaCaptioner:
    def __init__(self, use_auth=False, token=None, model_path="Salesforce/blip-image-captioning-large"):
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
        
        # Load BLIP model (fallback since CoCa has compatibility issues)
        try:
            print(f"Using BLIP model for CoCa captioner due to compatibility issues with the original CoCa model")
            self.processor = BlipProcessor.from_pretrained(model_path)
            self.model = BlipForConditionalGeneration.from_pretrained(model_path)
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        except OSError as e:
            print(f"Error loading model: {e}")
            print("Attempting to use local models or download without auth...")
            # Try with local files only option
            self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=False)
            self.model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=False)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def generate_caption(self, image_path):
        """Generate a caption for the image"""
        try:
            # Load and process the image
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            
            # Generate caption with beam search
            output = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                num_return_sequences=1,
                temperature=1.0
            )
            
            # Decode the generated caption
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Apply enhancement and simple analysis
            enhanced_caption = self.enhance_caption(caption)
            nlp_analysis = self.analyze_caption(caption)
            
            return {
                "original_caption": caption,
                "enhanced_caption": enhanced_caption,
                "nlp_analysis": nlp_analysis
            }
        except Exception as e:
            return {"error": str(e)}
    
    def enhance_caption(self, caption):
        """Apply basic enhancements to the caption"""
        if not caption:
            return ""
        
        # Ensure first letter is capitalized and has proper punctuation
        enhanced = caption[0].upper() + caption[1:]
        
        # Ensure proper punctuation at the end
        if not enhanced.endswith(('.', '!', '?')):
            enhanced += '.'
            
        # Fix common grammar issues
        enhanced = self._fix_common_grammar_issues(enhanced)
            
        return enhanced
    
    def _fix_common_grammar_issues(self, text):
        """Fix common grammar issues in generated captions"""
        # Replace double spaces
        text = text.replace("  ", " ")
        
        # Fix common article issues
        text = text.replace(" a orange", " an orange")
        text = text.replace(" a apple", " an apple")
        
        # Fix repetitions
        repeats = [" the the ", " a a ", " in in ", " on on ", " is is "]
        for repeat in repeats:
            text = text.replace(repeat, repeat.strip() + " ")
            
        return text
    
    def analyze_caption(self, caption):
        """Perform simple text analysis on the caption"""
        tokens = word_tokenize(caption)
        
        # Count content words (excluding stopwords)
        content_words = [word for word in tokens if word.lower() not in self.stop_words and word.isalnum()]
        
        return {
            "content_word_count": len(content_words),
            "total_word_count": len(tokens)
        }
    
    def generate_multiple_captions(self, image_path, num_captions=3):
        """Generate multiple diverse captions for the image"""
        try:
            # Load and process the image
            raw_image = Image.open(image_path).convert('RGB')
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
            
            # Generate multiple captions
            outputs = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                num_return_sequences=num_captions,
                temperature=1.0
            )
            
            # Decode captions
            captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
            
            # Apply enhancements to each caption
            enhanced_captions = [self.enhance_caption(caption) for caption in captions]
            
            return {
                "original_captions": captions,
                "enhanced_captions": enhanced_captions
            }
        except Exception as e:
            return {"error": str(e)} 