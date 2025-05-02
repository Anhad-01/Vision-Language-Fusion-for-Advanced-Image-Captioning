import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from huggingface_hub import login, HfFolder

class AdvancedCaptioner:
    def __init__(self, use_auth=False, token=None, model_path="Salesforce/blip-image-captioning-large", use_blip2=False):
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
        
        # Flag for tracking which model version is used
        self.using_blip2 = use_blip2
        
        # Load the selected model
        try:
            if use_blip2:
                # Use more advanced BLIP-2 model
                blip2_model_path = "Salesforce/blip2-opt-2.7b"
                self.processor = Blip2Processor.from_pretrained(blip2_model_path)
                self.model = Blip2ForConditionalGeneration.from_pretrained(blip2_model_path, torch_dtype=torch.float16)
            else:
                # Use standard BLIP model
                self.processor = BlipProcessor.from_pretrained(model_path)
                self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        except OSError as e:
            print(f"Error loading model: {e}")
            print("Attempting to use local models or download without auth...")
            # Try with local files only option
            if use_blip2:
                blip2_model_path = "Salesforce/blip2-opt-2.7b"
                self.processor = Blip2Processor.from_pretrained(blip2_model_path, local_files_only=False)
                self.model = Blip2ForConditionalGeneration.from_pretrained(blip2_model_path, local_files_only=False, torch_dtype=torch.float16)
            else:
                self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=False)
                self.model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=False)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_caption(self, image_path, use_beam_search=True, text_prompt="Describe this image in detail"):
        """Generate a caption for the image using BLIP or BLIP-2"""
        try:
            # Load and process the image
            raw_image = Image.open(image_path).convert('RGB')
            
            if self.using_blip2:
                # BLIP-2 specific processing
                inputs = self.processor(raw_image, text_prompt, return_tensors="pt").to(self.device, torch.float16)
                
                # Generate caption with BLIP-2
                if use_beam_search:
                    output = self.model.generate(
                        **inputs,
                        max_length=75,
                        num_beams=5,
                        num_return_sequences=1,
                        temperature=1.0
                    )
                else:
                    output = self.model.generate(
                        **inputs,
                        max_length=75,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                        temperature=0.7
                    )
                caption = self.processor.decode(output[0], skip_special_tokens=True)
            else:
                # Standard BLIP processing
                inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
                
                # Generate caption with BLIP
                if use_beam_search:
                    output = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=5,
                        num_return_sequences=1,
                        temperature=1.0
                    )
                else:
                    output = self.model.generate(
                        **inputs,
                        max_length=50,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                        temperature=0.7
                    )
                caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Apply enhancement
            enhanced_caption = self.enhance_caption(caption)
            # Simplified text analysis without POS tagging
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
        # Ensure first letter is capitalized and has proper punctuation
        if not caption:
            return ""
        
        # Fix common issues like missing articles
        enhanced = caption[0].upper() + caption[1:]
        
        # Ensure proper sentence structure
        if not enhanced.endswith(('.', '!', '?')):
            enhanced += '.'
        
        # Add post-processing to improve readability
        enhanced = self._fix_common_grammar_issues(enhanced)
            
        return enhanced
    
    def _fix_common_grammar_issues(self, text):
        """Fix common grammar issues in generated captions"""
        # Replace double spaces
        text = text.replace("  ", " ")
        
        # Fix common article issues
        text = text.replace(" a orange", " an orange")
        text = text.replace(" a apple", " an apple")
        
        # Fix repetitions (common in some models)
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
    
    def generate_multiple_captions(self, image_path, num_captions=3, text_prompts=None):
        """Generate multiple diverse captions for the image"""
        try:
            # Load and process the image
            raw_image = Image.open(image_path).convert('RGB')
            
            # Default prompts for diverse captions
            if text_prompts is None:
                text_prompts = [
                    "Describe this image in detail",
                    "What is shown in this image?",
                    "Write a detailed caption for this image"
                ]
                
                # Handle case where user requests more captions than default prompts
                if num_captions > len(text_prompts):
                    # Add generic prompts for additional captions
                    additional_prompts = ["Describe this image"] * (num_captions - len(text_prompts))
                    text_prompts.extend(additional_prompts)
            
            # Generate captions with different prompts for more diversity
            captions = []
            
            if self.using_blip2:
                # BLIP-2 specific approach with different prompts
                for i in range(min(num_captions, len(text_prompts))):
                    inputs = self.processor(raw_image, text_prompts[i], return_tensors="pt").to(self.device, torch.float16)
                    
                    output = self.model.generate(
                        **inputs,
                        max_length=75,
                        num_beams=5,
                        temperature=1.0
                    )
                    
                    caption = self.processor.decode(output[0], skip_special_tokens=True)
                    captions.append(caption)
            else:
                # Standard BLIP approach
                inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
                
                # Generate multiple captions with beam search
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    num_return_sequences=num_captions,
                    temperature=1.0
                )
                
                # Decode captions
                captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
            
            # Apply NLP enhancements to each caption
            enhanced_captions = [self.enhance_caption(caption) for caption in captions]
            
            return {
                "original_captions": captions,
                "enhanced_captions": enhanced_captions
            }
        except Exception as e:
            return {"error": str(e)} 