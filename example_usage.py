"""
Example script demonstrating how to use the image captioning models programmatically
"""

import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from image_captioner import ImageCaptioner
from advanced_captioner import AdvancedCaptioner

def display_caption_results(image_path, results, title):
    """Display the image and caption results"""
    plt.figure(figsize=(12, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Display the caption
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title(title)
    
    if "error" in results:
        plt.text(0.5, 0.5, f"Error: {results['error']}", 
                 ha='center', va='center', wrap=True, fontsize=12)
    else:
        y_pos = 0.9
        plt.text(0.1, y_pos, "Original Caption:", fontweight='bold', fontsize=12)
        y_pos -= 0.1
        plt.text(0.1, y_pos, results["original_caption"], fontsize=12, wrap=True)
        
        y_pos -= 0.2
        plt.text(0.1, y_pos, "Enhanced Caption:", fontweight='bold', fontsize=12)
        y_pos -= 0.1
        plt.text(0.1, y_pos, results["enhanced_caption"], fontsize=12, wrap=True)
        
        # If it's the advanced model and has NLP analysis
        if "nlp_analysis" in results:
            y_pos -= 0.2
            plt.text(0.1, y_pos, "NLP Analysis:", fontweight='bold', fontsize=12)
            y_pos -= 0.1
            analysis = results["nlp_analysis"]
            
            plt.text(0.1, y_pos, f"Content Words: {analysis['content_word_count']} / Total Words: {analysis['total_word_count']}", fontsize=10)
            y_pos -= 0.1
            
            if analysis["noun_phrases"]:
                phrases_text = "Noun Phrases: " + ", ".join(analysis["noun_phrases"])
                plt.text(0.1, y_pos, phrases_text, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate captions for an image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', choices=['standard', 'advanced'], default='standard',
                        help='Model to use (standard=ViT-GPT2, advanced=BLIP)')
    parser.add_argument('--beam-search', action='store_true', default=True,
                        help='Use beam search for the advanced model (default: True)')
    parser.add_argument('--no-beam-search', dest='beam_search', action='store_false',
                        help='Disable beam search for the advanced model')
    
    args = parser.parse_args()
    
    # Check if the image file exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        sys.exit(1)
    
    # Load the appropriate model
    print(f"Loading {args.model} model...")
    if args.model == 'standard':
        captioner = ImageCaptioner()
        results = captioner.generate_caption(args.image_path)
        title = "Standard Model (ViT-GPT2)"
    else:
        captioner = AdvancedCaptioner()
        results = captioner.generate_caption(args.image_path, use_beam_search=args.beam_search)
        title = f"Advanced Model (BLIP) - Beam Search: {args.beam_search}"
    
    print("\nResults:")
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Original Caption: {results['original_caption']}")
        print(f"Enhanced Caption: {results['enhanced_caption']}")
        
        # Print NLP analysis if available
        if "nlp_analysis" in results:
            analysis = results["nlp_analysis"]
            print("\nNLP Analysis:")
            print(f"Content Words: {analysis['content_word_count']} / Total Words: {analysis['total_word_count']}")
            if analysis["noun_phrases"]:
                print(f"Noun Phrases: {', '.join(analysis['noun_phrases'])}")
    
    # Display results graphically
    display_caption_results(args.image_path, results, title)

if __name__ == "__main__":
    main() 