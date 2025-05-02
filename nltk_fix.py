import nltk

# Download all necessary NLTK resources with correct names
print("Downloading required NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # correct name is 'averaged_perceptron_tagger', not 'averaged_perceptron_tagger_eng'

print("\nAll required NLTK resources have been downloaded.")
print("You can now run the application with 'streamlit run app.py'") 