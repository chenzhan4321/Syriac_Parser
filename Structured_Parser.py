import os
import json
from openai import OpenAI

class SyriacWord:
    def __init__(self, word, analysis):
        self.word = word
        self.root = analysis.get('root', '')
        self.part_of_speech = analysis.get('part_of_speech', '')
        self.has_article = analysis.get('has_article', False)
        self.has_pronoun_suffix = analysis.get('has_pronoun_suffix', False)
        self.has_object_suffix = analysis.get('has_object_suffix', False)
        self.person = analysis.get('person', '')
        self.number = analysis.get('number', '')
        self.tense = analysis.get('tense', '')
        self.meaning = analysis.get('meaning', '')  # Add this line

def parse_syriac_text(text):
    client = OpenAI(api_key="")
    
    functions = [
        {
            "name": "analyze_syriac_text",
            "description": "Analyze multiple Syriac words in context",
            "parameters": {
                "type": "object",
                "properties": {
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string", "description": "The Syriac word to analyze"},
                                "root": {"type": "string", "description": "The root of the word"},
                                "part_of_speech": {"type": "string", "description": "The part of speech of the word"},
                                "has_article": {"type": "boolean", "description": "Whether the word has an article (for nouns)"},
                                "has_pronoun_suffix": {"type": "boolean", "description": "Whether the word has a pronoun suffix (for nouns)"},
                                "has_object_suffix": {"type": "boolean", "description": "Whether the word has an object suffix (for verbs)"},
                                "person": {"type": "string", "description": "The person of the word (for verbs)"},
                                "number": {"type": "string", "description": "The number of the word (for verbs)"},
                                "tense": {"type": "string", "description": "The tense of the word (for verbs)"},
                                "meaning": {"type": "string", "description": "the basic one or two meanings of the word"}
                            },
                            "required": ["word", "root", "part_of_speech"]
                        }
                    }
                },
                "required": ["words"]
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Syriac language expert. Analyze the given Syriac text and provide detailed grammatical information for each word, considering the context."},
                {"role": "user", "content": f"Analyze the following Syriac text, considering the context of each word:\n\n{text}"}
            ],
            functions=functions,
            function_call={"name": "analyze_syriac_text"},
            stream=True  # Enable streaming
        )
        
        partial_data = ""
        for chunk in response:
            if chunk.choices[0].delta.function_call:
                partial_data += chunk.choices[0].delta.function_call.arguments
                try:
                    words_analysis = json.loads(partial_data)
                    for word_analysis in words_analysis.get('words', []):
                        yield SyriacWord(word_analysis['word'], word_analysis)
                except json.JSONDecodeError:
                    # If the JSON is incomplete, continue to the next chunk
                    continue
    except Exception as e:
        print(f"Error in API call: {str(e)}")

def main():
    # Read text from sample.txt in the same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file_path = os.path.join(script_dir, 'test.txt')
    
    try:
        with open(sample_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: The file 'sample.txt' was not found in {script_dir}")
        return
    except Exception as e:
        print(f"Error reading the file: {str(e)}")
        return

    # Parse the text and print results as they become available
    for word in parse_syriac_text(text):
        print(f"Word: {word.word}")
        print(f"Root: {word.root}")
        print(f"Part of Speech: {word.part_of_speech}")
        if "noun" in word.part_of_speech.lower():
            print(f"Has Article: {word.has_article}")
            print(f"Has Pronoun Suffix: {word.has_pronoun_suffix}")
        elif "verb" in word.part_of_speech.lower():
            print(f"Has Object Suffix: {word.has_object_suffix}")
            print(f"Person: {word.person}")
            print(f"Number: {word.number}")
            print(f"Tense: {word.tense}")
        print(f"Meaning: {word.meaning}") 
        print("---")

if __name__ == "__main__":
    main()
