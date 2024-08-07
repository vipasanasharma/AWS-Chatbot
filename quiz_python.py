import boto3
import os
import uuid
import re
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
AWS_REGION = os.getenv("AWS_REGION", 'us-east-1')
BUCKET_NAME = os.getenv("BUCKET_NAME", 'mirror-embeddings')

# Initialize AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
polly_client = boto3.client('polly', region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = os.path.join(os.path.expanduser("~"), "Documents")  # Adjusted for a more suitable local path

def get_unique_id():
    return str(uuid.uuid4())

def load_index():
    file_keys = [
        "9e81f4c4-3090-443f-a4f9-51dbedf75dd4.faiss",
        "9e81f4c4-3090-443f-a4f9-51dbedf75dd4.pkl",
        "a3189ab2-d34c-4b79-b520-86f3b13aa11b.faiss",
        "a3189ab2-d34c-4b79-b520-86f3b13aa11b.pkl"
    ]
    for key in file_keys:
        try:
            s3_client.download_file(Bucket=BUCKET_NAME, Key=key, Filename=os.path.join(folder_path, key))
            print(f"Downloaded {key} successfully.")
        except s3_client.exceptions.NoSuchKey:
            print(f"File {key} not found in bucket {BUCKET_NAME}.")
        except Exception as e:
            print(f"An error occurred while downloading {key}: {e}")

def get_llm():
    llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", client=bedrock_client,
                      model_kwargs={"messages": [
                          {"role": "system", "content": "You are a Compliance officer"}
                      ]})
    return llm

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide a concise answer to the question. Keep the responses
    to the point. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa.invoke({"query": question})  # use invoke instead of __call__
    text_response = answer['result']

    return text_response

def text_to_speech(text, output_file):
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Ruth',  # You can choose other voices available in Polly
            Engine='generative'  # Specify the correct engine
        )

        with open(output_file, 'wb') as file:
            file.write(response['AudioStream'].read())
        
        print(f"Audio saved to {output_file}")
        play_audio(output_file)

    except boto3.exceptions.Boto3Error as e:
        print(f"An error occurred: {e}")

def play_audio(file_path):
    if os.name == 'nt':  # Check if the system is Windows
        os.system(f'start {file_path}')  # Windows command to open the file with the default player
    else:
        os.system(f'xdg-open {file_path}')  # Linux command to open the file with the default player

def extract_relevant_parts(text):
    """ Extract relevant parts from the user's sentence. """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)  # Use a set to avoid duplicates

def is_match(user_input, option):
    """ Check if the user's input matches the option (case-insensitive). """
    normalized_input = normalize_answer(user_input)
    normalized_option = normalize_answer(option)
    # Check if any part of the normalized input is in the normalized option
    return normalized_input in normalized_option or normalized_option in normalized_input

def normalize_answer(answer):
    """ Normalize answers to handle formatting issues. """
    return answer.strip().lower()

def get_quiz_questions(vectorstore):
    # Retrieve quiz questions and answers from the vectorstore
    query = """Retrieve 5 random quiz questions, options, and answers. 
    Each time all 5 questions, options, and answers retrieved must be different. The options must 
    be separated by ; when differentiating between two options
    """
    llm = get_llm()
    response = get_response(llm, vectorstore, query)
    
    # Split response by double newlines to separate each Q&A pair
    qa_pairs = response.split('\n\n')
    questions = []
    options = []
    answers = []

    for qa in qa_pairs:
        # Further split each Q&A pair by newlines
        parts = qa.split('\n')
        if len(parts) >= 3:
            question = parts[0].replace('Q:', '').strip()
            options_text = parts[1].replace('Options:', '').strip()
            answer = parts[2].replace('Answer:', '').strip().lower()  # Store answers in lowercase
            
            # Clean up the answer prefix like "a: "
            if ':' in answer:
                answer = answer.split(':', 1)[1].strip()
            
            # Split options by newline or comma and strip whitespace
            option_list = [opt.strip() for opt in options_text.replace(';', '\n').split('\n') if opt.strip()]
            
            questions.append(question)
            options.append(option_list)
            answers.append(answer)
    
    return questions, options, answers

def ask_quiz(questions, options, answers):
    num_questions = min(5, len(questions))
    score = 0
    
    for i in range(num_questions):
        print(f"Question {i + 1}: {questions[i]}")
        
        # Print multiple-choice options
        for idx, option in enumerate(options[i]):
            print(f"  {chr(97 + idx)}. {option.strip()}")  # chr(97) is 'a', 98 is 'b', etc.
        
        attempts = 0
        max_attempts = 2
        
        while attempts < max_attempts:
            user_answer = input("Your answer (choose a, b, c, etc., or enter a full sentence): ").strip().lower()
            
            # Handle direct option input (e.g., 'a', 'b', etc.)
            if user_answer in map(chr, range(97, 97 + len(options[i]))):  # Check for 'a', 'b', etc.
                answer_index = ord(user_answer) - 97  # 'a' -> 0, 'b' -> 1, etc.
                if 0 <= answer_index < len(options[i]):
                    user_answer_text = normalize_answer(options[i][answer_index])
                    correct_answer = normalize_answer(answers[i])
                    if user_answer_text == correct_answer:
                        print("Correct!")
                        score += 1
                    else:
                        print(f"Wrong! The correct answer is: {answers[i]}")
                    break
                else:
                    print("Invalid option selected.")
            elif "option " in user_answer:
                # Handle options like 'option a', 'option b', etc.
                option_part = user_answer.split('option ', 1)[-1].strip()  # Extract 'a', 'b', etc.
                if option_part in map(chr, range(97, 97 + len(options[i]))):
                    answer_index = ord(option_part) - 97  # 'a' -> 0, 'b' -> 1, etc.
                    if 0 <= answer_index < len(options[i]):
                        user_answer_text = normalize_answer(options[i][answer_index])
                        correct_answer = normalize_answer(answers[i])
                        if user_answer_text == correct_answer:
                            print("Correct!")
                            score += 1
                        else:
                            print(f"Wrong! The correct answer is: {answers[i]}")
                        break
                    else:
                        print("Invalid option selected.")
                else:
                    print("Invalid option format.")
            else:
                # Extract relevant parts from the user's sentence
                relevant_parts = extract_relevant_parts(user_answer)
                
                # Normalize options
                normalized_options = [normalize_answer(option) for option in options[i]]
                
                matched_option = None
                for option in normalized_options:
                    # Check if any relevant part matches the option
                    if is_match(user_answer, option):
                        matched_option = option
                        break
                
                if matched_option:
                    correct_answer = normalize_answer(answers[i])
                    if matched_option == correct_answer:
                        print("Correct!")
                        score += 1
                    else:
                        print(f"Wrong! The correct answer is: {answers[i]}")
                    break
                else:
                    # Provide feedback if user's answer didn't match any options
                    print("Your answer didn't match any options. Please enter a valid choice or try again.")
            
            attempts += 1
            if attempts < max_attempts:
                print(f"Please try again. You have {max_attempts - attempts} attempt(s) left.")
    
    print(f"\nQuiz complete! You scored {score}/{num_questions}.")




def main():
    load_index()

    # Load FAISS index from local files
    faiss_index = FAISS.load_local(
        index_name="a3189ab2-d34c-4b79-b520-86f3b13aa11b",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    print("INDEX IS READY")

    while True:
        user_input = input("To exit the bot, type 'exit', 'quit', or 'bye'.\nUser: ").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        elif user_input == "quiz":
            questions, options, answers = get_quiz_questions(faiss_index)
            ask_quiz(questions, options, answers)

        else:
            response = get_response(get_llm(), faiss_index, user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    main()
