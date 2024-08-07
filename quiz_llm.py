import boto3
import os
import uuid
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
AWS_REGION = os.getenv("AWS_REGION", 'us-east-1')
BUCKET_NAME = os.getenv("BUCKET_NAME", 'mirror-embeddings')

# Initialize AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path = os.path.join(os.path.expanduser("~"), "Documents")

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

def get_conversational_chain(llm, vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True
    )
    
    return chain

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
    
    llm = get_llm()

    def fetch_question():
        question_prompt = """Extract a random quiz question from the document 
        with multiple-choice options. Ensure it is not a previously asked question.Dont need to explain why 
        you chose this question just give the question.Dont create your know question,use the questions from the 
        document only """
        response = conversational_chain.invoke({"question": question_prompt})
        question = response['answer']
        return question

    while True:
        user_input = input("To exit the bot, type 'exit', 'quit', or 'bye'.\nUser: ").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        elif user_input == "quiz":
            # Reset memory and score at the start of each quiz
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True
            )
            conversational_chain = get_conversational_chain(llm, faiss_index)
            score = 0
            num_questions = 5  # Number of questions to ask in the quiz
            asked_questions = []
            current_question_index = 0 # Reset the question index

            print("Quiz started! Here is your first question:")

            # Fetch and ask the first question
            question = fetch_question()
            while question in asked_questions:
                question = fetch_question()
            asked_questions.append(question)
            print(f"Question {current_question_index + 1}: {question}")
            
            while current_question_index < num_questions:
                user_answer = input("Your answer (a/b): ").strip().lower()
                answer_prompt = f"""Evaluate the user's answer '{user_answer}' for correctness.
                Provide only one line positive feedback and indicate if the answer is correct or wrong.
                No explanation is needed just state right or wrong and give positive feedback"""
                answer_response = conversational_chain.invoke({"question": answer_prompt})

                feedback = answer_response['answer']
                print(f"Bot: {feedback}")

                # Update score based on feedback
                if "correct" in feedback.lower():
                    score += 1

                current_question_index += 1
                if current_question_index < num_questions:
                    question = fetch_question()
                    while question in asked_questions:
                        question = fetch_question()
                    asked_questions.append(question)
                    print(f"Question {current_question_index + 1}: {question}")

            print(f"\nQuiz complete! You scored {score}/{num_questions}.")
            # Reset for potential next quiz
            current_question_index = 0
            asked_questions = []

        else:
            # Handle other user inputs
            response = conversational_chain.invoke({"question": user_input})
            print(f"Bot: {response['answer']}")

if __name__ == "__main__":
    main()
