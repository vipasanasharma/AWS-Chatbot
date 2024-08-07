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
    system_message = {
        "role": "system",
        "content": ("You are a Compliance officer. Respond to user inputs appropriately. "
                    "Provide with shorter explanations when asked about summaries")
    }
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        client=bedrock_client,
        model_kwargs={"messages": [system_message]}
    )
    return llm

def get_conversational_chain(llm, vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
         output_key="answer",
        return_messages=True
    )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True
    )
    return conversational_chain

def get_response(conversational_chain, question, chat_history):
    response = conversational_chain.invoke({"question": question, "chat_history": chat_history})
    text_response = response['answer']
    return text_response

def main():
    load_index()

    # Load FAISS index from local files
    faiss_index = FAISS.load_local(
        index_name="9e81f4c4-3090-443f-a4f9-51dbedf75dd4",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    print("INDEX IS READY")

    llm = get_llm()
    conversational_chain = get_conversational_chain(llm, faiss_index)
    chat_history = []

    while True:
        user_input = input("To exit the bot, type 'exit', 'quit', or 'bye'.\nUser: ").strip().lower()

        if user_input in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        response = get_response(conversational_chain, user_input, chat_history)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
