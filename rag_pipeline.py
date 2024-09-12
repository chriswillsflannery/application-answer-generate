import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['USER_AGENT'] = "chris bot (chriswillsflannery@gmail.com)"

def parse_html(soup):
    parsed_content = []
    for tag in soup.find_all(['h3', 'p']):
        if tag.name == 'h3':
            parsed_content.append({"type": "question", "content": tag.get_text().strip()})
        elif tag.name == 'p':
            parsed_content.append({"type": "answer", "content": tag.get_text().strip()})
    return parsed_content

class CustomWebLoader(WebBaseLoader):
    def load(self):
        try:
            soup = self.scrape()
            parsed_content = parse_html(soup)
            docs = []
            for item in parsed_content:
                metadata = {"type": item["type"]}
                docs.append(Document(page_content=item["content"], metadata=metadata))
            return docs
        except Exception as e:
            print(f"An error occurred while loading documents: {e}")
            return []

def split_qa_pairs(docs):
    qa_pairs = []
    current_qa = {'question':'', 'answer': '' }
    for doc in docs:
        if doc.metadata['type'] == 'question':
            # if we encounter new question and we have a previous qa pair, add to list
            if current_qa['question'] or current_qa['answer']:
                qa_pairs.append(Document(
                    page_content=f"Q: {current_qa['question']}\nA: {current_qa['answer']}",
                    metadata={'type': 'qa_pair'}
                ))
            # start a new qa pair
            current_qa = {'question': doc.page_content, 'answer': ''}
        elif doc.metadata['type'] == 'answer':
            # we encounter an answer without a question so use placeholder
            if not current_qa['question']:
                # TODO refactor with something better
                current_qa['question'] = 'Unknown Question'
            current_qa['answer'] = doc.page_content
    # add last qa pair if exists
    if current_qa['question'] or current_qa['answer']:
        qa_pairs.append(Document(
            page_content=f"Q: {current_qa['question']}\nA: {current_qa['answer']}",
            metadata={'type': 'qa_pair'}
        ))
    return qa_pairs

# Initialize the RAG pipeline components
url = "https://chriswillsflannery.vercel.app/posts/ragExamplesForJobApplication"
loader = CustomWebLoader(url)
try:
    docs = loader.load()
    if not docs:
        print("No documents were loaded.")
    else:
        print(f"Number of documents: {len(docs)}")
        for doc in docs[:2]:  # Print first two documents as an example
            print(f"Type: {doc.metadata['type']}, Content: {doc.page_content[:100]}...")
except Exception as e:
    print(f"An error occurred while loading documents: {e}")

# text split if content loaded
if docs:
    qa_splits = split_qa_pairs(docs)

    try:
        vectorstore = Chroma.from_documents(documents=qa_splits, embedding=OpenAIEmbeddings())
        print("vectorstore created")
    except Exception as e:
        print(f"vectorstore creation failed: {e}")
    retriever = vectorstore.as_retriever()

prompt_template = PromptTemplate.from_template("""
You are an assistant helping a job applicant complete job applications.
For the purpose of this excercise, you will pretend to be the job applicant.
Use the following retrieved context to inform the tone and style of your answers.
The context contains examples of the applicant's previous responses to job application questions.
You should ALWAYS answer in the first person, as if you are the job applicant.
You should NEVER refer to the job applicant in the third person.
Retrieved context:
{context}
Now, based on the style and content of the above context, please answer the following job application question:
Question: {question}
Answer:
""")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

def format_docs(docs):
    return "\n\n".join(f"{doc.metadata['type'].capitalize()}: {doc.page_content}" for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

def process_rag_query(question):
    try:
        docs = retriever.get_relevant_documents(question)
        print(f"returned {len(docs)} documents")
        print(docs)

        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# If the script is run directly, test the pipeline
if __name__ == "__main__":
    test_question = "What makes you a good fit for this position?"
    print(process_rag_query(test_question))