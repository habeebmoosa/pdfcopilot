from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import time

document = SimpleDirectoryReader("data").load_data()

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.llm = Ollama(
    model = "llama3",
    request_timeout=360.0
)

index = VectorStoreIndex.from_documents(document)
query_engine = index.as_query_engine()


while True:
    query = input("Enter your query: ")
    start_time = time.time()
    response = query_engine.query(query)
    end_time = time.time()
    total_time = end_time - start_time

    print("Time taken: ", total_time)
    print(response)
    