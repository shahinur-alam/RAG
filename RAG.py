import os
import warnings
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment variable to handle OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load pretrained RAG components
model_name = "facebook/rag-sequence-nq"
tokenizer = RagTokenizer.from_pretrained(model_name)
retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare input
question = "What is the next invention after AI?"
input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)

# Generate answer
outputs = model.generate(input_ids)
answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Question: {question}")
print(f"Answer: {answer}")
