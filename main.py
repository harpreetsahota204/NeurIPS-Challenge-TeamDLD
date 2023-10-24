from fastapi import FastAPI, HTTPException, Query
import asyncio

app = FastAPI()

@app.post("/process")
async def process_text(text: str):
    # Simulate a time-consuming async operation
    await asyncio.sleep(1)
    # Your processing logic here
    processed_text = f"Processed: {text}"
    return {"result": processed_text}

@app.post("/tokenize")
async def tokenize_text(text: str):
    # Simulate a time-consuming async operation
    await asyncio.sleep(1)
    # Your tokenization logic here
    tokenized_text = text.split()
    return {"tokens": tokenized_text}

@app.post("/decode")
async def decode_text(tokens: str):
    # Simulate a time-consuming async operation
    await asyncio.sleep(1)
    # Your decoding logic here
    decoded_text = " ".join(tokens)
    return {"decoded_text": decoded_text}
