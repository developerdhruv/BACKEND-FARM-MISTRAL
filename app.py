# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, HTTPException, Query


access_token = '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'
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", token = access_token)



device = torch.device("cuda") # the device to load the model onto
app = FastAPI()


@app.get("/answer")
async def answer(query: str = Query(..., description="Text prompt for music generation")):
    if query != "":
        # Encode the user's query
        encoded = tokenizer.encode(query, return_tensors="pt")
        # Move the input to the appropriate device
        encoded = encoded.to(device)
        # Generate a response from the model
        generated_ids = model.generate(encoded, max_length=50, num_return_sequences=1)
        # Decode the generated response
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {"answer": response}
    else:
        # If an error occurs, return an HTTP 500 error
        raise HTTPException(status_code=500)

@app.get("/translate")
async def translate(text: str = Query(..., description="Text to translate"), target_lang: str = "ur"):
    if text:
        translated = translator.translate(text, dest=target_lang)
        return {"translated_text": translated.text}
    else:
        raise HTTPException(status_code=400, detail="Text to translate cannot be empty")
