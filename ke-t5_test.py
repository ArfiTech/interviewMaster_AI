from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 'KETI-AIR/ke-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def translate_korean_to_english(text):
    model_name = 'KETI-AIR/ke-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model.generate(input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def translate_english_to_korean(text):
    model_name = 'KETI-AIR/ke-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model.generate(input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    text_kr = "안녕하세요 저는 컴퓨터공학을 전공하고 있는 이아무개라고 합니다."
    text_en = "Hello my name is mosi"
    
    tr_text = translate_korean_to_english(text_kr)
    print(tr_text)
