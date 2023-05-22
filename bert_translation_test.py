from transformers import AutoTokenizer, AutoModelWithLMHead
from sentence_transformers import SentenceTransformer, util

def translate_korean_to_english(text):
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base")

    input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model.generate(input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def translate_english_to_korean(text):
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base")
    
    input_ids = tokenizer.encode(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model.generate(input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def score_question_answer_k2ehdn(question_sequences, answer_sequences):
    # 한국어 -> 영어 변환을 위한 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    
    # 문장 벡터화 모델 (영어)
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    # 영어로 변환된 문장을 저장할 리스트
    english_questions = []
    english_answers = []
    
    # 영어로 변환된 문장을 저장
    for question_seq in question_sequences:
        english_question = tokenizer(question_seq, return_tensors='pt')
        english_questions.append(english_question)
        
    for answer_seq in answer_sequences:
        english_answer = tokenizer(answer_seq, return_tensors='pt')
        english_answers.append(english_answer)
    
    print("english_questions: ", english_questions)
    print("english_answers: ", english_answers)
    
    # 문장 벡터화
    question_embeddings = model.encode(english_questions)
    answer_embeddings = model.encode(english_answers)
    print("q_embeddings: ", question_embeddings)
    print("a_embeddings: ", answer_embeddings)
    
    # Similarity 계산
    cos_sim = util.cos_sim(question_embeddings, answer_embeddings)
    
    # Similarity 출력
    for i in range(len(question_sequences)):
        for j in range(len(answer_sequences)):
            print(f"Question {i+1} - Answer {j+1}: {cos_sim[i][j]}")
        
    return english_questions, english_answers

if __name__ == "__main__":
    text_kr = "안녕하세요 저는 컴퓨터공학을 전공하고 있는 이아무개라고 합니다."
    text_en = "Hello my name is mosi"
    
    inter_q = [
        "회사에서 가장 선호하는 프로그래밍 언어는 무엇인가요?",
        "특정 언어나 기술 스택에 대한 지식이 없는 지원자에게 회사에서 기대하는 것은 무엇인가요?",
        "어떤 프로젝트에서 힘들었던 점이나 어려움을 어떻게 해결했는지 알려주세요.",
        "당신에게 있어서 자유란 무엇입니까?"
    ]

    inter_a = [
        "저희 회사에서는 파이썬이 가장 선호되는 프로그래밍 언어입니다.",
        "저희 회사에서는 지원자가 특정 언어나 기술 스택에 대한 전문 지식보다는 학습 및 적응 능력, 문제 해결 능력, 그리고 협업 능력을 중요하게 여깁니다.",
        "저는 지난번 프로젝트에서 X 기술을 활용하려다가 어려움을 겪었지만, Y 기술로 대체하여 문제를 해결하였습니다.",
        "저는 피자가 먹고싶습니다."
    ]
    
    score_question_answer_k2ehdn(inter_q, inter_a)
    
    #tr_text = translate_english_to_korean(text_en)
    #print(tr_text)