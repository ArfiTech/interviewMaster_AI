import os, json
import openai
import torch
import transformers
import math
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from long_sentence_split import split_sentences
from vector_visualization import visualize_word_distances


from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SECURITY WARNING: keep the secret key used in production secret!

secret_file = os.path.join(BASE_DIR, 'interviewMaster_AI/secrets.json')

with open(secret_file) as f:
    secrets = json.loads(f.read())

def get_secret(setting, secrets=secrets):
    try:
        return secrets[setting]
    except KeyError:
        error_msg = "Set the {} environment variable".format(setting)
        raise ImproperlyConfigured(error_msg)

SECRET_KEY = get_secret("OPENAI_KEY")

class ScoreAns:
    
    openai.api_key = SECRET_KEY
    
    def __init__(self, inter_q, inter_a,
                 max_tokens=500,
                 n=1,
                 stop=None,
                 temperature=0.5,
                 frequency_penalty=0.5,
                 presence_penalty=0.5):
        self.inter_q = inter_q
        self.inter_a = inter_a
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
    
    # BertForNextSentencePrediction: 두 문장이 이어져 있는지 여부를 예측하기 위한 모델로, 두 문장이 이어져 있다고 예측하는 값이 1에 가까움
    def score_question_answer_tf(self, question, answer):
        model = transformers.BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        
        input_text = f"{question} [SEP] {answer} [SEP]"
        input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
        
        outputs = model(input_ids)
        print("model output: ", outputs)
        logits = outputs[0][0][0].item()
        score = 1 / (1 + math.exp(-logits))
        return score
    
    # 단순 sentence_transformers 적용 후 cosine similarity 계산
    def score_question_answer_st(self, question, answer):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        
        question_embeddings = model.encode(question)
        answer_embeddings = model.encode(answer)
        print("question_emb = {}, answer_emb = {}".format(question_embeddings.shape, answer_embeddings.shape))
        
        similarity = cosine_similarity([question_embeddings], [answer_embeddings])[0][0]
        
        return similarity
    
    # nltk를 이용한 질문/대답 전처리 및 hidden states 결과 임배딩, 코사인 유사도 계산
    def score_question_answer(self, question, answer):
        model = transformers.BertForNextSentencePrediction.from_pretrained('bert-base-uncased', output_hidden_states=True)
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 질문 대답 전처리(불용어 제거)
        stop_words = set(stopwords.words('english'))
        question_tokens = [token.lower() for token in word_tokenize(question) if token.lower() not in stop_words]
        answer_tokens = [token.lower() for token in word_tokenize(answer) if token.lower() not in stop_words]
        question_text = ' '.join(question_tokens)
        answer_text = ' '.join(answer_tokens)

        # inputs 인코딩
        encoded_input = tokenizer(question_text, answer_text, return_tensors='pt')
        with torch.no_grad(): # 모델의 weight를 업데이트 하지 않음
            model_output = model(**encoded_input) # 딕셔너리를 unpacking 하여 인자로 전달
        print("model_output: ", model_output)

        # 임베딩을 얻고 코사인 유사도 계산
        question_embedding = model_output.hidden_states[-1][0].numpy()
        answer_embedding = model_output.hidden_states[0][0].numpy()
        question_embedding = question_embedding.reshape(1, -1)
        answer_embedding = answer_embedding.reshape(1, -1)
        similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]

        return similarity
    
    def score_question_answer_hdn(self, question, answer):
        # BERT 모델 및 토크나이저 초기화
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        
        # 입력 문장을 토큰화하고 텐서로 변환
        input_ids = torch.tensor([tokenizer.encode(question, add_special_tokens=True)])
        answer_ids = torch.tensor([tokenizer.encode(answer, add_special_tokens=True)])
        
        # BERT 모델의 hidden states 추출
        with torch.no_grad():
            outputs = model(input_ids)
            question_hidden_states = outputs[0].squeeze(0)  # 첫 번째 문장의 hidden states
            outputs = model(answer_ids)
            answer_hidden_states = outputs[0].squeeze(0)  # 첫 번째 문장의 hidden states

        # 벡터 값을 추출하여 코사인 유사도 계산
        question_vector = torch.mean(question_hidden_states, dim=0).numpy()
        answer_vector = torch.mean(answer_hidden_states, dim=0).numpy()
        similarity = cosine_similarity([question_vector], [answer_vector])[0][0]

        return similarity
    
    # BERT를 이용한 한영/영한 번역 (fail)
    def score_question_answer_k2e(self, question, answer):
        # 한국어 -> 영어 변환을 위한 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        
        # 문장 벡터화 모델 (영어)
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        
        # 영어로 변환된 문장을 저장할 리스트
        english_questions = []
        english_answers = []
        
        # 영어로 변환된 문장을 저장
        for question_seq in question_sequences:
            english_question = tokenizer(question_seq, return_tensors='pt')['input_ids']
            english_questions.append(english_question)
            
        for answer_seq in answer_sequences:
            english_answer = tokenizer(answer_seq, return_tensors='pt')['input_ids']
            english_answers.append(english_answer)
    
    
    def translate_text(self, texts):
        # Join sentences with a delimiter
        delimiter = "|||" # 특정 delimiter 설정
        text_with_delimiter = delimiter.join(texts)
        
        # Create the translation prompt
        prompt = f"Translate the following Korean texts(Please do not remove the symbol ('|||')) to English:\n{text_with_delimiter}"
        print("tr_prompt: ", prompt)
        
        # Perform the translation using OpenAI API
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=100*len(texts),
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=None,
            n=1
        )
    
        # Extract the translated texts from the API response
        #translations = [choice['text'].strip() for choice in response['choices']]
        #translations = translations[0].split('\n')
        print("response: ", response)
        
        # Extract the translated text from the API response
        translation = response['choices'][0]['text'].strip()
        
        # Split the translated text into individual sentences
        translations = translation.split(delimiter)
        print("translations: ", translations)
        
        return translations
    
    def score_answer(self):
        scores = []
        feedbacks = []
        prompt_template = "Please generate feedback to evaluate if the following answer to the question is appropriate in Korean using high honorifics(one listener): Quesion: {question}\nAnswer: {answer}\n\nFeedback:"
        bound_max = 10 # 받을 수 있는 점수 중 가장 높은 점수
        pass_score = 4 # 합격 기준 score

        # Translate Korean questions and answers to English
        translated_questions = self.translate_text(self.inter_q)
        translated_answers = self.translate_text(self.inter_a)
        
        print("translated_questions: ", translated_questions)
        print("translated_answers: ", translated_answers)
        
        for i in range(len(self.inter_q)):
            #prompt = prompt_template.format(question=self.inter_q[i], answer=self.inter_a[i])
            #prompt_score = prompt_template_score.format(question=self.inter_q[i], answer=self.inter_a[i])
            prompt = prompt_template.format(question=translated_questions[i], answer=translated_answers[i])
            print("prompt: ", prompt)
            
            score = self.score_question_answer_hdn(question=translated_questions[i], answer=translated_answers[i])
            #score = self.score_question_answer(question=translated_questions[i], answer=translated_answers[i])
            score = float(score)*10
            print("score: ", score)
            
            # 피드백 생성
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=self.max_tokens,
                n=self.n,
                stop=self.stop,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )

            # 대답으로부터 feedback 가져오기
            response_text = response.choices[0].text
            feedback = response.choices[0].text.strip()
            print("{}th feedback: {}".format(i, feedback))
            '''
            score = self.score_question_answer(question=self.inter_q[i], answer=self.inter_a[i])
            print(score)
            '''
            # output list에 요소 append
            feedbacks.append(feedback)
            scores.append(score)

        return scores, feedbacks, bound_max, pass_score


if __name__ == "__main__":
    '''
    question = "What motivated you to apply for our company?"
    answer = "Your company is showing very rapid growth. The current size is small, but I want to work with you by spreading the possibility of future growth."
    '''
    '''
    inter_q = [
        "Can you tell me about yourself?",
        "What are your strengths?",
        "What are your weaknesses?",
        "Why do you want to work for our company?",
        "What is freedom in your opinion?",
    ]

    inter_a = [
        "I am a recent graduate with a degree in computer science. During my studies, I worked on several projects that involved software development, and I also completed an internship at a technology company. In my free time, I enjoy reading books and hiking.",
        "One of my strengths is my ability to work well in a team. I am a good listener and communicator, and I always make an effort to understand my team members' perspectives. I am also a quick learner, which allows me to adapt to new situations easily.",
        "One of my weaknesses is that I tend to be a perfectionist, which can sometimes lead to spending too much time on a task. I have been working on finding a balance between striving for excellence and not letting perfect be the enemy of good.",
        "I am excited about the work that your company is doing in the field of artificial intelligence, and I admire your commitment to innovation and collaboration. I believe that my skills and interests align well with the work that your company is doing, and I would love to be a part of that.",
        "I want to eat pizza.",
    ]
    '''
    
    inter_q = [
        "당신이 가장 선호하는 프로그래밍 언어는 무엇인가요?",
        "특정 언어나 기술 스택에 대한 지식이 없는 지원자에게 회사에서 기대하는 것은 무엇인가요?",
        "어떤 프로젝트에서 힘들었던 점이나 어려움을 어떻게 해결했는지 알려주세요.",
        "당신에게 있어서 자유란 무엇입니까?"
    ]

    inter_a = [
        "저는 파이썬이 가장 선호되는 프로그래밍 언어입니다.",
        "저는 지원자가 특정 언어나 기술 스택에 대한 전문 지식보다는 학습 및 적응 능력, 문제 해결 능력, 그리고 협업 능력을 중요하게 여깁니다.",
        "저는 지난번 프로젝트에서 X 기술을 활용하려다가 어려움을 겪었지만, Y 기술로 대체하여 문제를 해결하였습니다.",
        "저는 피자가 먹고싶습니다."
    ]
    
    scoring = ScoreAns(inter_q, inter_a)
    score, feedback, bound_max, pass_score= scoring.score_answer()
    print("score: ", score)
    print("feedback: ", feedback)