import os, json
import openai

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SECURITY WARNING: keep the secret key used in production secret!

secret_file = os.path.join(BASE_DIR, 'secrets.json')

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
    
    # openAI API를 이용해 두 문장 간 유사도 측정
    def score_answer(self):
        scores = []
        feedbacks = []
        prompt_template_score = "Score the following answer based on its relevance to the question:\nQuestion: {question}\nAnswer: {answer}\nScore:"
        prompt_template = "Q: {question}\nA: {answer}\n\nFeedback:"

        for i in range(len(self.inter_q)):
            prompt = prompt_template.format(question=self.inter_q[i], answer=self.inter_a[i])
            prompt_score = prompt_template_score.format(question=self.inter_q[i], answer=self.inter_a[i])
            
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
                #logprobs=10
            )
            #print("response: ", response)

            # 대답으로부터 feedback 가져오기
            feedback = response.choices[0].text.strip()
            
            response_score = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt_score,
                max_tokens=1, # 유사도 점수만 반환
                n=self.n,
                stop=self.stop,
                temperature=0.0, # 유사도 점수의 일관성 높이기
            )
            #print(response_score)
            score = response_score.choices[0].text.strip()
            #print("{}th score: {}".format(i, score))
            
            # output list에 요소 append
            feedbacks.append(feedback)
            scores.append(float(score))

        return scores, feedbacks


if __name__ == "__main__":
    '''
    question = "What motivated you to apply for our company?"
    answer = "Your company is showing very rapid growth. The current size is small, but I want to work with you by spreading the possibility of future growth."
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
    '''
    scoring = ScoreAns(inter_q, inter_a)
    score, feedback = scoring.score_answer()
    print("score: ", score)
    print("feedback: ", feedback)