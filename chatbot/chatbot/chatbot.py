"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

# Import pynecone.
import openai
import os
from datetime import datetime

import pynecone as pc
from pynecone.base import Base

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate



CUR_DIR = os.path.dirname(os.path.abspath('chatbot'))
kakao_sync_txt = os.path.join(CUR_DIR, "data/project_data.txt")


# openai.api_key = "<YOUR_OPENAI_API_KEY>"
os.environ["OPENAI_API_KEY"] = open("./key.txt", "r").read()

# 카카오싱크 파일 읽기 
def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

# 이렇게 받은 데이터 내용을 어떻게 좀 정리하면 좋을까?
# 딕셔너리로 좀 나눠놓으까... 

# 받은 내용 디비에 저장 

# create_chain
def create_chain(llm, kakao_sync_txt, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(kakao_sync_txt),
        ),
        output_key=output_key,
        verbose=True, # 얘네들이 진행되는 상황을 프린트 문으로 찍어줌
    )


from langchain.chains import SequentialChain

def generate_answer(question) -> dict[str, str]:

  llm = ChatOpenAI(temperature=0.1, max_tokens=300, model="gpt-3.5-turbo-16k")

  # 아이디어 뽑기 체인 생성
  answer_chain1 = create_chain(llm, kakao_sync_txt, "답변")

  preprocess_chain = SequentialChain( #SequentialChain 순서대로 실행
      chains=[
          answer_chain1,
          #answer_chain2
      ],
      input_variables=["kakao_sync_txt", "question"],
      output_variables=["답변"],
      verbose=True,
  )

  context = dict(
      kakao_sync_txt=read_prompt_template(kakao_sync_txt),
      question=question,
  )
  context = preprocess_chain(context)
  '''
  context["novel_chapter"] = []
  for chapter_number in range(1, 3):
      context["chapter_number"] = chapter_number
      context = novel_chapter_chain(context)
      context["novel_chapter"].append(context["output"])
  '''
  #contents = "\n\n".join(context["novel_chapter"])
  #return {"results": contents}

  context = answer_chain1(context)

  return context["답변"]

# 답변 요약 알고리즘 추가 
# 답변 번호 매기기 추가 
#question = "카카오싱크 기능이 무엇이 있는지 설명해주세요"
#print(generate_answer(question))

def answer_using_chatgpt(user_input: str):
    print("create llm ...")
    llm = ChatOpenAI(temperature=0.1, max_tokens=300, model="gpt-3.5-turbo-16k")

    print("create prompt template ...")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                f"""
                assistant는 챗봇으로서 동작한다. 가이드를 참고하여 질문에 답변한다.

                가이드
                '''
                {read_prompt_template(kakao_sync_txt)}
                '''
                """
            ),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    print("create chain and run ...")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = chain.run(
        question=user_input,
        verbose=True
    )
    return answer


# llm 정의 
#llm = ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo-16k')


parallel_example = {
    "한국어": ["오늘 날씨 어때", "딥러닝 기반의 AI기술이 인기를끌고 있다."],
    "영어": ["How is the weather today", "Deep learning-based AI technology is gaining popularity."],
    "일본어": ["今日の天気はどうですか", "ディープラーニングベースのAIテクノロジーが人気を集めています。"]
}


def translate_text_using_text_davinci(text, src_lang, trg_lang) -> str: #text : 한국문장 
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=f"Translate the following {src_lang} text to {trg_lang}: {text}",
                                        max_tokens=200,
                                        n=1,
                                        temperature=1
                                        )
    translated_text = response.choices[0].text.strip()
    return translated_text


def translate_text_using_chatgpt(text, src_lang, trg_lang) -> str:
    # fewshot 예제를 만들고
    def build_fewshot(src_lang, trg_lang):
        src_examples = parallel_example[src_lang]
        trg_examples = parallel_example[trg_lang]

        fewshot_messages = [] # 메세지를 여러개 질문 하는 방식 

        for src_text, trg_text in zip(src_examples, trg_examples):
            fewshot_messages.append({"role": "user", "content": src_text})
            fewshot_messages.append({"role": "assistant", "content": trg_text})

        return fewshot_messages

    # system instruction 만들고
    system_instruction = f"assistant는 번역앱으로서 동작한다. {src_lang}를 {trg_lang}로 적절하게 번역하고 번역된 텍스트만 출력한다." #롤 적용 > 어시스턴트는 번역앱 

    # messages를만들고
    fewshot_messages = build_fewshot(src_lang=src_lang, trg_lang=trg_lang)

    messages = [{"role": "system", "content": system_instruction},
                *fewshot_messages, # 한국어 영어 
                {"role": "user", "content": text}
                ]

    # API 호출
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
    translated_text = response['choices'][0]['message']['content']
    # Return
    return translated_text




class Message(Base):
    original_text: str
    text: str
    created_at: str
    to_lang: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []
    src_lang: str = "한국어"
    trg_lang: str = "영어"

    @pc.var
    def output(self) -> str:
        if not self.text.strip():
            return "Translations will appear here."
        translated = translate_text_using_chatgpt(
            self.text, src_lang=self.src_lang, trg_lang=self.trg_lang)
        return answer_using_chatgpt(self.text)

    def post(self):
        self.messages = [
            Message(
                original_text=self.text,
                text=self.output,
                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                to_lang=self.trg_lang,
            )
        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Translator 🗺", font_size="2rem"),
        pc.text(
            "Translate things and post them as messages!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            down_arrow(),
            text_box(message.text),
            pc.box(
                pc.text(message.to_lang),
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "Output",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.output),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Text to translate",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        pc.select(
            list(parallel_example.keys()),
            value=State.src_lang,
            placeholder="Select a language",
            on_change=State.set_src_lang,
            margin_top="1rem",
        ),
        pc.select(
            list(parallel_example.keys()),
            value=State.trg_lang,
            placeholder="Select a language",
            on_change=State.set_trg_lang,
            margin_top="1rem",
        ),
        output(),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="Translator")
app.compile()
