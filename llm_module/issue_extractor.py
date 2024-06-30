from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
あなたの役割は、ユーザーが提供したテキストから具体的な困りごとや質問を抽出し、それをリスト形式で整理することです。ユーザーのメッセージを注意深く読み取り、以下のポイントに従ってください：

1. 各質問や困りごとを個別の項目としてリスト化してください。
2. 質問が複数の部分に分かれている場合、それぞれの部分を別々の項目として抽出してください。
3. 不明瞭な部分がある場合でも、可能な限り具体的な項目として整理してください。
4. 抽出した項目は、ユーザーが明確な回答を求めているものに集中してください。

このプロセスを通じて、ユーザーの質問や困りごとをわかりやすく整理することで、迅速かつ的確な対応を可能にします。
"""

class UserIssueExtractor(BaseModel):
    """ユーザの困りごと単位に分割してリストする"""
    questions: List[str] = Field(
        default_factory=list,
        description="ユーザの困りごとを単位に分割してリストする"
    )


def structured_issue_extractor(llm):
    """
    ユーザーのテキストから具体的な困りごとや質問を抽出し、リスト形式で整理する関数。

    使用例:
    from langchain_openai import ChatOpenAI
    from my_llm_module import structured_issue_extractor

    # LLMの定義
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    # Issue Extractorの設定と実行
    issue_extractor_instance = structured_issue_extractor(llm)
    question = "こんにちは、先日注文した商品（注文番号：12345）の発送状況についてお伺いしたいのですが、まだ発送通知が届いていないので現在のステータスを教えてください。また、発送がまだであればいつ頃発送される予定かも知りたいです。さらに、もし商品に不具合があった場合や配送中に破損が発生した場合の対応方法についても教えてもらえますか？よろしくお願いします。"
    generated_issue = issue_extractor_instance.invoke({"input": question})

    # 出力例:
    # generated_issue.questions
    # [
    #     "注文番号：12345の発送状況について教えてください。",
    #     "発送がまだであれば、いつ頃発送される予定か教えてください。",
    #     "商品に不具合があった場合の対応方法を教えてください。",
    #     "配送中に破損が発生した場合の対応方法を教えてください。"
    # ]

    Parameters:
    llm : LLMのインスタンス。

    Returns:
    ChatPromptTemplate: ユーザーの入力を処理し、質問や困りごとを抽出するプロンプトテンプレート。
    """
    structured_llm = llm.with_structured_output(UserIssueExtractor)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}")
        ]
    )
    return prompt | structured_llm


