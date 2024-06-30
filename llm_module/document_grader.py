from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are a grader assessing the relevance of a retrieved document to a user question.
    Your goal is to filter out erroneous retrievals without being overly strict.
    If the document contains keywords or semantic meanings related to the user question, grade it as relevant.
    Give a binary score ('yes' or 'no') to indicate whether the document is relevant to the question.
"""

class GradeDocumentsWithReasoning(BaseModel):
    """Binary score and reasoning for relevance check on retrieved documents."""
    reasoning: str = Field(
        description="Thinking process to give a correct binary score shortly."
    )
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeDocumentsWithoutReasoning(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def structured_document_grader(llm, reasoning: bool):
    """
    ユーザーの質問に対するretrieverから取得されたdocumentの関連性を評価する関数。

    この関数は、retrieverから取得されたdocumentがユーザーの質問に関連しているかどうかをバイナリスコア（'yes' or 'no'）で評価します。
    関数の出力はchainであり、chainは以下の形式の文字列を引数として受け取ります
    - document: retrieverから取得されたドキュメントの内容
    - question: ユーザーの質問

    Parameters:
    llm : LLMのインスタンス。
    reasoning : bool
        Trueの場合、関連性評価の理由を含む出力を生成します。Falseの場合、バイナリスコアのみを生成します。

    Returns:
    ChatPromptTemplate: ユーザーの入力を処理し、関連性を評価するプロンプトテンプレート。

    使用例:
    from langchain_openai import ChatOpenAI
    from llm_module.document_grader import structured_document_grader

    # LLMの定義
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    # 実行
    document_grader = structured_document_grader(llm, reasoning=True)
    document = "このドキュメントは、AI技術の最新動向について説明しています。特に、機械学習とディープラーニングの進展に焦点を当てています。また、AIの倫理的な側面についても議論しています。"
    question = "AI技術の最新動向について教えてください。"
    result = document_grader.invoke({"document": document, "question": question})

    # 出力例:
    # result
    # {
    #     "reasoning": "ドキュメントはAI技術の最新動向について説明しているため、関連性があります。",
    #     "binary_score": "yes"
    # }
    """
    if reasoning:
        structured_llm = llm.with_structured_output(GradeDocumentsWithReasoning)
    else:
        structured_llm = llm.with_structured_output(GradeDocumentsWithoutReasoning)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Retrieved document: \n\n {document} \n\n User question: \n\n {question}")
        ]
    )
    return prompt | structured_llm
