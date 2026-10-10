"""Evaluate RAG answers with RAGAS and Bedrock; see evaluation/questions.json first."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference

from config.settings import AWS_REGION, BEDROCK_GENERATION_MODEL, EMBED_MODEL
from pipeline.graph import run_query_with_trace


def main() -> None:
    questions_path = Path("evaluation/questions.json")
    if not questions_path.exists():
        raise RuntimeError("Create evaluation/questions.json from evaluation/questions.example.json first.")

    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    samples = []
    for case in cases:
        state = run_query_with_trace(case["question"])
        answer = state["answer"]
        samples.append(SingleTurnSample(
            user_input=case["question"],
            response=answer.text,
            retrieved_contexts=[chunk.content for chunk in state["retrieved_chunks"]],
            reference=case["reference"],
        ))

    llm = ChatBedrockConverse(
        model_id=os.getenv("BEDROCK_GENERATION_MODEL", BEDROCK_GENERATION_MODEL),
        region_name=os.getenv("AWS_REGION", AWS_REGION),
    )
    embeddings = BedrockEmbeddings(
        model_id=os.getenv("EMBED_MODEL", EMBED_MODEL),
        region_name=os.getenv("AWS_REGION", AWS_REGION),
    )
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()],
        llm=llm,
        embeddings=embeddings,
    )
    print(result)


if __name__ == "__main__":
    main()
