import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
import boto3

from generation.prompt_builder import PromptMessages
from config.settings import AWS_REGION, BEDROCK_GENERATION_MODEL

load_dotenv()


@dataclass
class Answer:
    text: str
    citations: List[str]   # extracted citation strings from the answer text
    query: str
    chunks_used: int


def generate(prompt: PromptMessages, query: str, chunks_used: int) -> Answer:
    """
    Call Amazon Bedrock with the prompt messages.
    Parse citation strings from the response (lines matching '[Source: ..., Section: ...]').
    Return an Answer dataclass.
    Model: configured Bedrock Converse model
    Max tokens: 1024
    AWS credentials use the normal boto3 credential chain (AWS profile,
    environment variables, or an IAM role).
    """
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", AWS_REGION))
    response = client.converse(
        modelId=os.getenv("BEDROCK_GENERATION_MODEL", BEDROCK_GENERATION_MODEL),
        system=[{"text": prompt.system}],
        messages=[{"role": "user", "content": [{"text": prompt.user}]}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0},
    )
    text = response["output"]["message"]["content"][0]["text"]
    citations = re.findall(r'\[Source: .+?, Section: .+?\]', text)

    return Answer(
        text=text,
        citations=citations,
        query=query,
        chunks_used=chunks_used,
    )
