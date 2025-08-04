from dotenv import load_dotenv
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRACeloop_API_KEY = os.getenv("TRACeloop_API_KEY")

Traceloop.init(
  disable_batch=True,
  api_key=TRACeloop_API_KEY
)

client = OpenAI(api_key=OPENAI_API_KEY)

@task(name="translate_hi2en")
def translate_hindi(hindi_text: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Translate this Hindi text to English: " + hindi_text
    )
    return response.output_text


@task(name="translate_en2hi")
def translate_english(english_text: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Translate this English text to Hindi: " + english_text
    )
    return response.output_text


@workflow(name="translate")
def translate(hindi_text: str) -> tuple[str, str]:
    english_text = translate_hindi(hindi_text)
    regenerated_hindi_text = translate_english(english_text)
    return english_text, regenerated_hindi_text


def main():
    hindi_text = "आज का मौसम कैसा है?"
    english_text, regenerated_hindi_text = translate(hindi_text)
    print(f"{hindi_text} -> {english_text} -> {regenerated_hindi_text}")


if __name__ == "__main__":
    main()