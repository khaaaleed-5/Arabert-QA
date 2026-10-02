import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_ID = "Khaaaleed5/arabert-qa"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID).eval()


def answer(question: str, context: str) -> str:
    if not question.strip() or not context.strip():
        return "الرجاء إدخال السؤال والنص. / Please enter both a question and a context."

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    start = outputs.start_logits.argmax()
    end = outputs.end_logits.argmax() + 1
    text = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
    return text.strip() or "لم يتم العثور على إجابة. / No answer found."


examples = [
    [
        "ما هي عاصمة مصر؟",
        "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها، وتقع على ضفاف نهر النيل.",
    ],
    [
        "أين تقع جامعة القرويين؟",
        "جامعة القرويين هي جامعة تقع في مدينة فاس بالمغرب، وتأسست عام 859 ميلادية على يد فاطمة الفهرية.",
    ],
]

demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="السؤال (Question)", lines=2, rtl=True),
        gr.Textbox(label="النص (Context)", lines=8, rtl=True),
    ],
    outputs=gr.Textbox(label="الإجابة (Answer)", rtl=True),
    examples=examples,
    title="AraBERT Arabic Question Answering",
    description="Extractive QA: the answer is a span taken from the context. Model: "
    f"[{MODEL_ID}](https://huggingface.co/{MODEL_ID})",
)

if __name__ == "__main__":
    demo.launch()
