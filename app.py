# app.py
import gradio as gr
from fanpower_tracker import collect

def run(query):
    artists = [x.strip() for x in query.split(",") if x.strip()]
    if not artists:
        return "쉼표(,)로 구분하여 최소 1개 입력하세요."
    df = collect(artists)
    return gr.Dataframe(df, visible=True)

with gr.Blocks(title="Artist 팬덤 수집기") as demo:
    gr.Markdown("아티스트/그룹명을 입력하고 버튼을 눌러보세요")
    inp = gr.Textbox(label="예) BTS, BLACKPINK, NewJeans")
    btn = gr.Button("조회하기")
    out = gr.Dataframe(visible=False)
    btn.click(fn=run, inputs=inp, outputs=out)

demo.launch(share=True)  # ← 실행 시 자동으로 퍼블릭 URL 출력
