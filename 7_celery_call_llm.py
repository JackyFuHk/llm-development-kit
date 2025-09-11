'''
    call the chat openai api, and get the response

    how to call the task? 
    you can use the following code:

        from fastapi import FastAPI
        from tasks import call_openai

        app = FastAPI()

        @app.post("/ask")
        def ask(prompt: str):
            task = call_openai.delay(prompt)
            return {"task_id": task.id}

'''

import openai
from celery import Celery
import os
import requests
celery_app = Celery(
    "openai_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

'''
increased the task’s time limits: the soft timeout is now set to 1,200 seconds and the hard timeout to 2,400 seconds. 
This way, even if the task runs longer than the default limit, the worker process won’t receive a warm shutdown.
'''
celery_app.conf.update(
    task_soft_time_limit=1200,   
    task_time_limit=2400,       
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


openai.api_key = os.getenv("OPENAI_API_KEY")

def fake_callback(response_text: str):
    response = requests.post(
        "http://127.0.0.1:6999/chat_callback/",
        json={"task_id": "fake_task_id", "response_text": response_text},
        timeout=30
    )


@celery_app.task(
    name="call_openai",
    soft_time_limit=1200,
    time_limit=2400,
)
def call_openai(prompt: str) -> str:
    """
    调用 OpenAI ChatCompletion 并把结果通过回调回传
    """
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    answer = resp.choices[0].message.content.strip()

    # 回调
    fake_callback(answer)

    return answer