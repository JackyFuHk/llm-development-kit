'''
Rate-limiting the OpenAI API

1. Vendor-enforced caps  
   • Both OpenAI and Azure OpenAI set hard per-key, per-deployment limits on RPM, TPM (tokens per minute) and QPS.  
   • Exceeding any limit instantly triggers `429 Too Many Requests` or `503 Rate Limited`; retries are not guaranteed to succeed.  
   • Proactively rate-limiting in your code reduces the odds of 429/503 and raises overall success rates.

2. Cost control  
   • GPT is billed per token; the higher the concurrency, the faster you can burn through your quota or monthly budget.  
   • A rate limiter paired with a usage meter lets you hit the brakes before the budget is gone.

3. Downstream stability  
   • Your own gateway, database and logging stack can collapse under a sudden traffic spike.  
   • Rate limiting shaves off the peaks and fills the valleys, preventing cascading failures.

4. User experience & compliance  
   • In multi-tenant SaaS you must allocate fair quotas per user or project.  
   • Certain sectors (finance, healthcare) require “predictable traffic”; rate limiting is often a contractual compliance item.
'''


import time
import base64
import json
import os
import time
from typing import NamedTuple

import openai
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_unless_exception_type

from PIL import Image
import base64
import io

class Config:
    AZURE_GPT_GENAI_PROXY = os.environ.get("AZURE_GPT_GENAI_PROXY", "https://genai.azure-api.net")
    AZURE_OPENAI_TOKEN = os.environ.get("AZURE_OPENAI_TOKEN")
    OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", 100))
    OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", 0.9))
    OPENAI_RATE_LIMIT = int(os.environ.get("OPENAI_RATE_LIMIT", 10))
    OPENAI_RATE_LIMIT_SLEEP = float(os.environ.get("OPENAI_RATE_LIMIT_SLEEP", 1.1))
    OPENAI_REQUESTS_BATCH_SIZE = int(os.environ.get("OPENAI_REQUESTS_BATCH_SIZE", 20))

class ManagementPolicyException(Exception):
    def __init__(self,
                 message="Sensitive terms detected. Please adjust your input. You may try refining your query, for example by adding clarifying conditions or constraints."):
        super().__init__(message)

class RateLimiter:
    """Rate control class, each call goes through wait_if_needed, sleep if rate control is needed"""

    def __init__(self, rpm):
        self.last_call_time = 0
        self.interval = 1.1 * 60 / rpm  # Here 1.1 is used because even if the calls are made strictly according to time, they will still be QOS'd; consider switching to simple error retry later
        self.rpm = rpm

    def split_batches(self, batch):
        return [batch[i:i + self.rpm] for i in range(0, len(batch), self.rpm)]

    async def wait_if_needed(self, num_requests):
        current_time = time.time()
        elapsed_time = current_time - self.last_call_time

        if elapsed_time < self.interval * num_requests:
            remaining_time = self.interval * num_requests - elapsed_time
            print(f"sleep {remaining_time}")
            time.sleep(remaining_time)

        self.last_call_time = time.time()

class CHATPGAPI(RateLimiter):
    """
    Check https://platform.openai.com/examples for examples
    """
    def __init__(self, model="GPT-41-2025-04-14", prompt_save_path=None, max_tokens=None):
        # prepare credential
        self.GENAI_PROXY = Config.AZURE_GPT_GENAI_PROXY
        self.CONGNITIVE_SERVICES = "https://cognitiveservices.azure.com/.default"
        self.EMBEDDING_API_VERSION = "2024-02-01"
        self.HEADERS = None

        self.llm = openai
        self.model = model
        self.rpm = 10
        self.max_tokens = max_tokens if max_tokens else Config.OPENAI_MAX_TOKENS
        RateLimiter.__init__(self, rpm=self.rpm)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.2, min=0.3, max=5.0))
    def function_calling(self, messages: list[dict], functions: list[dict], user_tag={}, agent='other') -> dict:
        response = self._chat_function_calling(messages, functions, user_tag, agent)
        content = response.content
        additional_kwargs = response.additional_kwargs
        name = additional_kwargs["function_call"]["name"]
        arguments = additional_kwargs["function_call"]["arguments"]
        return content, name, arguments

    def _chat_function_calling(self, messages: list[dict], functions: list[dict], user_tag={}, agent='other') -> dict:
        start_time = time.time()
        llm = AzureChatOpenAI(
            azure_endpoint=self.GENAI_PROXY, 
            azure_deployment=self.model,
            api_version=self.api_version_config.get(self.model, "2024-02-01"),
            api_key=Config.AZURE_OPENAI_TOKEN,
            temperature=self.get_temperature(self.model),
            default_headers=self.HEADERS,
        )
        rsp = llm.predict_messages(self.form_message(messages), functions=functions)
        print(f"{self.model} function_calling consume time: {time.time() - start_time}")
        print(f"{self.model} function_calling response: {str(rsp)}")
        return rsp

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.2, min=0.1, max=5.0),
           retry=retry_if_exception_type(Exception) & retry_unless_exception_type(ManagementPolicyException))
    def ask_stream(self, messages: list[dict], conversation_id=None, stop=[]):
        rep = self._chat_completion(messages)
        result = self.get_choice_text(rep)
        return result

    def _chat_completion(self, messages: list[dict]) -> dict:
        try:
            start_time = time.time()
            llm = AzureChatOpenAI(
                azure_endpoint=self.GENAI_PROXY,
                azure_deployment=self.model,
                api_version=self.api_version_config.get(self.model, "2024-02-01"),
                api_key=Config.AZURE_OPENAI_TOKEN,
                temperature=self.get_temperature(self.model),
                default_headers=self.HEADERS,
                max_tokens=self.max_tokens
            )
            rsp = llm.generate([self.form_message(messages)])
            print(f"{self.model} chat consume time: {time.time() - start_time}")
        except openai.BadRequestError as err:
            print(f"{self.model} chat error:{str(err)}")
            if err.status_code == 400:
                raise ManagementPolicyException()
        except Exception as err:
            print(f"{self.model} chat error:{str(err)}")
            self.refresh()
            raise err
        return rsp

    def form_message(self, message):
        formed_msg = []
        for m in message:
            if m['role'] == "system":
                formed_msg.append(SystemMessage(content=m['content']))
            elif m['role'] == "assistant":
                formed_msg.append(AIMessage(content=m['content']))
            elif m['role'] == "user":
                formed_msg.append(HumanMessage(content=m['content']))
            else:
                formed_msg.append(HumanMessage(content=m['content']))
        return formed_msg

    def get_choice_text(self, rsp) -> str:
        """Required to provide the first text of choice"""
        return rsp.generations[0][0].text

    @retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1.2, min=0.1, max=5.0))
    def get_embedding(self, message, user_tag={}, agent='other'):
        try:
            start_time = time.time()
            llm = AzureOpenAIEmbeddings(
                azure_endpoint=self.GENAI_PROXY,
                deployment="TEXT-EMBEDDING-ADA-002",
                api_version=self.EMBEDDING_API_VERSION,
                api_key=Config.AZURE_OPENAI_TOKEN,
                default_headers=self.HEADERS
            )
            result = llm.embed_query(message)
            print(f"{self.model} embedding consume time: {time.time() - start_time}")
            if result is None:
                raise Exception("get_embedding result is None")
            return result
        except Exception as err:
            print(f"{self.model} embedding error:{str(err)}")
            self.refresh()
            raise err