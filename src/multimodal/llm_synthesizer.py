"""大模型问答整理 — 基于本地/联网资料生成有针对性的回答"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是智能座舱建筑讲解员。根据参考资料回答乘客关于窗外建筑的问题。\n"
    "要求：紧扣问题、突出关键信息、语言简洁专业；2-4句话；"
    "仅使用参考资料中的事实，不要编造；若资料不足请如实说明。"
)


class LLMAnswerSynthesizer:
    """用小参数量指令模型整理问答（懒加载 + 线程锁）"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def load(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("加载问答大模型: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info("问答大模型就绪")

    def preload_async(self):
        threading.Thread(target=self.load, daemon=True, name="llm-preload").start()

    def synthesize(self, building: str, question: str, context: str) -> str:
        if not context.strip():
            return f"暂未找到关于「{building}」的足够资料，无法回答「{question}」。"

        try:
            self.load()
        except Exception as e:
            logger.warning("大模型加载失败，使用规则整理: %s", e)
            return self._rule_fallback(building, question, context)

        user = (
            f"建筑：{building}\n"
            f"乘客问题：{question}\n\n"
            f"参考资料：\n{context.strip()[:1200]}\n\n"
            f"请针对乘客问题作答："
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            with self._gen_lock:
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._tokenizer([text], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=self.temperature > 0,
                        top_p=0.9,
                        repetition_penalty=1.05,
                    )
                gen = out[0][inputs["input_ids"].shape[1] :]
                answer = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
                answer = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", " ", answer).strip()
                if answer:
                    return answer
        except Exception as e:
            logger.warning("大模型生成失败: %s", e)

        return self._rule_fallback(building, question, context)

    @staticmethod
    def _rule_fallback(building: str, question: str, context: str) -> str:
        ctx = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", " ", context).strip()
        sents = re.split(r"(?<=[。！？!?])\s*", ctx)
        sents = [s.strip() for s in sents if len(s.strip()) > 4]
        if not sents:
            return ctx[:500] if ctx else f"暂未找到关于「{building}」的资料。"

        q_kws = [w for w in re.split(r"[，。；？?！!、\s]+", question) if len(w) >= 2]
        scored = []
        for s in sents:
            score = sum(1 for k in q_kws if k in s)
            if building and building in s:
                score += 2
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [s for sc, s in scored if sc > 0][:3]
        if not picked:
            picked = sents[:3]
        return " ".join(picked)
