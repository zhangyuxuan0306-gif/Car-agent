"""Writer Agent — 车载导游文案润色智能体"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from src.agents.dashboard import AgentDashboard
from src.multimodal.llm_synthesizer import LLMAnswerSynthesizer

logger = logging.getLogger(__name__)

_WRITER_SYSTEM = (
    "你是智能座舱导游文案编剧。根据建筑知识资料，写一段温和亲切的车载语音播报词。\n"
    "要求：2-4句话，自然口语化，串联多个地标；仅使用资料中的事实，不编造；"
    "不要出现夸张广告词（如「必吃」「全网最低」）。"
)


class WriterAgent:
    """将 RAG 检索到的零散资料转换为车载导游播报词"""

    def __init__(
        self,
        dashboard: AgentDashboard,
        synthesizer: Optional[LLMAnswerSynthesizer] = None,
        use_llm: bool = False,
    ):
        self.dashboard = dashboard
        self.synthesizer = synthesizer
        self.use_llm = use_llm and synthesizer is not None

    def write(
        self,
        buildings: List[str],
        rag_context: Dict[str, str],
        critic_feedback: str = "",
    ) -> str:
        self.dashboard.log("Handover", "Writer_Agent 正在生成车载导游播报词...")

        if critic_feedback:
            self.dashboard.log("Handover", f"Writer_Agent 收到质检反馈，重写中: {critic_feedback[:60]}")

        if self.use_llm and self.synthesizer:
            return self._write_with_llm(buildings, rag_context, critic_feedback)
        return self._write_template(buildings, rag_context, critic_feedback)

    def _write_template(
        self,
        buildings: List[str],
        rag_context: Dict[str, str],
        critic_feedback: str,
    ) -> str:
        if not buildings:
            return "乘客您好，当前窗外暂未识别到明确地标，请稍后再试。"

        parts = ["乘客您好，"]
        for i, name in enumerate(buildings):
            ctx = rag_context.get(name, "")
            brief = self._extract_brief(ctx) or f"{name}是北京CBD区域的知名地标"
            brief = brief.rstrip("。！？")
            if i == 0:
                parts.append(f"您右手边正在经过的是{name}。{brief}。")
            else:
                parts.append(f"不远处还有{name}，{brief}。")

        script = "".join(parts)
        if critic_feedback and "地址" in critic_feedback:
            script = re.sub(r"位于[^，。]+", "位于北京CBD", script)
        return script

    def _write_with_llm(
        self,
        buildings: List[str],
        rag_context: Dict[str, str],
        critic_feedback: str,
    ) -> str:
        combined = "\n\n".join(
            f"【{name}】\n{rag_context.get(name, '')}" for name in buildings
        )
        question = f"请为以下建筑生成一段串联的车载导游播报词：{'、'.join(buildings)}"
        if critic_feedback:
            question += f"\n\n质检反馈（请修正）：{critic_feedback}"

        try:
            self.synthesizer.load()
            user = f"{question}\n\n参考资料：\n{combined[:1500]}"
            messages = [
                {"role": "system", "content": _WRITER_SYSTEM},
                {"role": "user", "content": user},
            ]
            import torch
            with self.synthesizer._gen_lock:
                text = self.synthesizer._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.synthesizer._tokenizer([text], return_tensors="pt").to(
                    self.synthesizer.device
                )
                with torch.no_grad():
                    out = self.synthesizer._model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.4,
                        do_sample=True,
                        top_p=0.9,
                    )
                gen = out[0][inputs["input_ids"].shape[1]:]
                answer = self.synthesizer._tokenizer.decode(gen, skip_special_tokens=True).strip()
                if answer:
                    return answer
        except Exception as e:
            logger.warning("Writer LLM 失败，回退模板: %s", e)

        return self._write_template(buildings, rag_context, critic_feedback)

    @staticmethod
    def _extract_brief(ctx: str) -> str:
        if not ctx:
            return ""
        if "简介:" in ctx:
            return ctx.split("简介:")[-1].strip().split("\n")[0]
        for line in ctx.split("\n"):
            if line.startswith("description:") or "描述" in line:
                return line.split(":", 1)[-1].strip()
        return ctx.replace("\n", " ")[:80]
