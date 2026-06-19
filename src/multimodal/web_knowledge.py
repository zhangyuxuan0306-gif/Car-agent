"""联网建筑知识补全 — 国内优先百度，维基/DDG 备用"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from html import unescape
from typing import Optional

logger = logging.getLogger(__name__)

_UA_MOBILE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
)
_HEADERS_JSON = {
    "User-Agent": _UA_MOBILE,
    "Accept": "application/json",
}
_HEADERS_HTML = {
    "User-Agent": _UA_MOBILE,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

_WIKI_EN = {
    "东方明珠": "Oriental_Pearl_Tower",
    "上海中心大厦": "Shanghai_Tower",
    "上海环球金融中心": "Shanghai_World_Financial_Center",
    "金茂大厦": "Jin_Mao_Tower",
    "陆家嘴标志性建筑群": "Lujiazui",
    "中国尊": "China_Zun",
    "中央电视台总部大楼": "CCTV_Headquarters",
    "国贸三期A座": "China_World_Trade_Center_Tower_3",
    "国贸三期B座": "China_World_Trade_Center_Tower_3",
    "三星大厦": "Samsung_China_Headquarters",
}


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class WebKnowledgeSearcher:
    """建筑介绍联网检索"""

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self._cache: dict[str, str] = {}

    def search(self, building_name: str) -> str:
        if not building_name or building_name in ("建筑", "天空", "河流", "摩天大楼"):
            return ""
        if building_name in self._cache:
            return self._cache[building_name]

        for fn in [self._baidu_search, self._wiki_summary, self._wiki_en_summary, self._duckduckgo]:
            text = fn(building_name)
            if text and len(text) > 6:
                self._cache[building_name] = text
                logger.info("联网检索成功 [%s]: %s", fn.__name__, building_name)
                return text

        logger.warning("联网检索无结果: %s", building_name)
        return ""

    def search_answer(self, building_name: str, question: str) -> str:
        key = f"{building_name}|{question}"
        if key in self._cache:
            return self._cache[key]

        queries = [
            f"{building_name} {question}",
            f"{building_name} {question.rstrip('？?')}",
            f"{building_name} 简介",
            building_name,
        ]
        for q in queries:
            text = self._baidu_search(q)
            if not text and building_name:
                for fn in [self._wiki_summary, self._wiki_en_summary]:
                    text = fn(building_name if fn == self._wiki_en_summary else q)
                    if text:
                        break
            if not text:
                text = self._duckduckgo(q)
            if text and len(text) > 6:
                self._cache[key] = text
                logger.info("联网问答成功: %s / %s", building_name, question)
                return text

        logger.warning("联网问答无结果: %s / %s", building_name, question)
        return ""

    def _get_bytes(self, url: str, headers: dict) -> Optional[bytes]:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except Exception as e:
            logger.debug("请求失败 %s: %s", url, e)
            return None

    def _get_json(self, url: str) -> Optional[dict]:
        raw = self._get_bytes(url, _HEADERS_JSON)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _get_html(self, url: str) -> str:
        raw = self._get_bytes(url, _HEADERS_HTML)
        if not raw:
            return ""
        for enc in ("utf-8", "gb18030", "gbk"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    def _baidu_search(self, query: str) -> str:
        """百度移动端搜索 — 国内环境可用"""
        if not query.strip():
            return ""
        url = f"https://m.baidu.com/s?word={urllib.parse.quote(query.strip())}"
        html = self._get_html(url)
        if not html or len(html) < 500:
            return ""

        # AI 摘要 / marklang 段落
        for pat in (
            r'<p class="marklang-paragraph">(.*?)</p>',
            r'class="marklang-paragraph"[^>]*>(.*?)<',
            r'<!--s-text-->(.*?)<!--/s-text-->',
        ):
            m = re.search(pat, html, re.S | re.I)
            if m:
                text = _strip_html(m.group(1))
                if len(text) > 10:
                    return text[:400]

        # ml-text 片段拼接
        parts = re.findall(r'<ml-text[^>]*>([^<]+)</ml-text>', html)
        if parts:
            text = "".join(parts[:8])
            text = re.sub(r"\s+", "", text)
            if len(text) > 8:
                return text[:400]

        # 普通摘要
        for pat in (
            r'<div class="c-font-normal[^"]*"[^>]*>(.*?)</div>',
            r'class="cos-line-clamp[^"]*"[^>]*>(.*?)</',
        ):
            m = re.search(pat, html, re.S | re.I)
            if m:
                text = _strip_html(m.group(1))
                if len(text) > 15:
                    return text[:400]
        return ""

    def _wiki_summary(self, name: str) -> str:
        title = urllib.parse.quote(name.replace(" ", "_"))
        url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{title}"
        data = self._get_json(url)
        if data and data.get("extract"):
            return data["extract"][:500]
        for suffix in ["塔", "大厦", " (北京)"]:
            url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(name + suffix)}"
            data = self._get_json(url)
            if data and data.get("extract"):
                return data["extract"][:500]
        return ""

    def _wiki_en_summary(self, name: str) -> str:
        title = _WIKI_EN.get(name, name.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        data = self._get_json(url)
        if data and data.get("extract"):
            return data["extract"][:500]
        return ""

    def _duckduckgo(self, query: str) -> str:
        try:
            q = urllib.parse.quote(f"{query} 建筑 简介")
            url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1"
            data = self._get_json(url)
            if data:
                if data.get("AbstractText"):
                    return data["AbstractText"][:500]
                for topic in data.get("RelatedTopics", [])[:5]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        return topic["Text"][:500]
        except Exception:
            pass
        return ""
