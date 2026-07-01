"""联网建筑知识补全 — 桌面百度 + Bing/Sogou 备用"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from html import unescape
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_UA_DESKTOP = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_UA_MOBILE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
)
_HEADERS_JSON = {
    "User-Agent": _UA_DESKTOP,
    "Accept": "application/json",
}
_HEADERS_HTML = {
    "User-Agent": _UA_DESKTOP,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.baidu.com/",
}
_READ_LIMIT = 1_200_000

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
    "中央公园广场": "Central_Park_Plaza_Beijing",
    "世界华商中心": "World_Chinese_Merchants_Center",
}

_PRONOUN_RE = re.compile(r"^(他|她|它|这|那|这个|那个|该|此)(的|是|有)?")


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_question(building_name: str, question: str) -> str:
    q = question.strip().rstrip("？?")
    q = _PRONOUN_RE.sub("", q).strip()
    if not q:
        return building_name
    if building_name and building_name not in q:
        return f"{building_name} {q}"
    return q


class WebKnowledgeSearcher:
    """建筑介绍联网检索 — 多源并行，命中即返"""

    def __init__(
        self,
        timeout: float = 8.0,
        connect_timeout: float = 3.0,
        read_timeout: float = 10.0,
        enable_duckduckgo: bool = False,
        enable_bing: bool = True,
        enable_sogou: bool = True,
    ):
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.enable_duckduckgo = enable_duckduckgo
        self.enable_bing = enable_bing
        self.enable_sogou = enable_sogou
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="web_kb")
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookie_jar)
        )
        self._baidu_warmed = False

    def search(self, building_name: str) -> str:
        if not building_name or building_name in ("建筑", "天空", "河流", "摩天大楼"):
            return ""
        with self._lock:
            if building_name in self._cache:
                return self._cache[building_name]

        text = self._fetch_parallel(building_name, building_name, question_mode=False)
        if text:
            self._store(building_name, text)
            return text

        logger.warning("联网检索无结果: %s", building_name)
        return ""

    def search_answer(self, building_name: str, question: str) -> str:
        key = f"{building_name}|{question}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        normalized = _normalize_question(building_name, question)
        queries: List[str] = []
        for q in (normalized, f"{building_name} {question.rstrip('？?')}", building_name):
            q = q.strip()
            if q and q not in queries:
                queries.append(q)

        for q in queries:
            text = self._fetch_parallel(q, building_name, question_mode=True)
            if text and len(text) > 6:
                self._store(key, text)
                logger.info("联网问答成功: %s / %s", building_name, question)
                return text

        logger.warning("联网问答无结果: %s / %s", building_name, question)
        return ""

    def search_poi(
        self,
        building_name: str,
        question: str,
        location: str = "",
        keywords: tuple[str, ...] = (),
    ) -> str:
        from src.multimodal.question_router import build_poi_queries

        key = f"poi|{building_name}|{question}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        queries = build_poi_queries(building_name, question, location)
        kw = keywords or tuple(k for k in ("咖啡", "餐厅", "停车", "地铁") if k in question)

        for q in queries:
            text = self._fetch_poi(q, kw)
            if text and len(text) > 10:
                self._store(key, text)
                logger.info("POI 联网成功: %s / %s", building_name, question[:30])
                return text

        logger.warning("POI 联网无结果: %s / %s", building_name, question)
        return ""

    def _fetch_poi(self, query: str, keywords: tuple[str, ...]) -> str:
        self._ensure_baidu_warm()
        futures: dict = {
            self._pool.submit(self._baidu_search, query, keywords): "baidu",
        }
        if self.enable_bing:
            futures[self._pool.submit(self._bing_search, query, keywords)] = "bing"
        if self.enable_sogou:
            futures[self._pool.submit(self._sogou_search, query, keywords)] = "sogou"

        deadline = time.monotonic() + max(self.timeout + 3.0, self.read_timeout + 1.0)
        pending = set(futures.keys())

        while pending and time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            done, pending = wait(
                pending,
                timeout=min(0.4, remaining),
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                try:
                    text = fut.result()
                    if text and len(text) > 10:
                        for other in pending:
                            other.cancel()
                        return text
                except Exception:
                    pass
        return ""

    def _ensure_baidu_warm(self) -> None:
        if self._baidu_warmed:
            return
        try:
            self._get_html("https://www.baidu.com/")
        except Exception:
            pass
        self._baidu_warmed = True

    def _store(self, key: str, value: str) -> None:
        with self._lock:
            self._cache[key] = value

    def _fetch_parallel(
        self, query: str, building_name: str, *, question_mode: bool
    ) -> str:
        self._ensure_baidu_warm()
        providers: List[tuple[str, Callable[[], str]]] = [
            ("baidu", lambda: self._baidu_search(query)),
            ("wiki_zh", lambda: self._wiki_summary(building_name or query)),
            ("wiki_en", lambda: self._wiki_en_summary(building_name or query)),
        ]
        if self.enable_duckduckgo:
            providers.append(("ddg", lambda: self._duckduckgo(query)))

        overall = self.timeout + (1.5 if question_mode else 0.0)
        futures: dict[Future[str], str] = {
            self._pool.submit(fn): name for name, fn in providers
        }
        deadline = time.monotonic() + overall
        pending = set(futures.keys())

        while pending and time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            done, pending = wait(
                pending,
                timeout=min(0.3, remaining),
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                name = futures[fut]
                try:
                    text = fut.result()
                    if text and len(text) > 6:
                        logger.info("联网检索成功 [%s]: %s", name, query[:48])
                        for other in pending:
                            other.cancel()
                        return text
                except Exception as e:
                    logger.debug("provider %s failed: %s", name, e)

        for fut in pending:
            fut.cancel()
        return ""

    def _get_bytes(self, url: str, headers: dict) -> Optional[bytes]:
        try:
            req = urllib.request.Request(url, headers=headers)
            # 单 float 超时：部分 Python/urllib 不支持 (connect, read) 元组
            with self._opener.open(req, timeout=self.read_timeout) as resp:
                return resp.read(_READ_LIMIT)
        except Exception as e:
            logger.debug("请求失败 %s: %s", url[:80], e)
            return None

    def _get_json(self, url: str) -> Optional[dict]:
        raw = self._get_bytes(url, _HEADERS_JSON)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _get_html(self, url: str, mobile: bool = False) -> str:
        headers = dict(_HEADERS_HTML)
        if mobile:
            headers["User-Agent"] = _UA_MOBILE
        raw = self._get_bytes(url, headers)
        if not raw:
            return ""
        for enc in ("utf-8", "gb18030", "gbk"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    def _is_captcha_page(self, html: str) -> bool:
        if not html or len(html) < 800:
            return True
        markers = ("百度安全验证", "wappass.baidu.com", "网络不给力", "请稍后重试")
        return any(m in html for m in markers)

    def _parse_baidu_html(self, html: str, keywords: tuple[str, ...] = ()) -> str:
        if self._is_captcha_page(html):
            return ""

        blocks = re.findall(r"<!--s-text-->(.*?)<!--/s-text-->", html, re.S)
        if keywords and blocks:
            scored: List[tuple[int, str]] = []
            for block in blocks:
                text = _strip_html(block)
                if len(text) < 10:
                    continue
                score = sum(2 for k in keywords if k in text)
                if score > 0:
                    scored.append((score, text))
            if scored:
                scored.sort(key=lambda x: -x[0])
                return "\n".join(t[:220] for _, t in scored[:4])[:500]

        for block in blocks:
            text = _strip_html(block)
            if len(text) > 12:
                if not keywords or any(k in text for k in keywords):
                    return text[:450]

        for pat in (
            r'"contentText"\s*:\s*"([^"]{12,})"',
            r'<p class="marklang-paragraph">(.*?)</p>',
            r'class="marklang-paragraph"[^>]*>(.*?)<',
            r'<div class="c-abstract[^"]*"[^>]*>(.*?)</div>',
            r'class="cos-line-clamp[^"]*"[^>]*>(.*?)</',
        ):
            m = re.search(pat, html, re.S | re.I)
            if m:
                text = _strip_html(m.group(1))
                if len(text) > 12:
                    return text[:450]
        return ""

    def _baidu_search(self, query: str, keywords: tuple[str, ...] = ()) -> str:
        if not query.strip():
            return ""
        q = urllib.parse.quote(query.strip())

        desktop_url = f"https://www.baidu.com/s?wd={q}&rn=10"
        html = self._get_html(desktop_url, mobile=False)
        text = self._parse_baidu_html(html, keywords=keywords)
        if text:
            return text

        if self._is_captcha_page(html):
            logger.info("百度触发验证，尝试备用搜索引擎: %s", query[:36])

        mobile_url = f"https://m.baidu.com/s?word={q}"
        text = self._parse_baidu_html(
            self._get_html(mobile_url, mobile=True), keywords=keywords
        )
        return text

    def _bing_search(self, query: str, keywords: tuple[str, ...] = ()) -> str:
        if not query.strip():
            return ""
        url = f"https://cn.bing.com/search?q={urllib.parse.quote(query.strip())}&setlang=zh-Hans"
        html = self._get_html(url, mobile=False)
        if not html or len(html) < 1000:
            return ""

        section = html.split('id="b_results"')[-1] if 'id="b_results"' in html else html
        results: List[tuple[int, str]] = []
        for block in re.finditer(
            r'<li class="b_algo"[^>]*>(.*?)(?=<li class="b_algo"|$)',
            section[:90000],
            re.S,
        ):
            chunk = block.group(1)
            title_m = re.search(r"<h2[^>]*>\s*<a[^>]*>(.*?)</a>", chunk, re.S)
            if not title_m:
                continue
            title = _strip_html(title_m.group(1))
            desc = ""
            for pat in (
                r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>',
                r"<p[^>]*>(.*?)</p>",
            ):
                dm = re.search(pat, chunk, re.S)
                if dm:
                    desc = _strip_html(dm.group(1))
                    if len(desc) > 12:
                        break
            line = f"{title}：{desc}" if desc else title
            if keywords:
                score = sum(2 for k in keywords if k in title)
                score += sum(1 for k in keywords if k in desc)
                if score <= 0:
                    continue
            else:
                score = 1
            results.append((score, line[:240]))

        if results:
            results.sort(key=lambda x: -x[0])
            return "\n".join(t for _, t in results[:4])[:500]
        return ""

    def _sogou_search(self, query: str, keywords: tuple[str, ...] = ()) -> str:
        if not query.strip():
            return ""
        url = f"https://www.sogou.com/web?query={urllib.parse.quote(query.strip())}"
        html = self._get_html(url, mobile=False)
        if not html or len(html) < 5000:
            return ""

        results: List[tuple[int, str]] = []
        for block in re.finditer(
            r'<div class="vrwrap[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html[:600000],
            re.S,
        ):
            chunk = block.group(1)
            title_m = re.search(r"<h3[^>]*>(.*?)</h3>", chunk, re.S)
            if not title_m:
                continue
            title = _strip_html(title_m.group(1))
            desc_m = re.search(r'class="(?:str-text|star-wenan)[^"]*"[^>]*>(.*?)</', chunk, re.S)
            desc = _strip_html(desc_m.group(1)) if desc_m else ""
            line = f"{title}：{desc}" if desc else title
            if len(line) < 8:
                continue
            if keywords:
                score = sum(1 for k in keywords if k in line)
                if score <= 0:
                    continue
            else:
                score = 1
            results.append((score, line[:220]))

        if not results:
            for m in re.finditer(r"<h3[^>]*><a[^>]*>(.*?)</a></h3>", html[:400000], re.S):
                title = _strip_html(m.group(1))
                if keywords and not any(k in title for k in keywords):
                    continue
                results.append((1, title[:200]))

        if results:
            results.sort(key=lambda x: -x[0])
            return "\n".join(t for _, t in results[:4])[:500]
        return ""

    def _wiki_summary(self, name: str) -> str:
        if not name.strip():
            return ""
        candidates = [name.replace(" ", "_")]
        for suffix in (" (北京)", "塔", "大厦"):
            candidates.append(f"{name}{suffix}".replace(" ", "_"))
        seen = set()
        for title in candidates:
            if title in seen:
                continue
            seen.add(title)
            url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            data = self._get_json(url)
            if data and data.get("extract"):
                return data["extract"][:500]
        return ""

    def _wiki_en_summary(self, name: str) -> str:
        if not name.strip():
            return ""
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


def web_searcher_from_config(kb_cfg: dict) -> Optional[WebKnowledgeSearcher]:
    if not kb_cfg.get("web_search", False):
        return None
    return WebKnowledgeSearcher(
        timeout=float(kb_cfg.get("web_timeout", 8.0)),
        connect_timeout=float(kb_cfg.get("web_connect_timeout", 3.0)),
        read_timeout=float(kb_cfg.get("web_read_timeout", 10.0)),
        enable_duckduckgo=bool(kb_cfg.get("web_enable_ddg", False)),
        enable_bing=bool(kb_cfg.get("web_enable_bing", True)),
        enable_sogou=bool(kb_cfg.get("web_enable_sogou", True)),
    )
