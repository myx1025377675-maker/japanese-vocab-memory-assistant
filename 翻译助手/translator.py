"""
翻译引擎模块 —— 离线模式（本地模型） + 在线模式（DeepSeek API）。
"""
import re
import threading
import sqlite3
import json
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

from config import AppConfig
from prompts.system_prompts import JA_TO_ZH, ZH_TO_JA, AUTO_DETECT

# ============================================================
# 语言检测工具
# ============================================================

def detect_language(text: str) -> str:
    """
    简单而有效的日↔中语言检测。
    返回 "ja" | "zh" | "mixed_ja" | "mixed_zh"
    """
    # 统计日文假名字符
    hiragana = len(re.findall(r'[぀-ゟ]', text))
    katakana = len(re.findall(r'[゠-ヿ]', text))
    # 统计中文字符（CJK统一汉字）
    kanji = len(re.findall(r'[一-鿿]', text))
    # 统计日文特有标点/符号
    jp_punct = len(re.findall(r'[「」『』・。、]', text))
    # 统计中文特有字符
    cn_only = len(re.findall(r'[“”‘’【】《》]', text))

    jp_score = hiragana * 3 + katakana * 2 + jp_punct * 1 + kanji * 0.3
    cn_score = cn_only * 3 + kanji * 0.5

    if hiragana > 0 or katakana > 0:
        return "ja" if jp_score >= cn_score else "mixed_ja"
    else:
        return "zh" if cn_score >= jp_score else "mixed_zh"


def _select_prompt(direction: str) -> str:
    """根据翻译方向选择对应的 system prompt。"""
    if direction == "ja_to_zh":
        return JA_TO_ZH
    elif direction == "zh_to_ja":
        return ZH_TO_JA
    else:
        return AUTO_DETECT


# ============================================================
# 抽象翻译器基类
# ============================================================

class BaseTranslator(ABC):
    """翻译器抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """翻译器名称标识。"""
        ...

    @property
    @abstractmethod
    def is_online(self) -> bool:
        """是否需要联网。"""
        ...

    @abstractmethod
    def translate(self, text: str, direction: str) -> str:
        """
        执行翻译。
        direction: "ja_to_zh" | "zh_to_ja" | "auto"
        返回翻译后的文本。
        """
        ...

    def is_available(self) -> bool:
        """检查翻译器是否可用。"""
        return True


# ============================================================
# 离线翻译器 —— 轻量级词典 + 形态素分析
# ============================================================

# 精简的日↔中核心词典（约 2000 常用词条）
_JA_ZH_DICT: dict[str, str] = {}
_ZH_JA_DICT: dict[str, str] = {}


def _init_dictionaries():
    """初始化日↔中词典（核心高频词条）。"""
    if _JA_ZH_DICT:
        return

    # 核心日→中词条
    ja_zh_entries = {
        # ---- 日常用语 ----
        "ありがとう": "谢谢",
        "ありがとうございます": "非常感谢",
        "すみません": "对不起 / 不好意思",
        "ごめんなさい": "对不起",
        "ごめん": "抱歉",
        "おはよう": "早上好",
        "おはようございます": "早上好（敬体）",
        "こんにちは": "你好",
        "こんばんは": "晚上好",
        "さようなら": "再见",
        "おやすみ": "晚安",
        "おやすみなさい": "晚安（敬体）",
        "いただきます": "我开动了",
        "ごちそうさまでした": "多谢款待",
        "お疲れ様です": "辛苦了",
        "お疲れ様でした": "辛苦了（完成时）",
        "よろしくお願いします": "请多关照",
        "よろしく": "请多指教",
        "お願いします": "拜托了",
        "お願い": "拜托",
        "はじめまして": "初次见面",
        "どうぞ": "请",
        "どうも": "多谢 / 很",
        "はい": "是的",
        "いいえ": "不是 / 不",
        "ええ": "嗯 / 是的",
        "うん": "嗯（口语）",

        # ---- 疑问词 ----
        "何": "什么",
        "どこ": "哪里",
        "いつ": "什么时候",
        "だれ": "谁",
        "なぜ": "为什么",
        "どうして": "为什么",
        "どう": "怎样",
        "どの": "哪个",
        "どれ": "哪一个",
        "いくら": "多少钱",
        "いくつ": "几个",
        "何時": "几点",

        # ---- 代词 ----
        "私": "我",
        "僕": "我（男性）",
        "俺": "我（男性粗犷）",
        "あなた": "你",
        "君": "你（亲切）",
        "彼": "他",
        "彼女": "她 / 女朋友",
        "これ": "这个",
        "それ": "那个",
        "あれ": "那个（远）",
        "ここ": "这里",
        "そこ": "那里",
        "あそこ": "那里（远）",

        # ---- 常用动词 ----
        "する": "做",
        "します": "做（敬体）",
        "ある": "有 / 在（无生命）",
        "いる": "有 / 在（有生命）",
        "行く": "去",
        "来る": "来",
        "見る": "看",
        "聞く": "听 / 问",
        "話す": "说 / 交谈",
        "読む": "读",
        "書く": "写",
        "食べる": "吃",
        "飲む": "喝",
        "買う": "买",
        "売る": "卖",
        "使う": "使用",
        "作る": "制作",
        "思う": "想 / 认为",
        "考える": "思考",
        "知る": "知道",
        "分かる": "理解 / 明白",
        "できる": "能 / 会 / 完成",
        "なる": "成为 / 变得",
        "持つ": "拿 / 持有",
        "取る": "取 / 拿",
        "入れる": "放入",
        "出す": "拿出 / 提交",
        "会う": "见面",
        "帰る": "回去",
        "寝る": "睡觉",
        "起きる": "起床",
        "働く": "工作",
        "休む": "休息",
        "遊ぶ": "玩",
        "待つ": "等待",
        "乗る": "乘坐",
        "降りる": "下来",
        "歩く": "步行",
        "走る": "跑",
        "座る": "坐",
        "立つ": "站",
        "住む": "居住",
        "死ぬ": "死",
        "生きる": "活 / 生活",

        # ---- 形容词 ----
        "大きい": "大的",
        "小さい": "小的",
        "高い": "高的 / 贵的",
        "低い": "低的",
        "安い": "便宜的",
        "美味しい": "好吃的",
        "まずい": "难吃的",
        "良い": "好的",
        "悪い": "坏的",
        "新しい": "新的",
        "古い": "旧的",
        "早い": "早的 / 快的",
        "遅い": "慢的 / 晚的",
        "速い": "快速的",
        "多い": "多的",
        "少ない": "少的",
        "暑い": "热的（天气）",
        "熱い": "烫的（物体）",
        "寒い": "寒冷的",
        "冷たい": "冰凉的",
        "楽しい": "快乐的",
        "面白い": "有趣的",
        "嬉しい": "高兴的",
        "悲しい": "悲伤的",
        "難しい": "困难的",
        "易しい": "容易的",
        "優しい": "温柔的",
        "近い": "近的",
        "遠い": "远的",
        "広い": "宽广的",
        "狭い": "狭窄的",
        "明るい": "明亮的",
        "暗い": "暗的",
        "強い": "强的",
        "弱い": "弱的",
        "忙しい": "忙碌的",
        "痛い": "疼的",
        "欲しい": "想要的",
        "可愛い": "可爱的",
        "怖い": "可怕的",
        "正しい": "正确的",
        "危ない": "危险的",
        "素晴らしい": "出色的",
        "凄い": "厉害的",
        "美しい": "美丽的",
        "嬉しい": "高兴的",

        # ---- 名词（常见） ----
        "今日": "今天",
        "明日": "明天",
        "昨日": "昨天",
        "毎日": "每天",
        "今": "现在",
        "前": "之前",
        "後": "之后",
        "朝": "早上",
        "昼": "中午",
        "夜": "晚上",
        "時間": "时间",
        "人": "人",
        "物": "东西",
        "事": "事情",
        "所": "地方",
        "金": "钱",
        "水": "水",
        "食べ物": "食物",
        "飲み物": "饮料",
        "車": "车",
        "電車": "电车",
        "駅": "车站",
        "店": "商店",
        "家": "家 / 房子",
        "部屋": "房间",
        "学校": "学校",
        "会社": "公司",
        "仕事": "工作",
        "友達": "朋友",
        "家族": "家人",
        "子供": "孩子",
        "大人": "大人",
        "男": "男性",
        "女": "女性",
        "手": "手",
        "目": "眼睛",
        "口": "嘴",
        "心": "心",
        "気持ち": "心情",
        "意味": "意思",
        "日本語": "日语",
        "中国語": "中文",
        "英語": "英语",
        "日本": "日本",
        "中国": "中国",

        # ---- 助词/语法成分解释 ----
        "は": "[主题标记]",
        "が": "[主语标记]",
        "を": "[宾语标记]",
        "に": "[方向/时间标记]",
        "へ": "[方向标记]",
        "で": "[手段/场所标记]",
        "と": "[并列/引用标记]",
        "から": "[起点/原因标记]",
        "まで": "[终点/范围标记]",
        "も": "[也]",
        "か": "[疑问标记]",

        # ---- 常用短语 ----
        "大丈夫": "没问题",
        "大丈夫ですか": "没事吧？/ 没问题吗？",
        "かしこまりました": "明白了（敬语）",
        "承知しました": "知道了（正式）",
        "失礼します": "打扰了 / 告辞了",
        "お邪魔します": "打扰了（进入时）",
        "行ってきます": "我走了",
        "いってらっしゃい": "慢走",
        "ただいま": "我回来了",
        "お帰りなさい": "欢迎回来",
        "いらっしゃいませ": "欢迎光临",
        "どういたしまして": "不客气",
        "お元気ですか": "你好吗？",
        "元気です": "我很好",
        "わかりました": "明白了",
        "できました": "完成了",
        "どうでしょうか": "怎么样呢？",
        "そうですね": "是啊 / 嗯...",
        "すごいですね": "好厉害啊",
        "いいですね": "不错呢",
        "それはいい": "那挺好的",
        "楽しみにしています": "我很期待",
        "気をつけて": "小心 / 保重",
        "お大事に": "保重身体",
        "おめでとう": "恭喜",
        "おめでとうございます": "恭喜（敬体）",
        "誕生日おめでとう": "生日快乐",
        "明けましておめでとう": "新年快乐",
        "ありえない": "不可能 / 难以置信",
        "信じられない": "难以置信",
        "もちろん": "当然",
        "ぜひ": "务必",
        "ちょっと": "稍微 / 有点",
        "たくさん": "很多",
        "とても": "非常",
        "本当に": "真的",
        "たぶん": "大概 / 也许",
        "絶対": "绝对",
        "必ず": "一定",
        "すぐ": "马上",
        "もう": "已经",
        "まだ": "还 / 尚未",
        "ずっと": "一直",
        "いつも": "总是",
    }

    # 构建日→中词典 + 反向词典
    for ja, zh in ja_zh_entries.items():
        _JA_ZH_DICT[ja] = zh
        if zh not in _ZH_JA_DICT and not zh.startswith("["):
            _ZH_JA_DICT[zh] = ja


class OfflineTranslator(BaseTranslator):
    """
    离线翻译器 —— 纯 Python 实现，零外部依赖。
    使用内置词典 + 纯 Python 分词，无需 fugashi/jieba 也能正常工作。
    适合单词和短句的快速查阅。
    """

    # 日语助词/后缀列表，用于分词切割
    _JA_PARTICLES = {
        "は", "が", "を", "に", "へ", "で", "と", "から", "まで",
        "より", "ので", "のに", "ても", "では", "には", "とは",
        "か", "も", "や", "の", "な", "ね", "よ", "わ", "さ",
    }
    # 日语常见后缀（动词/形容词活用）
    _JA_SUFFIXES = {
        "ます", "ました", "ません", "ましょう",
        "です", "でした", "ではありません",
        "た", "だ", "ている", "ていた", "ています",
        "ない", "なかった", "ありません",
        "れば", "たら", "なら", "ば",
        "う", "よう", "たい", "たく", "たかった",
        "れる", "られる", "せる", "させる",
        "く", "かっ", "ければ", "かった",
    }

    # 中文常见词列表（用于最长匹配分词）
    _ZH_COMMON_WORDS: list[str] = []

    def __init__(self):
        _init_dictionaries()
        # 从词典中提取所有中文词用于分词
        if not self._ZH_COMMON_WORDS:
            self._ZH_COMMON_WORDS = sorted(
                [w for w in _ZH_JA_DICT.keys() if len(w) >= 2],
                key=len, reverse=True
            )

    @property
    def name(self) -> str:
        return "offline"

    @property
    def is_online(self) -> bool:
        return False

    def is_available(self) -> bool:
        return True  # 始终可用，零外部依赖

    # ---------- 纯 Python 日语分词 ----------

    def _tokenize_ja(self, text: str) -> list[str]:
        """
        纯 Python 日语分词 —— 基于假名边界 + 助词/后缀切割。
        不依赖 MeCab/fugashi，在 PyInstaller 打包环境中可靠运行。
        """
        import re

        # Step 1: 在标点符号处切分
        text = re.sub(r'([、。！？，．,\.!\?\s「」『』（）\(\)])', r' \1 ', text)

        # Step 2: 在助词前后加空格
        for particle in sorted(self._JA_PARTICLES, key=len, reverse=True):
            # 助词通常跟在词后面
            text = re.sub(rf'([^\s])({re.escape(particle)})(\s|[、。！？，]|$)', r'\1 \2 \3', text)

        # Step 3: 处理常见的活用后缀
        for suffix in sorted(self._JA_SUFFIXES, key=len, reverse=True):
            text = re.sub(rf'([^\s])({re.escape(suffix)})(\s|[、。！？，]|$)', r'\1\2 \3', text)

        # Step 4: 在假名-汉字边界切分（かんじ漢字 → かんじ 漢字）
        text = re.sub(r'([぀-ゟ]+)([一-鿿]+)', r'\1 \2', text)
        text = re.sub(r'([一-鿿]+)([぀-ゟ]+)', r'\1 \2', text)

        # Step 5: 清理多余空格并分割
        tokens = [t for t in text.split() if t.strip()]

        return tokens if tokens else [text]

    # ---------- 纯 Python 中文分词 ----------

    def _tokenize_zh(self, text: str) -> list[str]:
        """
        纯 Python 中文分词 —— 基于词典的最长匹配算法。
        不依赖 jieba，在 PyInstaller 打包环境中可靠运行。
        """
        if not text:
            return []

        tokens = []
        i = 0
        while i < len(text):
            matched = False
            # 尝试最长匹配（从词典中查找）
            for word in self._ZH_COMMON_WORDS:
                wlen = len(word)
                if text[i:i+wlen] == word:
                    tokens.append(word)
                    i += wlen
                    matched = True
                    break
            if not matched:
                # 单个字符作为一个词
                tokens.append(text[i])
                i += 1

        return tokens

    # ---------- 翻译逻辑 ----------

    def _translate_ja_to_zh(self, text: str) -> str:
        """日→中翻译。"""
        # 1. 精确匹配整个文本
        if text in _JA_ZH_DICT:
            return _JA_ZH_DICT[text]

        # 2. 分词后逐词匹配
        tokens = self._tokenize_ja(text)
        results = []
        unmatched = []

        for token in tokens:
            if token in _JA_ZH_DICT:
                zh = _JA_ZH_DICT[token]
                if not zh.startswith("["):  # 过滤语法标记
                    results.append(zh)
                unmatched = []
            else:
                unmatched.append(token)

        if results:
            if unmatched:
                return "".join(results) + f" ({' '.join(unmatched)})"
            return "".join(results)

        # 3. 模糊匹配：尝试在词典中找最长匹配子串
        best_match = None
        best_len = 0
        best_key = ""
        for ja_word, zh_word in _JA_ZH_DICT.items():
            if ja_word in text and len(ja_word) > best_len:
                best_match = zh_word
                best_len = len(ja_word)
                best_key = ja_word

        if best_match and not best_match.startswith("[") and best_key:
            remaining = text.replace(best_key, "", 1).strip()
            if remaining:
                return f"{best_match} + [{remaining}]"
            return best_match

        # 4. 完全无法匹配
        return f"[未找到匹配: {text}]"

    def _translate_zh_to_ja(self, text: str) -> str:
        """中→日翻译（反向词典查找）。"""
        # 1. 精确匹配
        if text in _ZH_JA_DICT:
            return _ZH_JA_DICT[text]

        # 2. 分词后逐词匹配
        tokens = self._tokenize_zh(text)
        results = []

        for token in tokens:
            if token in _ZH_JA_DICT:
                results.append(_ZH_JA_DICT[token])
            else:
                # 尝试在日→中词典的 value 中找匹配
                found = False
                for ja_word, zh_word in _JA_ZH_DICT.items():
                    if token in zh_word and len(token) > 1:
                        results.append(ja_word)
                        found = True
                        break
                if not found:
                    results.append(f"[{token}]")

        return " ".join(results) if results else f"[未找到匹配: {text}]"

    def translate(self, text: str, direction: str) -> str:
        if not text.strip():
            return ""

        if direction == "auto":
            lang = detect_language(text)
            direction = "ja_to_zh" if lang.startswith("ja") else "zh_to_ja"

        if direction == "ja_to_zh":
            return self._translate_ja_to_zh(text)
        elif direction == "zh_to_ja":
            return self._translate_zh_to_ja(text)
        else:
            raise ValueError(f"未知翻译方向: {direction}")

    def get_status(self) -> dict:
        return {
            "loaded": True,
            "error": None,
            "available": True,
            "tokenizer": "built-in (pure Python)",
        }


# ============================================================
# 在线翻译器 —— DeepSeek API
# ============================================================

class FreeWebTranslator(BaseTranslator):
    """
    免费联网翻译器，调用公开网页翻译接口。
    不需要 API Key；接口不可用时由 TranslationManager 自动退回离线翻译。
    """

    @property
    def name(self) -> str:
        return "free_web"

    @property
    def is_online(self) -> bool:
        return True

    def _lang_pair(self, text: str, direction: str) -> tuple[str, str]:
        if direction == "auto":
            lang = detect_language(text)
            direction = "ja_to_zh" if lang.startswith("ja") else "zh_to_ja"

        if direction == "ja_to_zh":
            return "ja", "zh-CN"
        if direction == "zh_to_ja":
            return "zh-CN", "ja"
        raise ValueError(f"未知翻译方向: {direction}")

    def _translate_mymemory(self, text: str, source_lang: str, target_lang: str) -> str:
        target_for_api = "ja-JP" if target_lang == "ja" else target_lang
        params = urllib.parse.urlencode({
            "q": text,
            "langpair": f"{source_lang}|{target_for_api}",
        })
        url = f"https://api.mymemory.translated.net/get?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(request, timeout=12) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        if data.get("responseStatus") != 200:
            raise RuntimeError(data.get("responseDetails") or "MyMemory 返回错误")
        translated = data.get("responseData", {}).get("translatedText", "")
        if not translated.strip():
            raise RuntimeError("MyMemory 返回空结果")
        return translated.strip()

    def _translate_google_public(self, text: str, source_lang: str, target_lang: str) -> str:
        params = urllib.parse.urlencode({
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        })
        url = f"https://translate.googleapis.com/translate_a/single?{params}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
            },
        )

        with urllib.request.urlopen(request, timeout=12) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        translated = "".join(part[0] for part in data[0] if part and part[0])
        if not translated.strip():
            raise RuntimeError("Google 免费翻译接口返回空结果")
        return translated.strip()

    def translate(self, text: str, direction: str) -> str:
        if not text.strip():
            return ""

        source_lang, target_lang = self._lang_pair(text, direction)
        errors = []
        try:
            return self._translate_mymemory(text, source_lang, target_lang)
        except Exception as e:
            errors.append(str(e))

        try:
            return self._translate_google_public(text, source_lang, target_lang)
        except Exception as e:
            errors.append(str(e))

        raise RuntimeError(f"免费联网翻译失败：{'；'.join(errors)}")


class DeepSeekTranslator(BaseTranslator):
    """
    在线翻译器，调用 DeepSeek API（OpenAI 兼容接口）。
    需要有效的 API Key。
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = None

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def is_online(self) -> bool:
        return True

    def is_available(self) -> bool:
        return bool(self.config.api_key.strip())

    def _get_client(self):
        """懒初始化 OpenAI 客户端。"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
        return self._client

    def translate(self, text: str, direction: str) -> str:
        if not text.strip():
            return ""
        if not self.is_available():
            raise RuntimeError("DeepSeek API Key 未设置，请在设置中填入 API Key")

        if direction == "auto":
            lang = detect_language(text)
            direction = "ja_to_zh" if lang.startswith("ja") else "zh_to_ja"

        system_prompt = _select_prompt(direction)
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=4096,
                timeout=30,
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 调用失败：{e}")


# ============================================================
# 翻译管理器 —— 统一入口
# ============================================================

class TranslationManager:
    """
    统一翻译管理器。
    - 根据配置选择翻译器（离线/DeepSeek）
    - 集成缓存层，命中缓存直接返回
    - 线程安全，适合在 UI 线程外调用
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._offline: Optional[OfflineTranslator] = None
        self._free_web: Optional[FreeWebTranslator] = None
        self._deepseek: Optional[DeepSeekTranslator] = None
        self._lock = threading.Lock()

    def _get_translator(self) -> BaseTranslator:
        """根据当前配置获取翻译器实例。"""
        if self.config.mode == "deepseek":
            if self._deepseek is None:
                self._deepseek = DeepSeekTranslator(self.config)
            return self._deepseek
        elif self.config.mode == "free_web":
            if self._free_web is None:
                self._free_web = FreeWebTranslator()
            return self._free_web
        else:
            if self._offline is None:
                self._offline = OfflineTranslator()
            return self._offline

    def _read_cache(self, text: str, direction: str, mode: str) -> str | None:
        """缓存不可用时不影响翻译主流程。"""
        if not self.config.cache_enabled:
            return None
        try:
            from db import cache_get
            return cache_get(text, direction, mode)
        except (OSError, sqlite3.Error):
            return None

    def _save_history(self, text: str, result: str, direction: str, mode: str):
        """写缓存/历史失败时忽略，避免用户看不到翻译结果。"""
        try:
            if self.config.cache_enabled and result:
                from db import cache_set
                cache_set(text, result, direction, mode)

            from db import prune
            prune(self.config.max_history)
        except (OSError, sqlite3.Error):
            pass

    def translate(self, text: str, direction: str = "auto") -> dict:
        """
        执行翻译（含缓存）。
        返回 {"source": str, "result": str, "mode": str, "cached": bool, "direction": str}
        """
        text = text.strip()
        if not text:
            return {"source": text, "result": "", "mode": "", "cached": False, "direction": direction}

        # 自动检测方向
        if direction == "auto":
            lang = detect_language(text)
            direction = "ja_to_zh" if lang.startswith("ja") else "zh_to_ja"

        translator = self._get_translator()

        # 检查缓存
        cached = self._read_cache(text, direction, translator.name)
        if cached:
            return {
                "source": text,
                "result": cached,
                "mode": "cached",
                "cached": True,
                "direction": direction,
            }

        # 执行翻译
        try:
            result = translator.translate(text, direction)
        except Exception:
            if translator.name != "free_web":
                raise
            if self._offline is None:
                self._offline = OfflineTranslator()
            translator = self._offline
            result = translator.translate(text, direction)

        self._save_history(text, result, direction, translator.name)

        return {
            "source": text,
            "result": result,
            "mode": translator.name,
            "cached": False,
            "direction": direction,
        }

    def translate_async(self, text: str, direction: str, callback):
        """
        异步翻译 —— 在后台线程运行，完成后回调 callback(dict)。
        避免阻塞 UI。
        """
        def _run():
            try:
                result = self.translate(text, direction)
                callback(result, None)
            except Exception as e:
                callback(None, str(e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
