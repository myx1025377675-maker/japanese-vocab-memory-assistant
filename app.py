import asyncio
import io
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
import zlib
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.font_manager as fm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud


APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "japanese_learning.db"
MOJI_DECK_NAME = "MOJi N2汉字考前对策"

LEVEL_WEIGHT = {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}
RATING_VALUE = {"认识": 1.0, "模糊": 0.55, "不认识": 0.1}

EDITORIAL_PAPER = "#F5F2EB"
EDITORIAL_PANEL = "#FAF8F5"
EDITORIAL_INK = "#2A2825"
EDITORIAL_MUTED = "#6E665C"
EDITORIAL_LINE = "#D8CDBF"
EDITORIAL_RED = "#8B2626"
EDITORIAL_BLUE = "#4F6575"
EDITORIAL_GREEN = "#4E6246"
EDITORIAL_GOLD = "#9A7A3F"


@st.cache_resource
def get_web_translator():
    translator_dir = APP_DIR / "翻译助手"
    if not (translator_dir / "translator.py").exists():
        raise FileNotFoundError("未找到联网翻译模块：翻译助手/translator.py")
    translator_dir_str = str(translator_dir)
    if translator_dir_str not in sys.path:
        sys.path.insert(0, translator_dir_str)
    from translator import FreeWebTranslator

    return FreeWebTranslator()


def web_translate(text: str, direction: str) -> str:
    return get_web_translator().translate(text.strip(), direction)


def configure_plot_fonts():
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            try:
                fm.fontManager.addfont(font_path)
                font_name = fm.FontProperties(fname=font_path).get_name()
                plt.rcParams["font.family"] = font_name
                plt.rcParams["font.sans-serif"] = [font_name, "Microsoft YaHei", "Yu Gothic", "SimHei"]
                break
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False


configure_plot_fonts()


def apply_editorial_chart_style(ax, grid_axis="y"):
    """Give Matplotlib charts the same quiet printed-paper style as the app."""
    ax.set_facecolor(EDITORIAL_PAPER)
    ax.figure.set_facecolor(EDITORIAL_PAPER)
    for side in ["top", "right"]:
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        if side in ax.spines:
            ax.spines[side].set_color(EDITORIAL_LINE)
            ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=EDITORIAL_MUTED, labelsize=9)
    ax.title.set_color(EDITORIAL_INK)
    ax.xaxis.label.set_color(EDITORIAL_MUTED)
    ax.yaxis.label.set_color(EDITORIAL_MUTED)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=EDITORIAL_LINE, linewidth=0.6, alpha=0.55)
    return ax

SAMPLE_VOCAB = [
    ("日本語", "にほんご", "日语", "名词", "N5", "私は日本語を勉強しています。", "学习主题"),
    ("学生", "がくせい", "学生", "名词", "N5", "彼は大学の学生です。", "身份"),
    ("先生", "せんせい", "老师", "名词", "N5", "田中先生に質問しました。", "学校"),
    ("食べる", "たべる", "吃", "动词", "N5", "朝ご飯を食べます。", "生活"),
    ("読む", "よむ", "读", "动词", "N5", "毎日ニュースを読みます。", "学习"),
    ("書く", "かく", "写", "动词", "N5", "ノートに漢字を書きます。", "学习"),
    ("静か", "しずか", "安静", "形容动词", "N5", "図書館は静かです。", "环境"),
    ("便利", "べんり", "方便", "形容动词", "N5", "このアプリは便利です。", "评价"),
    ("旅行", "りょこう", "旅行", "名词", "N4", "夏休みに京都へ旅行します。", "兴趣"),
    ("必要", "ひつよう", "必要", "形容动词", "N4", "復習は必要です。", "学习"),
    ("経験", "けいけん", "经验", "名词", "N4", "良い経験になりました。", "抽象"),
    ("説明", "せつめい", "说明", "名词/动词", "N4", "文法を説明してください。", "课堂"),
    ("忘れる", "わすれる", "忘记", "动词", "N4", "単語を忘れないように復習します。", "记忆"),
    ("覚える", "おぼえる", "记住", "动词", "N4", "新しい言葉を覚えます。", "记忆"),
    ("習慣", "しゅうかん", "习惯", "名词", "N3", "毎日の復習を習慣にします。", "学习策略"),
    ("資料", "しりょう", "资料", "名词", "N3", "授業の資料を読みます。", "学习"),
    ("確認", "かくにん", "确认", "名词/动词", "N3", "答えを確認しましょう。", "操作"),
    ("改善", "かいぜん", "改进", "名词/动词", "N3", "学習方法を改善します。", "策略"),
    ("理解", "りかい", "理解", "名词/动词", "N3", "文章の意味を理解しました。", "认知"),
    ("可能性", "かのうせい", "可能性", "名词", "N3", "合格の可能性があります。", "抽象"),
    ("傾向", "けいこう", "倾向", "名词", "N2", "試験には出題の傾向があります。", "分析"),
    ("分析", "ぶんせき", "分析", "名词/动词", "N2", "学習データを分析します。", "数据"),
    ("維持", "いじ", "维持", "名词/动词", "N2", "記憶を維持するために復習します。", "记忆"),
    ("効率", "こうりつ", "效率", "名词", "N2", "効率よく単語を覚えたいです。", "策略"),
    ("獲得", "かくとく", "获得", "名词/动词", "N2", "語彙力を獲得します。", "学习"),
    ("抽象", "ちゅうしょう", "抽象", "名词/形容动词", "N1", "抽象的な概念を説明します。", "高阶"),
    ("体系", "たいけい", "体系", "名词", "N1", "文法を体系的に整理します。", "高阶"),
    ("推論", "すいろん", "推论", "名词/动词", "N1", "文脈から意味を推論します。", "高阶"),
    ("妥当", "だとう", "妥当", "streamlit run app.py形容动词", "N1", "妥当な判断が必要です。", "高阶"),
    ("継続", "けいぞく", "持续", "名词/动词", "N1", "継続は力なり。", "学习策略"),
]

HIGH_FREQ_DATA = """
私|わたし|我|代词|N5|自我介绍、表达个人想法时最常用。|私は日本語を勉強しています。;私は中国から来ました。
あなた|あなた|你|代词|N5|直接称呼对方，正式场景可用姓名代替。|あなたの名前は何ですか。;あなたは学生ですか。
人|ひと|人|名词|N5|表示人、别人、某类人。|駅に人が多いです。;親切な人に会いました。
日本|にほん|日本|名词|N5|国家名，也常出现在文化、旅行话题。|日本へ旅行します。;日本の文化に興味があります。
日本語|にほんご|日语|名词|N5|表示语言、课程和学习对象。|日本語を話します。;日本語の勉強は楽しいです。
学生|がくせい|学生|名词|N5|说明身份或职业。|私は大学の学生です。;学生が教室にいます。
先生|せんせい|老师|名词|N5|称呼教师、医生等受尊敬的人。|先生に質問します。;田中先生は親切です。
友達|ともだち|朋友|名词|N5|表示朋友、同伴。|友達と映画を見ます。;友達に電話します。
家|いえ|家|名词|N5|表示住宅、家里。|家へ帰ります。;私の家は駅の近くです。
学校|がっこう|学校|名词|N5|表示学习场所。|学校へ行きます。;学校で日本語を勉強します。
会社|かいしゃ|公司|名词|N5|工作、商务话题常用。|会社で働きます。;会社は駅の前です。
駅|えき|车站|名词|N5|交通场景高频词。|駅で友達に会います。;駅まで歩きます。
店|みせ|商店|名词|N5|购物、餐饮场景常用。|店でパンを買います。;この店は有名です。
本|ほん|书|名词|N5|学习、阅读话题常用。|本を読みます。;日本語の本を買いました。
水|みず|水|名词|N5|生活基础词。|水を飲みます。;冷たい水がほしいです。
時間|じかん|时间|名词|N5|表示时间长度或空闲。|時間があります。;勉強する時間が足りません。
今日|きょう|今天|名词|N5|日期表达高频。|今日は暑いです。;今日、テストがあります。
明日|あした|明天|名词|N5|计划表达高频。|明日、学校へ行きます。;明日は休みです。
昨日|きのう|昨天|名词|N5|过去事件表达。|昨日、映画を見ました。;昨日は雨でした。
今|いま|现在|名词/副词|N5|表示当前时间或状态。|今、勉強しています。;今は忙しいです。
朝|あさ|早上|名词|N5|时间段表达。|朝ご飯を食べます。;朝早く起きます。
昼|ひる|中午|名词|N5|时间段表达。|昼に友達と会います。;昼ご飯を食べました。
夜|よる|晚上|名词|N5|时间段表达。|夜、音楽を聞きます。;夜は静かです。
毎日|まいにち|每天|名词/副词|N5|习惯表达常用。|毎日単語を覚えます。;毎日学校へ行きます。
何|なに|什么|代词|N5|疑问句核心词。|これは何ですか。;何を食べますか。
誰|だれ|谁|代词|N5|询问人物。|あの人は誰ですか。;誰と行きますか。
どこ|どこ|哪里|代词|N5|询问地点。|駅はどこですか。;どこで勉強しますか。
いつ|いつ|什么时候|代词|N5|询问时间。|試験はいつですか。;いつ日本へ行きますか。
なぜ|なぜ|为什么|副词|N4|询问原因，较书面。|なぜ遅れましたか。;なぜ日本語を勉強しますか。
どう|どう|怎么样|副词|N5|询问状态、方法或意见。|日本語はどうですか。;どうやって覚えますか。
行く|いく|去|动词|N5|表示移动到某地。|学校へ行きます。;週末に京都へ行きます。
来る|くる|来|动词|N5|表示向说话者方向移动。|友達が家に来ます。;明日来てください。
帰る|かえる|回去|动词|N5|回家、返回。|家へ帰ります。;早く帰りたいです。
食べる|たべる|吃|动词|N5|饮食场景高频。|朝ご飯を食べます。;寿司を食べたいです。
飲む|のむ|喝|动词|N5|饮食场景高频。|水を飲みます。;コーヒーを飲みました。
見る|みる|看|动词|N5|看电视、看资料等。|映画を見ます。;写真を見てください。
聞く|きく|听；问|动词|N5|听声音或询问信息。|音楽を聞きます。;先生に聞きます。
話す|はなす|说话|动词|N5|语言交流场景。|日本語で話します。;友達と話しました。
読む|よむ|读|动词|N5|阅读文章、书、新闻。|本を読みます。;ニュースを読みました。
書く|かく|写|动词|N5|写字、写文章、写答案。|漢字を書きます。;名前を書いてください。
買う|かう|买|动词|N5|购物场景。|本を買います。;新しい服を買いました。
使う|つかう|使用|动词|N5|使用工具、方法。|辞書を使います。;このアプリを使います。
作る|つくる|制作|动词|N5|做饭、做东西、建立计划。|料理を作ります。;学習計画を作ります。
待つ|まつ|等待|动词|N5|等待人或时间。|駅で友達を待ちます。;少し待ってください。
会う|あう|见面|动词|N5|和某人见面。|友達に会います。;先生に会いました。
知る|しる|知道|动词|N5|获知信息。|その言葉を知っています。;答えを知りたいです。
思う|おもう|想；认为|动词|N4|表达观点、判断。|いいと思います。;難しいと思いました。
考える|かんがえる|思考|动词|N4|思考、考虑。|答えを考えます。;将来について考えています。
分かる|わかる|明白|动词|N5|理解内容。|意味が分かります。;説明がよく分かりました。
覚える|おぼえる|记住|动词|N4|记忆词汇、知识。|単語を覚えます。;名前を覚えてください。
忘れる|わすれる|忘记|动词|N4|遗忘信息。|宿題を忘れました。;単語を忘れないように復習します。
勉強|べんきょう|学习|名词/动词|N5|学习行为。|日本語を勉強します。;毎日勉強しています。
練習|れんしゅう|练习|名词/动词|N4|技能训练。|会話を練習します。;発音の練習をします。
復習|ふくしゅう|复习|名词/动词|N3|回顾已学内容。|単語を復習します。;復習は大切です。
質問|しつもん|问题；提问|名词/动词|N5|课堂和交流常用。|先生に質問します。;質問があります。
答え|こたえ|答案|名词|N5|题目答案。|答えを確認します。;正しい答えを選びます。
問題|もんだい|问题；题目|名词|N4|考试、社会话题常用。|この問題は難しいです。;環境問題を考えます。
意味|いみ|意思|名词|N4|词语含义。|この言葉の意味は何ですか。;意味を調べます。
言葉|ことば|词语；语言|名词|N4|语言学习核心词。|新しい言葉を覚えます。;言葉の使い方を学びます。
単語|たんご|单词|名词|N4|词汇学习核心词。|単語を覚えます。;高頻単語を整理します。
文法|ぶんぽう|语法|名词|N4|语言结构。|文法を勉強します。;この文法は大切です。
漢字|かんじ|汉字|名词|N5|日语文字系统。|漢字を書きます。;漢字の読み方を覚えます。
例文|れいぶん|例句|名词|N3|展示用法的句子。|例文を読みます。;例文で使い方を確認します。
大きい|おおきい|大的|形容词|N5|形容尺寸。|大きい部屋です。;声が大きいです。
小さい|ちいさい|小的|形容词|N5|形容尺寸。|小さいかばんです。;小さい町に住んでいます。
新しい|あたらしい|新的|形容词|N5|形容新旧。|新しい単語を覚えます。;新しい本を買いました。
古い|ふるい|旧的|形容词|N5|形容新旧。|古い写真を見ました。;この建物は古いです。
良い|よい|好的|形容词|N5|评价事物。|良い方法です。;今日は良い天気です。
悪い|わるい|坏的|形容词|N5|评价事物或状态。|天気が悪いです。;気分が悪いです。
高い|たかい|高的；贵的|形容词|N5|表示高度或价格。|この時計は高いです。;山が高いです。
安い|やすい|便宜的|形容词|N5|价格低。|この店は安いです。;安い切符を買いました。
早い|はやい|早；快|形容词|N5|时间早或速度快。|朝早く起きます。;電車は早いです。
遅い|おそい|晚；慢|形容词|N5|时间晚或速度慢。|返事が遅いです。;今日は帰りが遅いです。
多い|おおい|多的|形容词|N5|数量多。|宿題が多いです。;人が多い場所です。
少ない|すくない|少的|形容词|N5|数量少。|時間が少ないです。;学生が少ないです。
難しい|むずかしい|难的|形容词|N5|表示难度高。|この問題は難しいです。;漢字は難しいです。
易しい|やさしい|容易的|形容词|N5|表示难度低。|この本は易しいです。;易しい言葉で説明します。
楽しい|たのしい|开心的|形容词|N5|表达愉快感受。|日本語の勉強は楽しいです。;旅行は楽しかったです。
忙しい|いそがしい|忙的|形容词|N5|表示忙碌状态。|今日は忙しいです。;仕事が忙しいです。
便利|べんり|方便|形容动词|N5|评价工具、地点。|このアプリは便利です。;駅に近くて便利です。
大切|たいせつ|重要|形容动词|N4|表示重要性。|復習は大切です。;友達を大切にします。
必要|ひつよう|必要|形容动词|N4|表示需要。|準備が必要です。;辞書が必要です。
安全|あんぜん|安全|名词/形容动词|N4|安全状态。|安全を確認します。;安全な場所へ行きます。
可能|かのう|可能|形容动词|N3|表示可行性。|オンラインで申請可能です。;参加が可能です。
理由|りゆう|理由|名词|N4|说明原因。|理由を説明します。;遅れた理由を聞きました。
場合|ばあい|情况；场合|名词|N4|条件表达常用。|雨の場合は中止です。;困った場合は連絡してください。
方法|ほうほう|方法|名词|N3|做事方式。|良い方法を探します。;勉強方法を変えます。
目的|もくてき|目的|名词|N3|行动目标。|学習の目的を決めます。;目的を説明してください。
結果|けっか|结果|名词|N3|考试、调查、行动后的结果。|テストの結果を見ます。;良い結果が出ました。
原因|げんいん|原因|名词|N3|导致结果的理由。|失敗の原因を調べます。;原因が分かりました。
情報|じょうほう|信息|名词|N3|资料、新闻、数据。|情報を集めます。;正しい情報が必要です。
資料|しりょう|资料|名词|N3|学习或说明材料。|授業の資料を読みます。;資料を準備します。
説明|せつめい|说明|名词/动词|N4|解释内容。|文法を説明します。;説明を聞きました。
確認|かくにん|确认|名词/动词|N3|检查是否正确。|答えを確認します。;予定を確認してください。
準備|じゅんび|准备|名词/动词|N4|事前安排。|試験の準備をします。;旅行の準備ができました。
連絡|れんらく|联系|名词/动词|N4|发送消息、通知。|あとで連絡します。;先生に連絡してください。
予約|よやく|预约|名词/动词|N4|预定服务。|ホテルを予約します。;予約を確認しました。
経験|けいけん|经验|名词/动词|N4|经历、体验。|良い経験になりました。;日本で生活した経験があります。
文化|ぶんか|文化|名词|N4|文化主题高频。|日本の文化を学びます。;文化の違いを理解します。
社会|しゃかい|社会|名词|N4|社会话题高频。|社会について考えます。;社会問題を調べます。
経済|けいざい|经济|名词|N3|新闻、社会话题常用。|経済のニュースを読みます。;経済が発展しています。
環境|かんきょう|环境|名词|N3|自然、生活、社会环境。|環境を守ります。;学習環境を整えます。
技術|ぎじゅつ|技术|名词|N3|科技、技能。|新しい技術を学びます。;技術が進歩しています。
政府|せいふ|政府|名词|N3|新闻高频。|政府は政策を発表しました。;政府の対応を確認します。
発表|はっぴょう|发表；公布|名词/动词|N3|新闻、课堂、会议常用。|結果を発表します。;新しい計画が発表されました。
計画|けいかく|计划|名词/动词|N3|安排未来行动。|学習計画を作ります。;旅行を計画しています。
地域|ちいき|地区|名词|N3|新闻、地理话题。|この地域は雨が多いです。;地域の人と交流します。
影響|えいきょう|影响|名词/动词|N3|说明影响关系。|生活に影響があります。;天気が交通に影響します。
対応|たいおう|应对|名词/动词|N2|处理问题、回应情况。|問題に対応します。;早い対応が必要です。
支援|しえん|支援|名词/动词|N2|帮助、援助。|学生を支援します。;被害を受けた地域を支援します。
改善|かいぜん|改善|名词/动词|N3|使情况变好。|学習方法を改善します。;効率を改善したいです。
分析|ぶんせき|分析|名词/动词|N2|数据、文章、原因分析。|学習データを分析します。;結果を分析します。
効率|こうりつ|效率|名词|N2|学习、工作效果。|効率よく単語を覚えます。;効率を上げたいです。
理解|りかい|理解|名词/动词|N3|理解意义、内容。|文章の意味を理解します。;文法を理解しました。
記憶|きおく|记忆|名词/动词|N3|记住内容的能力。|記憶を維持します。;新しい単語を記憶します。
維持|いじ|维持|名词/动词|N2|保持某种状态。|記憶を維持するために復習します。;健康を維持します。
継続|けいぞく|持续|名词/动词|N1|持续进行。|学習を継続します。;継続は力なり。
""".strip()


def parse_high_freq_blob(blob, start_rank=1):
    rows = []
    for idx, line in enumerate(blob.splitlines(), start=start_rank):
        word, kana, meaning, pos, level, usage, examples = line.split("|")
        example_list = [item.strip() for item in examples.split(";") if item.strip()]
        rows.append(
            {
                "rank": idx,
                "word": word,
                "kana": kana,
                "meaning": meaning,
                "part_of_speech": pos,
                "jlpt_level": level,
                "usage": usage,
                "examples": example_list,
            }
        )
    return rows


def parse_high_freq_words():
    words = parse_high_freq_blob(HIGH_FREQ_DATA)
    if MORE_HIGH_FREQ_DATA:
        words.extend(parse_high_freq_blob(MORE_HIGH_FREQ_DATA, len(words) + 1))
    dedup = {}
    for row in words:
        dedup[(row["word"], row["kana"])] = row
    out = list(dedup.values())
    for index, row in enumerate(out, start=1):
        row["rank"] = index
    return out

MORE_HIGH_FREQ_DATA = """
以内|いない|以内|名词|N4|表示范围的上限。|三日以内に提出してください。;一時間以内に戻ります。
以上|いじょう|以上|名词/接尾|N4|表示数量、程度超过某基准。|十八歳以上の人が参加できます。;これ以上待てません。
以下|いか|以下|名词/接尾|N4|表示数量、程度不超过某基准。|五千円以下の商品を探します。;詳しくは以下を見てください。
以外|いがい|以外|名词|N4|表示排除某项。|日曜日以外は毎日勉強します。;私以外は全員知っています。
以内|いない|以内|名词|N4|时间或范围内。|二週間以内に返事します。;予算以内で買います。
必要性|ひつようせい|必要性|名词|N2|表示某事物的必要程度。|復習の必要性を感じます。;制度の必要性を説明します。
実際|じっさい|实际|名词/副词|N3|表示真实情况。|実際に使ってみます。;実際の会話で練習します。
事実|じじつ|事实|名词|N3|表示真实发生的事情。|事実を確認します。;それは事実ではありません。
場合|ばあい|情况|名词|N4|表示条件或场合。|困った場合は連絡してください。;雨の場合は中止です。
状況|じょうきょう|状况|名词|N3|表示事情的发展状态。|状況を確認します。;今の状況を説明してください。
状態|じょうたい|状态|名词|N3|表示人或事物的样子。|健康状態が良いです。;保存状態を見ます。
関係|かんけい|关系|名词/动词|N4|表示联系或关联。|二つの問題は関係があります。;仕事に関係する資料です。
関連|かんれん|关联|名词/动词|N2|表示内容之间有联系。|関連する単語を覚えます。;事件との関連を調べます。
対象|たいしょう|对象|名词|N2|表示行为或研究所面向的人或物。|学生を対象に調査します。;分析の対象を決めます。
内容|ないよう|内容|名词|N3|表示里面包含的信息。|授業の内容を復習します。;内容を確認してください。
項目|こうもく|项目|名词|N2|列表中的条目。|必要な項目を入力します。;重要な項目を整理します。
条件|じょうけん|条件|名词|N3|表示前提、要求。|条件を満たします。;参加条件を確認します。
基準|きじゅん|标准|名词|N2|判断或比较的依据。|評価基準を作ります。;基準に合っています。
程度|ていど|程度|名词|N3|表示水平、范围。|どの程度分かりますか。;ある程度理解しました。
機会|きかい|机会|名词|N3|表示时机、机会。|話す機会を増やします。;良い機会になりました。
態度|たいど|态度|名词|N3|表示面对事情的姿态。|学習態度が大切です。;態度を改めます。
行動|こうどう|行动|名词/动词|N3|表示实际行为。|すぐに行動します。;行動を記録します。
活動|かつどう|活动|名词/动词|N3|表示持续进行的行为。|クラブ活動に参加します。;学習活動を続けます。
生活|せいかつ|生活|名词/动词|N4|日常生活。|日本で生活します。;生活習慣を変えます。
習慣|しゅうかん|习惯|名词|N3|反复形成的行为。|復習を習慣にします。;良い習慣を作ります。
能力|のうりょく|能力|名词|N3|表示可以完成某事的力量。|読む能力を高めます。;能力を伸ばします。
知識|ちしき|知识|名词|N3|通过学习得到的信息。|文法の知識が必要です。;知識を増やします。
記録|きろく|记录|名词/动词|N3|保存信息或成绩。|学習記録を残します。;結果を記録します。
目標|もくひょう|目标|名词|N3|想达到的标准。|今日の目標を決めます。;目標に近づきます。
予定|よてい|预定|名词|N4|计划中的安排。|明日の予定を確認します。;予定を変更します。
変更|へんこう|变更|名词/动词|N3|改变原有安排。|予定を変更します。;設定を変更できます。
選択|せんたく|选择|名词/动词|N3|从多个选项中挑选。|答えを選択します。;選択肢を読みます。
追加|ついか|追加|名词/动词|N3|在已有基础上增加。|単語を追加します。;機能を追加しました。
削除|さくじょ|删除|名词/动词|N2|去掉不需要的内容。|不要なデータを削除します。;単語を削除しました。
保存|ほぞん|保存|名词/动词|N3|保留数据或状态。|結果を保存します。;ファイルを保存してください。
登録|とうろく|登记|名词/动词|N3|录入系统或名单。|新しい単語を登録します。;会員登録をします。
表示|ひょうじ|显示|名词/动词|N3|把内容显示出来。|結果を表示します。;画面に表示されます。
検索|けんさく|搜索|名词/动词|N3|查找信息。|辞書で検索します。;単語を検索してください。
入力|にゅうりょく|输入|名词/动词|N3|向系统写入信息。|名前を入力します。;答えを入力してください。
出力|しゅつりょく|输出|名词/动词|N2|系统输出结果。|分析結果を出力します。;データを出力します。
分類|ぶんるい|分类|名词/动词|N2|按类别整理。|単語を分類します。;品詞ごとに分類します。
評価|ひょうか|评价|名词/动词|N3|判断价值或水平。|学習効果を評価します。;自分の力を評価します。
予測|よそく|预测|名词/动词|N2|推测未来结果。|記憶率を予測します。;結果を予測します。
推薦|すいせん|推荐|名词/动词|N2|建议选择某项。|復習単語を推薦します。;友達に本を推薦します。
重要|じゅうよう|重要|形容动词|N3|表示价值高、不能忽视。|重要な単語を覚えます。;重要な点を確認します。
正確|せいかく|准确|形容动词|N3|没有错误。|正確に答えます。;正確な情報が必要です。
詳細|しょうさい|详细|名词/形容动词|N3|内容细致。|詳細を確認します。;詳細な説明を読みます。
簡単|かんたん|简单|形容动词|N5|容易、不复杂。|簡単な問題です。;簡単に説明します。
複雑|ふくざつ|复杂|形容动词|N3|结构或关系不简单。|文法が複雑です。;複雑な問題を分析します。
十分|じゅうぶん|充分|形容动词/副词|N4|数量或程度足够。|十分に復習しました。;時間は十分あります。
不足|ふそく|不足|名词/动词|N3|不够。|練習が不足しています。;語彙力の不足を感じます。
効果|こうか|效果|名词|N3|行为带来的结果。|復習の効果があります。;効果を測定します。
成果|せいか|成果|名词|N2|努力后得到的结果。|学習の成果が出ました。;成果を報告します。
課題|かだい|课题|名词|N2|需要解决的问题或作业。|次の課題を提出します。;課題を分析します。
提出|ていしゅつ|提交|名词/动词|N3|交出作业或资料。|レポートを提出します。;期限までに提出してください。
期限|きげん|期限|名词|N2|规定的时间界限。|提出期限を確認します。;期限に間に合います。
資料|しりょう|资料|名词|N3|参考材料。|資料を読みます。;資料を準備します。
報告|ほうこく|报告|名词/动词|N3|汇报情况或结果。|結果を報告します。;報告書を書きます。
経験|けいけん|经验|名词/动词|N4|亲身经历。|良い経験になりました。;経験を積みます。
成績|せいせき|成绩|名词|N3|学习或工作结果。|成績が上がりました。;試験の成績を確認します。
合格|ごうかく|合格|名词/动词|N3|考试通过。|試験に合格しました。;合格を目指します。
失敗|しっぱい|失败|名词/动词|N4|没有成功。|失敗から学びます。;失敗を恐れません。
成功|せいこう|成功|名词/动词|N3|达到目标。|計画が成功しました。;成功の理由を考えます。
努力|どりょく|努力|名词/动词|N3|为目标付出行动。|毎日努力します。;努力が必要です。
成長|せいちょう|成长|名词/动词|N3|能力或状态提高。|語彙力が成長しました。;成長を感じます。
進歩|しんぽ|进步|名词/动词|N3|能力向前发展。|日本語が進歩しました。;技術が進歩しています。
集中|しゅうちゅう|集中|名词/动词|N3|注意力聚集。|勉強に集中します。;集中力を高めます。
確認|かくにん|确认|名词/动词|N3|检查是否正确。|答えを確認します。;予定を確認してください。
整理|せいり|整理|名词/动词|N3|把内容理顺。|単語を整理します。;資料を整理しました。
説明|せつめい|说明|名词/动词|N4|把事情讲清楚。|使い方を説明します。;説明を聞きます。
認識|にんしき|认识；认知|名词/动词|N2|表示理解并把握事物。|問題を正しく認識します。;現状の認識が必要です。
判断|はんだん|判断|名词/动词|N3|根据情况做决定。|状況を判断します。;正しい判断が必要です。
検討|けんとう|研究；讨论|名词/动词|N2|仔细考虑可行性。|改善案を検討します。;内容を検討してください。
達成|たっせい|达成|名词/动词|N2|完成目标。|目標を達成します。;計画を達成しました。
実現|じつげん|实现|名词/动词|N2|让计划成为现实。|夢を実現します。;機能を実現しました。
構成|こうせい|构成|名词/动词|N2|组成整体结构。|文章の構成を考えます。;画面を構成します。
設計|せっけい|设计|名词/动词|N2|规划系统或结构。|システムを設計します。;設計を見直します。
処理|しょり|处理|名词/动词|N2|解决或加工事务。|データを処理します。;問題を処理します。
操作|そうさ|操作|名词/动词|N2|使用设备或系统。|画面を操作します。;操作方法を確認します。
機能|きのう|功能|名词|N2|工具或系统所具备的作用。|新しい機能を追加します。;機能を改善します。
性能|せいのう|性能|名词|N2|机器、系统的能力。|モデルの性能を評価します。;性能が高いです。
精度|せいど|精度|名词|N2|准确程度。|予測精度を高めます。;精度を確認します。
傾向|けいこう|倾向|名词|N2|事物发展的方向。|間違いの傾向を分析します。;出題傾向を確認します。
対策|たいさく|对策|名词|N2|应对问题的方法。|試験対策をします。;対策を考えます。
要因|よういん|主要因素|名词|N2|造成结果的因素。|失敗の要因を分析します。;要因を調べます。
仮定|かてい|假定|名词/动词|N2|暂时设定条件。|条件を仮定します。;仮定に基づいて考えます。
推測|すいそく|推测|名词/动词|N2|根据线索判断。|意味を推測します。;結果を推測します。
抽出|ちゅうしゅつ|抽取|名词/动词|N1|从整体中取出需要部分。|重要語を抽出します。;特徴を抽出します。
変換|へんかん|转换|名词/动词|N2|把形式变成另一种。|文字を変換します。;データを変換します。
適用|てきよう|应用；适用|名词/动词|N2|把规则或方法用于对象。|規則を適用します。;モデルを適用します。
応用|おうよう|应用|名词/动词|N2|把知识用于实际。|文法を応用します。;知識を応用できます。
考慮|こうりょ|考虑|名词/动词|N2|把因素纳入判断。|時間を考慮します。;利用者を考慮した設計です。
把握|はあく|掌握；把握|名词/动词|N2|理解整体情况。|学習状況を把握します。;要点を把握します。
維持|いじ|维持|名词/动词|N2|保持状态不变。|記憶を維持します。;品質を維持します。
向上|こうじょう|提高|名词/动词|N2|水平变高。|語彙力を向上させます。;精度が向上しました。
解釈|かいしゃく|解释；理解|名词/动词|N2|理解并说明含义。|文を解釈します。;意味の解釈が難しいです。
誤解|ごかい|误解|名词/动词|N2|错误理解。|意味を誤解しました。;誤解を避けます。
妥当|だとう|妥当|形容动词|N1|合适、有道理。|妥当な判断です。;方法が妥当か確認します。
抽象|ちゅうしょう|抽象|名词/形容动词|N1|不具体、概念化。|抽象的な概念を学びます。;抽象度が高いです。
具体|ぐたい|具体|名词/形容动词|N2|清楚、实际。|具体的な例を出します。;具体的に説明します。
体系|たいけい|体系|名词|N1|有组织的整体结构。|知識を体系的に整理します。;文法体系を学びます。
概念|がいねん|概念|名词|N2|对事物的抽象认识。|新しい概念を理解します。;概念を説明します。
論理|ろんり|逻辑|名词|N2|思考和说明的条理。|論理的に説明します。;論理を確認します。
根拠|こんきょ|根据|名词|N2|判断的依据。|根拠を示します。;根拠のある説明です。
仮説|かせつ|假说|名词|N1|暂时提出的解释。|仮説を立てます。;仮説を検証します。
検証|けんしょう|验证|名词/动词|N1|确认是否正确。|結果を検証します。;仮説を検証します。
矛盾|むじゅん|矛盾|名词/动词|N1|前后不一致。|説明に矛盾があります。;矛盾を見つけます。
曖昧|あいまい|暧昧；模糊|形容动词|N1|不清楚、不明确。|曖昧な表現を避けます。;意味が曖昧です。
顕著|けんちょ|显著|形容动词|N1|非常明显。|効果が顕著です。;違いが顕著に現れます。
補足|ほそく|补充|名词/动词|N2|添加不足内容。|説明を補足します。;補足資料を読みます。
省略|しょうりゃく|省略|名词/动词|N2|删去一部分。|主語を省略します。;説明を省略します。
強調|きょうちょう|强调|名词/动词|N2|突出重点。|重要点を強調します。;違いを強調します。
比較|ひかく|比较|名词/动词|N3|对照差异。|二つの方法を比較します。;結果を比較します。
分析|ぶんせき|分析|名词/动词|N2|拆分研究。|データを分析します。;原因を分析します。
総合|そうごう|综合|名词/动词|N2|把多个方面合起来。|結果を総合します。;総合的に判断します。
推論|すいろん|推论|名词/动词|N1|根据事实推出结论。|文脈から推論します。;論理的に推論します。
""".strip()

HIGH_FREQ_WORDS = parse_high_freq_words()

BUILTIN_DECKS = {
    "综合高频100词库": [
        (
            row["word"],
            row["kana"],
            row["meaning"],
            row["part_of_speech"],
            row["jlpt_level"],
            row["examples"][0],
            "高频100",
        )
        for row in HIGH_FREQ_WORDS
    ],
    "高频基础词库": [
        ("私", "わたし", "我", "代词", "N5", "私は学生です。", "高频"),
        ("あなた", "あなた", "你", "代词", "N5", "あなたの名前は何ですか。", "高频"),
        ("人", "ひと", "人", "名词", "N5", "駅に人が多いです。", "高频"),
        ("何", "なに", "什么", "代词", "N5", "これは何ですか。", "高频"),
        ("今", "いま", "现在", "名词/副词", "N5", "今、勉強しています。", "高频"),
        ("今日", "きょう", "今天", "名词", "N5", "今日は忙しいです。", "高频"),
        ("明日", "あした", "明天", "名词", "N5", "明日、学校へ行きます。", "高频"),
        ("昨日", "きのう", "昨天", "名词", "N5", "昨日、映画を見ました。", "高频"),
        ("行く", "いく", "去", "动词", "N5", "駅へ行きます。", "高频"),
        ("来る", "くる", "来", "动词", "N5", "友達が家に来ます。", "高频"),
        ("見る", "みる", "看", "动词", "N5", "テレビを見ます。", "高频"),
        ("聞く", "きく", "听；问", "动词", "N5", "音楽を聞きます。", "高频"),
        ("話す", "はなす", "说话", "动词", "N5", "日本語で話します。", "高频"),
        ("買う", "かう", "买", "动词", "N5", "本を買います。", "高频"),
        ("使う", "つかう", "使用", "动词", "N5", "辞書を使います。", "高频"),
        ("作る", "つくる", "制作", "动词", "N5", "料理を作ります。", "高频"),
        ("大きい", "おおきい", "大的", "形容词", "N5", "大きいかばんです。", "高频"),
        ("小さい", "ちいさい", "小的", "形容词", "N5", "小さい部屋です。", "高频"),
        ("新しい", "あたらしい", "新的", "形容词", "N5", "新しい単語を覚えます。", "高频"),
        ("古い", "ふるい", "旧的", "形容词", "N5", "古い写真を見ました。", "高频"),
        ("良い", "よい", "好的", "形容词", "N5", "良い方法です。", "高频"),
        ("悪い", "わるい", "坏的", "形容词", "N5", "天気が悪いです。", "高频"),
        ("早い", "はやい", "早；快", "形容词", "N5", "朝早く起きます。", "高频"),
        ("多い", "おおい", "多的", "形容词", "N5", "宿題が多いです。", "高频"),
        ("少ない", "すくない", "少的", "形容词", "N5", "時間が少ないです。", "高频"),
        ("好き", "すき", "喜欢", "形容动词", "N5", "日本語が好きです。", "高频"),
        ("大切", "たいせつ", "重要", "形容动词", "N4", "復習は大切です。", "高频"),
        ("問題", "もんだい", "问题", "名词", "N4", "この問題は難しいです。", "高频"),
        ("時間", "じかん", "时间", "名词", "N5", "時間があります。", "高频"),
        ("場所", "ばしょ", "地点", "名词", "N4", "静かな場所で勉強します。", "高频"),
    ],
    "JLPT N4-N3 扩展词库": [
        ("場合", "ばあい", "情况；场合", "名词", "N4", "雨の場合は中止です。", "JLPT"),
        ("理由", "りゆう", "理由", "名词", "N4", "理由を説明してください。", "JLPT"),
        ("準備", "じゅんび", "准备", "名词/动词", "N4", "試験の準備をします。", "JLPT"),
        ("予約", "よやく", "预约", "名词/动词", "N4", "ホテルを予約しました。", "JLPT"),
        ("連絡", "れんらく", "联系", "名词/动词", "N4", "あとで連絡します。", "JLPT"),
        ("約束", "やくそく", "约定", "名词/动词", "N4", "友達と約束があります。", "JLPT"),
        ("意見", "いけん", "意见", "名词", "N4", "意見を言ってください。", "JLPT"),
        ("文化", "ぶんか", "文化", "名词", "N4", "日本の文化に興味があります。", "JLPT"),
        ("社会", "しゃかい", "社会", "名词", "N4", "社会について勉強します。", "JLPT"),
        ("経済", "けいざい", "经济", "名词", "N3", "経済のニュースを読みます。", "JLPT"),
        ("自然", "しぜん", "自然", "名词", "N3", "自然を守りたいです。", "JLPT"),
        ("環境", "かんきょう", "环境", "名词", "N3", "環境問題を考えます。", "JLPT"),
        ("技術", "ぎじゅつ", "技术", "名词", "N3", "新しい技術を学びます。", "JLPT"),
        ("情報", "じょうほう", "信息", "名词", "N3", "情報を集めます。", "JLPT"),
        ("結果", "けっか", "结果", "名词", "N3", "テストの結果を確認します。", "JLPT"),
        ("原因", "げんいん", "原因", "名词", "N3", "失敗の原因を調べます。", "JLPT"),
        ("目的", "もくてき", "目的", "名词", "N3", "学習の目的を決めます。", "JLPT"),
        ("方法", "ほうほう", "方法", "名词", "N3", "良い方法を探します。", "JLPT"),
        ("努力", "どりょく", "努力", "名词/动词", "N3", "毎日努力しています。", "JLPT"),
        ("成長", "せいちょう", "成长", "名词/动词", "N3", "語彙力が成長しました。", "JLPT"),
        ("参加", "さんか", "参加", "名词/动词", "N3", "大会に参加します。", "JLPT"),
        ("提出", "ていしゅつ", "提交", "名词/动词", "N3", "レポートを提出します。", "JLPT"),
        ("選択", "せんたく", "选择", "名词/动词", "N3", "答えを選択します。", "JLPT"),
        ("比較", "ひかく", "比较", "名词/动词", "N3", "二つの方法を比較します。", "JLPT"),
        ("判断", "はんだん", "判断", "名词/动词", "N3", "状況を判断します。", "JLPT"),
    ],
    "新闻阅读高频词库": [
        ("政府", "せいふ", "政府", "名词", "N3", "政府は新しい政策を発表しました。", "新闻"),
        ("発表", "はっぴょう", "发表；公布", "名词/动词", "N3", "結果を発表します。", "新闻"),
        ("計画", "けいかく", "计划", "名词/动词", "N3", "新しい計画を立てます。", "新闻"),
        ("政策", "せいさく", "政策", "名词", "N2", "政策について議論します。", "新闻"),
        ("地域", "ちいき", "地区", "名词", "N3", "この地域は雨が多いです。", "新闻"),
        ("住民", "じゅうみん", "居民", "名词", "N2", "住民に説明しました。", "新闻"),
        ("被害", "ひがい", "受害；损失", "名词", "N2", "台風で被害が出ました。", "新闻"),
        ("安全", "あんぜん", "安全", "名词/形容动词", "N4", "安全を確認します。", "新闻"),
        ("確認", "かくにん", "确认", "名词/动词", "N3", "情報を確認してください。", "新闻"),
        ("調査", "ちょうさ", "调查", "名词/动词", "N3", "原因を調査します。", "新闻"),
        ("増加", "ぞうか", "增加", "名词/动词", "N3", "利用者が増加しています。", "新闻"),
        ("減少", "げんしょう", "减少", "名词/动词", "N3", "人口が減少しています。", "新闻"),
        ("影響", "えいきょう", "影响", "名词/动词", "N3", "生活に影響があります。", "新闻"),
        ("対応", "たいおう", "应对", "名词/动词", "N2", "問題に対応します。", "新闻"),
        ("支援", "しえん", "支援", "名词/动词", "N2", "学生を支援します。", "新闻"),
        ("利用", "りよう", "利用", "名词/动词", "N3", "図書館を利用します。", "新闻"),
        ("開始", "かいし", "开始", "名词/动词", "N3", "授業を開始します。", "新闻"),
        ("終了", "しゅうりょう", "结束", "名词/动词", "N3", "試験が終了しました。", "新闻"),
        ("予定", "よてい", "预定", "名词", "N4", "明日の予定を確認します。", "新闻"),
        ("可能", "かのう", "可能", "形容动词", "N3", "オンラインで申請可能です。", "新闻"),
    ],
}


def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def run_sql(sql, params=()):
    with connect() as conn:
        conn.execute(sql, params)
        conn.commit()


def query_df(sql, params=()):
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def init_db():
    with connect() as conn:
        conn.executescript(
            """
            create table if not exists decks (
                id integer primary key autoincrement,
                name text unique not null,
                source text not null,
                created_at text not null
            );
            create table if not exists vocabulary (
                id integer primary key autoincrement,
                deck_id integer not null,
                word text not null,
                kana text not null,
                meaning text not null,
                part_of_speech text default '',
                jlpt_level text default 'N5',
                example text default '',
                tags text default '',
                created_at text not null,
                unique(deck_id, word, kana)
            );
            create table if not exists reviews (
                id integer primary key autoincrement,
                word_id integer not null,
                rating text not null,
                score real not null,
                reviewed_at text not null
            );
            create table if not exists test_results (
                id integer primary key autoincrement,
                word_id integer not null,
                is_correct integer not null,
                question_type text not null,
                answered_at text not null
            );
            create table if not exists feedback (
                id integer primary key autoincrement,
                word_id integer,
                message text not null,
                created_at text not null
            );
            create table if not exists wrong_book (
                id integer primary key autoincrement,
                word_id integer not null unique,
                note text default '',
                created_at text not null
            );
            """
        )
        deck_count = conn.execute("select count(*) from decks").fetchone()[0]
        if deck_count == 0:
            now = datetime.now().isoformat(timespec="seconds")
            cur = conn.execute(
                "insert into decks(name, source, created_at) values(?,?,?)",
                ("示例 JLPT 核心词库", "系统内置", now),
            )
            deck_id = cur.lastrowid
            conn.executemany(
                """
                insert or ignore into vocabulary
                (deck_id, word, kana, meaning, part_of_speech, jlpt_level, example, tags, created_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(deck_id, *row, now) for row in SAMPLE_VOCAB],
            )
            seed_reviews(conn, deck_id)
        ensure_builtin_decks(conn)
        ensure_moji_pdf_deck(conn)
        conn.commit()


def ensure_builtin_decks(conn):
    now = datetime.now().isoformat(timespec="seconds")
    for deck_name, rows in BUILTIN_DECKS.items():
        cur = conn.execute(
            "insert or ignore into decks(name, source, created_at) values(?,?,?)",
            (deck_name, "公开高频/JLPT词表整理", now),
        )
        deck_id = cur.lastrowid or conn.execute("select id from decks where name=?", (deck_name,)).fetchone()[0]
        conn.executemany(
            """
            insert or ignore into vocabulary
            (deck_id, word, kana, meaning, part_of_speech, jlpt_level, example, tags, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(deck_id, *row, now) for row in rows],
        )


def parse_pdf_tounicode(data):
    cmap = {}
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = match.group(1).strip(b"\r\n")
        try:
            decoded = zlib.decompress(raw)
        except Exception:
            continue
        if b"beginbfrange" not in decoded and b"beginbfchar" not in decoded:
            continue
        text = decoded.decode("latin1", errors="ignore")
        for item in re.finditer(r"<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>\s*\[(.*?)\]", text, re.S):
            start = int(item.group(1), 16)
            values = re.findall(r"<([0-9A-Fa-f]+)>", item.group(3))
            for offset, value in enumerate(values):
                try:
                    cmap[start + offset] = bytes.fromhex(value).decode("utf-16-be", errors="ignore")
                except Exception:
                    pass
        for item in re.finditer(r"<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]+)>", text):
            try:
                cmap[int(item.group(1), 16)] = bytes.fromhex(item.group(2)).decode("utf-16-be", errors="ignore")
            except Exception:
                pass
    return cmap


def decode_pdf_hex(hex_text, cmap):
    raw = bytes.fromhex(hex_text)
    chars = []
    for pos in range(0, len(raw), 2):
        chars.append(cmap.get(int.from_bytes(raw[pos : pos + 2], "big"), ""))
    return "".join(chars)


def extract_pdf_text(path):
    data = path.read_bytes()
    cmap = parse_pdf_tounicode(data)
    chunks = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        raw = match.group(1).strip(b"\r\n")
        try:
            decoded = zlib.decompress(raw)
        except Exception:
            continue
        if b"TJ" not in decoded and b"Tj" not in decoded:
            continue
        stream_text = decoded.decode("latin1", errors="ignore")
        for array_text in re.findall(r"\[(.*?)\]\s*TJ", stream_text, re.S):
            parts = re.findall(r"<([0-9A-Fa-f]+)>", array_text)
            if parts:
                chunks.append("".join(decode_pdf_hex(part, cmap) for part in parts))
        for hex_text in re.findall(r"<([0-9A-Fa-f]+)>\s*Tj", stream_text):
            chunks.append(decode_pdf_hex(hex_text, cmap))
    return "\n".join(chunk for chunk in chunks if chunk.strip())


def has_kana(text):
    return any(0x3040 <= ord(ch) <= 0x30FF for ch in text)


def has_cjk(text):
    return any(0x3400 <= ord(ch) <= 0x9FFF for ch in text)


def is_moji_index(text):
    return bool(re.fullmatch(r"[0-9０-９伴兴]+", text))


def clean_moji_text(text):
    cleaned = (
        text.replace("·", "/")
        .replace("oe", "サ")
        .replace("o", "/")
        .replace(")","，")
        .replace("x", "")
        .replace("O汉", "")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，；")
    return cleaned


def parse_moji_entries(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headers = {"序号", "发音", "单词", "释义"}

    def is_kana_line(value):
        return (
            has_kana(value)
            and len(value) <= 30
            and not value.startswith("[")
            and all((0x3040 <= ord(ch) <= 0x30FF) or ch == "ー" for ch in value)
        )

    def is_word_line(value):
        return (
            has_cjk(value)
            and not value.startswith("[")
            and value not in headers
            and not value.startswith("词单")
            and all((0x3040 <= ord(ch) <= 0x30FF) or (0x3400 <= ord(ch) <= 0x9FFF) or ch in "々ヶー" for ch in value)
        )

    entries = []
    index = 0
    while index < len(lines) - 3:
        if (
            is_moji_index(lines[index])
            and is_kana_line(lines[index + 1])
            and is_word_line(lines[index + 2])
            and lines[index + 3].startswith("[")
        ):
            kana = clean_moji_text(lines[index + 1])
            word = clean_moji_text(lines[index + 2])
            cursor = index + 3
            meaning_parts = []
            while cursor < len(lines):
                is_next = (
                    cursor > index + 3
                    and cursor + 3 < len(lines)
                    and is_moji_index(lines[cursor])
                    and is_kana_line(lines[cursor + 1])
                    and is_word_line(lines[cursor + 2])
                    and lines[cursor + 3].startswith("[")
                )
                if is_next:
                    break
                if lines[cursor] not in headers and not lines[cursor].startswith("词单"):
                    meaning_parts.append(lines[cursor])
                cursor += 1
            meaning = clean_moji_text(" ".join(meaning_parts))
            noisy = sum(kana.count(ch) + word.count(ch) + meaning.count(ch) for ch in "O汉忘b}=rPst")
            leaked_entry = (
                bool(re.search(r"\s[0-9０-９伴兴]{1,4}\s+[ぁ-んァ-ンーa-zA-ZO汉忘b]{2,}\s+[\u3400-\u9fff]", meaning))
                or bool(re.search(r"\s[0-9０-９伴兴]{1,4}\s+\S{1,24}\s+\[", meaning))
                or "词单" in meaning
                or "序号 发音 单词 释义" in meaning
                or "考前对策" in meaning
                or "MOちゃん" in meaning
            )
            if (
                1 <= len(word) <= 14
                and 2 <= len(kana) <= 24
                and has_cjk(word)
                and has_kana(kana)
                and has_cjk(meaning)
                and noisy <= max(1, len(meaning) // 24)
                and not leaked_entry
            ):
                pos_match = re.match(r"\[([^\]]+)\]", meaning)
                part_of_speech = pos_match.group(1) if pos_match else "N2词汇"
                entries.append((word, kana, meaning[:180], part_of_speech[:20], "N2"))
            index = cursor
        else:
            index += 1
    dedup = {}
    for word, kana, meaning, part_of_speech, level in entries:
        dedup[(word, kana)] = (word, kana, meaning, part_of_speech, level)
    return list(dedup.values())


def make_moji_example(word, kana="", meaning="", part_of_speech=""):
    """Generate a varied study sentence for MOJi PDF entries without real examples."""
    templates = [
        "本文の中に「{word}」という表現が出てきました。",
        "先生は「{word}」の意味と読み方を説明しました。",
        "ノートに「{word}」を使った短い文を書きました。",
        "試験前に「{word}」の使い方を確認しました。",
        "ニュース記事で「{word}」という言葉を見つけました。",
        "会話の中で「{word}」を自然に使えるように練習します。",
        "この単語帳では「{word}」を重点語として覚えます。",
        "辞書で「{word}」の意味を調べて例文を読みました。",
    ]
    seed = sum(ord(ch) for ch in f"{word}{kana}{meaning}{part_of_speech}")
    return templates[seed % len(templates)].format(word=word)


def ensure_moji_pdf_deck(conn):
    pdfs = sorted(APP_DIR.glob("MOJi辞書 - 考前对策N2汉字（新整理版）*.pdf"))
    if not pdfs:
        return
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "insert or ignore into decks(name, source, created_at) values(?,?,?)",
        (MOJI_DECK_NAME, "本地 MOJi PDF 自动解析", now),
    )
    deck_id = cur.lastrowid or conn.execute("select id from decks where name=?", (MOJI_DECK_NAME,)).fetchone()[0]
    existing = conn.execute("select count(*) from vocabulary where deck_id=?", (deck_id,)).fetchone()[0]
    if existing >= 100:
        return
    rows = []
    for pdf in pdfs:
        for word, kana, meaning, part_of_speech, level in parse_moji_entries(extract_pdf_text(pdf)):
            rows.append((deck_id, word, kana, meaning, part_of_speech, level, make_moji_example(word, kana, meaning, part_of_speech), "MOJi N2", now))
    conn.executemany(
        """
        insert or ignore into vocabulary
        (deck_id, word, kana, meaning, part_of_speech, jlpt_level, example, tags, created_at)
        values(?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )


def seed_reviews(conn, deck_id):
    ids = [r[0] for r in conn.execute("select id from vocabulary where deck_id=?", (deck_id,))]
    now = datetime.now()
    rows = []
    for word_id in random.sample(ids, min(len(ids), 18)):
        for _ in range(random.randint(1, 4)):
            rating = random.choices(["认识", "模糊", "不认识"], weights=[5, 3, 2])[0]
            rows.append(
                (
                    word_id,
                    rating,
                    RATING_VALUE[rating],
                    (now - timedelta(days=random.randint(0, 18))).isoformat(timespec="seconds"),
                )
            )
    conn.executemany(
        "insert into reviews(word_id, rating, score, reviewed_at) values(?,?,?,?)",
        rows,
    )


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --paper:#F5F2EB;
            --paper-soft:#FAF8F5;
            --paper-deep:#ECE4D8;
            --ink:#2A2825;
            --muted:#6E665C;
            --line:#D8CDBF;
            --line-strong:#8F8171;
            --red:#8B2626;
            --blue:#4F6575;
            --green:#4E6246;
            --gold:#9A7A3F;
            --danger:#7C1F1F;
            --font-serif:"Noto Serif SC","Source Han Serif SC","Songti SC","SimSun",serif;
            --font-jp:"Yu Mincho","Yu Gothic","MS Mincho","Noto Serif JP",serif;
            --font-body:"Microsoft YaHei","Noto Sans SC","PingFang SC",sans-serif;
        }
        .stApp {
            background:
                linear-gradient(rgba(42,40,37,.025) 1px, transparent 1px),
                var(--paper);
            background-size: 100% 32px;
            color: var(--ink);
            font-family: var(--font-body);
        }
        .block-container { padding-top: 1.2rem; max-width: 1180px; }
        .editorial-masthead {
            border-top: 4px solid var(--ink);
            border-bottom: 1px solid var(--ink);
            margin: 4px 0 22px;
            padding: 18px 0 13px;
            text-align: center;
        }
        .editorial-kicker {
            color: var(--red);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .editorial-title {
            color: var(--ink);
            font-family: var(--font-serif);
            font-size: clamp(34px, 5vw, 58px);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.08;
            margin-top: 5px;
        }
        .editorial-subtitle {
            border-top: 1px solid var(--line);
            color: var(--muted);
            display: inline-block;
            font-size: 12px;
            letter-spacing: .08em;
            margin-top: 12px;
            padding-top: 8px;
        }
        [data-testid="stSidebar"] {
            background: var(--paper-deep);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebarCollapseButton"],
        button[title="Close sidebar"],
        button[title="Open sidebar"] {
            display: none !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
        [data-testid="stSidebar"] * {
            color: var(--ink) !important;
            font-family: var(--font-body);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            border-bottom: 1px solid var(--line-strong);
            font-family: var(--font-serif);
            font-size: 18px;
            letter-spacing: .08em;
            padding-bottom: 10px;
            text-align: center;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background: var(--paper-soft);
            color: var(--ink);
            border: 1px solid var(--line-strong);
            border-radius: 2px;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-left: 3px solid transparent;
            color: var(--ink) !important;
            margin: 2px 0;
            opacity: 1 !important;
            padding: 8px 8px 8px 12px;
            transition: background .12s ease, border-color .12s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(139,38,38,.06);
            border-left-color: var(--line-strong);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: rgba(139,38,38,.08);
            border-left-color: var(--red);
            font-weight: 800;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
            display: none;
        }
        div[role="radiogroup"] label,
        div[role="radiogroup"] label *,
        div[role="radiogroup"] p {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        .stRadio [data-testid="stMarkdownContainer"] p {
            color: var(--ink) !important;
        }
        .stTextArea label,
        .stTextInput label,
        .stSelectbox label,
        .stSlider label,
        .stCheckbox label,
        .stFileUploader label {
            color: var(--ink) !important;
            font-weight: 700;
            opacity: 1 !important;
        }
        .stCheckbox label *,
        .stCheckbox [data-testid="stMarkdownContainer"] p,
        .stFileUploader [data-testid="stMarkdownContainer"] p,
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stWidgetLabel"] * {
            color: var(--ink) !important;
            opacity: 1 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--paper-soft) !important;
            border: 1px dashed var(--line-strong) !important;
            border-radius: 2px !important;
        }
        textarea, input {
            background: var(--paper-soft) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line-strong) !important;
            border-radius: 2px !important;
        }
        .stDataFrame, [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 2px;
            overflow: hidden;
        }
        h1, h2, h3 {
            color: var(--ink);
            font-family: var(--font-serif);
            letter-spacing: 0;
        }
        h2, h3 {
            border-bottom: 1px solid var(--line);
            padding-bottom: .35rem;
        }
        hr { border-color: var(--line); }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--ink);
        }
        .metric-card {
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 2px;
            padding: 18px 18px 14px;
            box-shadow: none;
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            letter-spacing: .08em;
        }
        .metric-value {
            color: var(--ink);
            font-family: var(--font-serif);
            font-size: 30px;
            font-weight: 800;
            margin-top: 6px;
        }
        [data-testid="stMetric"] {
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 2px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-size: 12px;
            letter-spacing: .08em;
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: var(--font-serif);
        }
        .word-card {
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-left: 4px solid var(--green);
            border-radius: 2px;
            padding: 24px;
            margin: 10px 0 16px;
        }
        .jp-word {
            color: var(--ink);
            font-family: var(--font-jp);
            font-size: 46px;
            font-weight: 750;
            line-height: 1.2;
        }
        .kana { color: var(--blue); font-size: 22px; margin-top: 6px; }
        .meaning { font-size: 20px; margin-top: 16px; }
        .subtle { color: var(--muted); font-size: 14px; }
        .pill {
            display:inline-block;
            border:1px solid var(--line);
            border-radius:2px;
            padding:4px 9px;
            margin-right:6px;
            background:var(--paper-soft);
            color:var(--green);
            font-size:13px;
        }
        .editorial-page {
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 2px;
            padding: 22px 24px;
            min-height: 280px;
        }
        .editorial-page.right {
            border-left: 1px solid var(--line-strong);
        }
        .editorial-label {
            color: var(--red);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .editorial-list-row {
            align-items: baseline;
            border-bottom: 1px solid var(--line);
            display: grid;
            gap: 16px;
            grid-template-columns: 64px 1fr auto;
            padding: 14px 0;
        }
        .editorial-list-no {
            color: var(--red);
            font-family: var(--font-serif);
            font-size: 30px;
            font-weight: 800;
            line-height: 1;
        }
        .editorial-list-word {
            color: var(--ink);
            font-family: var(--font-jp);
            font-size: 22px;
            font-weight: 800;
        }
        .editorial-list-meta {
            color: var(--muted);
            font-size: 13px;
            margin-top: 3px;
        }
        .stButton > button {
            background: var(--paper-soft) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line-strong) !important;
            border-radius: 2px !important;
            box-shadow: none !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover {
            background: #EFE6D9 !important;
            border-color: var(--red) !important;
            color: var(--ink) !important;
        }
        button[kind="primary"], .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: var(--ink) !important;
            color: var(--paper-soft) !important;
            border-color: var(--ink) !important;
        }
        button[kind="primary"] *, .stButton > button[kind="primary"] *, .stFormSubmitButton > button[kind="primary"] * {
            color: var(--paper-soft) !important;
            opacity: 1 !important;
        }
        .translation-result {
            background: var(--paper-soft);
            border: 1px solid var(--line-strong);
            border-radius: 2px;
            color: var(--ink);
            font-size: 16px;
            line-height: 1.8;
            min-height: 150px;
            padding: 14px 16px;
            white-space: pre-wrap;
        }
        .translation-result.muted {
            color: var(--muted);
        }
        div[data-baseweb="select"] > div {
            background: var(--paper-soft);
            color: var(--ink);
            border-color: var(--line-strong);
            border-radius: 2px;
        }
        .stAlert {
            border-radius: 2px;
            border: 1px solid var(--line);
        }
        div[data-testid="stExpander"] {
            background: var(--paper-soft);
            border: 1px solid var(--line);
            border-radius: 2px;
        }
        div[data-testid="stTabs"] button {
            color: var(--muted) !important;
            font-weight: 700;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--red) !important;
        }
        .stSlider [data-baseweb="slider"] * {
            color: var(--ink) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_decks():
    return query_df("select id, name, source from decks order by id")


def today_learned_count(deck_id):
    today = date.today().isoformat()
    learned = query_df(
        """
        select count(distinct r.word_id) as count
        from reviews r
        join vocabulary v on v.id = r.word_id
        where v.deck_id=? and r.score >= 0.95 and date(r.reviewed_at)=?
        """,
        (deck_id, today),
    )
    return int(learned.iloc[0]["count"]) if not learned.empty else 0


def deck_stats(deck_id, daily_goal):
    vocab = query_df("select * from vocabulary where deck_id=?", (deck_id,))
    reviews = query_df(
        """
        select v.id,
               coalesce((
                   select r2.score
                   from reviews r2
                   where r2.word_id = v.id
                   order by r2.reviewed_at desc, r2.id desc
                   limit 1
               ), 0) as last_score
        from vocabulary v
        where v.deck_id=?
        """,
        (deck_id,),
    )
    total = len(vocab)
    mastered = int((reviews["last_score"] >= 0.95).sum()) if total else 0
    pending = max(total - mastered, 0)
    due = max(min(daily_goal, pending) - today_learned_count(deck_id), 0)
    return total, mastered, pending, due


def build_features(deck_id=None):
    where = "where v.deck_id=?" if deck_id else ""
    params = (deck_id,) if deck_id else ()
    vocab = query_df(f"select v.* from vocabulary v {where}", params)
    reviews = query_df("select * from reviews")
    rows = []
    now = datetime.now()
    for _, word in vocab.iterrows():
        wr = reviews[reviews["word_id"] == word["id"]].sort_values("reviewed_at")
        if wr.empty:
            review_count = 0
            avg_score = 0.0
            last_score = 0.0
            interval_days = 365
        else:
            review_count = len(wr)
            avg_score = float(wr["score"].mean())
            last_score = float(wr.iloc[-1]["score"])
            last_dt = datetime.fromisoformat(wr.iloc[-1]["reviewed_at"])
            interval_days = max((now - last_dt).days, 0)
        level_num = LEVEL_WEIGHT.get(word["jlpt_level"], 1)
        rows.append(
            {
                "word_id": int(word["id"]),
                "word": word["word"],
                "kana": word["kana"],
                "meaning": word["meaning"],
                "part_of_speech": word["part_of_speech"],
                "jlpt_level": word["jlpt_level"],
                "level_num": level_num,
                "review_count": review_count,
                "avg_score": avg_score,
                "last_score": last_score,
                "interval_days": interval_days,
                "days_x_level": interval_days * level_num,
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def train_memory_model():
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(900):
        level_num = int(rng.integers(1, 6))
        review_count = int(rng.integers(0, 9))
        avg_score = float(rng.beta(2.2, 1.8))
        last_score = float(np.clip(avg_score + rng.normal(0, 0.18), 0, 1))
        interval_days = int(rng.integers(0, 45))
        stability = 1.55 * avg_score + 1.05 * last_score + 0.24 * review_count
        difficulty = 0.36 * level_num + 0.065 * interval_days
        p = 1 / (1 + math.exp(-(stability - difficulty + 0.15)))
        outcome = int(rng.random() < p)
        rows.append([level_num, review_count, avg_score, last_score, interval_days, interval_days * level_num, outcome])
    df = pd.DataFrame(
        rows,
        columns=["level_num", "review_count", "avg_score", "last_score", "interval_days", "days_x_level", "memory_outcome"],
    )
    x = df.drop(columns=["memory_outcome"])
    y = df["memory_outcome"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7, stratify=y)
    lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
    rf = RandomForestClassifier(n_estimators=120, max_depth=7, random_state=7)
    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)
    scores = []
    for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
        pred = model.predict(x_test)
        scores.append({"模型": name, "Accuracy": accuracy_score(y_test, pred), "F1": f1_score(y_test, pred)})
    return rf, pd.DataFrame(scores)


def add_memory_probability(df):
    if df.empty:
        return df
    model, _ = train_memory_model()
    cols = ["level_num", "review_count", "avg_score", "last_score", "interval_days", "days_x_level"]
    out = df.copy()
    out["memory_probability"] = model.predict_proba(out[cols])[:, 1]
    out["forget_risk"] = 1 - out["memory_probability"]
    return out


def recommend_words(deck_id, limit=8):
    features = build_features(deck_id)
    if features.empty:
        return features
    scored = add_memory_probability(features)
    scored["priority"] = (
        scored["forget_risk"] * 0.55
        + (scored["interval_days"].clip(0, 30) / 30) * 0.25
        + (1 / (scored["review_count"] + 1)) * 0.20
    )
    return scored.sort_values(["priority", "level_num"], ascending=[False, False]).head(limit)


def next_recommendation(deck_id, limit):
    recs = recommend_words(deck_id, limit)
    if recs.empty:
        return recs
    unmastered = recs[recs["last_score"] < 0.95]
    return unmastered if not unmastered.empty else recs


def speak_button(text, key, reading=None):
    speech_text = str(reading).strip() if reading and str(reading).strip() else str(text)
    safe_text = json.dumps(speech_text, ensure_ascii=False)
    components.html(
        f"""
        <div style="height:46px;display:flex;align-items:center;overflow:visible;">
        <button id="speak-button" type="button" style="box-sizing:border-box;border:1px solid #8F8171;border-radius:2px;padding:8px 14px;background:#FAF8F5;color:#2A2825;cursor:pointer;font:inherit;font-weight:700;line-height:1.2;min-height:36px;">
            发音
        </button>
        <span id="speak-status" style="margin-left:8px;color:#8b7f70;font-size:12px;"></span>
        </div>
        <script>
        const speechText = {safe_text};
        const button = document.getElementById("speak-button");
        const status = document.getElementById("speak-status");

        function findSpeechTarget() {{
            const candidates = [window.parent, window];
            for (const target of candidates) {{
                try {{
                    if (target && "speechSynthesis" in target && "SpeechSynthesisUtterance" in target) {{
                        return target;
                    }}
                }} catch (error) {{}}
            }}
            return null;
        }}

        function getJapaneseVoice(synthesis) {{
            const voices = synthesis.getVoices();
            return voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith("ja")) || null;
        }}

        function speakJapanese(event) {{
            event.preventDefault();
            const target = findSpeechTarget();
            if (!target) {{
                status.textContent = "当前浏览器不支持发音";
                return;
            }}
            const synthesis = target.speechSynthesis;
            const utterance = new target.SpeechSynthesisUtterance(speechText);
            utterance.lang = "ja-JP";
            utterance.rate = 0.85;
            const voice = getJapaneseVoice(synthesis);
            if (voice) {{
                utterance.voice = voice;
            }}
            utterance.onstart = () => status.textContent = "";
            utterance.onerror = () => status.textContent = "发音失败，请确认浏览器允许语音播放";
            synthesis.cancel();
            synthesis.speak(utterance);
            if (typeof synthesis.resume === "function") {{
                synthesis.resume();
            }}
        }}

        button.addEventListener("click", speakJapanese);
        const target = findSpeechTarget();
        if (target) {{
            target.speechSynthesis.onvoiceschanged = () => target.speechSynthesis.getVoices();
        }}
        </script>
        """,
        height=56,
    )


def render_metrics(deck_id, daily_goal):
    total, mastered, pending, due = deck_stats(deck_id, daily_goal)
    cols = st.columns(4)
    labels = [("词汇总量", total), ("已掌握", mastered), ("待学习", pending), ("今日建议复习", due)]
    for col, (label, value) in zip(cols, labels):
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )


def page_learn(deck_id):
    st.subheader("智能背词")
    daily_goal = int(st.session_state.get("daily_goal", 10))
    render_metrics(deck_id, daily_goal)

    # ---- "全部单词" 模式切换 ----
    if "show_all_words" not in st.session_state:
        st.session_state.show_all_words = False

    total_count = int(deck_stats(deck_id, daily_goal)[0])
    show_all = st.session_state.show_all_words
    limit = total_count if show_all else daily_goal

    recs = next_recommendation(deck_id, limit)
    if recs.empty:
        st.info("当前词库没有词汇。请先导入或选择其他词库。")
        return
    if "current_word_id" not in st.session_state or st.session_state.current_word_id not in recs["word_id"].tolist():
        st.session_state.current_word_id = int(recs.iloc[0]["word_id"])
    options = {f"{r.word} / {r.kana}": int(r.word_id) for r in recs.itertuples()}
    selected_label = st.selectbox("推荐队列", list(options.keys()), index=list(options.values()).index(st.session_state.current_word_id))
    st.session_state.current_word_id = options[selected_label]

    # 全部单词切换按钮 + 当前模式提示
    col_mode, col_info, _ = st.columns([1.2, 1, 2])
    with col_mode:
        if show_all:
            if st.button("📋 恢复推荐", use_container_width=True):
                st.session_state.show_all_words = False
                st.rerun()
        else:
            if st.button("📚 全部单词", use_container_width=True):
                st.session_state.show_all_words = True
                st.rerun()
    with col_info:
        if show_all:
            st.caption(f"全部 {total_count} 词 | 按遗忘风险排序")
        else:
            st.caption(f"今日推荐 {daily_goal} 词 | 滑动侧边栏调整数量")

    row = recs[recs["word_id"] == st.session_state.current_word_id].iloc[0]
    vocab = query_df("select * from vocabulary where id=?", (int(row["word_id"]),)).iloc[0]
    left_page, right_page = st.columns([5, 3], gap="large")
    with left_page:
        st.markdown(
            f"""
            <div class='editorial-page'>
                <div class='editorial-label'>Vocabulary</div>
                <div class='jp-word'>{vocab['word']}</div>
                <div class='kana'>{vocab['kana']}</div>
                <div style='margin-top:24px;'>
                    <span class='pill'>{vocab['jlpt_level']}</span>
                    <span class='pill'>{vocab['part_of_speech']}</span>
                    <span class='pill'>遗忘风险 {row['forget_risk']:.0%}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        speak_button(vocab["word"], f"word-{vocab['id']}", vocab["kana"])
    with right_page:
        st.markdown(
            f"""
            <div class='editorial-page right'>
                <div class='editorial-label'>Meaning & Example</div>
                <div class='meaning'>{vocab['meaning']}</div>
                <p>{vocab['example']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        wrong_exists = query_df("select 1 from wrong_book where word_id=?", (int(vocab["id"]),))
        if wrong_exists.empty:
            if st.button("加入错题本", use_container_width=True):
                add_to_wrong_book(int(vocab["id"]))
                st.toast("已加入错题本")
                st.rerun()
        else:
            st.caption("已在错题本中")
        cols = st.columns(3)
        for col, rating in zip(cols, ["认识", "模糊", "不认识"]):
            if col.button(rating, key=f"rate-{rating}", use_container_width=True):
                run_sql(
                    "insert into reviews(word_id, rating, score, reviewed_at) values(?,?,?,?)",
                    (int(vocab["id"]), rating, RATING_VALUE[rating], datetime.now().isoformat(timespec="seconds")),
                )
                st.toast(f"已记录：{rating}")
                next_words = next_recommendation(deck_id, limit)
                if not next_words.empty:
                    st.session_state.current_word_id = int(next_words.iloc[0]["word_id"])
                st.rerun()
    with st.expander("报错 / 建议"):
        msg = st.text_area("如果读音、释义或例句有问题，可以记录在这里。")
        if st.button("提交反馈"):
            if msg.strip():
                run_sql(
                    "insert into feedback(word_id, message, created_at) values(?,?,?)",
                    (int(vocab["id"]), msg.strip(), datetime.now().isoformat(timespec="seconds")),
                )
                st.success("反馈已保存，管理员页可查看。")


def load_test_vocab(deck_id):
    vocab = query_df("select * from vocabulary where deck_id=?", (deck_id,))
    if len(vocab) < 60:
        extra = query_df(
            """
            select v.*
            from vocabulary v
            join decks d on d.id = v.deck_id
            where d.name in ('综合高频100词库', '高频基础词库', 'JLPT N4-N3 扩展词库', '新闻阅读高频词库')
            """
        )
        vocab = pd.concat([vocab, extra], ignore_index=True).drop_duplicates(subset=["id"])
    return vocab


def jp_surface(row, prefer="mixed"):
    if prefer == "kana":
        return str(row["kana"])
    if prefer == "word":
        return str(row["word"])
    return f"{row['word']} / {row['kana']}"


def sample_choices(vocab, row, column, count=3):
    pool = vocab[vocab["id"] != row["id"]][column].dropna().astype(str).unique().tolist()
    if len(pool) < count:
        return pool
    return random.sample(pool, count)


def make_question(deck_id, seen_word_ids=None):
    seen_word_ids = set(seen_word_ids or [])
    vocab = load_test_vocab(deck_id)
    if len(vocab) < 4:
        return None
    available = vocab[~vocab["id"].isin(seen_word_ids)]
    if len(available) < 4:
        seen_word_ids.clear()
        available = vocab
    recent = query_df("select word_id, avg(is_correct) as acc from test_results group by word_id")
    merged = available.merge(recent, left_on="id", right_on="word_id", how="left")
    merged["acc"] = merged["acc"].astype(float).fillna(0.5)
    merged["weight"] = 1.4 - merged["acc"] + merged["jlpt_level"].map(LEVEL_WEIGHT).fillna(1) * 0.08
    row = merged.sample(1, weights=merged["weight"], random_state=random.randint(1, 999999)).iloc[0]
    qtype = random.choice(["日译中", "中译日", "假名识别", "词形识别"])
    if qtype == "日译中":
        prompt = f"{jp_surface(row)} 的中文意思是？"
        answer = row["meaning"]
        pool = sample_choices(vocab, row, "meaning")
    else:
        if qtype == "中译日":
            prompt = f"“{row['meaning']}” 对应的日语表达是？"
            answer = jp_surface(row)
            sample = vocab[vocab["id"] != row["id"]].sample(min(3, len(vocab) - 1))
            pool = [jp_surface(item) for _, item in sample.iterrows()]
        elif qtype == "假名识别":
            prompt = f"{row['word']} 的读音是假名哪一个？"
            answer = row["kana"]
            pool = sample_choices(vocab, row, "kana")
        else:
            prompt = f"{row['kana']} / {row['meaning']} 对应的词形是哪一个？"
            answer = row["word"]
            pool = sample_choices(vocab, row, "word")
    choices = pool + [answer]
    choices = list(dict.fromkeys(str(choice) for choice in choices if str(choice).strip()))
    while len(choices) < 4:
        fallback = jp_surface(vocab.sample(1).iloc[0])
        if fallback not in choices:
            choices.append(fallback)
    random.shuffle(choices)
    return {"word_id": int(row["id"]), "prompt": prompt, "answer": answer, "choices": choices, "qtype": qtype}


LEVEL_ORDER = ["N5", "N4", "N3", "N2", "N1"]


ABILITY_ADVICE = {
    "N5": [
        "先稳定掌握基础名词、时间、地点和日常动作，每天复习 20 个 N5 高频词。",
        "开始把单词放进短句中记忆，重点练习「名词 + です」「动词ます形」。",
        "增加中译日训练，避免只会看懂不会主动回忆。",
        "把错误词按生活场景分类，例如学校、购物、交通、饮食。",
        "开始少量接触 N4 词，保持 N5 正确率在 80% 以上再提升难度。",
    ],
    "N4": [
        "巩固基础动词变形和常见副词，减少意思相近词的混淆。",
        "用例句区分自动词/他动词，错题本优先复习动词类错误。",
        "加入 N4 阅读短文训练，把词汇和语境绑定。",
        "对抽象名词建立中文释义和日语例句的双向联系。",
        "开始混入 N3 高频词，自测中 N4 题稳定后逐步提高难度。",
    ],
    "N3": [
        "重点补齐抽象名词、复合动词和常见书面表达。",
        "每次复习后用一个例句复述单词用法，减少只记中文释义。",
        "统计词性错误，针对最高错误词性做专项 15 分钟练习。",
        "阅读新闻简文或教材文章，标记反复出现但不熟的词。",
        "逐步加入 N2 词汇，优先学习高频、可在阅读中复现的词。",
    ],
    "N2": [
        "提高对抽象概念词、新闻词和学术表达的辨析能力。",
        "将近义词分组比较，例如「検討/確認/判断/評価」。",
        "对错误词写出一个日语解释或搭配，提升深层掌握度。",
        "增加长句阅读，训练从上下文推断词义。",
        "开始少量接触 N1 词，但保持 N2 高频词的稳定正确率。",
    ],
    "N1": [
        "重点处理低频抽象词、书面词和语义细微差别。",
        "建立同义/反义/搭配网络，而不是孤立背单词。",
        "用真实文章复盘错词，记录其上下文和固定搭配。",
        "加强 N1 词在复杂句中的理解，避免只认识单词本释义。",
        "定期回测 N2/N1 混合词，防止高级词学习挤压基础稳定性。",
    ],
}


POS_ADVICE = {
    "名词": "名词错误偏高：建议按主题建立词群，例如学习、社会、新闻、抽象概念，并用例句记搭配。",
    "动词": "动词错误偏高：建议重点复习自他动词、サ变动词和常见搭配，做中译日主动回忆。",
    "形容词": "形容词错误偏高：建议把い形容词/な形容词分开复习，并用反义词和程度副词一起记。",
    "其他": "其他词性错误偏高：建议补充副词、接续表达和固定说法，结合短文语境复习。",
}


def pos_group(part_of_speech):
    text = str(part_of_speech)
    if "动" in text or "五段" in text or "一段" in text or "サ变" in text:
        return "动词"
    if "形" in text:
        return "形容词"
    if "名" in text:
        return "名词"
    return "其他"


def estimate_ability(history):
    if not history:
        return None
    df = pd.DataFrame(history)
    df["level_num"] = df["jlpt_level"].map(LEVEL_WEIGHT).fillna(1).astype(float)
    total = len(df)
    accuracy = float(df["is_correct"].mean())
    weighted_accuracy = float((df["is_correct"] * df["level_num"]).sum() / df["level_num"].sum())
    attempted_level = float(df["level_num"].mean())
    ability_score = attempted_level + (weighted_accuracy - 0.55) * 2.0
    ability_score = max(1.0, min(5.0, ability_score))
    estimated_level = "N5" if ability_score < 1.7 else "N4" if ability_score < 2.5 else "N3" if ability_score < 3.3 else "N2" if ability_score < 4.2 else "N1"
    stage = max(1, min(5, int((ability_score - math.floor(ability_score)) * 5) + 1))
    if accuracy < 0.45:
        stage = 1
    elif accuracy >= 0.85:
        stage = min(5, stage + 1)
    wrong = df[df["is_correct"] == 0].copy()
    if wrong.empty:
        weak_pos = []
    else:
        wrong["pos_group"] = wrong["part_of_speech"].apply(pos_group)
        weak_pos = wrong["pos_group"].value_counts().head(2).index.tolist()
    suggestions = [ABILITY_ADVICE[estimated_level][stage - 1]]
    suggestions.extend(POS_ADVICE[pos] for pos in weak_pos)
    if weighted_accuracy < 0.55:
        suggestions.append("当前加权正确率偏低：建议降低一个难度层级复习 1-2 轮，再回到当前级别测试。")
    elif weighted_accuracy > 0.82 and estimated_level != "N1":
        suggestions.append("当前加权正确率较高：可以适当增加下一等级词汇，但错题仍要当天复盘。")
    return {
        "total": total,
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "ability_score": ability_score,
        "estimated_level": estimated_level,
        "stage": stage,
        "weak_pos": weak_pos,
        "suggestions": suggestions,
    }


def _render_test_analysis(history):
    """渲染自测答题数据分析（题型正确率、等级正确率、薄弱词性）"""
    if not history:
        return
    df = pd.DataFrame(history)

    st.markdown("#### 按题型正确率")
    qtype_stats = df.groupby("question_type")["is_correct"].agg(["mean", "count"]).reset_index()
    qtype_stats.columns = ["题型", "正确率", "答题数"]
    qtype_stats["正确率"] = qtype_stats["正确率"].round(2)
    st.dataframe(qtype_stats, use_container_width=True, hide_index=True)

    st.markdown("#### 按 JLPT 等级正确率")
    level_stats = df.groupby("jlpt_level")["is_correct"].agg(["mean", "count"]).reset_index()
    level_stats.columns = ["JLPT等级", "正确率", "答题数"]
    level_stats["正确率"] = level_stats["正确率"].round(2)
    level_order = [l for l in LEVEL_ORDER if l in level_stats["JLPT等级"].tolist()]
    if level_order:
        level_stats["sort"] = level_stats["JLPT等级"].apply(lambda x: level_order.index(x) if x in level_order else 99)
        level_stats = level_stats.sort_values("sort").drop(columns=["sort"])
    st.dataframe(level_stats, use_container_width=True, hide_index=True)

    if len(level_stats) >= 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [EDITORIAL_GREEN if v >= 0.6 else EDITORIAL_RED for v in level_stats["正确率"]]
        ax.bar(level_stats["JLPT等级"], level_stats["正确率"], color=colors)
        ax.set_ylabel("正确率")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.6, color=EDITORIAL_GOLD, linestyle="--", alpha=0.6, label="60% 基准线")
        ax.legend(fontsize=9)
        apply_editorial_chart_style(ax)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown("#### 薄弱词性")
    wrong_df = df[df["is_correct"] == 0]
    if not wrong_df.empty:
        wrong_df = wrong_df.copy()
        wrong_df["pos_group"] = wrong_df["part_of_speech"].apply(pos_group)
        pos_err = wrong_df["pos_group"].value_counts().reset_index()
        pos_err.columns = ["词性类别", "错误次数"]
        pos_err["占比"] = (pos_err["错误次数"] / pos_err["错误次数"].sum()).round(2)
        st.dataframe(pos_err, use_container_width=True, hide_index=True)
        worst_pos = pos_err.iloc[0]["词性类别"]
        advice_map = POS_ADVICE.copy()
        advice_map["其他"] = "副词、接续词等错误偏高：建议结合短文语境记忆，注意固定表达和搭配。"
        st.info(advice_map.get(worst_pos, "建议针对薄弱词性进行专项复习。"))
    else:
        st.success("暂无错题数据。")


def render_ability_report(history):
    report = estimate_ability(history)
    if not report:
        return
    st.markdown("### 能力评估与学习建议")
    cols = st.columns(4)
    metrics = [
        ("答题数", report["total"]),
        ("普通正确率", f"{report['accuracy']:.0%}"),
        ("难度加权正确率", f"{report['weighted_accuracy']:.0%}"),
        ("估计水平", f"{report['estimated_level']} 第{report['stage']}阶段"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )
    if report["weak_pos"]:
        st.write("主要薄弱词性：" + "、".join(report["weak_pos"]))
    st.markdown("#### 针对性建议")
    for item in report["suggestions"]:
        st.markdown(f"- {item}")


def page_test(deck_id):
    st.subheader("自适应词汇自测")
    reset_keys = [
        "question",
        "question_deck_id",
        "test_count",
        "test_correct",
        "level_points",
        "test_history",
        "test_seen_words",
        "last_answer",
    ]
    if st.button("重新开始自测"):
        for key in reset_keys:
            st.session_state.pop(key, None)
        st.rerun()
    if "question" not in st.session_state or st.session_state.get("question_deck_id") != deck_id:
        st.session_state.test_seen_words = []
        st.session_state.question = make_question(deck_id, st.session_state.test_seen_words)
        st.session_state.question_deck_id = deck_id
        st.session_state.test_count = 0
        st.session_state.test_correct = 0
        st.session_state.level_points = 0.0
        st.session_state.test_history = []
        st.session_state.last_answer = None
    if st.session_state.get("test_count", 0) >= 50:
        st.info("本轮自测已达到 50 题上限，系统已自动开始新一轮。")
        for key in reset_keys:
            st.session_state.pop(key, None)
        st.rerun()
    q = st.session_state.question
    if not q:
        st.info("至少需要 4 个词才能生成选择题。")
        return
    test_left, test_right = st.columns([5, 3], gap="large")
    with test_left:
        st.markdown(
            f"""
            <div class='editorial-page'>
                <div class='editorial-label'>Adaptive Test</div>
                <div class='jp-word'>{q['prompt']}</div>
                <div class='subtle'>本轮题目不会重复出现同一个单词，答对后自动进入下一题。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with test_right:
        st.markdown("<div class='editorial-label'>Answer Sheet</div>", unsafe_allow_html=True)
        choice = st.radio("选择答案", q["choices"], label_visibility="collapsed")
        if st.session_state.get("last_answer") and st.session_state.last_answer.get("word_id") == q["word_id"]:
            last = st.session_state.last_answer
            if last["ok"]:
                st.success("回答正确，正在进入下一题。")
            else:
                st.error(f"回答错误，正确答案：{last['answer']}")
        if st.button("提交答案", type="primary"):
            ok = int(choice == q["answer"])
            st.session_state.test_count += 1
            st.session_state.test_correct += ok
            word_meta = query_df("select word, kana, meaning, part_of_speech, jlpt_level from vocabulary where id=?", (q["word_id"],))
            level_num = LEVEL_WEIGHT.get(word_meta.iloc[0]["jlpt_level"], 1) if not word_meta.empty else 1
            st.session_state.level_points += (level_num if ok else -0.35 * level_num)
            if not word_meta.empty:
                meta = word_meta.iloc[0]
                st.session_state.test_history.append(
                    {
                        "word_id": q["word_id"],
                        "word": meta["word"],
                        "kana": meta["kana"],
                        "meaning": meta["meaning"],
                        "part_of_speech": meta["part_of_speech"],
                        "jlpt_level": meta["jlpt_level"],
                        "is_correct": ok,
                        "question_type": q["qtype"],
                    }
                )
            if q["word_id"] not in st.session_state.test_seen_words:
                st.session_state.test_seen_words.append(q["word_id"])
            run_sql(
                "insert into test_results(word_id, is_correct, question_type, answered_at) values(?,?,?,?)",
                (q["word_id"], ok, q["qtype"], datetime.now().isoformat(timespec="seconds")),
            )
            if ok:
                st.success("回答正确。")
                st.session_state.last_answer = {"word_id": q["word_id"], "ok": True, "answer": q["answer"]}
                time.sleep(1)
                if st.session_state.test_count >= 50:
                    for key in reset_keys:
                        st.session_state.pop(key, None)
                else:
                    st.session_state.question = make_question(deck_id, st.session_state.test_seen_words)
                    st.session_state.last_answer = None
                st.rerun()
            else:
                st.error(f"回答错误，正确答案：{q['answer']}")
                st.session_state.last_answer = {"word_id": q["word_id"], "ok": False, "answer": q["answer"]}
    total = st.session_state.get("test_count", 0)
    correct = st.session_state.get("test_correct", 0)
    if total:
        acc = correct / total
        st.progress(acc)
        st.write(f"本轮进度：{total} 题，正确率：{acc:.0%}")
        if total < 8:
            st.info(f"至少回答 8 题后才会给出水平评级。还需要 {8 - total} 题。")
        else:
            avg_level = max(st.session_state.level_points / total, 0)
            adjusted = avg_level + (acc - 0.55) * 1.7
            estimated = "N5" if adjusted < 1.7 else "N4" if adjusted < 2.5 else "N3" if adjusted < 3.3 else "N2" if adjusted < 4.2 else "N1"
            trend = "上调" if acc >= 0.75 else "下调" if acc < 0.5 else "保持观察"
            st.success(f"近似词汇水平：{estimated}。系统会根据后续答题表现继续{trend}难度。")
        if total < 20:
            st.info(f"数据分析模块将在 20 题后生成能力评估与学习建议。还需要 {20 - total} 题。")
        else:
            render_ability_report(st.session_state.get("test_history", []))
        if total >= 10:
            with st.expander("答题数据分析", expanded=(total >= 20)):
                _render_test_analysis(st.session_state.get("test_history", []))
    if st.button("下一题"):
        st.session_state.question = make_question(deck_id, st.session_state.get("test_seen_words", []))
        st.session_state.last_answer = None
        st.rerun()


def japanese_tokens(text):
    tokens = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]{2,}", text)
    stop = {"です", "ます", "する", "した", "して", "から", "こと", "これ", "それ", "ため", "よう", "ある", "いる"}
    return [t for t in tokens if t not in stop]


def tokenize_with_janome(text):
    """使用 Janome 进行日语分词，返回 [{'surface': , 'base': , 'pos': , 'reading': }, ...]"""
    from janome.tokenizer import Tokenizer

    t = Tokenizer()
    results = []
    for token in t.tokenize(text):
        surface = token.surface
        base = token.base_form
        pos = token.part_of_speech.split(",")[0]
        reading = token.reading if token.reading else surface
        if pos not in ("助詞", "助動詞", "記号", "フィラー"):
            results.append({"surface": surface, "base": base, "pos": pos, "reading": reading})
    return results


def extract_ngrams(token_list, n=2):
    """从 Janome token 列表中提取 n-gram，返回 Counter"""
    surfaces = [t["surface"] for t in token_list]
    return Counter(zip(*(surfaces[i:] for i in range(n))))


def page_frequency(deck_id):
    st.subheader("高频词汇分析")
    vocab = query_df("select word from vocabulary where deck_id=?", (deck_id,))
    known_set = set(vocab["word"].tolist())
    levels = ["全部", "N5", "N4", "N3", "N2", "N1"]
    cols = st.columns([1, 2])
    level_filter = cols[0].selectbox("按 JLPT 难度筛选", levels)
    candidates = HIGH_FREQ_WORDS if level_filter == "全部" else [row for row in HIGH_FREQ_WORDS if row["jlpt_level"] == level_filter]
    options = [f"{row['rank']:03d}. {row['word']} / {row['kana']} / {row['meaning']}" for row in candidates]
    selected = cols[1].selectbox("选择一个高频词", options)
    row = candidates[options.index(selected)]
    st.markdown(
        f"""
        <div class='word-card'>
            <div class='jp-word'>{row['word']}</div>
            <div class='kana'>{row['kana']}</div>
            <div class='meaning'>{row['meaning']}</div>
            <p>{row['usage']}</p>
            <span class='pill'>频序 #{row['rank']}</span>
            <span class='pill'>{row['jlpt_level']}</span>
            <span class='pill'>{row['part_of_speech']}</span>
            <span class='pill'>当前词库：{"已包含" if row['word'] in known_set else "未包含"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    speak_button(row["word"], f"freq-{row['rank']}", row["kana"])
    st.markdown("### 常见用法例句")
    for example in row["examples"]:
        st.markdown(f"- {example}")
    overview = pd.DataFrame(
        [
            {
                "频序": item["rank"],
                "词": item["word"],
                "假名": item["kana"],
                "含义": item["meaning"],
                "JLPT": item["jlpt_level"],
                "词性": item["part_of_speech"],
                "当前词库": "是" if item["word"] in known_set else "否",
            }
            for item in candidates
        ]
    )
    st.markdown("### 高频词总览")
    st.dataframe(overview, use_container_width=True, hide_index=True)
    default_text = "日本語を勉強するために、毎日単語を読みます。復習を継続すると、記憶を維持できます。学習データを分析して、効率を改善します。"
    with st.expander("分析你自己的日语文本"):
        text = st.text_area("粘贴日语文章或教材片段", value=default_text, height=140)
        try:
            janome_tokens = tokenize_with_janome(text)
            st.caption("分词引擎：Janome（形态素解析）")
        except ImportError:
            st.warning("建议安装 janome 获得更准确的分词和 N-gram 分析：`pip install janome`")
            raw_tokens = japanese_tokens(text)
            janome_tokens = [{"surface": t, "base": t, "pos": "", "reading": t} for t in raw_tokens]

        tokens = [t["surface"] for t in janome_tokens]
        counts = Counter(tokens)
        if counts:
            freq = pd.DataFrame(counts.most_common(30), columns=["词", "频次"])
            freq["是否在当前词库"] = freq["词"].apply(lambda x: "是" if x in known_set else "否")
            st.markdown("#### 词频统计")
            st.dataframe(freq, use_container_width=True, hide_index=True)
        else:
            st.warning("没有识别到足够的日语词。")
            return

        if len(janome_tokens) >= 4:
            st.markdown("#### N-gram 词组搭配分析")
            ng_choice = st.radio("N-gram 类型", ["二元词组 (Bigram)", "三元词组 (Trigram)"], horizontal=True)
            n_val = 2 if "二元" in ng_choice else 3
            ngrams = extract_ngrams(janome_tokens, n_val)
            if ngrams:
                top_ngrams = ngrams.most_common(20)
                ng_rows = []
                for ng_tuple, count in top_ngrams:
                    phrase = "".join(ng_tuple) if n_val == 2 else "".join(ng_tuple)
                    unknown_words = [w for w in ng_tuple if w not in known_set]
                    ng_rows.append({
                        "词组": phrase,
                        "拆分为": " + ".join(ng_tuple),
                        "频次": count,
                        "含生词": f"{len(unknown_words)}个" if unknown_words else "无",
                    })
                st.dataframe(pd.DataFrame(ng_rows), use_container_width=True, hide_index=True)
                unknown_ngrams = [r for r in ng_rows if r["含生词"] != "无"]
                if unknown_ngrams:
                    st.info(f"有 {len(unknown_ngrams)} 个高频搭配包含你当前词库中未掌握的词汇，建议优先学习。")
            else:
                st.caption("文本过短，无法提取 N-gram。")
    font_candidates = [
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    wc = WordCloud(width=900, height=360, background_color=EDITORIAL_PAPER, colormap="copper", font_path=font_path)
    wc.generate_from_frequencies({item["word"]: max(120 - item["rank"], 10) for item in HIGH_FREQ_WORDS})
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor(EDITORIAL_PAPER)
    ax.set_facecolor(EDITORIAL_PAPER)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


def validate_vocab(deck_id):
    vocab = query_df("select * from vocabulary where deck_id=?", (deck_id,))
    issues = []
    kana_re = re.compile(r"^[\u3040-\u309f\u30a0-\u30ffー]+$")
    for _, row in vocab.iterrows():
        if not row["word"] or not row["kana"] or not row["meaning"]:
            issues.append((row["id"], row["word"], "字段缺失", "词、假名、释义都不能为空"))
        if row["kana"] and not kana_re.match(str(row["kana"])):
            issues.append((row["id"], row["word"], "假名格式异常", row["kana"]))
        if len(str(row["meaning"]).strip()) < 1:
            issues.append((row["id"], row["word"], "释义过短", row["meaning"]))
    dup = vocab.groupby(["word", "kana"]).size().reset_index(name="count")
    for _, row in dup[dup["count"] > 1].iterrows():
        issues.append(("", row["word"], "重复词条", f"{row['word']} / {row['kana']} 出现 {row['count']} 次"))
    features = build_features(deck_id)
    if not features.empty:
        hard = features[(features["review_count"] >= 3) & (features["avg_score"] < 0.35)]
        for _, row in hard.iterrows():
            issues.append((row["word_id"], row["word"], "学习异常", "多次复习后仍错误率较高，建议检查释义或例句"))
    return pd.DataFrame(issues, columns=["词条ID", "词", "问题类型", "说明"])


def page_validation(deck_id):
    st.subheader("自检与数据质量")
    issues = validate_vocab(deck_id)
    if issues.empty:
        st.success("当前词库未发现明显格式问题。")
    else:
        st.dataframe(issues, use_container_width=True, hide_index=True)
    feedback = query_df(
        """
        select f.created_at as 时间, v.word as 词, f.message as 反馈
        from feedback f left join vocabulary v on f.word_id=v.id
        order by f.id desc
        """
    )
    st.markdown("### 用户反馈")
    if feedback.empty:
        st.caption("暂无反馈。")
    else:
        st.dataframe(feedback, use_container_width=True, hide_index=True)
    _, score_df = train_memory_model()
    st.markdown("### 记忆预测模型评估")
    st.dataframe(score_df.assign(Accuracy=lambda d: d["Accuracy"].round(3), F1=lambda d: d["F1"].round(3)), hide_index=True)


def import_deck(file, deck_name):
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    mapping = {
        "word": ["word", "单词", "词", "日语", "vocabulary"],
        "kana": ["kana", "假名", "读音", "reading"],
        "meaning": ["meaning", "释义", "中文", "意思", "translation"],
        "part_of_speech": ["part_of_speech", "词性", "pos"],
        "jlpt_level": ["jlpt_level", "JLPT", "等级", "level"],
        "example": ["example", "例句", "sentence"],
        "tags": ["tags", "标签", "分类"],
    }
    colmap = {}
    for target, names in mapping.items():
        for name in names:
            if name in df.columns:
                colmap[target] = name
                break
    required = {"word", "kana", "meaning"}
    missing = required - set(colmap)
    if missing:
        raise ValueError("导入文件至少需要包含：word/kana/meaning，或中文列名：单词/假名/释义。")
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cur = conn.execute("insert or ignore into decks(name, source, created_at) values(?,?,?)", (deck_name, file.name, now))
        deck_id = cur.lastrowid or conn.execute("select id from decks where name=?", (deck_name,)).fetchone()[0]
        rows = []
        for _, row in df.iterrows():
            if pd.isna(row[colmap["word"]]) or pd.isna(row[colmap["kana"]]) or pd.isna(row[colmap["meaning"]]):
                continue
            rows.append(
                (
                    deck_id,
                    str(row[colmap["word"]]).strip(),
                    str(row[colmap["kana"]]).strip(),
                    str(row[colmap["meaning"]]).strip(),
                    str(row[colmap.get("part_of_speech", "")]).strip() if colmap.get("part_of_speech") else "",
                    str(row[colmap.get("jlpt_level", "")]).strip() if colmap.get("jlpt_level") else "N5",
                    str(row[colmap.get("example", "")]).strip() if colmap.get("example") else "",
                    str(row[colmap.get("tags", "")]).strip() if colmap.get("tags") else "",
                    now,
                )
            )
        conn.executemany(
            """
            insert or ignore into vocabulary
            (deck_id, word, kana, meaning, part_of_speech, jlpt_level, example, tags, created_at)
            values(?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def delete_deck(deck_id):
    with connect() as conn:
        word_ids = [row[0] for row in conn.execute("select id from vocabulary where deck_id=?", (deck_id,))]
        if word_ids:
            placeholders = ",".join("?" for _ in word_ids)
            conn.execute(f"delete from reviews where word_id in ({placeholders})", word_ids)
            conn.execute(f"delete from test_results where word_id in ({placeholders})", word_ids)
            conn.execute(f"delete from feedback where word_id in ({placeholders})", word_ids)
            conn.execute(f"delete from wrong_book where word_id in ({placeholders})", word_ids)
        conn.execute("delete from vocabulary where deck_id=?", (deck_id,))
        conn.execute("delete from decks where id=?", (deck_id,))
        conn.commit()


def rename_deck(deck_id, new_name):
    with connect() as conn:
        conn.execute("update decks set name=? where id=?", (new_name, deck_id))
        conn.commit()


def add_to_wrong_book(word_id, note=""):
    run_sql(
        "insert or ignore into wrong_book(word_id, note, created_at) values(?,?,?)",
        (int(word_id), note, datetime.now().isoformat(timespec="seconds")),
    )


def remove_from_wrong_book(word_id):
    run_sql("delete from wrong_book where word_id=?", (int(word_id),))


def get_wrong_book():
    return query_df(
        """
        select wb.created_at as added_at,
               v.id as word_id,
               d.name as deck_name,
               v.word,
               v.kana,
               v.meaning,
               v.part_of_speech,
               v.jlpt_level,
               v.example
        from wrong_book wb
        join vocabulary v on v.id = wb.word_id
        join decks d on d.id = v.deck_id
        order by wb.id desc
        """
    )


def page_deck(deck_id):
    st.subheader("词库管理")
    decks = get_decks()
    current_name = decks.loc[decks["id"] == deck_id, "name"].iloc[0]
    current_source = decks.loc[decks["id"] == deck_id, "source"].iloc[0]
    total, mastered, pending, _ = deck_stats(deck_id, int(st.session_state.get("daily_goal", 10)))
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>当前词库</div>
            <div class='metric-value'>{current_name}</div>
            <div class='subtle'>来源：{current_source} ｜ 总量 {total} ｜ 已掌握 {mastered} ｜ 待学习 {pending}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    vocab = query_df(
        "select word as 单词, kana as 假名, meaning as 释义, part_of_speech as 词性, jlpt_level as JLPT, example as 例句 from vocabulary where deck_id=? order by id",
        (deck_id,),
    )
    st.dataframe(vocab, use_container_width=True, hide_index=True)
    st.markdown("### 词库维护")
    col_rename, col_import = st.columns([1, 1])
    with col_rename:
        st.markdown("### 重命名已有词库")
        new_name = st.text_input("新的词库名称", value=current_name)
        if st.button("保存词库名称"):
            cleaned = new_name.strip()
            if not cleaned:
                st.warning("词库名称不能为空。")
            elif cleaned == current_name:
                st.info("名称没有变化。")
            elif cleaned in set(decks["name"].tolist()):
                st.warning("这个词库名称已经存在。")
            else:
                rename_deck(deck_id, cleaned)
                st.success(f"已重命名为「{cleaned}」。")
                st.rerun()
    with col_import:
        st.markdown("### 导入 CSV / Excel")
        deck_name = st.text_input("导入后的词库名称", value=f"自定义词库 {date.today().isoformat()}")
        file = st.file_uploader("支持列名：word/kana/meaning，或 单词/假名/释义", type=["csv", "xlsx"])
        if st.button("导入为新词库", type="primary"):
            if not file:
                st.warning("请先选择文件。")
            elif not deck_name.strip():
                st.warning("请填写词库名称。")
            else:
                try:
                    count = import_deck(file, deck_name.strip())
                    st.success(f"已导入 {count} 条词汇到「{deck_name.strip()}」。")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    _, col_delete = st.columns([1, 1])
    with col_delete:
        st.markdown("### 删除词库")
        deletable = decks[decks["name"] != "示例 JLPT 核心词库"]
        if deletable.empty:
            st.caption("暂无可删除词库。")
        else:
            delete_name = st.selectbox("选择要删除的词库", deletable["name"].tolist())
            confirm = st.checkbox(f"确认删除「{delete_name}」")
            if st.button("删除词库"):
                if not confirm:
                    st.warning("请先勾选确认删除。")
                else:
                    delete_id = int(deletable.loc[deletable["name"] == delete_name, "id"].iloc[0])
                    delete_deck(delete_id)
                    for key in ["current_word_id", "question", "question_deck_id"]:
                        st.session_state.pop(key, None)
                    st.success(f"已删除「{delete_name}」。")
                    st.rerun()


def page_wrong_book():
    st.subheader("错题本")
    wrong = get_wrong_book()
    if wrong.empty:
        st.info("错题本里还没有单词。你可以在背单词页面把容易错的词加入这里。")
        return
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>错题本单词</div>
            <div class='metric-value'>{len(wrong)}</div>
            <div class='subtle'>点击下拉框中的单词查看读音、例句和来源词库。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    options = {
        f"{row.word} / {row.kana} / {row.meaning[:28]}": int(row.word_id)
        for row in wrong.itertuples()
    }
    selected = st.selectbox("选择错题单词", list(options.keys()))
    word_id = options[selected]
    row = wrong[wrong["word_id"] == word_id].iloc[0]
    st.markdown(
        f"""
        <div class='word-card'>
            <div class='jp-word'>{row['word']}</div>
            <div class='kana'>{row['kana']}</div>
            <div class='meaning'>{row['meaning']}</div>
            <p>{row['example']}</p>
            <span class='pill'>{row['jlpt_level']}</span>
            <span class='pill'>{row['part_of_speech']}</span>
            <span class='pill'>{row['deck_name']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    speak_button(row["word"], f"wrong-{word_id}", row["kana"])
    col_remove, _ = st.columns([1, 3])
    if col_remove.button("移出错题本", use_container_width=True):
        remove_from_wrong_book(word_id)
        st.success("已移出错题本。")
        st.rerun()
    st.markdown("### 错题列表")
    for idx, item in enumerate(wrong.head(12).itertuples(), start=1):
        st.markdown(
            f"""
            <div class='editorial-list-row'>
                <div class='editorial-list-no'>{idx:02d}</div>
                <div>
                    <div class='editorial-list-word'>{item.word} / {item.kana}</div>
                    <div class='editorial-list-meta'>{item.meaning} · {item.jlpt_level} · {item.deck_name}</div>
                </div>
                <div class='editorial-list-meta'>{str(item.added_at)[:10]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.dataframe(
        wrong[["word", "kana", "meaning", "jlpt_level", "deck_name", "added_at"]].rename(
            columns={
                "word": "单词",
                "kana": "假名",
                "meaning": "释义",
                "jlpt_level": "JLPT",
                "deck_name": "来源词库",
                "added_at": "加入时间",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def page_analytics():
    st.subheader("学习数据分析")
    refresh_col, time_col = st.columns([1, 3])
    if refresh_col.button("刷新分析数据", use_container_width=True):
        st.rerun()
    time_col.caption(f"最近一次更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_reviews = query_df("select count(*) as n from reviews").iloc[0]["n"]
    total_tests = query_df("select count(*) as n from test_results").iloc[0]["n"]
    review_dates = query_df("select distinct date(reviewed_at) as d from reviews order by d")
    test_dates = query_df("select distinct date(answered_at) as d from test_results order by d")
    all_dates = sorted(set(review_dates["d"].tolist() + test_dates["d"].tolist()))
    total_days = len(all_dates)
    streak = 0
    today = date.today()
    for d in reversed(all_dates):
        if (today - datetime.strptime(d, "%Y-%m-%d").date()).days == streak:
            streak += 1
        else:
            break

    cols = st.columns(4)
    metrics = [
        ("累计学习天数", total_days),
        ("总复习次数", total_reviews),
        ("总自测题数", total_tests),
        ("连续学习天数", streak),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

    if total_reviews == 0 and total_tests == 0:
        st.info("还没有学习记录。去背单词或自测页面开始学习吧。")
        return

    st.markdown("### 每日学习趋势")
    daily_reviews = query_df(
        "select date(reviewed_at) as d, count(*) as cnt from reviews group by d order by d"
    )
    daily_tests = query_df(
        "select date(answered_at) as d, count(*) as cnt from test_results group by d order by d"
    )
    if not daily_reviews.empty or not daily_tests.empty:
        daily_reviews["d"] = pd.to_datetime(daily_reviews["d"])
        daily_tests["d"] = pd.to_datetime(daily_tests["d"])
        merged = pd.merge(daily_reviews, daily_tests, on="d", how="outer", suffixes=("_review", "_test")).fillna(0)
        merged = merged.sort_values("d")
        if len(merged) > 30:
            merged = merged.tail(30)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

        ax1.plot(merged["d"], merged["cnt_review"], marker="o", color=EDITORIAL_GREEN, linewidth=1.5, label="复习次数")
        ax1.plot(merged["d"], merged["cnt_test"], marker="s", color=EDITORIAL_GOLD, linewidth=1.5, label="自测题数")
        ax1.set_title("每日学习量")
        ax1.legend(fontsize=9)
        ax1.tick_params(axis="x", rotation=30)
        ax1.set_ylabel("数量")
        apply_editorial_chart_style(ax1)

        test_acc = query_df(
            "select date(answered_at) as d, avg(cast(is_correct as float)) as acc from test_results group by d order by d"
        )
        if not test_acc.empty:
            test_acc["d"] = pd.to_datetime(test_acc["d"])
            if len(test_acc) > 30:
                test_acc = test_acc.tail(30)
            ax2.plot(test_acc["d"], test_acc["acc"], marker="o", color=EDITORIAL_BLUE, linewidth=1.5)
            ax2.set_title("每日自测正确率")
            ax2.set_ylim(0, 1)
            ax2.axhline(y=0.6, color=EDITORIAL_RED, linestyle="--", alpha=0.5, label="60% 基准")
            ax2.legend(fontsize=9)
        ax2.tick_params(axis="x", rotation=30)
        apply_editorial_chart_style(ax2)

        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown("### JLPT 等级掌握度")
    level_mastery = query_df(
        """
        select v.jlpt_level,
               count(*) as total,
               sum(case when coalesce(last_scores.s, 0) >= 0.95 then 1 else 0 end) as mastered
        from vocabulary v
        left join (
            select word_id, max(score) as s
            from reviews
            group by word_id
        ) last_scores on last_scores.word_id = v.id
        group by v.jlpt_level
        order by v.jlpt_level
        """
    )
    if not level_mastery.empty:
        level_mastery["jlpt_level"] = pd.Categorical(level_mastery["jlpt_level"], categories=LEVEL_ORDER, ordered=True)
        level_mastery = level_mastery.sort_values("jlpt_level")
        level_mastery["learning"] = level_mastery["total"] - level_mastery["mastered"]
        level_mastery["mastery_rate"] = level_mastery["mastered"] / level_mastery["total"].replace(0, np.nan)
        level_mastery["mastery_rate"] = level_mastery["mastery_rate"].fillna(0)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = range(len(level_mastery))
        colors = [EDITORIAL_GREEN if value >= 0.6 else EDITORIAL_GOLD if value >= 0.3 else EDITORIAL_RED for value in level_mastery["mastery_rate"]]
        ax.bar(x, level_mastery["mastery_rate"], color=colors, label="掌握率")
        ax.set_xticks(x)
        ax.set_xticklabels(level_mastery["jlpt_level"])
        ax.set_ylabel("掌握率")
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.axhline(y=0.6, color=EDITORIAL_BLUE, linestyle="--", alpha=0.45, label="60% 基准")
        ax.legend(fontsize=9)
        for i, row in level_mastery.iterrows():
            xpos = list(x)[level_mastery.index.get_loc(i)]
            pct = row["mastery_rate"] * 100
            ax.text(xpos, min(row["mastery_rate"] + 0.04, 0.96), f"{pct:.0f}%", ha="center", fontsize=9)
            ax.text(xpos, 0.03, f"{int(row['mastered'])}/{int(row['total'])}", ha="center", fontsize=8, color=EDITORIAL_INK)
        apply_editorial_chart_style(ax)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 累计掌握词汇增长")
        mastered_growth = query_df(
            """
            select d, sum(cnt) over (order by d) as total
            from (
                select date(reviewed_at) as d, count(distinct word_id) as cnt
                from reviews
                where score >= 0.95
                group by d
            )
            """
        )
        if not mastered_growth.empty:
            mastered_growth["d"] = pd.to_datetime(mastered_growth["d"])
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.fill_between(mastered_growth["d"], mastered_growth["total"], alpha=0.28, color=EDITORIAL_GREEN)
            ax.plot(mastered_growth["d"], mastered_growth["total"], color=EDITORIAL_GREEN, linewidth=1.5)
            ax.set_ylabel("累计掌握词汇")
            ax.tick_params(axis="x", rotation=30)
            apply_editorial_chart_style(ax)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            st.caption("复习数据不足，暂无掌握增长曲线。")

    with col_right:
        st.markdown("### 错题词性分布")
        pos_errors = query_df(
            """
            select v.part_of_speech, count(*) as cnt
            from test_results tr
            join vocabulary v on v.id = tr.word_id
            where tr.is_correct = 0
            group by v.part_of_speech
            order by cnt desc
            limit 8
            """
        )
        if not pos_errors.empty:
            pos_errors["pos_group"] = pos_errors["part_of_speech"].apply(pos_group)
            grouped = pos_errors.groupby("pos_group")["cnt"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(5.6, 3.2))
            grouped = grouped.sort_values(ascending=True)
            colors = [EDITORIAL_RED if value == grouped.max() else EDITORIAL_GOLD for value in grouped.values]
            ax.barh(grouped.index, grouped.values, color=colors)
            ax.set_xlabel("错误次数")
            ax.set_title("错题词性分布")
            for i, value in enumerate(grouped.values):
                ax.text(value + 0.05, i, str(int(value)), va="center", fontsize=9)
            apply_editorial_chart_style(ax, grid_axis="x")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            st.caption("自测数据不足，暂无错题分布。")

    st.markdown("### 遗忘风险排行榜")
    risk_features = build_features()
    if not risk_features.empty:
        risk_df = add_memory_probability(risk_features).sort_values("forget_risk", ascending=False).head(10)
        risk_df = risk_df[["word", "kana", "meaning", "jlpt_level", "review_count", "interval_days", "forget_risk"]].copy()
        risk_df["遗忘风险"] = (risk_df["forget_risk"] * 100).round(0).astype(int).astype(str) + "%"
        risk_df = risk_df.rename(
            columns={
                "word": "单词",
                "kana": "假名",
                "meaning": "释义",
                "jlpt_level": "JLPT",
                "review_count": "复习次数",
                "interval_days": "距上次复习天数",
            }
        ).drop(columns=["forget_risk"])
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        st.caption("风险根据复习次数、最近掌握度、复习间隔和词汇难度综合估计，数值越高越建议优先复习。")
    else:
        st.caption("暂无词汇数据，无法生成遗忘风险排行。")

    st.markdown("### 学习画像与建议报告")
    total_vocab = query_df("select count(*) as n from vocabulary").iloc[0]["n"]
    mastered_vocab = query_df(
        """
        select count(distinct word_id) as n
        from reviews
        where score >= 0.95
        """
    ).iloc[0]["n"]
    recent_reviews = query_df(
        "select count(*) as n from reviews where date(reviewed_at) >= date('now', '-7 day')"
    ).iloc[0]["n"]
    recent_tests = query_df(
        "select avg(cast(is_correct as float)) as acc from test_results where date(answered_at) >= date('now', '-7 day')"
    ).iloc[0]["acc"]
    wrong_book_count = query_df("select count(*) as n from wrong_book").iloc[0]["n"]

    vocab_score = min(float(mastered_vocab) / max(float(total_vocab), 1.0), 1.0)
    stability_score = 0.0
    if not risk_features.empty:
        stability_score = 1.0 - float(add_memory_probability(risk_features)["forget_risk"].mean())
        stability_score = max(0.0, min(stability_score, 1.0))
    test_score = 0.0 if pd.isna(recent_tests) else max(0.0, min(float(recent_tests), 1.0))
    activity_score = min(float(recent_reviews + total_tests) / 80.0, 1.0)
    review_score = min(float(wrong_book_count) / 30.0, 1.0)

    radar_labels = ["词汇掌握", "记忆稳定", "自测表现", "学习活跃", "错题复盘"]
    radar_values = [vocab_score, stability_score, test_score, activity_score, review_score]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    values = radar_values + radar_values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.8, 5.2), subplot_kw={"polar": True})
    ax.plot(angles_closed, values, color=EDITORIAL_BLUE, linewidth=2)
    ax.fill(angles_closed, values, color=EDITORIAL_BLUE, alpha=0.18)
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.set_ylim(0, 1)
    ax.set_title("学习画像雷达图", pad=18)
    ax.set_facecolor(EDITORIAL_PAPER)
    fig.patch.set_facecolor(EDITORIAL_PAPER)
    ax.tick_params(colors=EDITORIAL_MUTED)
    ax.grid(color=EDITORIAL_LINE, linewidth=0.6, alpha=0.65)
    ax.spines["polar"].set_color(EDITORIAL_LINE)
    st.pyplot(fig, clear_figure=True)

    weakest_index = int(np.argmin(radar_values))
    weakest = radar_labels[weakest_index]
    report_advice = {
        "词汇掌握": "当前掌握词汇占比偏低：建议把每日目标集中在高频词和当前等级核心词，先扩大基础覆盖面。",
        "记忆稳定": "记忆稳定度偏低：建议优先处理遗忘风险排行榜中的词，并缩短复习间隔。",
        "自测表现": "自测表现偏弱：建议降低一个难度层级进行 20 题测试，确认基础后再上调难度。",
        "学习活跃": "学习活跃度不足：建议保持每天至少一次背词或自测，形成连续学习记录。",
        "错题复盘": "错题复盘不足：建议把答错和模糊的词加入错题本，并在当天二次复习。",
    }
    st.info(report_advice[weakest])


def page_project_intro():
    st.subheader("项目说明")
    st.markdown(
        """
        本项目面向日语学习者，设计一个本地运行的智能词汇记忆辅助系统。系统围绕
        “背词记录、自适应自测、遗忘风险预测、词频分析、学习数据可视化”形成完整闭环，
        既能满足日常学习使用，也能体现人工智能课程大作业对数据处理、模型分析和交互应用的要求。
        """
    )

    total_vocab = query_df("select count(*) as n from vocabulary").iloc[0]["n"]
    total_decks = query_df("select count(*) as n from decks").iloc[0]["n"]
    total_reviews = query_df("select count(*) as n from reviews").iloc[0]["n"]
    total_tests = query_df("select count(*) as n from test_results").iloc[0]["n"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("词库数量", int(total_decks))
    c2.metric("词汇总量", int(total_vocab))
    c3.metric("背词记录", int(total_reviews))
    c4.metric("自测记录", int(total_tests))

    st.markdown("### 功能模块")
    module_df = pd.DataFrame(
        [
            ["背单词", "根据词库、每日目标和遗忘风险推荐学习队列，记录认识、模糊、不认识等学习反馈。"],
            ["自测", "使用不同 JLPT 难度词汇进行随机测验，20 题后给出带权语言能力评估和学习建议。"],
            ["高频词分析", "支持查看高频词的含义、用法、例句，也支持对输入文本做词频、词云和搭配分析。"],
            ["学习分析", "根据背词、自测、错题本等交互数据生成趋势图、掌握率、错题分布和学习画像。"],
            ["自检", "检查词库字段、重复词、假名格式和长期低掌握词，辅助保证数据质量。"],
            ["词库管理", "支持自定义命名词库、导入 CSV/Excel、删除词库和维护本地学习数据。"],
            ["错题本", "保存重点复习词，查看读音、释义和例句，形成二次复盘入口。"],
            ["文本转语音", "将日语文本转为语音，辅助读音模仿和听力练习。"],
            ["中日互译", "使用免费联网翻译接口，支持日语与中文快速互译，适合查词和短句理解。"],
        ],
        columns=["模块", "作用"],
    )
    st.dataframe(module_df, use_container_width=True, hide_index=True)

    st.markdown("### AI 与数据分析体现")
    ai_df = pd.DataFrame(
        [
            [
                "记忆效果预测",
                "Logistic Regression、Random Forest",
                "使用 JLPT 难度、复习次数、最近掌握度、平均掌握度、复习间隔等特征，预测词汇保持概率。",
            ],
            [
                "智能复习推荐",
                "遗忘风险排序",
                "将模型输出转化为遗忘风险，优先推荐高风险词和未学习词。",
            ],
            [
                "自适应水平评估",
                "难度加权统计",
                "自测达到 20 题后，按 N1-N5 词汇难度加权估计真实语言能力。",
            ],
            [
                "个性化学习建议",
                "阶段规则 + 错误类型分析",
                "按等级阶段、正确率、名词/动词/形容词错误频率组合生成建议。",
            ],
            [
                "文本词频分析",
                "分词、词频、N-gram、词云",
                "对用户输入文本提取高频词和常见搭配，辅助发现真实语料中的重点词。",
            ],
            [
                "学习画像可视化",
                "雷达图与统计图表",
                "综合词汇掌握、记忆稳定、自测表现、学习活跃、错题复盘形成画像。",
            ],
        ],
        columns=["分析点", "方法", "说明"],
    )
    st.dataframe(ai_df, use_container_width=True, hide_index=True)

    st.markdown("### 数据库设计")
    db_df = pd.DataFrame(
        [
            ["decks", "词库信息，包括默认词库和用户自定义词库。"],
            ["vocabulary", "词汇主体数据，包括单词、假名、释义、词性、等级、例句和标签。"],
            ["reviews", "背单词反馈记录，用于统计掌握度和训练记忆预测模型。"],
            ["test_results", "自测答题记录，用于计算正确率、水平评估和错题类型。"],
            ["feedback", "用户反馈和自检问题记录。"],
            ["wrong_book", "错题本词汇记录，用于重点复习和学习画像。"],
        ],
        columns=["数据表", "用途"],
    )
    st.dataframe(db_df, use_container_width=True, hide_index=True)

    st.markdown("### 大作业要求对应")
    requirement_df = pd.DataFrame(
        [
            ["有明确应用场景", "面向日语学习者的词汇记忆、复习、自测和读音训练。"],
            ["有数据采集与处理", "采集背词反馈、自测结果、错题本、词库导入数据并写入 SQLite。"],
            ["有算法或模型", "使用机器学习模型预测记忆保持概率，并结合规则完成自适应推荐和水平评估。"],
            ["有数据分析结果", "展示掌握率、趋势、错题词性、遗忘风险、学习画像等分析结果。"],
            ["有可视化界面", "使用 Streamlit 构建可交互页面，支持图表、表格、按钮、导入和下载。"],
            ["有完整可运行作品", "本地通过 Python 运行，依赖集中在 requirements.txt，数据保存在 SQLite。"],
        ],
        columns=["要求", "本系统对应实现"],
    )
    st.dataframe(requirement_df, use_container_width=True, hide_index=True)

    st.markdown("### 运行方式")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="powershell")

    st.markdown("### 可继续改进方向")
    st.markdown(
        """
        - 增加更长期的学习记录后，重新训练更稳定的记忆预测模型。
        - 扩展听力题、例句填空题和语境选择题，让自测不只考察词义。
        - 为不同 JLPT 等级加入更完整的阶段任务和周报总结。
        - 支持导出学习报告，便于提交课程展示或长期跟踪学习效果。
        """
    )


async def _edge_tts_generate(text, voice, speed, on_progress, is_cancelled):
    """使用 Microsoft Edge TTS 生成 MP3 音频数据。"""
    import edge_tts

    rate_pct = int((speed - 1) * 100)
    rate_str = f"{rate_pct:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)

    total_sentences = len(re.findall(r"[。！？!?\n]", text))
    if total_sentences == 0:
        total_sentences = 1

    mp3_buffer = io.BytesIO()
    sentence_count = 0

    async for chunk in communicate.stream():
        if is_cancelled():
            raise asyncio.CancelledError("用户取消了语音生成")

        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
        elif chunk["type"] == "SentenceBoundary":
            sentence_count += 1
            on_progress(min(sentence_count / total_sentences, 1.0))

    on_progress(1.0)
    return mp3_buffer.getvalue()


def _tts_thread_runner(text, voice, speed, tracker):
    """在后台线程中运行 TTS 生成，通过 tracker dict 报告进度和结果。"""
    # 保存并绕过系统代理设置（aiohttp 的 trust_env=True 会读取
    # Windows 系统代理，如果代理进程未运行则所有声线都会连接失败）
    saved_no_proxy = os.environ.get("NO_PROXY")
    os.environ["NO_PROXY"] = "*"
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _edge_tts_generate(
                text, voice, speed,
                on_progress=lambda v: tracker.update({"progress": v}),
                is_cancelled=lambda: tracker.get("cancel", False),
            )
        )
        loop.close()
        tracker["result"] = result
        tracker["state"] = "complete"
    except asyncio.CancelledError:
        tracker["state"] = "idle"
        tracker["cancel"] = False
    except Exception as exc:
        tracker["state"] = "error"
        tracker["error_msg"] = str(exc)
    finally:
        if saved_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = saved_no_proxy


def page_translation():
    st.subheader("中日互译")
    st.caption("使用免费联网翻译接口，不需要 API Key；适合快速查词、短句理解和阅读辅助。")

    direction_options = {
        "自动识别": "auto",
        "日语 → 中文": "ja_to_zh",
        "中文 → 日语": "zh_to_ja",
    }

    st.info("当前模式需要联网，不消耗 API token。网络不可用或接口临时不可用时，翻译可能失败。")

    with st.form("web_translate_form"):
        direction_label = st.selectbox("翻译方向", list(direction_options.keys()), index=0)
        source_text = st.text_area(
            "输入中文或日语",
            height=180,
            placeholder="例：こんにちは / 你好 / 今日はいい天気です",
        )
        submitted = st.form_submit_button(
            "翻译",
            type="primary",
            use_container_width=True,
        )

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("清空结果", use_container_width=True):
            st.session_state.web_translate_result = ""
            st.session_state.web_translate_source = ""
            st.rerun()

    if submitted:
        if not source_text.strip():
            st.warning("请先输入要翻译的中文或日语。")
        else:
            try:
                result = web_translate(source_text, direction_options[direction_label])
                st.session_state.web_translate_result = result
                st.session_state.web_translate_source = source_text.strip()
                st.success("翻译完成")
            except Exception as exc:
                st.session_state.web_translate_result = ""
                st.session_state.web_translate_source = ""
                st.error(f"联网翻译失败：{exc}")

    result_text = st.session_state.get("web_translate_result", "")
    st.markdown("**翻译结果**")
    if result_text:
        st.markdown(f'<div class="translation-result">{result_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="translation-result muted">等待输入并点击翻译。</div>', unsafe_allow_html=True)


def page_tts():
    st.subheader("日语文本转语音")
    st.caption("使用 Microsoft Azure 神经语音，声线自然流畅，支持语速调节和 MP3 下载。")

    try:
        import edge_tts  # noqa: F401
    except ImportError:
        st.warning("需要安装 edge-tts 才能使用此功能，请在终端执行：`pip install edge-tts`")
        return

    VOICES = {
        "Nanami (女声·自然清澈)": "ja-JP-NanamiNeural",
        "Aoi (女声·温柔亲切)": "ja-JP-AoiNeural",
        "Mayu (女声·活泼明快)": "ja-JP-MayuNeural",
        "Keita (男声·标准稳重)": "ja-JP-KeitaNeural",
        "Naoki (男声·成熟沉稳)": "ja-JP-NaokiNeural",
    }

    # ---- 初始化会话状态 ----
    if "tts_state" not in st.session_state:
        st.session_state.tts_state = "idle"

    is_generating = st.session_state.tts_state == "generating"

    # ---- 输入控件（生成中禁用） ----
    col1, col2 = st.columns(2)
    with col1:
        selected_label = st.selectbox(
            "选择声线",
            list(VOICES.keys()),
            index=0,
            help="Nanami 是最受欢迎的日语女声，自然度极高。",
            disabled=is_generating,
        )
    with col2:
        speed = st.slider(
            "语速",
            min_value=0.5,
            max_value=2.0,
            value=0.9,
            step=0.1,
            help="0.5 = 最慢  |  0.9-1.0 = 自然语速  |  2.0 = 最快",
            disabled=is_generating,
        )

    text = st.text_area(
        "输入日语文本",
        height=200,
        placeholder=(
            "ここに日本語のテキストを貼り付けてください。\n"
            "例：今日はいい天気ですね。一緒に図書館へ行きませんか。"
        ),
        key="tts_input",
        disabled=is_generating,
    )

    char_count = len(text.strip()) if text else 0
    if char_count > 0:
        st.caption(
            f"已输入 {char_count} 个字符"
            + ("（长文本可能需要较长时间生成，请耐心等待）" if char_count > 200 else "")
        )

    # ============================================================
    # 状态机渲染（通过 tracker dict 与后台线程通信）
    # ============================================================

    # ---- 状态: idle ----
    if st.session_state.tts_state == "idle":
        col_btn1, _ = st.columns([1, 2])
        with col_btn1:
            if st.button(
                "朗读",
                type="primary",
                use_container_width=True,
                disabled=not text.strip(),
            ):
                # 创建 tracker dict 用于线程安全通信
                tracker = {
                    "progress": 0.0,
                    "cancel": False,
                    "state": "running",
                    "result": None,
                    "error_msg": "",
                }
                st.session_state.tts_tracker = tracker
                st.session_state.tts_state = "generating"
                st.session_state.tts_voice = selected_label
                st.session_state.tts_text_preview = text.strip()[:80]

                thread = threading.Thread(
                    target=_tts_thread_runner,
                    args=(text.strip(), VOICES[selected_label], speed, tracker),
                    daemon=True,
                )
                thread.start()
                st.session_state._tts_thread = thread

                st.rerun()

        if "tts_audio" in st.session_state and st.session_state.tts_audio:
            st.divider()
            st.markdown(
                f"**声线:** {st.session_state.get('tts_voice', '')}"
                f"  |  **预览:** {st.session_state.get('tts_text_preview', '')}..."
            )
            st.audio(st.session_state.tts_audio, format="audio/mp3")
            voice_label = st.session_state.get("tts_voice", "voice")
            voice_short = voice_label.split("(")[0] if "(" in voice_label else voice_label
            dl_name = f"{voice_short}.mp3"
            st.download_button(
                label="下载 MP3",
                data=st.session_state.tts_audio,
                file_name=dl_name,
                mime="audio/mpeg",
                key="dl_mp3_idle",
            )

    # ---- 状态: generating ----
    elif st.session_state.tts_state == "generating":
        tracker = st.session_state.get("tts_tracker", {})
        progress = tracker.get("progress", 0.0)
        tracker_state = tracker.get("state", "running")

        # 检查线程是否已结束（通过 tracker state 判断）
        if tracker_state == "complete":
            st.session_state.tts_audio = tracker["result"]
            st.session_state.tts_state = "complete"
            st.rerun()
        elif tracker_state == "error":
            st.session_state.tts_error = tracker.get("error_msg", "未知错误")
            st.session_state.tts_state = "error"
            st.rerun()
        elif tracker_state == "idle":
            st.session_state.tts_state = "idle"
            st.rerun()

        st.progress(progress, text=f"正在生成语音... {int(progress * 100)}%")

        col_stop, _ = st.columns([1, 2])
        with col_stop:
            if st.button("停止生成", type="secondary", use_container_width=True):
                tracker["cancel"] = True
                st.rerun()

        time.sleep(0.5)
        st.rerun()

    # ---- 状态: complete ----
    elif st.session_state.tts_state == "complete":
        st.success("语音生成完毕！")
        st.divider()
        st.markdown(
            f"**声线:** {st.session_state.get('tts_voice', '')}"
            f"  |  **预览:** {st.session_state.get('tts_text_preview', '')}..."
        )
        st.audio(st.session_state.tts_audio, format="audio/mp3")

        col_dl, col_new, _ = st.columns([1, 1, 1])
        with col_dl:
            voice_label = st.session_state.get("tts_voice", "voice")
            voice_short = voice_label.split("(")[0] if "(" in voice_label else voice_label
            dl_name = f"{voice_short}.mp3"
            st.download_button(
                label="下载 MP3",
                data=st.session_state.tts_audio,
                file_name=dl_name,
                mime="audio/mpeg",
                key="dl_mp3_complete",
            )
        with col_new:
            if st.button("重新生成", type="primary", use_container_width=True):
                st.session_state.tts_state = "idle"
                st.rerun()

    # ---- 状态: error ----
    elif st.session_state.tts_state == "error":
        error_msg = st.session_state.get("tts_error", "未知错误")
        st.error(f"语音生成失败：{error_msg}")
        if st.button("重试", type="primary"):
            st.session_state.tts_state = "idle"
            st.rerun()


def main():
    st.set_page_config(page_title="智能日语词汇记忆辅助系统", page_icon="学", layout="wide")
    init_db()
    inject_css()
    st.markdown(
        """
        <div class="editorial-masthead">
            <div class="editorial-kicker">Japanese Vocabulary Review System</div>
            <div class="editorial-title">智能日语词汇记忆辅助系统</div>
            <div class="editorial-subtitle">背单词 / 自测 / 高频词分析 / 学习分析 / 项目说明 / 自检 / 词库管理 / 错题本 / 文本转语音 / 中日互译</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    decks = get_decks()
    deck_names = decks["name"].tolist()
    with st.sidebar:
        st.header("学习控制台")
        deck_name = st.selectbox("当前词库", deck_names)
        deck_id = int(decks.loc[decks["name"] == deck_name, "id"].iloc[0])
        st.session_state.daily_goal = st.slider(
            "今日学习目标",
            min_value=5,
            max_value=50,
            value=int(st.session_state.get("daily_goal", 10)),
            step=5,
            help="控制背单词页面的推荐队列长度和今日建议复习数量。",
        )
        page = st.radio(
            "功能",
            ["背单词", "自测", "高频词分析", "学习分析", "项目说明", "自检", "词库管理", "错题本", "文本转语音", "中日互译"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"数据库：{DB_PATH.name}")
        st.caption("作者：萌芽熊")
    if page == "背单词":
        page_learn(deck_id)
    elif page == "自测":
        page_test(deck_id)
    elif page == "高频词分析":
        page_frequency(deck_id)
    elif page == "学习分析":
        page_analytics()
    elif page == "项目说明":
        page_project_intro()
    elif page == "自检":
        page_validation(deck_id)
    elif page == "错题本":
        page_wrong_book()
    elif page == "文本转语音":
        page_tts()
    elif page == "中日互译":
        page_translation()
    else:
        page_deck(deck_id)


if __name__ == "__main__":
    main()
