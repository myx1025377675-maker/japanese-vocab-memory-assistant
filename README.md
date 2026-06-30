# 智能日语词汇记忆辅助系统

这是一个可在 Trae CN / 本地 Python 环境运行的 Streamlit 大作业项目。程序围绕日语学习场景，实现词库管理、智能背词、机器学习记忆预测、自适应词汇自测、高频词分析和数据自检。

## 运行方式

```powershell
pip install -r requirements.txt
streamlit run app.py
```

首次运行会自动创建 `japanese_learning.db`，并写入一份示例 JLPT 词库和模拟学习记录。

## 功能对应

- 智能推荐：根据复习次数、平均掌握度、距离上次复习天数、JLPT 难度计算遗忘风险，优先推荐需要复习的词。
- 记忆效果预测：使用模拟学习行为数据训练 Logistic Regression 与 Random Forest，并在应用中展示评估指标。
- 自适应自测：根据历史答题正确率和词汇难度动态提高低掌握词的出题概率。
- 高频词总结：对用户粘贴的日语文本做规则分词、词频统计和词云可视化。
- 词库导入：支持 CSV / Excel，必要列为 `word/kana/meaning` 或 `单词/假名/释义`。
- 自检机制：检查字段缺失、假名格式、重复词条、长期低掌握词，并记录用户反馈。
- 发音：使用浏览器内置 SpeechSynthesis，离线可用，不依赖外部 TTS API。
- 扩展词库：内置“综合高频100词库”“高频基础词库”“JLPT N4-N3 扩展词库”“新闻阅读高频词库”，参考 Wiktionary Japanese frequency lists、ManyThings Japanese vocabulary frequency、Wokabulary common Japanese words、MLC Japanese JLPT 词表等公开资料整理，程序运行不依赖联网。
- MOJi PDF 词库：会自动解析当前目录下的 `MOJi辞書 - 考前对策N2汉字（新整理版）*.pdf`，合并为“MOJi N2汉字考前对策”词库。
- 今日学习目标：侧边栏可调整每日推荐数量，点击“认识”后已掌握、待学习、今日建议复习会即时更新。
- 词库管理：支持自定义命名导入 CSV/Excel，并可在界面中删除不需要的词库。
- 自测刷新：自测页面提供“重新开始自测”，可清空本轮进度重新测试。
- 错题本：背单词页面可将当前单词加入错题本，左侧“错题本”页面可查看错题单词、例句和读音。

## 建议提交材料

- `app.py`：主程序源码。
- `requirements.txt`：依赖说明。
- `README.md`：运行说明和功能说明。
- 运行截图：背单词、自测、词频分析、自检四个页面各一张。
- 课程报告：说明数据表设计、推荐算法、机器学习模型、实验指标和不足之处。
