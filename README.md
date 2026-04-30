# 特斯拉问答 Agent
## 项目介绍
基于 LangChain + 通义千问 实现的轻量化智能问答程序，自定义Agent工具调用，**杜绝AI幻觉**，仅依靠本地预设数据回答问题。

## 功能
1. 特斯拉多款车型价格与介绍查询
2. 特斯拉企业信息问答
3. 命令行交互式对话

## 技术栈
Python、LangChain、通义千问大模型

## 项目结构
- python qa_bot.py  主程序

## 运行
1. 配置环境变量：`DASHSCOPE_API_KEY`
2. 安装依赖：`pip install langchain langchain-community dashscope`
3. 运行：`python qa_bot.py`

## 流程图示
<img width="705" height="3332" alt="agent流程图 drawio" src="https://github.com/user-attachments/assets/93e11683-04e6-481a-b516-0ed54bbf874f" />
