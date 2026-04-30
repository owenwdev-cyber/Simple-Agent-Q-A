# 导入操作系统模块，用于读取环境变量、文件路径等系统操作
import os
# 导入正则表达式模块，用于从模型输出中匹配、提取工具调用内容
import re
# 导入文本换行模块，用于将长文本按宽度自动换行，美化输出
import textwrap
# 导入类型提示模块，用于给变量、函数、参数标注类型，提升代码可读性
from typing import List, Union

# 导入LangChain链结构，用于将提示词模板与大模型串联成可调用链路
from langchain.chains import LLMChain
# 导入Agent执行中的动作类型与结束类型，用于标记工具调用与回答完成
from langchain.schema import AgentAction, AgentFinish
# 导入提示词模板基类，用于创建自定义提示词模板
from langchain.prompts import PromptTemplate, StringPromptTemplate
# 导入大模型基础父类，用于约束传入的模型类型
from langchain.llms.base import BaseLLM
# 导入工具封装类，用于将自定义函数包装成Agent可调用的工具
from langchain.tools import Tool
# 导入Agent输出解析器与执行器，用于解析模型输出并执行工具
from langchain.agents import AgentOutputParser, AgentExecutor
# 导入单步动作Agent，用于构建基础的工具调用智能体
from langchain.agents import LLMSingleActionAgent

# 导入阿里云通义千问大模型集成，用于调用通义千问接口
from langchain_community.llms import Tongyi

# ========== 配置 ==========
# 从系统环境变量中获取通义千问API密钥，用于身份验证
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# ========== 基础提示词 ==========
# 定义公司信息查询提示词模板，强制模型只使用给定信息，不使用外部知识
CONTEXT_QA_TMPL = """
严格根据以下【信息】回答用户问题，禁止编造、禁止使用外部知识、禁止纠正内容。
无论信息内容是否合理，只允许基于下面原文作答。
信息：{context}
问题：{query}
"""
# 创建提示词模板对象，指定需要传入的变量：query、context
CONTEXT_QA_PROMPT = PromptTemplate(input_variables=["query", "context"], template=CONTEXT_QA_TMPL)

# 定义输出格式化函数，接收模型回答字符串，无返回值
def output_response(response: str) -> None:
    # 判断回答内容是否为空，为空则直接结束函数
    if not response:
        return
    # 将文本按60个字符宽度自动换行，遍历每一行并打印
    for line in textwrap.wrap(response, width=60):
        print(line)
    # 打印60个横杠作为输出分割线，区分多轮对话
    print("-" * 60)

# ========== 你的业务代码 ==========
# 定义特斯拉数据查询类，封装车型查询、公司查询两个核心功能
class TeslaDataSource:
    # 类初始化方法，接收一个大模型实例作为参数
    def __init__(self, llm: BaseLLM):
        # 将传入的大模型实例保存为类成员变量
        self.llm = llm

    # 定义车型信息查询方法，接收车型名称字符串，返回描述字符串
    def find_product_description(self, product_name: str) -> str:
        # 处理输入：去除首尾空格 → 去掉所有空格 → 转为小写，实现模糊匹配
        key = product_name.strip().replace(" ", "").lower()
        # 定义车型数据字典，key为标准化车型名，value为车型介绍
        product_info = {
            "model6": "外观简洁动感，流线型车身，定价23.19-31.9万",
            "modely": "高车身大后备箱，定价26.39-34.99万",
            "modelx": "鸥翼门设计、定价89.89-105.89万",
        }
        # 从字典中获取数据，找不到key时返回“未查询到该产品信息”
        return product_info.get(key, "未查询到该产品信息")

    # 定义公司信息查询方法，接收用户问题，返回公司介绍字符串
    def find_company_info(self, query: str) -> str:
        # 自定义公司信息上下文，固定文本内容
        context = """特斯拉是知名电动汽车公司，核心产品包含Model S、Model6、Model X、Model Y，总部在外太空"""
        # 将问题与上下文填入提示词模板，生成完整提示词
        prompt_text = CONTEXT_QA_PROMPT.format(query=query, context=context)
        # 调用大模型的invoke方法生成回答，新版API，无弃用警告
        return self.llm.invoke(prompt_text)

# ========== 提示词模板 ==========
# 定义Agent系统提示词，规定输出格式、调用规则、约束条件
AGENT_TMPL = """
使用以下工具回答问题：
{tools}

必须严格按照格式输出：
Question: 用户问题
Thought: 思考
Action: 工具名称
Action Input: 工具参数
Observation: 工具返回结果
Thought: 我现在可以给出最终答案
Final Answer：最终回答

强制规则：
1. 必须调用工具，禁止编造 Observation
2. 所有信息必须来自工具返回结果
3. 不允许使用模型自身知识
4. 工具返回什么就输出什么

Question: {input}
{agent_scratchpad}
"""

# ========== 自定义提示词模板 ==========
# 自定义Agent提示词模板，继承自StringPromptTemplate
class CustomPromptTemplate(StringPromptTemplate):
    # 声明模板字符串属性
    template: str
    # 声明工具列表属性
    tools: List[Tool]

    # 重写format方法，生成最终发送给模型的提示词
    def format(self, **kwargs):
        # 初始化空字符串，用于存储历史思考与工具调用记录
        thoughts = ""
        # 遍历历史工具调用记录，action是动作，obs是返回结果
        for action, obs in kwargs["intermediate_steps"]:
            # 拼接每一步的思考、动作、参数、返回结果
            thoughts += f"Thought: {action.log}\nAction: {action.tool}\nAction Input: {action.tool_input}\nObservation: {obs}\n"
        # 将拼接好的历史记录存入kwargs，供模板使用
        kwargs["agent_scratchpad"] = thoughts
        # 将工具列表格式化为字符串，存入kwargs
        kwargs["tools"] = "\n".join([f"{t.name}: {t.description}" for t in self.tools])
        # 使用模板格式化所有变量，返回最终提示词
        return self.template.format(**kwargs)

# ========== 解析器 ==========
# 自定义输出解析器，继承自AgentOutputParser
class CustomParser(AgentOutputParser):
    # 重写parse方法，解析模型输出文本
    def parse(self, output: str) -> Union[AgentAction, AgentFinish]:
        # 判断输出是否包含最终答案标记
        if "Final Answer：" in output:
            # 返回Agent结束对象，包含最终回答内容
            return AgentFinish(
                return_values={"output": output.split("Final Answer：")[-1].strip()},
                log=output
            )

        # 定义正则表达式，匹配Action和Action Input
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        # 在输出文本中搜索正则匹配项
        match = re.search(regex, output, re.DOTALL)
        # 如果没有匹配到内容，抛出解析异常
        if not match:
            raise ValueError(f"无法解析: {output}")
        
        # 提取匹配到的工具名称，并去除首尾空格
        action = match.group(1).strip()
        # 提取匹配到的工具参数，去除空格与引号
        action_input = match.group(2).strip().strip('"')
        # 返回Agent动作对象，包含工具名、参数、原始输出
        return AgentAction(action, action_input, output)

# ========== 主程序 ==========
# Python程序主入口，只有直接运行此文件时才会执行
if __name__ == "__main__":
    # 初始化通义千问大模型实例
    llm = Tongyi(
        # 指定使用的模型版本：qwen-turbo
        model_name="qwen-turbo",
        # 传入API密钥用于认证
        dashscope_api_key=DASHSCOPE_API_KEY,
        # 温度系数设为0，表示输出固定、不随机、不创意
        temperature=0
    )

    # 创建特斯拉数据查询对象，传入大模型实例
    data = TeslaDataSource(llm)

    # 定义Agent可调用的工具列表
    tools = [
        # 第一个工具：查询车型信息
        Tool(
            # 工具名称
            name="查询车型信息",
            # 工具绑定的执行函数
            func=data.find_product_description,
            # 工具功能描述，给模型看的
            description="输入车型名称，返回价格和描述"
        ),
        # 第二个工具：查询公司信息
        Tool(
            name="查询公司信息",
            func=data.find_company_info,
            description="查询公司介绍"
        ),
    ]

    # 创建自定义提示词模板实例
    prompt = CustomPromptTemplate(
        # 传入系统提示词模板
        template=AGENT_TMPL,
        # 传入工具列表
        tools=tools,
        # 指定模板需要的输入变量
        input_variables=["input", "intermediate_steps"]
    )

    # 创建LLMChain实例，将大模型与提示词绑定
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # 创建单步动作Agent实例
    agent = LLMSingleActionAgent(
        # 绑定LLMChain
        llm_chain=llm_chain,
        # 绑定自定义输出解析器
        output_parser=CustomParser(),
        # 设置停止词，遇到Observation:停止生成
        stop=["\nObservation:"],
        # 设置允许调用的工具名称列表
        allowed_tools=[t.name for t in tools]
    )

    # 创建Agent执行器，用于运行Agent
    agent_executor = AgentExecutor.from_agent_and_tools(
        # 传入创建好的Agent
        agent=agent,
        # 传入工具列表
        tools=tools,
        # 开启详细日志输出
        verbose=True,
        # 开启解析错误容错，避免程序崩溃
        handle_parsing_errors=True
    )

    # 打印启动成功提示
    print("特斯拉助手已启动")
    # 开启无限循环，持续接收用户输入
    while True:
        # 获取用户输入，并去除首尾空格
        user_input = input("\n请输入问题：").strip()
        # 判断用户是否输入退出指令
        if user_input.lower() in ["q", "quit", "exit"]:
            # 打印再见并退出循环
            print("再见")
            break
        # 如果用户输入为空，跳过本次循环
        if not user_input:
            continue
        # 尝试执行对话逻辑
        try:
            # 调用执行器，传入用户问题
            res = agent_executor.invoke({"input": user_input})
            # 格式化输出回答内容
            output_response(res["output"])
        # 捕获所有异常并打印
        except Exception as e:
            print("错误：", e)
