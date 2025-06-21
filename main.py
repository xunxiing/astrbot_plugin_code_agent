# main.py (最终智能解析版)

import os
import sys
import zipfile
import uuid
import re # <-- 新增导入，用于正则表达式解析

# 将插件目录添加到 Python 搜索路径
plugin_dir = os.path.dirname(__file__)
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

import asyncio
from typing import List, Dict, Any

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig 
import astrbot.api.message_components as Comp

# ------------------- 打包的 smol-agent 和工具 -------------------
try:
    from smolagents.agents import CodeAgent
    from smolagents.default_tools import WebSearchTool
    from smolagents.models import ChatMessage
except ImportError as e:
    logger.error(f"导入 CodeAgent 或其工具/模型失败: {e}。请检查 'smolagents' 文件夹及依赖。")
    raise ImportError(f"无法导入 smolagents 模块: {e}")
# ----------------------------------------------------------------

class AstrBotLLMBridge:
    def __init__(self, context: Context, event: AstrMessageEvent):
        self.context = context
        self.event = event
        self.provider = self.context.get_using_provider()
        if not self.provider:
            raise ValueError("AstrBot 未配置或启用任何 LLM Provider。")
        self.loop = asyncio.get_running_loop()

    def generate(self, messages: List[Any], **kwargs) -> ChatMessage:
        async def async_generate():
            try:
                converted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            except AttributeError:
                logger.error(f"消息格式转换失败，收到的原始消息: {messages}")
                raise TypeError("无法将收到的消息对象转换为 'role'/'content' 字典。")

            prompt = converted_messages[-1]["content"] if converted_messages else ""
            contexts = converted_messages[:-1] if len(converted_messages) > 1 else []

            logger.info(f"[CodeAgent] Calling LLM with prompt: {prompt}")
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(self.event.unified_msg_origin)
            conversation = None
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(self.event.unified_msg_origin, curr_cid)
            
            llm_response = await self.provider.text_chat(
                prompt=prompt, contexts=contexts, session_id=curr_cid, conversation=conversation, **kwargs 
            )
            
            if llm_response.role == "assistant":
                full_response_text = llm_response.completion_text
                
                # --- 关键变更：智能解析器，提前提取最干净的代码 ---
                # 使用正则表达式从 LLM 的完整输出中提取 <code>...</code> 之间的内容
                # re.DOTALL 标志让 . 可以匹配包括换行符在内的任何字符
                match = re.search(r"<code>(.*)</code>", full_response_text, re.DOTALL)
                
                if match:
                    # 如果找到了匹配项，只返回代码部分
                    clean_code = match.group(1).strip()
                    logger.info(f"[CodeAgent] 成功提取代码:\n{clean_code}")
                    return ChatMessage(role="assistant", content=clean_code)
                else:
                    # 如果没有找到 <code> 标签，则按原样返回，但这种情况很少见
                    logger.warning("[CodeAgent] 未在LLM响应中找到 <code> 标签，将返回完整响应。")
                    return ChatMessage(role="assistant", content=full_response_text)
                # ---------------------------------------------------

            else:
                logger.error(f"[CodeAgent] Unexpected LLM response role: {llm_response.role}")
                return ChatMessage(role="assistant", content="错误: LLM 未返回有效消息。")

        future = asyncio.run_coroutine_threadsafe(async_generate(), self.loop)
        return future.result()


@register("CodeAgentPlugin", "YourName", "集成 Smol-Agent 的编程智能体", "1.0.0")
class CodeAgentPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        logger.warning("编程智能体插件已加载。警告：此插件会直接执行由大语言模型生成的代码，存在安全风险。")

    @filter.command("code_agent", alias={'ca', '编程'})
    async def run_code_agent(self, event: AstrMessageEvent, task: str):
        if not self.context.get_using_provider():
            yield event.plain_result("错误：机器人管理员尚未配置大语言模型。")
            return

        yield event.plain_result(f"🤖 收到任务：【{task}】\n智能体开始思考... 预计需要3-4分钟")

        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        run_id = uuid.uuid4().hex
        py_file_path = os.path.join(temp_dir, f"{run_id}.py")
        zip_file_path = os.path.join(temp_dir, f"{run_id}.zip")

        try:
            llm_bridge = AstrBotLLMBridge(self.context, event)
            agent_tools = [WebSearchTool()]
            agent = CodeAgent(model=llm_bridge, tools=agent_tools)

            logger.warning(f"即将执行 CodeAgent.run，任务: '{task}'.")
            loop = asyncio.get_event_loop()
            max_steps = self.config.get('max_iterations', 7)
            final_result = await loop.run_in_executor(
                None, 
                lambda: agent.run(task, max_steps=max_steps)
            )

            logger.info(f"智能体执行完成，生成的代码: {final_result}")

            code_string = str(final_result)
            with open(py_file_path, "w", encoding="utf-8") as f:
                f.write(code_string)

            with zipfile.ZipFile(zip_file_path, 'w') as zipf:
                zipf.write(py_file_path, arcname='main.py')
            
            yield event.chain_result([
                Comp.Plain(f"✅ 任务【{task}】完成！\n为您生成了代码压缩包:"),
                Comp.File(file=zip_file_path, name=f"code_{run_id[:8]}.zip")
            ])

        except Exception as e:
            logger.error(f"编程智能体执行出错: {e}", exc_info=True)
            yield event.plain_result(f"❌ 智能体在执行任务时遇到错误：\n{e}")
        finally:
            if os.path.exists(py_file_path):
                os.remove(py_file_path)
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            event.stop_event()