import subprocess
import pyautogui
import os
import time
import platform
import logging
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger("AgentCore")


class AgentCore:
    def __init__(self, config_manager: Any, prompt_builder: Any, gemini_client: Any):
        self.config_manager = config_manager
        self.prompt_builder = prompt_builder
        self.gemini_client = gemini_client
        self.is_agent_mode_active = False
        self.tools: Dict[str, Callable] = {
            "open_application": self.open_application,
            "type_text": self.type_text,
            "press_key": self.press_key,
            "click_at": self.click_at,
            "create_file_with_content": self.create_file_with_content,
            "read_file_content": self.read_file_content,
            "get_active_window_title": self.get_active_window_title,
        }
        logger.info("Agent核心已初始化")

    def set_agent_mode(self, active: bool):
        self.is_agent_mode_active = active
        logger.info(f"Agent模式已设置为: {active}")
        if active:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = self.config_manager.get_agent_pyautogui_pause()
        else:
            pass

    async def process_user_request(
        self, user_request: str, media_files: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        if not self.is_agent_mode_active:
            return {
                "text": "Agent模式未激活，无法执行操作。",
                "emotion": "confused",
                "action_performed": False,
            }
        try:
            agent_prompt = self.prompt_builder.build_agent_decision_prompt(
                user_request, list(self.tools.keys()), media_files=media_files
            )
            logger.info(
                f"AgentCore: Sending to LLM for multi-step plan: {agent_prompt[:200]}..."
            )
            llm_plan_response = await self.gemini_client.send_message(
                message_text=agent_prompt, hippocampus_manager=None, is_agent_mode=True
            )
            logger.info(f"AgentCore: LLM raw plan response: {llm_plan_response}")
            if llm_plan_response.get("is_error", False) or not llm_plan_response.get(
                "steps"
            ):
                error_text = llm_plan_response.get(
                    "text", "抱歉，我无法规划如何执行您的指令。"
                )
                thinking_process = llm_plan_response.get(
                    "thinking_process", "LLM未能提供有效的执行计划。"
                )
                logger.warning(
                    f"AgentCore: LLM failed to provide valid steps. Response: {llm_plan_response}"
                )
                return {
                    "text": error_text,
                    "emotion": llm_plan_response.get("emotion", "sad"),
                    "action_performed": False,
                    "action_summary": thinking_process,
                }
            planned_steps: List[Dict[str, Any]] = llm_plan_response.get("steps", [])
            llm_initial_text = llm_plan_response.get("text", "正在处理您的指令...")
            llm_overall_thinking = llm_plan_response.get(
                "thinking_process", "LLM正在处理..."
            )
            llm_emotion = llm_plan_response.get("emotion", "neutral")
            cumulative_feedback_for_user = [llm_initial_text]
            all_steps_succeeded = True
            if not planned_steps:
                cannot_do_text = (
                    llm_initial_text
                    if "无法" in llm_initial_text or "抱歉" in llm_initial_text
                    else llm_overall_thinking
                )
                if cannot_do_text not in cumulative_feedback_for_user:
                    cumulative_feedback_for_user.append(cannot_do_text)
                return {
                    "text": "\n".join(filter(None, cumulative_feedback_for_user)),
                    "emotion": llm_emotion,
                    "action_performed": False,
                    "action_summary": llm_overall_thinking,
                }
            step_delay_seconds = self.config_manager.get_agent_step_delay_seconds()
            for i, step in enumerate(planned_steps):
                tool_name = step.get("tool_to_call")
                tool_args = step.get("tool_arguments", {})
                step_description = step.get(
                    "step_description", f"执行步骤 {i + 1}: {tool_name}"
                )
                cumulative_feedback_for_user.append(
                    f"\n[步骤 {i + 1}/{len(planned_steps)}] {step_description}"
                )
                if tool_name and tool_name in self.tools:
                    tool_function = self.tools[tool_name]
                    logger.info(
                        f"AgentCore: Executing step {i + 1}: Tool '{tool_name}' with args: {tool_args}"
                    )
                    if step_delay_seconds > 0 and i > 0:
                        logger.info(
                            f"Waiting {step_delay_seconds}s before executing step {i + 1}..."
                        )
                        time.sleep(step_delay_seconds)
                    try:
                        if not isinstance(tool_args, dict):
                            raise ValueError(
                                f"Tool arguments for '{tool_name}' (step {i + 1}) must be a dictionary, got {type(tool_args)}"
                            )
                        execution_result = tool_function(**tool_args)
                        step_result_message = ""
                        if isinstance(execution_result, dict):
                            if execution_result.get("success"):
                                step_result_message = f"成功: {execution_result.get('message', '操作完成')}"
                                if execution_result.get("content"):
                                    step_result_message += f"\n读取内容 (部分): {str(execution_result.get('content'))[:100]}..."
                                if execution_result.get("title"):
                                    step_result_message += (
                                        f"\n窗口标题: {execution_result.get('title')}"
                                    )
                            else:
                                error_detail = execution_result.get(
                                    "error", "未知工具错误"
                                )
                                step_result_message = f"失败: {error_detail}"
                                logger.error(
                                    f"Tool execution failed (step {i + 1}): {tool_name}, Error: {error_detail}"
                                )
                                all_steps_succeeded = False
                        elif isinstance(execution_result, str):
                            step_result_message = f"结果: {execution_result}"
                        cumulative_feedback_for_user.append(step_result_message)
                        if not all_steps_succeeded:
                            break
                    except Exception as e_exec:
                        error_msg = f"执行工具 '{tool_name}' (步骤 {i + 1}) 时发生错误: {e_exec}"
                        logger.error(error_msg, exc_info=True)
                        cumulative_feedback_for_user.append(f"错误: {error_msg}")
                        all_steps_succeeded = False
                        break
                else:
                    error_msg = f"步骤 {i + 1}: 未知或无效的工具 '{tool_name}'。"
                    logger.error(error_msg)
                    cumulative_feedback_for_user.append(error_msg)
                    all_steps_succeeded = False
                    break
            final_user_message = "\n".join(filter(None, cumulative_feedback_for_user))
            if all_steps_succeeded and planned_steps:
                final_user_message += "\n\n所有计划步骤已尝试执行完毕。"
            elif not all_steps_succeeded:
                final_user_message += "\n\n由于上述错误，部分或全部步骤未能完成。"
            return {
                "text": final_user_message,
                "emotion": llm_emotion if all_steps_succeeded else "sad",
                "action_performed": True if planned_steps else False,
                "action_summary": llm_overall_thinking,
            }
        except Exception as e:
            logger.error(
                f"AgentCore: Error processing multi-step user request: {e}",
                exc_info=True,
            )
            return {
                "text": f"处理您的多步骤请求时发生内部错误: {str(e)}",
                "emotion": "sad",
                "action_performed": False,
            }

    def open_application(self, app_name: str) -> Dict[str, Any]:
        try:
            logger.info(f"Attempting to open application: {app_name}")
            system = platform.system().lower()
            app_name_lower = app_name.lower()
            success_msg = f"已尝试启动应用程序 '{app_name}'。"
            if system == "windows":
                if app_name_lower in ["vscode", "code", "visual studio code"]:
                    common_paths = [
                        os.path.expandvars(
                            r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"
                        ),
                        os.path.expandvars(
                            r"%ProgramFiles%\Microsoft VS Code\Code.exe"
                        ),
                        "code",
                    ]
                    for path_or_cmd in common_paths:
                        if path_or_cmd == "code":
                            try:
                                subprocess.Popen(
                                    ["code"],
                                    creationflags=subprocess.CREATE_NO_WINDOW,
                                    shell=True,
                                )
                                success_msg = "已尝试通过 'code' 命令启动 VS Code。"
                                logger.info(success_msg)
                                return {"success": True, "message": success_msg}
                            except FileNotFoundError:
                                continue
                        elif os.path.exists(path_or_cmd):
                            subprocess.Popen(
                                [path_or_cmd], creationflags=subprocess.CREATE_NO_WINDOW
                            )
                            success_msg = f"已通过路径 '{path_or_cmd}' 启动 VS Code。"
                            logger.info(success_msg)
                            return {"success": True, "message": success_msg}
                    if app_name_lower in ["vscode", "code"]:
                        subprocess.Popen(
                            'start "" "code"',
                            shell=True,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                        )
                        success_msg = (
                            "已尝试通过 'start code' 启动 VS Code (作为后备)。"
                        )
                        logger.info(success_msg)
                        return {"success": True, "message": success_msg}
                subprocess.Popen(
                    f'start "" "{app_name}"',
                    shell=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                logger.info(success_msg)
                return {"success": True, "message": success_msg}
            elif system == "darwin":
                if app_name_lower in ["vscode", "visual studio code", "code"]:
                    app_to_open = "Visual Studio Code"
                else:
                    app_to_open = app_name
                subprocess.Popen(["open", "-a", app_to_open])
                success_msg = f"已尝试启动应用程序 '{app_to_open}'。"
                logger.info(success_msg)
                return {"success": True, "message": success_msg}
            elif system == "linux":
                cmd_to_run: List[str]
                if app_name_lower in ["vscode", "visual studio code", "code"]:
                    cmd_to_run = ["code"]
                else:
                    cmd_to_run = [app_name]
                subprocess.Popen(
                    cmd_to_run,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                success_msg = f"已尝试启动应用程序 '{' '.join(cmd_to_run)}'。"
                logger.info(success_msg)
                return {"success": True, "message": success_msg}
            else:
                return {"success": False, "error": f"不支持的操作系统: {system}"}
        except FileNotFoundError:
            err_msg = f"应用程序 '{app_name}' 未找到或路径不正确 (FileNotFoundError)。"
            logger.error(err_msg)
            return {"success": False, "error": err_msg}
        except Exception as e:
            err_msg = f"打开应用程序 '{app_name}' 时发生错误: {e}"
            logger.error(err_msg, exc_info=True)
            return {"success": False, "error": err_msg}

    def type_text(self, text: str, interval: float = 0.01) -> Dict[str, Any]:
        try:
            logger.info(f"Typing text: '{text[:50]}...'")
            active_window_delay = (
                self.config_manager.get_agent_active_window_delay_before_type()
            )
            if active_window_delay > 0:
                logger.info(
                    f"Waiting {active_window_delay}s for window to gain focus before typing..."
                )
                time.sleep(active_window_delay)
            pyautogui.typewrite(text, interval=interval)
            return {"success": True, "message": f"已尝试输入文本: {text[:30]}..."}
        except Exception as e:
            err_msg = f"输入文本时发生错误: {e}"
            logger.error(err_msg, exc_info=True)
            return {"success": False, "error": err_msg}

    def press_key(
        self, key_name: str, presses: int = 1, interval: float = 0.1
    ) -> Dict[str, Any]:
        try:
            actual_keys_list = [k.strip().lower() for k in key_name.split("+")]
            logger.info(f"Pressing key(s): {actual_keys_list} ({presses} times)")
            for i in range(presses):
                if len(actual_keys_list) > 1:
                    pyautogui.hotkey(*actual_keys_list)
                else:
                    pyautogui.press(actual_keys_list[0])
                if presses > 1 and i < presses - 1:
                    time.sleep(interval)
            return {
                "success": True,
                "message": f"已尝试按下按键: {key_name} ({presses}次)",
            }
        except Exception as e:
            err_msg = f"按下按键 '{key_name}' 时发生错误: {e}"
            logger.error(err_msg, exc_info=True)
            return {"success": False, "error": err_msg}

    def click_at(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> Dict[str, Any]:
        try:
            valid_buttons = ["left", "middle", "right"]
            button_lower = button.lower()
            if button_lower not in valid_buttons:
                logger.warning(
                    f"Invalid mouse button '{button}'. Defaulting to 'left'."
                )
                button_lower = "left"
            if x is not None and y is not None:
                logger.info(
                    f"Clicking {button_lower} button {clicks} times at ({x}, {y})"
                )
                pyautogui.click(
                    x=x, y=y, clicks=clicks, interval=interval, button=button_lower
                )
            else:
                logger.info(
                    f"Clicking {button_lower} button {clicks} times at current mouse position"
                )
                pyautogui.click(clicks=clicks, interval=interval, button=button_lower)
            return {
                "success": True,
                "message": f"已尝试在 {'({x},{y})' if x is not None else '当前位置'} 点击 {clicks} 次",
            }
        except Exception as e:
            err_msg = f"点击时发生错误: {e}"
            logger.error(err_msg, exc_info=True)
            return {"success": False, "error": err_msg}

    def create_file_with_content(
        self, file_path: str, content: str = ""
    ) -> Dict[str, Any]:
        expanded_file_path = ""
        abs_file_path = ""
        try:
            expanded_file_path = os.path.expanduser(file_path)
            abs_file_path = os.path.abspath(expanded_file_path)
            logger.info(
                f"Attempting to create/overwrite file '{abs_file_path}' with content (length: {len(content)})"
            )
            os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
            with open(abs_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {
                "success": True,
                "message": f"文件 '{abs_file_path}' 已成功创建/更新。",
            }
        except Exception as e:
            resolved_path_for_error = (
                abs_file_path
                if abs_file_path
                else expanded_file_path
                if expanded_file_path
                else file_path
            )
            err_msg = f"创建/写入文件 '{file_path}' (解析为 '{resolved_path_for_error}') 时发生错误: {e}"
            logger.error(err_msg, exc_info=True)
            return {"success": False, "error": err_msg}

    def read_file_content(self, file_path: str) -> Dict[str, Any]:
        expanded_file_path = ""
        abs_file_path = ""
        try:
            expanded_file_path = os.path.expanduser(file_path)
            abs_file_path = os.path.abspath(expanded_file_path)
            logger.info(f"Attempting to read file content from '{abs_file_path}'")
            if not os.path.exists(abs_file_path):
                return {"success": False, "error": f"文件 '{abs_file_path}' 不存在。"}
            if not os.path.isfile(abs_file_path):
                return {
                    "success": False,
                    "error": f"路径 '{abs_file_path}' 不是一个文件。",
                }
            with open(abs_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            max_read_len = self.config_manager.get_agent_max_read_file_length()
            if len(content) > max_read_len:
                summary_content = (
                    content[:max_read_len]
                    + f"\n... [内容过长 ({len(content)} bytes)，已截断至 {max_read_len} bytes]"
                )
            else:
                summary_content = content
            return {
                "success": True,
                "message": f"成功读取文件 '{abs_file_path}'。",
                "content": summary_content,
            }
        except Exception as e:
            resolved_path_for_error = (
                abs_file_path
                if abs_file_path
                else expanded_file_path
                if expanded_file_path
                else file_path
            )
            err_msg = f"读取文件 '{file_path}' (解析为 '{resolved_path_for_error}') 时发生错误: {e}"
            logger.error(err_msg, exc_info=True)
            return {"success": False, "error": err_msg, "content": None}

    def get_active_window_title(self) -> Dict[str, Any]:
        try:
            time.sleep(self.config_manager.get_agent_get_window_title_delay())
            active_window = pyautogui.getActiveWindow()
            if active_window and hasattr(active_window, "title"):
                title = active_window.title
                logger.info(f"Active window title: {title}")
                return {
                    "success": True,
                    "title": title,
                    "message": f"当前活动窗口标题是: {title}",
                }
            elif active_window is None:
                logger.warning(
                    "Could not get active window (pyautogui.getActiveWindow() returned None)."
                )
                return {
                    "success": False,
                    "title": None,
                    "error": "无法获取活动窗口 (getActiveWindow 返回 None)。",
                }
            else:
                logger.warning(
                    f"Got active window object, but it lacks a 'title' attribute: {type(active_window)}"
                )
                return {
                    "success": False,
                    "title": None,
                    "error": "获取到的活动窗口对象缺少标题属性。",
                }
        except Exception as e:
            err_msg = f"获取活动窗口标题时发生错误: {e}"
            logger.error(err_msg, exc_info=True)
            if (
                platform.system().lower() == "windows"
                and "pygetwindow" in str(e).lower()
            ):
                err_msg += (
                    " 这可能是由于没有活动窗口或权限问题。尝试点击一个窗口使其激活。"
                )
            return {"success": False, "title": None, "error": err_msg}
