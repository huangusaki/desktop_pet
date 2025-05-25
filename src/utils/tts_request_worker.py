import asyncio
import aiohttp
from typing import Optional, Any
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread


class TTSRequestWorker(QObject):
    audio_ready = pyqtSignal(bytes, str)
    tts_error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self, text_to_speak: str, config_manager: Any, parent: Optional[QObject] = None
    ):
        super().__init__(parent)
        self.text_to_speak = text_to_speak
        self.config_manager = config_manager
        self._is_running = True
        self.tts_api_base_url: Optional[str] = (
            self.config_manager.get_tts_api_base_url()
        )
        self.tts_api_endpoint: Optional[str] = (
            self.config_manager.get_tts_api_endpoint()
        )
        self.tts_refer_wav_path: Optional[str] = (
            self.config_manager.get_tts_refer_wav_path()
        )
        self.tts_prompt_text: Optional[str] = self.config_manager.get_tts_prompt_text()
        self.tts_prompt_language: Optional[str] = (
            self.config_manager.get_tts_prompt_language()
        )
        self.tts_text_language: Optional[str] = (
            self.config_manager.get_tts_text_language()
        )
        self.tts_cut_punc_method: Optional[str] = (
            self.config_manager.get_tts_cut_punc_method()
        )
        self.tts_media_type: str = self.config_manager.get_tts_media_type() or "wav"
        self.tts_timeout: int = (
            self.config_manager.get_tts_play_audio_timeout_seconds() or 30
        )

    def stop(self):
        self._is_running = False

    @pyqtSlot()
    def start_tts_request(self):
        """
        这个槽被连接到QThread的started信号，或者由ScreenAnalyzer在线程启动后调用。
        它负责设置并运行asyncio事件循环来执行异步的TTS请求。
        """
        if not self._is_running:
            self.finished.emit()
            return
        loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            if not self._is_running:
                if loop and loop.is_running():
                    loop.stop()
                self.finished.emit()
                return
            task = asyncio.ensure_future(self._perform_tts_request(), loop=loop)

            def task_done_callback(fut: asyncio.Future):
                try:
                    fut.result()
                except Exception as e:
                    if self._is_running:
                        self.tts_error.emit(f"TTSRequestWorker: Async task error: {e}")
                finally:
                    if loop and loop.is_running():
                        loop.stop()
                    self.finished.emit()

            task.add_done_callback(task_done_callback)
            if self._is_running:
                loop.run_forever()
            else:
                if loop and loop.is_running():
                    loop.stop()
                self.finished.emit()
        except Exception as e:
            if self._is_running:
                self.tts_error.emit(
                    f"TTSRequestWorker: Error in start_tts_request setup: {e}"
                )
            if loop and loop.is_running():
                loop.stop()
            self.finished.emit()

    async def _perform_tts_request(self):
        if not self._is_running:
            return
        if not self.tts_api_base_url or not self.tts_api_endpoint:
            self.tts_error.emit("TTS API URL or endpoint not configured.")
            return
        params = {
            "text": self.text_to_speak,
            "text_lang": self.tts_text_language,
            "ref_audio_path": self.tts_refer_wav_path,
            "prompt_lang": self.tts_prompt_language,
            "media_type": self.tts_media_type,
            "top_k": 5,
            "temperature": 1.0,
        }
        if self.tts_prompt_text:
            params["prompt_text"] = self.tts_prompt_text
        if self.tts_cut_punc_method:
            params["text_split_method"] = self.tts_cut_punc_method
        full_url = (
            f"{self.tts_api_base_url.rstrip('/')}/{self.tts_api_endpoint.lstrip('/')}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                print(
                    f"TTSRequestWorker: Requesting TTS from {full_url} with params: {params}"
                )
                if not self._is_running:
                    return
                async with session.get(
                    full_url, params=params, timeout=self.tts_timeout
                ) as response:
                    if not self._is_running:
                        return
                    response.raise_for_status()
                    audio_content = await response.read()
                    if not self._is_running:
                        return
                    print(
                        f"TTSRequestWorker: TTS audio received, size: {len(audio_content)} bytes, type: {self.tts_media_type}"
                    )
                    self.audio_ready.emit(audio_content, self.tts_media_type)
        except aiohttp.ClientResponseError as e_http:
            if self._is_running:
                error_text = (
                    await e_http.text()
                    if hasattr(e_http, "text") and callable(e_http.text)
                    else e_http.message
                )
                self.tts_error.emit(
                    f"TTS API request failed (status {e_http.status}): {error_text[:500]}"
                )
        except asyncio.TimeoutError:
            if self._is_running:
                self.tts_error.emit(
                    f"TTS API request timed out after {self.tts_timeout}s."
                )
        except Exception as e:
            if self._is_running:
                self.tts_error.emit(f"TTSRequestWorker: TTS request failed: {e}")
