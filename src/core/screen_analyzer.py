import io
import os
import random
import asyncio
import tempfile
import collections
from typing import Optional, List, Dict, Any
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer, QUrl, pyqtSlot
from PyQt6.QtWidgets import QApplication
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from ..utils.prompt_builder import PromptBuilder
from ..utils.tts_request_worker import TTSRequestWorker
from PIL import ImageGrab, Image
import logging

logger = logging.getLogger("ScreenAnalyzer")


class ScreenAnalysisWorker(QObject):
    request_gui_hide_before_grab = pyqtSignal()
    request_gui_show_after_grab = pyqtSignal()
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    gui_is_ready_for_grab = pyqtSignal()

    def __init__(
        self,
        gemini_client: Any,
        prompt_builder: PromptBuilder,
        config_manager: Any,
        pet_name: str,
        user_name: str,
        available_emotions: List[str],
    ):
        super().__init__()
        self.gemini_client = gemini_client
        self.prompt_builder = prompt_builder
        self.pet_name = pet_name
        self.config_manager = config_manager
        self.user_name = user_name
        self.available_emotions_list = available_emotions
        self._is_running = True
        self.gui_is_ready_for_grab.connect(self._trigger_async_grab_and_process)

    def stop(self):
        self._is_running = False

    def start_screenshot_sequence(self):
        if not self._is_running:
            return
        self.request_gui_hide_before_grab.emit()

    @pyqtSlot()
    def _trigger_async_grab_and_process(self):
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
                return
            task = asyncio.ensure_future(
                self._async_grab_and_process_llm_with_timeout(), loop=loop
            )

            def task_done_callback(fut: asyncio.Future):
                try:
                    fut.result()
                except Exception as e:
                    if self._is_running:
                        self.error_occurred.emit(
                            f"ScreenAnalysisWorker: Async task error: {e}"
                        )
                finally:
                    if loop and loop.is_running():
                        loop.stop()

            task.add_done_callback(task_done_callback)
            if self._is_running:
                loop.run_forever()
            else:
                if loop and loop.is_running():
                    loop.stop()
        except Exception as e:
            if self._is_running:
                self.error_occurred.emit(
                    f"ScreenAnalysisWorker: Error in _trigger_async_grab_and_process: {e}"
                )
            if loop and loop.is_running():
                loop.stop()

    async def _async_grab_and_process_llm_with_timeout(self):
        """包装器方法，使用 asyncio.wait_for 实现超时。"""
        if not self.config_manager:
            logger.error(
                "ScreenAnalysisWorker: ConfigManager is None, cannot get timeout."
            )
            self.error_occurred.emit(
                "ScreenAnalysisWorker: Configuration error (timeout)."
            )
            return
        timeout_seconds = self.config_manager.get_screen_analysis_task_timeout_seconds()
        try:
            logger.debug(
                f"Screen analysis task started with timeout: {timeout_seconds}s"
            )
            await asyncio.wait_for(
                self._async_grab_and_process_llm_core(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Screen analysis task timed out after {timeout_seconds} seconds."
            )
            if self._is_running:
                self.error_occurred.emit(f"屏幕分析任务超时 ({timeout_seconds}秒)。")
        except Exception as e:
            logger.error(
                f"Error in _async_grab_and_process_llm_with_timeout: {e}", exc_info=True
            )

    async def _async_grab_and_process_llm_core(self):
        original_screenshot = None
        screenshot_taken_successfully = False
        try:
            if not self._is_running:
                return
            if ImageGrab is None or Image is None:
                self.error_occurred.emit(
                    "Pillow library is not installed for screenshot."
                )
                return
            original_screenshot = await asyncio.to_thread(ImageGrab.grab)
            screenshot_taken_successfully = True
            self.request_gui_show_after_grab.emit()
        except Exception as e_grab:
            logger.error(f"Error during ImageGrab.grab(): {e_grab}", exc_info=True)
        finally:
            pass
        if (
            not self._is_running
            or not screenshot_taken_successfully
            or not original_screenshot
        ):
            if not screenshot_taken_successfully and self._is_running:
                self.error_occurred.emit("未能成功截取屏幕。")
            return
        try:
            img_byte_arr = io.BytesIO()
            await asyncio.to_thread(
                original_screenshot.save, img_byte_arr, format="JPEG", quality=75
            )
            img_bytes = img_byte_arr.getvalue()
            mime_type = "image/jpeg"
            if not self._is_running:
                return
            user_supplementary_notes_for_image = ""
            response_data = await self.gemini_client.send_message_with_image(
                image_bytes=img_bytes,
                mime_type=mime_type,
                prompt_text=user_supplementary_notes_for_image,
            )
            if not self._is_running:
                return
            if response_data and "text" in response_data and "emotion" in response_data:
                self.analysis_complete.emit(response_data)
            else:
                self.error_occurred.emit(
                    f"Invalid response from Gemini: {response_data}"
                )
        except Exception as e_process_llm:
            raise


class ScreenAnalyzer(QObject):
    pet_reaction_ready = pyqtSignal(str, str)
    ready_for_worker_grab = pyqtSignal()

    def __init__(
        self,
        gemini_client: Any,
        prompt_builder: PromptBuilder,
        config_manager: Any,
        pet_window: Any,
        pet_name: str,
        user_name: str,
        available_emotions: List[str],
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.gemini_client = gemini_client
        self.config_manager = config_manager
        self.pet_window = pet_window
        self.prompt_builder = prompt_builder
        self.pet_name = pet_name
        self.user_name = user_name
        self.available_emotions_list = available_emotions
        self._pet_was_visible_before_grab = False
        self._is_enabled = False
        self._analysis_chance = 0.1
        self.tts_enabled_globally = False
        self.tts_queue = collections.deque()
        self.is_tts_processing = False
        self.tts_request_thread: Optional[QThread] = None
        self.tts_request_worker: Optional[TTSRequestWorker] = None
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self._temp_audio_file_path = None
        self.player.mediaStatusChanged.connect(self._handle_media_status_changed)
        self.player.errorOccurred.connect(self._handle_player_error)
        self._load_config()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check_and_analyze_wrapper)
        self.analysis_thread: Optional[QThread] = None
        self.analysis_worker: Optional[ScreenAnalysisWorker] = None
        if not (ImageGrab and Image):
            self._is_enabled = False
            logger.warning("Pillow not found, screen analysis feature disabled.")
            if not self.tts_enabled_globally:
                logger.warning(
                    "Pillow not found and global TTS is disabled. ScreenAnalyzer will be largely inactive."
                )

    def _load_config(self):
        self._is_enabled = self.config_manager.get_screen_analysis_enabled()
        min_interval_seconds = (
            self.config_manager.get_screen_analysis_min_interval_seconds()
        )
        max_interval_seconds = (
            self.config_manager.get_screen_analysis_max_interval_seconds()
        )
        self._min_interval_ms = min_interval_seconds * 1000
        self._max_interval_ms = max_interval_seconds * 1000
        self._analysis_chance = self.config_manager.get_screen_analysis_chance()
        self.tts_enabled_globally = self.config_manager.get_tts_enabled()
        logger.info(
            f"Config: 屏幕截图={self._is_enabled}, 最小触发时间={min_interval_seconds}s,最大触发时间={max_interval_seconds}s, 触发概率={self._analysis_chance*100}%, TTS开启状态={self.tts_enabled_globally}"
        )

    def start_monitoring(self):
        if not self._is_enabled:
            logger.info(
                "Screen analysis monitoring not started (feature disabled). TTS can still be triggered by chat."
            )
            return
        if not self.gemini_client or not self.pet_window:
            logger.warning(
                "Cannot start screen analysis monitoring, Gemini client or PetWindow not available."
            )
            return
        initial_interval_ms = random.randint(
            self._min_interval_ms, self._max_interval_ms
        )
        self.timer.start(initial_interval_ms)

    def stop_monitoring(self):
        self.timer.stop()
        logger.info("Attempting to stop monitoring and clean up threads...")
        if self.analysis_thread and self.analysis_thread.isRunning():
            if self.analysis_worker:
                self.analysis_worker.stop()
            self.analysis_thread.quit()
            if not self.analysis_thread.wait(3000):
                logger.warning("Analysis thread did not quit gracefully, terminating.")
                self.analysis_thread.terminate()
                self.analysis_thread.wait()
        self._cleanup_analysis_thread_and_worker()
        if self.tts_request_thread and self.tts_request_thread.isRunning():
            if self.tts_request_worker:
                self.tts_request_worker.stop()
            self.tts_request_thread.quit()
            if not self.tts_request_thread.wait(1000):
                logger.warning(
                    "TTS request thread did not quit gracefully, terminating."
                )
                self.tts_request_thread.terminate()
                self.tts_request_thread.wait()
        self._cleanup_tts_request_thread_and_worker()
        self.tts_queue.clear()
        self.is_tts_processing = False
        logger.info("Monitoring stopped, TTS queue cleared, and cleanup attempted.")

    def _check_and_analyze_wrapper(self):
        if not self._is_enabled:
            self.timer.stop()
            return
        should_analyze = random.random() < self._analysis_chance
        if should_analyze:
            if self.analysis_thread and self.analysis_thread.isRunning():
                logger.info("屏幕分析线程运行中，跳过本次任务")
            else:
                logger.debug(f"开始屏幕分析")
                self._initiate_analysis_sequence()
        else:
            logger.debug(f"跳过本次屏幕分析任务")
        if self.timer.isActive():
            next_interval_ms = self._min_interval_ms
            next_interval_ms = random.randint(
                self._min_interval_ms, self._max_interval_ms
            )
            self.timer.setInterval(next_interval_ms)

    def _initiate_analysis_sequence(self):
        if self.analysis_thread:
            self._cleanup_analysis_thread_and_worker()
        if self.analysis_thread and self.analysis_thread.isRunning():
            logger.warning(
                "Attempted to initiate analysis while a thread was still unexpectedly running."
            )
            return
        self.analysis_worker = ScreenAnalysisWorker(
            gemini_client=self.gemini_client,
            prompt_builder=self.prompt_builder,
            config_manager=self.config_manager,
            pet_name=self.pet_name,
            user_name=self.user_name,
            available_emotions=self.available_emotions_list,
        )
        self.analysis_thread = QThread()
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_worker.request_gui_hide_before_grab.connect(
            self._handle_hide_request
        )
        self.analysis_worker.request_gui_show_after_grab.connect(
            self._handle_show_request
        )
        self.analysis_worker.analysis_complete.connect(self._handle_llm_response)
        self.analysis_worker.error_occurred.connect(self._handle_screen_worker_error)
        self.ready_for_worker_grab.connect(self.analysis_worker.gui_is_ready_for_grab)
        self.analysis_thread.started.connect(
            self.analysis_worker.start_screenshot_sequence
        )
        self.analysis_thread.finished.connect(self._cleanup_analysis_thread_and_worker)
        self.analysis_thread.start()

    @pyqtSlot()
    def _handle_hide_request(self):
        if self.pet_window:
            self._pet_was_visible_before_grab = self.pet_window.isVisible()
            if self._pet_was_visible_before_grab:
                self.pet_window.setWindowOpacity(0.50)
                QApplication.processEvents()
        self.ready_for_worker_grab.emit()

    @pyqtSlot()
    def _handle_show_request(self):
        if self.pet_window:
            if self._pet_was_visible_before_grab:
                self.pet_window.setWindowOpacity(1.0)
                if not self.pet_window.isVisible():
                    self.pet_window.show()
                self.pet_window.activateWindow()
                self.pet_window.raise_()
                QApplication.processEvents()

    @pyqtSlot(dict)
    def _handle_llm_response(self, response_data: Dict[str, Any]):
        text_chinese = response_data.get("text", "Hmm...")
        emotion = response_data.get("emotion", "default")
        text_japanese = response_data.get("text_japanese")
        self.pet_reaction_ready.emit(text_chinese, emotion)
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
        else:
            self._cleanup_analysis_thread_and_worker()
        if self.tts_enabled_globally:
            if text_japanese and text_japanese.strip():
                self.request_tts(text_japanese, source="ScreenAnalysis")
            elif not (text_japanese and text_japanese.strip()):
                logger.info(
                    "TTS (Screen Analysis) - Japanese text is missing or empty. Skipping TTS."
                )
        else:
            if not self.tts_enabled_globally:
                logger.info("TTS (Screen Analysis) - Global TTS is disabled.")

    @pyqtSlot(str)
    def play_tts_from_chat(self, japanese_text: str):
        if not self.tts_enabled_globally:
            logger.info("TTS (Chat) - Global TTS is disabled, not playing.")
            return
        if japanese_text and japanese_text.strip():
            self.request_tts(japanese_text, source="ChatDialog")
        else:
            logger.info("TTS (Chat) - Japanese text is missing or empty, skipping.")

    def request_tts(self, text_to_speak: str, source: str = "Unknown"):
        if not self.tts_enabled_globally:
            logger.info(f"TTS request from {source} ignored, global TTS is disabled.")
            return
        if not text_to_speak or not text_to_speak.strip():
            logger.info(f"TTS request from {source} ignored, text is empty.")
            return
        self.tts_queue.append(text_to_speak)
        logger.info(
            f"Added TTS request from {source} to queue (size: {len(self.tts_queue)}): '{text_to_speak[:30]}...'"
        )
        self._try_process_next_tts_in_queue()

    def _try_process_next_tts_in_queue(self):
        if self.is_tts_processing:
            return
        if not self.tts_queue:
            return
        if not self.tts_enabled_globally:
            self.tts_queue.clear()
            return
        if self.tts_request_thread and not self.tts_request_thread.isRunning():
            self._cleanup_tts_request_thread_and_worker()
        elif not self.tts_request_thread and self.tts_request_worker:
            self.tts_request_worker.deleteLater()
            self.tts_request_worker = None
        if self.tts_request_thread and self.tts_request_thread.isRunning():
            logger.warning(
                "_try_process_next_tts_in_queue called while a TTS thread is unexpectedly running."
            )
            return
        self.is_tts_processing = True
        text_to_speak = self.tts_queue.popleft()
        logger.info(f"Processing next TTS from queue: '{text_to_speak[:50]}...'")
        self.tts_request_worker = TTSRequestWorker(text_to_speak, self.config_manager)
        self.tts_request_thread = QThread()
        self.tts_request_worker.moveToThread(self.tts_request_thread)
        self.tts_request_worker.audio_ready.connect(self._handle_audio_playback)
        self.tts_request_worker.tts_error.connect(self._handle_tts_request_error_queued)
        self.tts_request_worker.finished.connect(
            self._handle_tts_worker_finished_queued
        )
        self.tts_request_thread.started.connect(
            self.tts_request_worker.start_tts_request
        )
        self.tts_request_thread.finished.connect(
            self._cleanup_tts_request_thread_and_worker
        )
        self.tts_request_thread.start()

    @pyqtSlot()
    def _handle_tts_worker_finished_queued(self):
        logger.info("TTS worker finished task (queued).")
        if self.tts_request_thread and self.tts_request_thread.isRunning():
            self.tts_request_thread.quit()

    @pyqtSlot(str)
    def _handle_tts_request_error_queued(self, error_message: str):
        logger.error(f"TTSRequestWorker Error (queued) - {error_message}")
        if self.tts_request_thread and self.tts_request_thread.isRunning():
            self.tts_request_thread.quit()

    @pyqtSlot(bytes, str)
    def _handle_audio_playback(self, audio_data: bytes, media_type: str):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.stop()
            if self._temp_audio_file_path and os.path.exists(
                self._temp_audio_file_path
            ):
                try:
                    os.remove(self._temp_audio_file_path)
                except Exception as e_remove:
                    logger.warning(
                        f"Could not remove temp audio file {self._temp_audio_file_path}: {e_remove}"
                    )
                self._temp_audio_file_path = None
        try:
            valid_suffix = "." + (
                media_type.lower() if media_type and media_type.strip() else "wav"
            )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=valid_suffix
            ) as temp_file:
                temp_file.write(audio_data)
                self._temp_audio_file_path = temp_file.name
            self.player.setSource(QUrl.fromLocalFile(self._temp_audio_file_path))
            self.player.play()
        except Exception as e:
            logger.error(f"Error handling audio playback: {e}", exc_info=True)
            self._delete_temp_file(self._temp_audio_file_path)

    def _handle_media_status_changed(self, status: QMediaPlayer.MediaStatus):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.setSource(QUrl())
            if self._temp_audio_file_path and os.path.exists(
                self._temp_audio_file_path
            ):
                QTimer.singleShot(
                    100, lambda p=self._temp_audio_file_path: self._delete_temp_file(p)
                )
            else:
                self._temp_audio_file_path = None
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            logger.error(
                f"QMediaPlayer: Invalid media. Source was: {self.player.source().toString()}"
            )
            self.player.setSource(QUrl())
            self._delete_temp_file(self._temp_audio_file_path)

    def _delete_temp_file(self, file_path: Optional[str]):
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Successfully deleted temp audio file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp audio file {file_path}: {e}")
        if file_path == self._temp_audio_file_path:
            self._temp_audio_file_path = None

    def _handle_player_error(
        self, error: QMediaPlayer.Error, error_string: Optional[str] = None
    ):
        actual_error_string = self.player.errorString()
        logger.error(
            f"QMediaPlayer Error: ({error}) {actual_error_string}. Source: {self.player.source().toString()}"
        )
        self.player.setSource(QUrl())
        self._delete_temp_file(self._temp_audio_file_path)

    @pyqtSlot(str)
    def _handle_screen_worker_error(self, error_message: str):
        logger.error(f"ScreenAnalysisWorker Error - {error_message}")
        if self.pet_window and self.pet_window.windowOpacity() < 1.0:
            if self._pet_was_visible_before_grab:
                self.pet_window.setWindowOpacity(1.0)
                if not self.pet_window.isVisible():
                    self.pet_window.show()
                self.pet_window.activateWindow()
                self.pet_window.raise_()
                QApplication.processEvents()
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
        else:
            self._cleanup_analysis_thread_and_worker()

    def _cleanup_analysis_thread_and_worker(self):
        if self.analysis_worker:
            try:
                self.analysis_worker.request_gui_hide_before_grab.disconnect(
                    self._handle_hide_request
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.analysis_worker.request_gui_show_after_grab.disconnect(
                    self._handle_show_request
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.analysis_worker.analysis_complete.disconnect(
                    self._handle_llm_response
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.analysis_worker.error_occurred.disconnect(
                    self._handle_screen_worker_error
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.ready_for_worker_grab.disconnect(
                    self.analysis_worker.gui_is_ready_for_grab
                )
            except (TypeError, RuntimeError):
                pass
            self.analysis_worker.deleteLater()
            self.analysis_worker = None
        if self.analysis_thread:
            if self.analysis_thread.isRunning():
                logger.debug("Cleaning up running analysis_thread.")
                self.analysis_thread.quit()
                if not self.analysis_thread.wait(1500):
                    logger.warning(
                        "Analysis thread did not quit gracefully in _cleanup, terminating."
                    )
                    self.analysis_thread.terminate()
                    self.analysis_thread.wait()
            self.analysis_thread.deleteLater()
            self.analysis_thread = None

    def _cleanup_tts_request_thread_and_worker(self):
        if self.tts_request_worker:
            try:
                self.tts_request_worker.audio_ready.disconnect(
                    self._handle_audio_playback
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.tts_request_worker.tts_error.disconnect(
                    self._handle_tts_request_error_queued
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.tts_request_worker.finished.disconnect(
                    self._handle_tts_worker_finished_queued
                )
            except (TypeError, RuntimeError):
                pass
            self.tts_request_worker.deleteLater()
            self.tts_request_worker = None
        if self.tts_request_thread:
            if self.tts_request_thread.isRunning():
                logger.debug("Cleaning up running tts_request_thread.")
                self.tts_request_thread.quit()
                if not self.tts_request_thread.wait(500):
                    logger.warning(
                        "TTS request thread did not quit gracefully in _cleanup, terminating."
                    )
                    self.tts_request_thread.terminate()
                    self.tts_request_thread.wait()
            self.tts_request_thread.deleteLater()
            self.tts_request_thread = None
        self.is_tts_processing = False
        QTimer.singleShot(0, self._try_process_next_tts_in_queue)
