import io
import os
import random
import asyncio
import tempfile
from typing import Optional, List, Dict, Any
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer, QUrl, pyqtSlot
from PyQt6.QtWidgets import QApplication
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from ..utils.prompt_builder import PromptBuilder
from ..utils.tts_request_worker import TTSRequestWorker

try:
    from PIL import ImageGrab, Image
except ImportError:
    print("Pillow library not found, screen analysis might not work as expected.")
    ImageGrab = None
    Image = None


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
        pet_name: str,
        user_name: str,
        available_emotions: List[str],
    ):
        super().__init__()
        self.gemini_client = gemini_client
        self.prompt_builder = prompt_builder
        self.pet_name = pet_name
        self.user_name = user_name
        self.available_emotions_list = available_emotions
        self._is_running = True
        self.gui_is_ready_for_grab.connect(self._trigger_async_grab_and_process)

    def stop(self):
        self._is_running = False

    def start_screenshot_sequence(self):
        if not self._is_running:
            return
        print(f"ScreenAnalysisWorker ({id(self)}): Requesting GUI hide for screenshot.")
        self.request_gui_hide_before_grab.emit()

    @pyqtSlot()
    def _trigger_async_grab_and_process(self):
        worker_id = id(self)
        print(
            f"ScreenAnalysisWorker ({worker_id}): GUI is ready, starting async grab and process."
        )
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
            task = asyncio.ensure_future(self._async_grab_and_process_llm(), loop=loop)

            def task_done_callback(fut: asyncio.Future):
                try:
                    fut.result()
                except Exception as e:
                    if self._is_running:
                        print(
                            f"ScreenAnalysisWorker ({worker_id}): Async task error: {e}"
                        )
                        self.error_occurred.emit(
                            f"ScreenAnalysisWorker: Async task error: {e}"
                        )
                finally:
                    if loop and loop.is_running():
                        print(
                            f"ScreenAnalysisWorker ({worker_id}): Stopping asyncio loop from task_done_callback."
                        )
                        loop.stop()
                    print(
                        f"ScreenAnalysisWorker ({worker_id}): Asyncio task and loop finished."
                    )

            task.add_done_callback(task_done_callback)
            if self._is_running:
                print(
                    f"ScreenAnalysisWorker ({worker_id}): Starting asyncio event loop..."
                )
                loop.run_forever()
                print(
                    f"ScreenAnalysisWorker ({worker_id}): Asyncio event loop has naturally stopped."
                )
            else:
                if loop and loop.is_running():
                    loop.stop()
                print(
                    f"ScreenAnalysisWorker ({worker_id}): Worker was stopped before loop.run_forever()."
                )
        except Exception as e:
            if self._is_running:
                print(
                    f"ScreenAnalysisWorker ({worker_id}): Error in _trigger_async_grab_and_process: {e}"
                )
                self.error_occurred.emit(
                    f"ScreenAnalysisWorker: Error in _trigger_async_grab_and_process: {e}"
                )
            if loop and loop.is_running():
                loop.stop()
        print(
            f"ScreenAnalysisWorker ({worker_id}): _trigger_async_grab_and_process method finished."
        )

    async def _async_grab_and_process_llm(self):
        worker_id = id(self)
        original_screenshot = None
        screenshot_taken_successfully = False
        try:
            if not self._is_running:
                return
            print(f"ScreenAnalysisWorker ({worker_id}): Checking Pillow.")
            if ImageGrab is None or Image is None:
                self.error_occurred.emit(
                    "Pillow library is not installed for screenshot."
                )
                return
            print(f"ScreenAnalysisWorker ({worker_id}): Performing ImageGrab.grab()")
            original_screenshot = await asyncio.to_thread(ImageGrab.grab)
            screenshot_taken_successfully = True
            print(f"ScreenAnalysisWorker ({worker_id}): Screenshot taken successfully.")
        except Exception as e_grab:
            self.error_occurred.emit(f"Error during ImageGrab.grab(): {e_grab}")
        finally:
            print(
                f"ScreenAnalysisWorker ({worker_id}): Requesting GUI show after grab attempt."
            )
            self.request_gui_show_after_grab.emit()
        if (
            not self._is_running
            or not screenshot_taken_successfully
            or not original_screenshot
        ):
            print(
                f"ScreenAnalysisWorker ({worker_id}): Bailing out early. is_running={self._is_running}, success={screenshot_taken_successfully}"
            )
            return
        try:
            print(f"ScreenAnalysisWorker ({worker_id}): Processing screenshot for LLM.")
            img_byte_arr = io.BytesIO()
            await asyncio.to_thread(
                original_screenshot.save, img_byte_arr, format="JPEG", quality=75
            )
            img_bytes = img_byte_arr.getvalue()
            mime_type = "image/jpeg"
            if not self._is_running:
                return
            user_supplementary_notes_for_image = ""
            print(
                f"ScreenAnalysisWorker ({worker_id}): Sending image to Gemini. Supplementary notes: '{user_supplementary_notes_for_image}'"
            )
            response_data = await asyncio.to_thread(
                self.gemini_client.send_message_with_image,
                image_bytes=img_bytes,
                mime_type=mime_type,
                prompt_text=user_supplementary_notes_for_image,
            )
            print(f"ScreenAnalysisWorker ({worker_id}): Gemini response received.")
            if not self._is_running:
                return
            if response_data and "text" in response_data and "emotion" in response_data:
                print(
                    f"ScreenAnalysisWorker ({worker_id}): Emitting analysis_complete."
                )
                self.analysis_complete.emit(response_data)
            else:
                self.error_occurred.emit(
                    f"Invalid response from Gemini: {response_data}"
                )
        except Exception as e_process_llm:
            if self._is_running:
                self.error_occurred.emit(
                    f"Error during screenshot processing or LLM call: {e_process_llm}"
                )
        print(
            f"ScreenAnalysisWorker ({worker_id}): _async_grab_and_process_llm finished."
        )


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
        self._interval_ms = 60000
        self._analysis_chance = 0.1
        self.tts_enabled_globally = False
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
        self.tts_request_thread: Optional[QThread] = None
        self.tts_request_worker: Optional[TTSRequestWorker] = None
        if not (ImageGrab and Image):
            self._is_enabled = False
            print(
                "ScreenAnalyzer: Pillow not found, screen analysis and related TTS disabled."
            )

    def _load_config(self):
        self._is_enabled = self.config_manager.get_screen_analysis_enabled()
        interval_seconds = self.config_manager.get_screen_analysis_interval_seconds()
        self._interval_ms = interval_seconds * 1000
        self._analysis_chance = self.config_manager.get_screen_analysis_chance()
        self.tts_enabled_globally = self.config_manager.get_tts_enabled()
        print(
            f"ScreenAnalyzer Config: Enabled={self._is_enabled}, Interval={interval_seconds}s, Chance={self._analysis_chance*100}%, TTS Globally Enabled={self.tts_enabled_globally}"
        )

    def start_monitoring(self):
        if not self._is_enabled:
            print("ScreenAnalyzer: Monitoring not started (disabled).")
            return
        if not self.gemini_client or not self.pet_window:
            print(
                "ScreenAnalyzer: Cannot start, Gemini client or PetWindow not available."
            )
            return
        self.timer.start(self._interval_ms)
        print(
            f"ScreenAnalyzer: Monitoring started. Interval: {self._interval_ms / 1000}s"
        )

    def stop_monitoring(self):
        self.timer.stop()
        print("ScreenAnalyzer: Attempting to stop monitoring and clean up threads...")
        if self.analysis_thread and self.analysis_thread.isRunning():
            print(
                f"ScreenAnalyzer: Stopping active analysis_worker ({id(self.analysis_worker)} if exists)."
            )
            if self.analysis_worker:
                self.analysis_worker.stop()
            print(
                f"ScreenAnalyzer: Quitting analysis_thread ({id(self.analysis_thread)})."
            )
            self.analysis_thread.quit()
            if not self.analysis_thread.wait(3000):
                print(
                    "ScreenAnalyzer: Analysis worker thread did not finish gracefully, terminating."
                )
                self.analysis_thread.terminate()
                self.analysis_thread.wait()
        else:
            self._cleanup_analysis_thread_and_worker()
        if self.tts_request_thread and self.tts_request_thread.isRunning():
            print(
                f"ScreenAnalyzer: Stopping active tts_request_worker ({id(self.tts_request_worker)} if exists)."
            )
            if self.tts_request_worker:
                self.tts_request_worker.stop()
            print(
                f"ScreenAnalyzer: Quitting tts_request_thread ({id(self.tts_request_thread)})."
            )
            self.tts_request_thread.quit()
            if not self.tts_request_thread.wait(1000):
                print(
                    "ScreenAnalyzer: TTS request thread did not finish gracefully, terminating."
                )
                self.tts_request_thread.terminate()
                self.tts_request_thread.wait()
        else:
            self._cleanup_tts_request_thread_and_worker()
        print("ScreenAnalyzer: Monitoring stopped and cleanup attempted.")

    def _check_and_analyze_wrapper(self):
        if not self._is_enabled:
            self.timer.stop()
            return
        thread_obj_id = id(self.analysis_thread) if self.analysis_thread else "None"
        is_running_status = (
            self.analysis_thread.isRunning() if self.analysis_thread else "N/A"
        )
        print(
            f"ScreenAnalyzer: _check_and_analyze_wrapper: self.analysis_thread (ID: {thread_obj_id}) is {self.analysis_thread}, isRunning: {is_running_status}"
        )
        if random.random() < self._analysis_chance:
            if self.analysis_thread and self.analysis_thread.isRunning():
                print("ScreenAnalyzer: Screen analysis already in progress. Skipping.")
                return
            self._initiate_analysis_sequence()
        else:
            print(
                f"ScreenAnalyzer: Tick - no analysis this time (chance: {self._analysis_chance*100}%)."
            )

    def _initiate_analysis_sequence(self):
        if self.analysis_thread:
            print(
                f"ScreenAnalyzer: INITIATE_ANALYSIS - Found existing analysis_thread (ID: {id(self.analysis_thread)}), attempting cleanup first."
            )
            self._cleanup_analysis_thread_and_worker()
        if self.analysis_thread and self.analysis_thread.isRunning():
            print(
                "ScreenAnalyzer: INITIATE_ANALYSIS - Previous analysis_thread still running after cleanup attempt. Aborting new sequence."
            )
            return
        print(
            "ScreenAnalyzer: INITIATE_ANALYSIS - Initiating new screen analysis sequence."
        )
        self.analysis_worker = ScreenAnalysisWorker(
            gemini_client=self.gemini_client,
            prompt_builder=self.prompt_builder,
            pet_name=self.pet_name,
            user_name=self.user_name,
            available_emotions=self.available_emotions_list,
        )
        self.analysis_thread = QThread()
        print(
            f"ScreenAnalyzer: INITIATE_ANALYSIS - Created new analysis_thread (ID: {id(self.analysis_thread)}) for worker (ID: {id(self.analysis_worker)})."
        )
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
        print("ScreenAnalyzer: Handling hide request. Making pet window transparent.")
        if self.pet_window:
            self._pet_was_visible_before_grab = self.pet_window.isVisible()
            if self._pet_was_visible_before_grab:
                self.pet_window.setWindowOpacity(0.01)
                QApplication.processEvents()
        self.ready_for_worker_grab.emit()

    @pyqtSlot()
    def _handle_show_request(self):
        print("ScreenAnalyzer: Handling show request. Restoring pet window opacity.")
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
        print(
            f"ScreenAnalyzer: LLM Response - Text (CN): '{text_chinese}', Emotion: '{emotion}', Text (JP): '{text_japanese}'"
        )
        self.pet_reaction_ready.emit(text_chinese, emotion)
        if self.analysis_thread and self.analysis_thread.isRunning():
            print(
                f"ScreenAnalyzer: LLM response handled, requesting analysis_thread (ID: {id(self.analysis_thread)}) to quit."
            )
            self.analysis_thread.quit()
        else:
            print(
                f"ScreenAnalyzer: LLM response handled, but analysis_thread (ID: {id(self.analysis_thread) if self.analysis_thread else 'None'}) was not running or None."
            )
        if self.tts_enabled_globally:
            if text_japanese and text_japanese.strip():
                print(
                    f"ScreenAnalyzer: TTS globally enabled and Japanese text found. Initiating TTS for: '{text_japanese[:50]}...'"
                )
                self._initiate_tts_request(text_japanese)
            elif not (text_japanese and text_japanese.strip()):
                print(
                    "ScreenAnalyzer: TTS globally enabled, but Japanese text for TTS is missing, empty, or null. Skipping TTS for this interaction."
                )
        else:
            if not self.tts_enabled_globally:
                print("ScreenAnalyzer: TTS is globally disabled.")
            if not text_japanese:
                print(
                    "ScreenAnalyzer: Japanese text from LLM for TTS was not provided or was null."
                )

    def _initiate_tts_request(self, text_to_speak: str):
        if self.tts_request_thread:
            print(
                f"ScreenAnalyzer: INITIATE_TTS - Found existing tts_request_thread (ID: {id(self.tts_request_thread)}), attempting cleanup first."
            )
            self._cleanup_tts_request_thread_and_worker()
        if self.tts_request_thread and self.tts_request_thread.isRunning():
            print(
                "ScreenAnalyzer: INITIATE_TTS - Previous tts_request_thread still running. Skipping new one."
            )
            return
        print(
            f"ScreenAnalyzer: INITIATE_TTS - Initiating new TTS request for: '{text_to_speak[:50]}...'"
        )
        self.tts_request_worker = TTSRequestWorker(text_to_speak, self.config_manager)
        self.tts_request_thread = QThread()
        print(
            f"ScreenAnalyzer: INITIATE_TTS - Created new tts_request_thread (ID: {id(self.tts_request_thread)}) for worker (ID: {id(self.tts_request_worker)})."
        )
        self.tts_request_worker.moveToThread(self.tts_request_thread)
        self.tts_request_worker.audio_ready.connect(self._handle_audio_playback)
        self.tts_request_worker.tts_error.connect(self._handle_tts_request_error)
        self.tts_request_worker.finished.connect(
            self._cleanup_tts_request_thread_and_worker
        )
        self.tts_request_thread.started.connect(
            self.tts_request_worker.start_tts_request
        )
        self.tts_request_thread.start()

    @pyqtSlot(bytes, str)
    def _handle_audio_playback(self, audio_data: bytes, media_type: str):
        print(
            f"ScreenAnalyzer: Received audio data for playback (type: {media_type}). Size: {len(audio_data)} bytes."
        )
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            print("ScreenAnalyzer: Player is already playing, stopping previous audio.")
            self.player.stop()
            if self._temp_audio_file_path and os.path.exists(
                self._temp_audio_file_path
            ):
                try:
                    os.remove(self._temp_audio_file_path)
                except Exception as e:
                    print(f"ScreenAnalyzer: Error removing previous temp file: {e}")
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
            print(
                f"ScreenAnalyzer: Audio data written to temporary file: {self._temp_audio_file_path}"
            )
            self.player.setSource(QUrl.fromLocalFile(self._temp_audio_file_path))
            self.player.play()
            print("ScreenAnalyzer: Initiated audio playback.")
        except Exception as e:
            print(f"ScreenAnalyzer: Error preparing/playing audio: {e}")
            self._delete_temp_file(self._temp_audio_file_path)

    def _handle_media_status_changed(self, status: QMediaPlayer.MediaStatus):
        print(f"ScreenAnalyzer: Media status changed to: {status}")
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            print("ScreenAnalyzer: Audio playback finished.")
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
            print("ScreenAnalyzer: Invalid media for playback.")
            self.player.setSource(QUrl())
            self._delete_temp_file(self._temp_audio_file_path)

    def _delete_temp_file(self, file_path: Optional[str]):
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"ScreenAnalyzer: Temporary audio file {file_path} deleted.")
            except Exception as e:
                print(
                    f"ScreenAnalyzer: Error deleting temporary audio file {file_path}: {e}"
                )
        if file_path == self._temp_audio_file_path:
            self._temp_audio_file_path = None

    def _handle_player_error(
        self, error: QMediaPlayer.Error, error_string: Optional[str] = None
    ):
        actual_error_string = self.player.errorString()
        print(f"ScreenAnalyzer: QMediaPlayer Error: ({error}) {actual_error_string}")
        self.player.setSource(QUrl())
        self._delete_temp_file(self._temp_audio_file_path)

    @pyqtSlot(str)
    def _handle_screen_worker_error(self, error_message: str):
        print(f"ScreenAnalyzer: ScreenAnalysisWorker Error - {error_message}")
        if self.pet_window and self.pet_window.windowOpacity() < 1.0:
            if self._pet_was_visible_before_grab:
                self.pet_window.setWindowOpacity(1.0)
                if not self.pet_window.isVisible():
                    self.pet_window.show()
                self.pet_window.activateWindow()
                self.pet_window.raise_()
                QApplication.processEvents()
        if self.analysis_thread and self.analysis_thread.isRunning():
            print(
                f"ScreenAnalyzer: Screen worker error, requesting analysis_thread (ID: {id(self.analysis_thread)}) to quit."
            )
            self.analysis_thread.quit()
        else:
            print(
                f"ScreenAnalyzer: Screen worker error, but analysis_thread (ID: {id(self.analysis_thread) if self.analysis_thread else 'None'}) was not running or None."
            )

    @pyqtSlot(str)
    def _handle_tts_request_error(self, error_message: str):
        print(f"ScreenAnalyzer: TTSRequestWorker Error - {error_message}")

    def _cleanup_analysis_thread_and_worker(self):
        thread_id_before = id(self.analysis_thread) if self.analysis_thread else "None"
        worker_id_before = id(self.analysis_worker) if self.analysis_worker else "None"
        is_running_before = (
            self.analysis_thread.isRunning() if self.analysis_thread else "N/A"
        )
        print(
            f"ScreenAnalyzer: CLEANUP_ANALYSIS - START. Target thread (ID: {thread_id_before}), worker (ID: {worker_id_before}), isRunning: {is_running_before}"
        )
        if self.analysis_worker:
            print(
                f"ScreenAnalyzer: CLEANUP_ANALYSIS - Disconnecting signals from analysis_worker (ID: {id(self.analysis_worker)})."
            )
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
            print(
                f"ScreenAnalyzer: CLEANUP_ANALYSIS - analysis_worker (formerly ID: {worker_id_before}) set to None and scheduled for deletion."
            )
        if self.analysis_thread:
            target_thread_id = id(self.analysis_thread)
            print(
                f"ScreenAnalyzer: CLEANUP_ANALYSIS - Processing analysis_thread (ID: {target_thread_id})."
            )
            if self.analysis_thread.isRunning():
                print(
                    f"ScreenAnalyzer: CLEANUP_ANALYSIS - analysis_thread (ID: {target_thread_id}) is still running. Quitting and waiting..."
                )
                self.analysis_thread.quit()
                if not self.analysis_thread.wait(1500):
                    print(
                        f"ScreenAnalyzer: CLEANUP_ANALYSIS - analysis_thread (ID: {target_thread_id}) did not quit gracefully. Terminating."
                    )
                    self.analysis_thread.terminate()
                    self.analysis_thread.wait()
                else:
                    print(
                        f"ScreenAnalyzer: CLEANUP_ANALYSIS - analysis_thread (ID: {target_thread_id}) quit gracefully."
                    )
            else:
                print(
                    f"ScreenAnalyzer: CLEANUP_ANALYSIS - analysis_thread (ID: {target_thread_id}) was not running."
                )
            self.analysis_thread.deleteLater()
            self.analysis_thread = None
            print(
                f"ScreenAnalyzer: CLEANUP_ANALYSIS - analysis_thread (formerly ID: {target_thread_id}) set to None and scheduled for deletion."
            )
        else:
            print(
                f"ScreenAnalyzer: CLEANUP_ANALYSIS - self.analysis_thread was already None when cleanup was called for it."
            )
        print(
            f"ScreenAnalyzer: CLEANUP_ANALYSIS - END. self.analysis_thread is now: {self.analysis_thread}"
        )

    def _cleanup_tts_request_thread_and_worker(self):
        thread_id_before = (
            id(self.tts_request_thread) if self.tts_request_thread else "None"
        )
        worker_id_before = (
            id(self.tts_request_worker) if self.tts_request_worker else "None"
        )
        is_running_before = (
            self.tts_request_thread.isRunning() if self.tts_request_thread else "N/A"
        )
        print(
            f"ScreenAnalyzer: CLEANUP_TTS - START. Target thread (ID: {thread_id_before}), worker (ID: {worker_id_before}), isRunning: {is_running_before}"
        )
        if self.tts_request_worker:
            print(
                f"ScreenAnalyzer: CLEANUP_TTS - Disconnecting signals from tts_request_worker (ID: {id(self.tts_request_worker)})."
            )
            try:
                self.tts_request_worker.audio_ready.disconnect(
                    self._handle_audio_playback
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.tts_request_worker.tts_error.disconnect(
                    self._handle_tts_request_error
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.tts_request_worker.finished.disconnect(
                    self._cleanup_tts_request_thread_and_worker
                )
            except (TypeError, RuntimeError):
                pass
            self.tts_request_worker.deleteLater()
            self.tts_request_worker = None
            print(
                f"ScreenAnalyzer: CLEANUP_TTS - tts_request_worker (formerly ID: {worker_id_before}) set to None and scheduled for deletion."
            )
        if self.tts_request_thread:
            target_thread_id = id(self.tts_request_thread)
            print(
                f"ScreenAnalyzer: CLEANUP_TTS - Processing tts_request_thread (ID: {target_thread_id})."
            )
            if self.tts_request_thread.isRunning():
                print(
                    f"ScreenAnalyzer: CLEANUP_TTS - tts_request_thread (ID: {target_thread_id}) is still running. Quitting and waiting..."
                )
                self.tts_request_thread.quit()
                if not self.tts_request_thread.wait(1000):
                    print(
                        f"ScreenAnalyzer: CLEANUP_TTS - tts_request_thread (ID: {target_thread_id}) did not quit gracefully. Terminating."
                    )
                    self.tts_request_thread.terminate()
                    self.tts_request_thread.wait()
                else:
                    print(
                        f"ScreenAnalyzer: CLEANUP_TTS - tts_request_thread (ID: {target_thread_id}) quit gracefully."
                    )
            else:
                print(
                    f"ScreenAnalyzer: CLEANUP_TTS - tts_request_thread (ID: {target_thread_id}) was not running."
                )
            self.tts_request_thread.deleteLater()
            self.tts_request_thread = None
            print(
                f"ScreenAnalyzer: CLEANUP_TTS - tts_request_thread (formerly ID: {target_thread_id}) set to None and scheduled for deletion."
            )
        else:
            print(
                f"ScreenAnalyzer: CLEANUP_TTS - self.tts_request_thread was already None when cleanup was called for it."
            )
        print(
            f"ScreenAnalyzer: CLEANUP_TTS - END. self.tts_request_thread is now: {self.tts_request_thread}"
        )
