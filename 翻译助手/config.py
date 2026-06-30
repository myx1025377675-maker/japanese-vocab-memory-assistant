"""
配置管理模块 —— 负责读写用户设置。
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field


CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home() / ".config")) / "desktop_translator"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class AppConfig:
    """应用配置，支持序列化到 JSON。"""
    # 翻译模式: "offline" | "free_web" | "deepseek"
    mode: str = "offline"

    # DeepSeek 配置
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"

    # UI 设置
    language_direction: str = "auto"   # "auto" | "ja_to_zh" | "zh_to_ja"
    window_x: int = 200
    window_y: int = 200
    window_width: int = 420
    window_height: int = 360
    always_on_top: bool = True
    opacity: float = 0.92

    # 快捷键
    hotkey_toggle: str = "ctrl+alt+t"
    hotkey_clipboard: str = "ctrl+alt+c"

    # 缓存设置
    max_history: int = 500
    cache_enabled: bool = True

    def save(self):
        """保存配置到文件。"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.mode,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "model": self.model,
            "language_direction": self.language_direction,
            "window_x": self.window_x,
            "window_y": self.window_y,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "always_on_top": self.always_on_top,
            "opacity": self.opacity,
            "hotkey_toggle": self.hotkey_toggle,
            "hotkey_clipboard": self.hotkey_clipboard,
            "max_history": self.max_history,
            "cache_enabled": self.cache_enabled,
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> "AppConfig":
        """从文件加载配置，不存在则返回默认值。"""
        if not CONFIG_FILE.exists():
            return cls()
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config
        except (json.JSONDecodeError, OSError):
            return cls()
