import os
import sys
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError, HfHubHTTPError, EntryNotFoundError

# Попытка импорта dotenv для автоматической загрузки .env файла (если библиотека установлена)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

# ==========================================
# ⚙️ КОНФИГУРАЦИЯ (Меняйте только эти поля)
# ==========================================
HF_REPO_MODEL = os.getenv("HF_REPO_MODEL")
FILE_NAME = os.getenv("FILE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_DIR_NAME = os.getenv("MODEL_DIR_NAME")

# Запасной репозиторий для гарантированной загрузки vocab.txt
FALLBACK_VOCAB_REPO = os.getenv("FALLBACK_VOCAB_REPO")
# ==========================================

# Динамическое определение локальной директории
LOCAL_DIR = os.path.join("/content/F5-TTS/ckpts", MODEL_DIR_NAME)
os.makedirs(LOCAL_DIR, exist_ok=True)

# Авторизация/Токен
token = HF_TOKEN if HF_TOKEN and not HF_TOKEN.startswith('hf_xxx') else os.getenv('HF_TOKEN', None)

try:
    print(f"🚀 Подключаемся к Hugging Face репозиторию: {HF_REPO_MODEL}...")

    # 1. Скачивание конкретного файла модели
    print(f"📦 Скачиваем файл весов '{FILE_NAME}'...")
    model_path = hf_hub_download(
        repo_id=HF_REPO_MODEL,
        filename=FILE_NAME,
        local_dir=LOCAL_DIR,
        token=token
    )
    print(f"✅ Файл весов сохранен: {model_path}")

    # 2. Скачивание vocab.txt из текущего или запасного репозитория
    vocab_target_path = os.path.join(LOCAL_DIR, "vocab.txt")
    if not os.path.exists(vocab_target_path):
        try:
            print("📦 Скачиваем 'vocab.txt' из основного репозитория...")
            hf_hub_download(
                repo_id=HF_REPO_MODEL,
                filename="vocab.txt",
                local_dir=LOCAL_DIR,
                token=token
            )
            print("✅ Файл 'vocab.txt' успешно загружен!")
        except (EntryNotFoundError, Exception) as e:
            print(f"⚠️ 'vocab.txt' не найден в {HF_REPO_MODEL}. Скачиваем из fallback-репозитория ({FALLBACK_VOCAB_REPO})...")
            hf_hub_download(
                repo_id=FALLBACK_VOCAB_REPO,
                filename="vocab.txt",
                local_dir=LOCAL_DIR,
                token=token
            )
            print("✅ Файл 'vocab.txt' из fallback-репозитория успешно загружен!")
    else:
        print("ℹ️ Файл 'vocab.txt' уже присутствует локально.")

    print(f"🎉 Готово! Все файлы сохранены в '{LOCAL_DIR}'")

except (RepositoryNotFoundError, GatedRepoError) as e:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Репозиторий '{HF_REPO_MODEL}' не найден или доступ ограничен.\n{e}")

except EntryNotFoundError:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл '{FILE_NAME}' не найден в репозитории '{HF_REPO_MODEL}'.")

except HfHubHTTPError as e:
    if '401' in str(e) or '403' in str(e):
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Ошибка авторизации 401/403 (Невалидный токен).")
    else:
        print(f"❌ Ошибка сети или сервера HF: {e}")

except Exception as e:
    if os.path.exists(os.path.join(LOCAL_DIR, FILE_NAME)):
        print(f"⚠️ Предупреждение: Не удалось обновить файл из HF ({e}). Используем локальную копию.")
    else:
        print(f"❌ Ошибка при скачивании файла и нет локальной копии: {e}")