import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

# Попытка импорта dotenv для автоматической загрузки .env файла
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ==========================================
# ⚙️ КОНФИГУРАЦИЯ ДАТАСЕТА (Чтение из .env / Переменных окружения)
# ==========================================
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
WAV_FILENAME = os.getenv("WAV_FILENAME")
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_DIR = os.getenv("DATASET_DIR")
# ==========================================

# Определяем точный целевой путь для сохранения
target_file_path = os.path.join(DATASET_DIR, WAV_FILENAME)
os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

token = (
    HF_TOKEN if HF_TOKEN and not HF_TOKEN.startswith("hf_xxx") else None
)

if not os.path.exists(target_file_path):
    print(
        f"🔍 Файл '{WAV_FILENAME}' не найден локально. Скачиваем из HF Dataset..."
    )
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=WAV_FILENAME,
            repo_type="dataset",  # Важно: указываем, что это датасет
            local_dir=DATASET_DIR,
            token=token,
        )
        print(f"✅ Файл успешно скачан и сохранен в: {downloaded_path}")
    except (RepositoryNotFoundError, GatedRepoError) as e:
        print(
            f"❌ ОШИБКА: Датасет '{HF_DATASET_REPO}' не найден или нет доступа.\n{e}"
        )
    except EntryNotFoundError:
        print(
            f"❌ ОШИБКА: Файл '{WAV_FILENAME}' не найден в датасете '{HF_DATASET_REPO}'."
        )
    except Exception as e:
        print(f"❌ Ошибка при скачивании аудиофайла: {e}")
else:
    print(f"ℹ️ Файл уже существует: {target_file_path}")