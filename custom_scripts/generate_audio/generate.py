import csv
import os
import shutil
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)

# ============================================================
# ⚙️ КОНФИГУРАЦИЯ (.env)
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
WORKER_ID = os.getenv("WORKER_ID", "Colab_1")
HF_COLLECTION_SLUG = os.getenv("HF_COLLECTION_SLUG")

MANIFEST_FILENAME = os.getenv("MANIFEST_FILENAME", "dataset_manifest.csv")
HF_WAV_DIR = os.getenv("HF_WAV_DIR", "generated_wavs")

# Пути к директориям
F5_DIR = os.getenv("F5_DIR", "/content/F5-TTS")
WORK_DIR = os.path.join(F5_DIR, "manifest_work")
OUTPUT_DIR = os.path.join(F5_DIR, "output_audio")

# Чекпоинты и модель
MODEL_DIR_NAME = os.getenv("MODEL_DIR_NAME")
LOCAL_CKPT_DIR = os.path.join(F5_DIR, "ckpts", MODEL_DIR_NAME)
CKPT_FILE_NAME = os.getenv("FILE_NAME")
VOCAB_FILE_NAME = os.getenv("VOCAB_FILE_NAME")

CKPT_PATH = os.path.join(LOCAL_CKPT_DIR, CKPT_FILE_NAME)
VOCAB_PATH = os.path.join(LOCAL_CKPT_DIR, VOCAB_FILE_NAME)

# Референсы
REF_AUDIO = os.path.join(
    os.getenv("DATASET_DIR", os.path.join(F5_DIR, "dataset")),
    os.getenv("WAV_FILENAME", ""),
)
REF_TEXT = os.getenv("REF_TEXT")

UPLOAD_EVERY = int(os.getenv("UPLOAD_EVERY", "50"))
GENERATED_MANIFEST_FILENAME = os.getenv("GENERATED_MANIFEST_FILENAME", "generated_manifest.csv")
LOCK_FILENAME = "LOCK_STATUS.txt"

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not HF_TOKEN:
    raise RuntimeError("\n❌ HF_TOKEN не найден в .env файле.")

api = HfApi(token=HF_TOKEN)


# ============================================================
# 🎯 МЕХАНИЗМ БЛОКИРОВКИ И ПОИСКА СВОБОДНОГО РЕПОЗИТОРИЯ
# ============================================================

def get_collection_repositories(collection_slug):
    """Получает список всех dataset-репозиториев из коллекции."""
    print(f"\n📚 Запрос коллекции: {collection_slug}...")
    try:
        collection = api.get_collection(collection_slug)
        repos = [item.item_id for item in collection.items if item.item_type == "dataset"]
        print(f"✅ Найдено репозиториев в коллекции: {len(repos)}")
        return repos
    except Exception as e:
        print(f"❌ Ошибка получения коллекции: {e}")
        return []


def try_lock_repository(repo_id, worker_id):
    """
    Пытается заблокировать репозиторий под текущий WORKER_ID.
    Возвращает True, если репозиторий успешно заблокирован за нами.
    """
    lock_file_path = os.path.join(WORK_DIR, LOCK_FILENAME)

    # 1. Проверяем текущее состояние LOCK_STATUS.txt на HF
    try:
        downloaded_lock = hf_hub_download(
            repo_id=repo_id,
            filename=LOCK_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True
        )
        with open(downloaded_lock, "r", encoding="utf-8") as f:
            current_status = f.read().strip()

        if current_status.startswith("IN_PROGRESS:") and not current_status.endswith(worker_id):
            print(f"   ⏩ Занят другим воркером: {current_status}")
            return False
        elif current_status == "DONE":
            print("   ⏩ Уже завершён (DONE).")
            return False

    except EntryNotFoundError:
        # Файла блокировки ещё нет — репозиторий свободен
        pass
    except Exception as e:
        print(f"   ⚠️ Ошибка чтения lock-файла: {e}")

    # 2. Пишем свою метку и загружаем на HF
    print(f"   ✍️ Запись метки {worker_id} в {repo_id}...")
    with open(lock_file_path, "w", encoding="utf-8") as f:
        f.write(f"IN_PROGRESS:{worker_id}")

    try:
        api.upload_file(
            path_or_fileobj=lock_file_path,
            path_in_repo=LOCK_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Lock repo for {worker_id}"
        )
    except Exception as e:
        print(f"   ❌ Ошибка загрузки lock-файла: {e}")
        return False

    # 3. Пауза 10 секунд (гонка процессов)
    print("   ⏳ Ожидание 10 сек для проверки гонки процессов...")
    time.sleep(10)

    # 4. Повторно скачиваем и проверяем, кто остался записан
    try:
        downloaded_lock = hf_hub_download(
            repo_id=repo_id,
            filename=LOCK_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True
        )
        with open(downloaded_lock, "r", encoding="utf-8") as f:
            final_status = f.read().strip()

        if final_status == f"IN_PROGRESS:{worker_id}":
            print(f"   🎉 УСПЕШНО ЗАБЛОКИРОВАНО за {worker_id}!")
            return True
        else:
            print(f"   ⚠️ Блокировка перехвачена: {final_status}")
            return False
    except Exception as e:
        print(f"   ❌ Ошибка проверки блокировки: {e}")
        return False


def mark_repository_done(repo_id, worker_id):
    """Помечает репозиторий как полностью обработанный."""
    lock_file_path = os.path.join(WORK_DIR, LOCK_FILENAME)
    with open(lock_file_path, "w", encoding="utf-8") as f:
        f.write("DONE")
    try:
        api.upload_file(
            path_or_fileobj=lock_file_path,
            path_in_repo=LOCK_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Mark done by {worker_id}"
        )
        print(f"✅ Репозиторий {repo_id} помечен как DONE.")
    except Exception as e:
        print(f"⚠️ Не удалось обновить статус DONE: {e}")


def clear_local_work_dirs():
    """Очищает локальные рабочие папки перед обработкой нового репозитория."""
    for folder in [WORK_DIR, OUTPUT_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)


# ============================================================
# GPU & ИНИЦИАЛИЗАЦИЯ МОДЕЛИ (Один раз на весь запуск)
# ============================================================

print("=" * 70)
print(f"🤖 ВОРКЕР: {WORKER_ID}")
print("=" * 70)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("⚠️ Используется CPU.")

if not os.path.exists(CKPT_PATH) or not os.path.exists(VOCAB_PATH) or not os.path.exists(REF_AUDIO):
    raise FileNotFoundError("❌ Ошибка: Не найдены чекпоинты или референсный WAV!")

SRC_DIR = os.path.join(F5_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from f5_tts.api import F5TTS

print("🧠 Загрузка модели F5-TTS в память...")
f5tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=CKPT_PATH,
    vocab_file=VOCAB_PATH,
    device=DEVICE,
)
print("✅ Модель готова к работе!")

# ============================================================
# ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ ВСЕХ РЕПОЗИТОРИЕВ
# ============================================================

while True:
    repos_to_process = get_collection_repositories(HF_COLLECTION_SLUG)

    TARGET_REPO_ID = None

    for repo in repos_to_process:
        print(f"\n🔍 Проверка репозитория: {repo}")
        if try_lock_repository(repo, WORKER_ID):
            TARGET_REPO_ID = repo
            break

    if not TARGET_REPO_ID:
        print("\n" + "=" * 70)
        print("🎉 ВСЕ РЕПОЗИТОРИИ В КОЛЛЕКЦИИ УЖЕ ОБРАБОТАНЫ ИЛИ ЗАНЯТЫ!")
        print("=" * 70)
        break

    # Очищаем локальные папки от предыдущего репозитория
    clear_local_work_dirs()

    MANIFEST_REPO_ID = TARGET_REPO_ID
    OUTPUT_DATASET_REPO = TARGET_REPO_ID

    print(f"\n🚀 НАЧИНАЕМ РАБОТУ С РЕПОЗИТОРИЕМ: {TARGET_REPO_ID}")

    # --- СКАЧИВАНИЕ MANIFEST & СУЩЕСТВУЮЩИХ WAV ---
    local_manifest_path = hf_hub_download(
        repo_id=MANIFEST_REPO_ID,
        filename=MANIFEST_FILENAME,
        repo_type="dataset",
        local_dir=WORK_DIR,
        token=HF_TOKEN,
    )

    try:
        snapshot_dir = snapshot_download(
            repo_id=OUTPUT_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=f"{HF_WAV_DIR}/*.wav",
            token=HF_TOKEN,
        )
        remote_wavs_dir = os.path.join(snapshot_dir, HF_WAV_DIR)
        if os.path.exists(remote_wavs_dir):
            for source_path in Path(remote_wavs_dir).glob("*.wav"):
                destination_path = os.path.join(OUTPUT_DIR, source_path.name)
                if not os.path.exists(destination_path):
                    shutil.copy2(source_path, destination_path)
    except Exception as e:
        print("ℹ️ WAV файлы ранее не загружались или ошибка:", e)

    # --- ЧТЕНИЕ MANIFEST И ИСТОРИИ ---
    rows = []
    with open(local_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filename") and row.get("text"):
                rows.append({
                    "filename": row["filename"].strip(),
                    "speaker": row.get("speaker", "").strip(),
                    "text": row["text"].strip(),
                })

    GENERATED_MANIFEST_PATH = os.path.join(OUTPUT_DIR, GENERATED_MANIFEST_FILENAME)
    previous_generated = {}

    try:
        remote_manifest = hf_hub_download(
            repo_id=OUTPUT_DATASET_REPO,
            filename=GENERATED_MANIFEST_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(remote_manifest, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("filename"):
                    previous_generated[r["filename"]] = {
                        "text": r.get("text", ""),
                        "speaker": r.get("speaker", "")
                    }
    except Exception:
        pass

    pending = []
    for row in rows:
        fn = row["filename"]
        out_p = os.path.join(OUTPUT_DIR, fn)
        if not os.path.exists(out_p):
            pending.append(row)
        else:
            prev = previous_generated.get(fn)
            if prev and prev.get("text", "").strip() != row["text"].strip():
                pending.append(row)

    print(f"📊 К генерации: {len(pending)} из {len(rows)}")


    def save_generated_manifest():
        with open(GENERATED_MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "speaker", "text"])
            writer.writeheader()
            for row in rows:
                if os.path.exists(os.path.join(OUTPUT_DIR, row["filename"])):
                    writer.writerow(row)


    # --- ЦИКЛ ГЕНЕРАЦИИ ДЛЯ ТЕКУЩЕГО РЕПОЗИТОРИЯ ---
    if pending:
        success = 0
        for index, row in enumerate(pending, start=1):
            filename = row["filename"]
            output_path = os.path.join(OUTPUT_DIR, filename)

            print(f"[{index}/{len(pending)}] {filename}...")
            try:
                wav, sr, _ = f5tts.infer(
                    ref_file=REF_AUDIO,
                    ref_text=REF_TEXT,
                    gen_text=row["text"],
                    seed=None,
                )
                sf.write(output_path, wav, sr)
                success += 1
                save_generated_manifest()

                if success % UPLOAD_EVERY == 0:
                    print("📤 Промежуточный upload на HF...")
                    api.upload_folder(
                        folder_path=OUTPUT_DIR, path_in_repo=HF_WAV_DIR,
                        repo_id=OUTPUT_DATASET_REPO, repo_type="dataset", allow_patterns="*.wav"
                    )
                    api.upload_file(
                        path_or_fileobj=GENERATED_MANIFEST_PATH,
                        path_in_repo=GENERATED_MANIFEST_FILENAME,
                        repo_id=OUTPUT_DATASET_REPO, repo_type="dataset"
                    )
            except Exception as e:
                print(f"❌ Ошибка {filename}: {e}")

    # --- ФИНАЛЬНАЯ СИНХРОНИЗАЦИЯ И МЕТКА DONE ---
    print("\n📤 Финальная загрузка результатов на HF...")
    save_generated_manifest()

    try:
        api.upload_folder(
            folder_path=OUTPUT_DIR, path_in_repo=HF_WAV_DIR,
            repo_id=OUTPUT_DATASET_REPO, repo_type="dataset", allow_patterns="*.wav"
        )
        api.upload_file(
            path_or_fileobj=GENERATED_MANIFEST_PATH,
            path_in_repo=GENERATED_MANIFEST_FILENAME,
            repo_id=OUTPUT_DATASET_REPO, repo_type="dataset"
        )
        print("✅ Все данные успешно загружены!")

        # Помечаем главу как выполнившую работу
        mark_repository_done(TARGET_REPO_ID, WORKER_ID)

    except Exception as e:
        print(f"❌ Ошибка финальной загрузки: {e}")

    print("=" * 70)
    print(f"🎉 РАБОТА ВОРКЕРА {WORKER_ID} НАД {TARGET_REPO_ID} ЗАВЕРШЕНА!")
    print("=" * 70)
    # Цикл переходит к следующей итерации (while True) и ищет следующий свободный репозиторий
