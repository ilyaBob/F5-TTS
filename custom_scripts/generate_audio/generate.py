import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

# ============================================================
# ⚙️ КОНФИГУРАЦИЯ (.env)
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
WORKER_ID = os.getenv("WORKER_ID", "Colab_1")
HF_COLLECTION_SLUG = os.getenv("HF_COLLECTION_SLUG")

MANIFEST_FILENAME = os.getenv("MANIFEST_FILENAME", "dataset_manifest.csv")
HF_WAV_DIR = os.getenv("HF_WAV_DIR", "generated_wavs")

F5_DIR = os.getenv("F5_DIR", "/content/F5-TTS")
WORK_DIR = os.path.join(F5_DIR, "manifest_work")
OUTPUT_DIR = os.path.join(F5_DIR, "output_audio")

MODEL_DIR_NAME = os.getenv("MODEL_DIR_NAME")
LOCAL_CKPT_DIR = os.path.join(F5_DIR, "ckpts", MODEL_DIR_NAME)
CKPT_FILE_NAME = os.getenv("FILE_NAME")
VOCAB_FILE_NAME = os.getenv("VOCAB_FILE_NAME")

CKPT_PATH = os.path.join(LOCAL_CKPT_DIR, CKPT_FILE_NAME)
VOCAB_PATH = os.path.join(LOCAL_CKPT_DIR, VOCAB_FILE_NAME)

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
# 🛠️ ГЛОБАЛЬНАЯ ЗАЩИТА ОТ ОБРЕЗАНИЙ (ДЛЯ ВСЕХ ДЛИН ТЕКСТА)
# ============================================================

def clean_text_for_stats(text: str) -> str:
    """Удаляет символы ударений (+) и пунктуацию для чистого подсчета длины."""
    text_no_accents = text.replace("+", "")
    return re.sub(r"[^\w\s]", "", text_no_accents).strip()


def calculate_robust_speed(
        text: str,
        chars_per_sec: float = 11.0,
        short_target_sec: float = 3.5,
        max_safe_speed: float = 0.82,  # Выше 0.82 в F5-TTS лучше не подниматься
        min_speed: float = 0.1,
) -> float:
    """
    Рассчитывает запас времени и безопасную скорость для ТЕКСТА ЛЮБОЙ ДЛИНЫ.
    - До 25 символов: фиксированная цель 3.5 сек (speed ~0.35-0.45).
    - Более 25 символов: базовая длительность + 20% запаса на договаривание концовки.
    """
    cleaned = clean_text_for_stats(text)
    char_len = len(cleaned)

    if char_len == 0:
        return 1.0

    # Естественная базовая длительность на основе длины текста
    base_duration = char_len / chars_per_sec

    if char_len <= 25:
        # Для коротких фраз (включая "Глав+а трин+адцатая")
        target_duration = max(short_target_sec, base_duration * 1.5)
    else:
        # Для средних и длинных фраз: добавляем 20% запаса времени на интонации и концовки
        target_duration = base_duration * 1.20

    calculated_speed = base_duration / target_duration
    speed = float(np.clip(calculated_speed, min_speed, max_safe_speed))

    return round(speed, 4)


def prepare_text_with_tail_padding(text: str) -> str:
    """
    Добавляет безопасную микро-паузу в конец текста, чтобы модель
    гарантированно успевала уйти в тишину и не обрезала последнюю букву.
    """
    text = text.strip()
    # Если текст не заканчивается знаками препинания, добавляем точку
    if not text.endswith((".", "!", "?", "...", "…")):
        text += "."
    return text


# ============================================================
# 🎯 БЛОКИРОВКА И УПРАВЛЕНИЕ РЕПОЗИТОРИЯМИ
# ============================================================

def get_collection_repositories(collection_slug):
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
    lock_file_path = os.path.join(WORK_DIR, LOCK_FILENAME)

    try:
        downloaded_lock = hf_hub_download(
            repo_id=repo_id,
            filename=LOCK_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
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
        pass
    except Exception as e:
        print(f"   ⚠ Ошибка чтения lock-файла: {e}")

    print(f"   ✍️ Запись метки {worker_id} в {repo_id}...")
    with open(lock_file_path, "w", encoding="utf-8") as f:
        f.write(f"IN_PROGRESS:{worker_id}")

    try:
        api.upload_file(
            path_or_fileobj=lock_file_path,
            path_in_repo=LOCK_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Lock repo for {worker_id}",
        )
    except Exception as e:
        print(f"   ❌ Ошибка загрузки lock-файла: {e}")
        return False

    print("   ⏳ Ожидание 10 сек для проверки гонки процессов...")
    time.sleep(10)

    try:
        downloaded_lock = hf_hub_download(
            repo_id=repo_id,
            filename=LOCK_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
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
    lock_file_path = os.path.join(WORK_DIR, LOCK_FILENAME)
    with open(lock_file_path, "w", encoding="utf-8") as f:
        f.write("DONE")
    try:
        api.upload_file(
            path_or_fileobj=lock_file_path,
            path_in_repo=LOCK_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Mark done by {worker_id}",
        )
        print(f"✅ Репозиторий {repo_id} помечен как DONE.")
    except Exception as e:
        print(f"⚠️ Не удалось обновить статус DONE: {e}")


def clear_local_work_dirs():
    for folder in [WORK_DIR, OUTPUT_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)


# ============================================================
# GPU & ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
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

print("🔥 Прогрев модели (Warm-up)...")
try:
    _ = f5tts.infer(
        ref_file=REF_AUDIO,
        ref_text=REF_TEXT,
        gen_text="Прогрев модели.",
        seed=1,
        speed=0.8,
    )
    print("✅ Прогрев завершён успешно.")
except Exception as e:
    print(f"⚠️ Предупреждение при прогреве: {e}")

# ============================================================
# ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ
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

    clear_local_work_dirs()

    MANIFEST_REPO_ID = TARGET_REPO_ID
    OUTPUT_DATASET_REPO = TARGET_REPO_ID

    print(f"\n🚀 НАЧИНАЕМ РАБОТУ С РЕПОЗИТОРИЕМ: {TARGET_REPO_ID}")

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
                        "speaker": r.get("speaker", ""),
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


    # --- ЦИКЛ ГЕНЕРАЦИИ С ДВОЙНОЙ ЗАЩИТОЙ ---
    if pending:
        success = 0
        for index, row in enumerate(pending, start=1):
            filename = row["filename"]
            raw_text = row["text"]
            output_path = os.path.join(OUTPUT_DIR, filename)

            # 1. Подготовка текста с tail-padding
            text_to_gen = prepare_text_with_tail_padding(raw_text)

            # 2. Подсчет надежной скорости
            calculated_speed = calculate_robust_speed(raw_text)
            char_count = len(clean_text_for_stats(raw_text))

            print(
                f"[{index}/{len(pending)}] {filename} | "
                f"Символов: {char_count} | Speed: {calculated_speed}..."
            )

            try:
                # Первая попытка генерации
                wav, sr, _ = f5tts.infer(
                    ref_file=REF_AUDIO,
                    ref_text=REF_TEXT,
                    gen_text=text_to_gen,
                    speed=calculated_speed,
                    seed=1,
                )

                duration_sec = len(wav) / sr

                # 3. АВТО-ПРОВЕРКА (Safety Check):
                # Если сгенерированный звук подозрительно короткий (< 14 символов в секунду),
                # значит модель проглотила/обрезала фразу. Делаем автоматический ретрай.
                expected_min_duration = max(1.5, char_count / 16.0)

                if duration_sec < expected_min_duration:
                    retry_speed = round(calculated_speed * 0.75, 4)
                    print(
                        f"⚠️ [SAFETY] Слишком короткий звук ({duration_sec:.2f}s < {expected_min_duration:.2f}s). "
                        f"Повтор с пониженной скоростью speed={retry_speed}..."
                    )
                    wav, sr, _ = f5tts.infer(
                        ref_file=REF_AUDIO,
                        ref_text=REF_TEXT,
                        gen_text=text_to_gen,
                        speed=retry_speed,
                        seed=1,
                    )

                sf.write(output_path, wav, sr)
                success += 1
                save_generated_manifest()

                if success % UPLOAD_EVERY == 0:
                    print("📤 Промежуточный upload на HF...")
                    api.upload_folder(
                        folder_path=OUTPUT_DIR,
                        path_in_repo=HF_WAV_DIR,
                        repo_id=OUTPUT_DATASET_REPO,
                        repo_type="dataset",
                        allow_patterns="*.wav",
                    )
                    api.upload_file(
                        path_or_fileobj=GENERATED_MANIFEST_PATH,
                        path_in_repo=GENERATED_MANIFEST_FILENAME,
                        repo_id=OUTPUT_DATASET_REPO,
                        repo_type="dataset",
                    )
            except Exception as e:
                print(f"❌ Ошибка {filename}: {e}")

    print("\n📤 Финальная загрузка результатов на HF...")
    save_generated_manifest()

    try:
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            path_in_repo=HF_WAV_DIR,
            repo_id=OUTPUT_DATASET_REPO,
            repo_type="dataset",
            allow_patterns="*.wav",
        )
        api.upload_file(
            path_or_fileobj=GENERATED_MANIFEST_PATH,
            path_in_repo=GENERATED_MANIFEST_FILENAME,
            repo_id=OUTPUT_DATASET_REPO,
            repo_type="dataset",
        )
        print("✅ Все данные успешно загружены!")
        mark_repository_done(TARGET_REPO_ID, WORKER_ID)

    except Exception as e:
        print(f"❌ Ошибка финальной загрузки: {e}")

    print("=" * 70)
    print(f"🎉 РАБОТА ВОРКЕРА {WORKER_ID} НАД {TARGET_REPO_ID} ЗАВЕРШЕНА!")
    print("=" * 70)