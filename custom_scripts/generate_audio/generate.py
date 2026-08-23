import csv
import os
import sys
import time
from pathlib import Path

# Попытка загрузить переменные из .env
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

# Репозитории Hugging Face
MANIFEST_REPO_ID = os.getenv(
    "MANIFEST_REPO_ID", "ilya-75/erof-vadim-nebesniy-tron-13"
)
MANIFEST_FILENAME = os.getenv("MANIFEST_FILENAME", "dataset_manifest.csv")
OUTPUT_DATASET_REPO = os.getenv(
    "OUTPUT_DATASET_REPO", "ilya-75/erof-vadim-nebesniy-tron-13"
)
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

# Референсное аудио и текст
REF_AUDIO = os.path.join(
    os.getenv("DATASET_DIR", os.path.join(F5_DIR, "dataset")),
    os.getenv("WAV_FILENAME"),
)
REF_TEXT = os.getenv(
    "REF_TEXT"
)

# Настройки процесса
UPLOAD_EVERY = int(os.getenv("UPLOAD_EVERY", "50"))
GENERATED_MANIFEST_FILENAME = os.getenv(
    "GENERATED_MANIFEST_FILENAME", "generated_manifest.csv"
)
GENERATED_MANIFEST_PATH = os.path.join(
    OUTPUT_DIR, GENERATED_MANIFEST_FILENAME
)

# ============================================================
# ПОДГОТОВКА
# ============================================================

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# TOKEN
# ============================================================

if not HF_TOKEN:
    raise RuntimeError(
        "\n❌ HF_TOKEN не найден.\nУбедитесь, что он задан в .env файле."
    )

token = (
    HF_TOKEN if HF_TOKEN and not HF_TOKEN.startswith("hf_xxx") else None
)

# ============================================================
# START
# ============================================================

print("=" * 70)
print("F5-TTS SMART BULK GENERATION")
print("=" * 70)

print()
print(f"📂 F5-TTS:     {F5_DIR}")
print(f"📂 Output:     {OUTPUT_DIR}")
print(f"🤗 HF Dataset: {OUTPUT_DATASET_REPO}")
print()

# ============================================================
# GPU
# ============================================================

print("=" * 70)
print("🔥 ПРОВЕРКА GPU")
print("=" * 70)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    total_memory = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )
    print(f"💾 VRAM: {total_memory:.1f} GB")
    torch.cuda.empty_cache()
else:
    DEVICE = "cpu"
    print("⚠️ CUDA не обнаружена!")
    print("⚠️ Используется CPU.")

# ============================================================
# 1. ПРОВЕРКА ФАЙЛОВ F5-TTS
# ============================================================

print()
print("=" * 70)
print("🔍 ПРОВЕРКА ФАЙЛОВ")
print("=" * 70)

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"\n❌ Не найден checkpoint:\n{CKPT_PATH}")

if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"\n❌ Не найден vocab:\n{VOCAB_PATH}")

if not os.path.exists(REF_AUDIO):
    raise FileNotFoundError(f"\n❌ Не найден reference audio:\n{REF_AUDIO}")

print(f"✅ Checkpoint: {CKPT_PATH}")
print(f"✅ Vocab:      {VOCAB_PATH}")
print(f"✅ Reference:  {REF_AUDIO}")

# ============================================================
# 2. HUGGING FACE API
# ============================================================

api = HfApi(token=token)

# ============================================================
# 3. СКАЧИВАЕМ DATASET MANIFEST
# ============================================================

print()
print("=" * 70)
print("📥 СКАЧИВАНИЕ DATASET MANIFEST")
print("=" * 70)

try:
    local_manifest_path = hf_hub_download(
        repo_id=MANIFEST_REPO_ID,
        filename=MANIFEST_FILENAME,
        repo_type="dataset",
        local_dir=WORK_DIR,
        token=token,
    )
    print(f"✅ Manifest скачан:\n{local_manifest_path}")

except (RepositoryNotFoundError, GatedRepoError) as e:
    print(f"❌ Ошибка доступа к HF:\n{e}")
    raise

except EntryNotFoundError:
    print(f"❌ Файл {MANIFEST_FILENAME} не найден в {MANIFEST_REPO_ID}")
    raise

# ============================================================
# 4. СКАЧИВАЕМ ВСЕ WAV ОДНИМ ВЫЗОВОМ
# ============================================================

print()
print("=" * 70)
print("📥 СКАЧИВАНИЕ ГОТОВЫХ WAV")
print("=" * 70)

print("☁️ Скачиваем generated_wavs/*.wav одним вызовом...")

try:
    snapshot_dir = snapshot_download(
        repo_id=OUTPUT_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=f"{HF_WAV_DIR}/*.wav",
        token=token,
    )

    remote_wavs_dir = os.path.join(snapshot_dir, HF_WAV_DIR)

    if not os.path.exists(remote_wavs_dir):
        print("ℹ️ Папка generated_wavs на HF пока отсутствует.")
    else:
        wav_files = list(Path(remote_wavs_dir).glob("*.wav"))
        print(f"☁️ Найдено WAV: {len(wav_files)}")

        import shutil

        downloaded = 0
        skipped = 0

        for source_path in wav_files:
            filename = source_path.name
            destination_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(destination_path):
                skipped += 1
                continue

            shutil.copy2(source_path, destination_path)
            downloaded += 1

        print()
        print(f"📥 Скопировано новых WAV: {downloaded}")
        print(f"⏩ Уже было локально: {skipped}")

except Exception as e:
    print("\n⚠️ Ошибка скачивания WAV:\n", e)
    print("\nℹ️ Продолжаем с локальными файлами.")

# ============================================================
# 5. ИМПОРТ F5-TTS
# ============================================================

print()
print("=" * 70)
print("🔥 ЗАГРУЗКА F5-TTS")
print("=" * 70)

SRC_DIR = os.path.join(F5_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from f5_tts.api import F5TTS

# ============================================================
# 6. ЗАГРУЖАЕМ МОДЕЛЬ ОДИН РАЗ
# ============================================================

start_model = time.time()

f5tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=CKPT_PATH,
    vocab_file=VOCAB_PATH,
    device=DEVICE,
)

model_time = time.time() - start_model
print(f"✅ Модель загружена за {model_time:.1f} сек.")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU allocated: {allocated:.2f} GB")
    print(f"GPU reserved:  {reserved:.2f} GB")

# ============================================================
# 7. ЧИТАЕМ DATASET MANIFEST
# ============================================================

print()
print("=" * 70)
print("📚 ЧТЕНИЕ MANIFEST")
print("=" * 70)

rows = []

with open(local_manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row.get("filename")
        speaker = row.get("speaker")
        text = row.get("text")

        if not filename or not text:
            continue

        rows.append(
            {
                "filename": filename.strip(),
                "speaker": speaker.strip() if speaker else "",
                "text": text.strip(),
            }
        )

print(f"📚 Всего записей: {len(rows)}")

# ============================================================
# 8. СКАЧИВАЕМ GENERATED MANIFEST С HF
# ============================================================

print()
print("=" * 70)
print("📋 ПРОВЕРКА ИСТОРИИ ГЕНЕРАЦИИ")
print("=" * 70)

previous_generated = {}

if os.path.exists(GENERATED_MANIFEST_PATH):
    print("📋 Найден локальный generated_manifest.csv")
    try:
        with open(GENERATED_MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename")
                if not filename:
                    continue
                previous_generated[filename] = {
                    "text": row.get("text", ""),
                    "speaker": row.get("speaker", ""),
                }
        print(f"✅ Загружено записей: {len(previous_generated)}")
    except Exception as e:
        print(f"⚠️ Ошибка чтения manifest: {e}")
        previous_generated = {}

if not previous_generated:
    print("📥 Проверяем generated_manifest.csv на HF...")
    try:
        remote_manifest = hf_hub_download(
            repo_id=OUTPUT_DATASET_REPO,
            filename=GENERATED_MANIFEST_FILENAME,
            repo_type="dataset",
            token=token,
        )

        with open(remote_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename")
                if not filename:
                    continue
                previous_generated[filename] = {
                    "text": row.get("text", ""),
                    "speaker": row.get("speaker", ""),
                }

        with open(
            GENERATED_MANIFEST_PATH, "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "speaker", "text"]
            )
            writer.writeheader()
            for filename, data in previous_generated.items():
                writer.writerow(
                    {
                        "filename": filename,
                        "speaker": data["speaker"],
                        "text": data["text"],
                    }
                )

        print(f"✅ История генерации: {len(previous_generated)}")

    except EntryNotFoundError:
        print("ℹ️ generated_manifest.csv на HF ещё нет.")
    except Exception as e:
        print(f"⚠️ Не удалось скачать generated_manifest.csv: {e}")

# ============================================================
# 9. ОПРЕДЕЛЯЕМ НЕДОСТАЮЩИЕ / ИЗМЕНЁННЫЕ
# ============================================================

print()
print("=" * 70)
print("🔍 ПРОВЕРКА WAV")
print("=" * 70)

pending = []
already_ready = 0
missing = 0
changed = 0
unknown = 0

for row in rows:
    filename = row["filename"]
    current_text = row["text"]
    output_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(output_path):
        print(f"🎙️ ОТСУТСТВУЕТ: {filename}")
        pending.append(row)
        missing += 1
        continue

    previous = previous_generated.get(filename)

    if previous is None:
        print(f"⚠️ НЕТ ИСТОРИИ: {filename}\n   ⏩ Оставляем существующий WAV")
        already_ready += 1
        unknown += 1
        continue

    old_text = previous.get("text", "").strip()
    new_text = current_text.strip()

    if old_text != new_text:
        print(
            f"\n🔄 ТЕКСТ ИЗМЕНЁН: {filename}\n   Было: {old_text[:150]}\n   Стало: {new_text[:150]}"
        )
        pending.append(row)
        changed += 1
        continue

    already_ready += 1

# ============================================================
# СТАТИСТИКА
# ============================================================

print()
print("=" * 70)
print("📊 РЕЗУЛЬТАТ ПРОВЕРКИ")
print("=" * 70)
print(f"Всего записей:       {len(rows)}")
print(f"Уже готовы:          {already_ready}")
print(f"Отсутствуют:         {missing}")
print(f"Текст изменён:       {changed}")
print(f"Без истории:         {unknown}")
print(f"Нужно генерировать:  {len(pending)}")

if not pending:
    print(f"\n🎉 Все WAV уже готовы!\n📂 {OUTPUT_DIR}")
    sys.exit(0)


# ============================================================
# 10. ФУНКЦИЯ СОХРАНЕНИЯ MANIFEST
# ============================================================


def save_generated_manifest():
    with open(
        GENERATED_MANIFEST_PATH, "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "speaker", "text"]
        )
        writer.writeheader()
        for row in rows:
            filename = row["filename"]
            output_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(output_path):
                writer.writerow(
                    {
                        "filename": filename,
                        "speaker": row["speaker"],
                        "text": row["text"],
                    }
                )


# ============================================================
# 11. ГЕНЕРАЦИЯ
# ============================================================

print()
print("=" * 70)
print("🎙️ НАЧИНАЕМ ГЕНЕРАЦИЮ")
print("=" * 70)
print(f"Файлов к генерации: {len(pending)}")

total_start = time.time()
success = 0
errors = 0

for index, row in enumerate(pending, start=1):
    filename = row["filename"]
    gen_text = row["text"]
    speaker = row["speaker"]
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"\n{'-' * 70}")
    print(f"[{index}/{len(pending)}] {filename}")
    print(f"Speaker: {speaker}")
    print(
        f"Text: {gen_text[:200]}"
        + ("..." if len(gen_text) > 200 else "")
    )

    start = time.time()

    try:
        wav, sr, _ = f5tts.infer(
            ref_file=REF_AUDIO,
            ref_text=REF_TEXT,
            gen_text=gen_text,
            seed=None,
        )

        sf.write(output_path, wav, sr)
        elapsed = time.time() - start
        success += 1
        print(f"✅ Готово за {elapsed:.2f} сек.")

        previous_generated[filename] = {"text": gen_text, "speaker": speaker}
        save_generated_manifest()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"GPU: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved"
            )

        if success % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"🧹 GPU cache очищен после {success} файлов")

        if success % UPLOAD_EVERY == 0:
            print("\n📤 Промежуточная синхронизация с HF...")
            try:
                api.upload_folder(
                    folder_path=OUTPUT_DIR,
                    path_in_repo=HF_WAV_DIR,
                    repo_id=OUTPUT_DATASET_REPO,
                    repo_type="dataset",
                    allow_patterns="*.wav",
                    commit_message=f"Auto upload: {success} generated files",
                )
                api.upload_file(
                    path_or_fileobj=GENERATED_MANIFEST_PATH,
                    path_in_repo=GENERATED_MANIFEST_FILENAME,
                    repo_id=OUTPUT_DATASET_REPO,
                    repo_type="dataset",
                    commit_message="Update generated manifest",
                )
                print("✅ HF синхронизация завершена")
            except Exception as e:
                print(f"⚠️ HF upload ошибка: {e}")

    except Exception as e:
        errors += 1
        print(f"\n❌ Ошибка генерации {filename}:\n{e}\n➡️ Продолжаем...")

# ============================================================
# 12. ФИНАЛЬНЫЙ MANIFEST И СИНХРОНИЗАЦИЯ
# ============================================================

print("\n" + "=" * 70 + "\n📝 СОХРАНЕНИЕ И СИНХРОНИЗАЦИЯ\n" + "=" * 70)
save_generated_manifest()

try:
    print("📤 Загружаем WAV...")
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        path_in_repo=HF_WAV_DIR,
        repo_id=OUTPUT_DATASET_REPO,
        repo_type="dataset",
        allow_patterns="*.wav",
        commit_message="Update generated TTS audio",
    )
    print("✅ WAV загружены")

    print("📤 Загружаем generated_manifest.csv...")
    api.upload_file(
        path_or_fileobj=GENERATED_MANIFEST_PATH,
        path_in_repo=GENERATED_MANIFEST_FILENAME,
        repo_id=OUTPUT_DATASET_REPO,
        repo_type="dataset",
        commit_message="Update generated TTS manifest",
    )
    print("✅ generated_manifest.csv загружен")

except Exception as e:
    print(f"❌ Ошибка финальной загрузки: {e}")

# ============================================================
# 14. СТАТИСТИКА
# ============================================================

total_time = time.time() - total_start
print("\n" + "=" * 70 + "\n🎉 ГОТОВО\n" + "=" * 70)
print(f"Успешно:       {success}")
print(f"Ошибок:        {errors}")
print(f"Время:         {total_time / 60:.1f} минут")
if success > 0:
    print(f"Среднее:       {total_time / success:.2f} сек/файл")
print(f"Результаты:    {OUTPUT_DIR}")
print(f"HF WAV:        {OUTPUT_DATASET_REPO}/{HF_WAV_DIR}")
print(
    f"HF Manifest:   {OUTPUT_DATASET_REPO}/{GENERATED_MANIFEST_FILENAME}"
)
print("=" * 70)