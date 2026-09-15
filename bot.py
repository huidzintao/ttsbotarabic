#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот озвучки арабских диалогов через Gemini TTS.

Что умеет:
1. Принимает ТЕКСТ диалога на арабском -> определяет персонажей и их пол/возраст -> озвучивает в MP3.
2. Принимает ФОТО/скрин диалога -> распознаёт текст (OCR через Gemini) -> то же самое.
3. Перед генерацией показывает кнопки: пользователь одним нажатием меняет пол/возраст персонажа.
"""

import io
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, MessageHandler, filters)
from telegram.error import Conflict
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tts-bot")


# ----------------------- НАСТРОЙКИ (из переменных окружения) -----------------------
def _req(name: str) -> str:
    """Читает переменную окружения; при отсутствии выводит понятную диагностику."""
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            "\nОШИБКА: переменная окружения '" + name + "' не найдена.\n"
            "Видимые имена переменных в контейнере:\n  "
            + str(sorted(os.environ.keys())) +
            "\nПроверьте: Render -> сервис -> Environment -> точное имя (латиница!)"
            " и значение -> Save, rebuild, and deploy.\n")
    return v


TG_TOKEN = _req("TG_TOKEN")                             # токен от @BotFather
GEMINI_API_KEY = _req("GEMINI_API_KEY")                 # ключ из aistudio.google.com
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemini-2.5-flash")
SERVICE_URL = os.environ.get("SERVICE_URL", "").rstrip("/")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "wh" + TG_TOKEN.split(":", 1)[0])
PORT = int(os.environ.get("PORT", "8080"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "4000"))
# DROP_PENDING=1 — при старте выбросить накопившуюся очередь сообщений.
# По умолчанию 0: не теряем сообщения, отправленные пока сервис спал.
DROP_PENDING = os.environ.get("DROP_PENDING", "0") == "1"

client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------------- ГОЛОСА (30 пресетов Gemini TTS) -----------------------
ROLE_VOICE = {
    "m-mid":   ("Charon",     "Speak as a calm 50-year-old Arabic male teacher, warm and informative"),
    "m-young": ("Puck",       "Speak as a young Arabic man, 25 years old, upbeat and friendly"),
    "m-teen":  ("Leda",       "Speak as a 12-year-old Arab schoolboy: light male voice, slightly higher than adult, bright, clean diction, no rasp"),
    "m-old":   ("Rasalgethi", "Speak as an elderly Arabic sheikh, 70 years old, slow and grave"),
    "f-mid":   ("Sulafat",    "Speak as a calm Arabic mother, 40 years old, warm gentle female voice"),
    "f-young": ("Aoede",      "Speak as a young Arabic woman, 25 years old, lively and clear"),
    "f-teen":  ("Leda",       "Speak as a 12-year-old Arab schoolgirl: light bright female voice, youthful"),
    "f-old":   ("Gacrux",     "Speak as an elderly Arabic woman, 70 years old, slow and mature"),
}

TAG_LABEL = {
    "m-mid":   "👨 Мужчина",
    "m-young": "🧑 Парень",
    "m-teen":  "👦 Мальчик",
    "m-old":   "👴 Пожилой мужчина",
    "f-mid":   "👩 Женщина",
    "f-young": "👱‍♀️ Девушка",
    "f-teen":  "👧 Девочка",
    "f-old":   "👵 Пожилая женщина",
}

CYCLE = ["m-mid", "m-young", "m-teen", "m-old",
         "f-mid", "f-young", "f-teen", "f-old"]

# chat_id -> {"utterances": [...], "order": [спикеры], "tags": {спикер: tag}}
PENDING = {}

# ----------------------- ПРОМПТ ДЛЯ ОПРЕДЕЛЕНИЯ РОЛЕЙ -----------------------
TAGGER_RULES = """Ты — лингвист-арабист. Тебе дают арабский учебный диалог.
Разбей его на реплики и верни ТОЛЬКО JSON-массив (без пояснений, без markdown):
[{"speaker": "имя/роль", "sex": "m|f", "age_band": "teen|young|mid|old", "text": "текст реплики дословно"}]

Правила определения пола и возраста:
1) Явные префиксы важнее всего: الشيخ/المعلم/الأستاذ -> m-mid; الأستاذة/المعلمة -> f-mid;
   الأم -> f; الأب/العم/الخال -> m; الطفل/الولد/الصبي -> m-teen; الطفلة/البنت -> f-teen;
   السائل/الطالب -> m-young (если нет других признаков).
2) Женские имена: فاطمة, مريم, هند, سميرة, ليلى, نور, ريم, عائشة, زينب -> f.
3) Мужские имена: محمد, أحمد, يوسف, علي, خالد, عمر, حسن, إبراهيم -> m.
4) Глагольные окончания говорящего: فعلتْ/ذهبتْ/كتبتْ -> f; فعلَ/ذهبَ/كتبَ -> m.
   Обращения: يا أُمِّي -> собеседник f; يا أبي -> собеседник m; يا بُنَيَّ -> собеседник m-teen.
5) Дети: упоминания مدرسة/واجب/تلميذ + детский контекст -> age_band=teen.
6) Если пол определить нельзя — ставь sex по глаголу, age_band=mid.
Сохраняй арабский текст реплик ДОСЛОВНО, со всей диакритикой. Ничего не переводи."""


def parse_json_array(raw: str):
    m = re.search(r"\[.*\]", raw, flags=re.S)
    if not m:
        raise ValueError("модель не вернула JSON-массив")
    return json.loads(m.group(0))


def tag_text(text: str):
    r = client.models.generate_content(
        model=TEXT_MODEL,
        contents=TAGGER_RULES + "\n\nТЕКСТ ДИАЛОГА:\n" + text)
    return parse_json_array(r.text)


def tag_photo(img_bytes: bytes, mime: str):
    prompt = (TAGGER_RULES +
              "\n\nСначала распознай арабский текст диалога с изображения дословно "
              "(сохрани диакритику и порядок реплик), затем верни JSON-массив.")
    r = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), prompt])
    return parse_json_array(r.text)


# ----------------------- ИНТЕРФЕЙС -----------------------

def build_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    st = PENDING[chat_id]
    rows = []
    for i, sp in enumerate(st["order"]):
        rows.append([InlineKeyboardButton(
            f"{sp}: {TAG_LABEL[st['tags'][sp]]}", callback_data=f"tgl:{i}")])
    rows.append([InlineKeyboardButton("✅ Озвучить", callback_data="gen")])
    return InlineKeyboardMarkup(rows)


def summary_text(chat_id: int) -> str:
    st = PENDING[chat_id]
    lines = [f"• {sp} → {TAG_LABEL[st['tags'][sp]]}" for sp in st["order"]]
    return ("Персонажи диалога:\n" + "\n".join(lines) +
            "\n\nНажми на персонажа, чтобы поменять голос, или сразу «✅ Озвучить».")


async def show_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Безопасный индикатор «печатает» (не роняет бота при сбое)."""
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass


async def present(chat_id: int, utts, context: ContextTypes.DEFAULT_TYPE):
    clean = []
    for u in utts:
        sp = str(u.get("speaker", "")).strip() or "الصوت"
        tag = f"{u.get('sex', 'm')}-{u.get('age_band', 'mid')}"
        if tag not in ROLE_VOICE:
            tag = "m-mid"
        txt = str(u.get("text", "")).strip()
        if txt:
            clean.append({"speaker": sp, "tag": tag, "text": txt})
    if not clean:
        await context.bot.send_message(chat_id, "Не нашёл реплик. Пришли текст диалога.")
        return
    order, tags = [], {}
    for c in clean:
        if c["speaker"] not in order:
            order.append(c["speaker"])
            tags[c["speaker"]] = c["tag"]
    PENDING[chat_id] = {"utterances": clean, "order": order, "tags": tags}
    await context.bot.send_message(chat_id, summary_text(chat_id),
                                   reply_markup=build_keyboard(chat_id))


# ----------------------- ОБРАБОТЧИКИ -----------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "السلام عليكم! 👋\n\n"
        "Я озвучиваю арабские диалоги голосами Gemini TTS.\n\n"
        "Что прислать:\n"
        "1️⃣ Текст диалога на арабском — сам определю, кто говорит (шейх/мать/ребёнок), "
        "и предложу голоса.\n"
        "2️⃣ Фото/скрин страницы учебника — распознаю текст и сделаю то же самое.\n\n"
        "Перед генерацией покажу кнопки: пол/возраст любого персонажа можно поменять "
        "одним нажатием. Потом пришлю готовый MP3 🎧"
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return
    if len(text) > MAX_CHARS:
        await update.message.reply_text(
            f"Текст длиннее {MAX_CHARS} символов — разбей на части.")
        return
    await show_typing(context, update.message.chat_id)
    try:
        utts = tag_text(text)
    except Exception as e:
        log.exception("tag_text failed")
        await update.message.reply_text(f"Не смог разобрать диалог: {e}")
        return
    await present(update.message.chat_id, utts, context)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    await show_typing(context, msg.chat_id)
    try:
        photo = msg.photo[-1]  # самое большое превью
        f = await context.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await f.download_to_memory(buf)
        utts = tag_photo(buf.getvalue(), "image/jpeg")
    except Exception as e:
        log.exception("tag_photo failed")
        await msg.reply_text(
            f"Не смог распознать текст на фото: {e}\n"
            "Попробуй фото поярче или пришли текстом.")
        return
    await present(msg.chat_id, utts, context)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat_id
    st = PENDING.get(chat_id)
    if not st:
        await q.edit_message_text("Сессия устарела. Пришли диалог заново.")
        return
    if q.data.startswith("tgl:"):
        i = int(q.data.split(":", 1)[1])
        sp = st["order"][i]
        cur = st["tags"][sp]
        st["tags"][sp] = CYCLE[(CYCLE.index(cur) + 1) % len(CYCLE)]
        for u in st["utterances"]:
            if u["speaker"] == sp:
                u["tag"] = st["tags"][sp]
        await q.edit_message_text(summary_text(chat_id),
                                  reply_markup=build_keyboard(chat_id))
    elif q.data == "gen":
        await q.edit_message_text("⏳ Генерирую озвучку, это займёт ~30–60 секунд…")
        await generate_and_send(chat_id, context)


# ----------------------- TTS -----------------------

def tts_utterance(text: str, tag: str) -> bytes:
    voice, instr = ROLE_VOICE[tag]
    prompt = (f"{instr}. Modern Standard Arabic (fusha), clear diction, "
              f"natural pace: {text}")
    r = client.models.generate_content(
        model=TTS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice)))))
    return r.candidates[0].content.parts[0].inline_data.data


async def generate_and_send(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    st = PENDING.get(chat_id)
    if not st:
        return
    utts = st["utterances"]
    silence = b"\x00\x00" * int(24000 * 0.45)  # 0.45 сек между репликами
    chunks = []
    try:
        for i, u in enumerate(utts, 1):
            chunks.append(tts_utterance(u["text"], u["tag"]))
            if i % 5 == 0:
                await context.bot.send_message(
                    chat_id, f"⏳ {i}/{len(utts)} реплик готово…")
    except Exception as e:
        log.exception("tts failed")
        await context.bot.send_message(chat_id, f"Ошибка TTS: {e}")
        return

    pcm = silence.join(chunks)
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "out.wav")
        mp3_path = os.path.join(td, "out.mp3")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame",
             "-b:a", "64k", "-ac", "1", mp3_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(mp3_path, "rb") as f:
            mp3 = f.read()

    speakers = " | ".join(f"{sp}={TAG_LABEL[st['tags'][sp]]}" for sp in st["order"])
    await context.bot.send_audio(chat_id, audio=mp3,
                                 title="Озвучка диалога",
                                 caption=f"🎧 {speakers}"[:1000])


# ----------------------- ЗАПУСК -----------------------

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Не даёт боту падать на временных сетевых конфликтах Telegram."""
    err = context.error
    if isinstance(err, Conflict):
        log.warning("409 Conflict: тот же токен опрашивает другой экземпляр. "
                    "Обычно проходит сам через несколько секунд, продолжаю работу.")
        return
    log.error("Необработанная ошибка: %s", err, exc_info=err)


class _Health(BaseHTTPRequestHandler):
    """Тривиальный health-эндпоинт: Render считает сервис живым,
    только если процесс слушает порт."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_HEAD(self):
        self.do_GET()

    def log_message(self, *args):
        pass


def start_health_server():
    """Поднимает HTTP-ответ на PORT, пока бот работает в режиме polling."""
    def run():
        try:
            srv = HTTPServer(("0.0.0.0", PORT), _Health)
            log.info("Health-сервер слушает порт %s", PORT)
            srv.serve_forever()
        except Exception as e:
            log.warning("Health-сервер не поднялся: %s", e)
    threading.Thread(target=run, daemon=True).start()


def main():
    app = Application.builder().token(TG_TOKEN).build()
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    if SERVICE_URL:
        # Продакшен: Telegram сам присылает обновления на наш HTTPS-адрес.
        log.info("Webhook mode: %s/%s", SERVICE_URL, WEBHOOK_SECRET)
        app.run_webhook(listen="0.0.0.0", port=PORT,
                        url_path=WEBHOOK_SECRET,
                        webhook_url=f"{SERVICE_URL}/{WEBHOOK_SECRET}")
    else:
        # Пока SERVICE_URL не задан: polling + health-сервер, чтобы деплой стал Live.
        log.info("Polling mode, health-сервер на порту %s", PORT)
        start_health_server()
        app.run_polling(drop_pending_updates=DROP_PENDING)


if __name__ == "__main__":
    main()
