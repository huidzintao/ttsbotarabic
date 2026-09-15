#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот озвучки арабских диалогов.

Что умеет:
1. Принимает ТЕКСТ диалога на арабском -> определяет персонажей и их пол/возраст -> озвучивает в MP3.
2. Принимает ФОТО/скрин диалога -> распознаёт текст (OCR через нейросеть) -> то же самое.
3. Перед генерацией показывает кнопки: пользователь одним нажатием меняет пол/возраст персонажа.
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
GEMINI_API_KEYS = [k.strip() for k in os.environ.get(
    "GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",")
    if k.strip()]
if not GEMINI_API_KEYS:
    GEMINI_API_KEYS = [_req("GEMINI_API_KEY")]   # понятная диагностика
# ВАЖНО: провайдер периодически закрывает старые модели для новых аккаунтов.
# Поэтому у каждой модели есть основной id и цепочка запасных: при ошибке 404
# код сам пробует следующую, править ничего не нужно.
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_MODEL_FALLBACKS = [m.strip() for m in os.environ.get(
    "TTS_MODEL_FALLBACKS",
    "gemini-3.1-flash-tts-preview,gemini-2.5-pro-preview-tts").split(",") if m.strip()]

TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemini-3.6-flash")
TEXT_MODEL_FALLBACKS = [m.strip() for m in os.environ.get(
    "TEXT_MODEL_FALLBACKS",
    "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-3.7-flash").split(",") if m.strip()]
SERVICE_URL = os.environ.get("SERVICE_URL", "").rstrip("/")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "wh" + TG_TOKEN.split(":", 1)[0])
PORT = int(os.environ.get("PORT", "8080"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "4000"))
# DROP_PENDING=1 — при старте выбросить накопившуюся очередь сообщений.
# По умолчанию 0: не теряем сообщения, отправленные пока сервис спал.
DROP_PENDING = os.environ.get("DROP_PENDING", "0") == "1"

# ---- Лимиты бесплатного тарифа: 3 запроса озвучки в минуту на проект ----
# Поэтому озвучка идёт «по одной реплике с паузой», а не пачкой.
TTS_RPM_LIMIT = int(os.environ.get("TTS_RPM_LIMIT", "3"))
TTS_MIN_INTERVAL = float(os.environ.get("TTS_MIN_INTERVAL", "22"))  # сек между запросами
TTS_MAX_CHUNKS = int(os.environ.get("TTS_MAX_CHUNKS", "8"))        # реплик за один прогон
TTS_MAX_RETRIES = int(os.environ.get("TTS_MAX_RETRIES", "3"))
_last_tts_call = 0.0  # время последнего TTS-запроса (для ограничителя темпа)

# ---------------- КЛЮЧИ: автоматическая ротация ----------------
# Лимиты сервиса считаются НА ПРОЕКТ, а не на ключ. Несколько ключей одного
# проекта лимит НЕ увеличивают — свободных запросов больше становится только
# за счёт РАЗНЫХ проектов/аккаунтов. Ротация ниже — ровно для такого случая.
class AllKeysExhausted(Exception):
    """Все ключи исчерпали суточную квоту (сброс — полночь по Тихоокеанскому времени)."""


_clients = {k: genai.Client(api_key=k) for k in GEMINI_API_KEYS}
# Метки исчерпания ключей хранит общий счётчик (см. блок «ОБЩИЙ СЧЁТЧИК» ниже).
_key_cursor = [0]
_key_in_use = [GEMINI_API_KEYS[0]]


def _mask(key: str) -> str:
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "ключ"


def _next_midnight_pt() -> float:
    """Ближайшая полночь по Тихоокеанскому времени — момент сброса суточных квот."""
    from datetime import datetime, timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/Los_Angeles")
    except Exception:
        tz = timezone(timedelta(hours=-7))
    now = datetime.now(tz)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=5,
                                            second=0, microsecond=0)
    return nxt.timestamp()


def _is_daily_quota(err: Exception) -> bool:
    """Отличает СУТОЧНЫЙ лимит (RPD) от МИНУТНОГО (RPM)."""
    s = str(err)
    return ("PerDay" in s) or ("RequestsPerDay" in s) or ("per day" in s.lower())


def _pick_key() -> str:
    now = time.time()
    for step in range(len(GEMINI_API_KEYS)):
        i = (_key_cursor[0] + step) % len(GEMINI_API_KEYS)
        k = GEMINI_API_KEYS[i]
        if key_blocked_until(k) <= now:
            _key_cursor[0] = i
            return k
    return GEMINI_API_KEYS[_key_cursor[0]]


def active_key() -> str:
    return _key_in_use[0]


def call_model(model: str, contents, config=None, kind: str = "tts"):
    """Один запрос с перебором ключей: ключ исчерпал СУТОЧНУЮ квоту —
    помечаем его до сброса и молча берём следующий."""
    now = time.time()
    if all(key_blocked_until(k) > now for k in GEMINI_API_KEYS):
        raise AllKeysExhausted("все ключи исчерпали суточную квоту")
    last_err = None
    tried = set()
    for _ in range(len(GEMINI_API_KEYS)):
        key = _pick_key()
        if key in tried:
            break
        tried.add(key)
        try:
            kw = {"model": model, "contents": contents}
            if config is not None:
                kw["config"] = config
            r = _clients[key].models.generate_content(**kw)
            _bump(key, kind)
            if _key_in_use[0] != key:
                log.info("Переключился на резервный ключ %s", _mask(key))
                _key_in_use[0] = key
            return r
        except Exception as e:
            last_err = e
            _bump(key, kind)          # отказ по лимиту тоже расходует квоту
            if _is_quota(e) and _is_daily_quota(e):
                _block(key, _next_midnight_pt())
                free = sum(1 for k in GEMINI_API_KEYS
                           if key_blocked_until(k) <= time.time())
                log.warning("Ключ %s исчерпал суточную квоту; свободных ещё: %d",
                            _mask(key), free)
                continue
            raise
    if _is_quota(last_err) and _is_daily_quota(last_err):
        raise AllKeysExhausted("все ключи исчерпали суточную квоту") from last_err
    raise last_err


# ---------------- ОБЩИЙ СЧЁТЧИК ЛИМИТА (скрытно от пользователя) ----------------
# Пользователь видит только «озвучено X из Y» и время следующей доступной
# озвучки. Ни одного слова о ключах, их числе или ротации наружу не уходит.
# Состояние — JSON рядом с ботом. ВАЖНО: на бесплатном тарифе хостинга диск
# непостоянный: при перезапуске/передеплое счётчик обнуляется.
STATE_FILE = os.environ.get("STATE_FILE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "usage_state.json"))
RPD_PER_PROJECT = int(os.environ.get("RPD_PER_PROJECT", "10"))  # озвучек в сутки на один источник
TZ_NAME = os.environ.get("TZ_NAME", "Europe/Moscow")            # часовой пояс пользователя
KEY_ID = {k: hashlib.sha1(k.encode()).hexdigest()[:10] for k in GEMINI_API_KEYS}
_state_lock = threading.Lock()
_state = {"day": "", "keys": {}}


def _user_tz():
    from datetime import timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(TZ_NAME)
    except Exception:
        return timezone(timedelta(hours=3))


def _fmt_local(ts: float) -> str:
    """Момент времени по часам пользователя."""
    from datetime import datetime
    return datetime.fromtimestamp(ts, _user_tz()).strftime("%d.%m в %H:%M")


def _pt_day() -> str:
    """Дата по Тихоокеанскому времени: в полночь по нему сбрасываются квоты."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        from datetime import datetime as d, timedelta, timezone
        return d.now(timezone(timedelta(hours=-7))).strftime("%Y-%m-%d")


def _blank_state(day: str) -> dict:
    return {"day": day,
            "keys": {kid: {"used": 0, "until": 0.0} for kid in KEY_ID.values()}}


def _load_state() -> None:
    global _state
    day = _pt_day()
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            cur = json.load(f)
    except Exception:
        cur = None
    if not isinstance(cur, dict) or cur.get("day") != day:
        _state = _blank_state(day)
        return
    for kid in KEY_ID.values():
        cur.setdefault("keys", {}).setdefault(kid, {"used": 0, "until": 0.0})
    _state = cur


def _save_state() -> None:
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_state, f)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        log.warning("счётчик не сохранён: %s", e)


def _rollover() -> None:
    """Новые тихоокеанские сутки -> счётчики и метки обнуляются."""
    global _state
    if _state.get("day") != _pt_day():
        _state = _blank_state(_pt_day())


def _entry(key: str) -> dict:
    return _state["keys"].setdefault(KEY_ID[key], {"used": 0, "until": 0.0})


def _bump(key: str, kind: str) -> None:
    """+1 к числу попыток: считаем и удачные, и отклонённые по лимиту запросы."""
    try:
        with _state_lock:
            _rollover()
            e = _entry(key)
            e["used"] = int(e.get("used", 0)) + 1
            _save_state()
    except Exception as ex:
        log.warning("счётчик недоступен: %s", ex)


def _block(key: str, until: float) -> None:
    """Ключ исчерпал суточную квоту; после указанного момента он снова свободен."""
    try:
        with _state_lock:
            _rollover()
            _entry(key)["until"] = float(until)
            _save_state()
    except Exception as ex:
        log.warning("не смог отметить лимит: %s", ex)


def key_blocked_until(key: str) -> float:
    with _state_lock:
        _rollover()
        return float(_entry(key).get("until", 0.0))


def quota_snapshot() -> dict:
    """Сводка без единого упоминания ключей: только цифры и время."""
    now = time.time()
    with _state_lock:
        _rollover()
        used = sum(int(e.get("used", 0)) for e in _state["keys"].values())
        untils = [float(e.get("until", 0.0)) for e in _state["keys"].values()]
    total = max(1, len(GEMINI_API_KEYS) * RPD_PER_PROJECT)
    blocked = [u for u in untils if u > now]
    all_out = (bool(untils) and all(u > now for u in untils)) or used >= total
    if blocked:
        nxt = min(blocked)
    elif all_out:
        nxt = _next_midnight_pt()
    else:
        nxt = 0.0
    return {"used": used, "total": total, "left": max(0, total - used),
            "all_exhausted": bool(all_out), "next_available": nxt}


def _quota_footer() -> str:
    """Строка под готовым аудио: сколько израсходовано и когда будет снова."""
    s = quota_snapshot()
    if s["all_exhausted"]:
        return (f"⏳ Лимит озвучки на сегодня исчерпан. Следующая озвучка будет "
                f"доступна после {_fmt_local(s['next_available'])} (по вашему времени).")
    return (f"📊 Озвучено сегодня: {s['used']} из {s['total']} · осталось {s['left']}")


_load_state()


# ----------------------- ВЫБОР МОДЕЛИ С ЗАПАСНЫМИ ВАРИАНТАМИ -----------------------
def _is_model_gone(exc: Exception) -> bool:
    """True, если ошибка означает «модель не обслуживается» (404 / NOT_FOUND)."""
    s = str(exc)
    return ("404" in s) or ("NOT_FOUND" in s) or ("no longer available" in s)


def _is_quota(exc: Exception) -> bool:
    """True для 429 / RESOURCE_EXHAUSTED — временный лимит запросов."""
    if isinstance(exc, AllKeysExhausted):
        return False
    s = str(exc)
    return (("429" in s) or ("RESOURCE_EXHAUSTED" in s)
            or ("exceeded your current quota" in s))


def _parse_retry_delay(exc: Exception, default: float = 25.0) -> float:
    """Достаёт из ответа сервиса рекомендованную паузу ('retryDelay': '34s')."""
    s = str(exc)
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s)
    if not m:
        m = re.search(r"retry in (\d+(?:\.\d+)?)s", s, flags=re.I)
    if m:
        return min(float(m.group(1)) + 2.0, 65.0)
    return default


# Запоминаем уже найденную рабочую модель, чтобы не долбиться в битую на каждой реплике.
_RESOLVED = {"text": None, "tts": None}


def _model_chain(primary: str, fallbacks, kind: str):
    """Основная модель + запасные, начиная с уже проверенной рабочей."""
    chain = []
    done = _RESOLVED.get(kind)
    if done:
        chain.append(done)
    for m in [primary] + list(fallbacks):
        if m and m not in chain:
            chain.append(m)
    return chain


def _remember(kind: str, model: str):
    if _RESOLVED.get(kind) != model:
        log.info("Использую %s-модель: %s", kind, model)
        _RESOLVED[kind] = model


def call_text_model(contents):
    """Разбор ролей и распознавание фото: перебирает модели, пока не найдёт рабочую.
    Раньше падал сразу на 429 (лимит запросов): теперь ждёт паузу из ответа
    и повторяет, а потом переходит к запасной модели."""
    last_err = None
    for m in _model_chain(TEXT_MODEL, TEXT_MODEL_FALLBACKS, "text"):
        for attempt in range(2):
            try:
                r = call_model(m, contents, kind="text")
                _remember("text", m)
                return r
            except Exception as e:
                last_err = e
                if _is_model_gone(e):
                    log.warning("Модель '%s' недоступна, пробую следующую", m)
                    break
                if _is_quota(e):
                    wait = _parse_retry_delay(e, default=30.0)
                    log.warning("429 quota у text-модели '%s', жду %.0f сек", m, wait)
                    time.sleep(wait)
                    continue
                raise
    raise last_err


def call_tts_model(contents, config):
    """Озвучка: то же самое, но со своим аудио-конфигом."""
    last_err = None
    for m in _model_chain(TTS_MODEL, TTS_MODEL_FALLBACKS, "tts"):
        try:
            r = call_model(m, contents, config, kind="tts")
            _remember("tts", m)
            return r
        except Exception as e:
            if _is_model_gone(e):
                log.warning("TTS-модель '%s' недоступна, пробую следующую", m)
                last_err = e
                continue
            raise
    raise last_err

# ----------------------- ГОЛОСА (30 пресетов) -----------------------
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
    r = call_text_model(TAGGER_RULES + "\n\nТЕКСТ ДИАЛОГА:\n" + text)
    return parse_json_array(r.text)


def tag_photo(img_bytes: bytes, mime: str):
    prompt = (TAGGER_RULES +
              "\n\nСначала распознай арабский текст диалога с изображения дословно "
              "(сохрани диакритику и порядок реплик), затем верни JSON-массив.")
    r = call_text_model(
        [types.Part.from_bytes(data=img_bytes, mime_type=mime), prompt])
    return parse_json_array(r.text)


# ----------------------- ИНТЕРФЕЙС -----------------------

def build_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    st = PENDING[chat_id]
    rows = []
    for i, sp in enumerate(st["order"]):
        rows.append([InlineKeyboardButton(
            f"{sp}: {TAG_LABEL[st['tags'][sp]]}", callback_data=f"tgl:{i}")])
    done = st.get("cursor", 0)
    total = len(st["utterances"])
    if done <= 0:
        label = "✅ Озвучить"
    elif done < total:
        label = (f"▶️ Продолжить (реплики {done + 1}–"
                 f"{min(done + TTS_MAX_CHUNKS, total)} из {total})")
    else:
        label = "🔄 Озвучить заново (с начала)"
    rows.append([InlineKeyboardButton(label, callback_data="gen")])
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
    PENDING[chat_id] = {"utterances": clean, "order": order, "tags": tags,
                        "cursor": 0, "pcm_parts": []}
    await context.bot.send_message(chat_id, summary_text(chat_id),
                                   reply_markup=build_keyboard(chat_id))


# ----------------------- ОБРАБОТЧИКИ -----------------------

async def diag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Проверка связи с сервисами: какая модель разметки сейчас отвечает."""
    try:
        await asyncio.to_thread(call_text_model, "Ответь одним словом: ок")
        await update.message.reply_text("✅ Озвучка и разбор диалога работают.")
    except Exception as e:
        await update.message.reply_text(
            "❌ Сервис сейчас не отвечает.\n" + _tag_error_text(e))


async def limits_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Общий лимит озвучки на сутки — без каких-либо данных о ключах."""
    s = quota_snapshot()
    lines = [f"Сегодня озвучено: {s['used']} из {s['total']}",
             f"Осталось озвучек: {s['left']}"]
    if s["all_exhausted"]:
        lines += ["", "⏳ Лимит на сегодня исчерпан.",
                  "Следующая озвучка будет доступна после "
                  + _fmt_local(s["next_available"]) + " (по вашему времени)."]
    else:
        lines += ["", "Лимит обновляется каждый день в 00:00 по тихоокеанскому "
                      "времени.", "Пока лимит есть — просто пришли диалог."]
    await update.message.reply_text("📊 Общий лимит озвучки\n" + "\n".join(lines))


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    PENDING.pop(update.message.chat_id, None)
    await update.message.reply_text(
        "Начнём заново. Пришли диалог текстом или фото страницы учебника.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "السلام عليكم! 👋\n\n"
        "Я озвучиваю арабские диалоги и присылаю готовый MP3.\n\n"
        "Что прислать:\n"
        "1️⃣ Текст диалога на арабском — сам определю, кто говорит (шейх/мать/ребёнок), "
        "и предложу голоса.\n"
        "2️⃣ Фото/скрин страницы учебника — распознаю текст и сделаю то же самое.\n\n"
        "Перед генерацией покажу кнопки: пол/возраст любого персонажа можно поменять "
        "одним нажатием. Потом пришлю готовый MP3 🎧\n\n"
        "Проверить общий лимит на сутки: /limits"
    )


def _tag_error_text(err: Exception, photo: bool = False) -> str:
    """Сообщение об ошибке разметки с понятной причиной (без служебных названий)."""
    if isinstance(err, AllKeysExhausted):
        s = quota_snapshot()
        when = _fmt_local(s["next_available"] or _next_midnight_pt())
        return ("⏳ Лимит озвучки на сегодня исчерпан. Следующая озвучка будет "
                "доступна после " + when + " (по вашему времени).")
    if _is_quota(err):
        return ("⏳ Сервис разбора текста сейчас ограничивает частоту запросов. "
                "Подожди минуту и пришли то же самое ещё раз — дальше пойдёт.")
    if _is_model_gone(err):
        return ("⚠️ Модель разбора текста временно недоступна. "
                "Попробуй ещё раз через пару минут.")
    s = str(err).lower()
    if "api key" in s or "api_key" in s or "401" in s or "403" in s or "unauth" in s:
        return ("⚠️ Проблема с ключом доступа на стороне бота. "
                "Напиши владельцу бота — нужно проверить ключ.")
    if photo:
        return ("Не смог распознать текст на фото.\n"
                "Попробуй фото поярче/чётче или пришли диалог текстом.")
    return "Не смог разобрать диалог. Пришли его ещё раз или разбей на две части."


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
        log.exception("tag_text failed: %s", e)
        await update.message.reply_text(_tag_error_text(e))
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
        mime = "image/png" if (f.file_path or "").lower().endswith(".png") else "image/jpeg"
        utts = tag_photo(buf.getvalue(), mime)
    except Exception as e:
        log.exception("tag_photo failed: %s", e)
        await msg.reply_text(_tag_error_text(e, photo=True))
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
        # Смена голоса обнуляет собранные куски: иначе в одном файле
        # встретятся старый и новый тембр одного и того же персонажа.
        st["cursor"] = 0
        st["pcm_parts"] = []
        await q.edit_message_text(summary_text(chat_id),
                                  reply_markup=build_keyboard(chat_id))
    elif q.data == "gen":
        await q.edit_message_text("⏳ Генерирую озвучку, это займёт ~30–60 секунд…")
        await generate_and_send(chat_id, context)


# ----------------------- TTS -----------------------

def tts_utterance(text: str, tag: str) -> bytes:
    """Одна реплика. При 429 ждёт столько, сколько просит сервис, и повторяет."""
    voice, instr = ROLE_VOICE[tag]
    prompt = (f"{instr}. Modern Standard Arabic (fusha), clear diction, "
              f"natural pace: {text}")
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice))))
    last_err = None
    for attempt in range(1, TTS_MAX_RETRIES + 1):
        try:
            r = call_tts_model(prompt, config)
            return r.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            if not _is_quota(e):
                raise
            last_err = e
            wait = _parse_retry_delay(e, default=max(20.0, TTS_MIN_INTERVAL))
            log.warning("429 quota (попытка %d/%d), жду %.0f сек",
                        attempt, TTS_MAX_RETRIES, wait)
            time.sleep(wait)
    raise last_err


def tts_mono(utts, order, tags) -> bytes:
    """ОДИН говорящий -> ОДИН запрос с одним голосом на весь текст.
    Без этого одиночный диалог уходил в путь «по реплике» и тратил N запросов
    вместо одного."""
    sp = list(order)[0]
    voice, instr = ROLE_VOICE[tags[sp]]
    parts_text = [u["text"] for u in utts if u["speaker"] == sp]
    if not parts_text:
        parts_text = [u["text"] for u in utts]
    text = " ".join(parts_text)
    prompt = (instr + ". Read the following Arabic text aloud. "
              "Modern Standard Arabic (fusha), clear diction, natural pace: " + text)
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice))))
    last_err = None
    for attempt in range(1, TTS_MAX_RETRIES + 1):
        try:
            r = call_tts_model(prompt, config)
            return r.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            if not _is_quota(e):
                raise
            last_err = e
            time.sleep(_parse_retry_delay(e, default=30.0))
    raise last_err


def tts_dialogue_multispeaker(utts, order, tags) -> bytes:
    """ВЕСЬ диалог одним запросом: два голоса сразу, без склейки из кусков.
    Именно это снимает лимит частоты: 1 запрос на диалог вместо N запросов."""
    pairs = list(order)[:2]
    label = {sp: f"Speaker{i}" for i, sp in enumerate(pairs, 1)}
    transcript = "\n".join(
        f"{label.get(u['speaker'], 'Speaker1')}: {u['text']}" for u in utts)
    prompt = ("Read aloud this Arabic dialogue between two people. "
              "Modern Standard Arabic (fusha), clear diction, natural pace, "
              "natural pauses between turns. Do not read the speaker labels.\n\n"
              + transcript)
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker=label[sp],
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=ROLE_VOICE[tags[sp]][0])))
                    for sp in pairs])))
    last_err = None
    for attempt in range(1, TTS_MAX_RETRIES + 1):
        try:
            r = call_tts_model(prompt, config)
            return r.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            if not _is_quota(e):
                raise
            last_err = e
            wait = _parse_retry_delay(e, default=30.0)
            log.warning("429 в двухголосом режиме (попытка %d/%d), жду %.0f сек",
                        attempt, TTS_MAX_RETRIES, wait)
            time.sleep(wait)
    raise last_err


def pcm_to_mp3(pcm: bytes) -> bytes:
    """PCM 24 кГц mono -> MP3 64 kbps."""
    if not pcm:
        raise ValueError("нет аудио для склейки")
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
            return f.read()


def build_mp3(pcm_parts) -> bytes:
    """Склейка отдельных реплик в один MP3 (запасной путь: 3+ персонажа)."""
    silence = b"\x00\x00" * int(24000 * 0.45)  # 0.45 сек между репликами
    return pcm_to_mp3(silence.join(pcm_parts))


def _friendly_error(err: Exception) -> str:
    """Понятное объяснение сбоя — без служебных названий сервисов."""
    if isinstance(err, AllKeysExhausted):
        s = quota_snapshot()
        when = _fmt_local(s["next_available"] or _next_midnight_pt())
        return ("⏳ Лимит озвучки на сегодня исчерпан. Следующая озвучка будет "
                "доступна после " + when + " (по вашему времени).")
    if _is_quota(err):
        return (" Сервис озвучки сейчас ограничивает частоту запросов "
                "(несколько в минуту). Подожди минуту — продолжим с того же места.")
    if _is_model_gone(err):
        return "⚠️ Голосовой движок переключился на резервный. Нажми ещё раз."
    if isinstance(err, ValueError):
        return "⚠️ Нечего склеивать: реплики не записались."
    return "⚠️ Сбой на стороне сервиса озвучки. Попробуй ещё раз через минуту."


async def report_failure(chat_id: int, context: ContextTypes.DEFAULT_TYPE,
                         err: Exception, st: dict, total: int):
    """Говорит, на чём прервались, и ВСЕГДА возвращает кнопку продолжения.
    Без этого после сбоя в чате не остаётся кнопок и бот кажется мёртвым."""
    done = int(st.get("cursor", 0) or 0)
    head = _friendly_error(err)
    if done <= 0:
        tail = "\n\nНажми «✅ Озвучить», чтобы попробовать снова."
    elif done < total:
        tail = (f"\n\nЗаписано {done} из {total} реплик. "
                f"Нажми «▶️ Продолжить» — допишу остальные в тот же файл.")
    else:
        tail = "\n\nВсе реплики записаны, осталась склейка. Нажми «🔄 Озвучить заново»."
    try:
        await context.bot.send_message(chat_id, head + tail,
                                       reply_markup=build_keyboard(chat_id))
    except Exception:
        log.exception("не смог отправить сообщение об ошибке")


async def generate_and_send(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    st = PENDING.get(chat_id)
    if not st:
        return
    global _last_tts_call

    all_utts = st["utterances"]
    total = len(all_utts)

    # Два персонажа -> ВЕСЬ диалог одним запросом (два голоса сразу).
    # Это главное лекарство от «ограничивает частоту запросов»:
    # вместо 19 обращений к сервису уходит ровно ОДНО.
    n_spk = len(st["order"])
    # 1 говорящий -> моно (1 запрос). 2 -> двухголосый (1 запрос).
    # 3 и больше -> путь по репликам: сервис принимает максимум 2 голоса.
    if (n_spk in (1, 2) and not st.get("pcm_parts")
            and st.get("cursor", 0) == 0):
        await context.bot.send_message(
            chat_id,
            "⏳Сабр — это половина веры.Озвучка идёт в несколько заходов, "
            "потом склеиваю. Жду — и ты жди.")
        try:
            fn = tts_dialogue_multispeaker if n_spk == 2 else tts_mono
            pcm = await asyncio.to_thread(fn, all_utts, st["order"], st["tags"])
            mp3 = await asyncio.to_thread(pcm_to_mp3, pcm)
            speakers = " | ".join(
                f"{sp}={TAG_LABEL[st['tags'][sp]]}" for sp in st["order"])
            mode_txt = "два голоса" if n_spk == 2 else "один голос"
            st["cursor"] = total
            cap = (f"🎧 Весь диалог целиком ({total} реплик, {mode_txt}) · "
                   f"{speakers}")[:900]
            await context.bot.send_audio(
                chat_id, audio=mp3, title="Озвучка диалога",
                caption=cap + "\n\n" + _quota_footer())
            return
        except Exception as e:
            log.warning("двухголосый режим не сработал (%s) — перехожу "
                        "к озвучке по репликам", e)
            # не выходим: ниже сработает обычный путь по репликам
    start = st.get("cursor", 0)
    if start >= total:            # весь диалог уже озвучен -> начинаем заново
        start = 0
        st["pcm_parts"] = []
    end = min(start + TTS_MAX_CHUNKS, total)
    batch = all_utts[start:end]
    st["cursor"] = start

    if total > TTS_MAX_CHUNKS:
        await context.bot.send_message(
            chat_id,
            f"ℹ️ В диалоге {total} реплик. Озвучиваю частями по "
            f"{TTS_MAX_CHUNKS}, но всё складываю в ОДИН общий файл. "
            f"Сейчас — реплики {start + 1}–{end}.")

    est_min = max(0.0, (len(batch) - 1) * TTS_MIN_INTERVAL) / 60.0
    if est_min >= 0.4:
        await context.bot.send_message(
            chat_id,
            "⏳Сабр — это половина веры.Озвучка идёт в несколько заходов, "
            "потом склеиваю. Жду — и ты жди.")
    try:
        for n, u in enumerate(batch, 1):
            pos = start + n
            wait = TTS_MIN_INTERVAL - (time.monotonic() - _last_tts_call)
            if wait > 5 and n > 1:
                await context.bot.send_message(
                    chat_id, f"⏳ Реплика {pos}/{total} — готовлю звук…")
            if wait > 0:
                await asyncio.sleep(wait)
            # Реплика целиком (включая паузы-ретраи) выполняется в потоке:
            # синхронный time.sleep внутри async-хендлера замораживал бота,
            # и он переставал отвечать даже на /start.
            part = await asyncio.to_thread(tts_utterance, u["text"], u["tag"])
            st["pcm_parts"].append(part)
            _last_tts_call = time.monotonic()
            st["cursor"] = pos      # при сбое продолжим с этого места
            if n % 3 == 0:
                await context.bot.send_message(
                    chat_id, f"⏳ {pos}/{total} реплик готово…")
    except Exception as e:
        log.exception("tts failed")
        await report_failure(chat_id, context, e, st, total)
        return

    try:
        mp3 = await asyncio.to_thread(build_mp3, st["pcm_parts"])
    except Exception as e:
        log.exception("mp3 build failed")
        await report_failure(chat_id, context, e, st, total)
        return

    speakers = " | ".join(f"{sp}={TAG_LABEL[st['tags'][sp]]}" for sp in st["order"])
    done = st["cursor"]
    if done < total:
        cap = (f"🎧 Реплики 1–{done} из {total} · {speakers}\n"
               f"Файл пока неполный — нажми «▶️ Продолжить», "
               f"и я допишу следующие реплики в этот же диалог.")
    else:
        cap = f"🎧 Весь диалог целиком ({total} реплик) · {speakers}"
    await context.bot.send_audio(
        chat_id, audio=mp3, title="Озвучка диалога",
        caption=(cap[:900] + "\n\n" + _quota_footer())[:1024])
    if done < total:
        await context.bot.send_message(
            chat_id,
            f"Осталось реплик: {total - done}. Продолжаем?",
            reply_markup=build_keyboard(chat_id))


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
            srv = ThreadingHTTPServer(("0.0.0.0", PORT), _Health)
            log.info("Health-сервер слушает порт %s", PORT)
            srv.serve_forever()
        except Exception as e:
            log.warning("Health-сервер не поднялся: %s", e)
    threading.Thread(target=run, daemon=True).start()


def main():
    app = Application.builder().token(TG_TOKEN).build()
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("diag", diag))
    app.add_handler(CommandHandler("limits", limits_status))
    app.add_handler(CommandHandler("stats", limits_status))
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
