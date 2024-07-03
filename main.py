import telebot
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import os
import re
import requests
import openai
import sqlite3
import time
from dotenv import load_dotenv
from telebot import types
from loguru import logger

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logger.add("bot.log", rotation="1 MB")  # Логирование в файл с ротацией

# Проверка загрузки переменных окружения
admin_usernames = os.getenv("ADMIN_USERNAMES", "")
logger.info(f"Loaded admin usernames: {admin_usernames}")


# Инициализация базы данных
def init_db():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS allowed_users (username TEXT PRIMARY KEY)''')
        c.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            message TEXT,
            direction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


# Функции для работы с базой данных
def log_message(username, message, direction):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO messages (username, message, direction) VALUES (?, ?, ?)",
                  (username, message, direction))
        conn.commit()
        conn.close()
        logger.debug(f"Logged message from {username} (direction: {direction}): {message}")
    except Exception as e:
        logger.error(f"Error logging message: {e}")


def fetch_dialogue(username):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT message, direction, timestamp FROM messages WHERE username = ? ORDER BY timestamp",
                  (username,))
        messages = c.fetchall()
        conn.close()
        dialogue = []
        for message, direction, timestamp in messages:
            dialogue.append(f"{timestamp} {'Входящее' if direction == 'incoming' else 'Исходящее'}: {message}")
        logger.debug(f"Fetched dialogue for {username}: {dialogue}")
        return "\n".join(dialogue)
    except Exception as e:
        logger.error(f"Error fetching dialogue: {e}")
        return "Ошибка при загрузке диалога."


def add_user_to_db(username):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO allowed_users (username) VALUES (?)", (username,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error adding user to database: {e}")


def remove_user_from_db(username):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("DELETE FROM allowed_users WHERE username = ?", (username,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error removing user from database: {e}")


def delete_messages_user(username):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        logger.info(f"Все сообщения пользователя {username} удалены.")
    except Exception as e:
        logger.error(f"Error deleting messages for user: {e}")


def is_user_allowed(username):
    admin_usernames = os.getenv("ADMIN_USERNAMES", "").split(',')
    logger.debug(f"Admin usernames: {admin_usernames}")
    if username in admin_usernames:
        return True
    else:
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT username FROM allowed_users WHERE username = ?", (username,))
            user = c.fetchone()
            conn.close()
            return user is not None
        except Exception as e:
            logger.error(f"Error checking if user is allowed: {e}")
            return False


def get_all_users():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT username FROM allowed_users")
        users = c.fetchall()
        conn.close()
        return [user[0] for user in users]
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        return []


# Инициализация базы данных
init_db()


def load_document_text(url: str) -> str:
    """Загружает текст документа по URL Google Docs."""
    match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Docs URL')
    doc_id = match_.group(1)
    response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
    response.raise_for_status()
    return response.text


# Проверка и загрузка API ключей
api_key = os.getenv("YOUR_API_KEY")
if api_key is None:
    raise Exception("API key for OpenAI is not set.")
openai.api_key = api_key

# Загрузка и обработка документов
try:
    system = load_document_text(
        'https://docs.google.com/document/d/1U5Y5OumzWOLod48hft018ipcuG2B7o4CzrnMnMgaBfU/edit?usp=sharing')
    database = load_document_text(
        'https://docs.google.com/document/d/1tA9FUbaDA768R2idA42xohrtz8-HdPJHJAvPet0mgFI/edit?usp=sharing')
except Exception as e:
    logger.error(f"Error loading documents: {e}")
    raise

splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)
source_chunks = [Document(page_content=chunk, metadata={}) for chunk in splitter.split_text(database)]

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(source_chunks, embeddings)


class TelegramBot:
    def __init__(self, gpt_instance, search_index):
        self.gpt = gpt_instance
        self.index = search_index
        token = os.getenv("YOUR_BOT_TOKEN")
        if token is None:
            raise Exception("Telegram Bot Token не определен в переменных окружения")
        self.bot = telebot.TeleBot(token)


bot = TelegramBot(gpt_instance=embeddings, search_index=db).bot
chat_histories = {}
chat_summaries = {}
dialog_states = {}


# Функция для создания клавиатуры с одной кнопкой
def create_single_button_keyboard(button_text):
    keyboard = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    button = types.KeyboardButton(button_text)
    keyboard.add(button)
    return keyboard


# Функция для отправки длинных сообщений с проверкой ссылок
def send_long_text(chat_id: int, text: str, bot):
    MAX_MESSAGE_LENGTH = 4096  # Максимальная длина сообщения в Telegram
    contains_link = bool(re.search(r'http[s]?://', text))

    # Отправка сообщения, деление на части если необходимо
    if len(text) <= MAX_MESSAGE_LENGTH:
        bot.send_message(chat_id=chat_id, text=text)
    else:
        parts = [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]
        for part in parts:
            bot.send_message(chat_id=chat_id, text=part)

    # Проверка наличия ссылки
    if contains_link:
<<<<<<< HEAD
        time.sleep(3)  # Задержка в 3 секунд
=======
        time.sleep(3)  # Задержка в 3 секунды
>>>>>>> origin/master

        # Отправляем стикер
        sticker_file_id = 'CAACAgIAAxkBAAIeeGZ6eXPrVYYAAWRJIHuhRDscfGvq9wACzDcAAkQsqUpvTd4i2f0HnTUE'  # Замените на ваш file_id стикера
        bot.send_sticker(chat_id, sticker_file_id)

        # Отправляем текстовое сообщение отдельно
        magic_message = "IT сфера - это современная магия! Поздравляю! Ты большой молодец! Теперь ты знаешь в каком направлении тебе обучаться!"
        bot.send_message(chat_id, magic_message)

        # Устанавливаем состояние завершения диалога
        dialog_states[chat_id] = "finished"

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    username = message.from_user.username
    logger.debug(f"Received /start command from {username} in chat_id: {chat_id}")

    bot.send_sticker(chat_id, 'CAACAgIAAxkBAAIedWZ6eTB3dgFVRP0ammpMpEqFR138AAKxOgACR_2hSkN5bfKbzeJFNQQ')
    welcome_message = """
Привет, я — Сова!
Я прилетела к тебе из онлайн школы Реботика,
в которой ребята изучают цифровые технологии и навыки XXI века, 
чтобы помочь тебе выбрать направление или профессию, 
которые тебе подойдут наилучшим образом. 
Для этого просто ответь на несколько моих вопросов. 
Договорились?"""
    bot.send_message(chat_id, welcome_message, reply_markup=create_single_button_keyboard("Хорошо"))

    dialog_states[chat_id] = "awaiting_confirmation"
    logger.debug(f"State set to awaiting_confirmation for chat_id: {chat_id}")

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_message(message):
    chat_id = message.chat.id
    user_message = message.text
    username = message.from_user.username
    logger.debug(f"Received message: {user_message} from {username} in chat_id: {chat_id}")

    if dialog_states.get(chat_id) == "finished":
        bot.send_message(chat_id,
                         "👇Ты уже завершил тестирование. Пожалуйста! Если хочешь пройти ещё раз, то нажми кнопку Cтарт в меню.")
        return

    if dialog_states.get(chat_id) == "awaiting_confirmation":
        if user_message.lower() == "хорошо":
            bot.send_message(chat_id, "Отлично! И чтобы у нас всё получилось, пожалуйста, отвечай честно! Начнём?", reply_markup=create_single_button_keyboard("Погнали"))
            bot.send_sticker(chat_id, 'CAACAgIAAxkBAAIfFWaDwyfZI-2yLIza5jHlPCqUBFpeAALsRwACdA2gS_Z0OaZBctWSNQQ')
            dialog_states[chat_id] = "awaiting_ready"
            logger.debug(f"State set to awaiting_ready for chat_id: {chat_id}")
        else:
            bot.send_message(chat_id, "Чтобы продолжить, просто нажми на кнопку.", reply_markup=create_single_button_keyboard("Хорошо"))
        return

    elif dialog_states.get(chat_id) == "awaiting_ready":
        if user_message.lower() == "погнали":
            dialog_states[chat_id] = "active"
            logger.debug(f"State set to active for chat_id: {chat_id}")
            bot.send_message(chat_id, "Как тебя зовут?", reply_markup=types.ReplyKeyboardRemove())
        else:
            bot.send_message(chat_id, "Чтобы продолжить, просто нажми на кнопку.", reply_markup=create_single_button_keyboard("Погнали"))
        return

    if dialog_states.get(chat_id) == "active":
        # Здесь ваша логика взаимодействия с GPT
        if not is_user_allowed(username):
            bot.reply_to(message, "Вы не имеете доступа к этому боту.")
            logger.debug(f"Access denied for user {username}")
            return

        if chat_id not in chat_histories:
            chat_histories[chat_id] = []
            chat_summaries[chat_id] = ""

        chat_histories[chat_id].append(("user", user_message))
        log_message(username, user_message, 'incoming')

        # Обновление суммаризированной истории
        current_summary = f"{chat_summaries[chat_id]} User: {user_message}"
        if len(current_summary) > 5000:  # Увеличиваем ограничение длины суммаризации до 5000 символов
            current_summary = current_summary[-5000:]
        chat_summaries[chat_id] = current_summary

        # Поиск релевантных отрезков из базы знаний
        docs = db.similarity_search(user_message, k=4)
        message_content = '\n '.join(
            [f'\nОтрывок документа №{i + 1}\n=====================' + doc.page_content + '\n' for i, doc in
             enumerate(docs)])

        # Формирование запроса к OpenAI
        messages = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Документ с информацией для ответа клиента: {message_content}\n\nВопрос клиента: {current_summary}"}
        ]
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o-2024-05-13",
                messages=messages,
                temperature=0.6,
                frequency_penalty=2.0
            )
            answer = completion.choices[0].message.content
            logger.info(f"Sending answer to {chat_id} ({username}): {answer}")
            chat_histories[chat_id].append(("bot", answer))
            chat_summaries[chat_id] += f" Bot: {answer}"
            log_message(username, answer, 'outgoing')
            send_long_text(chat_id, answer, bot)  # Используем send_long_text для отправки сообщения
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            bot.reply_to(message, "Произошла ошибка при обработке вашего запроса. Попробуйте позже.")

# Запуск бота
bot.polling(none_stop=True)
