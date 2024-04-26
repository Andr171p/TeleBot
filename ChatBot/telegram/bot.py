import telebot
import time
from ChatBot.telegram.config import bot_token
from ChatBot.giga_chat.chat import GigaChatBot
from langchain.memory import ConversationBufferMemory


user_conversation = {}

bot = telebot.TeleBot(bot_token)

giga_chat = GigaChatBot()
giga_chat.create_giga_model()
giga_chat.add_prompt(
    file_path=r"C:\Users\andre\TyuiuProjectParser\TestTeleBot\ChatBot\giga_chat\prompt\system.txt"
)


@bot.message_handler(content_types=['audio', 'video', 'document', 'photo'])
def not_text(message):
    user_id = message.chat.id
    bot.send_message(user_id, "Я работаю только с текстовыми сообщениями")


@bot.message_handler(content_types=['text'])
def bot_answer_message(message):
    user_id = message.chat.id
    if user_id not in user_conversation:
        user_conversation[user_id] = ConversationBufferMemory()

    giga_chat.conversation.memory = user_conversation[user_id]

    response = giga_chat.giga_answer(user_text=message.text)
    bot.send_message(user_id, giga_chat.conversation.memory.chat_memory.messages[-1].content)
    time.sleep(2)


bot.polling(none_stop=True)
