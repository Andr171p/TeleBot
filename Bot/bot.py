import logging
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils import executor
from Bot.config import token
from GigaChat.chat import GigaChat
from GigaChat.auth_data import client_id, client_secret

logging.basicConfig(level=logging.INFO)

bot = Bot(token=token)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    '''button_question = types.KeyboardButton('question')
    button_dialog = types.KeyboardButton('dialog')
    keyboard.add(button_question, button_dialog)'''
    await message.answer("Я еще не сделал кнопки поятому просто напиши ботяре",
                         reply_markup=keyboard)


@dp.message_handler()
async def get_bot_answer(message: types.Message):
    giga_chat = GigaChat(
        client_id=client_id,
        client_secret=client_secret
    )
    history = []
    _, bot_answer = giga_chat.get_chat_dialog(user_message=message.text,
                                              conversation_history=history)

    await message.reply(bot_answer)


def main():
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)


if __name__ == "__main__":
    main()
