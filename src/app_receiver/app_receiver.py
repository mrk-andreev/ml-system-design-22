import json
import logging
import os

import redis
from telegram import ForceReply
from telegram import Update
from telegram.ext import CallbackContext
from telegram.ext import CommandHandler
from telegram.ext import Filters
from telegram.ext import MessageHandler
from telegram.ext import Updater

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


class TaskCreator:
    def __init__(self):
        self._r = redis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], db=0)

    def create(self, img_url, chart_id):
        self._r.rpush(os.environ["REDIS_QUEUE"], json.dumps({
            "img": img_url,
            "chat_id": chart_id
        }))


class BotCommandDispatcher:
    def __init__(self, task_creator: TaskCreator):
        self._task_creator = task_creator

    def start(self, update: Update, context: CallbackContext) -> None:
        user = update.effective_user
        update.message.reply_markdown_v2(
            fr'Hi {user.mention_markdown_v2()}\!',
            reply_markup=ForceReply(selective=True),
        )

    def help_command(self, update: Update, context: CallbackContext) -> None:
        update.message.reply_text('Help!')

    def echo(self, update: Update, context: CallbackContext) -> None:
        if len(update.message.photo) > 0:
            self._task_creator.create(
                context.bot.get_file(update.message.photo[-1].file_id).file_path,
                update.message.chat_id
            )
            return
        if (update.message.effective_attachment is not None
                and update.message.effective_attachment.thumb is not None):
            self._task_creator.create(
                context.bot.get_file(update.message.effective_attachment.thumb.file_id).file_path,
                update.message.chat_id
            )


def main() -> None:
    bot_command_dispatcher = BotCommandDispatcher(TaskCreator())
    updater = Updater(os.environ["TELEGRAM_BOT_TOKEN"])
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", bot_command_dispatcher.start))
    dispatcher.add_handler(CommandHandler("help", bot_command_dispatcher.help_command))
    dispatcher.add_handler(MessageHandler(~Filters.command, bot_command_dispatcher.echo))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
