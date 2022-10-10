import abc
import json
import logging
import os
import uuid

import boto3
import redis
import requests
from telegram import ForceReply
from telegram import Update
from telegram.ext import CallbackContext
from telegram.ext import CallbackQueryHandler
from telegram.ext import CommandHandler
from telegram.ext import Filters
from telegram.ext import MessageHandler
from telegram.ext import Updater

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


class BlobStorageWriter(abc.ABC):
    @abc.abstractmethod
    def save(self, file_link):
        pass


class RemoteLinkBlobStorageWriter(BlobStorageWriter):
    def save(self, file_link):
        return {
            "storage": "REMOTE_LINK",
            "href": file_link,
        }


class S3BlobStorageWriter(BlobStorageWriter):
    def __init__(self, endpoint_url, access_key_id, secret_access_key, bucket_name, path_prefix, verify=False):
        self._s3 = boto3.resource(
            's3',
            endpoint_url=endpoint_url,  # 'https://<minio>:9000'
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=None,
            config=boto3.session.Config(signature_version='s3v4'),
            verify=verify
        )
        self._bucket_name = bucket_name
        self._path_prefix = path_prefix

    def save(self, file_link):
        obj_name = f"{self._path_prefix}/{uuid.uuid4()}.jpg"
        obj = self._s3.Object(self._bucket_name, obj_name)
        obj.put(Body=self._load_content(file_link))
        return {
            "storage": "S3",
            "href": {
                "bucket": self._bucket_name,
                "obj_name": obj_name
            },
        }

    @classmethod
    def _load_content(cls, file_link):
        resp = requests.get(file_link)
        if not resp.ok:
            raise ValueError(resp)
        return resp.content


def new_blob_storage():
    provider = os.environ.get("BLOB_STORAGE_PROVIDER", "REMOTE_LINK")
    if provider == "REMOTE_LINK":
        return RemoteLinkBlobStorageWriter()
    elif provider == "S3":
        return S3BlobStorageWriter(
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            bucket_name=os.environ["S3_BUCKET_NAME"],
            path_prefix=os.environ["S3_PATH_PREFIX"],
            verify=os.environ.get("S3_VERIFY", "false") == "true"
        )
    else:
        raise ValueError(f"Unknown blobStorageProvider = '{provider}'")


class TaskCreator:
    def __init__(self):
        self._r = redis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], db=0)
        self._q = os.environ["REDIS_QUEUE"]

    def create(self, img_data, chart_id, message_id):
        self._r.rpush(self._q, json.dumps({
            "img": img_data,
            "chat_id": chart_id,
            "message_id": message_id,
        }))


class BotCommandDispatcher:
    def __init__(self,
                 task_creator: TaskCreator,
                 blob_storage: BlobStorageWriter):
        self._task_creator = task_creator
        self._blob_storage = blob_storage

    def start(self, update: Update, context: CallbackContext) -> None:
        user = update.effective_user
        update.message.reply_markdown_v2(
            fr'Hi {user.mention_markdown_v2()}\! Upload picture with faces.',
            reply_markup=ForceReply(selective=True),
        )

    def help_command(self, update: Update, context: CallbackContext) -> None:
        update.message.reply_text('Upload image as picture. Bot will blur all detected faces.')

    def receive_callback(self, update: Update, context: CallbackContext) -> None:
        pass

    def echo(self, update: Update, context: CallbackContext) -> None:
        if len(update.message.photo) > 0:
            self._task_creator.create(
                self._blob_storage.save(context.bot.get_file(update.message.photo[-1].file_id).file_path),
                update.message.chat_id,
                update.message.message_id,
            )
            return
        if (update.message.effective_attachment is not None
                and update.message.effective_attachment.thumb is not None):
            self._task_creator.create(
                context.bot.get_file(update.message.effective_attachment.thumb.file_id).file_path,
                update.message.chat_id,
                update.message.message_id,
            )


def main() -> None:
    bot_command_dispatcher = BotCommandDispatcher(TaskCreator(), new_blob_storage())
    updater = Updater(os.environ["TELEGRAM_BOT_TOKEN"])
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", bot_command_dispatcher.start))
    dispatcher.add_handler(CommandHandler("help", bot_command_dispatcher.help_command))
    dispatcher.add_handler(MessageHandler(~Filters.command, bot_command_dispatcher.echo))
    dispatcher.add_handler(CallbackQueryHandler(bot_command_dispatcher.receive_callback))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
