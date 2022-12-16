import abc
import hashlib
import json
import logging
import os
import uuid

import boto3
import redis
import requests
from botocore.exceptions import ClientError
from telegram import ForceReply, Update
from telegram.ext import (
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    Filters,
    MessageHandler,
    Updater,
)
from prometheus_client import start_http_server, Summary


# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


FETCH_IMAGE_TIME = Summary('fetch_image_request_processing_seconds', 'Time spent processing request')
FETCH_FEEDBACK_TIME = Summary('fetch_feedback_request_processing_seconds', 'Time spent processing request')
FETCH_HELP_TIME = Summary('fetch_help_request_processing_seconds', 'Time spent processing request')
FETCH_START_TIME = Summary('fetch_start_request_processing_seconds', 'Time spent processing request')


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
    def __init__(self, endpoint_url, access_key_id, secret_access_key, bucket_name, path_prefix,
                 verify=False):
        self._s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,  # 'https://<minio>:9000'
            aws_access_key_id=access_key_id if access_key_id else None,
            aws_secret_access_key=secret_access_key if secret_access_key else None,
            aws_session_token=None,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=verify,
        )
        self._bucket_name = bucket_name
        self._path_prefix = path_prefix

    def save(self, file_link):
        obj_name = f"{self._path_prefix}/{uuid.uuid4()}.jpg"
        obj = self._s3.Object(self._bucket_name, obj_name)
        obj.put(Body=self._load_content(file_link))
        return {
            "storage": "S3",
            "href": {"bucket": self._bucket_name, "obj_name": obj_name},
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
    if provider == "S3":
        return S3BlobStorageWriter(
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            bucket_name=os.environ["S3_BUCKET_NAME"],
            path_prefix=os.environ["S3_PATH_PREFIX"],
            verify=os.environ.get("S3_VERIFY", "false") == "true",
        )

    raise ValueError(f"Unknown blobStorageProvider = '{provider}'")


class FeedbackStorage(abc.ABC):
    @abc.abstractmethod
    def save(self, message_id, action):
        pass


class NoneFeedbackStorage(FeedbackStorage):
    def save(self, message_id, action):
        # ignore
        pass


class S3FeedbackStorage(FeedbackStorage):
    def __init__(self, endpoint_url, access_key_id, secret_access_key, bucket_name, path_prefix,
                 verify=False):
        self._s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,  # 'https://<minio>:9000'
            aws_access_key_id=access_key_id if access_key_id else None,
            aws_secret_access_key=secret_access_key if secret_access_key else None,
            aws_session_token=None,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=verify,
        )

        self._bucket_name = bucket_name
        self._path_prefix = path_prefix

    def _is_exists(self, obj_name):
        obj = self._s3.Object(self._bucket_name, obj_name)
        try:
            m = obj.metadata
            return m is not None
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "-") == "404":
                return False
            raise e

    @classmethod
    def _evaluate_hash(cls, name):
        return hashlib.md5(name.encode("utf-8")).hexdigest()

    def save(self, message_id, action):
        obj_name = f"{self._path_prefix}/{message_id}-{self._evaluate_hash(action)}.json"
        if self._is_exists(obj_name):
            return

        obj = self._s3.Object(self._bucket_name, obj_name)
        obj.put(Body=json.dumps({"message_id": message_id, "action": action}).encode())


def new_feedback_storage():
    provider = os.environ.get("FEEDBACK_STORAGE_PROVIDER", "NONE")
    if provider == "NONE":
        return NoneFeedbackStorage()
    if provider == "S3":
        return S3FeedbackStorage(
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            bucket_name=os.environ["S3_BUCKET_NAME"],
            path_prefix=os.environ["S3_FEEDBACK_PATH_PREFIX"],
            verify=os.environ.get("S3_VERIFY", "false") == "true",
        )

    raise ValueError(f"Unknown feedbackStorageProvider = '{provider}'")


class TaskCreator:
    def __init__(self, host, port, queue, username, password):
        self._r = redis.Redis(host=host, port=port, db=0, username=username, password=password)
        self._q = queue
        self._r.ping()

    def create(self, img_data, chart_id, message_id):
        self._r.rpush(
            self._q,
            json.dumps(
                {
                    "img": img_data,
                    "chat_id": chart_id,
                    "message_id": message_id,
                }
            ),
        )


def init_task_creator():
    return TaskCreator(
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        queue=os.environ["REDIS_QUEUE"],
        username=os.environ.get("REDIS_USERNAME"),
        password=os.environ.get("REDIS_PASSWORD"),
    )


class BotCommandDispatcher:
    def __init__(self, task_creator: TaskCreator, blob_storage: BlobStorageWriter, feedback_storage: FeedbackStorage):
        self._task_creator = task_creator
        self._blob_storage = blob_storage
        self._feedback_storage = feedback_storage

    @classmethod
    @FETCH_START_TIME.time()
    def start(cls, update: Update, _: CallbackContext) -> None:
        user = update.effective_user
        update.message.reply_markdown_v2(
            rf"Hi {user.mention_markdown_v2()}\! Upload picture with faces.",
            reply_markup=ForceReply(selective=True),
        )

    @classmethod
    @FETCH_HELP_TIME.time()
    def help_command(cls, update: Update, _: CallbackContext) -> None:
        update.message.reply_text("Upload image as picture. Bot will blur all detected faces.")

    @FETCH_FEEDBACK_TIME.time()
    def receive_callback(self, update: Update, _: CallbackContext) -> None:
        message_id = update.callback_query.message.reply_to_message.message_id
        action = update.callback_query.data

        self._feedback_storage.save(message_id, action)

    @FETCH_IMAGE_TIME.time()
    def echo(self, update: Update, context: CallbackContext) -> None:
        if len(update.message.photo) > 0:
            self._task_creator.create(
                self._blob_storage.save(context.bot.get_file(update.message.photo[-1].file_id).file_path),
                update.message.chat_id,
                update.message.message_id,
            )
            return
        if update.message.effective_attachment is not None and update.message.effective_attachment.thumb is not None:
            self._task_creator.create(
                context.bot.get_file(update.message.effective_attachment.thumb.file_id).file_path,
                update.message.chat_id,
                update.message.message_id,
            )


def main() -> None:
    start_http_server(8001)
    bot_command_dispatcher = BotCommandDispatcher(init_task_creator(), new_blob_storage(), new_feedback_storage())
    updater = Updater(os.environ["TELEGRAM_BOT_TOKEN"])
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", bot_command_dispatcher.start))
    dispatcher.add_handler(CommandHandler("help", bot_command_dispatcher.help_command))
    dispatcher.add_handler(MessageHandler(~Filters.command, bot_command_dispatcher.echo))
    dispatcher.add_handler(CallbackQueryHandler(bot_command_dispatcher.receive_callback))
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
