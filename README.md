# Project "Photo Anonymization"

Telegram bot "Photo Anonymization". You upload a photo, the faces are programmatically blured.

![Image](docs/images/v0-prototype.png)

## Authors

- [Mark Andreev](https://github.com/mrk-andreev)
- [Yuliya Fokina](https://github.com/foookinaaa)

## How to deploy

### Local e2e

- Redis
- app_receiver
- app_predictor

```
cd envs/local
export TELEGRAM_BOT_TOKEN="<TELEGRAM_BOT_TOKEN>"
docker-compose -f docker-compose-app.yaml up --build
```

## Components

### experiments

### app_predictor (src/app_predictor)

Envs:

- REDIS_QUEUE
- REDIS_HOST
- REDIS_PORT
- TELEGRAM_BOT_TOKEN

### app_receiver (src/app_receiver)

Envs:

- REDIS_QUEUE
- REDIS_HOST
- REDIS_PORT
- TELEGRAM_BOT_TOKEN
