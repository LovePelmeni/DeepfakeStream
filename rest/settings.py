import fastapi 
import os 
import logging 
from rest.controllers import predict_human_deepfake, healthcheck
from fastapi.middleware import cors

logger = logging.getLogger("startup_logger")
file_handler = logging.FileHandler(filename="startup_log.log")
logger.addHandler(file_handler)

VERSION = os.environ.get("VERSION", "1.0.0")
ALLOWED_METHODS = os.environ.get("ALLOWED_METHODS", "*")
ALLOWED_HEADERS = os.environ.get("ALLOWED_HEADERS", "*")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

application = fastapi.FastAPI(version=VERSION)

try:
    application.add_middleware(
        middleware_class=cors.CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_headers=ALLOWED_HEADERS,
        allow_methods=ALLOWED_METHODS,
    )
except(Exception) as err:
    logger.error("failed to add core middlewares for protecting application")

try:
    application.add_api_route(
        path='/predict/human/deepfake/',
        endpoint=predict_human_deepfake,
        methods=['POST']
    )
    application.add_api_route(
        path='/healthcheck/',
        endpoint=healthcheck,
        methods=['GET']
    )
except(Exception) as err:
    logger.fatal("failed to start ASGI Server application")
    raise SystemExit("Failed to start application, check logs.")
