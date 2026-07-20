from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from regulatory_compliance.core.config import settings
from regulatory_compliance.routes.upload_routes import router as upload_router
from regulatory_compliance.routes.query_routes import router as query_router
from contextlib import asynccontextmanager
from regulatory_compliance.core.db import Database
from regulatory_compliance.utils.exceptions import (
    InvalidFileException,
    DocumentNotFoundException,
    DatabaseException,
    LLMException,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Application startup completed")

    yield

    print("Application shutdown completed")


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)
app.include_router(upload_router)
app.include_router(query_router)


@app.exception_handler(InvalidFileException)
async def invalid_file_handler(request: Request, exc: InvalidFileException):

    return JSONResponse(
        status_code=400, content={"success": False, "message": exc.message}
    )


@app.exception_handler(DocumentNotFoundException)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundException):

    return JSONResponse(
        status_code=404, content={"success": False, "message": exc.message}
    )


@app.exception_handler(DatabaseException)
async def database_handler(request: Request, exc: DatabaseException):

    return JSONResponse(
        status_code=500, content={"success": False, "message": exc.message}
    )


@app.exception_handler(LLMException)
async def llm_handler(request: Request, exc: LLMException):

    return JSONResponse(
        status_code=500, content={"success": False, "message": exc.message}
    )
