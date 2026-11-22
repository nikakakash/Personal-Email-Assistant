"""Exception → JSON response mapping. Responses carry fixed codes, never exception text.

- `HTTPException(detail="some_code")` → `{"error": "some_code"}`
- request validation → 422 `{"error": "validation_error", "errors": [{loc, msg, type}]}`
  (the submitted input is not echoed back: it may contain the passphrase)
- `ProposalNotFound` / `ReviewNotFound` → 404 `{"error": "not_found"}`
- `InvalidEdit` → 422 `{"error": "invalid_edit", "errors": [...]}`
- `DecisionConflict` → 409 `{"error": "conflict", "reason": ...}`
- anything else → 500 `{"error": "internal_error"}` (the traceback goes to the server log)
"""

import logging
import re

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.actions.decisions import DecisionConflict, InvalidEdit, ProposalNotFound
from app.actions.reviews import ReviewNotFound

log = logging.getLogger(__name__)

_CODE = re.compile(r"[a-z][a-z0-9_]*")
_DEFAULT_CODES = {
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    429: "too_many_attempts",
    503: "unavailable",
}


def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    detail = exc.detail if isinstance(exc.detail, str) and _CODE.fullmatch(exc.detail) else None
    code = detail or _DEFAULT_CODES.get(exc.status_code, "http_error")
    return JSONResponse({"error": code}, status_code=exc.status_code, headers=exc.headers)


def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {"loc": list(e.get("loc", ())), "msg": str(e.get("msg", "")), "type": e.get("type")}
        for e in exc.errors()
    ]
    return JSONResponse({"error": "validation_error", "errors": errors}, status_code=422)


def _not_found(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"error": "not_found"}, status_code=404)


def _invalid_edit(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, InvalidEdit)
    return JSONResponse({"error": "invalid_edit", "errors": exc.errors}, status_code=422)


def _conflict(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DecisionConflict)
    return JSONResponse({"error": "conflict", "reason": exc.reason}, status_code=409)


def _internal(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse({"error": "internal_error"}, status_code=500)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(ProposalNotFound, _not_found)
    app.add_exception_handler(ReviewNotFound, _not_found)
    app.add_exception_handler(InvalidEdit, _invalid_edit)
    app.add_exception_handler(DecisionConflict, _conflict)
    app.add_exception_handler(Exception, _internal)
