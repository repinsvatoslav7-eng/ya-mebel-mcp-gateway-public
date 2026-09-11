import os
import time
import uuid
import hmac
import hashlib
import json
import logging
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP

SITE_ID = os.environ.get("ROSTOV_SITE_ID", "rostov-test").strip()
SITE_URL = os.environ.get("ROSTOV_SITE_URL", "https://rostov.yamebel.pro").rstrip("/")
SHARED_SECRET = os.environ.get("ROSTOV_SHARED_SECRET", "")
REQUEST_TIMEOUT = float(os.environ.get("YMB_REQUEST_TIMEOUT", "20"))
PORT = int(os.environ.get("PORT", "10000"))

if not SHARED_SECRET:
    raise RuntimeError("ROSTOV_SHARED_SECRET is not configured.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ya-mebel-mcp-gateway")

mcp = FastMCP(
    "Я Мебель WordPress Bridge",
    stateless_http=True,
    json_response=True,
    host="0.0.0.0",
    port=PORT,
)

NS = "/ya-mebel-bridge/v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signed_headers(method: str, route: str, body: bytes = b"") -> dict[str, str]:
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    canonical = "\n".join([
        method.upper(),
        route,
        timestamp,
        nonce,
        _sha256_bytes(body),
    ])
    signature = hmac.new(
        SHARED_SECRET.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "X-YMB-Site": SITE_ID,
        "X-YMB-Timestamp": timestamp,
        "X-YMB-Nonce": nonce,
        "X-YMB-Signature": signature,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _safe_response_text(response: httpx.Response, limit: int = 4000) -> str:
    try:
        text = response.text
    except Exception:
        return "<unable to decode response body>"
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


async def _wp(
    method: str,
    route: str,
    *,
    params: Optional[dict[str, Any]] = None,
    payload: Optional[dict[str, Any]] = None,
    signed: bool = True,
) -> dict[str, Any]:
    method = method.upper()
    body = b""
    if payload is not None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    headers = {"Accept": "application/json"}
    if signed:
        headers = _signed_headers(method, route, body)
    elif payload is not None:
        headers["Content-Type"] = "application/json"

    url = f"{SITE_URL}/wp-json{route}"

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.request(
                method,
                url,
                params=params,
                content=body if payload is not None else None,
                headers=headers,
            )
    except httpx.TimeoutException as exc:
        logger.exception(
            "Timeout calling WordPress: method=%s route=%s timeout=%s",
            method,
            route,
            REQUEST_TIMEOUT,
        )
        raise RuntimeError(
            f"WordPress request timed out after {REQUEST_TIMEOUT}s: {method} {route}"
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception(
            "HTTP transport error calling WordPress: method=%s route=%s",
            method,
            route,
        )
        raise RuntimeError(
            f"WordPress transport error: {method} {route}: {exc}"
        ) from exc

    if response.status_code >= 400:
        response_text = _safe_response_text(response)
        logger.error(
            "WordPress returned HTTP %s: method=%s route=%s body=%s",
            response.status_code,
            method,
            route,
            response_text,
        )
        raise RuntimeError(
            f"WordPress returned HTTP {response.status_code} for {method} {route}: "
            f"{response_text}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        response_text = _safe_response_text(response)
        logger.error(
            "WordPress returned invalid JSON: method=%s route=%s status=%s body=%s",
            method,
            route,
            response.status_code,
            response_text,
        )
        raise RuntimeError(
            f"WordPress returned invalid JSON for {method} {route}: {response_text}"
        ) from exc

    return {
        "status_code": response.status_code,
        "data": data,
    }


@mcp.tool()
async def bridge_health() -> dict[str, Any]:
    """Проверить публичный статус WordPress Bridge без выполнения изменений."""
    return await _wp("GET", f"{NS}/health", signed=False)


@mcp.tool()
async def get_site_info() -> dict[str, Any]:
    """Прочитать основные сведения WordPress, PHP и активной темы."""
    result = await _wp("GET", f"{NS}/site/info")
    return result["data"]


@mcp.tool()
async def list_plugins() -> dict[str, Any]:
    """Получить список плагинов WordPress и их статус."""
    result = await _wp("GET", f"{NS}/plugins")
    return result["data"]


@mcp.tool()
async def list_theme_files() -> dict[str, Any]:
    """Получить список разрешённых файлов активной/родительской темы."""
    result = await _wp("GET", f"{NS}/theme/files")
    return result["data"]


@mcp.tool()
async def read_theme_file(path: str) -> dict[str, Any]:
    """Прочитать файл темы. Пример path: header.php или assets/css/main.css."""
    result = await _wp("GET", f"{NS}/theme/file", params={"path": path})
    return result["data"]


@mcp.tool()
async def list_content(
    post_type: str = "any",
    search: str = "",
    page: int = 1,
    per_page: int = 20,
) -> dict[str, Any]:
    """Найти/получить страницы и записи WordPress."""
    params = {
        "post_type": post_type,
        "search": search,
        "page": page,
        "per_page": per_page,
    }
    result = await _wp("GET", f"{NS}/content", params=params)
    return result["data"]


@mcp.tool()
async def get_content(content_id: int) -> dict[str, Any]:
    """Прочитать конкретную страницу/запись, включая ACF при наличии."""
    result = await _wp("GET", f"{NS}/content/{content_id}")
    return result["data"]


@mcp.tool()
async def get_theme_mods() -> dict[str, Any]:
    """Прочитать безопасные theme_mod настройки активной темы."""
    result = await _wp("GET", f"{NS}/theme-mods")
    return result["data"]


@mcp.tool()
async def get_menus() -> dict[str, Any]:
    """Прочитать меню WordPress и пункты меню."""
    result = await _wp("GET", f"{NS}/menus")
    return result["data"]


@mcp.tool()
async def get_audit_log(limit: int = 50) -> dict[str, Any]:
    """Прочитать журнал действий Я Мебель Bridge."""
    result = await _wp("GET", f"{NS}/audit", params={"limit": limit})
    return result["data"]


@mcp.tool()
async def list_backups(limit: int = 50) -> dict[str, Any]:
    """Получить список резервных копий, созданных Bridge."""
    result = await _wp("GET", f"{NS}/backups", params={"limit": limit})
    return result["data"]


@mcp.tool()
async def preview_theme_file_update(
    path: str,
    expected_sha256: str,
    content_base64: str,
) -> dict[str, Any]:
    """ТОЛЬКО PREVIEW изменения существующего файла темы. Ничего не записывает."""
    payload = {
        "path": path,
        "expected_sha256": expected_sha256,
        "content_base64": content_base64,
    }
    result = await _wp("POST", f"{NS}/theme/file/preview", payload=payload)
    return result["data"]


@mcp.tool()
async def preview_theme_file_patch(
    path: str,
    expected_sha256: str,
    search_base64: str,
    replacement_base64: str,
    expected_occurrences: int = 1,
) -> dict[str, Any]:
    """ТОЛЬКО PREVIEW точечной замены фрагмента файла темы. Ничего не записывает."""
    payload = {
        "path": path,
        "expected_sha256": expected_sha256,
        "search_base64": search_base64,
        "replace_base64": replacement_base64,
        "expected_occurrences": expected_occurrences,
    }
    result = await _wp("POST", f"{NS}/theme/file/patch/preview", payload=payload)
    return result["data"]


@mcp.tool()
async def apply_theme_file_update(
    plan_id: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """ПРИМЕНИТЬ ранее подготовленное изменение файла темы."""
    payload = {
        "plan_id": plan_id,
        "confirmation_token": confirmation_token,
    }
    result = await _wp("POST", f"{NS}/theme/file/apply", payload=payload)
    return result["data"]


@mcp.tool()
async def preview_about_page_update(
    expected_sha256: str,
    operations: list[str],
) -> dict[str, Any]:
    """
    Подготовить безопасное изменение страницы «О нас» только из разрешённого
    набора операций. Произвольный PHP/CSS и произвольные пути не принимаются.
    Доступные операции сейчас: advantages_approved_v1, slogan_style_v1.
    Ничего не записывает.
    """
    payload = {
        "expected_sha256": expected_sha256,
        "operations": operations,
    }
    result = await _wp("POST", f"{NS}/about/page/preview", payload=payload)
    return result["data"]


@mcp.tool()
async def apply_about_page_update(
    plan_id: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Применить ранее подготовленный ограниченный план страницы «О нас»."""
    payload = {
        "plan_id": plan_id,
        "confirmation_token": confirmation_token,
    }
    result = await _wp("POST", f"{NS}/about/page/apply", payload=payload)
    return result["data"]


@mcp.tool()
async def preview_cache_clear() -> dict[str, Any]:
    """ТОЛЬКО PREVIEW очистки кэша. Ничего не очищает."""
    result = await _wp("POST", f"{NS}/cache/clear/preview", payload={})
    return result["data"]


@mcp.tool()
async def apply_cache_clear(
    plan_id: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Очистить кэш по ранее созданному preview."""
    payload = {
        "plan_id": plan_id,
        "confirmation_token": confirmation_token,
    }
    result = await _wp("POST", f"{NS}/cache/clear/apply", payload=payload)
    return result["data"]


@mcp.tool()
async def preview_backup_restore(backup_id: str) -> dict[str, Any]:
    """ТОЛЬКО PREVIEW восстановления резервной копии. Ничего не восстанавливает."""
    result = await _wp(
        "POST",
        f"{NS}/backup/restore/preview",
        payload={"backup_id": backup_id},
    )
    return result["data"]


@mcp.tool()
async def apply_backup_restore(
    plan_id: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Восстановить backup по preview."""
    payload = {
        "plan_id": plan_id,
        "confirmation_token": confirmation_token,
    }
    result = await _wp("POST", f"{NS}/backup/restore/apply", payload=payload)
    return result["data"]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
