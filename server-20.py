import os
import re
import base64
import time
import uuid
import hmac
import hashlib
import json
import logging
import tempfile
import threading
from pathlib import PurePosixPath
from typing import Annotated, Any, Optional

import httpx
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

SITE_ID = os.environ.get("ROSTOV_SITE_ID", "rostov-test").strip()
SITE_URL = os.environ.get("ROSTOV_SITE_URL", "https://rostov.yamebel.pro").rstrip("/")
SHARED_SECRET = os.environ.get("ROSTOV_SHARED_SECRET", "")
REQUEST_TIMEOUT = float(os.environ.get("YMB_REQUEST_TIMEOUT", "20"))
PORT = int(os.environ.get("PORT", "10000"))

YANDEX_MARKETING_CLIENT_ID = os.environ.get("YANDEX_MARKETING_CLIENT_ID", "").strip()
YANDEX_MARKETING_OAUTH_TOKEN = os.environ.get("YANDEX_MARKETING_OAUTH_TOKEN", "").strip()
YANDEX_DIRECT_CLIENT_LOGIN = os.environ.get("YANDEX_DIRECT_CLIENT_LOGIN", "").strip()

METRIKA_BASE = "https://api-metrika.yandex.net"
DIRECT_BASE = "https://api.direct.yandex.com/json/v5"
DIRECT_REPORTS_URL = "https://api.direct.yandex.com/json/v5/reports"

# Marketing writes are intentionally two-step. Preview creates an immutable,
# short-lived in-memory plan; apply requires the exact plan id + confirmation
# token. Render restart invalidates pending plans by design.
MARKETING_PLAN_TTL_SECONDS = int(os.environ.get("YMB_MARKETING_PLAN_TTL_SECONDS", "900"))
MARKETING_PLAN_HISTORY_SECONDS = int(os.environ.get("YMB_MARKETING_PLAN_HISTORY_SECONDS", "86400"))
MARKETING_PLAN_STORE_PATH = os.environ.get(
    "YMB_MARKETING_PLAN_STORE_PATH",
    os.path.join(tempfile.gettempdir(), "ya_mebel_marketing_plans.json"),
)
_MARKETING_PLAN_LOCK = threading.RLock()

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

ThemeRelativePath = Annotated[
    str,
    Field(
        min_length=1,
        max_length=240,
        description=(
            "Relative path of an existing text/source file inside the active or parent "
            "WordPress theme, for example 'template-o-nas.php' or 'assets/css/main.css'. "
            "Absolute paths, '..' traversal, backslashes, NUL bytes, WordPress secrets, "
            "and files outside the allowed theme roots are not permitted."
        ),
    ),
]

_ALLOWED_THEME_TEXT_EXTENSIONS = {
    ".php", ".css", ".js", ".json", ".txt", ".md", ".xml", ".svg",
}
_BLOCKED_THEME_BASENAMES = {
    "wp-config.php",
    ".env",
    ".htaccess",
    "php.ini",
    "user.ini",
    # This site's functions.php currently contains hard-coded credentials/tokens.
    # Block it from MCP reads until those secrets have been moved and rotated.
    "functions.php",
}

# Keep each MCP result comfortably below large-tool-response limits.
# WordPress may return up to 1 MiB to the gateway; the gateway exposes it to the
# MCP client in explicit, lossless line windows instead of silently truncating.
DEFAULT_READ_MAX_LINES = 5000
MAX_READ_MAX_LINES = 5000
MAX_READ_CHUNK_BYTES = 750 * 1024


def _validate_theme_relative_path(path: str) -> str:
    """Defense-in-depth validation before the WordPress REST request."""
    candidate = path.strip()

    if not candidate:
        raise ValueError("Theme file path must not be empty.")
    if "\x00" in candidate:
        raise ValueError("NUL bytes are not allowed in theme file paths.")
    if "\\" in candidate:
        raise ValueError("Backslashes are not allowed; use a POSIX-style relative path.")
    if candidate.startswith("/"):
        raise ValueError("Absolute paths are not allowed.")

    posix_path = PurePosixPath(candidate)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError("Path traversal is not allowed.")

    basename = posix_path.name.lower()
    if basename in _BLOCKED_THEME_BASENAMES or basename.startswith(".env"):
        raise ValueError("This file is not readable through the theme-file tool.")

    suffix = posix_path.suffix.lower()
    if suffix not in _ALLOWED_THEME_TEXT_EXTENSIONS:
        raise ValueError(
            "Only allow-listed text/source file types from the theme may be read."
        )

    return posix_path.as_posix()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_sha256_hex(value: str) -> str:
    """Normalize and validate an externally supplied SHA-256 digest."""
    candidate = value.strip().lower()
    if len(candidate) != 64 or any(ch not in "0123456789abcdef" for ch in candidate):
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters.")
    return candidate


def _bounded_text_result(
    data: dict[str, Any],
    *,
    start_line: int = 1,
    max_lines: int = DEFAULT_READ_MAX_LINES,
) -> dict[str, Any]:
    """Return a lossless, byte-bounded line window from a theme-file response."""
    content = data.get("content")
    if not isinstance(content, str):
        raise RuntimeError("WordPress theme-file response does not contain text content.")

    if start_line < 1:
        raise ValueError("start_line must be >= 1.")
    if max_lines < 1 or max_lines > MAX_READ_MAX_LINES:
        raise ValueError(f"max_lines must be between 1 and {MAX_READ_MAX_LINES}.")

    raw = content.encode("utf-8")
    declared_size = data.get("size")
    if not isinstance(declared_size, int):
        declared_size = len(raw)
    if declared_size != len(raw):
        raise RuntimeError(
            f"WordPress size mismatch: endpoint declared {declared_size} bytes, "
            f"gateway decoded {len(raw)} UTF-8 bytes."
        )

    declared_sha256 = str(data.get("sha256") or "").lower()
    actual_sha256 = _sha256_bytes(raw)
    if declared_sha256 and not hmac.compare_digest(declared_sha256, actual_sha256):
        raise RuntimeError("WordPress SHA-256 mismatch while reading theme file.")

    # keepends=True guarantees that concatenating sequential chunks reproduces
    # the exact UTF-8 source byte-for-byte. The MCP-facing chunk is capped by
    # both line count and bytes, so a large source cannot overflow a tool result.
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        if start_line != 1:
            raise ValueError("start_line is beyond the end of the file.")
        chunk = ""
        end_line = 0
        returned_bytes = 0
        has_more = False
        next_start_line = None
    else:
        if start_line > total_lines:
            raise ValueError(
                f"start_line {start_line} is beyond total_lines {total_lines}."
            )
        start_index = start_line - 1
        selected: list[str] = []
        returned_bytes = 0
        end_index = start_index
        hard_end = min(start_index + max_lines, total_lines)
        for idx in range(start_index, hard_end):
            line = lines[idx]
            line_bytes = len(line.encode("utf-8"))
            if line_bytes > MAX_READ_CHUNK_BYTES:
                raise RuntimeError(
                    f"Source line {idx + 1} is {line_bytes} bytes, exceeding the "
                    f"{MAX_READ_CHUNK_BYTES}-byte per-result safety cap. "
                    "Use a byte-range reader for this exceptional file."
                )
            if selected and returned_bytes + line_bytes > MAX_READ_CHUNK_BYTES:
                break
            selected.append(line)
            returned_bytes += line_bytes
            end_index = idx + 1
        chunk = "".join(selected)
        end_line = end_index
        has_more = end_index < total_lines
        next_start_line = end_index + 1 if has_more else None

    return {
        "ok": bool(data.get("ok", True)),
        "site_id": data.get("site_id", SITE_ID),
        "warnings": data.get("warnings", []),
        "path": data.get("path"),
        "sha256": actual_sha256,
        "size": declared_size,
        "encoding": "utf-8",
        "total_lines": total_lines,
        "start_line": start_line,
        "end_line": end_line,
        "returned_bytes": returned_bytes,
        "content": chunk,
        "has_more": has_more,
        "next_start_line": next_start_line,
    }


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


def _marketing_auth_headers() -> dict[str, str]:
    if not YANDEX_MARKETING_OAUTH_TOKEN:
        raise RuntimeError("YANDEX_MARKETING_OAUTH_TOKEN is not configured.")
    return {
        "Authorization": f"OAuth {YANDEX_MARKETING_OAUTH_TOKEN}",
        "Accept": "application/json",
    }


def _direct_auth_headers() -> dict[str, str]:
    if not YANDEX_MARKETING_OAUTH_TOKEN:
        raise RuntimeError("YANDEX_MARKETING_OAUTH_TOKEN is not configured.")
    return {
        "Authorization": f"Bearer {YANDEX_MARKETING_OAUTH_TOKEN}",
        "Accept": "application/json",
    }


def _direct_response_meta(response: httpx.Response) -> dict[str, Any]:
    """Safe Direct diagnostics. Never includes auth/request payload secrets."""
    return {
        "http_status": response.status_code,
        "request_id": response.headers.get("RequestId"),
        "units": response.headers.get("Units"),
        "units_used_login": response.headers.get("Units-Used-Login"),
    }


def _parse_direct_json(response: httpx.Response, *, operation: str) -> dict[str, Any]:
    response_text = _safe_response_text(response, limit=12000)
    meta = _direct_response_meta(response)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Yandex Direct HTTP error during {operation}: {meta}; body={response_text}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Yandex Direct returned non-JSON during {operation}: {meta}; body={response_text}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Yandex Direct returned unexpected JSON during {operation}: {meta}; body={response_text}"
        )
    if data.get("error"):
        raise RuntimeError(
            f"Yandex Direct API error during {operation}: {meta}; error={data['error']}"
        )
    return data


def _direct_action_issues(data: dict[str, Any]) -> dict[str, Any]:
    """Collect per-object Errors/Warnings from AddResults/UpdateResults/etc."""
    result = data.get("result")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if isinstance(result, dict):
        for result_key, value in result.items():
            if not isinstance(value, list):
                continue
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    continue
                if item.get("Errors"):
                    errors.append({"result_key": result_key, "index": index, "errors": item["Errors"]})
                if item.get("Warnings"):
                    warnings.append({"result_key": result_key, "index": index, "warnings": item["Warnings"]})
    return {"errors": errors, "warnings": warnings}


def _clean_api_path(path: str) -> str:
    candidate = str(path or "").strip()
    if not candidate.startswith("/"):
        candidate = "/" + candidate
    if "://" in candidate or "\\" in candidate or "\x00" in candidate:
        raise ValueError("Only a relative Yandex API path is allowed.")
    if ".." in PurePosixPath(candidate).parts:
        raise ValueError("Path traversal is not allowed.")
    return candidate


async def _yandex_http(
    method: str,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    payload: Optional[Any] = None,
    content: Optional[bytes] = None,
    content_type: Optional[str] = None,
) -> Any:
    headers = _marketing_auth_headers()
    if content_type:
        headers["Content-Type"] = content_type
    elif payload is not None:
        headers["Content-Type"] = "application/json"

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            response = await client.request(
                method.upper(),
                url,
                params=params,
                json=payload if content is None else None,
                content=content,
                headers=headers,
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Yandex API timeout after {REQUEST_TIMEOUT}s.") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Yandex API transport error: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Yandex API returned HTTP {response.status_code}: {_safe_response_text(response)}"
        )

    ctype = response.headers.get("content-type", "")
    if "json" in ctype:
        return response.json()
    return {
        "status_code": response.status_code,
        "content_type": ctype,
        "text": response.text,
        "headers": {
            k: v for k, v in response.headers.items()
            if k.lower().startswith("reports-") or k.lower() in {"retry-in"}
        },
    }



def _load_marketing_plan_store() -> dict[str, dict[str, Any]]:
    """Load durable short-lived marketing plans from a local atomic JSON store."""
    with _MARKETING_PLAN_LOCK:
        try:
            with open(MARKETING_PLAN_STORE_PATH, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.error("Marketing plan store read failed: %s", exc)
        return {}


def _save_marketing_plan_store(store: dict[str, dict[str, Any]]) -> None:
    """Atomic replace so preview/apply cannot observe a partially written store."""
    with _MARKETING_PLAN_LOCK:
        directory = os.path.dirname(MARKETING_PLAN_STORE_PATH) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".ymb-plans-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(store, fh, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, MARKETING_PLAN_STORE_PATH)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass


def _marketing_audit_event(plan: dict[str, Any], event: str, **data: Any) -> None:
    """Append safe diagnostics to the plan record; never store OAuth secrets/tokens."""
    events = plan.setdefault("events", [])
    events.append({
        "time": time.time(),
        "event": event,
        **data,
    })
    # Bound history per plan.
    if len(events) > 50:
        del events[:-50]


def _prune_marketing_plans(store: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, dict[str, Any]]:
    now = time.time()
    current = store if store is not None else _load_marketing_plan_store()
    changed = False
    for plan_id, plan in list(current.items()):
        terminal_at = float(plan.get("terminal_at") or 0)
        expires_at = float(plan.get("expires_at") or 0)
        if terminal_at and terminal_at + MARKETING_PLAN_HISTORY_SECONDS <= now:
            current.pop(plan_id, None)
            changed = True
        elif not terminal_at and expires_at + MARKETING_PLAN_HISTORY_SECONDS <= now:
            current.pop(plan_id, None)
            changed = True
    if changed:
        _save_marketing_plan_store(current)
    return current


def _make_marketing_plan(
    api: str,
    method: str,
    target: str,
    *,
    params: Optional[dict[str, Any]] = None,
    payload: Optional[Any] = None,
    content_base64: Optional[str] = None,
    content_type: Optional[str] = None,
) -> dict[str, Any]:
    method = method.upper()
    if method not in {"POST", "PUT", "DELETE"}:
        raise ValueError("Marketing write plan supports POST, PUT or DELETE only.")

    plan_id = uuid.uuid4().hex
    confirmation_token = uuid.uuid4().hex
    now = time.time()
    canonical = json.dumps(
        {
            "api": api,
            "method": method,
            "target": target,
            "params": params or {},
            "payload": payload,
            "content_base64": content_base64,
            "content_type": content_type,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    plan = {
        "plan_id": plan_id,
        "confirmation_token_hash": _sha256_bytes(confirmation_token.encode("utf-8")),
        "api": api,
        "method": method,
        "target": target,
        "params": params or {},
        "payload": payload,
        "content_base64": content_base64,
        "content_type": content_type,
        "created_at": now,
        "expires_at": now + MARKETING_PLAN_TTL_SECONDS,
        "sha256": _sha256_bytes(canonical.encode("utf-8")),
        "state": "pending",
        "used": False,
        "terminal_at": None,
        "result_summary": None,
        "events": [],
    }
    _marketing_audit_event(plan, "preview_created")
    with _MARKETING_PLAN_LOCK:
        store = _prune_marketing_plans(_load_marketing_plan_store())
        store[plan_id] = plan
        _save_marketing_plan_store(store)

    return {
        "ok": True,
        "preview_only": True,
        "plan_id": plan_id,
        "confirmation_token": confirmation_token,
        "expires_in_seconds": MARKETING_PLAN_TTL_SECONDS,
        "operation_sha256": plan["sha256"],
        "api": api,
        "method": method,
        "target": target,
        "params": params or {},
        "payload": payload,
        "content_type": content_type,
        "requires_explicit_user_confirmation": True,
        "plan_store": "durable_atomic_file",
    }


async def _execute_marketing_plan(plan: dict[str, Any]) -> Any:
    api = plan["api"]
    method = plan["method"]
    params = plan.get("params") or None
    payload = plan.get("payload")
    content = None
    if plan.get("content_base64") is not None:
        try:
            content = base64.b64decode(plan["content_base64"], validate=True)
        except Exception as exc:
            raise ValueError("Invalid strict Base64 content in marketing plan.") from exc

    if api == "metrika":
        url = METRIKA_BASE + _clean_api_path(plan["target"])
        return await _yandex_http(
            method, url, params=params, payload=payload, content=content,
            content_type=plan.get("content_type"),
        )

    if api == "direct":
        if not YANDEX_DIRECT_CLIENT_LOGIN:
            raise RuntimeError("YANDEX_DIRECT_CLIENT_LOGIN is required for confirmed Direct writes.")
        service = str(plan["target"]).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,80}", service):
            raise ValueError("Invalid Direct service name.")
        headers = _direct_auth_headers()
        headers["Content-Type"] = "application/json; charset=utf-8"
        headers["Accept-Language"] = "ru"
        if YANDEX_DIRECT_CLIENT_LOGIN:
            headers["Client-Login"] = YANDEX_DIRECT_CLIENT_LOGIN
        body = plan.get("payload")
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            response = await client.post(f"{DIRECT_BASE}/{service}", headers=headers, json=body)
        meta = _direct_response_meta(response)
        response_text = _safe_response_text(response, limit=12000)
        logger.info(
            "Yandex Direct write response: service=%s method=%s meta=%s body=%s",
            service, (body or {}).get("method"), meta, response_text,
        )
        data = _parse_direct_json(
            response, operation=f"{service}.{(body or {}).get('method', 'unknown')}"
        )
        issues = _direct_action_issues(data)
        if issues["errors"]:
            # HTTP 200 can still mean that one or more requested objects were rejected.
            # Treat that as a failed apply so callers never see a false success.
            logger.error("Yandex Direct item errors: meta=%s errors=%s", meta, issues["errors"])
            raise RuntimeError(
                f"Yandex Direct rejected one or more objects: {meta}; errors={issues['errors']}"
            )
        if issues["warnings"]:
            logger.warning("Yandex Direct item warnings: meta=%s warnings=%s", meta, issues["warnings"])
        return {
            "ok": True,
            **meta,
            "warnings": issues["warnings"],
            "direct_response": data,
        }

    raise ValueError("Unknown marketing API.")


@mcp.tool(
    title="Yandex Marketing API status",
    description="Read-only check of Marketing API configuration. Never exposes OAuth secrets.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def marketing_api_status() -> dict[str, Any]:
    return {
        "ok": True,
        "server_revision": "v5.4-rsya-excluded-sites",
        "oauth_token_configured": bool(YANDEX_MARKETING_OAUTH_TOKEN),
        "client_id_configured": bool(YANDEX_MARKETING_CLIENT_ID),
        "direct_client_login_configured": bool(YANDEX_DIRECT_CLIENT_LOGIN),
        "write_policy": "fixed predefined operations -> preview -> explicit confirmation -> fixed apply",
        "plan_ttl_seconds": MARKETING_PLAN_TTL_SECONDS,
        "plan_store": "durable_atomic_file",
        "plan_store_path_configured": bool(MARKETING_PLAN_STORE_PATH),
        "direct_apply_tool": "apply_rostov_kitchen_launch_operation",
        "direct_write_capabilities": [
            "rostov_kitchen_launch_fixed_operations",
            "rostov_campaign_negatives_714564245",
            "rostov_rsya_excluded_sites_714590883",
        ],
        "rostov_rsya_excluded_sites_tools": [
            "get_rostov_rsya_excluded_sites",
            "preview_set_rostov_rsya_excluded_sites",
            "preview_add_rostov_rsya_excluded_sites",
            "preview_remove_rostov_rsya_excluded_sites",
            "apply_set_rostov_rsya_excluded_sites",
        ],
        "metrika_apply_tool": "no public generic write; fixed operations only",
    }


@mcp.tool(
    title="Metrica read",
    description=(
        "READ-ONLY universal Yandex Metrica REST reader. Supports Management API, "
        "Reporting API and Logs API GET resources by relative path."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def metrika_read(
    path: str,
    params: Optional[dict[str, Any]] = None,
) -> Any:
    safe_path = _clean_api_path(path)
    return await _yandex_http("GET", METRIKA_BASE + safe_path, params=params)


@mcp.tool(
    title="List Yandex Metrica counters",
    description="READ-ONLY: list counters available to the OAuth user.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def metrika_list_counters(
    fields: str = "goals,mirrors,grants,filters,operations,counter_flags,measurement_tokens",
) -> Any:
    return await _yandex_http(
        "GET",
        f"{METRIKA_BASE}/management/v1/counters",
        params={"field": fields},
    )


@mcp.tool(
    title="Metrica report",
    description="READ-ONLY: run a Yandex Metrica Reporting API request (/stat/v1/data).",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def metrika_report(params: dict[str, Any]) -> Any:
    return await _yandex_http("GET", f"{METRIKA_BASE}/stat/v1/data", params=params)


async def _preview_metrika_write_internal(
    method: str,
    path: str,
    params: Optional[dict[str, Any]] = None,
    payload: Optional[Any] = None,
    content_base64: Optional[str] = None,
    content_type: Optional[str] = None,
) -> dict[str, Any]:
    safe_path = _clean_api_path(path)
    if content_base64 is not None:
        base64.b64decode(content_base64, validate=True)
    return _make_marketing_plan(
        "metrika", method, safe_path, params=params, payload=payload,
        content_base64=content_base64, content_type=content_type,
    )


@mcp.tool(
    title="Yandex Direct read",
    description=(
        "READ-ONLY universal Direct API v5 reader. Calls only the get method of the "
        "specified Direct service; mutating Direct methods are rejected here."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def direct_get(
    service: str,
    params: dict[str, Any],
) -> Any:
    service_name = str(service).strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,80}", service_name):
        raise ValueError("Invalid Direct service name.")
    headers = _direct_auth_headers()
    headers["Content-Type"] = "application/json; charset=utf-8"
    headers["Accept-Language"] = "ru"
    if YANDEX_DIRECT_CLIENT_LOGIN:
        headers["Client-Login"] = YANDEX_DIRECT_CLIENT_LOGIN
    body = {"method": "get", "params": params}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        response = await client.post(f"{DIRECT_BASE}/{service_name}", headers=headers, json=body)
    return _parse_direct_json(response, operation=f"{service_name}.get")


@mcp.tool(
    title="Yandex Direct report",
    description="READ-ONLY: request a Yandex Direct Reports API v5 report.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def direct_report(
    params: dict[str, Any],
    return_money_in_micros: bool = False,
    skip_report_header: bool = True,
    skip_report_summary: bool = True,
) -> Any:
    headers = _direct_auth_headers()
    headers["Content-Type"] = "application/json; charset=utf-8"
    headers["Accept-Language"] = "ru"
    headers["processingMode"] = "auto"
    headers["returnMoneyInMicros"] = "true" if return_money_in_micros else "false"
    headers["skipReportHeader"] = "true" if skip_report_header else "false"
    headers["skipReportSummary"] = "true" if skip_report_summary else "false"
    if YANDEX_DIRECT_CLIENT_LOGIN:
        headers["Client-Login"] = YANDEX_DIRECT_CLIENT_LOGIN
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        response = await client.post(DIRECT_REPORTS_URL, headers=headers, json={"params": params})
    if response.status_code >= 400:
        raise RuntimeError(
            f"Yandex Direct Reports returned HTTP {response.status_code}: {_safe_response_text(response)}"
        )
    return {
        "status_code": response.status_code,
        "text": response.text,
        "headers": {
            k: v for k, v in response.headers.items()
            if k.lower().startswith("reports-") or k.lower() == "retry-in"
        },
    }


async def _preview_direct_write_internal(
    service: str,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    service_name = str(service).strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,80}", service_name):
        raise ValueError("Invalid Direct service name.")
    method_name = str(method).strip()
    if method_name.lower() == "get":
        raise ValueError("Use direct_get for read-only Direct requests.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,80}", method_name):
        raise ValueError("Invalid Direct method name.")
    if not isinstance(params, dict) or not params:
        raise ValueError("Direct write params must be a non-empty object.")

    # Guard a known campaigns.add schema trap: BudgetType is not part of
    # StrategyMaximumConversionRateAdd in the current Direct v5 add schema.
    if service_name == "campaigns" and method_name.lower() == "add":
        for campaign in params.get("Campaigns", []) if isinstance(params.get("Campaigns"), list) else []:
            tc = campaign.get("TextCampaign") if isinstance(campaign, dict) else None
            search = ((tc or {}).get("BiddingStrategy") or {}).get("Search") if isinstance(tc, dict) else None
            wb = (search or {}).get("WbMaximumConversionRate") if isinstance(search, dict) else None
            if isinstance(wb, dict) and "BudgetType" in wb:
                raise ValueError(
                    "campaigns.add: remove WbMaximumConversionRate.BudgetType; "
                    "use WeeklySpendLimit for the weekly budget in the current Direct v5 add schema."
                )
    return _make_marketing_plan(
        "direct",
        "POST",
        service_name,
        payload={"method": method_name, "params": params},
    )


async def _apply_confirmed_marketing_plan(
    plan_id: str,
    confirmation_token: str,
    *,
    required_api: str,
) -> Any:
    """Internal executor. Generic execution is never exposed as an MCP write tool."""
    now = time.time()

    # Critical: preview/apply may arrive in different stateless HTTP requests.
    # Never rely on process memory for authorization state.
    with _MARKETING_PLAN_LOCK:
        store = _prune_marketing_plans(_load_marketing_plan_store())
        plan = store.get(plan_id)
        if not plan:
            raise ValueError("Marketing plan not found. It may have expired or the durable store was reset.")
        if str(plan.get("api")) != required_api:
            raise ValueError(f"This confirmation tool accepts only {required_api} plans.")
        expected = str(plan["confirmation_token_hash"])
        actual = _sha256_bytes(str(confirmation_token).encode("utf-8"))
        if not hmac.compare_digest(expected, actual):
            _marketing_audit_event(plan, "confirmation_rejected", reason="invalid_token")
            store[plan_id] = plan
            _save_marketing_plan_store(store)
            raise ValueError("Invalid confirmation token.")
        if plan.get("state") != "pending" or plan.get("used"):
            raise ValueError(f"Marketing plan is not pending; current state={plan.get('state')}.")
        if float(plan["expires_at"]) <= now:
            plan["state"] = "expired"
            plan["terminal_at"] = now
            _marketing_audit_event(plan, "expired_before_apply")
            store[plan_id] = plan
            _save_marketing_plan_store(store)
            raise ValueError("Marketing plan has expired.")

        # Consume BEFORE any external request. This prevents duplicate mutation if the
        # client retries after an ambiguous timeout or tool transport failure.
        plan["used"] = True
        plan["state"] = "executing"
        plan["apply_started_at"] = now
        _marketing_audit_event(plan, "apply_started", api=required_api)
        store[plan_id] = plan
        _save_marketing_plan_store(store)

    try:
        result = await _execute_marketing_plan(plan)
    except Exception as exc:
        # Do not make the plan reusable. A transport failure can be ambiguous:
        # Yandex may have accepted the mutation even if the response was lost.
        with _MARKETING_PLAN_LOCK:
            store = _load_marketing_plan_store()
            current = store.get(plan_id, plan)
            current["state"] = "failed_or_ambiguous"
            current["terminal_at"] = time.time()
            current["result_summary"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
            }
            _marketing_audit_event(
                current, "apply_failed_or_ambiguous",
                error_type=type(exc).__name__,
                error=str(exc)[:4000],
            )
            store[plan_id] = current
            _save_marketing_plan_store(store)
        raise

    with _MARKETING_PLAN_LOCK:
        store = _load_marketing_plan_store()
        current = store.get(plan_id, plan)
        current["state"] = "succeeded"
        current["terminal_at"] = time.time()
        # Keep only safe compact diagnostics in durable history.
        summary = {"ok": True}
        if isinstance(result, dict):
            for key in ("http_status", "request_id", "units", "units_used_login", "warnings"):
                if key in result:
                    summary[key] = result[key]
        current["result_summary"] = summary
        _marketing_audit_event(current, "apply_succeeded", **summary)
        store[plan_id] = current
        _save_marketing_plan_store(store)

    return {
        "ok": True,
        "applied": True,
        "operation_sha256": plan["sha256"],
        "api": plan["api"],
        "method": plan["method"],
        "target": plan["target"],
        "result": result,
    }


ROSTOV_KITCHEN_CAMPAIGN_V1_NAME = "Ростов | Поиск | Кухни"
ROSTOV_KITCHEN_CAMPAIGN_V1_PARAMS = {
    "Campaigns": [{
        "Name": ROSTOV_KITCHEN_CAMPAIGN_V1_NAME,
        "StartDate": "2026-09-18",
        "TextCampaign": {
            "BiddingStrategy": {
                "Search": {
                    "BiddingStrategyType": "WB_MAXIMUM_CONVERSION_RATE",
                    "WbMaximumConversionRate": {
                        "GoalId": 626377274,
                        "WeeklySpendLimit": 6500000000,
                    },
                },
                "Network": {"BiddingStrategyType": "SERVING_OFF"},
            },
            "Settings": [
                {"Option": "ADD_METRICA_TAG", "Value": "NO"},
                {"Option": "ADD_TO_FAVORITES", "Value": "NO"},
                {"Option": "ENABLE_SITE_MONITORING", "Value": "YES"},
                {"Option": "ALTERNATIVE_TEXTS_ENABLED", "Value": "NO"},
            ],
            "CounterIds": {"Items": [112645576]},
        },
    }]
}


async def _direct_request(service: str, body: dict[str, Any]) -> httpx.Response:
    """Low-level Direct HTTP helper used only by fixed-operation pre/post read checks."""
    service_name = str(service or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,80}", service_name):
        raise ValueError("Invalid Direct service name.")
    if not isinstance(body, dict) or not body:
        raise ValueError("Direct request body must be a non-empty object.")

    headers = _direct_auth_headers()
    headers["Content-Type"] = "application/json; charset=utf-8"
    headers["Accept-Language"] = "ru"
    if YANDEX_DIRECT_CLIENT_LOGIN:
        headers["Client-Login"] = YANDEX_DIRECT_CLIENT_LOGIN

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        return await client.post(
            f"{DIRECT_BASE}/{service_name}",
            headers=headers,
            json=body,
        )


async def _direct_find_campaign_by_name_exact(name: str) -> list[dict[str, Any]]:
    """READ-ONLY pre/post-condition helper. Exact name match only."""
    response = await _direct_request(
        "campaigns",
        {
            "method": "get",
            "params": {
                "SelectionCriteria": {},
                "FieldNames": ["Id", "Name", "State", "Status", "Type"],
            },
        },
    )
    data = _parse_direct_json(response, operation="campaigns.get fixed-operation verification")
    campaigns = ((data.get("result") or {}).get("Campaigns") or [])
    return [c for c in campaigns if str(c.get("Name") or "") == name]


@mcp.tool(
    title="Preview create Rostov kitchen campaign",
    description=(
        "PREVIEW ONLY for one fixed predefined Direct operation. It first checks that no "
        "campaign named 'Ростов | Поиск | Кухни' already exists, then prepares an immutable "
        "plan for exactly that campaign shell. The caller supplies no Direct service, method, "
        "payload, budget, goal, counter, targeting, URL, group, keyword or ad."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_create_rostov_kitchen_campaign() -> dict[str, Any]:
    existing = await _direct_find_campaign_by_name_exact(ROSTOV_KITCHEN_CAMPAIGN_V1_NAME)
    if existing:
        return {
            "ok": False,
            "preview_only": True,
            "blocked_by_precondition": True,
            "reason": "campaign_already_exists",
            "existing": existing,
        }
    preview = _make_marketing_plan(
        "direct",
        "POST",
        "campaigns",
        payload={"method": "add", "params": ROSTOV_KITCHEN_CAMPAIGN_V1_PARAMS},
    )
    preview["fixed_operation"] = "create_rostov_kitchen_campaign_v1"
    preview["precondition"] = "no exact-name campaign exists"
    return preview


@mcp.tool(
    title="Apply create Rostov kitchen campaign",
    description=(
        "WRITE for one fixed predefined Direct operation only. Accepts only the immutable "
        "plan_id and one-time confirmation_token from preview_create_rostov_kitchen_campaign. "
        "Before mutation it re-checks that the campaign does not already exist. It cannot "
        "alter any Direct field and creates no groups, keywords or ads and requests no moderation."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True),
)
async def apply_create_rostov_kitchen_campaign(
    plan_id: Annotated[str, Field(min_length=1, max_length=120)],
    confirmation_token: Annotated[str, Field(min_length=1, max_length=256)],
) -> dict[str, Any]:
    # Validate the durable immutable plan is exactly this predefined operation.
    store = _prune_marketing_plans(_load_marketing_plan_store())
    plan = store.get(plan_id)
    if not plan:
        raise ValueError("Fixed Rostov campaign plan not found or expired.")
    expected_payload = {"method": "add", "params": ROSTOV_KITCHEN_CAMPAIGN_V1_PARAMS}
    if (
        plan.get("api") != "direct"
        or plan.get("method") != "POST"
        or plan.get("target") != "campaigns"
        or plan.get("payload") != expected_payload
    ):
        raise ValueError("Plan does not match fixed operation create_rostov_kitchen_campaign_v1.")

    # Second precondition check immediately before the mutation.
    existing = await _direct_find_campaign_by_name_exact(ROSTOV_KITCHEN_CAMPAIGN_V1_NAME)
    if existing:
        raise ValueError("Precondition failed: Rostov kitchen campaign already exists; refusing duplicate add.")

    try:
        applied = await _apply_confirmed_marketing_plan(
            plan_id,
            confirmation_token,
            required_api="direct",
        )
    except Exception:
        # Never retry add automatically. First perform a read-back because a transport
        # failure can be ambiguous: Yandex may have accepted the mutation.
        found_after_error = await _direct_find_campaign_by_name_exact(ROSTOV_KITCHEN_CAMPAIGN_V1_NAME)
        if found_after_error:
            return {
                "ok": True,
                "applied": "verified_after_ambiguous_error",
                "fixed_operation": "create_rostov_kitchen_campaign_v1",
                "campaigns": found_after_error,
                "warning": "Write response was ambiguous, but read-back found the exact campaign. No retry performed.",
            }
        raise

    # Mandatory post-write read-back. Success is not declared from HTTP response alone.
    created = await _direct_find_campaign_by_name_exact(ROSTOV_KITCHEN_CAMPAIGN_V1_NAME)
    if len(created) != 1:
        raise RuntimeError(
            f"Post-write verification failed: expected exactly 1 exact-name campaign, found {len(created)}. "
            "No automatic retry will be performed."
        )
    return {
        "ok": True,
        "applied": True,
        "fixed_operation": "create_rostov_kitchen_campaign_v1",
        "campaign": created[0],
        "direct_result": applied,
        "post_write_verified": True,
    }


# =============================================================================
# Rostov kitchen launch pack
# One stable server; routine launch steps are selected as predefined operations.
# No arbitrary Direct service/method/payload is exposed to ChatGPT.
# =============================================================================

ROSTOV_KITCHEN_CAMPAIGN_ID = 714564245
ROSTOV_KITCHEN_REGION_IDS = [11029]
ROSTOV_KITCHEN_LANDING = "https://rostov.yamebel.pro/services/kitchens/"
ROSTOV_KITCHEN_UTM = (
    "?utm_source=yandex&utm_medium=cpc&utm_campaign={campaign_id}"
    "&utm_content={gbid}.{ad_id}&utm_term={keyword}"
    "&source={source}&device={device_type}"
)

ROSTOV_KITCHEN_GROUP_NAMES = [
    "01 Кухни на заказ",
    "02 Ростов",
    "03 По размерам",
    "04 Цена",
    "05 Расчёт",
    "06 Конфигурация",
]

# Cross-negatives implement an intent hierarchy:
# calculation > price > configuration > dimensions > geo > core.
ROSTOV_KITCHEN_GROUP_NEGATIVES = {
    "01 Кухни на заказ": [
        "ростов", "цена", "стоимость", "сколько", "рассчитать", "расчет",
        "калькулятор", "размер", "угловая", "прямая", "п образная",
        "п-образная", "остров",
    ],
    "02 Ростов": [
        "цена", "стоимость", "сколько", "рассчитать", "расчет", "калькулятор",
        "размер", "угловая", "прямая", "п образная", "п-образная", "остров",
    ],
    "03 По размерам": [
        "цена", "стоимость", "сколько", "рассчитать", "расчет", "калькулятор",
        "угловая", "прямая", "п образная", "п-образная", "остров",
    ],
    "04 Цена": ["рассчитать", "расчет", "калькулятор"],
    "05 Расчёт": [],
    "06 Конфигурация": [
        "цена", "стоимость", "сколько", "рассчитать", "расчет", "калькулятор",
    ],
}

ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES = [
    "своими руками", "чертеж", "чертежи", "схема", "схемы", "скачать",
    "торрент", "работа", "вакансии", "вакансия", "зарплата", "обучение",
    "курс", "курсы", "реферат", "диплом", "картинки для детей",
    "игрушечная", "кукольная", "детская игрушка", "ремонт техники",
    "ремонт холодильника", "ремонт плиты", "рецепт", "рецепты", "игра", "игры",
    "недорого", "дешево", "дешевый", "дешёвый",
    "эконом", "эконом класс", "бюджетная", "бюджетный", "бюджетные",
    "бу", "авито",
    "ремонт кухни", "реставрация кухни", "замена фасадов",
    "ремонт фасадов", "покраска фасадов",
]

ROSTOV_KITCHEN_KEYWORDS = {
    "01 Кухни на заказ": [
        "кухня на заказ",
        "кухни на заказ",
        "заказать кухню",
        "заказать кухню на заказ",
        "изготовление кухни на заказ",
        "изготовление кухонь на заказ",
        "сделать кухню на заказ",
    ],
    "02 Ростов": [
        "кухни на заказ ростов",
        "кухня на заказ ростов",
        "кухни на заказ ростов на дону",
        "кухня на заказ ростов на дону",
        "заказать кухню ростов",
        "заказать кухню ростов на дону",
        "изготовление кухонь ростов",
        "изготовление кухни ростов",
    ],
    "03 По размерам": [
        "кухня по размерам",
        "кухни по размерам",
        "кухня по индивидуальным размерам",
        "кухни по индивидуальным размерам",
        "заказать кухню по размерам",
        "изготовление кухни по размерам",
        "изготовление кухни по индивидуальным размерам",
        "кухни по индивидуальным размерам ростов",
        "кухни по размерам ростов",
    ],
    "04 Цена": [
        "кухня на заказ цена",
        "кухни на заказ цены",
        "стоимость кухни на заказ",
        "сколько стоит кухня на заказ",
        "цена кухни по размерам",
        "стоимость кухни по размерам",
        "кухни на заказ цены ростов",
        "кухня на заказ цена ростов",
        "стоимость кухни на заказ ростов",
    ],
    "05 Расчёт": [
        "рассчитать кухню",
        "рассчитать стоимость кухни",
        "расчет кухни на заказ",
        "расчет стоимости кухни",
        "рассчитать кухню по размерам",
        "расчет кухни по размерам",
        "предварительный расчет кухни",
        "калькулятор кухни на заказ",
        "калькулятор стоимости кухни",
    ],
    "06 Конфигурация": [
        "угловая кухня на заказ",
        "угловые кухни на заказ",
        "угловая кухня по размерам",
        "прямая кухня на заказ",
        "п образная кухня на заказ",
        "п образная кухня по размерам",
        "кухня с островом на заказ",
        "кухни с островом на заказ",
    ],
}

ROSTOV_KITCHEN_SITELINKS = [
    {
        "Title": "Рассчитать стоимость",
        "Href": (
            "https://rostov.yamebel.pro/services/kitchens/"
            "?utm_source=yandex&utm_medium=cpc&utm_campaign={campaign_id}"
            "&utm_content=sitelink_calc.{ad_id}&utm_term={keyword}"
            "&source={source}&device={device_type}"
        ),
        "Description": "Предварительный расчёт кухни по вашим размерам.",
    },
    {
        "Title": "Наши проекты",
        "Href": (
            "https://rostov.yamebel.pro/project/"
            "?utm_source=yandex&utm_medium=cpc&utm_campaign={campaign_id}"
            "&utm_content=sitelink_projects.{ad_id}&utm_term={keyword}"
            "&source={source}&device={device_type}"
        ),
        "Description": "Реальные проекты мебели и примеры работ.",
    },
    {
        "Title": "О компании",
        "Href": (
            "https://rostov.yamebel.pro/about/"
            "?utm_source=yandex&utm_medium=cpc&utm_campaign={campaign_id}"
            "&utm_content=sitelink_about.{ad_id}&utm_term={keyword}"
            "&source={source}&device={device_type}"
        ),
        "Description": "Как мы работаем и что важно в заказе.",
    },
    {
        "Title": "Контакты",
        "Href": (
            "https://rostov.yamebel.pro/contacts/"
            "?utm_source=yandex&utm_medium=cpc&utm_campaign={campaign_id}"
            "&utm_content=sitelink_contacts.{ad_id}&utm_term={keyword}"
            "&source={source}&device={device_type}"
        ),
        "Description": "Связаться с командой Я Мебель в Ростове.",
    },
]

ROSTOV_KITCHEN_CALLOUTS = [
    "Гарантия 5 лет",
    "Индивидуальный проект",
    "По вашим размерам",
    "Ростов-на-Дону и область",
]

ROSTOV_KITCHEN_ADS_BALANCED_V1 = {
    "01 Кухни на заказ": {
        "Title": "Кухни на заказ в Ростове-на-Дону",
        "Title2": "Мебель для вашей истории",
        "Text": "Создадим кухню под ваш образ жизни. Проект, доставка, монтаж. Гарантия 5 лет.",
    },
    "02 Ростов": {
        "Title": "Кухни на заказ в Ростове-на-Дону",
        "Title2": "По вашим размерам",
        "Text": "Индивидуальный проект под ваш дом и привычки. Доставка, монтаж, гарантия 5 лет.",
    },
    "03 По размерам": {
        "Title": "Кухня по вашим размерам",
        "Title2": "Для вашей истории",
        "Text": "Проект под помещение и сценарии жизни. Доставка, монтаж, гарантия 5 лет.",
    },
    "04 Цена": {
        "Title": "Узнайте стоимость кухни до заказа",
        "Title2": "Предварительный расчёт",
        "Text": "Пришлите размеры, планировку или фото — подготовим предварительный расчёт кухни.",
    },
    "05 Расчёт": {
        "Title": "Рассчитайте стоимость кухни",
        "Title2": "По размерам или фото",
        "Text": "Расчёт кухни по размерам, планировке или фото. Реальные проекты и цены.",
    },
    "06 Конфигурация": {
        "Title": "Кухня вашей планировки на заказ",
        "Title2": "По вашим размерам",
        "Text": "Прямые, угловые, П-образные и с островом. Проект, доставка и монтаж.",
    },
}

ROSTOV_KITCHEN_ALLOWED_OPERATIONS = {
    "set_campaign_negatives_fixed_v2",
    "create_sitelinks_fixed_v1",
    "create_callouts_fixed_v1",
    "submit_ads_moderation_fixed_v1",
    "resume_campaign_fixed_v1",
    "suspend_autotargeting_g01_v1",
    "add_keywords_g01_v1",
    "create_ad_g01_v1",
    "suspend_autotargeting_g02_v1",
    "add_keywords_g02_v1",
    "create_ad_g02_v1",
    "suspend_autotargeting_g03_v1",
    "add_keywords_g03_v1",
    "create_ad_g03_v1",
    "suspend_autotargeting_g04_v1",
    "add_keywords_g04_v1",
    "create_ad_g04_v1",
    "suspend_autotargeting_g05_v1",
    "add_keywords_g05_v1",
    "create_ad_g05_v1",
    "suspend_autotargeting_g06_v1",
    "add_keywords_g06_v1",
    "create_ad_g06_v1",
    "create_groups_v1",
    "sync_regions_from_group_06_v1",
    "suspend_autotargeting_v1",
    "resume_autotargeting_v1",
    "set_campaign_negatives_v1",
    "add_keywords_v1",
    "create_sitelinks_v1",
    "create_callouts_v1",
    "create_ads_balanced_v1",
    "submit_ads_moderation_v1",
    "resume_campaign_v1",
    "suspend_campaign_v1",
}


def _rostov_norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _rostov_sitelinks_signature(items: list[dict[str, Any]]) -> tuple:
    return tuple(
        (
            str(x.get("Title") or ""),
            str(x.get("Href") or ""),
            str(x.get("Description") or ""),
        )
        for x in items
    )


async def _rostov_campaign_get() -> Optional[dict[str, Any]]:
    response = await _direct_request(
        "campaigns",
        {
            "method": "get",
            "params": {
                "SelectionCriteria": {"Ids": [ROSTOV_KITCHEN_CAMPAIGN_ID]},
                "FieldNames": [
                    "Id", "Name", "State", "Status", "Type", "NegativeKeywords", "StartDate"
                ],
                "TextCampaignFieldNames": ["BiddingStrategy", "CounterIds", "Settings"],
            },
        },
    )
    data = _parse_direct_json(response, operation="campaigns.get rostov launch")
    campaigns = ((data.get("result") or {}).get("Campaigns") or [])
    return campaigns[0] if campaigns else None


async def _rostov_groups_get() -> list[dict[str, Any]]:
    response = await _direct_request(
        "adgroups",
        {
            "method": "get",
            "params": {
                "SelectionCriteria": {"CampaignIds": [ROSTOV_KITCHEN_CAMPAIGN_ID]},
                "FieldNames": [
                    "Id", "Name", "CampaignId", "RegionIds", "NegativeKeywords",
                    "Status", "ServingStatus", "Type"
                ],
            },
        },
    )
    data = _parse_direct_json(response, operation="adgroups.get rostov launch")
    return ((data.get("result") or {}).get("AdGroups") or [])


async def _rostov_group_map(require_all: bool = True) -> dict[str, dict[str, Any]]:
    groups = await _rostov_groups_get()
    by_name = {
        str(g.get("Name") or ""): g
        for g in groups
        if str(g.get("Name") or "") in ROSTOV_KITCHEN_GROUP_NAMES
    }
    if require_all:
        missing = [n for n in ROSTOV_KITCHEN_GROUP_NAMES if n not in by_name]
        if missing:
            raise ValueError(f"Rostov launch prerequisite failed: missing ad groups: {missing}")
    return by_name


def _rostov_autotargeting_check(keywords: list[dict[str, Any]], group_ids: list[int]) -> dict[str, Any]:
    """Fail closed unless each target group has the approved search settings."""
    expected = {
        "Categories": {
            "Exact": "YES", "Narrow": "NO", "Alternative": "NO",
            "Accessory": "NO", "Broader": "NO",
        },
        "BrandOptions": {
            "WithoutBrands": "YES", "WithAdvertiserBrand": "YES",
            "WithCompetitorsBrand": "NO",
        },
    }
    checks = []
    for gid in group_ids:
        autos = [k for k in keywords if int(k.get("AdGroupId") or 0) == gid
                 and k.get("Keyword") == "---autotargeting"]
        differences = []
        if len(autos) != 1:
            differences.append({"field": "count", "expected": 1, "actual": len(autos)})
        else:
            auto = autos[0]
            if auto.get("State") != "ON":
                differences.append({"field": "State", "expected": "ON", "actual": auto.get("State")})
            settings = auto.get("AutotargetingSettings")
            settings = settings if isinstance(settings, dict) else {}
            for section, fields in expected.items():
                actual = settings.get(section)
                actual = actual if isinstance(actual, dict) else {}
                for field, wanted in fields.items():
                    if actual.get(field) != wanted:
                        differences.append({"field": section + "." + field,
                                            "expected": wanted, "actual": actual.get(field)})
        checks.append({"group_id": gid, "ok": not differences, "differences": differences})
    return {"ok": bool(checks) and all(c["ok"] for c in checks),
            "expected_settings": expected, "groups": checks}


async def _rostov_keywords_get() -> list[dict[str, Any]]:
    response = await _direct_request(
        "keywords",
        {
            "method": "get",
            "params": {
                "SelectionCriteria": {"CampaignIds": [ROSTOV_KITCHEN_CAMPAIGN_ID]},
                "FieldNames": [
                    "Id", "Keyword", "AdGroupId", "CampaignId", "State", "Status", "ServingStatus"
                ],
                "AutotargetingSettingsCategoriesFieldNames": [
                    "Exact", "Narrow", "Alternative", "Accessory", "Broader"
                ],
                "AutotargetingSettingsBrandOptionsFieldNames": [
                    "WithoutBrands", "WithAdvertiserBrand", "WithCompetitorsBrand"
                ],
            },
        },
    )
    data = _parse_direct_json(response, operation="keywords.get rostov launch")
    return ((data.get("result") or {}).get("Keywords") or [])


async def _rostov_ads_get() -> list[dict[str, Any]]:
    groups = await _rostov_group_map(require_all=False)
    ids = [int(g["Id"]) for g in groups.values() if g.get("Id") is not None]
    if not ids:
        return []
    response = await _direct_request(
        "ads",
        {
            "method": "get",
            "params": {
                "SelectionCriteria": {"AdGroupIds": ids},
                "FieldNames": [
                    "Id", "AdGroupId", "CampaignId", "Type", "State", "Status", "StatusClarification"
                ],
                "TextAdFieldNames": [
                    "Title", "Title2", "Text", "Href", "Mobile", "SitelinkSetId", "AdExtensions"
                ],
            },
        },
    )
    data = _parse_direct_json(response, operation="ads.get rostov launch")
    return ((data.get("result") or {}).get("Ads") or [])


async def _rostov_sitelinks_get_all() -> list[dict[str, Any]]:
    response = await _direct_request(
        "sitelinks",
        {
            "method": "get",
            "params": {
                "FieldNames": ["Id", "Sitelinks"],
                "Page": {"Limit": 10000, "Offset": 0},
            },
        },
    )
    data = _parse_direct_json(response, operation="sitelinks.get rostov launch")
    return ((data.get("result") or {}).get("SitelinksSets") or [])


async def _rostov_find_sitelink_set() -> list[dict[str, Any]]:
    wanted = _rostov_sitelinks_signature(ROSTOV_KITCHEN_SITELINKS)
    found = []
    for item in await _rostov_sitelinks_get_all():
        if _rostov_sitelinks_signature(item.get("Sitelinks") or []) == wanted:
            found.append(item)
    return found


async def _rostov_callouts_get_all() -> list[dict[str, Any]]:
    response = await _direct_request(
        "adextensions",
        {
            "method": "get",
            "params": {
                "SelectionCriteria": {"Types": ["CALLOUT"]},
                "FieldNames": ["Id", "Type", "Status", "Associated"],
                "CalloutFieldNames": ["CalloutText"],
                "Page": {"Limit": 10000, "Offset": 0},
            },
        },
    )
    data = _parse_direct_json(response, operation="adextensions.get rostov launch")
    return ((data.get("result") or {}).get("AdExtensions") or [])


async def _rostov_callout_map() -> dict[str, dict[str, Any]]:
    wanted = {_rostov_norm_text(x): x for x in ROSTOV_KITCHEN_CALLOUTS}
    out = {}
    for ext in await _rostov_callouts_get_all():
        text = _rostov_norm_text(((ext.get("Callout") or {}).get("CalloutText")))
        if text in wanted:
            out[wanted[text]] = ext
    return out


def _rostov_groups_add_payload() -> dict[str, Any]:
    groups = []
    for name in ROSTOV_KITCHEN_GROUP_NAMES:
        item = {
            "Name": name,
            "CampaignId": ROSTOV_KITCHEN_CAMPAIGN_ID,
            "RegionIds": ROSTOV_KITCHEN_REGION_IDS,
        }
        negatives = ROSTOV_KITCHEN_GROUP_NEGATIVES.get(name) or []
        if negatives:
            item["NegativeKeywords"] = {"Items": negatives}
        groups.append(item)
    return {"method": "add", "params": {"AdGroups": groups}}


async def _rostov_keywords_add_payload() -> dict[str, Any]:
    groups = await _rostov_group_map(require_all=True)
    items = []
    for name in ROSTOV_KITCHEN_GROUP_NAMES:
        gid = int(groups[name]["Id"])
        for keyword in ROSTOV_KITCHEN_KEYWORDS[name]:
            items.append({"Keyword": keyword, "AdGroupId": gid})
    return {"method": "add", "params": {"Keywords": items}}


async def _rostov_ads_add_payload() -> dict[str, Any]:
    groups = await _rostov_group_map(require_all=True)
    sitelink_sets = await _rostov_find_sitelink_set()
    if len(sitelink_sets) != 1:
        raise ValueError(
            f"Rostov launch prerequisite failed: expected exactly one approved sitelink set candidate, found {len(sitelink_sets)}."
        )
    callouts = await _rostov_callout_map()
    missing_callouts = [x for x in ROSTOV_KITCHEN_CALLOUTS if x not in callouts]
    if missing_callouts:
        raise ValueError(f"Rostov launch prerequisite failed: missing callouts: {missing_callouts}")

    sitelink_id = int(sitelink_sets[0]["Id"])
    callout_ids = [int(callouts[x]["Id"]) for x in ROSTOV_KITCHEN_CALLOUTS]
    ads = []
    for name in ROSTOV_KITCHEN_GROUP_NAMES:
        copy_item = ROSTOV_KITCHEN_ADS_BALANCED_V1[name]
        text_ad = {
            "Title": copy_item["Title"],
            "Title2": copy_item["Title2"],
            "Text": copy_item["Text"],
            "Href": ROSTOV_KITCHEN_LANDING + ROSTOV_KITCHEN_UTM,
            "Mobile": "NO",
            "SitelinkSetId": sitelink_id,
            "AdExtensions": [{"AdExtensionId": ext_id} for ext_id in callout_ids],
        }
        ads.append({"AdGroupId": int(groups[name]["Id"]), "TextAd": text_ad})
    return {"method": "add", "params": {"Ads": ads}}


async def _rostov_expected_ads_by_group_id() -> dict[int, dict[str, str]]:
    groups = await _rostov_group_map(require_all=True)
    return {
        int(groups[name]["Id"]): ROSTOV_KITCHEN_ADS_BALANCED_V1[name]
        for name in ROSTOV_KITCHEN_GROUP_NAMES
    }


ROSTOV_GROUP_INDEX_TO_NAME = {
    "01": "01 Кухни на заказ",
    "02": "02 Ростов",
    "03": "03 По размерам",
    "04": "04 Цена",
    "05": "05 Расчёт",
    "06": "06 Конфигурация",
}


def _rostov_group_name_from_fixed_operation(operation: str, prefix: str) -> Optional[str]:
    m = re.fullmatch(rf"{re.escape(prefix)}_g(0[1-6])_v1", str(operation or ""))
    if not m:
        return None
    return ROSTOV_GROUP_INDEX_TO_NAME[m.group(1)]


async def _rostov_single_group_ad_payload(group_name: str) -> dict[str, Any]:
    groups = await _rostov_group_map(require_all=True)
    sitelink_sets = await _rostov_find_sitelink_set()
    if len(sitelink_sets) != 1:
        raise ValueError(
            f"Rostov launch prerequisite failed: expected exactly one sitelink set, found {len(sitelink_sets)}."
        )
    callouts = await _rostov_callout_map()
    missing = [x for x in ROSTOV_KITCHEN_CALLOUTS if x not in callouts]
    if missing:
        raise ValueError(f"Rostov launch prerequisite failed: missing callouts: {missing}")

    copy_item = ROSTOV_KITCHEN_ADS_BALANCED_V1[group_name]
    text_ad = {
        "Title": copy_item["Title"],
        "Title2": copy_item["Title2"],
        "Text": copy_item["Text"],
        "Href": ROSTOV_KITCHEN_LANDING + ROSTOV_KITCHEN_UTM,
        "Mobile": "NO",
        "SitelinkSetId": int(sitelink_sets[0]["Id"]),
        "AdExtensions": [
            {"AdExtensionId": int(callouts[x]["Id"])}
            for x in ROSTOV_KITCHEN_CALLOUTS
        ],
    }
    return {
        "method": "add",
        "params": {
            "Ads": [{
                "AdGroupId": int(groups[group_name]["Id"]),
                "TextAd": text_ad,
            }]
        },
    }


async def _rostov_prepare_operation(operation: str) -> dict[str, Any]:
    operation = str(operation or "").strip()
    if operation not in ROSTOV_KITCHEN_ALLOWED_OPERATIONS:
        raise ValueError(
            "Unsupported Rostov operation. Allowed: "
            + ", ".join(sorted(ROSTOV_KITCHEN_ALLOWED_OPERATIONS))
        )

    campaign = await _rostov_campaign_get()
    if not campaign:
        return {"ok": False, "blocked_by_precondition": True, "reason": "campaign_missing"}

    if operation == "set_campaign_negatives_fixed_v2":
        existing = ((campaign.get("NegativeKeywords") or {}).get("Items") or [])
        if {_rostov_norm_text(x) for x in existing} == {_rostov_norm_text(x) for x in ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES}:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "campaign_negatives_already_match",
            }
        return {
            "service": "campaigns",
            "payload": {
                "method": "update",
                "params": {
                    "Campaigns": [{
                        "Id": ROSTOV_KITCHEN_CAMPAIGN_ID,
                        "NegativeKeywords": {"Items": ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES},
                    }]
                },
            },
            "summary": {
                "negative_keywords": ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES,
                "count": len(ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES),
                "fixed": True,
            },
        }

    group_name = _rostov_group_name_from_fixed_operation(operation, "suspend_autotargeting")
    if group_name:
        groups = await _rostov_group_map(require_all=True)
        gid = int(groups[group_name]["Id"])
        keywords = await _rostov_keywords_get()
        autos = [
            k for k in keywords
            if int(k.get("AdGroupId") or 0) == gid
            and str(k.get("Keyword") or "") == "---autotargeting"
        ]
        if len(autos) != 1:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "expected_one_autotargeting_in_group",
                "group": group_name,
                "found": len(autos),
            }
        auto = autos[0]
        if str(auto.get("State") or "") == "SUSPENDED":
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "autotargeting_already_suspended",
                "group": group_name,
            }
        return {
            "service": "keywords",
            "payload": {
                "method": "suspend",
                "params": {"SelectionCriteria": {"Ids": [int(auto["Id"])]}},
            },
            "summary": {
                "group": group_name,
                "autotargeting_id": int(auto["Id"]),
                "action": "suspend",
            },
        }

    group_name = _rostov_group_name_from_fixed_operation(operation, "add_keywords")
    if group_name:
        groups = await _rostov_group_map(require_all=True)
        gid = int(groups[group_name]["Id"])
        keywords = await _rostov_keywords_get()
        existing_manual = [
            k for k in keywords
            if int(k.get("AdGroupId") or 0) == gid
            and str(k.get("Keyword") or "") != "---autotargeting"
        ]
        if existing_manual:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "group_already_has_manual_keywords",
                "group": group_name,
                "existing_count": len(existing_manual),
            }
        auto_check = _rostov_autotargeting_check(keywords, [gid])
        if not auto_check["ok"]:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "group_autotargeting_settings_must_match_approved",
                "group": group_name,
                "autotargeting_check": auto_check,
            }
        items = [
            {"Keyword": keyword, "AdGroupId": gid}
            for keyword in ROSTOV_KITCHEN_KEYWORDS[group_name]
        ]
        return {
            "service": "keywords",
            "payload": {"method": "add", "params": {"Keywords": items}},
            "summary": {
                "group": group_name,
                "group_id": gid,
                "keywords": ROSTOV_KITCHEN_KEYWORDS[group_name],
                "count": len(items),
                "autotargeting_check": auto_check,
            },
        }

    if operation == "create_sitelinks_fixed_v1":
        existing = await _rostov_find_sitelink_set()
        if existing:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "exact_sitelink_set_already_exists",
                "existing_ids": [x.get("Id") for x in existing],
            }
        return {
            "service": "sitelinks",
            "payload": {
                "method": "add",
                "params": {"SitelinksSets": [{"Sitelinks": ROSTOV_KITCHEN_SITELINKS}]},
            },
            "summary": {"sitelinks": ROSTOV_KITCHEN_SITELINKS, "fixed": True},
        }

    if operation == "create_callouts_fixed_v1":
        current = await _rostov_callout_map()
        missing = [x for x in ROSTOV_KITCHEN_CALLOUTS if x not in current]
        if not missing:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "all_callouts_already_exist",
            }
        return {
            "service": "adextensions",
            "payload": {
                "method": "add",
                "params": {
                    "AdExtensions": [{"Callout": {"CalloutText": x}} for x in missing]
                },
            },
            "summary": {"callouts_to_create": missing, "fixed": True},
        }

    group_name = _rostov_group_name_from_fixed_operation(operation, "create_ad")
    if group_name:
        groups = await _rostov_group_map(require_all=True)
        gid = int(groups[group_name]["Id"])
        existing_ads = await _rostov_ads_get()
        group_ads = [a for a in existing_ads if int(a.get("AdGroupId") or 0) == gid]
        if group_ads:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "group_already_has_ad",
                "group": group_name,
                "ads": [{"Id": a.get("Id"), "Status": a.get("Status")} for a in group_ads],
            }
        payload = await _rostov_single_group_ad_payload(group_name)
        return {
            "service": "ads",
            "payload": payload,
            "summary": {
                "group": group_name,
                "ad": ROSTOV_KITCHEN_ADS_BALANCED_V1[group_name],
                "landing": ROSTOV_KITCHEN_LANDING,
            },
        }

    if operation == "submit_ads_moderation_fixed_v1":
        groups = await _rostov_group_map(require_all=True)
        ads = await _rostov_ads_get()
        target_gids = {int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES}
        target_ads = [a for a in ads if int(a.get("AdGroupId") or 0) in target_gids]
        if len(target_ads) != 6:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "expected_six_ads_before_moderation",
                "found": len(target_ads),
            }
        draft_ads = [a for a in target_ads if str(a.get("Status") or "") == "DRAFT"]
        if len(draft_ads) != 6:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "all_six_ads_must_be_draft",
                "statuses": [{"Id": a.get("Id"), "Status": a.get("Status")} for a in target_ads],
            }
        ids = [int(a["Id"]) for a in draft_ads]
        return {
            "service": "ads",
            "payload": {"method": "moderate", "params": {"SelectionCriteria": {"Ids": ids}}},
            "summary": {"ad_ids": ids, "action": "submit_for_moderation", "fixed": True},
        }

    if operation == "resume_campaign_fixed_v1":
        ads = await _rostov_ads_get()
        if len(ads) != 6 or any(str(a.get("Status") or "") != "ACCEPTED" for a in ads):
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "all_six_ads_must_be_accepted_before_resume",
                "ads": [{"Id": a.get("Id"), "Status": a.get("Status")} for a in ads],
            }
        if str(campaign.get("State") or "") == "ON":
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "campaign_already_on",
            }
        return {
            "service": "campaigns",
            "payload": {
                "method": "resume",
                "params": {"SelectionCriteria": {"Ids": [ROSTOV_KITCHEN_CAMPAIGN_ID]}},
            },
            "summary": {"campaign_id": ROSTOV_KITCHEN_CAMPAIGN_ID, "action": "resume", "fixed": True},
        }

    if operation == "create_groups_v1":
        groups = await _rostov_group_map(require_all=False)
        existing = [groups[n] for n in ROSTOV_KITCHEN_GROUP_NAMES if n in groups]
        if existing:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "one_or_more_target_groups_already_exist",
                "existing": existing,
            }
        return {
            "service": "adgroups",
            "payload": _rostov_groups_add_payload(),
            "summary": {
                "campaign_id": ROSTOV_KITCHEN_CAMPAIGN_ID,
                "groups": ROSTOV_KITCHEN_GROUP_NAMES,
                "region_ids": ROSTOV_KITCHEN_REGION_IDS,
                "group_cross_negatives": ROSTOV_KITCHEN_GROUP_NEGATIVES,
            },
        }

    if operation == "set_campaign_negatives_v1":
        existing = ((campaign.get("NegativeKeywords") or {}).get("Items") or [])
        if {_rostov_norm_text(x) for x in existing} == {_rostov_norm_text(x) for x in ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES}:
            return {"ok": False, "blocked_by_precondition": True, "reason": "campaign_negatives_already_match"}
        return {
            "service": "campaigns",
            "payload": {
                "method": "update",
                "params": {
                    "Campaigns": [{
                        "Id": ROSTOV_KITCHEN_CAMPAIGN_ID,
                        "NegativeKeywords": {"Items": ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES},
                    }]
                },
            },
            "summary": {"negative_keywords": ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES},
        }

    if operation == "sync_regions_from_group_06_v1":
        groups = await _rostov_group_map(require_all=True)
        source_regions = list(groups["06 Конфигурация"].get("RegionIds") or [])
        if source_regions != ROSTOV_KITCHEN_REGION_IDS:
            return {"ok": False, "blocked_by_precondition": True, "reason": "source_group_06_regions_must_be_11029"}
        mismatched = [n for n in ROSTOV_KITCHEN_GROUP_NAMES if list(groups[n].get("RegionIds") or []) != source_regions]
        if not mismatched:
            return {"ok": False, "blocked_by_precondition": True, "reason": "all_groups_already_match_group_06_regions", "region_ids": source_regions}
        updates = [{"Id": int(groups[n]["Id"]), "RegionIds": source_regions} for n in mismatched]
        return {
            "service": "adgroups",
            "payload": {"method": "update", "params": {"AdGroups": updates}},
            "summary": {"source_group": "06 Конфигурация", "source_region_ids": source_regions, "groups_to_update": mismatched},
        }

    if operation == "suspend_autotargeting_v1":
        groups = await _rostov_group_map(require_all=True)
        target_gids = {int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES}
        keywords = await _rostov_keywords_get()
        autos = [k for k in keywords if int(k.get("AdGroupId") or 0) in target_gids and str(k.get("Keyword") or "") == "---autotargeting"]
        if len(autos) != 6:
            return {"ok": False, "blocked_by_precondition": True, "reason": "expected_six_autotargetings", "found": len(autos)}
        ids = [int(k["Id"]) for k in autos if str(k.get("State") or "") != "SUSPENDED"]
        if not ids:
            return {"ok": False, "blocked_by_precondition": True, "reason": "all_autotargetings_already_suspended"}
        return {
            "service": "keywords",
            "payload": {"method": "suspend", "params": {"SelectionCriteria": {"Ids": ids}}},
            "summary": {"autotargeting_ids": ids, "action": "suspend"},
        }

    if operation == "resume_autotargeting_v1":
        groups = await _rostov_group_map(require_all=True)
        target_gids = {int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES}
        keywords = await _rostov_keywords_get()
        autos = [k for k in keywords if int(k.get("AdGroupId") or 0) in target_gids and str(k.get("Keyword") or "") == "---autotargeting"]
        if len(autos) != 6:
            return {"ok": False, "blocked_by_precondition": True, "reason": "expected_six_autotargetings", "found": len(autos)}
        ids = [int(k["Id"]) for k in autos if str(k.get("State") or "") == "SUSPENDED"]
        if not ids:
            return {"ok": False, "blocked_by_precondition": True, "reason": "no_suspended_autotargetings_to_resume"}
        return {
            "service": "keywords",
            "payload": {"method": "resume", "params": {"SelectionCriteria": {"Ids": ids}}},
            "summary": {"autotargeting_ids": ids, "action": "resume"},
        }

    if operation == "add_keywords_v1":
        groups = await _rostov_group_map(require_all=True)
        existing = await _rostov_keywords_get()
        target_gids = {int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES}
        existing_manual = [
            k for k in existing
            if int(k.get("AdGroupId") or 0) in target_gids
            and str(k.get("Keyword") or "") != "---autotargeting"
        ]
        if existing_manual:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "target_groups_already_have_manual_keywords",
                "existing_count": len(existing_manual),
            }
        auto_check = _rostov_autotargeting_check(existing, sorted(target_gids))
        if not auto_check["ok"]:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "autotargeting_settings_must_match_approved",
                "autotargeting_check": auto_check,
            }
        payload = await _rostov_keywords_add_payload()
        return {
            "service": "keywords",
            "payload": payload,
            "summary": {
                "keyword_count": sum(len(v) for v in ROSTOV_KITCHEN_KEYWORDS.values()),
                "keywords_by_group": ROSTOV_KITCHEN_KEYWORDS,
                "autotargeting_created": False,
                "autotargeting_check": auto_check,
            },
        }

    if operation == "create_sitelinks_v1":
        existing = await _rostov_find_sitelink_set()
        if existing:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "exact_sitelink_set_already_exists",
                "existing_ids": [x.get("Id") for x in existing],
            }
        return {
            "service": "sitelinks",
            "payload": {
                "method": "add",
                "params": {"SitelinksSets": [{"Sitelinks": ROSTOV_KITCHEN_SITELINKS}]},
            },
            "summary": {"sitelinks": ROSTOV_KITCHEN_SITELINKS},
        }

    if operation == "create_callouts_v1":
        current = await _rostov_callout_map()
        missing = [x for x in ROSTOV_KITCHEN_CALLOUTS if x not in current]
        if not missing:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "all_callouts_already_exist",
                "existing_ids": {x: current[x].get("Id") for x in ROSTOV_KITCHEN_CALLOUTS},
            }
        return {
            "service": "adextensions",
            "payload": {
                "method": "add",
                "params": {
                    "AdExtensions": [{"Callout": {"CalloutText": x}} for x in missing]
                },
            },
            "summary": {"missing_callouts_to_create": missing},
        }

    if operation == "create_ads_balanced_v1":
        groups = await _rostov_group_map(require_all=True)
        existing_ads = await _rostov_ads_get()
        target_gids = {int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES}
        existing_target = [a for a in existing_ads if int(a.get("AdGroupId") or 0) in target_gids]
        if existing_target:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "one_or_more_target_groups_already_have_ads",
                "existing_ads": [
                    {"Id": a.get("Id"), "AdGroupId": a.get("AdGroupId"), "Status": a.get("Status")}
                    for a in existing_target
                ],
            }
        payload = await _rostov_ads_add_payload()
        return {
            "service": "ads",
            "payload": payload,
            "summary": {
                "ads_by_group": ROSTOV_KITCHEN_ADS_BALANCED_V1,
                "landing": ROSTOV_KITCHEN_LANDING,
                "utm_location": "ad Href",
                "requests_moderation": False,
            },
        }

    if operation == "submit_ads_moderation_v1":
        groups = await _rostov_group_map(require_all=True)
        ads = await _rostov_ads_get()
        target_gids = {int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES}
        target_ads = [a for a in ads if int(a.get("AdGroupId") or 0) in target_gids]
        if len(target_ads) != 6:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "expected_six_ads_before_moderation",
                "found": len(target_ads),
            }
        not_draft = [a for a in target_ads if str(a.get("Status") or "") != "DRAFT"]
        if not_draft:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "one_or_more_ads_not_draft",
                "ads": [
                    {"Id": a.get("Id"), "Status": a.get("Status"), "State": a.get("State")}
                    for a in not_draft
                ],
            }
        ids = [int(a["Id"]) for a in target_ads]
        return {
            "service": "ads",
            "payload": {
                "method": "moderate",
                "params": {"SelectionCriteria": {"Ids": ids}},
            },
            "summary": {"ad_ids": ids, "action": "submit_for_moderation"},
        }

    if operation == "resume_campaign_v1":
        ads = await _rostov_ads_get()
        if len(ads) != 6:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "expected_six_ads_before_campaign_resume",
                "found": len(ads),
            }
        not_accepted = [a for a in ads if str(a.get("Status") or "") != "ACCEPTED"]
        if not_accepted:
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "ads_not_all_accepted",
                "ads": [
                    {"Id": a.get("Id"), "Status": a.get("Status"), "StatusClarification": a.get("StatusClarification")}
                    for a in not_accepted
                ],
            }
        if str(campaign.get("State") or "") == "ON":
            return {"ok": False, "blocked_by_precondition": True, "reason": "campaign_already_on"}
        return {
            "service": "campaigns",
            "payload": {
                "method": "resume",
                "params": {"SelectionCriteria": {"Ids": [ROSTOV_KITCHEN_CAMPAIGN_ID]}},
            },
            "summary": {"campaign_id": ROSTOV_KITCHEN_CAMPAIGN_ID, "action": "resume"},
        }

    if operation == "suspend_campaign_v1":
        if str(campaign.get("State") or "") != "ON":
            return {
                "ok": False,
                "blocked_by_precondition": True,
                "reason": "campaign_not_on",
                "state": campaign.get("State"),
            }
        return {
            "service": "campaigns",
            "payload": {
                "method": "suspend",
                "params": {"SelectionCriteria": {"Ids": [ROSTOV_KITCHEN_CAMPAIGN_ID]}},
            },
            "summary": {"campaign_id": ROSTOV_KITCHEN_CAMPAIGN_ID, "action": "suspend"},
        }

    raise AssertionError("Unreachable Rostov operation branch.")


async def _rostov_verify_operation(operation: str) -> dict[str, Any]:
    if operation == "set_campaign_negatives_fixed_v2":
        campaign = await _rostov_campaign_get()
        actual = ((campaign or {}).get("NegativeKeywords") or {}).get("Items") or []
        exp = {_rostov_norm_text(x) for x in ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES}
        got = {_rostov_norm_text(x) for x in actual}
        return {
            "ok": got == exp,
            "expected_count": len(exp),
            "actual_count": len(got),
        }

    group_name = _rostov_group_name_from_fixed_operation(operation, "suspend_autotargeting")
    if group_name:
        groups = await _rostov_group_map(require_all=True)
        gid = int(groups[group_name]["Id"])
        keywords = await _rostov_keywords_get()
        autos = [
            k for k in keywords
            if int(k.get("AdGroupId") or 0) == gid
            and str(k.get("Keyword") or "") == "---autotargeting"
        ]
        return {
            "ok": len(autos) == 1 and str(autos[0].get("State") or "") == "SUSPENDED",
            "group": group_name,
            "autotargeting": [
                {"Id": k.get("Id"), "State": k.get("State")} for k in autos
            ],
        }

    group_name = _rostov_group_name_from_fixed_operation(operation, "add_keywords")
    if group_name:
        groups = await _rostov_group_map(require_all=True)
        gid = int(groups[group_name]["Id"])
        keywords = await _rostov_keywords_get()
        actual = {
            _rostov_norm_text(k.get("Keyword"))
            for k in keywords
            if int(k.get("AdGroupId") or 0) == gid
            and str(k.get("Keyword") or "") != "---autotargeting"
        }
        expected = {_rostov_norm_text(x) for x in ROSTOV_KITCHEN_KEYWORDS[group_name]}
        auto_check = _rostov_autotargeting_check(keywords, [gid])
        return {
            "ok": actual == expected and auto_check["ok"],
            "autotargeting_check": auto_check,
            "group": group_name,
            "expected_count": len(expected),
            "actual_count": len(actual),
            "missing": sorted(expected - actual),
            "extra": sorted(actual - expected),
        }

    if operation == "create_sitelinks_fixed_v1":
        found = await _rostov_find_sitelink_set()
        return {"ok": len(found) == 1, "ids": [x.get("Id") for x in found]}

    if operation == "create_callouts_fixed_v1":
        current = await _rostov_callout_map()
        missing = [x for x in ROSTOV_KITCHEN_CALLOUTS if x not in current]
        return {
            "ok": not missing,
            "missing": missing,
            "ids": {x: current[x].get("Id") for x in ROSTOV_KITCHEN_CALLOUTS if x in current},
        }

    group_name = _rostov_group_name_from_fixed_operation(operation, "create_ad")
    if group_name:
        groups = await _rostov_group_map(require_all=True)
        gid = int(groups[group_name]["Id"])
        ads = await _rostov_ads_get()
        matches = [a for a in ads if int(a.get("AdGroupId") or 0) == gid]
        return {
            "ok": len(matches) == 1,
            "group": group_name,
            "ads": [
                {"Id": a.get("Id"), "Status": a.get("Status"), "State": a.get("State")}
                for a in matches
            ],
        }

    if operation == "submit_ads_moderation_fixed_v1":
        ads = await _rostov_ads_get()
        statuses = [str(a.get("Status") or "") for a in ads]
        return {
            "ok": len(ads) == 6 and all(
                x in {"MODERATION", "PREACCEPTED", "ACCEPTED"} for x in statuses
            ),
            "ads": [
                {"Id": a.get("Id"), "Status": a.get("Status"), "State": a.get("State")}
                for a in ads
            ],
        }

    if operation == "resume_campaign_fixed_v1":
        campaign = await _rostov_campaign_get()
        return {
            "ok": bool(campaign) and str(campaign.get("State") or "") == "ON",
            "campaign": campaign,
        }

    if operation == "create_groups_v1":
        groups = await _rostov_group_map(require_all=False)
        missing = [n for n in ROSTOV_KITCHEN_GROUP_NAMES if n not in groups]
        wrong_geo = [
            n for n in ROSTOV_KITCHEN_GROUP_NAMES
            if n in groups and list(groups[n].get("RegionIds") or []) != ROSTOV_KITCHEN_REGION_IDS
        ]
        return {
            "ok": not missing and not wrong_geo,
            "missing": missing,
            "wrong_geo": wrong_geo,
            "groups": [
                {"Id": groups[n].get("Id"), "Name": n, "RegionIds": groups[n].get("RegionIds")}
                for n in ROSTOV_KITCHEN_GROUP_NAMES if n in groups
            ],
        }

    if operation == "set_campaign_negatives_v1":
        campaign = await _rostov_campaign_get()
        actual = ((campaign or {}).get("NegativeKeywords") or {}).get("Items") or []
        expected = {_rostov_norm_text(x) for x in ROSTOV_KITCHEN_CAMPAIGN_NEGATIVES}
        got = {_rostov_norm_text(x) for x in actual}
        return {"ok": got == expected, "expected_count": len(expected), "actual_count": len(got)}

    if operation == "sync_regions_from_group_06_v1":
        groups = await _rostov_group_map(require_all=True)
        source_regions = list(groups["06 Конфигурация"].get("RegionIds") or [])
        mismatched = [n for n in ROSTOV_KITCHEN_GROUP_NAMES if list(groups[n].get("RegionIds") or []) != source_regions]
        return {"ok": source_regions == ROSTOV_KITCHEN_REGION_IDS and not mismatched, "region_ids": source_regions, "mismatched": mismatched}

    if operation == "suspend_autotargeting_v1":
        keywords = await _rostov_keywords_get()
        autos = [k for k in keywords if str(k.get("Keyword") or "") == "---autotargeting"]
        return {
            "ok": len(autos) == 6 and all(str(k.get("State") or "") == "SUSPENDED" for k in autos),
            "autotargetings": [{"Id": k.get("Id"), "AdGroupId": k.get("AdGroupId"), "State": k.get("State")} for k in autos],
        }

    if operation == "resume_autotargeting_v1":
        keywords = await _rostov_keywords_get()
        autos = [k for k in keywords if str(k.get("Keyword") or "") == "---autotargeting"]
        return {
            "ok": len(autos) == 6 and all(str(k.get("State") or "") == "ON" for k in autos),
            "autotargetings": [{"Id": k.get("Id"), "AdGroupId": k.get("AdGroupId"), "State": k.get("State")} for k in autos],
        }

    if operation == "add_keywords_v1":
        groups = await _rostov_group_map(require_all=True)
        keywords = await _rostov_keywords_get()
        actual = {}
        for name in ROSTOV_KITCHEN_GROUP_NAMES:
            gid = int(groups[name]["Id"])
            actual[name] = {
                _rostov_norm_text(k.get("Keyword"))
                for k in keywords
                if int(k.get("AdGroupId") or 0) == gid and str(k.get("Keyword") or "") != "---autotargeting"
            }
        missing = {}
        for name in ROSTOV_KITCHEN_GROUP_NAMES:
            exp = {_rostov_norm_text(x) for x in ROSTOV_KITCHEN_KEYWORDS[name]}
            diff = sorted(exp - actual.get(name, set()))
            if diff:
                missing[name] = diff
        autotargeting = [k for k in keywords if str(k.get("Keyword") or "") == "---autotargeting"]
        auto_check = _rostov_autotargeting_check(
            keywords, [int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES]
        )
        return {
            "ok": not missing and auto_check["ok"],
            "autotargeting_check": auto_check,
            "missing": missing,
            "autotargeting_count": len(autotargeting),
            "autotargeting_states": [{"Id": k.get("Id"), "State": k.get("State")} for k in autotargeting],
            "keyword_count": len([k for k in keywords if str(k.get("Keyword") or "") != "---autotargeting"]),
        }

    if operation == "create_sitelinks_v1":
        found = await _rostov_find_sitelink_set()
        return {"ok": len(found) == 1, "ids": [x.get("Id") for x in found]}

    if operation == "create_callouts_v1":
        current = await _rostov_callout_map()
        missing = [x for x in ROSTOV_KITCHEN_CALLOUTS if x not in current]
        return {
            "ok": not missing,
            "missing": missing,
            "ids": {x: current[x].get("Id") for x in ROSTOV_KITCHEN_CALLOUTS if x in current},
        }

    if operation == "create_ads_balanced_v1":
        expected = await _rostov_expected_ads_by_group_id()
        ads = await _rostov_ads_get()
        matched = {}
        for ad in ads:
            gid = int(ad.get("AdGroupId") or 0)
            if gid not in expected:
                continue
            ta = ad.get("TextAd") or {}
            e = expected[gid]
            if (
                str(ta.get("Title") or "") == e["Title"]
                and str(ta.get("Title2") or "") == e["Title2"]
                and str(ta.get("Text") or "") == e["Text"]
                and str(ta.get("Href") or "") == ROSTOV_KITCHEN_LANDING + ROSTOV_KITCHEN_UTM
            ):
                matched[gid] = ad
        missing = [gid for gid in expected if gid not in matched]
        return {
            "ok": not missing and len(matched) == 6,
            "missing_group_ids": missing,
            "ads": [
                {"Id": a.get("Id"), "AdGroupId": a.get("AdGroupId"), "Status": a.get("Status"), "State": a.get("State")}
                for a in matched.values()
            ],
        }

    if operation == "submit_ads_moderation_v1":
        ads = await _rostov_ads_get()
        statuses = [str(a.get("Status") or "") for a in ads]
        # After submit, ads may already be MODERATION, PREACCEPTED or ACCEPTED.
        ok = len(ads) == 6 and all(x in {"MODERATION", "PREACCEPTED", "ACCEPTED"} for x in statuses)
        return {
            "ok": ok,
            "ads": [
                {"Id": a.get("Id"), "Status": a.get("Status"), "State": a.get("State")}
                for a in ads
            ],
        }

    if operation == "resume_campaign_v1":
        campaign = await _rostov_campaign_get()
        return {"ok": bool(campaign) and str(campaign.get("State") or "") == "ON", "campaign": campaign}

    if operation == "suspend_campaign_v1":
        campaign = await _rostov_campaign_get()
        return {"ok": bool(campaign) and str(campaign.get("State") or "") != "ON", "campaign": campaign}

    return {"ok": False, "reason": "unknown_operation"}


@mcp.tool(
    title="Rostov kitchen launch status",
    description=(
        "READ-ONLY status of the fixed Rostov kitchen Direct launch: campaign, six planned "
        "groups, keywords/autotargeting, ads, sitelinks and callouts. Makes no changes."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def rostov_kitchen_launch_status() -> dict[str, Any]:
    campaign = await _rostov_campaign_get()
    groups = await _rostov_group_map(require_all=False)
    keywords = await _rostov_keywords_get()
    ads = await _rostov_ads_get()
    sitelinks = await _rostov_find_sitelink_set()
    callouts = await _rostov_callout_map()
    return {
        "ok": True,
        "campaign": campaign,
        "planned_groups": [
            {
                "name": n,
                "exists": n in groups,
                "id": (groups.get(n) or {}).get("Id"),
                "region_ids": (groups.get(n) or {}).get("RegionIds"),
            }
            for n in ROSTOV_KITCHEN_GROUP_NAMES
        ],
        "autotargeting_check": _rostov_autotargeting_check(
            keywords, [int(groups[n]["Id"]) for n in ROSTOV_KITCHEN_GROUP_NAMES if n in groups]
        ),
        "keyword_count": len(keywords),
        "autotargeting_count": len([k for k in keywords if str(k.get("Keyword") or "") == "---autotargeting"]),
        "autotargeting_on_count": len([k for k in keywords if str(k.get("Keyword") or "") == "---autotargeting" and str(k.get("State") or "") == "ON"]),
        "autotargeting_suspended_count": len([k for k in keywords if str(k.get("Keyword") or "") == "---autotargeting" and str(k.get("State") or "") == "SUSPENDED"]),
        "ads": [
            {
                "Id": a.get("Id"),
                "AdGroupId": a.get("AdGroupId"),
                "Status": a.get("Status"),
                "State": a.get("State"),
                "StatusClarification": a.get("StatusClarification"),
            }
            for a in ads
        ],
        "sitelink_set_ids": [x.get("Id") for x in sitelinks],
        "callout_ids": {k: v.get("Id") for k, v in callouts.items()},
        "allowed_write_operations": sorted(ROSTOV_KITCHEN_ALLOWED_OPERATIONS),
    }


async def _rostov_fixed_preview(operation: str) -> dict[str, Any]:
    prepared = await _rostov_prepare_operation(operation)
    if prepared.get("ok") is False:
        return {"preview_only": True, "fixed_operation": operation, **prepared}
    preview = _make_marketing_plan(
        "direct",
        "POST",
        prepared["service"],
        params={"fixed_operation": operation, "scope": "rostov_kitchen_launch_v1"},
        payload=prepared["payload"],
    )
    preview["fixed_operation"] = operation
    preview["scope"] = "rostov_kitchen_launch_v1"
    preview["summary"] = prepared["summary"]
    return preview


@mcp.tool(
    title="Preview fixed Rostov campaign negatives",
    description=(
        "PREVIEW ONLY. No arguments. Prepares the exact hard-coded 43 campaign-level negative "
        "phrases approved for campaign 714564245. No arbitrary negative list is accepted."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_negatives_fixed_v2() -> dict[str, Any]:
    return await _rostov_fixed_preview("set_campaign_negatives_fixed_v2")


@mcp.tool(
    title="Preview fixed Rostov sitelinks",
    description="PREVIEW ONLY. No arguments. Creates only the four hard-coded Rostov kitchen sitelinks.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_sitelinks_fixed_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_sitelinks_fixed_v1")


@mcp.tool(
    title="Preview fixed Rostov callouts",
    description="PREVIEW ONLY. No arguments. Creates only the hard-coded Rostov kitchen callouts.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_callouts_fixed_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_callouts_fixed_v1")


@mcp.tool(
    title="Preview fixed Rostov ads moderation",
    description="PREVIEW ONLY. No arguments. Submits exactly the six fixed Rostov kitchen ads for moderation after they exist.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_moderation_fixed_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("submit_ads_moderation_fixed_v1")


@mcp.tool(
    title="Preview fixed Rostov campaign resume",
    description="PREVIEW ONLY. No arguments. Resumes only campaign 714564245 after all six ads are accepted.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_resume_fixed_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("resume_campaign_fixed_v1")


@mcp.tool(
    title="Preview suspend autotargeting group 01",
    description="PREVIEW ONLY. No arguments. Suspends only the autotargeting keyword in Rostov group 01: 01 Кухни на заказ.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_suspend_auto_core_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("suspend_autotargeting_g01_v1")


@mcp.tool(
    title="Preview add keywords group 01",
    description="PREVIEW ONLY. No arguments. Adds only the hard-coded manual keyword list to Rostov group 01: 01 Кухни на заказ.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_keywords_core_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("add_keywords_g01_v1")


@mcp.tool(
    title="Preview create ad group 01",
    description="PREVIEW ONLY. No arguments. Creates exactly one hard-coded text ad in Rostov group 01: 01 Кухни на заказ.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_ad_core_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_ad_g01_v1")


@mcp.tool(
    title="Preview suspend autotargeting group 02",
    description="PREVIEW ONLY. No arguments. Suspends only the autotargeting keyword in Rostov group 02: 02 Ростов.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_suspend_auto_geo_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("suspend_autotargeting_g02_v1")


@mcp.tool(
    title="Preview add keywords group 02",
    description="PREVIEW ONLY. No arguments. Adds only the hard-coded manual keyword list to Rostov group 02: 02 Ростов.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_keywords_geo_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("add_keywords_g02_v1")


@mcp.tool(
    title="Preview create ad group 02",
    description="PREVIEW ONLY. No arguments. Creates exactly one hard-coded text ad in Rostov group 02: 02 Ростов.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_ad_geo_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_ad_g02_v1")


@mcp.tool(
    title="Preview suspend autotargeting group 03",
    description="PREVIEW ONLY. No arguments. Suspends only the autotargeting keyword in Rostov group 03: 03 По размерам.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_suspend_auto_dimensions_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("suspend_autotargeting_g03_v1")


@mcp.tool(
    title="Preview add keywords group 03",
    description="PREVIEW ONLY. No arguments. Adds only the hard-coded manual keyword list to Rostov group 03: 03 По размерам.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_keywords_dimensions_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("add_keywords_g03_v1")


@mcp.tool(
    title="Preview create ad group 03",
    description="PREVIEW ONLY. No arguments. Creates exactly one hard-coded text ad in Rostov group 03: 03 По размерам.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_ad_dimensions_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_ad_g03_v1")


@mcp.tool(
    title="Preview suspend autotargeting group 04",
    description="PREVIEW ONLY. No arguments. Suspends only the autotargeting keyword in Rostov group 04: 04 Цена.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_suspend_auto_price_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("suspend_autotargeting_g04_v1")


@mcp.tool(
    title="Preview add keywords group 04",
    description="PREVIEW ONLY. No arguments. Adds only the hard-coded manual keyword list to Rostov group 04: 04 Цена.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_keywords_price_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("add_keywords_g04_v1")


@mcp.tool(
    title="Preview create ad group 04",
    description="PREVIEW ONLY. No arguments. Creates exactly one hard-coded text ad in Rostov group 04: 04 Цена.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_ad_price_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_ad_g04_v1")


@mcp.tool(
    title="Preview suspend autotargeting group 05",
    description="PREVIEW ONLY. No arguments. Suspends only the autotargeting keyword in Rostov group 05: 05 Расчёт.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_suspend_auto_calculation_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("suspend_autotargeting_g05_v1")


@mcp.tool(
    title="Preview add keywords group 05",
    description="PREVIEW ONLY. No arguments. Adds only the hard-coded manual keyword list to Rostov group 05: 05 Расчёт.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_keywords_calculation_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("add_keywords_g05_v1")


@mcp.tool(
    title="Preview create ad group 05",
    description="PREVIEW ONLY. No arguments. Creates exactly one hard-coded text ad in Rostov group 05: 05 Расчёт.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_ad_calculation_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_ad_g05_v1")


@mcp.tool(
    title="Preview suspend autotargeting group 06",
    description="PREVIEW ONLY. No arguments. Suspends only the autotargeting keyword in Rostov group 06: 06 Конфигурация.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_suspend_auto_configuration_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("suspend_autotargeting_g06_v1")


@mcp.tool(
    title="Preview add keywords group 06",
    description="PREVIEW ONLY. No arguments. Adds only the hard-coded manual keyword list to Rostov group 06: 06 Конфигурация.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_keywords_configuration_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("add_keywords_g06_v1")


@mcp.tool(
    title="Preview create ad group 06",
    description="PREVIEW ONLY. No arguments. Creates exactly one hard-coded text ad in Rostov group 06: 06 Конфигурация.",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_ad_configuration_v1() -> dict[str, Any]:
    return await _rostov_fixed_preview("create_ad_g06_v1")


@mcp.tool(
    title="Preview Rostov kitchen launch operation",
    description=(
        "PREVIEW ONLY for the fixed Rostov kitchen Direct launch. The only allowed operation "
        "names are create_groups_v1, sync_regions_from_group_06_v1, suspend_autotargeting_v1, resume_autotargeting_v1, set_campaign_negatives_v1, add_keywords_v1, "
        "create_sitelinks_v1, create_callouts_v1, create_ads_balanced_v1, "
        "submit_ads_moderation_v1, resume_campaign_v1, suspend_campaign_v1. "
        "The caller cannot choose a Direct service, campaign id, group id, region, keyword, "
        "URL, ad copy, budget or arbitrary API payload."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_rostov_kitchen_launch_operation(
    operation: Annotated[str, Field(min_length=3, max_length=80)],
) -> dict[str, Any]:
    prepared = await _rostov_prepare_operation(operation)
    if prepared.get("ok") is False:
        return {"preview_only": True, "operation": operation, **prepared}

    preview = _make_marketing_plan(
        "direct",
        "POST",
        prepared["service"],
        params={"fixed_operation": operation, "scope": "rostov_kitchen_launch_v1"},
        payload=prepared["payload"],
    )
    preview["fixed_operation"] = operation
    preview["scope"] = "rostov_kitchen_launch_v1"
    preview["summary"] = prepared["summary"]
    return preview


@mcp.tool(
    title="Apply Rostov kitchen launch operation",
    description=(
        "WRITE. Applies exactly one immutable, explicitly confirmed plan previously created "
        "by preview_rostov_kitchen_launch_operation for the fixed Rostov kitchen campaign "
        "714564245. The caller supplies only plan_id and confirmation_token and cannot alter "
        "campaign, groups, regions, keywords, URLs, ad copy, budget or API payload at apply time."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True),
)
async def apply_rostov_kitchen_launch_operation(
    plan_id: Annotated[str, Field(min_length=1, max_length=120)],
    confirmation_token: Annotated[str, Field(min_length=1, max_length=256)],
) -> dict[str, Any]:
    store = _prune_marketing_plans(_load_marketing_plan_store())
    plan = store.get(plan_id)
    if not plan:
        raise ValueError("Rostov launch plan not found or expired.")
    meta = plan.get("params") or {}
    operation = str(meta.get("fixed_operation") or "")
    if meta.get("scope") != "rostov_kitchen_launch_v1" or operation not in ROSTOV_KITCHEN_ALLOWED_OPERATIONS:
        raise ValueError("Plan is not a permitted Rostov kitchen launch plan.")
    if plan.get("api") != "direct" or plan.get("method") != "POST":
        raise ValueError("Rostov launch plan has an invalid API or HTTP method.")

    # Rebuild the expected operation immediately before mutation.
    # If state changed after preview, refuse rather than blindly applying stale intent.
    prepared = await _rostov_prepare_operation(operation)
    if prepared.get("ok") is False:
        raise ValueError(
            f"Precondition changed before apply: {prepared.get('reason')}; refusing write."
        )
    if plan.get("target") != prepared["service"] or plan.get("payload") != prepared["payload"]:
        raise ValueError("Immutable Rostov launch plan no longer matches current fixed operation.")

    try:
        applied = await _apply_confirmed_marketing_plan(
            plan_id,
            confirmation_token,
            required_api="direct",
        )
    except Exception:
        # Never retry a write automatically. First read back the actual state.
        verification = await _rostov_verify_operation(operation)
        if verification.get("ok"):
            return {
                "ok": True,
                "applied": "verified_after_ambiguous_error",
                "fixed_operation": operation,
                "verification": verification,
                "warning": "Write response was ambiguous, but read-back verified the intended state. No retry performed.",
            }
        raise

    verification = await _rostov_verify_operation(operation)
    if not verification.get("ok"):
        return {
            "ok": False,
            "applied": "post_write_verification_failed",
            "fixed_operation": operation,
            "verification": verification,
            "direct_result": applied,
            "warning": "No automatic retry will be performed.",
        }
    return {
        "ok": True,
        "applied": True,
        "fixed_operation": operation,
        "verification": verification,
        "direct_result": applied,
        "post_write_verified": True,
    }


def _validate_rostov_custom_negatives(items: list[str]) -> list[str]:
    if not isinstance(items, list) or not items:
        raise ValueError("negative_keywords must be a non-empty list.")
    if len(items) > 200:
        raise ValueError("Too many negative keywords for this scoped tool (max 200).")
    cleaned = []
    seen = set()
    for raw in items:
        value = re.sub(r"\s+", " ", str(raw or "").strip())
        if not value:
            continue
        if len(value) > 80:
            raise ValueError(f"Negative keyword is too long: {value[:40]}...")
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(value)
    if not cleaned:
        raise ValueError("No usable negative keywords after normalization.")
    return cleaned


@mcp.tool(
    title="Preview Rostov campaign negative keywords",
    description=(
        "PREVIEW ONLY. Sets only campaign-level negative keywords for the fixed Rostov kitchen "
        "campaign 714564245. The caller can provide only the negative keyword list; campaign id, "
        "Direct service, method and all other fields are fixed by the server."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_set_rostov_campaign_negatives(negative_keywords: list[str]) -> dict[str, Any]:
    cleaned = _validate_rostov_custom_negatives(negative_keywords)
    campaign = await _rostov_campaign_get()
    if not campaign:
        return {"ok": False, "preview_only": True, "blocked_by_precondition": True, "reason": "campaign_missing"}
    current = ((campaign.get("NegativeKeywords") or {}).get("Items") or [])
    if {_rostov_norm_text(x) for x in current} == {_rostov_norm_text(x) for x in cleaned}:
        return {"ok": False, "preview_only": True, "blocked_by_precondition": True, "reason": "campaign_negatives_already_match"}
    payload = {
        "method": "update",
        "params": {"Campaigns": [{"Id": ROSTOV_KITCHEN_CAMPAIGN_ID, "NegativeKeywords": {"Items": cleaned}}]},
    }
    preview = _make_marketing_plan(
        "direct", "POST", "campaigns",
        params={"scope": "rostov_campaign_negatives_v1"},
        payload=payload,
    )
    preview["fixed_operation"] = "set_rostov_campaign_negatives_custom_v1"
    preview["scope"] = "rostov_campaign_negatives_v1"
    preview["summary"] = {"negative_keywords": cleaned, "count": len(cleaned)}
    return preview


@mcp.tool(
    title="Apply Rostov campaign negative keywords",
    description=(
        "WRITE. Applies exactly one immutable, explicitly confirmed campaign-negative plan "
        "for fixed campaign 714564245. The caller supplies only plan_id and confirmation_token."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True),
)
async def apply_set_rostov_campaign_negatives(
    plan_id: Annotated[str, Field(min_length=1, max_length=120)],
    confirmation_token: Annotated[str, Field(min_length=1, max_length=256)],
) -> dict[str, Any]:
    store = _prune_marketing_plans(_load_marketing_plan_store())
    plan = store.get(plan_id)
    if not plan:
        raise ValueError("Rostov campaign-negative plan not found or expired.")
    if (
        plan.get("api") != "direct"
        or plan.get("method") != "POST"
        or plan.get("target") != "campaigns"
        or (plan.get("params") or {}).get("scope") != "rostov_campaign_negatives_v1"
    ):
        raise ValueError("Plan is not a permitted Rostov campaign-negative plan.")
    payload = plan.get("payload") or {}
    campaigns = (((payload.get("params") or {}).get("Campaigns")) or [])
    if (
        payload.get("method") != "update"
        or len(campaigns) != 1
        or campaigns[0].get("Id") != ROSTOV_KITCHEN_CAMPAIGN_ID
        or set(campaigns[0].keys()) != {"Id", "NegativeKeywords"}
    ):
        raise ValueError("Campaign-negative plan payload is outside the fixed safe scope.")
    expected = _validate_rostov_custom_negatives(((campaigns[0].get("NegativeKeywords") or {}).get("Items") or []))
    applied = await _apply_confirmed_marketing_plan(plan_id, confirmation_token, required_api="direct")
    campaign = await _rostov_campaign_get()
    actual = ((campaign or {}).get("NegativeKeywords") or {}).get("Items") or []
    ok = {_rostov_norm_text(x) for x in actual} == {_rostov_norm_text(x) for x in expected}
    return {
        "ok": ok,
        "applied": bool(ok),
        "fixed_operation": "set_rostov_campaign_negatives_custom_v1",
        "expected_count": len(expected),
        "actual_count": len(actual),
        "post_write_verified": bool(ok),
        "direct_result": applied,
    }



# =============================================================================
# Rostov RSYA excluded sites controls
# Fixed campaign only. No generic Direct write is exposed.
# Yandex Direct API v5 Campaigns.update documents ExcludedSites as nillable,
# with max 1000 items and max 255 characters per item. Passing null clears it.
# =============================================================================

ROSTOV_RSYA_CAMPAIGN_ID = 714590883
ROSTOV_RSYA_EXCLUDED_SITES_SCOPE = "rostov_rsya_excluded_sites_v1"
ROSTOV_RSYA_EXCLUDED_SITES_MAX_ITEMS = 1000
ROSTOV_RSYA_EXCLUDED_SITE_MAX_LENGTH = 255


def _rostov_rsya_site_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _validate_rostov_rsya_excluded_sites(
    items: list[str],
    *,
    allow_empty: bool,
    argument_name: str,
) -> list[str]:
    if not isinstance(items, list):
        raise ValueError(f"{argument_name} must be a list of strings.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw or "").strip()
        if not value:
            continue
        if len(value) > ROSTOV_RSYA_EXCLUDED_SITE_MAX_LENGTH:
            raise ValueError(
                f"Excluded site is too long (max {ROSTOV_RSYA_EXCLUDED_SITE_MAX_LENGTH} characters): "
                f"{value[:80]}..."
            )
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            raise ValueError("Excluded site contains a control character.")
        # JSON is serialized by the HTTP client, so injection is not possible through quoting;
        # still reject structural/code-like characters that are not needed by domains, app IDs
        # or SSP names and usually indicate malformed input.
        if any(ch in value for ch in '{}[]<>"\\'):
            raise ValueError(f"Excluded site contains a forbidden character: {value}")
        if "://" in value:
            raise ValueError(
                f"Excluded site must be a domain, mobile-app ID or SSP name, not a URL: {value}"
            )
        key = _rostov_rsya_site_key(value)
        if key not in seen:
            seen.add(key)
            cleaned.append(value)

    if len(cleaned) > ROSTOV_RSYA_EXCLUDED_SITES_MAX_ITEMS:
        raise ValueError(
            f"Too many excluded sites (max {ROSTOV_RSYA_EXCLUDED_SITES_MAX_ITEMS})."
        )
    if not allow_empty and not cleaned:
        raise ValueError(f"{argument_name} must contain at least one usable site.")
    return cleaned


def _rostov_rsya_sites_equal(left: list[str], right: list[str]) -> bool:
    return {_rostov_rsya_site_key(x) for x in left} == {
        _rostov_rsya_site_key(x) for x in right
    }


def _rostov_rsya_sites_fingerprint(items: list[str]) -> str:
    canonical = json.dumps(
        sorted({_rostov_rsya_site_key(x) for x in items}),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _sha256_bytes(canonical.encode("utf-8"))


def _marketing_plan_recomputed_sha256(plan: dict[str, Any]) -> str:
    canonical = json.dumps(
        {
            "api": plan.get("api"),
            "method": plan.get("method"),
            "target": plan.get("target"),
            "params": plan.get("params") or {},
            "payload": plan.get("payload"),
            "content_base64": plan.get("content_base64"),
            "content_type": plan.get("content_type"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(canonical.encode("utf-8"))


def _record_marketing_plan_postcheck(plan_id: str, event: str, **data: Any) -> None:
    with _MARKETING_PLAN_LOCK:
        store = _load_marketing_plan_store()
        plan = store.get(plan_id)
        if not plan:
            return
        _marketing_audit_event(plan, event, **data)
        store[plan_id] = plan
        _save_marketing_plan_store(store)


async def _rostov_rsya_campaign_get() -> Optional[dict[str, Any]]:
    response = await _direct_request(
        "campaigns",
        {
            "method": "get",
            "params": {
                "SelectionCriteria": {"Ids": [ROSTOV_RSYA_CAMPAIGN_ID]},
                "FieldNames": [
                    "Id", "Name", "State", "Status", "Type", "ExcludedSites", "StartDate"
                ],
            },
        },
    )
    data = _parse_direct_json(response, operation="campaigns.get rostov rsya excluded sites")
    campaigns = ((data.get("result") or {}).get("Campaigns") or [])
    return campaigns[0] if campaigns else None


def _rostov_rsya_current_sites_from_campaign(campaign: Optional[dict[str, Any]]) -> list[str]:
    if not campaign:
        return []
    raw = (campaign.get("ExcludedSites") or {}).get("Items") or []
    return _validate_rostov_rsya_excluded_sites(
        list(raw), allow_empty=True, argument_name="current ExcludedSites"
    )


def _rostov_rsya_payload_for_sites(sites: list[str]) -> dict[str, Any]:
    # ExcludedSites is nillable in Campaigns.update. Per Direct API's generic
    # nillable rule, null clears the parameter; use it instead of guessing that
    # an empty Items array is accepted.
    excluded_sites: Optional[dict[str, list[str]]]
    excluded_sites = {"Items": sites} if sites else None
    return {
        "method": "update",
        "params": {
            "Campaigns": [{
                "Id": ROSTOV_RSYA_CAMPAIGN_ID,
                "ExcludedSites": excluded_sites,
            }]
        },
    }


def _rostov_rsya_diff(current: list[str], proposed: list[str]) -> tuple[list[str], list[str]]:
    current_keys = {_rostov_rsya_site_key(x) for x in current}
    proposed_keys = {_rostov_rsya_site_key(x) for x in proposed}
    added = [x for x in proposed if _rostov_rsya_site_key(x) not in current_keys]
    removed = [x for x in current if _rostov_rsya_site_key(x) not in proposed_keys]
    return added, removed


async def _preview_rostov_rsya_excluded_sites_plan(
    proposed_sites: list[str],
    *,
    current_sites: list[str],
    mode: str,
) -> dict[str, Any]:
    if _rostov_rsya_sites_equal(current_sites, proposed_sites):
        return {
            "ok": False,
            "preview_only": True,
            "blocked_by_precondition": True,
            "reason": "excluded_sites_already_match",
            "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
            "current_sites": current_sites,
            "proposed_sites": proposed_sites,
        }

    added, removed = _rostov_rsya_diff(current_sites, proposed_sites)
    payload = _rostov_rsya_payload_for_sites(proposed_sites)
    preview = _make_marketing_plan(
        "direct",
        "POST",
        "campaigns",
        params={
            "scope": ROSTOV_RSYA_EXCLUDED_SITES_SCOPE,
            "fixed_operation": "set_rostov_rsya_excluded_sites_v1",
            "mode": mode,
            "base_sites_sha256": _rostov_rsya_sites_fingerprint(current_sites),
        },
        payload=payload,
    )
    preview["fixed_operation"] = "set_rostov_rsya_excluded_sites_v1"
    preview["scope"] = ROSTOV_RSYA_EXCLUDED_SITES_SCOPE
    preview["summary"] = {
        "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
        "mode": mode,
        "current_sites": current_sites,
        "proposed_sites": proposed_sites,
        "added": added,
        "removed": removed,
        "count": len(proposed_sites),
        "clear_via_null": not proposed_sites,
    }
    return preview


@mcp.tool(
    title="Get Rostov RSYA excluded sites",
    description=(
        "READ-ONLY. Returns ExcludedSites only for the fixed Rostov RSYA campaign 714590883. "
        "No campaign id or arbitrary Direct request is accepted."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def get_rostov_rsya_excluded_sites() -> dict[str, Any]:
    campaign = await _rostov_rsya_campaign_get()
    if not campaign:
        return {
            "ok": False,
            "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
            "reason": "campaign_missing",
        }
    sites = _rostov_rsya_current_sites_from_campaign(campaign)
    return {
        "ok": True,
        "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
        "campaign_name": campaign.get("Name"),
        "state": campaign.get("State"),
        "status": campaign.get("Status"),
        "excluded_sites": sites,
        "count": len(sites),
    }


@mcp.tool(
    title="Preview set Rostov RSYA excluded sites",
    description=(
        "PREVIEW ONLY. Replaces ExcludedSites only for fixed campaign 714590883. "
        "The caller provides only the final site/app/SSP list. An empty list prepares a safe "
        "clear using ExcludedSites=null. Campaign id, Direct service, method and payload are fixed server-side."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_set_rostov_rsya_excluded_sites(excluded_sites: list[str]) -> dict[str, Any]:
    proposed = _validate_rostov_rsya_excluded_sites(
        excluded_sites, allow_empty=True, argument_name="excluded_sites"
    )
    campaign = await _rostov_rsya_campaign_get()
    if not campaign:
        return {
            "ok": False,
            "preview_only": True,
            "blocked_by_precondition": True,
            "reason": "campaign_missing",
            "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
        }
    current = _rostov_rsya_current_sites_from_campaign(campaign)
    return await _preview_rostov_rsya_excluded_sites_plan(
        proposed, current_sites=current, mode="set"
    )


@mcp.tool(
    title="Preview add Rostov RSYA excluded sites",
    description=(
        "PREVIEW ONLY. Adds specified sites/apps/SSPs to ExcludedSites for fixed campaign 714590883. "
        "Reads the current list, preserves it, deduplicates values and creates an immutable plan."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_add_rostov_rsya_excluded_sites(sites: list[str]) -> dict[str, Any]:
    additions = _validate_rostov_rsya_excluded_sites(
        sites, allow_empty=False, argument_name="sites"
    )
    campaign = await _rostov_rsya_campaign_get()
    if not campaign:
        return {
            "ok": False,
            "preview_only": True,
            "blocked_by_precondition": True,
            "reason": "campaign_missing",
            "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
        }
    current = _rostov_rsya_current_sites_from_campaign(campaign)
    combined = list(current)
    seen = {_rostov_rsya_site_key(x) for x in current}
    for value in additions:
        key = _rostov_rsya_site_key(value)
        if key not in seen:
            seen.add(key)
            combined.append(value)
    if len(combined) > ROSTOV_RSYA_EXCLUDED_SITES_MAX_ITEMS:
        raise ValueError(
            f"Result would exceed Direct limit of {ROSTOV_RSYA_EXCLUDED_SITES_MAX_ITEMS} excluded sites."
        )
    return await _preview_rostov_rsya_excluded_sites_plan(
        combined, current_sites=current, mode="add"
    )


@mcp.tool(
    title="Preview remove Rostov RSYA excluded sites",
    description=(
        "PREVIEW ONLY. Removes only specified sites/apps/SSPs from ExcludedSites for fixed campaign 714590883. "
        "All other exclusions are preserved. If the result is empty, the plan clears ExcludedSites with null."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True),
)
async def preview_remove_rostov_rsya_excluded_sites(sites: list[str]) -> dict[str, Any]:
    removals = _validate_rostov_rsya_excluded_sites(
        sites, allow_empty=False, argument_name="sites"
    )
    campaign = await _rostov_rsya_campaign_get()
    if not campaign:
        return {
            "ok": False,
            "preview_only": True,
            "blocked_by_precondition": True,
            "reason": "campaign_missing",
            "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
        }
    current = _rostov_rsya_current_sites_from_campaign(campaign)
    remove_keys = {_rostov_rsya_site_key(x) for x in removals}
    proposed = [x for x in current if _rostov_rsya_site_key(x) not in remove_keys]
    return await _preview_rostov_rsya_excluded_sites_plan(
        proposed, current_sites=current, mode="remove"
    )


@mcp.tool(
    title="Apply Rostov RSYA excluded sites",
    description=(
        "WRITE. Applies exactly one immutable, explicitly confirmed ExcludedSites plan for fixed "
        "campaign 714590883. The caller supplies only plan_id and confirmation_token and cannot "
        "choose a campaign, Direct service, method, sites or arbitrary API payload at apply time."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True),
)
async def apply_set_rostov_rsya_excluded_sites(
    plan_id: Annotated[str, Field(min_length=1, max_length=120)],
    confirmation_token: Annotated[str, Field(min_length=1, max_length=256)],
) -> dict[str, Any]:
    store = _prune_marketing_plans(_load_marketing_plan_store())
    plan = store.get(plan_id)
    if not plan:
        raise ValueError("Rostov RSYA excluded-sites plan not found or expired.")

    now = time.time()
    if float(plan.get("expires_at") or 0) <= now:
        raise ValueError("Rostov RSYA excluded-sites plan has expired.")
    if plan.get("state") != "pending" or plan.get("used"):
        raise ValueError(
            f"Rostov RSYA excluded-sites plan is not pending; state={plan.get('state')}."
        )
    if (
        plan.get("api") != "direct"
        or plan.get("method") != "POST"
        or plan.get("target") != "campaigns"
        or (plan.get("params") or {}).get("scope") != ROSTOV_RSYA_EXCLUDED_SITES_SCOPE
        or (plan.get("params") or {}).get("fixed_operation") != "set_rostov_rsya_excluded_sites_v1"
    ):
        raise ValueError("Plan is not a permitted Rostov RSYA excluded-sites plan.")

    recomputed_sha = _marketing_plan_recomputed_sha256(plan)
    if not hmac.compare_digest(str(plan.get("sha256") or ""), recomputed_sha):
        raise ValueError("Rostov RSYA excluded-sites plan integrity hash mismatch.")

    payload = plan.get("payload") or {}
    params = payload.get("params") or {}
    campaigns = params.get("Campaigns") or []
    if payload.get("method") != "update" or len(campaigns) != 1:
        raise ValueError("Excluded-sites plan payload is outside the fixed safe scope.")
    item = campaigns[0]
    if (
        not isinstance(item, dict)
        or item.get("Id") != ROSTOV_RSYA_CAMPAIGN_ID
        or set(item.keys()) != {"Id", "ExcludedSites"}
    ):
        raise ValueError("Excluded-sites plan can mutate only ExcludedSites of campaign 714590883.")

    raw_expected = item.get("ExcludedSites")
    if raw_expected is None:
        expected: list[str] = []
    else:
        if not isinstance(raw_expected, dict) or set(raw_expected.keys()) != {"Items"}:
            raise ValueError("ExcludedSites plan has an invalid structure.")
        expected = _validate_rostov_rsya_excluded_sites(
            raw_expected.get("Items") or [],
            allow_empty=False,
            argument_name="planned ExcludedSites.Items",
        )

    # Protect add/remove/set previews from overwriting a manual change that happened
    # after preview. The preview stores a fingerprint of the exact base list it read.
    campaign_before = await _rostov_rsya_campaign_get()
    if not campaign_before:
        raise ValueError("Rostov RSYA campaign is missing before apply.")
    current_before = _rostov_rsya_current_sites_from_campaign(campaign_before)
    base_sha = str((plan.get("params") or {}).get("base_sites_sha256") or "")
    if not base_sha or not hmac.compare_digest(
        base_sha, _rostov_rsya_sites_fingerprint(current_before)
    ):
        raise ValueError(
            "ExcludedSites changed after preview. Refusing stale write; create a new preview."
        )

    try:
        applied = await _apply_confirmed_marketing_plan(
            plan_id, confirmation_token, required_api="direct"
        )
    except Exception:
        # Never retry automatically. Read back once because an HTTP/transport error can be
        # ambiguous: Yandex may have committed the update even if the response was lost.
        campaign_after_error = await _rostov_rsya_campaign_get()
        actual_after_error = _rostov_rsya_current_sites_from_campaign(campaign_after_error)
        if _rostov_rsya_sites_equal(actual_after_error, expected):
            _record_marketing_plan_postcheck(
                plan_id,
                "post_write_verified_after_ambiguous_error",
                campaign_id=ROSTOV_RSYA_CAMPAIGN_ID,
                expected_count=len(expected),
                actual_count=len(actual_after_error),
            )
            return {
                "ok": True,
                "applied": True,
                "applied_state": "verified_after_ambiguous_error",
                "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
                "expected_sites": expected,
                "actual_sites": actual_after_error,
                "post_write_verified": True,
                "warning": "Write response was ambiguous; no retry was performed. Read-back matched the intended state.",
            }
        raise

    campaign_after = await _rostov_rsya_campaign_get()
    actual = _rostov_rsya_current_sites_from_campaign(campaign_after)
    verified = _rostov_rsya_sites_equal(actual, expected)
    _record_marketing_plan_postcheck(
        plan_id,
        "post_write_verification",
        campaign_id=ROSTOV_RSYA_CAMPAIGN_ID,
        expected_count=len(expected),
        actual_count=len(actual),
        verified=bool(verified),
    )
    return {
        "ok": bool(verified),
        "applied": True,
        "campaign_id": ROSTOV_RSYA_CAMPAIGN_ID,
        "expected_sites": expected,
        "actual_sites": actual,
        "post_write_verified": bool(verified),
        "direct_result": applied,
    }


@mcp.tool(
    title="Marketing plan diagnostics",
    description=(
        "READ-ONLY diagnostics for recent Yandex Direct/Metrica preview/apply plans. "
        "Returns state, expiry, operation hash and safe event history; never returns "
        "OAuth secrets or confirmation tokens and cannot execute a plan."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
)
async def marketing_plan_diagnostics(
    plan_id: Optional[str] = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
) -> dict[str, Any]:
    store = _prune_marketing_plans(_load_marketing_plan_store())
    items = []
    for pid, plan in store.items():
        if plan_id and pid != plan_id:
            continue
        items.append({
            "plan_id": pid,
            "api": plan.get("api"),
            "method": plan.get("method"),
            "target": plan.get("target"),
            "operation_sha256": plan.get("sha256"),
            "state": plan.get("state"),
            "used": bool(plan.get("used")),
            "created_at": plan.get("created_at"),
            "expires_at": plan.get("expires_at"),
            "terminal_at": plan.get("terminal_at"),
            "result_summary": plan.get("result_summary"),
            "events": plan.get("events", []),
        })
    items.sort(key=lambda x: float(x.get("created_at") or 0), reverse=True)
    return {
        "ok": True,
        "plan_store": "durable_atomic_file",
        "plan_ttl_seconds": MARKETING_PLAN_TTL_SECONDS,
        "history_seconds": MARKETING_PLAN_HISTORY_SECONDS,
        "items": items[:limit],
    }


async def _apply_confirmed_direct_write_internal(
    plan_id: Annotated[str, Field(min_length=1, max_length=120, description="Opaque immutable Direct plan id returned by preview_direct_write.")],
    confirmation_token: Annotated[str, Field(min_length=1, max_length=256, description="One-time token for this exact Direct preview after explicit user approval.")],
) -> Any:
    return await _apply_confirmed_marketing_plan(
        plan_id, confirmation_token, required_api="direct"
    )


async def _apply_confirmed_metrika_write_internal(
    plan_id: Annotated[str, Field(min_length=1, max_length=120, description="Opaque immutable Metrica plan id returned by preview_metrika_write.")],
    confirmation_token: Annotated[str, Field(min_length=1, max_length=256, description="One-time token for this exact Metrica preview after explicit user approval.")],
) -> Any:
    return await _apply_confirmed_marketing_plan(
        plan_id, confirmation_token, required_api="metrika"
    )


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


@mcp.tool(
    title="List allowed WordPress theme files",
    description=(
        "List existing files that the WordPress Bridge permits for theme-file reads. "
        "READ-ONLY: this operation does not create, modify, delete, execute, or write "
        "WordPress files, database records, settings, or external resources."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def list_theme_files() -> dict[str, Any]:
    result = await _wp("GET", f"{NS}/theme/files")
    return result["data"]


@mcp.tool(
    title="Read allowed WordPress theme file",
    description=(
        "Read and return the contents of one existing allow-listed text/source file "
        "inside the configured active or parent WordPress theme. READ-ONLY: the tool "
        "uses an HTTP GET request and does not create, modify, replace, patch, delete, "
        "rename, upload, execute, or evaluate files or PHP code; it does not change the "
        "WordPress database, options, configuration, cache, or any external resource. "
        "The caller supplies only a relative theme path. The gateway rejects absolute "
        "paths, '..' traversal, backslashes, secret/config filenames, and non-text "
        "extensions; the WordPress endpoint must independently enforce canonical-path "
        "containment within its allow-listed theme roots."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def read_theme_file(
    path: ThemeRelativePath,
    start_line: Annotated[int, Field(ge=1, description="1-based first source line to return.")] = 1,
    max_lines: Annotated[int, Field(ge=1, le=MAX_READ_MAX_LINES, description="Maximum source lines to return in this chunk.")] = DEFAULT_READ_MAX_LINES,
) -> dict[str, Any]:
    safe_path = _validate_theme_relative_path(path)
    result = await _wp("GET", f"{NS}/theme/file", params={"path": safe_path})
    return _bounded_text_result(
        result["data"],
        start_line=start_line,
        max_lines=max_lines,
    )


@mcp.tool(
    title="Read About page template",
    description=(
        "Read the current WordPress theme file template-o-nas.php only. "
        "READ-ONLY: this tool has no arguments, uses a fixed allow-listed relative path, "
        "performs only an HTTP GET request, and cannot select, create, modify, replace, "
        "patch, delete, rename, upload, execute, or evaluate any file or PHP code. "
        "It does not change WordPress content, database records, options, settings, cache, "
        "configuration, or any external resource."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def read_about_template(
    start_line: Annotated[int, Field(ge=1, description="1-based first source line to return.")] = 1,
    max_lines: Annotated[int, Field(ge=1, le=MAX_READ_MAX_LINES, description="Maximum source lines to return in this chunk.")] = DEFAULT_READ_MAX_LINES,
) -> dict[str, Any]:
    """Read a lossless line window from template-o-nas.php only."""
    fixed_path = "template-o-nas.php"
    result = await _wp("GET", f"{NS}/theme/file", params={"path": fixed_path})
    return _bounded_text_result(
        result["data"],
        start_line=start_line,
        max_lines=max_lines,
    )


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
    safe_path = _validate_theme_relative_path(path)
    safe_sha256 = _validate_sha256_hex(expected_sha256)
    payload = {
        "path": safe_path,
        "expected_sha256": safe_sha256,
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
    safe_path = _validate_theme_relative_path(path)
    safe_sha256 = _validate_sha256_hex(expected_sha256)
    if expected_occurrences < 1:
        raise ValueError("expected_occurrences must be >= 1.")
    payload = {
        "path": safe_path,
        "expected_sha256": safe_sha256,
        "search_base64": search_base64,
        "replace_base64": replacement_base64,
        "expected_occurrences": expected_occurrences,
    }
    result = await _wp("POST", f"{NS}/theme/file/patch/preview", payload=payload)
    return result["data"]


@mcp.tool(
    title="Preview fixed About-page trust slider repair",
    description=(
        "Prepare one fixed, predefined repair for the About page trust-review slider. "
        "The caller can provide only the current SHA-256 of template-o-nas.php. The file path, "
        "search fragment, PHP, JavaScript, and replacement content are hard-coded in this gateway; "
        "the caller cannot supply or alter code. PREVIEW ONLY: this creates an immutable server-side "
        "plan and does not write the theme file."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def preview_about_trust_slider_fix(
    expected_sha256: Annotated[
        str,
        Field(
            min_length=64,
            max_length=64,
            description="Current SHA-256 of template-o-nas.php before preparing the fixed slider repair.",
        ),
    ],
) -> dict[str, Any]:
    """Prepare the fixed review-slider repair without accepting arbitrary path or code input."""
    search = '            <div class="ymb-trust__dots" aria-hidden="true"><span class="is-active"></span><span></span><span></span><span></span><span></span></div>'

    replacement = r'''<?php
        $trust_reviews = [];
        if (isset($reviews) && is_array($reviews)) {
            foreach ($reviews as $review) {
                if (!is_array($review)) {
                    continue;
                }

                $review_text = trim((string) ($review['review_text'] ?? ''));
                if ($review_text === '') {
                    continue;
                }

                $trust_reviews[] = [
                    'text' => $review_text,
                    'name' => trim((string) ($review['review_name'] ?? '')),
                    'city' => trim((string) ($review['review_city'] ?? '')),
                ];
            }
        }

        $trust_dots_count = max(1, count($trust_reviews));
        ?>
        <div class="ymb-trust__dots" aria-hidden="true">
            <?php for ($trust_dot_index = 0; $trust_dot_index < $trust_dots_count; $trust_dot_index++) : ?>
                <span<?php echo $trust_dot_index === 0 ? ' class="is-active"' : ''; ?>></span>
            <?php endfor; ?>
        </div>

        <?php if (count($trust_reviews) > 1) : ?>
            <script>
            (() => {
                const reviews = <?php echo wp_json_encode(
                    $trust_reviews,
                    JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT
                ); ?>;
                const section = document.querySelector('.ymb-trust');
                if (!section || !Array.isArray(reviews) || reviews.length < 2) {
                    return;
                }

                const slider = section.querySelector('.ymb-trust__slider');
                const card = section.querySelector('.ymb-trust__card');
                const text = card ? card.querySelector('p') : null;
                const arrows = slider ? slider.querySelectorAll('.ymb-trust__arrow') : [];
                const dots = section.querySelectorAll('.ymb-trust__dots span');

                if (!slider || !card || !text || arrows.length < 2) {
                    return;
                }

                let author = card.querySelector('.ymb-trust__author');
                let currentIndex = 0;

                const renderReview = (index) => {
                    const review = reviews[index];
                    if (!review) {
                        return;
                    }

                    text.textContent = review.text || '';
                    const authorText = [review.name, review.city].filter(Boolean).join(', ');

                    if (authorText) {
                        if (!author) {
                            author = document.createElement('div');
                            author.className = 'ymb-trust__author';
                            card.appendChild(author);
                        }
                        author.textContent = authorText;
                    } else if (author) {
                        author.textContent = '';
                    }

                    dots.forEach((dot, dotIndex) => {
                        dot.classList.toggle('is-active', dotIndex === index);
                    });
                };

                arrows[0].addEventListener('click', () => {
                    currentIndex = (currentIndex - 1 + reviews.length) % reviews.length;
                    renderReview(currentIndex);
                });

                arrows[1].addEventListener('click', () => {
                    currentIndex = (currentIndex + 1) % reviews.length;
                    renderReview(currentIndex);
                });
            })();
            </script>
        <?php endif; ?>'''

    payload = {
        "path": "template-o-nas.php",
        "expected_sha256": _validate_sha256_hex(expected_sha256),
        "search_base64": base64.b64encode(search.encode("utf-8")).decode("ascii"),
        "replace_base64": base64.b64encode(replacement.encode("utf-8")).decode("ascii"),
        "expected_occurrences": 1,
    }
    result = await _wp("POST", f"{NS}/theme/file/patch/preview", payload=payload)
    return result["data"]


@mcp.tool(
    title="Apply confirmed WordPress theme patch",
    description=(
        "Apply exactly one previously prepared and user-confirmed immutable theme-file plan. "
        "This tool cannot choose a file path, cannot supply PHP/CSS/JS content, cannot create a "
        "new patch, and cannot change the prepared operation. It accepts only the server-issued "
        "plan identifier and the matching confirmation token returned by a prior preview call. "
        "The WordPress Bridge must revalidate the plan, current file SHA, expiry, confirmation "
        "token, backup policy, and syntax checks before writing."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def apply_confirmed_theme_patch(
    plan_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=120,
            description=(
                "Opaque immutable plan identifier returned by preview_theme_file_patch or "
                "preview_theme_file_update. The caller cannot alter the file path or patch "
                "through this value."
            ),
        ),
    ],
    confirmation_token: Annotated[
        str,
        Field(
            min_length=1,
            max_length=256,
            description=(
                "One-time confirmation token issued by the WordPress Bridge for this exact "
                "preview plan after explicit user approval."
            ),
        ),
    ],
) -> dict[str, Any]:
    """Apply only the exact server-side preview plan already approved by the user."""
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
    Доступные операции сейчас: advantages_approved_v1, slogan_style_v1, mobile_polish_v1, approved_mobile_layout_v1, about_desktop_layout_v1.
    Ничего не записывает.
    """
    payload = {
        "expected_sha256": expected_sha256,
        "operations": operations,
    }
    result = await _wp("POST", f"{NS}/about/page/preview", payload=payload)
    return result["data"]


@mcp.tool(
    title="Preview approved About-page desktop layout",
    description=(
        "Prepare the fixed approved desktop layout for template-o-nas.php. "
        "The operation is hard-coded as about_desktop_layout_v1; the caller can provide only "
        "the current SHA-256 and cannot supply arbitrary CSS, PHP, JavaScript, paths, selectors, "
        "or search/replace content. PREVIEW ONLY: no file is written."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def preview_about_desktop_layout(
    expected_sha256: Annotated[
        str,
        Field(min_length=64, max_length=64, description="Current SHA-256 of template-o-nas.php."),
    ],
) -> dict[str, Any]:
    """Prepare only the fixed approved desktop About-page layout."""
    payload = {
        "expected_sha256": _validate_sha256_hex(expected_sha256),
        "operations": ["about_desktop_layout_v1"],
    }
    result = await _wp("POST", f"{NS}/about/page/preview", payload=payload)
    return result["data"]


@mcp.tool()
async def preview_about_mobile_polish(
    expected_sha256: str,
) -> dict[str, Any]:
    """
    Подготовить только утверждённую мобильную доводку страницы «О нас».
    Операция фиксирована как mobile_polish_v1; произвольный CSS/PHP не принимается.
    Ничего не записывает.
    """
    payload = {
        "expected_sha256": expected_sha256,
        "operations": ["mobile_polish_v1"],
    }
    result = await _wp("POST", f"{NS}/about/page/preview", payload=payload)
    return result["data"]


@mcp.tool()
async def preview_about_approved_mobile_layout(
    expected_sha256: str,
) -> dict[str, Any]:
    """
    Подготовить утверждённую мобильную компоновку страницы «О нас».
    Операция фиксирована как approved_mobile_layout_v1; произвольный CSS/PHP не принимается.
    Ничего не записывает.
    """
    payload = {
        "expected_sha256": expected_sha256,
        "operations": ["approved_mobile_layout_v1"],
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
