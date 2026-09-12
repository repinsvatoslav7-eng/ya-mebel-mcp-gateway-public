import os
import base64
import time
import uuid
import hmac
import hashlib
import json
import logging
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
DEFAULT_READ_MAX_LINES = 400
MAX_READ_MAX_LINES = 800
MAX_READ_CHUNK_BYTES = 60 * 1024


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
