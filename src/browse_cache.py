"""
Browse result and thumbnail caching for the media browser.

The remote is slow to show browse pages when every item carries a Kodi ``/image/`` URL: the remote has to fetch
each one itself (with credentials embedded in the URL), and an image missing from Kodi's texture cache makes Kodi
download it from the original source first (e.g. a Plex transcode) before answering.

This module instead:

* downloads thumbnails in the integration, shrinks them and embeds them as small base64 JPEGs (the integration API
  accepts ``data:`` thumbnails up to 32768 characters), with an in-memory LRU cache;
* caches built browse pages for a short time and builds the next page in the background after each request, so
  scrolling to the next page is answered from memory.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable

from aiohttp import ClientSession, ClientTimeout
from ucapi.media_player import BrowseMediaItem

_LOG = logging.getLogger(__name__)

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - only when Pillow is missing from the build
    PIL_AVAILABLE = False
    _LOG.warning("Pillow not available, browse thumbnails will be sent as URLs")

DATA_URI_PREFIX = "data:image/jpeg;base64,"
MAX_THUMBNAIL_LEN = 32768
THUMBNAIL_BOX = (240, 240)
THUMBNAIL_QUALITIES = (75, 60, 45, 30)
FETCH_TIMEOUT = ClientTimeout(total=8)
# How long a browse request waits for thumbnails before answering with whatever is ready.
# Thumbnails still downloading keep going in the background and are cached for the next request.
THUMBNAIL_DEADLINE = 2.5
MAX_CONCURRENT_FETCHES = 6
THUMBNAIL_CACHE_SIZE = 600
PAGE_CACHE_TTL = 90.0
PAGE_CACHE_SIZE = 40

BrowseResult = tuple[BrowseMediaItem, Any]
BrowseFn = Callable[[], Awaitable[BrowseResult | None]]


def _encode_thumbnail(data: bytes) -> str | None:
    """Shrink an image and return it as a base64 JPEG data URI that fits the integration API limit."""
    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail(THUMBNAIL_BOX)
        for quality in THUMBNAIL_QUALITIES:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded = DATA_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode(
                "ascii"
            )
            if len(encoded) <= MAX_THUMBNAIL_LEN:
                return encoded
    return None


class ThumbnailCache:
    """Download, shrink and cache browse thumbnails."""

    def __init__(self, address: str):
        """Create the cache."""
        self._address = address
        self._cache: OrderedDict[str, str | None] = OrderedDict()
        self._pending: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        self._session: ClientSession | None = None

    def _get_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(timeout=FETCH_TIMEOUT)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _store(self, url: str, value: str | None) -> None:
        self._cache[url] = value
        self._cache.move_to_end(url)
        while len(self._cache) > THUMBNAIL_CACHE_SIZE:
            self._cache.popitem(last=False)

    async def _fetch(self, url: str) -> str | None:
        async with self._semaphore:
            try:
                async with self._get_session().get(url) as response:
                    if response.status != 200:
                        _LOG.debug(
                            "[%s] Thumbnail HTTP %s: %s",
                            self._address,
                            response.status,
                            url,
                        )
                        self._store(url, None)
                        return None
                    data = await response.read()
                encoded = await asyncio.to_thread(_encode_thumbnail, data)
            except Exception as ex:  # pylint: disable = W0718
                # Do not cache transient failures (timeouts...), only bad responses above
                _LOG.debug(
                    "[%s] Thumbnail fetch failed (%s): %s", self._address, ex, url
                )
                return None
        self._store(url, encoded)
        return encoded

    def _task(self, url: str) -> asyncio.Task:
        task = self._pending.get(url)
        if task is None:
            task = asyncio.create_task(self._fetch(url))
            self._pending[url] = task
            task.add_done_callback(lambda _: self._pending.pop(url, None))
        return task

    async def embed(
        self, items: list[BrowseMediaItem], deadline: float = THUMBNAIL_DEADLINE
    ) -> None:
        """Replace HTTP thumbnails of the given items with embedded images, waiting at most ``deadline`` seconds."""
        if not PIL_AVAILABLE:
            return
        waiting: dict[asyncio.Task, list[BrowseMediaItem]] = {}
        for item in items:
            url = item.thumbnail
            if not url or not url.startswith(("http://", "https://")):
                continue
            if url in self._cache:
                self._cache.move_to_end(url)
                # None = Kodi has no usable image: drop the URL so the remote shows its default icon
                item.thumbnail = self._cache[url]
                continue
            waiting.setdefault(self._task(url), []).append(item)
        if not waiting:
            return
        done, _ = await asyncio.wait(waiting.keys(), timeout=deadline)
        for task in done:
            if (
                not task.cancelled()
                and task.exception() is None
                and (encoded := task.result())
            ):
                for item in waiting[task]:
                    item.thumbnail = encoded
        if len(done) < len(waiting):
            _LOG.debug(
                "[%s] %d/%d thumbnails not ready after %.1fs, still loading in background",
                self._address,
                len(waiting) - len(done),
                len(waiting),
                deadline,
            )


class BrowseCache:
    """Short-lived cache of built browse pages, with background prefetch of the next page."""

    def __init__(self, address: str):
        """Create the cache."""
        self._address = address
        self.thumbnails = ThumbnailCache(address)
        self._pages: OrderedDict[tuple, tuple[float, BrowseResult]] = OrderedDict()
        self._pending: dict[tuple, asyncio.Task] = {}

    def clear(self) -> None:
        """Drop cached pages (thumbnails are kept, they do not change)."""
        self._pages.clear()

    def _get(self, key: tuple) -> BrowseResult | None:
        entry = self._pages.get(key)
        if entry is None:
            return None
        timestamp, result = entry
        if time.monotonic() - timestamp > PAGE_CACHE_TTL:
            self._pages.pop(key, None)
            return None
        return result

    def _put(self, key: tuple, result: BrowseResult) -> None:
        self._pages[key] = (time.monotonic(), result)
        self._pages.move_to_end(key)
        while len(self._pages) > PAGE_CACHE_SIZE:
            self._pages.popitem(last=False)

    async def _build(self, key: tuple, browse: BrowseFn) -> BrowseResult | None:
        start = time.monotonic()
        result = await browse()
        if result is None:
            return None
        item, _ = result
        await self.thumbnails.embed(list(item.items or []) + [item])
        _LOG.debug(
            "[%s] Built browse page %s in %.2fs",
            self._address,
            key,
            time.monotonic() - start,
        )
        self._put(key, result)
        return result

    async def browse(
        self,
        media_id: str | None,
        media_type: str | None,
        page: int,
        limit: int,
        browse: Callable[[int], Awaitable[BrowseResult | None]],
    ) -> BrowseResult | None:
        """
        Return a browse page, from cache when possible, and prefetch the following page.

        :param browse: coroutine factory building the page for the given page number
        """
        key = (media_id, media_type, page, limit)
        result = self._get(key)
        if result is None:
            pending = self._pending.get(key)
            if pending is not None:
                result = await asyncio.shield(pending)
            if result is None:
                result = await self._build(key, lambda: browse(page))
        else:
            # Fill in thumbnails that finished loading in the background since the page was built
            item, _ = result
            await self.thumbnails.embed(list(item.items or []) + [item], deadline=0.5)
            _LOG.debug("[%s] Browse page %s served from cache", self._address, key)
        if result is None:
            return None

        _, pagination = result
        count = getattr(pagination, "count", 0) or 0
        if page * limit < count:
            self._prefetch(
                (media_id, media_type, page + 1, limit), lambda: browse(page + 1)
            )
        return result

    def _prefetch(self, key: tuple, browse: BrowseFn) -> None:
        if key in self._pending or self._get(key) is not None:
            return

        async def run() -> BrowseResult | None:
            try:
                return await self._build(key, browse)
            except Exception as ex:  # pylint: disable = W0718
                _LOG.debug("[%s] Prefetch of %s failed: %s", self._address, key, ex)
                return None
            finally:
                self._pending.pop(key, None)

        self._pending[key] = asyncio.create_task(run())

    async def embed_thumbnails(self, items: list[BrowseMediaItem]) -> None:
        """Embed thumbnails for items not going through the page cache (search results)."""
        await self.thumbnails.embed(items)
