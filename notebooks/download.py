import tempfile
from typing import Optional

import httpx
from tqdm import tqdm


def download_file(
    url: str, headers: Optional[dict] = None, cookies: Optional[httpx.Cookies] = None
) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as file:
        with httpx.stream("GET", url, headers=headers, cookies=cookies) as res:
            total = int(res.headers["Content-Length"])
            with tqdm(
                total=total, unit_scale=True, unit_divisor=1024, unit="B"
            ) as progress:
                num_bytes_downloaded = res.num_bytes_downloaded
                for chunk in res.iter_bytes():
                    file.write(chunk)
                    progress.update(res.num_bytes_downloaded - num_bytes_downloaded)
                    num_bytes_downloaded = res.num_bytes_downloaded
    return file.name
