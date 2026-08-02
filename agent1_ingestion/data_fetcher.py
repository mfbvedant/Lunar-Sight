"""
Data Fetcher
==============
Download clients for Chandrayaan-2 DFSAR data (PRADAN/ISDA) and
LOLA DEM tiles from NASA PDS Geosciences Node.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class BaseFetcher:
    """Base class for data download clients."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _download_file(
        self,
        url: str,
        local_path: Path,
        expected_checksum: Optional[str] = None,
        chunk_size: int = 8192,
    ) -> Path:
        """Download a file with progress reporting and optional checksum verification.

        Args:
            url: URL to download from.
            local_path: Local path to save to.
            expected_checksum: Expected MD5 hex digest, if available.
            chunk_size: Download chunk size in bytes.

        Returns:
            Path to downloaded file.
        """
        import requests

        # Skip if already downloaded and checksum matches
        if local_path.exists() and expected_checksum:
            if self._verify_checksum(local_path, expected_checksum):
                logger.info(f"File already exists and checksum OK: {local_path}")
                return local_path

        logger.info(f"Downloading: {url}")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (chunk_size * 100) == 0:
                        logger.info(f"  Progress: {pct:.1f}% ({downloaded}/{total_size})")

        logger.info(f"Downloaded: {local_path} ({local_path.stat().st_size} bytes)")

        if expected_checksum:
            if not self._verify_checksum(local_path, expected_checksum):
                raise ValueError(
                    f"Checksum mismatch for {local_path}. "
                    f"Expected {expected_checksum}"
                )

        return local_path

    @staticmethod
    def _verify_checksum(path: Path, expected_md5: str) -> bool:
        """Verify MD5 checksum of a file."""
        md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        actual = md5.hexdigest()
        return actual.lower() == expected_md5.lower()


class DFSARFetcher(BaseFetcher):
    """Download client for Chandrayaan-2 DFSAR data.

    Supports manual file placement and authenticated download from PRADAN/ISDA.
    Since PRADAN may require interactive login, this class primarily manages
    local file organization and validation.
    """

    # Base URL for ISDA (Indian Space Data Archive)
    ISDA_BASE_URL = "https://pradan.issdc.gov.in/pradan/"

    def __init__(
        self,
        output_dir: str | Path,
        band: str = "L",
    ):
        """
        Args:
            output_dir: Directory to store downloaded DFSAR files.
            band: SAR band — "L" or "S".
        """
        super().__init__(output_dir)
        self.band = band.upper()
        self.data_dir = self.output_dir / f"dfsar_{self.band.lower()}_band"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def register_local_files(
        self,
        data_file: str | Path,
        label_file: Optional[str | Path] = None,
    ) -> dict[str, Path]:
        """Register locally available DFSAR files (e.g., already downloaded).

        Copies or symlinks the files into the project data directory.

        Args:
            data_file: Path to the .img binary data file.
            label_file: Path to the .xml PDS4 label file (optional).

        Returns:
            Dict with 'data' and optionally 'label' paths in the project.
        """
        data_file = Path(data_file)
        result: dict[str, Path] = {}

        if not data_file.exists():
            raise FileNotFoundError(f"DFSAR data file not found: {data_file}")

        dest_data = self.data_dir / data_file.name
        if not dest_data.exists():
            shutil.copy2(str(data_file), str(dest_data))
            logger.info(f"Copied DFSAR data → {dest_data}")
        result["data"] = dest_data

        if label_file:
            label_file = Path(label_file)
            if label_file.exists():
                dest_label = self.data_dir / label_file.name
                if not dest_label.exists():
                    shutil.copy2(str(label_file), str(dest_label))
                result["label"] = dest_label

        return result

    def list_local_files(self) -> list[Path]:
        """List all DFSAR files currently in the data directory."""
        return sorted(self.data_dir.glob("*"))

    def fetch(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> dict[str, Path]:
        """Fetch DFSAR data from PRADAN/ISDA.

        NOTE: PRADAN requires interactive authentication. This method provides
        guidance for manual download if automated access is not configured.

        Args:
            product_url: Direct URL to the product (if known).
            product_id: PRADAN product identifier.

        Returns:
            Dict with paths to downloaded files.
        """
        if product_url:
            fname = product_url.split("/")[-1]
            local_path = self.data_dir / fname
            return {"data": self._download_file(product_url, local_path)}

        logger.warning(
            "PRADAN requires interactive login. "
            "Please download DFSAR data manually from:\n"
            f"  {self.ISDA_BASE_URL}\n"
            "Then use register_local_files() to register the downloaded files."
        )
        return {}


class LOLAFetcher(BaseFetcher):
    """Download client for LOLA DEM data from NASA PDS Geosciences Node.

    Supports both the global 118m DEM and higher-resolution SLDEM2015.
    """

    # NASA PDS Geosciences Node URLs
    LOLA_BASE_URL = (
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        "lrolol_1xxx/data/lola_gdr/"
    )
    SLDEM_BASE_URL = (
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        "lrolol_1xxx/data/sldem2015/"
    )

    def __init__(
        self,
        output_dir: str | Path,
        resolution: str = "118m",
    ):
        """
        Args:
            output_dir: Directory to store downloaded DEM files.
            resolution: "118m" for global LOLA GDR, or "sldem" for SLDEM2015.
        """
        super().__init__(output_dir)
        self.resolution = resolution
        self.dem_dir = self.output_dir / "lola_dem"
        self.dem_dir.mkdir(parents=True, exist_ok=True)

    def fetch_south_pole_dem(
        self,
        lat_min: float = -90.0,
        lat_max: float = -60.0,
    ) -> Path:
        """Fetch the LOLA south polar DEM tile.

        Args:
            lat_min: Southern boundary (default: -90°).
            lat_max: Northern boundary (default: -60°).

        Returns:
            Path to downloaded DEM file.
        """
        # LOLA GDR south polar tile naming convention
        tile_name = f"ldem_sp_{abs(int(lat_max))}s_{abs(int(lat_min))}s.img"
        tile_url = urljoin(self.LOLA_BASE_URL, f"polar/{tile_name}")

        local_path = self.dem_dir / tile_name

        try:
            return self._download_file(tile_url, local_path)
        except Exception as e:
            logger.warning(
                f"Could not download LOLA DEM from {tile_url}: {e}\n"
                "Please download manually and place in: "
                f"{self.dem_dir}"
            )
            return local_path

    def register_local_dem(self, dem_path: str | Path) -> Path:
        """Register a locally available DEM file.

        Args:
            dem_path: Path to the DEM file.

        Returns:
            Path to the file in the project data directory.
        """
        dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")

        dest = self.dem_dir / dem_path.name
        if not dest.exists():
            shutil.copy2(str(dem_path), str(dest))
            logger.info(f"Copied DEM → {dest}")

        return dest

    def list_local_files(self) -> list[Path]:
        """List all DEM files currently in the data directory."""
        return sorted(self.dem_dir.glob("*"))
