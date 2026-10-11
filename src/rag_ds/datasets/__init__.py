"""公开研究数据到项目统一契约的可复现转换。"""

from rag_ds.datasets.averitec import (
    AVERITEC_DOWNLOAD_URLS,
    AveritecDataError,
    build_averitec_dataset,
)
from rag_ds.datasets.climate_fever import (
    CLIMATE_FEVER_DOWNLOAD_URL,
    ClimateFeverDataError,
    build_climate_fever_dataset,
)

__all__ = [
    "AVERITEC_DOWNLOAD_URLS",
    "CLIMATE_FEVER_DOWNLOAD_URL",
    "AveritecDataError",
    "ClimateFeverDataError",
    "build_averitec_dataset",
    "build_climate_fever_dataset",
]
