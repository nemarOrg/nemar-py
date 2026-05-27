"""NEMAR S3 file-delivery contract — public submodule.

Re-exports of the constants and helper defined in :mod:`nemar._s3`.
The names below describe the bucket layout that every NEMAR dataset
shares — verified end-to-end against the official DataLad clones.
"""

from __future__ import annotations

from nemar._s3 import NEMAR_S3_BUCKET as NEMAR_S3_BUCKET
from nemar._s3 import NEMAR_S3_HOST as NEMAR_S3_HOST
from nemar._s3 import NEMAR_S3_REGION as NEMAR_S3_REGION
from nemar._s3 import s3_object_url as s3_object_url
