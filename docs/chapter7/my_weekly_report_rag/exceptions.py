#!/usr/bin/env python
# -*- coding: utf-8 -*-


class WeeklyReportRagError(Exception):
    """周报 RAG 基础异常。"""


class EnvConfigError(WeeklyReportRagError):
    """环境变量未配置。"""


class StorageNotFoundError(WeeklyReportRagError):
    """向量库不存在或不完整。"""


class NoDataError(WeeklyReportRagError):
    """数据目录下没有可索引文件。"""


class IndexExistsError(WeeklyReportRagError):
    """索引已存在且未指定强制重建。"""
