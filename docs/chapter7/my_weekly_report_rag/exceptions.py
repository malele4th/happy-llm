#!/usr/bin/env python
# -*- coding: utf-8 -*-


class WeeklyReportRagError(Exception):
    """周报 RAG 基础异常。"""


class EnvConfigError(WeeklyReportRagError):
    """环境变量未配置。"""


class IndexNotFoundError(WeeklyReportRagError):
    """索引目录不存在或不完整。"""


class IndexCorruptError(WeeklyReportRagError):
    """索引数据不一致或损坏。"""


class NoDataError(WeeklyReportRagError):
    """数据目录下没有可索引文件。"""
