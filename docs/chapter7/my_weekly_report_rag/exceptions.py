#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""周报 RAG 统一异常层次。"""


class WeeklyReportRagError(Exception):
    """所有业务异常的基类。"""


class EnvConfigError(WeeklyReportRagError):
    """环境变量未配置。"""


class IndexNotFoundError(WeeklyReportRagError):
    """索引目录不存在或不完整。"""


class IndexCorruptError(WeeklyReportRagError):
    """索引数据不一致或损坏。"""


class NoDataError(WeeklyReportRagError):
    """数据目录下没有可索引文件。"""


class EmbeddingError(WeeklyReportRagError):
    """Embedding 请求或结果异常。"""


class ApiRequestError(WeeklyReportRagError):
    """OpenAI 兼容 API 调用失败。"""
