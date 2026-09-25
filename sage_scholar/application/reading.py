from enum import StrEnum
import re


class ReadingStrategy(StrEnum):
    TARGETED = "targeted"
    OVERVIEW = "overview"

    @classmethod
    def for_query(cls, query: str):
        if re.search(r"概括|概览|总结.{0,8}(全文|论文)|梳理.{0,16}(方法|实验|结论)|详细介绍|全文|summari[sz]e|overview", query, re.I):
            return cls.OVERVIEW
        return cls.TARGETED
