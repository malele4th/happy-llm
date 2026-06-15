import datetime

import requests
import wikipedia

REQUEST_TIMEOUT = 10

def count_letter_in_string(a: str, b: str) -> str:
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    string = a.lower()
    letter = b.lower()
    count = string.count(letter)
    return f"The letter '{letter}' appears {count} times in the string."


def get_current_datetime() -> str:
    """
    获取真实的当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.datetime.now()
    return current_datetime.strftime("%Y-%m-%d %H:%M:%S")


def get_current_temperature(latitude: float, longitude: float) -> str:
    """
    获取指定经纬度位置的当前温度。
    :param latitude: 纬度坐标。
    :param longitude: 经度坐标。
    :return: 当前温度的字符串表示。
    """
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "forecast_days": 1,
    }

    response = requests.get(open_meteo_url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    results = response.json()

    current_utc_time = datetime.datetime.now(datetime.UTC)
    time_list = [
        datetime.datetime.fromisoformat(time_str).replace(tzinfo=datetime.timezone.utc)
        for time_str in results["hourly"]["time"]
    ]
    temperature_list = results["hourly"]["temperature_2m"]

    closest_time_index = min(
        range(len(time_list)),
        key=lambda i: abs(time_list[i] - current_utc_time),
    )
    current_temperature = temperature_list[closest_time_index]
    return f"现在温度是 {current_temperature}°C"


def search_wikipedia(query: str) -> str:
    """
    在维基百科中搜索指定查询的前三个页面摘要。
    :param query: 要搜索的查询字符串。
    :return: 包含前三个页面摘要的字符串。
    """
    try:
        page_titles = wikipedia.search(query)
    except Exception as e:
        return f"维基百科搜索失败: {e}"

    summaries = []
    skipped = []
    for page_title in page_titles[:3]:
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"页面: {page_title}\n摘要: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ) as e:
            skipped.append(f"{page_title}({e})")

    if not summaries:
        detail = "; ".join(skipped) if skipped else "无匹配页面"
        return f"维基百科没有搜索到合适的结果（{detail}）"
    return "\n\n".join(summaries)


