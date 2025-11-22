import urllib.request

def fetch_verdict_text():
    """
        Fetch Verdict Text
    """
    
    url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")

    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")

    return text

