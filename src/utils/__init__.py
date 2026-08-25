
def line(sep: str = "=", count: int = 50):
    print(sep * count)

def log_title(title: str):
    print(f"\n{title}")
    line()

def pretty_log(title: str, content):
    line()
    print(f"  {title}  ")
    line()
    print(content)
    line('-')