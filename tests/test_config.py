from src.config import Config


def test_config_path():
    print("-" * 30)
    for attr in dir(Config):
        if attr.startswith("__"):
            continue
        print(f"| Config.{attr} = {getattr(Config, attr)}")
    print("_" * 30)
