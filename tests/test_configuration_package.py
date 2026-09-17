"""Application configuration assets stay colocated with their loader."""

from importlib.resources import files

from react_agent.configuration.settings import Settings
from react_agent.metering.pricing import pricing_path


def test_configuration_assets_are_available_in_package(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_PRICING_PATH", raising=False)

    package = files("react_agent.configuration")
    assert package.joinpath("config.yaml").is_file()
    assert package.joinpath("deepseek_pricing.yaml").is_file()
    assert pricing_path() == package.joinpath("deepseek_pricing.yaml")
    assert Settings().tools.rag.max_content_chars == 800
