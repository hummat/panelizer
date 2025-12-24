from panel_flow import hello


def test_hello_default() -> None:
    assert hello() == "hello, world"
