from libs.main import hello_world


def test_hello_world(caplog):
    hello_world()
    assert "Hello world" in caplog.text
