import ssl
import certifi


def configure_ssl_certifi() -> None:
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
