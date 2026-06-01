from __future__ import annotations

import os


def main() -> None:
    client_id = os.getenv("CLIENT_ID", "A")
    server_host = os.getenv("SERVER_HOST", "fl_server")
    epsilon = float(os.getenv("TARGET_EPSILON", "1.0"))
    print({"client_id": client_id, "server_host": server_host, "target_epsilon": epsilon})


if __name__ == "__main__":
    main()
