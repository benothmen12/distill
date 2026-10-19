"""`python -m distill` prints the tool definitions sent to the model."""

import json

from distill.tools import to_openai_tools


def main() -> None:
    print(json.dumps(to_openai_tools(), indent=2))


if __name__ == "__main__":
    main()
