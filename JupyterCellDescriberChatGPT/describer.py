import openai
import re
import __main__


class CellNotFoundError(Exception):
    """
    Exception raised when the specified cell cannot be found.

    """

class JupyterCellDescriberChatGPT:
    """
    A class that describes the processing performed by a Jupyter Notebook cell using the OpenAI GPT-3 model.

    Attributes:
        OPENAI_APIKEY (str): The API key for accessing the OpenAI API.
        lang (str): The language to be used for describing the cell. Default is "en".

    Methods:
        __init__(self, OPENAI_APIKEY, lang="en"): Constructs a CellDescriberOpenAI object.
        describe(self, number=-1): Describes the processing performed by a Jupyter Notebook cell.
        __call__(self, number=-1): Returns the description of the processing performed by a Jupyter Notebook cell.

    """

    def __init__(self, OPENAI_APIKEY, lang="en"):
        """
        Constructs a CellDescriberOpenAI object.

        Args:
            OPENAI_APIKEY (str): The API key for accessing the OpenAI API.
            lang (str): The language to be used for describing the cell. Default is "en".

        """
        self.api_key = OPENAI_APIKEY
        self.lang = lang
        self.messages = []
        self.last_number = -1

    def describe(self, number=-1):
        """
        Describes the processing performed by a Jupyter Notebook cell.

        Args:
            number (int): The index of the cell to be described. Default is -1, which describes the most recent cell.

        Returns:
            The description of the processing performed by the specified Jupyter Notebook cell.

        Raises:
            CellNotFoundError: If the specified cell cannot be found.

        """

        _globals = vars(__main__)

        keys = set([k for k in _globals.keys() if re.match("^_i[0-9]+$", k)])

        if number == -1:
            number = max([int(k.split("i")[-1]) for k in keys])-1

        key = f"_i{number}"

        if key in keys:
            if self.lang in ["ja", "en"]:
                if self.last_number != number:
                    _src = _globals[key]
                    self.messages = []
                    if self.lang == "ja":
                        self.messages.append({"role": "user", "content": f"これは、jupyterのセルです。このセルが行っている処理について説明してください {_src}"})
                    elif self.lang == "en":
                        self.messages.append({"role": "user", "content": f"This is a jupyter cell. Describe the processing this cell is doing {_src}"})
                    else:
                        raise NotImplementedError

                else:
                    if self.lang == "ja":
                        self.messages.append({"role": "user", "content": "もっと詳しく教えてください"})
                    elif self.lang == "en":
                        self.messages.append({"role": "user", "content": "Tell me more about it."})
                    else:
                        raise NotImplementedError

                completion = openai.ChatCompletion.create(
                    api_key=self.api_key,
                    model="gpt-3.5-turbo",
                    messages=self.messages
                )

                self.messages.append(completion["choices"][0]["message"])
                self.last_number = number
                return completion
            else:
                raise NotImplementedError

        else:
            raise CellNotFoundError

    def __call__(self, number=-1):
        """
        Returns the description of the processing performed by a Jupyter Notebook cell.

        Args:
            number (int): The index of the cell to be described. Default is -1, which describes the most recent cell.

        Returns:
            The description of the processing performed by the specified Jupyter Notebook cell.

        """
        return self.describe(number)["choices"][0]["message"]["content"]