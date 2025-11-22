import tiktoken

from .simple_tokenizer import SimpleTokenizer

def gpt_encoding():
    
    """
    GPT 2 encoding using Tiktoken
    """
    return tiktoken.get_encoding("gpt2")