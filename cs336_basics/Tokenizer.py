import os
import multiprocessing
from typing import BinaryIO
import regex as re
from collections import defaultdict
from typing import Iterable, Iterator
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_func(text: str, special_tokens: list[str]):
    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter = "|".join(escaped_tokens)
    temp = re.split(delimiter, text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_counts = defaultdict(int)
    for item in temp:
        for match in re.finditer(PAT, item):
            token = tuple(match.group().encode("utf-8"))
            token_counts[token] += 1
    return dict(token_counts)

def pretokenize(input_path: str | os.PathLike, special_tokens: list[str]):
    core_count = multiprocessing.cpu_count()
    print(f"Available core number:{core_count}")
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, core_count, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    from functools import partial
    func = partial(process_func, special_tokens=special_tokens)
    import pickle
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    with multiprocessing.Pool() as pool:
        results = pool.map(func, chunks)
    total_counts = defaultdict(int)
    for counts in results:
        for token, cnt in counts.items():
            total_counts[token] += cnt
    return total_counts

def seek_pair(target: tuple[int], pair: list[int]):
    count = 0
    for i in range(len(target) - 1):
        if target[i] == pair[0]:
            if target[i + 1] == pair[1]:
                count += 1
    return count

def merge(pair: tuple[int], target: tuple[int], new_id: int):
    if pair[0] in target and pair[1] in target:
        num_list = list(target)
        i = 0
        status = 0
        while i < len(num_list) - 1:
            if (num_list[i], num_list[i+1]) == pair:
                status = 1
                num_list[i] = new_id
                del num_list[i+1]
            else:
                i += 1
        if status:
            return tuple(num_list), target
        else:
            return None, None
    return None, None

def calculate_difference(old_dict, new_dict):
    result = {}
    all_keys = set(old_dict.keys()).union(set(new_dict.keys()))
    for key in all_keys:
        old_val = old_dict.get(key, 0)
        new_val = new_dict.get(key, 0)
        if new_val != old_val:
            result[key] = new_val - old_val 
    return result

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocab
    t1 = time.time()
    vocab: dict[int, bytes] = dict()
    merges: list[tuple[bytes, bytes]] = []
    if vocab_size < 256 + len(special_tokens):
        print("Error! Unable to construct dictionary.")
    for i in range(256):
        vocab[i] = bytes([i])
    t2 = time.time()
    print(f"初始化用时：{t2-t1}")
    # pre-tokenize
    t1 = time.time()
    dic_word = pretokenize(input_path, special_tokens)
    t2 = time.time()
    print(f"pretoken用时：{t2-t1}")
    # construct pair-wise counts dictionary
    t1 = time.time()
    dic_pair = defaultdict(int)
    for key, value in dic_word.items():
        for i in range(len(key)-1):
            dic_pair[(key[i], key[i+1])] += value
    dic_pair = dict(dic_pair)
    t2 = time.time()
    print(f"构建pair对词典用时：{t2-t1}")
    # merge
    t1 = time.time()
    id = 256
    max_num = vocab_size - len(special_tokens)
    while id < max_num:
        max_key = max(dic_pair, key=lambda k: (dic_pair[k], (vocab[k[0]],vocab[k[1]]))) #following dictonary（字典序）
        merges.append((vocab[max_key[0]], vocab[max_key[1]]))
        vocab[id] = vocab[max_key[0]] + vocab[max_key[1]]
        # update dic_pair and dic_word
        temp_list = list(dic_word.keys())
        for item in temp_list:
            new, old = merge(max_key, item, id)
            if new:
                dic_word[new] = dic_word[old] + dic_word.get(new, 0)
                del dic_word[old]
                # compare change of pair-wise count between new and old
                new_dic = dict()
                old_dic = dict()
                for i in range(len(new) - 1):
                    new_dic[(new[i], new[i+1])] = new_dic.get((new[i], new[i+1]), 0) + 1
                for i in range(len(old) - 1):
                    old_dic[(old[i], old[i+1])] = old_dic.get((old[i], old[i+1]), 0) + 1
                dif = calculate_difference(old_dic, new_dic)
                for key,value in dif.items():
                    dic_pair[key] = dic_pair.get(key,0) + value * dic_word[new]
        del dic_pair[max_key]               
        id += 1
    t2 = time.time()
    print(f"合并用时：{t2-t1}")
    for i, t in enumerate(special_tokens):
        vocab[i + id] = t.encode("utf-8")
    return vocab, merges

# This part is for test(debugging)
# if __name__ == "__main__":
#     import pathlib
#     test_input_path = pathlib.Path("../data/TinyStoriesV2-GPT4-train.txt")  
#     test_vocab_size = 10000
#     test_special_tokens = ["<|endoftext|>"]
#     trained_vocab, trained_merges = train_bpe(
#         input_path=test_input_path,
#         vocab_size=test_vocab_size,
#         special_tokens=test_special_tokens
#     )
#     import pickle
#     with open("vocab.pkl", "wb") as f:
#         pickle.dump(trained_vocab, f) 
#     with open("merges.pkl", "wb") as f:
#         pickle.dump(trained_merges, f) 
#     m = max(trained_vocab, key = lambda k: len(trained_vocab[k]))
#     print(f"The longest vocab is {trained_vocab[m]}")
if __name__ == "__main__":
    import pathlib
    test_input_path = pathlib.Path("../data/owt_train.txt")  
    test_vocab_size = 32000
    test_special_tokens = ["<|endoftext|>"]
    trained_vocab, trained_merges = train_bpe(
        input_path=test_input_path,
        vocab_size=test_vocab_size,
        special_tokens=test_special_tokens
    )
    import pickle
    with open("vocab_owt.pkl", "wb") as f:
        pickle.dump(trained_vocab, f) 
    with open("merges_owt.pkl", "wb") as f:
        pickle.dump(trained_merges, f) 
    m = max(trained_vocab, key = lambda k: len(trained_vocab[k]))
    print(f"The longest vocab is {trained_vocab[m]}")


#####################################################################
##########################Tokenizer class############################
#####################################################################
class Tokenizer():
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        import pickle
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
        return cls(vocab, merges, special_tokens)
    # def encode(self, text: str) -> list[]:
    #     escaped_tokens = [re.escape(token) for token in self.special_tokens]
    #     pattern = f"({'|'.join(escaped_tokens)})"
    #     delimiter = "|".join(escaped_tokens)
    def encode(self, text: str) -> list[int]:
        if len(text) == 0:
            return []
        vocab_dict = {k: v for v, k in self.vocab.items()}
        # deal with special tokens and pre-tokenize
        if self.special_tokens == None:
            text = [text]
            captured_tokens = []
        else:
            special_token = sorted(self.special_tokens, key=lambda s: len(s), reverse=True)
            escaped_tokens = [re.escape(token) for token in special_token]
            pattern = f"({'|'.join(escaped_tokens)})"
            delimiter = "|".join(escaped_tokens)
            captured_tokens = re.findall(pattern, text)
            text = re.split(delimiter, text)
        result = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for i in range(len(text) - 1):
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            item = re.findall(PAT, text[i])
            for t in item:
                result.append([vocab_dict[bytes([i])] for i in t.encode("utf-8")])
            if captured_tokens:
                result.append([vocab_dict[captured_tokens[i].encode("utf-8")]])
        for t in re.findall(PAT, text[-1]):
            result.append([vocab_dict[bytes([i])] for i in t.encode("utf-8")])
        
        # merge
        for vocab in result:
            for merge in self.merges:
                pair = [vocab_dict[merge[0]], vocab_dict[merge[1]]]
                if pair[0] in vocab and pair[1] in vocab:
                    i = 0
                    while i < len(vocab) - 1:
                        status = 0
                        if vocab[i] == pair[0]:
                            if vocab[i + 1] == pair[1]:
                                vocab[i] = vocab_dict[merge[0] + merge[1]]
                                del vocab[i + 1]
                                status = 1
                        if not status:
                            i += 1
        final_result = []
        for item in result:
            final_result += item
        return final_result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        def result():
            for t in iterable:
                t = self.encode(str(t))
                for i in t:
                    yield i
        return result()

    def decode(self, ids: list[int]) -> str:
        if len(ids) == 0:
            return ""
        result = self.vocab[ids[0]]
        for i in range(1, len(ids)):
            result += self.vocab[ids[i]]
        return result.decode("utf-8", errors="replace")   