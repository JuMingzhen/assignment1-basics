import os
import multiprocessing
from typing import List, Callable
from typing import BinaryIO
import regex as re
import itertools

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

def process_func(text: str):
    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter = "|".join(escaped_tokens)
    temp = re.split(delimiter, text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    result = [re.finditer(PAT, item) for item in temp]
    return result

def pretokenize(input_path: str | os.PathLike, special_tokens: list[str],):
    core_count = multiprocessing.cpu_count()
    print(f"Available core number:{core_count}")
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, core_count, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
            # Run pre-tokenization on your chunk and store the counts for each pre-token
    with multiprocessing.Pool() as pool:
        results = pool.map(process_func, chunks)
    return results

#given result from pretokenization, return stats(count) dictionary for next step
def build_vocab_dict(l):
    result = dict()
    for i in l:
        for j in i:
            for item in j:
                t = tuple(item.encode("utf-8"))
                result[t] = result.get(t, 0) + 1

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
        while i < len(num_list) - 1:
            if (num_list[i], num_list[i+1]) == pair:
                num_list[i] = new_id
                del num_list[i+1]
            else:
                i += 1
        return tuple(num_list), target
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
    vocab: dict[int, bytes] = dict()
    merges: list[tuple[bytes, bytes]] = []
    if vocab_size < 256 + len(special_tokens):
        print("Error! Unable to construct dictionary.")
    for i in range(256):
        vocab[i] = bytes(i)

    # pre-tokenize
    tokens = pretokenize(input_path, special_tokens)
    dic_word = build_vocab_dict(tokens)
    
    # construct pair-wise counts dictionary
    dic_pair = dict()
    for i in range(256):
        for j in range(256):
            count = 0
            for key, value in dic_word.items():
                count += seek_pair(key, [i,j]) * value
            dic_pair[(i,j)] = count

    # merge
    id = 256
    max_num = vocab_size - len(special_tokens)
    while id <= max_num:
        max_key = max(dic_pair, key=lambda k: dic_pair[k])
        merges.append(tuple(vocab[max_key[0]], vocab[max_key[1]]))
        vocab[id] = vocab[max_key[0]] + vocab[max_key[1]]
        # update dic_pair and dic_word
        for item in dic_word.keys():
            new, old = merge(max_key, item, id)
            if new:
                dic_word[new] = dic_word[old]
                del dic_word[old]
                # compare change of pair-wise count between new and old
                new_dic = dict()
                old_dic = dict()
                for i in range(len(new) - 1):
                    new_dic[(new[i], new[i+1])] = new_dic.get((new[i], new[i+1]), 0) + 1
                for i in range(len(old) - 1):
                    new_dic[(old[i], old[i+1])] = old_dic.get((old[i], old[i+1]), 0) + 1
                dif = calculate_difference(old_dic, new_dic)
                for key,value in dif.items():
                    dic_pair[key] = dic_pair.get(key,0) + value * dic_word[new]
        del dic_pair[max_key]               
        id += 1
    for i, t in enumerate(special_tokens):
        vocab[i + id] = t.encode("utf-8")
    return vocab, merges