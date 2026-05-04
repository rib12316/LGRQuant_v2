"""
Unified data loader for LGRQuant_v2.
Merged from data_utils.py (stage1/stage2/precompute) and datautils.py (inference/restorative_lora).
"""
import os
import random
import numpy as np
import torch
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Tokenizer wrapper (used by both versions)
# ---------------------------------------------------------------------------
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# ---------------------------------------------------------------------------
# Version 1: data_utils.py based functions (used by stage1 eval, stage2, precompute)
# These accept tokenizer object directly.
# ---------------------------------------------------------------------------

def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = load_dataset('/data1/xx/xxquant/datasets/wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = load_dataset('/data1/xx/xxquant/datasets/wikitext', 'wikitext-2-raw-v1', split='train')
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        valdata = load_dataset(
            "json", data_files='/data1/xx/xxquant/datasets/allenai/c4/en/c4-validation.00000-of-00008.gz', split='train'
        )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = load_dataset(
            "json", data_files='/data1/xx/xxquant/datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz', split='train'
        )
        trainloader = []
        eos_id = tokenizer.eos_token_id
        for idx in range(nsamples):
            token_ids = []
            while len(token_ids) < seqlen + 1:
                i = random.randint(0, len(traindata) - 1)
                enc = tokenizer(traindata[i]["text"], add_special_tokens=False, return_tensors=None)
                ids = enc.get("input_ids", [])
                if len(ids) == 0:
                    continue
                token_ids.extend(ids)
                if eos_id is not None:
                    token_ids.append(eos_id)

            start = random.randint(0, len(token_ids) - seqlen - 1)
            window = token_ids[start : start + seqlen]
            inp = torch.tensor(window, dtype=torch.long).unsqueeze(0)
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            if (idx + 1) % 50 == 0:
                print(f"[c4] prepared {idx + 1}/{nsamples} samples (seqlen={seqlen})")
        return trainloader


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seqlen, tokenizer):
    traindata = load_dataset("./datasets/pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(args, name, tokenizer, nsamples=128, seqlen=2048, eval_mode=False):
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode)
    elif 'ptb' in name:
        dataset = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'c4' in name:
        dataset = get_c4_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'pile' in name:
        dataset = get_pile(nsamples, seqlen, tokenizer)

    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset


# ---------------------------------------------------------------------------
# Version 2: datautils.py based functions (used by stage1 PTQ training, inference)
# These accept model path and create tokenizer internally.
# ---------------------------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2_legacy(nsamples, seed, seqlen, model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset('/data1/xx/xxquant/datasets/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('/data1/xx/xxquant/datasets/wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb_legacy(nsamples, seed, seqlen, model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_legacy(nsamples, seed, seqlen, model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset(
        "json", data_files='/data1/xx/xxquant/datasets/allenai/c4/en/c4-train.00000-of-01024.json.gz', split='train'
    )
    valdata = load_dataset(
        "json", data_files='/data1/xx/xxquant/datasets/allenai/c4/en/c4-validation.00000-of-00008.gz', split='train'
    )

    random.seed(seed)
    trainloader = []
    choose = set()
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen and i not in choose:
                break
        choose.add(i)
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_ptb_new_legacy(nsamples, seed, seqlen, model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new_legacy(nsamples, seed, seqlen, model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders_legacy(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2_legacy(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new_legacy(nsamples, seed, seqlen, model)
        return get_ptb_legacy(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new_legacy(nsamples, seed, seqlen, model)
        return get_c4_legacy(nsamples, seed, seqlen, model)


# ---------------------------------------------------------------------------
# Additional datasets from datautils.py (used by stage2, restorative_lora)
# ---------------------------------------------------------------------------

def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)
    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_alpaca(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("tatsu-lab/alpaca", split=split)
    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_deita_10k(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def format_deita(example):
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        roles = {"human": "USER", "gpt": "ASSISTANT"}
        sep = " "
        sep2 = "</s>"
        conversations = example["conversations"]
        formatted_text = system_prompt + sep
        for sentence in conversations:
            role = roles.get(sentence["from"], sentence["from"])
            content = sentence["value"]
            if role == "USER":
                formatted_text += role + ": " + content + sep
            else:
                formatted_text += role + ": " + content + sep2
        return {"text": formatted_text}

    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{percent}%]"
    else:
        split = "train"
    dataset = load_dataset("hkust-nlp/deita-10k-v0", split=split)
    dataset = dataset.map(format_deita, num_proc=os.cpu_count())
    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)
    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    elif name == "alpaca":
        data = get_alpaca(tokenizer, data_percent)
    elif name == "deita_10k":
        data = get_deita_10k(tokenizer, data_percent)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


def get_ptq_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    print(f" get_ptq_calib_data {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = (
        f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors='pt')
        return testenc
    if "ptb" in name:
        testdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["sentence"]), return_tensors='pt')
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors='pt')
        return testenc
    raise NotImplementedError
