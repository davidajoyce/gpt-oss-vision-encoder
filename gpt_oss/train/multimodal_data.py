import json
import os
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import logging

import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers

# Constants for special tokens
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200  # Special index for image tokens
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


@dataclass
class DataArguments:
    """Arguments for data loading and processing."""
    data_path: str = None
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = None
    image_aspect_ratio: str = 'square'
    image_processor = None
    mm_use_im_start_end: bool = False
    version: str = "simple"


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    """
    Preprocess multimodal data by handling image tokens.
    
    Args:
        sources: List of conversation data
        data_args: Data configuration
        
    Returns:
        Dictionary with preprocessed data
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Preprocess for LLaMA-2 style conversations.
    Adapted from LLaVA's preprocessing pipeline.
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.LLAMA_2

    # Mask targets for training
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess plain text data for stage 1 training (vision-language alignment).
    """
    # Tokenize text
    conversations = [f"{source[0]['value']}" for source in sources]
    
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest", 
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    
    targets = input_ids.clone()
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning with lazy loading.
    Supports both stage 1 (pretraining) and stage 2 (instruction following) data.
    """

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                if processor is not None:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    # Basic preprocessing if no processor available
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    image = transform(image)
                
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
            except Exception as e:
                logging.warning(f"Error loading image {image_file}: {e}")
                # Skip this sample or use a placeholder
                image = torch.zeros((3, 224, 224))
                sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            image = None

        has_image = ('image' in self.list_data_dict[i])
        
        # Choose preprocessing function based on data type
        if 'plain' in getattr(self.data_args, 'version', ''):
            data_dict = preprocess_plain(sources, self.tokenizer)
        else:
            data_dict = preprocess_llama_2(sources, self.tokenizer, has_image=has_image)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                           labels=data_dict["labels"][0])

        # Add image if present
        if image is not None:
            data_dict['image'] = image
        
        return data_dict


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = tuple(instance["input_ids"] for instance in instances)
        labels = tuple(instance["labels"] for instance in instances)
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        # Handle images if present
        images = [instance.get("image") for instance in instances]
        if any(img is not None for img in images):
            # Stack images, using zeros for missing images
            batch_images = []
            for img in images:
                if img is not None:
                    batch_images.append(img)
                else:
                    # Create placeholder image
                    batch_images.append(torch.zeros((3, 224, 224)))
            images = torch.stack(batch_images, dim=0)
        else:
            images = None

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        if images is not None:
            batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                               data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                        data_path=data_args.data_path,
                                        data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# Helper functions
def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    """
    Tokenize a prompt that contains image tokens.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


# Conversation handling (simplified version)
class ConversationLib:
    """Simple conversation handling for different formats."""
    
    def __init__(self):
        self.default_conversation = SimpleConversation()


class SeparatorStyle:
    """Conversation separator styles."""
    LLAMA_2 = "llama_2"


class SimpleConversation:
    """Simple conversation template."""
    
    def __init__(self):
        self.system = ""
        self.roles = ["USER", "ASSISTANT"]
        self.messages = []
        self.sep = "\n"
        self.sep2 = "</s>"
        self.sep_style = SeparatorStyle.LLAMA_2
        self.version = "simple"
        
    def copy(self):
        return copy.deepcopy(self)
        
    def append_message(self, role, message):
        self.messages.append([role, message])
        
    def get_prompt(self):
        ret = self.system
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ":"
        return ret


# Global conversation lib instance
conversation_lib = ConversationLib()